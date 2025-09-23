# ppo_train_balatro.py
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm.auto import tqdm  # 进度条

# === 若 BalatroEnv 在单独文件，请修改为你的文件名；若同一文件，请删除本行并把 Env 放到本文件前部
from envs.BalatroEnv import BalatroEnv

def smooth_curve(y, window=100):
    """简单滑动平均。window 越大越平滑；自动截断到不超过样本长度。"""
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y
    w = int(max(1, min(window, len(y))))
    if w == 1:
        return y
    kernel = np.ones(w, dtype=np.float32) / w
    # valid 模式会短 (len - w + 1)，这样更稳定；需要对 x 对齐
    return np.convolve(y, kernel, mode='valid')

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, max_hand_size: int, hidden=(512, 512)):
        super().__init__()
        self.max_hand_size = max_hand_size
        layers, last = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.backbone = nn.Sequential(*layers)
        self.logits_type = nn.Linear(last, 2)              # 0=弃, 1=出
        self.logits_select = nn.Linear(last, max_hand_size) # hand mask (Bernoulli)
        self.value_head = nn.Linear(last, 1)
        nn.init.zeros_(self.logits_type.bias)
        nn.init.zeros_(self.logits_select.bias)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, x):
        z = self.backbone(x)
        return self.logits_type(z), self.logits_select(z), self.value_head(z).squeeze(-1)

class PPOAgent:
    def __init__(self, obs_dim, max_hand_size, device,
                 lr=3e-4, gamma=0.99, gae_lambda=0.95, clip=0.2,
                 vcoef=0.5, ecoef=0.01, epochs=4, mb_size=1024):
        self.device = device
        self.gamma, self.lmbda = gamma, gae_lambda
        self.clip, self.vcoef, self.ecoef = clip, vcoef, ecoef
        self.epochs, self.mb_size = epochs, mb_size
        self.max_hand_size = max_hand_size
        self.net = ActorCritic(obs_dim, max_hand_size).to(device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, obs_np):
        self.net.eval()
        x = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits_type, logits_sel, value = self.net(x)
        dist_type = torch.distributions.Categorical(logits=logits_type)
        a_type = dist_type.sample()  # (1,)

        # 依据前 52 维手牌 one-hot 个数决定有效位 k
        hand_count = int(np.clip(np.round(obs_np[:52].sum()), 0, self.max_hand_size))
        if hand_count == 0:
            a_mask = torch.zeros((1, self.max_hand_size), dtype=torch.int64, device=self.device)
            logprob_sel = torch.zeros((1,), device=self.device)
            entropy_sel = torch.zeros((1,), device=self.device)
        else:
            logits_valid = logits_sel[:, :hand_count]
            dist_bern = torch.distributions.Bernoulli(logits=logits_valid)
            sampled = dist_bern.sample()
            if sampled.sum() < 1:  # 至少选一张
                idx = torch.argmax(logits_valid, dim=1)
                sampled[0, idx] = 1.0
            pad = torch.zeros((1, self.max_hand_size - hand_count), device=self.device)
            a_mask = torch.cat([sampled, pad], dim=1).to(torch.int64)
            logprob_sel = dist_bern.log_prob(sampled).sum(dim=1)
            entropy_sel = dist_bern.entropy().sum(dim=1)

        logprob_type = dist_type.log_prob(a_type)  # (1,)
        entropy_type = dist_type.entropy()         # (1,)
        total_logprob = (logprob_type + logprob_sel).squeeze(0)
        total_entropy = (entropy_type + entropy_sel).squeeze(0)
        value = value.squeeze(0)

        return int(a_type.item()), a_mask.squeeze(0).detach().cpu().numpy().tolist(), \
               float(total_logprob.item()), float(value.item()), float(total_entropy.item())

    def evaluate_actions(self, obs_b, a_type_b, a_mask_b):
        logits_type, logits_sel, values = self.net(obs_b)
        dist_type = torch.distributions.Categorical(logits=logits_type)
        logprob_type = dist_type.log_prob(a_type_b)
        entropy_type = dist_type.entropy()

        B = obs_b.size(0)
        logprob_sel = torch.zeros(B, device=self.device)
        entropy_sel = torch.zeros(B, device=self.device)
        for i in range(B):
            k = int(torch.clamp(torch.round(obs_b[i, :52].sum()), 0, self.max_hand_size).item())
            if k > 0:
                dist_bern = torch.distributions.Bernoulli(logits=logits_sel[i, :k])
                a_mask_i = a_mask_b[i, :k].float()
                logprob_sel[i] = dist_bern.log_prob(a_mask_i).sum()
                entropy_sel[i] = dist_bern.entropy().sum()

        return logprob_type + logprob_sel, values, entropy_type + entropy_sel

    @staticmethod
    def gae(rews, dones, vals, gamma, lam):
        T = len(rews)
        adv = np.zeros(T, dtype=np.float32)
        last = 0.0
        for t in reversed(range(T)):
            nonterm = 1.0 - float(dones[t])
            next_v = vals[t + 1] if t + 1 < len(vals) else 0.0
            delta = rews[t] + gamma * next_v * nonterm - vals[t]
            last = delta + gamma * lam * nonterm * last
            adv[t] = last
        rets = adv + np.array(vals[:T], dtype=np.float32)
        return rets, adv

    def update(self, traj):
        obs = torch.as_tensor(np.array(traj["obs"]), dtype=torch.float32, device=self.device)
        a_type = torch.as_tensor(np.array(traj["a_type"]), dtype=torch.long, device=self.device)
        a_mask = torch.as_tensor(np.array(traj["a_mask"]), dtype=torch.long, device=self.device)

        with torch.no_grad():
            old_logp = torch.as_tensor(np.array(traj["logp"]), dtype=torch.float32, device=self.device)
            values = torch.as_tensor(np.array(traj["val"]), dtype=torch.float32, device=self.device)
            rewards = np.array(traj["rew"], dtype=np.float32)
            dones = np.array(traj["done"], dtype=np.float32)
            vals_ext = np.concatenate([values.detach().cpu().numpy(), np.array([0.0], dtype=np.float32)], 0)
            rets, adv = self.gae(rewards, dones, vals_ext, self.gamma, self.lmbda)
            returns = torch.as_tensor(rets, dtype=torch.float32, device=self.device)
            advantages = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        N = obs.size(0)
        idx = np.arange(N)
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for s in range(0, N, self.mb_size):
                mb = idx[s:s + self.mb_size]
                mb_obs, mb_type, mb_mask = obs[mb], a_type[mb], a_mask[mb]
                mb_old_logp = old_logp[mb]
                mb_ret, mb_adv = returns[mb], advantages[mb]

                new_logp, new_v, ent = self.evaluate_actions(mb_obs, mb_type, mb_mask)
                ratio = torch.exp(new_logp - mb_old_logp)
                pol_loss = torch.max(-ratio * mb_adv,
                                     -torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * mb_adv).mean()
                v_loss = (mb_ret - new_v).pow(2).mean()
                loss = pol_loss + self.vcoef * v_loss - self.ecoef * ent.mean()

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()

def train(total_steps=200_000, update_steps=4096, seed=0, lr=3e-4,
          gamma=0.99, gae_lambda=0.95, clip=0.2, vcoef=0.5, ecoef=0.01,
          epochs=4, mb_size=1024, save_path=r"C:\Users\17140\Documents\GitHub\Balatro_SimpleVersion\model\ppo_balatro.pt",
          plot_path="training_curve.png",
          max_hand_size=8, max_play=5, max_discard=3):

    set_seed(seed)
    device = get_device()
    print(f"[Init] device={device}, seed={seed}, total_steps={total_steps}, update_steps={update_steps}")

    env = BalatroEnv(max_hand_size=max_hand_size, max_play=max_play, max_discard=max_discard)
    obs = env.reset()
    obs_dim = env.observation_space.shape[0]
    agent = PPOAgent(obs_dim, max_hand_size, device, lr, gamma, gae_lambda, clip, vcoef, ecoef, epochs, mb_size)

    ep_returns, ep_ret = [], 0.0
    total_collected = 0

    # === 进度条 ===
    pbar = tqdm(total=total_steps, desc="Training", unit="step", dynamic_ncols=True)

    try:
        while total_collected < total_steps:
            traj = {"obs": [], "a_type": [], "a_mask": [], "logp": [], "val": [], "rew": [], "done": []}
            steps = 0
            while steps < update_steps and total_collected < total_steps:
                a_type, a_mask, logp, val, _ = agent.act(obs)
                next_obs, reward, done, _ = env.step((a_type, a_mask))

                traj["obs"].append(obs)
                traj["a_type"].append(a_type)
                traj["a_mask"].append(a_mask)
                traj["logp"].append(logp)
                traj["val"].append(val)
                traj["rew"].append(reward)
                traj["done"].append(done)

                ep_ret += reward
                obs = next_obs
                steps += 1
                total_collected += 1

                # === 更新进度条 ===
                pbar.update(1)

                # 每步在进度条右侧显示一些关键指标
                if len(ep_returns) > 0:
                    recent5 = float(np.mean(ep_returns[-5:]))
                    pbar.set_postfix({
                        "episodes": len(ep_returns),
                        "recent5": f"{recent5:.1f}",
                        "cur_ep_ret": f"{ep_ret:.1f}"
                    })
                else:
                    pbar.set_postfix({
                        "episodes": 0,
                        "cur_ep_ret": f"{ep_ret:.1f}"
                    })

                if done:
                    ep_returns.append(ep_ret)
                    ep_ret = 0.0
                    obs = env.reset()

            # === 每收集完一批进行一次 PPO 更新 ===
            agent.update(traj)

            # 批次完成后再用最近 5 局均值反馈一次（可选）
            if len(ep_returns) >= 5:
                recent5 = float(np.mean(ep_returns[-5:]))
                pbar.set_postfix({
                    "episodes": len(ep_returns),
                    "recent5": f"{recent5:.1f}",
                    "last_batch": steps
                })

        pbar.close()

    except KeyboardInterrupt:
        pbar.close()
        print("\n[Info] 手动中断，保存已训练的权重…")

    # 保存模型
    pkg = {
        "state_dict": agent.net.state_dict(),
        "config": {
            "obs_dim": obs_dim,
            "max_hand_size": max_hand_size,
            "max_play": max_play,
            "max_discard": max_discard,
            "seed": seed,
        }
    }
    torch.save(pkg, save_path)
    print(f"[Save] 模型已保存到 {save_path}")

    # 画训练曲线
        # 画训练曲线
    if len(ep_returns) > 0:
        plt.figure(figsize=(8, 5))

        # 原始曲线（浅色细线）
        x_raw = np.arange(len(ep_returns))
        plt.plot(x_raw, ep_returns, linewidth=0.8, alpha=0.35, label="Episode Return (raw)")

        # 平滑曲线（滑动平均）
        window = 100  # 你可以改成 50/100；越大越平滑
        y_smooth = smooth_curve(ep_returns, window=window)
        x_smooth = np.arange(window - 1, window - 1 + len(y_smooth))
        plt.plot(x_smooth, y_smooth, linewidth=2.0, label=f"Moving Avg (window={window})")

        plt.xlabel("Episode")
        plt.ylabel("Cumulative Reward")
        plt.title("PPO Training Curve (BalatroEnv)")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=160)
        print(f"[Plot] 训练曲线已保存到 {plot_path}")



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total_steps", type=int, default=1000000)
    p.add_argument("--update_steps", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--mb_size", type=int, default=1024)
    p.add_argument("--save_path", type=str, default="ppo_balatro.pt")
    p.add_argument("--plot_path", type=str, default="training_curve.png")
    p.add_argument("--max_hand_size", type=int, default=8)
    p.add_argument("--max_play", type=int, default=5)
    p.add_argument("--max_discard", type=int, default=3)
    args = p.parse_args()

    train(total_steps=args.total_steps,
          update_steps=args.update_steps,
          seed=args.seed,
          lr=args.lr,
          gamma=args.gamma,
          gae_lambda=args.gae_lambda,
          clip=args.clip,
          vcoef=args.value_coef,
          ecoef=args.entropy_coef,
          epochs=args.epochs,
          mb_size=args.mb_size,
          save_path=args.save_path,
          plot_path=args.plot_path,
          max_hand_size=args.max_hand_size,
          max_play=args.max_play,
          max_discard=args.max_discard)

if __name__ == "__main__":
    main()
