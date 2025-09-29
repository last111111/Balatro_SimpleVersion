# PPO_train.py
# -*- coding: utf-8 -*-
import argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from envs.BalatroEnv import BalatroEnv


def smooth_curve(y, window=100):
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0: return y
    w = int(max(1, min(window, len(y))))
    if w == 1: return y
    kernel = np.ones(w, dtype=np.float32) / w
    return np.convolve(y, kernel, mode='valid')


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ================== 模型：分离的出/弃 mask 头 ==================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, max_hand_size: int, hidden=(512, 512)):
        super().__init__()
        self.max_hand_size = max_hand_size
        layers, last = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.backbone = nn.Sequential(*layers)

        self.logits_type     = nn.Linear(last, 2)                 # 0=弃, 1=出
        self.logits_sel_play = nn.Linear(last, max_hand_size)     # 出牌 mask 头
        self.logits_sel_dis  = nn.Linear(last, max_hand_size)     # 弃牌 mask 头
        self.value_head      = nn.Linear(last, 1)

        nn.init.zeros_(self.logits_type.bias)
        nn.init.zeros_(self.logits_sel_play.bias)
        nn.init.zeros_(self.logits_sel_dis.bias)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, x):
        z = self.backbone(x)
        return (
            self.logits_type(z),
            self.logits_sel_play(z),
            self.logits_sel_dis(z),
            self.value_head(z).squeeze(-1)
        )


class PPOAgent:
    def __init__(self, obs_dim, max_hand_size, device,
                 lr=3e-4, gamma=0.99, gae_lambda=0.95, clip=0.2,
                 vcoef=0.5, ecoef=0.05, epochs=4, mb_size=1024):
        self.device = device
        self.gamma, self.lmbda = gamma, gae_lambda
        self.clip, self.vcoef, self.ecoef = clip, vcoef, ecoef
        self.epochs, self.mb_size = epochs, mb_size
        self.max_hand_size = max_hand_size

        self.net = ActorCritic(obs_dim, max_hand_size).to(device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

        # 弃牌 mask 的损失系数（会在训练中渐增）
        self.coef_sel_dis = 0.0
        self.coef_sel_dis_target = 0.10

    @torch.no_grad()
    def act(self, obs_np):
        """
        采样策略：
          - 先采类型头 type(0/1)
          - 再根据类型选择对应的 mask 头（出=logits_sel_play / 弃=logits_sel_dis）
          - 返回：a_type, a_mask, logp_type, logp_mask, value, entropy_total
        """
        self.net.eval()
        x = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits_type, logits_sel_play, logits_sel_dis, value = self.net(x)

        # 类型分布
        dist_type = torch.distributions.Categorical(logits=logits_type)
        a_type_t  = dist_type.sample()    # tensor 0/1
        a_type    = int(a_type_t.item())

        # 当前手牌数（依据 obs 前 52 维手牌 one-hot）
        hand_count = int(np.clip(np.round(float(np.asarray(obs_np[:52]).sum())),
                                 0, self.max_hand_size))

        if hand_count <= 0:
            a_mask = torch.zeros((1, self.max_hand_size), dtype=torch.int64, device=self.device)
            logprob_mask = torch.zeros((1,), device=self.device)
            entropy_mask = torch.zeros((1,), device=self.device)
        else:
            # 选择对应的 mask 头
            logits_active = logits_sel_play if a_type == 1 else logits_sel_dis
            logits_valid  = logits_active[:, :hand_count]
            dist_bern     = torch.distributions.Bernoulli(logits=logits_valid)
            sampled       = dist_bern.sample()  # (1,k)

            # 出牌时，至少 1 张
            if a_type == 1 and sampled.sum() < 1:
                idx = torch.argmax(logits_valid, dim=1)
                sampled[0, idx] = 1.0

            pad = torch.zeros((1, self.max_hand_size - hand_count), device=self.device)
            a_mask_t = torch.cat([sampled, pad], dim=1).to(torch.int64)

            a_mask = a_mask_t.squeeze(0).detach().cpu().tolist()
            logprob_mask = dist_bern.log_prob(sampled).sum(dim=1)   # (1,)
            entropy_mask = dist_bern.entropy().sum(dim=1)           # (1,)

        logprob_type = dist_type.log_prob(a_type_t)   # (1,)
        entropy_type = dist_type.entropy()            # (1,)

        total_entropy = entropy_type + entropy_mask
        value = value.squeeze(0)

        return (
            a_type,
            a_mask,
            float(logprob_type.item()),
            float(logprob_mask.item()),
            float(value.item()),
            float(total_entropy.item())
        )

    def evaluate_actions(self, obs_b, a_type_b, a_mask_b):
        """
        训练时，对一个 batch 评估：
          - 类型头 logp / 熵
          - 按 a_type 每样本选择出/弃对应 mask 头，算 logp_mask / 熵
          - 价值头 v
        """
        logits_type, logits_sel_play, logits_sel_dis, values = self.net(obs_b)
        dist_type = torch.distributions.Categorical(logits=logits_type)
        logprob_type = dist_type.log_prob(a_type_b)    # (B,)
        entropy_type = dist_type.entropy()             # (B,)

        B = obs_b.size(0)
        logprob_mask = torch.zeros(B, device=self.device)
        entropy_mask = torch.zeros(B, device=self.device)

        for i in range(B):
            # 有效位只在当前手牌数上
            k = int(torch.clamp(torch.round(obs_b[i, :52].sum()), 0, self.max_hand_size).item())
            if k > 0:
                if a_type_b[i].item() == 1:
                    logits_valid = logits_sel_play[i, :k]
                else:
                    logits_valid = logits_sel_dis[i, :k]
                dist_bern = torch.distributions.Bernoulli(logits=logits_valid)
                a_mask_i  = a_mask_b[i, :k].float()
                logprob_mask[i] = dist_bern.log_prob(a_mask_i).sum()
                entropy_mask[i] = dist_bern.entropy().sum()

        return logprob_type, logprob_mask, values, entropy_type, entropy_mask

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
        obs   = torch.as_tensor(np.array(traj["obs"]), dtype=torch.float32, device=self.device)
        a_type= torch.as_tensor(np.array(traj["a_type"]), dtype=torch.long, device=self.device)
        a_mask= torch.as_tensor(np.array(traj["a_mask"]), dtype=torch.long, device=self.device)

        with torch.no_grad():
            old_logp_type = torch.as_tensor(np.array(traj["logp_type"]), dtype=torch.float32, device=self.device)
            old_logp_mask = torch.as_tensor(np.array(traj["logp_mask"]), dtype=torch.float32, device=self.device)
            values   = torch.as_tensor(np.array(traj["val"]),  dtype=torch.float32, device=self.device)
            rewards  = np.array(traj["rew"],  dtype=np.float32)
            dones    = np.array(traj["done"], dtype=np.float32)

            # --- GAE ---
            vals_ext = np.concatenate([values.detach().cpu().numpy(), np.array([0.0], dtype=np.float32)], 0)
            rets, adv = self.gae(rewards, dones, vals_ext, self.gamma, self.lmbda)

            returns    = torch.as_tensor(rets, dtype=torch.float32, device=self.device)
            advantages = torch.as_tensor(adv,  dtype=torch.float32, device=self.device)

            # 标准化（避免尺度失衡），轻度裁剪优势
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            returns    = (returns    - returns.mean())    / (returns.std(unbiased=False)    + 1e-8)
            advantages.clamp_(-10.0, 10.0)

        N = obs.size(0)
        idx = np.arange(N)
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for s in range(0, N, self.mb_size):
                mb = idx[s:s + self.mb_size]
                mb_obs, mb_type, mb_mask = obs[mb], a_type[mb], a_mask[mb]
                mb_old_logp_type = old_logp_type[mb]
                mb_old_logp_mask = old_logp_mask[mb]
                mb_ret, mb_adv   = returns[mb], advantages[mb]
                mb_old_v         = values[mb].detach()

                new_logp_type, new_logp_mask, new_v, ent_type, ent_mask = self.evaluate_actions(
                    mb_obs, mb_type, mb_mask
                )

                # ---- 策略损失：类型头 ----
                ratio_type   = torch.exp(new_logp_type - mb_old_logp_type)
                unclipped_t  = -ratio_type * mb_adv
                clipped_t    = -torch.clamp(ratio_type, 1 - self.clip, 1 + self.clip) * mb_adv
                pol_loss_type= torch.max(unclipped_t, clipped_t).mean()

                # ---- 策略损失：mask 头（分开计算 play 与 discard）----
                ratio_mask   = torch.exp(new_logp_mask - mb_old_logp_mask)
                unclipped_m  = -ratio_mask * mb_adv
                clipped_m    = -torch.clamp(ratio_mask, 1 - self.clip, 1 + self.clip) * mb_adv
                per_sample_m = torch.max(unclipped_m, clipped_m)

                m_play = (mb_type == 1)
                m_dis  = (mb_type == 0)

                # 避免除 0：当某类样本在这个 batch 中不存在，将对应均值设为 0
                if m_play.any():
                    pol_loss_mask_play = per_sample_m[m_play].mean()
                else:
                    pol_loss_mask_play = torch.tensor(0.0, device=self.device)

                if m_dis.any():
                    pol_loss_mask_dis  = per_sample_m[m_dis].mean()
                else:
                    pol_loss_mask_dis  = torch.tensor(0.0, device=self.device)

                # 合并：出牌 mask 权重=1.0；弃牌 mask 逐步增至 0.1
                coef_sel_play = 1.0
                coef_sel_dis  = self.coef_sel_dis
                pol_loss = pol_loss_type \
                           + coef_sel_play * pol_loss_mask_play \
                           + coef_sel_dis  * pol_loss_mask_dis

                # ---- 值函数裁剪（PPO-style）----
                vclip = 0.2
                v_clipped = mb_old_v + torch.clamp(new_v - mb_old_v, -vclip, vclip)
                v_loss = torch.max((new_v - mb_ret).pow(2), (v_clipped - mb_ret).pow(2)).mean()

                # ---- 熵（类型 + 小权重的mask 熵）----
                ent = ent_type.mean() + 0.01 * ent_mask.mean()

                loss = pol_loss + self.vcoef * v_loss - self.ecoef * ent
                if not torch.isfinite(loss):
                    self.opt.zero_grad()
                    continue

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()


def train(total_steps=200_000, update_steps=4096, seed=0, lr=3e-4,
          gamma=0.99, gae_lambda=0.95, clip=0.2, vcoef=0.5,
          ecoef_start=0.05, ecoef_end=0.01,
          epochs=4, mb_size=1024,
          save_path="ppo_balatro.pt", plot_path="training_curve.png",
          # Env 超参
          max_hand_size=8, max_play=5,
          shaping_beta=1.0):

    set_seed(seed)
    device = get_device()
    print(f"[Init] device={device}, seed={seed}, total_steps={total_steps}, update_steps={update_steps}")

    env = BalatroEnv(
        max_hand_size=max_hand_size,
        max_play=max_play,
        shaping_beta=shaping_beta
    )
    obs = env.reset()
    obs_dim = env.observation_space.shape[0]

    agent = PPOAgent(obs_dim, max_hand_size, device, lr, gamma, gae_lambda, clip,
                     vcoef, ecoef_start, epochs, mb_size)

    # 总更新次数，用于退火
    total_updates = int(math.ceil(total_steps / float(update_steps)))
    updates_done = 0

    # 熵系数退火
    def anneal_ecoef():
        frac = min(1.0, updates_done / max(1, total_updates))
        return float(ecoef_start + (ecoef_end - ecoef_start) * frac)

    # 弃牌 mask 系数退火（建议 20% 更新内线性拉到 0.1）
    def anneal_coef_dis():
        warmup_frac = 0.2
        frac = min(1.0, updates_done / max(1, int(total_updates * warmup_frac)))
        return float(agent.coef_sel_dis_target * frac)

    # 统计：训练真实回报（含弃牌塑形）vs 仅统计出牌回报（画图）
    ep_returns_train, ep_ret_train = [], 0.0
    ep_returns_plot,  ep_ret_plot  = [], 0.0

    total_collected = 0
    pbar = tqdm(total=total_steps, desc="Training", unit="step", dynamic_ncols=True)

    try:
        while total_collected < total_steps:
            traj = {"obs": [], "a_type": [], "a_mask": [],
                    "logp_type": [], "logp_mask": [],
                    "val": [], "rew": [], "done": []}
            steps = 0
            while steps < update_steps and total_collected < total_steps:
                a_type, a_mask, logp_type, logp_mask, val, _ = agent.act(obs)
                next_obs, reward, done, _ = env.step((a_type, a_mask))

                # 训练轨迹
                traj["obs"].append(obs)
                traj["a_type"].append(a_type)
                traj["a_mask"].append(a_mask)
                traj["logp_type"].append(logp_type)
                traj["logp_mask"].append(logp_mask)
                traj["val"].append(val)
                traj["rew"].append(reward)
                traj["done"].append(done)

                # 统计：训练总回报（含弃牌塑形）
                ep_ret_train += reward
                # 统计：只记录出牌的 reward（画图）
                if a_type == 1:
                    ep_ret_plot += reward

                obs = next_obs
                steps += 1
                total_collected += 1

                pbar.update(1)
                if len(ep_returns_train) > 0:
                    recent5_train = float(np.mean(ep_returns_train[-5:]))
                    recent5_plot  = float(np.mean(ep_returns_plot[-5:])) if len(ep_returns_plot)>0 else 0.0
                    pbar.set_postfix({
                        "episodes": len(ep_returns_train),
                        "train_recent5": f"{recent5_train:.1f}",
                        "plot_recent5":  f"{recent5_plot:.1f}",
                        "ret_train":     f"{ep_ret_train:.1f}",
                        "ret_plot":      f"{ep_ret_plot:.1f}",
                        "ecoef":         f"{agent.ecoef:.3f}",
                        "coef_dis":      f"{agent.coef_sel_dis:.3f}",
                    })
                else:
                    pbar.set_postfix({
                        "episodes": 0,
                        "ret_train": f"{ep_ret_train:.1f}",
                        "ret_plot":  f"{ep_ret_plot:.1f}",
                        "ecoef":     f"{agent.ecoef:.3f}",
                        "coef_dis":  f"{agent.coef_sel_dis:.3f}",
                    })

                if done:
                    ep_returns_train.append(ep_ret_train)
                    ep_returns_plot.append(ep_ret_plot)
                    ep_ret_train = 0.0
                    ep_ret_plot  = 0.0
                    obs = env.reset()

            # PPO 更新
            agent.update(traj)

            # 退火：熵系数 & 弃牌 mask 系数
            updates_done += 1
            agent.ecoef = anneal_ecoef()
            agent.coef_sel_dis = anneal_coef_dis()

        pbar.close()

    except KeyboardInterrupt:
        pbar.close()
        print("\n[Info] 手动中断，保存已训练的权重…")

    # 保存模型（结构与从前一致；仅多存 shaping_beta 以便复现）
    pkg = {
        "state_dict": agent.net.state_dict(),
        "config": {
            "obs_dim": obs_dim,
            "max_hand_size": max_hand_size,
            "max_play": max_play,
            "seed": seed,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip": clip,
            "vcoef": vcoef,
            "ecoef_start": ecoef_start,
            "ecoef_end": ecoef_end,
            "shaping_beta": shaping_beta,
        }
    }
    torch.save(pkg, save_path)
    print(f"[Save] 模型已保存到 {save_path}")

    # 画两条曲线：训练总回报（含弃牌塑形）与只统计出牌的回报
    if len(ep_returns_train) > 0:
        plt.figure(figsize=(9, 5))
        x_raw = np.arange(len(ep_returns_train))
        plt.plot(x_raw, ep_returns_train, linewidth=0.8, alpha=0.35, label="Episode Return (train, incl. discard shaping)")
        x_raw2 = np.arange(len(ep_returns_plot))
        plt.plot(x_raw2, ep_returns_plot,  linewidth=1.6, alpha=0.9,  label="Episode Return (plot, only play rewards)")

        window = min(100, max(5, len(ep_returns_plot)//5))
        if window >= 5:
            y_smooth = smooth_curve(ep_returns_plot, window=window)
            x_smooth = np.arange(window - 1, window - 1 + len(y_smooth))
            plt.plot(x_smooth, y_smooth, linewidth=2.0, label=f"Moving Avg (only play rewards, window={window})")

        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("PPO Training Curve (train vs. only-play-rewards)")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=160)
        print(f"[Plot] 训练曲线已保存到 {plot_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total_steps", type=int, default=1_000_000)
    p.add_argument("--update_steps", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--entropy_coef_start", type=float, default=0.05)
    p.add_argument("--entropy_coef_end", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--mb_size", type=int, default=1024)
    p.add_argument("--save_path", type=str, default="ppo_balatro.pt")
    p.add_argument("--plot_path", type=str, default="training_curve.png")

    # Env 参数：只保留 shaping_beta
    p.add_argument("--max_hand_size", type=int, default=8)
    p.add_argument("--max_play", type=int, default=5)
    p.add_argument("--shaping_beta", type=float, default=1.0)

    args = p.parse_args()

    train(total_steps=args.total_steps,
          update_steps=args.update_steps,
          seed=args.seed,
          lr=args.lr,
          gamma=args.gamma,
          gae_lambda=args.gae_lambda,
          clip=args.clip,
          vcoef=args.value_coef,
          ecoef_start=args.entropy_coef_start,
          ecoef_end=args.entropy_coef_end,
          epochs=args.epochs,
          mb_size=args.mb_size,
          save_path=args.save_path,
          plot_path=args.plot_path,
          max_hand_size=args.max_hand_size,
          max_play=args.max_play,
          shaping_beta=args.shaping_beta)


if __name__ == "__main__":
    main()