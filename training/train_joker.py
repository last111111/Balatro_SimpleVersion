# training/train_joker.py
# -*- coding: utf-8 -*-
"""JokerPPOAgent + 单独小丑牌训练循环 (ChatGPT reward)"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse, math, csv
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from models.joker_agent import JokerSelectNet
from training.ppo_utils import get_device, set_seed, smooth_curve, gae
from envs.joker_env import JokerSelectEnv


class JokerPPOAgent:
    def __init__(self, obs_dim=41, num_actions=25, device='cpu',
                 lr=3e-4, gamma=0.99, gae_lambda=0.95, clip=0.2,
                 vcoef=0.5, ecoef=0.05, epochs=4, mb_size=256,
                 total_updates=1):
        self.device = device
        self.gamma, self.lmbda = gamma, gae_lambda
        self.clip, self.vcoef, self.ecoef = clip, vcoef, ecoef
        self.epochs, self.mb_size = epochs, mb_size

        self.net = JokerSelectNet(obs_dim=obs_dim, num_actions=num_actions).to(device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

        # LR 调度
        warmup_steps = max(1, total_updates // 20)
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup_sched = LinearLR(self.opt, start_factor=0.1, total_iters=warmup_steps)
        cosine_sched = CosineAnnealingLR(self.opt, T_max=max(1, total_updates - warmup_steps), eta_min=lr * 0.1)
        self.scheduler = SequentialLR(self.opt, [warmup_sched, cosine_sched], milestones=[warmup_steps])

    @torch.no_grad()
    def act(self, obs_np, action_mask_np):
        """
        采样动作
        Returns: action, logprob, value, entropy
        """
        self.net.eval()
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask = torch.as_tensor(action_mask_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        logits, value = self.net(obs, mask)

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()

        return (
            int(action.item()),
            float(dist.log_prob(action).item()),
            float(value.item()),
            float(dist.entropy().item())
        )

    def evaluate_actions(self, obs_b, action_b, mask_b):
        """
        训练时对 batch 评估
        Returns: logprob, value, entropy
        """
        logits, values = self.net(obs_b, mask_b)
        dist = torch.distributions.Categorical(logits=logits)
        logprob = dist.log_prob(action_b)
        entropy = dist.entropy()
        return logprob, values, entropy

    def update(self, traj):
        """PPO 更新"""
        obs = torch.as_tensor(np.array(traj["obs"]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.array(traj["action"]), dtype=torch.long, device=self.device)
        masks = torch.as_tensor(np.array(traj["mask"]), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            old_logp = torch.as_tensor(np.array(traj["logp"]), dtype=torch.float32, device=self.device)
            values = torch.as_tensor(np.array(traj["val"]), dtype=torch.float32, device=self.device)
            rewards = np.array(traj["rew"], dtype=np.float32)
            dones = np.array(traj["done"], dtype=np.float32)

            vals_ext = np.concatenate([values.detach().cpu().numpy(), np.array([0.0], dtype=np.float32)], 0)
            rets, adv = gae(rewards, dones, vals_ext, self.gamma, self.lmbda)

            returns = torch.as_tensor(rets, dtype=torch.float32, device=self.device)
            advantages = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            advantages.clamp_(-10.0, 10.0)

        N = obs.size(0)
        idx = np.arange(N)
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for s in range(0, N, self.mb_size):
                mb = idx[s:s + self.mb_size]

                new_logp, new_v, ent = self.evaluate_actions(obs[mb], actions[mb], masks[mb])

                ratio = torch.exp(new_logp - old_logp[mb])
                mb_adv = advantages[mb]
                unclipped = -ratio * mb_adv
                clipped = -torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * mb_adv
                pol_loss = torch.max(unclipped, clipped).mean()

                # 值函数裁剪
                mb_old_v = values[mb].detach()
                mb_ret = returns[mb]
                vclip = 0.2
                v_clipped = mb_old_v + torch.clamp(new_v - mb_old_v, -vclip, vclip)
                v_loss = torch.max((new_v - mb_ret).pow(2), (v_clipped - mb_ret).pow(2)).mean()

                loss = pol_loss + self.vcoef * v_loss - self.ecoef * ent.mean()
                if not torch.isfinite(loss):
                    self.opt.zero_grad()
                    continue

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()


def train_joker_standalone(total_episodes=10000, update_episodes=128, seed=0, lr=3e-4,
                           gamma=0.99, gae_lambda=0.95, clip=0.2, vcoef=0.5,
                           ecoef=0.05, epochs=4, mb_size=256,
                           checkpoint_interval=2000, log_interval=100,
                           use_chatgpt=False, chatgpt_model="gpt-4o-mini"):
    """
    单独训练小丑牌选择 agent

    Args:
        use_chatgpt: 是否使用 ChatGPT 评分作为 reward（否则用随机 reward 验证训练流程）
    """
    set_seed(seed)
    device = get_device()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"joker_run_{timestamp}"

    os.makedirs("outputs/joker/checkpoints", exist_ok=True)
    os.makedirs("outputs/joker/logs", exist_ok=True)
    os.makedirs("outputs/joker/plots", exist_ok=True)

    log_path = f"outputs/joker/logs/{run_name}.csv"
    checkpoint_dir = "outputs/joker/checkpoints"

    print(f"[Init] device={device}, seed={seed}, total_episodes={total_episodes}")
    print(f"[Init] Run name: {run_name}")
    print(f"[Init] use_chatgpt={use_chatgpt}")

    env = JokerSelectEnv()

    total_updates = int(math.ceil(total_episodes / float(update_episodes)))
    agent = JokerPPOAgent(device=device, lr=lr, gamma=gamma, gae_lambda=gae_lambda,
                          clip=clip, vcoef=vcoef, ecoef=ecoef, epochs=epochs,
                          mb_size=mb_size, total_updates=total_updates)

    # ChatGPT reward 函数（可选）
    reward_fn = None
    if use_chatgpt:
        from utils.chatgpt_reward import get_joker_rating
        reward_fn = get_joker_rating

    # CSV 日志
    csv_file = open(log_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['episode', 'reward', 'avg_reward_100', 'num_held'])
    csv_file.flush()

    ep_rewards = []
    updates_done = 0
    pbar = tqdm(total=total_episodes, desc="Joker Training", unit="ep", dynamic_ncols=True)

    try:
        ep = 0
        while ep < total_episodes:
            # 收集 update_episodes 个 episode 的 trajectory
            traj = {"obs": [], "action": [], "mask": [],
                    "logp": [], "val": [], "rew": [], "done": []}

            batch_rewards = []
            for _ in range(update_episodes):
                if ep >= total_episodes:
                    break

                obs = env.reset()
                episode_traj = []
                done = False

                while not done:
                    action_mask = env.get_action_mask()
                    action, logp, val, ent = agent.act(obs, action_mask)
                    next_obs, _, done, info = env.step(action)

                    episode_traj.append({
                        "obs": obs, "action": action, "mask": action_mask,
                        "logp": logp, "val": val
                    })
                    obs = next_obs

                # 计算 episode reward
                held_jokers = env.held
                if reward_fn is not None:
                    rating = reward_fn(held_jokers)
                    ep_reward = (rating - 5.0) / 5.0
                else:
                    # 没有 ChatGPT 时使用简单启发式：持有越多越好 + 多样性奖励
                    ep_reward = len(held_jokers) / 5.0 - 0.5

                # 分配 reward：中间步 reward=0，最后一步 reward=ep_reward
                for i, step in enumerate(episode_traj):
                    traj["obs"].append(step["obs"])
                    traj["action"].append(step["action"])
                    traj["mask"].append(step["mask"])
                    traj["logp"].append(step["logp"])
                    traj["val"].append(step["val"])
                    traj["done"].append(i == len(episode_traj) - 1)
                    traj["rew"].append(ep_reward if i == len(episode_traj) - 1 else 0.0)

                ep_rewards.append(ep_reward)
                batch_rewards.append(ep_reward)
                ep += 1
                pbar.update(1)

                # CSV 日志
                if ep % log_interval == 0:
                    avg_100 = float(np.mean(ep_rewards[-100:])) if len(ep_rewards) >= 100 else float(np.mean(ep_rewards))
                    csv_writer.writerow([ep, ep_reward, avg_100, len(held_jokers)])
                    csv_file.flush()

                # 进度条
                if len(ep_rewards) > 0:
                    recent = float(np.mean(ep_rewards[-min(50, len(ep_rewards)):]))
                    pbar.set_postfix({"avg_50": f"{recent:.3f}", "last": f"{ep_reward:.3f}"})

            # PPO 更新
            if len(traj["obs"]) > 0:
                agent.update(traj)
                updates_done += 1
                agent.scheduler.step()

            # Checkpoint
            if ep % checkpoint_interval < update_episodes:
                ckpt_path = os.path.join(checkpoint_dir, f"{run_name}_ep_{ep}.pt")
                torch.save({
                    "state_dict": agent.net.state_dict(),
                    "optimizer": agent.opt.state_dict(),
                    "episode": ep,
                    "config": {"obs_dim": 41, "num_actions": 25}
                }, ckpt_path)
                print(f"\n[Checkpoint] Saved to {ckpt_path}")

        pbar.close()

    except KeyboardInterrupt:
        pbar.close()
        print("\n[Info] 手动中断")

    finally:
        csv_file.close()

    # 保存最终模型
    final_path = os.path.join(checkpoint_dir, f"{run_name}_final.pt")
    torch.save({
        "state_dict": agent.net.state_dict(),
        "optimizer": agent.opt.state_dict(),
        "episode": ep,
        "config": {"obs_dim": 41, "num_actions": 25}
    }, final_path)
    print(f"[Save] 最终模型已保存到 {final_path}")

    # 画训练曲线
    if len(ep_rewards) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Joker Training - {run_name}', fontsize=14, fontweight='bold')

        window = min(100, max(5, len(ep_rewards) // 5))

        ax1 = axes[0]
        x_raw = np.arange(len(ep_rewards))
        ax1.plot(x_raw, ep_rewards, linewidth=0.5, alpha=0.3, color='blue', label="Raw")
        if window >= 5 and len(ep_rewards) >= window:
            y_smooth = smooth_curve(ep_rewards, window=window)
            x_smooth = np.arange(window - 1, window - 1 + len(y_smooth))
            ax1.plot(x_smooth, y_smooth, linewidth=2.0, color='red', label=f"MA({window})")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.set_title("Joker Selection Reward")
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend()

        ax2 = axes[1]
        if len(ep_rewards) >= 100:
            avg_100 = [np.mean(ep_rewards[max(0, i - 100):i]) for i in range(100, len(ep_rewards) + 1)]
            ax2.plot(range(100, len(ep_rewards) + 1), avg_100, linewidth=2.0, color='green')
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Avg Reward (100 ep)")
        ax2.set_title("Rolling Average Reward")
        ax2.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        plot_path = f"outputs/joker/plots/{run_name}_curve.png"
        plt.savefig(plot_path, dpi=160)
        plt.close('all')
        print(f"[Plot] 训练曲线已保存到 {plot_path}")

    return ep_rewards


def main():
    p = argparse.ArgumentParser(description='Joker Selection PPO Training')
    p.add_argument("--total_episodes", type=int, default=10000)
    p.add_argument("--update_episodes", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--vcoef", type=float, default=0.5)
    p.add_argument("--ecoef", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--mb_size", type=int, default=256)
    p.add_argument("--checkpoint_interval", type=int, default=2000)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--use_chatgpt", action='store_true')
    p.add_argument("--chatgpt_model", type=str, default="gpt-4o-mini")
    args = p.parse_args()

    train_joker_standalone(
        total_episodes=args.total_episodes,
        update_episodes=args.update_episodes,
        seed=args.seed, lr=args.lr, gamma=args.gamma,
        gae_lambda=args.gae_lambda, clip=args.clip,
        vcoef=args.vcoef, ecoef=args.ecoef,
        epochs=args.epochs, mb_size=args.mb_size,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        use_chatgpt=args.use_chatgpt,
        chatgpt_model=args.chatgpt_model
    )


if __name__ == "__main__":
    main()
