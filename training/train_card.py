# training/train_card.py
# -*- coding: utf-8 -*-
"""PPOAgent + 打牌训练循环"""

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

from models.card_agent import ActorCritic
from training.ppo_utils import get_device, set_seed, smooth_curve, gae
from envs.BalatroEnv import BalatroEnv


class PPOAgent:
    def __init__(self, obs_dim, max_hand_size, device,
                 lr=3e-4, gamma=0.99, gae_lambda=0.95, clip=0.2,
                 vcoef=0.5, ecoef=0.05, epochs=4, mb_size=1024,
                 total_updates=1):
        self.device = device
        self.gamma, self.lmbda = gamma, gae_lambda
        self.clip, self.vcoef, self.ecoef = clip, vcoef, ecoef
        self.epochs, self.mb_size = epochs, mb_size
        self.max_hand_size = max_hand_size

        self.net = ActorCritic(obs_dim, max_hand_size).to(device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

        # H6: 学习率调度 — warmup 5% + cosine decay
        warmup_steps = max(1, total_updates // 20)
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup_sched = LinearLR(self.opt, start_factor=0.1, total_iters=warmup_steps)
        cosine_sched = CosineAnnealingLR(self.opt, T_max=max(1, total_updates - warmup_steps), eta_min=lr * 0.1)
        self.scheduler = SequentialLR(self.opt, [warmup_sched, cosine_sched], milestones=[warmup_steps])

        # 弃牌 mask 的损失系数（H3）
        self.coef_sel_dis = 0.3
        self.coef_sel_dis_target = 0.80

        # H5: mask 熵系数
        self.mask_ent_coef = 0.1
        self.mask_ent_coef_start = 0.1
        self.mask_ent_coef_end = 0.02

    @torch.no_grad()
    def act(self, obs_np):
        self.net.eval()
        x = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits_type, logits_sel_play, logits_sel_dis, value = self.net(x)

        # H7: 弃牌次数为 0 时 mask 弃牌选项
        if obs_np[209] < 0.01:
            logits_type[0, 0] = -1e8

        dist_type = torch.distributions.Categorical(logits=logits_type)
        a_type_t = dist_type.sample()
        a_type = int(a_type_t.item())

        hand_mask = torch.as_tensor(obs_np[:52], dtype=torch.float32, device=self.device).unsqueeze(0)
        hand_indices = torch.where(hand_mask[0] > 0.5)[0]

        if len(hand_indices) == 0:
            a_mask = torch.zeros((1, 52), dtype=torch.int64, device=self.device)
            logprob_mask = torch.zeros((1,), device=self.device)
            entropy_mask = torch.zeros((1,), device=self.device)
        else:
            logits_active = logits_sel_play if a_type == 1 else logits_sel_dis
            logits_valid = logits_active[:, hand_indices]
            dist_bern = torch.distributions.Bernoulli(logits=logits_valid)
            sampled = dist_bern.sample()

            if a_type == 1 and sampled.sum() < 1:
                idx = torch.argmax(logits_valid, dim=1)
                sampled[0, idx] = 1.0

            a_mask_52 = torch.zeros((1, 52), dtype=torch.float32, device=self.device)
            a_mask_52[0, hand_indices] = sampled[0]

            a_mask = a_mask_52.squeeze(0).to(torch.int64).detach().cpu().tolist()
            logprob_mask = dist_bern.log_prob(sampled).sum(dim=1)
            entropy_mask = dist_bern.entropy().sum(dim=1)

        logprob_type = dist_type.log_prob(a_type_t)
        entropy_type = dist_type.entropy()
        total_entropy = entropy_type + entropy_mask
        value = value.squeeze(0)

        return (
            a_type, a_mask,
            float(logprob_type.item()), float(logprob_mask.item()),
            float(value.item()), float(total_entropy.item())
        )

    def evaluate_actions(self, obs_b, a_type_b, a_mask_b):
        logits_type, logits_sel_play, logits_sel_dis, values = self.net(obs_b)

        # H7
        no_discard = obs_b[:, 209] < 0.01
        logits_type[no_discard, 0] = -1e8

        dist_type = torch.distributions.Categorical(logits=logits_type)
        logprob_type = dist_type.log_prob(a_type_b)
        entropy_type = dist_type.entropy()

        hand_mask = obs_b[:, :52]
        is_play = (a_type_b == 1).unsqueeze(1)
        logits_active = torch.where(is_play, logits_sel_play, logits_sel_dis)

        logits_masked = logits_active.clone()
        logits_masked[hand_mask < 0.5] = -1e8

        dist_bern = torch.distributions.Bernoulli(logits=logits_masked)
        logprob_mask = (dist_bern.log_prob(a_mask_b.float()) * hand_mask).sum(dim=1)
        entropy_mask = (dist_bern.entropy() * hand_mask).sum(dim=1)

        return logprob_type, logprob_mask, values, entropy_type, entropy_mask

    def update(self, traj):
        obs = torch.as_tensor(np.array(traj["obs"]), dtype=torch.float32, device=self.device)
        a_type = torch.as_tensor(np.array(traj["a_type"]), dtype=torch.long, device=self.device)
        a_mask = torch.as_tensor(np.array(traj["a_mask"]), dtype=torch.long, device=self.device)

        with torch.no_grad():
            old_logp_type = torch.as_tensor(np.array(traj["logp_type"]), dtype=torch.float32, device=self.device)
            old_logp_mask = torch.as_tensor(np.array(traj["logp_mask"]), dtype=torch.float32, device=self.device)
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
                mb_obs, mb_type, mb_mask = obs[mb], a_type[mb], a_mask[mb]
                mb_old_logp_type = old_logp_type[mb]
                mb_old_logp_mask = old_logp_mask[mb]
                mb_ret, mb_adv = returns[mb], advantages[mb]
                mb_old_v = values[mb].detach()

                new_logp_type, new_logp_mask, new_v, ent_type, ent_mask = self.evaluate_actions(
                    mb_obs, mb_type, mb_mask
                )

                ratio_type = torch.exp(new_logp_type - mb_old_logp_type)
                unclipped_t = -ratio_type * mb_adv
                clipped_t = -torch.clamp(ratio_type, 1 - self.clip, 1 + self.clip) * mb_adv
                pol_loss_type = torch.max(unclipped_t, clipped_t).mean()

                ratio_mask = torch.exp(new_logp_mask - mb_old_logp_mask)
                unclipped_m = -ratio_mask * mb_adv
                clipped_m = -torch.clamp(ratio_mask, 1 - self.clip, 1 + self.clip) * mb_adv
                per_sample_m = torch.max(unclipped_m, clipped_m)

                m_play = (mb_type == 1)
                m_dis = (mb_type == 0)

                pol_loss_mask_play = per_sample_m[m_play].mean() if m_play.any() else torch.tensor(0.0, device=self.device)
                pol_loss_mask_dis = per_sample_m[m_dis].mean() if m_dis.any() else torch.tensor(0.0, device=self.device)

                pol_loss = pol_loss_type + 1.0 * pol_loss_mask_play + self.coef_sel_dis * pol_loss_mask_dis

                vclip = 0.2
                v_clipped = mb_old_v + torch.clamp(new_v - mb_old_v, -vclip, vclip)
                v_loss = torch.max((new_v - mb_ret).pow(2), (v_clipped - mb_ret).pow(2)).mean()

                ent = ent_type.mean() + self.mask_ent_coef * ent_mask.mean()

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
          save_path="outputs/card/checkpoints/ppo_balatro_final.pt",
          plot_path="outputs/card/plots/training_curve.png",
          max_hand_size=8, max_play=5,
          shaping_beta=0.8,
          discard_cost=-2.0,
          checkpoint_interval=50000,
          log_interval=100):

    set_seed(seed)
    device = get_device()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_run_{timestamp}"

    os.makedirs("outputs/card/checkpoints", exist_ok=True)
    os.makedirs("outputs/card/logs", exist_ok=True)
    os.makedirs("outputs/card/plots", exist_ok=True)

    checkpoint_dir = "outputs/card/checkpoints"
    log_path = f"outputs/card/logs/{run_name}.csv"
    plot_path_with_name = f"outputs/card/plots/{run_name}_curve.png"

    print(f"[Init] device={device}, seed={seed}, total_steps={total_steps}, update_steps={update_steps}")
    print(f"[Init] Run name: {run_name}")
    print(f"[Init] Checkpoint dir: {checkpoint_dir}")
    print(f"[Init] Log path: {log_path}")

    env = BalatroEnv(max_hand_size=max_hand_size, max_play=max_play, shaping_beta=shaping_beta, discard_cost=discard_cost)
    obs = env.reset()
    obs_dim = env.observation_space.shape[0]

    total_updates = int(math.ceil(total_steps / float(update_steps)))

    agent = PPOAgent(obs_dim, max_hand_size, device, lr, gamma, gae_lambda, clip,
                     vcoef, ecoef_start, epochs, mb_size, total_updates=total_updates)
    updates_done = 0

    def anneal_ecoef():
        frac = min(1.0, updates_done / max(1, total_updates))
        return float(ecoef_start + (ecoef_end - ecoef_start) * frac)

    def anneal_coef_dis():
        warmup_frac = 0.1
        frac = min(1.0, updates_done / max(1, int(total_updates * warmup_frac)))
        return float(0.3 + (agent.coef_sel_dis_target - 0.3) * frac)

    def anneal_mask_ent_coef():
        frac = min(1.0, updates_done / max(1, total_updates))
        return float(agent.mask_ent_coef_start + (agent.mask_ent_coef_end - agent.mask_ent_coef_start) * frac)

    ep_returns_train, ep_ret_train = [], 0.0
    ep_returns_plot, ep_ret_plot = [], 0.0
    ep_actual_scores = []
    ep_play_ratios = []
    ep_play_count, ep_discard_count = 0, 0

    csv_file = open(log_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['episode', 'steps', 'return_train', 'return_plot', 'actual_score',
                         'avg_train_100', 'avg_plot_100', 'avg_score_100', 'max_plot', 'ecoef', 'coef_dis', 'play_ratio'])
    csv_file.flush()

    total_collected = 0
    last_checkpoint_step = 0
    pbar = tqdm(total=total_steps, desc="Training", unit="step", dynamic_ncols=True)

    try:
        while total_collected < total_steps:
            traj = {"obs": [], "a_type": [], "a_mask": [],
                    "logp_type": [], "logp_mask": [],
                    "val": [], "rew": [], "done": []}
            steps = 0
            while steps < update_steps and total_collected < total_steps:
                a_type_val, a_mask_val, logp_type, logp_mask, val, _ = agent.act(obs)
                next_obs, reward, done, _ = env.step((a_type_val, a_mask_val))

                traj["obs"].append(obs)
                traj["a_type"].append(a_type_val)
                traj["a_mask"].append(a_mask_val)
                traj["logp_type"].append(logp_type)
                traj["logp_mask"].append(logp_mask)
                traj["val"].append(val)
                traj["rew"].append(reward)
                traj["done"].append(done)

                ep_ret_train += reward
                if a_type_val == 1:
                    ep_ret_plot += reward
                    ep_play_count += 1
                else:
                    ep_discard_count += 1

                obs = next_obs
                steps += 1
                total_collected += 1

                pbar.update(1)
                if len(ep_returns_train) > 0:
                    recent5_train = float(np.mean(ep_returns_train[-5:]))
                    recent5_plot = float(np.mean(ep_returns_plot[-5:])) if len(ep_returns_plot) > 0 else 0.0
                    recent_pr = float(np.mean(ep_play_ratios[-5:])) if len(ep_play_ratios) > 0 else 0.0
                    pbar.set_postfix({
                        "episodes": len(ep_returns_train),
                        "train_recent5": f"{recent5_train:.1f}",
                        "plot_recent5": f"{recent5_plot:.1f}",
                        "play_ratio": f"{recent_pr:.2f}",
                        "ret_train": f"{ep_ret_train:.1f}",
                        "ret_plot": f"{ep_ret_plot:.1f}",
                        "ecoef": f"{agent.ecoef:.3f}",
                        "coef_dis": f"{agent.coef_sel_dis:.3f}",
                    })
                else:
                    pbar.set_postfix({
                        "episodes": 0,
                        "ret_train": f"{ep_ret_train:.1f}",
                        "ret_plot": f"{ep_ret_plot:.1f}",
                        "ecoef": f"{agent.ecoef:.3f}",
                        "coef_dis": f"{agent.coef_sel_dis:.3f}",
                    })

                if done:
                    ep_returns_train.append(ep_ret_train)
                    ep_returns_plot.append(ep_ret_plot)
                    ep_actual_scores.append(env.cumulative_score)
                    total_actions = ep_play_count + ep_discard_count
                    play_ratio = ep_play_count / max(1, total_actions)
                    ep_play_ratios.append(play_ratio)

                    if len(ep_returns_train) % log_interval == 0:
                        avg_train_100 = float(np.mean(ep_returns_train[-100:])) if len(ep_returns_train) >= 100 else float(np.mean(ep_returns_train))
                        avg_plot_100 = float(np.mean(ep_returns_plot[-100:])) if len(ep_returns_plot) >= 100 else float(np.mean(ep_returns_plot))
                        avg_score_100 = float(np.mean(ep_actual_scores[-100:])) if len(ep_actual_scores) >= 100 else float(np.mean(ep_actual_scores))
                        max_plot = float(np.max(ep_returns_plot)) if len(ep_returns_plot) > 0 else 0.0

                        csv_writer.writerow([
                            len(ep_returns_train), total_collected,
                            ep_ret_train, ep_ret_plot, env.cumulative_score,
                            avg_train_100, avg_plot_100, avg_score_100, max_plot,
                            agent.ecoef, agent.coef_sel_dis, play_ratio
                        ])
                        csv_file.flush()

                    ep_ret_train = 0.0
                    ep_ret_plot = 0.0
                    ep_play_count, ep_discard_count = 0, 0
                    obs = env.reset()

            if total_collected - last_checkpoint_step >= checkpoint_interval:
                checkpoint_path = os.path.join(checkpoint_dir, f"{run_name}_step_{total_collected}.pt")
                pkg = {
                    "state_dict": agent.net.state_dict(),
                    "optimizer": agent.opt.state_dict(),
                    "step": total_collected,
                    "episode": len(ep_returns_train),
                    "config": {
                        "obs_dim": obs_dim, "max_hand_size": max_hand_size,
                        "max_play": max_play, "seed": seed,
                        "gamma": gamma, "gae_lambda": gae_lambda,
                        "clip": clip, "vcoef": vcoef,
                        "ecoef_start": ecoef_start, "ecoef_end": ecoef_end,
                        "shaping_beta": shaping_beta,
                        "discard_cost": discard_cost,
                    }
                }
                torch.save(pkg, checkpoint_path)
                print(f"\n[Checkpoint] Saved to {checkpoint_path}")
                last_checkpoint_step = total_collected

            agent.update(traj)

            updates_done += 1
            agent.ecoef = anneal_ecoef()
            agent.coef_sel_dis = anneal_coef_dis()
            agent.mask_ent_coef = anneal_mask_ent_coef()
            agent.scheduler.step()

        pbar.close()

    except KeyboardInterrupt:
        pbar.close()
        print("\n[Info] 手动中断，保存已训练的权重…")

    finally:
        csv_file.close()

    # 保存最终模型
    final_model_path = os.path.join(checkpoint_dir, f"{run_name}_final.pt")
    pkg = {
        "state_dict": agent.net.state_dict(),
        "optimizer": agent.opt.state_dict(),
        "step": total_collected,
        "episode": len(ep_returns_train),
        "config": {
            "obs_dim": obs_dim, "max_hand_size": max_hand_size,
            "max_play": max_play, "seed": seed,
            "gamma": gamma, "gae_lambda": gae_lambda,
            "clip": clip, "vcoef": vcoef,
            "ecoef_start": ecoef_start, "ecoef_end": ecoef_end,
            "shaping_beta": shaping_beta,
            "discard_cost": discard_cost,
        }
    }
    torch.save(pkg, final_model_path)
    print(f"[Save] 最终模型已保存到 {final_model_path}")

    # 画训练曲线
    if len(ep_returns_train) > 0:
        fig, axes = plt.subplots(3, 2, figsize=(14, 15))
        fig.suptitle(f'PPO Training - {run_name}', fontsize=14, fontweight='bold')

        window = min(100, max(5, len(ep_returns_plot) // 5))

        ax1 = axes[0, 0]
        x_raw = np.arange(len(ep_actual_scores))
        ax1.plot(x_raw, ep_actual_scores, linewidth=0.8, alpha=0.35, color='blue', label="Raw")
        if window >= 5 and len(ep_actual_scores) >= window:
            y_smooth = smooth_curve(ep_actual_scores, window=window)
            x_smooth = np.arange(window - 1, window - 1 + len(y_smooth))
            ax1.plot(x_smooth, y_smooth, linewidth=2.0, color='red', label=f"MA({window})")
        ax1.set_xlabel("Episode"); ax1.set_ylabel("Score (raw points)")
        ax1.set_title("Play Score Per Episode"); ax1.grid(True, linestyle="--", alpha=0.4); ax1.legend()

        ax2 = axes[0, 1]
        x_raw_t = np.arange(len(ep_returns_train))
        ax2.plot(x_raw_t, ep_returns_train, linewidth=0.8, alpha=0.35, color='darkorange', label="Raw")
        if window >= 5 and len(ep_returns_train) >= window:
            y_smooth_t = smooth_curve(ep_returns_train, window=window)
            x_smooth_t = np.arange(window - 1, window - 1 + len(y_smooth_t))
            ax2.plot(x_smooth_t, y_smooth_t, linewidth=2.0, color='red', label=f"MA({window})")
        ax2.set_xlabel("Episode"); ax2.set_ylabel("Total Return")
        ax2.set_title("Training Return Per Episode"); ax2.grid(True, linestyle="--", alpha=0.4); ax2.legend()

        ax3 = axes[1, 0]
        if len(ep_actual_scores) >= 100:
            avg_score_100 = [np.mean(ep_actual_scores[max(0, i - 100):i]) for i in range(100, len(ep_actual_scores) + 1)]
            ax3.plot(range(100, len(ep_actual_scores) + 1), avg_score_100, linewidth=2.0, color='green')
        ax3.set_xlabel("Episode"); ax3.set_ylabel("Avg Score (100 ep)")
        ax3.set_title("Rolling Average Score"); ax3.grid(True, linestyle="--", alpha=0.4)

        ax4 = axes[1, 1]
        if len(ep_actual_scores) > 0:
            max_actual = [np.max(ep_actual_scores[:i + 1]) for i in range(len(ep_actual_scores))]
            ax4.plot(range(len(ep_actual_scores)), max_actual, linewidth=2.0, color='purple', label='Max Score')
        ax4.set_xlabel("Episode"); ax4.set_ylabel("Max Score")
        ax4.set_title("Best Score Over Time"); ax4.grid(True, linestyle="--", alpha=0.4); ax4.legend()

        ax5 = axes[2, 0]
        if len(ep_actual_scores) > 500:
            ax5.hist(ep_actual_scores[:500], bins=30, alpha=0.5, color='blue', label='First 500', density=True)
            ax5.hist(ep_actual_scores[-500:], bins=30, alpha=0.5, color='red', label='Last 500', density=True)
        else:
            ax5.hist(ep_actual_scores, bins=30, alpha=0.7, color='blue', label='All', density=True)
        ax5.set_xlabel("Score"); ax5.set_ylabel("Density")
        ax5.set_title("Score Distribution"); ax5.grid(True, linestyle="--", alpha=0.4); ax5.legend()

        ax6 = axes[2, 1]
        if len(ep_play_ratios) > 0:
            x_pr = np.arange(len(ep_play_ratios))
            ax6.plot(x_pr, ep_play_ratios, linewidth=0.8, alpha=0.35, color='teal', label="Raw")
            if window >= 5 and len(ep_play_ratios) >= window:
                pr_smooth = smooth_curve(ep_play_ratios, window=window)
                x_pr_s = np.arange(window - 1, window - 1 + len(pr_smooth))
                ax6.plot(x_pr_s, pr_smooth, linewidth=2.0, color='darkred', label=f"MA({window})")
            ax6.axhline(y=5/8, color='gray', linestyle='--', alpha=0.5, label='Baseline (5/8)')
        ax6.set_xlabel("Episode"); ax6.set_ylabel("Play Ratio")
        ax6.set_title("Play / (Play+Discard) Ratio"); ax6.set_ylim(0, 1)
        ax6.grid(True, linestyle="--", alpha=0.4); ax6.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(plot_path_with_name, dpi=160)
        print(f"[Plot] 训练曲线已保存到 {plot_path_with_name}")

        plt.figure(figsize=(9, 5))
        x_raw_s = np.arange(len(ep_actual_scores))
        plt.plot(x_raw_s, ep_actual_scores, linewidth=0.8, alpha=0.35, color='darkorange', label="Actual Score")
        if window >= 5 and len(ep_actual_scores) >= window:
            y_smooth_s = smooth_curve(ep_actual_scores, window=window)
            x_smooth_s = np.arange(window - 1, window - 1 + len(y_smooth_s))
            plt.plot(x_smooth_s, y_smooth_s, linewidth=2.0, color='red', label=f"Moving Avg ({window})")
        plt.xlabel("Episode"); plt.ylabel("Actual Score (raw points)")
        plt.title("PPO Training - Actual Play Score")
        plt.grid(True, linestyle="--", alpha=0.4); plt.legend(); plt.tight_layout()
        plt.savefig("outputs/card/plots/training_curve.png", dpi=160)
        plt.close('all')


def main():
    p = argparse.ArgumentParser(description='PPO Training for Balatro Environment')
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
    p.add_argument("--max_hand_size", type=int, default=8)
    p.add_argument("--max_play", type=int, default=5)
    p.add_argument("--shaping_beta", type=float, default=0.8)
    p.add_argument("--discard_cost", type=float, default=-2.0)
    p.add_argument("--checkpoint_interval", type=int, default=50000)
    p.add_argument("--log_interval", type=int, default=100)
    args = p.parse_args()

    train(total_steps=args.total_steps, update_steps=args.update_steps,
          seed=args.seed, lr=args.lr, gamma=args.gamma, gae_lambda=args.gae_lambda,
          clip=args.clip, vcoef=args.value_coef,
          ecoef_start=args.entropy_coef_start, ecoef_end=args.entropy_coef_end,
          epochs=args.epochs, mb_size=args.mb_size,
          max_hand_size=args.max_hand_size, max_play=args.max_play,
          shaping_beta=args.shaping_beta, discard_cost=args.discard_cost,
          checkpoint_interval=args.checkpoint_interval, log_interval=args.log_interval)


if __name__ == "__main__":
    main()
