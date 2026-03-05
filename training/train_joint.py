# training/train_joint.py
# -*- coding: utf-8 -*-
"""联动训练：小丑牌选择 agent + 打牌 agent"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse, math, csv
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from models.card_agent import ActorCritic
from models.joker_agent import JokerSelectNet
from training.train_card import PPOAgent
from training.train_joker import JokerPPOAgent
from training.ppo_utils import get_device, set_seed, smooth_curve
from envs.joint_env import JointEnv


def train_joint(total_episodes=5000,
                joker_update_episodes=128,
                card_update_steps=4096,
                seed=0,
                card_lr=1e-4,
                joker_lr=3e-4,
                gamma=0.99,
                gae_lambda=0.95,
                clip=0.2,
                vcoef=0.5,
                ecoef=0.03,
                epochs=4,
                card_mb_size=1024,
                joker_mb_size=256,
                card_checkpoint=None,
                joker_checkpoint=None,
                max_hand_size=8,
                max_play=5,
                shaping_beta=1.0,
                checkpoint_interval=1000,
                log_interval=50):
    """
    联动训练

    Args:
        card_checkpoint: 打牌 agent 的预训练 checkpoint 路径
        joker_checkpoint: 小丑牌 agent 的预训练 checkpoint 路径
        card_lr: 打牌 agent 的学习率（较低，微调）
    """
    set_seed(seed)
    device = get_device()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"joint_run_{timestamp}"

    os.makedirs("outputs/joint/checkpoints", exist_ok=True)
    os.makedirs("outputs/joint/logs", exist_ok=True)
    os.makedirs("outputs/joint/plots", exist_ok=True)

    log_path = f"outputs/joint/logs/{run_name}.csv"
    checkpoint_dir = "outputs/joint/checkpoints"

    print(f"[Init] device={device}, seed={seed}, total_episodes={total_episodes}")
    print(f"[Init] Run name: {run_name}")
    print(f"[Init] card_lr={card_lr}, joker_lr={joker_lr}")

    # 创建联动环境
    joint_env = JointEnv(max_hand_size=max_hand_size, max_play=max_play, shaping_beta=shaping_beta)

    obs_dim = joint_env.card_env.observation_space.shape[0]

    # 预估总更新次数
    joker_total_updates = int(math.ceil(total_episodes / float(joker_update_episodes)))
    card_total_updates = total_episodes * 7  # 估算（每 episode 7 轮，每轮约 8 步）

    # 创建 agents
    card_agent = PPOAgent(obs_dim, max_hand_size, device, lr=card_lr,
                          gamma=gamma, gae_lambda=gae_lambda, clip=clip,
                          vcoef=vcoef, ecoef=ecoef, epochs=epochs,
                          mb_size=card_mb_size, total_updates=max(1, card_total_updates // card_update_steps))

    joker_agent = JokerPPOAgent(device=device, lr=joker_lr,
                                gamma=gamma, gae_lambda=gae_lambda,
                                clip=clip, vcoef=vcoef, ecoef=ecoef,
                                epochs=epochs, mb_size=joker_mb_size,
                                total_updates=joker_total_updates)

    # 加载预训练 checkpoint
    if card_checkpoint:
        ckpt = torch.load(card_checkpoint, map_location=device)
        card_agent.net.load_state_dict(ckpt["state_dict"])
        print(f"[Init] Loaded card agent from {card_checkpoint}")

    if joker_checkpoint:
        ckpt = torch.load(joker_checkpoint, map_location=device)
        joker_agent.net.load_state_dict(ckpt["state_dict"])
        print(f"[Init] Loaded joker agent from {joker_checkpoint}")

    # CSV 日志
    csv_file = open(log_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['episode', 'avg_score_7rounds', 'total_score', 'num_jokers',
                         'avg_score_50', 'joker_ids'])
    csv_file.flush()

    # 统计
    ep_total_scores = []      # 每个 episode (7轮) 的总打牌得分
    ep_avg_scores = []         # 每个 episode 的平均每轮得分
    joker_updates_done = 0
    card_steps_collected = 0
    card_traj_buffer = {"obs": [], "a_type": [], "a_mask": [],
                        "logp_type": [], "logp_mask": [],
                        "val": [], "rew": [], "done": []}

    pbar = tqdm(total=total_episodes, desc="Joint Training", unit="ep", dynamic_ncols=True)

    try:
        ep = 0
        joker_traj = {"obs": [], "action": [], "mask": [],
                      "logp": [], "val": [], "rew": [], "done": []}

        while ep < total_episodes:
            # ===== 一个完整 episode: 7 轮 =====
            joker_obs = joint_env.reset()
            episode_scores = []

            for round_idx in range(7):
                # 1. Joker agent 选择小丑牌
                joker_mask = joint_env.get_joker_action_mask()
                joker_action, joker_logp, joker_val, joker_ent = joker_agent.act(joker_obs, joker_mask)

                # 记录 joker trajectory（reward 稍后填入）
                joker_traj["obs"].append(joker_obs)
                joker_traj["action"].append(joker_action)
                joker_traj["mask"].append(joker_mask)
                joker_traj["logp"].append(joker_logp)
                joker_traj["val"].append(joker_val)

                # 2. 执行 joker 选择
                joker_obs, joker_done, joker_info = joint_env.joker_step(joker_action)

                # 3. 打牌 agent 打一局
                card_traj, card_score = joint_env.play_card_episode(card_agent)
                episode_scores.append(card_score)

                # 4. Joker reward = log压缩的平均每次出牌得分
                # 原始分数跨度太大（10~5M），log压缩到合理范围（~1~15）
                avg_score = card_score / joint_env.card_env.max_play
                joker_reward = math.log1p(avg_score)
                joker_traj["rew"].append(joker_reward)
                joker_traj["done"].append(joker_done)

                # 5. 积累打牌 trajectory
                for key in card_traj_buffer:
                    card_traj_buffer[key].extend(card_traj[key])
                card_steps_collected += len(card_traj["obs"])

            # Episode 结束
            total_score = sum(episode_scores)
            avg_score = np.mean(episode_scores)
            ep_total_scores.append(total_score)
            ep_avg_scores.append(avg_score)
            ep += 1
            pbar.update(1)

            # 进度条
            if len(ep_avg_scores) > 0:
                recent = float(np.mean(ep_avg_scores[-min(50, len(ep_avg_scores)):]))
                pbar.set_postfix({
                    "avg_score": f"{avg_score:.0f}",
                    "recent_50": f"{recent:.0f}",
                    "jokers": str(joint_env.held_jokers[:3]) + "..."
                })

            # CSV 日志
            if ep % log_interval == 0:
                avg_50 = float(np.mean(ep_avg_scores[-50:])) if len(ep_avg_scores) >= 50 else float(np.mean(ep_avg_scores))
                csv_writer.writerow([
                    ep, avg_score, total_score, len(joint_env.held_jokers),
                    avg_50, str(joint_env.held_jokers)
                ])
                csv_file.flush()

            # ===== PPO 更新 =====

            # Card agent 更新（每 card_update_steps 步）
            if card_steps_collected >= card_update_steps:
                card_agent.update(card_traj_buffer)
                card_agent.scheduler.step()
                card_traj_buffer = {"obs": [], "a_type": [], "a_mask": [],
                                    "logp_type": [], "logp_mask": [],
                                    "val": [], "rew": [], "done": []}
                card_steps_collected = 0

            # Joker agent 更新（每 joker_update_episodes 个 episode）
            if ep % joker_update_episodes == 0 and len(joker_traj["obs"]) > 0:
                joker_agent.update(joker_traj)
                joker_agent.scheduler.step()
                joker_updates_done += 1
                joker_traj = {"obs": [], "action": [], "mask": [],
                              "logp": [], "val": [], "rew": [], "done": []}

            # Checkpoint
            if ep % checkpoint_interval == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"{run_name}_ep_{ep}.pt")
                torch.save({
                    "card_state_dict": card_agent.net.state_dict(),
                    "card_optimizer": card_agent.opt.state_dict(),
                    "joker_state_dict": joker_agent.net.state_dict(),
                    "joker_optimizer": joker_agent.opt.state_dict(),
                    "episode": ep,
                    "config": {
                        "card_obs_dim": obs_dim,
                        "joker_obs_dim": 41,
                        "max_hand_size": max_hand_size,
                        "max_play": max_play,
                    }
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
        "card_state_dict": card_agent.net.state_dict(),
        "card_optimizer": card_agent.opt.state_dict(),
        "joker_state_dict": joker_agent.net.state_dict(),
        "joker_optimizer": joker_agent.opt.state_dict(),
        "episode": ep,
        "config": {
            "card_obs_dim": obs_dim,
            "joker_obs_dim": 41,
            "max_hand_size": max_hand_size,
            "max_play": max_play,
        }
    }, final_path)
    print(f"[Save] 最终模型已保存到 {final_path}")

    # 画训练曲线（2×2：上排MA清晰曲线，下排Raw参考）
    if len(ep_avg_scores) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Joint Training - {run_name}', fontsize=14, fontweight='bold')

        window = min(100, max(5, len(ep_avg_scores) // 5))
        x_raw = np.arange(len(ep_avg_scores))
        x_raw_t = np.arange(len(ep_total_scores))

        # 上左：Avg Score MA（主要看这个）
        ax_ma1 = axes[0, 0]
        if window >= 5 and len(ep_avg_scores) >= window:
            y_smooth = smooth_curve(ep_avg_scores, window=window)
            x_smooth = np.arange(window - 1, window - 1 + len(y_smooth))
            ax_ma1.plot(x_smooth, y_smooth, linewidth=2.0, color='red', label=f"MA({window})")
            ax_ma1.legend()
        ax_ma1.set_xlabel("Episode")
        ax_ma1.set_ylabel("Avg Score per Round")
        ax_ma1.set_title("Avg Score per Round (Smoothed)")
        ax_ma1.grid(True, linestyle="--", alpha=0.4)

        # 上右：Total Score MA
        ax_ma2 = axes[0, 1]
        if window >= 5 and len(ep_total_scores) >= window:
            y_smooth_t = smooth_curve(ep_total_scores, window=window)
            x_smooth_t = np.arange(window - 1, window - 1 + len(y_smooth_t))
            ax_ma2.plot(x_smooth_t, y_smooth_t, linewidth=2.0, color='red', label=f"MA({window})")
            ax_ma2.legend()
        ax_ma2.set_xlabel("Episode")
        ax_ma2.set_ylabel("Total Score (7 rounds)")
        ax_ma2.set_title("Total Score per Episode (Smoothed)")
        ax_ma2.grid(True, linestyle="--", alpha=0.4)

        # 下左：Avg Score Raw
        ax_raw1 = axes[1, 0]
        ax_raw1.plot(x_raw, ep_avg_scores, linewidth=0.5, alpha=0.4, color='blue', label="Raw")
        ax_raw1.set_xlabel("Episode")
        ax_raw1.set_ylabel("Avg Score per Round")
        ax_raw1.set_title("Avg Score per Round (Raw)")
        ax_raw1.grid(True, linestyle="--", alpha=0.4)
        ax_raw1.legend()

        # 下右：Total Score Raw
        ax_raw2 = axes[1, 1]
        ax_raw2.plot(x_raw_t, ep_total_scores, linewidth=0.5, alpha=0.4, color='green', label="Raw")
        ax_raw2.set_xlabel("Episode")
        ax_raw2.set_ylabel("Total Score (7 rounds)")
        ax_raw2.set_title("Total Score per Episode (Raw)")
        ax_raw2.grid(True, linestyle="--", alpha=0.4)
        ax_raw2.legend()

        plt.tight_layout()
        plot_path = f"outputs/joint/plots/{run_name}_curve.png"
        plt.savefig(plot_path, dpi=160)
        plt.close('all')
        print(f"[Plot] 训练曲线已保存到 {plot_path}")

    return ep_avg_scores


def main():
    p = argparse.ArgumentParser(description='Joint Training: Joker Selection + Card Playing')
    p.add_argument("--total_episodes", type=int, default=5000)
    p.add_argument("--joker_update_episodes", type=int, default=128)
    p.add_argument("--card_update_steps", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--card_lr", type=float, default=1e-4, help='Card agent LR (lower for finetuning)')
    p.add_argument("--joker_lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--vcoef", type=float, default=0.5)
    p.add_argument("--ecoef", type=float, default=0.03)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--card_mb_size", type=int, default=1024)
    p.add_argument("--joker_mb_size", type=int, default=256)
    p.add_argument("--card_checkpoint", type=str, default=None, help='Pre-trained card agent checkpoint')
    p.add_argument("--joker_checkpoint", type=str, default=None, help='Pre-trained joker agent checkpoint')
    p.add_argument("--max_hand_size", type=int, default=8)
    p.add_argument("--max_play", type=int, default=5)
    p.add_argument("--shaping_beta", type=float, default=1.0)
    p.add_argument("--checkpoint_interval", type=int, default=1000)
    p.add_argument("--log_interval", type=int, default=50)
    args = p.parse_args()

    train_joint(
        total_episodes=args.total_episodes,
        joker_update_episodes=args.joker_update_episodes,
        card_update_steps=args.card_update_steps,
        seed=args.seed,
        card_lr=args.card_lr, joker_lr=args.joker_lr,
        gamma=args.gamma, gae_lambda=args.gae_lambda,
        clip=args.clip, vcoef=args.vcoef, ecoef=args.ecoef,
        epochs=args.epochs,
        card_mb_size=args.card_mb_size, joker_mb_size=args.joker_mb_size,
        card_checkpoint=args.card_checkpoint, joker_checkpoint=args.joker_checkpoint,
        max_hand_size=args.max_hand_size, max_play=args.max_play,
        shaping_beta=args.shaping_beta,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()
