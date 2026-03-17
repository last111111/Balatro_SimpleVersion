# evaluation/eval_joint.py
# -*- coding: utf-8 -*-
"""联动评估：小丑牌选择 + 打牌的完整 episode 评估"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import glob
import argparse
import torch
import numpy as np

from models.card_agent import (
    ActorCritic, NUM_COMBOS, NUM_ACTIONS,
    combo_idx_to_card_mask, get_valid_action_mask,
)
from models.joker_agent import JokerSelectNet
from envs.joint_env import JointEnv
from utils.visualization import GameVisualizer
from utils.chatgpt_reward import JOKER_DESCRIPTIONS


def load_joint_checkpoint(checkpoint_path=None, checkpoint_dir="outputs/joint/checkpoints"):
    """加载联动训练的 checkpoint"""
    if checkpoint_path is None:
        # 查找最新的 joint checkpoint
        joint_files = glob.glob(os.path.join(checkpoint_dir, "joint_*_final.pt"))
        if not joint_files:
            joint_files = glob.glob(os.path.join(checkpoint_dir, "joint_*.pt"))
        if not joint_files:
            raise FileNotFoundError(f"在 {checkpoint_dir} 中没有找到 joint checkpoint")
        checkpoint_path = max(joint_files, key=os.path.getmtime)

    print(f"加载 joint checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint, checkpoint_path


def evaluate_joint(checkpoint_path=None, num_episodes=3, render=True, seed=None):
    """
    评估联动训练的结果

    每个 episode:
      1. Joker agent 选 7 轮小丑牌
      2. 每轮选完后，Card agent 用当前 joker 组合打一局
      3. 输出每轮的 joker 选择、打牌得分
    """
    checkpoint, ckpt_path = load_joint_checkpoint(checkpoint_path)
    config = checkpoint['config']

    device = torch.device('cpu')

    # 加载打牌模型
    card_model = ActorCritic(config['card_obs_dim'], config.get('max_hand_size', 8)).to(device)
    card_model.load_state_dict(checkpoint['card_state_dict'])
    card_model.eval()

    # 加载小丑牌模型
    joker_model = JokerSelectNet(obs_dim=config.get('joker_obs_dim', 41)).to(device)
    joker_model.load_state_dict(checkpoint['joker_state_dict'])
    joker_model.eval()

    joint_env = JointEnv(
        max_hand_size=config.get('max_hand_size', 8),
        max_play=config.get('max_play', 5),
        shaping_beta=0.0  # 评估时不用 shaping
    )

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    all_total_scores = []

    for ep in range(num_episodes):
        if render:
            GameVisualizer.print_header(f"Joint Episode {ep + 1}/{num_episodes}")

        joker_obs = joint_env.reset()
        episode_scores = []

        for round_idx in range(7):
            # Joker agent 选择
            joker_mask = joint_env.get_joker_action_mask()
            with torch.no_grad():
                obs_t = torch.as_tensor(joker_obs, dtype=torch.float32, device=device).unsqueeze(0)
                mask_t = torch.as_tensor(joker_mask, dtype=torch.float32, device=device).unsqueeze(0)
                logits, _ = joker_model(obs_t, mask_t)
                # 确定性选择（argmax）
                action = int(logits.argmax(dim=1).item())

            joker_obs, joker_done, joker_info = joint_env.joker_step(action)

            if render:
                held = joint_env.held_jokers
                print(f"\n  Round {round_idx + 1}/7 — Joker action: {action}")
                print(f"  Held jokers ({len(held)}): {[JOKER_DESCRIPTIONS.get(j, f'#{j}') for j in held]}")

            # Card agent 打一局（使用简单的 act 包装）
            card_agent_wrapper = _CardAgentWrapper(card_model, device)
            card_traj, card_score = joint_env.play_card_episode(card_agent_wrapper)
            episode_scores.append(card_score)

            if render:
                print(f"  Card score this round: {GameVisualizer.BOLD}{card_score:.0f}{GameVisualizer.RESET}")

        total_score = sum(episode_scores)
        avg_score = np.mean(episode_scores)
        all_total_scores.append(total_score)

        if render:
            print(f"\n{GameVisualizer.BOLD}{GameVisualizer.GREEN}Episode {ep + 1} 结束!{GameVisualizer.RESET}")
            print(f"  Total score (7 rounds): {total_score:.0f}")
            print(f"  Average per round: {avg_score:.0f}")
            print(f"  Round scores: {[f'{s:.0f}' for s in episode_scores]}")
            print()

    # 统计
    if render:
        GameVisualizer.print_header("联动评估统计")
        print(f"  Episodes: {num_episodes}")
        print(f"  Avg total score: {GameVisualizer.BOLD}{np.mean(all_total_scores):.0f}{GameVisualizer.RESET}")
        print(f"  Std: {np.std(all_total_scores):.0f}")
        print(f"  Max: {GameVisualizer.GREEN}{np.max(all_total_scores):.0f}{GameVisualizer.RESET}")
        print(f"  Min: {np.min(all_total_scores):.0f}")
        GameVisualizer.print_separator()

    return all_total_scores


class _CardAgentWrapper:
    """将 ActorCritic model 包装成和 PPOAgent 相同的 .act() 接口 (Categorical 436)"""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def act(self, obs_np):
        x = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_logits, value = self.model(x)

        # Build valid action mask
        hand_indices = list(np.where(obs_np[:52] > 0.5)[0])
        hand_size = len(hand_indices)
        can_discard = obs_np[209] >= 0.01
        valid_mask = get_valid_action_mask(hand_size, can_discard, device=self.device)

        action_logits[0, ~valid_mask] = -1e8

        dist = torch.distributions.Categorical(logits=action_logits[0])
        combo_idx_t = dist.sample()
        combo_idx = int(combo_idx_t.item())
        logprob = float(dist.log_prob(combo_idx_t).item())

        a_type, a_mask = combo_idx_to_card_mask(combo_idx, hand_indices)
        val = float(value.squeeze().item())

        return a_type, a_mask, combo_idx, logprob, val, 0.0


def main():
    parser = argparse.ArgumentParser(description='Joint Evaluation: Joker + Card')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--no-render', action='store_true')
    args = parser.parse_args()

    evaluate_joint(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        render=not args.no_render,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
