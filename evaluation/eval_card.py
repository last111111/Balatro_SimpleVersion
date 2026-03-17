# evaluation/eval_card.py
# -*- coding: utf-8 -*-
"""评估训练好的PPO模型，可视化展示游戏过程"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import glob
import argparse
import torch
import numpy as np
from envs.BalatroEnv import BalatroEnv
from models.card_agent import (
    ActorCritic, NUM_COMBOS, NUM_ACTIONS,
    combo_idx_to_card_mask, get_valid_action_mask,
)
from utils.visualization import GameVisualizer


def load_latest_checkpoint(checkpoint_dir="outputs/card/checkpoints"):
    """加载最新的checkpoint，也兼容旧路径 outputs/checkpoints/"""
    search_dirs = [checkpoint_dir, "outputs/checkpoints"]
    all_files = []
    for d in search_dirs:
        all_files.extend(glob.glob(os.path.join(d, "*_final.pt")))
        all_files.extend(glob.glob(os.path.join(d, "*_step_*.pt")))
    # 也检查根目录的 ppo_balatro.pt
    if os.path.isfile("ppo_balatro.pt"):
        all_files.append("ppo_balatro.pt")

    # 过滤掉 joker/joint 的 checkpoint
    all_files = [f for f in all_files if 'joker_' not in os.path.basename(f) and 'joint_' not in os.path.basename(f)]

    if not all_files:
        raise FileNotFoundError(f"在 {search_dirs} 中没有找到 card checkpoint 文件")

    latest_file = max(all_files, key=os.path.getmtime)
    print(f"加载checkpoint: {latest_file}")

    checkpoint = torch.load(latest_file, map_location='cpu')
    return checkpoint, latest_file


def get_hand_name(env, cards):
    """L5: 使用 _calculate_best_score(return_hand_type=True) 获取牌型名称"""
    if not cards:
        return "无牌"
    try:
        score, hand_type = env._calculate_best_score(cards, return_hand_type=True)
        return hand_type if hand_type else "高牌"
    except:
        return "未知牌型"


def evaluate_model(checkpoint_path=None, num_episodes=5, render=True, seed=None):
    """评估模型并可视化游戏过程"""

    if checkpoint_path is None:
        checkpoint, checkpoint_path = load_latest_checkpoint()
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

    config = checkpoint['config']

    print(f"\n{GameVisualizer.BOLD}模型信息:{GameVisualizer.RESET}")
    print(f"  训练步数: {checkpoint.get('step', 'Unknown')}")
    print(f"  完成Episode: {checkpoint.get('episode', 'Unknown')}")
    print(f"  观测维度: {config['obs_dim']}")
    print(f"  最大手牌数: {config['max_hand_size']}")
    print(f"  最大出牌次数: {config['max_play']}")

    env = BalatroEnv(
        max_hand_size=config['max_hand_size'],
        max_play=config['max_play'],
        shaping_beta=0.0
    )

    device = torch.device('cpu')
    model = ActorCritic(config['obs_dim'], config['max_hand_size']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    episode_rewards = []

    for ep in range(num_episodes):
        GameVisualizer.print_header(f"Episode {ep + 1}/{num_episodes}")

        obs = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0

        while not done:
            step_count += 1

            if render:
                GameVisualizer.print_game_state(env, step_count, total_reward)

            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action_logits, value = model(obs_tensor)

                # Build valid action mask
                hand_indices = list(np.where(obs[:52] > 0.5)[0])
                hand_size = len(hand_indices)
                can_discard = obs[209] >= 0.01
                valid_mask = get_valid_action_mask(hand_size, can_discard, device=device)

                action_logits[0, ~valid_mask] = -1e8

                # Deterministic: argmax for evaluation
                combo_idx = int(action_logits[0].argmax().item())
                action_type, action_mask = combo_idx_to_card_mask(combo_idx, hand_indices)

            selected_cards = []
            current_hand = env.hand.copy()
            for card in current_hand:
                card_idx = env._card_index(card)
                if card_idx < len(action_mask) and action_mask[card_idx] == 1:
                    selected_cards.append(card)

            next_obs, reward, done, info = env.step((action_type, action_mask))
            total_reward += reward

            hand_name = None
            if action_type == 1 and selected_cards:
                hand_name = get_hand_name(env, selected_cards)

            if render:
                GameVisualizer.print_action(action_type, selected_cards, reward, hand_name)

            obs = next_obs

        episode_rewards.append(total_reward)

        if render:
            print(f"\n{GameVisualizer.BOLD}{GameVisualizer.GREEN}Episode {ep + 1} 结束!{GameVisualizer.RESET}")
            print(f"  总步数: {step_count}")
            print(f"  总奖励: {GameVisualizer.BOLD}{total_reward:.1f}{GameVisualizer.RESET}")
            print()

    GameVisualizer.print_header("评估统计")
    print(f"  Episodes数量: {num_episodes}")
    print(f"  平均奖励: {GameVisualizer.BOLD}{np.mean(episode_rewards):.2f}{GameVisualizer.RESET}")
    print(f"  标准差: {np.std(episode_rewards):.2f}")
    print(f"  最高奖励: {GameVisualizer.GREEN}{np.max(episode_rewards):.2f}{GameVisualizer.RESET}")
    print(f"  最低奖励: {np.min(episode_rewards):.2f}")
    print(f"  所有奖励: {[f'{r:.1f}' for r in episode_rewards]}")
    GameVisualizer.print_separator()

    return episode_rewards


def main():
    parser = argparse.ArgumentParser(description='评估PPO模型并可视化游戏过程')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint文件路径（默认使用最新的）')
    parser.add_argument('--episodes', type=int, default=5,
                        help='评估的episode数量')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子')
    parser.add_argument('--no-render', action='store_true',
                        help='不显示详细过程，只显示统计')

    args = parser.parse_args()

    evaluate_model(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        render=not args.no_render,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
