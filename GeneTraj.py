# -*- coding: utf-8 -*-
# 适配“新模型（分离出/弃 mask 头）”的轨迹导出脚本
# 输出字段与老脚本完全一致：episode, t, action, reward, remaining_plays, remaining_discards,
#                            hand_before, cards, deck_remaining, advantage

import argparse, csv, json, os
import numpy as np
import torch
import torch.nn as nn

from envs.BalatroEnv import BalatroEnv  # 保持你的工程结构


# ====== 与“新模型（分离mask头）”一致的网络结构 ======
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, max_hand_size: int, hidden=(512, 512)):
        super().__init__()
        self.max_hand_size = max_hand_size
        layers, last = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.backbone = nn.Sequential(*layers)
        # 新模型：类型头 + 出牌mask头 + 弃牌mask头 + 值函数
        self.logits_type     = nn.Linear(last, 2)                 # 0=discard, 1=play
        self.logits_sel_play = nn.Linear(last, max_hand_size)     # 出牌 mask（Bernoulli）
        self.logits_sel_dis  = nn.Linear(last, max_hand_size)     # 弃牌 mask（Bernoulli）
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


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ====== 扑克牌元组 (rank:int 1-13, suit:'H/D/C/S') 转字符串（保持你之前的格式） ======
def card_tuple_to_str(card):
    rank, suit = card  # e.g. (1,'H')
    # 维持你之前脚本的映射（1->"1", 11->"11", ...）
    rank_map = {1: "1", 11: "11", 12: "12", 13: "13"}
    r = rank_map.get(rank, str(rank))
    s = suit  # 已是 'H','D','C','S'
    return f"{r}{s}"


def list_cards_to_str(cards):
    return [card_tuple_to_str(c) for c in cards]


@torch.no_grad()
def eval_value(net, obs_np, device) -> float:
    x = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
    _, _, _, v = net(x)
    return float(v.squeeze(0).item())


@torch.no_grad()
def policy_act(net, obs_np, device, max_hand_size):
    """
    与“新模型”训练时一致的采样策略：
      - 先采动作类型（0=discard, 1=play）
      - 再根据类型选择对应的 mask 头：play->logits_sel_play，discard->logits_sel_dis
      - 对有效位（当前手牌张数）做 Bernoulli 逐位采样
      - 出牌时保证至少选一张
    返回 (a_type, a_mask)
    """
    x = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
    logits_type, logits_sel_play, logits_sel_dis, _ = net(x)

    # 0=discard, 1=play
    dist_type = torch.distributions.Categorical(logits=logits_type)
    a_type = int(dist_type.sample().item())

    # 依据 obs 前 52 维估算当前手牌张数
    hand_count = int(np.clip(np.round(float(np.asarray(obs_np[:52]).sum())), 0, max_hand_size))
    if hand_count <= 0:
        a_mask = [0] * max_hand_size
    else:
        # 根据类型选择对应头
        logits_active = logits_sel_play if a_type == 1 else logits_sel_dis
        logits_valid = logits_active[:, :hand_count]              # (1,k)
        dist_bern = torch.distributions.Bernoulli(logits=logits_valid)
        sampled = dist_bern.sample()                              # (1,k)

        # 出牌时，保证至少 1 张
        if a_type == 1 and sampled.sum() < 1:
            idx = int(torch.argmax(logits_valid, dim=1).item())
            sampled[0, idx] = 1.0

        pad = torch.zeros((1, max_hand_size - hand_count), device=device)
        mask = torch.cat([sampled, pad], dim=1).to(torch.int64)  # (1, max_hand_size)
        a_mask = mask.squeeze(0).detach().cpu().tolist()

    return a_type, a_mask


def load_model(pt_path, device, env_kwargs):
    """
    从新模型的 .pt 载入：
      - 优先从 pkg['config'] 读 obs_dim / max_hand_size；否则从临时 env 推断 obs_dim
      - state_dict 应包含分离头的权重：logits_type / logits_sel_play / logits_sel_dis / value_head
    """
    pkg = torch.load(pt_path, map_location=device)
    cfg = pkg.get("config", {})

    # obs_dim
    if isinstance(cfg.get("obs_dim"), int):
        obs_dim = cfg["obs_dim"]
    else:
        tmp_env = BalatroEnv(**env_kwargs)
        obs_dim = tmp_env.observation_space.shape[0]

    # max_hand_size
    max_hand_size = cfg.get("max_hand_size", env_kwargs.get("max_hand_size", 8))

    # 构建并加载
    net = ActorCritic(obs_dim, max_hand_size).to(device)
    # 如果你确认是“新模型”的权重，下面严格加载即可；若需兼容旧模型，可用 strict=False
    net.load_state_dict(pkg["state_dict"], strict=True)
    net.eval()
    return net, max_hand_size


def rollout_to_csv_noenvchange(model_path: str,
                               csv_path: str = "traj.csv",
                               episodes: int = 3,
                               gamma: float = 0.99,
                               max_steps_per_ep: int = 1000,
                               max_discard: int = 3,
                               env_kwargs=None):
    """
    不改动 env：
    - step 前读取 env.hand / env.play_count
    - 用 mask 在 hand_before 上选出 cards
    - step 后读取 env.deck / env.play_count
    - remaining_discards 由脚本本地倒数（a_type==0 时减 1）
    - CSV 字段与旧脚本完全一致
    """
    env_kwargs = env_kwargs or {}
    device = get_device()
    net, max_hand_size = load_model(model_path, device, env_kwargs)

    env = BalatroEnv(**env_kwargs)

    fieldnames = [
        "episode", "t", "action", "reward",
        "remaining_plays", "remaining_discards",
        "hand_before", "cards", "deck_remaining", "advantage"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        for ep in range(1, episodes + 1):
            obs = env.reset()
            done = False
            t = 0

            # 本地维护“剩余弃牌次数”（用于 CSV）
            rem_discards = max_discard

            while not done and t < max_steps_per_ep:
                t += 1
                obs_np = np.asarray(obs, dtype=np.float32)

                # 估计 V(s)
                V_s = eval_value(net, obs_np, device)

                # --- 记录动作前的状态 ---
                hand_before = list(env.hand)  # 手牌 [(rank,suit), ...]
                # 采样（新模型：分离 mask 头）
                a_type, a_mask = policy_act(net, obs_np, device, max_hand_size)
                action_name = "play" if a_type == 1 else "discard"

                # 用 mask 在 hand_before 上取子集（保持与 env.update_hand 的 zip 逻辑一致）
                cards_involved = [
                    card for card, m in zip(hand_before, a_mask) if m and len(hand_before) > 0
                ]

                # --- 环境前向 ---
                next_obs, reward, done, _ = env.step((a_type, a_mask))

                # 估计 V(s')
                V_next = 0.0 if done else eval_value(net, next_obs, device)
                advantage = float(reward + gamma * V_next - V_s)

                # step() 后：读取牌库剩余（cnt>0 的牌）
                deck_remaining = [card for card, cnt in env.deck.items() if cnt > 0]

                # 读取剩余出牌次数（env 内部已有）
                remaining_plays = env.play_count

                # 本地维护剩余弃牌次数（与旧脚本一致）
                if a_type == 0:
                    rem_discards = max(0, rem_discards - 1)

                # 写入 CSV（与旧版完全一致的格式）
                writer.writerow({
                    "episode": ep,
                    "t": t,
                    "action": action_name,
                    "reward": float(reward),
                    "remaining_plays": int(remaining_plays),
                    "remaining_discards": int(rem_discards),
                    "hand_before": json.dumps(list_cards_to_str(hand_before), ensure_ascii=False),
                    "cards": json.dumps(list_cards_to_str(cards_involved), ensure_ascii=False),
                    "deck_remaining": json.dumps(list_cards_to_str(deck_remaining), ensure_ascii=False),
                    "advantage": float(advantage),
                })

                obs = next_obs

    print(f"[Save] CSV -> {csv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="ppo_balatro.pt")
    p.add_argument("--csv_path", type=str, default="traj.csv")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--max_steps_per_ep", type=int, default=1000)
    # 本地维护弃牌上限（用于 CSV 的 remaining_discards 字段）
    p.add_argument("--max_discard", type=int, default=3)
    # 如需覆盖 env 的构造参数（不会修改 env 源码）
    p.add_argument("--max_hand_size", type=int, default=8)
    p.add_argument("--max_play", type=int, default=5)
    args = p.parse_args()

    env_kwargs = dict(
        max_hand_size=args.max_hand_size,
        max_play=args.max_play
    )

    rollout_to_csv_noenvchange(
        model_path=args.model_path,
        csv_path=args.csv_path,
        episodes=args.episodes,
        gamma=args.gamma,
        max_steps_per_ep=args.max_steps_per_ep,
        max_discard=args.max_discard,
        env_kwargs=env_kwargs
    )