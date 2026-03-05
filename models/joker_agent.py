# models/joker_agent.py
# -*- coding: utf-8 -*-
"""小丑牌选择网络：JokerSelectNet（位置感知 embedding + 3层512编码器）"""

import torch
import torch.nn as nn
from models.card_agent import orthogonal_init

NUM_JOKER_TYPES = 30


class JokerSelectNet(nn.Module):
    """
    小丑牌选择网络（位置感知版）

    obs (41 维):
      obs[0:30]   = held multi-hot → mean-pool embedding (16)
      obs[30:35]  = held slot IDs / 29 → per-slot embedding (5×16=80)
      obs[35:39]  = offered slot IDs / 29 → per-slot embedding (4×16=64)
      obs[39:41]  = round, held_count (2)

    encoder 输入: 16 + 80 + 64 + 2 = 162
      ↓
    Linear(162, 512) → Tanh → Linear(512, 512) → Tanh → Linear(512, 256) → Tanh
      ↓
    Policy: Linear(256, 25) + action_mask → Categorical
    Value:  Linear(256, 256) → Tanh → Linear(256, 1)
    """

    def __init__(self, obs_dim=41, joker_embed_dim=16, num_actions=25):
        super().__init__()
        self.joker_embed_dim = joker_embed_dim
        self.num_actions = num_actions

        # Joker embedding: 30种 → 16维
        self.joker_embed = nn.Embedding(NUM_JOKER_TYPES, joker_embed_dim)

        # 共享编码器 (3层 512→512→256)
        # 输入: held_repr(16) + offered_embeds(64) + held_pos(80) + scalars(2) = 162
        encoder_input_dim = joker_embed_dim + 4 * joker_embed_dim + 5 * joker_embed_dim + 2
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, 512), nn.Tanh(),
            nn.Linear(512, 512), nn.Tanh(),
            nn.Linear(512, 256), nn.Tanh(),
        )

        # 策略头
        self.policy_head = nn.Linear(256, num_actions)

        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 1)
        )

        # 正交初始化
        self.apply(lambda m: orthogonal_init(m, gain=1.0))
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)

    def _joker_repr(self, obs):
        """从 obs 的 held multi-hot (前30维) 计算 joker embedding mean-pool"""
        held_multihot = obs[:, :NUM_JOKER_TYPES]  # (B, 30)
        all_embeds = self.joker_embed.weight  # (30, 16)
        masked = all_embeds.unsqueeze(0) * held_multihot.unsqueeze(-1)  # (B, 30, 16)
        count = held_multihot.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
        return masked.sum(dim=1) / count  # (B, 16)

    def forward(self, obs, action_mask=None):
        """
        obs: (B, 41)
        action_mask: (B, 25) float, 1=valid, 0=invalid

        Returns: logits (B, 25), value (B,)
        """
        B = obs.size(0)

        # 1. held mean-pool（全局策略感知）
        held_repr = self._joker_repr(obs)  # (B, 16)

        # 2. offered 位置 embedding
        offered_ids = (obs[:, 35:39] * 29).round().long().clamp(0, 29)  # (B, 4)
        offered_embeds = self.joker_embed(offered_ids)  # (B, 4, 16)
        offered_flat = offered_embeds.reshape(B, -1)  # (B, 64)

        # 3. held 位置 embedding（空槽用 zero embedding）
        held_slot_raw = obs[:, 30:35]  # (B, 5)
        held_ids = (held_slot_raw * 29).round().long().clamp(0, 29)  # (B, 5)
        held_valid = (held_slot_raw >= 0).float().unsqueeze(-1)  # (B, 5, 1)
        held_pos_embeds = self.joker_embed(held_ids) * held_valid  # (B, 5, 16)
        held_pos_flat = held_pos_embeds.reshape(B, -1)  # (B, 80)

        # 4. 标量特征
        scalars = obs[:, 39:41]  # (B, 2): round, held_count

        # 5. 拼接 → encoder
        z = self.encoder(torch.cat([held_repr, offered_flat, held_pos_flat, scalars], dim=-1))

        logits = self.policy_head(z)  # (B, 25)

        # 应用 action mask
        if action_mask is not None:
            logits = logits + (1.0 - action_mask) * (-1e8)

        value = self.value_head(z).squeeze(-1)  # (B,)

        return logits, value
