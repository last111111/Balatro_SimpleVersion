# models/card_agent.py
# -*- coding: utf-8 -*-
"""打牌 Agent 的网络定义：CardEncoder + ActorCritic"""

import torch
import torch.nn as nn


def orthogonal_init(module, gain=1.0):
    """正交初始化（参考 danzero_plus）"""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class CardEncoder(nn.Module):
    """将 52 维 one-hot 手牌编码为包含 rank/suit 结构信息的稠密向量"""
    def __init__(self, embed_dim=16):
        super().__init__()
        self.rank_embed = nn.Embedding(13, embed_dim // 2)
        self.suit_embed = nn.Embedding(4, embed_dim // 2)
        self.embed_dim = embed_dim
        card_indices = torch.arange(52)
        self.register_buffer('ranks', card_indices // 4)
        self.register_buffer('suits', card_indices % 4)

    def forward(self, hand_onehot):
        """hand_onehot: (B, 52) → (B, embed_dim)"""
        card_feats = torch.cat([
            self.rank_embed(self.ranks),
            self.suit_embed(self.suits)
        ], dim=-1)
        masked = card_feats.unsqueeze(0) * hand_onehot.unsqueeze(-1)
        card_count = hand_onehot.sum(dim=1, keepdim=True).clamp(min=1.0)
        hand_repr = masked.sum(dim=1) / card_count
        return hand_repr


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, max_hand_size: int, hidden=(512, 512),
                 card_embed_dim=16):
        super().__init__()
        self.max_hand_size = max_hand_size
        self.card_embed_dim = card_embed_dim

        self.card_encoder = CardEncoder(embed_dim=card_embed_dim)

        # 共享编码器（3层，512→512→256，Tanh）
        encoder_input_dim = obs_dim + card_embed_dim
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, 512), nn.Tanh(),
            nn.Linear(512, 512), nn.Tanh(),
            nn.Linear(512, 256), nn.Tanh(),
        )

        # 类型决策头（MLP，从单层线性升级为 256→128→2）
        self.type_head = nn.Sequential(
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 2)
        )

        # 出牌专用子网络（2隐层，512→256→52）
        self.play_net = nn.Sequential(
            nn.Linear(256, 512), nn.Tanh(),
            nn.Linear(512, 256), nn.Tanh(),
            nn.Linear(256, 52)
        )

        # 弃牌专用子网络（2隐层，512→256→52）
        self.discard_net = nn.Sequential(
            nn.Linear(256, 512), nn.Tanh(),
            nn.Linear(512, 256), nn.Tanh(),
            nn.Linear(256, 52)
        )

        # 价值头（256→256→256→1，加深一层提高 V(s) 精度）
        self.value_head = nn.Sequential(
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 1)
        )

        # 正交初始化所有层
        self.apply(lambda m: orthogonal_init(m, gain=1.0))
        nn.init.orthogonal_(self.type_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.play_net[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.discard_net[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)

    def forward(self, x):
        hand_onehot = x[:, :52]
        hand_repr = self.card_encoder(hand_onehot)
        z = self.encoder(torch.cat([x, hand_repr], dim=-1))
        return (
            self.type_head(z),
            self.play_net(z),
            self.discard_net(z),
            self.value_head(z).squeeze(-1)
        )
