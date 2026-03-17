# models/card_agent.py
# -*- coding: utf-8 -*-
"""打牌 Agent 的网络定义：CardEncoder + ActorCritic (Categorical 436-dim action space)"""

import torch
import torch.nn as nn
from itertools import combinations


def orthogonal_init(module, gain=1.0):
    """正交初始化（参考 danzero_plus）"""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ════════════════════════════════════════════════════════════════
# Combination enumeration: precompute all C(8,k) for k=1..5
# ════════════════════════════════════════════════════════════════

def _build_combo_table(max_hand_size=8, max_select=5):
    """
    Precompute all subsets of positions {0,...,max_hand_size-1} with size 1..max_select.

    Returns:
        combos: list of tuples, each is a subset of positions (sorted)
                len = C(8,1)+C(8,2)+C(8,3)+C(8,4)+C(8,5) = 218
        combo_masks: LongTensor (218, max_hand_size) — binary masks
    """
    combos = []
    for k in range(1, max_select + 1):
        for c in combinations(range(max_hand_size), k):
            combos.append(c)
    # Build binary masks
    masks = torch.zeros(len(combos), max_hand_size, dtype=torch.long)
    for i, c in enumerate(combos):
        for pos in c:
            masks[i, pos] = 1
    return combos, masks


# Module-level constants
COMBO_TABLE, COMBO_MASKS = _build_combo_table(max_hand_size=8, max_select=5)
NUM_COMBOS = len(COMBO_TABLE)   # 218
NUM_ACTIONS = NUM_COMBOS * 2     # 436: first 218 = play, last 218 = discard
COMBO_MAX_POS = torch.tensor([max(c) for c in COMBO_TABLE], dtype=torch.long)  # (218,)


def combo_idx_to_card_mask(combo_idx, hand_indices_52):
    """
    Convert a combo index (0..435) to a 52-dim card mask.

    Args:
        combo_idx: int, 0..435 (0-217=play, 218-435=discard)
        hand_indices_52: list/array of int, length <= 8, the 52-dim indices
                         of cards currently in hand (ordered by position)

    Returns:
        a_type: int (0=discard, 1=play)
        card_mask_52: list of int, length 52, binary mask
    """
    if combo_idx < NUM_COMBOS:
        a_type = 1  # play
        local_idx = combo_idx
    else:
        a_type = 0  # discard
        local_idx = combo_idx - NUM_COMBOS

    positions = COMBO_TABLE[local_idx]
    card_mask = [0] * 52
    for pos in positions:
        if pos < len(hand_indices_52):
            card_mask[hand_indices_52[pos]] = 1
    return a_type, card_mask


def get_valid_action_mask(hand_size, can_discard=True, device=None):
    """
    Build a boolean mask of shape (436,) for valid actions.

    Args:
        hand_size: int, number of cards currently in hand (0..8)
        can_discard: bool, whether discard actions are available
        device: torch device

    Returns:
        valid_mask: BoolTensor (436,)
    """
    valid = torch.zeros(NUM_ACTIONS, dtype=torch.bool, device=device)
    for i, combo in enumerate(COMBO_TABLE):
        # Check all positions in combo are within hand_size
        if all(pos < hand_size for pos in combo):
            valid[i] = True                        # play combo
            if can_discard:
                valid[i + NUM_COMBOS] = True       # discard combo
    return valid


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

        # 共享编码器（5层，512→512→512→512→256，Tanh，对齐 DanZero+）
        encoder_input_dim = obs_dim + card_embed_dim
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, 512), nn.Tanh(),
            nn.Linear(512, 512), nn.Tanh(),
            nn.Linear(512, 512), nn.Tanh(),
            nn.Linear(512, 512), nn.Tanh(),
            nn.Linear(512, 256), nn.Tanh(),
        )

        # 统一动作头：Categorical(436)
        # 前 218 = play combos, 后 218 = discard combos
        self.action_head = nn.Sequential(
            nn.Linear(256, 512), nn.Tanh(),
            nn.Linear(512, 256), nn.Tanh(),
            nn.Linear(256, NUM_ACTIONS)
        )

        # 价值头（256→256→256→1，加深一层提高 V(s) 精度）
        self.value_head = nn.Sequential(
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 1)
        )

        # 正交初始化所有层
        self.apply(lambda m: orthogonal_init(m, gain=1.0))
        nn.init.orthogonal_(self.action_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)

    def forward(self, x):
        """
        Returns:
            action_logits: (B, 436)
            value: (B,)
        """
        hand_onehot = x[:, :52]
        hand_repr = self.card_encoder(hand_onehot)
        z = self.encoder(torch.cat([x, hand_repr], dim=-1))
        return (
            self.action_head(z),
            self.value_head(z).squeeze(-1)
        )
