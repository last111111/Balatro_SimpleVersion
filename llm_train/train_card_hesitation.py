# llm_train/train_card_hesitation.py
# -*- coding: utf-8 -*-
"""
Hesitation-Gated LLM Prior for Card Play/Discard PPO
=====================================================

Paper Eq. 4-6 implementation for card agent (Categorical 436-dim action space):
  - LLM Action Prior: N-vote empirical distribution via vLLM (local)
  - Hesitation Gate: g(s) = 1 if h(s) < tau (uncertain -> query LLM)
  - Modified PPO loss: L = clip + vcoef*v_loss - ecoef*H + alpha*g*KL(pi||p_LLM)

LLM: Qwen3-32B via vLLM (本地加载，不需要 API server)
Colab A100 quick start:
  !pip install vllm torch numpy matplotlib tqdm gym
  !git clone https://github.com/YOUR_REPO/Balatro_SimpleVersion.git /content/Balatro_SimpleVersion

  import sys; sys.path.insert(0, '/content/Balatro_SimpleVersion')
  from llm_train.train_card_hesitation import train_card_hesitation

  train_card_hesitation(
      llm_model="Qwen/Qwen3-32B",
      total_steps=50000,
      gpu_memory_utilization=0.85,  # A100-80GB: Qwen3-32B needs ~65GB
  )
"""

import os
import sys
import re
import math
import csv
import json
import time
import argparse
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# ── Path setup ──────────────────────────────────────────────────
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.card_agent import (
    ActorCritic, NUM_COMBOS, NUM_ACTIONS, COMBO_TABLE, COMBO_MAX_POS,
    combo_idx_to_card_mask, get_valid_action_mask,
)
from training.ppo_utils import get_device, set_seed, smooth_curve, gae
from envs.BalatroEnv import BalatroEnv

# Colab detection
if 'google.colab' in sys.modules:
    matplotlib.use('module://matplotlib_inline.backend_inline')

RANK_NAMES = {1: 'A', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
              8: '8', 9: '9', 10: '10', 11: 'J', 12: 'Q', 13: 'K'}
SUIT_SYMBOLS = {'H': '\u2665', 'D': '\u2666', 'C': '\u2663', 'S': '\u2660'}
SUIT_LIST = ['H', 'D', 'C', 'S']


# ════════════════════════════════════════════════════════════════
# Helper: card <-> string conversion
# ════════════════════════════════════════════════════════════════

def _card_to_str(rank, suit):
    """(rank, suit) -> 'A\u2660', '10\u2665' etc."""
    return f"{RANK_NAMES[rank]}{SUIT_SYMBOLS[suit]}"


def _index_to_card(idx):
    """0-51 index -> (rank, suit)"""
    rank = idx // 4 + 1
    suit = SUIT_LIST[idx % 4]
    return (rank, suit)


def _card_index(rank, suit):
    """(rank, suit) -> 0-51 index"""
    suit_map = {'H': 0, 'D': 1, 'C': 2, 'S': 3}
    return (rank - 1) * 4 + suit_map[suit]


def _parse_card_str(s):
    """Parse 'A\u2660' -> (rank, suit) or None"""
    s = s.strip()
    if not s:
        return None
    sym_to_suit = {'\u2665': 'H', '\u2666': 'D', '\u2663': 'C', '\u2660': 'S',
                   'H': 'H', 'D': 'D', 'C': 'C', 'S': 'S'}
    name_to_rank = {'A': 1, 'J': 11, 'Q': 12, 'K': 13}
    suit_char = s[-1]
    suit = sym_to_suit.get(suit_char)
    if suit is None:
        return None
    rank_str = s[:-1]
    if rank_str in name_to_rank:
        rank = name_to_rank[rank_str]
    elif rank_str.isdigit() and 1 <= int(rank_str) <= 13:
        rank = int(rank_str)
    else:
        return None
    return (rank, suit)


def _get_hand_indices(obs_np):
    """Extract sorted list of 52-dim card indices that are in hand."""
    return list(np.where(obs_np[:52] > 0.5)[0])


def _card_selection_to_combo_idx(a_type, card_indices_52, hand_indices_52):
    """
    Map LLM's (action_type, selected_card_indices) to a combo_idx (0..435).

    Args:
        a_type: int, 0=discard, 1=play
        card_indices_52: list of int, selected card indices in 52-dim space
        hand_indices_52: list of int, hand card indices in 52-dim space (ordered)

    Returns:
        combo_idx: int (0..435) or None if no match
    """
    # Convert 52-dim indices to hand positions (0..7)
    idx_to_pos = {ci: pos for pos, ci in enumerate(hand_indices_52)}
    positions = sorted([idx_to_pos[ci] for ci in card_indices_52 if ci in idx_to_pos])
    if not positions:
        return None
    positions_tuple = tuple(positions)

    # Find in COMBO_TABLE
    for i, combo in enumerate(COMBO_TABLE):
        if combo == positions_tuple:
            if a_type == 1:
                return i  # play
            else:
                return i + NUM_COMBOS  # discard
    return None


# ════════════════════════════════════════════════════════════════
# 1. CardLLMActionPrior -- 本地 vLLM 推理
# ════════════════════════════════════════════════════════════════

LLM_SYSTEM_PROMPT = """\
You are an expert Balatro card game player. Given a hand, reply with EXACTLY one line:
PLAY card1 card2 ...
or
DISCARD card1 card2 ...

Game rules:
- Standard 52-card deck (ranks A,2-10,J,Q,K; suits \u2665\u2666\u2663\u2660). You hold 8 cards.
- Each round you have 5 play actions and 3 discard actions.
- PLAY: select 1-5 cards from your hand to score. Score = base_chips x base_mult + card chip bonuses.
- DISCARD: select cards you don't want, they are removed and you draw replacements from the deck.
- Goal: maximize total score across all plays.

Scoring -- hand types (base_chips x base_mult):
  High Card:        5 x 1 =     5   (any single card)
  Pair:            10 x 2 =    20   (two cards of same rank)
  Two Pair:        20 x 2 =    40   (two different pairs)
  Three of a Kind: 30 x 3 =    90   (three cards of same rank)
  Straight:        30 x 4 =   120   (5 cards with consecutive ranks, e.g. 5-6-7-8-9)
  Flush:           35 x 4 =   140   (5 cards of the same suit)
  Full House:      40 x 4 =   160   (three of a kind + a pair)
  Four of a Kind:  60 x 7 =   420   (four cards of same rank)
  Straight Flush: 100 x 8 =   800   (straight + flush combined)

Card chip values (added to base chips for each scored card):
  A=11, K=10, Q=10, J=10, 10=10, 9=9, 8=8, 7=7, 6=6, 5=5, 4=4, 3=3, 2=2

Strategy tips:
- Four of a Kind (420+) and Straight Flush (800+) are extremely powerful. Prioritize building toward them.
- Flush (140) and Straight (120) are strong; Full House (160) is better if achievable.
- Pair (20) and High Card (5) are weak. Discard to improve if discards remain.
- When discards remain and your hand is weak, DISCARD low-value cards to draw better ones.
- When plays are limited or you already have a strong hand, PLAY immediately.
- High-chip cards (A, K, Q, J, 10) contribute more chips when scored.
- Playing fewer but better cards is sometimes optimal (e.g. PLAY a Pair rather than a weak 5-card hand).

Example 1:
Hand: A\u2660, A\u2665, K\u2666, 10\u2663, 9\u2663, 7\u2666, 5\u2665, 3\u2663 | Plays: 4/5, Discards: 2/3
PLAY A\u2660 A\u2665

Example 2:
Hand: Q\u2663, J\u2663, 8\u2663, 7\u2666, 6\u2660, 4\u2665, 3\u2663, 2\u2666 | Plays: 3/5, Discards: 3/3
DISCARD 7\u2666 6\u2660 4\u2665 2\u2666

Example 3:
Hand: K\u2665, K\u2666, K\u2663, 9\u2660, 8\u2665, 5\u2666, 4\u2663, 2\u2660 | Plays: 5/5, Discards: 0/3
PLAY K\u2665 K\u2666 K\u2663

IMPORTANT: Output ONLY the action line. No explanation, no analysis."""


class CardLLMActionPrior:
    """Queries LLM N times per state to form empirical card action prior (436-dim)."""

    def __init__(self, model_name="Qwen/Qwen3-32B", num_votes=5,
                 temperature=0.7, cache_maxsize=2048,
                 gpu_memory_utilization=0.85, llm_log_path=None,
                 quantization=None):
        self.model_name = model_name
        self.num_votes = num_votes
        self.temperature = temperature

        # Local vLLM loading
        from vllm import LLM, SamplingParams
        quant_info = f", quantization={quantization}" if quantization else ""
        print(f"[LLM] Loading {model_name} via vLLM (local{quant_info})...")
        llm_kwargs = dict(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
            max_model_len=4096,
        )
        if quantization:
            llm_kwargs["quantization"] = quantization
        self.llm = LLM(**llm_kwargs)
        self.SamplingParams = SamplingParams
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=300,
            stop=["<|im_end|>"],
        )
        print(f"[LLM] Model loaded successfully.")

        # LRU cache
        self._cache = OrderedDict()
        self._cache_maxsize = cache_maxsize
        self._total_queries = 0
        self._cache_hits = 0
        self._valid_votes_history = []
        self._valid_votes_max = 500

        # JSONL logger
        self._llm_log_file = None
        self._llm_log_count = 0
        self._llm_log_flush_interval = 50
        if llm_log_path:
            os.makedirs(os.path.dirname(llm_log_path), exist_ok=True)
            self._llm_log_file = open(llm_log_path, 'a')
            print(f"[LLM] Response log -> {llm_log_path}")

    # ── Prompt construction ───────────────────────────────────
    def _build_prompt(self, env):
        """Build natural language prompt from BalatroEnv state."""
        hand = env.hand
        hand_strs = [_card_to_str(r, s) for r, s in hand]
        plays_left = env.play_count
        discards_left = env.discard_count
        score = env.cumulative_score
        deck_remaining = sum(env.deck.values())

        lines = []
        lines.append(f"Hand ({len(hand)} cards): {', '.join(hand_strs)}")
        lines.append(f"Plays remaining: {plays_left}/{env.max_play}, "
                     f"Discards remaining: {discards_left}/{env.max_discard}")
        lines.append(f"Score so far: {score:.0f}, Deck: {deck_remaining} cards left")
        lines.append("")
        lines.append("Choose your action:")
        return "\n".join(lines)

    # ── Cache key ─────────────────────────────────────────────
    def _state_key(self, env):
        hand_key = tuple(sorted(env.hand))
        return (hand_key, env.play_count, env.discard_count)

    # ── Parse LLM response ───────────────────────────────────
    def _parse_response(self, text, hand):
        """
        Parse "PLAY A\u2660 K\u2665 ..." or "DISCARD 3\u2663 5\u2665 ..." -> (type_int, card_indices)
        Returns (None, None) if parse fails.
        """
        text = text.strip()
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        action_type = None
        card_part = None

        for line in text.split('\n'):
            line = line.strip().strip('`').strip('*').strip()
            upper = line.upper()
            if upper.startswith("PLAY ") or upper == "PLAY":
                action_type = 1
                card_part = line[4:].strip()
                break
            elif upper.startswith("DISCARD ") or upper == "DISCARD":
                action_type = 0
                card_part = line[7:].strip()
                break

        if action_type is None:
            m = re.search(
                r'\b(PLAY|DISCARD)\s+((?:[AKQJ2-9]|10)[\u2665\u2666\u2663\u2660HDCS](?:\s+(?:[AKQJ2-9]|10)[\u2665\u2666\u2663\u2660HDCS])*)',
                text, re.IGNORECASE
            )
            if m:
                action_type = 1 if m.group(1).upper() == "PLAY" else 0
                card_part = m.group(2).strip()

        if action_type is None or card_part is None:
            return None, None

        hand_set = set(hand)
        tokens = re.split(r'[,\s]+', card_part)
        card_indices = []
        for tok in tokens:
            parsed = _parse_card_str(tok)
            if parsed and parsed in hand_set:
                card_indices.append(_card_index(parsed[0], parsed[1]))

        if not card_indices:
            return None, None

        return action_type, card_indices

    # ── N-vote prior ──────────────────────────────────────────
    def get_prior(self, env):
        """
        Query LLM N times, return empirical action distribution over 436 combo actions.

        Returns:
            p_combo: np.ndarray (436,) -- probability per combo action with Laplace smoothing
        """
        key = self._state_key(env)

        # Cache hit
        if key in self._cache:
            self._cache_hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]

        prompt = self._build_prompt(env)
        hand = env.hand

        # Hand card 52-dim indices — MUST be sorted to match act()'s _get_hand_indices()
        hand_indices_52 = sorted([_card_index(r, s) for r, s in hand])

        combo_counts = np.zeros(NUM_ACTIONS, dtype=np.float32)
        valid_votes = 0
        raw_responses = []

        # Batch all N votes
        full_prompt = (
            f"<|im_start|>system\n{LLM_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}\n/no_think<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        try:
            batch_prompts = [full_prompt] * self.num_votes
            outputs = self.llm.generate(batch_prompts, self.sampling_params)
            self._total_queries += self.num_votes

            for out in outputs:
                raw_text = out.outputs[0].text.strip()
                raw_responses.append(raw_text)
                action_type, card_idx = self._parse_response(raw_text, hand)
                if action_type is not None and card_idx:
                    cidx = _card_selection_to_combo_idx(action_type, card_idx, hand_indices_52)
                    if cidx is not None:
                        combo_counts[cidx] += 1
                        valid_votes += 1
                    else:
                        print(f"[LLM WARNING] Could not map to combo: type={action_type}, cards={card_idx}")
                else:
                    print(f"[LLM WARNING] Unparseable response: {raw_text!r}")
        except Exception as e:
            print(f"[LLM WARNING] Batch query failed: {e}")
            self._total_queries += self.num_votes

        # Laplace smoothing
        eps = 0.001
        if valid_votes > 0:
            p_combo = (combo_counts + eps) / (valid_votes + eps * NUM_ACTIONS)
        else:
            p_combo = np.ones(NUM_ACTIONS, dtype=np.float32) / NUM_ACTIONS

        # Normalize
        p_combo = p_combo / p_combo.sum()

        self._valid_votes_history.append(valid_votes)
        if len(self._valid_votes_history) > self._valid_votes_max:
            self._valid_votes_history = self._valid_votes_history[-self._valid_votes_max:]

        # Cache with LRU eviction
        self._cache[key] = p_combo
        if len(self._cache) > self._cache_maxsize:
            self._cache.popitem(last=False)

        # Log
        if self._llm_log_file is not None:
            hand_strs = [_card_to_str(r, s) for r, s in hand]
            log_entry = {
                "query_id": self._total_queries,
                "responses": raw_responses,
                "valid_votes": valid_votes,
                "hand": hand_strs,
            }
            self._llm_log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            self._llm_log_count += 1
            if self._llm_log_count % self._llm_log_flush_interval == 0:
                self._llm_log_file.flush()

        return p_combo

    @property
    def total_queries(self):
        return self._total_queries

    @property
    def cache_hits(self):
        return self._cache_hits

    @property
    def valid_vote_rate(self):
        if not self._valid_votes_history:
            return 0.0
        return float(np.mean(self._valid_votes_history)) / max(1, self.num_votes)


# ════════════════════════════════════════════════════════════════
# 2. HesitationGate -- softmax variance over 436 actions
# ════════════════════════════════════════════════════════════════

class HesitationGate:
    """
    Top-k probability gate for Categorical(436) action space.

    h(s) = sum of top-k probabilities from π(·|s)
    h near 0 -> uniform (uncertain) -> gate ON (ask LLM)
    h near 1 -> peaked (decisive) -> gate OFF (trust PPO)

    Dimension-agnostic: works the same regardless of action space size.
    """
    def __init__(self, tau=0.5, top_k=5):
        self.tau = tau
        self.top_k = top_k

    def compute_h(self, action_logits, valid_mask):
        """
        action_logits: (B, 436)
        valid_mask:    (B, 436) bool

        Returns h: (B,) in [0, 1] — sum of top-k probabilities
        """
        masked_logits = action_logits.clone()
        masked_logits[~valid_mask] = -1e8

        probs = F.softmax(masked_logits, dim=-1)
        top_k_probs, _ = probs.topk(self.top_k, dim=-1)  # (B, k)
        h = top_k_probs.sum(dim=-1)  # (B,)
        return h.clamp(0.0, 1.0)

    def __call__(self, action_logits, valid_mask):
        h = self.compute_h(action_logits, valid_mask)
        gate = (h < self.tau).float()
        return gate, h


# ════════════════════════════════════════════════════════════════
# 3. HesitationCardPPOAgent -- PPO + hesitation gate + KL
# ════════════════════════════════════════════════════════════════

class HesitationCardPPOAgent:
    """
    PPO agent for card play/discard with Categorical(436) action space
    and hesitation-gated LLM prior.

    Loss = standard_ppo_loss + alpha * E[g(s) * KL(pi||p_LLM)]
    """

    def __init__(self, obs_dim, max_hand_size, device,
                 lr=3e-4, gamma=0.99, gae_lambda=0.95, clip=0.2,
                 vcoef=0.5, ecoef=0.05, epochs=4, mb_size=1024,
                 total_updates=1,
                 # Hesitation
                 tau=0.5, alpha=0.1,
                 kl_target=0.5, alpha_min=0.01, alpha_max=10.0,
                 llm_prior=None):
        self.device = device
        self.gamma, self.lmbda = gamma, gae_lambda
        self.clip, self.vcoef, self.ecoef = clip, vcoef, ecoef
        self.epochs, self.mb_size = epochs, mb_size
        self.max_hand_size = max_hand_size
        self.alpha = alpha

        # Adaptive alpha
        self.kl_target = kl_target
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        # Network + optimizer
        self.net = ActorCritic(obs_dim, max_hand_size).to(device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

        # LR schedule: 5% warmup + cosine decay
        warmup_steps = max(1, total_updates // 20)
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup_sched = LinearLR(self.opt, start_factor=0.1, total_iters=warmup_steps)
        cosine_sched = CosineAnnealingLR(
            self.opt, T_max=max(1, total_updates - warmup_steps), eta_min=lr * 0.1
        )
        self.scheduler = SequentialLR(
            self.opt, [warmup_sched, cosine_sched], milestones=[warmup_steps]
        )

        # Hesitation gate + LLM prior
        self.gate = HesitationGate(tau=tau)
        self.llm_prior = llm_prior

        # Statistics
        self.gate_stats = {
            "total": 0, "active": 0,
            "h_sum": 0.0, "h_count": 0,
            "agreements": 0,
            "llm_samples": 0,
        }

    # ── Act (single step) ────────────────────────────────────
    @torch.no_grad()
    def act(self, obs_np, env=None):
        """
        Sample action; query LLM if hesitation gate fires.

        Returns:
            a_type, card_mask_52, combo_idx, logprob, value, entropy,
            p_llm_combo (np.array(436,) or None),
            gate_active (bool), h_value (float)
        """
        self.net.eval()
        x = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_logits, value = self.net(x)

        # Build valid action mask
        hand_indices = _get_hand_indices(obs_np)
        hand_size = len(hand_indices)
        can_discard = obs_np[209] >= 0.01
        valid_mask = get_valid_action_mask(hand_size, can_discard, device=self.device).unsqueeze(0)  # (1, 436)

        # Mask invalid actions
        action_logits[~valid_mask] = -1e8

        # Hesitation gate
        gate_val, h_val = self.gate(action_logits, valid_mask)
        gate_active = bool(gate_val.item() > 0.5)
        h_value = float(h_val.item())

        # Sample from Categorical
        dist = torch.distributions.Categorical(logits=action_logits[0])
        combo_idx_t = dist.sample()
        combo_idx = int(combo_idx_t.item())

        logprob = dist.log_prob(combo_idx_t)
        entropy = dist.entropy()

        self.gate_stats["total"] += 1
        self.gate_stats["h_sum"] += h_value
        self.gate_stats["h_count"] += 1

        # Query LLM if gate active
        p_llm_combo = None
        if gate_active and self.llm_prior is not None and env is not None:
            self.gate_stats["active"] += 1
            p_llm_combo = self.llm_prior.get_prior(env)

            # LLM action override: use LLM's most-voted combo
            if p_llm_combo is not None:
                # Find best valid combo from LLM prior
                llm_probs_t = torch.as_tensor(p_llm_combo, dtype=torch.float32, device=self.device)
                llm_probs_t[~valid_mask[0]] = 0.0
                if llm_probs_t.sum() > 0:
                    llm_combo_idx = int(llm_probs_t.argmax().item())
                    combo_idx = llm_combo_idx
                    combo_idx_t = torch.tensor(combo_idx, device=self.device)

                    # Recompute logprob under policy for LLM-chosen action
                    logprob = dist.log_prob(combo_idx_t)

        # Track LLM-agent agreement
        if p_llm_combo is not None:
            llm_best = int(np.argmax(p_llm_combo))
            # Check if same type (play vs discard)
            agent_type = 1 if combo_idx < NUM_COMBOS else 0
            llm_type = 1 if llm_best < NUM_COMBOS else 0
            if agent_type == llm_type:
                self.gate_stats["agreements"] += 1
            self.gate_stats["llm_samples"] += 1

        # Convert to env action
        a_type, card_mask_52 = combo_idx_to_card_mask(combo_idx, hand_indices)

        value = value.squeeze(0)

        return (
            a_type, card_mask_52, combo_idx,
            float(logprob.item()),
            float(value.item()), float(entropy.item()),
            p_llm_combo, gate_active, h_value,
        )

    # ── Evaluate actions (batch) ──────────────────────────────
    def evaluate_actions(self, obs_b, combo_idx_b):
        """
        Returns: logprob, values, entropy, action_logits
        """
        action_logits, values = self.net(obs_b)

        # Build valid masks for entire batch
        hand_sizes = obs_b[:, :52].sum(dim=1).long()  # (N,)
        can_discard = obs_b[:, 209] >= 0.01  # (N,)

        cmp = COMBO_MAX_POS.to(self.device)  # (218,) — precomputed
        play_valid = cmp.unsqueeze(0) < hand_sizes.unsqueeze(1)  # (N, 218)
        discard_valid = play_valid & can_discard.unsqueeze(1)  # (N, 218)
        valid_mask = torch.cat([play_valid, discard_valid], dim=1)  # (N, 436)

        # Clone to avoid in-place mutation of forward output (breaks autograd)
        masked_logits = action_logits.clone()
        masked_logits[~valid_mask] = -1e8

        dist = torch.distributions.Categorical(logits=masked_logits)
        logprob = dist.log_prob(combo_idx_b)
        entropy = dist.entropy()

        return logprob, values, entropy, masked_logits, valid_mask

    # ── PPO Update with hesitation-gated KL ──────────────────
    def update(self, traj):
        obs = torch.as_tensor(np.array(traj["obs"]), dtype=torch.float32, device=self.device)
        combo_idx = torch.as_tensor(np.array(traj["combo_idx"]), dtype=torch.long, device=self.device)

        with torch.no_grad():
            old_logp = torch.as_tensor(np.array(traj["logp"]), dtype=torch.float32, device=self.device)
            values = torch.as_tensor(np.array(traj["val"]), dtype=torch.float32, device=self.device)
            rewards = np.array(traj["rew"], dtype=np.float32)
            dones = np.array(traj["done"], dtype=np.float32)

            vals_ext = np.concatenate([values.detach().cpu().numpy(),
                                       np.array([0.0], dtype=np.float32)])
            rets, adv = gae(rewards, dones, vals_ext, self.gamma, self.lmbda)

            returns = torch.as_tensor(rets, dtype=torch.float32, device=self.device)
            advantages = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            advantages.clamp_(-10.0, 10.0)

        # Gate flags
        gate_flags = np.array(traj["gate"], dtype=np.float32)
        gate_t = torch.as_tensor(gate_flags, dtype=torch.float32, device=self.device)

        # LLM priors -- fill None with uniform (masked out by gate=0)
        uniform_combo = np.ones(NUM_ACTIONS, dtype=np.float32) / NUM_ACTIONS
        p_llm_list = [p if p is not None else uniform_combo for p in traj["p_llm_combo"]]
        p_llm_t = torch.as_tensor(np.array(p_llm_list), dtype=torch.float32, device=self.device)

        N = obs.size(0)
        idx = np.arange(N)

        log_info = {"pol_loss": 0.0, "v_loss": 0.0, "ent": 0.0,
                     "kl_loss": 0.0, "n_batches": 0}

        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for s in range(0, N, self.mb_size):
                mb = idx[s:s + self.mb_size]
                mb_obs = obs[mb]
                mb_combo = combo_idx[mb]
                mb_old_logp = old_logp[mb]
                mb_ret, mb_adv = returns[mb], advantages[mb]
                mb_old_v = values[mb].detach()

                new_logp, new_v, ent, cur_logits, valid_mask = self.evaluate_actions(
                    mb_obs, mb_combo
                )

                # PPO clipped policy loss
                ratio = torch.exp(new_logp - mb_old_logp)
                unclipped = -ratio * mb_adv
                clipped = -torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * mb_adv
                pol_loss = torch.max(unclipped, clipped).mean()

                # Value loss with clipping
                vclip = 0.2
                v_clipped = mb_old_v + torch.clamp(new_v - mb_old_v, -vclip, vclip)
                v_loss = torch.max((new_v - mb_ret).pow(2), (v_clipped - mb_ret).pow(2)).mean()

                # Entropy
                ent_mean = ent.mean()

                # Gated Categorical KL: KL(pi || p_LLM)
                pi = F.softmax(cur_logits, dim=-1)  # (mb, 436)
                mb_p_llm = p_llm_t[mb].clamp(min=1e-8)
                # Only compute KL over valid actions
                kl_per = pi * (pi.clamp(min=1e-8).log() - mb_p_llm.log())
                kl_per[~valid_mask] = 0.0
                kl = kl_per.sum(dim=-1)  # (mb,)
                kl = kl.clamp(0.0, 50.0)

                mb_gate = gate_t[mb]
                gated_kl = (mb_gate * kl).mean()

                # Total loss
                loss = (pol_loss
                        + self.vcoef * v_loss
                        - self.ecoef * ent_mean
                        + self.alpha * gated_kl)

                if not torch.isfinite(loss):
                    self.opt.zero_grad()
                    continue

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()

                log_info["pol_loss"] += pol_loss.item()
                log_info["v_loss"] += v_loss.item()
                log_info["ent"] += ent_mean.item()
                log_info["kl_loss"] += gated_kl.item()
                log_info["n_batches"] += 1

        # Adaptive alpha
        nb = max(1, log_info["n_batches"])
        avg_kl = log_info["kl_loss"] / nb
        old_alpha = self.alpha
        if avg_kl > self.kl_target * 1.5:
            self.alpha = min(self.alpha * 2.0, self.alpha_max)
        elif avg_kl < self.kl_target / 1.5:
            self.alpha = max(self.alpha * 0.5, self.alpha_min)
        log_info["alpha"] = self.alpha
        log_info["alpha_changed"] = (self.alpha != old_alpha)

        return log_info

    @property
    def gate_activation_rate(self):
        if self.gate_stats["total"] == 0:
            return 0.0
        return self.gate_stats["active"] / self.gate_stats["total"]

    @property
    def llm_agreement_rate(self):
        if self.gate_stats["llm_samples"] == 0:
            return 0.0
        return self.gate_stats["agreements"] / self.gate_stats["llm_samples"]

    @property
    def avg_h_value(self):
        if self.gate_stats["h_count"] == 0:
            return 0.0
        return self.gate_stats["h_sum"] / self.gate_stats["h_count"]


# ════════════════════════════════════════════════════════════════
# 4. Training Loop
# ════════════════════════════════════════════════════════════════

def train_card_hesitation(
    total_steps=1_000_000,
    update_steps=4096,
    seed=0,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip=0.2,
    vcoef=0.5,
    ecoef_start=0.05,
    ecoef_end=0.01,
    epochs=4,
    mb_size=1024,
    max_hand_size=8,
    max_play=5,
    shaping_beta=0.8,
    discard_cost=-2.0,
    checkpoint_interval=50000,
    log_interval=100,
    # Hesitation gate
    tau=0.5,
    tau_min=0.05,
    alpha=0.1,
    kl_target=0.5,
    alpha_min=0.01,
    alpha_max=10.0,
    # LLM (本地 vLLM)
    llm_model="",
    num_votes=5,
    llm_temperature=0.7,
    gpu_memory_utilization=0.85,
    quantization="",
    # Resume from checkpoint
    resume_checkpoint="",
    # Output directory
    output_dir="outputs/card_hesitation",
):
    """
    Train card agent with hesitation-gated LLM prior (Categorical 436-dim action space).

    If llm_model is empty, falls back to pure PPO (no LLM).
    If resume_checkpoint is given, loads weights + optimizer and continues from saved step.
    """
    set_seed(seed)
    device = get_device()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"card_hesitation_{timestamp}"

    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{output_dir}/logs", exist_ok=True)
    os.makedirs(f"{output_dir}/plots", exist_ok=True)

    checkpoint_dir = f"{output_dir}/checkpoints"
    log_path = f"{output_dir}/logs/{run_name}.csv"

    # ── LLM prior ────────────────────────────────────────────
    llm_prior = None
    if llm_model:
        llm_log_path = f"{output_dir}/logs/{run_name}_llm_responses.jsonl"
        llm_prior = CardLLMActionPrior(
            model_name=llm_model,
            num_votes=num_votes,
            temperature=llm_temperature,
            gpu_memory_utilization=gpu_memory_utilization,
            llm_log_path=llm_log_path,
            quantization=quantization if quantization else None,
        )
        print(f"[Init] LLM prior: {llm_model} (local vLLM), N={num_votes}, tau={tau}, alpha={alpha}")
    else:
        print(f"[Init] No LLM -- pure PPO mode (tau={tau}, alpha={alpha} ignored)")

    # ── Environment ───────────────────────────────────────────
    env = BalatroEnv(max_hand_size=max_hand_size, max_play=max_play,
                     shaping_beta=shaping_beta, discard_cost=discard_cost)
    obs = env.reset()
    obs_dim = env.observation_space.shape[0]

    total_updates = int(math.ceil(total_steps / float(update_steps)))

    print(f"[Init] Action space: Categorical({NUM_ACTIONS}) (play={NUM_COMBOS} + discard={NUM_COMBOS})")

    # ── Agent ─────────────────────────────────────────────────
    agent = HesitationCardPPOAgent(
        obs_dim, max_hand_size, device,
        lr=lr, gamma=gamma, gae_lambda=gae_lambda, clip=clip,
        vcoef=vcoef, ecoef=ecoef_start, epochs=epochs, mb_size=mb_size,
        total_updates=total_updates,
        tau=tau, alpha=alpha,
        kl_target=kl_target, alpha_min=alpha_min, alpha_max=alpha_max,
        llm_prior=llm_prior,
    )

    # ── Resume from checkpoint ─────────────────────────────────
    resume_step = 0
    resume_episode = 0
    if resume_checkpoint:
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        agent.net.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            agent.opt.load_state_dict(ckpt["optimizer"])
        resume_step = ckpt.get("step", 0)
        resume_episode = ckpt.get("episode", 0)
        if "config" in ckpt and "alpha" in ckpt["config"]:
            agent.alpha = ckpt["config"]["alpha"]
        print(f"[Resume] Loaded {resume_checkpoint}")
        print(f"[Resume] Continuing from step={resume_step}, episode={resume_episode}, alpha={agent.alpha:.4f}")

    print(f"[Init] device={device}, seed={seed}, total_steps={total_steps}")
    print(f"[Init] Run name: {run_name}")

    updates_done = resume_step // update_steps

    def anneal_ecoef():
        frac = min(1.0, updates_done / max(1, total_updates))
        return float(ecoef_start + (ecoef_end - ecoef_start) * frac)

    # ── Statistics ────────────────────────────────────────────
    ep_returns_train, ep_ret_train = [], 0.0
    ep_returns_plot, ep_ret_plot = [], 0.0
    ep_actual_scores = []
    ep_play_ratios = []
    ep_play_count, ep_discard_count = 0, 0
    gate_rates = []
    h_value_history = []
    kl_values = []
    alpha_history = []
    agreement_rates = []
    valid_vote_rates = []

    # Resume: append to existing CSV; fresh: write header
    csv_header = [
        'episode', 'steps', 'return_train', 'return_plot', 'actual_score',
        'avg_score_100', 'play_ratio', 'gate_rate', 'avg_h', 'avg_kl',
        'alpha', 'llm_agree_rate', 'valid_vote_rate',
        'ecoef', 'llm_queries', 'cache_hits',
    ]
    if resume_checkpoint and os.path.isfile(log_path):
        csv_file = open(log_path, 'a', newline='')
    else:
        csv_file = open(log_path, 'w', newline='')
        csv.writer(csv_file).writerow(csv_header)
    csv_writer = csv.writer(csv_file)
    csv_file.flush()

    total_collected = resume_step
    last_checkpoint_step = resume_step
    pbar = tqdm(total=total_steps, initial=resume_step, desc="Card Hesitation Training", unit="step", dynamic_ncols=True)

    try:
        while total_collected < total_steps:
            traj = {
                "obs": [], "combo_idx": [],
                "logp": [],
                "val": [], "rew": [], "done": [],
                "p_llm_combo": [], "gate": [],
            }
            steps = 0

            while steps < update_steps and total_collected < total_steps:
                result = agent.act(obs, env=env)
                a_type_val, card_mask_52, combo_idx = result[0], result[1], result[2]
                logp = result[3]
                val = result[4]
                p_llm_combo, gate_active = result[6], result[7]
                h_val = result[8]

                next_obs, reward, done, _ = env.step((a_type_val, card_mask_52))

                traj["obs"].append(obs)
                traj["combo_idx"].append(combo_idx)
                traj["logp"].append(logp)
                traj["val"].append(val)
                traj["rew"].append(reward)
                traj["done"].append(done)
                traj["p_llm_combo"].append(p_llm_combo)
                traj["gate"].append(gate_active)

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
                gate_rate = agent.gate_activation_rate
                if total_collected % 50 == 0 and len(ep_returns_train) > 0:
                    recent5_plot = float(np.mean(ep_returns_plot[-5:])) if ep_returns_plot else 0.0
                    recent_pr = float(np.mean(ep_play_ratios[-5:])) if ep_play_ratios else 0.0
                    pbar.set_postfix({
                        "ep": len(ep_returns_train),
                        "score5": f"{recent5_plot:.0f}",
                        "pr": f"{recent_pr:.2f}",
                        "gate": f"{gate_rate:.1%}",
                        "tau": f"{agent.gate.tau:.3f}",
                        "llm": llm_prior.total_queries if llm_prior else 0,
                    })

                if done:
                    ep_returns_train.append(ep_ret_train)
                    ep_returns_plot.append(ep_ret_plot)
                    ep_actual_scores.append(env.cumulative_score)
                    total_actions = ep_play_count + ep_discard_count
                    play_ratio = ep_play_count / max(1, total_actions)
                    ep_play_ratios.append(play_ratio)
                    gate_rates.append(gate_rate)

                    if len(ep_returns_train) % log_interval == 0:
                        avg_score_100 = float(np.mean(ep_actual_scores[-100:])) if len(ep_actual_scores) >= 100 else float(np.mean(ep_actual_scores))
                        avg_kl = kl_values[-1] if kl_values else 0.0
                        csv_writer.writerow([
                            len(ep_returns_train), total_collected,
                            ep_ret_train, ep_ret_plot, env.cumulative_score,
                            avg_score_100, play_ratio, gate_rate, agent.avg_h_value,
                            avg_kl, agent.alpha,
                            agent.llm_agreement_rate,
                            llm_prior.valid_vote_rate if llm_prior else 0.0,
                            agent.ecoef,
                            llm_prior.total_queries if llm_prior else 0,
                            llm_prior.cache_hits if llm_prior else 0,
                        ])
                        csv_file.flush()

                    ep_ret_train = 0.0
                    ep_ret_plot = 0.0
                    ep_play_count, ep_discard_count = 0, 0
                    obs = env.reset()

                # ── Checkpoint (per step check) ──────────────
                if total_collected - last_checkpoint_step >= checkpoint_interval:
                    ckpt_path = os.path.join(checkpoint_dir, f"{run_name}_step_{total_collected}.pt")
                    torch.save({
                        "state_dict": agent.net.state_dict(),
                        "optimizer": agent.opt.state_dict(),
                        "step": total_collected,
                        "episode": len(ep_returns_train),
                        "config": {
                            "obs_dim": obs_dim, "max_hand_size": max_hand_size,
                            "max_play": max_play, "tau": tau, "alpha": agent.alpha,
                            "shaping_beta": shaping_beta, "discard_cost": discard_cost,
                            "action_space": "categorical_436",
                        },
                    }, ckpt_path)
                    csv_file.flush()  # 保证 Colab 崩溃时 log 不丢
                    print(f"\n[Checkpoint] Saved to {ckpt_path}")
                    last_checkpoint_step = total_collected

            # ── PPO Update ────────────────────────────────────
            log_info = agent.update(traj)

            updates_done += 1
            agent.ecoef = anneal_ecoef()
            agent.scheduler.step()

            # Anneal tau: linearly decay from tau → tau_min
            frac = min(1.0, updates_done / max(1, total_updates))
            agent.gate.tau = tau + (tau_min - tau) * frac

            # Track KL + alpha
            nb = max(1, log_info["n_batches"])
            avg_kl_val = log_info["kl_loss"] / nb
            kl_values.append(avg_kl_val)
            alpha_history.append(agent.alpha)
            h_value_history.append(agent.avg_h_value)
            agreement_rates.append(agent.llm_agreement_rate)
            valid_vote_rates.append(llm_prior.valid_vote_rate if llm_prior else 0.0)

            # Reset gate stats per update cycle
            agent.gate_stats = {
                "total": 0, "active": 0,
                "h_sum": 0.0, "h_count": 0, "agreements": 0, "llm_samples": 0,
            }

        pbar.close()

    except KeyboardInterrupt:
        pbar.close()
        print("\n[Info] Interrupted -- saving...")

    finally:
        csv_file.close()

    # ── Save final model ─────────────────────────────────────
    final_path = os.path.join(checkpoint_dir, f"{run_name}_final.pt")
    torch.save({
        "state_dict": agent.net.state_dict(),
        "optimizer": agent.opt.state_dict(),
        "step": total_collected,
        "episode": len(ep_returns_train),
        "config": {
            "obs_dim": obs_dim, "max_hand_size": max_hand_size,
            "max_play": max_play, "tau": tau, "alpha": agent.alpha,
            "shaping_beta": shaping_beta, "discard_cost": discard_cost,
            "action_space": "categorical_436",
        },
    }, final_path)
    print(f"[Save] Final model -> {final_path}")

    # ── Plot training curves (4x2) ───────────────────────────
    if len(ep_returns_train) > 0:
        fig, axes = plt.subplots(5, 2, figsize=(14, 25))
        fig.suptitle(f'Card Hesitation-Gated LLM Prior -- {run_name}', fontsize=14, fontweight='bold')

        window = min(100, max(5, len(ep_actual_scores) // 5))

        # (0,0) Actual score
        ax = axes[0, 0]
        x_raw = np.arange(len(ep_actual_scores))
        ax.plot(x_raw, ep_actual_scores, lw=0.8, alpha=0.35, color='blue', label='Raw')
        if window >= 5 and len(ep_actual_scores) >= window:
            y_s = smooth_curve(ep_actual_scores, window=window)
            x_s = np.arange(window - 1, window - 1 + len(y_s))
            ax.plot(x_s, y_s, lw=2, color='red', label=f'MA({window})')
        ax.set_xlabel('Episode'); ax.set_ylabel('Score')
        ax.set_title('Play Score Per Episode'); ax.grid(True, ls='--', alpha=0.4); ax.legend()

        # (0,1) Training return
        ax = axes[0, 1]
        x_raw_t = np.arange(len(ep_returns_train))
        ax.plot(x_raw_t, ep_returns_train, lw=0.8, alpha=0.35, color='darkorange', label='Raw')
        if window >= 5 and len(ep_returns_train) >= window:
            y_st = smooth_curve(ep_returns_train, window=window)
            x_st = np.arange(window - 1, window - 1 + len(y_st))
            ax.plot(x_st, y_st, lw=2, color='red', label=f'MA({window})')
        ax.set_xlabel('Episode'); ax.set_ylabel('Return')
        ax.set_title('Training Return'); ax.grid(True, ls='--', alpha=0.4); ax.legend()

        # (1,0) Rolling avg score
        ax = axes[1, 0]
        if len(ep_actual_scores) >= 100:
            avg_100 = [np.mean(ep_actual_scores[max(0, i-100):i]) for i in range(100, len(ep_actual_scores)+1)]
            ax.plot(range(100, len(ep_actual_scores)+1), avg_100, lw=2, color='green')
        ax.set_xlabel('Episode'); ax.set_ylabel('Avg Score (100 ep)')
        ax.set_title('Rolling Average Score'); ax.grid(True, ls='--', alpha=0.4)

        # (1,1) Play ratio
        ax = axes[1, 1]
        if ep_play_ratios:
            x_pr = np.arange(len(ep_play_ratios))
            ax.plot(x_pr, ep_play_ratios, lw=0.8, alpha=0.35, color='teal', label='Raw')
            if window >= 5 and len(ep_play_ratios) >= window:
                pr_s = smooth_curve(ep_play_ratios, window=window)
                x_prs = np.arange(window-1, window-1+len(pr_s))
                ax.plot(x_prs, pr_s, lw=2, color='darkred', label=f'MA({window})')
            ax.axhline(y=5/8, color='gray', ls='--', alpha=0.5, label='Baseline (5/8)')
        ax.set_xlabel('Episode'); ax.set_ylabel('Play Ratio')
        ax.set_title('Play / (Play+Discard) Ratio'); ax.set_ylim(0, 1)
        ax.grid(True, ls='--', alpha=0.4); ax.legend(fontsize=8)

        # (2,0) Gate activation rate
        ax = axes[2, 0]
        if gate_rates:
            ax.plot(np.arange(len(gate_rates)), gate_rates, lw=1, color='purple')
            if len(gate_rates) >= window:
                gr_s = smooth_curve(gate_rates, window=window)
                x_gr = np.arange(window-1, window-1+len(gr_s))
                ax.plot(x_gr, gr_s, lw=2, color='red', label=f'MA({window})')
                ax.legend()
        ax.set_xlabel('Episode'); ax.set_ylabel('Gate Rate')
        ax.set_title('Hesitation Gate Activation Rate'); ax.grid(True, ls='--', alpha=0.4)
        ax.set_ylim(-0.05, 1.05)

        # (2,1) KL + alpha
        ax = axes[2, 1]
        if kl_values:
            ax.plot(np.arange(len(kl_values)), kl_values, lw=1.5, color='darkorange', label='KL')
            ax.axhline(y=agent.kl_target, ls=':', color='gray', lw=1, label=f'KL target={agent.kl_target}')
            ax.set_ylabel('Avg KL', color='darkorange')
            ax.legend(loc='upper left')
        if alpha_history:
            ax2 = ax.twinx()
            ax2.plot(np.arange(len(alpha_history)), alpha_history, lw=1.5, color='steelblue', label='alpha')
            ax2.set_ylabel('alpha (adaptive)', color='steelblue')
            ax2.legend(loc='upper right')
        ax.set_xlabel('Update')
        ax.set_title('KL Divergence + Adaptive alpha'); ax.grid(True, ls='--', alpha=0.4)

        # (3,0) h(s) over updates
        ax = axes[3, 0]
        if h_value_history:
            ax.plot(np.arange(len(h_value_history)), h_value_history, lw=1.5, color='teal', label='Avg h(s)')
            ax.axhline(y=agent.gate.tau, ls='--', color='red', lw=1, label=f'tau={agent.gate.tau}')
            ax.set_xlabel('Update'); ax.set_ylabel('Avg h(s)')
            ax.set_title('Hesitation h(s) Over Updates'); ax.grid(True, ls='--', alpha=0.4)
            ax.set_ylim(-0.05, 1.05); ax.legend()

        # (3,1) LLM agreement rate + valid vote rate
        ax = axes[3, 1]
        if agreement_rates:
            ax.plot(np.arange(len(agreement_rates)), agreement_rates, lw=1.5, color='green', label='LLM Agree')
        if valid_vote_rates:
            ax.plot(np.arange(len(valid_vote_rates)), valid_vote_rates, lw=1.5, color='orange', label='Valid Vote')
        ax.set_xlabel('Update'); ax.set_ylabel('Rate')
        ax.set_title('LLM Agreement & Valid Vote Rate'); ax.set_ylim(-0.05, 1.05)
        ax.grid(True, ls='--', alpha=0.4); ax.legend()

        # (4,0) Score distribution
        ax = axes[4, 0]
        if len(ep_actual_scores) > 500:
            ax.hist(ep_actual_scores[:500], bins=30, alpha=0.5, color='blue', label='First 500', density=True)
            ax.hist(ep_actual_scores[-500:], bins=30, alpha=0.5, color='red', label='Last 500', density=True)
        else:
            ax.hist(ep_actual_scores, bins=30, alpha=0.7, color='blue', label='All', density=True)
        ax.set_xlabel('Score'); ax.set_ylabel('Density')
        ax.set_title('Score Distribution'); ax.grid(True, ls='--', alpha=0.4); ax.legend()

        # (4,1) empty or summary text
        ax = axes[4, 1]
        ax.axis('off')
        if len(ep_actual_scores) > 0:
            final_100 = float(np.mean(ep_actual_scores[-100:])) if len(ep_actual_scores) >= 100 else float(np.mean(ep_actual_scores))
            summary = (
                f"Action space: Categorical({NUM_ACTIONS})\n"
                f"Final avg score (last 100): {final_100:.0f}\n"
                f"Max score: {max(ep_actual_scores):.0f}\n"
                f"Final alpha: {agent.alpha:.4f}\n"
                f"LLM queries: {llm_prior.total_queries if llm_prior else 0}\n"
                f"Cache hits: {llm_prior.cache_hits if llm_prior else 0}"
            )
            ax.text(0.1, 0.5, summary, transform=ax.transAxes, fontsize=12,
                    verticalalignment='center', family='monospace')

        plt.tight_layout()
        plot_path = f"{output_dir}/plots/{run_name}_curve.png"
        plt.savefig(plot_path, dpi=160)
        if 'google.colab' in sys.modules:
            plt.show()
        plt.close('all')
        print(f"[Plot] Saved -> {plot_path}")

    # ── Print summary ────────────────────────────────────────
    if len(ep_actual_scores) > 0:
        final_100 = float(np.mean(ep_actual_scores[-100:])) if len(ep_actual_scores) >= 100 \
            else float(np.mean(ep_actual_scores))
        print(f"\n{'='*60}")
        print(f"  Action space:                  Categorical({NUM_ACTIONS})")
        print(f"  Final avg score (last 100 ep): {final_100:.0f}")
        print(f"  Max score:                     {max(ep_actual_scores):.0f}")
        print(f"  Final alpha:                   {agent.alpha:.4f} (init={alpha})")
        if llm_prior:
            print(f"  Total LLM queries:             {llm_prior.total_queries}")
            print(f"  Cache hits:                    {llm_prior.cache_hits}")
        print(f"{'='*60}")

    return ep_actual_scores


# ════════════════════════════════════════════════════════════════
# 5. CLI Entry Point
# ════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description='Hesitation-Gated LLM Prior for Card Play/Discard PPO (Categorical 436)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Pure PPO (no LLM, baseline)
  python llm_train/train_card_hesitation.py --total_steps 1000000

  # With LLM (Colab A100, local vLLM)
  python llm_train/train_card_hesitation.py \\
      --llm_model Qwen/Qwen3-32B \\
      --total_steps 50000 --num_votes 5
""",
    )

    # Training
    p.add_argument("--total_steps", type=int, default=1_000_000)
    p.add_argument("--update_steps", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--vcoef", type=float, default=0.5)
    p.add_argument("--ecoef_start", type=float, default=0.05)
    p.add_argument("--ecoef_end", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--mb_size", type=int, default=1024)
    p.add_argument("--max_hand_size", type=int, default=8)
    p.add_argument("--max_play", type=int, default=5)
    p.add_argument("--shaping_beta", type=float, default=0.8)
    p.add_argument("--discard_cost", type=float, default=-2.0)
    p.add_argument("--checkpoint_interval", type=int, default=50000)
    p.add_argument("--log_interval", type=int, default=100)

    # Hesitation gate
    p.add_argument("--tau", type=float, default=0.5,
                   help="Gate threshold: query LLM when top-5 prob sum < tau")
    p.add_argument("--tau_min", type=float, default=0.05,
                   help="Final tau after linear decay")
    p.add_argument("--alpha", type=float, default=0.1,
                   help="Initial KL regularization coefficient")
    p.add_argument("--kl_target", type=float, default=0.5,
                   help="Target KL for adaptive alpha")
    p.add_argument("--alpha_min", type=float, default=0.01)
    p.add_argument("--alpha_max", type=float, default=10.0)

    # LLM (本地 vLLM)
    p.add_argument("--llm_model", type=str, default="",
                   help="HuggingFace model name (empty = pure PPO)")
    p.add_argument("--num_votes", type=int, default=5,
                   help="N: LLM queries per state for voting")
    p.add_argument("--llm_temperature", type=float, default=0.7)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                   help="vLLM GPU memory fraction (A100-80GB: 0.85 for Qwen3-32B)")
    p.add_argument("--quantization", type=str, default="",
                   help="Quantization method: awq, gptq, etc. (empty=FP16)")

    # Resume
    p.add_argument("--resume_checkpoint", type=str, default="",
                   help="Path to checkpoint .pt file to resume training from")

    # Output directory
    p.add_argument("--output_dir", type=str, default="outputs/card_hesitation",
                   help="Output directory for checkpoints/logs/plots")

    args = p.parse_args()

    train_card_hesitation(
        total_steps=args.total_steps,
        update_steps=args.update_steps,
        seed=args.seed,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip=args.clip,
        vcoef=args.vcoef,
        ecoef_start=args.ecoef_start,
        ecoef_end=args.ecoef_end,
        epochs=args.epochs,
        mb_size=args.mb_size,
        max_hand_size=args.max_hand_size,
        max_play=args.max_play,
        shaping_beta=args.shaping_beta,
        discard_cost=args.discard_cost,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        tau=args.tau,
        tau_min=args.tau_min,
        alpha=args.alpha,
        kl_target=args.kl_target,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        llm_model=args.llm_model,
        num_votes=args.num_votes,
        llm_temperature=args.llm_temperature,
        gpu_memory_utilization=args.gpu_memory_utilization,
        quantization=args.quantization,
        resume_checkpoint=args.resume_checkpoint,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
