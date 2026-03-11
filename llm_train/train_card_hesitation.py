# llm_train/train_card_hesitation.py
# -*- coding: utf-8 -*-
"""
Hesitation-Gated LLM Prior for Card Play/Discard PPO
=====================================================

Paper Eq. 4-6 implementation for card agent:
  - LLM Action Prior: N-vote empirical distribution via vLLM (local)
  - Hesitation Gate: g(s) = 1 if h(s) < τ (uncertain → query LLM)
  - Modified PPO loss: L = clip + vcoef*v_loss - ecoef*H + α*g*(KL_type + KL_card)

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

from models.card_agent import ActorCritic
from training.ppo_utils import get_device, set_seed, smooth_curve, gae
from envs.BalatroEnv import BalatroEnv

# Colab detection
if 'google.colab' in sys.modules:
    matplotlib.use('module://matplotlib_inline.backend_inline')

# ── Joker descriptions (for LLM prompt context) ────────────────
JOKER_DESCRIPTIONS = {
    0:  "Joker - +4 Mult",
    1:  "Greedy Joker - +3 Mult for each Diamond card scored",
    2:  "Lusty Joker - +3 Mult for each Heart card scored",
    3:  "Wrathful Joker - +3 Mult for each Spade card scored",
    4:  "Gluttonous Joker - +3 Mult for each Club card scored",
    5:  "Jolly Joker - +8 Mult if hand contains a Pair",
    6:  "Zany Joker - +12 Mult if hand contains Three of a Kind",
    7:  "Mad Joker - +10 Mult if hand contains Two Pair",
    8:  "Crazy Joker - +12 Mult if hand contains a Straight",
    9:  "Droll Joker - +10 Mult if hand contains a Flush",
    10: "Half Joker - +20 Mult if hand has 3 or fewer cards",
    11: "Steel Joker - +0.2 X Mult per Steel card in hand",
    12: "Joker Stencil - X1 Mult for each empty Joker slot",
    13: "Four Fingers - Flushes and Straights can be made with 4 cards",
    14: "Banner - +30 Chips for each discard remaining",
    15: "Mystic Summit - +15 Mult if 0 discards remaining",
    16: "Misprint - +? Mult (random 0 to 23)",
    17: "Raised Fist - Adds 2x the rank of lowest held card to Mult",
    18: "Fibonacci - +8 Mult for each A, 2, 3, 5, 8 scored",
    19: "Even Steven - +4 Mult for each even rank card scored (2,4,6,8,10)",
    20: "Odd Todd - +31 Chips for each odd rank card scored (A,3,5,7,9)",
    21: "Blackboard - X3 Mult if all held cards are Spades or Clubs",
    22: "Ice Cream - +100 Chips, but loses 5 Chips per round played",
    23: "Blue Joker - +2 Chips for each remaining card in the deck",
    24: "Runner - +15 Chips if hand contains a Straight (grows each time)",
    25: "Supernova - +Mult equal to the number of times this hand type has been played",
    26: "Ride the Bus - +1 Mult per consecutive hand played without a face card",
    27: "Spare Trousers - +2 Mult if hand contains Two Pair (grows each time)",
    28: "Abstract Joker - +3 Mult for each Joker you own",
    29: "Loyalty Card - X4 Mult every 6 hands played",
}

RANK_NAMES = {1: 'A', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
              8: '8', 9: '9', 10: '10', 11: 'J', 12: 'Q', 13: 'K'}
SUIT_SYMBOLS = {'H': '♥', 'D': '♦', 'C': '♣', 'S': '♠'}
SUIT_LIST = ['H', 'D', 'C', 'S']


# ════════════════════════════════════════════════════════════════
# Helper: card ↔ string conversion
# ════════════════════════════════════════════════════════════════

def _card_to_str(rank, suit):
    """(rank, suit) → 'A♠', '10♥' etc."""
    return f"{RANK_NAMES[rank]}{SUIT_SYMBOLS[suit]}"


def _index_to_card(idx):
    """0-51 index → (rank, suit)"""
    rank = idx // 4 + 1
    suit = SUIT_LIST[idx % 4]
    return (rank, suit)


def _card_index(rank, suit):
    """(rank, suit) → 0-51 index"""
    suit_map = {'H': 0, 'D': 1, 'C': 2, 'S': 3}
    return (rank - 1) * 4 + suit_map[suit]


def _parse_card_str(s):
    """Parse 'A♠' → (rank, suit) or None"""
    s = s.strip()
    if not s:
        return None
    # Map symbols back to letters
    sym_to_suit = {'♥': 'H', '♦': 'D', '♣': 'C', '♠': 'S',
                   'H': 'H', 'D': 'D', 'C': 'C', 'S': 'S'}
    name_to_rank = {'A': 1, 'J': 11, 'Q': 12, 'K': 13}
    # Last char is suit
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


# ════════════════════════════════════════════════════════════════
# 1. CardLLMActionPrior — 本地 vLLM 推理
# ════════════════════════════════════════════════════════════════

LLM_SYSTEM_PROMPT = """\
You are an expert Balatro card game player deciding which cards to play or discard.

Game rules:
- 52-card deck, 8-card hand, 5 plays + 3 discards total.
- Play 1-5 cards to score. Discard unwanted cards to draw replacements.
- Score = chips × mult. Chips and mult start from the hand type base, then card chips and joker effects are added.

Hand types (base_chips × base_mult = base score):
  High Card:        5 × 1 =    5
  Pair:            10 × 2 =   20
  Two Pair:        20 × 2 =   40
  Three of a Kind: 30 × 3 =   90
  Straight:        30 × 4 =  120   (5 consecutive ranks)
  Flush:           35 × 4 =  140   (5 same suit)
  Full House:      40 × 4 =  160
  Four of a Kind:  60 × 7 =  420
  Straight Flush: 100 × 8 =  800

Card chips added to base: A=11, K/Q/J/10=10, 9=9, ..., 2=2.
Joker effects: some add chips, some add mult, some MULTIPLY mult (very powerful).

Strategy:
- Four of a Kind (420) and Straight Flush (800) are extremely strong.
- Flush (140) and Straight (120) are good; Pair (20) and High Card (5) are weak.
- Discard weak cards to draw for better hands, but only if discards remain.
- With few plays left, play your best available hand immediately.
- Consider active joker synergies (suit bonuses, hand type bonuses).

Reply with ONLY: "PLAY card1 card2 ..." or "DISCARD card1 card2 ..."
Nothing else."""


class CardLLMActionPrior:
    """Queries LLM N times per state to form empirical card action prior."""

    def __init__(self, model_name="Qwen/Qwen3-32B", num_votes=5,
                 temperature=0.7, cache_maxsize=2048,
                 gpu_memory_utilization=0.85):
        self.model_name = model_name
        self.num_votes = num_votes
        self.temperature = temperature

        # Local vLLM loading
        from vllm import LLM, SamplingParams
        print(f"[LLM] Loading {model_name} via vLLM (local)...")
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.SamplingParams = SamplingParams
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=80,
            stop=["\n"],
        )
        print(f"[LLM] Model loaded successfully.")

        # LRU cache
        self._cache = OrderedDict()
        self._cache_maxsize = cache_maxsize
        self._total_queries = 0
        self._cache_hits = 0
        self._valid_votes_history = []  # valid votes per non-cached get_prior call

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

        # Jokers
        if hasattr(env, 'jokers') and env.jokers:
            joker_strs = []
            for j in env.jokers:
                jid = j.type_id if hasattr(j, 'type_id') else getattr(j, 'joker_type', -1)
                desc = JOKER_DESCRIPTIONS.get(jid, f"Joker(id={jid})")
                joker_strs.append(desc)
            lines.append(f"Active Jokers: {'; '.join(joker_strs)}")
        else:
            lines.append("Active Jokers: None")

        lines.append("")
        lines.append("Choose your action:")
        return "\n".join(lines)

    # ── Cache key ─────────────────────────────────────────────
    def _state_key(self, env):
        hand_key = tuple(sorted(env.hand))
        joker_ids = []
        if hasattr(env, 'jokers') and env.jokers:
            for j in env.jokers:
                jid = j.type_id if hasattr(j, 'type_id') else getattr(j, 'joker_type', -1)
                joker_ids.append(jid)
        return (hand_key, env.play_count, env.discard_count, tuple(joker_ids))

    # ── Parse LLM response ───────────────────────────────────
    def _parse_response(self, text, hand):
        """
        Parse "PLAY A♠ K♥ ..." or "DISCARD 3♣ 5♥ ..." → (type_int, card_indices)
        Returns (None, None) if parse fails.
        """
        text = text.strip()
        # Strip Qwen3 thinking tags
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        # Detect type
        upper = text.upper()
        if upper.startswith("PLAY"):
            action_type = 1
            card_part = text[4:].strip()
        elif upper.startswith("DISCARD"):
            action_type = 0
            card_part = text[7:].strip()
        else:
            return None, None

        # Parse cards
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

    # ── Single query ──────────────────────────────────────────
    def _query_single(self, prompt, hand):
        try:
            full_prompt = f"{LLM_SYSTEM_PROMPT}\n\n{prompt}"
            outputs = self.llm.generate([full_prompt], self.sampling_params)
            text = outputs[0].outputs[0].text.strip()
            result = self._parse_response(text, hand)
            if result[0] is None:
                print(f"[LLM WARNING] Unparseable response: {text!r}")
            return result
        except Exception as e:
            print(f"[LLM WARNING] Query failed: {e}")
            return None, None

    # ── N-vote prior ──────────────────────────────────────────
    def get_prior(self, env):
        """
        Query LLM N times, return empirical action distributions.

        Returns:
            p_type: np.ndarray (2,) — [P(discard), P(play)] with Laplace smoothing
            p_card: np.ndarray (52,) — per-card selection frequency with smoothing
        """
        key = self._state_key(env)

        # Cache hit
        if key in self._cache:
            self._cache_hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]

        prompt = self._build_prompt(env)
        hand = env.hand

        # Hand card indices for masking
        hand_indices = set()
        for r, s in hand:
            hand_indices.add(_card_index(r, s))

        type_counts = np.zeros(2, dtype=np.float32)  # [discard, play]
        card_counts = np.zeros(52, dtype=np.float32)
        valid_votes = 0

        for _ in range(self.num_votes):
            action_type, card_indices = self._query_single(prompt, hand)
            self._total_queries += 1

            if action_type is not None and card_indices:
                type_counts[action_type] += 1
                for ci in card_indices:
                    card_counts[ci] += 1
                valid_votes += 1

        # Laplace smoothing for type prior
        eps_t = 0.01
        if valid_votes > 0:
            p_type = (type_counts + eps_t) / (valid_votes + eps_t * 2)
        else:
            p_type = np.array([0.5, 0.5], dtype=np.float32)

        # Normalize
        p_type = p_type / p_type.sum()

        # Per-card Bernoulli prior (only for hand cards)
        eps_c = 0.01
        if valid_votes > 0:
            p_card = np.full(52, eps_c / (valid_votes + 1), dtype=np.float32)
            for ci in hand_indices:
                p_card[ci] = (card_counts[ci] + eps_c) / (valid_votes + eps_c * 2)
        else:
            p_card = np.full(52, 0.5, dtype=np.float32)

        # Clamp to [eps, 1-eps] for Bernoulli KL stability
        p_card = np.clip(p_card, 0.01, 0.99)

        self._valid_votes_history.append(valid_votes)

        # Cache with LRU eviction
        result = (p_type, p_card)
        self._cache[key] = result
        if len(self._cache) > self._cache_maxsize:
            self._cache.popitem(last=False)

        return result

    @property
    def total_queries(self):
        return self._total_queries

    @property
    def cache_hits(self):
        return self._cache_hits

    @property
    def valid_vote_rate(self):
        """Average fraction of valid votes per LLM query (0~1)."""
        if not self._valid_votes_history:
            return 0.0
        return float(np.mean(self._valid_votes_history)) / max(1, self.num_votes)


# ════════════════════════════════════════════════════════════════
# 2. HesitationGate — 直接复用
# ════════════════════════════════════════════════════════════════

try:
    from llm_train.train_joker_hesitation import HesitationGate
except ImportError:
    # Fallback: inline implementation if joker module not importable
    class HesitationGate:
        """h(s) = Var(π)/σ²_max, gate when h < τ"""
        def __init__(self, num_actions=2, tau=0.3):
            self.num_actions = num_actions
            self.tau = tau
            self.sigma2_max = (num_actions - 1) / (num_actions ** 2)

        def compute_h(self, logits, action_mask):
            masked_logits = logits + (1.0 - action_mask) * (-1e8)
            probs = F.softmax(masked_logits, dim=-1)
            mean_p = probs.mean(dim=-1, keepdim=True)
            var_p = ((probs - mean_p) ** 2).mean(dim=-1)
            h = var_p / max(self.sigma2_max, 1e-12)
            return h.clamp(0.0, 1.0)

        def __call__(self, logits, action_mask):
            h = self.compute_h(logits, action_mask)
            gate = (h < self.tau).float()
            return gate, h


# ════════════════════════════════════════════════════════════════
# 3. HesitationCardPPOAgent — PPO + hesitation gate + KL
# ════════════════════════════════════════════════════════════════

class HesitationCardPPOAgent:
    """
    PPO agent for card play/discard with hesitation-gated LLM prior.

    Loss = standard_ppo_loss + α * E[g(s) * (KL_type + KL_card)]
    """

    def __init__(self, obs_dim, max_hand_size, device,
                 lr=3e-4, gamma=0.99, gae_lambda=0.95, clip=0.2,
                 vcoef=0.5, ecoef=0.05, epochs=4, mb_size=1024,
                 total_updates=1,
                 # Hesitation
                 tau=0.3, alpha=0.1,
                 kl_target=0.5, alpha_min=0.01, alpha_max=10.0,
                 llm_prior=None):
        self.device = device
        self.gamma, self.lmbda = gamma, gae_lambda
        self.clip, self.vcoef, self.ecoef = clip, vcoef, ecoef
        self.epochs, self.mb_size = epochs, mb_size
        self.max_hand_size = max_hand_size
        self.alpha = alpha

        # Adaptive α
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

        # H3: discard mask loss coef
        self.coef_sel_dis = 0.3
        self.coef_sel_dis_target = 0.80

        # H5: mask entropy coef
        self.mask_ent_coef = 0.1
        self.mask_ent_coef_start = 0.1
        self.mask_ent_coef_end = 0.02

        # Hesitation gate + LLM prior
        self.gate = HesitationGate(num_actions=2, tau=tau)
        self.llm_prior = llm_prior

        # Statistics
        self.gate_stats = {
            "total": 0, "active": 0,
            "h_values": [],        # continuous h(s) per step
            "agreements": 0,       # LLM type == agent type count
            "llm_samples": 0,      # total gated steps with LLM response
        }

    # ── Act (single step) ────────────────────────────────────
    @torch.no_grad()
    def act(self, obs_np, env=None):
        """
        Sample action; query LLM if hesitation gate fires.

        Returns:
            a_type, a_mask, logp_type, logp_mask, value, entropy,
            p_llm_type (np.array(2,) or None),
            p_llm_card (np.array(52,) or None),
            gate_active (bool)
        """
        self.net.eval()
        x = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits_type, logits_sel_play, logits_sel_dis, value = self.net(x)

        # H7: mask discard when discard_count=0
        if obs_np[209] < 0.01:
            logits_type[0, 0] = -1e8

        # Hesitation gate on type head
        type_mask = torch.ones((1, 2), device=self.device)
        if obs_np[209] < 0.01:
            type_mask[0, 0] = 0.0
        gate_val, h_val = self.gate(logits_type, type_mask)
        gate_active = bool(gate_val.item() > 0.5)
        h_value = float(h_val.item())

        self.gate_stats["total"] += 1
        self.gate_stats["h_values"].append(h_value)

        # Query LLM if gate active
        p_llm_type, p_llm_card = None, None
        if gate_active and self.llm_prior is not None and env is not None:
            self.gate_stats["active"] += 1
            p_llm_type, p_llm_card = self.llm_prior.get_prior(env)

        # Sample type (from policy, NOT LLM)
        dist_type = torch.distributions.Categorical(logits=logits_type)
        a_type_t = dist_type.sample()
        a_type = int(a_type_t.item())

        # Sample card mask
        hand_mask = torch.as_tensor(obs_np[:52], dtype=torch.float32, device=self.device).unsqueeze(0)
        hand_indices = torch.where(hand_mask[0] > 0.5)[0]

        if len(hand_indices) == 0:
            a_mask = [0] * 52
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

        # Track LLM-agent agreement
        if p_llm_type is not None:
            llm_preferred_type = int(np.argmax(p_llm_type))
            if llm_preferred_type == a_type:
                self.gate_stats["agreements"] += 1
            self.gate_stats["llm_samples"] += 1

        logprob_type = dist_type.log_prob(a_type_t)
        entropy_type = dist_type.entropy()
        total_entropy = entropy_type + entropy_mask
        value = value.squeeze(0)

        return (
            a_type, a_mask,
            float(logprob_type.item()), float(logprob_mask.item()),
            float(value.item()), float(total_entropy.item()),
            p_llm_type, p_llm_card, gate_active, h_value,
        )

    # ── Evaluate actions (batch) ──────────────────────────────
    def evaluate_actions(self, obs_b, a_type_b, a_mask_b):
        """Returns: logp_type, logp_mask, values, ent_type, ent_mask, logits_type, logits_card"""
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

        return (logprob_type, logprob_mask, values, entropy_type, entropy_mask,
                logits_type, logits_masked)

    # ── PPO Update with hesitation-gated KL ──────────────────
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

        # LLM priors — fill None with uniform (masked out by gate=0)
        uniform_type = np.array([0.5, 0.5], dtype=np.float32)
        uniform_card = np.full(52, 0.5, dtype=np.float32)
        p_llm_type_list = [p if p is not None else uniform_type for p in traj["p_llm_type"]]
        p_llm_card_list = [p if p is not None else uniform_card for p in traj["p_llm_card"]]
        p_llm_type_t = torch.as_tensor(np.array(p_llm_type_list), dtype=torch.float32, device=self.device)
        p_llm_card_t = torch.as_tensor(np.array(p_llm_card_list), dtype=torch.float32, device=self.device)

        N = obs.size(0)
        idx = np.arange(N)

        log_info = {"pol_loss": 0.0, "v_loss": 0.0, "ent": 0.0,
                     "kl_loss": 0.0, "kl_type_loss": 0.0, "kl_card_loss": 0.0,
                     "n_batches": 0}

        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for s in range(0, N, self.mb_size):
                mb = idx[s:s + self.mb_size]
                mb_obs, mb_type, mb_mask = obs[mb], a_type[mb], a_mask[mb]
                mb_old_logp_type = old_logp_type[mb]
                mb_old_logp_mask = old_logp_mask[mb]
                mb_ret, mb_adv = returns[mb], advantages[mb]
                mb_old_v = values[mb].detach()

                (new_logp_type, new_logp_mask, new_v, ent_type, ent_mask,
                 cur_logits_type, cur_logits_card) = self.evaluate_actions(
                    mb_obs, mb_type, mb_mask
                )

                # ── Type head PPO loss ────────────────────────
                ratio_type = torch.exp(new_logp_type - mb_old_logp_type)
                unclipped_t = -ratio_type * mb_adv
                clipped_t = -torch.clamp(ratio_type, 1 - self.clip, 1 + self.clip) * mb_adv
                pol_loss_type = torch.max(unclipped_t, clipped_t).mean()

                # ── Card mask PPO loss (play/discard separate) ─
                ratio_mask = torch.exp(new_logp_mask - mb_old_logp_mask)
                unclipped_m = -ratio_mask * mb_adv
                clipped_m = -torch.clamp(ratio_mask, 1 - self.clip, 1 + self.clip) * mb_adv
                per_sample_m = torch.max(unclipped_m, clipped_m)

                m_play = (mb_type == 1)
                m_dis = (mb_type == 0)
                pol_loss_mask_play = per_sample_m[m_play].mean() if m_play.any() else torch.tensor(0.0, device=self.device)
                pol_loss_mask_dis = per_sample_m[m_dis].mean() if m_dis.any() else torch.tensor(0.0, device=self.device)

                pol_loss = pol_loss_type + pol_loss_mask_play + self.coef_sel_dis * pol_loss_mask_dis

                # ── Value loss with clipping ──────────────────
                vclip = 0.2
                v_clipped = mb_old_v + torch.clamp(new_v - mb_old_v, -vclip, vclip)
                v_loss = torch.max((new_v - mb_ret).pow(2), (v_clipped - mb_ret).pow(2)).mean()

                # ── Entropy ───────────────────────────────────
                ent = ent_type.mean() + self.mask_ent_coef * ent_mask.mean()

                # ── Gated KL (type + card) ────────────────────
                # Type KL: KL(Categorical(π_type) || Categorical(p_llm_type))
                pi_type = F.softmax(cur_logits_type, dim=-1)
                mb_p_llm_type = p_llm_type_t[mb].clamp(min=1e-8)
                kl_type = (pi_type * (pi_type.clamp(min=1e-8).log() - mb_p_llm_type.log())).sum(dim=-1)
                kl_type = kl_type.clamp(0.0, 10.0)

                # Card KL: Σ_i∈hand KL(Bernoulli(σ(logit_i)) || Bernoulli(p_llm_card_i))
                hand_mask_mb = mb_obs[:, :52]
                pi_card = torch.sigmoid(cur_logits_card)  # (mb, 52)
                mb_p_llm_card = p_llm_card_t[mb].clamp(min=1e-8, max=1.0 - 1e-8)
                pi_card_safe = pi_card.clamp(min=1e-8, max=1.0 - 1e-8)

                kl_card_per = (
                    pi_card_safe * (pi_card_safe.log() - mb_p_llm_card.log())
                    + (1 - pi_card_safe) * ((1 - pi_card_safe).log() - (1 - mb_p_llm_card).log())
                )
                kl_card = (kl_card_per * hand_mask_mb).sum(dim=-1)  # (mb,)
                kl_card = kl_card.clamp(0.0, 50.0)

                # Gate
                mb_gate = gate_t[mb]
                gated_kl = (mb_gate * (kl_type + kl_card)).mean()

                # ── Total loss ────────────────────────────────
                loss = (pol_loss
                        + self.vcoef * v_loss
                        - self.ecoef * ent
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
                log_info["ent"] += ent.mean().item()
                log_info["kl_loss"] += gated_kl.item()
                log_info["kl_type_loss"] += (mb_gate * kl_type).mean().item()
                log_info["kl_card_loss"] += (mb_gate * kl_card).mean().item()
                log_info["n_batches"] += 1

        # ── Adaptive α ─────────────────────────────────────
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
        if not self.gate_stats["h_values"]:
            return 0.0
        return float(np.mean(self.gate_stats["h_values"]))


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
    tau=0.3,
    alpha=0.1,
    kl_target=0.5,
    alpha_min=0.01,
    alpha_max=10.0,
    # LLM (本地 vLLM)
    llm_model="",
    num_votes=5,
    llm_temperature=0.7,
    gpu_memory_utilization=0.85,
):
    """
    Train card agent with hesitation-gated LLM prior.

    If llm_model is empty, falls back to pure PPO (no LLM).
    """
    set_seed(seed)
    device = get_device()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"card_hesitation_{timestamp}"

    os.makedirs("outputs/card_hesitation/checkpoints", exist_ok=True)
    os.makedirs("outputs/card_hesitation/logs", exist_ok=True)
    os.makedirs("outputs/card_hesitation/plots", exist_ok=True)

    checkpoint_dir = "outputs/card_hesitation/checkpoints"
    log_path = f"outputs/card_hesitation/logs/{run_name}.csv"

    # ── LLM prior ────────────────────────────────────────────
    llm_prior = None
    if llm_model:
        llm_prior = CardLLMActionPrior(
            model_name=llm_model,
            num_votes=num_votes,
            temperature=llm_temperature,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        print(f"[Init] LLM prior: {llm_model} (local vLLM), N={num_votes}, τ={tau}, α={alpha}")
    else:
        print(f"[Init] No LLM — pure PPO mode (τ={tau}, α={alpha} ignored)")

    # ── Environment ───────────────────────────────────────────
    env = BalatroEnv(max_hand_size=max_hand_size, max_play=max_play,
                     shaping_beta=shaping_beta, discard_cost=discard_cost)
    obs = env.reset()
    obs_dim = env.observation_space.shape[0]

    total_updates = int(math.ceil(total_steps / float(update_steps)))

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

    print(f"[Init] device={device}, seed={seed}, total_steps={total_steps}")
    print(f"[Init] Run name: {run_name}")

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

    # ── Statistics ────────────────────────────────────────────
    ep_returns_train, ep_ret_train = [], 0.0
    ep_returns_plot, ep_ret_plot = [], 0.0
    ep_actual_scores = []
    ep_play_ratios = []
    ep_play_count, ep_discard_count = 0, 0
    gate_rates = []
    h_value_history = []      # avg h(s) per update cycle
    kl_values = []
    kl_type_values = []
    kl_card_values = []
    alpha_history = []
    agreement_rates = []
    valid_vote_rates = []

    csv_file = open(log_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'episode', 'steps', 'return_train', 'return_plot', 'actual_score',
        'avg_score_100', 'play_ratio', 'gate_rate', 'avg_h', 'avg_kl',
        'kl_type', 'kl_card', 'alpha', 'llm_agree_rate', 'valid_vote_rate',
        'ecoef', 'coef_dis', 'llm_queries', 'cache_hits',
    ])
    csv_file.flush()

    total_collected = 0
    last_checkpoint_step = 0
    pbar = tqdm(total=total_steps, desc="Card Hesitation Training", unit="step", dynamic_ncols=True)

    try:
        while total_collected < total_steps:
            traj = {
                "obs": [], "a_type": [], "a_mask": [],
                "logp_type": [], "logp_mask": [],
                "val": [], "rew": [], "done": [],
                "p_llm_type": [], "p_llm_card": [], "gate": [],
            }
            steps = 0

            while steps < update_steps and total_collected < total_steps:
                result = agent.act(obs, env=env)
                a_type_val, a_mask_val = result[0], result[1]
                logp_type, logp_mask = result[2], result[3]
                val = result[4]
                p_llm_type, p_llm_card, gate_active = result[6], result[7], result[8]
                h_val = result[9]

                next_obs, reward, done, _ = env.step((a_type_val, a_mask_val))

                traj["obs"].append(obs)
                traj["a_type"].append(a_type_val)
                traj["a_mask"].append(a_mask_val)
                traj["logp_type"].append(logp_type)
                traj["logp_mask"].append(logp_mask)
                traj["val"].append(val)
                traj["rew"].append(reward)
                traj["done"].append(done)
                traj["p_llm_type"].append(p_llm_type)
                traj["p_llm_card"].append(p_llm_card)
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
                if len(ep_returns_train) > 0:
                    recent5_plot = float(np.mean(ep_returns_plot[-5:])) if ep_returns_plot else 0.0
                    recent_pr = float(np.mean(ep_play_ratios[-5:])) if ep_play_ratios else 0.0
                    pbar.set_postfix({
                        "ep": len(ep_returns_train),
                        "score5": f"{recent5_plot:.0f}",
                        "pr": f"{recent_pr:.2f}",
                        "gate": f"{gate_rate:.1%}",
                        "α": f"{agent.alpha:.3f}",
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
                        avg_kl_t = kl_type_values[-1] if kl_type_values else 0.0
                        avg_kl_c = kl_card_values[-1] if kl_card_values else 0.0
                        csv_writer.writerow([
                            len(ep_returns_train), total_collected,
                            ep_ret_train, ep_ret_plot, env.cumulative_score,
                            avg_score_100, play_ratio, gate_rate, agent.avg_h_value,
                            avg_kl, avg_kl_t, avg_kl_c, agent.alpha,
                            agent.llm_agreement_rate,
                            llm_prior.valid_vote_rate if llm_prior else 0.0,
                            agent.ecoef, agent.coef_sel_dis,
                            llm_prior.total_queries if llm_prior else 0,
                            llm_prior.cache_hits if llm_prior else 0,
                        ])
                        csv_file.flush()

                    ep_ret_train = 0.0
                    ep_ret_plot = 0.0
                    ep_play_count, ep_discard_count = 0, 0
                    obs = env.reset()

            # ── Checkpoint ────────────────────────────────────
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
                    },
                }, ckpt_path)
                print(f"\n[Checkpoint] Saved to {ckpt_path}")
                last_checkpoint_step = total_collected

            # ── PPO Update ────────────────────────────────────
            log_info = agent.update(traj)

            updates_done += 1
            agent.ecoef = anneal_ecoef()
            agent.coef_sel_dis = anneal_coef_dis()
            agent.mask_ent_coef = anneal_mask_ent_coef()
            agent.scheduler.step()

            # Track KL + alpha + new metrics
            nb = max(1, log_info["n_batches"])
            avg_kl_val = log_info["kl_loss"] / nb
            kl_values.append(avg_kl_val)
            kl_type_values.append(log_info["kl_type_loss"] / nb)
            kl_card_values.append(log_info["kl_card_loss"] / nb)
            alpha_history.append(agent.alpha)
            h_value_history.append(agent.avg_h_value)
            agreement_rates.append(agent.llm_agreement_rate)
            valid_vote_rates.append(llm_prior.valid_vote_rate if llm_prior else 0.0)

            # Reset gate stats per update cycle
            agent.gate_stats = {
                "total": 0, "active": 0,
                "h_values": [], "agreements": 0, "llm_samples": 0,
            }

        pbar.close()

    except KeyboardInterrupt:
        pbar.close()
        print("\n[Info] Interrupted — saving...")

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
        },
    }, final_path)
    print(f"[Save] Final model → {final_path}")

    # ── Plot training curves (4×2) ───────────────────────────
    if len(ep_returns_train) > 0:
        fig, axes = plt.subplots(5, 2, figsize=(14, 25))
        fig.suptitle(f'Card Hesitation-Gated LLM Prior — {run_name}', fontsize=14, fontweight='bold')

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
            ax2.plot(np.arange(len(alpha_history)), alpha_history, lw=1.5, color='steelblue', label='α')
            ax2.set_ylabel('α (adaptive)', color='steelblue')
            ax2.legend(loc='upper right')
        ax.set_xlabel('Update')
        ax.set_title('KL Divergence + Adaptive α'); ax.grid(True, ls='--', alpha=0.4)

        # (3,0) h(s) distribution + avg h over updates
        ax = axes[3, 0]
        if h_value_history:
            ax.plot(np.arange(len(h_value_history)), h_value_history, lw=1.5, color='teal', label='Avg h(s)')
            ax.axhline(y=agent.gate.tau, ls='--', color='red', lw=1, label=f'τ={agent.gate.tau}')
            ax.set_xlabel('Update'); ax.set_ylabel('Avg h(s)')
            ax.set_title('Hesitation h(s) Over Updates'); ax.grid(True, ls='--', alpha=0.4)
            ax.set_ylim(-0.05, 1.05); ax.legend()

        # (3,1) KL_type vs KL_card over updates
        ax = axes[3, 1]
        if kl_type_values:
            ax.plot(np.arange(len(kl_type_values)), kl_type_values, lw=1.5, color='coral', label='KL_type')
        if kl_card_values:
            ax.plot(np.arange(len(kl_card_values)), kl_card_values, lw=1.5, color='steelblue', label='KL_card')
        ax.set_xlabel('Update'); ax.set_ylabel('Avg Gated KL')
        ax.set_title('KL_type vs KL_card'); ax.grid(True, ls='--', alpha=0.4); ax.legend()

        # (4,0) LLM agreement rate + valid vote rate
        ax = axes[4, 0]
        if agreement_rates:
            ax.plot(np.arange(len(agreement_rates)), agreement_rates, lw=1.5, color='green', label='LLM Agree')
        if valid_vote_rates:
            ax.plot(np.arange(len(valid_vote_rates)), valid_vote_rates, lw=1.5, color='orange', label='Valid Vote')
        ax.set_xlabel('Update'); ax.set_ylabel('Rate')
        ax.set_title('LLM Agreement & Valid Vote Rate'); ax.set_ylim(-0.05, 1.05)
        ax.grid(True, ls='--', alpha=0.4); ax.legend()

        # (4,1) Score distribution
        ax = axes[4, 1]
        if len(ep_actual_scores) > 500:
            ax.hist(ep_actual_scores[:500], bins=30, alpha=0.5, color='blue', label='First 500', density=True)
            ax.hist(ep_actual_scores[-500:], bins=30, alpha=0.5, color='red', label='Last 500', density=True)
        else:
            ax.hist(ep_actual_scores, bins=30, alpha=0.7, color='blue', label='All', density=True)
        ax.set_xlabel('Score'); ax.set_ylabel('Density')
        ax.set_title('Score Distribution'); ax.grid(True, ls='--', alpha=0.4); ax.legend()

        plt.tight_layout()
        plot_path = f"outputs/card_hesitation/plots/{run_name}_curve.png"
        plt.savefig(plot_path, dpi=160)
        if 'google.colab' in sys.modules:
            plt.show()
        plt.close('all')
        print(f"[Plot] Saved → {plot_path}")

    # ── Print summary ────────────────────────────────────────
    if len(ep_actual_scores) > 0:
        final_100 = float(np.mean(ep_actual_scores[-100:])) if len(ep_actual_scores) >= 100 \
            else float(np.mean(ep_actual_scores))
        print(f"\n{'='*60}")
        print(f"  Final avg score (last 100 ep): {final_100:.0f}")
        print(f"  Max score:                     {max(ep_actual_scores):.0f}")
        print(f"  Final α:                       {agent.alpha:.4f} (init={alpha})")
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
        description='Hesitation-Gated LLM Prior for Card Play/Discard PPO',
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
    p.add_argument("--tau", type=float, default=0.3,
                   help="Gate threshold: query LLM when h(s) < tau")
    p.add_argument("--alpha", type=float, default=0.1,
                   help="Initial KL regularization coefficient")
    p.add_argument("--kl_target", type=float, default=0.5,
                   help="Target KL for adaptive alpha")
    p.add_argument("--alpha_min", type=float, default=0.01)
    p.add_argument("--alpha_max", type=float, default=10.0)

    # LLM (本地 vLLM)
    p.add_argument("--llm_model", type=str, default="",
                   help="HuggingFace model name (不传则纯 PPO)")
    p.add_argument("--num_votes", type=int, default=5,
                   help="N: LLM queries per state for voting")
    p.add_argument("--llm_temperature", type=float, default=0.7)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                   help="vLLM GPU memory fraction (A100-80GB: 0.85 for Qwen3-32B)")

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
        alpha=args.alpha,
        kl_target=args.kl_target,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        llm_model=args.llm_model,
        num_votes=args.num_votes,
        llm_temperature=args.llm_temperature,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


if __name__ == "__main__":
    main()
