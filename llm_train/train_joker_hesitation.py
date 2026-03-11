# llm_train/train_joker_hesitation.py
# -*- coding: utf-8 -*-
"""
Hesitation-Gated LLM Prior for Joker Selection PPO
===================================================

Paper Eq. 4-6 implementation:
  - LLM Action Prior: N-vote empirical distribution p_LLM(a|s)
  - Hesitation Gate: g(s) = 1 if h(s) < τ  (uncertain → query LLM)
  - Modified PPO loss: L = clip + vcoef*v_loss - ecoef*H + α*g*KL(π||p_LLM)

LLM: Qwen3-32B via vLLM (OpenAI-compatible endpoint)
Reward: Joint mode — card agent plays with selected jokers, cumulative score as reward

Google Colab quick start:
  !pip install openai torch numpy matplotlib tqdm gym

  import sys; sys.path.insert(0, '/content/Balatro_SimpleVersion')
  from llm_train.train_joker_hesitation import train_joker_hesitation

  train_joker_hesitation(
      total_episodes=5000,
      api_base="http://YOUR_VLLM_HOST:8000/v1",
      api_key="token-abc123",
      llm_model="Qwen/Qwen3-32B",
      card_checkpoint="outputs/card/checkpoints/xxx.pt",
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

from models.joker_agent import JokerSelectNet
from training.ppo_utils import get_device, set_seed, smooth_curve, gae
from training.train_card import PPOAgent
from envs.joker_env import JokerSelectEnv, NUM_JOKER_TYPES, MAX_HELD, NUM_OFFERED, NUM_ROUNDS
from envs.joint_env import JointEnv

# Colab detection — use inline backend
if 'google.colab' in sys.modules:
    matplotlib.use('module://matplotlib_inline.backend_inline')

# ── Joker descriptions (from chatgpt_reward.py) ────────────────
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


# ════════════════════════════════════════════════════════════════
# 1. LLM Action Prior — N-vote empirical distribution
# ════════════════════════════════════════════════════════════════

LLM_SYSTEM_PROMPT = """\
You are an expert Balatro player choosing Joker cards.

Game rules:
- 7 rounds of Joker selection, 4 jokers offered per round, max 5 held.
- During card play: 52-card deck, 8-card hand, 5 plays + 3 discards per round.
- Score = (base_chips + card_chips) × (base_mult + joker_mult).
- Joker effects are applied left-to-right (additive chips first, then additive mult, then multiplicative mult).

Strategy tips:
- Pick jokers that SYNERGIZE (e.g., suit-specific mult + flush enablers).
- Multiplicative jokers (X Mult) are very powerful, prioritize them.
- Consider coverage: jokers that trigger on common hand types (Pair, Two Pair) are more reliable.
- Later rounds: consider replacing weak jokers with stronger ones.

Reply with ONLY the action number (0, 1, 2, ...). Nothing else."""


def _joker_name(jid):
    return JOKER_DESCRIPTIONS.get(jid, f"Unknown({jid})").split(" - ")[0]


def _joker_desc(jid):
    return JOKER_DESCRIPTIONS.get(jid, f"Unknown Joker (id={jid})")


class LLMActionPrior:
    """Queries LLM N times per state to form empirical action prior p_LLM(a|s)."""

    def __init__(self, api_base, api_key, model="Qwen/Qwen3-32B",
                 num_votes=5, num_actions=25,
                 temperature=0.7, timeout=30,
                 cache_maxsize=2048, rate_limit_delay=0.02,
                 llm_log_path=None):
        self.model = model
        self.num_votes = num_votes
        self.num_actions = num_actions
        self.temperature = temperature
        self.rate_limit_delay = rate_limit_delay

        from openai import OpenAI
        self.client = OpenAI(base_url=api_base, api_key=api_key, timeout=timeout)

        # LRU cache
        self._cache = OrderedDict()
        self._cache_maxsize = cache_maxsize
        self._total_queries = 0
        self._cache_hits = 0
        self._valid_votes_history = []  # valid votes per non-cached get_prior call

        # JSONL logger for LLM responses
        self._llm_log_file = None
        if llm_log_path:
            os.makedirs(os.path.dirname(llm_log_path), exist_ok=True)
            self._llm_log_file = open(llm_log_path, 'a')
            print(f"[LLM] Response log → {llm_log_path}")

    # ── Prompt construction ───────────────────────────────────
    def _build_prompt(self, state_dict):
        """Build natural language prompt from env state."""
        held = state_dict["held"]
        offered = state_dict["offered"]
        rnd = state_dict["round"]
        action_mask = state_dict["action_mask"]

        lines = [f"Round {rnd + 1}/{NUM_ROUNDS}"]

        # Currently held
        lines.append(f"\nCurrently held jokers ({len(held)}/{MAX_HELD}):")
        if held:
            for i, jid in enumerate(held):
                lines.append(f"  {i+1}. {_joker_desc(jid)}")
        else:
            lines.append("  (none)")

        # Offered this round
        lines.append(f"\nOffered this round:")
        for i, jid in enumerate(offered):
            lines.append(f"  {chr(65+i)}. {_joker_desc(jid)}")

        # Available actions
        lines.append(f"\nAvailable actions:")
        actions_desc = ["0: Skip (don't pick any joker)"]

        if len(held) < MAX_HELD:
            for i in range(NUM_OFFERED):
                jid = offered[i]
                actions_desc.append(f"{1+i}: Pick \"{_joker_desc(jid)}\"")
        else:
            for i in range(NUM_OFFERED):
                offered_jid = offered[i]
                for j in range(MAX_HELD):
                    held_jid = held[j]
                    action_id = 5 + i * MAX_HELD + j
                    actions_desc.append(
                        f"{action_id}: Pick \"{_joker_desc(offered_jid)}\" "
                        f"→ replace slot {j+1} ({_joker_name(held_jid)})"
                    )

        lines.append("  " + "\n  ".join(actions_desc))
        lines.append("\nChoose the best action:")
        return "\n".join(lines)

    # ── Cache key ─────────────────────────────────────────────
    @staticmethod
    def _state_key(state_dict):
        return (tuple(sorted(state_dict["held"])),
                tuple(state_dict["offered"]),
                state_dict["round"])

    # ── Single LLM query ─────────────────────────────────────
    def _query_single(self, prompt):
        raw_text = ""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=10,
                temperature=self.temperature,
            )
            raw_text = response.choices[0].message.content.strip()
            # Handle Qwen3 thinking tags: strip <think>...</think>
            text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
            match = re.search(r'\d+', text)
            if match:
                return int(match.group()), raw_text
            print(f"[LLM WARNING] Unparseable response: {raw_text!r}")
        except Exception as e:
            raw_text = f"ERROR: {e}"
            print(f"[LLM WARNING] Query failed: {e}")
        return None, raw_text

    # ── N-vote prior ──────────────────────────────────────────
    def get_prior(self, state_dict):
        """
        Query LLM N times, return empirical action distribution.

        Returns:
            np.ndarray (25,) — probability distribution with Laplace smoothing
        """
        key = self._state_key(state_dict)

        # Cache hit
        if key in self._cache:
            self._cache_hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]

        prompt = self._build_prompt(state_dict)
        action_mask = state_dict["action_mask"]

        counts = np.zeros(self.num_actions, dtype=np.float32)
        valid_votes = 0
        raw_responses = []

        for _ in range(self.num_votes):
            action, raw_text = self._query_single(prompt)
            self._total_queries += 1
            raw_responses.append(raw_text)

            if action is not None and 0 <= action < self.num_actions:
                if action_mask[action] > 0:
                    counts[action] += 1
                    valid_votes += 1

            if self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)

        self._valid_votes_history.append(valid_votes)

        # Laplace smoothing → avoid zero probabilities (KL would explode)
        eps = 0.01
        if valid_votes > 0:
            prior = (counts + eps) / (valid_votes + eps * self.num_actions)
        else:
            prior = np.ones(self.num_actions, dtype=np.float32) / self.num_actions

        # Mask invalid actions and renormalize
        prior = prior * action_mask
        prior_sum = prior.sum()
        if prior_sum > 0:
            prior = prior / prior_sum
        else:
            # Fallback: uniform over valid
            valid = action_mask / max(1.0, action_mask.sum())
            prior = valid

        # Cache with LRU eviction
        self._cache[key] = prior
        if len(self._cache) > self._cache_maxsize:
            self._cache.popitem(last=False)

        # Log LLM responses to JSONL
        if self._llm_log_file is not None:
            log_entry = {
                "query_id": self._total_queries,
                "round": state_dict["round"],
                "held": [int(j) for j in state_dict["held"]],
                "offered": [int(j) for j in state_dict["offered"]],
                "prompt": prompt,
                "responses": raw_responses,
                "valid_votes": valid_votes,
                "prior_top3": sorted(
                    [(int(i), float(prior[i])) for i in range(len(prior)) if prior[i] > 0.01],
                    key=lambda x: -x[1]
                )[:3],
            }
            self._llm_log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            self._llm_log_file.flush()

        return prior

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
# 2. Hesitation Gate — Eq. 4-5
# ════════════════════════════════════════════════════════════════

class HesitationGate:
    """
    Confidence-based gate that activates LLM prior only when policy is uncertain.

    h(s) = Var(π(·|s)) / σ²_max          (Eq. 4)
    g(s) = 1 if h(s) < τ else 0          (Eq. 5)

    h(s) ∈ [0, 1]:
      h ≈ 0  → uniform policy → maximum uncertainty → gate OPENS (query LLM)
      h ≈ 1  → peaked policy  → maximum confidence  → gate CLOSES (skip LLM)
    """

    def __init__(self, num_actions=25, tau=0.3):
        self.num_actions = num_actions
        self.tau = tau
        # σ²_max = (n-1)/n² — variance of a one-hot distribution
        self.sigma2_max = (num_actions - 1) / (num_actions ** 2)

    def compute_h(self, logits, action_mask):
        """
        Compute normalized variance h(s).

        Args:
            logits: (B, 25) raw logits
            action_mask: (B, 25) float mask (1=valid, 0=invalid)

        Returns:
            h: (B,) values in [0, 1]
        """
        masked_logits = logits + (1.0 - action_mask) * (-1e8)
        probs = F.softmax(masked_logits, dim=-1)              # (B, 25)
        mean_p = probs.mean(dim=-1, keepdim=True)             # (B, 1)
        var_p = ((probs - mean_p) ** 2).mean(dim=-1)          # (B,)
        h = var_p / max(self.sigma2_max, 1e-12)
        return h.clamp(0.0, 1.0)

    def __call__(self, logits, action_mask):
        """
        Returns:
            gate: (B,) float — 1.0 where uncertain (h < τ), 0.0 where confident
            h:    (B,) float — raw h(s) values for logging
        """
        h = self.compute_h(logits, action_mask)
        gate = (h < self.tau).float()
        return gate, h


# ════════════════════════════════════════════════════════════════
# 3. HesitationJokerPPOAgent — PPO + hesitation gate + KL
# ════════════════════════════════════════════════════════════════

class HesitationJokerPPOAgent:
    """
    PPO agent for Joker selection with hesitation-gated LLM prior.

    Loss = clip_loss + vcoef * v_loss - ecoef * H(π) + α * E[g(s) * KL(π || p_LLM)]
    """

    def __init__(self, obs_dim=41, num_actions=25, device='cpu',
                 lr=3e-4, gamma=0.99, gae_lambda=0.95, clip=0.2,
                 vcoef=0.5, ecoef=0.05, epochs=4, mb_size=256,
                 total_updates=1,
                 tau=0.3, alpha=0.1,
                 kl_target=0.5, alpha_min=0.01, alpha_max=10.0,
                 llm_prior=None):
        self.device = device
        self.gamma = gamma
        self.lmbda = gae_lambda
        self.clip = clip
        self.vcoef = vcoef
        self.ecoef = ecoef
        self.epochs = epochs
        self.mb_size = mb_size
        self.num_actions = num_actions
        self.alpha = alpha

        # Adaptive α (Ouyang et al., 2022 — InstructGPT style)
        self.kl_target = kl_target
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        # Network + optimizer
        self.net = JokerSelectNet(obs_dim=obs_dim, num_actions=num_actions).to(device)
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
        self.gate = HesitationGate(num_actions=num_actions, tau=tau)
        self.llm_prior = llm_prior

        # Statistics
        self.gate_stats = {
            "total": 0, "active": 0,
            "h_values": [],        # continuous h(s) per step
            "agreements": 0,       # LLM top action == agent action count
            "llm_samples": 0,      # total gated steps with LLM response
        }

    # ── Act (single step) ────────────────────────────────────
    @torch.no_grad()
    def act(self, obs_np, action_mask_np, env_state_dict=None):
        """
        Sample action; query LLM if hesitation gate fires.

        Args:
            obs_np: (41,) numpy
            action_mask_np: (25,) numpy
            env_state_dict: {"held", "offered", "round", "action_mask"} for LLM prompt

        Returns:
            action: int
            logprob: float
            value: float
            entropy: float
            p_llm: np.ndarray (25,) or None
            gate_active: bool
        """
        self.net.eval()
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask = torch.as_tensor(action_mask_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        logits, value = self.net(obs, mask)

        # Hesitation gate
        gate_val, h_val = self.gate(logits, mask)
        gate_active = bool(gate_val.item() > 0.5)
        h_value = float(h_val.item())

        self.gate_stats["total"] += 1
        self.gate_stats["h_values"].append(h_value)

        # Query LLM if gate is active
        p_llm = None
        if gate_active and self.llm_prior is not None and env_state_dict is not None:
            self.gate_stats["active"] += 1
            p_llm = self.llm_prior.get_prior(env_state_dict)

        # Sample from current policy (NOT LLM)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        a_int = int(action.item())

        # Track LLM-agent agreement
        if p_llm is not None:
            llm_preferred = int(np.argmax(p_llm))
            if llm_preferred == a_int:
                self.gate_stats["agreements"] += 1
            self.gate_stats["llm_samples"] += 1

        return (
            a_int,
            float(dist.log_prob(action).item()),
            float(value.squeeze().item()),
            float(dist.entropy().item()),
            p_llm,
            gate_active,
            h_value,
        )

    # ── Evaluate actions (batch, for training) ───────────────
    def evaluate_actions(self, obs_b, action_b, mask_b):
        """
        Returns:
            logprob: (B,)
            values:  (B,)
            entropy: (B,)
            logits:  (B, 25) — needed for KL computation
        """
        logits, values = self.net(obs_b, mask_b)
        dist = torch.distributions.Categorical(logits=logits)
        logprob = dist.log_prob(action_b)
        entropy = dist.entropy()
        return logprob, values, entropy, logits

    # ── PPO Update with hesitation-gated KL ──────────────────
    def update(self, traj):
        """
        Modified PPO update (Eq. 6):
          L = clip_loss + vcoef*v_loss - ecoef*H + α * mean(g * KL(π || p_LLM))

        traj must include extra keys: "p_llm" (list of (25,) or None), "gate" (list of bool)

        Returns:
            dict with loss components for logging
        """
        obs = torch.as_tensor(np.array(traj["obs"]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.array(traj["action"]), dtype=torch.long, device=self.device)
        masks = torch.as_tensor(np.array(traj["mask"]), dtype=torch.float32, device=self.device)

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

        # LLM prior distributions — fill None with uniform (masked out by gate=0)
        uniform = np.ones(self.num_actions, dtype=np.float32) / self.num_actions
        p_llm_list = [p if p is not None else uniform for p in traj["p_llm"]]
        p_llm_t = torch.as_tensor(np.array(p_llm_list), dtype=torch.float32, device=self.device)

        N = obs.size(0)
        idx = np.arange(N)

        log_info = {"pol_loss": 0.0, "v_loss": 0.0, "ent": 0.0,
                     "kl_loss": 0.0, "n_batches": 0}

        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for s in range(0, N, self.mb_size):
                mb = idx[s:s + self.mb_size]

                new_logp, new_v, ent, logits = self.evaluate_actions(
                    obs[mb], actions[mb], masks[mb]
                )

                # ── Standard PPO clipped loss ────────────────
                ratio = torch.exp(new_logp - old_logp[mb])
                mb_adv = advantages[mb]
                unclipped = -ratio * mb_adv
                clipped = -torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * mb_adv
                pol_loss = torch.max(unclipped, clipped).mean()

                # ── Value loss with clipping ─────────────────
                mb_old_v = values[mb].detach()
                mb_ret = returns[mb]
                vclip = 0.2
                v_clipped = mb_old_v + torch.clamp(new_v - mb_old_v, -vclip, vclip)
                v_loss = torch.max(
                    (new_v - mb_ret).pow(2),
                    (v_clipped - mb_ret).pow(2)
                ).mean()

                # ── Gated KL divergence (Eq. 6) ─────────────
                #   KL(π || p_LLM) = Σ_a π(a) * log(π(a) / p_LLM(a))
                masked_logits = logits + (1.0 - masks[mb]) * (-1e8)
                pi_probs = F.softmax(masked_logits, dim=-1)       # (mb, 25)
                mb_p_llm = p_llm_t[mb].clamp(min=1e-8)            # (mb, 25)

                kl_per_sample = (
                    pi_probs * (pi_probs.clamp(min=1e-8).log() - mb_p_llm.log())
                ).sum(dim=-1)                                      # (mb,)
                kl_per_sample = kl_per_sample.clamp(0.0, 10.0)    # prevent explosion

                mb_gate = gate_t[mb]
                gated_kl = (mb_gate * kl_per_sample).mean()

                # ── Total loss ───────────────────────────────
                loss = (pol_loss
                        + self.vcoef * v_loss
                        - self.ecoef * ent.mean()
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
                log_info["n_batches"] += 1

        # ── Adaptive α (Ouyang et al., 2022) ─────────────
        # If avg gated KL overshoots target → increase α to penalize more
        # If avg gated KL undershoots target → decrease α to relax constraint
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

def train_joker_hesitation(
    total_episodes=5000,
    update_episodes=64,
    seed=0,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip=0.2,
    vcoef=0.5,
    ecoef=0.05,
    epochs=4,
    mb_size=256,
    checkpoint_interval=500,
    log_interval=50,
    # ── Hesitation gate ──
    tau=0.3,
    alpha=0.1,
    kl_target=0.5,
    alpha_min=0.01,
    alpha_max=10.0,
    # ── LLM ──
    api_base="",
    api_key="",
    llm_model="Qwen/Qwen3-32B",
    num_votes=5,
    llm_temperature=0.7,
    llm_timeout=30,
    rate_limit_delay=0.02,
    # ── Joint reward ──
    card_checkpoint=None,
    max_hand_size=8,
    max_play=5,
    shaping_beta=0.3,
    # Output directory (可指向 Drive 路径以持久化)
    output_dir="outputs/hesitation",
):
    """
    Train joker selection agent with hesitation-gated LLM prior.

    Requires card_checkpoint for joint reward.
    If api_base is empty, falls back to pure PPO (no LLM).
    """
    set_seed(seed)
    device = get_device()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"hesitation_run_{timestamp}"

    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{output_dir}/logs", exist_ok=True)
    os.makedirs(f"{output_dir}/plots", exist_ok=True)

    log_path = f"{output_dir}/logs/{run_name}.csv"
    checkpoint_dir = f"{output_dir}/checkpoints"

    # ── LLM prior ────────────────────────────────────────────
    llm_prior = None
    if api_base:
        # vLLM 本地部署不需要真实密钥，用占位符即可
        actual_key = api_key if api_key else "EMPTY"
        llm_log_path = f"{output_dir}/logs/{run_name}_llm_responses.jsonl"
        llm_prior = LLMActionPrior(
            api_base=api_base, api_key=actual_key, model=llm_model,
            num_votes=num_votes, temperature=llm_temperature,
            timeout=llm_timeout, rate_limit_delay=rate_limit_delay,
            llm_log_path=llm_log_path,
        )
        print(f"[Init] LLM prior: {llm_model} via {api_base}, N={num_votes}, τ={tau}, α={alpha}")
    else:
        print(f"[Init] No LLM — pure PPO mode (τ={tau}, α={alpha} ignored)")

    # ── Joint env + card agent ───────────────────────────────
    joint_env = JointEnv(
        max_hand_size=max_hand_size, max_play=max_play, shaping_beta=shaping_beta,
    )
    obs_dim = joint_env.card_env.observation_space.shape[0]

    card_agent = PPOAgent(obs_dim, max_hand_size, device, lr=1e-4)
    if card_checkpoint:
        ckpt = torch.load(card_checkpoint, map_location=device, weights_only=False)
        card_agent.net.load_state_dict(ckpt["state_dict"])
        print(f"[Init] Card agent loaded from {card_checkpoint}")
    else:
        print("[Warn] No card_checkpoint — card agent is random! Scores will be noisy.")

    # ── Joker agent ──────────────────────────────────────────
    total_updates = int(math.ceil(total_episodes / float(update_episodes)))
    agent = HesitationJokerPPOAgent(
        device=device, lr=lr, gamma=gamma, gae_lambda=gae_lambda,
        clip=clip, vcoef=vcoef, ecoef=ecoef, epochs=epochs,
        mb_size=mb_size, total_updates=total_updates,
        tau=tau, alpha=alpha,
        kl_target=kl_target, alpha_min=alpha_min, alpha_max=alpha_max,
        llm_prior=llm_prior,
    )

    print(f"[Init] device={device}, seed={seed}, episodes={total_episodes}")
    print(f"[Init] Run name: {run_name}")

    # ── CSV logger ───────────────────────────────────────────
    csv_file = open(log_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'episode', 'avg_score_7r', 'total_score', 'num_jokers',
        'avg_score_50', 'gate_rate', 'avg_h', 'llm_queries', 'cache_hits',
        'avg_kl', 'alpha', 'llm_agree_rate', 'valid_vote_rate', 'joker_ids',
    ])
    csv_file.flush()

    # ── Statistics ───────────────────────────────────────────
    ep_avg_scores = []
    ep_total_scores = []
    gate_rates = []
    h_value_history = []      # avg h(s) per update cycle
    kl_values = []
    alpha_history = []
    agreement_rates = []
    valid_vote_rates = []
    updates_done = 0

    pbar = tqdm(total=total_episodes, desc="Hesitation Training", unit="ep", dynamic_ncols=True)

    try:
        ep = 0
        joker_traj = {
            "obs": [], "action": [], "mask": [], "logp": [], "val": [],
            "rew": [], "done": [],
            "p_llm": [], "gate": [],
        }

        while ep < total_episodes:
            # ===== One episode: 7 rounds of joker + card play =====
            joker_obs = joint_env.reset()
            episode_scores = []

            for round_idx in range(NUM_ROUNDS):
                joker_mask = joint_env.get_joker_action_mask()

                # Build env state for LLM prompt
                env_state_dict = {
                    "held": list(joint_env.joker_env.held),
                    "offered": list(joint_env.joker_env.offered),
                    "round": joint_env.joker_env.round,
                    "action_mask": joker_mask.copy(),
                }

                # Joker agent selects
                action, logp, val, ent, p_llm, gate_active, h_val = agent.act(
                    joker_obs, joker_mask, env_state_dict
                )

                # Store trajectory step
                joker_traj["obs"].append(joker_obs)
                joker_traj["action"].append(action)
                joker_traj["mask"].append(joker_mask)
                joker_traj["logp"].append(logp)
                joker_traj["val"].append(val)
                joker_traj["p_llm"].append(p_llm)
                joker_traj["gate"].append(gate_active)

                # Execute joker selection
                joker_obs, joker_done, joker_info = joint_env.joker_step(action)

                # Card agent plays a full round
                card_traj, card_score = joint_env.play_card_episode(card_agent)
                episode_scores.append(card_score)

                # Joker reward: log-compressed average play score
                avg_play = card_score / max(1, joint_env.card_env.max_play)
                joker_reward = math.log1p(avg_play)

                joker_traj["rew"].append(joker_reward)
                joker_traj["done"].append(joker_done)

            # Episode stats
            total_score = sum(episode_scores)
            avg_score = float(np.mean(episode_scores))
            ep_total_scores.append(total_score)
            ep_avg_scores.append(avg_score)
            ep += 1
            pbar.update(1)

            # Progress bar
            gate_rate = agent.gate_activation_rate
            gate_rates.append(gate_rate)

            if len(ep_avg_scores) > 0:
                recent = float(np.mean(ep_avg_scores[-min(50, len(ep_avg_scores)):]))
                pbar.set_postfix({
                    "avg": f"{avg_score:.0f}",
                    "recent50": f"{recent:.0f}",
                    "gate": f"{gate_rate:.1%}",
                    "α": f"{agent.alpha:.3f}",
                    "llm_q": llm_prior.total_queries if llm_prior else 0,
                })

            # CSV log
            if ep % log_interval == 0:
                avg_50 = float(np.mean(ep_avg_scores[-50:])) if len(ep_avg_scores) >= 50 \
                    else float(np.mean(ep_avg_scores))
                avg_kl = kl_values[-1] if kl_values else 0.0
                csv_writer.writerow([
                    ep, avg_score, total_score, len(joint_env.held_jokers),
                    avg_50, gate_rate, agent.avg_h_value,
                    llm_prior.total_queries if llm_prior else 0,
                    llm_prior.cache_hits if llm_prior else 0,
                    avg_kl, agent.alpha,
                    agent.llm_agreement_rate,
                    llm_prior.valid_vote_rate if llm_prior else 0.0,
                    str(joint_env.held_jokers),
                ])
                csv_file.flush()

            # ===== PPO Update =====
            if ep % update_episodes == 0 and len(joker_traj["obs"]) > 0:
                log_info = agent.update(joker_traj)
                agent.scheduler.step()
                updates_done += 1

                # Track KL + alpha + new metrics
                nb = max(1, log_info["n_batches"])
                avg_kl_val = log_info["kl_loss"] / nb
                kl_values.append(avg_kl_val)
                alpha_history.append(agent.alpha)
                h_value_history.append(agent.avg_h_value)
                agreement_rates.append(agent.llm_agreement_rate)
                valid_vote_rates.append(llm_prior.valid_vote_rate if llm_prior else 0.0)

                # Reset trajectory
                joker_traj = {
                    "obs": [], "action": [], "mask": [], "logp": [], "val": [],
                    "rew": [], "done": [],
                    "p_llm": [], "gate": [],
                }

                # Reset gate stats per update cycle
                agent.gate_stats = {
                    "total": 0, "active": 0,
                    "h_values": [], "agreements": 0, "llm_samples": 0,
                }

            # Checkpoint
            if ep % checkpoint_interval == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"{run_name}_ep_{ep}.pt")
                torch.save({
                    "state_dict": agent.net.state_dict(),
                    "optimizer": agent.opt.state_dict(),
                    "episode": ep,
                    "config": {
                        "obs_dim": 41, "num_actions": 25,
                        "tau": tau, "alpha": alpha,
                        "num_votes": num_votes,
                    },
                }, ckpt_path)
                print(f"\n[Checkpoint] Saved to {ckpt_path}")

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
        "episode": ep,
        "config": {
            "obs_dim": 41, "num_actions": 25,
            "tau": tau, "alpha": alpha, "num_votes": num_votes,
        },
    }, final_path)
    print(f"[Save] Final model → {final_path}")

    # ── Plot training curves (2×2) ───────────────────────────
    if len(ep_avg_scores) > 0:
        fig, axes = plt.subplots(3, 2, figsize=(14, 15))
        fig.suptitle(f'Hesitation-Gated LLM Prior — {run_name}', fontsize=14, fontweight='bold')

        window = min(100, max(5, len(ep_avg_scores) // 5))

        # (0,0) Avg score per round — smoothed
        ax = axes[0, 0]
        x_raw = np.arange(len(ep_avg_scores))
        ax.plot(x_raw, ep_avg_scores, lw=0.5, alpha=0.3, color='blue', label='Raw')
        if window >= 5 and len(ep_avg_scores) >= window:
            y_s = smooth_curve(ep_avg_scores, window=window)
            x_s = np.arange(window - 1, window - 1 + len(y_s))
            ax.plot(x_s, y_s, lw=2, color='red', label=f'MA({window})')
        ax.set_xlabel('Episode'); ax.set_ylabel('Avg Score / Round')
        ax.set_title('Avg Score per Round'); ax.grid(True, ls='--', alpha=0.4); ax.legend()

        # (0,1) Total score per episode — smoothed
        ax = axes[0, 1]
        x_raw_t = np.arange(len(ep_total_scores))
        ax.plot(x_raw_t, ep_total_scores, lw=0.5, alpha=0.3, color='green', label='Raw')
        if window >= 5 and len(ep_total_scores) >= window:
            y_st = smooth_curve(ep_total_scores, window=window)
            x_st = np.arange(window - 1, window - 1 + len(y_st))
            ax.plot(x_st, y_st, lw=2, color='red', label=f'MA({window})')
        ax.set_xlabel('Episode'); ax.set_ylabel('Total Score (7 rounds)')
        ax.set_title('Total Score per Episode'); ax.grid(True, ls='--', alpha=0.4); ax.legend()

        # (1,0) Gate activation rate over time
        ax = axes[1, 0]
        if gate_rates:
            ax.plot(np.arange(len(gate_rates)), gate_rates, lw=1, color='purple')
            if len(gate_rates) >= window:
                gr_s = smooth_curve(gate_rates, window=window)
                x_gr = np.arange(window - 1, window - 1 + len(gr_s))
                ax.plot(x_gr, gr_s, lw=2, color='red', label=f'MA({window})')
                ax.legend()
        ax.set_xlabel('Episode'); ax.set_ylabel('Gate Rate')
        ax.set_title('Hesitation Gate Activation Rate'); ax.grid(True, ls='--', alpha=0.4)
        ax.set_ylim(-0.05, 1.05)

        # (1,1) KL divergence + adaptive α over updates
        ax = axes[1, 1]
        if kl_values:
            ax.plot(np.arange(len(kl_values)), kl_values, lw=1.5, color='darkorange', label='KL')
            ax.axhline(y=agent.kl_target, ls=':', color='gray', lw=1, label=f'KL target={agent.kl_target}')
            ax.set_ylabel('Avg KL(π || p_LLM)', color='darkorange')
            ax.legend(loc='upper left')
        if alpha_history:
            ax2 = ax.twinx()
            ax2.plot(np.arange(len(alpha_history)), alpha_history, lw=1.5, color='steelblue', label='α')
            ax2.set_ylabel('α (adaptive)', color='steelblue')
            ax2.legend(loc='upper right')
        ax.set_xlabel('Update')
        ax.set_title('KL Divergence + Adaptive α'); ax.grid(True, ls='--', alpha=0.4)

        # (2,0) h(s) over updates
        ax = axes[2, 0]
        if h_value_history:
            ax.plot(np.arange(len(h_value_history)), h_value_history, lw=1.5, color='teal', label='Avg h(s)')
            ax.axhline(y=agent.gate.tau, ls='--', color='red', lw=1, label=f'τ={agent.gate.tau}')
            ax.set_xlabel('Update'); ax.set_ylabel('Avg h(s)')
            ax.set_title('Hesitation h(s) Over Updates'); ax.grid(True, ls='--', alpha=0.4)
            ax.set_ylim(-0.05, 1.05); ax.legend()

        # (2,1) LLM agreement rate + valid vote rate
        ax = axes[2, 1]
        if agreement_rates:
            ax.plot(np.arange(len(agreement_rates)), agreement_rates, lw=1.5, color='green', label='LLM Agree')
        if valid_vote_rates:
            ax.plot(np.arange(len(valid_vote_rates)), valid_vote_rates, lw=1.5, color='orange', label='Valid Vote')
        ax.set_xlabel('Update'); ax.set_ylabel('Rate')
        ax.set_title('LLM Agreement & Valid Vote Rate'); ax.set_ylim(-0.05, 1.05)
        ax.grid(True, ls='--', alpha=0.4); ax.legend()

        plt.tight_layout()
        plot_path = f"{output_dir}/plots/{run_name}_curve.png"
        plt.savefig(plot_path, dpi=160)
        if 'google.colab' in sys.modules:
            plt.show()
        plt.close('all')
        print(f"[Plot] Saved → {plot_path}")

    # ── Print summary ────────────────────────────────────────
    if len(ep_avg_scores) > 0:
        final_50 = float(np.mean(ep_avg_scores[-50:])) if len(ep_avg_scores) >= 50 \
            else float(np.mean(ep_avg_scores))
        print(f"\n{'='*60}")
        print(f"  Final avg score (last 50 ep): {final_50:.0f}")
        print(f"  Max total score:              {max(ep_total_scores):.0f}")
        print(f"  Final α:                      {agent.alpha:.4f} (init={alpha})")
        if llm_prior:
            print(f"  Total LLM queries:            {llm_prior.total_queries}")
            print(f"  Cache hits:                   {llm_prior.cache_hits}")
        print(f"{'='*60}")

    return ep_avg_scores


# ════════════════════════════════════════════════════════════════
# 5. CLI Entry Point
# ════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description='Hesitation-Gated LLM Prior for Joker Selection PPO',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Pure PPO (no LLM, baseline)
  python llm_train/train_joker_hesitation.py \\
      --card_checkpoint outputs/card/checkpoints/xxx.pt

  # With LLM prior via vLLM
  python llm_train/train_joker_hesitation.py \\
      --api_base http://localhost:8000/v1 \\
      --api_key token-abc123 \\
      --llm_model Qwen/Qwen3-32B \\
      --card_checkpoint outputs/card/checkpoints/xxx.pt \\
      --tau 0.3 --alpha 0.1 --num_votes 5
""",
    )

    # Training
    p.add_argument("--total_episodes", type=int, default=5000)
    p.add_argument("--update_episodes", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--vcoef", type=float, default=0.5)
    p.add_argument("--ecoef", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--mb_size", type=int, default=256)
    p.add_argument("--checkpoint_interval", type=int, default=500)
    p.add_argument("--log_interval", type=int, default=50)

    # Hesitation gate
    p.add_argument("--tau", type=float, default=0.3,
                   help="Gate threshold: query LLM when h(s) < tau (0=never, 1=always)")
    p.add_argument("--alpha", type=float, default=0.1,
                   help="Initial KL regularization coefficient (adaptive)")
    p.add_argument("--kl_target", type=float, default=0.5,
                   help="Target KL for adaptive alpha (Ouyang et al., 2022)")
    p.add_argument("--alpha_min", type=float, default=0.01,
                   help="Minimum alpha for adaptive KL")
    p.add_argument("--alpha_max", type=float, default=10.0,
                   help="Maximum alpha for adaptive KL")

    # LLM
    p.add_argument("--api_base", type=str, default="",
                   help="vLLM endpoint, e.g. http://localhost:8000/v1")
    p.add_argument("--api_key", type=str, default="",
                   help="API key (vLLM 本地部署可不填，自动用占位符)")
    p.add_argument("--llm_model", type=str, default="Qwen/Qwen3-32B")
    p.add_argument("--num_votes", type=int, default=5,
                   help="N: number of LLM queries per state for voting")
    p.add_argument("--llm_temperature", type=float, default=0.7)
    p.add_argument("--llm_timeout", type=int, default=30)
    p.add_argument("--rate_limit_delay", type=float, default=0.02)

    # Joint
    p.add_argument("--card_checkpoint", type=str, default=None,
                   help="Path to pre-trained card agent checkpoint (required)")
    p.add_argument("--max_hand_size", type=int, default=8)
    p.add_argument("--max_play", type=int, default=5)
    p.add_argument("--shaping_beta", type=float, default=0.3)

    # Output directory
    p.add_argument("--output_dir", type=str, default="outputs/hesitation",
                   help="Output directory for checkpoints/logs/plots (可指向 Drive 路径)")

    args = p.parse_args()

    train_joker_hesitation(
        total_episodes=args.total_episodes,
        update_episodes=args.update_episodes,
        seed=args.seed,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip=args.clip,
        vcoef=args.vcoef,
        ecoef=args.ecoef,
        epochs=args.epochs,
        mb_size=args.mb_size,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        tau=args.tau,
        alpha=args.alpha,
        kl_target=args.kl_target,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        api_base=args.api_base,
        api_key=args.api_key,
        llm_model=args.llm_model,
        num_votes=args.num_votes,
        llm_temperature=args.llm_temperature,
        llm_timeout=args.llm_timeout,
        rate_limit_delay=args.rate_limit_delay,
        card_checkpoint=args.card_checkpoint,
        max_hand_size=args.max_hand_size,
        max_play=args.max_play,
        shaping_beta=args.shaping_beta,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
