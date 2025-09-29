# -*- coding: utf-8 -*-
# Simplified BalatroEnv —— 单局、无盲注、手牌评分（保留原 done 行，其它增强）
import gym
from gym import spaces
import numpy as np
import random
from itertools import combinations
from collections import Counter


class BalatroEnv(gym.Env):
    metadata = {'render.modes':['human','rgb_array']}

    def __init__(self, max_hand_size=8, hand_multipliers=None, hand_basic_score=None, max_play=5, max_discard=3, shaping_beta=1.0):
        super().__init__()
        self.max_hand_size = max_hand_size
        self.shaping_beta = shaping_beta

        # 动作空间：多头决策 - Tuple of two separate heads
        self.action_space = spaces.Tuple((
            spaces.Discrete(2),                          # Head 1: 0=弃牌,1=出牌
            spaces.MultiBinary(self.max_hand_size)       # Head 2: 卡牌选择掩码
        ))

        # 默认牌型倍数
        self.mult = {
            "Straight Flush": 8,
            "Four of a Kind": 7,
            "Full House":     4,
            "Flush":          4,
            "Straight":       4,
            "Three of a Kind":3,
            "Two Pair":       2,
            "One Pair":       2,
            "High Card":      1,
        }

        # 默认基础分（当前未用到，可扩展）
        self.basic_score = {
            "Straight Flush": 100,
            "Four of a Kind": 60,
            "Full House":     40,
            "Flush":          35,
            "Straight":       30,
            "Three of a Kind":30,
            "Two Pair":       20,
            "One Pair":       10,
            "High Card":      5,
        }

        # 合并外部配置
        self.hand_multipliers = {**self.mult, **(hand_multipliers or {})}
        self.hand_basic_score = {**self.basic_score, **(hand_basic_score or {})}

        self.hand = []
        self.deck = {}
        self.played_cards = []
        self.discarded_cards = []
        self.max_play = max_play
        self.play_count = self.max_play
        self.max_discard = max_discard
        self.discard_count = self.max_discard
        self.step_history = []

        # 观测向量：手牌 one-hot 52 + 牌库可用 one-hot 52 + 出/弃归一化各1
        obs_dim = 52 + 52 + 1 + 1
        self.observation_space = spaces.Box(0.0, 1.0, shape=(obs_dim,), dtype=np.float32)

    # ---------- 核心接口 ----------
    def reset(self):
        self.deck = self._init_deck()
        self.hand = []
        self.played_cards = []
        self.discarded_cards = []
        self.play_count = self.max_play
        self.discard_count = self.max_discard
        self.step_history = []
        self.draw_card()
        return self._get_observation()

    def step(self, action):
        # 规范化动作：mask -> np.int8, 长度对齐 max_hand_size
        a_type, mask = action
        mask = np.asarray(mask, dtype=np.int8)
        if mask.shape[0] != self.max_hand_size:
            fixed = np.zeros(self.max_hand_size, dtype=np.int8)
            upto = min(self.max_hand_size, mask.shape[0])
            fixed[:upto] = mask[:upto]
            mask = fixed

        prev_obs = self._get_observation()
        prev_env_state = self.get_env_state()

        # 本次选中的牌（只取有效索引位）
        selected_cards = [self.hand[i] for i, m in enumerate(mask) if m and i < len(self.hand)]

        reward = 0.0
        if a_type == 0:  # 弃牌

            if self.discard_count > 0 and selected_cards:
                old_score, _ = self.best_hand_score()     # 弃牌前潜力
                self.discarded_cards.extend(selected_cards)
                self.update_hand(mask)
                self.draw_card()
                self.discard_count -= 1
                new_score, _ = self.best_hand_score()     # 弃牌后潜力
                delta = float(new_score - old_score)
                if delta <=0:
                    reward =0
                else:
                # 固定缩放常数（按你现在的分数量级很稳）
                   reward = self.shaping_beta * delta

        else:             # 出牌
            reward = float(self.best_mask_score(mask)) if selected_cards else 0.0
            self.played_cards.extend(selected_cards)
            self.update_hand(mask)
            self.draw_card()
            self.play_count -= 1

        obs = self._get_observation()
        no_play_left = (self.play_count <= 0)
        no_discard_left = (self.discard_count <= 0)
        no_hand = (len(self.hand) == 0)
        no_deck = all(cnt == 0 for cnt in self.deck.values())
        no_cards = no_hand and no_deck

        # *** 按你的要求：此行保持原样，不做任何修改 ***
        # done 为真当且仅当——出牌次数用尽 或者 弃牌次数用尽 或者 （手牌和牌库同时空）
        done = no_play_left or (no_discard_left and no_play_left) or no_cards

        step_info = {
            'state': prev_obs,
            'action': (int(a_type), mask.astype(int).tolist()),
            'reward': reward,
            'next_state': obs,
            'done': done,
            'env_state': prev_env_state,
            'selected_cards': selected_cards,
            'action_type': 'discard' if a_type == 0 else 'play'
        }
        self.step_history.append(step_info)

        return obs, reward, done, {}

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    # ---------- 牌堆 / 手牌 ----------
    def _init_deck(self):
        suits = ['H','D','C','S']
        ranks = list(range(1,14))
        return {(r,s):1 for s in suits for r in ranks}

    def update_hand(self, mask):
        to_remove = [self.hand[i] for i, m in enumerate(mask) if m and i < len(self.hand)]
        self.hand = [c for c in self.hand if c not in to_remove]

    def draw_card(self):
        # 抽至手牌满
        need = self.max_hand_size - len(self.hand)
        if need <= 0:
            return
        available = [card for card, cnt in self.deck.items() if cnt > 0]
        if not available:
            return
        k = min(need, len(available))
        drawn_cards = random.sample(available, k)
        for card in drawn_cards:
            self.deck[card] -= 1
            self.hand.append(card)

    # ---------- 计分 ----------
    def best_mask_score(self, mask):
        selected = [card for i, card in enumerate(self.hand) if i < len(self.hand) and mask[i]]
        if not selected:
            return 0

        def sum_ranks(cards):
            return sum(r for r,s in cards if isinstance(r, int))

        def is_flush(cards):
            if len(cards) != 5: return False
            suits = [s for _,s in cards]
            return len(set(suits)) == 1

        def is_straight(cards):
            if len(cards) != 5: return False
            vals = sorted({r for r,_ in cards if isinstance(r,int)})
            if len(vals) != 5: return False
            return vals == list(range(vals[0], vals[0]+5))

        best_score = 0
        for r in range(1, min(5, len(selected)) + 1):
            for combo in combinations(selected, r):
                cnt    = Counter(rv for rv,_ in combo if isinstance(rv,int))
                counts = sorted(cnt.values(), reverse=True)
                patterns = []

                if is_straight(combo) and is_flush(combo):
                    patterns.append(("Straight Flush", combo))
                if counts and counts[0] == 4:
                    four = cnt.most_common(1)[0][0]
                    patterns.append(("Four of a Kind", tuple(c for c in combo if c[0]==four)))
                if len(combo)==5 and counts[0]==3 and counts[1]==2:
                    patterns.append(("Full House", combo))
                if is_flush(combo):
                    patterns.append(("Flush", combo))
                if is_straight(combo):
                    patterns.append(("Straight", combo))
                if counts and counts[0] == 3:
                    three = cnt.most_common(1)[0][0]
                    patterns.append(("Three of a Kind", tuple(c for c in combo if c[0]==three)))
                pairs = [rv for rv,v in cnt.items() if v==2]
                if len(pairs) >= 2:
                    patterns.append(("Two Pair", tuple(c for c in combo if c[0] in pairs[:2])))
                if counts and counts[0] == 2:
                    one = cnt.most_common(1)[0][0]
                    patterns.append(("One Pair", tuple(c for c in combo if c[0]==one)))
                high = max((rv for rv,_ in combo if isinstance(rv,int)))
                patterns.append(("High Card",(next(c for c in combo if c[0]==high),)))

                for pat, pat_cards in patterns:
                    s = sum_ranks(pat_cards)
                    base = self.hand_basic_score.get(pat, 0)
                    mult = self.hand_multipliers.get(pat, 1)
                    score = (s + base) * mult   # <<< 核心修改：基础分先加上，再乘倍数
                    if score > best_score:
                        best_score = score

        return best_score

    def best_hand_score(self):
        selected = self.hand

        def sum_ranks(cards):
            return sum(r for r,s in cards if isinstance(r, int))

        def is_flush(cards):
            if len(cards) != 5: return False
            suits = [s for _,s in cards]
            return len(set(suits)) == 1

        def is_straight(cards):
            if len(cards) != 5: return False
            vals = sorted({r for r,_ in cards if isinstance(r,int)})
            if len(vals) != 5: return False
            return vals == list(range(vals[0], vals[0]+5))

        best_score = 0
        best_combo = []

        for r in range(1, min(5, len(selected)) + 1):
            for combo in combinations(selected, r):
                cnt    = Counter(rv for rv,_ in combo if isinstance(rv,int))
                counts = sorted(cnt.values(), reverse=True)
                patterns = []

                if is_straight(combo) and is_flush(combo):
                    patterns.append(("Straight Flush", combo))
                if counts and counts[0] == 4:
                    four = cnt.most_common(1)[0][0]
                    patterns.append(("Four of a Kind", tuple(c for c in combo if c[0]==four)))
                if len(combo)==5 and counts[0]==3 and counts[1]==2:
                    patterns.append(("Full House", combo))
                if is_flush(combo):
                    patterns.append(("Flush", combo))
                if is_straight(combo):
                    patterns.append(("Straight", combo))
                if counts and counts[0] == 3:
                    three = cnt.most_common(1)[0][0]
                    patterns.append(("Three of a Kind", tuple(c for c in combo if c[0]==three)))
                pairs = [rv for rv,v in cnt.items() if v==2]
                if len(pairs) >= 2:
                    patterns.append(("Two Pair", tuple(c for c in combo if c[0] in pairs[:2])))
                if counts and counts[0] == 2:
                    one = cnt.most_common(1)[0][0]
                    patterns.append(("One Pair", tuple(c for c in combo if c[0]==one)))
                high = max((rv for rv,_ in combo if isinstance(rv,int)))
                patterns.append(("High Card",(next(c for c in combo if c[0]==high),)))

                for pat, pat_cards in patterns:
                    s = sum_ranks(pat_cards)
                    score = s * self.mult[pat]
                    if score > best_score:
                        best_score = score
                        best_combo = list(pat_cards)

        return best_score, best_combo

    # ---------- 观测 ----------
    def _card_index(self, card):
        rank, suit = card
        suit_map = {'H':0,'D':1,'C':2,'S':3}
        return (rank-1)*4 + suit_map[suit]

    def _get_observation(self):
        # 1) 手牌 one-hot
        hand_mask = np.zeros(52, dtype=np.float32)
        for card in self.hand:
            hand_mask[self._card_index(card)] = 1.0

        # 2) 牌库 one-hot（是否可用）
        deck_mask = np.zeros(52, dtype=np.float32)
        for card, cnt in self.deck.items():
            deck_mask[self._card_index(card)] = 1.0 if cnt > 0 else 0.0

        # 3) 出/弃次数（归一化）
        play_feat = np.array([self.play_count / self.max_play], dtype=np.float32)
        discard_feat = np.array([self.discard_count / self.max_discard], dtype=np.float32)

        return np.concatenate([hand_mask, deck_mask, play_feat, discard_feat])

    # ---------- 辅助 ----------
    def get_env_state(self):
        return {
            'hand': self.hand.copy(),
            'played_cards': self.played_cards.copy(),
            'discarded_cards': self.discarded_cards.copy(),
            'deck': self.deck.copy(),
            'play_count': self.play_count,
            'discard_count': self.discard_count,
            'max_play': self.max_play,
            'max_discard': self.max_discard
        }

    def get_step_history(self):
        return self.step_history.copy()

    def save_episode_data(self, filepath=None):
        import json
        import datetime
        if filepath is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"episode_data_{timestamp}.json"
        episode_data = {
            'step_history': self.step_history,
            'final_env_state': self.get_env_state(),
            'total_steps': len(self.step_history)
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(episode_data, f, indent=2, ensure_ascii=False)
        return filepath

    def clear_history(self):
        self.step_history = []

