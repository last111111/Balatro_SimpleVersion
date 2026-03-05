# -*- coding: utf-8 -*-
# Simplified BalatroEnv —— 单局、无盲注、手牌评分（保留原 done 行，其它增强）
import gym
from gym import spaces
import numpy as np
import random
from itertools import combinations
from collections import Counter


# ============================================================
# 小丑牌系统
# ============================================================

class JokerType:
    """小丑牌类型常量（30种）"""
    JOKER = 0
    GREEDY_JOKER = 1        # +3 Mult per Diamond scored
    LUSTY_JOKER = 2         # +3 Mult per Heart scored
    WRATHFUL_JOKER = 3      # +3 Mult per Spade scored
    GLUTTONOUS_JOKER = 4    # +3 Mult per Club scored
    JOLLY_JOKER = 5         # +8 Mult if hand has a Pair
    ZANY_JOKER = 6          # +12 Mult if hand has Three of a Kind
    MAD_JOKER = 7           # +10 Mult if hand has Two Pair
    CRAZY_JOKER = 8         # +12 Mult if hand has Straight
    DROLL_JOKER = 9         # +10 Mult if hand has Flush
    HALF_JOKER = 10         # +20 Mult if hand <= 3 cards
    STEEL_JOKER = 11        # +0.2 X Mult per Steel card (需要card enhancement)
    JOKER_STENCIL = 12      # X1 Mult per empty Joker slot
    FOUR_FINGERS = 13       # Flushes/Straights可用4张牌（影响牌型评估）
    BANNER = 14             # +30 Chips per discard remaining
    MYSTIC_SUMMIT = 15      # +15 Mult if 0 discards remaining
    MISPRINT = 16           # +? Mult (random 0-23)
    RAISED_FIST = 17        # Adds 2x rank of lowest held card to Mult
    FIBONACCI = 18          # +8 Mult for each A,2,3,5,8 scored
    EVEN_STEVEN = 19        # +4 Mult for each even rank scored
    ODD_TODD = 20           # +31 Chips for each odd rank scored
    BLACKBOARD = 21         # X3 Mult if all held cards are Spades or Clubs
    ICE_CREAM = 22          # +100 Chips, loses 5 chips per round
    BLUE_JOKER = 23         # +2 Chips per remaining card in deck
    RUNNER = 24             # +15 Chips if hand has Straight (grows +15)
    SUPERNOVA = 25          # +Mult equal to times hand type played this run
    RIDE_THE_BUS = 26       # +1 Mult per consecutive hand without face card
    SPARE_TROUSERS = 27     # +2 Mult if hand has Two Pair (grows +2)
    ABSTRACT_JOKER = 28     # +3 Mult per Joker owned
    LOYALTY_CARD = 29       # X4 Mult every 6 hands played


class Joker:
    """小丑牌实例"""
    def __init__(self, joker_type):
        self.joker_type = joker_type
        # 可变状态（某些小丑牌会增长）
        self.counter = 0         # 通用计数器（如 RIDE_THE_BUS 的连续次数）
        self.extra_chips = 0     # 额外chips（如 ICE_CREAM, RUNNER）
        self.extra_mult = 0.0    # 额外mult（如 SPARE_TROUSERS）


class BalatroEnv(gym.Env):
    metadata = {'render.modes':['human','rgb_array']}

    def __init__(self, max_hand_size=8, hand_multipliers=None, hand_basic_score=None, max_play=5, max_discard=3, shaping_beta=0.3, discard_cost=-0.5, mc_sims=30):
        super().__init__()
        self.max_hand_size = max_hand_size
        self.shaping_beta = shaping_beta
        self.discard_cost = discard_cost
        self.mc_sims = mc_sims  # 弃牌奖励的蒙特卡洛模拟次数

        # 动作空间：集合式设计 - 选择52张牌空间中的具体牌
        self.action_space = spaces.Tuple((
            spaces.Discrete(2),                          # Head 1: 0=弃牌,1=出牌
            spaces.MultiBinary(52)                       # Head 2: 选择52张牌空间中的牌（而非位置）
        ))

        # 默认牌型倍数
        self.mult = {
            "Flush Five":     16,
            "Flush House":    14,
            "Five of a Kind": 12,
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

        # 默认基础分
        self.basic_score = {
            "Flush Five":     160,
            "Flush House":    140,
            "Five of a Kind": 120,
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

        # 小丑牌系统
        self.joker_slots = 5  # 最多5张小丑牌
        self.jokers = []      # List[Joker]，按位置顺序存储（当前训练为空）
        self.total_hands_played = 0  # 全局出牌计数（LOYALTY_CARD需要）
        self.hands_played_this_run = {}  # {hand_type_str: count}（SUPERNOVA需要）

        # M2: 增加历史信息；M3: 扩展 Joker 编码
        # 观测向量维度计算：
        # - 手牌 one-hot: 52
        # - 牌库可用 one-hot: 52
        # - 已出过的牌 one-hot: 52       (M2)
        # - 已弃过的牌 one-hot: 52       (M2)
        # - 出牌次数归一化: 1
        # - 弃牌次数归一化: 1
        # - 累计得分归一化: 1            (M2)
        # - 小丑牌 (type, chips, mult) × 5: 15  (M3)
        obs_dim = 52 + 52 + 52 + 52 + 1 + 1 + 1 + 15
        self.observation_space = spaces.Box(0.0, 1.0, shape=(obs_dim,), dtype=np.float32)

        self.cumulative_score = 0.0

    # ---------- 核心接口 ----------
    def reset(self):
        self.deck = self._init_deck()
        self.hand = []
        self.played_cards = []
        self.discarded_cards = []
        self.play_count = self.max_play
        self.discard_count = self.max_discard
        self.step_history = []
        self.cumulative_score = 0.0  # M2

        # 重置小丑牌系统（当前训练保持为空）
        self.jokers = []
        self.total_hands_played = 0
        self.hands_played_this_run = {}

        self.draw_card()
        return self._get_observation()

    def reset_with_jokers(self, joker_ids):
        """带小丑牌的 reset：先正常 reset，再注入指定的小丑牌"""
        self.reset()
        for jid in joker_ids:
            self.add_joker(jid)
        return self._get_observation()

    def step(self, action):
        # 新的动作格式：(a_type, card_mask)
        # card_mask: 52维，表示选中52张牌空间中的哪些牌
        a_type, card_mask = action
        card_mask = np.asarray(card_mask, dtype=np.int8)

        # 确保card_mask是52维
        if card_mask.shape[0] != 52:
            fixed = np.zeros(52, dtype=np.int8)
            upto = min(52, card_mask.shape[0])
            fixed[:upto] = card_mask[:upto]
            card_mask = fixed

        prev_obs = self._get_observation()
        prev_env_state = self.get_env_state()

        # 将card_mask映射到实际手牌：只选择手中有的牌
        selected_cards = []
        for card_idx in range(52):
            if card_mask[card_idx]:
                card = self._index_to_card(card_idx)
                if card in self.hand:
                    selected_cards.append(card)

        # M5: 出牌时截断为最多 5 张，并对超选施加惩罚
        overselect_penalty = 0.0
        if a_type == 1 and len(selected_cards) > 5:
            overselect_penalty = -1.0 * (len(selected_cards) - 5)  # 每多选一张扣1分
            selected_cards = selected_cards[:5]

        reward = overselect_penalty
        if a_type == 0:  # 弃牌

            if self.discard_count <= 0:
                # H7: 弃牌次数已用完，惩罚并强制转为出牌
                reward = -10.0
                a_type = 1
                # 下面会走出牌逻辑
            elif not selected_cards:
                # 空弃牌：没选任何牌，强制转为出牌（防止无限循环）
                a_type = 1
            elif selected_cards:
                old_score, _ = self.best_hand_score()     # 弃牌前潜力

                # MC模拟：估计弃牌的期望得分提升
                remaining_hand = [c for c in self.hand if c not in selected_cards]
                available_draw = [card for card, cnt in self.deck.items()
                                  if cnt > 0 and card not in selected_cards]
                n_to_draw = min(len(selected_cards), len(available_draw),
                                self.max_hand_size - len(remaining_hand))

                if self.mc_sims > 0 and n_to_draw > 0 and available_draw:
                    sim_scores = []
                    for _ in range(self.mc_sims):
                        drawn = random.sample(available_draw, n_to_draw)
                        sim_hand = remaining_hand + drawn
                        sim_scores.append(self._calculate_best_score(sim_hand) if sim_hand else 0)
                    expected_new_score = float(np.mean(sim_scores))
                else:
                    # 无牌可抽时直接用剩余手牌评分
                    expected_new_score = float(self._calculate_best_score(remaining_hand)) if remaining_hand else 0.0

                delta = expected_new_score - float(old_score)

                # 实际执行弃牌
                self.discarded_cards.extend(selected_cards)
                self.update_hand_by_cards(selected_cards)
                self.draw_card()
                self.discard_count -= 1

                # 弃牌奖励：对称缩放 + 固有代价
                reward = self.shaping_beta * delta + self.discard_cost

        if a_type == 1:   # 出牌（包括从弃牌强制转换的情况）
            if selected_cards:
                raw_score, hand_type_name = self._calculate_best_score(selected_cards, return_hand_type=True)
            else:
                raw_score, hand_type_name = 0.0, "High Card"
            raw_score = float(raw_score)
            # 出牌奖励：直接使用原始分数
            play_reward = raw_score
            reward = reward + play_reward if reward != 0.0 else play_reward

            # L3: 更新牌型统计（SUPERNOVA 等 Joker 需要）
            if selected_cards:
                self.total_hands_played += 1
                self.hands_played_this_run[hand_type_name] = \
                    self.hands_played_this_run.get(hand_type_name, 0) + 1

                # 出牌后的小丑牌维护
                self._post_play_joker_maintenance()

            self.played_cards.extend(selected_cards)
            self.update_hand_by_cards(selected_cards)
            self.draw_card()
            self.play_count -= 1
            self.cumulative_score += raw_score  # M2: 累计原始得分

        obs = self._get_observation()
        no_play_left = (self.play_count <= 0)
        no_hand = (len(self.hand) == 0)
        no_deck = all(cnt == 0 for cnt in self.deck.values())
        no_cards = no_hand and no_deck

        # L1: 简化冗余 done 条件
        done = no_play_left or no_cards

        step_info = {
            'state': prev_obs,
            'action': (int(a_type), card_mask.astype(int).tolist()),
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
        """旧方法：按位置掩码移除牌（保留用于兼容）"""
        to_remove = [self.hand[i] for i, m in enumerate(mask) if m and i < len(self.hand)]
        self.hand = [c for c in self.hand if c not in to_remove]

    def update_hand_by_cards(self, cards_to_remove):
        """新方法：按具体牌移除"""
        self.hand = [c for c in self.hand if c not in cards_to_remove]

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
        """旧方法：按位置掩码计算得分（保留用于兼容）"""
        selected = [card for i, card in enumerate(self.hand) if i < len(self.hand) and mask[i]]
        if not selected:
            return 0
        return self._calculate_best_score(selected)

    def best_cards_score(self, cards):
        """新方法：直接按具体牌计算得分"""
        if not cards:
            return 0
        return self._calculate_best_score(cards)

    def _calculate_best_score(self, selected, return_hand_type=False):
        """
        通用的计分逻辑（使用小丑牌系统）
        遍历所有可能的组合，找到得分最高的牌型
        如果 return_hand_type=True，返回 (score, hand_type_name)
        """

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
        best_hand_type = "High Card"
        for r in range(1, min(5, len(selected)) + 1):
            for combo in combinations(selected, r):
                cnt    = Counter(rv for rv,_ in combo if isinstance(rv,int))
                counts = sorted(cnt.values(), reverse=True)
                patterns = []

                # Check for advanced hand types (following Balatro's hand ranking order)
                has_five = counts and counts[0] >= 5
                has_four = counts and counts[0] >= 4
                has_three = counts and counts[0] >= 3
                pair_count = sum(1 for c in counts if c >= 2)
                has_full_house = has_three and pair_count >= 2

                # Flush Five: Five of a Kind + Flush
                if has_five and is_flush(combo):
                    five = cnt.most_common(1)[0][0]
                    patterns.append(("Flush Five", tuple(c for c in combo if c[0]==five)))

                # Flush House: Full House + Flush
                if has_full_house and is_flush(combo):
                    patterns.append(("Flush House", combo))

                # Five of a Kind
                if has_five:
                    five = cnt.most_common(1)[0][0]
                    patterns.append(("Five of a Kind", tuple(c for c in combo if c[0]==five)))

                # Straight Flush
                if is_straight(combo) and is_flush(combo):
                    patterns.append(("Straight Flush", combo))

                # Four of a Kind
                if has_four:
                    four = cnt.most_common(1)[0][0]
                    patterns.append(("Four of a Kind", tuple(c for c in combo if c[0]==four)))

                # Full House
                if has_full_house:
                    patterns.append(("Full House", combo))

                # Flush
                if is_flush(combo):
                    patterns.append(("Flush", combo))

                # Straight
                if is_straight(combo):
                    patterns.append(("Straight", combo))

                # Three of a Kind
                if has_three:
                    three = cnt.most_common(1)[0][0]
                    patterns.append(("Three of a Kind", tuple(c for c in combo if c[0]==three)))

                # Two Pair
                pairs = [rv for rv,v in cnt.items() if v==2]
                if len(pairs) >= 2:
                    patterns.append(("Two Pair", tuple(c for c in combo if c[0] in pairs[:2])))

                # One Pair
                if counts and counts[0] == 2:
                    one = cnt.most_common(1)[0][0]
                    patterns.append(("One Pair", tuple(c for c in combo if c[0]==one)))

                # High Card
                high = max((rv for rv,_ in combo if isinstance(rv,int)))
                patterns.append(("High Card",(next(c for c in combo if c[0]==high),)))

                for pat, pat_cards in patterns:
                    # 使用新的小丑牌算分系统
                    score = self._calculate_score_with_jokers(
                        played_cards=list(combo),
                        hand_type_name=pat,
                        scoring_cards=list(pat_cards)
                    )
                    if score > best_score:
                        best_score = score
                        best_hand_type = pat

        if return_hand_type:
            return best_score, best_hand_type
        return best_score

    def best_hand_score(self):
        """L2: 复用 _calculate_best_score 去重，返回 (best_score, best_combo)"""
        if not self.hand:
            return 0, []
        # 使用 joker-aware 评分（与 best_cards_score 一致）
        best_score = self._calculate_best_score(self.hand)
        # 简化：不再单独追踪 best_combo（弃牌奖励只需 score）
        return best_score, []

    # ---------- 观测 ----------
    def _card_index(self, card):
        """将牌(rank, suit)转换为0-51的索引"""
        rank, suit = card
        suit_map = {'H':0,'D':1,'C':2,'S':3}
        return (rank-1)*4 + suit_map[suit]

    def _index_to_card(self, idx):
        """将0-51的索引转换为牌(rank, suit)"""
        suits = ['H', 'D', 'C', 'S']
        rank = idx // 4 + 1
        suit = suits[idx % 4]
        return (rank, suit)

    def _get_observation(self):
        # 1) 手牌 one-hot (52)
        hand_mask = np.zeros(52, dtype=np.float32)
        for card in self.hand:
            hand_mask[self._card_index(card)] = 1.0

        # 2) 牌库 one-hot (52)
        deck_mask = np.zeros(52, dtype=np.float32)
        for card, cnt in self.deck.items():
            deck_mask[self._card_index(card)] = 1.0 if cnt > 0 else 0.0

        # M2: 3) 已出过的牌 one-hot (52)
        played_mask = np.zeros(52, dtype=np.float32)
        for card in self.played_cards:
            played_mask[self._card_index(card)] = 1.0

        # M2: 4) 已弃过的牌 one-hot (52)
        discarded_mask = np.zeros(52, dtype=np.float32)
        for card in self.discarded_cards:
            discarded_mask[self._card_index(card)] = 1.0

        # 5) 出/弃次数（归一化）
        play_feat = np.array([self.play_count / self.max_play], dtype=np.float32)
        discard_feat = np.array([self.discard_count / self.max_discard], dtype=np.float32)

        # M2: 6) 累计得分（归一化）
        score_feat = np.array([min(1.0, self.cumulative_score / 1000.0)], dtype=np.float32)

        # M3: 7) 小丑牌槽位编码（type + chips + mult per slot）
        joker_features = self._encode_jokers()

        return np.concatenate([hand_mask, deck_mask, played_mask, discarded_mask,
                               play_feat, discard_feat, score_feat, joker_features])

    def _encode_jokers(self):
        """
        M3: 编码小丑牌槽位信息（type + state）
        每个 slot 3 维: (type_id/29, extra_chips/200, extra_mult/20)
        返回 15 维向量 (5 slots × 3 features)
        """
        features_per_slot = 3
        joker_features = np.zeros(self.joker_slots * features_per_slot, dtype=np.float32)

        for i in range(min(len(self.jokers), self.joker_slots)):
            base = i * features_per_slot
            joker_features[base] = self.jokers[i].joker_type / 29.0
            joker_features[base + 1] = min(1.0, max(0.0, self.jokers[i].extra_chips / 200.0))
            joker_features[base + 2] = min(1.0, max(0.0, self.jokers[i].extra_mult / 20.0))

        return joker_features

    # ---------- 小丑牌管理接口 ----------
    def add_joker(self, joker_type):
        """添加小丑牌（用于测试和未来扩展）"""
        if len(self.jokers) < self.joker_slots:
            self.jokers.append(Joker(joker_type))
            return True
        return False

    def remove_joker(self, index):
        """移除指定位置的小丑牌"""
        if 0 <= index < len(self.jokers):
            self.jokers.pop(index)
            return True
        return False

    def clear_jokers(self):
        """清空所有小丑牌"""
        self.jokers = []

    def _post_play_joker_maintenance(self):
        """出牌后的小丑牌维护（如ICE_CREAM衰减）"""
        for joker in self.jokers:
            if joker.joker_type == JokerType.ICE_CREAM:
                joker.extra_chips -= 5

    # ---------- 辅助 ----------
    def _get_card_chips(self, card):
        """获取单张牌的chips值"""
        rank = card[0]
        if rank >= 10:  # J, Q, K
            return 10
        elif rank == 1:  # A
            return 11
        else:
            return rank

    def _rank_to_chips(self, rank):
        """将rank转换为chips值（用于RAISED_FIST等）"""
        if rank >= 10:
            return 10
        elif rank == 1:
            return 11
        else:
            return rank

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

    # ---------- 小丑牌算分系统 ----------
    _ADDITIVE_CHIPS_JOKERS = {14, 20, 22, 23, 24}   # Banner, Odd Todd, Ice Cream, Blue Joker, Runner
    _MULTIPLICATIVE_MULT_JOKERS = {11, 12, 21, 29}   # Steel, Stencil, Blackboard, Loyalty Card

    def _sort_jokers_for_scoring(self):
        """算分前自动排序：additive chips → additive mult → multiplicative mult"""
        def priority(joker):
            jtype = joker.joker_type
            if jtype in self._ADDITIVE_CHIPS_JOKERS:
                return 0
            elif jtype in self._MULTIPLICATIVE_MULT_JOKERS:
                return 2
            return 1
        self.jokers.sort(key=priority)

    def _calculate_score_with_jokers(self, played_cards, hand_type_name, scoring_cards):
        """
        完整的Balatro算分系统（包含小丑牌）

        Args:
            played_cards: 打出的所有牌
            hand_type_name: 牌型名称字符串（如 "One Pair"）
            scoring_cards: 参与计分的牌

        Returns:
            int: 最终得分
        """
        # 1. 基础chips和mult
        base_chips = self.hand_basic_score.get(hand_type_name, 0)
        base_mult = self.hand_multipliers.get(hand_type_name, 1)

        chips = float(base_chips)
        mult = float(base_mult)

        # 2. 加上scoring cards的chips
        for card in scoring_cards:
            chips += self._get_card_chips(card)

        # 3. 自动排序后依次应用小丑牌效果（从左到右）
        self._sort_jokers_for_scoring()
        for joker in self.jokers:
            chips, mult = self._apply_joker_effect(
                joker, chips, mult,
                played_cards, hand_type_name, scoring_cards
            )

        # 4. 最终得分
        score = int(chips * mult)
        return score

    def _apply_joker_effect(self, joker, chips, mult, played_cards, hand_type_name, scoring_cards):
        """
        应用单个小丑牌的效果（30种完整实现）

        Args:
            joker: Joker 实例
            chips: 当前chips值
            mult: 当前mult值
            played_cards: 打出的所有牌
            hand_type_name: 牌型名称
            scoring_cards: 参与计分的牌

        Returns:
            (chips, mult): 修改后的值
        """
        jt = joker.joker_type

        # 基础类
        if jt == JokerType.JOKER:
            mult += 4

        # 花色类（+mult per suit scored）
        elif jt == JokerType.GREEDY_JOKER:  # Diamonds
            for card in scoring_cards:
                if card[1] == 'D':
                    mult += 3

        elif jt == JokerType.LUSTY_JOKER:  # Hearts
            for card in scoring_cards:
                if card[1] == 'H':
                    mult += 3

        elif jt == JokerType.WRATHFUL_JOKER:  # Spades
            for card in scoring_cards:
                if card[1] == 'S':
                    mult += 3

        elif jt == JokerType.GLUTTONOUS_JOKER:  # Clubs
            for card in scoring_cards:
                if card[1] == 'C':
                    mult += 3

        # 牌型触发类
        elif jt == JokerType.JOLLY_JOKER:  # Pair
            if hand_type_name in ["One Pair", "Two Pair", "Full House",
                                  "Four of a Kind", "Five of a Kind",
                                  "Flush House", "Flush Five"]:
                mult += 8

        elif jt == JokerType.ZANY_JOKER:  # Three of a Kind
            if hand_type_name in ["Three of a Kind", "Full House",
                                  "Four of a Kind", "Five of a Kind",
                                  "Flush House", "Flush Five"]:
                mult += 12

        elif jt == JokerType.MAD_JOKER:  # Two Pair
            if hand_type_name in ["Two Pair", "Full House", "Flush House"]:
                mult += 10

        elif jt == JokerType.CRAZY_JOKER:  # Straight
            if hand_type_name in ["Straight", "Straight Flush"]:
                mult += 12

        elif jt == JokerType.DROLL_JOKER:  # Flush
            if hand_type_name in ["Flush", "Straight Flush",
                                  "Flush House", "Flush Five"]:
                mult += 10

        # 手牌数量类
        elif jt == JokerType.HALF_JOKER:  # <= 3 cards
            if len(played_cards) <= 3:
                mult += 20

        # 资源依赖类
        elif jt == JokerType.BANNER:  # +30 chips per discard remaining
            chips += 30 * self.discard_count

        elif jt == JokerType.MYSTIC_SUMMIT:  # +15 mult if 0 discards
            if self.discard_count == 0:
                mult += 15

        elif jt == JokerType.BLUE_JOKER:  # +2 chips per card in deck
            deck_size = sum(1 for card, cnt in self.deck.items() if cnt > 0)
            chips += 2 * deck_size

        # 随机类
        elif jt == JokerType.MISPRINT:  # +0-23 mult
            mult += random.randint(0, 23)

        # 手牌依赖类
        elif jt == JokerType.RAISED_FIST:  # +2x lowest rank in hand
            if self.hand:
                lowest_rank = min(card[0] for card in self.hand)
                mult += 2 * self._rank_to_chips(lowest_rank)

        # 点数类
        elif jt == JokerType.FIBONACCI:  # A,2,3,5,8
            fib_ranks = {1, 2, 3, 5, 8}
            for card in scoring_cards:
                if card[0] in fib_ranks:
                    mult += 8

        elif jt == JokerType.EVEN_STEVEN:  # Even ranks
            for card in scoring_cards:
                if card[0] % 2 == 0:
                    mult += 4

        elif jt == JokerType.ODD_TODD:  # Odd ranks
            for card in scoring_cards:
                if card[0] % 2 == 1:
                    chips += 31

        # ×mult 类（立即乘到mult）
        elif jt == JokerType.BLACKBOARD:  # ×3 if all spades/clubs
            if self.hand and all(card[1] in ('S', 'C') for card in self.hand):
                mult *= 3.0

        elif jt == JokerType.LOYALTY_CARD:  # ×4 every 6 hands
            if self.total_hands_played > 0 and self.total_hands_played % 6 == 0:
                mult *= 4.0

        elif jt == JokerType.JOKER_STENCIL:  # ×N per empty joker slot
            empty_slots = self.joker_slots - len(self.jokers)
            if empty_slots > 0:
                mult *= float(empty_slots)

        # 成长类（带状态）
        elif jt == JokerType.ICE_CREAM:  # +chips (decays)
            chips += max(0, 100 + joker.extra_chips)

        elif jt == JokerType.RUNNER:  # +chips if straight (grows)
            if hand_type_name in ["Straight", "Straight Flush"]:
                joker.extra_chips += 15
            chips += joker.extra_chips

        elif jt == JokerType.RIDE_THE_BUS:  # +mult if no face (grows)
            has_face = any(card[0] >= 11 for card in scoring_cards)
            if has_face:
                joker.counter = 0
            else:
                joker.counter += 1
            mult += joker.counter

        elif jt == JokerType.SPARE_TROUSERS:  # +mult if two pair (grows)
            if hand_type_name in ["Two Pair", "Full House", "Flush House"]:
                joker.extra_mult += 2
            mult += joker.extra_mult

        elif jt == JokerType.SUPERNOVA:  # +mult per time hand played
            times_played = self.hands_played_this_run.get(hand_type_name, 0)
            mult += times_played

        # 小丑牌依赖类
        elif jt == JokerType.ABSTRACT_JOKER:  # +3 mult per joker
            mult += 3 * len(self.jokers)

        # 特殊规则类（暂不实现或不影响算分）
        elif jt == JokerType.STEEL_JOKER:
            # 需要card enhancement系统
            pass

        elif jt == JokerType.FOUR_FINGERS:
            # 影响牌型评估，不是算分
            pass

        return chips, mult

