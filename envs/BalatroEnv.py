# Simplified BalatroEnv —— 单局、无盲注、手牌评分
import gym
from gym import spaces
import numpy as np
import itertools
import random
from itertools import combinations
from collections import Counter

class BalatroEnv(gym.Env):
    metadata = {'render.modes':['human','rgb_array']}
    def __init__(self, max_hand_size=8, hand_multipliers=None, hand_basic_score = None, max_play = 5, max_discard = 3):
        super().__init__()
        self.max_hand_size = max_hand_size     
        # 动作空间：多头决策 - Tuple of two separate heads
        self.action_space = spaces.Tuple((
            spaces.Discrete(2),                          # Head 1: 0=弃牌,1=出牌
            spaces.MultiBinary(self.max_hand_size)       # Head 2: 卡牌选择掩码
        ))

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

        # 合并：外部传的会覆盖默认
        self.hand_multipliers = {**self.mult, **(hand_multipliers or {})}
        self.hand_basic_score = {**self.basic_score, **(hand_basic_score or {})}
        self.hand = []
        self.deck = {}
        self.max_play = max_play
        self.play_count = self.max_play
        self.max_discard = max_discard
        self.discard_count = self.max_discard
        obs_dim = 52 + 52 + 1 + 1
        self.observation_space = spaces.Box(0.0, 1.0, shape=(obs_dim,), dtype=np.float32)

    def reset(self):
        # 1) 生成并洗牌
        self.deck = self._init_deck()
        self.hand = []  
        self.play_count = self.max_play
        self.discard_count = self.max_discard
        # 2) 发手牌
        self.draw_card()
        return self._get_observation()
    
    def step(self, action):
        # action 是一个 tuple: (action_type, mask)
        a_type = action[0]    # Head 1: 0 或 1
        mask   = action[1]    # Head 2: array([0,1,0,1,0,0,0], dtype=int8)
        reward = 0
        '''
        if a_type == 0:
            old_score, _ = self.best_hand_score()
        else:
            old_score = 0
        '''
        if a_type == 0:   # 弃牌
            if self.discard_count > 0:
                self.update_hand(mask)
                self.draw_card()
                self.discard_count -= 1
            # —— 新增：弃牌后的最优潜力分 —— 
            #new_score, _ = self.best_hand_score()
            # reward 设为分差
            #reward = new_score - old_score
            #self.update_hand(mask)
            #self.draw_card()

        else:             # 出牌
            # 从手牌移到公共区
            reward = self.best_mask_score(mask)
            self.update_hand(mask)
            self.draw_card()
            self.play_count -= 1
        obs = self._get_observation()
        no_play_left = (self.play_count <= 0)
        no_discard_left = (self.discard_count <= 0)
        # 条件二：手牌和牌库都为空
        no_hand = (len(self.hand) == 0)
        no_deck = all(cnt == 0 for cnt in self.deck.values())
        no_cards = no_hand and no_deck
        # done 为真当且仅当——出牌次数用尽 或者 弃牌次数用尽 或者 （手牌和牌库同时空）
        done = no_play_left or (no_discard_left and no_play_left) or no_cards
        return obs, reward, done, {}

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def _init_deck(self):
        suits = ['H','D','C','S']
        ranks = list(range(1,14))
        deck = {(r,s):1 for s in suits for r in ranks}
        return deck
    
    def _init_hand(self):
        self.hand = []
    
    def update_hand(self, mask):
        # 只移除那些既被 mask 标记，又确实在 hand 范围内的牌
        to_remove = [
            self.hand[i] 
            for i, m in enumerate(mask) 
            if m and i < len(self.hand)
        ]
        # 把这些牌从手牌里剔除
        self.hand = [c for c in self.hand if c not in to_remove]

    def draw_card(self):
        #抽完卡片后更新
        number = self.max_hand_size - len(self.hand)
        available = [card for card, cnt in self.deck.items() if cnt > 0]
       # 实际抽取数量不能超过可用牌
        k = min(number, len(available))
        # 随机抽 k 张
        drawn_cards = random.sample(available, k)
        # 更新 deck 和 hand
        for card in drawn_cards:
            self.deck[card] -= 1
            self.hand.append(card)

    def best_mask_score(self, mask):
        selected = [card for card, m in zip(self.hand, mask) if m]
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

        # 枚举所有 1–5 张子集
        for r in range(1, min(5, len(selected)) + 1):
            for combo in combinations(selected, r):
                cnt    = Counter(r for r,_ in combo if isinstance(r,int))
                counts = sorted(cnt.values(), reverse=True)
                patterns = []

                # 收集所有符合的牌型和它们的“构成牌”
                if is_straight(combo) and is_flush(combo):
                    patterns.append(("Straight Flush", combo))
                if counts and counts[0] == 4:
                    four = cnt.most_common(1)[0][0]
                    patterns.append(("Four of a Kind",
                                    tuple(c for c in combo if c[0]==four)))
                if len(combo)==5 and counts[0]==3 and counts[1]==2:
                    patterns.append(("Full House", combo))
                if is_flush(combo):
                    patterns.append(("Flush", combo))
                if is_straight(combo):
                    patterns.append(("Straight", combo))
                if counts and counts[0] == 3:
                    three = cnt.most_common(1)[0][0]
                    patterns.append(("Three of a Kind",
                                    tuple(c for c in combo if c[0]==three)))
                pairs = [r for r,v in cnt.items() if v==2]
                if len(pairs)>=2:
                    patterns.append(("Two Pair",
                                    tuple(c for c in combo if c[0] in pairs[:2])))
                if counts and counts[0] == 2:
                    one = cnt.most_common(1)[0][0]
                    patterns.append(("One Pair",
                                    tuple(c for c in combo if c[0]==one)))
                # 高牌一定有，取点数最大的那张
                high = max((r for r,_ in combo if isinstance(r,int)))
                patterns.append(("High Card",
                                (next(c for c in combo if c[0]==high),)))

                # 对每个牌型只给它构成该牌型的牌算分
                for pat, pat_cards in patterns:
                    s = sum_ranks(pat_cards)
                    score = s * self.mult[pat]
                    if score > best_score:
                        best_score = score

        # 一定要在所有子集都跑完后再返回
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
                cnt    = Counter(r for r,_ in combo if isinstance(r,int))
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
                pairs = [r for r,v in cnt.items() if v==2]
                if len(pairs) >= 2:
                    patterns.append(("Two Pair", tuple(c for c in combo if c[0] in pairs[:2])))
                if counts and counts[0] == 2:
                    one = cnt.most_common(1)[0][0]
                    patterns.append(("One Pair", tuple(c for c in combo if c[0]==one)))
                high = max((r for r,_ in combo if isinstance(r,int)))
                patterns.append(("High Card", (next(c for c in combo if c[0]==high),)))

                for pat, pat_cards in patterns:
                    s = sum_ranks(pat_cards)
                    score = s * self.mult[pat]
                    if score > best_score:
                        best_score = score
                        best_combo = list(pat_cards)

        return best_score, best_combo
    
    def _card_index(self, card):
        # 把 (rank,suit) 映射到 0–53；JOKER→52,53
        rank, suit = card
        suit_map = {'H':0,'D':1,'C':2,'S':3}
        return (rank-1)*4 + suit_map[suit]
    
    def _get_observation(self):
        # 1) 手牌 one-hot
        hand_mask = np.zeros(52, dtype=np.float32)
        for card in self.hand:
            idx = self._card_index(card)
            hand_mask[idx] = 1.0

        # 2) 牌库 one-hot
        deck_mask = np.zeros(52, dtype=np.float32)
        for card, cnt in self.deck.items():
            idx = self._card_index(card)
            deck_mask[idx] = 1.0 if cnt > 0 else 0.0

        # 3) 剩余可出牌次数（归一化到 [0,1]）
        play_feat = np.array([self.play_count / self.max_play], dtype=np.float32)
        # 4) 剩余弃牌次数（归一化到 [0,1]）
        discard_feat = np.array([self.discard_count / self.max_discard], dtype=np.float32)
        # 拼成一个向量返回
        return np.concatenate([hand_mask, deck_mask, play_feat, discard_feat])
    
    def print_self(self):
        print("手牌:", self.hand)
        print("剩余牌:", self.deck)
        print(f"剩余出牌次数: {self.play_count}")
        print(f"剩余弃牌次数: {self.discard_count}")
        
# 测试多头决策动作空间
if __name__ == "__main__":
    env = BalatroEnv()
    obs = env.reset()
    env.print_self()
    
    print(f"动作空间: {env.action_space}")
    print(f"观测空间形状: {env.observation_space.shape}")
    
    # 测试弃牌动作 (action_type=0, 弃掉前两张牌)
    action_discard = (0, [1, 1, 0, 0, 0, 0, 0, 0])
    print(f"\n执行弃牌动作: {action_discard}")
    obs, reward, done, info = env.step(action_discard)
    print(f"奖励: {reward}, 游戏结束: {done}")
    env.print_self()
    
    # 再次测试弃牌动作
    action_discard2 = (0, [1, 0, 0, 0, 0, 0, 0, 0])
    print(f"\n再次执行弃牌动作: {action_discard2}")
    obs, reward, done, info = env.step(action_discard2)
    print(f"奖励: {reward}, 游戏结束: {done}")
    env.print_self()
    
    # 测试出牌动作 (action_type=1, 出前三张牌)  
    action_play = (1, [1, 1, 1, 0, 0, 0, 0, 0])
    print(f"\n执行出牌动作: {action_play}")
    obs, reward, done, info = env.step(action_play)
    print(f"奖励: {reward}, 游戏结束: {done}")
    env.print_self()
    
    score, combo = env.best_hand_score()
    print(f"当前手牌最优组合得分: {score}")
    print(f"最佳出牌组合: {combo}")

