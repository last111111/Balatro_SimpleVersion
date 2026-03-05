"""
小丑牌算分系统完整实现
这个文件包含需要添加到 BalatroEnv.py 的所有方法
"""

# ============================================================
# 第一部分：辅助方法（添加到 BalatroEnv 类中）
# ============================================================

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


def _map_hand_type_to_str(self, hand_type_name):
    """
    将内部牌型名称映射为标准字符串
    用于 hands_played_this_run 的统计
    """
    # 直接返回牌型名称
    return hand_type_name


# ============================================================
# 第二部分：核心算分函数（添加到 BalatroEnv 类中）
# ============================================================

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

    # 3. 依次应用小丑牌效果（从左到右）
    for joker in self.jokers:
        chips, mult = self._apply_joker_effect(
            joker, chips, mult,
            played_cards, hand_type_name, scoring_cards
        )

    # 4. 最终得分
    score = int(chips * mult)
    return score


# ============================================================
# 第三部分：小丑牌效果实现（添加到 BalatroEnv 类中）
# ============================================================

def _apply_joker_effect(self, joker, chips, mult, played_cards, hand_type_name, scoring_cards):
    """
    应用单个小丑牌的效果

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


# ============================================================
# 第四部分：小丑牌管理接口（添加到 BalatroEnv 类中）
# ============================================================

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


# ============================================================
# 第五部分：修改 best_cards_score 方法
# ============================================================

# 需要修改现有的 best_cards_score 方法：

def best_cards_score_NEW(self, cards):
    """新方法：直接按具体牌计算得分，使用小丑牌系统"""
    if not cards:
        return 0

    # 使用现有的 _calculate_best_score 评估牌型
    # 但需要提取牌型信息
    best_score = 0
    best_hand_type = None
    best_scoring_cards = None

    def sum_ranks(cards_subset):
        return sum(r for r,s in cards_subset if isinstance(r, int))

    def is_flush(cards_subset):
        if len(cards_subset) != 5: return False
        suits = [s for _,s in cards_subset]
        return len(set(suits)) == 1

    def is_straight(cards_subset):
        if len(cards_subset) != 5: return False
        vals = sorted({r for r,_ in cards_subset if isinstance(r,int)})
        if len(vals) != 5: return False
        return vals == list(range(vals[0], vals[0]+5))

    # 遍历所有可能的组合，找到最高分
    for r in range(1, min(5, len(cards)) + 1):
        for combo in combinations(cards, r):
            cnt = Counter(rv for rv,_ in combo if isinstance(rv,int))
            counts = sorted(cnt.values(), reverse=True)

            # 识别牌型（复用现有逻辑）
            has_five = counts and counts[0] >= 5
            has_four = counts and counts[0] >= 4
            has_three = counts and counts[0] >= 3
            pair_count = sum(1 for c in counts if c >= 2)
            has_full_house = has_three and pair_count >= 2

            # 按优先级匹配牌型
            patterns = []

            if has_five and is_flush(combo):
                patterns.append(("Flush Five", combo))
            if has_full_house and is_flush(combo):
                patterns.append(("Flush House", combo))
            if has_five:
                five = cnt.most_common(1)[0][0]
                patterns.append(("Five of a Kind", tuple(c for c in combo if c[0]==five)))
            if is_straight(combo) and is_flush(combo):
                patterns.append(("Straight Flush", combo))
            if has_four:
                four = cnt.most_common(1)[0][0]
                patterns.append(("Four of a Kind", tuple(c for c in combo if c[0]==four)))
            if has_full_house:
                patterns.append(("Full House", combo))
            if is_flush(combo):
                patterns.append(("Flush", combo))
            if is_straight(combo):
                patterns.append(("Straight", combo))
            if has_three:
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

            # 对每个牌型计算得分（使用小丑牌系统）
            for pat, pat_cards in patterns:
                score = self._calculate_score_with_jokers(
                    played_cards=list(combo),
                    hand_type_name=pat,
                    scoring_cards=list(pat_cards)
                )
                if score > best_score:
                    best_score = score
                    best_hand_type = pat
                    best_scoring_cards = list(pat_cards)

    return best_score
