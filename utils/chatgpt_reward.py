# utils/chatgpt_reward.py
# -*- coding: utf-8 -*-
"""ChatGPT 小丑牌评分模块：调用 gpt-4o-mini 评估小丑牌组合质量"""

import os
import re

# 30 种小丑牌的效果描述
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

SYSTEM_PROMPT = """\
You are evaluating a Joker card collection in the game Balatro.

Game rules: 52-card standard deck, 8-card hand, 5 plays and 3 discards per round.
Score for each hand = (base_chips + card_chips) × (base_mult + joker_mult).
Higher scores are better. The player can hold up to 5 Joker cards.

Rate the given Joker collection on a scale of 1-10. Consider:
1. Individual strength of each Joker
2. Synergy between Jokers (do they complement each other?)
3. Coverage (do they work with different hand types or strategies?)
4. Scaling potential (do any grow stronger over time?)

Reply with ONLY a single integer from 1 to 10. Nothing else."""


def _build_user_prompt(held_joker_ids):
    """构建用户 prompt"""
    lines = ["Current Joker collection:"]
    for i, jid in enumerate(held_joker_ids):
        desc = JOKER_DESCRIPTIONS.get(jid, f"Unknown Joker (id={jid})")
        lines.append(f"  {i+1}. {desc}")
    if not held_joker_ids:
        lines.append("  (empty - no jokers selected)")
    return "\n".join(lines)


def get_joker_rating(held_joker_ids, model="gpt-4o-mini", timeout=10):
    """
    调用 OpenAI API 评估小丑牌组合质量

    Args:
        held_joker_ids: list of int, 已持有的小丑牌 type ID 列表
        model: OpenAI 模型名称
        timeout: 请求超时时间（秒）

    Returns:
        int: 1-10 的评分
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("需要安装 openai 包: pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("请设置 OPENAI_API_KEY 环境变量")

    client = OpenAI(api_key=api_key, timeout=timeout)

    user_prompt = _build_user_prompt(held_joker_ids)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=5,
        temperature=0.3,
    )

    text = response.choices[0].message.content.strip()

    # 提取数字
    match = re.search(r'\d+', text)
    if match:
        rating = int(match.group())
        return max(1, min(10, rating))

    # 解析失败，返回中间值
    return 5


def get_joker_rating_batch(batch_held_joker_ids, model="gpt-4o-mini", timeout=10):
    """
    批量评估多组小丑牌（串行调用，适合训练时每 episode 结束调用一次）

    Args:
        batch_held_joker_ids: list of list of int

    Returns:
        list of int: 每组的评分
    """
    return [get_joker_rating(held, model=model, timeout=timeout)
            for held in batch_held_joker_ids]
