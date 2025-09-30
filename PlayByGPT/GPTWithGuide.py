# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from openai import OpenAI
import numpy as np
from envs.BalatroEnv import BalatroEnv
from config import OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)
import pandas as pd

# You can change the model name here
GPT_MODEL_NAME = "gpt-4o-mini"

# Path of external guideline
GUIDELINE_PATH = "PlayByGPT/guideline_final.txt"

def load_guideline(path: str) -> str:
    """
    Load external guideline text. If missing/unreadable, return empty string and warn.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            print(f"[Warn] Guideline file is empty: {path}")
        else:
            print(f"[OK] Guideline loaded from: {path} (length={len(text)} chars)")
        return text
    except Exception as e:
        print(f"[Warn] Could not load guideline at {path}: {e}")
        return ""  # proceed without guideline

def build_system_prompt(env):
    lines = []
    lines.append("You are a Balatro game expert. Follow the rules strictly and output JSON only.")
    lines.append("")
    lines.append("[Action & Encoding]")
    lines.append("1) Each step you must choose exactly one action: 0=discard, 1=play.")
    lines.append(f"2) The mask is a length-{env.max_hand_size} array of 0/1; 1 means selecting the card at that hand position, and positions beyond current hand length must be 0.")
    lines.append("3) When action_type=0 (discard), the mask must include at least one 1; when action_type=1 (play), the selected cards will be played and scored.")
    lines.append("")
    lines.append("[Card Representation]")
    lines.append("Use (rank, suit) to represent a card, where rank∈[1..13], suit∈{H,D,C,S}. Example: (10,'H') is ♥10.")
    lines.append("")
    lines.append("[Allowed Number of Played Cards]")
    lines.append("You can select 1~5 cards in a single play. The environment automatically enumerates and evaluates whether the selection forms any of the listed patterns.")
    lines.append("")
    lines.append("[Patterns with Multipliers / Base Scores]")
    for name in env.hand_multipliers.keys():
        mult = env.hand_multipliers[name]
        base = env.hand_basic_score.get(name, 0)
        lines.append(f"- {name}: multiplier={mult}, base={base}")
    lines.append("")
    lines.append("[Scoring Formula]")
    lines.append("If the selected cards form a pattern, the score is: (sum of ranks in that combination + base score of the pattern) × multiplier.")
    lines.append("sum of ranks = sum of rank values of all cards in the combination (Ace is 1). The environment always picks the best-scoring valid combination from your selection.")
    lines.append("")
    lines.append("[Special Rule: High Card]")
    lines.append("If the selected cards do not form any combination like Pair/Straight/Flush, i.e. only isolated cards, only the single highest-rank card is used for scoring (not the sum of all selected cards).")
    lines.append("Example: You selected 5 isolated cards, the highest is King of Hearts, then scoring is based on K + base score for High Card, multiplied by its multiplier.")
    lines.append("")
    lines.append("[Pattern Identification Overview]")
    lines.append("- Straight Flush: simultaneously Straight + Flush (5 cards).")
    lines.append("- Four of a Kind: four of the same rank (4 cards).")
    lines.append("- Full House: three of a kind + a pair (5 cards).")
    lines.append("- Flush: all cards same suit (5 cards).")
    lines.append("- Straight: 5 cards with consecutive ranks (after de-duplication).")
    lines.append("- Three of a Kind: three cards of the same rank (3 cards).")
    lines.append("- Two Pair: two distinct pairs (at least 4 cards).")
    lines.append("- One Pair: a pair (2 cards).")
    lines.append("- High Card: no pattern; only the single highest card is used for scoring.")
    lines.append("")
    lines.append("[Done Condition] (Decided by the environment)")
    lines.append("done = (run out of play count) OR (run out of discard count AND run out of play count) OR (both hand and deck are empty).")
    lines.append("")
    return "\n".join(lines)

def gpt_choose_action(env, env_state, system_prompt, guideline_text):
    import json, sys
    import numpy as np

    # >>> Scheme A: LLM outputs positions (0-based indices); we convert to mask locally
    # 显式显示索引，帮助模型对齐“牌位-索引”
    hand_index_line = ", ".join([f"{i}: {c}" for i, c in enumerate(env_state["hand"])])

    user_prompt = (
        "You are an expert Balatro player. Based on the given hand and current game state, "
        "choose a reasonable, strategic action that strictly follows the rules. "
        "You need to appropriately choose whether to play or discard, and specify the selected cards via positions (0-based indices).\n\n"
        "Use the provided Guideline as advice; if it ever conflicts with the explicit rules, obey the rules.\n\n"
        "Action definition:\n"
        "- action_type = 0 → discard (select 1–5 positions)\n"
        "- action_type = 1 → play (select 1–5 positions; these cards will be played and scored)\n"
        f"- positions = an array of hand indices (0-based). Never select indices ≥ current hand length.\n\n"
        "Current state:\n"
        f"- Hand (index→card): {hand_index_line}\n"
        f"- Played cards: {env_state['played_cards']}\n"
        f"- Discarded cards: {env_state['discarded_cards']}\n"
        f"- Remaining plays: {env_state['play_count']}\n"
        f"- Remaining discards: {env_state['discard_count']}\n\n"
        "Output strictly in JSON format only (no explanations):\n"
        "{\n"
        '  "action_type": 0 or 1,\n'
        '  "positions": [one or more 0-based indices]\n'
        "}\n"
    )

    schema = {
        "type": "object",
        "properties": {
            "action_type": {"type": "integer", "enum": [0, 1]},
            "positions": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
                "minItems": 1,
                "maxItems": 5  # play 上限 5；弃牌我们运行时再限到 ≤3
            }
        },
        "required": ["action_type", "positions"],
        "additionalProperties": False
    }

    # Build messages list, injecting guideline as a dedicated system message (if present)
    messages = [{"role": "system", "content": system_prompt}]
    if guideline_text:
        messages.append({
            "role": "system",
            "content": (
                "[Guideline]\n"
                + guideline_text
                + "\n\nFollow the Guideline when deciding actions. "
                  "If any part of the Guideline conflicts with the explicit game rules, obey the rules."
            )
        })
    messages.append({"role": "user", "content": user_prompt})

    try:
        resp = client.chat.completions.create(
            model=GPT_MODEL_NAME,
            temperature=0,
            messages=messages,
            tools=[{
                "type": "function",
                "function": {
                    "name": "choose_balatro_action",
                    "description": "Choose a valid action for Balatro.",
                    "parameters": schema
                }
            }],
            tool_choice={"type": "function", "function": {"name": "choose_balatro_action"}}
        )

        calls = resp.choices[0].message.tool_calls
        if not calls:
            print("LLM exception: no tool_calls returned")
            sys.exit(1)

        args_json = calls[0].function.arguments  # JSON string
        print("LLM raw args:", args_json)        # debug
        parsed = json.loads(args_json)

        a_type = int(parsed["action_type"])
        positions = list(parsed["positions"])

    except Exception as e:
        print("LLM exception:", repr(e))
        sys.exit(1)   # Your requirement: exit if exception

    # ---- Local validation/fixes: positions -> mask ----
    hand_len = len(env_state["hand"])
    # 去重且排序，并且只保留合法索引
    positions = sorted(set([p for p in positions if isinstance(p, int) and 0 <= p < hand_len]))
    if a_type == 1:
        # play：1~5 张（schema 已限 ≤5），这里确保至少 1 张，且不超过当前手牌数
        if not (1 <= len(positions) <= min(5, hand_len)):
            positions = [0] if hand_len > 0 else []
    else:
        # discard：1~5 张，确保至少 1 张，且不超过当前手牌数
        if not (1 <= len(positions) <= min(5, hand_len)):
            positions = [0] if hand_len > 0 else []

    # 转换为定长 mask
    mask = np.zeros(env.max_hand_size, dtype=np.int8)
    for p in positions:
        if p < env.max_hand_size:
            mask[p] = 1

    # 兜底：仍然为空且手里有牌
    if hand_len > 0 and mask.sum() == 0:
        mask[0] = 1

    return int(a_type), mask

def play_n_episodes(env, n=100):
    all_steps, episode_summaries = [], []
    system_prompt = build_system_prompt(env)  # build once
    guideline_text = load_guideline(GUIDELINE_PATH)  # load once

    for ep in range(n):
        env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            env_state = env.get_env_state()
            action = gpt_choose_action(env, env_state, system_prompt, guideline_text)
            print(action)
            _, reward, done, _ = env.step(action)
            print(done)
            ep_reward += reward

            step_info = env.get_step_history()[-1]
            all_steps.append({
                "episode": ep,
                "hand": env_state["hand"],
                "played_cards": env_state["played_cards"],
                "discarded_cards": env_state["discarded_cards"],
                "selected_cards": step_info["selected_cards"],
                "reward": step_info["reward"],
            })

        episode_summaries.append({"episode": ep, "total_reward": ep_reward})
        print(f"Episode {ep} finished, total reward={ep_reward}")

    df_steps = pd.DataFrame(all_steps)
    df_steps.to_csv("NoGuideEpisode.csv", index=False, encoding="utf-8-sig")

    df_summary = pd.DataFrame(episode_summaries)
    df_summary.to_csv("NoGuideSummary.csv", index=False, encoding="utf-8-sig")
    avg_reward = df_summary["total_reward"].mean()
    print(f"\nPlayed {n} episodes, average total reward = {avg_reward:.2f}")
    return df_steps, df_summary, float(avg_reward)

if __name__ == "__main__":
    env = BalatroEnv()
    df_steps, df_summary, avg_reward = play_n_episodes(env, n=100)

    # Write run summary CSV
    # Columns: [script_file_name, gpt_model_name, average_score_of_100_eps]
    script_name = os.path.basename(__file__)
    out_row = {
        "script_file": script_name,
        "gpt_model": GPT_MODEL_NAME,
        "avg_score_100eps": avg_reward
    }
    df_out = pd.DataFrame([out_row])

    summary_file = "run_summary.csv"
    if not os.path.exists(summary_file):
        # Not exists -> create and write header + one row
        df_out.to_csv(summary_file, index=False, encoding="utf-8-sig", mode="w")
    else:
        # Exists -> append exactly one row, no header
        df_out.to_csv(summary_file, index=False, encoding="utf-8-sig", mode="a", header=False)

    print(f"[Save] Summary appended to -> {summary_file}")