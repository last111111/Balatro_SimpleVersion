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

def build_system_prompt(env):
    lines = []
    lines.append("你是 Balatro 游戏专家，请严格按以下规则行动，并只输出 JSON：")
    lines.append("")
    lines.append("【动作与编码】")
    lines.append("1) 每一步只能二选一：0=弃牌(discard)，1=出牌(play)。")
    lines.append(f"2) mask 是长度为 {env.max_hand_size} 的 0/1 数组；1 表示选中对应手牌位置，超出当前手牌长度的位必须为 0。")
    lines.append("3) 当 action_type=0（弃牌）时，mask 至少包含一张 1；当 action_type=1（出牌）时，mask 选中的牌会被打出并计分。")
    lines.append("")
    lines.append("【牌表示】")
    lines.append("用 (rank, suit) 表示一张牌，rank∈[1..13]，suit∈{H,D,C,S}。例：(10,'H') 表示红桃10。")
    lines.append("")
    lines.append("【允许的出牌张数】")
    lines.append("一次出牌可选 1～5 张，环境会自动枚举并评估组合是否构成下列牌型。")
    lines.append("")
    lines.append("【牌型与倍数/基础分】")
    for name in env.hand_multipliers.keys():
        mult = env.hand_multipliers[name]
        base = env.hand_basic_score.get(name, 0)
        lines.append(f"- {name}: 倍数={mult}，基础分={base}")
    lines.append("")
    lines.append("【计分公式】")
    lines.append("选择的牌若构成某一牌型，其得分：score = (所选组合的点数和 + 该牌型基础分) × 该牌型倍数。")
    lines.append("点数和 = 组合中所有牌的 rank 之和（A 记为 1）。环境会在你给的选择里自动寻找分数最高的合法组合。")
    lines.append("")
    lines.append("【特别规则：高牌 (High Card)】")
    lines.append("如果选出的牌不能组成对子、顺子、同花等牌型，而只有散牌，那么系统只会取 **其中点数最高的一张牌** 来计分，不会把所有牌加和。")
    lines.append("例如：你选出 5 张散牌，其中最大的一张是红桃K，那么只按 K 的点数 + 高牌基础分，再乘以高牌倍数来计分。")
    lines.append("")
    lines.append("【牌型判定概要】")
    lines.append("- Straight Flush: 同时满足 Straight 与 Flush（5 张）。")
    lines.append("- Four of a Kind: 四条（4 张）。")
    lines.append("- Full House: 三带二（5 张）。")
    lines.append("- Flush: 同花（5 张）。")
    lines.append("- Straight: 顺子（5 张，去重后连续）。")
    lines.append("- Three of a Kind: 三条（3 张）。")
    lines.append("- Two Pair: 两对（至少 4 张）。")
    lines.append("- One Pair: 一对（2 张）。")
    lines.append("- High Card: 散牌，只取其中最大的单牌来算。")
    lines.append("")
    lines.append("【局面结束（done）条件】（由环境判定）")
    lines.append("done = (出牌次数用尽) 或者 (弃牌次数用尽 且 出牌次数用尽) 或者 (手牌与牌库同时为空)。")
    lines.append("")
    return "\n".join(lines)

def gpt_choose_action(env, env_state, system_prompt):
    import json, sys
    import numpy as np

    user_prompt = (
        "当前状态：\n"
        f"- 手牌: {env_state['hand']}\n"
        f"- 已出牌: {env_state['played_cards']}\n"
        f"- 已弃牌: {env_state['discarded_cards']}\n"
        f"- 剩余出牌次数: {env_state['play_count']}\n"
        f"- 剩余弃牌次数: {env_state['discard_count']}\n\n"
        "请只输出严格 JSON：\n"
        "{\n"
        '  "action_type": 0 或 1,\n'
        f'  "mask": 长度为 {env.max_hand_size} 的 0/1 数组，超出当前手牌长度的位必须为 0\n'
        "}\n"
    )

    schema = {
        "type": "object",
        "properties": {
            "action_type": {"type": "integer", "enum": [0, 1]},
            "mask": {
                "type": "array",
                "items": {"type": "integer", "enum": [0, 1]},
                "minItems": env.max_hand_size,
                "maxItems": env.max_hand_size
            }
        },
        "required": ["action_type", "mask"],
        "additionalProperties": False
    }

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
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
        print("LLM raw args:", args_json)        # 调试
        parsed = json.loads(args_json)

        a_type = int(parsed["action_type"])
        raw_mask = list(parsed["mask"])

    except Exception as e:
        print("LLM exception:", repr(e))
        sys.exit(1)   # 你的要求：异常就终止

    # 本地校验 / 修正
    mask = np.zeros(env.max_hand_size, dtype=np.int8)
    hand_len = len(env_state["hand"])
    for i in range(min(env.max_hand_size, len(raw_mask), hand_len)):
        mask[i] = 1 if raw_mask[i] else 0
    for i in range(hand_len, env.max_hand_size):
        mask[i] = 0
    if hand_len > 0 and mask.sum() == 0:
        mask[0] = 1

    return int(a_type), mask

def play_n_episodes(env, n=100):
    all_steps, episode_summaries = [], []
    system_prompt = build_system_prompt(env)  # << 只构建一次

    for ep in range(n):
        env.reset()
        done = False
        ep_reward = 0.0
   
        while not done:
            env_state = env.get_env_state()
            action = gpt_choose_action(env, env_state, system_prompt)  # << 传入 system_prompt
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
    return df_steps, df_summary

if __name__ == "__main__":
    env = BalatroEnv()
    df_steps, df_summary = play_n_episodes(env, n=100)