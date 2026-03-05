# utils/generate_joker_expert.py
# -*- coding: utf-8 -*-
"""GPT 专家轨迹生成：模拟 JokerSelectEnv，让 GPT 选最优动作，保存 BC 训练数据"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import re
import numpy as np
from tqdm.auto import tqdm

from envs.joker_env import JokerSelectEnv, NUM_JOKER_TYPES, MAX_HELD, NUM_OFFERED, NUM_ROUNDS
from utils.chatgpt_reward import JOKER_DESCRIPTIONS


SYSTEM_PROMPT = """\
You are an expert Balatro player choosing Joker cards.

Game rules:
- 7 rounds of Joker selection, 4 jokers offered per round, max 5 held.
- During card play: 52-card deck, 8-card hand, 5 plays + 3 discards per round.
- Score = (base_chips + card_chips) × (base_mult + joker_mult).
- Joker effects are applied left-to-right (order is auto-optimized, you don't need to worry about ordering).

Strategy tips:
- Pick jokers that SYNERGIZE (e.g., suit-specific mult + flush enablers).
- Multiplicative jokers (X Mult) are very powerful, prioritize them.
- Consider coverage: jokers that trigger on common hand types (Pair, Two Pair) are more reliable.
- Later rounds: consider replacing weak jokers with stronger ones.

Reply with ONLY the action number (0, 1, 2, ...). Nothing else."""


def _get_joker_name(jid):
    """获取 joker 简短名称（不含效果描述）"""
    desc = JOKER_DESCRIPTIONS.get(jid, f"Unknown({jid})")
    return desc.split(" - ")[0]


def _get_joker_desc(jid):
    """获取 joker 完整描述"""
    return JOKER_DESCRIPTIONS.get(jid, f"Unknown Joker (id={jid})")


def _build_action_descriptions(env):
    """构建当前可选动作的描述列表"""
    actions = []
    actions.append("0: Skip (don't pick any joker)")

    if len(env.held) < MAX_HELD:
        # 可以直接添加
        for i in range(NUM_OFFERED):
            jid = env.offered[i]
            actions.append(f"{1+i}: Pick \"{_get_joker_desc(jid)}\"")
    else:
        # 已满，替换
        for i in range(NUM_OFFERED):
            offered_jid = env.offered[i]
            offered_name = _get_joker_name(offered_jid)
            for j in range(MAX_HELD):
                held_jid = env.held[j]
                held_name = _get_joker_name(held_jid)
                action_id = 5 + i * MAX_HELD + j
                actions.append(
                    f"{action_id}: Pick \"{_get_joker_desc(offered_jid)}\" → replace slot {j+1} ({held_name})"
                )

    return "\n  ".join(actions)


def _build_user_prompt(env):
    """构建单步的 user prompt"""
    lines = [f"Round {env.round + 1}/{NUM_ROUNDS}"]

    # 当前持有
    lines.append(f"\nCurrently held jokers ({len(env.held)}/{MAX_HELD}):")
    if env.held:
        for i, jid in enumerate(env.held):
            lines.append(f"  {i+1}. {_get_joker_desc(jid)}")
    else:
        lines.append("  (none)")

    # 本轮提供
    lines.append(f"\nOffered this round:")
    for i, jid in enumerate(env.offered):
        lines.append(f"  {chr(65+i)}. {_get_joker_desc(jid)}")

    # 可选动作
    lines.append(f"\nAvailable actions:")
    lines.append(f"  {_build_action_descriptions(env)}")

    lines.append("\nChoose the best action:")
    return "\n".join(lines)


def _query_gpt(client, model, user_prompt, max_retries=2):
    """调用 GPT 获取动作选择"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=10,
                temperature=0.3 if attempt == 0 else 0.1,
            )
            text = response.choices[0].message.content.strip()
            match = re.search(r'\d+', text)
            if match:
                return int(match.group())
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  [Warn] GPT call failed: {e}")
    return None


def generate_expert_data(num_episodes=500, model="gpt-4o-mini", output_path="outputs/joker/expert_data.npz",
                         timeout=15, seed=42):
    """生成 GPT 专家轨迹"""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("需要安装 openai 包: pip install openai")

    from utils.api_key import OPENAI_API_KEY
    api_key = OPENAI_API_KEY

    client = OpenAI(api_key=api_key, timeout=timeout)
    env = JokerSelectEnv()

    np.random.seed(seed)
    import random
    random.seed(seed)

    all_obs = []
    all_actions = []
    all_masks = []
    episode_summaries = []

    skipped = 0
    total_steps = 0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pbar = tqdm(total=num_episodes, desc="Generating expert data", unit="ep")

    for ep in range(num_episodes):
        obs = env.reset()

        for step in range(NUM_ROUNDS):
            action_mask = env.get_action_mask()
            user_prompt = _build_user_prompt(env)

            # 询问 GPT
            gpt_action = _query_gpt(client, model, user_prompt)

            # 验证动作有效性
            if gpt_action is None or gpt_action >= len(action_mask) or action_mask[gpt_action] == 0:
                # 无效 → skip
                gpt_action = 0
                skipped += 1

            # 记录
            all_obs.append(obs.copy())
            all_actions.append(gpt_action)
            all_masks.append(action_mask.copy())
            total_steps += 1

            # 执行
            obs, _, done, info = env.step(gpt_action)

        # Episode 结束
        held_names = [_get_joker_name(jid) for jid in env.held]
        episode_summaries.append({
            "ep": ep, "held": list(env.held), "names": held_names
        })

        pbar.update(1)
        if (ep + 1) % 50 == 0:
            pbar.set_postfix({
                "steps": total_steps,
                "skipped": f"{skipped}/{total_steps}",
                "last_held": str(held_names[:3]) + "..."
            })

    pbar.close()

    # 保存
    np.savez(
        output_path,
        obs=np.array(all_obs, dtype=np.float32),
        actions=np.array(all_actions, dtype=np.int64),
        action_masks=np.array(all_masks, dtype=np.float32),
    )

    print(f"\n[Save] 专家数据已保存到 {output_path}")
    print(f"  总步数: {total_steps} ({num_episodes} episodes × {NUM_ROUNDS} rounds)")
    print(f"  跳过 (无效动作 fallback): {skipped} ({100*skipped/total_steps:.1f}%)")

    # 打印一些 episode 样本
    print(f"\n[Sample] 前 5 个 episode 的 joker 选择:")
    for s in episode_summaries[:5]:
        print(f"  Ep {s['ep']}: {s['names']}")

    return output_path


def main():
    p = argparse.ArgumentParser(description='Generate GPT expert trajectories for Joker selection BC')
    p.add_argument("--num_episodes", type=int, default=500,
                   help="Number of episodes to generate")
    p.add_argument("--model", type=str, default="gpt-4o-mini",
                   help="OpenAI model to use")
    p.add_argument("--output", type=str, default="outputs/joker/expert_data.npz",
                   help="Output .npz file path")
    p.add_argument("--timeout", type=int, default=15,
                   help="API call timeout (seconds)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for environment")
    args = p.parse_args()

    generate_expert_data(
        num_episodes=args.num_episodes,
        model=args.model,
        output_path=args.output,
        timeout=args.timeout,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
