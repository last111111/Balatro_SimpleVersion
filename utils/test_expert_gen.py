# utils/test_expert_gen.py
# -*- coding: utf-8 -*-
"""测试 GPT 专家轨迹生成：跑 2 个 episode，打印每步的 prompt 和 GPT 回复"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import re
import numpy as np
import random

from envs.joker_env import JokerSelectEnv, NUM_ROUNDS
from utils.generate_joker_expert import _build_user_prompt, _get_joker_name, SYSTEM_PROMPT
from utils.api_key import OPENAI_API_KEY


def test_expert_generation(num_episodes=2, model="gpt-4o-mini"):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY, timeout=15)
    env = JokerSelectEnv()

    random.seed(42)
    np.random.seed(42)

    total_steps = 0
    valid_steps = 0
    invalid_steps = 0

    for ep in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {ep+1}/{num_episodes}")
        print(f"{'='*60}")

        obs = env.reset()

        for step in range(NUM_ROUNDS):
            action_mask = env.get_action_mask()
            user_prompt = _build_user_prompt(env)

            print(f"\n--- Round {step+1}/{NUM_ROUNDS} ---")
            print(f"[Prompt]\n{user_prompt}")

            # 调用 GPT
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=10,
                temperature=0.3,
            )
            raw_text = response.choices[0].message.content.strip()
            print(f"[GPT raw response]: \"{raw_text}\"")

            # 解析
            match = re.search(r'\d+', raw_text)
            if match:
                action = int(match.group())
                print(f"[Parsed action]: {action}")

                if action < len(action_mask) and action_mask[action] == 1.0:
                    print(f"[Valid]: YES")
                    valid_steps += 1
                else:
                    print(f"[Valid]: NO (mask={action_mask[action] if action < len(action_mask) else 'out of range'})")
                    invalid_steps += 1
                    action = 0  # fallback to skip
            else:
                print(f"[Valid]: NO (cannot parse number)")
                invalid_steps += 1
                action = 0

            total_steps += 1

            # 执行
            obs, _, done, info = env.step(action)

        # Episode 结束
        held_names = [_get_joker_name(jid) for jid in env.held]
        print(f"\n[Episode {ep+1} Result]")
        print(f"  Held jokers ({len(env.held)}): {held_names}")
        print(f"  Held IDs: {env.held}")

        # 验证 obs 格式
        print(f"\n[Obs check]")
        print(f"  obs shape: {obs.shape}")
        print(f"  obs[0:30] (held multi-hot): {np.where(obs[:30] > 0)[0].tolist()}")
        print(f"  obs[30:35] (held slot IDs): {obs[30:35]}")
        print(f"  obs[35:39] (offered slot IDs): {obs[35:39]}")
        print(f"  obs[39] (round): {obs[39]:.3f}")
        print(f"  obs[40] (held_count): {obs[40]:.3f}")

    # 汇总
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"  Total steps: {total_steps}")
    print(f"  Valid: {valid_steps} ({100*valid_steps/total_steps:.1f}%)")
    print(f"  Invalid (fallback to skip): {invalid_steps} ({100*invalid_steps/total_steps:.1f}%)")

    if invalid_steps == 0:
        print(f"\n  ALL GOOD - GPT responses are valid, ready for BC training!")
    else:
        print(f"\n  WARNING: {invalid_steps} invalid responses. Check prompt format.")


if __name__ == "__main__":
    test_expert_generation()
