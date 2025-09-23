from openai import OpenAI
import numpy as np
from envs.BalatroEnv import BalatroEnv

client = OpenAI()

def gpt_choose_action(env_state):
    """
    使用 GPT 生成动作，返回 (a_type, mask)
    """
    prompt = f"""
    你是一个 Balatro 游戏专家。请根据当前状态选择动作：
    - 手牌: {env_state['hand']}
    - 已出牌: {env_state['played_cards']}
    - 已弃牌: {env_state['discarded_cards']}
    - 剩余出牌次数: {env_state['play_count']}
    - 剩余弃牌次数: {env_state['discard_count']}

    请选择一个动作，要求输出 JSON 格式：
    {{
      "action_type": 0 或 1,    # 0=弃牌, 1=出牌
      "mask": [0,1,0,0,...]     # 长度={len(env_state['hand'])} 的数组，选中的牌为1
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "balatro_action",
                "schema": {
                    "type": "object",
                    "properties": {
                        "action_type": {"type": "integer", "enum": [0, 1]},
                        "mask": {
                            "type": "array",
                            "items": {"type": "integer", "enum": [0, 1]}
                        }
                    },
                    "required": ["action_type", "mask"]
                }
            }
        }
    )

    action = response.choices[0].message.parsed
    a_type = action["action_type"]
    # 注意：mask 长度要补齐到 max_hand_size
    mask = action["mask"]
    mask = mask + [0]*(env.max_hand_size - len(mask))
    return (a_type, np.array(mask, dtype=np.int8))


def play_one_episode(env):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        env_state = env.get_env_state()
        action = gpt_choose_action(env_state)
        obs, reward, done, _ = env.step(action)
        total_reward += reward

    return total_reward, env.get_step_history()


def play_n_episodes(env, n=100):
    results = []
    for i in range(n):
        total_reward, steps = play_one_episode(env)
        results.append({"episode": i, "reward": total_reward, "steps": steps})
        print(f"Episode {i} finished. Total reward={total_reward}")
    return results


env = BalatroEnv()
results = play_n_episodes(env, n=100)

# 保存结果
import json
with open("gpt_play_100.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
