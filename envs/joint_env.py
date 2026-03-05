# envs/joint_env.py
# -*- coding: utf-8 -*-
"""联动环境编排器：小丑牌选择 + 打牌交替进行"""

from envs.joker_env import JokerSelectEnv, NUM_ROUNDS
from envs.BalatroEnv import BalatroEnv


class JointEnv:
    """
    联动环境：每个 episode = 7 轮

    流程 (每轮):
      1. joker_env 提供观测 + action mask
      2. joker_agent 选择小丑牌
      3. joker_env.step(action)
      4. card_env.reset_with_jokers(held_jokers)
      5. card_agent 打完一局 (5 plays + 3 discards)
      6. joker_reward = card_env.cumulative_score / 100
    """

    def __init__(self, max_hand_size=8, max_play=5, shaping_beta=1.0):
        self.joker_env = JokerSelectEnv()
        self.card_env = BalatroEnv(
            max_hand_size=max_hand_size,
            max_play=max_play,
            shaping_beta=shaping_beta
        )
        self.current_round = 0

    def reset(self):
        """重置联动环境，返回 joker_env 的初始观测"""
        joker_obs = self.joker_env.reset()
        self.current_round = 0
        return joker_obs

    def get_joker_action_mask(self):
        """获取当前 joker 动作 mask"""
        return self.joker_env.get_action_mask()

    def joker_step(self, joker_action):
        """
        执行 joker 选择动作

        Returns:
            joker_obs: 下一轮 joker 观测
            joker_done: 是否 7 轮结束
            joker_info: 包含 held_jokers 等信息
        """
        joker_obs, _, joker_done, joker_info = self.joker_env.step(joker_action)
        self.current_round += 1
        return joker_obs, joker_done, joker_info

    def play_card_episode(self, card_agent):
        """
        用当前持有的小丑牌打一局

        Args:
            card_agent: PPOAgent 实例（打牌 agent）

        Returns:
            card_traj: 打牌 trajectory dict
            cumulative_score: 本局累计得分
        """
        held_joker_ids = [j for j in self.joker_env.held]
        card_obs = self.card_env.reset_with_jokers(held_joker_ids)

        card_traj = {"obs": [], "a_type": [], "a_mask": [],
                     "logp_type": [], "logp_mask": [],
                     "val": [], "rew": [], "done": []}

        done = False
        while not done:
            a_type, a_mask, logp_type, logp_mask, val, _ = card_agent.act(card_obs)
            next_obs, reward, done, info = self.card_env.step((a_type, a_mask))

            card_traj["obs"].append(card_obs)
            card_traj["a_type"].append(a_type)
            card_traj["a_mask"].append(a_mask)
            card_traj["logp_type"].append(logp_type)
            card_traj["logp_mask"].append(logp_mask)
            card_traj["val"].append(val)
            card_traj["rew"].append(reward)
            card_traj["done"].append(done)

            card_obs = next_obs

        return card_traj, self.card_env.cumulative_score

    @property
    def held_jokers(self):
        return list(self.joker_env.held)

    @property
    def is_done(self):
        return self.current_round >= NUM_ROUNDS
