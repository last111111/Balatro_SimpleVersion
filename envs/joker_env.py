# envs/joker_env.py
# -*- coding: utf-8 -*-
"""小丑牌选择环境：7轮，每轮提供4张，最多持有5张"""

import gym
from gym import spaces
import numpy as np
import random


NUM_JOKER_TYPES = 30
MAX_HELD = 5
NUM_ROUNDS = 7
NUM_OFFERED = 4


class JokerSelectEnv(gym.Env):
    """
    小丑牌选择环境

    观测空间 (41 维):
      obs[0:30]   = 已持有小丑牌 multi-hot
      obs[30:35]  = held 槽位 type ID / 29（空槽 = -1/29）
      obs[35:39]  = offered 位置 type ID / 29
      obs[39]     = round / 6 (归一化进度, 0~1)
      obs[40]     = len(held) / 5 (归一化持有量, 0~1)

    动作空间 Discrete(25):
      Action 0:    跳过 (永远有效)
      Action 1-4:  选提供的第 0-3 张 (仅 held < 5 时有效)
      Action 5-24: 选提供的第 (a-5)//5 张 并替换已有第 (a-5)%5 张
                   (仅 held == 5 时有效)
    """

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(41,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(25)

        self.held = []       # 已持有的 joker type ids
        self.offered = []    # 当前提供的 4 张 joker type ids
        self.round = 0

    def reset(self):
        self.held = []
        self.round = 0
        self._offer_new_jokers()
        return self._get_obs()

    def step(self, action):
        """
        执行动作，返回 (obs, reward, done, info)
        中间步 reward=0，最终 reward 由外部 (ChatGPT 或打牌 agent) 提供
        """
        action = int(action)
        mask = self.get_action_mask()

        # 如果动作无效，当作跳过
        if not mask[action]:
            action = 0

        if action == 0:
            # 跳过
            pass
        elif 1 <= action <= 4:
            # 选提供的第 (action-1) 张
            pick_idx = action - 1
            joker_id = self.offered[pick_idx]
            self.held.append(joker_id)
        else:
            # 选提供的第 offer_idx 张，替换已有的第 replace_idx 张
            a = action - 5
            offer_idx = a // MAX_HELD
            replace_idx = a % MAX_HELD
            joker_id = self.offered[offer_idx]
            self.held[replace_idx] = joker_id

        self.round += 1
        done = (self.round >= NUM_ROUNDS)

        if not done:
            self._offer_new_jokers()

        info = {"held_jokers": list(self.held), "round": self.round}
        return self._get_obs(), 0.0, done, info

    def get_action_mask(self):
        """返回 25 维布尔 mask，标记哪些动作有效"""
        mask = np.zeros(25, dtype=np.float32)

        # Action 0: 跳过永远有效
        mask[0] = 1.0

        if len(self.held) < MAX_HELD:
            # 可以直接添加：Action 1-4
            for i in range(NUM_OFFERED):
                mask[1 + i] = 1.0
        else:
            # 已满，只能替换：Action 5-24
            for i in range(NUM_OFFERED):
                for j in range(MAX_HELD):
                    mask[5 + i * MAX_HELD + j] = 1.0

        return mask

    def _offer_new_jokers(self):
        """从 30 种中随机选 4 张（排除已持有的）"""
        available = [j for j in range(NUM_JOKER_TYPES) if j not in self.held]
        # 如果可选不足 4 张，允许重复（极端情况）
        if len(available) >= NUM_OFFERED:
            self.offered = random.sample(available, NUM_OFFERED)
        else:
            self.offered = list(available)
            while len(self.offered) < NUM_OFFERED:
                self.offered.append(random.choice(range(NUM_JOKER_TYPES)))

    def _get_obs(self):
        obs = np.zeros(41, dtype=np.float32)

        # 已持有 multi-hot（用于网络端 mean-pool embedding）
        for jid in self.held:
            obs[jid] = 1.0

        # held 槽位 type ID / 29（空槽 = -1/29）
        for i in range(MAX_HELD):
            if i < len(self.held):
                obs[30 + i] = self.held[i] / 29.0
            else:
                obs[30 + i] = -1.0 / 29.0

        # offered 位置 type ID / 29
        for i in range(NUM_OFFERED):
            obs[35 + i] = self.offered[i] / 29.0

        # 归一化进度
        obs[39] = self.round / max(1, NUM_ROUNDS - 1)  # 0~1

        # 归一化持有量
        obs[40] = len(self.held) / MAX_HELD

        return obs
