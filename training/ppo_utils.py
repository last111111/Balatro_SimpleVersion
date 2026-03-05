# training/ppo_utils.py
# -*- coding: utf-8 -*-
"""PPO 训练的共享工具函数"""

import random
import numpy as np
import torch


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def smooth_curve(y, window=100):
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y
    w = int(max(1, min(window, len(y))))
    if w == 1:
        return y
    kernel = np.ones(w, dtype=np.float32) / w
    return np.convolve(y, kernel, mode='valid')


def gae(rews, dones, vals, gamma, lam):
    """Generalized Advantage Estimation"""
    T = len(rews)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        nonterm = 1.0 - float(dones[t])
        next_v = vals[t + 1] if t + 1 < len(vals) else 0.0
        delta = rews[t] + gamma * next_v * nonterm - vals[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    rets = adv + np.array(vals[:T], dtype=np.float32)
    return rets, adv


class SimpleVecEnv:
    """同步向量化环境：同时运行 N 个独立环境实例"""
    def __init__(self, env_fn, n_envs=8):
        self.envs = [env_fn() for _ in range(n_envs)]
        self.n_envs = n_envs

    def reset(self):
        return np.array([env.reset() for env in self.envs])

    def step(self, actions):
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        obs, rews, dones, infos = zip(*results)
        obs = list(obs)
        for i, d in enumerate(dones):
            if d:
                obs[i] = self.envs[i].reset()
        return np.array(obs), np.array(rews), np.array(dones), list(infos)

    @property
    def observation_space(self):
        return self.envs[0].observation_space

    @property
    def action_space(self):
        return self.envs[0].action_space
