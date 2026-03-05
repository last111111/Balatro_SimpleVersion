# PPO_train.py
# -*- coding: utf-8 -*-
"""薄包装：向后兼容入口，实际逻辑在 training/train_card.py"""

# 保持旧的 import 路径可用（evaluate_model.py 等可能 from PPO_train import ActorCritic）
from models.card_agent import CardEncoder, ActorCritic, orthogonal_init  # noqa: F401
from training.train_card import PPOAgent, train, main  # noqa: F401

if __name__ == "__main__":
    main()
