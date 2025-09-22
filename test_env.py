#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试BalatroEnv环境的(s,a)保存功能
"""

import numpy as np
import json
import os
from envs.BalatroEnv import BalatroEnv

def test_basic_functionality():
    """测试基本环境功能"""
    print("=== 测试基本环境功能 ===")
    env = BalatroEnv()
    obs = env.reset()
    
    print(f"观测空间维度: {obs.shape}")
    print(f"动作空间: {env.action_space}")
    print("初始状态:")
    env.print_self()
    print()

def test_step_history_recording():
    """测试step历史记录功能"""
    print("=== 测试step历史记录功能 ===")
    env = BalatroEnv()
    obs = env.reset()
    
    # 执行几个动作
    actions = [
        (0, [1, 1, 0, 0, 0, 0, 0, 0]),  # 弃牌前两张
        (1, [1, 0, 1, 0, 0, 0, 0, 0]),  # 出牌第1,3张
        (0, [0, 1, 0, 0, 0, 0, 0, 0]),  # 弃牌第2张
    ]
    
    for i, action in enumerate(actions):
        print(f"\n--- Step {i+1} ---")
        print(f"执行动作: {action}")
        print("执行前状态:")
        env.print_self()
        
        obs, reward, done, info = env.step(action)
        print(f"奖励: {reward}, 游戏结束: {done}")
        
        if done:
            break
    
    # 检查历史记录
    history = env.get_step_history()
    print(f"\n总共记录了 {len(history)} 步")
    
    for i, step_info in enumerate(history):
        print(f"\nStep {i+1} 记录:")
        print(f"  动作类型: {step_info['action_type']}")
        print(f"  选中卡牌: {step_info['selected_cards']}")
        print(f"  奖励: {step_info['reward']}")
        print(f"  游戏结束: {step_info['done']}")
    
    print()

def test_env_state_tracking():
    """测试环境状态跟踪"""
    print("=== 测试环境状态跟踪 ===")
    env = BalatroEnv()
    obs = env.reset()
    
    print("初始环境状态:")
    initial_state = env.get_env_state()
    print(f"  手牌数量: {len(initial_state['hand'])}")
    print(f"  已打出牌数量: {len(initial_state['played_cards'])}")
    print(f"  已弃牌数量: {len(initial_state['discarded_cards'])}")
    print(f"  剩余出牌次数: {initial_state['play_count']}")
    print(f"  剩余弃牌次数: {initial_state['discard_count']}")
    
    # 执行一个出牌动作
    action = (1, [1, 1, 1, 0, 0, 0, 0, 0])  # 出前三张牌
    obs, reward, done, info = env.step(action)
    
    print(f"\n执行出牌动作后:")
    current_state = env.get_env_state()
    print(f"  手牌数量: {len(current_state['hand'])}")
    print(f"  已打出牌数量: {len(current_state['played_cards'])}")
    print(f"  已弃牌数量: {len(current_state['discarded_cards'])}")
    print(f"  剩余出牌次数: {current_state['play_count']}")
    print(f"  剩余弃牌次数: {current_state['discard_count']}")
    
    print(f"  实际打出的牌: {current_state['played_cards']}")
    print()

def test_save_load_functionality():
    """测试保存加载功能"""
    print("=== 测试保存加载功能 ===")
    env = BalatroEnv()
    obs = env.reset()
    
    # 执行一些动作
    actions = [
        (0, [1, 0, 0, 0, 0, 0, 0, 0]),  # 弃一张牌
        (1, [1, 1, 0, 0, 0, 0, 0, 0]),  # 出两张牌
    ]
    
    for action in actions:
        obs, reward, done, info = env.step(action)
        if done:
            break
    
    # 保存数据
    test_filepath = "test_episode_data.json"
    saved_filepath = env.save_episode_data(test_filepath)
    print(f"数据已保存到: {saved_filepath}")
    
    # 验证保存的数据
    if os.path.exists(test_filepath):
        with open(test_filepath, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        print(f"保存的数据包含 {loaded_data['total_steps']} 步")
        print(f"最终环境状态: {loaded_data['final_env_state']['play_count']} 次出牌机会剩余")
        
        # 显示每步的详细信息
        for i, step in enumerate(loaded_data['step_history']):
            print(f"\nStep {i+1}:")
            print(f"  动作: {step['action']}")
            print(f"  动作类型: {step['action_type']}")
            print(f"  选中卡牌: {step['selected_cards']}")
            print(f"  奖励: {step['reward']}")
        
        # 清理测试文件
        os.remove(test_filepath)
        print(f"\n测试文件 {test_filepath} 已删除")
    else:
        print("保存失败!")
    
    print()

def test_complete_episode():
    """测试完整的一局游戏"""
    print("=== 测试完整的一局游戏 ===")
    env = BalatroEnv(max_play=3, max_discard=2)  # 限制次数以便快速结束
    obs = env.reset()
    
    step_count = 0
    total_reward = 0
    
    while True:
        # 随机选择动作
        action_type = np.random.choice([0, 1])  # 随机选择弃牌或出牌
        mask = np.random.choice([0, 1], size=env.max_hand_size)  # 随机选择卡牌
        action = (action_type, mask)
        
        print(f"\nStep {step_count + 1}:")
        print(f"执行动作: {action}")
        
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        print(f"奖励: {reward}, 累计奖励: {total_reward}")
        env.print_self()
        
        if done:
            print(f"\n游戏结束! 总步数: {step_count}, 总奖励: {total_reward}")
            break
    
    # 保存完整游戏数据
    filepath = env.save_episode_data("complete_episode_test.json")
    print(f"完整游戏数据已保存到: {filepath}")
    
    # 显示统计信息
    history = env.get_step_history()
    play_actions = sum(1 for step in history if step['action_type'] == 'play')
    discard_actions = sum(1 for step in history if step['action_type'] == 'discard')
    
    print(f"\n游戏统计:")
    print(f"  出牌动作: {play_actions} 次")
    print(f"  弃牌动作: {discard_actions} 次")
    print(f"  总动作: {len(history)} 次")
    
    # 清理
    if os.path.exists("complete_episode_test.json"):
        os.remove("complete_episode_test.json")
    
    print()

def main():
    """运行所有测试"""
    print("开始测试BalatroEnv环境的(s,a)保存功能\n")
    
    test_basic_functionality()
    test_step_history_recording()
    test_env_state_tracking()
    test_save_load_functionality()
    test_complete_episode()
    
    print("所有测试完成!")

if __name__ == "__main__":
    main()