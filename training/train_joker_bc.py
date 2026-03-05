# training/train_joker_bc.py
# -*- coding: utf-8 -*-
"""行为克隆 (Behavioral Cloning) 训练 JokerSelectNet：从 GPT 专家轨迹学习"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.joker_agent import JokerSelectNet
from training.ppo_utils import get_device, set_seed


def load_expert_data(data_path):
    """加载 .npz 专家数据"""
    data = np.load(data_path)
    obs = data["obs"]              # (N, 62)
    actions = data["actions"]      # (N,)
    action_masks = data["action_masks"]  # (N, 25)
    print(f"[Data] Loaded {len(obs)} samples from {data_path}")
    return obs, actions, action_masks


def train_bc(data_path, epochs=100, lr=1e-3, batch_size=256,
             output_path="outputs/joker/checkpoints/joker_bc.pt",
             val_split=0.2, patience=10, seed=42):
    """
    行为克隆训练

    Args:
        data_path: .npz 专家数据路径
        epochs: 最大训练轮数
        lr: 学习率
        batch_size: 批大小
        output_path: checkpoint 保存路径
        val_split: 验证集比例
        patience: early stopping 耐心
        seed: 随机种子
    """
    set_seed(seed)
    device = get_device()

    # 加载数据
    obs_np, actions_np, masks_np = load_expert_data(data_path)
    N = len(obs_np)

    # Train/Val split
    indices = np.random.permutation(N)
    val_size = int(N * val_split)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_obs = torch.as_tensor(obs_np[train_idx], dtype=torch.float32, device=device)
    train_actions = torch.as_tensor(actions_np[train_idx], dtype=torch.long, device=device)
    train_masks = torch.as_tensor(masks_np[train_idx], dtype=torch.float32, device=device)

    val_obs = torch.as_tensor(obs_np[val_idx], dtype=torch.float32, device=device)
    val_actions = torch.as_tensor(actions_np[val_idx], dtype=torch.long, device=device)
    val_masks = torch.as_tensor(masks_np[val_idx], dtype=torch.float32, device=device)

    print(f"[Split] Train: {len(train_idx)}, Val: {len(val_idx)}")

    # 创建网络
    net = JokerSelectNet(obs_dim=41, num_actions=25).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    best_val_loss = float('inf')
    best_epoch = 0
    no_improve = 0

    print(f"\n[Train] Starting BC training: {epochs} epochs, lr={lr}, batch_size={batch_size}")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>10} | {'Train Acc':>10} | {'Val Acc':>10} | {'LR':>10}")
    print("-" * 72)

    for epoch in range(1, epochs + 1):
        # --- Train ---
        net.train()
        perm = torch.randperm(len(train_obs), device=device)
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for start in range(0, len(train_obs), batch_size):
            mb = perm[start:start + batch_size]
            mb_obs = train_obs[mb]
            mb_actions = train_actions[mb]
            mb_masks = train_masks[mb]

            logits, _ = net(mb_obs, mb_masks)  # masked logits
            loss = criterion(logits, mb_actions)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * len(mb)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == mb_actions).sum().item()
            train_total += len(mb)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # --- Val ---
        net.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0

            for start in range(0, len(val_obs), batch_size):
                mb_obs = val_obs[start:start + batch_size]
                mb_actions = val_actions[start:start + batch_size]
                mb_masks = val_masks[start:start + batch_size]

                logits, _ = net(mb_obs, mb_masks)
                loss = criterion(logits, mb_actions)

                val_loss_sum += loss.item() * len(mb_obs)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == mb_actions).sum().item()
                val_total += len(mb_obs)

            val_loss = val_loss_sum / val_total
            val_acc = val_correct / val_total

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Logging
        if epoch <= 5 or epoch % 10 == 0 or val_loss < best_val_loss:
            marker = ""
            if val_loss < best_val_loss:
                marker = " *"
            print(f"{epoch:>6} | {train_loss:>10.4f} | {val_loss:>10.4f} | {train_acc:>9.1%} | {val_acc:>9.1%} | {current_lr:>10.6f}{marker}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0

            # 保存最佳模型
            torch.save({
                "state_dict": net.state_dict(),
                "config": {"obs_dim": 62, "num_actions": 25},
                "bc_info": {
                    "epochs": epoch,
                    "final_loss": val_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "data_size": N,
                }
            }, output_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n[Early Stop] No improvement for {patience} epochs. Best epoch: {best_epoch}")
                break

    print(f"\n[Done] Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"[Save] Checkpoint saved to {output_path}")

    return output_path


def main():
    p = argparse.ArgumentParser(description='Behavioral Cloning for Joker Selection')
    p.add_argument("--data", type=str, required=True,
                   help="Path to expert data .npz file")
    p.add_argument("--epochs", type=int, default=100,
                   help="Maximum training epochs")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate")
    p.add_argument("--batch_size", type=int, default=256,
                   help="Batch size")
    p.add_argument("--output", type=str, default="outputs/joker/checkpoints/joker_bc.pt",
                   help="Output checkpoint path")
    p.add_argument("--val_split", type=float, default=0.2,
                   help="Validation split ratio")
    p.add_argument("--patience", type=int, default=10,
                   help="Early stopping patience")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    args = p.parse_args()

    train_bc(
        data_path=args.data,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        output_path=args.output,
        val_split=args.val_split,
        patience=args.patience,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
