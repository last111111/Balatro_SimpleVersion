# 🎮 Balatro PPO训练指南

## 📁 文件结构

训练后会生成以下结构化的输出文件：

```
Balatro_SimpleVersion/
├── outputs/
│   ├── checkpoints/          # 模型checkpoint
│   │   ├── ppo_run_<timestamp>_step_50000.pt
│   │   ├── ppo_run_<timestamp>_step_100000.pt
│   │   └── ppo_run_<timestamp>_final.pt
│   ├── logs/                 # 训练日志（CSV格式）
│   │   └── ppo_run_<timestamp>.csv
│   └── plots/                # 训练曲线图
│       └── ppo_run_<timestamp>_curve.png
├── envs/
│   └── BalatroEnv.py        # 环境定义（包含小丑牌系统）
└── PPO_train.py             # 训练脚本
```

## 🚀 快速开始

### 1. 基础训练（默认参数）

```bash
python PPO_train.py
```

默认配置：
- 总训练步数: 1,000,000
- 学习率: 3e-4
- 每50,000步保存checkpoint
- 每100个episode记录日志

### 2. 快速测试

```bash
python PPO_train.py --total_steps 10000
```

短时间内完成训练，用于验证代码是否正常工作。

### 3. 自定义训练

```bash
python PPO_train.py \
  --total_steps 500000 \
  --lr 0.0003 \
  --max_play 5 \
  --checkpoint_interval 25000 \
  --log_interval 50
```

## 📊 输出文件说明

### Checkpoint文件 (`.pt`)

包含完整的训练状态，可用于恢复训练或评估：

```python
checkpoint = torch.load('outputs/checkpoints/ppo_run_xxx_final.pt')
# 包含:
# - state_dict: 模型权重
# - optimizer: 优化器状态
# - step: 训练步数
# - episode: 完成的episode数
# - config: 所有超参数配置
```

### 日志文件 (`.csv`)

记录训练过程中的关键指标：

| 列名 | 说明 |
|------|------|
| episode | Episode编号 |
| steps | 累计训练步数 |
| return_train | 当前episode总回报（含reward shaping） |
| return_plot | 当前episode回报（仅出牌奖励） |
| avg_train_100 | 最近100个episode平均回报（训练） |
| avg_plot_100 | 最近100个episode平均回报（绘图） |
| max_plot | 历史最高分 |
| ecoef | 当前熵系数 |
| coef_dis | 当前弃牌mask系数 |

### 训练曲线图 (`.png`)

**4个子图展示：**

1. **左上**: Episode Return 原始曲线 + 移动平均
2. **右上**: 最近100个episode的滚动平均
3. **左下**: 历史最高分变化
4. **右下**: 训练摘要统计信息

## ⚙️ 命令行参数完整列表

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--total_steps` | 1000000 | 总训练步数 |
| `--update_steps` | 4096 | 每次PPO更新收集的步数 |
| `--seed` | 0 | 随机种子 |

### PPO超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lr` | 3e-4 | 学习率 |
| `--gamma` | 0.99 | 折扣因子 |
| `--gae_lambda` | 0.95 | GAE lambda |
| `--clip` | 0.2 | PPO裁剪比率 |
| `--value_coef` | 0.5 | 价值损失系数 |
| `--entropy_coef_start` | 0.05 | 初始熵系数 |
| `--entropy_coef_end` | 0.01 | 最终熵系数 |
| `--epochs` | 4 | PPO每次更新的epoch数 |
| `--mb_size` | 1024 | Minibatch大小 |

### 环境参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max_hand_size` | 8 | 最大手牌数 |
| `--max_play` | 5 | 每轮最多出牌次数 |
| `--shaping_beta` | 1.0 | 奖励塑形系数（0=无塑形） |

### Checkpoint和日志

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint_interval` | 50000 | 每N步保存checkpoint |
| `--log_interval` | 100 | 每N个episode记录日志 |

## 🃏 小丑牌系统

当前训练**默认不使用小丑牌**（`jokers=[]`），但环境已完全支持30种小丑牌效果。

### 激活小丑牌训练

编辑 `envs/BalatroEnv.py` 的 `reset()` 方法：

```python
def reset(self):
    # ... 其他初始化代码 ...

    # 添加小丑牌
    self.add_joker(JokerType.JOKER)       # +4 mult
    self.add_joker(JokerType.RUNNER)      # 顺子成长型
    self.add_joker(JokerType.BLUE_JOKER)  # 牌库剩余越多越强

    return self._get_observation()
```

### 30种小丑牌类型

详见 `envs/BalatroEnv.py` 中的 `JokerType` 类定义，包括：
- 基础加成型: JOKER (+4 mult)
- 花色触发型: GREEDY/LUSTY/WRATHFUL/GLUTTONOUS
- 牌型触发型: JOLLY/ZANY/MAD/CRAZY/DROLL
- 成长型: RUNNER, RIDE_THE_BUS, ICE_CREAM
- 倍数型: BLACKBOARD, LOYALTY_CARD, JOKER_STENCIL
- 特殊触发型: FIBONACCI, EVEN_STEVEN, ODD_TODD等

## 📈 训练监控

### 实时监控

训练时进度条会显示：

```
Training: 45%|████▌     | 450000/1000000 [12:34<15:21, 597step/s]
episodes: 2341
train_recent5: 285.3    # 最近5个episode平均分（含塑形）
plot_recent5: 245.8     # 最近5个episode平均分（仅出牌）
ret_train: 312.4        # 当前episode总分（含塑形）
ret_plot: 268.0         # 当前episode总分（仅出牌）
ecoef: 0.032            # 当前熵系数
coef_dis: 0.100         # 弃牌mask系数
```

### 分析训练日志

使用pandas分析CSV日志：

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取日志
df = pd.read_csv('outputs/logs/ppo_run_xxx.csv')

# 绘制学习曲线
plt.plot(df['episode'], df['avg_plot_100'])
plt.xlabel('Episode')
plt.ylabel('Avg Return (100 episodes)')
plt.show()

# 查看最佳表现
print(f"最高分: {df['max_plot'].max()}")
print(f"最终100episode平均: {df['avg_plot_100'].iloc[-1]}")
```

## 🔄 从Checkpoint恢复训练

目前保存了checkpoint但未实现自动恢复功能。如需恢复训练，可手动加载：

```python
# 加载checkpoint
checkpoint = torch.load('outputs/checkpoints/ppo_run_xxx_step_50000.pt')

# 恢复模型
agent.net.load_state_dict(checkpoint['state_dict'])
agent.opt.load_state_dict(checkpoint['optimizer'])

# 查看训练进度
print(f"已训练步数: {checkpoint['step']}")
print(f"已完成episode: {checkpoint['episode']}")
```

## 💡 训练建议

### 推荐配置（平衡效果和时间）

```bash
python PPO_train.py \
  --total_steps 500000 \
  --checkpoint_interval 50000 \
  --shaping_beta 1.0
```

预计训练时间：15-30分钟（取决于硬件）

### 长时间训练（追求最佳性能）

```bash
python PPO_train.py \
  --total_steps 2000000 \
  --lr 0.0002 \
  --checkpoint_interval 100000
```

### 调试/快速验证

```bash
python PPO_train.py \
  --total_steps 5000 \
  --checkpoint_interval 2000 \
  --log_interval 10
```

## 🎯 性能基准

参考：DDQN实现（4出牌+3弃牌）平均分230

PPO目标：通过更好的长期规划和策略优化，超越DDQN的表现。

## 📝 注意事项

1. **Checkpoint保存**: 每次训练都会创建独立的时间戳目录，不会覆盖之前的训练结果
2. **兼容性**: 同时保存一份 `training_curve.png` 到根目录，保持向后兼容
3. **小丑牌**: 当前训练不包含小丑牌，系统已就绪但需手动激活
4. **奖励塑形**: `shaping_beta=1.0` 表示弃牌时会有额外的塑形奖励，设为0则关闭

## 🐛 故障排除

### 训练中断

Ctrl+C中断训练时会自动保存最终模型到 `outputs/checkpoints/`

### 内存不足

减小 `--mb_size` 或 `--update_steps`:

```bash
python PPO_train.py --mb_size 512 --update_steps 2048
```

### 训练不稳定

调整学习率和裁剪比率：

```bash
python PPO_train.py --lr 0.0001 --clip 0.1
```

---

**Happy Training! 🎮🚀**
