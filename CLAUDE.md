# Balatro RL Project

## Project Overview
PPO-based reinforcement learning agent for a simplified Balatro card game (single round, 52-card deck, 8-card hand, 5 plays, 3 discards). Includes LLM-guided training with Hesitation Gate mechanism.

## Setup & Installation
```bash
pip install torch numpy matplotlib tqdm gym
# For LLM-guided training (Colab):
pip install openai vllm
# For ChatGPT reward / expert generation:
export OPENAI_API_KEY="your-key"
```

## Project Structure
```
Balatro_SimpleVersion/
├── models/
│   ├── card_agent.py              — CardEncoder, ActorCritic, orthogonal_init
│   └── joker_agent.py             — JokerSelectNet (位置感知 Embedding + 3层512 MLP)
├── envs/
│   ├── BalatroEnv.py              — 打牌环境 (226-dim obs, reset_with_jokers)
│   ├── joker_env.py               — 小丑牌选择环境 (41-dim obs, Discrete(25))
│   └── joint_env.py               — 联动编排器
├── training/
│   ├── ppo_utils.py               — GAE, set_seed, get_device, SimpleVecEnv
│   ├── train_card.py              — PPOAgent + 打牌训练循环
│   ├── train_joker.py             — JokerPPOAgent + 单独小丑牌训练
│   ├── train_joker_bc.py          — 行为克隆预训练 (GPT 专家数据 → JokerSelectNet)
│   └── train_joint.py             — 联动训练 (joker + card)
├── llm_train/
│   └── train_joker_hesitation.py  — Hesitation-Gated LLM Prior (Qwen3-32B + vLLM)
├── evaluation/
│   ├── eval_card.py               — 打牌评估
│   └── eval_joint.py              — 联动评估
├── utils/
│   ├── chatgpt_reward.py          — ChatGPT 小丑牌评分 (gpt-4o-mini)
│   ├── generate_joker_expert.py   — GPT 专家轨迹生成 (BC 数据)
│   └── visualization.py           — GameVisualizer (ANSI 彩色终端输出)
├── paper/                         — LaTeX 论文源码 + 参考文献
├── game_gui.py                    — Tkinter GUI (人工/AI 游玩)
├── PPO_train.py                   — 薄包装 (向后兼容)
├── evaluate_model.py              — 薄包装 (向后兼容)
├── TRAINING_GUIDE.md              — 详细训练指南 (中文)
├── PlayByGPT/                     — GPT 直接打牌基线
├── GeneTraj.py                    — 轨迹生成脚本
├── SelectTraj.py                  — 轨迹筛选脚本
├── MCTS.py                        — MCTS 基线
└── outputs/                       — 训练输出 (checkpoints, logs, plots)
```

## Architecture
- **Network**: ActorCritic with:
  - `CardEncoder`: Rank/Suit embedding (16-dim) via learned embeddings for 13 ranks and 4 suits
  - Shared encoder (3-layer): Linear(242,512) → Tanh → Linear(512,512) → Tanh → Linear(512,256) → Tanh
  - Type head: Linear(256, 2) — 0=discard, 1=play
  - Independent play subnet (2-hidden): Linear(256,512) → Tanh → Linear(512,256) → Tanh → Linear(256,52)
  - Independent discard subnet (2-hidden): Linear(256,512) → Tanh → Linear(512,256) → Tanh → Linear(256,52)
  - Value head: Linear(256,256) → Tanh → Linear(256,1)
  - Orthogonal initialization (output layers gain=0.01, hidden layers gain=1.0)
- **Observation** (226 dims):
  - `obs[0:52]` = hand cards one-hot
  - `obs[52:104]` = deck availability one-hot
  - `obs[104:156]` = played cards history one-hot
  - `obs[156:208]` = discarded cards history one-hot
  - `obs[208]` = play_count / max_play
  - `obs[209]` = discard_count / max_discard
  - `obs[210]` = cumulative_score / 1000 (capped at 1.0)
  - `obs[211:226]` = joker features (5 slots × 3: type_id, chips, mult)
- **Action**: hierarchical — type head (0=discard, 1=play) + Bernoulli card mask (52-dim)
- **Card indexing**: `idx = (rank-1)*4 + suit_map[suit]` where H=0, D=1, C=2, S=3

## Reward Design
- **Play**: `raw_score` — raw hand score (e.g. High Card ~5, Pair ~20, Flush ~150)
- **Discard**: `shaping_beta * delta + discard_cost` — MC simulated expected score change (30 sims, symmetric scaling)
  - `shaping_beta=0.3` (reduced from 1.0 to prevent discard bias)
  - `discard_cost=-0.5` (intrinsic cost per discard action)
  - Symmetric delta: positive and negative deltas treated equally (removed 0.2x negative scaling)
- **Invalid discard** (discard_count=0): penalty -10, forced to play
- **Card limit**: Play actions truncated to max 5 cards

## Training Features
- LR scheduling: warmup (5%) + cosine decay (eta_min=lr*0.1)
- Discard mask loss coefficient: 0.3 → 0.8 (10% warmup)
- Mask entropy coefficient: 0.1 → 0.02 (linear anneal)
- Entropy coefficient: 0.05 → 0.01 (linear anneal)
- H7: Type head mask prevents discard when discard_count=0
- Vectorized `evaluate_actions()` (no Python for-loop)

## Hesitation-Gated LLM Prior (llm_train/)
Paper Eq. 4-6 implementation for LLM-guided joker selection:
- **LLM Action Prior**: N-vote empirical distribution p_LLM(a|s) via Qwen3-32B (vLLM)
- **Hesitation Gate**: g(s) = 1 if h(s) < τ (uncertain state → query LLM for guidance)
- **Modified PPO loss**: L = clip + vcoef*v_loss - ecoef*H + α*g*KL(π||p_LLM)
- Designed for Google Colab with vLLM backend

## Training Commands
```bash
# 打牌 agent 单独训练
python training/train_card.py --total_steps 1000000

# 小丑牌 agent 单独训练 (启发式 reward)
python training/train_joker.py --total_episodes 50000

# 小丑牌 agent 训练 (ChatGPT reward, 需要 OPENAI_API_KEY)
python training/train_joker.py --total_episodes 10000 --use_chatgpt

# GPT 专家轨迹生成 (需要 OPENAI_API_KEY)
python utils/generate_joker_expert.py --num_episodes 500 --output outputs/joker/expert_data.npz

# 行为克隆预训练 (BC warm start)
python training/train_joker_bc.py --data outputs/joker/expert_data.npz --output outputs/joker/checkpoints/joker_bc.pt

# 联动训练 (可加载预训练的打牌 + BC joker checkpoint)
python training/train_joint.py --card_checkpoint outputs/card/checkpoints/xxx.pt --joker_checkpoint outputs/joker/checkpoints/joker_bc.pt

# Hesitation-Gated LLM Prior 训练 (Colab + vLLM)
python llm_train/train_joker_hesitation.py \
  --api_base http://YOUR_VLLM_HOST:8000/v1 \
  --llm_model Qwen/Qwen3-32B \
  --card_checkpoint outputs/card/checkpoints/xxx.pt

# GUI 游玩 (人工或 AI)
python game_gui.py

# 评估
python evaluation/eval_card.py                          # 打牌评估
python evaluation/eval_joint.py                         # 联动评估
python evaluate_model.py                                # 向后兼容
```

## Output Structure
```
outputs/
├── card/              ← train_card.py 输出
│   ├── checkpoints/   — 模型权重 (.pt)
│   ├── logs/          — 训练指标 (CSV)
│   └── plots/         — 训练曲线 (PNG)
├── joker/             ← train_joker.py 输出
│   ├── checkpoints/
│   ├── logs/
│   └── plots/
└── joint/             ← train_joint.py 输出
    ├── checkpoints/
    ├── logs/
    └── plots/
```

## Joker Auto-Sort
- Jokers are auto-sorted before scoring: additive chips → additive mult → multiplicative mult
- Agent only needs to learn WHAT to pick, not WHERE to place
- Multiplicative jokers: {11=Steel, 12=Stencil, 21=Blackboard, 29=Loyalty Card}
- Additive chips: {14=Banner, 20=Odd Todd, 22=Ice Cream, 23=Blue Joker, 24=Runner}

## BC Warm Start Pipeline
1. `generate_joker_expert.py` → GPT picks jokers → saves (obs, action, mask) to .npz
2. `train_joker_bc.py` → CrossEntropyLoss on masked logits → saves checkpoint
3. `train_joint.py --joker_checkpoint joker_bc.pt` → fine-tune with real card scores

## Important Conventions
- `ep_returns_plot` tracks only play rewards; `ep_returns_train` includes shaping
- Old checkpoints (obs_dim=111) are incompatible with current architecture (obs_dim=226)
- `_calculate_best_score(cards, return_hand_type=True)` returns (score, hand_type_name)
- Joint checkpoint 包含 `card_state_dict` 和 `joker_state_dict` 两个网络的权重
- Hesitation checkpoint 包含 `joker_state_dict` + `hesitation_stats`
- 中英文混用: 代码注释和文档中使用中文, 变量名和函数名使用英文
