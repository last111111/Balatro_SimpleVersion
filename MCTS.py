# mcts_train.py
import time, math, copy, random, pickle, csv, sys, os
from collections import Counter
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt

# === 这里改成你的环境文件名 ===
from envs.BalatroEnv import BalatroEnv

# ----------------------
# 进度条（无第三方依赖）
# ----------------------
def fmt_time(s):
    if s < 60: return f"{int(s)}s"
    m, s = divmod(int(s), 60)
    if m < 60: return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"

def print_progress(ep, total, start_time, last_reward, last_avg):
    spent = time.time() - start_time
    rate = ep / spent if spent > 0 else 0
    remain = (total - ep) / rate if rate > 0 else 0
    width = 30
    filled = int(width * ep / total)
    bar = "█" * filled + "·" * (width - filled)
    msg = (f"\r[ {bar} ] {ep}/{total} | "
           f"R={last_reward:.1f} | Avg10={last_avg:.1f} | "
           f"ETA {fmt_time(remain)}")
    sys.stdout.write(msg)
    sys.stdout.flush()
    if ep == total:
        sys.stdout.write("\n")

# ----------------------
# 工具：状态键与候选动作
# ----------------------
def state_key(env: BalatroEnv):
    return (tuple(sorted(env.hand)), env.play_count)

def candidate_actions(env: BalatroEnv, max_actions=48):
    actions = []
    n = len(env.hand)
    idxs = list(range(n))

    # 出牌（1, mask）：所有 ≤5 张组合，按即时分排序，截断
    play_actions = []
    for r in range(1, min(5, n) + 1):
        for combo in combinations(idxs, r):
            mask = [0] * env.max_hand_size
            for i in combo:
                mask[i] = 1
            play_actions.append((1, mask))

    scored = []
    for a in play_actions:
        s = env.best_mask_score(a[1])
        scored.append((s, a))
    scored.sort(key=lambda x: x[0], reverse=True)
    actions.extend([a for _, a in scored[:max_actions // 2]])

    # 弃牌（0, mask）：所有单张 + 若干两张低点数组合
    for i in idxs:
        m = [0] * env.max_hand_size
        m[i] = 1
        actions.append((0, m))
    low = sorted([(env.hand[i][0], i) for i in idxs])[:min(6, n)]
    low_idxs = [i for _, i in low]
    for combo in combinations(low_idxs, 2):
        if len(actions) >= max_actions: break
        m = [0] * env.max_hand_size
        for i in combo: m[i] = 1
        actions.append((0, m))

    if not actions:
        if n > 0:
            m = [0]*env.max_hand_size; m[0]=1
            return [(0, m)]
        return [(1, [0]*env.max_hand_size)]
    return actions

# ----------------------
# MCTS
# ----------------------
class Node:
    __slots__ = ("parent","children","N","W","Q","action","terminal","key")
    def __init__(self, parent, action, key, terminal=False):
        self.parent = parent
        self.children = []       # list[(action, child)]
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.action = action
        self.terminal = terminal
        self.key = key

def ucb(parent, child, c_puct=1.4):
    if child.N == 0: return float("inf")
    return child.Q + c_puct * math.sqrt(math.log(parent.N + 1) / child.N)

def mcts_plan(env: BalatroEnv, n_sim=32, max_depth=20, c_puct=1.4):
    root = Node(None, None, state_key(env), terminal=False)
    cache = {root.key: root}

    for _ in range(n_sim):
        node = root
        sim_env = copy.deepcopy(env)
        depth = 0

        # 1) 选择
        while node.children and not node.terminal and depth < max_depth:
            scores = [(ucb(node, ch, c_puct), (a, ch)) for (a, ch) in node.children]
            _, (a, node) = max(scores, key=lambda t: t[0])
            _, _, done, _ = sim_env.step(a)
            depth += 1
            if done:
                node.terminal = True
                break

        # 2) 扩展
        if not node.terminal and depth < max_depth:
            kids = []
            for a in candidate_actions(sim_env):
                e2 = copy.deepcopy(sim_env)
                _, _, done, _ = e2.step(a)
                k = state_key(e2)
                ch = cache.get(k)
                if ch is None:
                    ch = Node(node, a, k, terminal=done)
                    cache[k] = ch
                kids.append((a, ch))
            node.children = kids

        # 3) 模拟
        total = 0.0
        d = depth
        roll_env = copy.deepcopy(sim_env)
        while d < max_depth:
            acts = candidate_actions(roll_env)
            a = random.choice(acts)
            _, r, done, _ = roll_env.step(a)
            total += r
            d += 1
            if done: break

        # 4) 回传
        while node is not None:
            node.N += 1
            node.W += total
            node.Q = node.W / node.N
            node = node.parent

    if not root.children:
        acts = candidate_actions(env)
        return random.choice(acts)
    best_act, _ = max(root.children, key=lambda ac: ac[1].Q)
    return best_act

# ----------------------
# 训练主函数
# ----------------------
def train(episodes=200, sims_per_move=32, seed=7, max_depth=20, window=10, out_dir="."):
    random.seed(seed); np.random.seed(seed)
    env = BalatroEnv()
    returns, avgs = [], []
    policy = {}   # state_key -> action

    os.makedirs(out_dir, exist_ok=True)
    csv_path   = os.path.join(out_dir, "training_log.csv")
    plot_path  = os.path.join(out_dir, "reward_plot.png")
    policy_path= os.path.join(out_dir, "balatro_mcts_policy.pkl")

    t0 = time.time()
    for ep in range(1, episodes+1):
        env.reset()
        done, total_r, steps = False, 0.0, 0
        while not done and steps < 200:
            k = state_key(env)
            a = policy.get(k)
            if a is None:
                a = mcts_plan(env, n_sim=sims_per_move, max_depth=max_depth)
                policy[k] = a
            _, r, done, _ = env.step(a)
            total_r += r
            steps += 1

        returns.append(total_r)
        avg = float(np.mean(returns[-window:]))
        avgs.append(avg)
        print_progress(ep, episodes, t0, total_r, avg)

    # 保存日志
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode","reward","avg_reward"])
        for i,(r,a) in enumerate(zip(returns, avgs), 1):
            w.writerow([i, r, a])

    # 保存策略
    with open(policy_path, "wb") as f:
        pickle.dump(policy, f)

    # 画图
    x = np.arange(1, len(returns)+1)
    plt.figure(figsize=(8,4.5))
    plt.plot(x, returns, label="Reward")
    plt.plot(x, avgs, label="AvgReward(10)")
    plt.xlabel("Episode"); plt.ylabel("Reward")
    plt.title("BalatroEnv - MCTS Training")
    plt.legend(); plt.tight_layout()
    plt.savefig(plot_path)

    print(f"\nSaved:\n  {policy_path}\n  {csv_path}\n  {plot_path}")
    return policy_path, csv_path, plot_path

# ----------------------
# 直接运行
# ----------------------
if __name__ == "__main__":
    # 可在命令行按需修改参数（简单起见，这里直接写死）
    train(
        episodes=200,       # 训练回合数
        sims_per_move=32,   # 每步 MCTS 模拟次数
        seed=7,
        max_depth=20,
        window=10,
        out_dir="."
    )