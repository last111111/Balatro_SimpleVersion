# -*- coding: utf-8 -*-
import argparse
import math
import pandas as pd

def select_extremes(df_action: pd.DataFrame, top_frac: float = 0.10, bottom_frac: float = 0.10,
                    action_name: str = "play"):
    """
    从单一 action 子集里选出 advantage 的前 top_frac 和后 bottom_frac。
    返回两个 DataFrame：top, bottom，并添加 subset 字段用于标记。
    """
    if df_action.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 确保 advantage 列是数值
    df = df_action.copy()
    df["advantage"] = pd.to_numeric(df["advantage"], errors="coerce")
    df = df.dropna(subset=["advantage"])

    n = len(df)
    if n == 0:
        return pd.DataFrame(), pd.DataFrame()

    n_top = max(1, int(math.ceil(n * top_frac)))
    n_bot = max(1, int(math.ceil(n * bottom_frac)))

    # Top: 按 advantage 降序
    top_df = df.sort_values(by="advantage", ascending=False).head(n_top).copy()
    top_df["subset"] = f"{action_name}_top10"

    # Bottom: 按 advantage 升序
    bot_df = df.sort_values(by="advantage", ascending=True).head(n_bot).copy()
    bot_df["subset"] = f"{action_name}_bottom10"

    # 保险起见：去重（同一条记录不会同时出现在 top/bottom，因为排序方向相反）
    # 但考虑到极端相等，我们可以确保按 episode,t,action 去重
    top_df = top_df.drop_duplicates(subset=["episode", "t", "action"])
    bot_df = bot_df.drop_duplicates(subset=["episode", "t", "action"])

    # 如果仍有重叠（理论上 play 子集不会和 discard 子集重叠），这里也可去重
    overlap_keys = set(zip(top_df["episode"], top_df["t"], top_df["action"])) & \
                   set(zip(bot_df["episode"], bot_df["t"], bot_df["action"]))
    if overlap_keys:
        # 如果真的重叠了（极少见），优先保留 top 集合，去掉 bottom 的重叠项
        mask = bot_df.apply(lambda r: (r["episode"], r["t"], r["action"]) not in overlap_keys, axis=1)
        bot_df = bot_df[mask]

    return top_df, bot_df


def main(input_csv: str, output_csv: str):
    # 读取 CSV
    df = pd.read_csv(input_csv)
    # 标准化 action 字段（可能有大小写差异）
    df["action"] = df["action"].astype(str).str.strip().str.lower()

    # 分为 play / discard 子集
    df_play = df[df["action"] == "play"].copy()
    df_dis  = df[df["action"] == "discard"].copy()

    # 分别选取极端样本
    play_top, play_bot = select_extremes(df_play, top_frac=0.10, bottom_frac=0.10, action_name="play")
    dis_top,  dis_bot  = select_extremes(df_dis,  top_frac=0.10, bottom_frac=0.10, action_name="discard")

    # 合并并再次按唯一键去重（避免任何重叠）
    merged = pd.concat([play_top, play_bot, dis_top, dis_bot], axis=0, ignore_index=True)
    merged = merged.drop_duplicates(subset=["episode", "t", "action"])

    # 输出 CSV
    merged.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[Save] Extracted extremes saved to: {output_csv}")

    # 简单打印一些统计
    def count_subset(tag):
        return int((merged["subset"] == tag).sum()) if "subset" in merged.columns else 0

    print("[Summary]")
    print(f"  Play top10:     {count_subset('play_top10')}")
    print(f"  Play bottom10:  {count_subset('play_bottom10')}")
    print(f"  Discard top10:  {count_subset('discard_top10')}")
    print(f"  Discard bottom10:{count_subset('discard_bottom10')}")
    print(f"  Total rows:     {len(merged)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", type=str, default="traj.csv", help="Input rollout CSV path")
    p.add_argument("--output_csv", type=str, default="selected_traj.csv", help="Output CSV path")
    args = p.parse_args()

    main(args.input_csv, args.output_csv)