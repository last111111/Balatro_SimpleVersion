# check_review_stats.py
# -*- coding: utf-8 -*-
"""
Check counts of Correct vs Incorrect in llm_review_results.csv
"""

import pandas as pd

def check_correct_incorrect(csv_path="llm_review_results.csv"):
    df = pd.read_csv(csv_path)

    # 确保列名正确
    if "is_correct" not in df.columns:
        raise ValueError("CSV must contain an 'is_correct' column")

    counts = df["is_correct"].value_counts(dropna=False)

    correct_count = counts.get("Correct", 0)
    incorrect_count = counts.get("Incorrect", 0)
    total = len(df)

    print("=== Review Statistics ===")
    print(f"Total rows: {total}")
    print(f"Correct:   {correct_count} ({correct_count/total:.2%})")
    print(f"Incorrect: {incorrect_count} ({incorrect_count/total:.2%})")

    return {"total": total, "correct": correct_count, "incorrect": incorrect_count}

if __name__ == "__main__":
    check_correct_incorrect("llm_review_results.csv")