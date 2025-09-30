# -*- coding: utf-8 -*-
"""
Iteratively build and refine a Balatro action guideline using LLM:
- Read llm_review_results.csv
- In batches of 5 rows:
    - First batch => build initial guideline (with rules included)
    - Each subsequent batch => refine/improve guideline with the new examples + rules
- Save intermediate guidelines and final guideline

Requirements:
  pip install openai pandas tqdm
  Ensure config.py contains OPENAI_API_KEY
"""

import os
import json
from typing import List, Dict, Any, Optional
import pandas as pd
from tqdm import tqdm

from openai import OpenAI
from config import OPENAI_API_KEY


# ======================
# OpenAI Client
# ======================
client = OpenAI(api_key=OPENAI_API_KEY)


# ======================
# 1) Balatro Rules (Static Text)
# ======================
def build_balatro_rules_text() -> str:
    """
    Provide a clear, stable rule text for Balatro to be injected into prompts.
    This ensures the LLM has consistent grounding when summarizing/refining guidelines.
    """
    lines = []
    lines.append("Balatro Rules Summary:")
    lines.append("- Each step, exactly one action is chosen: 0=discard, 1=play.")
    lines.append("- Hand size is up to 8 typically; a binary mask selects cards by hand positions (0/1 per slot).")
    lines.append("- For PLAY:")
    lines.append("  - You may select 1~5 cards by mask.")
    lines.append("  - The environment will automatically evaluate all valid combinations within the selected cards and use the best-scoring one.")
    lines.append("  - Scoring formula: (sum of ranks in the chosen combination + base score of the pattern) × multiplier.")
    lines.append("- For DISCARD:")
    lines.append("  - Selected cards are removed and replaced by drawing from the deck (if available).")
    lines.append("  - Discard does not score by itself, but may improve future hand potential.")
    lines.append("- Recognized patterns and typical logic:")
    lines.append("  - Straight Flush (5 cards): straight + flush.")
    lines.append("  - Four of a Kind (4 cards).")
    lines.append("  - Full House (5 cards): three of a kind + a pair.")
    lines.append("  - Flush (5 cards): all same suit.")
    lines.append("  - Straight (5 cards): 5 distinct ranks in sequence.")
    lines.append("  - Three of a Kind (3 cards).")
    lines.append("  - Two Pair (≥4 cards).")
    lines.append("  - One Pair (2 cards).")
    lines.append("  - High Card: if no pattern is formed, only a single highest card counts (not the sum of all selected).")
    lines.append("- The environment picks the best-scoring valid pattern among the selected cards.")
    lines.append("- Typical limits: max 5 plays, max 3 discards (per game) – exact values may vary by environment.")
    lines.append("- Strategic note: Discard can be used to improve the hand without scoring penalty, so it is often valuable early if the hand is weak.")
    return "\n".join(lines)


# ======================
# 2) Utilities
# ======================
def parse_json_list(x):
    """Safely parse JSON array string to Python list."""
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return []
    return [] if pd.isna(x) else x


def format_batch_for_prompt(batch_df: pd.DataFrame) -> str:
    """
    Format 5 rows of llm_review_results into a readable English block for the LLM.
    Extracts: hand_before, action, cards, is_correct, reason, recommended_action_type, recommended_selected_cards, recommended_reason.
    """
    lines = []
    for idx, row in batch_df.iterrows():
        hand = parse_json_list(row.get("hand_before", "[]"))
        selected = parse_json_list(row.get("cards", "[]"))

        action = str(row.get("action", "")).strip().lower()
        is_correct = str(row.get("is_correct", ""))
        reason = str(row.get("reason", ""))

        rec_type = row.get("recommended_action_type", None)
        rec_cards = row.get("recommended_selected_cards", None)
        rec_cards_list = []
        if isinstance(rec_cards, str):
            try:
                rec_cards_list = json.loads(rec_cards)
            except Exception:
                rec_cards_list = []
        rec_reason = row.get("recommended_reason", None)

        # Human-friendly formatting
        hand_str = ", ".join(map(str, hand)) if hand else "(empty)"
        sel_str  = ", ".join(map(str, selected)) if selected else "(none)"

        lines.append("---- Example ----")
        lines.append(f"- Hand: [{hand_str}]")
        lines.append(f"- Action: {action}")
        lines.append(f"- Selected cards: [{sel_str}]")
        lines.append(f"- LLM Review: {is_correct}")
        lines.append(f"- Reason: {reason}")
        if is_correct.lower() == "incorrect":
            lines.append(f"- Recommended action type: {rec_type}")
            lines.append(f"- Recommended selected cards: {rec_cards_list}")
            lines.append(f"- Recommended reason: {rec_reason}")
        lines.append("")
    return "\n".join(lines)


# ======================
# 3) Prompt Builders
# ======================
def build_initial_guideline_prompt(batch_text: str, rules_text: str) -> List[Dict[str, str]]:
    """
    Prompt to ask the LLM to build an initial guideline based on examples + rules.
    """
    system_prompt = (
        "You are a Balatro expert. Study real reviewed actions and distill them into a concise and practical guideline. "
        "The guideline should help players decide when to PLAY vs DISCARD and how to select cards well, based on the examples and the official rules. "
        "Make it self-contained, unambiguous, prioritized, and preferably in short bullet points."
    )
    user_prompt = (
        f"=== OFFICIAL RULES ===\n{rules_text}\n\n"
        "Here are 5 reviewed examples (with Correct/Incorrect decisions and reasons). "
        "Please summarize them into a guideline that captures the key decision principles.\n\n"
        f"{batch_text}\n\n"
        "Return ONLY the guideline text (no extra commentary)."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def refine_guideline_prompt(current_guideline: str, batch_text: str, rules_text: str) -> List[Dict[str, str]]:
    """
    Prompt to refine an existing guideline given new examples + rules.
    """
    system_prompt = (
        "You are a Balatro expert and an editorial coach. You refine and improve an existing guideline "
        "to better reflect more examples, resolve conflicts, and improve clarity and utility. "
        "Ground your decisions in both the official rules and the provided reviewed examples."
    )
    user_prompt = (
        f"=== OFFICIAL RULES ===\n{rules_text}\n\n"
        "Below is the CURRENT guideline and a set of 5 more reviewed examples.\n"
        "Please revise the guideline to incorporate new insights, resolve any contradictions, and keep it concise, prioritized, and practical. "
        "Preserve strong rules, adjust weak ones, and ensure the guideline stays self-contained.\n\n"
        "=== CURRENT GUIDELINE ===\n"
        f"{current_guideline}\n\n"
        "=== NEW EXAMPLES ===\n"
        f"{batch_text}\n\n"
        "Return ONLY the revised guideline text (no extra commentary)."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def ask_llm_for_guideline(messages: List[Dict[str, str]], model: str = "gpt-4o-mini") -> str:
    """
    Call OpenAI chat completions to get the guideline text.
    """
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=messages
    )
    text = resp.choices[0].message.content.strip()
    return text


# ======================
# 4) Main pipeline
# ======================
def iterative_guideline_from_csv(
    input_csv: str = "llm_review_results.csv",
    chunk_size: int = 5,
    out_dir: str = "guidelines",
    model: str = "gpt-4o-mini"
) -> str:
    """
    Iteratively build and refine a guideline based on llm_review_results.
    - Load CSV
    - Chunk rows by 5
    - Use first chunk to build initial guideline
    - Each subsequent chunk refines the guideline (rules included at every step)
    - Save intermediate versions and final version
    Returns the final guideline string.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(input_csv)

    n = len(df)
    if n == 0:
        raise ValueError("Input CSV is empty.")

    # Split into chunks
    chunks = []
    for i in range(0, n, chunk_size):
        chunks.append(df.iloc[i:i+chunk_size])

    guideline = None
    rules_text = build_balatro_rules_text()

    pbar = tqdm(total=len(chunks), desc="Guideline Iter Refinement", ncols=100)
    for idx, batch_df in enumerate(chunks, start=1):
        batch_text = format_batch_for_prompt(batch_df)

        if idx == 1:
            # Build initial guideline from first batch
            msgs = build_initial_guideline_prompt(batch_text, rules_text)
            guideline = ask_llm_for_guideline(msgs, model=model)
        else:
            # Refine using current guideline + new batch + rules
            msgs = refine_guideline_prompt(guideline, batch_text, rules_text)
            guideline = ask_llm_for_guideline(msgs, model=model)

        # Save intermediate
        iter_path = os.path.join(out_dir, f"guideline_iter_{idx}.txt")
        with open(iter_path, "w", encoding="utf-8") as f:
            f.write(guideline)

        pbar.set_postfix({"iter": idx})
        pbar.update(1)

    pbar.close()

    # Save final
    final_path = os.path.join(out_dir, "guideline_final.txt")
    with open(final_path, "w", encoding="utf-8") as f:
        f.write(guideline)

    print(f"[Save] Final guideline -> {final_path}")
    return guideline


if __name__ == "__main__":
    # You can adjust parameters here
    INPUT_CSV = "PlayByGPT/llm_review_results.csv"
    CHUNK_SIZE = 5
    OUT_DIR = "guidelines"
    MODEL = "gpt-4o-mini"

    iterative_guideline_from_csv(
        input_csv=INPUT_CSV,
        chunk_size=CHUNK_SIZE,
        out_dir=OUT_DIR,
        model=MODEL
    )