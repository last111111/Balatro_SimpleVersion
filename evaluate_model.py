# evaluate_model.py
# -*- coding: utf-8 -*-
"""薄包装：向后兼容入口，实际逻辑在 evaluation/eval_card.py"""

from evaluation.eval_card import (  # noqa: F401
    GameVisualizer, load_latest_checkpoint, get_hand_name,
    evaluate_model, main
)

if __name__ == "__main__":
    main()
