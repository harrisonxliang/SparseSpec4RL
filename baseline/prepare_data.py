"""Prepare MATH data for the verl GRPO baseline.

Train: MATH training split, levels 3-5 (DigitalLearningGmbH/MATH-lighteval).
Val:   MATH-500 held-out subset (HuggingFaceH4/MATH-500).

Output: parquet files in verl's RLHF dataset schema. The user prompt
matches the SparseSpec repo's "qwen3-math-thinking" template
(eval/benchmarks/utils.py) so both arms and the offline eval see the
exact same prompt text.

Usage (on the A100 box):
  python baseline/prepare_data.py --out-dir ~/data/sparsespec_baseline
"""

import argparse
import os

from datasets import load_dataset

from math_grader import extract_boxed_answer

# Must match eval/benchmarks/utils.py "qwen3-math-thinking".
INSTRUCTION = "\nPlease think step by step and then provide only the final answer within \\boxed{}."

TRAIN_DATASET = "DigitalLearningGmbH/MATH-lighteval"
VAL_DATASET = "HuggingFaceH4/MATH-500"
TRAIN_LEVELS = {"Level 3", "Level 4", "Level 5"}


def make_row(problem: str, ground_truth: str, data_source: str, split: str, idx: int, extra: dict):
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": problem + INSTRUCTION}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {"split": split, "index": idx, **extra},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=os.path.expanduser("~/data/sparsespec_baseline"))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Train: MATH levels 3-5 ----
    train = load_dataset(TRAIN_DATASET, "default", split="train")
    rows, skipped = [], 0
    for i, ex in enumerate(train):
        if ex["level"] not in TRAIN_LEVELS:
            continue
        gt = extract_boxed_answer(ex["solution"])
        if gt is None:
            skipped += 1
            continue
        rows.append(
            make_row(
                ex["problem"], gt, "math_l3to5_train", "train", i,
                {"level": ex["level"], "type": ex["type"]},
            )
        )
    print(f"train: kept {len(rows)} problems (levels 3-5), skipped {skipped} with no boxed answer")

    # ---- Val: MATH-500 ----
    val = load_dataset(VAL_DATASET, split="test")
    val_rows = []
    for i, ex in enumerate(val):
        val_rows.append(
            make_row(
                ex["problem"], ex["answer"], "math500", "test", i,
                {"level": ex["level"], "subject": ex["subject"], "unique_id": ex["unique_id"]},
            )
        )
    print(f"val: {len(val_rows)} problems")

    import pandas as pd

    train_path = os.path.join(args.out_dir, "math_l3to5_train.parquet")
    val_path = os.path.join(args.out_dir, "math500_test.parquet")
    pd.DataFrame(rows).to_parquet(train_path)
    pd.DataFrame(val_rows).to_parquet(val_path)
    print(f"wrote {train_path}")
    print(f"wrote {val_path}")


if __name__ == "__main__":
    main()
