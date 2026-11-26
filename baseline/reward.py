"""Rule-based reward for verl (custom_reward_function).

1.0 if the last \\boxed{} answer in the response matches the reference,
else 0.0. Unparseable responses (no \\boxed) score 0.0 but are counted
so we can track the unparseable rate during training.

Wired up via:
  custom_reward_function.path=baseline/reward.py
  custom_reward_function.name=compute_score
"""

import os
import sys

# verl loads this file by path with importlib, so make the sibling
# grader importable regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from math_grader import extract_boxed_answer, grade_answer  # noqa: E402


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    try:
        if extract_boxed_answer(solution_str) is None:
            # No boxed answer within the response budget -> reward 0.
            return 0.0
        return 1.0 if grade_answer(solution_str, ground_truth) else 0.0
    except Exception:
        return 0.0
