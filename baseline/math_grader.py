"""Shared answer extraction + grading for the MATH dataset.

Single source of truth for correctness: the verl training reward
(baseline/reward.py) and the offline MATH-500 eval
(baseline/eval_math500.py) both import from here, so train reward and
eval accuracy can never disagree.

Normalization follows the standard Hendrycks MATH comparison code
(same family as verl's built-in math scorer and the
math-evaluation-harness vendored in eval/benchmarks).
"""

from __future__ import annotations


def last_boxed_only_string(string: str) -> str | None:
    """Return the last \\boxed{...} (or \\fbox{...}) substring, or None."""
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def remove_boxed(s: str) -> str | None:
    if s is None:
        return None
    if "\\boxed " in s:
        left = "\\boxed "
        if not s.startswith(left):
            return None
        return s[len(left) :]
    left = "\\boxed{"
    if not (s.startswith(left) and s.endswith("}")):
        return None
    return s[len(left) : -1]


def extract_boxed_answer(text: str) -> str | None:
    """Extract the content of the last \\boxed{} in a model response."""
    return remove_boxed(last_boxed_only_string(text))


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if not substr:
                continue
            if substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a, b = substr[0], substr[1]
                if b != "{":
                    new_str += "{" + a + "}{" + b + "}" + substr[2:]
                else:
                    new_str += "{" + a + "}" + substr[1:]
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a_int, b_int = int(a), int(b)
        if string == f"{a_int}/{b_int}":
            return "\\frac{" + str(a_int) + "}{" + str(b_int) + "}"
        return string
    except ValueError:
        return string


def _remove_right_units(string: str) -> str:
    # "\\text{ " only ever appears (at least in the MATH dataset) when
    # units are appended to the answer.
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        return splits[0]
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def _strip_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "").replace("%", "")
    # " 0." -> " ." and leading "0." -> "."
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if string and string[0] == ".":
        string = "0" + string
    # Keep only the RHS of a short "x = ..." style answer.
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def is_equiv(str1: str | None, str2: str | None) -> bool:
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        return _strip_string(str1) == _strip_string(str2)
    except Exception:
        return str1 == str2


def grade_answer(solution_str: str, ground_truth: str) -> bool:
    """Grade a full model response against a ground-truth final answer.

    ground_truth may be the bare answer ("42") or still boxed
    ("\\boxed{42}"); both are handled.
    """
    answer = extract_boxed_answer(solution_str)
    if answer is None:
        return False
    gt = remove_boxed(last_boxed_only_string(ground_truth))
    if gt is None:
        gt = ground_truth
    return is_equiv(answer, gt)
