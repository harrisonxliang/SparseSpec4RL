"""Shared helpers for the correctness gates (see ``integration/gates/__init__.py``).

Importable WITHOUT ``serve.rl.engine`` / flashinfer / triton / vllm: only
torch, numpy, transformers and the pure-torch parts of ``serve.rl`` (``base``,
``weights``, ``reference_engine``) are used, so the same module serves the
SparseSpec env, the stock-vLLM baseline env and a CPU-only machine.

What is in here:

* prompts        ``load_math_prompts`` builds MATH prompts exactly as verl's
                 ``RLHFDataset`` does from the baseline parquet;
                 ``load_prompts_file`` reads token-id lists from JSON.
* engines        ``build_engine("sparsespec" | "reference", model, **opts)``.
* weights        ``read_state_dict`` (HF safetensors dir), ``bf16_diff_report``.
* trainer side   ``hf_trainer_logprobs`` scores (prompt, response) pairs the way
                 verl's actor does (``dp_actor._forward_micro_batch``);
                 ``hf_greedy`` / ``hf_teacher_forced_logits`` for the greedy gate.
* statistics     ``compare_logprobs``, ``tracking_coefficient``.
* gate 3b files  ``greedy_tokens_record`` / ``read_greedy_tokens`` (one token-file
                 schema for the engine, HF and stock-vLLM sides),
                 ``describe_prompt_mismatch``.
* reporting      ``Checks``, ``write_json``, ``write_text``, ``md_table``.
* shared box     ``gpu_preflight`` (nvidia-smi; refuses to start, never kills).
"""

from __future__ import annotations

import ast
import contextlib
import datetime
import glob
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
from typing import Sequence

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "baseline", "results", "sparsespec_checks")

# Same environment variables and fallbacks as baseline/config/grpo_base.sh.
DEFAULT_MODEL = os.environ.get("MODEL_PATH", "Qwen/Qwen3-0.6B")
DEFAULT_DATA_DIR = os.environ.get("DATA_DIR", os.path.expanduser("~/data/sparsespec_baseline"))
DEFAULT_CKPT_ROOT = os.environ.get("CKPT_ROOT", os.path.expanduser("~/ckpts/sparsespec_baseline"))
DEFAULT_TRAIN_PARQUET = os.path.join(DEFAULT_DATA_DIR, "math_l3to5_train.parquet")
DEFAULT_STEP25_CKPT = os.path.join(DEFAULT_CKPT_ROOT, "vllm_baseline_seed0", "hf", "global_step_25")

# KEEP IN SYNC with baseline/prepare_data.py (INSTRUCTION, TRAIN_DATASET,
# TRAIN_LEVELS); INSTRUCTION is also duplicated in baseline/eval_math500.py and
# eval/benchmarks/utils.py ("qwen3-math-thinking"). Only the HF-dataset
# fallback of load_math_prompts uses them (the parquet written by
# prepare_data.py already contains the instruction), and that fallback calls
# check_prompt_constants_in_sync() first, so a drift is an error, not a
# silently different prompt.
INSTRUCTION = "\nPlease think step by step and then provide only the final answer within \\boxed{}."
TRAIN_DATASET = "DigitalLearningGmbH/MATH-lighteval"
TRAIN_LEVELS = {"Level 3", "Level 4", "Level 5"}
# data.max_prompt_length in baseline/config/grpo_base.sh (verl drops longer prompts)
MAX_PROMPT_LENGTH = 1024

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

GIB = 1024**3


def log(msg: str = "") -> None:
    print(msg, flush=True)


# Exit status contract of every gate: 0 PASS, 1 FAIL, anything else = no verdict.
# 2 is argparse's usage error and gate 3b's REFUSED (inputs do not match).
EXIT_CRASH = 3


def run_main(main, prefix: str) -> None:
    """``sys.exit(main())`` for a gate's ``__main__`` block, except that an
    unexpected exception (CUDA OOM, an upstream AssertionError, an ImportError)
    exits with EXIT_CRASH after its traceback instead of CPython's 1, which the
    gate runner would read as a FAIL verdict. ``SystemExit`` (argparse errors,
    the gates' own "FAIL: ..." messages) and ``KeyboardInterrupt`` pass through."""
    try:
        status = main()
    except Exception as exc:  # noqa: BLE001 - reported as a crash, never swallowed
        # before the traceback, so that the traceback's last line stays the last error line of the log
        log(f"{prefix} CRASHED: unexpected {type(exc).__name__}; no verdict was computed "
            f"(exit status {EXIT_CRASH}); traceback below")  # fmt: skip
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        sys.exit(EXIT_CRASH)
    sys.exit(status)


# ----------------------------------------------------------------------------
# prompts
# ----------------------------------------------------------------------------
def _prepare_data_constants() -> dict:
    """The prompt constants as written in baseline/prepare_data.py.

    Parsed, not imported: that module imports ``datasets`` and ``math_grader``
    at import time.
    """
    path = os.path.join(REPO_ROOT, "baseline", "prepare_data.py")
    with open(path) as f:
        tree = ast.parse(f.read())
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("INSTRUCTION", "TRAIN_DATASET", "TRAIN_LEVELS"):
                found[name] = ast.literal_eval(node.value)
    return found


def check_prompt_constants_in_sync() -> None:
    """Raise if this module's copy of the prompt constants drifted from baseline/prepare_data.py."""
    theirs = _prepare_data_constants()
    mine = {"INSTRUCTION": INSTRUCTION, "TRAIN_DATASET": TRAIN_DATASET, "TRAIN_LEVELS": TRAIN_LEVELS}
    for name, value in mine.items():
        if name not in theirs:
            raise RuntimeError(f"baseline/prepare_data.py no longer defines {name}; update integration/gates/common.py")
        if theirs[name] != value:
            raise RuntimeError(
                f"{name} in integration/gates/common.py ({value!r}) differs from baseline/prepare_data.py "
                f"({theirs[name]!r}); the gates would not test the training prompts"
            )


def build_prompt_ids(tokenizer, messages) -> list[int]:
    """Token ids of one chat prompt, built like verl's ``RLHFDataset.__getitem__``
    (``utils/dataset/rl_dataset.py``): chat template with the generation
    prompt, rendered to text, then tokenized without extra special tokens."""
    raw_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return [int(t) for t in tokenizer.encode(raw_prompt, add_special_tokens=False)]


def _messages_from_cell(cell) -> list[dict]:
    # pandas returns the list<struct<content, role>> column as an array of dicts
    return [{"role": str(m["role"]), "content": str(m["content"])} for m in cell]


def _load_math_grader():
    import importlib.util

    path = os.path.join(REPO_ROOT, "baseline", "math_grader.py")
    spec = importlib.util.spec_from_file_location("_sparsespec_gates_math_grader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_math_messages(data_path: str | None = None) -> tuple[list[list[dict]], str]:
    """Chat messages of every training problem, plus a description of the source.

    ``data_path`` is the train parquet written by baseline/prepare_data.py
    (default ``$DATA_DIR/math_l3to5_train.parquet``). An explicitly given path
    must exist. Only when the DEFAULT parquet is absent are the rows rebuilt
    from the HF dataset the way prepare_data.py builds them.
    """
    path = os.path.expanduser(data_path) if data_path else DEFAULT_TRAIN_PARQUET
    if os.path.exists(path):
        import pandas as pd

        frame = pd.read_parquet(path, columns=["prompt"])
        return [_messages_from_cell(cell) for cell in frame["prompt"]], f"parquet:{path}"
    if data_path:
        raise FileNotFoundError(f"training parquet not found: {path}")

    log(f"[prompts] {path} not found; rebuilding the rows from {TRAIN_DATASET} like baseline/prepare_data.py")
    check_prompt_constants_in_sync()
    from datasets import load_dataset

    extract_boxed_answer = _load_math_grader().extract_boxed_answer
    train = load_dataset(TRAIN_DATASET, "default", split="train")
    rows = []
    for example in train:
        if example["level"] not in TRAIN_LEVELS:
            continue
        if extract_boxed_answer(example["solution"]) is None:  # prepare_data.py skips these rows
            continue
        rows.append([{"role": "user", "content": example["problem"] + INSTRUCTION}])
    return rows, f"hf:{TRAIN_DATASET} levels 3-5 (rebuilt like baseline/prepare_data.py)"


def load_math_prompts(
    tokenizer,
    n: int,
    data_path: str | None = None,
    seed: int = 0,
    *,
    max_prompt_length: int = MAX_PROMPT_LENGTH,
    return_meta: bool = False,
):
    """``n`` MATH training prompts as token-id lists, built exactly as verl builds them.

    Rows are taken in a ``seed``-shuffled order; prompts longer than
    ``max_prompt_length`` tokens are skipped (``data.filter_overlong_prompts``).
    The same (tokenizer, data, seed, n) gives the same prompts in every env,
    so the SparseSpec gate and the stock-vLLM reference use identical inputs.
    """
    messages, source = load_math_messages(data_path)
    order = list(range(len(messages)))
    random.Random(seed).shuffle(order)
    prompts: list[list[int]] = []
    rows: list[int] = []
    for index in order:
        ids = build_prompt_ids(tokenizer, messages[index])
        if len(ids) > max_prompt_length:
            continue
        prompts.append(ids)
        rows.append(index)
        if len(prompts) == n:
            break
    if len(prompts) < n:
        raise ValueError(f"only {len(prompts)} prompts of at most {max_prompt_length} tokens in {source}, need {n}")
    meta = {
        "source": source,
        "seed": seed,
        "rows": rows,
        "num_rows_available": len(messages),
        "max_prompt_length": max_prompt_length,
    }
    return (prompts, meta) if return_meta else prompts


def load_prompts_file(path: str) -> list[list[int]]:
    """Prompts as token-id lists from JSON: ``[[1, 2, ...], ...]`` or ``{"prompts": [[...], ...]}``."""
    with open(os.path.expanduser(path)) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("prompts")
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: expected a non-empty list of token-id lists")
    prompts = []
    for i, row in enumerate(data):
        if not isinstance(row, list) or not row or not all(isinstance(t, int) and not isinstance(t, bool) for t in row):
            raise ValueError(f"{path}: prompt {i} is not a non-empty list of ints")
        prompts.append(list(row))
    return prompts


def resolve_prompts(
    *, prompts_file: str | None, model_dir: str, n: int, data_path: str | None = None, seed: int = 0
) -> tuple[list[list[int]], dict]:
    """The gate's prompts: the first ``n`` of ``--prompts-file`` if given, else MATH training prompts."""
    if prompts_file:
        prompts = load_prompts_file(prompts_file)
        if len(prompts) < n:
            raise ValueError(f"{prompts_file} holds {len(prompts)} prompts but {n} were requested")
        return prompts[:n], {"source": f"prompts-file:{os.path.abspath(os.path.expanduser(prompts_file))}", "num": n}
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    prompts, meta = load_math_prompts(tokenizer, n, data_path=data_path, seed=seed, return_meta=True)
    meta["num"] = n
    return prompts, meta


def repeat_interleave(prompts: Sequence[Sequence[int]], n: int) -> list[list[int]]:
    """Each prompt ``n`` times in a row, as verl's ``gen_batch.repeat(repeat_times=n, interleave=True)``."""
    return [list(p) for p in prompts for _ in range(n)]


# ----------------------------------------------------------------------------
# model directories, state dicts, engines
# ----------------------------------------------------------------------------
def resolve_model_dir(model_path: str) -> str:
    """Local directory of a model: an existing directory, or the snapshot of a hub id."""
    path = os.path.expanduser(str(model_path))
    if os.path.isdir(path):
        return os.path.abspath(path)
    if path.startswith(("/", ".", "~")) or path.count("/") != 1:
        raise FileNotFoundError(f"model directory not found: {model_path}")
    from huggingface_hub import snapshot_download

    return snapshot_download(model_path)


def read_state_dict(model_dir: str) -> dict[str, torch.Tensor]:
    """All tensors of a local HF checkpoint directory (``*.safetensors``), on CPU, dtypes as stored."""
    from safetensors.torch import load_file

    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no *.safetensors files in {model_dir}")
    state: dict[str, torch.Tensor] = {}
    for path in files:
        part = load_file(path, device="cpu")
        overlap = state.keys() & part.keys()
        if overlap:
            raise ValueError(f"duplicate keys across shards of {model_dir}: {sorted(overlap)[:5]}")
        state.update(part)
    return state


def eos_ids_from_checkpoint(model_dir: str) -> list[int]:
    """EOS ids the way both engines derive their default (generation config + model config)."""
    from transformers import AutoConfig, GenerationConfig

    from serve.rl.base import normalize_eos_ids

    eos: list = []
    try:
        eos.extend(normalize_eos_ids(GenerationConfig.from_pretrained(model_dir).eos_token_id))
    except OSError:
        pass
    eos.extend(normalize_eos_ids(getattr(AutoConfig.from_pretrained(model_dir), "eos_token_id", None)))
    return list(normalize_eos_ids(eos))


def bf16_diff_report(state_a: dict, state_b: dict) -> dict:
    """How many weights differ once both state dicts are cast to bfloat16 (bitwise).

    This is what the engine and the trainer actually run with: both hold
    bf16 values, so two float32 checkpoints that round to the same bf16
    tensors are the same model for them.
    """
    from serve.rl.weights import normalize_hf_key

    a = {normalize_hf_key(k): v for k, v in state_a.items()}
    b = {normalize_hf_key(k): v for k, v in state_b.items()}
    common = sorted(set(a) & set(b))
    total = changed = tensors_changed = 0
    shape_mismatch = []
    for key in common:
        ta, tb = a[key], b[key]
        if not ta.is_floating_point() or not tb.is_floating_point():
            continue
        if ta.shape != tb.shape:
            shape_mismatch.append(key)
            continue
        bits_a = ta.to(torch.bfloat16).contiguous().view(torch.int16)
        bits_b = tb.to(torch.bfloat16).contiguous().view(torch.int16)
        num = int((bits_a != bits_b).sum().item())
        total += ta.numel()
        changed += num
        tensors_changed += int(num > 0)
    return {
        "num_tensors": len(common),
        "num_tensors_changed": tensors_changed,
        "num_elements": total,
        "num_changed": changed,
        "fraction_changed": (changed / total) if total else 0.0,
        "only_in_a": sorted(set(a) - set(b))[:10],
        "only_in_b": sorted(set(b) - set(a))[:10],
        "shape_mismatch": shape_mismatch[:10],
    }


def default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def add_engine_args(parser, *, mem_fraction: float = 0.45, max_batch_size: int = 64) -> None:
    """Engine flags shared by the gates; defaults are the training config's
    (baseline/config/grpo_base.sh + baseline/run_seed_sparsespec.sh)."""
    group = parser.add_argument_group("engine")
    group.add_argument("--engine", choices=["sparsespec", "reference"], default="sparsespec")
    group.add_argument("--device", default=default_device(), help="default: cuda:0 if available, else cpu")
    group.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16", help="engine weight dtype")
    group.add_argument(
        "--mem-fraction", type=float, default=mem_fraction, help="engine GPU share (rollout.gpu_memory_utilization)"
    )
    group.add_argument("--max-batch-size", type=int, default=max_batch_size)
    group.add_argument("--spec-stride", type=int, default=8)
    group.add_argument("--budget-ratio", type=float, default=0.05)
    group.add_argument("--num-min-budget", type=int, default=128)
    group.add_argument("--no-cuda-graph", action="store_true", help="sparsespec: run without CUDA graphs")
    group.add_argument("--engine-seed", type=int, default=0)
    group.add_argument(
        "--kv-capacity-tokens", type=int, default=None, help="sparsespec: override the planned KV pool size"
    )


def engine_opts_from_args(args) -> dict:
    opts = dict(
        mem_fraction=args.mem_fraction,
        max_batch_size=args.max_batch_size,
        spec_stride=args.spec_stride,
        budget_ratio=args.budget_ratio,
        num_min_budget=args.num_min_budget,
        kv_cache="pillar",
        cuda_graph=not args.no_cuda_graph,
        seed=args.engine_seed,
        dtype=args.dtype,
        device=args.device,
    )
    if args.engine == "sparsespec" and args.kv_capacity_tokens is not None:
        opts["kv_capacity_tokens"] = args.kv_capacity_tokens
    return opts


def build_engine(kind: str, model_path: str, **opts):
    """A rollout engine behind the ``serve.rl.base.RolloutEngine`` interface.

    ``serve.rl.engine`` (CUDA + flashinfer + triton) is imported only for
    ``kind == "sparsespec"``; ``"reference"`` needs torch + transformers only.
    Engines start awake. SparseSpecEngine: one instance per process.
    """
    if kind == "reference":
        from serve.rl.reference_engine import ReferenceEngine

        return ReferenceEngine(model_path, **opts)
    if kind == "sparsespec":
        from serve.rl.engine import SparseSpecEngine

        return SparseSpecEngine(model_path, **opts)
    raise ValueError(f"unknown engine kind {kind!r} (expected 'sparsespec' or 'reference')")


def check_responses(result, max_new_tokens: int, eos_ids, *, greedy: bool = False) -> list[str]:
    """Violations of the response contract in a ``GenerationResult`` (empty list = ok).

    Per response: non-empty, at most ``max_new_tokens`` tokens, either ends
    with the first EOS it contains or has exactly ``max_new_tokens`` tokens,
    one finite log-prob <= 0 per token (exactly 0.0 when greedy), and a
    finish reason that matches. ``result.logprobs`` may be None (generated
    with ``return_logprobs=False``): the log-prob checks are then skipped.
    """
    eos = set(int(t) for t in eos_ids)
    problems: list[str] = []
    all_logprobs = result.logprobs if result.logprobs is not None else [None] * len(result.token_ids)
    for i, (tokens, logprobs, reason) in enumerate(zip(result.token_ids, all_logprobs, result.finish_reasons)):
        if len(tokens) == 0:
            problems.append(f"response {i}: empty")
            continue
        if logprobs is not None and len(tokens) != len(logprobs):
            problems.append(f"response {i}: {len(tokens)} tokens but {len(logprobs)} log-probs")
        if len(tokens) > max_new_tokens:
            problems.append(f"response {i}: {len(tokens)} tokens > max_new_tokens={max_new_tokens}")
        eos_positions = [j for j, t in enumerate(tokens) if t in eos]
        if eos_positions:
            if eos_positions[0] != len(tokens) - 1:
                problems.append(f"response {i}: tokens after the first EOS (position {eos_positions[0]})")
            if reason != "stop":
                problems.append(f"response {i}: ends with EOS but finish_reason={reason!r}")
        else:
            if len(tokens) != max_new_tokens:
                problems.append(f"response {i}: no EOS and {len(tokens)} != max_new_tokens={max_new_tokens}")
            if reason != "length":
                problems.append(f"response {i}: no EOS but finish_reason={reason!r}")
        if logprobs is None:
            continue
        bad = [x for x in logprobs if not math.isfinite(x) or x > 0.0]
        if bad:
            problems.append(f"response {i}: {len(bad)} log-prob(s) not finite or > 0, e.g. {bad[0]!r}")
        if greedy and any(x != 0.0 for x in logprobs):
            problems.append(f"response {i}: greedy log-probs must be exactly 0.0")
    if not (len(result.token_ids) == len(all_logprobs) == len(result.finish_reasons)):
        problems.append("token_ids / logprobs / finish_reasons have different lengths")
    return problems


def fingerprint(token_lists: Sequence[Sequence[int]]) -> str:
    """Stable hash of a list of token-id lists."""
    payload = json.dumps([[int(t) for t in row] for row in token_lists], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def first_difference(a: Sequence[int], b: Sequence[int]) -> int:
    """First index where two token sequences differ; -1 if they are identical.

    If one is a strict prefix of the other the result is the shorter length.
    """
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1 if len(a) == len(b) else min(len(a), len(b))


# ----------------------------------------------------------------------------
# greedy token files (gate 3b): one schema for the engine, HF and stock-vLLM sides
# ----------------------------------------------------------------------------
GREEDY_TOKENS_KIND = "gate_b_greedy_tokens"
GREEDY_TOKENS_FORMAT = 1
GREEDY_SIDES = ("engine", "hf", "vllm")
# per-step arrays that a side may carry; each has one entry per response token
GREEDY_STEP_ARRAYS = ("logprobs", "sampled_logprobs", "sampled_ranks", "top2_token_ids", "top2_logprobs", "margins")


class GreedyFileError(ValueError):
    """A gate 3b token file is missing, malformed or internally inconsistent."""


def prompts_sha256(prompts: Sequence[Sequence[int]]) -> str:
    """Hash that identifies a prompt set (sha256 of the compact JSON of the token-id lists)."""
    return fingerprint(prompts)


def greedy_tokens_record(
    *,
    side: str,
    engine: str,
    model: str,
    prompts: Sequence[Sequence[int]],
    prompts_meta: dict | None,
    token_ids: Sequence[Sequence[int]],
    finish_reasons: Sequence[str],
    max_new_tokens: int,
    eos_token_ids: Sequence[int],
    **extra,
) -> dict:
    """The JSON object of one side of the greedy comparison (see ``read_greedy_tokens``)."""
    if side not in GREEDY_SIDES:
        raise ValueError(f"side must be one of {GREEDY_SIDES}, got {side!r}")
    record = {
        "kind": GREEDY_TOKENS_KIND,
        "format_version": GREEDY_TOKENS_FORMAT,
        "side": side,
        "engine": engine,
        "model": model,
        "temperature": 0.0,
        "max_new_tokens": int(max_new_tokens),
        "eos_token_ids": sorted(int(t) for t in eos_token_ids),
        "num_prompts": len(prompts),
        "prompts_sha256": prompts_sha256(prompts),
        "prompts_meta": prompts_meta,
        "prompts": [[int(t) for t in p] for p in prompts],
        "token_ids": [[int(t) for t in row] for row in token_ids],
        "finish_reasons": list(finish_reasons),
    }
    record.update(extra)
    return record


def _is_int_list(row) -> bool:
    return isinstance(row, list) and all(isinstance(t, int) and not isinstance(t, bool) for t in row)


def read_greedy_tokens(path: str, *, side: str | None = None) -> dict:
    """Load and validate a gate 3b token file (written by ``greedy_tokens_record``).

    Raises ``GreedyFileError`` naming the file and the first problem found:
    wrong kind / format / side, prompts that do not match their own stored
    hash (the file was edited or truncated), or per-prompt / per-step arrays
    of the wrong length.
    """
    where = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(where):
        raise GreedyFileError(f"{where}: file not found")
    try:
        with open(where) as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise GreedyFileError(f"{where}: not valid JSON ({exc})") from exc
    if not isinstance(data, dict) or data.get("kind") != GREEDY_TOKENS_KIND:
        found = data.get("kind") if isinstance(data, dict) else type(data).__name__
        raise GreedyFileError(f"{where}: not a gate 3b token file (kind={found!r}, expected {GREEDY_TOKENS_KIND!r})")
    if data.get("format_version") != GREEDY_TOKENS_FORMAT:
        raise GreedyFileError(
            f"{where}: format_version {data.get('format_version')!r}, this code reads {GREEDY_TOKENS_FORMAT}"
        )
    if side is not None and data.get("side") != side:
        raise GreedyFileError(f"{where}: holds the {data.get('side')!r} side, expected {side!r}")
    for key in ("prompts", "token_ids", "finish_reasons", "prompts_sha256", "max_new_tokens", "eos_token_ids"):
        if key not in data:
            raise GreedyFileError(f"{where}: missing key {key!r}")
    prompts, tokens = data["prompts"], data["token_ids"]
    if not isinstance(prompts, list) or not prompts or not all(_is_int_list(p) and p for p in prompts):
        raise GreedyFileError(f"{where}: 'prompts' must be a non-empty list of non-empty token-id lists")
    if prompts_sha256(prompts) != data["prompts_sha256"]:
        raise GreedyFileError(
            f"{where}: the stored prompts do not match the stored prompts_sha256 (file edited or corrupted)"
        )
    if not isinstance(tokens, list) or len(tokens) != len(prompts) or not all(_is_int_list(r) for r in tokens):
        raise GreedyFileError(f"{where}: 'token_ids' must hold one token-id list per prompt ({len(prompts)})")
    if not isinstance(data["finish_reasons"], list) or len(data["finish_reasons"]) != len(prompts):
        raise GreedyFileError(f"{where}: 'finish_reasons' must hold one entry per prompt ({len(prompts)})")
    for key in GREEDY_STEP_ARRAYS:
        if key not in data:
            continue
        rows = data[key]
        if not isinstance(rows, list) or len(rows) != len(prompts):
            raise GreedyFileError(f"{where}: {key!r} must hold one list per prompt ({len(prompts)})")
        for i, (row, toks) in enumerate(zip(rows, tokens)):
            if not isinstance(row, list) or len(row) != len(toks):
                raise GreedyFileError(
                    f"{where}: {key!r} of prompt {i} has {len(row) if isinstance(row, list) else '?'} entries for "
                    f"{len(toks)} response tokens"
                )
    return data


def describe_prompt_mismatch(expected: Sequence[Sequence[int]], actual: Sequence[Sequence[int]]) -> str | None:
    """None if the two prompt sets are identical (hash and exact ids), else where they first differ."""
    expected = [list(p) for p in expected]
    actual = [list(p) for p in actual]
    if prompts_sha256(expected) == prompts_sha256(actual) and expected == actual:
        return None
    head = f"sha256 {prompts_sha256(expected)[:16]}... vs {prompts_sha256(actual)[:16]}..."
    if len(expected) != len(actual):
        return f"{len(expected)} prompts vs {len(actual)} prompts ({head})"
    for i, (a, b) in enumerate(zip(expected, actual)):
        if a != b:
            j = first_difference(a, b)
            return (
                f"prompt {i} differs at token {j} (lengths {len(a)} vs {len(b)}; "
                f"ids {a[j] if j < len(a) else 'end'} vs {b[j] if j < len(b) else 'end'}) ({head})"
            )
    return f"same ids but different hashes ({head})"  # unreachable unless fingerprint changes


# ----------------------------------------------------------------------------
# trainer side: HuggingFace forward passes
# ----------------------------------------------------------------------------
def load_hf_model(model_dir: str, device: str, dtype: str = "bfloat16"):
    """HF causal LM with PARAMETERS in ``dtype`` on ``device``, eval mode, no grads.

    ``from_pretrained(torch_dtype=...)`` casts the parameters and leaves float
    buffers alone, in particular Qwen3's rotary ``inv_freq`` stays float32, as
    it does in the verl actor (FSDP mixed precision casts parameters only) and
    in vLLM. The model is therefore only MOVED here: ``Module.to(dtype=...)``
    would round ``inv_freq`` to bf16 and bend every rotary angle by up to
    0.4%, a different model at long positions.
    """
    from transformers import AutoModelForCausalLM

    torch_dtype = DTYPES[dtype]
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch_dtype)
    except TypeError:  # newer transformers renamed the argument
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch_dtype)
    wrong = [name for name, param in model.named_parameters() if param.is_floating_point() and param.dtype != torch_dtype]
    if wrong:
        raise RuntimeError(f"from_pretrained left {len(wrong)} parameter(s) not in {dtype}, e.g. {wrong[:3]}")
    model = model.to(torch.device(device)).eval()
    model.requires_grad_(False)
    return model


def _autocast(device: torch.device):
    # verl: `with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16)`
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def hf_trainer_logprobs(
    model,
    prompts: Sequence[Sequence[int]],
    responses: Sequence[Sequence[int]],
    *,
    temperature: float = 1.0,
    token_budget: int = 8192,
    pad_token_id: int = 0,
    progress: str | None = None,
) -> list[list[float]]:
    """Log-prob of every response token under ``model``, computed as verl's actor computes old log-probs.

    Mirrors ``DataParallelPPOActor._forward_micro_batch`` (workers/actor/dp_actor.py):
    the model holds bf16 weights, the forward runs under
    ``torch.autocast(dtype=bfloat16)`` (on CUDA), the logits are divided by
    the temperature IN PLACE in the model's dtype (``logits.div_(temperature)``),
    and the log-softmax is evaluated in float32 (flash-attn's cross-entropy
    kernel and autocast's ``log_softmax`` both upcast), then gathered at the
    response tokens.

    Differences that do not change the math: sequences are right padded with
    an explicit ``attention_mask`` and ``position_ids`` (verl left-pads the
    prompt, or removes padding with ``use_remove_padding``), and batches are
    formed by a padded-token budget, longest first.

    Returns ``out[i][j] = log p(responses[i][j] | prompts[i], responses[i][:j])``.
    """
    if len(prompts) != len(responses):
        raise ValueError(f"{len(prompts)} prompts but {len(responses)} responses")
    if temperature <= 0:
        raise ValueError("hf_trainer_logprobs needs temperature > 0")
    device = next(model.parameters()).device
    lengths = [len(p) + len(r) for p, r in zip(prompts, responses)]
    for i, (p, r) in enumerate(zip(prompts, responses)):
        if len(p) == 0 or len(r) == 0:
            raise ValueError(f"sequence {i}: empty prompt or response")
    order = sorted(range(len(prompts)), key=lambda i: -lengths[i])
    out: list = [None] * len(prompts)
    cursor = 0
    while cursor < len(order):
        longest = lengths[order[cursor]]
        size = max(1, token_budget // longest)
        chunk = order[cursor : cursor + size]
        cursor += size

        input_ids = torch.full((len(chunk), longest), int(pad_token_id), dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(chunk), longest), dtype=torch.long, device=device)
        for row, i in enumerate(chunk):
            seq = torch.tensor(list(prompts[i]) + list(responses[i]), dtype=torch.long, device=device)
            input_ids[row, : seq.numel()] = seq
            attention_mask[row, : seq.numel()] = 1
        # verl.utils.model.compute_position_id_with_mask
        position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0)

        with _autocast(device):
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
            ).logits
            logits.div_(temperature)
            for row, i in enumerate(chunk):
                start, num = len(prompts[i]) - 1, len(responses[i])
                labels = torch.tensor(list(responses[i]), dtype=torch.long, device=device)
                picked = []
                # the logits at position t predict token t + 1. The float32 copy and
                # its log-softmax are made 512 positions at a time and dropped before
                # the next slice (2 x 512 x vocab x 4 B = 0.58 GiB for Qwen3); a whole
                # 2048-token row costs 3 x 1.16 GiB, because the previous row's result
                # is still bound while the next one is built
                for lo in range(0, num, 512):
                    hi = min(lo + 512, num)
                    piece = torch.log_softmax(logits[row, start + lo : start + hi].to(torch.float32), dim=-1)
                    picked.append(piece.gather(-1, labels[lo:hi].unsqueeze(-1)).squeeze(-1))
                    del piece
                out[i] = torch.cat(picked).double().cpu().tolist()
        del logits
        if progress:
            log(f"[{progress}] scored {min(cursor, len(order))}/{len(order)} sequences")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


@torch.no_grad()
def hf_greedy(model, prompt: Sequence[int], max_new_tokens: int, eos_ids) -> list[int]:
    """Greedy continuation of ONE prompt (no batching, no padding) with a hand-written
    loop over the HF KV cache: ``generate()`` is avoided on purpose, the
    checkpoint's generation config (Qwen3: sampling with top-k/top-p) must not
    leak into the reference. The response ends with the first EOS (kept) or
    has ``max_new_tokens`` tokens: the engines' contract."""
    device = next(model.parameters()).device
    eos = set(int(t) for t in eos_ids)
    step_input = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    past = None
    tokens: list[int] = []
    with _autocast(device):
        for _ in range(max_new_tokens):
            out = model(input_ids=step_input, past_key_values=past, use_cache=True)
            past = out.past_key_values
            token = int(out.logits[0, -1].to(torch.float32).argmax().item())
            tokens.append(token)
            if token in eos:
                break
            step_input = torch.tensor([[token]], dtype=torch.long, device=device)
    return tokens


@torch.no_grad()
def hf_teacher_forced_logits(model, prompt: Sequence[int], response: Sequence[int]) -> torch.Tensor:
    """float32 logits ``[len(response), vocab]``: row ``j`` is the model's
    next-token distribution after ``prompt + response[:j]`` (one full forward
    pass without a KV cache)."""
    device = next(model.parameters()).device
    ids = torch.tensor([list(prompt) + list(response[:-1])], dtype=torch.long, device=device)
    with _autocast(device):
        logits = model(input_ids=ids, use_cache=False).logits
    return logits[0, len(prompt) - 1 :].to(torch.float32)


# ----------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------
def _flatten_pair(a, b) -> tuple[np.ndarray, np.ndarray]:
    if len(a) != len(b):
        raise ValueError(f"{len(a)} sequences vs {len(b)} sequences")
    flat_a: list[float] = []
    flat_b: list[float] = []
    for i, (row_a, row_b) in enumerate(zip(a, b)):
        if len(row_a) != len(row_b):
            raise ValueError(f"sequence {i}: {len(row_a)} log-probs vs {len(row_b)}")
        flat_a.extend(row_a)
        flat_b.extend(row_b)
    return np.asarray(flat_a, dtype=np.float64), np.asarray(flat_b, dtype=np.float64)


def compare_logprobs(a, b) -> dict:
    """Per-token agreement of two sets of log-probs (lists of per-sequence lists).

    ``count``            tokens compared (both values finite)
    ``num_nonfinite``    tokens dropped because either value is nan / inf
    ``mean`` / ``median`` / ``p99`` / ``max``   of ``|delta log p|``
    ``max_prob_ratio``   ``exp(max)``: the worst per-token probability ratio
    ``prob_diff_mean`` / ``prob_diff_max``      of ``|p_a - p_b|`` (what verl logs as
                         ``training/rollout_probs_diff_mean`` / ``_max``)
    """
    x, y = _flatten_pair(a, b)
    finite = np.isfinite(x) & np.isfinite(y)
    stats = {"count": int(finite.sum()), "num_nonfinite": int((~finite).sum())}
    if stats["count"] == 0:
        stats.update(
            mean=math.nan, median=math.nan, p99=math.nan, max=math.nan, max_prob_ratio=math.nan,
            prob_diff_mean=math.nan, prob_diff_max=math.nan,
        )  # fmt: skip
        return stats
    x, y = x[finite], y[finite]
    delta = np.abs(x - y)
    prob_diff = np.abs(np.exp(x) - np.exp(y))
    stats.update(
        mean=float(delta.mean()),
        median=float(np.median(delta)),
        p99=float(np.percentile(delta, 99)),
        max=float(delta.max()),
        max_prob_ratio=float(np.exp(delta.max())),
        prob_diff_mean=float(prob_diff.mean()),
        prob_diff_max=float(prob_diff.max()),
    )
    return stats


def tracking_coefficient(engine, ref_a, ref_b) -> dict:
    """Which of two reference models do the engine's log-probs follow?

    Least-squares ``beta`` in ``engine - ref_a = beta * (ref_b - ref_a) + noise``
    over all tokens: about 0 when the engine holds A's weights, about 1 when
    it holds B's. Unlike a ratio of mean ``|delta|`` it stays decisive when A
    and B are so close that their difference is below the bf16 noise floor,
    because the noise averages out over many tokens. ``se`` is the standard
    error of ``beta``; ``signal_rms`` is the RMS of ``ref_b - ref_a``.
    """
    e, a = _flatten_pair(engine, ref_a)
    _, b = _flatten_pair(engine, ref_b)
    finite = np.isfinite(e) & np.isfinite(a) & np.isfinite(b)
    e, a, b = e[finite], a[finite], b[finite]
    signal = b - a
    energy = float(np.dot(signal, signal))
    if e.size < 2 or energy == 0.0:
        return {"beta": math.nan, "se": math.nan, "count": int(e.size), "signal_rms": 0.0}
    beta = float(np.dot(e - a, signal) / energy)
    residual = (e - a) - beta * signal
    se = float(math.sqrt(float(np.dot(residual, residual)) / (e.size - 1) / energy))
    return {"beta": beta, "se": se, "count": int(e.size), "signal_rms": float(math.sqrt(energy / e.size))}


# ----------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------
class Checks:
    """Named pass/fail sub-checks; prints each one as it is recorded."""

    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        ok = bool(ok)
        self.items.append({"name": name, "ok": ok, "detail": detail})
        log(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}" if detail else f"[{'PASS' if ok else 'FAIL'}] {name}")
        return ok

    @property
    def failed(self) -> list[dict]:
        return [item for item in self.items if not item["ok"]]

    @property
    def ok(self) -> bool:
        return not self.failed

    def markdown(self) -> str:
        rows = [["PASS" if item["ok"] else "**FAIL**", item["name"], item["detail"]] for item in self.items]
        return md_table(["result", "check", "detail"], rows)


def fmt_num(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.{digits}g}"
    return str(value)


def md_table(headers: Sequence[str], rows: Sequence[Sequence]) -> str:
    def cell(value) -> str:
        return fmt_num(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(cell(h) for h in headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in rows)
    return "\n".join(lines)


STAT_COLUMNS = ("count", "mean", "median", "p99", "max", "max_prob_ratio")


def stats_table(named_stats: Sequence[tuple[str, dict]]) -> str:
    """Markdown table of ``compare_logprobs`` results, one row per (name, stats)."""
    rows = [[name] + [stats.get(col) for col in STAT_COLUMNS] for name, stats in named_stats]
    return md_table(["|delta log p| of"] + list(STAT_COLUMNS), rows)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in (sorted(obj) if isinstance(obj, (set, frozenset)) else obj)]
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None  # strict JSON has no nan / inf
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, torch.Tensor):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    return str(obj)


def write_json(path: str, obj) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_jsonable(obj), f, indent=2, allow_nan=False)
        f.write("\n")
    return path


def write_text(path: str, text: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(text if text.endswith("\n") else text + "\n")
    return path


def environment_info(device: str | None = None) -> dict:
    """Versions and hardware, recorded in every result file."""
    import transformers

    info = {
        "time_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "argv": list(sys.argv),
    }
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "-C", REPO_ROOT, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        info["git_commit"] = None
    if device is not None and torch.device(device).type == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.device(device))
        info["gpu"] = props.name
        info["gpu_total_gb"] = props.total_memory / GIB
    return info


# ----------------------------------------------------------------------------
# shared GPU box: look before allocating, never kill anything
# ----------------------------------------------------------------------------
_SMI_QUERY = "index,uuid,name,memory.total,memory.used,memory.free"


def parse_nvidia_smi(text: str) -> list[dict]:
    """Rows of ``nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits``."""
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "uuid": parts[1],
                    "name": parts[2],
                    "total_mib": float(parts[3]),
                    "used_mib": float(parts[4]),
                    "free_mib": float(parts[5]),
                }
            )
        except ValueError:
            continue
    return gpus


def _visible_gpu(gpus: list[dict], ordinal: int) -> dict | None:
    """The nvidia-smi row behind ``cuda:<ordinal>``, honouring CUDA_VISIBLE_DEVICES (indices or UUIDs)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or visible.strip() == "":
        candidates = gpus
    else:
        candidates = []
        for entry in (e.strip() for e in visible.split(",") if e.strip()):
            match = None
            if entry.isdigit():
                match = next((g for g in gpus if g["index"] == int(entry)), None)
            else:
                match = next((g for g in gpus if g["uuid"].startswith(entry)), None)
            if match is None:
                return None
            candidates.append(match)
    return candidates[ordinal] if 0 <= ordinal < len(candidates) else None


def gpu_preflight(
    required_free_gb: float | None,
    device: str = "cuda:0",
    *,
    share: float = 0.0,
    extra_gb: float = 0.0,
    smi_output: str | None = None,
) -> dict:
    """Print the GPU's memory state and refuse to continue (``SystemExit``) when
    too little is free on ``device``.

    The requirement is ``required_free_gb`` GiB, or, when that is None,
    ``share`` x the card's total memory + ``extra_gb``. The A100 is shared:
    this only LOOKS (nvidia-smi, no CUDA context is created) and stops this
    script; it never touches other processes. No-op for CPU devices.
    ``smi_output`` lets tests inject nvidia-smi's CSV output.
    """
    dev = torch.device(device)
    if dev.type != "cuda":
        log(f"[preflight] device {device}: no GPU preflight needed")
        return {"device": device, "skipped": True}
    ordinal = dev.index or 0
    apps = ""
    if smi_output is None:
        try:
            smi_output = subprocess.check_output(
                ["nvidia-smi", f"--query-gpu={_SMI_QUERY}", "--format=csv,noheader,nounits"], text=True
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SystemExit(
                f"[preflight] FAIL: cannot run nvidia-smi ({exc}); refusing to allocate on a shared GPU blind. "
                "Pass --skip-preflight to override."
            )
        try:
            apps = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            apps = ""
    gpus = parse_nvidia_smi(smi_output)
    for g in gpus:
        log(
            f"[preflight] GPU {g['index']} {g['name']}: {g['free_mib'] / 1024:.1f} GiB free, "
            f"{g['used_mib'] / 1024:.1f} GiB used, {g['total_mib'] / 1024:.1f} GiB total"
        )
    if apps:
        log("[preflight] compute processes (pid, name, memory):")
        for line in apps.splitlines():
            log(f"[preflight]   {line.strip()}")
    gpu = _visible_gpu(gpus, ordinal)
    if gpu is None:
        raise SystemExit(
            f"[preflight] FAIL: cannot map {device} to an nvidia-smi GPU "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}). Pass --skip-preflight to override."
        )
    free_gb = gpu["free_mib"] / 1024
    total_gb = gpu["total_mib"] / 1024
    if required_free_gb is None:
        required_free_gb = share * total_gb + extra_gb
    report = {
        "device": device,
        "gpu_index": gpu["index"],
        "gpu_name": gpu["name"],
        "free_gb": free_gb,
        "total_gb": total_gb,
        "required_free_gb": float(required_free_gb),
        "skipped": False,
    }
    if free_gb < required_free_gb:
        raise SystemExit(
            f"[preflight] FAIL: {device} (GPU {gpu['index']}) has {free_gb:.1f} GiB free but this run needs "
            f"{required_free_gb:.1f} GiB. The box is shared: nothing was touched. Wait for the other job, lower "
            "the engine's memory share, or pass --require-free-gb / --skip-preflight if you know better."
        )
    log(f"[preflight] ok: {free_gb:.1f} GiB free on {device} >= {required_free_gb:.1f} GiB required")
    return report


def add_preflight_args(parser) -> None:
    group = parser.add_argument_group("shared-GPU preflight")
    group.add_argument(
        "--require-free-gb",
        type=float,
        default=None,
        help="free GPU memory required to start (default: engine share x total + --hf-reserve-gb)",
    )
    group.add_argument(
        "--hf-reserve-gb", type=float, default=6.0, help="memory budgeted for the HF trainer model(s) and scoring"
    )
    group.add_argument("--skip-preflight", action="store_true", help="do not check free GPU memory first")


def run_preflight(args, *, engine_share: float) -> dict:
    """Preflight for a gate that holds one engine (``engine_share`` of the card) plus HF models."""
    if torch.device(args.device).type != "cuda":
        return {"device": args.device, "skipped": True}
    if args.skip_preflight:
        log("[preflight] skipped (--skip-preflight)")
        return {"device": args.device, "skipped": True}
    return gpu_preflight(args.require_free_gb, args.device, share=engine_share, extra_gb=args.hf_reserve_gb)


def copy_non_weight_files(src_dir: str, dst_dir: str) -> list[str]:
    """Copy config / generation config / tokenizer / chat-template files of an HF model directory."""
    copied = []
    for name in sorted(os.listdir(src_dir)):
        path = os.path.join(src_dir, name)
        if not os.path.isfile(path):
            continue
        if not name.endswith((".json", ".txt", ".jinja", ".model")) or name.endswith(".safetensors.index.json"):
            continue
        shutil.copy2(path, os.path.join(dst_dir, name))  # follows the symlinks of an HF snapshot
        copied.append(name)
    return copied
