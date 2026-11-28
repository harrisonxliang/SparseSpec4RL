"""Side-by-side comparison of several short training runs (pure CSV / Python, no GPU).

For the write-up and for tuning: reads the per-step timing CSVs that
``baseline/parse_timing.py`` writes (one row per step, one column per metric)
and puts the runs next to each other. From the repo root (PYTHONPATH=repo):

    python -m integration.gates.smoke_compare \
        --run graphs=baseline/results/sparsespec_smoke.timing.csv \
        --run eager=baseline/results/sparsespec_eager_smoke.timing.csv \
        --run nologprob=baseline/results/sparsespec_nologprob_smoke.timing.csv \
        --logprob-pair graphs,nologprob \
        --baseline-csv baseline/results/smoke.timing.csv \
        --host-mem-total-gb 503 --host-mem-available-gb 410 --gpu-total-gb 39.4

Per run: the steps found, and for each metric below the mean over the steps
that report it and the step-1 value ('-' where a run has no such column):
``timing_s/gen``, ``sparsespec/gen_s``, ``timing_s/step``,
``sparsespec/{batch_size_mean, inflight_peak, inflight_cap, kv_util_peak,
host_util_peak, num_offloads, accept_len_mean, accept_rate, wake_s,
graph_capture_s, peak_mem_gb}``, ``response_length/mean``,
``critic/score/mean``. Then

* gen time of every run relative to the FIRST ``--run`` (mean and step 1);
* ``--logprob-pair ON,OFF`` (two ``--run`` labels: calculate_log_probs True /
  False): the cost of the log-prob collection as the difference and ratio of
  ``sparsespec/gen_s`` and ``timing_s/gen`` on the steps both runs have.
  Only step 1 is like for like (same prompts, same base weights);
* ``--baseline-csv``: the vLLM baseline's ``timing_s/gen`` / ``timing_s/step``
  next to the runs, and each run's ratio to it;
* host RAM (with ``--host-mem-total-gb`` and/or ``--host-mem-available-gb``):
  ``serve.rl.memory.plan_memory`` with the Qwen3-0.6B / training constants
  (every one overridable by a flag) for ``cpu_offload_ratio`` 1.5, 3.0 and the
  smallest ratio that lets ``host_guard`` admission reach ``max_batch_size``:
  how much host RAM would be page-locked, how many requests ``worst_case`` and
  ``host_guard`` admission would allow in flight, and whether it fits in the
  available host RAM with ``--host-mem-margin-gb`` to spare;
* a one-line reading of each measured run: is the GPU KV pool or the
  admission cap the limit?

Missing or partial runs are reported, never fatal. Outputs (under --out-dir,
default ``baseline/results/sparsespec_checks/``): ``smoke_compare.json`` /
``.md``, or ``smoke_compare_<tag>.json`` / ``.md`` with ``--tag``; a console
table. The JSON's ``"pass"`` means "the report is usable" (it is not a
verdict on the runs). Exit status: 0 = report written; 2 = unusable inputs
(bad arguments, or not one ``--run`` CSV with a single step; the report is
still written in the latter case).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import re
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from serve.rl.memory import GIB, inflight_cap, plan_memory  # noqa: E402  (pure Python)

DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "baseline", "results", "sparsespec_checks")
PREFIX = "[smoke compare]"

METRICS = (
    "timing_s/gen",
    "sparsespec/gen_s",
    "timing_s/step",
    "sparsespec/batch_size_mean",
    "sparsespec/inflight_peak",
    "sparsespec/inflight_cap",
    "sparsespec/kv_util_peak",
    "sparsespec/host_util_peak",
    "sparsespec/num_offloads",
    "sparsespec/accept_len_mean",
    "sparsespec/accept_rate",
    "sparsespec/wake_s",
    "sparsespec/graph_capture_s",
    "sparsespec/peak_mem_gb",
    "response_length/mean",
    "critic/score/mean",
)
GEN_METRICS = ("timing_s/gen", "sparsespec/gen_s")
LOGPROB_METRICS = ("sparsespec/gen_s", "timing_s/gen")
BASELINE_METRICS = ("timing_s/gen", "timing_s/step")

# Qwen/Qwen3-0.6B (config.json) and the training config (baseline/config/grpo_base.sh,
# baseline/run_seed_sparsespec.sh); serve/rl/tests/test_memory.py checks the same numbers
QWEN3_06B = dict(
    weights_bytes=1_192_099_840,  # unique parameter bytes, bf16, tied lm_head counted once
    num_layers=28,
    num_kv_heads=8,
    head_dim=128,
    vocab_size=151_936,
    hidden_size=1024,
    dtype_bytes=2,
)
A100_40GB_TOTAL_BYTES = int(42.5e9)  # torch.cuda.get_device_properties(...).total_memory, about 39.6 GiB
KV_NEAR_FULL = 0.95  # kv_util_peak at or above this: the GPU pool is the limit
KV_WELL_BELOW = 0.80  # kv_util_peak below this: the GPU pool is not the limit

LIKE_FOR_LIKE_NOTE = (
    "Step 1 uses the same prompts and the same (base) weights in every run (same seed, same data order), so "
    "only step 1 is like for like; it also includes one-time costs (kernel compilation when the cache is cold, "
    "the first CUDA-graph capture). Later steps are NOT comparable one to one: each run generates from weights "
    "updated with its own sampled responses, so prompts' response lengths and the work per step differ."
)


def log(msg: str = "") -> None:
    print(msg, flush=True)


# ----------------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------------
def read_run(label: str, path: str) -> dict:
    """``{label, path, exists, error, steps: {step: {column: value}}, columns, notes}``; never raises."""
    path = os.path.abspath(os.path.expanduser(path))
    run = {"label": label, "path": path, "exists": os.path.isfile(path), "error": None, "steps": {}, "columns": [], "notes": []}
    if not run["exists"]:
        run["error"] = "file not found"
        return run
    try:
        with open(path, newline="", errors="replace") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            if not fields:
                run["error"] = "empty file (no header)"
                return run
            if "step" not in fields:
                run["error"] = f"not a parse_timing.py CSV: no 'step' column (columns: {fields[:6]})"
                return run
            run["columns"] = [c for c in fields if c and c != "step"]
            bad_steps = nonnumeric = nonfinite = 0
            for row in reader:
                try:
                    step = int(float((row.get("step") or "").strip()))
                except ValueError:
                    bad_steps += 1
                    continue
                values = run["steps"].setdefault(step, {})
                for column in run["columns"]:
                    cell = (row.get(column) or "").strip()
                    if not cell:
                        continue
                    try:
                        value = float(cell)
                    except ValueError:
                        nonnumeric += 1
                        continue
                    if not math.isfinite(value):
                        nonfinite += 1
                        continue
                    values[column] = value  # a later row for the same step wins (resumed runs)
    except (OSError, csv.Error, UnicodeError) as exc:
        run["error"] = f"cannot read: {type(exc).__name__}: {exc}"
        return run
    if bad_steps:
        run["notes"].append(f"{bad_steps} row(s) without a numeric step skipped")
    if nonnumeric:
        run["notes"].append(f"{nonnumeric} non-numeric cell(s) ignored")
    if nonfinite:
        run["notes"].append(f"{nonfinite} non-finite cell(s) (nan / inf) ignored")
    steps = sorted(run["steps"])
    if not steps:
        run["error"] = "no step rows"
    elif steps != list(range(1, steps[-1] + 1)):
        missing = sorted(set(range(1, steps[-1] + 1)) - set(steps))
        run["notes"].append(f"partial: steps {missing[:10]} missing")
    return run


def metric_summary(run: dict, column: str) -> dict:
    values = [(s, run["steps"][s][column]) for s in sorted(run["steps"]) if column in run["steps"][s]]
    if not values:
        return {"mean": None, "step1": None, "num_steps": 0, "min": None, "max": None}
    nums = [v for _, v in values]
    return {
        "mean": sum(nums) / len(nums),
        "step1": run["steps"].get(1, {}).get(column),
        "num_steps": len(nums),
        "min": min(nums),
        "max": max(nums),
    }


def summarize(run: dict, metrics=METRICS) -> dict:
    return {column: metric_summary(run, column) for column in metrics}


def ratio(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b


# ----------------------------------------------------------------------------
# comparisons
# ----------------------------------------------------------------------------
def gen_ratios(runs: list[dict]) -> dict:
    """Each run's gen time / the first run's (mean and step 1)."""
    first = runs[0]
    out = {"reference_run": first["label"], "rows": []}
    for run in runs:
        row = {"label": run["label"]}
        for column in GEN_METRICS:
            a, b = run["summary"][column], first["summary"][column]
            row[column] = {"mean_ratio": ratio(a["mean"], b["mean"]), "step1_ratio": ratio(a["step1"], b["step1"])}
        out["rows"].append(row)
    return out


def logprob_cost(on: dict, off: dict) -> dict:
    """calculate_log_probs cost: ON - OFF and ON / OFF on the steps both runs report."""
    out = {"on": on["label"], "off": off["label"], "note": LIKE_FOR_LIKE_NOTE, "metrics": {}}
    for column in LOGPROB_METRICS:
        steps = sorted(s for s in on["steps"] if column in on["steps"][s] and column in off["steps"].get(s, {}))
        per_step = []
        for s in steps:
            a, b = on["steps"][s][column], off["steps"][s][column]
            per_step.append({"step": s, "on": a, "off": b, "diff": a - b, "ratio": ratio(a, b)})
        step1 = next((p for p in per_step if p["step"] == 1), None)
        mean_on = sum(p["on"] for p in per_step) / len(per_step) if per_step else None
        mean_off = sum(p["off"] for p in per_step) / len(per_step) if per_step else None
        out["metrics"][column] = {
            "common_steps": steps,
            "per_step": per_step,
            "step1_diff": step1["diff"] if step1 else None,
            "step1_ratio": step1["ratio"] if step1 else None,
            "mean_on": mean_on,
            "mean_off": mean_off,
            "mean_diff": (mean_on - mean_off) if per_step else None,
            "mean_ratio": ratio(mean_on, mean_off),
        }
    return out


def baseline_comparison(baseline: dict, runs: list[dict]) -> dict:
    summary = summarize(baseline, BASELINE_METRICS)
    rows = []
    for run in runs:
        row = {"label": run["label"]}
        for column in BASELINE_METRICS:
            a, b = run["summary"][column], summary[column]
            row[column] = {"mean_ratio": ratio(a["mean"], b["mean"]), "step1_ratio": ratio(a["step1"], b["step1"])}
        rows.append(row)
    return {"path": baseline["path"], "exists": baseline["exists"], "error": baseline["error"],
            "steps": sorted(baseline["steps"]), "notes": baseline["notes"], "summary": summary, "ratios": rows}  # fmt: skip


def measured_reading(run: dict, max_batch_size: int) -> str:
    """One line: what limits concurrency in this run (from per-step peaks, max over steps)."""
    s = run["summary"]
    kv = s["sparsespec/kv_util_peak"]["max"]
    peak = s["sparsespec/inflight_peak"]["max"]
    cap = s["sparsespec/inflight_cap"]["max"]
    batch = s["sparsespec/batch_size_mean"]["mean"]
    if not run["steps"]:
        return "no data"
    if kv is None:
        return "no sparsespec/kv_util_peak column: cannot tell (reference engine, or a run from before these metrics)"
    extra = f" (batch_size_mean {fmt(batch)})" if batch is not None else ""
    if kv >= KV_NEAR_FULL:
        return (
            f"kv_util_peak {fmt(kv)} near 1: the GPU pool is the limit; a larger host pool (cpu_offload_ratio) "
            f"will not raise concurrency{extra}"
        )
    if peak is None or cap is None:
        return f"kv_util_peak {fmt(kv)} below 1; no inflight_peak / inflight_cap columns to say more{extra}"
    if peak >= cap:
        if cap >= max_batch_size:
            return (
                f"inflight_peak {fmt(peak)} at the cap {fmt(cap)} = max_batch_size {max_batch_size}: max_batch_size is "
                f"the limit, not memory (kv_util_peak {fmt(kv)}){extra}"
            )
        if kv < KV_WELL_BELOW:
            return (
                f"inflight_peak {fmt(peak)} at the cap {fmt(cap)} while kv_util_peak {fmt(kv)} is well below 1: the "
                f"admission cap is the limit, a larger host pool would help (see the host RAM table){extra}"
            )
        return (
            f"inflight_peak {fmt(peak)} at the cap {fmt(cap)} and kv_util_peak {fmt(kv)} close to 1: both limits are "
            f"near; a larger host pool would help little{extra}"
        )
    return (
        f"inflight_peak {fmt(peak)} below the cap {fmt(cap)} and kv_util_peak {fmt(kv)} below 1: neither memory "
        f"limit binds (not enough requests at once to fill either){extra}"
    )


# ----------------------------------------------------------------------------
# host RAM: what a larger host offload pool would lock and buy
# ----------------------------------------------------------------------------
def _plan(args, ratio_value: float):
    return plan_memory(
        total_bytes=int(args.gpu_total_gb * GIB) if args.gpu_total_gb else A100_40GB_TOTAL_BYTES,
        mem_fraction=args.mem_fraction,
        weights_bytes=args.weights_bytes,
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        dtype_bytes=args.dtype_bytes,
        max_batch_size=args.max_batch_size,
        cuda_graph=not args.no_cuda_graph,
        cpu_offload_ratio=ratio_value,
        transient_reserve_bytes=None if args.transient_reserve_gb is None else int(args.transient_reserve_gb * GIB),
        kv_capacity_tokens=args.kv_capacity_tokens,
        spec_stride=args.spec_stride,
        chunk_size=args.chunk_size,
    )


def _cap(args, plan, admission: str) -> int:
    return inflight_cap(
        admission, plan, args.max_batch_size, args.spec_stride, args.chunk_size, args.max_prompt_len, args.max_new_tokens
    )


def _caps(args, plan) -> dict:
    return {"worst_case": _cap(args, plan, "worst_case"), "host_guard": _cap(args, plan, "host_guard")}


def smallest_ratio(args, admission: str, limit_tenths: int = 200) -> float | None:
    """Smallest cpu_offload_ratio (0.1 steps, the engine warning's rule) up to 20 whose plan lets
    ``admission`` reach max_batch_size; None if there is none."""
    for tenths in range(0, limit_tenths + 1):
        candidate = tenths / 10
        if _cap(args, _plan(args, candidate), admission) >= args.max_batch_size:
            return candidate
    return None


def host_memory_table(args) -> dict:
    out = {
        "host_mem_total_gb": args.host_mem_total_gb,
        "host_mem_available_gb": args.host_mem_available_gb,
        "margin_gb": args.host_mem_margin_gb,
        "gpu_total_gb": args.gpu_total_gb if args.gpu_total_gb else A100_40GB_TOTAL_BYTES / GIB,
        "gpu_total_assumed": not args.gpu_total_gb,
        "workload": {"max_prompt_len": args.max_prompt_len, "max_new_tokens": args.max_new_tokens,
                     "max_batch_size": args.max_batch_size, "mem_fraction": args.mem_fraction},  # fmt: skip
        "rows": [],
        "notes": [],
    }
    try:
        hg_ratio = smallest_ratio(args, "host_guard")
        wc_ratio = smallest_ratio(args, "worst_case")
        ratios = [("default", 1.5), ("3.0", 3.0)]
        if hg_ratio is not None:
            ratios.append(("smallest for host_guard = max_batch_size", hg_ratio))
        else:
            out["notes"].append("no cpu_offload_ratio up to 20 lets host_guard reach max_batch_size")
        seen = set()
        for name, value in ratios:
            if value in seen:
                continue
            seen.add(value)
            plan = _plan(args, value)
            caps = _caps(args, plan)
            pinned_gb = plan.cpu_pinned_bytes / GIB
            needed_gb = pinned_gb + args.host_mem_margin_gb
            out["rows"].append({
                "name": name,
                "cpu_offload_ratio": value,
                "kv_capacity_tokens": plan.kv_capacity_tokens,
                "host_pool_tokens": plan.cpu_capacity_tokens,
                "pinned_gb": pinned_gb,
                "inflight_worst_case": caps["worst_case"],
                "inflight_host_guard": caps["host_guard"],
                "fits_available": None if args.host_mem_available_gb is None else needed_gb <= args.host_mem_available_gb,
                "fits_total": None if args.host_mem_total_gb is None else needed_gb <= args.host_mem_total_gb,
            })  # fmt: skip
        out["smallest_ratio_host_guard"] = hg_ratio
        out["smallest_ratio_worst_case"] = wc_ratio
        if wc_ratio is not None:
            plan = _plan(args, wc_ratio)
            out["notes"].append(
                f"worst_case admission reaches max_batch_size={args.max_batch_size} only from cpu_offload_ratio "
                f"{wc_ratio:g} (locks {plan.cpu_pinned_bytes / GIB:.0f} GiB)"
            )
        out["kv_capacity_tokens"] = _plan(args, 1.5).kv_capacity_tokens
        out["notes"].append(
            "pinned = page-locked host RAM held for the whole run (torch rounds the pool to a power-of-two block); "
            f"'fits' = pinned + {args.host_mem_margin_gb:g} GiB margin (--host-mem-margin-gb: the training process, "
            "Ray and the data loader need host RAM too) <= the host RAM figure given"
        )
    except (ValueError, MemoryError) as exc:  # e.g. a share too small for these constants
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


# ----------------------------------------------------------------------------
# formatting
# ----------------------------------------------------------------------------
def fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "NO"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        if value != 0 and abs(value) >= 1000:
            return f"{value:.0f}"
        return f"{value:.4g}"
    return str(value)


def text_tables(title: str, row_names: list[str], columns: list[tuple[str, list]], width: int) -> list[str]:
    """Plain-text table(s), split by columns so each line fits ``width``."""
    name_w = max([len(title)] + [len(n) for n in row_names])
    lines: list[str] = []
    chunk: list[tuple[str, list, int]] = []
    used = name_w

    def flush():
        if not chunk:
            return
        lines.append("  ".join([title.ljust(name_w)] + [h.rjust(w) for h, _, w in chunk]))
        lines.append("  ".join(["-" * name_w] + ["-" * w for _, _, w in chunk]))
        for i, name in enumerate(row_names):
            lines.append("  ".join([name.ljust(name_w)] + [fmt(vals[i]).rjust(w) for _, vals, w in chunk]))
        lines.append("")

    for header, values in columns:
        w = max([len(header)] + [len(fmt(v)) for v in values])
        if chunk and used + 2 + w > width:
            flush()
            chunk, used = [], name_w
        chunk.append((header, values, w))
        used += 2 + w
    flush()
    return lines


def md_table(headers: list[str], rows: list[list]) -> str:
    def cell(v):
        return fmt(v).replace("|", "\\|")

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _short(label: str, limit: int = 14) -> str:
    return label if len(label) <= limit else label[: limit - 1] + "~"


def build_markdown(report: dict) -> str:
    runs, labels = report["runs"], [r["label"] for r in report["runs"]]
    lines = [f"# Smoke runs side by side{(' - ' + report['tag']) if report.get('tag') else ''}", ""]
    lines += [md_table(["run", "CSV", "steps found", "status"],
                       [[r["label"], f"`{r['path']}`", ", ".join(map(str, r["steps"])) or "-",
                         r["error"] or ("; ".join(r["notes"]) or "ok")] for r in runs]), ""]  # fmt: skip
    base = report.get("baseline")
    for key, title in (("mean", "Mean over the steps that report the metric"), ("step1", "Step 1")):
        headers = ["metric"] + labels + (["vLLM baseline"] if base else [])
        rows = []
        for column in METRICS:
            row = [f"`{column}`"] + [r["summary"][column][key] for r in runs]
            if base:
                row.append(base["summary"][column][key] if column in base["summary"] else None)
            rows.append(row)
        lines += [f"## {title}", "", md_table(headers, rows), ""]
    ratios = report["gen_ratios"]
    lines += [f"## Gen time relative to `{ratios['reference_run']}` (the first run)", ""]
    lines += [md_table(["run"] + [f"{c} {k}" for c in GEN_METRICS for k in ("mean", "step 1")],
                       [[row["label"]] + [row[c][k] for c in GEN_METRICS for k in ("mean_ratio", "step1_ratio")]
                        for row in ratios["rows"]]), ""]  # fmt: skip
    cost = report.get("logprob_cost")
    if cost:
        lines += [f"## Cost of calculate_log_probs: `{cost['on']}` (on) vs `{cost['off']}` (off)", "", cost["note"], ""]
        rows = []
        for column, m in cost["metrics"].items():
            rows.append([f"`{column}`", ", ".join(map(str, m["common_steps"])) or "-", m["step1_diff"], m["step1_ratio"],
                         m["mean_diff"], m["mean_ratio"]])  # fmt: skip
        lines += [md_table(["metric", "common steps", "step 1 on - off (s)", "step 1 on / off",
                            "mean on - off (s)", "mean on / off"], rows), ""]  # fmt: skip
    if base:
        lines += ["## vLLM baseline", "", f"`{base['path']}`: " + (base["error"] or f"steps {base['steps']}"), ""]
        lines += [md_table(["run"] + [f"{c} / baseline, {k}" for c in BASELINE_METRICS for k in ("mean", "step 1")],
                           [[row["label"]] + [row[c][k] for c in BASELINE_METRICS for k in ("mean_ratio", "step1_ratio")]
                            for row in base["ratios"]]), ""]  # fmt: skip
    lines += ["## What limits concurrency (measured; per-step peaks, max over steps)", ""]
    lines += [f"- `{r['label']}`: {r['reading']}" for r in runs] + [""]
    host = report.get("host_memory")
    if host:
        lines += ["## Host RAM: a larger host offload pool", ""]
        lines += [
            f"Host RAM total {fmt(host['host_mem_total_gb'])} GiB, available {fmt(host['host_mem_available_gb'])} GiB, "
            f"GPU {fmt(host['gpu_total_gb'])} GiB{' (assumed: A100 40GB)' if host['gpu_total_assumed'] else ''}; "
            f"prompts up to {host['workload']['max_prompt_len']}, responses up to {host['workload']['max_new_tokens']}, "
            f"max_batch_size {host['workload']['max_batch_size']}, share {host['workload']['mem_fraction']}.",
            "",
        ]
        if host.get("error"):
            lines += [f"Could not plan: {host['error']}", ""]
        else:
            lines += [md_table(["cpu_offload_ratio", "", "host pool tokens", "pinned GiB", "in flight worst_case",
                                "in flight host_guard", "fits available", "fits total"],
                               [[r["cpu_offload_ratio"], r["name"], r["host_pool_tokens"], r["pinned_gb"],
                                 r["inflight_worst_case"], r["inflight_host_guard"], r["fits_available"], r["fits_total"]]
                                for r in host["rows"]]), ""]  # fmt: skip
        lines += [f"- {n}" for n in host["notes"]] + [""]
    lines += [f"- {n}" for n in report["notes"]]
    return "\n".join(lines)


def console_lines(report: dict, width: int) -> list[str]:
    runs = report["runs"]
    lines = [f"{PREFIX} runs:"]
    for r in runs:
        status = r["error"] or ("; ".join(r["notes"]) or "ok")
        lines.append(f"{PREFIX}   {r['label']}: {len(r['steps'])} step(s) {r['steps'][:10]}  {status}  ({r['path']})")
    lines.append("")
    base = report.get("baseline")
    for key, title in (("mean", "mean over steps"), ("step1", "step 1")):
        columns = [(_short(r["label"]), [r["summary"][c][key] for c in METRICS]) for r in runs]
        if base:
            columns.append(("vLLM base", [base["summary"].get(c, {}).get(key) for c in METRICS]))
        lines += text_tables(title, list(METRICS), columns, width)
    ratios = report["gen_ratios"]
    names = [f"{c} {k}" for c in GEN_METRICS for k in ("mean", "step1")]
    columns = [(_short(row["label"]), [row[c][k] for c in GEN_METRICS for k in ("mean_ratio", "step1_ratio")])
               for row in ratios["rows"]]  # fmt: skip
    lines += text_tables(f"/ {_short(ratios['reference_run'])}", names, columns, width)
    cost = report.get("logprob_cost")
    if cost:
        lines.append(f"{PREFIX} calculate_log_probs cost, {cost['on']} (on) vs {cost['off']} (off):")
        for column, m in cost["metrics"].items():
            lines.append(
                f"{PREFIX}   {column}: step 1 on-off {fmt(m['step1_diff'])} s, on/off {fmt(m['step1_ratio'])}; "
                f"mean over steps {m['common_steps']}: on-off {fmt(m['mean_diff'])} s, on/off {fmt(m['mean_ratio'])}"
            )
        lines.append(f"{PREFIX}   NOTE: {cost['note']}")
        lines.append("")
    if base:
        lines.append(f"{PREFIX} vLLM baseline {base['path']}: {base['error'] or 'steps ' + str(base['steps'])}")
        for row in base["ratios"]:
            parts = [f"{c} / baseline: mean {fmt(row[c]['mean_ratio'])}, step 1 {fmt(row[c]['step1_ratio'])}" for c in BASELINE_METRICS]
            lines.append(f"{PREFIX}   {row['label']}: " + "; ".join(parts))
        lines.append("")
    lines.append(f"{PREFIX} what limits concurrency (measured; per-step peaks, max over steps):")
    lines += [f"{PREFIX}   {r['label']}: {r['reading']}" for r in runs]
    host = report.get("host_memory")
    if host:
        lines.append("")
        lines.append(
            f"{PREFIX} host RAM (total {fmt(host['host_mem_total_gb'])} GiB, available {fmt(host['host_mem_available_gb'])} GiB, "
            f"margin {fmt(host['margin_gb'])} GiB, GPU {fmt(host['gpu_total_gb'])} GiB"
            f"{' assumed' if host['gpu_total_assumed'] else ''}):"
        )
        if host.get("error"):
            lines.append(f"{PREFIX}   could not plan: {host['error']}")
        else:
            names = ["host pool tokens", "pinned GiB", "in flight worst_case", "in flight host_guard", "fits available", "fits total"]
            columns = [(f"ratio {r['cpu_offload_ratio']:.1f}", [r["host_pool_tokens"], r["pinned_gb"], r["inflight_worst_case"],
                                                              r["inflight_host_guard"], r["fits_available"], r["fits_total"]])
                       for r in host["rows"]]  # fmt: skip
            lines += text_tables("cpu_offload_ratio", names, columns, width)
        lines += [f"{PREFIX}   note: {n}" for n in host["notes"]]
    lines += [f"{PREFIX} note: {n}" for n in report["notes"]]
    return lines


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", default=[], metavar="LABEL=CSV",
                        help="a run's timing CSV (parse_timing.py output); repeat; the first run is the ratio reference")  # fmt: skip
    parser.add_argument("--logprob-pair", default=None, metavar="ON_LABEL,OFF_LABEL",
                        help="two --run labels: calculate_log_probs True and False")  # fmt: skip
    parser.add_argument("--baseline-csv", default=None, help="timing CSV of a vLLM baseline run")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", default=None, help="outputs smoke_compare_<tag>.{json,md} (default smoke_compare.{json,md})")
    parser.add_argument("--width", type=int, default=None, help="console table width (default: terminal width, at least 80)")
    host = parser.add_argument_group("host RAM (GiB, e.g. from `free -g`) and the memory-plan constants")
    host.add_argument("--host-mem-total-gb", type=float, default=None)
    host.add_argument("--host-mem-available-gb", type=float, default=None)
    host.add_argument("--host-mem-margin-gb", type=float, default=16.0,
                      help="host RAM to leave for everything else besides the pinned pool (default %(default)s)")  # fmt: skip
    host.add_argument("--gpu-total-gb", type=float, default=None,
                      help="GPU memory in GiB as torch reports it (default: A100 40GB, 42.5e9 bytes = 39.6 GiB)")  # fmt: skip
    host.add_argument("--mem-fraction", type=float, default=0.45, help="rollout.gpu_memory_utilization (default %(default)s)")
    host.add_argument("--max-batch-size", type=int, default=64)
    host.add_argument("--spec-stride", type=int, default=8)
    host.add_argument("--chunk-size", type=int, default=256)
    host.add_argument("--max-prompt-len", type=int, default=1024, help="data.max_prompt_length (default %(default)s)")
    host.add_argument("--max-new-tokens", type=int, default=8192, help="data.max_response_length (default %(default)s)")
    host.add_argument("--kv-capacity-tokens", type=int, default=None,
                      help="GPU KV pool size (default: planned; pass the engine log's value to use the measured one)")  # fmt: skip
    host.add_argument("--transient-reserve-gb", type=float, default=None, help="default: planned (auto)")
    host.add_argument("--no-cuda-graph", action="store_true", help="plan without CUDA graphs")
    for key, value in QWEN3_06B.items():
        host.add_argument("--" + key.replace("_", "-"), type=int, default=value, help=f"Qwen3-0.6B: {value}")
    return parser


def parse_runs(parser, specs: list[str]) -> list[tuple[str, str]]:
    if not specs:
        parser.error("at least one --run LABEL=CSV is required")
    out, seen = [], set()
    for spec in specs:
        label, sep, path = spec.partition("=")
        label, path = label.strip(), path.strip()
        if not sep or not label or not path:
            parser.error(f"--run {spec!r}: expected LABEL=CSV")
        if "," in label:
            parser.error(f"--run {spec!r}: a label must not contain ','")
        if label in seen:
            parser.error(f"--run {spec!r}: label {label!r} given twice")
        seen.add(label)
        out.append((label, path))
    return out


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    specs = parse_runs(parser, args.run)
    labels = [label for label, _ in specs]
    pair = None
    if args.logprob_pair:
        pair = [p.strip() for p in args.logprob_pair.split(",")]
        if len(pair) != 2 or pair[0] == pair[1] or any(p not in labels for p in pair):
            parser.error(f"--logprob-pair {args.logprob_pair!r}: expected ON,OFF, two different --run labels of {labels}")
    if args.tag is not None and not re.fullmatch(r"[A-Za-z0-9._-]+", args.tag):
        parser.error(f"--tag {args.tag!r}: use letters, digits, '.', '_' and '-' only")
    width = args.width or max(80, shutil.get_terminal_size((100, 24)).columns)

    runs = [read_run(label, path) for label, path in specs]
    for run in runs:
        run["summary"] = summarize(run)
        run["reading"] = measured_reading(run, args.max_batch_size)
    usable = any(run["steps"] for run in runs)
    notes = []
    for run in runs:
        absent = [c for c in METRICS if run["steps"] and run["summary"][c]["num_steps"] == 0]
        if absent:
            notes.append(f"{run['label']}: no column {', '.join(absent)}")
    if not usable:
        notes.insert(0, "UNUSABLE: none of the --run CSVs has a single step row")
    report = {
        "kind": "smoke_compare",
        "pass": usable,
        "usable": usable,
        "tag": args.tag,
        "time_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runs": [{k: (sorted(v) if k == "steps" else v) for k, v in run.items()} for run in runs],
        "gen_ratios": gen_ratios(runs),
        "logprob_cost": logprob_cost(runs[labels.index(pair[0])], runs[labels.index(pair[1])]) if pair else None,
        "baseline": baseline_comparison(read_run("vllm_baseline", args.baseline_csv), runs) if args.baseline_csv else None,
        "host_memory": host_memory_table(args)
        if args.host_mem_total_gb is not None or args.host_mem_available_gb is not None
        else None,
        "notes": notes,
    }
    # per-step values in the JSON (the summaries alone hide partial runs)
    for out_run, run in zip(report["runs"], runs):
        out_run["values"] = {str(s): run["steps"][s] for s in sorted(run["steps"])}

    name = "smoke_compare" + (f"_{args.tag}" if args.tag else "")
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, name + ".json")
    md_path = os.path.join(args.out_dir, name + ".md")
    with open(json_path, "w") as f:
        json.dump(_jsonable(report), f, indent=2, allow_nan=False)
        f.write("\n")
    with open(md_path, "w") as f:
        f.write(build_markdown(report) + "\n")

    for line in console_lines(report, width):
        log(line)
    log(f"{PREFIX} wrote {json_path}")
    log(f"{PREFIX} wrote {md_path}")
    if not usable:
        log(f"{PREFIX} UNUSABLE: no --run CSV has any step (see the paths above)")
        return 2
    log(f"{PREFIX} OK: {sum(bool(r['steps']) for r in runs)} of {len(runs)} run(s) with data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
