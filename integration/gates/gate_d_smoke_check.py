"""Gate 3d: post-check of a short training run with the sparsespec rollout backend.

Reads the per-step metrics CSV that ``baseline/parse_timing.py`` writes at the
end of ``baseline/run_seed_sparsespec.sh`` (pure CSV, no GPU needed):

    STEPS=3 EXP_NAME=sparsespec_smoke bash baseline/run_seed_sparsespec.sh
    python -m integration.gates.gate_d_smoke_check --exp-name sparsespec_smoke --steps 3 --engine sparsespec

    # reference engine: reduced-size plumbing run; it calls this checker itself on
    # baseline/results/reference_plumbing.timing.csv (a full-size reference run is hours per step)
    bash integration/smoke_reference.sh

    # step-time comparison against the vLLM baseline's CSV
    python -m integration.gates.gate_d_smoke_check --exp-name sparsespec_smoke --steps 3 \
        --baseline-csv baseline/results/smoke.timing.csv

Checks (exit status 1 if any fails)
  * every step 1..--steps has a row (without --steps: 1..max step found);
  * ``critic/score/mean`` > 0 on at least one step;
  * ``timing_s/gen`` and ``timing_s/step`` are present and finite on every step;
  * the six engine columns are present and finite on every step:
    ``sparsespec/accept_len_mean``, ``sparsespec/num_cycles``,
    ``sparsespec/weight_sync_s``, ``sparsespec/wake_s``, ``sparsespec/gen_s``,
    ``sparsespec/sleep_s``;
  * ``--engine sparsespec``: accept_len_mean > 1 and num_cycles > 0 on every step
    (speculation happened and accepted something);
    ``--engine reference``: accept_len_mean == 0 and num_cycles == 0 (no speculation);
  * timings are sane: the four ``*_s`` columns >= 0, gen_s > 0, and
    ``timing_s/gen`` (the whole rollout call) >= ``sparsespec/gen_s`` (the engine inside it).

It prints the means of ``timing_s/gen``, ``timing_s/step`` and every
``sparsespec/*`` column, and with ``--baseline-csv`` a side-by-side step-time
comparison over the steps both runs have. Outputs:
``<out-dir>/gate_d_smoke_<exp>.json`` / ``.md``.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from integration.gates import common  # noqa: E402

PREFIX = "sparsespec/"
REQUIRED_ENGINE_COLUMNS = (
    "sparsespec/accept_len_mean",
    "sparsespec/num_cycles",
    "sparsespec/weight_sync_s",
    "sparsespec/wake_s",
    "sparsespec/gen_s",
    "sparsespec/sleep_s",
)
REQUIRED_TIMING_COLUMNS = ("timing_s/gen", "timing_s/step")
SCORE_COLUMN = "critic/score/mean"
COMPARE_COLUMNS = ("timing_s/gen", "timing_s/update_actor", "timing_s/step", "response_length/mean", SCORE_COLUMN)


def read_timing_csv(path: str) -> tuple[dict[int, dict[str, float]], list[str]]:
    """``{step: {column: value}}`` (empty cells omitted) and the column names of a parse_timing.py CSV."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "step" not in reader.fieldnames:
            raise SystemExit(f"{path}: not a parse_timing.py CSV (no 'step' column)")
        columns = [c for c in reader.fieldnames if c != "step"]
        steps: dict[int, dict[str, float]] = {}
        for row in reader:
            step = int(float(row["step"]))
            values = {}
            for column in columns:
                cell = (row.get(column) or "").strip()
                if cell == "":
                    continue
                try:
                    values[column] = float(cell)
                except ValueError:
                    values[column] = math.nan
            steps.setdefault(step, {}).update(values)
    return steps, columns


def column_problems(steps: dict, expected_steps: list[int], column: str) -> list[str]:
    """Steps where ``column`` is missing or not finite."""
    problems = []
    for step in expected_steps:
        value = steps.get(step, {}).get(column)
        if value is None:
            problems.append(f"step {step}: missing")
        elif not math.isfinite(value):
            problems.append(f"step {step}: {value}")
    return problems


def summarize(steps: dict, step_list: list[int], column: str) -> dict | None:
    values = [steps[s][column] for s in step_list if s in steps and column in steps[s] and math.isfinite(steps[s][column])]
    if not values:
        return None
    return {"mean": sum(values) / len(values), "min": min(values), "max": max(values), "last": values[-1], "num": len(values)}


def per_step_rule(steps: dict, expected_steps: list[int], column: str, predicate, describe: str) -> tuple[bool, str]:
    """``predicate`` on every reported value of ``column``. Missing / non-finite
    cells are the ``present_finite[...]`` checks' business and are not repeated
    here, but a column with no value at all cannot satisfy a rule."""
    values = {
        s: steps[s][column] for s in expected_steps if s in steps and column in steps[s] and math.isfinite(steps[s][column])
    }
    if not values:
        return False, f"{column} must be {describe}, but no step reports a finite value"
    bad = [f"step {s}: {v}" for s, v in values.items() if not predicate(v)]
    if bad:
        return False, f"{column} must be {describe} on every step; violated at {bad[:6]}"
    return True, (
        f"{column} {describe} on the {len(values)} step(s) that report it "
        f"(min {min(values.values()):.4g}, max {max(values.values()):.4g})"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp-name", default=None, help="reads baseline/results/<exp-name>.timing.csv")
    parser.add_argument("--csv", default=None, help="explicit path to the timing CSV (instead of --exp-name)")
    parser.add_argument("--steps", type=int, default=None, help="number of training steps the run was asked to do")
    parser.add_argument("--engine", choices=["sparsespec", "reference"], default="sparsespec")
    parser.add_argument("--baseline-csv", default=None, help="timing CSV of a vLLM baseline run to compare step times with")
    parser.add_argument("--out-dir", default=common.DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    if bool(args.exp_name) == bool(args.csv):
        raise SystemExit("pass exactly one of --exp-name and --csv")
    path = args.csv or os.path.join(common.REPO_ROOT, "baseline", "results", f"{args.exp_name}.timing.csv")
    path = os.path.abspath(os.path.expanduser(path))
    exp = args.exp_name or os.path.basename(path).replace(".timing.csv", "").replace(".csv", "")
    if not os.path.exists(path):
        common.log(
            f"[gate 3d] FAIL: {path} does not exist (did the run finish? baseline/parse_timing.py writes it from "
            "baseline/logs/<exp>.log)"
        )
        return 1
    steps, columns = read_timing_csv(path)
    common.log(f"[gate 3d] {path}: {len(steps)} step rows, {len(columns)} metric columns, engine={args.engine}")

    checks = common.Checks()
    found = sorted(steps)
    if not checks.add("csv_has_steps", bool(found), f"{len(found)} step rows"):
        common.log("[gate 3d] FAIL: no step rows (verl prints one 'step:N - ...' line per training step)")
        return 1
    if args.steps is None:
        common.log("[gate 3d] NOTE: --steps not given; expecting steps 1..max(step) found in the CSV")
    last = args.steps if args.steps is not None else max(found)
    expected = list(range(1, last + 1))
    missing = [s for s in expected if s not in steps]
    extra = [s for s in found if s not in expected]
    checks.add(
        "all_steps_present",
        not missing and not extra,
        f"expected steps 1..{last}; found {found[:12]}{'...' if len(found) > 12 else ''}; missing {missing[:12]}; "
        f"unexpected {extra[:12]}",
    )

    scores = {s: steps[s][SCORE_COLUMN] for s in found if SCORE_COLUMN in steps[s] and math.isfinite(steps[s][SCORE_COLUMN])}
    best = max(scores.items(), key=lambda kv: kv[1]) if scores else None
    checks.add(
        "reward_nonzero_on_some_step",
        best is not None and best[1] > 0,
        f"max {SCORE_COLUMN} = {best[1]:.4g} at step {best[0]} ({len(scores)} steps report it)"
        if best
        else f"{SCORE_COLUMN} is in no row",
    )

    for column in REQUIRED_TIMING_COLUMNS:
        problems = column_problems(steps, expected, column)
        checks.add(f"present_finite[{column}]", not problems, "ok on every step" if not problems else str(problems[:6]))

    unpatched = [c for c in columns if c.startswith("timing_s/" + PREFIX)]
    for column in REQUIRED_ENGINE_COLUMNS:
        problems = column_problems(steps, expected, column)
        detail = "ok on every step" if not problems else str(problems[:6])
        if problems and ("timing_s/" + column) in columns:
            detail += (
                f"; found 'timing_s/{column}' instead: the ray_trainer.py hunk of "
                "integration/verl-0.5.0-sparsespec.patch is not applied in this env"
            )
        checks.add(f"present_finite[{column}]", not problems, detail)

    if args.engine == "sparsespec":
        rules = [
            ("sparsespec/accept_len_mean", lambda v: v > 1.0, "> 1"),
            ("sparsespec/num_cycles", lambda v: v > 0.0, "> 0"),
        ]
    else:
        rules = [
            ("sparsespec/accept_len_mean", lambda v: v == 0.0, "== 0"),
            ("sparsespec/num_cycles", lambda v: v == 0.0, "== 0"),
        ]
    for column, predicate, describe in rules:
        ok, detail = per_step_rule(steps, expected, column, predicate, describe)
        checks.add(f"{args.engine}_rule[{column}]", ok, detail)

    for column, predicate, describe in (
        ("sparsespec/weight_sync_s", lambda v: v >= 0.0, ">= 0"),
        ("sparsespec/wake_s", lambda v: v >= 0.0, ">= 0"),
        ("sparsespec/sleep_s", lambda v: v >= 0.0, ">= 0"),
        ("sparsespec/gen_s", lambda v: v > 0.0, "> 0"),
    ):
        ok, detail = per_step_rule(steps, expected, column, predicate, describe)
        checks.add(f"timing_sane[{column}]", ok, detail)
    inconsistent = [
        f"step {s}: timing_s/gen={steps[s]['timing_s/gen']:.3f} < sparsespec/gen_s={steps[s]['sparsespec/gen_s']:.3f}"
        for s in expected
        if s in steps
        and "timing_s/gen" in steps[s]
        and "sparsespec/gen_s" in steps[s]
        and steps[s]["timing_s/gen"] < steps[s]["sparsespec/gen_s"]
    ]
    comparable = [s for s in expected if s in steps and "timing_s/gen" in steps[s] and "sparsespec/gen_s" in steps[s]]
    checks.add(
        "rollout_time_contains_engine_time",
        bool(comparable) and not inconsistent,
        f"timing_s/gen >= sparsespec/gen_s on {len(comparable)} steps" if comparable and not inconsistent else str(inconsistent[:6] or "no step has both columns"),
    )  # fmt: skip

    # ---------------- report ----------------
    present_steps = [s for s in expected if s in steps]
    report_columns = list(REQUIRED_TIMING_COLUMNS) + ["timing_s/update_actor", "response_length/mean", SCORE_COLUMN]
    report_columns += sorted(c for c in columns if c.startswith(PREFIX))
    if unpatched:
        report_columns += sorted(unpatched)
    summary = {c: summarize(steps, present_steps, c) for c in report_columns}
    rows = [[c, s["mean"], s["min"], s["max"], s["last"], s["num"]] if s else [c, None, None, None, None, 0] for c, s in summary.items()]
    table = common.md_table(["metric", "mean", "min", "max", "last", "steps"], rows)
    common.log("")
    common.log(f"[gate 3d] per-step metrics over steps {present_steps[:1]}..{present_steps[-1:]}:")
    for line in table.splitlines():
        common.log("  " + line)

    comparison = None
    comparison_table = ""
    base_path = os.path.abspath(os.path.expanduser(args.baseline_csv)) if args.baseline_csv else None
    if base_path and not os.path.isfile(base_path):
        common.log("")
        common.log(
            f"[gate 3d] NOTE: baseline CSV {base_path} does not exist; skipping the step-time comparison "
            "(it is not part of the verdict)"
        )
        base_path = None
    if base_path:
        base_steps, _ = read_timing_csv(base_path)
        shared = [s for s in present_steps if s in base_steps]
        comparison = {"baseline_csv": base_path, "steps": shared, "metrics": {}}
        comp_rows = []
        for column in COMPARE_COLUMNS:
            mine, theirs = summarize(steps, shared, column), summarize(base_steps, shared, column)
            ratio = (mine["mean"] / theirs["mean"]) if mine and theirs and theirs["mean"] else None
            comparison["metrics"][column] = {
                "this_run_mean": mine["mean"] if mine else None,
                "baseline_mean": theirs["mean"] if theirs else None,
                "this_over_baseline": ratio,
            }
            comp_rows.append(
                [column, mine["mean"] if mine else None, theirs["mean"] if theirs else None, ratio, (1.0 / ratio) if ratio else None]
            )
        comparison_table = common.md_table(
            ["metric (mean over shared steps)", f"{exp} ({args.engine})", "vLLM baseline", "this / baseline", "baseline / this"],
            comp_rows,
        )
        common.log("")
        if shared:
            common.log(f"[gate 3d] side by side with {base_path} over the {len(shared)} shared steps {shared[0]}..{shared[-1]}:")
            for line in comparison_table.splitlines():
                common.log("  " + line)
            common.log(
                "[gate 3d] NOTE: step times depend on the response lengths of those steps; compare "
                "response_length/mean before reading the ratio as a speedup."
            )
        else:
            common.log(f"[gate 3d] NOTE: {base_path} shares no step with this run; nothing to compare")

    passed = checks.ok
    output = {
        "gate": "3d_smoke_check",
        "pass": passed,
        "config": {"csv": path, "exp_name": exp, "engine": args.engine, "steps_expected": last, "steps_given": args.steps},
        "steps_found": found,
        "checks": checks.items,
        "summary": summary,
        "baseline_comparison": comparison,
    }
    json_path = common.write_json(os.path.join(args.out_dir, f"gate_d_smoke_{exp}.json"), output)
    markdown = [
        f"# Gate 3d - smoke run post-check: {exp} ({args.engine} engine)",
        "",
        f"**Verdict: {'PASS' if passed else 'FAIL'}**",
        "",
        f"- CSV: `{path}`",
        f"- steps expected: 1..{last}; found: {len(found)}",
        "",
        "## Checks",
        "",
        checks.markdown(),
        "",
        "## Per-step metrics",
        "",
        table,
        "",
    ]
    if comparison is not None:
        markdown += [
            "## Step time vs the vLLM baseline",
            "",
            f"Baseline CSV: `{comparison['baseline_csv']}`; shared steps: {comparison['steps']}",
            "",
            comparison_table,
            "",
            "Step times depend on the response lengths of the compared steps (`response_length/mean`).",
            "",
        ]
    md_path = common.write_text(os.path.join(args.out_dir, f"gate_d_smoke_{exp}.md"), "\n".join(markdown))
    common.log("")
    common.log(f"[gate 3d] wrote {json_path}")
    common.log(f"[gate 3d] wrote {md_path}")
    if passed:
        common.log(f"[gate 3d] PASS ({len(checks.items)} checks)")
        return 0
    for item in checks.failed:
        common.log(f"[gate 3d] FAIL {item['name']}: {item['detail']}")
    common.log(f"[gate 3d] FAIL ({len(checks.failed)} of {len(checks.items)} checks failed)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
