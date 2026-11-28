"""CPU tests of ``integration/gates/smoke_compare.py`` (pure CSV / Python).

    PYTHONPATH=. python -m pytest integration/gates/tests/test_smoke_compare.py -q

The timing CSVs are produced the way the run script produces them: a console
log in verl's ``step:N - key:value - ...`` format goes through the real
``baseline/parse_timing.py``. No ``__init__.py`` next to this file (see
test_gates_cpu.py).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from integration.gates import smoke_compare  # noqa: E402
from serve.rl import memory  # noqa: E402

PREFIX = smoke_compare.PREFIX


def _log_line(step: int, values: dict) -> str:
    return f"(TaskRunner pid=1) step:{step} - " + " - ".join(f"{k}:{v}" for k, v in values.items())


def _csv(tmp_path: Path, name: str, rows: dict[int, dict]) -> str:
    log = tmp_path / f"{name}.log"
    lines = ["Training Progress:   0%|          | 0/3 [00:00<?, ?it/s]"]
    lines += [_log_line(step, {"training/global_step": step, **values}) for step, values in sorted(rows.items())]
    log.write_text("\n".join(lines) + "\n")
    out = tmp_path / f"{name}.timing.csv"
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "baseline" / "parse_timing.py"), str(log), "-o", str(out)],
        check=True, capture_output=True, text=True,
    )  # fmt: skip
    assert out.exists()
    return str(out)


def _sparsespec_step(gen: float, *, kv=0.42, peak=27, cap=27, batch=24.5, score=0.2, **extra) -> dict:
    values = {
        "timing_s/gen": gen, "timing_s/generate_sequences": gen - 1, "timing_s/step": gen + 80,
        "sparsespec/gen_s": gen - 10, "sparsespec/batch_size_mean": batch, "sparsespec/inflight_peak": peak,
        "sparsespec/inflight_cap": cap, "sparsespec/kv_util_peak": kv, "sparsespec/host_util_peak": 0.1,
        "sparsespec/num_offloads": 3, "sparsespec/accept_len_mean": 3.2, "sparsespec/accept_rate": 0.4,
        "sparsespec/wake_s": 2.5, "sparsespec/graph_capture_s": 2.0, "sparsespec/peak_mem_gb": 15.5,
        "response_length/mean": 900.0, "critic/score/mean": score,
    }  # fmt: skip
    values.update(extra)
    return values


def _run(argv, capsys) -> tuple[int, dict, str, str]:
    status = smoke_compare.main(argv)
    out = capsys.readouterr().out
    out_dir = Path(argv[argv.index("--out-dir") + 1])
    tag = argv[argv.index("--tag") + 1] if "--tag" in argv else None
    name = "smoke_compare" + (f"_{tag}" if tag else "")
    with open(out_dir / f"{name}.json") as f:
        report = json.load(f)
    return status, report, (out_dir / f"{name}.md").read_text(), out


def _by_label(report: dict) -> dict:
    return {r["label"]: r for r in report["runs"]}


def test_full_comparison(tmp_path, capsys):
    graphs = _csv(tmp_path, "graphs", {1: _sparsespec_step(100.0), 2: _sparsespec_step(90.0), 3: _sparsespec_step(80.0)})
    eager = _csv(tmp_path, "eager", {s: _sparsespec_step(g, kv=0.97, peak=20, cap=27) for s, g in ((1, 150.0), (2, 140.0), (3, 130.0))})
    # partial: no step 3, no critic/score/mean, no inflight columns
    nolp_rows = {}
    for step, gen in ((1, 95.0), (2, 85.0)):
        row = _sparsespec_step(gen)
        for key in ("critic/score/mean", "sparsespec/inflight_peak", "sparsespec/inflight_cap"):
            row.pop(key)
        nolp_rows[step] = row
    nolp = _csv(tmp_path, "nolp", nolp_rows)
    baseline = _csv(tmp_path, "vllm", {1: {"timing_s/gen": 200.0, "timing_s/step": 300.0}, 2: {"timing_s/gen": 190.0, "timing_s/step": 290.0}})
    out_dir = tmp_path / "out"
    argv = ["--run", f"graphs={graphs}", "--run", f"eager={eager}", "--run", f"nolp={nolp}",
            "--run", f"gone={tmp_path / 'missing.timing.csv'}", "--logprob-pair", "graphs,nolp",
            "--baseline-csv", baseline, "--host-mem-total-gb", "64", "--host-mem-available-gb", "40",
            "--gpu-total-gb", str(int(42.5e9) / 2**30), "--out-dir", str(out_dir), "--width", "80"]  # fmt: skip
    status, report, markdown, out = _run(argv, capsys)

    assert status == 0 and report["pass"] is True and report["usable"] is True
    runs = _by_label(report)
    assert runs["graphs"]["steps"] == [1, 2, 3] and runs["nolp"]["steps"] == [1, 2] and runs["gone"]["steps"] == []
    assert runs["gone"]["error"] == "file not found" and runs["gone"]["reading"] == "no data"
    assert any("partial" in n for n in runs["nolp"]["notes"]) is False  # steps 1..2 are contiguous: not partial
    g = runs["graphs"]["summary"]
    assert g["timing_s/gen"] == {"mean": 90.0, "step1": 100.0, "num_steps": 3, "min": 80.0, "max": 100.0}
    assert g["sparsespec/gen_s"]["mean"] == pytest.approx(80.0) and g["sparsespec/peak_mem_gb"]["step1"] == 15.5
    for column in smoke_compare.METRICS:  # every metric reported for a complete run
        assert g[column]["num_steps"] == 3, column
    assert runs["nolp"]["summary"]["critic/score/mean"]["mean"] is None  # '-', not a crash
    assert all(v["mean"] is None for v in runs["gone"]["summary"].values())
    assert runs["graphs"]["values"]["2"]["timing_s/gen"] == 90.0

    # gen time relative to the first run
    rows = {r["label"]: r for r in report["gen_ratios"]["rows"]}
    assert report["gen_ratios"]["reference_run"] == "graphs"
    assert rows["eager"]["timing_s/gen"]["mean_ratio"] == pytest.approx(140.0 / 90.0)
    assert rows["eager"]["timing_s/gen"]["step1_ratio"] == pytest.approx(1.5)
    assert rows["gone"]["timing_s/gen"]["mean_ratio"] is None

    # calculate_log_probs cost on the common steps (1, 2)
    cost = report["logprob_cost"]
    assert cost["on"] == "graphs" and cost["off"] == "nolp" and "only step 1 is like for like" in cost["note"]
    gen = cost["metrics"]["sparsespec/gen_s"]
    assert gen["common_steps"] == [1, 2] and gen["step1_diff"] == pytest.approx(5.0)
    assert gen["step1_ratio"] == pytest.approx(90.0 / 85.0)
    assert gen["mean_diff"] == pytest.approx(5.0) and gen["mean_ratio"] == pytest.approx(85.0 / 80.0)
    assert cost["metrics"]["timing_s/gen"]["step1_diff"] == pytest.approx(5.0)

    # vLLM baseline next to the runs
    base = report["baseline"]
    assert base["summary"]["timing_s/gen"]["mean"] == pytest.approx(195.0) and base["steps"] == [1, 2]
    brows = {r["label"]: r for r in base["ratios"]}
    assert brows["graphs"]["timing_s/gen"]["step1_ratio"] == pytest.approx(0.5)

    # what limits concurrency
    assert "the admission cap is the limit, a larger host pool would help" in runs["graphs"]["reading"]
    assert "kv_util_peak 0.97 near 1: the GPU pool is the limit" in runs["eager"]["reading"]
    assert "no inflight_peak / inflight_cap" in runs["nolp"]["reading"]

    # host RAM: the planner's numbers for 1.5 / 3.0 / smallest host_guard ratio
    host = report["host_memory"]
    assert host["gpu_total_assumed"] is False and host["margin_gb"] == 16.0
    by_ratio = {r["cpu_offload_ratio"]: r for r in host["rows"]}
    hg = host["smallest_ratio_host_guard"]
    assert set(by_ratio) == {1.5, 3.0, hg} and hg is not None
    for value, row in by_ratio.items():
        plan = memory.plan_memory(
            total_bytes=int(42.5e9), mem_fraction=0.45, max_batch_size=64, cuda_graph=True, cpu_offload_ratio=value,
            spec_stride=8, chunk_size=256, **smoke_compare.QWEN3_06B,
        )  # fmt: skip
        assert row["pinned_gb"] == plan.cpu_pinned_bytes / memory.GIB
        assert row["inflight_worst_case"] == memory.inflight_cap("worst_case", plan, 64, 8, 256, 1024, 8192)
        assert row["inflight_host_guard"] == memory.inflight_cap("host_guard", plan, 64, 8, 256, 1024, 8192)
        assert row["fits_available"] == (row["pinned_gb"] + 16.0 <= 40.0)
    assert by_ratio[1.5]["pinned_gb"] == 16.0 and by_ratio[3.0]["pinned_gb"] == 32.0  # RUNBOOK section 6
    assert by_ratio[1.5]["fits_available"] is True and by_ratio[3.0]["fits_available"] is False
    assert by_ratio[hg]["inflight_host_guard"] == 64
    previous = smoke_compare.smallest_ratio(smoke_compare.build_parser().parse_args(["--run", "x=y"]), "host_guard")
    assert previous == hg and round(hg - 0.1, 1) not in by_ratio
    assert any("worst_case admission reaches max_batch_size=64" in n for n in host["notes"])

    # outputs
    assert "## Cost of calculate_log_probs" in markdown and "## Host RAM" in markdown and "| `timing_s/gen` |" in markdown
    assert "file not found" in markdown
    table_lines = [line for line in out.splitlines() if line and not line.startswith(PREFIX)]
    assert table_lines and max(len(line) for line in table_lines) <= 80, [l for l in table_lines if len(l) > 80]
    assert f"{PREFIX} OK: 3 of 4 run(s) with data" in out


def test_missing_and_garbage_inputs_never_crash(tmp_path, capsys):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    garbage = tmp_path / "garbage.csv"
    garbage.write_text("a,b\n1,2\n")
    weird = tmp_path / "weird.csv"
    weird.write_text("step,timing_s/gen,sparsespec/gen_s\n1,abc,5\nx,1,2\n2,nan,\n3,7,inf\n")
    out_dir = tmp_path / "out"
    argv = ["--run", f"empty={empty}", "--run", f"garbage={garbage}", "--run", f"weird={weird}",
            "--run", f"gone={tmp_path / 'nope.csv'}", "--logprob-pair", "weird,gone", "--baseline-csv",
            str(tmp_path / "no_baseline.csv"), "--out-dir", str(out_dir), "--tag", "t1"]  # fmt: skip
    status, report, markdown, out = _run(argv, capsys)
    assert status == 0 and report["usable"] is True
    runs = _by_label(report)
    assert "no header" in runs["empty"]["error"] and "no 'step' column" in runs["garbage"]["error"]
    weird_run = runs["weird"]
    assert weird_run["error"] is None and weird_run["steps"] == [1, 2, 3]
    assert weird_run["summary"]["timing_s/gen"]["mean"] == 7.0 and weird_run["summary"]["sparsespec/gen_s"]["mean"] == 5.0
    assert any("non-numeric" in n for n in weird_run["notes"]) and any("non-finite" in n for n in weird_run["notes"])
    assert any("without a numeric step" in n for n in weird_run["notes"])
    assert report["logprob_cost"]["metrics"]["sparsespec/gen_s"]["common_steps"] == []
    assert report["baseline"]["error"] == "file not found" and report["host_memory"] is None
    assert (out_dir / "smoke_compare_t1.json").exists() and (out_dir / "smoke_compare_t1.md").exists()


def test_partial_run_is_flagged(tmp_path, capsys):
    run = _csv(tmp_path, "partial", {1: _sparsespec_step(100.0), 3: _sparsespec_step(80.0)})
    status, report, _, _ = _run(["--run", f"p={run}", "--out-dir", str(tmp_path)], capsys)
    assert status == 0 and _by_label(report)["p"]["steps"] == [1, 3]
    assert any("partial: steps [2] missing" in n for n in _by_label(report)["p"]["notes"])


def test_readings(tmp_path):
    def reading(**summary):
        run = {"steps": {1: {}}, "summary": {c: {"max": None, "mean": None} for c in smoke_compare.METRICS}}
        for key, value in summary.items():
            run["summary"]["sparsespec/" + key] = {"max": value, "mean": value}
        return smoke_compare.measured_reading(run, 64)

    assert "the GPU pool is the limit" in reading(kv_util_peak=0.99, inflight_peak=27, inflight_cap=27)
    assert "the admission cap is the limit, a larger host pool would help" in reading(kv_util_peak=0.5, inflight_peak=27, inflight_cap=27)
    assert "max_batch_size is the limit" in reading(kv_util_peak=0.5, inflight_peak=64, inflight_cap=64)
    assert "both limits are near" in reading(kv_util_peak=0.9, inflight_peak=27, inflight_cap=27)
    assert "neither memory limit binds" in reading(kv_util_peak=0.5, inflight_peak=10, inflight_cap=27)
    assert "cannot tell" in reading()


def test_unusable_inputs_exit_2(tmp_path, capsys):
    status, report, _, out = _run(["--run", f"a={tmp_path / 'a.csv'}", "--run", f"b={tmp_path / 'b.csv'}",
                                   "--out-dir", str(tmp_path)], capsys)  # fmt: skip
    assert status == 2 and report["pass"] is False and report["usable"] is False
    assert "UNUSABLE" in out and report["notes"][0].startswith("UNUSABLE")
    for bad in (["--run", "nolabel"], ["--run", "=x.csv"], ["--run", "a=x", "--run", "a=y"], ["--run", "a,b=x"],
                ["--run", "a=x", "--logprob-pair", "a,zzz"], ["--run", "a=x", "--logprob-pair", "a"],
                ["--run", "a=x", "--tag", "../evil"], []):  # fmt: skip
        with pytest.raises(SystemExit) as exc:
            smoke_compare.main(bad + ["--out-dir", str(tmp_path / "never")])
        assert exc.value.code == 2, bad
    assert not (tmp_path / "never").exists()


def test_runs_as_a_module(tmp_path):
    run = _csv(tmp_path, "m", {1: _sparsespec_step(100.0)})
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    proc = subprocess.run(
        [sys.executable, "-m", "integration.gates.smoke_compare", "--run", f"m={run}", "--out-dir", str(tmp_path),
         "--tag", "mod"], cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (tmp_path / "smoke_compare_mod.json").exists()
    assert "--logprob-pair" in subprocess.run(
        [sys.executable, str(REPO_ROOT / "integration" / "gates" / "smoke_compare.py"), "--help"],
        env=env, capture_output=True, text=True, check=True,
    ).stdout
