"""sparsespec/* metrics: timing dict -> trainer metrics -> console log line ->
baseline/parse_timing.py -> CSV columns.

The chain is exercised with the real pieces: the statement
integration/verl-0.5.0-sparsespec.patch adds to ``RayPPOTrainer.fit``, verl's
console logger formatting (``concat_dict_to_str``) and the repo's log parser.
"""

from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys

import pytest

verl = pytest.importorskip("verl")

from verl.utils.logger.aggregate_logger import LocalLogger, concat_dict_to_str  # noqa: E402

# One value per sparsespec/* key, chosen to cover the number formats the logger
# produces: plain decimals, a value below 1e-4 (printed as 3.2e-05; the
# reference engine's sleep/wake really are microseconds), long mantissas, zero.
SYNTHETIC = {
    "sparsespec/weight_sync_s": 0.8125,
    "sparsespec/wake_s": 2.5,
    "sparsespec/graph_capture_s": 1.7523456789,
    "sparsespec/sleep_s": 3.2e-05,
    "sparsespec/accept_len_mean": 6.283185307179586,
    "sparsespec/accept_rate": 0.6603981633974483,
    "sparsespec/num_cycles": 123456.0,
    "sparsespec/num_iters": 2500000.0,
    "sparsespec/batch_size_mean": 37.5,
    "sparsespec/num_offloads": 0.0,
    "sparsespec/kv_util_peak": 0.9999999,
    "sparsespec/host_util_peak": 0.4375,
    "sparsespec/inflight_cap": 49.0,
    "sparsespec/inflight_peak": 48.0,
    "sparsespec/peak_mem_gb": 17.9,
    "sparsespec/process_peak_alloc_gb": 31.25,
    "sparsespec/process_peak_reserved_gb": 33.75,
    "sparsespec/gen_s": 1234.5678,
    "sparsespec/resident_asleep_gb": 2.171875,
}


@pytest.fixture(scope="module")
def parse_timing(tk):
    """baseline/parse_timing.py imported by path (baseline/ is not a package).
    Bytecode caching is off for the import so nothing is written into baseline/."""
    path = tk.REPO_ROOT / "baseline" / "parse_timing.py"
    spec = importlib.util.spec_from_file_location("baseline_parse_timing", path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def trainer_metrics(tk, worker_timing: dict, step: int, patched: bool = True) -> dict:
    """The part of RayPPOTrainer.fit that turns the worker's timing dict into
    logged metrics (ray_trainer.py l.1174-1175 and compute_timing_metrics)."""
    metrics = {}
    timing_raw = {"gen": 12.5}  # marked_timer("gen", timing_raw) around the rollout call
    timing_raw.update(worker_timing)
    if patched:
        tk.apply_trainer_metrics_statement(metrics, timing_raw)
    timing_raw.update({"step": 40.0, "update_actor": 20.25})
    # trainer/ppo/metric_utils.py::compute_timing_metrics, the part that needs no batch
    metrics.update({f"timing_s/{name}": value for name, value in timing_raw.items()})
    metrics.update(
        {
            "training/global_step": step,
            "training/epoch": 0,
            "critic/score/mean": 0.125,
            "response_length/mean": 3071.4,
            "actor/lr": 1e-06,
        }
    )
    return metrics


def test_synthetic_table_covers_every_published_key(tk):
    assert set(SYNTHETIC) == set(tk.SPARSESPEC_TIMING_KEYS)
    assert all(type(v) is float for v in SYNTHETIC.values())
    assert len(set(SYNTHETIC.values())) == len(SYNTHETIC)  # distinct: a mixed-up column cannot pass


def test_patch_statement_moves_sparsespec_keys_into_metrics(tk):
    statement = tk.trainer_metrics_statement()
    assert statement.startswith("metrics.update(") and "sparsespec/" in statement

    worker_timing = {"generate_sequences": 9.0, "reshard": 3.25, **SYNTHETIC}
    metrics, timing_raw = {"already": 1}, {"gen": 12.5, **worker_timing}
    tk.apply_trainer_metrics_statement(metrics, timing_raw)
    assert metrics == {"already": 1, **SYNTHETIC}  # logged under exactly their own names
    assert timing_raw == {"gen": 12.5, "generate_sequences": 9.0, "reshard": 3.25}  # the rest stays timing

    # a no-op for every other backend (vLLM's timing dict has no such keys)
    metrics, timing_raw = {}, {"gen": 12.5, "generate_sequences": 9.0, "reshard": 3.25}
    tk.apply_trainer_metrics_statement(metrics, timing_raw)
    assert metrics == {} and timing_raw == {"gen": 12.5, "generate_sequences": 9.0, "reshard": 3.25}


def test_console_line_is_parsed_back_with_every_sparsespec_column(tk, parse_timing, tmp_path, capsys):
    worker_timing = {"generate_sequences": 9.0, "reshard": 3.25, **SYNTHETIC}
    lines = []
    for step in (1, 2, 3):
        scaled = {k: (v * step if k.startswith("sparsespec/") else v) for k, v in worker_timing.items()}
        metrics = trainer_metrics(tk, scaled, step)
        LocalLogger(print_to_console=True).log(data=metrics, step=step)  # trainer.logger=['console']
        printed = capsys.readouterr().out
        assert printed == concat_dict_to_str(metrics, step=step) + "\n"
        lines.append(printed)

    log = tmp_path / "sparsespec_seed0.log"
    log.write_text(
        "[run_sparsespec] sparsespec_seed0: seed=0 steps=3 config=v1 engine=sparsespec\n"
        "INFO sparsespec weight sync: 311 engine tensors in 0.81s\n"
        + lines[0]
        + "(TaskRunner pid=4242) Training Progress:   1%|          | 1/100 [06:12<10:14:31, 372.44s/it]\n"
        + "(TaskRunner pid=4242) " + lines[1]  # ray prefixes the driver's output
        + "\x1b[36m(TaskRunner pid=4242)\x1b[0m " + lines[2]
    )  # fmt: skip

    steps = parse_timing.parse(str(log))
    assert sorted(steps) == [1, 2, 3]
    for step in (1, 2, 3):
        row = steps[step]
        for key, value in SYNTHETIC.items():
            assert key in row, f"step {step}: column {key} was not captured"
            assert row[key] == value * step, (step, key)
        assert {k for k in row if k.startswith("sparsespec/")} == set(tk.SPARSESPEC_TIMING_KEYS)
        assert not any(k.startswith("timing_s/sparsespec") for k in row)
        # the columns the baseline arm reports are still there, with the same names
        assert row["timing_s/gen"] == 12.5 and row["timing_s/reshard"] == 3.25
        assert row["timing_s/generate_sequences"] == 9.0
        assert row["timing_s/step"] == 40.0 and row["timing_s/update_actor"] == 20.25
        assert row["critic/score/mean"] == 0.125 and row["response_length/mean"] == 3071.4
        assert row["actor/lr"] == 1e-06
        assert "step" not in row


def test_without_the_trainer_patch_the_keys_arrive_with_the_timing_prefix(tk, parse_timing, tmp_path):
    """Documented fallback (SparseSpecShardingManager docstring): nothing is lost."""
    worker_timing = {"generate_sequences": 9.0, "reshard": 3.25, **SYNTHETIC}
    metrics = trainer_metrics(tk, worker_timing, step=7, patched=False)
    log = tmp_path / "unpatched.log"
    log.write_text(concat_dict_to_str(metrics, step=7) + "\n")
    row = parse_timing.parse(str(log))[7]
    for key, value in SYNTHETIC.items():
        assert row["timing_s/" + key] == value
    assert not any(k.startswith("sparsespec/") for k in row)


def test_parse_timing_cli_writes_the_columns_to_the_csv(tk, tmp_path):
    """The run script calls `python3 baseline/parse_timing.py LOG -o CSV`."""
    worker_timing = {"generate_sequences": 9.0, "reshard": 3.25, **SYNTHETIC}
    log = tmp_path / "run.log"
    log.write_text(
        "".join(concat_dict_to_str(trainer_metrics(tk, worker_timing, step), step=step) + "\n" for step in (1, 2))
    )
    out = tmp_path / "run.timing.csv"
    proc = subprocess.run(
        [sys.executable, str(tk.REPO_ROOT / "baseline" / "parse_timing.py"), str(log), "-o", str(out)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "2 steps" in proc.stdout
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [row["step"] for row in rows] == ["1", "2"]
    for row in rows:
        for key, value in SYNTHETIC.items():
            assert float(row[key]) == value, key
        assert float(row["timing_s/gen"]) == 12.5


def test_real_worker_timing_survives_the_whole_chain(tk, tiny_model, parse_timing, tmp_path):
    """Same chain, fed by an actual rollout through the sharding manager."""
    from serve.rl.verl_rollout import SparseSpecRollout, SparseSpecShardingManager

    config = tk.make_rollout_config(prompt_length=24, response_length=16)
    engine = tk.make_reference_engine(tk.make_tiny_qwen3(1))
    rollout = SparseSpecRollout("unused", config, tk.TinyTokenizer(), tiny_model.config, engine=engine)
    manager = SparseSpecShardingManager(module=tiny_model, rollout=rollout, rollout_config=config)

    logged = {}
    lines = []
    for step in (1, 2):
        _, gen_batch = tk.build_gen_batch(tk.make_prompts([5, 24, 2, 11], seed=step), 24, n=8, global_steps=step)
        output = tk.worker_generate_sequences(rollout, manager, gen_batch)
        worker_timing = output.meta_info["timing"]
        assert set(tk.SPARSESPEC_TIMING_KEYS) <= set(worker_timing)
        metrics = trainer_metrics(tk, worker_timing, step)
        logged[step] = metrics
        lines.append(concat_dict_to_str(metrics, step=step))
        # every value survives the logger's "is it a number" filter
        for key in tk.SPARSESPEC_TIMING_KEYS:
            assert f"{key}:" in lines[-1], key

    log = tmp_path / "reference_plumbing.log"
    log.write_text("\n".join(lines) + "\n")
    steps = parse_timing.parse(str(log))
    assert sorted(steps) == [1, 2]
    for step, metrics in logged.items():
        for key in tk.SPARSESPEC_TIMING_KEYS:
            assert steps[step][key] == metrics[key], (step, key)  # repr round-trips exactly
        assert steps[step]["sparsespec/gen_s"] > 0.0 and steps[step]["sparsespec/num_iters"] > 0.0
        assert steps[step]["sparsespec/weight_sync_s"] > 0.0
        assert steps[step]["timing_s/generate_sequences"] == metrics["timing_s/generate_sequences"]
