"""baseline/run_seed_sparsespec.sh: the hydra override list of the treatment arm.

* The list is taken from the script itself (``DRY_RUN=1``).
* The baseline's list is taken from baseline/run_seed.sh by running an
  unmodified COPY of it (in a temp dir) with a ``python3`` shim on PATH that
  records its argv, so nothing is assumed about how that script builds it.
* Both lists are composed with verl's real config (hydra) and compared.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from types import SimpleNamespace

import pytest

ROLLOUT = "actor_rollout_ref.rollout."
EXPERIMENT_KEYS = {"trainer.experiment_name", "trainer.default_local_dir"}  # differ by EXP_NAME only
RUN_ENV_VARS = ("SEED", "STEPS", "EXP_NAME", "SPARSESPEC_ENGINE", "MODEL_PATH", "DATA_DIR", "CKPT_ROOT",
                "DRY_RUN", "REPO_ROOT", "PYTHONPATH", "ARGV_FILE", "CALLS_FILE", "PYTHONPATH_FILE",
                "SPARSESPEC_CUDA_GRAPH", "SPARSESPEC_ADMISSION", "SPARSESPEC_CALC_LOGPROBS",
                "ALLOW_FULL_SIZE_REFERENCE")  # fmt: skip

EXPECTED_ROLLOUT_CHANGES = {
    ROLLOUT + "name": "sparsespec",
    ROLLOUT + "tensor_model_parallel_size": "1",
    ROLLOUT + "sparsespec.engine": "sparsespec",
    ROLLOUT + "sparsespec.spec_stride": "8",
    ROLLOUT + "sparsespec.budget_ratio": "0.05",
    ROLLOUT + "sparsespec.num_min_budget": "128",
    ROLLOUT + "sparsespec.kv_cache": "pillar",
    ROLLOUT + "sparsespec.max_batch_size": "64",
    ROLLOUT + "sparsespec.cuda_graph": "True",
    ROLLOUT + "sparsespec.admission": "worst_case",
    ROLLOUT + "calculate_log_probs": "True",
}

PYTHON3_SHIM = """#!/usr/bin/env bash
# Stand-in for python3 used by the run-script tests: records what it was asked to do.
printf '%s\\n' "$*" >> "$CALLS_FILE"
if [ "${1:-}" = "-m" ] && [ "${2:-}" = "verl.trainer.main_ppo" ]; then
    shift 2
    printf '%s\\n' "$@" > "$ARGV_FILE"
    printf '%s\\n' "${PYTHONPATH:-}" > "$PYTHONPATH_FILE"
    echo "step:1 - timing_s/gen:1.0"
fi
exit 0
"""


def override_key(override: str) -> str:
    return override.split("=", 1)[0].lstrip("+")


def effective(overrides: list[str]) -> dict[str, str]:
    """key -> value as hydra resolves the list: later overrides win."""
    out = {}
    for item in overrides:
        key, value = item.split("=", 1)
        out[key.lstrip("+")] = value
    return out


@pytest.fixture(scope="module")
def bash():
    path = shutil.which("bash")
    if path is None:
        pytest.skip("bash is not available")
    return path


@pytest.fixture
def run_env(tmp_path):
    """Environment that does not depend on the caller's HOME / exported run variables."""
    env = {k: v for k, v in os.environ.items() if k not in RUN_ENV_VARS}
    env["DATA_DIR"] = str(tmp_path / "data")
    env["CKPT_ROOT"] = str(tmp_path / "ckpts")
    return env


@pytest.fixture
def dry_run(tk, bash, run_env, tmp_path):
    def run(script=None, **env_overrides) -> list[str]:
        script = script or tk.REPO_ROOT / "baseline" / "run_seed_sparsespec.sh"
        env = dict(run_env, DRY_RUN="1", **env_overrides)
        # cwd is NOT the repo: the script must locate itself
        proc = subprocess.run([bash, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        assert proc.stderr == ""
        lines = proc.stdout.splitlines()
        assert lines and all("=" in line for line in lines), proc.stdout
        return lines

    return run


@pytest.fixture
def repo_copy(tk, tmp_path):
    """Unmodified copies of the run scripts + shared config, and a python3 shim."""
    root = tmp_path / "repo_copy"
    (root / "baseline" / "config").mkdir(parents=True)
    for rel in ("baseline/run_seed.sh", "baseline/run_seed_sparsespec.sh", "baseline/config/grpo_base.sh"):
        shutil.copyfile(tk.REPO_ROOT / rel, root / rel)
    shim_dir = tmp_path / "shim-bin"
    shim_dir.mkdir()
    shim = shim_dir / "python3"
    shim.write_text(PYTHON3_SHIM)
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return SimpleNamespace(root=root, shim_dir=shim_dir)


@pytest.fixture
def run_with_shim(tk, bash, run_env, repo_copy, tmp_path):
    """Run a script of the temp copy for real, with python3 replaced by the shim.
    Returns the argv given to ``python3 -m verl.trainer.main_ppo`` (paths of the
    copy mapped back to the repository) and everything else the shim saw."""

    def run(script_name: str, **env_overrides) -> SimpleNamespace:
        tag = f"{script_name}-{len(list(tmp_path.glob('argv-*')))}"
        files = {name: tmp_path / f"{name.lower()}-{tag}" for name in ("ARGV_FILE", "CALLS_FILE", "PYTHONPATH_FILE")}
        files["ARGV_FILE"] = tmp_path / f"argv-{tag}"
        env = dict(run_env, **{k: str(v) for k, v in files.items()}, **env_overrides)
        env["PATH"] = str(repo_copy.shim_dir) + os.pathsep + env.get("PATH", "")
        proc = subprocess.run(
            [bash, str(repo_copy.root / "baseline" / script_name)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        argv = files["ARGV_FILE"].read_text().splitlines()
        argv = [a.replace(str(repo_copy.root), str(tk.REPO_ROOT)) for a in argv]
        return SimpleNamespace(
            argv=argv,
            calls=files["CALLS_FILE"].read_text().splitlines(),
            pythonpath=files["PYTHONPATH_FILE"].read_text().strip(),
            stdout=proc.stdout,
        )

    return run


@pytest.fixture
def baseline_overrides(run_with_shim):
    def get(**env_overrides) -> list[str]:
        return run_with_shim("run_seed.sh", **env_overrides).argv

    return get


def compose(overrides: list[str]):
    pytest.importorskip("verl")
    hydra = pytest.importorskip("hydra")
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    try:
        with hydra.initialize_config_module("verl.trainer.config", version_base=None):
            return hydra.compose("ppo_trainer", overrides=list(overrides))
    finally:
        GlobalHydra.instance().clear()


def flatten(node, prefix="") -> dict:
    out = {}
    if isinstance(node, dict):
        if not node and prefix:
            out[prefix[:-1]] = {}
        for key, value in node.items():
            out.update(flatten(value, f"{prefix}{key}."))
    else:
        out[prefix[:-1]] = node
    return out


# ---------------------------------------------------------------------------
# the override list
# ---------------------------------------------------------------------------
def test_dry_run_prints_overrides_and_touches_nothing(tk, dry_run, tmp_path):
    watched = [tk.REPO_ROOT / "baseline" / "logs", tk.REPO_ROOT / "baseline" / "results"]
    before = [(p.exists(), sorted(x.name for x in p.iterdir()) if p.exists() else None) for p in watched]
    overrides = dry_run()
    after = [(p.exists(), sorted(x.name for x in p.iterdir()) if p.exists() else None) for p in watched]
    assert before == after
    assert not (tmp_path / "ckpts").exists() and not (tmp_path / "data").exists()

    values = effective(overrides)
    assert values[ROLLOUT + "name"] == "sparsespec"
    assert values["data.train_files"] == str(tmp_path / "data" / "math_l3to5_train.parquet")
    assert values["trainer.default_local_dir"] == str(tmp_path / "ckpts" / "sparsespec_seed0")
    assert values["custom_reward_function.path"] == str(tk.REPO_ROOT / "baseline" / "reward.py")
    assert values["actor_rollout_ref.model.path"] == "Qwen/Qwen3-0.6B"
    # one override per line, no stray shell syntax
    assert not any(line != line.strip() or line.startswith("#") for line in overrides)


def test_only_the_rollout_block_differs_from_the_baseline(dry_run, baseline_overrides):
    ours = dry_run()
    base = baseline_overrides()
    ours_eff, base_eff = effective(ours), effective(base)

    changed = {k for k in ours_eff.keys() | base_eff.keys() if ours_eff.get(k) != base_eff.get(k)}
    outside = {k for k in changed if not k.startswith(ROLLOUT)}
    assert (
        outside == EXPERIMENT_KEYS
    ), f"overrides outside the rollout block differ: {sorted(outside - EXPERIMENT_KEYS)}"
    assert {k: ours_eff[k] for k in changed if k.startswith(ROLLOUT)} == EXPECTED_ROLLOUT_CHANGES
    assert not set(base_eff) - set(ours_eff)  # nothing the baseline sets is dropped

    # experiment name / checkpoint dir differ by EXP_NAME and nothing else
    assert base_eff["trainer.experiment_name"] == "vllm_baseline_seed0"
    assert ours_eff["trainer.experiment_name"] == "sparsespec_seed0"
    assert ours_eff["trainer.default_local_dir"] == base_eff["trainer.default_local_dir"].replace(
        "vllm_baseline_seed0", "sparsespec_seed0"
    )

    # structure: BASE_ARGS first and unchanged, then the rollout block, then the baseline's trailer
    trailer = 5  # +data.seed, experiment_name, default_local_dir, total_epochs, total_training_steps
    base_args, base_trailer = base[:-trailer], base[-trailer:]
    assert ours[: len(base_args)] == base_args
    block = ours[len(base_args) : -trailer]
    assert block and all(override_key(item).startswith(ROLLOUT) for item in block)
    assert [override_key(item) for item in ours[-trailer:]] == [override_key(item) for item in base_trailer]
    assert [i for i in ours[-trailer:] if override_key(i) not in EXPERIMENT_KEYS] == [
        i for i in base_trailer if override_key(i) not in EXPERIMENT_KEYS
    ]
    # the baseline's own rollout settings are inherited, not restated with other values
    restated = {override_key(i) for i in block} & {override_key(i) for i in base_args}
    assert restated == {ROLLOUT + "name"}
    # "+" (add a new key) is used exactly for the keys verl's schema does not have
    assert {override_key(i) for i in block if i.startswith("+")} == {
        k for k in EXPECTED_ROLLOUT_CHANGES if ".sparsespec." in k
    }


def test_same_experiment_name_leaves_only_rollout_differences(dry_run, baseline_overrides):
    ours = effective(dry_run(EXP_NAME="shared_name"))
    base = effective(baseline_overrides(EXP_NAME="shared_name"))
    changed = {k for k in ours.keys() | base.keys() if ours.get(k) != base.get(k)}
    assert changed == set(EXPECTED_ROLLOUT_CHANGES)


def test_seed_steps_and_engine_variables(dry_run, baseline_overrides):
    ours = effective(dry_run(SEED="3", STEPS="7"))
    base = effective(baseline_overrides(SEED="3", STEPS="7"))
    assert ours["data.seed"] == base["data.seed"] == "3"  # seeds are shared across arms
    assert ours["trainer.total_training_steps"] == base["trainer.total_training_steps"] == "7"
    assert ours["trainer.total_epochs"] == base["trainer.total_epochs"] == "100"
    assert ours["trainer.experiment_name"] == "sparsespec_seed3"
    assert base["trainer.experiment_name"] == "vllm_baseline_seed3"

    reference = effective(dry_run(SPARSESPEC_ENGINE="reference", EXP_NAME="reference_smoke", STEPS="3"))
    assert reference[ROLLOUT + "sparsespec.engine"] == "reference"
    assert reference["trainer.experiment_name"] == "reference_smoke"
    default = effective(dry_run())
    assert {k: v for k, v in reference.items() if v != default[k]}.keys() == {
        ROLLOUT + "sparsespec.engine", "trainer.experiment_name", "trainer.default_local_dir",
        "trainer.total_training_steps",
    }  # fmt: skip


def test_dry_run_prints_exactly_what_the_real_run_passes_to_verl(tk, dry_run, run_with_shim, repo_copy):
    """Run the (copied) script for real with python3 replaced by a recorder."""
    real = run_with_shim("run_seed_sparsespec.sh")
    assert real.argv == dry_run()
    assert dry_run(script=repo_copy.root / "baseline" / "run_seed_sparsespec.sh") == [
        a.replace(str(tk.REPO_ROOT), str(repo_copy.root)) for a in real.argv
    ]
    # the Ray workers must be able to import serve.rl.verl_rollout: repo root first on PYTHONPATH
    assert real.pythonpath.split(os.pathsep)[0] == str(repo_copy.root)

    calls = real.calls
    assert calls[0] == "-c import serve.rl.verl_rollout"  # pre-flight 1: the backend is importable
    assert calls[1] == "-"  # pre-flight 2: verl has the patch (heredoc on stdin)
    assert calls[2].startswith("-m verl.trainer.main_ppo algorithm.adv_estimator=grpo")
    assert any("parse_timing.py" in c and c.endswith("sparsespec_seed0.timing.csv") for c in calls[3:])
    log_run = [c for c in calls if "log_run.py" in c]
    assert len(log_run) == 1
    assert "--engine sparsespec" in log_run[0] and "--exp-name sparsespec_seed0" in log_run[0]
    assert "--seed 0 --steps 100" in log_run[0] and "--config-version v1" in log_run[0]
    assert (repo_copy.root / "baseline" / "logs" / "sparsespec_seed0.log").read_text().startswith("step:1")


def test_reference_engine_runs_are_not_logged_as_sparsespec_runs(run_with_shim):
    real = run_with_shim(
        "run_seed_sparsespec.sh",
        SPARSESPEC_ENGINE="reference",
        EXP_NAME="reference_smoke",
        STEPS="3",
        ALLOW_FULL_SIZE_REFERENCE="1",
    )
    assert effective(real.argv)[ROLLOUT + "sparsespec.engine"] == "reference"
    assert not [c for c in real.calls if "log_run.py" in c]
    assert "not appending to runs.jsonl" in real.stdout


def test_full_size_reference_runs_are_refused_by_default(bash, run_env, repo_copy, tmp_path):
    """The reference engine at the full-size config is hours per step: the script must stop before
    launching anything unless ALLOW_FULL_SIZE_REFERENCE=1 (DRY_RUN is exempt, smoke_reference.sh uses it)."""
    calls = tmp_path / "guard-calls"
    env = dict(
        run_env,
        SPARSESPEC_ENGINE="reference",
        CALLS_FILE=str(calls),
        ARGV_FILE=str(tmp_path / "guard-argv"),
        PYTHONPATH_FILE=str(tmp_path / "guard-pythonpath"),
        PATH=str(repo_copy.shim_dir) + os.pathsep + run_env.get("PATH", ""),
    )
    script = str(repo_copy.root / "baseline" / "run_seed_sparsespec.sh")
    proc = subprocess.run([bash, script], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert proc.returncode == 1
    assert "integration/smoke_reference.sh" in proc.stderr and "ALLOW_FULL_SIZE_REFERENCE" in proc.stderr
    assert not calls.exists() or not any("verl.trainer.main_ppo" in c for c in calls.read_text().splitlines())
    assert not (repo_copy.root / "baseline" / "logs").exists()

    dry = subprocess.run([bash, script], cwd=tmp_path, env=dict(env, DRY_RUN="1"), capture_output=True, text=True)
    assert dry.returncode == 0, dry.stderr
    assert "+" + ROLLOUT + "sparsespec.engine=reference" in dry.stdout.splitlines()


def test_cuda_graph_switch_only_changes_the_sparsespec_block(dry_run):
    on, off = dry_run(), dry_run(SPARSESPEC_CUDA_GRAPH="False")
    key = "+" + ROLLOUT + "sparsespec.cuda_graph="
    assert key + "True" in on and key + "False" in off
    assert [a for a in on if not a.startswith(key)] == [a for a in off if not a.startswith(key)]


def test_preflight_failures_stop_the_run(tk, bash, run_env, repo_copy, tmp_path):
    """If serve.rl.verl_rollout cannot be imported the script must stop before launching verl."""
    failing = tmp_path / "failing-bin"
    failing.mkdir()
    shim = failing / "python3"
    shim.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$*" >> "$CALLS_FILE"\nexit 1\n')
    shim.chmod(0o755)
    calls = tmp_path / "failing-calls"
    env = dict(run_env, CALLS_FILE=str(calls), PATH=str(failing) + os.pathsep + run_env.get("PATH", ""))
    proc = subprocess.run(
        [bash, str(repo_copy.root / "baseline" / "run_seed_sparsespec.sh")],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )  # fmt: skip
    assert proc.returncode == 1
    assert "cannot import serve.rl.verl_rollout" in proc.stderr
    assert not any("verl.trainer.main_ppo" in c for c in calls.read_text().splitlines())
    assert not (repo_copy.root / "baseline" / "logs").exists()


# ---------------------------------------------------------------------------
# the overrides against verl's real config schema
# ---------------------------------------------------------------------------
def test_overrides_compose_with_verls_config(tk, dry_run):
    from omegaconf import OmegaConf

    from serve.rl.verl_rollout import SparseSpecRolloutConfig

    overrides = dry_run()
    cfg = compose(overrides)  # raises on a duplicate-key / "+" problem: that would be a bug in the script
    rollout = cfg.actor_rollout_ref.rollout

    assert rollout.name == "sparsespec"
    block = OmegaConf.to_container(rollout.sparsespec, resolve=True)
    assert block == {
        "engine": "sparsespec",
        "spec_stride": 8,
        "budget_ratio": 0.05,
        "num_min_budget": 128,
        "kv_cache": "pillar",
        "max_batch_size": 64,
        "cuda_graph": True,
        "admission": "worst_case",
    }
    assert {k: type(v) for k, v in block.items()} == {
        "engine": str, "spec_stride": int, "budget_ratio": float, "num_min_budget": int,
        "kv_cache": str, "max_batch_size": int, "cuda_graph": bool, "admission": str,
    }  # fmt: skip

    parsed = SparseSpecRolloutConfig.from_rollout_config(rollout)  # a struct-mode DictConfig, as in the worker
    assert parsed == SparseSpecRolloutConfig(
        engine="sparsespec", spec_stride=8, budget_ratio=0.05, num_min_budget=128, kv_cache="pillar",
        max_batch_size=64, cuda_graph=True, seed=0, chunk_size=256, cpu_offload_ratio=1.5,
        kv_capacity_tokens=None, transient_reserve_gb=None, admission="worst_case",
    )  # fmt: skip

    # what the rollout class reads, with the values the script intends
    assert rollout.tensor_model_parallel_size == 1
    assert rollout.calculate_log_probs is True
    assert rollout.mode == "sync" and rollout.dtype == "bfloat16" and rollout.ignore_eos is False
    assert rollout.prompt_length == cfg.data.max_prompt_length == 1024
    assert rollout.response_length == cfg.data.max_response_length == 8192
    assert rollout.val_kwargs.temperature == 0 and rollout.val_kwargs.top_p == 1.0 and rollout.val_kwargs.top_k == -1
    assert cfg.data.seed == 0
    assert cfg.trainer.logger == ["console"]
    assert cfg.trainer.total_training_steps == 100 and cfg.trainer.experiment_name == "sparsespec_seed0"


def test_inherited_rollout_values_equal_the_baselines(dry_run, baseline_overrides):
    from omegaconf import OmegaConf

    ours = compose(dry_run())
    base = compose(baseline_overrides())
    ours_rollout, base_rollout = ours.actor_rollout_ref.rollout, base.actor_rollout_ref.rollout

    inherited = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "n": 8,
        "gpu_memory_utilization": 0.45,
        "free_cache_engine": True,
        "enforce_eager": True,  # verl's default; neither script overrides it
        "do_sample": True,
    }
    for key, value in inherited.items():
        assert ours_rollout[key] == base_rollout[key] == value, key
        assert type(ours_rollout[key]) is type(base_rollout[key]), key
    assert base_rollout.name == "vllm" and "sparsespec" not in base_rollout

    # the fully resolved configs differ in the rollout block (+ experiment name) and nowhere else
    flat_ours = flatten(OmegaConf.to_container(ours, resolve=True))
    flat_base = flatten(OmegaConf.to_container(base, resolve=True))
    changed = {
        k for k in flat_ours.keys() | flat_base.keys() if flat_ours.get(k, "<absent>") != flat_base.get(k, "<absent>")
    }
    assert changed == set(EXPECTED_ROLLOUT_CHANGES) | EXPERIMENT_KEYS
    assert flat_base[ROLLOUT + "tensor_model_parallel_size"] == 2  # verl's default, harmless for vLLM on 1 GPU
    assert flat_base[ROLLOUT + "calculate_log_probs"] is False


def test_composed_config_is_accepted_by_the_rollout_class(tk, dry_run, tiny_model):
    """SparseSpecRollout.__init__ on the real composed rollout config (engine injected)."""
    from serve.rl.verl_rollout import SparseSpecRollout

    cfg = compose(dry_run(SPARSESPEC_ENGINE="reference"))
    engine = tk.make_reference_engine(tiny_model)
    qwen3_06b = SimpleNamespace(max_position_embeddings=40960)
    rollout = SparseSpecRollout(
        model_path="unused", config=cfg.actor_rollout_ref.rollout, tokenizer=tk.TinyTokenizer(),
        model_hf_config=qwen3_06b, engine=engine,
    )  # fmt: skip
    assert rollout.ss.engine == "reference" and rollout.ss.seed == 0
    assert engine.is_sleeping is True  # free_cache_engine=True is inherited
    kwargs = rollout._sampling_kwargs(SimpleNamespace(meta_info={}))
    assert kwargs == {"temperature": 1.0, "top_p": 1.0, "top_k": -1}
    engine.wake_up()
    result = engine.generate([[9, 10, 11]], 4, **kwargs)  # the baseline's sampling settings are supported
    assert len(result.token_ids) == 1


def test_composed_config_fits_the_memory_plan_on_the_a100(dry_run):
    """gpu_memory_utilization, max_batch_size and the prompt / response lengths
    of the run are a workload the engine's planner accepts for Qwen3-0.6B."""
    from serve.rl.memory import check_workload_fits, plan_memory
    from serve.rl.verl_rollout import SparseSpecRolloutConfig

    cfg = compose(dry_run())
    rollout = cfg.actor_rollout_ref.rollout
    ss = SparseSpecRolloutConfig.from_rollout_config(rollout)
    layers, hidden, heads, kv_heads, head_dim, inter, vocab = 28, 1024, 16, 8, 128, 3072, 151_936
    params = (
        vocab * hidden
        + hidden
        + layers
        * (
            2 * hidden * heads * head_dim
            + 2 * hidden * kv_heads * head_dim
            + 3 * hidden * inter
            + 2 * hidden
            + 2 * head_dim
        )
    )
    plan = plan_memory(
        total_bytes=int(42.5e9),
        mem_fraction=float(rollout.gpu_memory_utilization),
        weights_bytes=2 * params,
        num_layers=layers,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        vocab_size=vocab,
        hidden_size=hidden,
        dtype_bytes=2,
        max_batch_size=ss.max_batch_size,
        cuda_graph=ss.cuda_graph,
        cpu_offload_ratio=ss.cpu_offload_ratio,
        # as SparseSpecEngine.__init__ passes them: None = the automatic worst-case reserve
        transient_reserve_bytes=None if ss.transient_reserve_gb is None else int(ss.transient_reserve_gb * 1024**3),
        kv_capacity_tokens=ss.kv_capacity_tokens,
        spec_stride=ss.spec_stride,
        chunk_size=ss.chunk_size,
    )
    assert plan.planned_bytes <= plan.budget_bytes == int(42.5e9 * 0.45)
    assert plan.transient_bytes == 2 * 1024**3  # the automatic reserve keeps 2 GiB at max_batch_size 64
    check_workload_fits(
        prompt_lens=[int(rollout.prompt_length)],
        max_new_tokens=int(rollout.response_length),
        spec_stride=ss.spec_stride,
        chunk_size=ss.chunk_size,
        max_batch_size=ss.max_batch_size,
        plan=plan,
        max_position_embeddings=40960,
    )


    # the in-flight caps the engine will log at start-up (expected_max_prompt_len / expected_max_new_tokens)
    from serve.rl.memory import inflight_cap

    caps = {
        mode: inflight_cap(
            mode, plan, ss.max_batch_size, ss.spec_stride, ss.chunk_size,
            int(rollout.prompt_length), int(rollout.response_length),
        )
        for mode in ("worst_case", "host_guard")
    }  # fmt: skip
    assert 26 <= caps["worst_case"] <= 29 and 45 <= caps["host_guard"] <= 49, caps


# ---------------------------------------------------------------------------
# SPARSESPEC_ADMISSION / SPARSESPEC_CALC_LOGPROBS
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "env, key, default, value",
    [
        ({"SPARSESPEC_ADMISSION": "host_guard"}, "+" + ROLLOUT + "sparsespec.admission=", "worst_case", "host_guard"),
        ({"SPARSESPEC_CALC_LOGPROBS": "False"}, ROLLOUT + "calculate_log_probs=", "True", "False"),
    ],
)
def test_admission_and_logprob_switches_change_exactly_their_override(dry_run, env, key, default, value):
    on, switched = dry_run(), dry_run(**env)
    assert [a for a in on if a.startswith(key)] == [key + default]
    assert [a for a in switched if a.startswith(key)] == [key + value]
    # nothing else moves, and the order of the list is kept
    assert [a for a in on if not a.startswith(key)] == [a for a in switched if not a.startswith(key)]
    assert on.index(key + default) == switched.index(key + value)


def test_switched_overrides_compose_with_verls_config(dry_run):
    from omegaconf import OmegaConf

    from serve.rl.verl_rollout import SparseSpecRolloutConfig

    default = compose(dry_run())
    both = compose(dry_run(SPARSESPEC_ADMISSION="host_guard", SPARSESPEC_CALC_LOGPROBS="False"))
    rollout = both.actor_rollout_ref.rollout
    assert rollout.sparsespec.admission == "host_guard" and type(rollout.sparsespec.admission) is str
    assert rollout.calculate_log_probs is False  # hydra turns "False" into a real bool
    assert default.actor_rollout_ref.rollout.calculate_log_probs is True
    parsed = SparseSpecRolloutConfig.from_rollout_config(rollout)
    assert parsed.admission == "host_guard"
    assert parsed == SparseSpecRolloutConfig.from_rollout_config(
        OmegaConf.merge(default.actor_rollout_ref.rollout, {"sparsespec": {"admission": "host_guard"}})
    )
    # the two composed configs differ in exactly these two keys
    flat_default = flatten(OmegaConf.to_container(default, resolve=True))
    flat_both = flatten(OmegaConf.to_container(both, resolve=True))
    changed = {k for k in flat_default.keys() | flat_both.keys() if flat_default.get(k) != flat_both.get(k)}
    assert changed == {ROLLOUT + "sparsespec.admission", ROLLOUT + "calculate_log_probs"}


@pytest.mark.parametrize("value", ["hostguard", "worst-case", "HOST_GUARD", "none"])
def test_invalid_admission_value_is_rejected_by_the_rollout_config(dry_run, value):
    """The script passes SPARSESPEC_ADMISSION through unchecked; the worker's config
    parsing (SparseSpecRolloutConfig) is what refuses it, before any engine is built."""
    from serve.rl.verl_rollout import SparseSpecRolloutConfig

    overrides = dry_run(SPARSESPEC_ADMISSION=value)
    assert "+" + ROLLOUT + "sparsespec.admission=" + value in overrides
    rollout = compose(overrides).actor_rollout_ref.rollout
    with pytest.raises(ValueError, match="admission"):
        SparseSpecRolloutConfig.from_rollout_config(rollout)


def test_empty_switches_fall_back_to_the_defaults(dry_run):
    """``${VAR:-default}``: an exported but empty variable means the default, not an empty override."""
    assert dry_run(SPARSESPEC_ADMISSION="", SPARSESPEC_CALC_LOGPROBS="") == dry_run()
