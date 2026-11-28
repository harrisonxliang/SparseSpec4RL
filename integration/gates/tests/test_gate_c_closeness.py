"""CPU tests of gate 3c's failure classification (``failure_class``).

    PYTHONPATH=. python -m pytest integration/gates/tests/test_gate_c_closeness.py -q

The gate runs end to end with ``--engine reference`` on a tiny random Qwen3
(float32 safetensors, tied embeddings), like ``test_gates_cpu.py`` (whose
fixtures are not imported: that file is edited independently). The
"too close" checkpoint is the base model with FOUR weights of the last layer
moved by one bf16 step, stored under ``.../hf/global_step_25``: bitwise a different model, but
no engine could make the 3x mean rule hold. The verdict must stay FAIL
(exit 1) and the outputs must say why, with the step-50 rerun command; a
broken engine on the same checkpoint must still be classified ``engine``, and
the synthetic 2% checkpoint must still pass.

No ``__init__.py`` next to this file (see test_gates_cpu.py).
"""

from __future__ import annotations

import json
import random
import shlex
import sys
import types
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from integration.gates import common  # noqa: E402
from integration.gates import gate_c_reload  # noqa: E402
from integration.gates import gate_c_vllm_reference  # noqa: E402
from integration.gates import make_synthetic_ckpt as synth  # noqa: E402

VOCAB = 512
EOS_IDS = [VOCAB - 2, VOCAB - 1]
PAD_ID = VOCAB - 2
MOVED_KEY = "model.layers.1.mlp.down_proj.weight"  # last layer: tiny effect on log-probs (about 1e-4)
NUM_MOVED = 4


# ----------------------------------------------------------------------------
# fixtures: tiny model, prompts, checkpoints
# ----------------------------------------------------------------------------
def _write_tiny_qwen3(path: Path, seed: int = 0) -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, max_position_embeddings=512, tie_word_embeddings=True,
        initializer_range=0.5, eos_token_id=EOS_IDS[1], pad_token_id=PAD_ID, bos_token_id=None,
    )  # fmt: skip
    rng_state = torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        model = Qwen3ForCausalLM(config).to(torch.float32).eval()
    finally:
        torch.set_rng_state(rng_state)
    model.generation_config.eos_token_id = list(EOS_IDS)
    model.generation_config.pad_token_id = PAD_ID
    model.save_pretrained(str(path), safe_serialization=True)


def _move_by_one_bf16_step(state: dict, key: str, count: int) -> dict:
    """``state`` with the first ``count`` elements of ``key`` moved up by one bf16 step
    (values stay float32, exactly representable in bf16)."""
    new = {k: v.clone() for k, v in state.items()}
    flat = new[key].view(-1)
    old_bf16 = flat[:count].to(torch.bfloat16)
    bits = old_bf16.view(torch.int16) + 1
    flat[:count] = bits.view(torch.bfloat16).to(torch.float32)
    return new


@pytest.fixture(scope="module")
def workdir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("gate_c_closeness")


@pytest.fixture(scope="module")
def tiny_dir(workdir) -> str:
    path = workdir / "tiny-qwen3"
    _write_tiny_qwen3(path)
    return str(path)


@pytest.fixture(scope="module")
def prompts_file(workdir) -> str:
    rng = random.Random(0)
    prompts = [[rng.randrange(0, VOCAB - 8) for _ in range(rng.randrange(5, 40))] for _ in range(16)]
    path = workdir / "prompts.json"
    path.write_text(json.dumps(prompts))
    return str(path)


@pytest.fixture(scope="module")
def close_ckpt(workdir, tiny_dir) -> str:
    """Base model + 4 weights one bf16 step away, where a real run keeps its step-25 checkpoint."""
    from safetensors.torch import save_file

    out = workdir / "run" / "hf" / "global_step_25"
    out.mkdir(parents=True)
    base = common.read_state_dict(tiny_dir)
    assert MOVED_KEY in base and base[MOVED_KEY].dtype == torch.float32
    state = _move_by_one_bf16_step(base, MOVED_KEY, NUM_MOVED)
    common.copy_non_weight_files(tiny_dir, str(out))
    save_file(state, str(out / "model.safetensors"), metadata={"format": "pt"})
    diff = common.bf16_diff_report(base, common.read_state_dict(str(out)))
    assert diff["num_changed"] == NUM_MOVED and diff["num_tensors_changed"] == 1, diff
    return str(out)


@pytest.fixture(scope="module")
def synthetic_dir(workdir, tiny_dir) -> str:
    out = workdir / "tiny-synthetic"
    report = synth.make_synthetic_checkpoint(tiny_dir, str(out), rel_noise=synth.DEFAULT_REL_NOISE, seed=0)
    assert report["bf16_fraction_changed"] > 0.5
    return str(out)


def _gate_argv(tiny_dir, ckpt, prompts_file, out_dir, tag, *extra) -> list[str]:
    return ["--engine", "reference", "--device", "cpu", "--model", tiny_dir, "--ckpt", ckpt, "--prompts-file",
            prompts_file, "--num-prompts", "6", "--n", "4", "--max-new-tokens", "40", "--fingerprint-prompts", "3",
            "--fingerprint-tokens", "16", "--out-dir", str(out_dir), "--tag", tag, *extra]  # fmt: skip


def _run_gate(argv, out_dir, tag) -> tuple[int, dict]:
    status = gate_c_reload.main(argv)
    with open(Path(out_dir) / f"gate_c_reload_{tag}.json") as f:
        return status, json.load(f)


class _NoopReloadEngine:
    """A ReferenceEngine whose load_weights does nothing (the classic reload bug)."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def load_weights(self, named_tensors):
        from serve.rl.base import LoadReport

        return LoadReport(num_params=0, num_sources=0, num_bytes=0, seconds=0.0)


# ----------------------------------------------------------------------------
# pure helpers
# ----------------------------------------------------------------------------
def _items(failed=(), *, extra_ok=()):
    names = list(gate_c_reload.ENGINE_CHECKS) + list(gate_c_reload.MEAN_RULE_CHECKS) + list(extra_ok)
    return [{"name": n, "ok": n not in failed, "detail": ""} for n in names]


def test_classify_failure_rules():
    classify = gate_c_reload.classify_failure
    under, engine = gate_c_reload.FAILURE_UNDERPOWERED, gate_c_reload.FAILURE_ENGINE
    mean_rule = "negative_control_clearly_larger"
    inconclusive = gate_c_reload.FAILURE_INCONCLUSIVE
    ref = {"mean": 0.01, "usable": True}
    kw = dict(neg_control_mode="mean", mean_rule_ok=False, distance=0.001, needed=0.03, reloaded_mean=0.01, reference=ref)

    assert classify(_items(), **{**kw, "mean_rule_ok": True})[0] is None
    assert classify(_items([mean_rule]), **kw)[0] == under
    # without a usable same-checkpoint stock-vLLM reference the bar is the engine's own noise: inconclusive
    assert classify(_items([mean_rule]), **{**kw, "reference": None})[0] == inconclusive
    assert classify(_items([mean_rule]), **{**kw, "reference": {"mean": 0.01, "usable": False}})[0] == inconclusive
    assert classify(_items([mean_rule]), **{**kw, "reference": {"mean": float("nan"), "usable": True}})[0] == inconclusive
    # a noisy engine raises its own bar: 0.06 vs stock vLLM 0.02, checkpoint 0.08 apart -> engine, not underpowered
    noisy = {**kw, "reloaded_mean": 0.06, "needed": 0.18, "distance": 0.08, "reference": {"mean": 0.02, "usable": True}}
    assert classify(_items([mean_rule]), **noisy)[0] == engine
    # engine as precise as vLLM, but a vLLM-quality engine would have passed (0.08 > 3 x 0.02)
    assert classify(_items([mean_rule]), **{**noisy, "reloaded_mean": 0.035, "needed": 0.105})[0] == engine
    # both conditions hold: underpowered
    assert classify(_items([mean_rule]), **{**noisy, "reloaded_mean": 0.03, "needed": 0.09, "distance": 0.05})[0] == under
    # the models differ by more than the rule demands: a correct engine would have passed it
    assert classify(_items([mean_rule]), **{**kw, "distance": 0.05})[0] == engine
    assert classify(_items([mean_rule]), **{**kw, "distance": float("nan")})[0] == engine
    assert classify(_items([mean_rule]), **{**kw, "needed": float("inf")})[0] == engine
    # any other failed check is the engine's, whatever the distance
    for other in ("reload1_bitwise", "reload2_pointers_unchanged", "fingerprint_F2_equals_F1",
                  "sample_1_response_contract", "base_after_roundtrip_same_range", "reloaded_mean_small"):  # fmt: skip
        assert classify(_items([mean_rule, other]), **kw)[0] == engine, other
    tracking = ["tracking_follows_checkpoint_after_reload1", "tracking_follows_base_after_reload2"]
    assert classify(_items([mean_rule, tracking[0]], extra_ok=tracking), **kw)[0] == engine
    assert classify(_items([mean_rule], extra_ok=tracking), **kw)[0] == under
    # an engine check that did not run at all: conservative
    missing = [c for c in _items([mean_rule]) if c["name"] != "reload3_bitwise"]
    assert classify(missing, **kw)[0] == engine
    # tracking mode: the mean rule is not a check; a tracking failure is "engine"
    tracking_mode = {**kw, "neg_control_mode": "tracking"}
    assert classify(_items(["tracking_evaluable"], extra_ok=["tracking_evaluable"]), **tracking_mode)[0] == engine


def test_rerun_command_derivation(tmp_path):
    assert gate_c_reload.next_checkpoint("/x/run/hf/global_step_25")["dir"] == "/x/run/hf/global_step_50"
    assert gate_c_reload.next_checkpoint("/x/run/hf/global_step_25/")["next_step"] == 50
    assert gate_c_reload.next_checkpoint("/x/run/hf/global_step_75")["next_step"] == 100
    assert gate_c_reload.next_checkpoint("/x/run/hf/global_step_100") is None
    assert gate_c_reload.next_checkpoint("/x/synthetic/relnoise0.02_seed0") is None
    assert gate_c_reload.next_tag("sparsespec_step25", 25, 50) == "sparsespec_step50"
    assert gate_c_reload.next_tag("step_25_a", 25, 50) == "step_50_a"
    assert gate_c_reload.next_tag("sparsespec_step250", 25, 50) == "sparsespec_step250_step50"
    assert gate_c_reload.next_tag("sparsespec", 25, 50) == "sparsespec_step50"
    assert gate_c_reload._set_flag(["--a", "1", "--ckpt=x", "--b"], "--ckpt", "y") == ["--a", "1", "--ckpt", "y", "--b"]
    assert gate_c_reload._set_flag(["--a", "1"], "--tag", "t") == ["--a", "1", "--tag", "t"]
    assert gate_c_reload._flag_value(["--n", "4", "--num-prompts=6"], "--num-prompts") == "6"

    class Args:
        engine, vllm_reference = "sparsespec", None

    ckpt = tmp_path / "hf" / "global_step_25"
    rerun = gate_c_reload.rerun_commands(["--engine", "sparsespec", "--ckpt", str(ckpt)], Args, str(ckpt), "sparsespec")
    assert rerun["next_step"] == 50 and rerun["next_ckpt_exists"] is False and len(rerun["commands"]) == 1
    assert shlex.split(rerun["commands"][0]["command"]) == [
        "python", "-m", "integration.gates.gate_c_reload", "--engine", "sparsespec",
        "--ckpt", str(tmp_path / "hf" / "global_step_50"), "--tag", "sparsespec_step50",
    ]  # fmt: skip
    (tmp_path / "hf" / "global_step_50").mkdir(parents=True)
    assert gate_c_reload.rerun_commands([], Args, str(ckpt), "t")["next_ckpt_exists"] is True
    generic = gate_c_reload.rerun_commands([], Args, str(tmp_path / "somewhere"), "t")
    assert generic["commands"] == [] and "global_step_50" in generic["hint"]


# ----------------------------------------------------------------------------
# end to end
# ----------------------------------------------------------------------------
def test_close_checkpoint_is_classified_underpowered(tiny_dir, close_ckpt, prompts_file, tmp_path):
    # a stock-vLLM reference file for the same checkpoint (the gate only reads its stats)
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    ref_path = ref_dir / "gate_c_vllm_reference.json"
    ref_stats = {"count": 100, "num_nonfinite": 0, "mean": 0.01, "median": 0.005, "p99": 0.1, "max": 0.3,
                 "max_prob_ratio": 1.35}  # fmt: skip
    ref_path.write_text(json.dumps({"pass": True, "config": {"ckpt": close_ckpt, "num_sequences": 24, "max_new_tokens": 40},
                                    "stats": {"engine_vs_trainer": ref_stats}}))  # fmt: skip
    tag = "close_step25"
    argv = _gate_argv(tiny_dir, close_ckpt, prompts_file, tmp_path, tag, "--vllm-reference", str(ref_path))
    status, result = _run_gate(argv, tmp_path, tag)

    # the verdict itself is untouched
    assert status == 1 and result["pass"] is False
    assert result["failed_checks"] == ["negative_control_clearly_larger"], result["failed_checks"]
    assert result["failure_class"] == "negative_control_underpowered", result["failure_reason"]
    distance = result["stats"]["hf_base_vs_ckpt"]["mean"]
    needed = 3.0 * max(result["stats"]["reloaded"]["mean"], 1e-4)
    assert distance <= needed
    assert result["checkpoint_vs_base"]["num_changed"] == NUM_MOVED

    text = result["failure_explanation"]
    assert "too close to the base model" in text and "consistent with a correct engine" in text
    assert "not an engine failure" not in text and "however correct" not in text
    assert common.fmt_num(distance) in text and common.fmt_num(needed) in text
    assert "Tracking coefficients" in text and common.fmt_num(result["tracking"]["sample_1"]["se"]) in text
    # what the tracking coefficients say is in the paragraph either way (with four bf16 steps in the last
    # layer they are usually not evaluable; if they are, they were checks and passed)
    if result["negative_control"]["tracking_evaluable"]:
        assert "were part of the verdict and passed" in text
        assert {"tracking_follows_checkpoint_after_reload1", "tracking_follows_base_after_reload2"} <= {
            c["name"] for c in result["checks"] if c["ok"]
        }
    else:
        assert "NOT evaluable" in text and not any(c["name"].startswith("tracking") for c in result["checks"])

    # rerun: stock-vLLM reference on step 50 first, then the same gate command with step 50 everywhere
    rerun = result["rerun"]
    step50 = str(Path(close_ckpt).parent / "global_step_50")
    assert rerun["next_step"] == 50 and rerun["next_ckpt"] == step50 and rerun["next_ckpt_exists"] is False
    assert [c["env"] for c in rerun["commands"]] == ["verl-baseline", "verl-sparsespec-dev or verl-sparsespec"]
    ref_cmd, gate_cmd = (shlex.split(c["command"]) for c in rerun["commands"])
    assert ref_cmd[:2] == ["python", "integration/gates/gate_c_vllm_reference.py"]
    assert ref_cmd[ref_cmd.index("--ckpt") + 1] == step50 and ref_cmd[ref_cmd.index("--tag") + 1] == "step50"
    assert ref_cmd[ref_cmd.index("--out-dir") + 1] == str(ref_dir)
    for flag, value in (("--num-prompts", "6"), ("--n", "4"), ("--max-new-tokens", "40"), ("--prompts-file", prompts_file)):
        assert ref_cmd[ref_cmd.index(flag) + 1] == value, flag
    new_ref = ref_dir / (gate_c_vllm_reference.output_name("step50") + ".json")
    expected = list(argv)
    expected[expected.index("--ckpt") + 1] = step50
    expected[expected.index("--tag") + 1] = "close_step50"
    expected[expected.index("--vllm-reference") + 1] = str(new_ref)
    assert gate_cmd == ["python", "-m", "integration.gates.gate_c_reload", *expected]
    for command in rerun["commands"]:
        assert command["command"] in text

    markdown = (tmp_path / f"gate_c_reload_{tag}.md").read_text()
    assert "**Verdict: FAIL**" in markdown and "negative_control_underpowered" in markdown
    assert text in markdown and rerun["commands"][1]["command"] in markdown and "```bash" in markdown


def test_close_checkpoint_with_a_broken_engine_is_still_an_engine_failure(
    tiny_dir, close_ckpt, prompts_file, tmp_path, monkeypatch
):
    real_build = common.build_engine
    monkeypatch.setattr(common, "build_engine", lambda kind, path, **opts: _NoopReloadEngine(real_build(kind, path, **opts)))
    status, result = _run_gate(_gate_argv(tiny_dir, close_ckpt, prompts_file, tmp_path, "close_noop"), tmp_path, "close_noop")
    assert status == 1 and result["pass"] is False
    assert {"reload1_bitwise", "reload1_base_no_longer_matches", "reload3_bitwise"} <= set(result["failed_checks"])
    assert result["failure_class"] == "engine" and result["failure_explanation"] is None and result["rerun"] is None
    markdown = (tmp_path / "gate_c_reload_close_noop.md").read_text()
    assert "Failure class: `engine`" in markdown and "negative_control_underpowered" not in markdown


@pytest.mark.parametrize("mode", ["mean", "tracking"])
def test_synthetic_checkpoint_still_passes(tiny_dir, synthetic_dir, prompts_file, tmp_path, mode):
    tag = f"synthetic_{mode}"
    status, result = _run_gate(
        _gate_argv(tiny_dir, synthetic_dir, prompts_file, tmp_path, tag, "--neg-control", mode), tmp_path, tag
    )
    assert status == 0 and result["pass"] is True
    assert result["failure_class"] is None and result["failed_checks"] == [] and result["failure_explanation"] is None
    assert result["negative_control"]["mode"] == mode and result["negative_control"]["mean_rule_ok"] is True
    names = {c["name"] for c in result["checks"]}
    assert ("negative_control_clearly_larger" in names) == (mode == "mean")
    assert {"tracking_follows_checkpoint_after_reload1", "tracking_follows_base_after_reload2"} <= names


def test_underpowered_explanation_without_a_step_number():
    """A checkpoint path without global_step_<N> gets the generic hint instead of a command."""
    stats = {"reloaded": {"mean": 0.008}, "negative_control": {"mean": 0.009}}
    args = types.SimpleNamespace(neg_control_factor=3.0, noise_floor=1e-4, tracking_max_se=0.1, same_range_factor=2.0)
    tracking = {"sample_1": {"beta": 0.97, "se": 0.04}, "sample_2": {"beta": 0.02, "se": 0.05}}
    diff = {"fraction_changed": 0.001, "num_changed": 10, "num_elements": 10_000}
    rerun = {"next_step": None, "commands": [], "hint": gate_c_reload.GENERIC_RERUN_HINT}
    text = gate_c_reload.underpowered_explanation(
        stats=stats, weights_diff=diff, needed=0.024, distance=0.002, args=args, tracking=tracking,
        tracking_evaluable=True, rerun=rerun, reference_mean=0.007,
    )  # fmt: skip
    assert gate_c_reload.GENERIC_RERUN_HINT in text and "were part of the verdict and passed" in text
    assert "0.97 +/- 0.04" in text and "0.024" in text and "0.002" in text


class _NoisyLogprobEngine:
    """A ReferenceEngine whose sampled-token log-probs carry extra noise (reload itself is correct)."""

    def __init__(self, inner, scale):
        import numpy as np

        self._inner, self._scale, self._rng = inner, scale, np.random.default_rng(123)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def generate(self, *args, **kwargs):
        result = self._inner.generate(*args, **kwargs)
        if kwargs.get("temperature", 1.0) > 0 and result.logprobs is not None:
            result.logprobs = [[min(0.0, x + float(self._rng.laplace(0.0, self._scale))) for x in row]
                               for row in result.logprobs]  # fmt: skip
        return result


def test_noisy_engine_is_not_blamed_on_the_checkpoint(tiny_dir, prompts_file, tmp_path, monkeypatch):
    """A checkpoint a correct engine passes on; the same engine with 3x noisier log-probs (still far below
    --max-mean-abs-dlogp) fails only the mean rule, and must NOT be called underpowered."""
    ckpt = tmp_path / "run" / "hf" / "global_step_25"
    synth.make_synthetic_checkpoint(tiny_dir, str(ckpt), rel_noise=0.001, seed=0)
    argv = lambda tag, *extra: _gate_argv(tiny_dir, str(ckpt), prompts_file, tmp_path, tag, "--num-prompts", "16",
                                          "--n", "8", "--max-new-tokens", "64", *extra)  # fmt: skip
    status, correct = _run_gate(argv("correct"), tmp_path, "correct")
    assert status == 0 and correct["pass"] is True, correct["failed_checks"]

    ref_path = tmp_path / "ref.json"
    ref_stats = {"count": 1000, "num_nonfinite": 0, "mean": correct["stats"]["reloaded"]["mean"], "median": 0.0,
                 "p99": 0.0, "max": 0.0, "max_prob_ratio": 1.0}  # fmt: skip
    ref_path.write_text(json.dumps({"pass": True, "config": {"ckpt": str(ckpt), "max_new_tokens": 64},
                                    "stats": {"engine_vs_trainer": ref_stats}}))  # fmt: skip
    real_build = common.build_engine
    monkeypatch.setattr(common, "build_engine", lambda kind, path, **o: _NoisyLogprobEngine(real_build(kind, path, **o), 0.02))
    for tag, extra, want in (("noisy_ref", ("--vllm-reference", str(ref_path)), gate_c_reload.FAILURE_ENGINE),
                             ("noisy_noref", (), gate_c_reload.FAILURE_INCONCLUSIVE)):  # fmt: skip
        status, result = _run_gate(argv(tag, *extra), tmp_path, tag)
        assert status == 1 and result["failed_checks"] == ["negative_control_clearly_larger"], result["failed_checks"]
        assert result["stats"]["reloaded"]["mean"] < 0.1  # passes the absolute cap: only the mean rule catches it
        assert result["failure_class"] == want, (tag, result["failure_reason"])
        assert result["failure_explanation"] is None
