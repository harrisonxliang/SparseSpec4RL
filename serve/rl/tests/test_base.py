"""serve/rl/base.py: the response contract and argument validation."""

from __future__ import annotations

import dataclasses
import inspect
import os
import subprocess
import sys

import pytest

from serve.rl import base
from serve.rl.base import (
    FINISH_LENGTH,
    FINISH_STOP,
    STAT_KEYS,
    GenerationResult,
    LoadReport,
    RolloutEngine,
    empty_stats,
    finalize_response,
    first_eos_index,
    normalize_eos_ids,
    validate_prompts,
    validate_sampling_params,
)

EOS = (7, 3)


def _lps(n):
    return [-0.1 * (i + 1) for i in range(n)]


# ---------------------------------------------------------------------------
# finalize_response
# ---------------------------------------------------------------------------
def test_finalize_keeps_the_first_eos_and_drops_everything_after():
    tokens = [10, 11, 7, 12, 13]
    toks, lps, reason = finalize_response(tokens, _lps(5), max_new_tokens=10, eos_ids=EOS)
    assert toks == [10, 11, 7]
    assert lps == _lps(5)[:3]
    assert reason == FINISH_STOP == "stop"


def test_finalize_first_of_several_eos_wins_and_any_eos_id_counts():
    tokens = [10, 3, 11, 7, 3]
    toks, lps, reason = finalize_response(tokens, _lps(5), max_new_tokens=10, eos_ids=EOS)
    assert toks == [10, 3]
    assert lps == _lps(5)[:2]
    assert reason == "stop"


def test_finalize_eos_as_first_token():
    toks, lps, reason = finalize_response([7, 20, 21], _lps(3), max_new_tokens=8, eos_ids=EOS)
    assert (toks, lps, reason) == ([7], _lps(3)[:1], "stop")


def test_finalize_cuts_to_exactly_max_new_tokens():
    tokens = list(range(20, 28))
    toks, lps, reason = finalize_response(tokens, _lps(8), max_new_tokens=5, eos_ids=EOS)
    assert toks == tokens[:5]
    assert lps == _lps(8)[:5]
    assert reason == FINISH_LENGTH == "length"


def test_finalize_eos_beyond_the_cut_is_length():
    tokens = [20, 21, 22, 23, 7]
    toks, lps, reason = finalize_response(tokens, _lps(5), max_new_tokens=4, eos_ids=EOS)
    assert toks == [20, 21, 22, 23]
    assert lps == _lps(5)[:4]
    assert reason == "length"


def test_finalize_eos_exactly_at_the_last_allowed_position_is_stop():
    tokens = [20, 21, 22, 7, 30, 31]
    toks, lps, reason = finalize_response(tokens, _lps(6), max_new_tokens=4, eos_ids=EOS)
    assert toks == [20, 21, 22, 7]
    assert len(toks) == 4
    assert lps == _lps(6)[:4]
    assert reason == "stop"


def test_finalize_without_eos_and_exact_length():
    tokens = [20, 21, 22]
    toks, lps, reason = finalize_response(tokens, _lps(3), max_new_tokens=3, eos_ids=EOS)
    assert (toks, lps, reason) == (tokens, _lps(3), "length")


@pytest.mark.parametrize("num_logprobs", [0, 2, 4])
def test_finalize_mismatched_lengths_raise(num_logprobs):
    with pytest.raises(RuntimeError, match="3 response tokens"):
        finalize_response([20, 21, 22], _lps(num_logprobs), max_new_tokens=8, eos_ids=EOS)


@pytest.mark.parametrize(
    "tokens, max_new, expected, reason",
    [
        ([20, 7, 21, 22], 8, [20, 7], FINISH_STOP),  # EOS kept, the rest dropped
        ([20, 21, 22, 23, 24], 3, [20, 21, 22], FINISH_LENGTH),  # cut to max_new_tokens
        ([7], 1, [7], FINISH_STOP),
        ([20, 21, 3], 3, [20, 21, 3], FINISH_STOP),  # EOS exactly at the last allowed position
    ],
)
def test_finalize_without_logprobs(tokens, max_new, expected, reason):
    """return_logprobs=False: the engines pass None, which stays None (not [] and not trimmed)."""
    toks, lps, why = finalize_response(tokens, None, max_new_tokens=max_new, eos_ids=EOS)
    assert (toks, lps, why) == (expected, None, reason)
    # the same tokens and reason as with log-probs
    with_lps = finalize_response(tokens, _lps(len(tokens)), max_new_tokens=max_new, eos_ids=EOS)
    assert with_lps[0] == toks and with_lps[2] == why and len(with_lps[1]) == len(toks)


def test_finalize_without_logprobs_has_no_length_check():
    # nothing to compare against: only the token rule applies
    assert finalize_response([20, 21, 22], None, max_new_tokens=2, eos_ids=EOS) == ([20, 21], None, FINISH_LENGTH)


def test_finalize_mismatch_is_checked_before_trimming():
    # equal after the cut to max_new_tokens, but the engine's bookkeeping was already off
    with pytest.raises(RuntimeError):
        finalize_response([20, 21, 22, 23], _lps(3), max_new_tokens=2, eos_ids=EOS)


@pytest.mark.parametrize("eos_ids", [(7, 3), [7, 3], {7, 3}, frozenset({7, 3})])
def test_finalize_accepts_any_eos_container_and_does_not_mutate_inputs(eos_ids):
    tokens = (20, 7, 21)  # a tuple: must not be modified, result must be new lists
    logprobs = (-1.0, -2.0, -3.0)
    toks, lps, reason = finalize_response(tokens, logprobs, max_new_tokens=8, eos_ids=eos_ids)
    assert isinstance(toks, list) and isinstance(lps, list)
    assert toks == [20, 7] and lps == [-1.0, -2.0] and reason == "stop"
    assert tokens == (20, 7, 21) and logprobs == (-1.0, -2.0, -3.0)

    source = [20, 7, 21]
    out, _, _ = finalize_response(source, [0.0, 0.0, 0.0], max_new_tokens=8, eos_ids=eos_ids)
    assert source == [20, 7, 21]
    out.append(99)
    assert source == [20, 7, 21]


def test_finalize_tokens_and_logprobs_always_have_equal_length():
    for tokens in ([20] * 9, [20, 7] + [21] * 7, [7], [20, 21, 3]):
        for max_new in (1, 2, 3, 9, 20):
            toks, lps, reason = finalize_response(tokens, _lps(len(tokens)), max_new, EOS)
            assert len(toks) == len(lps) >= 1
            assert len(toks) <= max_new
            assert reason in ("stop", "length")
            if reason == "stop":
                assert toks[-1] in EOS and not any(t in EOS for t in toks[:-1])
            else:
                assert not any(t in EOS for t in toks)


# ---------------------------------------------------------------------------
# first_eos_index
# ---------------------------------------------------------------------------
def test_first_eos_index():
    tokens = [9, 7, 9, 3, 7]
    assert first_eos_index(tokens, EOS) == 1
    assert first_eos_index(tokens, EOS, start=1) == 1
    assert first_eos_index(tokens, EOS, start=2) == 3
    assert first_eos_index(tokens, EOS, start=4) == 4
    assert first_eos_index(tokens, EOS, start=5) == -1
    assert first_eos_index(tokens, {3}) == 3
    assert first_eos_index([9, 9, 9], EOS) == -1
    assert first_eos_index([], EOS) == -1
    assert first_eos_index(tokens, ()) == -1


# ---------------------------------------------------------------------------
# normalize_eos_ids
# ---------------------------------------------------------------------------
def test_normalize_eos_ids():
    assert normalize_eos_ids(None) == ()
    assert normalize_eos_ids(7) == (7,)
    assert normalize_eos_ids([151645, 151643]) == (151643, 151645)  # verl passes the generation-config list
    assert normalize_eos_ids((7, 3, 7, 3)) == (3, 7)
    assert normalize_eos_ids([7, None, 3]) == (3, 7)
    assert normalize_eos_ids([]) == ()
    assert normalize_eos_ids([None]) == ()
    assert normalize_eos_ids({5}) == (5,)
    result = normalize_eos_ids([7, 3])
    assert isinstance(result, tuple) and all(type(t) is int for t in result)


# ---------------------------------------------------------------------------
# validate_sampling_params
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "temperature, top_p, top_k",
    [
        (1.0, 1.0, -1),  # the baseline config
        (0.7, 1.0, -1),
        (0.0, 1.0, -1),  # greedy
        (0, 1, 0),  # ints as hydra / yaml deliver them; top_k=0 is HF's "disabled"
        (1.0, None, None),
    ],
)
def test_validate_sampling_params_accepts_disabled_filters(temperature, top_p, top_k):
    assert validate_sampling_params(temperature, top_p, top_k) is None


@pytest.mark.parametrize("top_p", [0.9, 0.0, 0.999])
def test_validate_sampling_params_rejects_top_p(top_p):
    with pytest.raises(NotImplementedError, match="top_p"):
        validate_sampling_params(1.0, top_p, -1)


@pytest.mark.parametrize("top_k", [1, 50, -2])
def test_validate_sampling_params_rejects_top_k(top_k):
    with pytest.raises(NotImplementedError, match="top_k"):
        validate_sampling_params(1.0, 1.0, top_k)


@pytest.mark.parametrize("temperature", [-0.1, -1, None])
def test_validate_sampling_params_rejects_bad_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        validate_sampling_params(temperature, 1.0, -1)


# ---------------------------------------------------------------------------
# validate_prompts
# ---------------------------------------------------------------------------
def test_validate_prompts():
    assert validate_prompts([[1, 2], [3]], 1) is None
    with pytest.raises(ValueError, match="max_new_tokens"):
        validate_prompts([[1]], 0)
    with pytest.raises(ValueError, match="at least one prompt"):
        validate_prompts([], 4)
    with pytest.raises(ValueError, match="prompt 1 is empty"):
        validate_prompts([[1], []], 4)


# ---------------------------------------------------------------------------
# containers / interface
# ---------------------------------------------------------------------------
def test_stat_keys_and_empty_stats():
    assert len(STAT_KEYS) == len(set(STAT_KEYS))
    for required in (
        "num_requests",
        "num_tokens",
        "num_cycles",
        "draft_tokens",
        "accepted_draft_tokens",
        "accept_len_mean",
        "accept_rate",
        "num_iters",
        "batch_size_mean",
        "num_offloads",
        "kv_util_peak",
        "host_util_peak",
        "inflight_cap",
        "inflight_peak",
        "peak_mem_gb",
        "gen_s",
    ):
        assert required in STAT_KEYS
    stats = empty_stats()
    assert tuple(stats) == STAT_KEYS
    assert all(type(v) is float and v == 0.0 for v in stats.values())
    stats["gen_s"] = 1.0
    assert empty_stats()["gen_s"] == 0.0  # a fresh dict every call


def test_generation_result_defaults():
    a = GenerationResult(token_ids=[[1]], logprobs=[[0.0]], finish_reasons=["length"])
    b = GenerationResult(token_ids=[[1]], logprobs=[[0.0]], finish_reasons=["length"])
    assert tuple(a.stats) == STAT_KEYS
    assert a.stats is not b.stats
    assert GenerationResult(token_ids=[[1]], logprobs=None, finish_reasons=["length"]).logprobs is None


def test_load_report_fields():
    assert [f.name for f in dataclasses.fields(LoadReport)] == ["num_params", "num_sources", "num_bytes", "seconds"]


def test_rollout_engine_is_abstract_with_the_documented_interface():
    with pytest.raises(TypeError):
        RolloutEngine()
    assert RolloutEngine.__abstractmethods__ == frozenset(
        {"generate", "load_weights", "sleep", "wake_up", "is_sleeping"}
    )

    signature = inspect.signature(RolloutEngine.generate)
    keyword_only = {n: p.default for n, p in signature.parameters.items() if p.kind is p.KEYWORD_ONLY}
    assert keyword_only == {"eos_token_ids": None, "return_logprobs": True}
    assert [n for n, p in signature.parameters.items() if p.kind is not p.KEYWORD_ONLY] == [
        "self", "prompt_token_ids", "max_new_tokens", "temperature", "top_p", "top_k",
    ]  # fmt: skip

    class Minimal(RolloutEngine):
        def generate(
            self,
            prompt_token_ids,
            max_new_tokens,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            *,
            eos_token_ids=None,
            return_logprobs=True,
        ):
            return GenerationResult([], [] if return_logprobs else None, [])

        def load_weights(self, named_tensors):
            return LoadReport(0, 0, 0, 0.0)

        def sleep(self):
            pass

        def wake_up(self):
            pass

        @property
        def is_sleeping(self):
            return False

    engine = Minimal()
    assert engine.last_wake_stats == {"wake_s": 0.0, "graph_capture_s": 0.0}
    assert engine.close() is None


def test_cpu_modules_do_not_import_the_gpu_stack(tk):
    """serve.rl (and the verl glue) must stay importable without flashinfer:
    the engine and its scheduler are only imported lazily. Checked in a fresh
    interpreter so other test modules cannot influence the result."""
    code = (
        "import sys\n"
        "import serve.rl, serve.rl.base, serve.rl.weights, serve.rl.memory\n"
        "import serve.rl.reference_engine, serve.rl.verl_rollout\n"
        "heavy = ('flashinfer', 'serve.model.model', 'serve.rl.engine', 'serve.rl.scheduler')\n"
        "loaded = [m for m in heavy if m in sys.modules]\n"
        "assert not loaded, loaded\n"
        "assert serve.rl.RolloutEngine is serve.rl.base.RolloutEngine\n"
        "assert serve.rl.ReferenceEngine is serve.rl.reference_engine.ReferenceEngine\n"
        "print('ok')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tk.REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", code], cwd=tk.REPO_ROOT, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().endswith("ok")


def test_lazy_package_attributes():
    import serve.rl as rl

    assert rl.GenerationResult is GenerationResult
    assert set(rl.__all__) == {"GenerationResult", "RolloutEngine", "ReferenceEngine", "SparseSpecEngine"}
    with pytest.raises(AttributeError):
        rl.NoSuchEngine  # noqa: B018
    assert base.NamedTensors is not None
