"""GPU tests for ``serve.rl.engine.SparseSpecEngine`` (run on the A100 box).

Three invocations, from the repo root (PYTHONPATH=repo), each its own pytest
process (``SparseSpecEngine`` allows one engine per process):

    # 1) everything except the offload stress test, default admission (worst_case)
    python -m pytest serve/rl/tests/test_engine_gpu.py -v -s --log-cli-level=INFO
    # 2) the same with the other admission mode
    SPARSESPEC_TEST_ADMISSION=host_guard python -m pytest serve/rl/tests/test_engine_gpu.py -v -s --log-cli-level=INFO
    # 3) offload stress: an engine with a deliberately small GPU KV pool; runs ONLY
    #    test_offload_stress, test_admission_stats and test_return_logprobs_false
    #    (everything else is skipped with a reason). Combine with 2) for host_guard.
    SPARSESPEC_TEST_OFFLOAD_STRESS=1 python -m pytest serve/rl/tests/test_engine_gpu.py -v -s --log-cli-level=INFO

The whole module is SKIPPED unless CUDA is available and the custom flashinfer
build imports (nothing is stubbed). If both are there, a failure to import
``serve.rl.engine`` itself is an ERROR, not a skip.

ONE engine is built for the module (``SparseSpecEngine`` allows one per
process) and every test restores what it changes (weights, sleep state).
The box is shared: the engine fixture first checks free GPU memory and skips
with a clear reason rather than squeezing next to somebody else's job.

Environment variables
    SPARSESPEC_TEST_MODEL            model dir or hub id   (default Qwen/Qwen3-0.6B)
    SPARSESPEC_TEST_MEM_FRACTION     engine GPU share      (default 0.25: a ~3.4 GiB KV pool on a 40 GB card)
    SPARSESPEC_TEST_MAX_BATCH_SIZE   engine batch size     (default 16)
    SPARSESPEC_TEST_DEVICE           cuda device           (default cuda:0)
    SPARSESPEC_TEST_CUDA_GRAPH       0 to disable graphs   (default 1)
    SPARSESPEC_TEST_NUM_MIN_BUDGET   PillarAttn min budget (default 128, the training value)
    SPARSESPEC_TEST_TIE_MARGIN       HF logit margin that still counts as a bf16 near-tie (default 1.0)
    SPARSESPEC_TEST_MAX_MEAN_DLOGP   bound on mean |delta log p| engine vs trainer forward (default 0.05)
    SPARSESPEC_TEST_ADMISSION        engine admission: worst_case (default) | host_guard
    SPARSESPEC_TEST_OFFLOAD_STRESS   1 = offload stress process (see 3 above; default 0)
  only read with SPARSESPEC_TEST_OFFLOAD_STRESS=1:
    SPARSESPEC_TEST_KV_CAPACITY_TOKENS        GPU KV pool size (default 8192 tokens: with the defaults
                                              below 7 (worst_case) or 9 (host_guard) requests in flight
                                              need ~13.7-17.6k tokens)
    SPARSESPEC_TEST_STRESS_NUM_REQUESTS       requests in the stress batch (default 24)
    SPARSESPEC_TEST_STRESS_MAX_NEW_TOKENS     response length of every stress request (default 1536: ~1.95k KV
                                              tokens per request, so offload victims often hold more than one
                                              Request.MAX_OFFLOADING_LEN = 1024-token step, as in training)
    SPARSESPEC_TEST_STRESS_MAX_RESPONSE_DLOGP bound on the worst single response's mean |delta log p|
                                              (default 3 x SPARSESPEC_TEST_MAX_MEAN_DLOGP)

The HF comparisons reuse the gate helpers (``integration/gates``): the same
"trainer forward" and the same near-tie rule as gates 3b / 3c.
"""

from __future__ import annotations

import gc
import math
import os
import sys
import warnings
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("SparseSpecEngine needs a CUDA device (run this module on the A100 box)", allow_module_level=True)
try:
    import flashinfer  # noqa: F401  (the fork in 3rdparty/flashinfer; never stubbed)
except Exception as exc:  # ImportError, or OSError from a broken CUDA extension
    pytest.skip(
        f"flashinfer does not import ({type(exc).__name__}: {exc}); SparseSpecEngine needs the custom build "
        "from 3rdparty/flashinfer",
        allow_module_level=True,
    )

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from integration.gates import common  # noqa: E402
from integration.gates.gate_b_greedy import analyse_prompt  # noqa: E402
from integration.gates.make_synthetic_ckpt import perturb_state_dict  # noqa: E402
from serve.rl import memory  # noqa: E402
from serve.rl.base import STAT_KEYS  # noqa: E402


def _env(name: str, default: str, cast):
    raw = os.environ.get(name, default)
    try:
        return cast(raw)
    except ValueError:
        raise ValueError(f"environment variable {name}={raw!r} is not a valid {cast.__name__}") from None


MODEL = os.environ.get("SPARSESPEC_TEST_MODEL", "Qwen/Qwen3-0.6B")
MEM_FRACTION = _env("SPARSESPEC_TEST_MEM_FRACTION", "0.25", float)
MAX_BATCH_SIZE = _env("SPARSESPEC_TEST_MAX_BATCH_SIZE", "16", int)
DEVICE = os.environ.get("SPARSESPEC_TEST_DEVICE", "cuda:0")
CUDA_GRAPH = os.environ.get("SPARSESPEC_TEST_CUDA_GRAPH", "1") != "0"
NUM_MIN_BUDGET = _env("SPARSESPEC_TEST_NUM_MIN_BUDGET", "128", int)
TIE_MARGIN = _env("SPARSESPEC_TEST_TIE_MARGIN", "1.0", float)
MAX_MEAN_DLOGP = _env("SPARSESPEC_TEST_MAX_MEAN_DLOGP", "0.05", float)
ADMISSION = os.environ.get("SPARSESPEC_TEST_ADMISSION", "worst_case")
if ADMISSION not in memory.ADMISSION_MODES:
    raise ValueError(f"SPARSESPEC_TEST_ADMISSION={ADMISSION!r}: expected one of {memory.ADMISSION_MODES}")
OFFLOAD_STRESS = os.environ.get("SPARSESPEC_TEST_OFFLOAD_STRESS", "0") not in ("", "0")
KV_CAPACITY_TOKENS = _env("SPARSESPEC_TEST_KV_CAPACITY_TOKENS", "8192", int)
STRESS_NUM_REQUESTS = _env("SPARSESPEC_TEST_STRESS_NUM_REQUESTS", "24", int)
STRESS_MAX_NEW_TOKENS = _env("SPARSESPEC_TEST_STRESS_MAX_NEW_TOKENS", "1536", int)
STRESS_MAX_RESPONSE_DLOGP = _env("SPARSESPEC_TEST_STRESS_MAX_RESPONSE_DLOGP", str(3 * MAX_MEAN_DLOGP), float)

# tests that run in the offload-stress process (SPARSESPEC_TEST_OFFLOAD_STRESS=1); all others are skipped there
STRESS_PROCESS_TESTS = ("test_offload_stress", "test_admission_stats", "test_return_logprobs_false")
# tests that run ONLY in the offload-stress process
STRESS_ONLY_TESTS = ("test_offload_stress",)

SPEC_STRIDE = 8
BUDGET_RATIO = 0.05
GIB = 1024**3
MIB = 1024**2

QUESTIONS = [
    "What is the sum of all positive divisors of 36?",
    "If $3x + 7 = 25$, what is the value of $x^2 - 1$?",
    "How many ways can 5 distinct books be arranged on a shelf if two particular books must be adjacent?",
    "A right triangle has legs of length 9 and 12. What is the radius of its inscribed circle?",
]


def _long_question(k: int, terms: int = 70) -> str:
    """A several-hundred-token problem: longer than the prefill chunk (256) and
    than PillarAttn's minimum budget (128), so chunked prefill and genuinely
    sparse drafting are both exercised."""
    values = [(7 * i * i + 3 * i + 11 * k) % 97 + 1 for i in range(1, terms + 1)]
    listing = ", ".join(f"a_{i} = {v}" for i, v in enumerate(values, 1))
    return (
        f"A sequence is given by the following values: {listing}. "
        f"What is the remainder when a_{k + 1} + a_{k + 2} + a_{k + 3} is divided by 5?"
    )


# ----------------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def model_dir() -> str:
    return common.resolve_model_dir(MODEL)


@pytest.fixture(scope="module")
def eos(model_dir) -> list[int]:
    ids = common.eos_ids_from_checkpoint(model_dir)
    assert ids, "the test model has no EOS ids"
    return ids


@pytest.fixture(scope="module")
def prompts(model_dir) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def build(question: str) -> list[int]:
        return common.build_prompt_ids(tokenizer, [{"role": "user", "content": question + common.INSTRUCTION}])

    short = [build(q) for q in QUESTIONS]
    long = [build(_long_question(k)) for k in range(3)]
    assert all(len(p) > 256 for p in long), [len(p) for p in long]
    # offload stress: eight distinct prompts of nearly the same length (chunked prefill, 2 chunks)
    stress = [build(_long_question(k, terms=40)) for k in range(3, 11)]
    assert all(len(p) > 256 for p in stress), [len(p) for p in stress]
    print(
        f"\n[prompts] short: {[len(p) for p in short]} tokens, long: {[len(p) for p in long]} tokens, "
        f"stress: {[len(p) for p in stress]} tokens"
    )
    return {"short": short, "long": long, "stress": stress}


@pytest.fixture(scope="module")
def engine(model_dir):
    device = torch.device(DEVICE)
    free, total = torch.cuda.mem_get_info(device)
    needed = MEM_FRACTION * total + 5 * GIB  # engine share + two HF bf16 copies and scoring scratch
    if free < needed:
        pytest.skip(
            f"{DEVICE} has {free / GIB:.1f} GiB free, the tests need {needed / GIB:.1f} GiB "
            f"(SPARSESPEC_TEST_MEM_FRACTION={MEM_FRACTION}); the box is shared, not starting"
        )
    from serve.rl.engine import SparseSpecEngine  # an import error here is a real failure

    extra = {"kv_capacity_tokens": KV_CAPACITY_TOKENS} if OFFLOAD_STRESS else {}
    print(
        f"\n[engine] admission={ADMISSION}, offload stress process: {OFFLOAD_STRESS}"
        + (f" (kv_capacity_tokens={KV_CAPACITY_TOKENS})" if OFFLOAD_STRESS else "")
    )
    eng = SparseSpecEngine(
        model_dir,
        mem_fraction=MEM_FRACTION,
        max_batch_size=MAX_BATCH_SIZE,
        spec_stride=SPEC_STRIDE,
        budget_ratio=BUDGET_RATIO,
        num_min_budget=NUM_MIN_BUDGET,
        kv_cache="pillar",
        cuda_graph=CUDA_GRAPH,
        seed=0,
        dtype="bfloat16",
        device=DEVICE,
        admission=ADMISSION,
        **extra,
    )
    assert eng.admission == ADMISSION
    if OFFLOAD_STRESS:
        assert eng.kv_pool.capacity == eng.plan.kv_capacity_tokens == KV_CAPACITY_TOKENS
    print("\n[engine] memory plan:\n" + eng.plan.describe())
    print(f"[engine] measured: {eng.memory_report()['measured']}")
    yield eng
    eng.close()
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def hf_model(model_dir, engine):
    """The trainer-side reference: HF model, bf16, base weights (built after the engine)."""
    model = common.load_hf_model(model_dir, DEVICE, "bfloat16")
    yield model
    del model
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def base_state(model_dir) -> dict:
    return common.read_state_dict(model_dir)


@pytest.fixture(autouse=True)
def _process_kind(request):
    """The offload-stress process (small KV pool) runs only STRESS_PROCESS_TESTS; the
    normal process skips STRESS_ONLY_TESTS. (The module-scoped engine may already be
    built when this runs: pytest sets up higher-scoped fixtures first. That is the same
    engine either way.)"""
    name = getattr(request.node, "originalname", None) or request.node.name
    if OFFLOAD_STRESS and name not in STRESS_PROCESS_TESTS:
        pytest.skip(
            f"SPARSESPEC_TEST_OFFLOAD_STRESS=1: this process has a {KV_CAPACITY_TOKENS}-token KV pool and runs only "
            f"{', '.join(STRESS_PROCESS_TESTS)}"
        )
    if not OFFLOAD_STRESS and name in STRESS_ONLY_TESTS:
        pytest.skip("needs its own pytest process: rerun with SPARSESPEC_TEST_OFFLOAD_STRESS=1 (see the module docstring)")
    yield


@pytest.fixture(autouse=True)
def _no_kv_pool_leak(caplog):
    """``SparseSpecEngine._check_pool_clean`` repairs a dirty KV pool and only
    LOGS an error, so a page leak would be invisible in the pool counters
    afterwards. Every test therefore fails if that error was logged."""
    yield
    # caplog.records is per phase: in teardown it no longer holds what the test body logged
    records = caplog.get_records("setup") + caplog.get_records("call")
    leaks = [r.getMessage() for r in records if "KV pool not clean" in r.getMessage()]
    assert not leaks, f"the engine leaked KV pages: {leaks}"


def _allocated() -> int:
    torch.cuda.synchronize(torch.device(DEVICE))
    return torch.cuda.memory_allocated(torch.device(DEVICE))


def _assert_torch_defaults_untouched():
    """The trainer lives in the same process: after an engine call torch's
    process-wide defaults must be what they were (float32 on CPU, and no
    leftover device context that would redirect tensor factories)."""
    assert torch.get_default_dtype() == torch.float32
    assert torch.get_default_device() == torch.device("cpu")
    assert torch.empty(1).device.type == "cpu" and torch.empty(1).dtype == torch.float32
    probe = torch.ones(2, device=torch.device(DEVICE))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # torch.tensor(tensor) recommends clone(); the copy is the point here
        # with a stale DeviceContext("cpu") mode installed this copy would land on the CPU
        assert torch.tensor(probe).device == probe.device


def _greedy(engine, batch, max_new_tokens, eos):
    result = engine.generate(batch, max_new_tokens=max_new_tokens, temperature=0.0, eos_token_ids=eos)
    return [list(t) for t in result.token_ids], result


def _compare_with_hf_greedy(model, batch, engine_tokens, max_new_tokens, eos, label):
    """Gate 3b's rule: identical, or diverging only where HF's own margin is a bf16 near-tie."""
    records = []
    for i, (prompt, tokens) in enumerate(zip(batch, engine_tokens)):
        hf_tokens = common.hf_greedy(model, prompt, max_new_tokens, eos)
        record = analyse_prompt(model, prompt, tokens, hf_tokens, TIE_MARGIN)
        records.append(record)
        print(
            f"[{label}] prompt {i} ({len(prompt)} tok): engine {record['engine_len']} / HF {record['hf_len']} tokens, "
            f"first diff {record['first_diff']}, margin {record['margin']}, status {record['status']}"
        )
    return records


# ----------------------------------------------------------------------------
# construction
# ----------------------------------------------------------------------------
def test_engine_starts_awake_with_checkpoint_weights(engine, base_state):
    assert engine.is_sleeping is False
    assert engine.name == "sparsespec"
    report = engine.verify_weights(base_state)
    assert report["ok"], report
    assert report["tied_ok"]
    if getattr(engine.config, "tie_word_embeddings", False):
        assert report["tied_shared_storage"] is True, "lm_head and embed_tokens should share one storage"
    assert engine.kv_pool.num_free_pages == engine.kv_pool.capacity == engine.plan.kv_capacity_tokens
    _assert_torch_defaults_untouched()  # after construction (graph capture included)


def test_second_engine_in_the_same_process_is_refused(engine, model_dir):
    from serve.rl.engine import SparseSpecEngine

    with pytest.raises(RuntimeError, match="one engine per process"):
        SparseSpecEngine(model_dir, mem_fraction=MEM_FRACTION, max_batch_size=MAX_BATCH_SIZE, device=DEVICE)


def test_unsupported_requests_are_rejected_before_any_forward(engine, prompts, eos):
    batch = prompts["short"][:1]
    with pytest.raises(NotImplementedError):
        engine.generate(batch, max_new_tokens=4, temperature=1.0, top_p=0.9, eos_token_ids=eos)
    with pytest.raises(NotImplementedError):
        engine.generate(batch, max_new_tokens=4, temperature=1.0, top_k=20, eos_token_ids=eos)
    with pytest.raises(ValueError):
        engine.generate([[]], max_new_tokens=4, eos_token_ids=eos)
    with pytest.raises(ValueError):
        engine.generate(batch, max_new_tokens=0, eos_token_ids=eos)
    with pytest.raises(ValueError):
        engine.generate([[engine.config.vocab_size]], max_new_tokens=4, eos_token_ids=eos)
    assert engine.kv_pool.num_free_pages == engine.kv_pool.capacity


# ----------------------------------------------------------------------------
# greedy == HF greedy
# ----------------------------------------------------------------------------
def test_greedy_matches_hf(engine, hf_model, prompts, eos):
    max_new_tokens = 64
    batch = prompts["short"][:3] + prompts["long"][:2]
    tokens, result = _greedy(engine, batch, max_new_tokens, eos)
    assert not common.check_responses(result, max_new_tokens, eos, greedy=True)

    records = _compare_with_hf_greedy(hf_model, batch, tokens, max_new_tokens, eos, "greedy")
    bad = [(i, r["status"], r["first_diff"], r["margin"]) for i, r in enumerate(records) if r["status"].isupper()]
    assert not bad, (
        f"engine greedy output diverges from HF greedy against a clear HF margin (> {TIE_MARGIN} logits) or by "
        f"length: (prompt, status, first diff, margin) = {bad}"
    )
    print(f"[greedy] engine stats: {result.stats}")


# ----------------------------------------------------------------------------
# response contract around the speculation stride
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("temperature", [1.0, 0.0])
@pytest.mark.parametrize("max_new_tokens", [1, 2, 8, 9, 10, 17, 33])
def test_response_contract(engine, prompts, eos, max_new_tokens, temperature):
    # five requests start their first verify cycle at five different offsets
    # (the scheduler staggers them), one prompt needs chunked prefill
    batch = prompts["short"] + prompts["long"][:1]
    result = engine.generate(batch, max_new_tokens=max_new_tokens, temperature=temperature, eos_token_ids=eos)

    assert len(result.token_ids) == len(result.logprobs) == len(result.finish_reasons) == len(batch)
    eos_set = set(eos)
    for i, (tokens, logprobs, reason) in enumerate(zip(result.token_ids, result.logprobs, result.finish_reasons)):
        assert 1 <= len(tokens) <= max_new_tokens, (i, len(tokens))
        assert len(logprobs) == len(tokens), (i, len(tokens), len(logprobs))
        assert all(isinstance(t, int) and 0 <= t < engine.config.vocab_size for t in tokens)
        if tokens[-1] in eos_set:
            assert reason == "stop"
        else:
            assert len(tokens) == max_new_tokens, f"response {i}: {len(tokens)} tokens, no EOS, limit {max_new_tokens}"
            assert reason == "length"
        assert not any(t in eos_set for t in tokens[:-1]), f"response {i}: tokens after an EOS"
        assert all(math.isfinite(x) and x <= 0.0 for x in logprobs), (i, logprobs)
        if temperature == 0.0:
            assert all(x == 0.0 for x in logprobs), "greedy log-probs are defined as 0.0"
    # the shared checker used by the gates must agree
    assert not common.check_responses(result, max_new_tokens, eos, greedy=temperature == 0.0)
    assert result.stats["num_requests"] == len(batch)
    assert result.stats["num_tokens"] == sum(len(t) for t in result.token_ids)


def test_eos_is_kept_and_nothing_follows_it(engine, prompts, eos):
    """Declare a token of the greedy response to be THE EOS and generate again:
    the response must be the old one cut right after the first occurrence of
    that token, wherever it falls in a draft / verify cycle (first token,
    inside a cycle, on a cycle boundary)."""
    max_new_tokens = 33
    prompt = prompts["long"][0]  # a single request: the schedule up to the stop is identical in both runs
    baseline, _ = _greedy(engine, [prompt], max_new_tokens, eos)
    baseline = baseline[0]
    positions = [j for j in (0, 1, 4, 7, 8, 9, 12, 16, 17, 25) if j < len(baseline)]
    assert positions, baseline
    for j in positions:
        fake_eos = baseline[j]
        expected = baseline[: baseline.index(fake_eos) + 1]
        result = engine.generate([prompt], max_new_tokens=max_new_tokens, temperature=0.0, eos_token_ids=[fake_eos])
        assert list(result.token_ids[0]) == expected, f"EOS at response position {baseline.index(fake_eos)}"
        assert result.finish_reasons[0] == "stop"
        assert result.logprobs[0] == [0.0] * len(expected)


# ----------------------------------------------------------------------------
# target log-probs
# ----------------------------------------------------------------------------
def test_target_logprobs_match_trainer_forward(engine, hf_model, prompts, eos):
    """The log-probs the engine reports at T=1 are those of the full-attention
    (target) model: they agree with the trainer forward on the same tokens."""
    max_new_tokens = 96
    batch = common.repeat_interleave(prompts["short"][:2] + prompts["long"][:2], 2)  # verl repeats prompts n times
    result = engine.generate(batch, max_new_tokens=max_new_tokens, temperature=1.0, eos_token_ids=eos)
    assert not common.check_responses(result, max_new_tokens, eos)

    trainer = common.hf_trainer_logprobs(hf_model, batch, result.token_ids, temperature=1.0, pad_token_id=eos[-1])
    stats = common.compare_logprobs(result.logprobs, trainer)
    print(f"\n[logprobs] engine vs HF trainer forward, T=1: {stats}")
    print(
        f"[logprobs] accept_len_mean={result.stats['accept_len_mean']:.2f} accept_rate={result.stats['accept_rate']:.3f} "
        f"cycles={result.stats['num_cycles']:.0f} draft_tokens={result.stats['draft_tokens']:.0f}"
    )
    assert stats["num_nonfinite"] == 0
    assert stats["count"] == sum(len(t) for t in result.token_ids)
    assert stats["mean"] <= MAX_MEAN_DLOGP, (
        f"mean |delta log p| = {stats['mean']:.4f} > {MAX_MEAN_DLOGP} (SPARSESPEC_TEST_MAX_MEAN_DLOGP); "
        f"median {stats['median']:.4f}, p99 {stats['p99']:.4f}, max {stats['max']:.4f}"
    )


# ----------------------------------------------------------------------------
# RNG
# ----------------------------------------------------------------------------
def test_rng_stream_is_private_and_continues(engine, prompts, eos):
    device = torch.device(DEVICE)
    batch = prompts["short"][:3]

    torch.manual_seed(4321)
    reference_draw = torch.rand(16, device=device)

    torch.manual_seed(4321)
    state_before = torch.cuda.get_rng_state(device)
    first = engine.generate(batch, max_new_tokens=32, temperature=1.0, eos_token_ids=eos)
    state_after = torch.cuda.get_rng_state(device)
    assert torch.equal(state_before, state_after), "generate() changed the caller's CUDA RNG state"
    assert torch.equal(torch.rand(16, device=device), reference_draw), "the caller's random stream was disturbed"

    # Same caller seed again: if the engine drew from the caller's stream the
    # second call would repeat the first. Its own stream continues instead.
    torch.manual_seed(4321)
    second = engine.generate(batch, max_new_tokens=32, temperature=1.0, eos_token_ids=eos)
    assert first.token_ids != second.token_ids, "two consecutive T=1 calls returned identical samples"
    _assert_torch_defaults_untouched()


# ----------------------------------------------------------------------------
# in-place weight reload
# ----------------------------------------------------------------------------
def test_inplace_reload_is_bitwise_and_changes_generation(engine, hf_model, base_state, prompts, eos):
    max_new_tokens = 32
    batch = prompts["short"][:2] + prompts["long"][:1]
    tied = bool(getattr(engine.config, "tie_word_embeddings", False))
    noisy_state = perturb_state_dict(base_state, rel_noise=0.02, seed=0, tied=tied)  # float32, like the trainer's
    pointers = engine.parameter_pointers()
    before, _ = _greedy(engine, batch, max_new_tokens, eos)

    try:
        report = engine.load_weights(noisy_state)
        assert report.num_params > 0 and report.num_bytes > 0
        assert engine.parameter_pointers() == pointers, "load_weights reallocated a parameter"
        verify = engine.verify_weights(noisy_state)
        assert verify["ok"], verify
        assert verify["tied_ok"]
        assert not engine.verify_weights(base_state)["ok"], "the engine still holds the base weights"

        after, result = _greedy(engine, batch, max_new_tokens, eos)
        assert not common.check_responses(result, max_new_tokens, eos, greedy=True)
        assert after != before, "greedy output did not change after loading different weights"

        # HF with the same new weights (the reference engine's load path, pure torch)
        from serve.rl.weights import load_weights_inplace

        with torch.no_grad():
            load_weights_inplace(hf_model, noisy_state)
        records = _compare_with_hf_greedy(hf_model, batch, after, max_new_tokens, eos, "reload")
        bad = [(i, r["status"], r["first_diff"], r["margin"]) for i, r in enumerate(records) if r["status"].isupper()]
        assert not bad, f"after the reload the engine does not follow HF with the new weights: {bad}"
    finally:
        from serve.rl.weights import load_weights_inplace

        engine.load_weights(base_state)
        with torch.no_grad():
            load_weights_inplace(hf_model, base_state)

    assert engine.verify_weights(base_state)["ok"]
    assert engine.parameter_pointers() == pointers
    restored, _ = _greedy(engine, batch, max_new_tokens, eos)
    assert restored == before, "greedy output with the base weights changed after a reload round trip"


# ----------------------------------------------------------------------------
# sleep / wake
# ----------------------------------------------------------------------------
def test_sleep_frees_the_kv_pool_and_wake_restores(engine, base_state, prompts, eos):
    max_new_tokens = 32
    batch = prompts["short"][:2] + prompts["long"][:1]
    kv_bytes = math.prod(engine._kv_shape) * torch.finfo(engine.dtype).bits // 8
    assert kv_bytes == engine.plan.kv_bytes
    slack = int(0.1 * kv_bytes) + 256 * MIB
    baseline, _ = _greedy(engine, batch, max_new_tokens, eos)
    awake_bytes = _allocated()
    pointers = engine.parameter_pointers()

    for cycle in range(2):
        engine.sleep()
        assert engine.is_sleeping
        asleep_bytes = _allocated()
        freed = awake_bytes - asleep_bytes
        print(
            f"\n[sleep {cycle}] allocated {awake_bytes / GIB:.2f} -> {asleep_bytes / GIB:.2f} GiB "
            f"(freed {freed / GIB:.2f} GiB; KV pool {kv_bytes / GIB:.2f} GiB)"
        )
        # the KV pool goes away (plus the CUDA graphs captured against it); the weights stay
        assert freed >= 0.95 * kv_bytes, f"sleep freed {freed / GIB:.2f} GiB, the KV pool alone is {kv_bytes / GIB:.2f} GiB"
        assert freed <= kv_bytes + engine.plan.graph_bytes + 1 * GIB, "sleep freed far more than KV pool + CUDA graphs"
        assert engine.parameter_pointers() == pointers
        assert engine.verify_weights(base_state)["ok"], "the weights must survive sleep()"
        with pytest.raises(RuntimeError, match="asleep"):
            engine.generate(batch, max_new_tokens=4, temperature=0.0, eos_token_ids=eos)
        engine.sleep()  # idempotent: nothing more to free, nothing allocated
        assert engine.is_sleeping and abs(_allocated() - asleep_bytes) <= 16 * MIB

        engine.wake_up()
        assert not engine.is_sleeping
        wake = engine.last_wake_stats
        print(f"[wake {cycle}] {wake}; allocated {_allocated() / GIB:.2f} GiB")
        assert wake["wake_s"] > 0.0
        if engine.cuda_graph:
            assert 0.0 < wake["graph_capture_s"] <= wake["wake_s"]
        else:
            assert wake["graph_capture_s"] == 0.0
        assert abs(_allocated() - awake_bytes) <= slack, (
            f"allocated memory after wake_up ({_allocated() / GIB:.2f} GiB) differs from before sleep "
            f"({awake_bytes / GIB:.2f} GiB) by more than {slack / GIB:.2f} GiB"
        )
        engine.wake_up()  # idempotent
        assert engine.kv_pool.num_free_pages == engine.kv_pool.capacity

        tokens, _ = _greedy(engine, batch, max_new_tokens, eos)
        assert tokens == baseline, f"greedy output changed after sleep/wake cycle {cycle + 1} (cuda_graph={engine.cuda_graph})"
    _assert_torch_defaults_untouched()


# ----------------------------------------------------------------------------
# KV pool and statistics
# ----------------------------------------------------------------------------
def test_kv_pool_is_clean_after_generate(engine, prompts, eos, caplog):
    pool = engine.kv_pool
    # more requests than the engine batches at once, so some wait in the queue
    batch = (prompts["short"] + prompts["long"]) * (2 * MAX_BATCH_SIZE // 7 + 1)
    assert len(batch) > MAX_BATCH_SIZE
    # same workload once before measuring, so one-time allocations do not count as a leak
    engine.generate(batch, max_new_tokens=24, temperature=1.0, eos_token_ids=eos)
    before = _allocated()

    result = engine.generate(batch, max_new_tokens=24, temperature=1.0, eos_token_ids=eos)

    assert not common.check_responses(result, 24, eos)
    assert pool.num_free_pages == pool.capacity, f"{pool.capacity - pool.num_free_pages} GPU KV pages still allocated"
    assert len(pool._free_cpu) == pool._MAX_CPU_CAPACITY, "host offload pages still allocated"
    assert len(set(pool._free_gpu)) == pool.capacity and min(pool._free_gpu) == 0 and max(pool._free_gpu) == pool.capacity - 1
    if hasattr(pool, "_cpu_page_free"):  # per-page "is free" flags of the host pool
        assert pool._cpu_page_free.count(1) == pool._MAX_CPU_CAPACITY, "host pages still flagged as allocated"
    # _check_pool_clean would have repaired a leak and logged it (see the autouse fixture)
    assert not [r for r in caplog.records if "KV pool not clean" in r.getMessage()]
    leaked = _allocated() - before
    print(f"\n[kv pool] {len(batch)} requests, kv_util_peak={result.stats['kv_util_peak']:.3f}, residual {leaked / MIB:.1f} MiB")
    assert leaked <= 64 * MIB, f"{leaked / MIB:.1f} MiB still allocated after generate (per-request buffers not released?)"
    assert 0.0 < result.stats["kv_util_peak"] <= 1.0


def test_stats_are_sane(engine, prompts, eos):
    max_new_tokens = 64
    batch = prompts["short"] + prompts["long"]
    result = engine.generate(batch, max_new_tokens=max_new_tokens, temperature=1.0, eos_token_ids=eos)
    stats = result.stats
    print(f"\n[stats] {stats}")

    assert set(stats) == set(STAT_KEYS)
    assert all(isinstance(v, float) and math.isfinite(v) for v in stats.values()), stats
    assert stats["num_requests"] == len(batch)
    assert stats["num_tokens"] == sum(len(t) for t in result.token_ids)
    assert stats["num_cycles"] > 0
    assert 1.0 <= stats["accept_len_mean"] <= SPEC_STRIDE + 1
    assert 0.0 <= stats["accept_rate"] <= 1.0
    assert 0 <= stats["accepted_draft_tokens"] <= stats["draft_tokens"]
    assert stats["draft_tokens"] <= stats["num_cycles"] * SPEC_STRIDE
    # every verify cycle emits its accepted draft tokens plus one recovered / bonus token
    assert stats["accept_len_mean"] == pytest.approx(stats["accepted_draft_tokens"] / stats["num_cycles"] + 1.0)
    assert stats["accept_rate"] == pytest.approx(stats["accepted_draft_tokens"] / stats["draft_tokens"])
    # tokens come from verify cycles plus one first token per request; cycles may overshoot the limit by < 1 cycle each
    emitted = stats["accepted_draft_tokens"] + stats["num_cycles"] + stats["num_requests"]
    assert stats["num_tokens"] <= emitted <= stats["num_tokens"] + len(batch) * (SPEC_STRIDE + 1)
    assert stats["num_iters"] > 0
    assert 1.0 <= stats["batch_size_mean"] <= min(MAX_BATCH_SIZE, len(batch))
    assert stats["num_offloads"] >= 0
    assert 0.0 < stats["kv_util_peak"] <= 1.0
    assert 0.0 < stats["peak_mem_gb"] < torch.cuda.get_device_properties(torch.device(DEVICE)).total_memory / GIB
    assert stats["gen_s"] > 0.0


# ----------------------------------------------------------------------------
# admission statistics (both admission modes; also in the offload-stress process)
# ----------------------------------------------------------------------------
def _assert_pool_clean(engine, where: str) -> None:
    pool = engine.kv_pool
    assert pool.num_free_pages == pool.capacity, f"{where}: {pool.capacity - pool.num_free_pages} GPU KV pages still allocated"
    assert len(set(pool._free_gpu)) == pool.capacity, f"{where}: duplicate GPU pages in the free list"
    assert len(pool._free_cpu) == pool._MAX_CPU_CAPACITY, (
        f"{where}: {pool._MAX_CPU_CAPACITY - len(pool._free_cpu)} host offload pages still allocated"
    )
    assert len(set(pool._free_cpu)) == pool._MAX_CPU_CAPACITY, f"{where}: duplicate host pages in the free list"
    if hasattr(pool, "_cpu_page_free"):
        assert pool._cpu_page_free.count(1) == pool._MAX_CPU_CAPACITY, f"{where}: host pages still flagged as allocated"


def test_admission_stats(engine, prompts, eos):
    """inflight_peak <= inflight_cap <= max_batch_size, the cap is serve.rl.memory.inflight_cap for
    the workload, host_util_peak in [0, 1]. All prompts have the SAME length: worst_case admission
    commits each request's own worst case, so with mixed lengths more short requests than the cap
    (which is computed for the longest prompt) could be in flight."""
    length = min(len(p) for p in prompts["long"])
    batch = [p[:length] for p in prompts["long"]] * 7  # 21 requests > max_batch_size 16 by default
    max_new_tokens = 64
    result = engine.generate(batch, max_new_tokens=max_new_tokens, temperature=1.0, eos_token_ids=eos)
    stats = result.stats
    assert not common.check_responses(result, max_new_tokens, eos)

    expected = {
        mode: memory.inflight_cap(
            mode, engine.plan, engine.max_batch_size, engine.spec_stride, engine.chunk_size, length, max_new_tokens
        )
        for mode in memory.ADMISSION_MODES
    }
    print(
        f"\n[admission] admission={engine.admission}: inflight_peak={stats['inflight_peak']:.0f} "
        f"inflight_cap={stats['inflight_cap']:.0f} (worst_case {expected['worst_case']}, host_guard "
        f"{expected['host_guard']}, max_batch_size {engine.max_batch_size}), batch_size_mean={stats['batch_size_mean']:.2f}, "
        f"num_offloads={stats['num_offloads']:.0f}, kv_util_peak={stats['kv_util_peak']:.3f}, "
        f"host_util_peak={stats['host_util_peak']:.3f}"
    )
    assert engine.admission == ADMISSION
    assert stats["inflight_cap"] == float(expected[ADMISSION])
    for mode in memory.ADMISSION_MODES:
        assert engine.inflight_cap(length, max_new_tokens, mode) == expected[mode], mode
    assert 1.0 <= stats["inflight_peak"] <= stats["inflight_cap"] <= engine.max_batch_size, stats
    assert stats["inflight_peak"] <= len(batch)
    # requests in flight = running + offloaded, so the mean running batch cannot exceed the peak
    assert stats["batch_size_mean"] <= stats["inflight_peak"] + 1e-9
    assert 0.0 <= stats["host_util_peak"] <= 1.0
    if stats["num_offloads"] == 0:
        assert stats["host_util_peak"] == 0.0, "host pages in use although nothing was offloaded"
    else:
        assert stats["host_util_peak"] > 0.0
    assert 0.0 < stats["kv_util_peak"] <= 1.0
    _assert_pool_clean(engine, "after the admission workload")


# ----------------------------------------------------------------------------
# return_logprobs=False (also in the offload-stress process)
# ----------------------------------------------------------------------------
def test_return_logprobs_false(engine, prompts, eos):
    batch = prompts["short"][:3] + prompts["long"][:1]
    max_new_tokens = 40
    with_lp = engine.generate(batch, max_new_tokens=max_new_tokens, temperature=0.0, eos_token_ids=eos)
    without = engine.generate(batch, max_new_tokens=max_new_tokens, temperature=0.0, eos_token_ids=eos, return_logprobs=False)
    assert with_lp.logprobs is not None and not common.check_responses(with_lp, max_new_tokens, eos, greedy=True)
    assert without.logprobs is None
    assert [list(t) for t in without.token_ids] == [list(t) for t in with_lp.token_ids], (
        "greedy tokens differ between return_logprobs=True and False"
    )
    assert without.finish_reasons == with_lp.finish_reasons
    assert set(without.stats) == set(STAT_KEYS) and without.stats["num_tokens"] == with_lp.stats["num_tokens"]

    # sampling without log-probs: the response contract still holds
    sampled = engine.generate(batch, max_new_tokens=max_new_tokens, temperature=1.0, eos_token_ids=eos, return_logprobs=False)
    assert sampled.logprobs is None and len(sampled.token_ids) == len(sampled.finish_reasons) == len(batch)
    eos_set = set(eos)
    for i, (tokens, reason) in enumerate(zip(sampled.token_ids, sampled.finish_reasons)):
        assert 1 <= len(tokens) <= max_new_tokens, (i, len(tokens))
        assert not any(t in eos_set for t in tokens[:-1]), f"response {i}: tokens after an EOS"
        if tokens[-1] in eos_set:
            assert reason == "stop"
        else:
            assert len(tokens) == max_new_tokens and reason == "length", (i, len(tokens), reason)
    _assert_pool_clean(engine, "after return_logprobs=False")
    _assert_torch_defaults_untouched()


# ----------------------------------------------------------------------------
# offload stress (SPARSESPEC_TEST_OFFLOAD_STRESS=1 only: small GPU KV pool)
# ----------------------------------------------------------------------------
def _per_response_mean_dlogp(a, b) -> list[float]:
    out = []
    for row_a, row_b in zip(a, b):
        deltas = [abs(x - y) for x, y in zip(row_a, row_b) if math.isfinite(x) and math.isfinite(y)]
        out.append(sum(deltas) / len(deltas) if deltas else math.inf)
    return out


def test_offload_stress(engine, hf_model, prompts, eos, monkeypatch):
    """More fixed-length requests than the GPU KV pool holds: the scheduler must offload KV to the
    host pool and restore it, and that must not corrupt anything.

    Every request runs to exactly max_new_tokens: the EOS list is one id >= vocab_size, which can
    never be sampled (the engine does not range-check eos_token_ids: serve.rl.base.normalize_eos_ids
    only makes them ints and RLSpecScheduler only tests membership). The trainer forward still pads
    with the model's real EOS id.

    1. no-offload reference: the same prompts, greedy, a few at a time (fewer than can fill the pool);
    2. all at once, T=1: offloads happen; every log-prob agrees with the HF trainer forward as tightly
       as in the non-offloaded test (and no single response is off);
    3. all at once, greedy: offloads happen; outputs equal the reference (or differ only at a bf16
       near-tie of HF's logits: batch composition changes the bf16 arithmetic);
    4. the first reference chunk again: bitwise identical (nothing stale after offloading);
    5. steps 2 and 3 left some request partly on the GPU and partly on the host, both ways: an offload
       and a restore call each moved only Request.MAX_OFFLOADING_LEN tokens and the rest followed in a
       later call (the multi-step path long training responses take), so 2 and 3 covered it;
    the KV pools are clean after every call."""
    pool = engine.kv_pool
    kv = pool.capacity
    max_new_tokens = STRESS_MAX_NEW_TOKENS
    never = [int(engine.config.vocab_size)]
    batch = [prompts["stress"][i % len(prompts["stress"])] for i in range(STRESS_NUM_REQUESTS)]
    longest, shortest = max(map(len, batch)), min(map(len, batch))
    need_max = memory.request_token_need(longest, max_new_tokens, engine.spec_stride)
    need_min = memory.request_token_need(shortest, max_new_tokens, engine.spec_stride)
    cap = engine.inflight_cap(longest, max_new_tokens)
    admit_tokens = memory.admission_capacity_tokens(engine.plan, engine.max_batch_size, engine.spec_stride, engine.chunk_size)
    # worst_case commits each request's own need; host_guard uses the cap for the longest prompt
    max_in_flight = cap if engine.admission == "host_guard" else min(engine.max_batch_size, admit_tokens // need_min)
    # fewer requests than can fill the GPU pool at all: the reference never offloads
    chunk = memory.gpu_full_min_active(kv, need_max, longest) - 1
    print(
        f"\n[stress] admission={engine.admission} kv_capacity={kv} host_capacity={pool._MAX_CPU_CAPACITY} "
        f"requests={len(batch)} prompt lengths {shortest}-{longest} max_new_tokens={max_new_tokens} "
        f"need/request {need_min}-{need_max} inflight_cap={cap} (at most {max_in_flight} in flight) "
        f"reference chunk={chunk}"
    )
    if need_max > kv:
        pytest.fail(f"one request needs {need_max} KV tokens > the {kv}-token pool: raise SPARSESPEC_TEST_KV_CAPACITY_TOKENS")
    if min(cap, len(batch)) * need_min <= kv:
        pytest.fail(
            f"this workload cannot overflow the GPU pool: {min(cap, len(batch))} requests in flight x {need_min} "
            f"tokens <= {kv}. Lower SPARSESPEC_TEST_KV_CAPACITY_TOKENS or raise SPARSESPEC_TEST_STRESS_MAX_NEW_TOKENS"
        )
    if chunk < 1:
        pytest.fail(f"no request count below the pool size: {kv} tokens, {need_max} per request")

    def check_fixed_length(result, label, greedy):
        problems = common.check_responses(result, max_new_tokens, never, greedy=greedy)
        assert not problems, f"{label}: {problems[:5]}"
        lengths = sorted({len(t) for t in result.token_ids})
        assert lengths == [max_new_tokens], f"{label}: response lengths {lengths}, all must be {max_new_tokens}"
        assert set(result.finish_reasons) == {"length"}, f"{label}: {set(result.finish_reasons)}"

    # 1. no-offload reference, a few requests at a time
    reference: list[list[int]] = []
    first_chunk = None
    for start in range(0, len(batch), chunk):
        part = batch[start : start + chunk]
        result = engine.generate(part, max_new_tokens=max_new_tokens, temperature=0.0, eos_token_ids=never)
        check_fixed_length(result, f"reference chunk at {start}", greedy=True)
        assert result.stats["num_offloads"] == 0, (
            f"reference chunk at {start} ({len(part)} requests) offloaded {result.stats['num_offloads']:.0f} times: "
            "serve.rl.memory.gpu_full_min_active says it cannot fill the pool"
        )
        _assert_pool_clean(engine, f"after reference chunk at {start}")
        reference += [list(t) for t in result.token_ids]
        if first_chunk is None:
            first_chunk = (part, [list(t) for t in result.token_ids])

    # count the offload / restore calls of steps 2 and 3 that leave a request split between the pools
    from serve.request.request import Request

    moves = {"offload_calls": 0, "partial_offloads": 0, "restore_calls": 0, "partial_restores": 0}
    real_offload, real_restore = Request.offload, Request.restore

    def counting_offload(self):
        out = real_offload(self)  # self while GPU pages remain: the scheduler offloads the rest later
        moves["offload_calls"] += 1
        moves["partial_offloads"] += out is not None
        return out

    def counting_restore(self):
        out = real_restore(self)  # self while host pages remain: the scheduler restores the rest later
        moves["restore_calls"] += 1
        moves["partial_restores"] += out is not None
        return out

    monkeypatch.setattr(Request, "offload", counting_offload)
    monkeypatch.setattr(Request, "restore", counting_restore)

    # 2. everything at once at T=1: offloads, and the log-probs still match the trainer forward
    sampled = engine.generate(batch, max_new_tokens=max_new_tokens, temperature=1.0, eos_token_ids=never)
    stats = sampled.stats
    print(f"[stress] T=1 stats: {stats}")
    check_fixed_length(sampled, "T=1 stress", greedy=False)
    _assert_pool_clean(engine, "after the T=1 stress run")
    assert stats["num_offloads"] > 0, f"no KV offload happened (kv_util_peak {stats['kv_util_peak']:.3f}): the test did not test anything"
    assert 0.0 < stats["host_util_peak"] <= 1.0, stats["host_util_peak"]
    assert 0.0 < stats["kv_util_peak"] <= 1.0
    assert 1.0 <= stats["inflight_peak"] <= max_in_flight, (stats["inflight_peak"], max_in_flight)
    assert stats["inflight_cap"] == float(cap)
    if engine.admission == "host_guard":
        assert stats["inflight_peak"] <= stats["inflight_cap"]

    trainer = common.hf_trainer_logprobs(hf_model, batch, sampled.token_ids, temperature=1.0, pad_token_id=eos[-1])
    lp = common.compare_logprobs(sampled.logprobs, trainer)
    per_response = _per_response_mean_dlogp(sampled.logprobs, trainer)
    worst = sorted(range(len(per_response)), key=lambda i: -per_response[i])[:5]
    print(f"[stress] log-probs vs HF trainer forward: {lp}")
    print(f"[stress] worst responses (index, mean |delta log p|): {[(i, round(per_response[i], 4)) for i in worst]}")
    assert lp["num_nonfinite"] == 0 and lp["count"] == len(batch) * max_new_tokens
    assert lp["mean"] <= MAX_MEAN_DLOGP, (
        f"with offloading, mean |delta log p| = {lp['mean']:.4f} > {MAX_MEAN_DLOGP} (SPARSESPEC_TEST_MAX_MEAN_DLOGP); "
        f"median {lp['median']:.4f}, p99 {lp['p99']:.4f}, max {lp['max']:.4f}: offload / restore corrupts KV?"
    )
    assert max(per_response) <= STRESS_MAX_RESPONSE_DLOGP, (
        f"response {worst[0]}: mean |delta log p| {per_response[worst[0]]:.4f} > {STRESS_MAX_RESPONSE_DLOGP} "
        "(SPARSESPEC_TEST_STRESS_MAX_RESPONSE_DLOGP): one request's KV was corrupted by offload / restore?"
    )

    # 3. everything at once, greedy: offloads, and the same tokens as without offloading
    greedy = engine.generate(batch, max_new_tokens=max_new_tokens, temperature=0.0, eos_token_ids=never)
    print(f"[stress] T=0 stats: {greedy.stats}")
    check_fixed_length(greedy, "greedy stress", greedy=True)
    _assert_pool_clean(engine, "after the greedy stress run")
    assert greedy.stats["num_offloads"] > 0, "the greedy stress run did not offload"
    tokens = [list(t) for t in greedy.token_ids]
    records = []
    for i, (prompt, got, want) in enumerate(zip(batch, tokens, reference)):
        if got == want:
            continue
        record = analyse_prompt(hf_model, prompt, got, want, TIE_MARGIN)
        records.append((i, record["status"], record["first_diff"], record["margin"]))
    print(f"[stress] greedy with offloading vs without: {len(batch) - len(records)}/{len(batch)} identical; divergences: {records}")
    bad = [r for r in records if r[1].isupper()]
    assert not bad, (
        "greedy output with offloading diverges from the no-offload output against a clear HF margin "
        f"(> {TIE_MARGIN} logits): (request, status, first diff, margin) = {bad}"
    )

    multi_step = dict(moves)  # steps 2 and 3 only
    print(f"[stress] offload / restore calls of the two stress runs: {multi_step}")

    # 4. nothing stale: the first reference chunk again, bitwise
    part, want = first_chunk
    again = engine.generate(part, max_new_tokens=max_new_tokens, temperature=0.0, eos_token_ids=never)
    assert again.stats["num_offloads"] == 0
    assert [list(t) for t in again.token_ids] == want, "after the offload runs the same small greedy batch changed"
    _assert_pool_clean(engine, "at the end of the stress test")
    _assert_torch_defaults_untouched()

    # 5. coverage, checked last so every correctness check above has run: steps 2 and 3 really took the
    #    multi-step path (a request partly on the GPU, partly on the host)
    assert multi_step["partial_offloads"] > 0 and multi_step["partial_restores"] > 0, (
        f"the stress runs never moved a request in more than one {Request.MAX_OFFLOADING_LEN}-token step each way "
        f"({multi_step}): the multi-step offload / restore path training uses was not exercised. Raise "
        "SPARSESPEC_TEST_STRESS_MAX_NEW_TOKENS (offload victims must hold more than "
        f"{Request.MAX_OFFLOADING_LEN} KV tokens)"
    )
