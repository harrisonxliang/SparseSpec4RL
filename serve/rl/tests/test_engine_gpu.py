"""GPU tests for ``serve.rl.engine.SparseSpecEngine`` (run on the A100 box).

    PYTHONPATH=. python -m pytest serve/rl/tests/test_engine_gpu.py -v -s --log-cli-level=INFO

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
from serve.rl.base import STAT_KEYS  # noqa: E402

MODEL = os.environ.get("SPARSESPEC_TEST_MODEL", "Qwen/Qwen3-0.6B")
MEM_FRACTION = float(os.environ.get("SPARSESPEC_TEST_MEM_FRACTION", "0.25"))
MAX_BATCH_SIZE = int(os.environ.get("SPARSESPEC_TEST_MAX_BATCH_SIZE", "16"))
DEVICE = os.environ.get("SPARSESPEC_TEST_DEVICE", "cuda:0")
CUDA_GRAPH = os.environ.get("SPARSESPEC_TEST_CUDA_GRAPH", "1") != "0"
NUM_MIN_BUDGET = int(os.environ.get("SPARSESPEC_TEST_NUM_MIN_BUDGET", "128"))
TIE_MARGIN = float(os.environ.get("SPARSESPEC_TEST_TIE_MARGIN", "1.0"))
MAX_MEAN_DLOGP = float(os.environ.get("SPARSESPEC_TEST_MAX_MEAN_DLOGP", "0.05"))

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
    print(f"\n[prompts] short: {[len(p) for p in short]} tokens, long: {[len(p) for p in long]} tokens")
    return {"short": short, "long": long}


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
    )
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
