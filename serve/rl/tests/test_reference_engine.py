"""serve/rl/reference_engine.py on CPU with the tiny Qwen3 injected via ``model=``.

Every expectation is computed independently of the engine: plain HF forwards
of ONE unpadded sequence without a KV cache (``tk.hf_greedy`` /
``tk.teacher_forced_logprobs``), while the engine runs left-padded batches
with a cache.
"""

from __future__ import annotations

import inspect
import math
import random

import numpy as np
import pytest
import torch

from serve.rl.base import STAT_KEYS, GenerationResult, LoadReport, RolloutEngine
from serve.rl.reference_engine import ReferenceEngine

MAX_NEW = 12
MIXED_LENGTHS = [5, 19, 2, 11, 7, 19, 3, 14, 9, 1]  # unsorted, with a tie and a 1-token prompt
# a wide EOS set (10% of the vocabulary) so that sampled responses stop early often
WIDE_EOS = list(range(40, 56))


@pytest.fixture
def prompts(tk):
    return tk.make_prompts(MIXED_LENGTHS, seed=1)


@pytest.fixture
def engine(tk, tiny_model):
    return tk.make_reference_engine(tiny_model)  # float32, seed 0, max_batch_size 8


def check_contract(result: GenerationResult, num_prompts: int, max_new_tokens: int, eos_ids, vocab: int) -> None:
    eos = set(eos_ids)
    assert len(result.token_ids) == len(result.logprobs) == len(result.finish_reasons) == num_prompts
    for tokens, logprobs, reason in zip(result.token_ids, result.logprobs, result.finish_reasons):
        assert isinstance(tokens, list) and isinstance(logprobs, list)
        assert all(type(t) is int for t in tokens) and all(type(lp) is float for lp in logprobs)
        assert 1 <= len(tokens) == len(logprobs) <= max_new_tokens
        assert all(0 <= t < vocab for t in tokens)
        assert all(math.isfinite(lp) and lp <= 0.0 for lp in logprobs)
        assert reason in ("stop", "length")
        if reason == "stop":  # ends with the FIRST EOS, which is kept
            assert tokens[-1] in eos
            assert not any(t in eos for t in tokens[:-1])
        else:  # exactly max_new_tokens tokens, none of them an EOS
            assert len(tokens) == max_new_tokens
            assert not any(t in eos for t in tokens)


def generate_until_both_finish_reasons(engine, prompts, max_new_tokens, temperature, eos_ids, attempts=6):
    """Sampled batches, repeated (the stream continues) until one contains both
    a stopped and a length-limited response. Returns all results generated."""
    results = []
    for _ in range(attempts):
        results.append(engine.generate(prompts, max_new_tokens, temperature=temperature, eos_token_ids=eos_ids))
        if {"stop", "length"} <= set(results[-1].finish_reasons):
            return results
    raise AssertionError("no batch with both finish reasons; the EOS set is too narrow or too wide for this model")


def unused_token(tk, rows) -> int:
    """A token id that occurs in none of the given continuations."""
    used = {t for row in rows for t in row}
    return next(t for t in range(tk.FIRST_NORMAL_ID, tk.VOCAB) if t not in used)


def pick_eos_for_greedy(greedy_rows) -> int:
    """A token that makes some greedy rows stop early and leaves others at full length."""
    for token in greedy_rows[0][2:]:
        hit = [token in row for row in greedy_rows]
        if any(hit) and not all(hit):
            return token
    raise AssertionError("could not find a discriminating EOS token; change the prompt seed")


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------
def test_is_a_rollout_engine_with_the_shared_signature(tk, tiny_model):
    engine = tk.make_reference_engine(tiny_model)
    assert isinstance(engine, RolloutEngine) and engine.name == "reference"
    names = list(inspect.signature(ReferenceEngine.__init__).parameters)
    assert names[1:10] == [
        "model_path",
        "mem_fraction",
        "max_batch_size",
        "spec_stride",
        "budget_ratio",
        "num_min_budget",
        "kv_cache",
        "cuda_graph",
        "seed",
    ]
    params = inspect.signature(ReferenceEngine.__init__).parameters
    assert params["dtype"].kind is inspect.Parameter.KEYWORD_ONLY and params["dtype"].default == "bfloat16"
    assert params["device"].kind is inspect.Parameter.KEYWORD_ONLY
    assert inspect.signature(ReferenceEngine.generate) == inspect.signature(RolloutEngine.generate)

    # positional construction as in the brief; SparseSpec-only options are accepted and ignored
    engine = ReferenceEngine(
        "unused-path", 0.45, 4, 8, 0.05, 128, "pillar", True, 3,
        dtype="float32", device="cpu", model=tk.make_tiny_qwen3(0), chunk_size=256, kv_capacity_tokens=None,
    )  # fmt: skip
    assert (engine.max_batch_size, engine.seed, engine.mem_fraction) == (4, 3, 0.45)
    assert not engine.is_sleeping  # engines start awake


def test_engine_freezes_and_casts_its_model(tk, tiny_model):
    tiny_model.train()
    engine = tk.make_reference_engine(tiny_model, dtype="bfloat16")
    assert engine.model is not tiny_model  # the fixture hands the engine a copy
    assert engine.model.training is False
    assert all(p.dtype == torch.bfloat16 and not p.requires_grad for p in engine.model.parameters())
    assert engine.dtype == torch.bfloat16 and engine.device == torch.device("cpu")
    # the caller's module is untouched by building an engine around a copy
    assert all(p.dtype == torch.float32 and p.requires_grad for p in tiny_model.parameters())


# ---------------------------------------------------------------------------
# response contract
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_response_contract_when_sampling(tk, engine, prompts, temperature):
    results = generate_until_both_finish_reasons(engine, prompts, MAX_NEW, temperature, WIDE_EOS)
    for result in results:
        check_contract(result, len(prompts), MAX_NEW, WIDE_EOS, tk.VOCAB)
    reasons = set(results[-1].finish_reasons)
    assert reasons == {"stop", "length"}


def test_response_contract_with_the_models_own_eos_list(tk, engine, prompts):
    """No eos_token_ids argument: the generation config's LIST [im_end, endoftext] applies."""
    for _ in range(3):
        result = engine.generate(prompts, 24, temperature=1.0)
        check_contract(result, len(prompts), 24, tk.EOS_IDS, tk.VOCAB)


@pytest.mark.parametrize("max_new_tokens", [1, 2])
def test_tiny_token_budgets(tk, engine, prompts, max_new_tokens):
    result = engine.generate(prompts, max_new_tokens, temperature=1.0, eos_token_ids=WIDE_EOS)
    check_contract(result, len(prompts), max_new_tokens, WIDE_EOS, tk.VOCAB)
    if max_new_tokens == 1:
        assert all(len(t) == 1 for t in result.token_ids)
        for tokens, reason in zip(result.token_ids, result.finish_reasons):
            assert reason == ("stop" if tokens[0] in WIDE_EOS else "length")


def test_bf16_engine_obeys_the_contract(tk, tiny_model, prompts):
    engine = tk.make_reference_engine(tiny_model, dtype="bfloat16")
    result = engine.generate(prompts, MAX_NEW, temperature=1.0, eos_token_ids=WIDE_EOS)
    check_contract(result, len(prompts), MAX_NEW, WIDE_EOS, tk.VOCAB)


# ---------------------------------------------------------------------------
# log-probs: independent teacher-forced recomputation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_logprobs_equal_teacher_forced_recomputation(tk, tiny_model, engine, prompts, temperature):
    results = generate_until_both_finish_reasons(engine, prompts, MAX_NEW, temperature, WIDE_EOS)
    other_temperature = 1.0 if temperature != 1.0 else 0.7
    worst, worst_other, checked = 0.0, 0.0, 0
    for result in results:
        for prompt, tokens, logprobs in zip(prompts, result.token_ids, result.logprobs):
            expected = tk.teacher_forced_logprobs(tiny_model, prompt, tokens, temperature)
            assert len(expected) == len(logprobs)
            worst = max(worst, max(abs(a - b) for a, b in zip(logprobs, expected)))
            wrong = tk.teacher_forced_logprobs(tiny_model, prompt, tokens, other_temperature)
            worst_other = max(worst_other, max(abs(a - b) for a, b in zip(logprobs, wrong)))
            checked += len(tokens)
    assert checked > len(prompts)
    assert worst < 1e-4, f"max |engine - teacher forced| = {worst}"
    # the comparison has teeth: scaling by the wrong temperature is far outside the tolerance
    assert worst_other > 1e-2


def test_logprobs_are_of_the_sampled_distribution(tk, tiny_model, engine):
    """exp(logprob) is the probability the token was drawn with: over many
    draws of the first token the empirical frequencies match it."""
    prompt = tk.make_prompts([6], seed=2)[0]
    with torch.no_grad():
        logits = tiny_model(input_ids=torch.tensor([prompt])).logits[0, -1].double()
    probs = torch.softmax(logits, dim=-1)

    draws = 4000
    counts = torch.zeros(tk.VOCAB, dtype=torch.float64)
    engine = tk.make_reference_engine(tiny_model, max_batch_size=500)
    for _ in range(draws // 500):
        result = engine.generate([prompt] * 500, 1, temperature=1.0)
        for tokens, logprobs in zip(result.token_ids, result.logprobs):
            counts[tokens[0]] += 1
            assert abs(logprobs[0] - math.log(float(probs[tokens[0]]))) < 1e-4
    # total variation distance between empirical and model distribution; for 4000
    # draws over 160 outcomes it concentrates around 0.08 (sampling noise only)
    tv = 0.5 * (counts / draws - probs).abs().sum().item()
    assert tv < 0.15, tv
    uniform_tv = 0.5 * (torch.full_like(probs, 1 / tk.VOCAB) - probs).abs().sum().item()
    assert uniform_tv > 2 * tv  # ... and it is the model's distribution, not just "some" distribution


# ---------------------------------------------------------------------------
# temperature 0
# ---------------------------------------------------------------------------
def test_temperature_zero_is_the_hf_greedy_continuation(tk, tiny_model, engine, prompts):
    greedy = [tk.hf_greedy(tiny_model, p, MAX_NEW) for p in prompts]
    eos = [pick_eos_for_greedy(greedy), tk.PAD_ID]
    expected = [tk.cut_at_eos(row, eos) for row in greedy]
    assert any(len(row) < MAX_NEW for row in expected) and any(len(row) == MAX_NEW for row in expected)

    result = engine.generate(prompts, MAX_NEW, temperature=0.0, eos_token_ids=eos)
    assert result.token_ids == expected
    assert result.finish_reasons == ["stop" if row[-1] in eos else "length" for row in expected]
    assert all(lp == 0.0 for row in result.logprobs for lp in row)  # greedy target is one-hot
    assert [len(row) for row in result.logprobs] == [len(row) for row in expected]
    check_contract(result, len(prompts), MAX_NEW, eos, tk.VOCAB)

    # deterministic: a second call (and temperature given as int 0, as verl passes it) is identical
    again = engine.generate(prompts, MAX_NEW, temperature=0, eos_token_ids=eos)
    assert again.token_ids == expected and again.finish_reasons == result.finish_reasons


def test_greedy_does_not_consume_the_sampling_stream(tk, tiny_model, prompts):
    a = tk.make_reference_engine(tiny_model, seed=5)
    b = tk.make_reference_engine(tiny_model, seed=5)
    a.generate(prompts, MAX_NEW, temperature=0.0)  # e.g. a validation pass between training steps
    ra = a.generate(prompts, MAX_NEW, temperature=1.0)
    rb = b.generate(prompts, MAX_NEW, temperature=1.0)
    assert ra.token_ids == rb.token_ids


# ---------------------------------------------------------------------------
# caller order, padding, position ids
# ---------------------------------------------------------------------------
def test_results_in_caller_order_and_independent_of_batching(tk, tiny_model, prompts):
    greedy = [tk.hf_greedy(tiny_model, p, MAX_NEW) for p in prompts]
    eos = [pick_eos_for_greedy(greedy)]
    expected = [tk.cut_at_eos(row, eos) for row in greedy]
    assert len({tuple(row) for row in expected}) > 1  # rows are distinguishable, so order is observable

    outputs = {}
    for max_batch_size in (1, 3, 8, 64):
        engine = tk.make_reference_engine(tiny_model, max_batch_size=max_batch_size)
        result = engine.generate(prompts, MAX_NEW, temperature=0.0, eos_token_ids=eos)
        outputs[max_batch_size] = result
        # caller order, whatever the engine's internal length-sorted micro-batches were
        assert result.token_ids == expected, f"max_batch_size={max_batch_size}"
        assert result.stats["num_requests"] == len(prompts)
    # 1 = no padding at all; 8 / 64 = left-padded batches with explicit position ids
    assert outputs[1].token_ids == outputs[8].token_ids == outputs[3].token_ids == outputs[64].token_ids
    assert outputs[1].finish_reasons == outputs[8].finish_reasons
    assert outputs[1].stats["batch_size_mean"] == 1.0
    assert outputs[8].stats["batch_size_mean"] > 1.0
    assert outputs[1].stats["num_iters"] > outputs[8].stats["num_iters"] > outputs[64].stats["num_iters"]


def test_permuting_the_prompts_permutes_the_results(tk, tiny_model, prompts):
    engine = tk.make_reference_engine(tiny_model, max_batch_size=4)
    base = engine.generate(prompts, MAX_NEW, temperature=0.0)
    order = [7, 2, 9, 0, 5, 3, 8, 1, 6, 4]
    shuffled = engine.generate([prompts[i] for i in order], MAX_NEW, temperature=0.0)
    assert shuffled.token_ids == [base.token_ids[i] for i in order]
    assert shuffled.finish_reasons == [base.finish_reasons[i] for i in order]


def test_padded_batches_give_unpadded_logprobs(tk, tiny_model, prompts):
    """Sampling in ONE left-padded batch of very different prompt lengths:
    each row's log-probs still match its own unpadded teacher-forced pass."""
    engine = tk.make_reference_engine(tiny_model, max_batch_size=64)
    result = engine.generate(prompts, MAX_NEW, temperature=1.0, eos_token_ids=WIDE_EOS)
    assert result.stats["batch_size_mean"] == len(prompts)
    for prompt, tokens, logprobs in zip(prompts, result.token_ids, result.logprobs):
        expected = tk.teacher_forced_logprobs(tiny_model, prompt, tokens, 1.0)
        assert max(abs(a - b) for a, b in zip(logprobs, expected)) < 1e-4, len(prompt)


def test_prompt_inputs_are_accepted_as_any_int_sequences(tk, tiny_model, prompts):
    engine = tk.make_reference_engine(tiny_model)
    as_lists = engine.generate(prompts, 4, temperature=0.0)
    as_arrays = engine.generate([np.array(p) for p in prompts], 4, temperature=0.0)
    as_tuples = engine.generate(tuple(tuple(p) for p in prompts), np.int64(4), temperature=0.0)
    assert as_lists.token_ids == as_arrays.token_ids == as_tuples.token_ids
    assert all(len(p) == n for p, n in zip(prompts, MIXED_LENGTHS))  # inputs are not modified


# ---------------------------------------------------------------------------
# RNG stream
# ---------------------------------------------------------------------------
def test_seeded_stream_continues_and_is_reproducible(tk, tiny_model, prompts):
    first = tk.make_reference_engine(tiny_model, seed=0)
    r1 = first.generate(prompts, MAX_NEW, temperature=1.0)
    r2 = first.generate(prompts, MAX_NEW, temperature=1.0)
    assert r1.token_ids != r2.token_ids  # the stream continues: no reseeding per call

    fresh = tk.make_reference_engine(tiny_model, seed=0)
    s1 = fresh.generate(prompts, MAX_NEW, temperature=1.0)
    s2 = fresh.generate(prompts, MAX_NEW, temperature=1.0)
    assert s1.token_ids == r1.token_ids and s1.logprobs == r1.logprobs
    assert s1.finish_reasons == r1.finish_reasons
    assert s2.token_ids == r2.token_ids and s2.logprobs == r2.logprobs  # the continuation too

    other = tk.make_reference_engine(tiny_model, seed=1)
    assert other.generate(prompts, MAX_NEW, temperature=1.0).token_ids != r1.token_ids


def test_generate_leaves_the_global_rngs_alone(tk, tiny_model, prompts):
    engine = tk.make_reference_engine(tiny_model, seed=0)
    torch.manual_seed(77)
    random.seed(77)
    np.random.seed(77)
    torch_state = torch.get_rng_state()
    python_state = random.getstate()
    numpy_state = np.random.get_state()

    r1 = engine.generate(prompts, MAX_NEW, temperature=1.0)
    engine.generate(prompts, MAX_NEW, temperature=0.0)

    assert torch.equal(torch.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    after = np.random.get_state()
    assert after[0] == numpy_state[0] and np.array_equal(after[1], numpy_state[1]) and after[2:] == numpy_state[2:]

    # ... and the samples do not depend on the global seed either
    torch.manual_seed(999)
    other = tk.make_reference_engine(tiny_model, seed=0)
    torch.manual_seed(31337)
    assert other.generate(prompts, MAX_NEW, temperature=1.0).token_ids == r1.token_ids


def test_constructing_an_engine_leaves_the_global_rng_alone(tk, tiny_model):
    torch.manual_seed(5)
    state = torch.get_rng_state()
    tk.make_reference_engine(tiny_model, seed=123)
    assert torch.equal(torch.get_rng_state(), state)


# ---------------------------------------------------------------------------
# sleep / wake
# ---------------------------------------------------------------------------
def test_sleep_wake_state_machine(tk, engine, prompts):
    assert engine.is_sleeping is False
    assert engine.last_wake_stats == {"wake_s": 0.0, "graph_capture_s": 0.0}
    engine.wake_up()  # idempotent while awake
    assert engine.is_sleeping is False

    engine.sleep()
    assert engine.is_sleeping is True
    with pytest.raises(RuntimeError, match="asleep"):
        engine.generate(prompts, 2)
    engine.sleep()  # idempotent
    assert engine.is_sleeping is True

    engine.wake_up()
    assert engine.is_sleeping is False
    stats = engine.last_wake_stats
    assert set(stats) == {"wake_s", "graph_capture_s"}
    assert all(type(v) is float and v >= 0.0 for v in stats.values())
    stats["wake_s"] = -1.0  # a copy: callers cannot corrupt the engine's record
    assert engine.last_wake_stats["wake_s"] >= 0.0
    engine.wake_up()
    result = engine.generate(prompts, 2)
    assert len(result.token_ids) == len(prompts)


def test_sleep_does_not_disturb_weights_or_the_stream(tk, tiny_model, prompts):
    napping = tk.make_reference_engine(tiny_model, seed=4)
    steady = tk.make_reference_engine(tiny_model, seed=4)
    a1 = napping.generate(prompts, MAX_NEW, temperature=1.0)
    napping.sleep()
    napping.wake_up()
    a2 = napping.generate(prompts, MAX_NEW, temperature=1.0)
    b1 = steady.generate(prompts, MAX_NEW, temperature=1.0)
    b2 = steady.generate(prompts, MAX_NEW, temperature=1.0)
    assert (a1.token_ids, a2.token_ids) == (b1.token_ids, b2.token_ids)


def test_close(tk, engine, prompts):
    engine.close()
    assert engine.is_sleeping is True
    engine.close()  # idempotent
    for call in (
        lambda: engine.generate(prompts, 2),
        engine.sleep,
        engine.wake_up,
        lambda: engine.load_weights({}),
    ):
        with pytest.raises(RuntimeError, match="closed"):
            call()


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------
def test_load_weights_switches_the_greedy_output(tk, tiny_model, tiny_model_b, prompts):
    greedy_a = [tk.hf_greedy(tiny_model, p, MAX_NEW) for p in prompts]
    greedy_b = [tk.hf_greedy(tiny_model_b, p, MAX_NEW) for p in prompts]
    assert all(a != b for a, b in zip(greedy_a, greedy_b))  # the two weight sets are clearly different models
    never = [unused_token(tk, greedy_a + greedy_b)]  # an EOS id neither model produces: full-length responses

    engine = tk.make_reference_engine(tiny_model)
    pointers = engine.parameter_pointers()
    state_a = {k: v.clone() for k, v in tiny_model.state_dict().items()}
    state_b = {k: v.clone() for k, v in tiny_model_b.state_dict().items()}
    assert engine.generate(prompts, MAX_NEW, temperature=0.0, eos_token_ids=never).token_ids == greedy_a
    assert engine.verify_weights(state_a)["ok"] is True
    assert engine.verify_weights(state_b)["ok"] is False

    report = engine.load_weights(state_b)
    assert isinstance(report, LoadReport)
    assert report.num_params == len(list(tiny_model.parameters()))
    assert report.num_sources == len(state_b)
    assert engine.generate(prompts, MAX_NEW, temperature=0.0, eos_token_ids=never).token_ids == greedy_b

    assert engine.parameter_pointers() == pointers  # in place
    assert pointers["lm_head.weight"] == pointers["embed_tokens.weight"]  # still tied
    assert set(pointers) == {(k[len("model.") :] if k.startswith("model.") else k) for k in state_a}
    verdict = engine.verify_weights(state_b)
    assert verdict["ok"] is True and verdict["tied_ok"] is True and verdict["num_mismatched"] == 0
    stale = engine.verify_weights(state_a)
    assert stale["ok"] is False and stale["num_mismatched"] == stale["num_params_checked"]

    # and back again, with the names an FSDP-wrapped actor produces, while asleep
    engine.sleep()
    engine.load_weights({"_fsdp_wrapped_module." + k: v for k, v in state_a.items()})
    engine.wake_up()
    assert engine.generate(prompts, MAX_NEW, temperature=0.0, eos_token_ids=never).token_ids == greedy_a
    assert engine.parameter_pointers() == pointers


def test_load_weights_does_not_alias_the_trainers_tensors(tk, tiny_model, tiny_model_b, prompts):
    engine = tk.make_reference_engine(tiny_model)
    engine.load_weights(tiny_model_b.state_dict())
    before = engine.generate(prompts, 6, temperature=0.0).token_ids
    with torch.no_grad():
        for param in tiny_model_b.parameters():  # the trainer takes an optimizer step
            param.add_(torch.randn(param.shape, generator=torch.Generator().manual_seed(0)) * 0.2)
    assert engine.generate(prompts, 6, temperature=0.0).token_ids == before
    assert engine.verify_weights(tiny_model_b.state_dict())["ok"] is False  # stale until the next sync


def test_failed_load_keeps_the_old_weights(tk, tiny_model, tiny_model_b, prompts):
    engine = tk.make_reference_engine(tiny_model)
    before = engine.generate(prompts, 6, temperature=0.0).token_ids
    bad = dict(tiny_model_b.state_dict())
    del bad["model.layers.1.mlp.down_proj.weight"]
    with pytest.raises(KeyError):
        engine.load_weights(bad)
    bad = dict(tiny_model_b.state_dict())
    bad["model.norm.weight"] = torch.ones(tk.HIDDEN + 1)
    with pytest.raises(ValueError):
        engine.load_weights(bad)
    assert engine.verify_weights(tiny_model.state_dict())["ok"] is True
    assert engine.generate(prompts, 6, temperature=0.0).token_ids == before


def test_bf16_engine_holds_the_rounded_weights(tk, tiny_model, tiny_model_b):
    engine = tk.make_reference_engine(tiny_model, dtype="bfloat16")
    state_b = tiny_model_b.state_dict()  # float32, as the actor's
    engine.load_weights(state_b)
    for name, param in engine.model.named_parameters(remove_duplicate=False):
        assert param.dtype == torch.bfloat16
        assert tk.bit_equal(param, state_b[name].to(torch.bfloat16)), name
    assert engine.verify_weights(state_b)["ok"] is True


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [{"top_p": 0.9}, {"top_k": 50}, {"top_p": 0.95, "top_k": 20}, {"top_k": 1}])
def test_top_p_top_k_are_rejected_not_ignored(engine, prompts, kwargs):
    with pytest.raises(NotImplementedError):
        engine.generate(prompts, 4, temperature=1.0, **kwargs)


@pytest.mark.parametrize(
    "kwargs", [{}, {"top_p": 1.0, "top_k": -1}, {"top_p": 1, "top_k": 0}, {"top_p": None, "top_k": None}]
)
def test_disabled_top_p_top_k_are_accepted(engine, prompts, kwargs):
    assert len(engine.generate(prompts, 2, temperature=1.0, **kwargs).token_ids) == len(prompts)


def test_rejected_calls_do_not_advance_the_stream(tk, tiny_model, prompts):
    a = tk.make_reference_engine(tiny_model, seed=2)
    b = tk.make_reference_engine(tiny_model, seed=2)
    for bad in (
        lambda: a.generate(prompts, 4, top_p=0.5),
        lambda: a.generate([], 4),
        lambda: a.generate(prompts, 0),
        lambda: a.generate(prompts, 4, temperature=-1.0),
    ):
        with pytest.raises((NotImplementedError, ValueError)):
            bad()
    assert a.generate(prompts, 4).token_ids == b.generate(prompts, 4).token_ids


def test_invalid_prompts_and_budgets(tk, engine, prompts):
    with pytest.raises(ValueError, match="at least one prompt"):
        engine.generate([], 4)
    with pytest.raises(ValueError, match="prompt 1 is empty"):
        engine.generate([prompts[0], []], 4)
    with pytest.raises(ValueError, match="max_new_tokens"):
        engine.generate(prompts, 0)
    with pytest.raises(ValueError, match="temperature"):
        engine.generate(prompts, 4, temperature=-0.5)
    with pytest.raises(ValueError, match="prompt 2 contains a token id outside"):
        engine.generate([prompts[0], prompts[1], [8, tk.VOCAB]], 4)
    with pytest.raises(ValueError, match="prompt 0 contains a token id outside"):
        engine.generate([[-1, 9]], 4)


# ---------------------------------------------------------------------------
# EOS sources
# ---------------------------------------------------------------------------
def test_eos_defaults_overrides_and_forms(tk, tiny_model):
    prompt = tk.make_prompts([9], seed=3)
    greedy = tk.hf_greedy(tiny_model, prompt[0], MAX_NEW)
    firsts = [i for i, t in enumerate(greedy) if t not in greedy[:i]]  # where each distinct token first appears
    assert len(firsts) >= 4, "greedy continuation too repetitive: pick another prompt seed"
    early, late = greedy[firsts[1]], greedy[firsts[3]]
    cut_early, cut_late = greedy[: firsts[1] + 1], greedy[: firsts[3] + 1]
    by_default = tk.cut_at_eos(greedy, tk.EOS_IDS)
    assert len({tuple(cut_early), tuple(cut_late), tuple(by_default)}) == 3  # three distinguishable outcomes

    def greedy_tokens(engine, **kwargs):
        return engine.generate(prompt, MAX_NEW, temperature=0.0, **kwargs).token_ids[0]

    # 1) default: generation_config.eos_token_id (a list for Qwen3) and config.eos_token_id
    assert greedy_tokens(tk.make_reference_engine(tiny_model)) == by_default

    model = tk.make_tiny_qwen3(0)
    model.generation_config.eos_token_id = [late, tk.PAD_ID]
    model.config.eos_token_id = None
    assert greedy_tokens(tk.make_reference_engine(model)) == tk.cut_at_eos(greedy, [late, tk.PAD_ID]) == cut_late
    model.config.eos_token_id = early  # the union of both sources applies
    engine = tk.make_reference_engine(model)
    result = engine.generate(prompt, MAX_NEW, temperature=0.0)
    assert result.token_ids == [cut_early] and result.finish_reasons == ["stop"]

    # 2) the constructor argument replaces the model's ids
    engine = tk.make_reference_engine(model, eos_token_ids=[late])
    assert greedy_tokens(engine) == cut_late

    # 3) the per-call argument wins over both; int, list, tuple and None entries are fine
    for form in (early, [early], (early, late), [None, early]):
        assert greedy_tokens(engine, eos_token_ids=form) == cut_early
    # ... and does not stick
    assert greedy_tokens(engine) == cut_late
    # None means "use the engine's default", exactly like not passing it
    assert greedy_tokens(engine, eos_token_ids=None) == cut_late


def test_no_eos_anywhere_is_an_error(tk):
    model = tk.make_tiny_qwen3(0)
    model.generation_config.eos_token_id = None
    model.config.eos_token_id = None
    engine = tk.make_reference_engine(model)
    with pytest.raises(ValueError, match="EOS"):
        engine.generate([[9, 10]], 4)
    assert len(engine.generate([[9, 10]], 4, eos_token_ids=[tk.IM_END_ID]).token_ids[0]) >= 1


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------
def test_stats_have_exactly_the_shared_keys(tk, engine, prompts):
    result = engine.generate(prompts, MAX_NEW, temperature=1.0, eos_token_ids=WIDE_EOS)
    stats = result.stats
    assert set(stats) == set(STAT_KEYS) and len(stats) == len(STAT_KEYS)
    assert all(type(v) is float for v in stats.values()), {k: type(v) for k, v in stats.items()}

    # no speculation in the reference engine: everything about acceptance / KV offload is zero
    for key in ("num_cycles", "draft_tokens", "accepted_draft_tokens", "accept_len_mean", "accept_rate",
                "num_offloads", "kv_util_peak"):  # fmt: skip
        assert stats[key] == 0.0, key
    assert stats["peak_mem_gb"] == 0.0  # CPU

    assert stats["num_requests"] == len(prompts)
    assert stats["num_tokens"] == sum(len(t) for t in result.token_ids)
    # 10 prompts, max_batch_size 8 -> two micro-batches (8 + 2), each at most MAX_NEW forwards
    assert 2 <= stats["num_iters"] <= 2 * MAX_NEW
    assert 2.0 <= stats["batch_size_mean"] <= 8.0
    assert stats["gen_s"] > 0.0

    greedy = engine.generate(prompts[:3], 5, temperature=0.0, eos_token_ids=[tk.VOCAB - 1])
    assert greedy.stats["num_requests"] == 3.0 and greedy.stats["num_tokens"] == 15.0
    assert greedy.stats["num_iters"] == 5.0 and greedy.stats["batch_size_mean"] == 3.0
    assert result.stats["num_requests"] == len(prompts)  # an earlier result is not rewritten


def test_early_stop_ends_the_batch_early(tk, tiny_model):
    """When every row has produced an EOS the engine stops running forwards."""
    prompts = tk.make_prompts([6, 9, 4], seed=7)
    greedy = [tk.hf_greedy(tiny_model, p, MAX_NEW) for p in prompts]
    eos = [row[1] for row in greedy]  # every row stops at its second token at the latest
    engine = tk.make_reference_engine(tiny_model)
    result = engine.generate(prompts, MAX_NEW, temperature=0.0, eos_token_ids=eos)
    assert result.token_ids == [tk.cut_at_eos(row, eos) for row in greedy]
    assert result.finish_reasons == ["stop"] * 3
    assert result.stats["num_iters"] <= 2.0
