"""serve/rl/memory.py: the admission-mode math (pure Python).

``worst_case`` admits a request only while the worst-case KV demand of
everything in flight fits GPU pool + host pool; ``host_guard`` caps the
number of requests in flight so that upstream's own offload rules can never
fill the host pool (``host_guard_host_bound``). Whether the bound really
holds against upstream's policy is test_offload_sim.py's job; this module
checks the arithmetic and the numbers for the run's configuration.
"""

from __future__ import annotations

import itertools

import pytest

from serve.rl import memory
from serve.rl.memory import (
    ADMISSION_MODES,
    GIB,
    MIN_TRANSIENT_BYTES,
    admission_capacity_tokens,
    gpu_full_min_active,
    host_guard_host_bound,
    host_guard_inflight_cap,
    inflight_cap,
    request_token_need,
    transient_bytes_worst_case,
)

STRIDE, CHUNK = 8, 256


def brute_k_full(kv: int, need: int, longest_prompt: int) -> int:
    """Fewest k with k * need + longest_prompt > kv (k active requests of fewer than
    ``need`` tokens each, plus one prompt still owed, can overflow the GPU pool)."""
    k = 1
    while k * need + longest_prompt <= kv:
        k += 1
    return k


def brute_bound(n: int, kv: int, need: int, longest_prompt: int) -> int:
    k_full = brute_k_full(kv, need, longest_prompt)
    total = 0
    for k in range(1, n + 1):
        if k >= k_full:
            total += min(kv // k, need)
    return total


def brute_cap(kv: int, cpu: int, need: int, longest_prompt: int, margin: int = 1024) -> int:
    """Largest n in [1, 10_000] whose bound plus one offload step fits the host pool
    (n below k_full never offloads, so it always fits)."""
    k_full = brute_k_full(kv, need, longest_prompt)
    best = 1
    for n in range(1, 10_001):
        if n < k_full or brute_bound(n, kv, need, longest_prompt) + margin <= cpu:
            best = n
        else:
            break
    return best


POOLS = [
    # kv, need, longest prompt
    (100_715, request_token_need(1024, 8192, STRIDE), 1024),  # the run: Qwen3-0.6B, 0.45 of the A100
    (100_715, request_token_need(16, 512, STRIDE), 16),
    (50_000, request_token_need(2048, 2048, STRIDE), 2048),
    (9_226, 9_226, 1024),  # the pool holds exactly one worst-case request
    (3_000, 700, 1),
    (1_200, 1_100, 600),
]


# ---------------------------------------------------------------------------
# gpu_full_min_active
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kv, need, lp", POOLS)
def test_gpu_full_min_active_is_the_fewest_requests_that_can_fill_the_pool(kv, need, lp):
    k_full = gpu_full_min_active(kv, need, lp)
    assert k_full == brute_k_full(kv, need, lp) >= 1
    # k_full requests can overflow the pool, one fewer cannot
    assert k_full * need + lp > kv
    assert (k_full - 1) * need + lp <= kv or k_full == 1


def test_gpu_full_min_active_numbers():
    need = request_token_need(1024, 8192, STRIDE)
    assert need == 9_226
    assert gpu_full_min_active(100_715, need, 1024) == 11  # 10 * 9226 + 1024 = 93_284 <= 100_715 < 102_510
    assert gpu_full_min_active(10, 100, 50) == 1  # a pool smaller than one request: any request can fill it
    assert gpu_full_min_active(0, 1, 0) == 1


# ---------------------------------------------------------------------------
# host_guard_host_bound
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kv, need, lp", POOLS)
def test_host_bound_matches_the_brute_force_sum_and_is_monotone(kv, need, lp):
    k_full = gpu_full_min_active(kv, need, lp)
    previous = 0
    for n in range(0, 300):
        bound = host_guard_host_bound(n, kv, need, lp)
        assert bound == brute_bound(n, kv, need, lp), n
        assert bound >= previous, n  # monotone in the number in flight
        if n < k_full:
            assert bound == 0, n  # the GPU pool cannot fill: nothing is ever offloaded
        else:
            # each extra request adds at most one request's worth, and at most its share of the GPU pool
            assert bound - previous == min(kv // n, need), n
        previous = bound


def test_host_bound_grows_harmonically_not_linearly():
    """The point of host_guard: the n-th request in flight adds kv // n, not need."""
    kv, lp = 100_715, 1024
    need = request_token_need(lp, 8192, STRIDE)
    assert host_guard_host_bound(10, kv, need, lp) == 0
    assert host_guard_host_bound(11, kv, need, lp) == kv // 11 == 9_155
    assert host_guard_host_bound(45, kv, need, lp) == sum(kv // k for k in range(11, 46)) == 147_625
    assert host_guard_host_bound(64, kv, need, lp) == 182_760
    assert host_guard_host_bound(64, kv, need, lp) < (64 - 10) * need  # what a linear bound would charge


# ---------------------------------------------------------------------------
# host_guard_inflight_cap
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kv, need, lp", POOLS)
@pytest.mark.parametrize("ratio", [0.0, 0.05, 0.5, 1.0, 1.5, 3.0])
def test_host_guard_cap_is_the_largest_n_whose_bound_fits(kv, need, lp, ratio):
    cpu = int(kv * ratio)
    cap = host_guard_inflight_cap(kv, cpu, need, lp)
    assert cap == brute_cap(kv, cpu, need, lp)
    assert cap >= 1
    k_full = gpu_full_min_active(kv, need, lp)
    if cap >= k_full:  # the host pool is used at all: the cap's bound fits with one offload step to spare
        clamped = cap == 1 and host_guard_host_bound(1, kv, need, lp) + 1024 > cpu
        # the only exception is the clamp to one request: a lone request never offloads, because the
        # engine refuses workloads with need > kv (check_workload_fits) and a request holds < need tokens
        assert host_guard_host_bound(cap, kv, need, lp) + 1024 <= cpu or (clamped and need <= kv)
    # one more request in flight would not fit any more
    assert cap + 1 < k_full or host_guard_host_bound(cap + 1, kv, need, lp) + 1024 > cpu


@pytest.mark.parametrize("kv, need, lp", POOLS)
def test_host_guard_cap_without_a_host_pool_is_what_the_gpu_pool_holds_outright(kv, need, lp):
    cap = host_guard_inflight_cap(kv, 0, need, lp)
    outright = (kv - lp) // need  # worst-case requests that fit next to one owed prompt: no offload possible
    assert cap == max(1, outright) == max(1, gpu_full_min_active(kv, need, lp) - 1)
    # less than one offload step of host pool is the same as none
    assert host_guard_inflight_cap(kv, 1023, need, lp) == cap


@pytest.mark.parametrize("kv, need, lp", POOLS)
def test_host_guard_cap_is_monotone_in_the_host_pool_and_never_below_one(kv, need, lp):
    previous = 0
    for cpu in sorted({0, 1, 1023, 1024, 1025, *range(0, 4 * kv + 1, max(1, kv // 37))}):
        cap = host_guard_inflight_cap(kv, cpu, need, lp)
        assert cap >= max(1, previous), cpu
        previous = cap


def test_host_guard_cap_terminates_for_an_unbounded_host_pool():
    # beyond kv requests the n-th adds kv // n == 0 host pages: the loop stops there
    assert host_guard_inflight_cap(100_715, 10**15, 9_226, 1024) == 100_715
    assert host_guard_inflight_cap(3_000, 10**15, 700, 1) == 3_000


def test_host_guard_cap_margin_is_one_offload_step():
    kv, lp = 100_715, 1024
    need = request_token_need(lp, 8192, STRIDE)
    bound_45 = host_guard_host_bound(45, kv, need, lp)
    assert host_guard_inflight_cap(kv, bound_45 + 1024, need, lp) == 45
    assert host_guard_inflight_cap(kv, bound_45 + 1023, need, lp) == 44
    assert host_guard_inflight_cap(kv, bound_45, need, lp, margin=0) == 45


# ---------------------------------------------------------------------------
# inflight_cap for the run's plan (Qwen3-0.6B, 0.45 of the A100, max_batch_size 64)
# ---------------------------------------------------------------------------
def test_qwen3_plan_numbers(tk):
    plan = tk.qwen3_plan()
    assert plan.weights_bytes == 2 * 596_049_920  # same model as test_memory.py
    assert (plan.kv_capacity_tokens, plan.cpu_capacity_tokens) == (100_715, 149_796)
    q = tk.QWEN3_06B
    caps = {
        mode: inflight_cap(mode, plan, q.max_batch_size, STRIDE, CHUNK, q.max_prompt, q.max_response)
        for mode in ADMISSION_MODES
    }
    assert ADMISSION_MODES == ("worst_case", "host_guard")
    assert 26 <= caps["worst_case"] <= 29, caps
    assert 45 <= caps["host_guard"] <= 49, caps
    assert caps == {"worst_case": 26, "host_guard": 45}  # exact values today, for the write-up
    # worst_case is the old formula: committed worst-case tokens <= GPU + host pool minus the reserve
    need = request_token_need(q.max_prompt, q.max_response, STRIDE)
    assert caps["worst_case"] == admission_capacity_tokens(plan, q.max_batch_size, STRIDE, CHUNK) // need
    assert caps["host_guard"] == host_guard_inflight_cap(
        plan.kv_capacity_tokens, plan.cpu_capacity_tokens, need, q.max_prompt
    )


@pytest.mark.parametrize("ratio, worst, guard", [(0.0, 10, 10), (1.0, 18, 21), (1.5, 26, 45), (3.0, 43, 64)])
def test_qwen3_caps_by_host_pool(tk, ratio, worst, guard):
    plan = tk.qwen3_plan(cpu_offload_ratio=ratio)
    q = tk.QWEN3_06B
    args = (plan, q.max_batch_size, STRIDE, CHUNK, q.max_prompt, q.max_response)
    assert (inflight_cap("worst_case", *args), inflight_cap("host_guard", *args)) == (worst, guard)


def test_inflight_cap_is_clamped_to_max_batch_size_and_one(tk):
    plan = tk.qwen3_plan(cpu_offload_ratio=6.0)
    for mode in ADMISSION_MODES:
        assert inflight_cap(mode, plan, 64, STRIDE, CHUNK, 1024, 8192) == 64
        assert inflight_cap(mode, plan, 7, STRIDE, CHUNK, 1024, 8192) == 7
    tiny = tk.synthetic_plan(10_000, 0)
    for mode in ADMISSION_MODES:
        assert inflight_cap(mode, tiny, 64, STRIDE, CHUNK, 1024, 8192) == 1  # not even one fits: never 0


@pytest.mark.parametrize("mode", ["", "worst-case", "hostguard", "HOST_GUARD", None])
def test_inflight_cap_rejects_unknown_modes(tk, mode):
    with pytest.raises(ValueError, match="admission"):
        inflight_cap(mode, tk.qwen3_plan(), 64, STRIDE, CHUNK, 1024, 8192)


def test_host_guard_admits_at_least_as_many_as_worst_case_on_every_plan(tk):
    """host_guard is offered as the mode that admits more. Sweep the dimensions a run can
    change: memory share, host pool ratio, max_batch_size, longest prompt, response length."""
    checked, below = 0, []
    for mem_fraction, ratio, mbs, lp, max_new in itertools.product(
        (0.3, 0.45, 0.6, 0.9),
        (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 6.0),
        (1, 8, 32, 64, 128, 256),
        (16, 256, 1024, 2048, 4096),
        (16, 512, 2048, 8192, 16384),
    ):
        try:
            plan = tk.qwen3_plan(mem_fraction=mem_fraction, cpu_offload_ratio=ratio, max_batch_size=mbs)
            memory.check_workload_fits(
                prompt_lens=[lp], max_new_tokens=max_new, spec_stride=STRIDE, chunk_size=CHUNK,
                max_batch_size=mbs, plan=plan, max_position_embeddings=None,
            )  # fmt: skip
        except (MemoryError, ValueError):
            continue  # the engine refuses this workload outright
        checked += 1
        args = (plan, mbs, STRIDE, CHUNK, lp, max_new)
        worst, guard = inflight_cap("worst_case", *args), inflight_cap("host_guard", *args)
        if guard < worst:
            below.append(
                f"mem_fraction={mem_fraction} cpu_offload_ratio={ratio} max_batch_size={mbs} prompt={lp} "
                f"max_new={max_new}: kv={plan.kv_capacity_tokens} host={plan.cpu_capacity_tokens} "
                f"worst_case={worst} host_guard={guard}"
            )
    assert checked > 3000
    assert not below, (
        f"host_guard admits fewer requests than worst_case on {len(below)} of {checked} plans "
        "(both caps are safe; inflight_cap('host_guard', ...) could take the larger of the two):\n  "
        + "\n  ".join(below[:12])
    )


# ---------------------------------------------------------------------------
# transient_bytes_worst_case and the automatic transient reserve
# ---------------------------------------------------------------------------
def test_transient_worst_case_formula():
    vocab, dtype = 151_936, 2
    for mbs in (1, 8, 64, 65, 128, 256):
        rows = max(mbs * (STRIDE + 1) + CHUNK, 640)
        sampling = rows * vocab * (dtype + 12)  # logits + noise + new probs + previous probs (fp32)
        verify = vocab * (rows * 8 + mbs * STRIDE * 8 + mbs * 4)
        assert transient_bytes_worst_case(mbs, STRIDE, CHUNK, vocab, dtype) == max(sampling, verify), mbs
    # a CUDA-graph pass is padded to its captured batch size: never fewer than 640 rows
    assert transient_bytes_worst_case(1, STRIDE, CHUNK, vocab, dtype) == transient_bytes_worst_case(
        40, STRIDE, CHUNK, vocab, dtype
    )


def test_auto_transient_reserve_keeps_two_gib_at_64_and_grows_beyond(tk):
    default = tk.qwen3_plan()
    assert default.transient_bytes == MIN_TRANSIENT_BYTES == 2 * GIB  # the plan STATUS.md quotes is unchanged
    assert 1.1 * transient_bytes_worst_case(64, STRIDE, CHUNK, 151_936, 2) < 2 * GIB
    assert tk.qwen3_plan(transient_reserve_bytes=2 * GIB).kv_capacity_tokens == default.kv_capacity_tokens

    previous = default.transient_bytes
    for mbs in (96, 128, 192, 256):
        plan = tk.qwen3_plan(max_batch_size=mbs)
        expected = int(1.1 * transient_bytes_worst_case(mbs, STRIDE, CHUNK, 151_936, 2))
        assert plan.transient_bytes == max(2 * GIB, expected) == expected > previous, mbs
        assert plan.planned_bytes <= plan.budget_bytes
        # the larger reserve comes out of the KV pool
        fixed = tk.qwen3_plan(max_batch_size=mbs, transient_reserve_bytes=2 * GIB)
        assert plan.kv_capacity_tokens < fixed.kv_capacity_tokens, mbs
        previous = plan.transient_bytes
    assert tk.qwen3_plan(max_batch_size=256).transient_bytes > 5 * GIB


def test_auto_transient_reserve_follows_the_scheduler_geometry(tk):
    base = tk.qwen3_plan(max_batch_size=128)
    assert tk.qwen3_plan(max_batch_size=128, spec_stride=4).transient_bytes < base.transient_bytes
    assert tk.qwen3_plan(max_batch_size=128, chunk_size=2048).transient_bytes > base.transient_bytes


@pytest.mark.parametrize("explicit", [GIB, 2 * GIB, 3 * GIB, 7 * GIB])
def test_explicit_transient_reserve_is_honoured(tk, explicit):
    for mbs in (64, 256):
        plan = tk.qwen3_plan(max_batch_size=mbs, transient_reserve_bytes=explicit)
        assert plan.transient_bytes == explicit, mbs  # also below the automatic value: the user decides
