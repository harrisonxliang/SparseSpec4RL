"""Both admission modes against upstream's real offloading rules, page by page.

``offload_sim.py`` transcribes upstream's ``offloading`` admission policy and
the KV page accounting (see its docstring for the exact upstream lines) and
puts ``RLSpecScheduler``'s guard on top. The adversary chooses every
request's final length. For ``host_guard`` with the cap from
``serve.rl.memory`` the tests assert, on every schedule tried:

* neither ``alloc_page`` / ``allocate_tokens`` (GPU) nor ``alloc_page_cpu``
  (host) runs out, and no other upstream assert fires;
* host pages in use never exceed ``host_guard_host_bound`` of the requests in
  flight (the most since the offload stack was last empty), which never
  exceeds the host pool; each victim obeys the bound's two lemmas (it holds
  at most kv // active tokens, and at least ``gpu_full_min_active``
  requests are active when it is picked);
* requests in flight never exceed the cap (nor max_batch_size);
* every request completes exactly once, at exactly its planned length, with
  its KV cache intact token by token, and its prompt prefilled once (nothing
  dropped, truncated, corrupted or restarted); the loop terminates.

``worst_case`` gets the same safety checks. Without any guard (upstream's
admission alone) the default pools DO run out, so the checks have teeth.

``SPARSESPEC_SIM_TRIALS=<n>`` scales the randomized search up for a deeper
hunt (default: a budget that keeps this module around 30 s).
"""

from __future__ import annotations

import ast
import hashlib
import os
import random
import textwrap

import pytest

from serve.rl import memory
from serve.rl.memory import (
    admission_capacity_tokens,
    gpu_full_min_active,
    host_guard_host_bound,
    host_guard_inflight_cap,
    inflight_cap,
    request_token_need,
)

RANDOM_TRIALS = int(os.environ.get("SPARSESPEC_SIM_TRIALS", "120"))


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def max_final_kv(prompt_len: int, max_new_tokens: int, spec_stride: int) -> int:
    """Largest KV length a request reaches in RLSpecScheduler: it completes at the first verify
    after its KV holds prompt + max_new_tokens - 1 tokens; the last cycle starts below that and adds
    at most spec_stride + 1 (request_token_need keeps 3 more tokens of slack on top)."""
    return prompt_len + max_new_tokens - 2 + spec_stride + 1


def simulate(
    sim,
    plan,
    guard,
    prompt_lens,
    final_kvs,
    max_new_tokens,
    *,
    max_batch_size=64,
    spec_stride=8,
    chunk_size=256,
    sawtooth_seed=None,
    finish_after_restore=False,
    inflight_cap_override=None,
):
    """One SparseSpecEngine.generate call, as the engine sets the scheduler up."""
    lp = max(prompt_lens)
    need = request_token_need(lp, max_new_tokens, spec_stride)
    kv, cpu = plan.kv_capacity_tokens, plan.cpu_capacity_tokens
    # what the engine computes per call (SparseSpecEngine.generate)
    admit_cap = admission_capacity_tokens(plan, max_batch_size, spec_stride, chunk_size)
    cap = inflight_cap(guard or "host_guard", plan, max_batch_size, spec_stride, chunk_size, lp, max_new_tokens)
    if inflight_cap_override is not None:
        cap = inflight_cap_override
    sched = sim.SchedulerSim(
        sim.KVPoolSim(kv, cpu),
        max_batch_size=max_batch_size,
        chunk_size=chunk_size,
        spec_stride=spec_stride,
        guard=guard,
        total_token_capacity=admit_cap,
        inflight_cap=cap,
        host_bound=lambda n: host_guard_host_bound(n, kv, need, lp),
        k_full=gpu_full_min_active(kv, need, lp),
        max_kv_len=need - 1,
        sawtooth_seed=sawtooth_seed,
    )
    requests = sim.make_requests(prompt_lens, final_kvs, max_new_tokens)
    for request in requests:
        request.finish_after_restore = finish_after_restore
    planned = [r.final_kv for r in requests]
    work = sum(final_kvs) + len(final_kvs) * (spec_stride + 2)
    res = sched.run(requests, max_iterations=20 * work + 10_000)

    # --- every request completed once, as planned, intact (the KV tags were checked at completion) ---
    assert sorted(r.rid for r in res.completed) == list(range(len(requests)))
    for request in requests:
        want = request.final_kv if finish_after_restore else planned[request.rid]
        assert request.completed_len == want, (request.rid, request.completed_len, want)
        assert request.prefill_tokens == request.len_input, f"request {request.rid} was prefilled twice"
        assert request.kv_cache_ptr is None  # retired: all its pages are back
    pool = sched.kv_pool
    assert pool.num_free_pages == kv and pool.host_used == 0  # nothing leaked
    assert sched.prefilling_request is None and not sched.offload_requests and not sched.active_requests

    # --- what the real scheduler would publish (RLSpecScheduler.rl.peak_*) is the true peak -------
    if guard is not None:
        assert res.rl_peak_host_used == res.peak_host_used
        assert res.rl_peak_inflight == res.peak_inflight
    res.cap, res.need, res.bound_at_peak = cap, need, host_guard_host_bound(res.peak_inflight, kv, need, lp)
    res.longest_prompt, res.max_batch_size, res.spec_stride, res.chunk_size = lp, max_batch_size, spec_stride, chunk_size
    res.sched = sched
    return res


def assert_host_guard_guarantee(res, plan):
    """peak host pages <= host_guard_host_bound(peak in flight) <= host pool, in flight <= cap."""
    cpu = plan.cpu_capacity_tokens
    assert res.peak_inflight <= res.cap
    assert res.bound_checks == res.iterations  # the bound was checked after every admission step
    assert res.peak_host_used <= res.bound_at_peak
    guard_cap = host_guard_inflight_cap(plan.kv_capacity_tokens, cpu, res.need, res.longest_prompt)
    if res.cap > guard_cap:
        # a cap above host_guard's own (inflight_cap taking worst_case's when that is larger) is only
        # safe if its worst-case demand fits the pools like worst_case admission requires
        admit_cap = admission_capacity_tokens(plan, res.max_batch_size, res.spec_stride, res.chunk_size)
        assert res.cap * res.need <= admit_cap
        return
    if res.cap == 1 and res.bound_at_peak + 1024 > cpu:
        # inflight_cap never returns 0; a lone request cannot overflow a GPU pool it fits in
        assert res.peak_host_used == 0 and res.num_offloads == 0
    else:
        assert res.bound_at_peak <= cpu
        assert res.bound_at_peak == 0 or res.bound_at_peak + 1024 <= cpu  # one offload step to spare


# ---------------------------------------------------------------------------
# the run's configuration: Qwen3-0.6B on the A100, 64 requests of 1024 + 8192 tokens
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def default_plan(tk):
    plan = tk.qwen3_plan()
    assert (plan.kv_capacity_tokens, plan.cpu_capacity_tokens) == (100_715, 149_796)
    return plan


@pytest.fixture(scope="module")
def all_maximal(tk, default_plan):
    q = tk.QWEN3_06B
    final = max_final_kv(q.max_prompt, q.max_response, q.spec_stride)
    runs = {
        guard: simulate(tk.offload_sim, default_plan, guard, [q.max_prompt] * 64, [final] * 64, q.max_response)
        for guard in ("host_guard", "worst_case")
    }
    return runs


def test_all_maximal_host_guard_is_safe_and_uses_the_host_pool(all_maximal, default_plan, capsys):
    res = all_maximal["host_guard"]
    assert res.cap == 45
    assert_host_guard_guarantee(res, default_plan)
    assert res.peak_inflight == 45  # the cap is reached ...
    assert res.num_offloads > 0 and res.peak_host_used > 0.9 * default_plan.cpu_capacity_tokens  # ... and used
    assert res.prefilling_victims == 0
    # how tight the bound is on the all-maximal schedule (printed with -s; measured 137624 / 147625 = 0.932
    # in a host pool of 149796; the same schedule with a larger cap first runs out of host pages at 52)
    ratio = res.peak_host_used / res.bound_at_peak
    with capsys.disabled():
        print(
            f"\n[offload sim] host_guard, 64 x (1024 + 8192): cap {res.cap}, peak in flight {res.peak_inflight}, "
            f"peak host pages {res.peak_host_used} / bound {res.bound_at_peak} = {ratio:.3f} "
            f"(host pool {default_plan.cpu_capacity_tokens}), {res.num_offloads} offloads, {res.iterations} iterations"
        )
    assert 0.9 < ratio <= 1.0


def test_all_maximal_worst_case_is_safe(all_maximal, default_plan):
    res = all_maximal["worst_case"]
    assert res.peak_inflight == 26  # inflight_cap("worst_case", ...) for this workload
    assert res.peak_committed <= admission_capacity_tokens(default_plan, 64, 8, 256)
    assert res.peak_host_used <= default_plan.cpu_capacity_tokens
    assert res.peak_host_used <= res.bound_at_peak  # the bound is about upstream's rules: it holds here too
    # host_guard keeps more requests in flight and finishes the same work in fewer iterations
    assert all_maximal["host_guard"].peak_inflight > res.peak_inflight
    assert all_maximal["host_guard"].iterations < res.iterations


def test_without_a_guard_the_default_host_pool_runs_out(tk, default_plan):
    """Upstream's admission alone (GPU free capacity only) on the same 64 requests: the host
    pool runs out, i.e. what alloc_page_cpu's "Out of memory" assert does on the box."""
    sim = tk.offload_sim
    q = tk.QWEN3_06B
    final = max_final_kv(q.max_prompt, q.max_response, q.spec_stride)
    with pytest.raises(sim.PoolExhausted) as excinfo:
        simulate(sim, default_plan, None, [q.max_prompt] * 64, [final] * 64, q.max_response)
    assert excinfo.value.pool == "host" and "alloc_page_cpu" in str(excinfo.value)


def test_a_cap_well_above_host_guards_does_run_out(tk, default_plan):
    """The cap is not arbitrary: on the all-maximal schedule a cap of 52 exhausts the host pool."""
    sim = tk.offload_sim
    q = tk.QWEN3_06B
    final = max_final_kv(q.max_prompt, q.max_response, q.spec_stride)
    with pytest.raises(sim.PoolExhausted, match="host pool"):
        simulate(
            sim, default_plan, "host_guard", [q.max_prompt] * 64, [final] * 64, q.max_response,
            inflight_cap_override=52,
        )  # fmt: skip


@pytest.mark.parametrize("guard", ["host_guard", "worst_case"])
@pytest.mark.parametrize("seed", [1, 2])
def test_default_pools_mixed_lengths(tk, default_plan, guard, seed):
    rng = random.Random(seed)
    q = tk.QWEN3_06B
    prompts = [rng.randint(100, q.max_prompt) for _ in range(63)] + [q.max_prompt]
    finals = [rng.randint(p, max_final_kv(p, q.max_response, q.spec_stride)) for p in prompts]
    res = simulate(tk.offload_sim, default_plan, guard, prompts, finals, q.max_response, sawtooth_seed=seed)
    if guard == "host_guard":
        assert_host_guard_guarantee(res, default_plan)
    assert res.peak_host_used <= default_plan.cpu_capacity_tokens


@pytest.mark.parametrize("guard", ["host_guard", "worst_case"])
def test_default_pools_bimodal_lengths(tk, default_plan, guard):
    """Most responses short, one in four runs to the limit (a GRPO batch with a few rambling samples)."""
    rng = random.Random(7)
    q = tk.QWEN3_06B
    prompts = [q.max_prompt] * 96
    finals = [
        max_final_kv(p, q.max_response, q.spec_stride) if rng.random() < 0.25 else p + rng.randint(1, 400)
        for p in prompts
    ]
    res = simulate(tk.offload_sim, default_plan, guard, prompts, finals, q.max_response)
    if guard == "host_guard":
        assert_host_guard_guarantee(res, default_plan)
        assert res.num_offloads > 0
    assert res.peak_host_used <= default_plan.cpu_capacity_tokens


@pytest.mark.parametrize("guard", ["host_guard", "worst_case"])
def test_default_pools_requests_finish_right_after_coming_back(tk, default_plan, guard):
    """Adversary: every request that was offloaded ends one token after it is restored, so restores
    are immediately followed by completions and the stack drains and refills as fast as it can."""
    q = tk.QWEN3_06B
    final = max_final_kv(q.max_prompt, q.max_response, q.spec_stride)
    res = simulate(
        tk.offload_sim, default_plan, guard, [q.max_prompt] * 80, [final] * 80, q.max_response,
        finish_after_restore=True,
    )  # fmt: skip
    restored_early = [r for r in res.completed if r.times_offloaded and r.completed_len < final]
    assert restored_early  # the adversary did bite
    if guard == "host_guard":
        assert_host_guard_guarantee(res, default_plan)
    assert res.peak_host_used <= default_plan.cpu_capacity_tokens


@pytest.mark.parametrize(
    "mem_fraction, ratio, mbs, lp, max_new",
    [
        (0.45, 0.0, 8, 4096, 16384),  # no host pool at all
        (0.3, 0.25, 64, 1024, 8192),  # a host pool smaller than one request
        (0.6, 0.5, 128, 16, 8192),
    ],
)
@pytest.mark.parametrize("guard", ["host_guard", "worst_case"])
def test_plans_where_the_two_caps_differ_by_one(tk, guard, mem_fraction, ratio, mbs, lp, max_new):
    """Qwen3 plans on which inflight_cap('host_guard') is one below inflight_cap('worst_case')
    today (see test_admission_math.py): whatever cap the module returns must be safe."""
    plan = tk.qwen3_plan(mem_fraction=mem_fraction, cpu_offload_ratio=ratio, max_batch_size=mbs)
    n = 3 * inflight_cap("worst_case", plan, mbs, 8, 256, lp, max_new)
    res = simulate(
        tk.offload_sim, plan, guard, [lp] * n, [max_final_kv(lp, max_new, 8)] * n, max_new, max_batch_size=mbs
    )
    if guard == "host_guard":
        assert_host_guard_guarantee(res, plan)
    assert res.peak_host_used <= plan.cpu_capacity_tokens


# ---------------------------------------------------------------------------
# randomized search: small pools, every host-pool ratio, up to 256 in flight
# ---------------------------------------------------------------------------
def random_case(rng, tk):
    """A workload the engine accepts (check_workload_fits). Three in four are "pressure" cases: the
    host_guard cap lets enough requests in to fill the GPU pool and more requests are waiting."""
    pressure = rng.random() < 0.75
    while True:
        kv = rng.choice([rng.randint(1_100, 3_000), rng.randint(3_000, 9_000)])
        ratio = rng.choice([0.5, 1.5, 3.0] if pressure else [0.0, 0.5, 1.5, 3.0])
        mbs = rng.choice([1, 2, 4, 8, 16, 32, 64, 128, 256])
        chunk = rng.choice([16, 64, 256])
        stride = rng.choice([2, 4, 8])
        lp = rng.choice([1, rng.randint(1, 32), rng.randint(1, 700)])
        max_new = rng.randint(64, 1_500) if pressure else rng.choice([rng.randint(1, 64), rng.randint(64, 1_500)])
        plan = tk.synthetic_plan(kv, int(kv * ratio))
        try:
            memory.check_workload_fits(
                prompt_lens=[lp], max_new_tokens=max_new, spec_stride=stride, chunk_size=chunk,
                max_batch_size=mbs, plan=plan, max_position_embeddings=None,
            )  # fmt: skip
        except (MemoryError, ValueError):
            continue
        cap = inflight_cap("host_guard", plan, mbs, stride, chunk, lp, max_new)
        k_full = gpu_full_min_active(kv, request_token_need(lp, max_new, stride), lp)
        if pressure and cap < k_full:
            continue
        break
    n = rng.randint(cap, 3 * cap + 2) if pressure else rng.randint(1, 2 * cap + 3)
    prompts = [rng.choice([lp, rng.randint(1, lp)]) for _ in range(n)]
    prompts[rng.randrange(n)] = lp  # the engine's cap uses the longest prompt of the call
    pattern = rng.choice(["max", "max", "uniform", "bimodal", "staircase", "finish_after_restore", "prompt_only"])
    finals = []
    for i, p in enumerate(prompts):
        top = max_final_kv(p, max_new, stride)
        if pattern in ("max", "finish_after_restore"):
            finals.append(top)
        elif pattern == "uniform":
            finals.append(rng.randint(p, top))
        elif pattern == "bimodal":
            finals.append(top if rng.random() < 0.6 else rng.randint(p, min(top, p + 30)))
        elif pattern == "staircase":
            finals.append(max(p, top - (i % 7) * rng.randint(0, 40)))
        else:
            finals.append(p)  # EOS as the first response token
    return dict(
        plan=plan, mbs=mbs, chunk=chunk, stride=stride, max_new=max_new, prompts=prompts, finals=finals,
        pattern=pattern, sawtooth=rng.choice([None, rng.randrange(10**6)]),
    )  # fmt: skip


def describe(case) -> str:
    plan = case["plan"]
    return (
        f"kv={plan.kv_capacity_tokens} host={plan.cpu_capacity_tokens} max_batch_size={case['mbs']} "
        f"chunk={case['chunk']} spec_stride={case['stride']} max_new={case['max_new']} n={len(case['prompts'])} "
        f"longest_prompt={max(case['prompts'])} pattern={case['pattern']} sawtooth={case['sawtooth']}"
    )


@pytest.mark.parametrize("guard", ["host_guard", "worst_case"])
def test_randomized_search(tk, guard):
    sim = tk.offload_sim
    rng = random.Random({"host_guard": 11, "worst_case": 22}[guard])
    with_offloads, prefilling_victims, tightest = 0, 0, 0.0
    for trial in range(RANDOM_TRIALS):
        case = random_case(rng, tk)
        try:
            res = simulate(
                sim, case["plan"], guard, case["prompts"], case["finals"], case["max_new"],
                max_batch_size=case["mbs"], spec_stride=case["stride"], chunk_size=case["chunk"],
                sawtooth_seed=case["sawtooth"], finish_after_restore=case["pattern"] == "finish_after_restore",
            )  # fmt: skip
        except AssertionError as exc:
            raise AssertionError(f"trial {trial}: {describe(case)}\n{type(exc).__name__}: {exc}") from exc
        if guard == "host_guard":
            assert_host_guard_guarantee(res, case["plan"])
        assert res.peak_host_used <= case["plan"].cpu_capacity_tokens
        with_offloads += res.num_offloads > 0
        prefilling_victims += res.prefilling_victims
        if res.bound_at_peak:
            tightest = max(tightest, res.peak_host_used / res.bound_at_peak)
    print(
        f"[offload sim] {guard}: {RANDOM_TRIALS} random schedules, {with_offloads} with offloads, "
        f"{prefilling_victims} prefilling victims, tightest peak host / bound {tightest:.3f}"
    )
    assert with_offloads >= RANDOM_TRIALS // 3  # the search does exercise the host pool
    # a request is admitted only with room to finish its prefill before any offload
    assert prefilling_victims == 0


# Schedules in which, before admission was restricted to scheduler calls that START with an empty
# offload stack, upstream restored the top of the stack and admitted a new request in the same call:
# the room check had been made before the restore took its pages, and the new request became the
# offload victim while still prefilling (one such victim per seed below).
@pytest.mark.parametrize(
    "guard, seed",
    [("host_guard", s) for s in (132, 193, 238, 296)] + [("worst_case", s) for s in (25, 133, 173, 193)],
)
def test_no_request_is_offloaded_while_prefilling(tk, guard, seed):
    rng = random.Random(seed)
    prompts = [rng.randint(380, 620) for _ in range(24)]
    finals = [max_final_kv(p, 512, 8) - rng.randint(0, 8) for p in prompts]
    res = simulate(
        tk.offload_sim, tk.synthetic_plan(4096, 4681), guard, prompts, finals, 512,
        max_batch_size=16, sawtooth_seed=seed,
    )  # fmt: skip
    assert res.num_offloads > 0  # the schedule does go through the offload stack
    assert res.prefilling_victims == 0


def test_the_simulation_catches_a_broken_host_pool(tk):
    """Teeth for the randomized search: the same driver with a too-generous cap on a tiny host pool."""
    sim = tk.offload_sim
    plan = tk.synthetic_plan(3_000, 1_500)
    lp, max_new = 16, 1_000
    cap = inflight_cap("host_guard", plan, 64, 8, 256, lp, max_new)
    final = max_final_kv(lp, max_new, 8)
    simulate(sim, plan, "host_guard", [lp] * 12, [final] * 12, max_new)  # the real cap is fine
    with pytest.raises(sim.PoolExhausted, match="host pool"):
        simulate(sim, plan, "host_guard", [lp] * 12, [final] * 12, max_new, inflight_cap_override=cap + 4)


# ---------------------------------------------------------------------------
# the transcription in offload_sim.py is of TODAY's upstream code
# ---------------------------------------------------------------------------
TRANSCRIBED = {
    # (file, class, member): sha256 of the normalised source (see _normalised)
    ("serve/scheduler/scheduler.py", "BaseScheduler", "admit_requests"):
        "e411e9f3cf9ce30398b1deff263641a257bc930c92d709870daab2b35b086547",
    ("serve/scheduler/scheduler.py", "BaseScheduler", "run"):
        "015ac3823768ac7bbcc97ea9b34a313e9097de8654dc644854f9dd3acc4cd865",
    ("serve/scheduler/spec_scheduler.py", "SpecScheduler", "add_request"):
        "7e8f7fab3796f40848cd96932eda3a42486cbee5c28d2f4caef2de89185c371a",
    ("serve/scheduler/spec_scheduler.py", "SpecScheduler", "prepare_forward_inputs"):
        "d1f1861774db5cdd90857ae99e8460f3b6bf029e1ba17070c90b927466fc70a3",
    ("serve/scheduler/spec_scheduler.py", "SpecScheduler", "verify_outputs"):
        "d52edae602f53044910b30fef413db9133cd33fab2fcdea74a58d91e38d04701",
    ("serve/request/request.py", "Request", "MAX_OFFLOADING_LEN"):
        "62f0fb052431df0dd9b3630d6467c89c3a0e3efbdb830bb27bcaee7342f1f8f2",
    ("serve/request/request.py", "Request", "retire"):
        "30ff5ac01fba8488ca50c0920c98ddc02ada67c05572dd1cc1f966d8c046fe63",
    ("serve/request/request.py", "Request", "offload"):
        "08cf7b0b10f25acfeb79d9e8c9eb7dcc2ac4705728483c1ceb873830ef42c8ae",
    ("serve/request/request.py", "Request", "restore"):
        "1a857f0bdd595afdf6f0c69b0ba1160949ba0bde3e82c79feb7b4ebe970d48df",
    ("serve/request/request.py", "Request", "prefilled"):
        "59ff95b6ab02d2a329764a701b7be66f6a104c7d60a2b10bd6ba07b42527feb0",
    ("serve/request/kv_cache_ptr/base.py", "FullCachePtr", "__del__"):
        "b7dc4a540a220aee07ef4db37a86cb175b705f9689dac79044719e31155e4ca9",
    ("serve/request/kv_cache_ptr/base.py", "FullCachePtr", "allocate_tokens"):
        "29bcc83cc2afc7d24a5e3da3257b1e41029d12e0fd8a58d96cc45a78604277d2",
    ("serve/request/kv_cache_ptr/base.py", "FullCachePtr", "retract_tokens"):
        "d2b1db6b217720e097fd66d6867cfb50c3d20bd46938baae569a38d4f318b045",
    ("serve/request/kv_cache_ptr/base.py", "FullCachePtr", "offload"):
        "ad2831bcec2178cea1513a526d548d96b4bfb69892c5b1fa1dd1531f21c76b9d",
    ("serve/request/kv_cache_ptr/base.py", "FullCachePtr", "restore"):
        "6560143696e78cba7528f6427a0fed2984251a4ae95e066d79f1ef585e26931a",
    ("serve/request/kv_cache_ptr/streaming.py", "StreamingCachePtr", "allocate_tokens"):
        "a1e0741dd0adcad06543c5ef19c903b01c3bbd73726785a2cb6b510f2f24f4c1",
    ("serve/request/kv_cache_ptr/streaming.py", "StreamingCachePtr", "retract_tokens"):
        "b89fdf80dbff0518b14187897912d4a199b0f54b2fd5ea5b07867098d0e5d006",
    ("serve/request/kv_cache_ptr/streaming.py", "StreamingCachePtr", "offload"):
        "051a03e4b08dd17e154aac2816150927842cb5c099f9e359cfae69daa12be84d",
    ("serve/request/kv_cache_ptr/streaming.py", "StreamingCachePtr", "restore"):
        "90cfcbbbf28319ea3baa1ca789f47953050eebcbdd1b2cad6918f3cb2c89477b",
    ("serve/request/kv_cache_ptr/pillar.py", "PillarCachePtr", "from_kv_cache"):
        "905623ad435ff0a3777e60ba06a981d473bc192ea6e6b5efe6151808d4ce64a9",
    ("serve/model/model.py", "KVPool", "num_free_pages"):
        "129de4e88930da446886d03a40cd18bfe820a05892e374f4e8e53bc62685892a",
    ("serve/model/model.py", "KVPool", "alloc_page"):
        "65fc38a6b8863f34ea4199a7f96f8787d49c5a3576b0dacf20acf02bc021c852",
    ("serve/model/model.py", "KVPool", "free_page"):
        "0224a2b3c32782ec8c46db7d4a23cf3e04fc063589695ef95d0b98cf9b62fd38",
    ("serve/model/model.py", "KVPool", "alloc_page_cpu"):
        "89d93e45b0da7ee5c1e6c46662f7e564d7b308f6bfb2d856111f47956a9b0e86",
    ("serve/model/model.py", "KVPool", "free_page_cpu"):
        "c02374b817506002cb3b27d7b74d2aa330ce6f9dd25103b5e91fa1bad1d4c59f",
    ("serve/model/model.py", "KVPool", "offload"):
        "728cf13d259705eb58cd03d9d1be970f5c941e43bcf2ce6848d43e8b3bc58691",
    ("serve/model/model.py", "KVPool", "restore"):
        "4cdf918710349d7a50e3043eea94609d954f69cf6c1e5a63d23cc66900ffc1f0",
    ("serve/rl/scheduler.py", "RLSpecScheduler", "_rl_may_admit"):
        "fd14d69bc9ff20ee84c65ed013786d8520b4c741e7e2899ec774d0101d375395",
    ("serve/rl/scheduler.py", "RLSpecScheduler", "admit_requests"):
        "2dfef0c82480d9276c6dd70c28f5fde881737e6cefb3fb14352fbccc9e6b1f0c",
    ("serve/rl/scheduler.py", "RLSpecScheduler", "add_request"):
        "90fad8fd48c9f380fd45e0130ba23fc1f254a6152d394d8300a237c6860d50c8",
}


def _member_source(tk, rel: str, class_name: str, member: str) -> str:
    text = (tk.REPO_ROOT / rel).read_text()
    tree = ast.parse(text)
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name)
    for node in cls.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == member:
            return ast.get_source_segment(text, node)
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == member for t in node.targets):
            return ast.get_source_segment(text, node)
    raise LookupError(f"{rel}: {class_name}.{member} not found")


def _normalised(source: str) -> str:
    """Indentation-independent and blind to blank lines and whole-line comments (text, not an AST
    dump, so the hash is the same on every Python version)."""
    lines = textwrap.dedent(source).splitlines()
    return "\n".join(line.rstrip() for line in lines if line.strip() and not line.strip().startswith("#"))


def _digest(tk, key) -> str:
    return hashlib.sha256(_normalised(_member_source(tk, *key)).encode()).hexdigest()


@pytest.mark.parametrize("key", sorted(TRANSCRIBED), ids=lambda k: f"{k[1]}.{k[2]}")
def test_transcribed_upstream_code_is_unchanged(tk, key):
    actual = _digest(tk, key)
    assert actual == TRANSCRIBED[key], (
        f"{key[0]}: {key[1]}.{key[2]} changed since serve/rl/tests/offload_sim.py transcribed it.\n"
        "The offload simulation (and with it the host_guard guarantee it validates) must be re-validated "
        "against the new code: update the transcription, rerun test_offload_sim.py (also with "
        f"SPARSESPEC_SIM_TRIALS=2000), then record the new hash {actual}"
    )


def test_offload_step_constants_agree(tk):
    """One offload / restore step is Request.MAX_OFFLOADING_LEN tokens everywhere it is assumed."""
    import inspect

    source = _member_source(tk, "serve/request/request.py", "Request", "MAX_OFFLOADING_LEN")
    upstream = ast.literal_eval(ast.parse(source).body[0].value)
    assert upstream == tk.offload_sim.MAX_OFFLOADING_LEN == memory.OFFLOAD_CHUNK_TOKENS == 1024
    margin = inspect.signature(memory.host_guard_inflight_cap).parameters["margin"].default
    assert margin == upstream
