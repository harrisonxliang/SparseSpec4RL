"""serve/rl/memory.py: the engine's memory plan and the fail-fast workload checks."""

from __future__ import annotations

import ast
import dataclasses

import pytest

from serve.rl import memory
from serve.rl.memory import (
    GIB,
    EngineLimits,
    check_prob_offsets,
    check_workload_fits,
    host_pool_tokens,
    kv_token_bytes,
    max_rows_per_pass,
    per_request_bytes,
    pinned_block_bytes,
    plan_memory,
    request_token_need,
)

# ---- Qwen/Qwen3-0.6B (config.json) on the A100-40GB box ---------------------
A100_TOTAL_BYTES = int(42.5e9)
MEM_FRACTION = 0.45  # baseline: actor_rollout_ref.rollout.gpu_memory_utilization
NUM_LAYERS, HIDDEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM = 28, 1024, 16, 8, 128
INTERMEDIATE, VOCAB = 3072, 151_936
BF16 = 2

# baseline workload (baseline/config/grpo_base.sh) and the treatment config
MAX_PROMPT, MAX_RESPONSE = 1024, 8192
SPEC_STRIDE, CHUNK_SIZE, MAX_BATCH = 8, 256, 64
QWEN3_MAX_POSITIONS = 40_960


def qwen3_06b_weight_bytes() -> int:
    per_layer = (
        HIDDEN * NUM_HEADS * HEAD_DIM  # q_proj
        + 2 * HIDDEN * NUM_KV_HEADS * HEAD_DIM  # k_proj, v_proj
        + NUM_HEADS * HEAD_DIM * HIDDEN  # o_proj
        + 3 * HIDDEN * INTERMEDIATE  # gate, up, down
        + 2 * HIDDEN  # input / post-attention RMSNorm
        + 2 * HEAD_DIM  # q_norm, k_norm
    )
    params = VOCAB * HIDDEN + NUM_LAYERS * per_layer + HIDDEN  # tied lm_head counted once
    assert params == 596_049_920  # the "0.6B" of Qwen3-0.6B
    return params * BF16


def qwen3_plan(**overrides):
    kwargs = dict(
        total_bytes=A100_TOTAL_BYTES,
        mem_fraction=MEM_FRACTION,
        weights_bytes=qwen3_06b_weight_bytes(),
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        dtype_bytes=BF16,
        max_batch_size=MAX_BATCH,
        cuda_graph=True,
    )
    kwargs.update(overrides)
    return plan_memory(**kwargs)


def component_sum(plan) -> int:
    return (
        plan.weights_bytes
        + plan.fixed_bytes
        + plan.graph_bytes
        + plan.request_bytes
        + plan.transient_bytes
        + plan.safety_bytes
        + plan.kv_bytes
    )


def admission_cap(plan, max_batch_size=MAX_BATCH, spec_stride=SPEC_STRIDE, chunk_size=CHUNK_SIZE) -> int:
    """Total tokens the scheduler may commit (GPU + host pool minus any reserve the module keeps)."""
    fn = getattr(memory, "admission_capacity_tokens", None)
    return plan.total_capacity_tokens if fn is None else fn(plan, max_batch_size, spec_stride, chunk_size)


def fits(plan, **overrides):
    kwargs = dict(
        prompt_lens=[MAX_PROMPT],
        max_new_tokens=MAX_RESPONSE,
        spec_stride=SPEC_STRIDE,
        chunk_size=CHUNK_SIZE,
        max_batch_size=MAX_BATCH,
        plan=plan,
        max_position_embeddings=QWEN3_MAX_POSITIONS,
    )
    kwargs.update(overrides)
    return check_workload_fits(**kwargs)


# ---------------------------------------------------------------------------
# plan_memory: Qwen3-0.6B at 0.45 of 42.5e9 bytes
# ---------------------------------------------------------------------------
def test_qwen3_plan_respects_the_budget():
    plan = qwen3_plan()

    assert plan.total_bytes == A100_TOTAL_BYTES
    assert plan.budget_bytes == int(A100_TOTAL_BYTES * MEM_FRACTION) == 19_125_000_000
    assert plan.weights_bytes == qwen3_06b_weight_bytes() == 1_192_099_840

    # the whole point: everything the engine holds fits in its share
    assert plan.planned_bytes <= plan.budget_bytes
    assert plan.planned_bytes == component_sum(plan)
    assert plan.kv_capacity_tokens > 0

    # the KV pool gets all that is left (not clamped at this size), to the token
    assert plan.notes == []
    assert plan.capacity_overridden is False
    assert 0 <= plan.budget_bytes - plan.planned_bytes < plan.kv_token_bytes

    assert plan.kv_token_bytes == NUM_LAYERS * NUM_KV_HEADS * HEAD_DIM * BF16 * 2 == 114_688
    assert plan.kv_bytes == plan.kv_capacity_tokens * plan.kv_token_bytes
    assert plan.request_bytes == plan.per_request_bytes * MAX_BATCH
    assert plan.transient_bytes == 2 * GIB
    assert plan.safety_bytes == GIB // 2
    assert plan.graph_bytes > 0 and plan.fixed_bytes > 0

    # host offload pool: asked for 1.5x the GPU pool (as serve/model/model.py KVPool), then sized to the
    # power-of-two block torch's pinned allocator really locks. 1.5x is 16.1 GiB, which would lock 32 GiB,
    # so the pool is shrunk to fill 16 GiB exactly.
    assert plan.cpu_requested_tokens == int(plan.kv_capacity_tokens * 1.5)
    assert plan.cpu_requested_tokens * plan.kv_token_bytes > 2**34  # the request is just above 16 GiB
    assert plan.cpu_capacity_tokens == 2**34 // plan.kv_token_bytes == 149_796
    assert plan.cpu_capacity_tokens <= plan.cpu_requested_tokens
    assert plan.cpu_pinned_bytes == pinned_block_bytes(plan.cpu_capacity_tokens * plan.kv_token_bytes) == 2**34
    assert plan.total_capacity_tokens == plan.kv_capacity_tokens + plan.cpu_capacity_tokens
    assert "asked for" in plan.describe() and str(plan.cpu_requested_tokens) in plan.describe()

    # every size is a plain int (they go into torch.empty / log lines)
    for f in dataclasses.fields(plan):
        value = getattr(plan, f.name)
        if f.name not in ("capacity_overridden", "notes"):
            assert type(value) is int, f.name


def test_qwen3_plan_holds_the_baseline_workload():
    plan = qwen3_plan()
    need = request_token_need(MAX_PROMPT, MAX_RESPONSE, SPEC_STRIDE)
    assert plan.kv_capacity_tokens >= need  # at least one worst-case request on the GPU
    assert fits(plan) is None
    assert fits(plan, prompt_lens=[17, MAX_PROMPT, 300]) is None  # the longest prompt decides


@pytest.mark.parametrize("mem_fraction", [0.3, 0.45, 0.6, 0.9])
@pytest.mark.parametrize("max_batch_size", [1, 16, 64, 128])
@pytest.mark.parametrize("cuda_graph", [True, False])
def test_plan_never_exceeds_the_share(mem_fraction, max_batch_size, cuda_graph):
    plan = qwen3_plan(mem_fraction=mem_fraction, max_batch_size=max_batch_size, cuda_graph=cuda_graph)
    assert plan.kv_capacity_tokens > 0
    assert plan.planned_bytes == component_sum(plan) <= plan.budget_bytes
    assert plan.budget_bytes == int(A100_TOTAL_BYTES * mem_fraction)
    if not plan.notes:  # not clamped: the pool takes everything that is left
        assert plan.budget_bytes - plan.planned_bytes < plan.kv_token_bytes
    if not cuda_graph:
        assert plan.graph_bytes == 0


def test_plan_monotonicity():
    base = qwen3_plan()
    assert qwen3_plan(max_batch_size=128).kv_capacity_tokens < base.kv_capacity_tokens
    assert qwen3_plan(max_batch_size=16).kv_capacity_tokens > base.kv_capacity_tokens
    assert qwen3_plan(cuda_graph=False).kv_capacity_tokens > base.kv_capacity_tokens
    assert qwen3_plan(mem_fraction=0.5).kv_capacity_tokens > base.kv_capacity_tokens
    assert qwen3_plan(transient_reserve_bytes=4 * GIB).kv_capacity_tokens < base.kv_capacity_tokens
    assert qwen3_plan(safety_bytes=0).kv_capacity_tokens > base.kv_capacity_tokens
    assert qwen3_plan(cpu_offload_ratio=0.0).cpu_capacity_tokens == 0
    assert qwen3_plan(cpu_offload_ratio=0.0).cpu_pinned_bytes == 0
    # the pool never exceeds the request and never shrinks when more is asked for
    previous = 0
    for ratio in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
        plan = qwen3_plan(cpu_offload_ratio=ratio)
        assert plan.cpu_requested_tokens == int(base.kv_capacity_tokens * ratio)
        assert previous <= plan.cpu_capacity_tokens <= plan.cpu_requested_tokens
        previous = plan.cpu_capacity_tokens
        # what is locked is a power of two, and at most 1/15 more than what the pool uses
        used = plan.cpu_capacity_tokens * plan.kv_token_bytes
        assert plan.cpu_pinned_bytes == pinned_block_bytes(used)
        assert plan.cpu_pinned_bytes & (plan.cpu_pinned_bytes - 1) == 0
        assert plan.cpu_pinned_bytes - used <= plan.cpu_pinned_bytes // 16 + plan.kv_token_bytes


def test_capacity_override_below_the_plan():
    auto = qwen3_plan()
    wanted = auto.kv_capacity_tokens // 2
    plan = qwen3_plan(kv_capacity_tokens=wanted)
    assert plan.kv_capacity_tokens == wanted
    assert plan.capacity_overridden is True
    assert plan.notes == []
    assert plan.kv_bytes == wanted * plan.kv_token_bytes
    assert plan.planned_bytes <= plan.budget_bytes
    assert plan.cpu_requested_tokens == int(wanted * 1.5)
    assert plan.cpu_capacity_tokens == host_pool_tokens(int(wanted * 1.5), plan.kv_token_bytes)
    assert "override" in plan.describe()
    assert "override" not in auto.describe()


def test_capacity_override_above_the_plan_is_honoured_and_flagged():
    auto = qwen3_plan()
    wanted = auto.kv_capacity_tokens + 10_000
    plan = qwen3_plan(kv_capacity_tokens=wanted)
    assert plan.kv_capacity_tokens == wanted
    assert plan.capacity_overridden is True
    assert len(plan.notes) == 1
    assert "exceed" in plan.notes[0] and str(wanted) in plan.notes[0] and str(auto.kv_capacity_tokens) in plan.notes[0]
    # the plan does not hide that the share is exceeded
    assert plan.planned_bytes > plan.budget_bytes
    assert plan.notes[0] in plan.describe()


def test_capacity_is_clamped_to_the_kernel_buffers():
    limits = EngineLimits()
    kernel_limit = min(limits.max_total_verify_kv_len, limits.max_kv_context_len)

    # explicit override far beyond what the kernels can index
    plan = qwen3_plan(kv_capacity_tokens=10_000_000)
    assert 0 < plan.kv_capacity_tokens < kernel_limit
    assert plan.kv_capacity_tokens >= kernel_limit - 1  # as much as the buffers allow
    assert any("clamp" in note for note in plan.notes)
    assert plan.kv_bytes == plan.kv_capacity_tokens * plan.kv_token_bytes
    assert plan.cpu_requested_tokens == int(plan.kv_capacity_tokens * 1.5)
    assert plan.cpu_capacity_tokens == host_pool_tokens(plan.cpu_requested_tokens, plan.kv_token_bytes)

    # automatic sizing on a card with far more memory than the buffers can use
    big = qwen3_plan(total_bytes=4 * A100_TOTAL_BYTES, mem_fraction=0.9)
    assert 0 < big.kv_capacity_tokens < kernel_limit
    assert any("clamp" in note for note in big.notes)
    assert big.capacity_overridden is False
    assert big.planned_bytes <= big.budget_bytes

    # the limit comes from EngineLimits, not from a constant in plan_memory
    small = dataclasses.replace(limits, max_total_verify_kv_len=5_000)
    plan = qwen3_plan(limits=small)
    assert 0 < plan.kv_capacity_tokens < 5_000
    assert any("clamp" in note for note in plan.notes)
    small = dataclasses.replace(limits, max_kv_context_len=7_000)
    plan = qwen3_plan(limits=small)
    assert 0 < plan.kv_capacity_tokens < 7_000


def test_plan_raises_when_nothing_is_left_for_the_kv_pool():
    with pytest.raises(MemoryError, match="gpu_memory_utilization"):
        qwen3_plan(mem_fraction=0.05)
    with pytest.raises(MemoryError):
        qwen3_plan(max_batch_size=2_000)  # 2000 x 11.6 MiB of per-request buffers alone exceed the share
    with pytest.raises(MemoryError):
        qwen3_plan(weights_bytes=int(20e9))
    with pytest.raises(MemoryError):
        qwen3_plan(kv_capacity_tokens=0)
    # exactly the non-KV needs: zero tokens left
    plan = qwen3_plan()
    non_kv = plan.planned_bytes - plan.kv_bytes
    with pytest.raises(MemoryError):
        qwen3_plan(total_bytes=non_kv, mem_fraction=1.0)
    # one token more than that is enough
    one = qwen3_plan(total_bytes=non_kv + plan.kv_token_bytes, mem_fraction=1.0)
    assert one.kv_capacity_tokens == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"mem_fraction": 0.0},
        {"mem_fraction": -0.1},
        {"mem_fraction": 1.01},
        {"max_batch_size": 0},
        {"max_batch_size": EngineLimits().max_batch_size},
        {"cpu_offload_ratio": -0.5},
    ],
)
def test_plan_rejects_invalid_arguments(overrides):
    with pytest.raises(ValueError):
        qwen3_plan(**overrides)


def test_plan_accepts_the_whole_card():
    plan = qwen3_plan(mem_fraction=1.0)
    assert plan.budget_bytes == A100_TOTAL_BYTES
    assert plan.planned_bytes <= plan.budget_bytes


def test_describe_lists_every_component():
    plan = qwen3_plan()
    text = plan.describe()
    for label in ("weights", "fixed buffers", "CUDA graphs", "per-request", "transient", "safety margin", "KV pool"):
        assert label in text
    assert f"{plan.kv_capacity_tokens} tokens" in text
    assert f"{plan.cpu_capacity_tokens} tokens" in text


# ---------------------------------------------------------------------------
# size helpers
# ---------------------------------------------------------------------------
def test_kv_token_bytes():
    assert kv_token_bytes(NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, BF16) == 114_688  # keys + values, all layers
    assert kv_token_bytes(2, 2, 8, 4) == 2 * 2 * 8 * 4 * 2


def test_per_request_bytes_matches_the_buffers_a_request_holds():
    limits = EngineLimits()
    # Request.draft_probs: [RejectionSampler.MAX_SPEC_LEN, vocab] float32   (serve/request/request.py)
    draft_probs = limits.max_spec_len * VOCAB * 4
    # PillarCachePtr.selected_indices_buffer_gpu: [layers, kv_heads, MAX_LEN_SELECTED_KV_CACHE] int32
    selected = NUM_LAYERS * NUM_KV_HEADS * limits.max_len_selected_kv_cache * 4
    assert per_request_bytes(NUM_LAYERS, NUM_KV_HEADS, VOCAB, limits) == draft_probs + selected == 11_582_464
    bigger = dataclasses.replace(limits, max_spec_len=12)
    assert per_request_bytes(NUM_LAYERS, NUM_KV_HEADS, VOCAB, bigger) == 12 * VOCAB * 4 + selected


def test_graph_bytes_only_counts_graphs_that_fit():
    full = memory.graph_bytes(VOCAB, HIDDEN, BF16)
    assert full > 2 * sum(memory.GRAPH_BATCH_SIZES) * VOCAB * BF16  # at least both phases' logits buffers
    assert memory.graph_bytes(VOCAB, HIDDEN, BF16, kv_capacity_hint=10**9) == full
    assert 0 < memory.graph_bytes(VOCAB, HIDDEN, BF16, kv_capacity_hint=10_000) < full


def test_request_token_need():
    assert request_token_need(1024, 8192, 8) == 1024 + 8192 + 8 + 2
    assert request_token_need(1, 1, 1) == 5
    # monotone in every argument
    assert request_token_need(11, 5, 8) == request_token_need(10, 5, 8) + 1
    assert request_token_need(10, 6, 8) == request_token_need(10, 5, 8) + 1
    assert request_token_need(10, 5, 9) == request_token_need(10, 5, 8) + 1


# ---------------------------------------------------------------------------
# check_workload_fits: every failure mode
# ---------------------------------------------------------------------------
def test_workload_too_long_for_the_position_embeddings():
    plan = qwen3_plan()
    need = request_token_need(MAX_PROMPT, MAX_RESPONSE, SPEC_STRIDE)
    assert fits(plan, max_position_embeddings=need) is None  # exactly fits
    with pytest.raises(ValueError, match="max_position_embeddings"):
        fits(plan, max_position_embeddings=need - 1)
    with pytest.raises(ValueError, match="max_position_embeddings"):
        fits(plan, max_position_embeddings=MAX_PROMPT + MAX_RESPONSE)  # no room for the speculation slack
    assert fits(plan, max_position_embeddings=None) is None  # unknown limit: not checked


def test_request_larger_than_gpu_plus_host_pool():
    plan = qwen3_plan(kv_capacity_tokens=4_000, cpu_offload_ratio=0.0)
    need = request_token_need(100, 6_000, SPEC_STRIDE)
    assert need > plan.total_capacity_tokens
    with pytest.raises(MemoryError, match="gpu_memory_utilization"):
        fits(plan, prompt_lens=[100], max_new_tokens=6_000)


def test_request_larger_than_the_gpu_pool_alone():
    """The host pool does not help a single request: it must fit on the GPU."""
    plan = qwen3_plan(kv_capacity_tokens=20_000)  # + 30,000 host tokens
    max_new = 25_000
    need = request_token_need(100, max_new, SPEC_STRIDE)
    assert plan.kv_capacity_tokens < need <= admission_cap(plan)  # only the GPU-pool condition is violated
    with pytest.raises(MemoryError) as err:
        fits(plan, prompt_lens=[100], max_new_tokens=max_new, max_position_embeddings=None)
    assert str(need) in str(err.value) and str(plan.kv_capacity_tokens) in str(err.value)
    # and it is accepted as soon as the GPU pool is large enough
    assert (
        fits(
            qwen3_plan(kv_capacity_tokens=need + 1),
            prompt_lens=[100],
            max_new_tokens=max_new,
            max_position_embeddings=None,
        )
        is None
    )


def test_no_headroom_to_prefill_a_lone_request():
    """With a tiny prefill chunk the prompt takes many iterations; the tokens
    other work could append meanwhile must still fit next to the prompt."""
    plan = qwen3_plan(kv_capacity_tokens=1_000, cpu_offload_ratio=10.0)
    longest, chunk = 100, 1
    need = request_token_need(longest, 1, SPEC_STRIDE)
    prefill_iters = -(-longest // chunk) + 1
    headroom = longest + (SPEC_STRIDE + 2) * prefill_iters
    assert need <= plan.kv_capacity_tokens and need <= admission_cap(plan, max_batch_size=1, chunk_size=chunk)
    assert headroom >= plan.kv_capacity_tokens  # only the headroom condition is violated
    with pytest.raises(MemoryError):
        fits(plan, prompt_lens=[longest], max_new_tokens=1, chunk_size=chunk, max_batch_size=1)
    # the same request with the default chunk size is fine
    assert fits(plan, prompt_lens=[longest], max_new_tokens=1, max_batch_size=1) is None


def test_batch_can_overflow_the_draft_index_buffer():
    plan = qwen3_plan()
    limits = EngineLimits()
    longest = 2_048
    worst = MAX_BATCH * (longest + SPEC_STRIDE + 2)
    assert worst >= limits.max_total_draft_kv_len  # 64 x 2058 = 131,712 > 128 Ki
    with pytest.raises(ValueError, match="max_batch_size") as err:
        fits(plan, prompt_lens=[longest], max_new_tokens=128)
    # the message recommends a batch size that does fit
    recommended = (limits.max_total_draft_kv_len - 1) // (longest + SPEC_STRIDE + 2)
    assert str(recommended) in str(err.value)
    assert fits(plan, prompt_lens=[longest], max_new_tokens=128, max_batch_size=recommended) is None
    with pytest.raises(ValueError):
        fits(plan, prompt_lens=[longest], max_new_tokens=128, max_batch_size=recommended + 1)
    # the limit is taken from EngineLimits
    roomy = dataclasses.replace(limits, max_total_draft_kv_len=4 * limits.max_total_draft_kv_len)
    assert fits(plan, prompt_lens=[longest], max_new_tokens=128, limits=roomy) is None


def test_baseline_batch_fits_the_draft_index_buffer():
    # 64 concurrent requests with 1024-token prompts: 64 x 1034 = 66,176 < 128 Ki
    assert MAX_BATCH * (MAX_PROMPT + SPEC_STRIDE + 2) < EngineLimits().max_total_draft_kv_len


# ---------------------------------------------------------------------------
# EngineLimits defaults mirror the constants in the engine sources (read with
# ast: importing those modules needs flashinfer)
# ---------------------------------------------------------------------------
def _const(node) -> int:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _const(node.left) * _const(node.right)
    raise ValueError(ast.dump(node))


def _class_constants(path, class_name: str) -> dict[str, int]:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            out = {}
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    try:
                        out[stmt.targets[0].id] = _const(stmt.value)
                    except ValueError:
                        pass
            return out
    raise AssertionError(f"class {class_name} not found in {path}")


def test_engine_limits_defaults_mirror_the_sources(tk):
    serve = tk.REPO_ROOT / "serve"
    pillar = _class_constants(serve / "request" / "kv_cache_ptr" / "pillar.py", "PillarCachePtr")
    full = _class_constants(serve / "request" / "kv_cache_ptr" / "base.py", "FullCachePtr")
    sampler = _class_constants(serve / "sampling" / "rejection_sampler.py", "RejectionSampler")
    limits = EngineLimits()
    assert limits.max_total_draft_kv_len == pillar["MAX_TOTAL_DRAFT_KV_LEN"]
    assert limits.max_total_verify_kv_len == pillar["MAX_TOTAL_VERIFY_KV_LEN"]
    assert limits.max_len_selected_kv_cache == pillar["MAX_LEN_SELECTED_KV_CACHE"]
    assert limits.max_kv_context_len == full["_MAX_KV_CONTEXT_LEN"]
    assert limits.max_nnz == full["_MAX_NNZ"]
    assert limits.max_batch_size == full["_MAX_BATCH_SIZE"]
    assert limits.max_spec_len == sampler["MAX_SPEC_LEN"]
    if hasattr(memory, "OFFLOAD_CHUNK_TOKENS"):
        request = _class_constants(serve / "request" / "request.py", "Request")
        assert memory.OFFLOAD_CHUNK_TOKENS == request["MAX_OFFLOADING_LEN"]


def test_graph_batch_sizes_mirror_the_capture_list(tk):
    """CUDAGraphModelWrapper.eager_cuda_graph_mode captures a hard-coded list of batch sizes."""
    tree = ast.parse((tk.REPO_ROOT / "serve" / "model" / "graph_model.py").read_text())
    captured = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "eager_cuda_graph_mode":
            for stmt in ast.walk(node):
                if (
                    isinstance(stmt, ast.Assign)
                    and isinstance(stmt.targets[0], ast.Name)
                    and stmt.targets[0].id == "captured_args"
                ):
                    captured = ast.literal_eval(stmt.value)
    assert captured is not None, "captured_args not found in eager_cuda_graph_mode"
    assert tuple(captured) == memory.GRAPH_BATCH_SIZES


# ---------------------------------------------------------------------------
# pinned host pool sizing and the int32 probability-offset limit
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "nbytes, block",
    [(0, 0), (-5, 0), (1, 1), (2, 2), (3, 4), (2**34, 2**34), (2**34 + 1, 2**35), (17_329_471_488, 2**35)],
)
def test_pinned_block_bytes_is_the_next_power_of_two(nbytes, block):
    assert pinned_block_bytes(nbytes) == block


def test_host_pool_tokens_fills_the_block_it_locks():
    token = 114_688
    assert host_pool_tokens(0, token) == 0
    assert host_pool_tokens(-3, token) == 0
    # a request that (nearly) fills its block is kept as is
    full = 2**34 // token
    assert host_pool_tokens(full, token) == full
    nearly = int(2**34 * 0.95) // token
    assert host_pool_tokens(nearly, token) == nearly
    # just above a power of two: shrink to fill the block below instead of locking twice as much
    assert host_pool_tokens(full + 1, token) == full
    assert host_pool_tokens(151_101, token) == full  # the canonical 1.5x request (16.14 GiB)
    # well inside a block but not close to full: also the block below
    assert host_pool_tokens(int(2**34 * 1.5) // token, token) == full
    for requested in (1, 7, 1000, 99_999, 151_101, 299_593, 300_000, 600_000):
        got = host_pool_tokens(requested, token)
        assert 0 < got <= requested
        assert pinned_block_bytes(got * token) - got * token <= pinned_block_bytes(got * token) // 16 + token


def test_max_rows_per_pass_and_the_offset_check():
    assert max_rows_per_pass(64, 8, 256) == 64 * 9 + 256 == 832
    # Qwen3's vocabulary: int32 byte offsets (row * 4 * vocab) wrap at row 3534
    limit = (2**31 - 1) // (4 * VOCAB)
    assert limit == 3533
    check_prob_offsets(64, 8, 256, VOCAB)  # the project configuration is far below the limit
    check_prob_offsets((limit - 256) // 9, 8, 256, VOCAB)  # the largest batch that still fits
    with pytest.raises(ValueError, match="int32"):
        check_prob_offsets((limit - 256) // 9 + 1, 8, 256, VOCAB)
    with pytest.raises(ValueError, match="max_batch_size"):
        check_prob_offsets(512, 8, 256, VOCAB)
