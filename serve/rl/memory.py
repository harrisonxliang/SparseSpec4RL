"""GPU memory planning for the persistent RL engine (pure Python).

``serve/run.py`` sizes the KV pool as ``total * mem_fraction - model_size``
and lets everything else (CUDA graphs, per-request buffers, sampler scratch)
land on top of that. Under verl the engine shares one 40 GB card with FSDP
training, so here ``mem_fraction`` is the engine's WHOLE share, matching the
meaning of vLLM's ``gpu_memory_utilization``: weights + fixed buffers + CUDA
graphs + per-request buffers + transient scratch + KV pool all have to fit in
``total * mem_fraction``. The KV pool gets what is left.

Sizes that cannot be known exactly without a GPU are deliberately rounded up;
the engine logs the measured numbers next to this plan after the first
wake-up so the estimates can be checked on the box.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MIB = 1024**2
GIB = 1024**3

# Batch sizes captured by CUDAGraphModelWrapper.eager_cuda_graph_mode
# (hard-coded there, independent of its arguments).
GRAPH_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 640)
GRAPH_CAPTURE_SEQ_LEN = 128  # dummy context length used during capture
GRAPH_CAPTURE_EXTRA_PAGES = 1024


@dataclass
class EngineLimits:
    """Compile-time buffer sizes of the serve/ kernels (read from the classes
    by the engine; defaults mirror the source for the CPU tests)."""

    max_total_draft_kv_len: int = 128 * 1024  # PillarCachePtr.MAX_TOTAL_DRAFT_KV_LEN
    max_total_verify_kv_len: int = 256 * 1024  # PillarCachePtr.MAX_TOTAL_VERIFY_KV_LEN
    max_len_selected_kv_cache: int = 6 * 1024  # PillarCachePtr.MAX_LEN_SELECTED_KV_CACHE
    max_kv_context_len: int = 2 * 1024 * 1024  # FullCachePtr._MAX_KV_CONTEXT_LEN
    max_nnz: int = 8192 * 4  # FullCachePtr._MAX_NNZ
    max_batch_size: int = 4 * 1024  # FullCachePtr._MAX_BATCH_SIZE
    max_spec_len: int = 10  # RejectionSampler.MAX_SPEC_LEN


@dataclass
class MemoryPlan:
    total_bytes: int
    budget_bytes: int
    weights_bytes: int
    fixed_bytes: int
    graph_bytes: int
    per_request_bytes: int
    request_bytes: int
    transient_bytes: int
    safety_bytes: int
    kv_token_bytes: int
    kv_bytes: int
    kv_capacity_tokens: int
    cpu_capacity_tokens: int
    cpu_pinned_bytes: int
    capacity_overridden: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def total_capacity_tokens(self) -> int:
        return self.kv_capacity_tokens + self.cpu_capacity_tokens

    @property
    def planned_bytes(self) -> int:
        return (
            self.weights_bytes
            + self.fixed_bytes
            + self.graph_bytes
            + self.request_bytes
            + self.transient_bytes
            + self.safety_bytes
            + self.kv_bytes
        )

    def describe(self) -> str:
        def gib(x):
            return f"{x / GIB:7.2f} GiB"

        lines = [
            f"engine share      {gib(self.budget_bytes)}  (of {gib(self.total_bytes).strip()} total)",
            f"  weights         {gib(self.weights_bytes)}",
            f"  fixed buffers   {gib(self.fixed_bytes)}  (attention metadata, top-k / attention workspaces)",
            f"  CUDA graphs     {gib(self.graph_bytes)}",
            f"  per-request     {gib(self.request_bytes)}  ({self.per_request_bytes / MIB:.1f} MiB x max_batch_size)",
            f"  transient       {gib(self.transient_bytes)}  (sampler / verification scratch)",
            f"  safety margin   {gib(self.safety_bytes)}",
            f"  KV pool         {gib(self.kv_bytes)}  = {self.kv_capacity_tokens} tokens "
            f"({self.kv_token_bytes / 1024:.1f} KiB/token)"
            + ("  [kv_capacity_tokens override]" if self.capacity_overridden else ""),
            f"host offload pool {gib(self.cpu_pinned_bytes)}  = {self.cpu_capacity_tokens} tokens (pinned)",
        ]
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)


def kv_token_bytes(num_layers: int, num_kv_heads: int, head_dim: int, dtype_bytes: int) -> int:
    """Bytes of KV cache per token (keys + values, all layers)."""
    return num_layers * num_kv_heads * head_dim * dtype_bytes * 2


def per_request_bytes(num_layers: int, num_kv_heads: int, vocab_size: int, limits: EngineLimits) -> int:
    """GPU memory every in-flight request holds OUTSIDE the KV pool.

    * ``Request.draft_probs``: [MAX_SPEC_LEN, vocab] float32
    * ``PillarCachePtr.selected_indices_buffer_gpu``:
      [num_layers, num_kv_heads, MAX_LEN_SELECTED_KV_CACHE] int32
    Neither is ever offloaded, so offloaded requests still hold them.
    """
    draft_probs = limits.max_spec_len * vocab_size * 4
    selected_indices = num_layers * num_kv_heads * limits.max_len_selected_kv_cache * 4
    return draft_probs + selected_indices


def fixed_buffer_bytes(num_layers: int, num_kv_heads: int, limits: EngineLimits) -> int:
    """Persistent non-KV buffers of the pillar backend (two CUDA-graph phases)."""
    lh = num_layers * num_kv_heads
    draft_idx = lh * limits.max_total_draft_kv_len * 4  # int32
    verify_logits = lh * limits.max_total_verify_kv_len * 2  # float16
    # PillarCachePtr class buffers: draft_kv_indices_tensor_buf + dump_logits_snapshot_tensor_buf
    class_level = draft_idx + verify_logits
    # per phase: kv_indices + flatten_indices + dump_logits + small index arrays
    small = 2 * limits.max_nnz * 4 + 8 * (limits.max_batch_size + 1) * 4
    per_phase = limits.max_kv_context_len * 4 + draft_idx + verify_logits + small
    # workspaces: shared 256 MiB attention float workspace, per-phase 8 MiB
    # int workspace and 64 MiB top-k workspace (serve/model/model.py,
    # serve/attention/backend.py, flashinfer/attention.py)
    workspaces = 256 * MIB + 2 * (8 * MIB + 64 * MIB)
    return class_level + 2 * per_phase + workspaces


def graph_bytes(vocab_size: int, hidden_size: int, dtype_bytes: int, kv_capacity_hint: int | None = None) -> int:
    """Memory held by the captured CUDA graphs (private pools).

    Dominated by each graph's logits output [batch, vocab]. Activations inside
    a graph are [batch, O(hidden)] tensors whose blocks the graph's private
    pool reuses from layer to layer; they are covered by an allowance of 32
    hidden-sized rows per token plus 20% headroom. Deliberately on the high
    side: compare with the measured "resident while awake" number the engine
    logs and set sparsespec.kv_capacity_tokens if the gap is worth reclaiming.
    """
    sizes = [
        b
        for b in GRAPH_BATCH_SIZES
        if kv_capacity_hint is None or GRAPH_CAPTURE_SEQ_LEN * b + GRAPH_CAPTURE_EXTRA_PAGES < kv_capacity_hint
    ]
    tokens = sum(sizes)
    per_token = vocab_size * dtype_bytes + 32 * hidden_size * dtype_bytes
    return int(2 * tokens * per_token * 1.2)


def plan_memory(
    *,
    total_bytes: int,
    mem_fraction: float,
    weights_bytes: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    vocab_size: int,
    hidden_size: int,
    dtype_bytes: int,
    max_batch_size: int,
    cuda_graph: bool,
    limits: EngineLimits | None = None,
    cpu_offload_ratio: float = 1.5,
    transient_reserve_bytes: int = 2 * GIB,
    safety_bytes: int = GIB // 2,
    kv_capacity_tokens: int | None = None,
) -> MemoryPlan:
    if not 0.0 < mem_fraction <= 1.0:
        raise ValueError(f"mem_fraction must be in (0, 1], got {mem_fraction}")
    if max_batch_size < 1:
        raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
    if cpu_offload_ratio < 0:
        raise ValueError(f"cpu_offload_ratio must be >= 0, got {cpu_offload_ratio}")
    limits = limits or EngineLimits()
    if max_batch_size >= limits.max_batch_size:
        raise ValueError(f"max_batch_size={max_batch_size} exceeds the kernel limit {limits.max_batch_size}")

    budget = int(total_bytes * mem_fraction)
    token_bytes = kv_token_bytes(num_layers, num_kv_heads, head_dim, dtype_bytes)
    fixed = fixed_buffer_bytes(num_layers, num_kv_heads, limits)
    req_each = per_request_bytes(num_layers, num_kv_heads, vocab_size, limits)
    req_total = req_each * max_batch_size
    graphs = graph_bytes(vocab_size, hidden_size, dtype_bytes) if cuda_graph else 0
    notes: list[str] = []

    non_kv = weights_bytes + fixed + graphs + req_total + transient_reserve_bytes + safety_bytes
    auto_kv_bytes = budget - non_kv
    auto_capacity = auto_kv_bytes // token_bytes if auto_kv_bytes > 0 else 0

    overridden = kv_capacity_tokens is not None
    capacity = int(kv_capacity_tokens) if overridden else int(auto_capacity)
    if overridden and capacity > auto_capacity:
        notes.append(
            f"kv_capacity_tokens={capacity} exceeds the planned {auto_capacity}: "
            "the engine will exceed its memory share by "
            f"{(capacity - auto_capacity) * token_bytes / GIB:.2f} GiB"
        )

    # Kernel buffer limits (see EngineLimits): a single verify iteration dumps
    # attention scores for every verifying request's full context, and the
    # paged-KV index buffer holds every active request's page list.
    hard_cap = min(limits.max_total_verify_kv_len, limits.max_kv_context_len) - 1
    if capacity > hard_cap:
        notes.append(f"KV capacity clamped from {capacity} to the kernel buffer limit {hard_cap}")
        capacity = hard_cap

    if capacity <= 0:
        raise MemoryError(
            "no memory left for the KV pool: "
            f"share {budget / GIB:.2f} GiB < non-KV needs {non_kv / GIB:.2f} GiB "
            f"(weights {weights_bytes / GIB:.2f}, fixed {fixed / GIB:.2f}, graphs {graphs / GIB:.2f}, "
            f"requests {req_total / GIB:.2f}, transient {transient_reserve_bytes / GIB:.2f}, "
            f"safety {safety_bytes / GIB:.2f}). Raise gpu_memory_utilization or lower max_batch_size."
        )

    cpu_capacity = int(capacity * cpu_offload_ratio)
    return MemoryPlan(
        total_bytes=int(total_bytes),
        budget_bytes=budget,
        weights_bytes=int(weights_bytes),
        fixed_bytes=fixed,
        graph_bytes=graphs,
        per_request_bytes=req_each,
        request_bytes=req_total,
        transient_bytes=int(transient_reserve_bytes),
        safety_bytes=int(safety_bytes),
        kv_token_bytes=token_bytes,
        kv_bytes=capacity * token_bytes,
        kv_capacity_tokens=capacity,
        cpu_capacity_tokens=cpu_capacity,
        cpu_pinned_bytes=cpu_capacity * token_bytes,
        capacity_overridden=overridden,
        notes=notes,
    )


def request_token_need(prompt_len: int, max_new_tokens: int, spec_stride: int) -> int:
    """Upper bound on the KV tokens one request can ever hold.

    A request finishes at the first verify after its KV length reaches
    ``prompt_len + max_new_tokens - 1`` (see RLSpecScheduler); one more
    draft/verify cycle can add ``spec_stride + 1`` tokens before that check.
    """
    return prompt_len + max_new_tokens + spec_stride + 2


def check_workload_fits(
    *,
    prompt_lens: list[int],
    max_new_tokens: int,
    spec_stride: int,
    chunk_size: int,
    max_batch_size: int,
    plan: MemoryPlan,
    max_position_embeddings: int | None,
    limits: EngineLimits | None = None,
) -> None:
    """Fail fast (before any forward pass) on workloads the engine cannot run."""
    limits = limits or EngineLimits()
    longest = max(prompt_lens)
    need = request_token_need(longest, max_new_tokens, spec_stride)
    if max_position_embeddings is not None and need > max_position_embeddings:
        raise ValueError(
            f"prompt ({longest}) + max_new_tokens ({max_new_tokens}) + speculation slack exceeds "
            f"max_position_embeddings ({max_position_embeddings})"
        )
    # A lone request must be admissible, otherwise the scheduler loop never starts.
    prefill_iters = -(-longest // chunk_size) + 1
    headroom = longest + (spec_stride + 2) * prefill_iters
    if need > plan.total_capacity_tokens or headroom >= plan.kv_capacity_tokens or need > plan.kv_capacity_tokens:
        raise MemoryError(
            f"a single request may need {need} KV tokens but the GPU pool holds {plan.kv_capacity_tokens} "
            f"(+{plan.cpu_capacity_tokens} on host). Raise gpu_memory_utilization or lower the response length."
        )
    # Cold start: the first draft cycle of a request loads its whole prompt
    # (PillarCachePtr.from_kv_cache); all in-flight requests share one
    # MAX_TOTAL_DRAFT_KV_LEN index buffer.
    worst_draft = max_batch_size * (longest + spec_stride + 2)
    if worst_draft >= limits.max_total_draft_kv_len:
        raise ValueError(
            f"max_batch_size ({max_batch_size}) x longest prompt ({longest}) can overflow the draft index buffer "
            f"({limits.max_total_draft_kv_len} entries). Lower sparsespec.max_batch_size to at most "
            f"{(limits.max_total_draft_kv_len - 1) // (longest + spec_stride + 2)}."
        )
