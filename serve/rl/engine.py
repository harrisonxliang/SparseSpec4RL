"""Persistent SparseSpec engine for RL rollouts.

``serve/run.py::run_inference`` builds model + KV pool + CUDA graphs, runs one
offline job and exits. RL needs one engine for the whole training run that

* generates a batch per training step and returns, for every response token,
  its log-probability under the target (full-attention) distribution;
* takes new weights in place before every step (``load_weights``);
* gives its rollout-only GPU memory back while the trainer runs
  (``sleep`` / ``wake_up``).

The engine is built the way ``run_inference`` builds it (same classes, same
order); the differences are listed in ``SparseSpecEngine.__init__``.

Requires CUDA, flashinfer (the fork in 3rdparty/), RAFT and triton. Importing
this module on a machine without them fails; that is intentional (no stubs).
One engine per process: the pillar KV-cache class keeps kernel buffers in
class attributes.
"""

from __future__ import annotations

import array
import contextlib
import gc
import glob
import logging
import math
import os
import tempfile
import time
from typing import Sequence

import torch
from safetensors.torch import load_file as safe_load
from transformers import AutoConfig

from serve.distribute.tp import setup_tp_worker
from serve.model.config import ModelArgs
from serve.model.graph_model import CUDAGraphModelWrapper
from serve.model.model import KVPool, Transformer
from serve.request.kv_cache_ptr.pillar import PillarCachePtr
from serve.request.request import Request
from serve.rl.base import (
    FINISH_LENGTH,
    GenerationResult,
    LoadReport,
    NamedTensors,
    RolloutEngine,
    empty_stats,
    finalize_response,
    normalize_eos_ids,
    validate_prompts,
    validate_sampling_params,
)
from serve.rl.memory import (
    GIB,
    EngineLimits,
    MemoryPlan,
    admission_capacity_tokens,
    check_prob_offsets,
    check_workload_fits,
    plan_memory,
    request_token_need,
)
from serve.rl.scheduler import RLSpecScheduler
from serve.rl.weights import load_weights_inplace, parameter_pointers, unique_parameter_bytes, verify_weights
from serve.sampling.rejection_sampler import RejectionSampler
from serve.utils import ctx_torch_default, dtype_to_bytes, global_logging_config, str_to_dtype

logger = logging.getLogger(__name__)


def resolve_model_dir(model_path: str) -> str:
    """Return a local directory ``Transformer.from_pretrained`` can load.

    Two upstream quirks are handled here rather than by editing serve/model:

    * For a hub id, ``from_pretrained`` probes ``model.safetensors.index.json``
      and only falls back to a single-file checkpoint on ``FileNotFoundError``.
      Single-file repos such as Qwen3-0.6B raise huggingface_hub's
      EntryNotFoundError instead, so hub ids are resolved to a local snapshot
      first (local directories take the working fallback path).
    * ``serve/model/model.py::Attention`` only creates Qwen3's q_norm/k_norm
      when the config PATH contains "qwen3". A Qwen3 checkpoint stored under
      e.g. ``.../hf/global_step_25`` would silently lose them, so such a
      directory is exposed through a symlink whose name contains "qwen3".
    """
    path = os.path.expanduser(str(model_path))
    if not os.path.isdir(path):
        from huggingface_hub import snapshot_download

        path = snapshot_download(model_path)
    path = os.path.realpath(path)
    model_type = getattr(AutoConfig.from_pretrained(path), "model_type", "") or ""
    if model_type.lower() == "qwen3" and "qwen3" not in path.lower():
        alias_root = tempfile.mkdtemp(prefix="sparsespec-qwen3-")
        alias = os.path.join(alias_root, "qwen3-model")
        os.symlink(path, alias)
        logger.info("model dir %s does not contain 'qwen3'; using alias %s", path, alias)
        path = alias
    return path


def read_hf_state_dict(model_dir: str) -> dict[str, torch.Tensor]:
    """All tensors of a local HF checkpoint (safetensors), on CPU."""
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no *.safetensors files in {model_dir}")
    state: dict[str, torch.Tensor] = {}
    for f in files:
        part = safe_load(f, device="cpu")
        overlap = state.keys() & part.keys()
        if overlap:
            raise ValueError(f"duplicate keys across shards: {sorted(overlap)[:5]}")
        state.update(part)
    return state


class SparseSpecEngine(RolloutEngine):
    name = "sparsespec"

    def __init__(
        self,
        model_path: str,
        mem_fraction: float = 0.45,
        max_batch_size: int = 64,
        spec_stride: int = 8,
        budget_ratio: float = 0.05,
        num_min_budget: int = 128,
        kv_cache: str = "pillar",
        cuda_graph: bool = True,
        seed: int = 0,
        *,
        dtype: str = "bfloat16",
        device: str = "cuda:0",
        chunk_size: int = 256,
        batch_policy: str = "greedy",
        cpu_offload_ratio: float = 1.5,
        kv_capacity_tokens: int | None = None,
        transient_reserve_gb: float = 2.0,
        safety_reserve_gb: float = 0.5,
        eos_token_ids=None,
        verify_init_weights: bool = True,
    ):
        """Build the engine like ``run_inference`` does. Differences:

        * ``mem_fraction`` is the engine's whole GPU share (weights, buffers,
          CUDA graphs, per-request buffers, scratch AND KV pool), like vLLM's
          ``gpu_memory_utilization``; see ``serve/rl/memory.py``.
        * Model size comes from the loaded parameters
          (``ModelArgs.element_size`` has no entry for Qwen3-0.6B).
        * Seeded once, into an engine-private CUDA RNG stream that continues
          across ``generate`` calls and never touches the caller's stream
          (``run_inference`` reseeds with 330 on every call).
        * torch's default device / dtype are set only inside engine calls and
          restored afterwards (``run_inference`` sets them globally, which
          would switch the trainer in the same process to bf16-on-cuda).
        * Tied ``lm_head`` / ``embed_tokens`` share one storage again
          (``from_pretrained`` + ``.to(device)`` leaves two equal copies).
        * Scheduling is synchronous (``async_cpu=False``), admission is
          ``offloading`` with the guards in ``RLSpecScheduler``.
        """
        if not logging.getLogger().hasHandlers():
            global_logging_config()
        if kv_cache != "pillar":
            raise NotImplementedError(f"kv_cache={kv_cache!r}: the RL engine implements PillarAttn ('pillar') only")
        if batch_policy not in ("greedy", "naive"):
            # 'random' draws from Python's global RNG, which the engine does not own
            raise ValueError(f"batch_policy must be 'greedy' or 'naive', got {batch_policy!r}")
        if spec_stride < 1:
            raise ValueError(f"spec_stride must be >= 1, got {spec_stride}")
        if not torch.cuda.is_available():
            raise RuntimeError("SparseSpecEngine needs a CUDA device")
        if PillarCachePtr._INITIALIZED != 0:
            raise RuntimeError(
                "a SparseSpecEngine already exists in this process (the pillar KV-cache class keeps its kernel "
                "buffers in class attributes); use one engine per process"
            )

        self.model_path = str(model_path)
        self.mem_fraction = float(mem_fraction)
        self.max_batch_size = int(max_batch_size)
        self.spec_stride = int(spec_stride)
        self.budget_ratio = float(budget_ratio)
        self.num_min_budget = int(num_min_budget)
        self.kv_cache = kv_cache
        self.cuda_graph = bool(cuda_graph)
        self.seed = int(seed)
        self.chunk_size = int(chunk_size)
        self.batch_policy = batch_policy
        self.device = torch.device(device)
        self.dtype = str_to_dtype(dtype)
        self._closed = False
        self._sleeping = True  # nothing allocated yet
        self._runner = None
        self._wake_stats = {"wake_s": 0.0, "graph_capture_s": 0.0}
        self._measured: dict[str, float] = {}
        # reserved-but-not-allocated memory held by the CUDA graphs' private pools
        self._graph_pool_bytes = 0
        # running process-wide peaks, kept because generate() resets torch's counters
        self._process_peak_alloc = 0
        self._process_peak_reserved = 0
        self._warned_inflight_cap: set = set()

        init_start = time.perf_counter()
        gc.collect()
        allocated_before = torch.cuda.memory_allocated(self.device)

        # engine-private RNG stream, seeded exactly once
        outer_rng = torch.cuda.get_rng_state(self.device)
        with torch.cuda.device(self.device):
            torch.cuda.manual_seed(self.seed)
        self._rng_state = torch.cuda.get_rng_state(self.device)
        torch.cuda.set_rng_state(outer_rng, self.device)

        self.model_dir = resolve_model_dir(self.model_path)
        hf_config = AutoConfig.from_pretrained(self.model_dir)
        self._default_eos = normalize_eos_ids(eos_token_ids) or self._eos_from_checkpoint(hf_config)

        # same adjustment run_inference makes; must precede any Request / memory plan
        if self.spec_stride + 1 >= RejectionSampler.MAX_SPEC_LEN:
            RejectionSampler.MAX_SPEC_LEN = math.ceil((self.spec_stride + 1) / 4) * 4
            logger.warning("RejectionSampler.MAX_SPEC_LEN raised to %d", RejectionSampler.MAX_SPEC_LEN)

        with self._ctx():
            model = Transformer.from_pretrained(repo_id=self.model_dir, device=str(self.device), dtype=self.dtype)
            model = setup_tp_worker(
                model=model, tp_ranks=1, device=str(self.device), dtype=self.dtype, reduce_method="customized"
            )
            self.model = model
            self.config: ModelArgs = model.config
            self._check_architecture(hf_config)
            self._retie_embeddings()
            if verify_init_weights:
                report = verify_weights(self.model, read_hf_state_dict(self.model_dir))
                if not report["ok"]:
                    raise RuntimeError(f"engine weights differ from the checkpoint right after loading: {report}")

            if (self.spec_stride + 1) * self.config.num_qo_heads // self.config.num_kv_heads >= 64:
                raise ValueError(
                    "spec_stride too large for the verification score kernel "
                    "((spec_stride + 1) * num_qo_heads / num_kv_heads must stay below 64)"
                )
            check_prob_offsets(self.max_batch_size, self.spec_stride, self.chunk_size, self.config.vocab_size)

            self.limits = EngineLimits(
                max_total_draft_kv_len=PillarCachePtr.MAX_TOTAL_DRAFT_KV_LEN,
                max_total_verify_kv_len=PillarCachePtr.MAX_TOTAL_VERIFY_KV_LEN,
                max_len_selected_kv_cache=PillarCachePtr.MAX_LEN_SELECTED_KV_CACHE,
                max_kv_context_len=PillarCachePtr._MAX_KV_CONTEXT_LEN,
                max_nnz=PillarCachePtr._MAX_NNZ,
                max_batch_size=PillarCachePtr._MAX_BATCH_SIZE,
                max_spec_len=RejectionSampler.MAX_SPEC_LEN,
            )
            self.plan: MemoryPlan = plan_memory(
                total_bytes=torch.cuda.get_device_properties(self.device).total_memory,
                mem_fraction=self.mem_fraction,
                weights_bytes=unique_parameter_bytes(self.model.parameters()),
                num_layers=self.config.num_layers,
                num_kv_heads=self.config.num_kv_heads,
                head_dim=self.config.head_dim,
                vocab_size=self.config.vocab_size,
                hidden_size=self.config.dim,
                dtype_bytes=dtype_to_bytes(self.dtype),
                max_batch_size=self.max_batch_size,
                cuda_graph=self.cuda_graph,
                limits=self.limits,
                cpu_offload_ratio=cpu_offload_ratio,
                transient_reserve_bytes=int(transient_reserve_gb * GIB),
                safety_bytes=int(safety_reserve_gb * GIB),
                kv_capacity_tokens=kv_capacity_tokens,
            )
            logger.info("SparseSpecEngine memory plan:\n%s", self.plan.describe())

            self._spec_args = {
                "spec_stride": self.spec_stride,
                "budget_ratio": self.budget_ratio,
                "num_min_budget": self.num_min_budget,
                "num_sink_tokens": 4,
                "max_batch_size": self.max_batch_size,
            }
            PillarCachePtr.initialize_class_metadata(
                dtype=self.dtype,
                head_dim=self.config.head_dim,
                num_qo_heads=self.config.num_qo_heads,
                num_kv_heads=self.config.num_kv_heads,
                num_layers=self.config.num_layers,
                capacity=self.plan.kv_capacity_tokens,
                device=str(self.device),
                max_batch_size=self.max_batch_size,
                spec_cache_args=self._spec_args,
            )
            pin_start = time.perf_counter()
            self.kv_pool = KVPool(
                config=self.config,
                capacity=self.plan.kv_capacity_tokens,
                device=str(self.device),
                dtype=self.dtype,
                kv_cache_class=PillarCachePtr,
                page_size=1,
                vanilla_backend=False,
                cpu_capacity=self.plan.cpu_capacity_tokens,
            )
            self._kv_shape = tuple(self.kv_pool._physical_mem_gpu.shape)
            logger.info("KV pool built in %.1fs (includes pinning the host offload pool)", time.perf_counter() - pin_start)
            self.model.attach_kv_pool(self.kv_pool)
            self._sleeping = False
            self._build_runner()

        torch.cuda.synchronize(self.device)
        # allocated-based (the trainer's share is derived from it in generate());
        # the graphs' private-pool slack is tracked separately and added to
        # everything that is reported
        self._resident_bytes = torch.cuda.memory_allocated(self.device) - allocated_before
        self._measured["init_s"] = time.perf_counter() - init_start
        self._measured["resident_awake_gb"] = (self._resident_bytes + self._graph_pool_bytes) / GIB
        logger.info(
            "SparseSpecEngine ready in %.1fs: %.2f GiB resident while awake (planned without transient scratch: "
            "%.2f GiB, share: %.2f GiB)",
            self._measured["init_s"],
            self._measured["resident_awake_gb"],
            (self.plan.planned_bytes - self.plan.transient_bytes - self.plan.safety_bytes - self.plan.request_bytes)
            / GIB,
            self.plan.budget_bytes / GIB,
        )

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def _ctx(self):
        """Everything the engine does on the GPU runs in here: inference mode,
        run_inference's torch defaults, and the engine's own RNG stream."""
        outer_rng = torch.cuda.get_rng_state(self.device)
        torch.cuda.set_rng_state(self._rng_state, self.device)
        # ctx_torch_default "restores" the default device with
        # set_default_device(cpu), which installs a DeviceContext('cpu')
        # torch-function mode instead of returning to "no context". Left in
        # place it would change torch.tensor(<cuda tensor>) in the trainer
        # thread, so remove it unless the caller had a context of its own.
        tls = getattr(torch, "_GLOBAL_DEVICE_CONTEXT", None)
        had_device_ctx = getattr(tls, "device_context", None) is not None
        try:
            with torch.inference_mode(), ctx_torch_default(device=str(self.device), dtype=self.dtype):
                yield
        finally:
            try:
                if not had_device_ctx:
                    torch.set_default_device(None)  # pops the leftover DeviceContext mode
                    if tls is not None and getattr(tls, "device_context", 0) is None:
                        # torch <= 2.7: get_default_device() dereferences this None
                        del tls.device_context
            finally:
                self._rng_state = torch.cuda.get_rng_state(self.device)
                torch.cuda.set_rng_state(outer_rng, self.device)

    def _eos_from_checkpoint(self, hf_config) -> tuple[int, ...]:
        eos: list = []
        try:
            from transformers import GenerationConfig

            gen = GenerationConfig.from_pretrained(self.model_dir)
            eos.extend(normalize_eos_ids(gen.eos_token_id))
        except OSError:
            pass
        eos.extend(normalize_eos_ids(getattr(hf_config, "eos_token_id", None)))
        return normalize_eos_ids(eos)

    def _check_architecture(self, hf_config) -> None:
        model_type = (getattr(hf_config, "model_type", "") or "").lower()
        if model_type == "qwen3":
            for i, layer in enumerate(self.model.layers):
                if layer.self_attn.q_norm is None or layer.self_attn.k_norm is None:
                    raise RuntimeError(
                        f"layer {i} was built without q_norm/k_norm although the checkpoint is Qwen3 "
                        "(serve/model/model.py keys this on the model path containing 'qwen3')"
                    )
        if getattr(self.config, "high_freq_factor", None) is not None:
            raise NotImplementedError("rope scaling variants are not supported by the engine")

    def _retie_embeddings(self) -> None:
        if not getattr(self.config, "tie_word_embeddings", False):
            return
        embed, head = self.model.embed_tokens.weight, self.model.lm_head.weight
        if embed.data_ptr() == head.data_ptr():
            return
        if embed.shape != head.shape or not bool(torch.equal(embed, head)):
            raise RuntimeError("tie_word_embeddings=True but lm_head.weight != embed_tokens.weight after loading")
        # Done before any CUDA graph is captured, so no captured pointer goes stale.
        self.model.lm_head.weight = self.model.embed_tokens.weight
        del head
        torch.cuda.empty_cache()
        logger.info("re-tied lm_head.weight to embed_tokens.weight (one shared storage)")

    def _build_runner(self) -> None:
        """Fresh CUDAGraphModelWrapper; captures graphs against the current KV tensor."""
        start = time.perf_counter()
        self.kv_pool.phase_idx = 0
        runner = CUDAGraphModelWrapper(self.model)
        self._graph_pool_bytes = 0
        if self.cuda_graph:
            torch.cuda.synchronize(self.device)
            # torch 2.7 / 2.8: torch.cuda.graph.__enter__ runs an unconditional
            # full gc.collect() for each of the 24 captures. Collect once (this
            # also drops trainer garbage that may hold CUDA memory), then park
            # the surviving heap in the permanent generation so those passes
            # only scan objects created during capture.
            gc.collect()
            torch.cuda.empty_cache()
            reserved_before = torch.cuda.memory_reserved(self.device)
            allocated_before = torch.cuda.memory_allocated(self.device)
            gc.freeze()
            try:
                # identical call to run_inference
                runner.eager_cuda_graph_mode(
                    max_batch_size=int(self.max_batch_size * 2.5),
                    num_cuda_graphs=14,
                    max_padded_ratio=1,
                )
            finally:
                gc.unfreeze()
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            # Tensors freed during capture stay reserved in each graph's
            # private pool but no longer count as allocated.
            reserved = torch.cuda.memory_reserved(self.device) - reserved_before
            allocated = torch.cuda.memory_allocated(self.device) - allocated_before
            self._graph_pool_bytes = max(0, reserved - allocated)
            self._measured["graph_reserved_gb"] = reserved / GIB
            self._measured["graph_pool_unallocated_gb"] = self._graph_pool_bytes / GIB
        self._runner = runner
        self._wake_stats["graph_capture_s"] = time.perf_counter() - start if self.cuda_graph else 0.0

    # ------------------------------------------------------------------
    # sleep / wake
    # ------------------------------------------------------------------
    @property
    def is_sleeping(self) -> bool:
        return self._sleeping

    @property
    def last_wake_stats(self) -> dict[str, float]:
        return dict(self._wake_stats)

    def sleep(self) -> None:
        """Free the GPU KV pool and the CUDA graphs captured against it.

        Weights and the fixed kernel buffers stay on the GPU (``load_weights``
        writes into the same parameter storage on every step). Per-request
        buffers are already gone: every request is retired by the end of
        ``generate``.
        """
        self._require_open()
        if self._sleeping:
            return
        gc.collect()
        before = torch.cuda.memory_allocated(self.device)
        self._runner = None  # drops the graphs, their private pools and output buffers
        gc.collect()
        self._check_pool_clean("sleep")
        self.kv_pool._physical_mem_gpu = None
        gc.collect()
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        freed = before - torch.cuda.memory_allocated(self.device)
        self._resident_bytes -= freed
        self._sleeping = True
        # the graphs' private pools go away with the graphs
        self._measured["sleep_freed_gb"] = (freed + self._graph_pool_bytes) / GIB
        self._graph_pool_bytes = 0
        self._measured["resident_asleep_gb"] = self._resident_bytes / GIB
        logger.info(
            "SparseSpecEngine asleep: freed %.2f GiB, %.2f GiB still resident (weights + fixed buffers)",
            self._measured["sleep_freed_gb"],
            self._resident_bytes / GIB,
        )

    def wake_up(self) -> None:
        """Re-allocate the GPU KV pool and re-capture the CUDA graphs.

        The old graphs recorded raw pointers into the freed KV tensor, so they
        are never reused: a new ``CUDAGraphModelWrapper`` is captured against
        the new tensor (timed as ``graph_capture_s``).
        """
        self._require_open()
        if not self._sleeping:
            return
        start = time.perf_counter()
        gc.collect()
        before = torch.cuda.memory_allocated(self.device)
        with self._ctx():
            self.kv_pool._physical_mem_gpu = torch.empty(self._kv_shape, dtype=self.dtype, device=self.device)
            self._sleeping = False
            try:
                self._build_runner()
            except BaseException:
                self._runner = None
                self.kv_pool._physical_mem_gpu = None
                self._sleeping = True
                raise
        torch.cuda.synchronize(self.device)
        self._resident_bytes += torch.cuda.memory_allocated(self.device) - before
        self._wake_stats["wake_s"] = time.perf_counter() - start
        self._measured["resident_awake_gb"] = (self._resident_bytes + self._graph_pool_bytes) / GIB

    def _check_pool_clean(self, where: str) -> None:
        """Every KV page must be back in the free lists between generate calls."""
        pool = self.kv_pool
        gpu_ok = pool.num_free_pages == pool.capacity
        cpu_ok = len(pool._free_cpu) == pool._MAX_CPU_CAPACITY
        if gpu_ok and cpu_ok:
            return
        logger.error(
            "KV pool not clean at %s: %d/%d GPU pages free, %d/%d host pages free; resetting the allocators",
            where,
            pool.num_free_pages,
            pool.capacity,
            len(pool._free_cpu),
            pool._MAX_CPU_CAPACITY,
        )
        pool._free_gpu = set(range(pool.capacity))
        pool._free_cpu = array.array("i", range(pool._MAX_CPU_CAPACITY))
        pool._cpu_page_free = bytearray(b"\x01") * pool._MAX_CPU_CAPACITY

    # ------------------------------------------------------------------
    # weights
    # ------------------------------------------------------------------
    def load_weights(self, named_tensors: NamedTensors) -> LoadReport:
        """In-place reload; valid asleep or awake (captured graphs keep working
        because no parameter tensor is reassigned)."""
        self._require_open()
        with torch.inference_mode():
            report = load_weights_inplace(self.model, named_tensors)
        torch.cuda.synchronize(self.device)
        return report

    def verify_weights(self, named_tensors: NamedTensors) -> dict:
        with torch.inference_mode():
            return verify_weights(self.model, named_tensors)

    def parameter_pointers(self) -> dict[str, int]:
        return parameter_pointers(self.model)

    # ------------------------------------------------------------------
    # generation
    # ------------------------------------------------------------------
    def _job_args(self, temperature: float, eos: Sequence[int], num_requests: int) -> dict:
        args = {
            # same keys run_inference passes to the scheduler
            "model_name": self.model_path,
            "max_batch_size": self.max_batch_size,
            "num_requests": num_requests,
            "admit_policy": "offloading",
            "tp_ranks": 1,
            "kv_cache": self.kv_cache,
            "eos_token_id": list(eos),
            "enable_wandb": False,
            "async_cpu": False,
            "enable_cuda_graph": self.cuda_graph,
            "chunk_size": self.chunk_size,
            "batch_policy": self.batch_policy,
            "trace_name": "rl",
            "temperature": float(temperature),
            "enable_torch_profiler": False,
            "profiler_result_dir": None,
        }
        args.update(self._spec_args)
        return args

    def generate(
        self,
        prompt_token_ids: Sequence[Sequence[int]],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        *,
        eos_token_ids=None,
    ) -> GenerationResult:
        self._require_open()
        if self._sleeping:
            raise RuntimeError("SparseSpecEngine is asleep; call wake_up() before generate()")
        max_new_tokens = int(max_new_tokens)
        validate_prompts(prompt_token_ids, max_new_tokens)
        validate_sampling_params(temperature, top_p, top_k)
        eos = normalize_eos_ids(eos_token_ids) or self._default_eos
        if not eos:
            raise ValueError("no EOS token ids: pass eos_token_ids or use a checkpoint with a generation config")
        eos_set = frozenset(eos)

        prompts = [[int(t) for t in p] for p in prompt_token_ids]
        vocab = self.config.vocab_size
        for i, p in enumerate(prompts):
            if min(p) < 0 or max(p) >= vocab:
                raise ValueError(f"prompt {i} contains a token id outside [0, {vocab})")
        prompt_lens = [len(p) for p in prompts]
        check_workload_fits(
            prompt_lens=prompt_lens,
            max_new_tokens=max_new_tokens,
            spec_stride=self.spec_stride,
            chunk_size=self.chunk_size,
            max_batch_size=self.max_batch_size,
            plan=self.plan,
            max_position_embeddings=getattr(self.config, "max_position_embeddings", None),
            limits=self.limits,
        )
        admit_cap = admission_capacity_tokens(self.plan, self.max_batch_size, self.spec_stride, self.chunk_size)
        self._warn_inflight_cap(admit_cap, max(prompt_lens), max_new_tokens, len(prompts))
        longest_ctx = max(prompt_lens) + max_new_tokens + self.spec_stride + 2
        if max(self.num_min_budget, math.ceil(longest_ctx * self.budget_ratio)) >= self.limits.max_len_selected_kv_cache:
            raise ValueError(
                f"budget_ratio={self.budget_ratio} x context {longest_ctx} exceeds the selected-KV buffer "
                f"({self.limits.max_len_selected_kv_cache} entries)"
            )

        start = time.perf_counter()
        requests: list[Request] = []
        scheduler = None
        try:
            with self._ctx():
                # reset_peak_memory_stats is device-global and verl's
                # perf/max_memory_* read the same counters: remember what the
                # reset destroys (published as process_peak_*_gb)
                self._process_peak_alloc = max(self._process_peak_alloc, torch.cuda.max_memory_allocated(self.device))
                self._process_peak_reserved = max(
                    self._process_peak_reserved, torch.cuda.max_memory_reserved(self.device)
                )
                torch.cuda.reset_peak_memory_stats(self.device)
                allocated_start = torch.cuda.memory_allocated(self.device)
                # A request completes at the first verify pass after it holds
                # max_new_tokens response tokens (see RLSpecScheduler); the
                # few extra tokens of that pass are cut below.
                requests = [Request(token_ids=p, desired_output_length=max_new_tokens - 1) for p in prompts]
                scheduler = RLSpecScheduler(
                    self.kv_pool,
                    self._runner,
                    total_token_capacity=admit_cap,
                    **self._job_args(temperature, eos, len(prompts)),
                )
                finished = scheduler.run(list(requests))
                torch.cuda.synchronize(self.device)
                peak = torch.cuda.max_memory_allocated(self.device)
                self._process_peak_alloc = max(self._process_peak_alloc, peak)
                self._process_peak_reserved = max(
                    self._process_peak_reserved, torch.cuda.max_memory_reserved(self.device)
                )
            if len(finished) != len(requests):
                raise RuntimeError(f"scheduler returned {len(finished)} of {len(requests)} requests")

            token_ids, logprobs, reasons = [], [], []
            for request in requests:
                response = list(request.token_ids[request.len_input :])
                toks, lps, reason = finalize_response(response, request.rl_logprobs, max_new_tokens, eos_set)
                if not toks:
                    raise RuntimeError(f"request {request.request_id} produced an empty response")
                if reason == FINISH_LENGTH and len(toks) != max_new_tokens:
                    raise RuntimeError(
                        f"request {request.request_id} stopped after {len(toks)} tokens without an EOS "
                        f"(max_new_tokens={max_new_tokens})"
                    )
                token_ids.append(toks)
                logprobs.append(lps)
                reasons.append(reason)

            rl = scheduler.rl
            stats = empty_stats()
            stats.update(
                num_requests=float(len(requests)),
                num_tokens=float(sum(len(t) for t in token_ids)),
                num_cycles=float(rl.num_cycles),
                draft_tokens=float(rl.draft_tokens),
                accepted_draft_tokens=float(rl.accepted_draft_tokens),
                accept_len_mean=(rl.emitted_tokens / rl.num_cycles) if rl.num_cycles else 0.0,
                accept_rate=(rl.accepted_draft_tokens / rl.draft_tokens) if rl.draft_tokens else 0.0,
                num_iters=float(rl.num_iters),
                batch_size_mean=(rl.sum_active / rl.num_iters) if rl.num_iters else 0.0,
                num_offloads=float(rl.num_offloads),
                kv_util_peak=rl.peak_kv_used / self.kv_pool.capacity,
                # peak of everything allocated during generate, minus what the
                # rest of the process (the trainer) already held at the start,
                # plus what the CUDA graphs' private pools reserve unallocated
                peak_mem_gb=(peak - (allocated_start - self._resident_bytes) + self._graph_pool_bytes) / GIB,
                process_peak_alloc_gb=self._process_peak_alloc / GIB,
                process_peak_reserved_gb=self._process_peak_reserved / GIB,
                gen_s=time.perf_counter() - start,
            )
            if stats["peak_mem_gb"] * GIB > self.plan.budget_bytes:
                logger.warning(
                    "engine peak memory %.2f GiB exceeded its share %.2f GiB during generate()",
                    stats["peak_mem_gb"],
                    self.plan.budget_bytes / GIB,
                )
            self._measured["last_peak_gb"] = stats["peak_mem_gb"]
            return GenerationResult(token_ids=token_ids, logprobs=logprobs, finish_reasons=reasons, stats=stats)
        finally:
            # Drop every request / cache pointer so all KV pages return to the
            # pool (also after an exception), then make sure they did.
            scheduler = None
            finished = None
            for request in requests:
                request.kv_cache_ptr = None
                request.draft_probs = None
            requests.clear()
            gc.collect()
            self._check_pool_clean("end of generate")

    # ------------------------------------------------------------------
    # misc
    # ------------------------------------------------------------------
    def inflight_cap(self, longest_prompt: int, max_new_tokens: int) -> int:
        """Most requests the admission guard lets run at once for this workload."""
        admit_cap = admission_capacity_tokens(self.plan, self.max_batch_size, self.spec_stride, self.chunk_size)
        need = request_token_need(longest_prompt, max_new_tokens, self.spec_stride)
        return min(self.max_batch_size, admit_cap // need)

    def _warn_inflight_cap(self, admit_cap: int, longest_prompt: int, max_new_tokens: int, num_prompts: int) -> None:
        need = request_token_need(longest_prompt, max_new_tokens, self.spec_stride)
        cap = admit_cap // need
        key = (max_new_tokens, cap)
        if cap >= min(self.max_batch_size, num_prompts) or key in self._warned_inflight_cap:
            return
        self._warned_inflight_cap.add(key)
        plan = self.plan
        reserve = plan.total_capacity_tokens - admit_cap
        ratio = (self.max_batch_size * need + reserve - plan.kv_capacity_tokens) / plan.kv_capacity_tokens
        logger.warning(
            "worst-case KV admission limits concurrency to %d in-flight requests although max_batch_size is %d: "
            "each request may need %d KV tokens and the pools hold %d (GPU) + %d (host). Realised concurrency is "
            "logged as sparsespec/batch_size_mean. sparsespec.cpu_offload_ratio >= %.1f (about %.0f GiB of pinned "
            "host RAM, locked in power-of-two blocks) would lift the cap to max_batch_size.",
            cap,
            self.max_batch_size,
            need,
            plan.kv_capacity_tokens,
            plan.cpu_capacity_tokens,
            ratio,
            ratio * plan.kv_capacity_tokens * plan.kv_token_bytes / GIB,
        )

    def memory_report(self) -> dict:
        """Planned vs measured memory, for logs and STATUS.md."""
        plan = self.plan
        return {
            "budget_gb": plan.budget_bytes / GIB,
            "planned_gb": {
                "weights": plan.weights_bytes / GIB,
                "fixed": plan.fixed_bytes / GIB,
                "graphs": plan.graph_bytes / GIB,
                "requests": plan.request_bytes / GIB,
                "transient": plan.transient_bytes / GIB,
                "safety": plan.safety_bytes / GIB,
                "kv": plan.kv_bytes / GIB,
            },
            "kv_capacity_tokens": plan.kv_capacity_tokens,
            "cpu_capacity_tokens": plan.cpu_capacity_tokens,
            "host_pinned_gb": plan.cpu_pinned_bytes / GIB,
            "measured": dict(self._measured),
            "notes": list(plan.notes),
        }

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("SparseSpecEngine is closed")

    def close(self) -> None:
        if self._closed:
            return
        self._runner = None
        if getattr(self, "kv_pool", None) is not None:
            self.kv_pool._physical_mem_gpu = None
        self._sleeping = True
        self._closed = True
        gc.collect()
        torch.cuda.empty_cache()
