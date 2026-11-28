"""Reference rollout engine backed by HuggingFace transformers.

Same interface and response contract as ``SparseSpecEngine`` (see
``serve/rl/base.py``), no speculation: every token is sampled from the full
model and the reported log-prob is ``log_softmax(logits / temperature)`` of
that same forward pass. It exists to test the verl plumbing (rollout class,
sharding manager, weight sync, metrics, run script) without flashinfer, and
as a second opinion in the correctness gates. It is slow; it is not a
baseline for timing.

Needs only torch + transformers and also runs on CPU (used by the unit tests).
"""

from __future__ import annotations

import logging
import time
from typing import Sequence

import torch

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
from serve.rl.weights import load_weights_inplace, parameter_pointers, unique_parameter_bytes, verify_weights

logger = logging.getLogger(__name__)

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class ReferenceEngine(RolloutEngine):
    name = "reference"

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
        eos_token_ids=None,
        model=None,
        **unused,
    ):
        """Same positional signature as ``SparseSpecEngine``.

        The speculation parameters are accepted and ignored. ``mem_fraction``
        bounds the micro-batch size on CUDA (the HF KV cache of a batch must
        fit in the share). ``model`` lets tests inject an already-built module.
        """
        from transformers import AutoModelForCausalLM

        if unused:
            logger.info("ReferenceEngine ignores SparseSpec-only options: %s", sorted(unused))
        self.model_path = str(model_path)
        self.mem_fraction = float(mem_fraction)
        self.max_batch_size = int(max_batch_size)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.dtype = _DTYPES[dtype]
        self._closed = False
        self._wake_stats = {"wake_s": 0.0, "graph_capture_s": 0.0}

        if model is None:
            try:
                model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=self.dtype)
            except TypeError:  # newer transformers renamed the argument
                model = AutoModelForCausalLM.from_pretrained(self.model_path, dtype=self.dtype)
        # Move, then cast PARAMETERS only: Module.to(dtype=...) would also
        # round float buffers such as Qwen3's rotary inv_freq, which
        # from_pretrained (and therefore vLLM and the verl actor) keep in fp32.
        model = model.to(self.device)
        for param in model.parameters():
            if param.dtype != self.dtype:
                param.data = param.data.to(self.dtype)
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.config = self.model.config

        # seeded once; the stream continues across generate() calls and is
        # independent of the global (trainer) RNG
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(self.seed)

        self._default_eos = normalize_eos_ids(eos_token_ids) or self._eos_from_model()
        self._sleeping = False
        # running process-wide peaks (generate() resets torch's counters)
        self._process_peak_alloc = 0
        self._process_peak_reserved = 0

    # ------------------------------------------------------------------
    def _eos_from_model(self) -> tuple[int, ...]:
        eos: list = []
        gen = getattr(self.model, "generation_config", None)
        if gen is not None:
            eos.extend(normalize_eos_ids(gen.eos_token_id))
        eos.extend(normalize_eos_ids(getattr(self.config, "eos_token_id", None)))
        return normalize_eos_ids(eos)

    def _kv_token_bytes(self) -> int:
        cfg = self.config
        heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        return cfg.num_hidden_layers * heads * head_dim * 2 * torch.finfo(self.dtype).bits // 8

    def _micro_batch_size(self, longest_prompt: int, max_new_tokens: int) -> int:
        """Largest batch whose KV cache fits in the engine's memory share."""
        if self.device.type != "cuda":
            return self.max_batch_size
        total = torch.cuda.get_device_properties(self.device).total_memory
        weights = unique_parameter_bytes(self.model.parameters())
        # 2 GiB for logits / softmax scratch. x2 on the KV estimate: HF's
        # DynamicCache re-allocates every K/V tensor each step, and the
        # allocator can strand one outgrown generation of cache segments
        # between the trims in _generate_batch (live memory is only ~1.07x).
        kv_budget = total * self.mem_fraction - weights - 2 * 1024**3
        per_request = (longest_prompt + max_new_tokens) * self._kv_token_bytes() * 2
        fit = int(kv_budget // per_request) if kv_budget > 0 else 0
        return max(1, min(self.max_batch_size, fit))

    # ------------------------------------------------------------------
    @property
    def is_sleeping(self) -> bool:
        return self._sleeping

    @property
    def last_wake_stats(self) -> dict[str, float]:
        return dict(self._wake_stats)

    def sleep(self) -> None:
        """No persistent KV pool to free: generation caches die with each
        micro-batch. Keeps the same state machine as SparseSpecEngine so
        ordering bugs in the verl glue show up with this engine too."""
        self._require_open()
        if self._sleeping:
            return
        self._sleeping = True
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def wake_up(self) -> None:
        self._require_open()
        if not self._sleeping:
            return
        start = time.perf_counter()
        self._sleeping = False
        self._wake_stats = {"wake_s": time.perf_counter() - start, "graph_capture_s": 0.0}

    # ------------------------------------------------------------------
    def load_weights(self, named_tensors: NamedTensors) -> LoadReport:
        self._require_open()
        with torch.no_grad():
            return load_weights_inplace(self.model, named_tensors)

    def verify_weights(self, named_tensors: NamedTensors) -> dict:
        with torch.no_grad():
            return verify_weights(self.model, named_tensors)

    def parameter_pointers(self) -> dict[str, int]:
        return parameter_pointers(self.model)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _generate_batch(
        self,
        prompts: list[list[int]],
        max_new_tokens: int,
        temperature: float,
        eos_set: frozenset,
    ) -> tuple[list[list[int]], list[list[float]], int]:
        device = self.device
        batch = len(prompts)
        longest = max(len(p) for p in prompts)
        input_ids = torch.zeros((batch, longest), dtype=torch.long, device=device)
        attention_mask = torch.zeros((batch, longest), dtype=torch.long, device=device)
        for i, p in enumerate(prompts):  # left padding, like verl's prompts
            input_ids[i, longest - len(p) :] = torch.tensor(p, dtype=torch.long, device=device)
            attention_mask[i, longest - len(p) :] = 1
        # same convention as verl.utils.model.compute_position_id_with_mask
        position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0)
        next_position = position_ids[:, -1:] + 1

        eos_tensor = torch.tensor(sorted(eos_set), dtype=torch.long, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        tokens: list[torch.Tensor] = []
        logps: list[torch.Tensor] = []
        past = None
        step_input, step_positions = input_ids, position_ids
        forwards = 0

        for _ in range(max_new_tokens):
            out = self.model(
                input_ids=step_input,
                attention_mask=attention_mask,
                position_ids=step_positions,
                past_key_values=past,
                use_cache=True,
                logits_to_keep=1,  # only the last position is sampled
            )
            forwards += 1
            past = out.past_key_values
            if device.type == "cuda" and forwards % 16 == 0:
                # Once the per-step re-allocated K/V tensors outgrow their
                # allocator segments, the old segments stay cached and can
                # never be reused; hand them back when they pile up.
                if torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device) > 2 * 1024**3:
                    torch.cuda.empty_cache()
            logits = out.logits[:, -1, :].to(torch.float32)
            if temperature == 0.0:
                next_token = logits.argmax(dim=-1)
                next_logp = torch.zeros(batch, dtype=torch.float32, device=device)
            else:
                logp = torch.log_softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(logp.exp(), num_samples=1, generator=self._generator).squeeze(-1)
                next_logp = logp.gather(-1, next_token.unsqueeze(-1)).squeeze(-1)
            tokens.append(next_token)
            logps.append(next_logp)
            finished |= torch.isin(next_token, eos_tensor)
            if bool(finished.all()):
                break
            step_input = next_token.unsqueeze(-1)
            step_positions = next_position
            next_position = next_position + 1
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((batch, 1))], dim=-1)

        token_mat = torch.stack(tokens, dim=1).cpu().tolist()
        logp_mat = torch.stack(logps, dim=1).double().cpu().tolist()
        return token_mat, logp_mat, forwards

    def generate(
        self,
        prompt_token_ids: Sequence[Sequence[int]],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        *,
        eos_token_ids=None,
        return_logprobs: bool = True,
    ) -> GenerationResult:
        self._require_open()
        if self._sleeping:
            raise RuntimeError("ReferenceEngine is asleep; call wake_up() before generate()")
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

        start = time.perf_counter()
        if self.device.type == "cuda":
            # same bookkeeping as SparseSpecEngine.generate (see STAT_KEYS)
            self._process_peak_alloc = max(self._process_peak_alloc, torch.cuda.max_memory_allocated(self.device))
            self._process_peak_reserved = max(self._process_peak_reserved, torch.cuda.max_memory_reserved(self.device))
            torch.cuda.reset_peak_memory_stats(self.device)
            allocated_start = torch.cuda.memory_allocated(self.device)

        # batch similar lengths together; results go back in the caller's order
        order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
        token_ids: list = [None] * len(prompts)
        logprobs: list = [None] * len(prompts)
        reasons: list = [None] * len(prompts)
        forwards = 0
        rows = 0
        cursor = 0
        batch_limit = 0  # largest micro-batch the memory share allowed
        batch_peak = 0  # largest micro-batch actually run
        while cursor < len(order):
            size = self._micro_batch_size(len(prompts[order[cursor]]), max_new_tokens)
            chunk = order[cursor : cursor + size]
            cursor += size
            batch_limit = max(batch_limit, size)
            batch_peak = max(batch_peak, len(chunk))
            toks, lps, n_fwd = self._generate_batch(
                [prompts[i] for i in chunk], max_new_tokens, float(temperature), eos_set
            )
            forwards += n_fwd
            rows += n_fwd * len(chunk)
            for i, row_toks, row_lps in zip(chunk, toks, lps):
                token_ids[i], logprobs[i], reasons[i] = finalize_response(row_toks, row_lps, max_new_tokens, eos_set)
                if reasons[i] == FINISH_LENGTH and len(token_ids[i]) != max_new_tokens:
                    raise RuntimeError(
                        f"prompt {i} stopped after {len(token_ids[i])} tokens without an EOS "
                        f"(max_new_tokens={max_new_tokens})"
                    )
            del toks, lps
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        stats = empty_stats()
        stats.update(
            num_requests=float(len(prompts)),
            num_tokens=float(sum(len(t) for t in token_ids)),
            num_iters=float(forwards),
            batch_size_mean=(rows / forwards) if forwards else 0.0,
            gen_s=time.perf_counter() - start,
        )
        if self.device.type == "cuda":
            weights = unique_parameter_bytes(self.model.parameters())
            peak = torch.cuda.max_memory_allocated(self.device)
            peak_reserved = torch.cuda.max_memory_reserved(self.device)
            self._process_peak_alloc = max(self._process_peak_alloc, peak)
            self._process_peak_reserved = max(self._process_peak_reserved, peak_reserved)
            stats["peak_mem_gb"] = (peak - (allocated_start - weights)) / 1024**3
            stats["process_peak_alloc_gb"] = self._process_peak_alloc / 1024**3
            stats["process_peak_reserved_gb"] = self._process_peak_reserved / 1024**3
            logger.info(
                "ReferenceEngine.generate: peak allocated %.2f GiB, peak reserved %.2f GiB (whole process)",
                peak / 1024**3,
                peak_reserved / 1024**3,
            )
        stats["inflight_cap"] = float(batch_limit)
        stats["inflight_peak"] = float(batch_peak)
        return GenerationResult(
            token_ids=token_ids,
            logprobs=logprobs if return_logprobs else None,
            finish_reasons=reasons,
            stats=stats,
        )

    # ------------------------------------------------------------------
    def memory_report(self) -> dict:
        return {"engine": self.name, "weights_gb": unique_parameter_bytes(self.model.parameters()) / 1024**3}

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("ReferenceEngine is closed")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._sleeping = True
        self.model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
