"""Engine-agnostic pieces of the RL rollout interface.

Pure Python/torch: importable without flashinfer, triton or CUDA, so the
reference engine, the verl glue and the CPU unit tests can share it.

The contract both engines implement (``ReferenceEngine`` and
``SparseSpecEngine``):

* ``generate`` returns, per prompt, the response token ids, the log-probability
  of every returned token under the TARGET (full-attention) distribution at the
  requested temperature (``log_softmax(logits / temperature)``, the same scaling
  verl's actor applies), and a stats dict.
* A response ends with the first EOS token it contains (the EOS is KEPT, as
  stock vLLM does) or is cut to exactly ``max_new_tokens``.
* ``load_weights`` copies new values into the existing parameters in place.
* ``sleep`` / ``wake_up`` release / re-acquire the rollout-only GPU memory.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence, Union

import torch

NamedTensors = Union[Mapping[str, torch.Tensor], Iterable[tuple[str, torch.Tensor]]]

FINISH_STOP = "stop"
FINISH_LENGTH = "length"

# Every engine reports exactly these keys (zeros where a quantity does not
# apply), so downstream metric code never has to branch on the engine type.
STAT_KEYS = (
    "num_requests",
    "num_tokens",  # response tokens returned to the caller
    "num_cycles",  # verify cycles (0 for the reference engine)
    "draft_tokens",  # draft tokens submitted to verification
    "accepted_draft_tokens",  # draft tokens accepted by verification
    "accept_len_mean",  # tokens emitted per verify cycle = accepted + 1
    "accept_rate",  # accepted_draft_tokens / draft_tokens
    "num_iters",  # engine forward passes
    "batch_size_mean",  # mean number of requests per forward pass
    "num_offloads",  # KV offload events (GPU -> host)
    "kv_util_peak",  # peak fraction of the GPU KV pool in use
    "host_util_peak",  # peak fraction of the host offload pool in use
    # In-flight requests the admission policy guarantees room for, sized for
    # the call's longest prompt. A hard limit under host_guard; worst_case
    # budgets tokens per request, so shorter prompts can push inflight_peak
    # slightly above it.
    "inflight_cap",
    "inflight_peak",  # most requests in flight at once (running + offloaded)
    "peak_mem_gb",  # peak GPU memory attributed to the engine during generate
    # Running process-wide peaks of torch's allocator up to the end of this
    # generate(). generate() resets torch's peak counters to measure
    # peak_mem_gb, so in this arm verl's perf/max_memory_* become per-step
    # peaks (generate() start -> update_actor); the process peak at step N is
    # max(process_peak_*_gb[N], perf/max_memory_*_gb[N]).
    "process_peak_alloc_gb",
    "process_peak_reserved_gb",
    "gen_s",  # wall time of generate()
)


def empty_stats() -> dict[str, float]:
    return {k: 0.0 for k in STAT_KEYS}


@dataclass
class GenerationResult:
    """Output of ``RolloutEngine.generate`` (plain Python containers only).

    ``token_ids[i]`` and ``logprobs[i]`` have the same length; ``logprobs`` is
    None when the caller passed ``return_logprobs=False``. Plain lists are
    returned rather than tensors so nothing created under
    ``torch.inference_mode`` leaks into the trainer's autograd graph.
    """

    token_ids: list[list[int]]
    logprobs: list[list[float]] | None
    finish_reasons: list[str]
    stats: dict[str, float] = field(default_factory=empty_stats)


@dataclass
class LoadReport:
    num_params: int  # engine parameters written
    num_sources: int  # source tensors consumed
    num_bytes: int  # bytes written into engine parameters
    seconds: float


def normalize_eos_ids(eos_token_ids) -> tuple[int, ...]:
    """Accept an int, a list (as verl passes ``generation_config.eos_token_id``)
    or None entries, and return a sorted tuple of unique ints."""
    if eos_token_ids is None:
        return ()
    if isinstance(eos_token_ids, int):
        return (int(eos_token_ids),)
    out = sorted({int(t) for t in eos_token_ids if t is not None})
    return tuple(out)


def validate_sampling_params(temperature: float, top_p: float, top_k: int) -> None:
    """Both engines sample from the plain temperature-scaled distribution.

    The SparseSpec sampler (serve/sampling/sampler.py) has no top-p / top-k
    path (it asserts both are None), and applying them to only one of p/q
    would break the speculative-sampling guarantee, so anything other than
    "disabled" is rejected instead of silently ignored.
    """
    if temperature is None or temperature < 0.0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if top_p is not None and float(top_p) != 1.0:
        raise NotImplementedError(
            f"top_p={top_p} is not supported: the SparseSpec sampler implements "
            "temperature sampling only (top_p must be 1.0)"
        )
    if top_k is not None and int(top_k) not in (-1, 0):
        raise NotImplementedError(
            f"top_k={top_k} is not supported: the SparseSpec sampler implements "
            "temperature sampling only (top_k must be -1 or 0)"
        )


def validate_prompts(prompt_token_ids: Sequence[Sequence[int]], max_new_tokens: int) -> None:
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
    if len(prompt_token_ids) == 0:
        raise ValueError("generate() needs at least one prompt")
    for i, p in enumerate(prompt_token_ids):
        if len(p) == 0:
            raise ValueError(f"prompt {i} is empty")


def first_eos_index(tokens: Sequence[int], eos_ids, start: int = 0) -> int:
    """Index of the first EOS token at or after ``start``; -1 if there is none."""
    for pos in range(start, len(tokens)):
        if tokens[pos] in eos_ids:
            return pos
    return -1


def finalize_response(
    tokens: Sequence[int],
    logprobs: Sequence[float] | None,
    max_new_tokens: int,
    eos_ids,
) -> tuple[list[int], list[float] | None, str]:
    """Apply the response contract: stop at the first EOS (kept), else cut to
    ``max_new_tokens``. Token ids and log-probs are trimmed together;
    ``logprobs=None`` (not collected) stays None."""
    if logprobs is not None and len(tokens) != len(logprobs):
        raise RuntimeError(
            f"engine bookkeeping error: {len(tokens)} response tokens but "
            f"{len(logprobs)} target log-probs"
        )
    tokens = list(tokens[:max_new_tokens])
    if logprobs is not None:
        logprobs = list(logprobs[:max_new_tokens])
    eos_pos = first_eos_index(tokens, eos_ids)
    if eos_pos >= 0:
        tokens = tokens[: eos_pos + 1]
        return tokens, (None if logprobs is None else logprobs[: eos_pos + 1]), FINISH_STOP
    return tokens, logprobs, FINISH_LENGTH


class RolloutEngine(abc.ABC):
    """Interface shared by ``ReferenceEngine`` and ``SparseSpecEngine``."""

    #: name used in logs / metrics
    name: str = "engine"

    @abc.abstractmethod
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
        """Sample one response per prompt. ``temperature == 0`` means greedy
        (reported log-probs are then 0.0: the greedy target is one-hot).
        With ``return_logprobs=False`` the log-probs are not collected at all
        and ``GenerationResult.logprobs`` is None."""

    @abc.abstractmethod
    def load_weights(self, named_tensors: NamedTensors) -> LoadReport:
        """Overwrite every parameter in place from HF-named tensors."""

    @abc.abstractmethod
    def sleep(self) -> None:
        """Release rollout-only GPU memory. Idempotent."""

    @abc.abstractmethod
    def wake_up(self) -> None:
        """Re-acquire what ``sleep`` released. Idempotent."""

    @property
    @abc.abstractmethod
    def is_sleeping(self) -> bool: ...

    @property
    def last_wake_stats(self) -> dict[str, float]:
        """Timing of the most recent wake_up (``wake_s``, ``graph_capture_s``)."""
        return {"wake_s": 0.0, "graph_capture_s": 0.0}

    def close(self) -> None:
        """Release everything. The engine is unusable afterwards."""
