"""SpecScheduler subclass for RL rollouts.

Needs CUDA + flashinfer + triton (it imports the real scheduler); only import
this from ``serve.rl.engine``.

What it adds on top of ``serve.scheduler.spec_scheduler.SpecScheduler``
(synchronous mode only), without changing how tokens are drafted, verified
or sampled:

1. Target log-probs. Every token that ends up in a response is distributed
   according to one row of the full-attention ("target") probability tensor
   of the forward pass that confirmed it:

   * the first response token comes from the NORMAL (full-attention) pass
     that finished prefill: row ``token_indptr[idx + 1] - 1``;
   * a verify pass feeds ``n + 1`` tokens (the last confirmed token and the
     ``n`` draft tokens; ``n == spec_stride`` except in a request's first
     cycle, which has ``spec_stride - initial_idx`` drafts) and its rows
     ``start + j`` are the target distributions for output position ``j``.
     The rejection sampler returns ``m`` accepted draft tokens followed by
     one recovered token (``m < n``, drawn from the normalized residual
     ``max(0, p - q)`` of row ``start + m``; its marginal law is that row)
     or the bonus token (``m == n``, sampled from row ``start + n`` by the
     regular sampler). So output token ``j`` always has target probability
     ``probs[start + j, token_j]``.

   Those probabilities are ``softmax(logits / temperature)`` in float32
   (``serve/sampling/fast_fused_op.py``), i.e. the same temperature scaling
   verl's actor uses, and they are the exact ``p`` the accept/reject test
   used. They are gathered on the GPU once per iteration.

2. The response contract of stock vLLM: the EOS token is kept as the last
   response token and nothing after it survives. Upstream strips the EOS
   after a verify pass, skips the EOS scan while fewer than ``spec_stride+1``
   tokens were generated, and looks at the last PROMPT token while a prompt
   is still being chunk-prefilled.

3. Admission control that cannot exhaust the KV pools (upstream would hit
   ``alloc_page_cpu``'s "Out of memory" assertion), in two modes:

   * ``worst_case`` (default): a request is admitted only if the worst-case
     KV demand of everything in flight (every request running to its length
     limit) still fits in GPU pool + host offload pool;
   * ``host_guard``: admit on current free capacity like upstream, but never
     more requests in flight than the host pool can absorb under upstream's
     own offload rules (shortest victim first, last offloaded restored
     first, no admission while anything is offloaded). The bound is
     ``serve.rl.memory.host_guard_host_bound``; it is a fixed in-flight cap
     per ``generate`` call, so no request is ever dropped, restarted or
     truncated, only admitted later.

   In both modes a request is admitted only in a scheduler call that starts
   with nothing offloaded, and only when the GPU pool has room to finish
   prefilling it without an offload in between. (Upstream may restore the
   last offloaded request and admit a new one in the same call; the room
   check would then be measured before the restore took its pages, so
   admission waits one iteration instead.) A request that is still
   prefilling is therefore never the offload victim.

4. Per-call statistics (verify cycles, acceptance, batch size, offloads).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from serve.request.request import Request
from serve.rl.base import FINISH_LENGTH, FINISH_STOP, first_eos_index
from serve.rl.memory import request_token_need
from serve.scheduler.spec_scheduler import SpecScheduler

logger = logging.getLogger(__name__)

_NORMAL = Request.ReqExecType.NORMAL
_DRAFT = Request.ReqExecType.DRAFT
_VERIFY = Request.ReqExecType.VERIFY


@dataclass
class SchedulerStats:
    num_cycles: int = 0  # verify passes (summed over requests)
    draft_tokens: int = 0  # draft tokens submitted to verification
    accepted_draft_tokens: int = 0  # draft tokens accepted
    emitted_tokens: int = 0  # tokens emitted by verify passes (accepted + recovered/bonus)
    normal_tokens: int = 0  # first response tokens, sampled in NORMAL mode
    num_iters: int = 0  # forward passes
    sum_active: int = 0  # sum over forward passes of the number of active requests
    num_offloads: int = 0  # requests pushed to the host offload pool
    peak_kv_used: int = 0  # peak number of GPU KV pages in use
    peak_host_used: int = 0  # peak number of host offload pages in use
    peak_inflight: int = 0  # most requests in flight at once (active + offloaded)


class RLSpecScheduler(SpecScheduler):
    def __init__(
        self,
        kv_pool,
        model,
        *,
        total_token_capacity: int,
        admission: str = "worst_case",
        inflight_cap: int | None = None,
        collect_logprobs: bool = True,
        **job_args,
    ):
        super().__init__(kv_pool=kv_pool, model=model, **job_args)
        if admission not in ("worst_case", "host_guard"):
            raise ValueError(f"admission must be 'worst_case' or 'host_guard', got {admission!r}")
        if admission == "host_guard" and (inflight_cap is None or inflight_cap < 1):
            raise ValueError("host_guard admission needs inflight_cap >= 1")
        self._rl_admission = admission
        self._rl_inflight_cap = inflight_cap
        self._rl_collect_logprobs_on = bool(collect_logprobs)
        if self.async_on:
            raise NotImplementedError(
                "RLSpecScheduler supports synchronous scheduling only (async_cpu=False): in async mode "
                "verification runs one iteration late against the previous pass' probabilities, and the "
                "target log-prob bookkeeping below has not been written or validated for that path"
            )
        if self.admit_policy != "offloading":
            raise NotImplementedError("RLSpecScheduler requires admit_policy='offloading'")
        self._rl_eos = frozenset(int(t) for t in self.eos_token_id)
        if not self._rl_eos:
            raise ValueError("at least one EOS token id is required")
        self._rl_total_capacity = int(total_token_capacity)
        self._rl_committed = 0  # worst-case KV tokens of all in-flight requests
        self._rl_verify_jobs: list[tuple[Request, int, list[int]]] = []
        self.rl = SchedulerStats()

    # ------------------------------------------------------------------
    # admission
    # ------------------------------------------------------------------
    def _rl_need(self, request: Request) -> int:
        # the engine sets len_output = max_new_tokens - 1
        return request_token_need(request.len_input, request.len_output + 1, self.spec_stride)

    def _rl_may_admit(self, request: Request) -> bool:
        if self._rl_admission == "host_guard":
            # Upstream only admits while nothing is offloaded, so this counts
            # the running requests; the cap bounds what they can ever push to
            # the host pool (see the module docstring).
            if len(self.active_requests) + len(self.offload_requests) + 1 > self._rl_inflight_cap:
                return False
        elif self._rl_committed + self._rl_need(request) > self._rl_total_capacity:
            return False
        if not self.active_requests and not self.offload_requests:
            # Nothing in flight: the whole GPU pool is free. The engine checked
            # up front that a lone request fits, so this guarantees progress.
            return True
        page = self.kv_pool.page_size
        free = self.kv_pool.num_free_pages * page
        owed = sum(r.len_token_ids - r.len_kv_cache for r in self.active_requests)
        # Room for the prompt plus what every active request can append while
        # the prompt is being chunk-prefilled, so no offload is needed (and
        # the half-prefilled request is never the offload victim) before it
        # leaves NORMAL mode.
        prefill_iters = -(-request.len_token_ids // self._args["chunk_size"]) + 1
        margin = (len(self.active_requests) + 1) * (self.spec_stride + 2) * prefill_iters
        return free - owed >= request.len_token_ids + margin

    def admit_requests(self, pending_requests) -> None:
        head = pending_requests[0] if pending_requests else None
        # Not while anything is offloaded at the start of the call: upstream
        # would first restore the top of the stack (up to 1024 GPU pages that
        # _rl_may_admit has not seen) and could then admit in the same call.
        allow = head is not None and not self.offload_requests and self._rl_may_admit(head)
        offloaded_before = len(self.offload_requests)
        # With an empty list upstream still runs its offload / restore logic
        # but cannot admit anything.
        super().admit_requests(pending_requests if allow else [])
        self.rl.num_offloads += max(0, len(self.offload_requests) - offloaded_before)
        if allow and (not pending_requests or pending_requests[0] is not head):
            head._rl_need = self._rl_need(head)
            self._rl_committed += head._rl_need
            head.rl_logprobs = []
            head.rl_finish_reason = None
        in_flight = len(self.active_requests) + len(self.offload_requests)
        if in_flight > self.rl.peak_inflight:
            self.rl.peak_inflight = in_flight
        host_used = self.kv_pool._MAX_CPU_CAPACITY - len(self.kv_pool._free_cpu)
        if host_used > self.rl.peak_host_used:
            self.rl.peak_host_used = host_used

    def add_request(self, request: Request, is_restore: bool = False) -> None:
        if is_restore and request.status == _NORMAL:
            # Safety net only. Upstream asserts an offloaded request is never
            # in NORMAL state, although its victim selection (shortest active
            # request) could pick one that is still prefilling. The admission
            # rule in admit_requests keeps that from happening (a request is
            # admitted with room to finish its prefill before any offload).
            # If it happens anyway, resuming prefill is safe for prompts up
            # to 1025 tokens (one restore step); do not crash a run over it,
            # but say so loudly.
            logger.warning(
                "restoring an offloaded request that is still prefilling (prompt %d tokens, %d in KV): "
                "the admission invariant was broken; please report this",
                request.len_input,
                request.len_kv_cache,
            )
            self.active_requests.append(request)
            return
        super().add_request(request, is_restore=is_restore)

    # ------------------------------------------------------------------
    # completion: EOS kept, nothing after it
    # ------------------------------------------------------------------
    def is_request_complete(self, request: Request, check_idx: int = -1) -> bool:
        assert check_idx == -1, "synchronous mode only"
        status = request.status
        if status == _NORMAL:
            if not request.prefilled:
                # Still (chunk-)prefilling: the last token is a prompt token.
                return False
            # Exactly one response token exists (NORMAL -> DRAFT happens right
            # after the first one).
            if request.token_ids[-1] in self._rl_eos:
                request.rl_finish_reason = FINISH_STOP
                return True
            if request.len_kv_cache >= request.len_oracle_ctx:
                request.rl_finish_reason = FINISH_LENGTH
                return True
            return False
        if status == _DRAFT:
            # draft tokens are provisional until verified
            return False
        if status == _VERIFY:
            # Called right after verify_outputs(): scan only the tokens that
            # pass emitted, earliest first.
            emitted = request._rl_last_emitted
            eos_pos = first_eos_index(request.token_ids, self._rl_eos, start=len(request.token_ids) - emitted)
            if eos_pos >= 0:
                del request.token_ids[eos_pos + 1 :]
                request.rl_finish_reason = FINISH_STOP
                return True
            if request.len_kv_cache >= request.len_oracle_ctx:
                request.rl_finish_reason = FINISH_LENGTH
                return True
            return False
        raise AssertionError(f"unexpected request status in synchronous mode: {status}")

    # ------------------------------------------------------------------
    # verification bookkeeping
    # ------------------------------------------------------------------
    def verify_outputs(self, request: Request, output_token_ids, **kwargs):
        emitted = len(output_token_ids)
        self.rl.num_cycles += 1
        self.rl.draft_tokens += request.len_draft_probs
        self.rl.accepted_draft_tokens += emitted - 1
        self.rl.emitted_tokens += emitted
        request._rl_last_emitted = emitted
        self._rl_verify_jobs.append((request, request._rl_verify_row, [int(t) for t in output_token_ids]))
        return super().verify_outputs(request, output_token_ids, **kwargs)

    def update_requests(self, token_indptr, gen_token_ids_gpu, gen_token_probs_gpu, input_metadata):
        self._rl_verify_jobs = []
        normal_jobs: list[tuple[Request, int]] = []
        if token_indptr is not None:
            self.rl.num_iters += 1
            self.rl.sum_active += len(self.active_requests)
            used = self.kv_pool.capacity - self.kv_pool.num_free_pages
            if used > self.rl.peak_kv_used:
                self.rl.peak_kv_used = used
            for idx, request in enumerate(self.active_requests):
                if request.status == _VERIFY:
                    # first row of this request's verify pass
                    request._rl_verify_row = token_indptr[idx]
                elif request.status == _NORMAL and request.len_kv_cache == request.len_token_ids:
                    # prefill finished in this pass: its last row samples the
                    # first response token (same condition upstream uses)
                    normal_jobs.append((request, token_indptr[idx + 1] - 1))

        completed = super().update_requests(token_indptr, gen_token_ids_gpu, gen_token_probs_gpu, input_metadata)

        self._rl_collect_logprobs(normal_jobs, gen_token_probs_gpu)
        self._rl_verify_jobs = []

        for request in completed:
            self._rl_committed -= request._rl_need
            num_response = request.len_token_ids - request.len_input
            if self._rl_collect_logprobs_on and len(request.rl_logprobs) != num_response:
                raise RuntimeError(
                    f"request {getattr(request, 'request_id', '?')}: {num_response} response tokens but "
                    f"{len(request.rl_logprobs)} target log-probs"
                )
        return completed

    def _rl_collect_logprobs(self, normal_jobs, gen_token_probs_gpu) -> None:
        """One GPU gather + one D2H copy for every token confirmed this pass."""
        if not self._rl_collect_logprobs_on:
            self.rl.normal_tokens += len(normal_jobs)
            return
        rows: list[int] = []
        toks: list[int] = []
        for request, row in normal_jobs:
            rows.append(int(row))
            toks.append(int(request.token_ids[request.len_input]))
        for _, start_row, out_tokens in self._rl_verify_jobs:
            rows.extend(int(start_row) + j for j in range(len(out_tokens)))
            toks.extend(out_tokens)
        if not rows:
            return

        device = gen_token_probs_gpu.device
        rows_t = torch.tensor(rows, dtype=torch.long, device=device)
        toks_t = torch.tensor(toks, dtype=torch.long, device=device)
        probs = gen_token_probs_gpu[rows_t, toks_t].to(torch.float64)
        logprobs = torch.log(probs).cpu().tolist()

        k = 0
        for request, _ in normal_jobs:
            self.rl.normal_tokens += 1
            request.rl_logprobs.append(logprobs[k])
            k += 1
        for request, _, out_tokens in self._rl_verify_jobs:
            request.rl_logprobs.extend(logprobs[k : k + len(out_tokens)])
            k += len(out_tokens)
            # All tokens are confirmed right after a verify pass, so the
            # response length is exact; it is shorter than what the pass
            # emitted only when tokens after an EOS were dropped.
            num_response = request.len_token_ids - request.len_input
            if len(request.rl_logprobs) > num_response:
                del request.rl_logprobs[num_response:]
            elif len(request.rl_logprobs) < num_response:
                raise RuntimeError(
                    f"request {getattr(request, 'request_id', '?')}: {num_response} confirmed response tokens "
                    f"but only {len(request.rl_logprobs)} target log-probs"
                )
