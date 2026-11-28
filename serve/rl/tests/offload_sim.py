"""Page-level CPU simulation of upstream's ``offloading`` admission policy.

``serve/scheduler/scheduler.py`` imports torch.cuda / triton at module level
and cannot be imported here, and nothing is stubbed. Instead this module is a
line-by-line TRANSCRIPTION of the code that decides when KV pages move between
the GPU pool and the host offload pool, with every assert kept:

* ``BaseScheduler.admit_requests``, ``admit_policy == "offloading"`` branch
  (serve/scheduler/scheduler.py l.194-287)
* ``SpecScheduler.add_request`` restore assert (serve/scheduler/spec_scheduler.py l.51-63)
* ``Request.MAX_OFFLOADING_LEN`` / ``offload`` / ``restore`` / ``prefilled``
  (serve/request/request.py l.13, l.70-83)
* ``FullCachePtr.allocate_tokens`` / ``retract_tokens`` / ``offload`` /
  ``restore`` / ``__del__`` (serve/request/kv_cache_ptr/base.py l.100-164),
  the cache of a request until its prefill is done; after that its cache is a
  ``PillarCachePtr`` (``from_kv_cache``, serve/request/kv_cache_ptr/pillar.py
  l.93-120) whose page handling is ``StreamingCachePtr.allocate_tokens`` /
  ``offload`` / ``restore`` (serve/request/kv_cache_ptr/streaming.py
  l.93-222: same pages, asserts in a different order; the selected-index
  bookkeeping there does not touch pages)
* ``KVPool.num_free_pages`` / ``alloc_page`` / ``free_page`` /
  ``alloc_page_cpu`` / ``free_page_cpu`` / ``offload`` / ``restore``
  (serve/model/model.py l.151-190), page_size 1
* ``RLSpecScheduler._rl_may_admit`` / ``admit_requests`` / ``add_request``
  (serve/rl/scheduler.py), both admission modes

test_offload_sim.py pins a hash of each of those upstream sources: if one
changes, the transcription (and the guarantee it validates) must be redone.

The parts of the scheduler loop that do not touch the admission decision
(forward pass, sampling, verification) are replaced by a model of what they
do to a request's KV length per iteration: chunked prefill of ``chunk_size``
tokens per iteration until the prompt is in (``SpecScheduler.
prepare_forward_inputs``, l.162-171), then one KV token per iteration
(``allocate_tokens(1)`` for DRAFT and VERIFY requests, l.158-171), optionally
with a verify pass every ``spec_stride + 1`` iterations that retracts the
rejected drafts (``verify_outputs``, l.287-290), and completion at the
adversary's chosen final KV length, which frees every page (``retire``).
Physical memory is modelled too: every GPU / host page holds a
``(request, position)`` tag, copied the way ``KVPool.offload`` /
``KVPool.restore`` copy slices, so a lost, misplaced or overwritten KV token
is detected when its request completes.
"""

from __future__ import annotations

import array
import random
from dataclasses import dataclass, field
from enum import Enum

# Request.MAX_OFFLOADING_LEN (serve/request/request.py l.13)
MAX_OFFLOADING_LEN = 1024


class ReqExecType(Enum):  # Request.ReqExecType (serve/request/request.py l.15-19)
    NORMAL = 0
    DRAFT = 1
    VERIFY = 2
    STALL = 3


class PoolExhausted(AssertionError):
    """An upstream ``assert`` on free pages fired (GPU or host pool)."""

    def __init__(self, pool: str, message: str):
        super().__init__(f"{pool} pool exhausted: {message}")
        self.pool = pool


class UpstreamAssert(AssertionError):
    """Any other upstream ``assert`` / shape error the transcription keeps."""


class SimInvariant(AssertionError):
    """A property this simulation checks on top of upstream's own asserts."""


# ---------------------------------------------------------------------------
# serve/model/model.py KVPool (page_size 1), with tagged physical memory
# ---------------------------------------------------------------------------
class KVPoolSim:
    def __init__(self, capacity: int, cpu_capacity: int):
        self._free_gpu = set(range(capacity))  # l.46
        self.capacity = capacity
        self.page_size = 1
        self._MAX_CPU_CAPACITY = int(cpu_capacity)  # l.72-74
        self._free_cpu = array.array("i", range(self._MAX_CPU_CAPACITY))  # l.75
        self._cpu_page_free = bytearray(b"\x01") * self._MAX_CPU_CAPACITY  # l.78
        # stand-ins for _physical_mem_gpu / _physical_mem_cpu: one tag per page
        self.gpu_mem: list = [None] * capacity
        self.cpu_mem: list = [None] * self._MAX_CPU_CAPACITY

    @property
    def num_free_pages(self) -> int:  # l.151-153
        return len(self._free_gpu)

    def alloc_page(self) -> int:  # l.158-161
        if not len(self._free_gpu) > 0:
            raise PoolExhausted("gpu", "alloc_page: Out of memory")
        return self._free_gpu.pop()

    def free_page(self, idx: int) -> None:  # l.163-166
        if idx in self._free_gpu:
            raise UpstreamAssert(f"Page {idx} already free")
        if not 0 <= idx < self.capacity:
            raise UpstreamAssert(f"page {idx} out of range")
        self._free_gpu.add(idx)

    def alloc_page_cpu(self) -> int:  # l.168-172
        if not len(self._free_cpu) > 0:
            raise PoolExhausted("host", "alloc_page_cpu: Out of memory")
        idx = self._free_cpu.pop()
        self._cpu_page_free[idx] = 0
        return idx

    def free_page_cpu(self, idx: int) -> None:  # l.174-178
        if not 0 <= idx < self._MAX_CPU_CAPACITY:
            raise UpstreamAssert(f"host page {idx} out of range")
        if self._cpu_page_free[idx]:
            raise UpstreamAssert(f"Page {idx} already restored")
        self._cpu_page_free[idx] = 1
        self._free_cpu.append(idx)

    @staticmethod
    def _cpu_rows(idx_cpu, n: int) -> range:
        # upstream copies the SLICE _physical_mem_cpu[idx_cpu[0] : idx_cpu[-1] + 1],
        # not the listed pages; torch raises on a size mismatch
        if len(idx_cpu) == 0:
            raise UpstreamAssert("IndexError: KVPool.offload / restore with an empty host index list")
        rows = range(idx_cpu[0], idx_cpu[-1] + 1)
        if len(rows) != n:
            raise UpstreamAssert(
                f"shape mismatch: host slice [{idx_cpu[0]}:{idx_cpu[-1] + 1}] has {len(rows)} rows for {n} GPU pages"
            )
        if list(rows) != list(idx_cpu):
            raise SimInvariant(f"host pages {list(idx_cpu)[:8]}... are not the contiguous slice that is copied")
        return rows

    def offload(self, idx_gpu, idx_cpu) -> None:  # l.180-183
        for row, page in zip(self._cpu_rows(idx_cpu, len(idx_gpu)), idx_gpu):
            self.cpu_mem[row] = self.gpu_mem[page]

    def restore(self, idx_cpu, idx_gpu) -> None:  # l.185-190
        for row, page in zip(self._cpu_rows(idx_cpu, len(idx_gpu)), idx_gpu):
            self.gpu_mem[page] = self.cpu_mem[row]

    @property
    def host_used(self) -> int:
        return self._MAX_CPU_CAPACITY - len(self._free_cpu)


# ---------------------------------------------------------------------------
# serve/request/kv_cache_ptr/base.py FullCachePtr (page_size 1)
# ---------------------------------------------------------------------------
class CachePtrSim:
    def __init__(self, kv_pool: KVPoolSim):  # l.86-98
        self.kv_pool = kv_pool
        self.indices = array.array("i", [])
        self.indices_cpu = array.array("i", [])
        self.len_kv_cache = 0
        self.page_size = kv_pool.page_size
        self.streaming = False  # True once PillarCachePtr.from_kv_cache has replaced the FullCachePtr

    def to_pillar(self) -> None:
        """PillarCachePtr.from_kv_cache (pillar.py l.93-120): the pages move to the new pointer;
        the old FullCachePtr's indices are emptied, its host pages (if any) would be orphaned."""
        if not self.len_kv_cache > 0 or len(self.indices) != self.len_kv_cache:
            raise UpstreamAssert("from_kv_cache: source cache not fully on the GPU")
        if len(self.indices_cpu):
            raise SimInvariant("NORMAL -> DRAFT with host pages: from_kv_cache would leak them")
        self.streaming = True

    def release(self) -> None:  # __del__, l.100-102 (called explicitly at retire)
        for page_index in self.indices:
            self.kv_pool.free_page(page_index)
        self.indices = array.array("i", [])

    @property
    def last_page_offset(self) -> int:  # l.104-106
        return (self.len_kv_cache - 1) % self.page_size + 1

    def allocate_tokens(self, num_tokens: int, restore: bool = False) -> int:  # l.108-118
        if self.streaming and not (num_tokens == 1 or restore):  # StreamingCachePtr.allocate_tokens l.94
            raise UpstreamAssert("assert num_tokens == 1 or restore == True")
        if not num_tokens > 0:
            raise UpstreamAssert(f"Cannot allocate {num_tokens} tokens")
        if not num_tokens <= self.kv_pool.num_free_pages * self.page_size:
            raise PoolExhausted("gpu", f"Cannot allocate {num_tokens} tokens (exceeds available memory)")
        num_new_pages = (num_tokens + self.last_page_offset - 1) // self.page_size
        for _ in range(num_new_pages):
            self.indices.append(self.kv_pool.alloc_page())
        self.len_kv_cache += num_tokens
        return num_new_pages

    def retract_tokens(self, num_tokens: int) -> None:  # l.120-128
        if not 0 <= num_tokens <= self.len_kv_cache:
            raise UpstreamAssert(f"Cannot retract {num_tokens} tokens (only {self.len_kv_cache} exist)")
        for _ in range(num_tokens):
            if self.last_page_offset == 1:
                self.kv_pool.free_page(self.indices.pop())
            self.len_kv_cache -= 1

    def offload(self, length: int) -> int:  # l.130-141 (StreamingCachePtr: l.110-162)
        assert self.page_size == 1
        if self.streaming:
            length = min(length, len(self.indices))
        if not length > 0:
            raise UpstreamAssert("offload length must be > 0")
        length = min(length, len(self.indices))
        for _ in range(length):
            self.indices_cpu.append(self.kv_pool.alloc_page_cpu())
        # NB: with length == 0, upstream's [-0:] slices select EVERYTHING
        idx_gpu = self.indices[-length:] if length else self.indices[:]
        idx_cpu = self.indices_cpu[-length:][::-1] if length else self.indices_cpu[::-1]
        self.kv_pool.offload(idx_gpu, idx_cpu)
        self.retract_tokens(length)
        return len(self.indices)

    def restore(self, length: int) -> int:  # l.143-164 (StreamingCachePtr: l.164-222)
        assert self.page_size == 1
        if self.streaming:
            length = min(length, len(self.indices_cpu))
        if not length > 0:
            raise UpstreamAssert("restore length must be > 0")
        length = min(length, len(self.indices_cpu))
        self.allocate_tokens(length, restore=True)
        idx_cpu = self.indices_cpu[-length:][::-1] if length else self.indices_cpu[::-1]
        idx_gpu = self.indices[-length:] if length else self.indices[:]
        self.kv_pool.restore(idx_cpu, idx_gpu)
        for _ in range(length):
            self.kv_pool.free_page_cpu(self.indices_cpu.pop())
        return len(self.indices_cpu)


# ---------------------------------------------------------------------------
# serve/request/request.py Request, plus the adversary's plan for it
# ---------------------------------------------------------------------------
class RequestSim:
    MAX_OFFLOADING_LEN = MAX_OFFLOADING_LEN

    def __init__(self, rid: int, prompt_len: int, desired_output_length: int, final_kv: int):
        self.rid = rid
        self.len_input = prompt_len
        self.len_output = desired_output_length
        self._len_token_ids = prompt_len  # token_ids = the prompt
        self.status = ReqExecType.NORMAL
        self.kv_cache_ptr: CachePtrSim | None = None
        self.initial_idx = None
        # adversary / bookkeeping (not upstream)
        self.final_kv = final_kv  # KV length at which the request completes
        self.finish_after_restore = False
        self.prefill_tokens = 0  # prompt tokens fed through the model (restarts would double this)
        self.cycle_pos = 0
        self.completed_at: int | None = None
        self.completed_len: int | None = None
        self.times_offloaded = 0

    def attach_kv_pool(self, kv_pool: KVPoolSim) -> None:  # l.42-57
        self.kv_cache_ptr = CachePtrSim(kv_pool)

    def retire(self) -> None:  # l.59-68
        if len(self.kv_cache_ptr.indices_cpu):
            raise SimInvariant(f"request {self.rid} retired with {len(self.kv_cache_ptr.indices_cpu)} host pages")
        self.kv_cache_ptr.release()
        self.kv_cache_ptr = None

    def offload(self):  # l.70-72
        rest_len = self.kv_cache_ptr.offload(RequestSim.MAX_OFFLOADING_LEN)
        return self if rest_len != 0 else None

    def restore(self):  # l.74-76
        rest_len = self.kv_cache_ptr.restore(RequestSim.MAX_OFFLOADING_LEN)
        return self if rest_len != 0 else None

    @property
    def prefilled(self) -> bool:  # l.79-83
        return self.len_token_ids > self.len_input

    @property
    def len_kv_cache(self) -> int:
        return self.kv_cache_ptr.len_kv_cache

    @property
    def len_token_ids(self) -> int:
        return self._len_token_ids

    @property
    def host_pages(self) -> int:
        return 0 if self.kv_cache_ptr is None else len(self.kv_cache_ptr.indices_cpu)


@dataclass
class SimResult:
    iterations: int = 0
    completed: list = field(default_factory=list)
    peak_host_used: int = 0
    peak_gpu_used: int = 0
    peak_inflight: int = 0
    peak_active: int = 0
    num_offloads: int = 0  # victims picked (RLSpecScheduler.rl.num_offloads)
    prefilling_victims: int = 0  # victims picked while still prefilling (RLSpecScheduler.add_request warns)
    victim_picks: list = field(default_factory=list)  # (iteration, num_active, victim_len)
    # RLSpecScheduler.rl.* as the real scheduler records them (after each admit_requests)
    rl_peak_host_used: int = 0
    rl_peak_inflight: int = 0
    peak_committed: int = 0
    bound_checks: int = 0
    worst_bound_ratio: float = 0.0  # max over iterations of host_used / bound(epoch in-flight)


class SchedulerSim:
    """``BaseScheduler`` / ``SpecScheduler`` state relevant to admission, with
    ``RLSpecScheduler``'s guard on top (``guard=None``: upstream alone)."""

    def __init__(
        self,
        kv_pool: KVPoolSim,
        *,
        max_batch_size: int,
        chunk_size: int,
        spec_stride: int,
        guard: str | None,
        total_token_capacity: int | None = None,
        inflight_cap: int | None = None,
        host_bound=None,
        k_full: int | None = None,
        max_kv_len: int | None = None,
        sawtooth_seed: int | None = None,
    ):
        from serve.rl.memory import request_token_need

        self._request_token_need = request_token_need
        self.active_requests: list[RequestSim] = []
        self.offload_requests: list[RequestSim] = []
        self.offloading_request: RequestSim | None = None
        self.prefilling_request: RequestSim | None = None
        self.kv_pool = kv_pool
        self.max_batch_size = max_batch_size
        self._args = {"chunk_size": chunk_size}
        self.spec_stride = spec_stride
        if guard not in (None, "worst_case", "host_guard"):
            raise ValueError(guard)
        if guard == "host_guard" and (inflight_cap is None or inflight_cap < 1):
            raise ValueError("host_guard admission needs inflight_cap >= 1")  # RLSpecScheduler.__init__
        self._rl_admission = guard
        self._rl_inflight_cap = inflight_cap
        self._rl_total_capacity = None if total_token_capacity is None else int(total_token_capacity)
        self._rl_committed = 0
        self.host_bound = host_bound  # n -> bound on host pages (serve.rl.memory.host_guard_host_bound)
        self.k_full = k_full
        self.max_kv_len = max_kv_len  # premise of the bound: every request stays below `need`
        self.iteration = 0
        self.result = SimResult()
        self._epoch_inflight = 0
        self._sawtooth = random.Random(sawtooth_seed) if sawtooth_seed is not None else None

    # ---- SpecScheduler.add_request (l.51-63) / RLSpecScheduler.add_request -----------------------
    def add_request(self, request: RequestSim, is_restore: bool = False) -> None:
        if self._rl_admission is not None and is_restore and request.status == ReqExecType.NORMAL:
            # RLSpecScheduler.add_request: restoring a request that is still prefilling is allowed
            self.active_requests.append(request)
            return
        self.active_requests.append(request)
        if is_restore:
            if request.status == ReqExecType.NORMAL:
                raise UpstreamAssert("SpecScheduler.add_request: restored request in NORMAL state")
        else:
            request.initial_idx = 0

    # ---- BaseScheduler.admit_requests, admit_policy == "offloading" (l.194-287) -------------------
    def upstream_admit_requests(self, pending_requests: list[RequestSim]) -> None:
        # Do not include chunk size here to leave headroom
        tokens_needed_sum = sum(req.len_token_ids - req.len_kv_cache for req in self.active_requests)
        available_capacity = self.kv_pool.num_free_pages * self.kv_pool.page_size - tokens_needed_sum

        # Offload requests if needed
        if available_capacity < 0:
            # Sort active requests by len_kv_cache
            self.active_requests.sort(key=lambda x: x.len_kv_cache, reverse=True)
            while available_capacity < 0:
                if self.offloading_request:
                    self.offloading_request = self.offloading_request.offload()
                    available_capacity = self.kv_pool.num_free_pages * self.kv_pool.page_size - tokens_needed_sum
                else:
                    request = None
                    for req in self.active_requests[::-1]:
                        if req.status == ReqExecType.STALL:
                            continue
                        request = req
                        break
                    if request is None:
                        raise UpstreamAssert("admit_requests: no request left to offload (assert request is not None)")
                    self._on_victim(request)
                    self.offloading_request = request.offload()
                    self.offload_requests.append(request)
                    self.active_requests.remove(request)
                    available_capacity = self.kv_pool.num_free_pages * self.kv_pool.page_size - tokens_needed_sum
        else:
            # If there are offloaded requests, try to restore them
            if (
                len(self.offload_requests) > 0
                and len(self.active_requests) < self.max_batch_size
                and self.offload_requests[-1].MAX_OFFLOADING_LEN < available_capacity
            ):
                request = self.offload_requests[-1]
                self.offloading_request = request.restore()
                if self.offloading_request is None:
                    self.add_request(request, is_restore=True)
                    self.offload_requests.pop()
                    self._on_restored(request)
                available_capacity = self.kv_pool.num_free_pages * self.kv_pool.page_size - tokens_needed_sum

            if len(self.offload_requests) == 0:
                # If no offloaded requests, try to add pending requests
                # Only try first one to assure one per-iteration
                if (
                    len(pending_requests) > 0
                    and len(self.active_requests) < self.max_batch_size
                    and pending_requests[0].len_token_ids < available_capacity
                    and self.prefilling_request is None
                ):
                    request = pending_requests[0]
                    available_capacity -= request.len_token_ids
                    request.attach_kv_pool(self.kv_pool)
                    self.prefilling_request = request
                    self.add_request(request)
                    pending_requests.remove(request)

    # ---- RLSpecScheduler (serve/rl/scheduler.py) --------------------------------------------------
    def _rl_need(self, request: RequestSim) -> int:
        return self._request_token_need(request.len_input, request.len_output + 1, self.spec_stride)

    def _rl_may_admit(self, request: RequestSim) -> bool:
        if self._rl_admission == "host_guard":
            if len(self.active_requests) + len(self.offload_requests) + 1 > self._rl_inflight_cap:
                return False
        elif self._rl_committed + self._rl_need(request) > self._rl_total_capacity:
            return False
        if not self.active_requests and not self.offload_requests:
            return True
        page = self.kv_pool.page_size
        free = self.kv_pool.num_free_pages * page
        owed = sum(r.len_token_ids - r.len_kv_cache for r in self.active_requests)
        prefill_iters = -(-request.len_token_ids // self._args["chunk_size"]) + 1
        margin = (len(self.active_requests) + 1) * (self.spec_stride + 2) * prefill_iters
        return free - owed >= request.len_token_ids + margin

    def admit_requests(self, pending_requests: list[RequestSim]) -> None:
        if self._rl_admission is None:
            self.upstream_admit_requests(pending_requests)
            return
        head = pending_requests[0] if pending_requests else None
        allow = head is not None and not self.offload_requests and self._rl_may_admit(head)
        self.upstream_admit_requests(pending_requests if allow else [])
        if allow and (not pending_requests or pending_requests[0] is not head):
            head._rl_need = self._rl_need(head)
            self._rl_committed += head._rl_need
        in_flight = len(self.active_requests) + len(self.offload_requests)
        self.result.rl_peak_inflight = max(self.result.rl_peak_inflight, in_flight)
        host_used = self.kv_pool._MAX_CPU_CAPACITY - len(self.kv_pool._free_cpu)
        self.result.rl_peak_host_used = max(self.result.rl_peak_host_used, host_used)

    # ---- observations ------------------------------------------------------------------------------
    def _on_victim(self, request: RequestSim) -> None:
        res = self.result
        res.num_offloads += 1
        request.times_offloaded += 1
        active = len(self.active_requests)
        res.victim_picks.append((self.iteration, active, request.len_kv_cache))
        if not request.prefilled:
            res.prefilling_victims += 1
        # the two lemmas behind serve.rl.memory.host_guard_host_bound
        kv = self.kv_pool.capacity
        if request.len_kv_cache > kv // active:
            raise SimInvariant(
                f"iteration {self.iteration}: victim holds {request.len_kv_cache} tokens > kv // {active} active"
            )
        if self.k_full is not None and active < self.k_full:
            raise SimInvariant(
                f"iteration {self.iteration}: offload with only {active} active requests < k_full={self.k_full} "
                f"(gpu_full_min_active); victim len {request.len_kv_cache}, "
                f"active lens {sorted(r.len_kv_cache for r in self.active_requests)}"
            )

    def _on_restored(self, request: RequestSim) -> None:
        if request.finish_after_restore:
            # adversary: the request ends right after it is back on the GPU
            request.final_kv = min(request.final_kv, max(request.len_kv_cache + 1, request.len_input))

    # ---- the driver loop: BaseScheduler.run (l.524-698) with the forward pass modelled ------------
    def run(self, requests: list[RequestSim], max_iterations: int) -> SimResult:
        pending = list(requests)
        total = len(pending)
        completed: list[RequestSim] = []
        pool = self.kv_pool
        res = self.result
        while pending or self.active_requests or self.offload_requests:
            self.iteration += 1
            if self.iteration > max_iterations:
                raise SimInvariant(
                    f"no termination after {max_iterations} iterations: {len(pending)} pending, "
                    f"{len(self.active_requests)} active, {len(self.offload_requests)} offloaded"
                )
            if not self.offload_requests:
                self._epoch_inflight = len(self.active_requests) + len(self.offload_requests)
            self.admit_requests(pending)
            if len(self.active_requests) + len(pending) + len(completed) + len(self.offload_requests) != total:
                raise UpstreamAssert("BaseScheduler.run request count assert")
            self._check_after_admit()

            # ---- prepare_forward_inputs: KV allocation (SpecScheduler l.156-171) ----
            for request in self.active_requests:
                cache = request.kv_cache_ptr
                if request.status == ReqExecType.NORMAL:
                    num_tokens = min(request.len_token_ids - request.len_kv_cache, self._args["chunk_size"])
                    request.prefill_tokens += num_tokens
                else:
                    num_tokens = 1
                start = cache.len_kv_cache
                cache.allocate_tokens(num_tokens)
                for pos in range(start, start + num_tokens):
                    pool.gpu_mem[cache.indices[pos]] = (request.rid, pos)
            res.peak_gpu_used = max(res.peak_gpu_used, pool.capacity - pool.num_free_pages)

            # ---- update_requests: new token, verify retraction, completion (SpecScheduler l.453-613) ----
            remaining: list[RequestSim] = []
            for request in self.active_requests:
                cache = request.kv_cache_ptr
                if request.status == ReqExecType.NORMAL:
                    if request.len_kv_cache == request.len_token_ids:
                        request._len_token_ids += 1  # first response token
                        if self.prefilling_request is not None and self.prefilling_request == request:
                            if not request.prefilled:
                                raise UpstreamAssert("assert request.prefilled")
                            self.prefilling_request = None
                else:
                    request._len_token_ids += 1  # next (draft / bonus) token
                    request.cycle_pos += 1
                    if self._sawtooth is not None and request.cycle_pos % (self.spec_stride + 1) == 0:
                        # verify pass: rejected drafts leave the KV cache (verify_outputs l.287-290)
                        floor = request.len_input + 1
                        retract = min(self._sawtooth.randint(0, self.spec_stride), cache.len_kv_cache - floor)
                        if retract > 0:
                            cache.retract_tokens(retract)
                            request._len_token_ids -= retract
                if self.max_kv_len is not None and request.len_kv_cache > self.max_kv_len:
                    raise SimInvariant(f"request {request.rid} exceeded the modelled maximum KV length")
                if request.prefilled and request.len_kv_cache >= request.final_kv:
                    self._complete(request)
                    completed.append(request)
                    continue
                remaining.append(request)
                if request.status == ReqExecType.NORMAL and request.prefilled:
                    request.status = ReqExecType.DRAFT
                    cache.to_pillar()  # spec_scheduler.py l.539-551
            self.active_requests = remaining
            res.peak_active = max(res.peak_active, len(self.active_requests))
        res.iterations = self.iteration
        res.completed = completed
        return res

    def _complete(self, request: RequestSim) -> None:
        cache = request.kv_cache_ptr
        tags = [self.kv_pool.gpu_mem[p] for p in cache.indices]
        expected = [(request.rid, i) for i in range(cache.len_kv_cache)]
        if tags != expected:
            bad = next(i for i, (a, b) in enumerate(zip(tags, expected)) if a != b)
            raise SimInvariant(f"request {request.rid}: KV position {bad} holds {tags[bad]} (data lost or misplaced)")
        request.completed_at = self.iteration
        request.completed_len = cache.len_kv_cache
        if self._rl_admission is not None:
            self._rl_committed -= request._rl_need
        request.retire()

    def _check_after_admit(self) -> None:
        res = self.result
        pool = self.kv_pool
        in_flight = len(self.active_requests) + len(self.offload_requests)
        self._epoch_inflight = max(self._epoch_inflight, in_flight)
        res.peak_inflight = max(res.peak_inflight, in_flight)
        host_used = pool.host_used
        res.peak_host_used = max(res.peak_host_used, host_used)
        res.peak_committed = max(res.peak_committed, self._rl_committed)
        if in_flight > self.max_batch_size:
            raise SimInvariant(f"{in_flight} requests in flight > max_batch_size (per-request buffers)")
        if self._rl_admission == "host_guard" and in_flight > self._rl_inflight_cap:
            raise SimInvariant(f"{in_flight} requests in flight > host_guard cap {self._rl_inflight_cap}")
        if self._rl_admission == "worst_case" and self._rl_committed > self._rl_total_capacity:
            raise SimInvariant(f"committed {self._rl_committed} > admission capacity {self._rl_total_capacity}")
        if sum(r.host_pages for r in self.offload_requests) != host_used:
            raise SimInvariant("host pages are held by something other than the offloaded requests")
        if self.host_bound is not None:
            bound = self.host_bound(self._epoch_inflight)
            res.bound_checks += 1
            if host_used > bound:
                raise SimInvariant(
                    f"iteration {self.iteration}: {host_used} host pages in use > host_guard_host_bound("
                    f"{self._epoch_inflight}) = {bound}; stack lens "
                    f"{[r.host_pages for r in self.offload_requests]}"
                )
            if bound:
                res.worst_bound_ratio = max(res.worst_bound_ratio, host_used / bound)


def make_requests(prompt_lens, final_kvs, max_new_tokens: int) -> list[RequestSim]:
    """Requests as SparseSpecEngine.generate builds them: desired_output_length = max_new_tokens - 1."""
    out = []
    for rid, (p, f) in enumerate(zip(prompt_lens, final_kvs)):
        if f < p:
            raise ValueError(f"final KV length {f} below the prompt length {p}")
        out.append(RequestSim(rid, int(p), max_new_tokens - 1, int(f)))
    return out
