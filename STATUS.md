# STATUS: SparseSpec as a verl 0.5.0 rollout backend

Branch `sparsespec-verl-backend`, written 2026-11-27. How to run everything on the A100 box: `RUNBOOK.md`.

## Where this stands

**The code for steps 0 to 4 is written and committed, but none of it has run on a GPU.** The session that
wrote it was on a Mac with no CUDA, no conda and no route to the A100 box. Everything that can run on a
CPU was run and passes. The milestone gates that need the real engine (3b greedy, 3c reload, 3d smoke)
and the GPU unit tests are **not run**; their scripts, thresholds and exact commands are ready.

So the weight-reload log-prob gate is **not passed yet**, and the real run cannot be launched until the
gates have been run on the box in Harrison's `verl-sparsespec` env. The launch deadline is Nov 29.

## Gate results

| Gate | Status | Result |
|---|---|---|
| 3a sampler review | done, by code reading | Standard speculative sampling; no off-by-one in the row bookkeeping; see below and `baseline/results/sparsespec_checks/3a_sampler_review.md` |
| Engine unit tests (GPU) | **not run** (no GPU) | `serve/rl/tests/test_engine_gpu.py`, RUNBOOK section 4 |
| 3b greedy, SparseSpec vs HF | **not run** (no GPU) | `python -m integration.gates.gate_b_greedy --engine sparsespec` |
| 3c reload gate, synthetic checkpoint | **not run** (no GPU) | `python -m integration.gates.gate_c_reload --engine sparsespec --synthetic` |
| 3c reload gate, real step-25 checkpoint | **not run** (no GPU; checkpoint not seen) | RUNBOOK section 4 |
| 3c stock-vLLM reference numbers | **not run** (no GPU, no vLLM) | `gate_c_vllm_reference.py`; written against the vLLM 0.9.1 sources, never executed |
| 3d smoke, 3 steps | **not run** (no GPU) | `STEPS=3 EXP_NAME=sparsespec_smoke bash baseline/run_seed_sparsespec.sh` |
| Reference-engine versions of 3b / 3c / smoke | **not run on GPU**; control flow exercised on CPU with a tiny random Qwen3 | RUNBOOK section 3 |

Numbers to fill in on the box (leave blank until measured):

| Quantity | SparseSpec | stock vLLM |
|---|---|---|
| 3b: prompts identical / 16, earliest divergence position, largest HF margin at a divergence | | n/a |
| 3c reloaded: mean, p99 \|Δ log p\|, max probability ratio | | |
| 3c base after round trip: mean \|Δ log p\| | | n/a |
| 3c negative control: mean \|Δ log p\| (and tracking coefficients) | | n/a |
| 3c bitwise equal after reload / tied / pointers unchanged | | n/a |
| 3d: step time, `timing_s/gen`, reward on step 3 | | (baseline smoke) |
| engine resident awake / asleep, peak during rollout (GiB) | | |
| `accept_len_mean`, `batch_size_mean`, `kv_util_peak`, `num_offloads` | | n/a |

### What was run (CPU only)

| Check | Result |
|---|---|
| `python -m pytest serve/rl/tests -q` (local: torch 2.15.0, transformers 5.24.1, numpy 1.26.4, verl 0.5.0) | 427 passed, 1 skipped (the GPU module) |
| Same suite before the review fixes, on a box-like overlay (torch 2.7.1, transformers 4.57.1 and 4.51.3) | 415 passed, 1 skipped; not re-run on the overlay after the fixes |
| `integration/verl-0.5.0-sparsespec.patch` against the pristine verl 0.5.0 wheel | applies with `git apply` and `patch -p1`, both files compile, reverse-applies |
| `DRY_RUN=1 bash baseline/run_seed_sparsespec.sh` composed with verl's real hydra config | composes; differs from `run_seed.sh` only in `actor_rollout_ref.rollout.*`, experiment name and checkpoint dir |
| verl metrics path: sharding manager timing -> patched trainer statement -> console line -> `baseline/parse_timing.py` | every `sparsespec/*` column parsed with the right value |
| Rollout output vs a verbatim copy of the tail of `vLLMRollout.generate_sequences` | identical keys, shapes, dtypes, padding, masks, position ids |
| Gate scripts with `--engine reference` on a tiny random Qwen3 | see "Gate scripts" below |
| 3a: CPU transcription of the sampler arithmetic, chi-square against the exact law (20M sequences per variant) | sampler as written p = 0.44 to 0.86; three deliberately broken variants rejected with p = 0 |

Review: five independent reviewers read the wrapper against upstream `serve/`, verl 0.5.0, vLLM 0.9.1 and
torch 2.7 sources, and each finding was checked by a second reader. 20 findings were confirmed, none of
them a wrong-result bug; all code-level ones are fixed on the branch (list under "Review fixes"). The
log-prob row mapping, the EOS handling and the weight mapping were traced by hand and found correct.
That is still reading, not running.

### Versions

| | this Mac (CPU checks) | `verl-baseline` | `verl-sparsespec` |
|---|---|---|---|
| torch | 2.15.0 (and 2.7.1 overlay) | fill in | fill in |
| transformers | 5.24.1 (and 4.57.1 / 4.51.3 overlay) | fill in | fill in |
| verl | 0.5.0 | 0.5.0 expected | 0.5.0 + patch |
| vLLM | none | 0.9.1 expected | fill in |
| flashinfer | none | none | fork, fill in |

The command that prints these on the box is in RUNBOOK section 1b.

## What is on the branch

| Path | What |
|---|---|
| `serve/rl/engine.py` | `SparseSpecEngine`: built like `run_inference`, lives for the whole run; `generate` returns target log-probs; in-place `load_weights`; `sleep` / `wake_up` |
| `serve/rl/scheduler.py` | `RLSpecScheduler(SpecScheduler)`: log-prob capture from the verify pass, EOS kept, admission guards, per-call stats |
| `serve/rl/reference_engine.py` | `ReferenceEngine`: same interface on HF transformers; acceptance stats 0 |
| `serve/rl/verl_rollout.py` | `SparseSpecRollout`, `SparseSpecShardingManager`, config parsing; `rollout.sparsespec.engine=reference\|sparsespec` |
| `serve/rl/weights.py`, `memory.py`, `base.py` | weight mapping (fused `wqkv` / `w12`, tied embeddings), memory plan, shared contract |
| `integration/verl-0.5.0-sparsespec.patch` | the only change to verl: one branch in `_build_rollout`, one statement in the trainer for metric names |
| `integration/setup_env.sh`, `smoke_reference.sh` | patched verl in a clone of `verl-baseline`; reduced-size plumbing run |
| `integration/gates/` | gates 3b, 3c (+ stock-vLLM reference), 3d checker, synthetic checkpoint |
| `serve/rl/tests/` | CPU tests and `test_engine_gpu.py` |
| `baseline/run_seed_sparsespec.sh` | real keys, `exit 1` guard removed, SEED=0, everything else inherited |
| `baseline/results/sparsespec_checks/3a_sampler_review.md` | gate 3a |

Config block the run script adds (nothing outside `actor_rollout_ref.rollout.*`):

```
actor_rollout_ref.rollout.name=sparsespec
actor_rollout_ref.rollout.tensor_model_parallel_size=1
+actor_rollout_ref.rollout.sparsespec.engine=sparsespec
+actor_rollout_ref.rollout.sparsespec.spec_stride=8
+actor_rollout_ref.rollout.sparsespec.budget_ratio=0.05
+actor_rollout_ref.rollout.sparsespec.num_min_budget=128
+actor_rollout_ref.rollout.sparsespec.kv_cache=pillar
+actor_rollout_ref.rollout.sparsespec.max_batch_size=64
+actor_rollout_ref.rollout.sparsespec.cuda_graph=True
actor_rollout_ref.rollout.calculate_log_probs=True
```

`rollout.gpu_memory_utilization=0.45` (inherited) is the engine's `mem_fraction`.

Per-step metrics in the timing CSV: `sparsespec/accept_len_mean`, `num_cycles`, `weight_sync_s`,
`wake_s` (includes `graph_capture_s`), `gen_s`, `sleep_s`, plus `accept_rate`, `num_iters`,
`batch_size_mean`, `num_offloads`, `kv_util_peak`, `peak_mem_gb`, `resident_asleep_gb`,
`process_peak_alloc_gb`, `process_peak_reserved_gb`. `accept_len_mean` is tokens emitted per verify
cycle (accepted drafts + 1).

### Gate 3a in one paragraph

By code reading, `serve/sampling/rejection_sampler.py` is standard speculative sampling, a port of
vLLM's: accept draft token x when `p(x)/q(x) >= u` with u uniform in float64 (reject if q(x) = 0);
on the first rejection draw the replacement from normalized `max(0, p - q)` by an exponential-race
argmax with independent noise per request; if all drafts are accepted, the bonus token is the regular
sampler's draw from the target row after the last draft. p and q are both float32
`softmax(logits / T)` from the same kernel: p from the full-attention verify pass, q the exact row the
draft token was sampled from. Temperature is applied identically to both. **top_p and top_k are not
implemented on the path that runs**; the wrapper raises on anything other than top_p = 1.0 and top_k in
{-1, 0}, which is the project config. At T = 0 both become one-hot and acceptance is exact match.
No gate tests the sampling law itself on the GPU (3b and 3c test that p is the right row of the right
model); the review proposes a surprisal-calibration test that reuses 3c's scoring pass.

## Things the brief assumed that turned out different

1. **Per-request memory is about 11 MiB, not 7.** Besides `draft_probs` (10 x 151,936 x 4 B = 5.8 MiB)
   every in-flight request holds a PillarAttn selected-index buffer of 28 x 8 x 6144 int32 = 5.25 MiB.
   Both are accounted for.
2. **`run_inference` cannot build Qwen3-0.6B as is.** `ModelArgs.element_size()` has no entry for it; a
   hub id for a single-file checkpoint fails in `Transformer.from_pretrained` (it only falls back on
   `FileNotFoundError`); and `Attention` creates Qwen3's `q_norm` / `k_norm` only when the model **path**
   contains "qwen3", so a checkpoint under `.../hf/global_step_25` would silently lose them. The engine
   sizes memory from the loaded parameters, resolves hub ids to a local snapshot, aliases such
   directories through a symlink named `qwen3-model`, and verifies its weights bitwise against the
   checkpoint right after loading.
3. **Upstream EOS handling does not match vLLM's contract.** After a verify pass `SpecScheduler` strips
   the EOS token, skips the EOS scan while fewer than 9 tokens exist (an early EOS can be missed for
   good), looks at the last prompt token during chunked prefill, and overshoots the length limit by up to
   9 tokens. `RLSpecScheduler` overrides completion: the EOS is kept, nothing after it survives, and
   responses are cut to exactly `max_new_tokens`, as stock vLLM returns them.
4. **`mem_fraction` in `run_inference` covers only weights + KV pool.** Here it is the whole engine
   share, like vLLM's `gpu_memory_utilization`: weights 1.11 GiB, fixed kernel buffers 1.06, CUDA graphs
   1.69 (estimate), per-request buffers 0.69, scratch 2.0, safety 0.5, and the KV pool gets the rest:
   **10.76 GiB = 100,735 tokens** on a 39.6 GiB card (planned; the engine logs measured values).
5. **Tied embeddings are not tied in the stock engine.** `from_pretrained` + `.to(device)` leaves two
   equal copies (311 MB extra). The wrapper re-ties them to one storage before any graph is captured.
6. **Asynchronous CPU scheduling is off.** Upstream's benchmark scripts use `--async_cpu`; there,
   verification runs one iteration late against the previous pass's tensors and the target log-prob
   bookkeeping would have to thread through `run()`. The wrapper supports synchronous mode only and
   refuses async. This costs the SparseSpec arm some throughput.
7. **Asserts are on.** Upstream scripts run with `PYTHONOPTIMIZE=1`. Under verl they are enabled, which
   exposed an O(pool) assert per restored host page (fixed, see below) and means any other upstream
   assert our workload trips will stop a run. The 3-step smoke at full size is the test for that.

Changes to `serve/` outside `serve/rl/` (both in `serve/model/model.py::KVPool`, defaults unchanged):
an optional `cpu_capacity` argument, and an O(1) double-free check for host pages.

### Review fixes already applied

- Host-page free assert scanned the whole free list per page: about 2 s per 1024-page restore with
  asserts on. Now a per-page flag.
- The 1.5x host offload pool (16.1 GiB) would have been rounded up to **32 GiB of page-locked RAM** by
  torch's pinned allocator. It is now sized to fill a 16 GiB block exactly and the plan reports what is
  really locked.
- The engine left a `DeviceContext('cpu')` torch-function mode installed in the trainer thread after
  every call. Removed on exit.
- CUDA-graph re-capture on every wake ran 24 full `gc.collect()` passes (torch 2.7); now one.
- Reported engine memory missed the graphs' private pools (about 0.4 GiB by an allocator model).
- `generate()` resets torch's peak-memory counters; the process peak is now preserved and published.
- Worst-case admission silently capped concurrency below `max_batch_size`; now logged once, with the
  knob that lifts it.
- Run-script pre-flight checked only one of the two patch hunks; a full-size reference-engine run
  (hours per step) was advertised as a smoke test and is now refused by default.
- Reference engine rounded Qwen3's rotary `inv_freq` to bf16, computed logits for every prompt position,
  and could strand KV-cache segments.

## Fairness: what differs between the arms besides the decoding algorithm

Facts from the sources (verl 0.5.0, vLLM 0.9.1, this branch); nothing here is measured.

| Topic | Baseline (vLLM) | SparseSpec arm | Favors |
|---|---|---|---|
| CUDA graphs | **eager**: verl's default `rollout.enforce_eager=True`, not overridden in `grpo_base.sh`; no graphs, no torch.compile | graphs on (`cuda_graph=True`), re-captured every step (`sparsespec/graph_capture_s`, inside `timing_s/gen`) | SparseSpec, size unknown |
| Engine version | vLLM V1 in-process by source reading (`VLLM_USE_V1` unset); confirm from the baseline log | `serve/` synchronous scheduler; async CPU off | baseline |
| GPU KV pool inside the 0.45 share | roughly 15 GiB (share minus weights and profiling peak) | 10.76 GiB planned (fixed buffers, graphs, per-request buffers, scratch reserves come out of the share) | baseline |
| Concurrency | up to 256 sequences (vLLM default; verl does not forward `max_num_seqs`), optimistic admission, preempt and recompute | worst-case admission: about 26 to 29 in flight with the default host pool, `max_batch_size` 64 | baseline |
| Host RAM | weights copy while asleep (about 1.2 GB) | 16 GiB pinned offload pool for the whole run | n/a (resource) |
| GPU memory during training | about 0 (level-1 sleep unmaps weights and KV) | about 2.2 GiB planned stays resident (weights + fixed buffers) | baseline (headroom) |
| Prefix caching | on: the 8 samples of a prompt share prompt KV | none | baseline, small |
| Per-step lifecycle | weights restored from host on wake | no weight restore, but KV re-allocation + graph capture | unknown |
| Sampling seed / RNG | global CUDA stream, seeded 1000 | engine-private stream, seeded 0; trainer RNG untouched | none |
| `calculate_log_probs` | False | True (diagnostics only; small extra cost inside `timing_s/gen`) | baseline |
| torch / vLLM versions | torch pinned by vLLM 0.9.1 | same torch **only if** Harrison's env keeps stock vLLM 0.9.1 (the fork pins torch 2.8.0); `serve/` needs only `vllm._custom_ops`, which stock 0.9.1 has | n/a |
| verl `perf/max_memory_*` | lifetime peak, includes vLLM's sleeping pool | per-step peak (engine resets the counters) | not comparable; use `sparsespec/*` memory columns |

`timing_s/gen` wraps the same verl code in both arms (weight sync, wake, generation, sleep, Ray
transfer). For a like-for-like generation time also compare `timing_s/generate_sequences`.

## Open decisions for you

1. **The baseline config looks like it cannot start on one GPU.** `grpo_base.sh` does not set
   `rollout.tensor_model_parallel_size`; verl 0.5.0's default is 2 and `_build_rollout` asserts it
   divides the world size (1). The composed baseline config was checked on CPU and has the value 2. If
   the baseline did run on the box, something there differs from the committed config (local edit, other
   verl build). I did not touch `grpo_base.sh`. The SparseSpec script sets the key to 1 inside its own
   rollout block; if the baseline needs the same line, that is a `CONFIG_VERSION` bump for the group.
2. **CUDA graphs: eager baseline vs graph-replaying SparseSpec.** Options: (a) baseline owner sets
   `rollout.enforce_eager=False` (CONFIG_VERSION bump, needs a short check that vLLM's graphs survive
   sleep / weight sync); (b) keep both as they are and label the headline accordingly; (c) also report an
   eager SparseSpec timing from a few steps (`SPARSESPEC_CUDA_GRAPH=False`). I recommend (b) + (c) given
   the time left: graphs are the engine's intended mode, and (c) costs a few steps.
3. **Host RAM and concurrency.** Default: 16 GiB pinned, about 26 to 29 requests in flight. If the box
   has the RAM, `cpu_offload_ratio=3.0` (32 GiB pinned) gives about 46. Whether it helps depends on the
   response-length distribution: decide after the smoke run from `kv_util_peak` and `batch_size_mean`
   (RUNBOOK section 6). I left the conservative default because the box is shared.
4. **`calculate_log_probs=True` in the SparseSpec arm only.** It is inside the rollout block and gives a
   per-step engine-vs-actor check (`training/rollout_probs_diff_*`) for the whole run. Drop the line in
   `run_seed_sparsespec.sh` if you want the two arms' rollout blocks to differ by engine keys only.
5. **Negative control at step 25.** 25 steps at lr 1e-6 may leave the checkpoint almost identical to the
   base model in bf16, in which case "the gap must be clearly larger" cannot hold for any correct
   engine. The gate reports the model distance and a tracking coefficient for that case, but the default
   verdict rule is the one you specified. If it fails for this reason, the numbers decide, not a
   threshold change: rerun on step 50 or 100, or accept the tracking statistic.
6. **Async scheduling.** Supporting `async_cpu=True` would recover an upstream optimization but needs
   new log-prob bookkeeping and its own gate run. Not attempted.
7. **Tell Harrison** the env does not need the vLLM fork (keeps torch equal across arms), and that the
   engine needs RAFT headers and `librmm.so` under `$CONDA_PREFIX` plus `numba`, `jinja2`, `wandb`.
8. **Sampling-law test.** Optional extra gate (surprisal calibration) described in the 3a note,
   section 7. Not implemented.

## Launching the real run

Only after 3b, 3c and 3d pass on the box (RUNBOOK section 4):

```bash
conda activate verl-sparsespec
nvidia-smi
bash baseline/run_seed_sparsespec.sh        # SEED=0, STEPS=100, EXP_NAME=sparsespec_seed0
```
