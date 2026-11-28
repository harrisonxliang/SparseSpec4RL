# STATUS: SparseSpec as a verl 0.5.0 rollout backend

Branch `sparsespec-verl-backend`, updated 2026-11-28. For Harrison: `HANDOFF.md`. Details per gate and
manual reruns: `RUNBOOK.md`.

## Where this stands

**All code is written and committed. None of it has run on a GPU.** It was written on a Mac with no CUDA
and no conda. Everything that can run on a CPU was run and passes. The gates that need the real engine
(GPU unit tests, 3b, 3c, 3d) are **not run**, so the weight-reload gate is **not passed yet** and the
real run must not be launched.

Harrison is the only person with GPU access, so the plan is one round trip:

1. Harrison builds the `verl-sparsespec` env: `integration/build_sparsespec_env.sh` (HANDOFF.md section 1).
2. Harrison runs `bash integration/run_all_gates.sh` (5 to 12 hours, estimated) and sends back the tarball
   it names.
3. You read `SUMMARY.txt` in the bundle, fill the tables below, and give the go-ahead for
   `bash baseline/run_seed_sparsespec.sh`.

The launch deadline is Nov 29 and the gate run takes most of a working day. If that is too long, the
two smoke stages that exist only for the write-up can be left for after the launch:
`SKIP_STAGES=smoke_no_graph,smoke_no_logprobs bash integration/run_all_gates.sh` saves roughly 1 to 3.5
hours and does not change the verdict.

The build script, the gate runner, `setup_env.sh` and `smoke_reference.sh` have never been executed for
real (only against stub commands), so a failure on the box may be a script bug. Each one logs every
step, names the failed command and says what to send back.

## Decisions recorded (round 2)

| Topic | Decision | Where it shows |
|---|---|---|
| CUDA graphs | Baseline stays eager, SparseSpec keeps graphs. The smoke stage also runs 3 steps with `cuda_graph=False`. | `smoke_compare` reports gen time for default / host_guard / no-graph side by side |
| `calculate_log_probs` | Stays `True` in the SparseSpec arm. Its cost is measured: 1 step with, 1 step without. | `smoke_compare`, "log-prob cost" |
| Negative control (3c) | Default verdict unchanged: 3x mean rule, plus the tracking coefficients when their standard error is small. If step 25 fails only because the checkpoint is too close to the base model, the bundle says so (`failure_class: negative_control_underpowered`) and prints the step-50 rerun commands. | gate JSON / markdown, `SUMMARY.txt`, RUNBOOK section 5 |
| Async scheduling, sampling-law test | Out of scope, unchanged. | |
| Host RAM | Total and available RAM are recorded in the bundle; `smoke_compare` says whether a larger `cpu_offload_ratio` would fit. | `env/host.txt`, `smoke_compare.md` |
| Admission | New flag `rollout.sparsespec.admission=worst_case\|host_guard`. `worst_case` stays the default. | section "Concurrency" below |
| `max_batch_size` | Stays 64. It cannot usefully be raised to vLLM's 256; numbers below. | section "Concurrency" below |

## Choices I made that you should confirm

1. **Gate 3b, "no worse than vLLM" is decided by a significance test, not literally.** With 16 prompts
   the literal rule (SparseSpec diverges on no more prompts than vLLM, and not earlier) fails a correct
   engine 43% to 68% of the time once divergences happen at all, because both engines' divergences are
   random bf16 near-ties. The default is a one-sided exact test of "SparseSpec's per-token divergence
   rate is higher than vLLM's" at alpha 0.05. Both rules are always computed and printed;
   `--profile-rule strict` makes the literal rule decide. "Every divergence is a near-tie" is checked
   as written (HF logit margin at most 1.0) for all three pairs.
2. **Gate 3b passes only as a three-way comparison.** If the stock-vLLM half fails, the runner still
   runs SparseSpec vs HF for the information, but the verdict says "3b two-way only" and the run does
   not pass.
3. **Stage order: the stock-vLLM reference for 3c now runs before the step-25 gate**, not after it as you
   listed. The step-25 gate needs it: "the checkpoint is too close" can only be told apart from "the
   engine's log-probs are noisy" by comparing with stock vLLM on the same checkpoint. Without a usable
   reference the class is `negative_control_inconclusive`, never "underpowered".
4. **The env is built with `pip install --no-deps -e .`**, not `pip install -e .`. The project's pinned
   dependencies would downgrade `datasets` and force `antlr4-python3-runtime==4.7.2`, which breaks
   hydra / omegaconf (they need 4.9) and with it verl's config parser. Everything `serve/` imports is
   already in a clone of `verl-baseline`. Every other install into the env is `--no-deps` too, and the
   build ends with a drift report that fails if torch, transformers, verl, vllm, numpy or tensordict
   differ from `verl-baseline`.
5. **The runner refuses to start (exit 2, in the first minutes) on things Harrison can fix at once**:
   GPU busy, env missing, pytest or the verl patch missing, a stray `INSTALL_PREFIX` / `PREFIX`, no CUDA
   compiler where the engine's JIT looks, training data missing. Each refusal prints the command that
   fixes it. Anything else is recorded and the run continues.
6. **The GPU offload stress test now requires that multi-step offload and restore really happened**
   (requests above 1024 KV tokens, as in training). If the scheduler never takes that path in the
   test, the test fails with a message saying so; that would be a coverage failure, not an engine bug.

## Gate results

| Gate | Status | Notes |
|---|---|---|
| 3a sampler review | done, by code reading | `baseline/results/sparsespec_checks/3a_sampler_review.md` |
| GPU unit tests (4 variants: default, host_guard, offload stress x 2 admissions) | **not run** | `serve/rl/tests/test_engine_gpu.py` |
| 3b three-way greedy (SparseSpec / HF / stock vLLM) | **not run** | `gate_b_vllm_greedy.py` has never run against a real vLLM |
| 3c reload, synthetic checkpoint | **not run** | |
| 3c stock-vLLM reference | **not run** | written against the vLLM 0.9.1 sources, never executed |
| 3c reload, step-25 checkpoint | **not run** | checkpoint not seen |
| 3d smoke, 3 steps: default / host_guard / no graphs; 1 step without log-probs | **not run** | |
| Reference-engine plumbing run | **not run** | `integration/smoke_reference.sh` |

Numbers to fill in from the bundle (`SUMMARY.txt`, `gates/*.md`, `gates/smoke_compare.md`):

| Quantity | SparseSpec | stock vLLM |
|---|---|---|
| 3b: prompts diverging from HF / 16, earliest divergence, largest HF margin at a divergence | | |
| 3b: SparseSpec vs vLLM: prompts diverging, earliest, largest margin; R2 p-value | | n/a |
| 3c reloaded: mean, p99 \|Δ log p\|, max probability ratio | | |
| 3c base after round trip: mean \|Δ log p\| | | n/a |
| 3c negative control: mean \|Δ log p\|, tracking coefficients, `failure_class` | | n/a |
| 3c bitwise equal after reload / tied / pointers unchanged | | n/a |
| 3d: `timing_s/gen` step 1 and steps 2+, reward on step 3 | | (baseline smoke) |
| Smoke: gen time default / host_guard / no graphs | | n/a |
| Smoke: `batch_size_mean`, `kv_util_peak`, `num_offloads`, worst_case vs host_guard | | n/a |
| Smoke: gen time with / without log-probs (1 step each) | | n/a |
| Engine resident awake / asleep, peak during rollout (GiB) | | |
| Host RAM total / available; largest `cpu_offload_ratio` that fits | | n/a |
| Versions: torch, CUDA, transformers, verl, vllm, flashinfer, driver | | |

### What was run (CPU only)

| Check | Result |
|---|---|
| `python -m pytest serve/rl/tests integration/gates/tests integration/tests -q` (local: torch 2.15, transformers 5.24, verl 0.5.0) | 783 passed, 3 skipped (the GPU module, one test that needs `datasets`, shellcheck not installed) |
| Same on a box-like overlay (torch 2.7.1, transformers 4.57.1, verl 0.5.0) | 783 passed, 3 skipped. In both, four tests that send Ctrl-C to the gate runner time out if two copies of the suite run at the same time on a loaded machine; run on their own they pass |
| `integration/verl-0.5.0-sparsespec.patch` against the pristine verl 0.5.0 wheel | applies with `git apply` and `patch -p1`, both files compile, reverse-applies |
| `DRY_RUN=1 bash baseline/run_seed_sparsespec.sh` composed with verl's real hydra config | composes; differs from `run_seed.sh` only in `actor_rollout_ref.rollout.*`, experiment name and checkpoint dir |
| Rollout output vs a verbatim copy of the tail of `vLLMRollout.generate_sequences` | identical keys, shapes, dtypes, padding, masks, position ids |
| Gates 3b / 3c with `--engine reference`, tiny random Qwen3 on CPU | pass on a working engine; seven deliberately broken engines each caught; three-way 3b from files with synthetic vLLM data |
| Admission: page-level transcription of upstream's offloading scheduler (source hashes pinned), 2 x 2000 random schedules per mode | no pool ever exhausted, no request offloaded while prefilling, host peak within the proven bound |
| `run_all_gates.sh`, `build_sparsespec_env.sh`, `setup_env.sh`, `smoke_reference.sh` against stub `conda` / `nvidia-smi` / `python` / `pip` / `git` under bash 3.2 | stage order, env switching, skips, refusals, timeouts, interrupt, bundle and summary content |
| 3a: CPU transcription of the sampler arithmetic, chi-square against the exact law | sampler as written p = 0.44 to 0.86; three broken variants rejected |

Review: after each round, independent reviewers read the code against upstream `serve/`, verl 0.5.0,
vLLM 0.9.1 and torch sources, and every finding was checked by a second reader. Round 1: 34 confirmed,
all fixed except one latent case verl never triggers (calling the engine from inside a caller's
`with torch.device(...)` block). Round 2: 3 in the admission code; 22 of 24 confirmed in the gates,
the runner and the build scripts (19 distinct); then 10 more in a second review of those fixes. All
fixed; the ones that mattered are listed under "Review fixes that mattered". This is reading and CPU
simulation, not running.

## What is on the branch

| Path | What |
|---|---|
| `HANDOFF.md` | one page for Harrison |
| `integration/build_sparsespec_env.sh` | builds `verl-sparsespec`: clone of `verl-baseline` + FlashInfer fork + RAFT / `librmm.so` + this repo + pytest; `--check`, `--verify`, drift report |
| `integration/setup_env.sh` | installs the patched verl 0.5.0 into an env; `--check` |
| `integration/run_all_gates.sh`, `summarize_bundle.py` | every gate in one command; bundle + one-screen summary |
| `integration/gates/` | 3b (three-way) + its stock-vLLM half, 3c + stock-vLLM reference, 3d checker, `smoke_compare`, synthetic checkpoint |
| `integration/smoke_reference.sh` | reduced-size verl run on the HF reference engine |
| `integration/verl-0.5.0-sparsespec.patch` | the only change to verl: one branch in `_build_rollout`, one statement for metric names |
| `serve/rl/engine.py` | `SparseSpecEngine`: persistent, `generate` returns target log-probs, in-place `load_weights`, `sleep` / `wake_up` |
| `serve/rl/scheduler.py` | `RLSpecScheduler`: log-prob capture, EOS kept, admission (`worst_case` / `host_guard`), stats |
| `serve/rl/reference_engine.py` | same interface on HF transformers |
| `serve/rl/verl_rollout.py` | `SparseSpecRollout`, sharding manager, config |
| `serve/rl/weights.py`, `memory.py`, `base.py` | weight mapping, memory plan and admission bounds, shared contract |
| `serve/rl/tests/`, `integration/gates/tests/`, `integration/tests/` | CPU tests; `test_engine_gpu.py` for the box |
| `baseline/run_seed_sparsespec.sh` | the run script (SEED=0, 100 steps) |

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
+actor_rollout_ref.rollout.sparsespec.cuda_graph=True          # SPARSESPEC_CUDA_GRAPH
+actor_rollout_ref.rollout.sparsespec.admission=worst_case     # SPARSESPEC_ADMISSION
actor_rollout_ref.rollout.calculate_log_probs=True             # SPARSESPEC_CALC_LOGPROBS
```

`rollout.gpu_memory_utilization=0.45` (inherited) is the engine's whole GPU share.

Per-step metrics in the timing CSV (`sparsespec/...`): `accept_len_mean`, `num_cycles`, `weight_sync_s`,
`wake_s` (includes `graph_capture_s`), `gen_s`, `sleep_s`, `accept_rate`, `num_iters`,
`batch_size_mean`, `num_offloads`, `kv_util_peak`, `host_util_peak`, `inflight_cap`, `inflight_peak`,
`peak_mem_gb`, `resident_asleep_gb`, `process_peak_alloc_gb`, `process_peak_reserved_gb`.
`accept_len_mean` is tokens emitted per verify cycle (accepted drafts + 1).

## Concurrency: `worst_case` vs `host_guard`, and why `max_batch_size` stays 64

Numbers are from the planning code for an A100 40 GB at the 0.45 share, 1024-token prompts and 8192 new
tokens (the config's limits). Nothing here is measured.

| | In flight at once | Host RAM page-locked |
|---|---|---|
| stock vLLM | up to 256 by its default `max_num_seqs`, in practice whatever fits its KV pool (about 15 GiB, roughly 140K tokens) | none |
| SparseSpec, `worst_case`, default host pool | 26 (a few more when prompts are short) | 16 GiB |
| SparseSpec, `host_guard`, default host pool | 45 | 16 GiB |
| SparseSpec, `worst_case`, `cpu_offload_ratio=3.0` | 43 | 32 GiB |
| SparseSpec, `host_guard`, `cpu_offload_ratio=3.0` | 64 (= `max_batch_size`) | 32 GiB |

**`worst_case`** admits a request only if everything in flight would still fit in GPU pool + host pool
with every request running to 8192 tokens. It can never run out of memory and almost never offloads,
but it leaves the KV pool mostly empty: most responses are far shorter than the limit.

**`host_guard`** admits on current free GPU capacity, as upstream's `offloading` policy does, and
offloads the shortest sequence to the host when the GPU pool fills. Upstream's failure is running out
of host pool (`alloc_page_cpu` "Out of memory"). The guard is a fixed cap on requests in flight, derived
from upstream's own rules (the victim is the shortest active request; offloaded requests are restored
last-in first-out; nothing is admitted while anything is offloaded): with those, the host pool can
never hold more than the sum of `kv // k` over the active counts k at which an offload can happen, so
the cap is the largest count whose bound fits. A request is only ever admitted later, never dropped,
restarted or truncated. One change to upstream's behaviour was needed: a request is not admitted in the
same scheduler call that restores an offloaded one (upstream can do both, and could then offload a
request that is still prefilling, which its own code asserts never happens).

Evidence for the bound: proof in `serve/rl/memory.py::host_guard_host_bound`; an independent reviewer
who tried to break it over thousands of schedules (tight to 0.999 of the bound, never above); and the
CPU simulator in `serve/rl/tests/offload_sim.py`, a transcription of upstream's scheduler whose source
hashes are pinned so that an upstream change fails the test. On the box, the offload stress GPU test
runs in both modes with a pool small enough to force multi-step offloads.

**Why not `max_batch_size=256`.** Every unit of `max_batch_size` reserves GPU memory inside the same
0.45 share, whether or not that many requests ever run:

| `max_batch_size` | per-request buffers | verification scratch | KV pool left | in flight (`worst_case` / `host_guard`) |
|---|---|---|---|---|
| 64 | 0.69 GiB | 2.00 GiB | 10.76 GiB = 100.7K tokens | 26 / 45 |
| 128 | 1.38 GiB | 3.11 GiB | 8.96 GiB = 83.9K tokens | 16 / 20 |
| 256 | 2.76 GiB | 5.90 GiB | 4.79 GiB = 44.8K tokens | 8 / 9 |

So a larger `max_batch_size` lowers real concurrency. On top of that, the engine's shared draft index
buffer allows at most 126 requests with 1024-token prompts. What decides concurrency is KV memory, for
vLLM as well: its 256 is reachable only while sequences average under about 550 tokens. The honest
comparison is realised concurrency, which both smoke runs report (`batch_size_mean`, `inflight_peak`).

Recommendation: decide after the smoke comparison. If `host_guard` passes its GPU tests and its smoke,
and its gen time is better, launch the real run with `SPARSESPEC_ADMISSION=host_guard`: dynamic KV
offloading is part of the method under test, and `worst_case` hides it. If the box has the RAM
(`smoke_compare.md` says), `cpu_offload_ratio=3.0` on top lifts `host_guard` to the full 64.

## Things the brief assumed that turned out different

1. **Per-request memory is about 11 MiB, not 7.** Besides `draft_probs` (5.8 MiB) every in-flight
   request holds a PillarAttn selected-index buffer of 5.25 MiB.
2. **`run_inference` cannot build Qwen3-0.6B as is.** No entry in `ModelArgs.element_size()`; hub ids
   for single-file checkpoints fail; and `Attention` creates Qwen3's `q_norm` / `k_norm` only when the
   model **path** contains "qwen3", so a checkpoint under `.../hf/global_step_25` would silently lose
   them. The engine sizes memory from the loaded parameters, resolves hub ids, aliases such directories
   through a symlink named `qwen3-model`, and verifies its weights bitwise after loading.
3. **Upstream EOS handling does not match vLLM's contract** (EOS stripped, early EOS missed, overshoot by
   up to 9 tokens). `RLSpecScheduler` keeps the EOS, drops everything after it and cuts to exactly
   `max_new_tokens`.
4. **`mem_fraction` in `run_inference` covers only weights + KV pool.** Here it is the whole engine
   share, like vLLM's `gpu_memory_utilization`: weights 1.11 GiB, fixed kernel buffers 1.06, CUDA graphs
   1.69 (estimate), per-request buffers 0.69, scratch 2.0, safety 0.5, KV pool 10.76 GiB.
5. **Tied embeddings are not tied in the stock engine** (311 MB extra). The wrapper re-ties them.
6. **Asynchronous CPU scheduling is off** (upstream benchmarks use it). Synchronous only; costs the
   SparseSpec arm some throughput.
7. **Asserts are on.** Upstream scripts run with `PYTHONOPTIMIZE=1`; under verl they are enabled, so any
   upstream assert our workload trips stops a run. The full-size smoke is the test for that.
8. **Upstream can offload a request that is still prefilling**, which its restore path asserts never
   happens. Prevented by the admission rule above.

Changes to `serve/` outside `serve/rl/` (both in `serve/model/model.py::KVPool`, defaults unchanged):
an optional `cpu_capacity` argument, and an O(1) double-free check for host pages.

### Review fixes that mattered

Round 1: a per-page host-free assert that cost about 2 s per restore; the host pool being rounded up to
32 GiB of locked RAM; a torch device context left installed in the trainer thread; 24 `gc.collect()`
passes per wake; engine INFO logs suppressed under verl; memory accounting gaps.

Round 2:
- Admission: a restore and an admission in the same scheduler call bypassed the GPU-headroom check.
- Gate 3c could label an engine with noisy log-probs "checkpoint too close, not an engine failure",
  because the bar came from the engine's own noise. The label now needs agreement with stock vLLM on
  the same checkpoint.
- The runner counted a two-way 3b result as "3b PASS" and could exit 0 without any vLLM comparison.
- The build script's final check ran `ldd` outside python on a library without rpath and would have
  reported every correct build as failed.
- The offload stress test never reached the multi-step offload path that long training responses take.
- A timed-out stage could leave its own processes holding the GPU; Ray worker logs were not bundled;
  each smoke left a 7 GB checkpoint behind; a crashed gate was indistinguishable from a FAIL verdict.
- The build told Harrison to start the gates from a fresh shell, which loses the CUDA toolkit export the
  engine needs to compile its kernels; missing training data was recorded but would only have failed
  hours into the run. Both are now refusals at start with the fix printed.

## Fairness: what differs between the arms besides the decoding algorithm

Facts from the sources (verl 0.5.0, vLLM 0.9.1, this branch); nothing here is measured.

| Topic | Baseline (vLLM) | SparseSpec arm | Favors |
|---|---|---|---|
| CUDA graphs | **eager** (verl's default `enforce_eager=True`) | graphs on, re-captured every step (`sparsespec/graph_capture_s`, inside `timing_s/gen`) | SparseSpec; the no-graph smoke measures how much |
| Engine version | vLLM V1 in-process by source reading; confirm from the baseline log | `serve/` synchronous scheduler; async CPU off | baseline |
| GPU KV pool inside the 0.45 share | roughly 15 GiB | 10.76 GiB planned | baseline |
| Concurrency | KV-bound, nominal limit 256, optimistic admission, preempt and recompute | 26 (`worst_case`) or 45 (`host_guard`) in flight with the default host pool | baseline |
| Host RAM | weights copy while asleep (about 1.2 GB) | 16 GiB pinned offload pool for the whole run | n/a (resource) |
| GPU memory during training | about 0 | about 2.2 GiB planned stays resident (weights + fixed buffers) | baseline (headroom) |
| Prefix caching | on: the 8 samples of a prompt share prompt KV | none | baseline, small |
| Per-step lifecycle | weights restored from host on wake | no weight restore, but KV re-allocation + graph capture | unknown |
| Sampling seed / RNG | global CUDA stream, seeded 1000 | engine-private stream, seeded 0; trainer RNG untouched | none |
| `calculate_log_probs` | False | True (diagnostics; cost measured by the smoke) | baseline |
| torch / vLLM versions | torch pinned by vLLM 0.9.1 | same: the env is a clone and the drift report enforces it | none |
| verl `perf/max_memory_*` | lifetime peak | per-step peak (engine resets the counters) | not comparable; use `sparsespec/*` memory columns |

`timing_s/gen` wraps the same verl code in both arms (weight sync, wake, generation, sleep, Ray
transfer). For a like-for-like generation time also compare `timing_s/generate_sequences`.

## Still open

1. **The baseline config looks like it cannot start on one GPU.** `grpo_base.sh` does not set
   `rollout.tensor_model_parallel_size`; verl 0.5.0's default is 2 and `_build_rollout` asserts it
   divides the world size (1). If the baseline did run on the box, something there differs from the
   committed config. I did not touch `grpo_base.sh`; the SparseSpec script sets the key to 1 inside its
   own rollout block. If the baseline needs the same line, that is a `CONFIG_VERSION` bump for the group.
2. **Which admission mode the real run uses** (see the recommendation above). Decide from the bundle.
3. **Step-25 negative control.** If it comes back `negative_control_underpowered`, rerun on step 50
   (RUNBOOK section 5) or accept the tracking coefficients; a threshold change is not on the table.
4. **Upstream observation, not fixed:** after a KV restore the sparse-attention selection of that request
   can be stale for one draft cycle (the restore path assumes position-ordered indices, the top-k kernel
   does not write them in order). It cannot change outputs or target log-probs (rejection sampling is
   exact for whatever the draft was); it can only lower acceptance slightly right after a restore.

## Launching the real run

Only after you have read the bundle and the required gates passed:

```bash
conda activate verl-sparsespec
nvidia-smi
bash baseline/run_seed_sparsespec.sh        # SEED=0, STEPS=100, EXP_NAME=sparsespec_seed0
# or, if you choose host_guard:  SPARSESPEC_ADMISSION=host_guard bash baseline/run_seed_sparsespec.sh
```
