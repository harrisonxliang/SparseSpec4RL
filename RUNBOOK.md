# RUNBOOK: SparseSpec rollout backend on the A100 box

Everything on branch `sparsespec-verl-backend` was written on a machine without a GPU. The CPU-only parts
were run there; nothing that needs CUDA, flashinfer or vLLM has been executed yet. `HANDOFF.md` is the
one-page version of sections 1 to 3 for whoever runs the box. `STATUS.md` says what is verified, records
the decisions, and has the tables the numbers go into.

Run every command from the repo root.

## 0. Ground rules on the shared box

- `nvidia-smi` before every GPU job. `run_all_gates.sh` and the gate scripts check free memory themselves
  and refuse to start if there is not enough. Nothing here kills, signals or renices a process it did not
  start. If a teammate's run is using the card, wait.
- Never modify the `verl-baseline` conda env. The build and setup scripts refuse to, and fingerprint that
  env before and after as a tripwire.
- `baseline/config/grpo_base.sh`, `reward.py`, data prep and the baseline scripts are untouched by this
  branch. The SparseSpec arm differs from the baseline only in `actor_rollout_ref.rollout.*`
  (`DRY_RUN=1 bash baseline/run_seed_sparsespec.sh` prints the full override list).
- If a gate fails, record the numbers and stop. Do not change a threshold to get a pass.
- `baseline/results/` is git-ignored (`*results`). To commit a result file:
  `git add -f baseline/results/sparsespec_checks/<file>`.

```bash
git fetch origin && git checkout sparsespec-verl-backend && git pull
```

## 1. Build the `verl-sparsespec` env

```bash
bash integration/build_sparsespec_env.sh --check     # read-only: prerequisites and the plan
bash integration/build_sparsespec_env.sh             # build; every step is restartable, rerun after a failure
bash integration/setup_env.sh verl-sparsespec        # patched verl 0.5.0 (integration/verl-0.5.0-sparsespec.patch)
bash integration/build_sparsespec_env.sh --verify    # verification + drift report against verl-baseline
```

What the env is: a `--copy` clone of `verl-baseline` (stock vLLM 0.9.1, verl 0.5.0, same torch), plus

- the FlashInfer fork, `pip install --no-build-isolation --no-deps -e 3rdparty/flashinfer`;
- RAFT headers and `librmm.so`, built by `3rdparty/raft/build.sh libraft` and installed into the env
  prefix (where `serve/attention/backend.py` looks);
- this repo, `pip install --no-deps -e .` (`--no-deps` because the project's pins would downgrade
  `datasets` and break hydra's `antlr4` runtime; see STATUS.md);
- pytest (the gate runner needs it and installs nothing itself);
- **not** the vLLM fork: `serve/` only needs `vllm._custom_ops`, which stock 0.9.1 has, and the fork pins
  another torch.

The build needs a CUDA toolkit with `nvcc` matching torch's CUDA major version (the engine JIT-compiles
kernels at first use), gcc 9.3+, cmake 3.30.4+ (installed into a private tools venv), network access.
`--check` reports each of these and how to fix it. `--verify` imports `serve.rl.engine`, JIT-compiles
the RAFT top-k op, and fails if torch, transformers, verl, vllm, numpy or tensordict differ from
`verl-baseline`. Logs: `integration/logs/`; reports: `integration/build/verl-sparsespec/report/`.

The build ends by printing the packaging commands for the shared Drive (conda-pack plus manifests).

Without Harrison's build, `bash integration/setup_env.sh` alone creates `verl-sparsespec-dev` (clone +
patched verl), in which only `engine=reference` works. `VERL_IN_PLACE=1` patches the installed verl
instead of cloning it from GitHub.

## 2. Run every gate: one command

```bash
tmux new -s gates
unset INSTALL_PREFIX PREFIX; export CUDA_HOME=...    # the line the build printed as step 3 of "next"
bash integration/run_all_gates.sh          # any conda env or none; 5 to 12 hours (estimate)
bash integration/run_all_gates.sh --help   # stages, options, timeouts, rough durations
```

The export line matters: the engine JIT-compiles its kernels at first start with the CUDA toolkit that
`CUDA_HOME` names (plus `FLASHINFER_EXTRA_LDFLAGS` / `CXX` if the build needed them), and a new tmux
session does not inherit the exports of the shell it was started from. `--verify` prints the line again.

Stages, in order. A failed stage does not stop the run; a stage whose dependency failed is SKIPped with
the reason.

| Stage | Env | What |
|---|---|---|
| `env` | both | versions (torch, CUDA, transformers, verl, vllm, flashinfer, driver), `pip freeze`, verl patch state, `nvidia-smi`, host RAM, disk |
| `cpu_tests` | sparsespec | the CPU tests of the engine wrapper and the gate scripts (`serve/rl/tests`, `integration/gates/tests`) |
| `engine_import` | sparsespec | `import serve.rl.engine` |
| `reference_plumbing` | sparsespec | `integration/smoke_reference.sh`: verl end to end on the HF reference engine, reduced size |
| `gpu_unit_tests` (+ `_host_guard`, `_offload`, `_offload_host_guard`) | sparsespec | `serve/rl/tests/test_engine_gpu.py` in both admission modes, plus the offload stress in both |
| `gate_3b_vllm` | baseline | stock vLLM 0.9.1, `enforce_eager`, greedy: token ids + per-step top-2 logit margins to JSON |
| `gate_3b` | sparsespec | three-way greedy comparison from files |
| `gate_3c_synthetic` | sparsespec | reload gate on a synthetic checkpoint (base + 2% noise) |
| `gate_3c_vllm_reference` | baseline | the same log-prob measurement for stock vLLM (step-25 checkpoint, else the synthetic one) |
| `gate_3c_step25` | sparsespec | reload gate on `$CKPT_ROOT/vllm_baseline_seed0/hf/global_step_25`; SKIP with the path if missing |
| `smoke_worst_case` (+ `_check`) | sparsespec | gate 3d: 3 full-size training steps, default config |
| `smoke_host_guard` (+ `_check`) | sparsespec | the same with `admission=host_guard` |
| `smoke_no_graph` (+ `_check`) | sparsespec | the same with `cuda_graph=False` |
| `smoke_no_logprobs` (+ `_check`) | sparsespec | 1 step with `calculate_log_probs=False` |
| `smoke_compare` | sparsespec | the four smoke runs side by side |

It refuses to start (exit 2, within minutes, with the command that fixes each problem) when:

- the GPU is not free, or conda or the `verl-sparsespec` env is missing;
- pytest or the verl patch is missing from that env;
- `INSTALL_PREFIX` / `PREFIX` is set and points somewhere without `lib/librmm.so`;
- the engine's JIT would find no CUDA compiler (`$PYTORCH_NVCC`, else `nvcc` under torch's `CUDA_HOME` or
  `/usr/local/cuda`): paste the export line from the build;
- `$DATA_DIR/math_l3to5_train.parquet` or `math500_test.parquet` is missing.

Each check applies only when a selected stage needs it. A missing `verl-baseline` only skips the two
stock-vLLM stages (gate 3b then cannot pass). Less than 24 GiB of available host RAM is a warning.
Before every later GPU stage the runner waits up to 15 minutes for the memory to be free, then SKIPs that
stage. After every stage command, and when the runner is interrupted, it stops what is left of that
stage's own process group and nothing else.

Output: `baseline/results/gate_bundles/gates_<host>_<UTC time>/` and the same name `.tar.gz`. The last
lines printed are the one-screen summary and `send this file back: <tarball>`. The bundle holds one log
per stage with full tracebacks, `gates/*.json|md`, the smoke CSVs and logs, the env record and
`SUMMARY.txt`. Exit status: 0 all required gates passed (a missing step-25 checkpoint is allowed), 1 a
required gate failed or did not run, 2 refused to start. Required: GPU unit tests (all four), 3b
(three-way), 3c synthetic, 3c step-25 if the checkpoint exists, 3d (`smoke_worst_case` and its check).

Useful options (environment variables):

```bash
STAGES=gate_3b_vllm,gate_3b bash integration/run_all_gates.sh              # only these stages
SKIP_STAGES=smoke_no_graph,smoke_no_logprobs bash integration/run_all_gates.sh   # leave the write-up-only smokes for later
MIN_FREE_GPU_GIB=30 bash integration/run_all_gates.sh                      # default: card total minus 1.5 GiB
CKPT_ROOT=/data/ckpts DATA_DIR=/data/math bash integration/run_all_gates.sh
```

Leave the timeouts on (`TIMEOUT_BIN=none` also removes the process group that lets the runner clean up
after an interrupted stage; then check `nvidia-smi` yourself before rerunning).

Each smoke run is a normal run of `baseline/run_seed_sparsespec.sh` under the experiment name
`gates_smoke_<label>`: it appends one entry to `baseline/results/runs.jsonl`, and its checkpoint
directory under `$CKPT_ROOT` is removed again after the run.

## 3. Reading the result

`SUMMARY.txt` has one line per stage, then the numbers that matter, then `VERDICT`. To stay on one
screen, stages that all passed can share a row (`gpu_unit_tests + 3 variants`, `smoke runs, 3d check
PASS`). A stage that produced no verdict is labelled `REFUSED`, `no verdict`, `crashed` or `timed out`,
with the gate's last message or the exception line underneath; none of these counts as passed.

**3b, three-way greedy.** 16 MATH prompts, temperature 0, 256 new tokens, the same prompts in all three
engines. For each pair (SparseSpec vs HF, vLLM vs HF, SparseSpec vs vLLM) the report gives the number of
diverging prompts, the first divergence position and the HF logit margin there. PASS needs:

- R1: every divergence of every pair is a near-tie (HF margin at most `--tie-margin`, default 1.0
  logits). A vLLM-vs-HF divergence that is not a near-tie also fails, labelled as a problem of the
  reference engines, not of SparseSpec.
- R2: SparseSpec's divergence profile against HF is no worse than vLLM's against HF. Default
  `--profile-rule significance`: fail if SparseSpec's per-token divergence rate is significantly higher
  (one-sided exact test, alpha 0.05). `--profile-rule strict` is the literal rule (no more diverging
  prompts and not earlier). Both are printed whichever decides; why the literal rule is not the default
  is in STATUS.md.

A two-way result (no vLLM JSON) is information only and never passes 3b. `REFUSED` means the inputs of
the two envs do not match (different prompts, EOS ids or model): nothing was compared.

**3c, reload.** 64 prompts x 4 samples, T=1.0, up to 2048 new tokens. One engine instance goes base ->
checkpoint (reloaded while awake, under live CUDA graphs) -> base -> checkpoint. Checks: every parameter
bitwise equal to the checkpoint after reload, tied weights tied, no parameter pointer changed;
`reloaded` mean / p99 / max |Δ log p| engine vs trainer forward; `base_after_roundtrip` in the same
range; `negative_control` (engine on base weights vs trainer on checkpoint weights) clearly larger,
default rule 3x the reloaded mean plus tracking coefficients when their standard error is small; greedy
fingerprints identical before and after a round trip. `failure_class` in the JSON:

| `failure_class` | Meaning |
|---|---|
| `null` | passed |
| `negative_control_underpowered` | only the negative-control mean rule failed, the two trainer models differ by less than the rule demands, and stock vLLM on the same checkpoint is as noisy as this engine: the checkpoint is too close to the base model. Rerun on step 50 (section 5). |
| `negative_control_inconclusive` | the same failure without a usable stock-vLLM reference: cannot tell a close checkpoint from a noisy engine. Get the reference, rerun. |
| `engine` | anything else. A real failure. |

**3d and the smoke comparison.** `smoke_worst_case_check` is gate 3d: no OOM, three steps, reward > 0 on
some step, all `sparsespec/*` columns present and sane. `gates/smoke_compare.md` puts the runs side by
side: `timing_s/gen` (step 1 and steps 2+) for default / host_guard / no graphs; `batch_size_mean`,
`inflight_peak`, `kv_util_peak`, `host_util_peak`, `num_offloads` for the two admission modes; gen time
with and without log-probs; host RAM total and available, and which `cpu_offload_ratio` would fit.

How to read the concurrency columns: `batch_size_mean` near `inflight_cap` with `kv_util_peak` well
below 1 means the admission cap, not GPU memory, limits throughput (expected under `worst_case`).
`kv_util_peak` near 1 with `num_offloads` > 0 means the GPU pool is the limit and offloading is doing
its job (expected under `host_guard`).

Other things worth reading in a smoke log (`smoke/gates_smoke_<label>.log`, lines `SparseSpecEngine ...`
prefixed by the Ray worker name): the memory plan, `X GiB resident while awake`, `asleep: freed ...`,
and `sparsespec/peak_mem_gb` against 0.45 x the card.

**GPU unit tests.** One small engine per process: greedy vs HF, the response contract around the
speculation stride, log-prob agreement with the trainer forward, RNG stream continuity and isolation,
in-place bitwise reload, sleep / wake with CUDA graphs, KV pool clean after generate. The offload
variants use an 8192-token KV pool and 24 requests of 1536 new tokens, so sequences are offloaded to the
host and restored in several steps, and check that outputs and log-probs still match the no-offload
reference.

## 4. Running one piece by hand

Rerun selected stages into a new bundle with `STAGES=...` (section 2). The direct commands, for
debugging (`export PYTHONPATH=$PWD` first; outputs go to `baseline/results/sparsespec_checks/` unless
`--out-dir` is given):

```bash
# GPU unit tests (verl-sparsespec)
python -m pytest serve/rl/tests/test_engine_gpu.py -v -s --log-cli-level=INFO --color=no
SPARSESPEC_TEST_ADMISSION=host_guard python -m pytest serve/rl/tests/test_engine_gpu.py -v -s --log-cli-level=INFO
SPARSESPEC_TEST_OFFLOAD_STRESS=1 python -m pytest serve/rl/tests/test_engine_gpu.py -v -s --log-cli-level=INFO

# 3b: stock vLLM half (verl-baseline), then the gate (verl-sparsespec)
conda run --no-capture-output -n verl-baseline env PYTHONPATH=$PWD \
    python integration/gates/gate_b_vllm_greedy.py --out baseline/results/sparsespec_checks/gate_b_vllm_greedy.json
python -m integration.gates.gate_b_greedy --engine sparsespec \
    --vllm-json baseline/results/sparsespec_checks/gate_b_vllm_greedy.json --tag three_way
python -m integration.gates.gate_b_greedy --compare-only --stored-margins --tag three_way \
    --vllm-json baseline/results/sparsespec_checks/gate_b_vllm_greedy.json --profile-rule strict   # recompute the verdict from the three token files

# 3c (verl-sparsespec; the reference in verl-baseline)
python -m integration.gates.gate_c_reload --engine sparsespec --synthetic --tag sparsespec_synthetic

# 3d: one smoke run and its check (needs the whole card)
rm -rf "${CKPT_ROOT:?source baseline/config/grpo_base.sh first}/sparsespec_smoke"
STEPS=3 EXP_NAME=sparsespec_smoke bash baseline/run_seed_sparsespec.sh
python -m integration.gates.gate_d_smoke_check --exp-name sparsespec_smoke --steps 3 --engine sparsespec

# CPU tests (no GPU; never in verl-baseline)
python -m pytest serve/rl/tests integration/gates/tests integration/tests -q --ignore=serve/rl/tests/test_engine_gpu.py
```

A smoke run that reaches its last step saves a checkpoint, and verl's default `resume_mode=auto` makes
the next run with the same `EXP_NAME` resume from it; hence the `rm -rf` (it removes only that smoke's
own directory). The runner does this itself.

`run_seed_sparsespec.sh` environment: `SEED` (0), `STEPS` (100), `EXP_NAME` (`sparsespec_seed$SEED`),
`SPARSESPEC_ENGINE` (`sparsespec` | `reference`), `SPARSESPEC_CUDA_GRAPH` (`True`),
`SPARSESPEC_ADMISSION` (`worst_case` | `host_guard`), `SPARSESPEC_CALC_LOGPROBS` (`True`), `DRY_RUN=1`.

## 5. If step 25 comes back `negative_control_underpowered` or `_inconclusive`: rerun on step 50

25 steps at lr 1e-6 move most weights by less than half a bf16 step, so the step-25 model may be nearly
the base model in bf16 and no engine could satisfy "negative control clearly larger". The gate then says
so explicitly, with the numbers. `SUMMARY.txt`, the console and the end of the stage's log
(`logs/*_gate_3c_step25.log` in the bundle) print these two commands with the real paths filled in, to
paste as they are:

```bash
cd <repo>
C=<CKPT_ROOT>/vllm_baseline_seed0/hf/global_step_50      # needs the baseline's step-50 checkpoint
nvidia-smi                                               # both need the card
conda run --no-capture-output -n verl-baseline env PYTHONPATH=. \
    python integration/gates/gate_c_vllm_reference.py --ckpt "$C" --tag step50
conda run --no-capture-output -n verl-sparsespec env PYTHONPATH=. \
    python -m integration.gates.gate_c_reload --engine sparsespec --ckpt "$C" --tag sparsespec_step50 \
    --vllm-reference baseline/results/sparsespec_checks/gate_c_vllm_reference_step50.json
# -> baseline/results/sparsespec_checks/gate_c_reload_sparsespec_step50.{json,md}
```

The bitwise reload check, the reloaded means and the tracking coefficients do not depend on how far the
checkpoint is from the base model; only the mean-based negative control does. `--neg-control tracking`
makes the tracking coefficients the deciding rule instead; that is a decision for the team, not a
default.

## 6. Launching the real run

After the required gates passed and Dylan has confirmed:

```bash
tmux new -s sparsespec          # a new shell: exports and env activation happen inside it
unset INSTALL_PREFIX PREFIX; export CUDA_HOME=...    # the same line as for the gate run
conda activate verl-sparsespec
echo "$CONDA_DEFAULT_ENV"       # must print verl-sparsespec
# re-export any non-default CKPT_ROOT / DATA_DIR / MODEL_PATH here
nvidia-smi
bash baseline/run_seed_sparsespec.sh
# or: SPARSESPEC_ADMISSION=host_guard bash baseline/run_seed_sparsespec.sh
# seed 0, 100 steps, checkpoints every 25 under $CKPT_ROOT/sparsespec_seed0/
# log: baseline/logs/sparsespec_seed0.log    metrics: baseline/results/sparsespec_seed0.timing.csv
```

Restarting after a crash: the same command resumes from the last checkpoint under
`$CKPT_ROOT/sparsespec_seed0/`, but the script truncates the log and rewrites the CSV. Move them aside
first and merge afterwards:

```bash
mv baseline/logs/sparsespec_seed0.log baseline/logs/sparsespec_seed0.attempt1.log
cp baseline/results/sparsespec_seed0.timing.csv baseline/results/sparsespec_seed0.attempt1.timing.csv
bash baseline/run_seed_sparsespec.sh
cat baseline/logs/sparsespec_seed0.attempt1.log baseline/logs/sparsespec_seed0.log > baseline/logs/sparsespec_seed0.merged.log
python baseline/parse_timing.py baseline/logs/sparsespec_seed0.merged.log -o baseline/results/sparsespec_seed0.timing.csv
```

`runs.jsonl` gets one entry per attempt; the run's GPU-hours are their sum. Afterwards, as for the
baseline: `bash baseline/merge_checkpoints.sh sparsespec_seed0` and `bash baseline/eval_all.sh
sparsespec_seed0` (in the baseline env: the offline eval uses stock vLLM for both arms).

## 7. Knobs (all inside `actor_rollout_ref.rollout.sparsespec.*`)

`max_batch_size`, `cuda_graph` and `admission` are already set in `SPARSESPEC_ARGS` of
`baseline/run_seed_sparsespec.sh`: change `cuda_graph` and `admission` with `SPARSESPEC_CUDA_GRAPH` /
`SPARSESPEC_ADMISSION` in the environment, `max_batch_size` by editing its existing line. Do not add a
second line for a key that is already there (hydra: `Could not append to config`). Other keys: add
`+actor_rollout_ref.rollout.sparsespec.<key>=<value>` lines.

| Key | Default | Effect |
|---|---|---|
| `admission` | `worst_case` | `worst_case`: admit only what fits if every in-flight sequence runs to the length limit (26 in flight). `host_guard`: admit on current free GPU capacity and offload to host, capped so the host pool cannot run out (45 in flight). STATUS.md has the numbers. |
| `max_batch_size` | 64 | upper bound on in-flight requests; reserves about 11 MiB of GPU memory each plus scratch, so raising it shrinks the KV pool and lowers real concurrency |
| `cpu_offload_ratio` | 1.5 | host offload pool as a multiple of the GPU KV pool, sized to a power-of-two pinned block: 1.5 locks 16 GiB, 3.0 locks 32 GiB (43 in flight under `worst_case`, 64 under `host_guard`) |
| `kv_capacity_tokens` | auto | GPU KV pool size; auto = what is left of the 0.45 share |
| `transient_reserve_gb` | auto (2.0 at batch 64) | GPU memory kept free for sampler / verification scratch |
| `cuda_graph` | True | `SPARSESPEC_CUDA_GRAPH=False` to run eager |
| `chunk_size` | 256 | prefill chunk |

## 8. If something breaks

| Symptom | Likely cause / what to do |
|---|---|
| `run_all_gates.sh` exits 2 at once | it printed why and the command to fix it (GPU busy, env / pytest / verl patch missing, stray `INSTALL_PREFIX`); the bundle is still written |
| a stage says `no verdict` or `crashed` | the gate stopped before computing a result: `SUMMARY.txt` shows its last message or the exception line under the stage's row, and the traceback is in that stage's log in the bundle. Not a FAIL verdict, but the gate has not passed. |
| Ctrl-C on the runner seems to do nothing | press it again after a few seconds (bash can swallow an interrupt that lands while it starts a helper process). The summary and the tarball are still written. |
| a stage says `timed out`, later GPU stages are SKIPped as `GPU busy` | a hung stage; the runner stopped that stage's own processes. If memory is still held, the log lists the processes; nothing outside the runner's own process group is ever signalled. |
| 3b `REFUSED` | prompts, EOS ids or model differ between the two envs (for example a different transformers version renders the chat template differently); nothing was compared |
| `AssertionError: rollout world_size: 1 is not divisible by infer_tp: 2` in a **baseline** run | `grpo_base.sh` does not set `rollout.tensor_model_parallel_size`; verl's default is 2. Baseline owner's call (STATUS.md, still open). The SparseSpec script sets it to 1. |
| `this env's verl does not have ... patch applied` | `bash integration/setup_env.sh verl-sparsespec` |
| `ModuleNotFoundError: flashinfer`, `librmm.so is not found`, RAFT headers missing | the build is incomplete or `INSTALL_PREFIX` / `PREFIX` is set in the shell: `unset INSTALL_PREFIX PREFIX`, then `bash integration/build_sparsespec_env.sh --verify` |
| the first engine build fails compiling a kernel (`nvcc`, `cannot find -lcudart`) | CUDA toolkit problem: `bash integration/build_sparsespec_env.sh --check` names it |
| `a SparseSpecEngine already exists in this process` | one engine per process (kernel buffers live in class attributes) |
| `MemoryError: no memory left for the KV pool` | share too small for the non-KV parts: lower `max_batch_size` |
| engine build fails pinning host memory, or the worker is OOM-killed at start | not enough free host RAM for the 16 GiB pinned pool: free RAM, or `cpu_offload_ratio=1.0` (locks 8 GiB, lowers concurrency) |
| CUDA OOM during rollout | raise `transient_reserve_gb` or set a smaller `kv_capacity_tokens`; the plan and measured numbers are in the log |
| CUDA OOM during `update_actor` that the baseline does not have | the engine keeps about 2.2 GiB (weights + fixed buffers) on the GPU while asleep |
| an upstream `assert` inside `serve/` fires mid-rollout | upstream's own scripts run with `PYTHONOPTIMIZE=1`; here asserts are on. Record the traceback; do not set `PYTHONOPTIMIZE` for the training process |
| `restoring an offloaded request that is still prefilling` warning | the admission invariant was broken; the run continues safely, but report it with the log |
| a verl stage dies with `ActorDiedError` / `Check python-core-worker-*.log` | the Ray worker died natively; for a stage that did not pass, its Ray session logs are copied into the bundle under `ray_logs/<stage>/` |
