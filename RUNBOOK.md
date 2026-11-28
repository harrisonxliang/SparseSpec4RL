# RUNBOOK: SparseSpec rollout backend on the A100 box

Everything on branch `sparsespec-verl-backend` was written on a machine without a GPU. The CPU-only parts
were run there (unit tests, hydra composition of the run script, patch application); nothing that needs
CUDA, flashinfer or vLLM has been executed yet. This file is the exact sequence to run on the box.
`STATUS.md` says what is and is not verified and has the table the numbers go into.

Run every command from the repo root.

## 0. Ground rules on the shared box

- `nvidia-smi` before every GPU job. The gate scripts check free memory themselves and refuse to start if
  there is not enough (they never touch other processes). If a teammate's run is using the card, wait.
- Never modify the `verl-baseline` conda env. `integration/setup_env.sh` refuses to, and hashes that env's
  verl files before and after as a tripwire.
- `baseline/config/grpo_base.sh`, `reward.py`, data prep and the baseline scripts are untouched by this
  branch. The SparseSpec arm differs from the baseline only in `actor_rollout_ref.rollout.*`
  (`DRY_RUN=1 bash baseline/run_seed_sparsespec.sh` prints the full override list).
- Gate outputs land in `baseline/results/sparsespec_checks/`. The repo's `.gitignore` ignores `*results`,
  so commit them with `git add -f baseline/results/sparsespec_checks/<file>`.

```bash
git fetch origin && git checkout sparsespec-verl-backend && git pull
export PYTHONPATH=$PWD            # serve/ is a namespace package, not installed
source baseline/config/grpo_base.sh   # exports MODEL_PATH, DATA_DIR, CKPT_ROOT (REPO_ROOT may be unset here; fine)
```

## 1. Environments

### 1a. Now: a private clone of verl-baseline (reference engine only)

```bash
bash integration/setup_env.sh              # creates verl-sparsespec-dev from verl-baseline, installs patched verl 0.5.0
conda activate verl-sparsespec-dev
python -m pip install pytest               # test runner for sections 2 and 4; no setup step or dependency installs it
```

The script clones verl `v0.5.0` into `integration/.verl-0.5.0` (git-ignored), applies
`integration/verl-0.5.0-sparsespec.patch` and installs it editable with `--no-deps`. No network or no git
tag: `VERL_IN_PLACE=1 bash integration/setup_env.sh` patches the clone's installed verl in place instead.
It ends by printing the versions and whether the SparseSpec engine is importable (it is not in this env;
`engine=reference` works).

### 1b. When Harrison's `verl-sparsespec` env exists

```bash
bash integration/setup_env.sh verl-sparsespec     # only installs the patched verl into it
conda activate verl-sparsespec
python -m pip install pytest                      # test runner for sections 2 and 4
python -c "import serve.rl.engine; print('SparseSpec engine importable')"
```

`setup_env.sh` expects the target env to already contain verl 0.5.0 (true for any clone of `verl-baseline`)
and has itself never been executed (no conda on the machine it was written on), so read its output on the
first run.

For Harrison's build (see STATUS.md, fairness): the engine needs the flashinfer fork, RAFT headers and
`librmm.so` under `$CONDA_PREFIX`, plus `numba`, `jinja2`, `wandb`. It does **not** need the vLLM fork:
`serve/` only imports `vllm._custom_ops`, which stock vLLM 0.9.1 has. Keeping stock vLLM 0.9.1 keeps
torch at the baseline's version (the fork pins torch 2.8.0), so the training path runs on the same torch
in both arms.

Record versions in both envs for STATUS.md:

```bash
for env in verl-baseline verl-sparsespec; do
  conda run -n $env python -c "import torch, transformers, verl; print('$env', 'torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__, 'verl', verl.__version__)"
  conda run -n $env python -c "import vllm; print('$env', 'vllm', vllm.__version__)"
  conda run -n $env python -c "import flashinfer; print('$env', 'flashinfer', flashinfer.__version__)" || true
done
```

## 2. CPU tests (`verl-sparsespec-dev` or `verl-sparsespec`, no GPU needed, about a minute)

```bash
python -m pytest serve/rl/tests integration/gates/tests -q --ignore=serve/rl/tests/test_engine_gpu.py
```

Expected: everything passes; four tests of `test_verl_patch.py` skip in an env whose verl is already
patched. `test_engine_gpu.py` is left out on purpose: without `--ignore` it is skipped only where CUDA or
flashinfer is missing, so in `verl-sparsespec` this command would build the real engine on the shared card.
Run it with the command in section 4. Do not run the tests in `verl-baseline`: pytest is not installed
there and that env must not be modified.

## 3. Reference-engine plumbing (dev env, before the real engine exists)

These prove the verl side (rollout contract, weight sync, sleep/wake order, metrics, run script) and the
gate scripts themselves. The reference engine is HF transformers; it is slow and is not a timing baseline.

```bash
nvidia-smi
# gate scripts end to end with the reference engine (about 24 GiB free needed at the default share;
# add --mem-fraction 0.2 if the card is busy)
python -m integration.gates.gate_b_greedy --engine reference --tag reference
python -m integration.gates.gate_c_reload --engine reference --synthetic \
    --num-prompts 16 --n 2 --max-new-tokens 512 --tag reference_synthetic

# verl end to end at reduced size (8 prompts x 8 samples, 512-token responses, 3 steps; minutes)
bash integration/smoke_reference.sh
```

`smoke_reference.sh` uses the override list of the real run script with the engine switched to
`reference`, shrinks batch and response length (so it is NOT a comparable run and never writes
`runs.jsonl`), then runs gate 3d's checker on `baseline/results/reference_plumbing.timing.csv`.
Pass = exit 0: three steps, reward > 0 on some step, all `sparsespec/*` columns present, acceptance
stats 0. `baseline/run_seed_sparsespec.sh` refuses `SPARSESPEC_ENGINE=reference` at full size (hours per
step) unless `ALLOW_FULL_SIZE_REFERENCE=1`.

## 4. The gates with the real engine (`verl-sparsespec` env)

In order. Each writes JSON + markdown under `baseline/results/sparsespec_checks/` and exits non-zero on
FAIL. If a gate fails, stop and record the numbers; do not change thresholds to get a pass.

### 3a. Sampler review (already done, by code reading)

`baseline/results/sparsespec_checks/3a_sampler_review.md`. Nothing to run.

### Engine unit tests (GPU, about 15 GiB free)

```bash
nvidia-smi
mkdir -p baseline/results/sparsespec_checks
python -m pytest serve/rl/tests/test_engine_gpu.py -v -s --log-cli-level=INFO 2>&1 | tee baseline/results/sparsespec_checks/engine_gpu_tests.log
```

One small engine (default share 0.25, `SPARSESPEC_TEST_MEM_FRACTION` to change; the module skips itself
if less than share + 5 GiB is free) runs: greedy vs HF, the
response contract around the speculation stride, log-prob agreement with the trainer forward, RNG stream
continuity and isolation, in-place bitwise reload, sleep/wake with CUDA graphs (memory freed, identical
greedy output after two cycles), KV pool clean after generate. The first run compiles the triton and
flashinfer kernels, so it is slow once.

Read the engine's log lines `SparseSpecEngine memory plan`, `ready in ...: X GiB resident while awake` and
`asleep: freed ..., Y GiB still resident` (pytest shows them under "live log setup" / "live log call"; they
need `--log-cli-level=INFO`), and copy them into STATUS.md. Do not use the test's own
`[sleep N] allocated ...` line for the asleep figure: it is process-wide and includes the HF reference model.

### 3b. Greedy check

```bash
nvidia-smi
python -m integration.gates.gate_b_greedy --engine sparsespec
# -> baseline/results/sparsespec_checks/gate_b_greedy_sparsespec.{json,md}
```

16 MATH prompts, 256 new tokens, temperature 0, engine vs HF on the same weights. The report lists the
first divergence position per prompt and the HF logit margin there. Late divergences at near-ties are
bf16 noise; a divergence against a large margin, or any early one, is a bug. The verdict rule and its
thresholds are printed (`--tie-margin`, default 1.0).

### 3c. Reload gate

Until the real checkpoint is uploaded, use the synthetic one (base weights + 2% relative noise, created
on the fly under `$CKPT_ROOT/synthetic/`):

```bash
nvidia-smi
python -m integration.gates.gate_c_reload --engine sparsespec --synthetic --tag sparsespec_synthetic
```

With the baseline's step-25 checkpoint (`$CKPT_ROOT/vllm_baseline_seed0/hf/global_step_25`):

```bash
# 1) reference numbers from STOCK vLLM, in the baseline env (it must not see the SparseSpec env)
conda activate verl-baseline
export PYTHONPATH=$PWD
nvidia-smi
python integration/gates/gate_c_vllm_reference.py --ckpt $CKPT_ROOT/vllm_baseline_seed0/hf/global_step_25
# -> baseline/results/sparsespec_checks/gate_c_vllm_reference.{json,md}

# 2) the gate itself, in the SparseSpec env, with the vLLM numbers side by side
conda activate verl-sparsespec
export PYTHONPATH=$PWD
nvidia-smi
python -m integration.gates.gate_c_reload --engine sparsespec \
    --ckpt $CKPT_ROOT/vllm_baseline_seed0/hf/global_step_25 \
    --vllm-reference baseline/results/sparsespec_checks/gate_c_vllm_reference.json \
    --tag sparsespec_step25
```

Defaults are the spec: 64 prompts x 4 samples, T=1.0, up to 2048 new tokens. One engine instance does
base -> checkpoint (reload while awake, under live CUDA graphs) -> base -> checkpoint, and checks:

- every parameter bitwise equal to the checkpoint after reload, tied weights tied, no parameter pointer
  changed (in place);
- `reloaded`: mean / p99 / max |Δ log p| and max probability ratio, engine vs trainer forward on the
  checkpoint weights;
- `base_after_roundtrip`: the same with base weights on both sides (same small range);
- `negative_control`: engine has base weights, trainer has checkpoint weights (must be clearly larger);
- greedy fingerprints identical before and after a round trip (stale-state check).

One thing to expect with the real checkpoint: 25 steps at lr 1e-6 move most weights by less than half a
bf16 step, so the step-25 model may be nearly the base model in bf16 and the mean-based negative control
may not separate. The report prints how far apart the two models are (fraction of bf16 weights that
differ, mean |Δ log p| between HF(base) and HF(checkpoint)) and a tracking coefficient (about 1 when the
engine's log-probs follow the checkpoint, about 0 when they follow base) with its standard error. If the
default rule fails for that reason, record the numbers and decide with the team; `--neg-control tracking`
makes the tracking coefficients the deciding rule, and a later checkpoint (step 50/100) gives the mean
rule more power. The bitwise check does not depend on any of this.

### 3d. Smoke run (3 steps, full size)

```bash
nvidia-smi          # needs the whole card, like a baseline run
rm -rf "${CKPT_ROOT:?source baseline/config/grpo_base.sh first}/sparsespec_smoke"   # every attempt starts from scratch, see below
STEPS=3 EXP_NAME=sparsespec_smoke bash baseline/run_seed_sparsespec.sh
python -m integration.gates.gate_d_smoke_check --exp-name sparsespec_smoke --steps 3 --engine sparsespec \
    --baseline-csv baseline/results/smoke.timing.csv        # the baseline's smoke CSV, if it is on the box
```

Pass = no OOM, three steps, `critic/score/mean` > 0 on some step, the six `sparsespec/*` columns present
and sane. The checker prints the step-time comparison with the baseline smoke when that CSV exists
(otherwise a NOTE; the verdict does not depend on it).

Every smoke attempt has to start without a checkpoint, hence the `rm -rf` (it only removes this smoke's own
directory). A smoke that reaches step 3 saves `$CKPT_ROOT/sparsespec_smoke/global_step_3` (verl saves on the
last step whenever `save_freq` > 0), and verl's default `trainer.resume_mode=auto` makes the next run with
the same `EXP_NAME` resume from it: only `step:4` runs, the log and CSV are overwritten, and gate 3d fails
with missing steps 1 to 3. A fresh `EXP_NAME` (and the same `--exp-name` for the checker) works too.

The engine's `SparseSpecEngine ...` lines appear in the run log prefixed by the Ray worker name.

Things to read in `baseline/logs/sparsespec_smoke.log` and the CSV before launching the real run:

| What | Where | Why |
|---|---|---|
| memory plan, resident awake / asleep | worker log lines `SparseSpecEngine ...` | share respected; what the trainer loses during updates (`sparsespec/resident_asleep_gb`) |
| `sparsespec/peak_mem_gb` vs 0.45 x card | CSV | engine stayed inside its share; a warning is logged if not |
| `worst-case KV admission limits concurrency to N` | worker log (warning, once) | realised concurrency; see section 6 |
| `sparsespec/batch_size_mean`, `kv_util_peak`, `num_offloads` | CSV | whether the GPU pool or the admission cap is binding |
| `sparsespec/wake_s`, `graph_capture_s`, `weight_sync_s`, `sleep_s` | CSV | per-step lifecycle cost (included in `timing_s/gen`) |
| `training/rollout_probs_diff_mean` / `_max` | CSV | engine log-probs vs the actor's recomputed ones, every step |
| `timing_s/gen`, `timing_s/generate_sequences`, `timing_s/step` | CSV | step time vs the baseline smoke; step 1 includes one-time kernel compilation |

Optional, to quantify the CUDA-graph effect (the baseline's vLLM runs eager; see STATUS.md):

```bash
rm -rf "${CKPT_ROOT:?source baseline/config/grpo_base.sh first}/sparsespec_eager_smoke"
SPARSESPEC_CUDA_GRAPH=False STEPS=3 EXP_NAME=sparsespec_eager_smoke bash baseline/run_seed_sparsespec.sh
```

## 5. Launching the real run

After 3b, 3c and 3d pass:

```bash
tmux new -s sparsespec          # from the repo root; the run takes about a day
```

Then inside the tmux session. It is a new shell: an env activated before `tmux new` does not carry over,
and with an already running tmux server neither do exported `CKPT_ROOT` / `DATA_DIR` / `MODEL_PATH`.

```bash
conda activate verl-sparsespec
echo "$CONDA_DEFAULT_ENV"       # must print verl-sparsespec
# re-export any non-default CKPT_ROOT / DATA_DIR / MODEL_PATH here
nvidia-smi
bash baseline/run_seed_sparsespec.sh
# seed 0, 100 steps, checkpoints every 25 under $CKPT_ROOT/sparsespec_seed0/
# log: baseline/logs/sparsespec_seed0.log    metrics: baseline/results/sparsespec_seed0.timing.csv
```

Without tmux: `conda activate verl-sparsespec`, then `nohup bash baseline/run_seed_sparsespec.sh &` in the
same shell.

Restarting after a crash: re-running the same command resumes from the last checkpoint under
`$CKPT_ROOT/sparsespec_seed0/` (step 25/50/75; from scratch if it died before step 25). That is intended,
but the script truncates the log and rewrites the CSV, so the steps before that checkpoint would disappear
from both. Move the log aside first and merge afterwards:

```bash
mv baseline/logs/sparsespec_seed0.log baseline/logs/sparsespec_seed0.attempt1.log
cp baseline/results/sparsespec_seed0.timing.csv baseline/results/sparsespec_seed0.attempt1.timing.csv
bash baseline/run_seed_sparsespec.sh
# one CSV over all steps; for steps both attempts ran, the later (resumed) lines win
cat baseline/logs/sparsespec_seed0.attempt1.log baseline/logs/sparsespec_seed0.log > baseline/logs/sparsespec_seed0.merged.log
python baseline/parse_timing.py baseline/logs/sparsespec_seed0.merged.log -o baseline/results/sparsespec_seed0.timing.csv
```

`runs.jsonl` gets one entry per attempt; the run's GPU-hours are their sum.

Afterwards, as for the baseline: `bash baseline/merge_checkpoints.sh sparsespec_seed0` and
`bash baseline/eval_all.sh sparsespec_seed0` (in the baseline env: the offline eval uses stock vLLM for
both arms).

## 6. Knobs (all inside `actor_rollout_ref.rollout.sparsespec.*`)

`max_batch_size` and `cuda_graph` are already set in `SPARSESPEC_ARGS` of `baseline/run_seed_sparsespec.sh`:
change `max_batch_size` by editing the value on its existing line there, and `cuda_graph` with
`SPARSESPEC_CUDA_GRAPH=False` in the environment. Do not add a second line for either: hydra stops with
`Could not append to config. An item is already at ...`. The other four keys are not set by the script: add
them as `+actor_rollout_ref.rollout.sparsespec.<key>=<value>` lines in `SPARSESPEC_ARGS`.

| Key | Default | Effect |
|---|---|---|
| `max_batch_size` | 64 | upper bound on in-flight requests; reserves about 11 MiB of GPU memory each |
| `cpu_offload_ratio` | 1.5 | host offload pool as a multiple of the GPU KV pool, then sized to a power-of-two pinned block: 1.5 locks 16 GiB of host RAM, 3.0 locks 32 GiB, about 5.6 or more locks 64 GiB (values in between fall back to the block below; the engine's warning prints the exact ratio for the card) |
| `kv_capacity_tokens` | auto | GPU KV pool size; auto = what is left of the 0.45 share after weights, buffers, graphs and scratch |
| `transient_reserve_gb` | 2.0 | GPU memory kept free for sampler / verification scratch |
| `cuda_graph` | True | `SPARSESPEC_CUDA_GRAPH=False` in the run script's environment |
| `chunk_size` | 256 | prefill chunk |

Concurrency. The scheduler admits a request only if the worst case (every in-flight request running to
8192 tokens) still fits in GPU pool + host pool, because upstream asserts when the host pool runs out.
With the defaults that is about 26 to 29 requests in flight, not 64 (the engine logs this once). If
`sparsespec/kv_util_peak` stays well below 1 and `batch_size_mean` sits at that cap, the cap is costing
throughput: raise `cpu_offload_ratio` if `free -g` shows the host RAM (3.0 gives about 46 in flight for
32 GiB pinned). If `kv_util_peak` is near 1, the GPU pool is the limit and the ratio will not help.

## 7. If something breaks

| Symptom | Likely cause / what to do |
|---|---|
| `AssertionError: rollout world_size: 1 is not divisible by infer_tp: 2` in a **baseline** run | `grpo_base.sh` does not set `rollout.tensor_model_parallel_size`; verl's default is 2. Baseline owner's call (STATUS.md, open decisions). The SparseSpec script sets it to 1. |
| `this env's verl does not have ... patch applied` | run `bash integration/setup_env.sh <env>` |
| `cannot import serve.rl.verl_rollout` | wrong env or `PYTHONPATH` not the repo root |
| `ModuleNotFoundError: flashinfer` / RAFT / `librmm.so` assertion at engine build | Harrison's build is not in this env; only `engine=reference` works |
| `a SparseSpecEngine already exists in this process` | one engine per process (kernel buffers live in class attributes) |
| `MemoryError: no memory left for the KV pool` | share too small for the non-KV parts: raise `gpu_memory_utilization` or lower `max_batch_size` |
| engine build fails pinning host memory | lower `cpu_offload_ratio` (1.0 locks 8 GiB) |
| CUDA OOM during rollout | raise `transient_reserve_gb` or set a smaller `kv_capacity_tokens`; the plan and measured numbers are in the log |
| CUDA OOM during `update_actor` that the baseline does not have | the engine keeps about 2.2 GiB (weights + fixed buffers) on the GPU while asleep; see STATUS.md |
| an upstream `assert` inside `serve/` fires mid-rollout | upstream's own scripts run with `PYTHONOPTIMIZE=1`; here asserts are on. Record the traceback; do not set `PYTHONOPTIMIZE` for the training process (it would also change the verl side of this arm only) |
| `KV pool not clean at ...` error line | pages leaked by a failed generate; the allocators were reset. Report it with the preceding traceback |
