# vLLM GRPO baseline arm

Everything for the **vanilla verl + stock vLLM** baseline runs (David's
part of the work split). This directory is self-contained and does
**not** use the SparseSpec engine — it only shares this repo's prompt
convention (`qwen3-math-thinking`) and hands its checkpoints to the
SparseSpec/replay side.

All scripts run on the **A100 Linux box**, not locally.

## Order of operations

```bash
# 1. One-time environment (SEPARATE env from the SparseSpec build —
#    SparseSpec pins a custom vLLM fork; this arm needs stock vLLM).
bash baseline/setup_env.sh
conda activate verl-baseline

# 2. Data: MATH levels 3-5 train + MATH-500 val -> parquet
python baseline/prepare_data.py

# 3. Smoke test: 10 steps. Check reward > 0, thinking-mode lengths,
#    no OOM, and REPORT STEP TIME TO THE GROUP (fixes the step budget).
bash baseline/run_smoke.sh

# 4. The two baseline seeds (~100 steps each, checkpoints every 25)
SEED=0 bash baseline/run_seed.sh
SEED=1 bash baseline/run_seed.sh

# 5. Convert checkpoints to weight-only HF format (the handoff artifact)
bash baseline/merge_checkpoints.sh vllm_baseline_seed0
bash baseline/merge_checkpoints.sh vllm_baseline_seed1
# ... then upload $CKPT_ROOT/<exp>/hf/ to the shared Drive.

# 6. Offline MATH-500 eval on every checkpoint (same stock-vLLM engine
#    is used for BOTH arms' checkpoints, so the evaluator isn't a confound)
bash baseline/eval_all.sh vllm_baseline_seed0
bash baseline/eval_all.sh vllm_baseline_seed1
```

## The config contract

[`config/grpo_base.sh`](config/grpo_base.sh) is the **canonical shared
config** for both arms (model, data, GRPO hyperparameters, sampling,
batch limits, memory settings). Rules:

- The SparseSpec arm reuses this file and may change **only** the
  `actor_rollout_ref.rollout.*` block (the engine).
- Any other change requires bumping `CONFIG_VERSION` and telling the
  group — otherwise the two arms silently stop being comparable.
- If 8K responses blow the step budget, the sanctioned fallback is
  `data.max_response_length=4096` — as a `CONFIG_VERSION` bump applied
  to *both* arms, never as a one-off.

## Outputs

| Path | What |
|---|---|
| `$CKPT_ROOT/<exp>/global_step_N/` | verl FSDP checkpoints (every 25 steps) |
| `$CKPT_ROOT/<exp>/hf/global_step_N/` | weight-only HF checkpoints — **handoff to replay/SparseSpec** |
| `baseline/logs/<exp>.log` | full console log |
| `baseline/results/<exp>.timing.csv` | per-step metrics: `timing_s/gen` (rollout), `timing_s/update_actor` (train), `timing_s/step` (total), reward, response lengths |
| `baseline/results/runs.jsonl` | one entry per run: config version, git hash, GPU type, GPU-hours, checkpoint list |
| `baseline/results/math500_summary.jsonl` | pass@1 (4 samples/problem) per checkpoint |

## Conventions shared with the rest of the project

- **Prompt**: user message is the problem +
  `"\nPlease think step by step and then provide only the final answer within \boxed{}."`,
  rendered through the Qwen3 chat template with thinking enabled —
  identical to `eval/benchmarks/utils.py` (`qwen3-math-thinking`).
- **Reward/grading**: `math_grader.py` is the single source of truth;
  the training reward (`reward.py`) and the eval (`eval_math500.py`)
  both import it. Unparseable (no `\boxed{}`) → reward 0, and the
  unparseable rate is reported by the eval.
- **Seeds** control data order (`data.seed`); vLLM sampling itself is
  not bit-deterministic, which is fine — the two seeds exist to
  estimate run-to-run spread.

## Known risks / first things to check on the A100

- **Memory split (40 GB)**: training FSDP + vLLM engine share the GPU.
  `rollout.gpu_memory_utilization=0.45` leaves ~22 GB for policy +
  Adam states + activations. If rollout OOMs late in training (long
  responses), lower it before touching anything else.
- **`+data.seed` override**: on some verl versions `data.seed` already
  exists in the schema and hydra will error with "key already exists" —
  drop the `+` in `run_seed.sh` if so.
- **Metric names**: `parse_timing.py` captures *all* numeric metrics,
  so if a verl version renames `timing_s/gen`, the data is still in the
  CSV — only the printed summary keys may need updating.
