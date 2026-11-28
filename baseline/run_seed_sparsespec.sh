#!/usr/bin/env bash
# SparseSpec TREATMENT arm — runs from the SAME canonical config as the
# vLLM baseline (config/grpo_base.sh). Per the config contract, the
# rollout engine block below is the ONLY thing allowed to differ
# between the two arms.
#
# Usage (A100 box, in the env with the patched verl — see RUNBOOK.md):
#   bash baseline/run_seed_sparsespec.sh                         # the real run: seed 0, 100 steps
#   STEPS=3 EXP_NAME=sparsespec_smoke bash baseline/run_seed_sparsespec.sh
#   SPARSESPEC_ENGINE=reference STEPS=3 EXP_NAME=reference_smoke bash baseline/run_seed_sparsespec.sh
#   DRY_RUN=1 bash baseline/run_seed_sparsespec.sh               # print the hydra overrides and exit
#
# Everything else (data, reward, GRPO hyperparams, checkpoint cadence,
# logging) is inherited and must not be overridden here.
#
# IMPORTANT: SEED stays 0 — the proposal (section 3.1) shares seeds and
# data order across arms, so this run must mirror vllm_baseline_seed0.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
export REPO_ROOT
source "$REPO_ROOT/baseline/config/grpo_base.sh"

SEED=${SEED:-0}
STEPS=${STEPS:-100}
EXP_NAME=${EXP_NAME:-sparsespec_seed${SEED}}

# sparsespec = the real engine (serve/, needs Harrison's build);
# reference  = HF transformers behind the same interface (plumbing tests only).
SPARSESPEC_ENGINE=${SPARSESPEC_ENGINE:-sparsespec}

SPARSESPEC_ARGS=(
    # ---- rollout engine block: the only sanctioned diff vs baseline ----
    # Later overrides win in hydra, so these replace rollout.name=vllm from
    # BASE_ARGS. Sampling (T=1.0, top_p=1.0, top_k=-1, n=8), the memory share
    # (gpu_memory_utilization=0.45) and free_cache_engine=True are inherited.
    actor_rollout_ref.rollout.name=sparsespec
    # The engine runs on one GPU. verl's default is 2, and _build_rollout
    # asserts it divides the world size (1 here).
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    # Proposal 3.1 treatment config: draft length k=8, PillarAttn s=0.05,
    # minimum sparse budget 128 tokens (paper default).
    +actor_rollout_ref.rollout.sparsespec.engine="$SPARSESPEC_ENGINE"
    +actor_rollout_ref.rollout.sparsespec.spec_stride=8
    +actor_rollout_ref.rollout.sparsespec.budget_ratio=0.05
    +actor_rollout_ref.rollout.sparsespec.num_min_budget=128
    +actor_rollout_ref.rollout.sparsespec.kv_cache=pillar
    +actor_rollout_ref.rollout.sparsespec.max_batch_size=64
    +actor_rollout_ref.rollout.sparsespec.cuda_graph=True
    # Diagnostics only: also return the engine's target log-probs so verl logs
    # training/rollout_probs_diff_* (engine vs actor) every step. The actor
    # still recomputes old_log_probs itself; the loss does not use these.
    actor_rollout_ref.rollout.calculate_log_probs=True
    # Weight reload needs no flag: the sharding manager copies the FSDP
    # actor's weights into the engine before every rollout.
)

TRAIN_ARGS=(
    "${BASE_ARGS[@]}"
    "${SPARSESPEC_ARGS[@]}"
    +data.seed="$SEED"
    trainer.experiment_name="$EXP_NAME"
    trainer.default_local_dir="$CKPT_ROOT/$EXP_NAME"
    trainer.total_epochs=100
    trainer.total_training_steps="$STEPS"
)

if [ "${DRY_RUN:-0}" = "1" ]; then
    # One hydra override per line (also used by serve/rl/tests to validate
    # the overrides against verl's config schema).
    printf '%s\n' "${TRAIN_ARGS[@]}"
    exit 0
fi

# The patched verl imports serve.rl.verl_rollout inside the Ray workers,
# which inherit this environment.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if ! python3 -c "import serve.rl.verl_rollout" 2>/dev/null; then
    echo "ERROR: cannot import serve.rl.verl_rollout (wrong env? see RUNBOOK.md)." >&2
    python3 -c "import serve.rl.verl_rollout" >&2 || true
    exit 1
fi
if ! python3 - <<'EOF'
import inspect, sys
from verl.workers import fsdp_workers
sys.exit(0 if '"sparsespec"' in inspect.getsource(fsdp_workers.ActorRolloutRefWorker._build_rollout) else 1)
EOF
then
    echo "ERROR: this env's verl does not have integration/verl-0.5.0-sparsespec.patch applied." >&2
    echo "       Run: bash integration/setup_env.sh <env-name>   (see RUNBOOK.md)" >&2
    exit 1
fi

LOG_DIR="$REPO_ROOT/baseline/logs"
RESULTS_DIR="$REPO_ROOT/baseline/results"
mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$CKPT_ROOT/$EXP_NAME"
LOG_FILE="$LOG_DIR/$EXP_NAME.log"

START_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "[run_sparsespec] $EXP_NAME: seed=$SEED steps=$STEPS config=$CONFIG_VERSION engine=$SPARSESPEC_ENGINE"
echo "[run_sparsespec] logging to $LOG_FILE"

set +e
python3 -m verl.trainer.main_ppo "${TRAIN_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
STATUS=${PIPESTATUS[0]}
set -e
# NOTE: if the +data.seed override fails with "key already exists",
# your verl version has data.seed in its schema — drop the leading '+'
# (same note as run_seed.sh; keep the two scripts in sync).

END_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

python3 "$REPO_ROOT/baseline/parse_timing.py" \
    "$LOG_FILE" -o "$RESULTS_DIR/$EXP_NAME.timing.csv" || true

if [ "$SPARSESPEC_ENGINE" = "sparsespec" ]; then
    python3 "$REPO_ROOT/baseline/log_run.py" \
        --exp-name "$EXP_NAME" --seed "$SEED" --steps "$STEPS" \
        --config-version "$CONFIG_VERSION" --engine sparsespec \
        --started-at "$START_TS" --ended-at "$END_TS" \
        --exit-status "$STATUS" \
        --ckpt-dir "$CKPT_ROOT/$EXP_NAME" \
        -o "$RESULTS_DIR/runs.jsonl"
else
    # log_run.py only knows the engines vllm|sparsespec; a reference-engine
    # plumbing run must not be recorded as a SparseSpec run.
    echo "[run_sparsespec] engine=$SPARSESPEC_ENGINE: not appending to runs.jsonl"
fi

echo "[run_sparsespec] done, exit status $STATUS"
exit "$STATUS"
