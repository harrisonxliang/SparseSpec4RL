#!/usr/bin/env bash
# Run one GRPO baseline arm (verl + stock vLLM rollout).
#
# Usage:
#   bash baseline/run_seed.sh                 # seed 0, 100 steps
#   SEED=1 bash baseline/run_seed.sh          # seed 1, 100 steps
#   STEPS=10 EXP_NAME=smoke bash baseline/run_seed.sh
#
# Produces:
#   $CKPT_ROOT/$EXP_NAME/global_step_{25,50,75,100}/   verl checkpoints
#   baseline/logs/$EXP_NAME.log                        full console log
#   baseline/results/$EXP_NAME.timing.csv              per-step metrics
#   baseline/results/runs.jsonl                        run metadata entry
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
export REPO_ROOT
source "$REPO_ROOT/baseline/config/grpo_base.sh"

SEED=${SEED:-0}
STEPS=${STEPS:-100}
EXP_NAME=${EXP_NAME:-vllm_baseline_seed${SEED}}

LOG_DIR="$REPO_ROOT/baseline/logs"
RESULTS_DIR="$REPO_ROOT/baseline/results"
mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$CKPT_ROOT/$EXP_NAME"
LOG_FILE="$LOG_DIR/$EXP_NAME.log"

START_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "[run_seed] $EXP_NAME: seed=$SEED steps=$STEPS config=$CONFIG_VERSION"
echo "[run_seed] logging to $LOG_FILE"

set +e
python3 -m verl.trainer.main_ppo \
    "${BASE_ARGS[@]}" \
    +data.seed="$SEED" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.default_local_dir="$CKPT_ROOT/$EXP_NAME" \
    trainer.total_epochs=100 \
    trainer.total_training_steps="$STEPS" \
    2>&1 | tee "$LOG_FILE"
STATUS=${PIPESTATUS[0]}
set -e
# NOTE: if the +data.seed override fails with "key already exists",
# your verl version has data.seed in its schema — drop the leading '+'.

END_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

python3 "$REPO_ROOT/baseline/parse_timing.py" \
    "$LOG_FILE" -o "$RESULTS_DIR/$EXP_NAME.timing.csv" || true

python3 "$REPO_ROOT/baseline/log_run.py" \
    --exp-name "$EXP_NAME" --seed "$SEED" --steps "$STEPS" \
    --config-version "$CONFIG_VERSION" --engine vllm \
    --started-at "$START_TS" --ended-at "$END_TS" \
    --exit-status "$STATUS" \
    --ckpt-dir "$CKPT_ROOT/$EXP_NAME" \
    -o "$RESULTS_DIR/runs.jsonl"

echo "[run_seed] done, exit status $STATUS"
exit "$STATUS"
