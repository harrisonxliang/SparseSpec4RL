#!/usr/bin/env bash
# SparseSpec TREATMENT arm — runs from the SAME canonical config as the
# vLLM baseline (config/grpo_base.sh). Per the config contract, the
# rollout engine block below is the ONLY thing allowed to differ
# between the two arms.
#
# TODO(Dylan): wire the real backend keys once SparseSpec registers as
# a verl rollout backend, then delete the `exit 1` guard. Everything
# else (data, reward, GRPO hyperparams, checkpoint cadence, logging)
# is inherited and must not be overridden here.
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

SPARSESPEC_ARGS=(
    # ---- rollout engine block: the only sanctioned diff vs baseline ----
    actor_rollout_ref.rollout.name=sparsespec  # TODO(Dylan): actual backend name
    # Proposal 3.1 treatment config: draft length k=8, PillarAttn s=0.05.
    # TODO(Dylan): map onto the backend's real hydra keys, e.g.:
    # +actor_rollout_ref.rollout.num_draft_tokens=8
    # +actor_rollout_ref.rollout.kv_cache=pillar
    # +actor_rollout_ref.rollout.sparsity=0.05
    # TODO(Dylan): weight-reload path flag(s), if any, go here too.
)

echo "ERROR: engine wiring not filled in yet — see TODO(Dylan) above." >&2
exit 1

LOG_DIR="$REPO_ROOT/baseline/logs"
RESULTS_DIR="$REPO_ROOT/baseline/results"
mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$CKPT_ROOT/$EXP_NAME"
LOG_FILE="$LOG_DIR/$EXP_NAME.log"

START_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "[run_sparsespec] $EXP_NAME: seed=$SEED steps=$STEPS config=$CONFIG_VERSION"

set +e
python3 -m verl.trainer.main_ppo \
    "${BASE_ARGS[@]}" \
    "${SPARSESPEC_ARGS[@]}" \
    +data.seed="$SEED" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.default_local_dir="$CKPT_ROOT/$EXP_NAME" \
    trainer.total_epochs=100 \
    trainer.total_training_steps="$STEPS" \
    2>&1 | tee "$LOG_FILE"
STATUS=${PIPESTATUS[0]}
set -e

END_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

python3 "$REPO_ROOT/baseline/parse_timing.py" \
    "$LOG_FILE" -o "$RESULTS_DIR/$EXP_NAME.timing.csv" || true

python3 "$REPO_ROOT/baseline/log_run.py" \
    --exp-name "$EXP_NAME" --seed "$SEED" --steps "$STEPS" \
    --config-version "$CONFIG_VERSION" --engine sparsespec \
    --started-at "$START_TS" --ended-at "$END_TS" \
    --exit-status "$STATUS" \
    --ckpt-dir "$CKPT_ROOT/$EXP_NAME" \
    -o "$RESULTS_DIR/runs.jsonl"

echo "[run_sparsespec] done, exit status $STATUS"
exit "$STATUS"
