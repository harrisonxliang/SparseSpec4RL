#!/usr/bin/env bash
# Plumbing smoke test: the patched verl + sparsespec rollout backend with the
# REFERENCE engine (HF transformers), at a reduced problem size so it finishes
# in minutes. It proves the verl side works end to end (rollout contract,
# weight sync, sleep/wake ordering, metrics, timing CSV) before the real
# engine exists.
#
# NOT a comparable run: it shrinks the batch and the response length, which
# are outside the rollout block of the config contract. It never writes to
# baseline/results/runs.jsonl and uses its own experiment name.
#
#   bash integration/smoke_reference.sh              # 3 steps
#   STEPS=2 bash integration/smoke_reference.sh
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
export REPO_ROOT
STEPS=${STEPS:-3}
EXP_NAME=${EXP_NAME:-reference_plumbing}
PROMPTS=${PROMPTS:-8}                 # prompts per step (x rollout.n=8 samples)
RESPONSE_LEN=${RESPONSE_LEN:-512}

# The exact override list of the real run script, engine switched to reference.
mapfile -t ARGS < <(DRY_RUN=1 SPARSESPEC_ENGINE=reference STEPS="$STEPS" EXP_NAME="$EXP_NAME" \
    bash "$REPO_ROOT/baseline/run_seed_sparsespec.sh")

# Reductions (later overrides win). ppo_mini_batch_size must equal the number
# of prompts so one mini-batch covers the step, as in the full-size config.
ARGS+=(
    data.train_batch_size="$PROMPTS"
    actor_rollout_ref.actor.ppo_mini_batch_size="$PROMPTS"
    data.max_response_length="$RESPONSE_LEN"
    trainer.save_freq=-1
)

source "$REPO_ROOT/baseline/config/grpo_base.sh"   # for CKPT_ROOT / DATA_DIR defaults
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "[smoke_reference] free GPU memory before starting:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv

LOG_DIR="$REPO_ROOT/baseline/logs"
RESULTS_DIR="$REPO_ROOT/baseline/results"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"
LOG_FILE="$LOG_DIR/$EXP_NAME.log"

echo "[smoke_reference] $EXP_NAME: steps=$STEPS prompts/step=$PROMPTS response_len=$RESPONSE_LEN"
set +e
python3 -m verl.trainer.main_ppo "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"
STATUS=${PIPESTATUS[0]}
set -e

python3 "$REPO_ROOT/baseline/parse_timing.py" "$LOG_FILE" -o "$RESULTS_DIR/$EXP_NAME.timing.csv" || true
if [ "$STATUS" -eq 0 ]; then
    python3 "$REPO_ROOT/integration/gates/gate_d_smoke_check.py" \
        --csv "$RESULTS_DIR/$EXP_NAME.timing.csv" --engine reference --steps "$STEPS" || STATUS=1
fi
echo "[smoke_reference] done, exit status $STATUS"
exit "$STATUS"
