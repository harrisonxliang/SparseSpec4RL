#!/usr/bin/env bash
# Run MATH-500 eval on the base model + every merged checkpoint of a run.
# Usage: bash baseline/eval_all.sh vllm_baseline_seed0
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
export REPO_ROOT
source "$REPO_ROOT/baseline/config/grpo_base.sh"

EXP_NAME=${1:?usage: eval_all.sh <exp_name>}
EXP_DIR="$CKPT_ROOT/$EXP_NAME"
DATA_ARG="$DATA_DIR/math500_test.parquet"

# Step 0 = the base model (shared across all runs; eval it once under
# exp name "base" rather than per seed if you prefer).
python3 "$REPO_ROOT/baseline/eval_math500.py" \
    --ckpt "$MODEL_PATH" --exp-name "$EXP_NAME" --step 0 --data "$DATA_ARG"

shopt -s nullglob
for hf_dir in "$EXP_DIR"/hf/global_step_*; do
    step=${hf_dir##*global_step_}
    python3 "$REPO_ROOT/baseline/eval_math500.py" \
        --ckpt "$hf_dir" --exp-name "$EXP_NAME" --step "$step" --data "$DATA_ARG"
done

echo "all evals appended to baseline/results/math500_summary.jsonl"
