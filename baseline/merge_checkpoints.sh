#!/usr/bin/env bash
# Convert verl FSDP checkpoints to weight-only HuggingFace format.
#
# This is the HANDOFF ARTIFACT: the replay harness, the offline MATH-500
# eval, and the SparseSpec engine all load these HF directories. Layout:
#   $CKPT_ROOT/$EXP_NAME/hf/global_step_25/   (config + safetensors + tokenizer)
#   $CKPT_ROOT/$EXP_NAME/hf/global_step_50/   ...
#
# Usage: bash baseline/merge_checkpoints.sh vllm_baseline_seed0
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
export REPO_ROOT
source "$REPO_ROOT/baseline/config/grpo_base.sh"

EXP_NAME=${1:?usage: merge_checkpoints.sh <exp_name>}
EXP_DIR="$CKPT_ROOT/$EXP_NAME"

shopt -s nullglob
STEPS=("$EXP_DIR"/global_step_*)
if [ ${#STEPS[@]} -eq 0 ]; then
    echo "no checkpoints under $EXP_DIR" >&2
    exit 1
fi

for step_dir in "${STEPS[@]}"; do
    step=$(basename "$step_dir")
    target="$EXP_DIR/hf/$step"
    if [ -d "$target" ]; then
        echo "skip $step (already merged)"
        continue
    fi
    echo "merging $step -> $target"
    python3 -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$step_dir/actor" \
        --target_dir "$target"
    # Make each HF dir self-contained: replay/eval should not need to
    # know which base model to pull the tokenizer from.
    if [ ! -f "$target/tokenizer_config.json" ]; then
        python3 - "$MODEL_PATH" "$target" <<'EOF'
import sys
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained(sys.argv[1]).save_pretrained(sys.argv[2])
EOF
    fi
done

echo "merged checkpoints:"
ls -d "$EXP_DIR"/hf/global_step_* 2>/dev/null
echo "next: upload $EXP_DIR/hf to the shared Drive"
