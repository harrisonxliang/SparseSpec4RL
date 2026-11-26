#!/usr/bin/env bash
# Environment for the vLLM BASELINE arm. Run on the A100 Linux box.
#
# IMPORTANT: this is a SEPARATE conda env from the SparseSpec build.
# SparseSpec pins a custom vLLM fork (3rdparty/vllm) and a custom
# FlashInfer; the baseline arm must use STOCK vLLM. Never install both
# into the same env.
set -euo pipefail

ENV_NAME=${ENV_NAME:-verl-baseline}

conda create -n "$ENV_NAME" python=3.11 -y
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Pinned combo documented as compatible by verl 0.5.x. If this box has a
# wheel problem, the known-good fallback pair is verl==0.4.1 + vllm==0.8.5.post1.
pip install vllm==0.9.1
pip install verl==0.5.0

# FSDP training path wants flash-attn; qwen3-0.6B falls back to SDPA
# without it, which is slower but correct — so failure here is non-fatal.
pip install flash-attn --no-build-isolation || echo "WARN: flash-attn build failed, continuing with SDPA"

pip install datasets pandas pyarrow

python -c "import verl, vllm; print('verl', verl.__version__, '| vllm', vllm.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "setup_env: OK — activate with: conda activate $ENV_NAME"
