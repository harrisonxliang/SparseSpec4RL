#!/usr/bin/env bash
# Put the patched verl 0.5.0 (integration/verl-0.5.0-sparsespec.patch) into a
# conda env for the SparseSpec arm. Run on the A100 Linux box.
#
#   bash integration/setup_env.sh                    # env "verl-sparsespec-dev": cloned from verl-baseline if missing
#   bash integration/setup_env.sh verl-sparsespec    # Harrison's env, once it exists: only installs the patched verl
#
# What it does to the TARGET env, and nothing else:
#   1. creates it as a clone of $BASE_ENV (default verl-baseline) if it does not exist;
#   2. checks out verl v0.5.0 into integration/.verl-0.5.0 (git-ignored), applies the patch,
#      and installs it editable with --no-deps (replacing the env's stock verl).
# It does NOT build SparseSpec / FlashInfer / RAFT / the vLLM fork (Harrison owns that build),
# and it refuses to touch $BASE_ENV. The baseline env's verl files are hashed before and
# after as a tripwire.
#
# Options (environment variables):
#   BASE_ENV=verl-baseline     env to clone from when the target does not exist
#   VERL_SRC=<dir>             where the editable verl checkout lives
#   VERL_IN_PLACE=1            no network / no git tag: patch the target env's installed verl in place
set -euo pipefail

ENV_NAME=${1:-${ENV_NAME:-verl-sparsespec-dev}}
BASE_ENV=${BASE_ENV:-verl-baseline}
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
PATCH="$REPO_ROOT/integration/verl-0.5.0-sparsespec.patch"
VERL_SRC=${VERL_SRC:-$REPO_ROOT/integration/.verl-0.5.0}

# sha256 of the two files in the verl==0.5.0 PyPI wheel the patch was generated against
SHA_FSDP_WORKERS=79598ab019228a331825e315ff474322460162ae7adb436e1097ec2a32780f62
SHA_RAY_TRAINER=dc6de1e241259e6384bf8299fca4152d4162d0505e4ff99840058afbdd7c1b70

if [ "$ENV_NAME" = "$BASE_ENV" ]; then
    echo "ERROR: refusing to modify the baseline env '$BASE_ENV'. Pass a different env name." >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

env_prefix() { conda env list | awk -v n="$1" '$1 == n {print $NF}'; }
verl_dir_of() {  # path of the verl package inside an env prefix, without activating it
    "$1/bin/python" - <<'EOF' 2>/dev/null || true
import importlib.util, os
spec = importlib.util.find_spec("verl")
print(os.path.dirname(spec.origin) if spec and spec.origin else "")
EOF
}
hash_baseline() {
    local prefix dir
    prefix=$(env_prefix "$BASE_ENV")
    [ -n "$prefix" ] || { echo "none"; return; }
    dir=$(verl_dir_of "$prefix")
    [ -n "$dir" ] || { echo "none"; return; }
    cat "$dir/workers/fsdp_workers.py" "$dir/trainer/ppo/ray_trainer.py" | sha256sum | cut -d' ' -f1
}

BASELINE_HASH_BEFORE=$(hash_baseline)

if [ -z "$(env_prefix "$ENV_NAME")" ]; then
    if [ -z "$(env_prefix "$BASE_ENV")" ]; then
        echo "ERROR: neither '$ENV_NAME' nor '$BASE_ENV' exists. Run baseline/setup_env.sh first." >&2
        exit 1
    fi
    echo "[setup_env] creating '$ENV_NAME' as a clone of '$BASE_ENV'"
    conda create -y -n "$ENV_NAME" --clone "$BASE_ENV"
fi
conda activate "$ENV_NAME"
if [ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]; then
    echo "ERROR: failed to activate '$ENV_NAME' (active: ${CONDA_DEFAULT_ENV:-none})" >&2
    exit 1
fi

INSTALLED=$(python -c "import importlib.metadata as m; print(m.version('verl'))" 2>/dev/null || echo "none")
echo "[setup_env] env '$ENV_NAME': installed verl = $INSTALLED"
if [ "$INSTALLED" != "0.5.0" ]; then
    echo "ERROR: expected verl 0.5.0 in '$ENV_NAME' (the patch targets 0.5.0), found '$INSTALLED'." >&2
    exit 1
fi

apply_patch_in() {  # $1 = directory that contains verl/
    local root=$1
    if (cd "$root" && patch -p1 --dry-run -R -s -f < "$PATCH" >/dev/null 2>&1); then
        echo "[setup_env] patch already applied in $root"
    elif (cd "$root" && patch -p1 --dry-run -s -f < "$PATCH" >/dev/null 2>&1); then
        (cd "$root" && patch -p1 -s < "$PATCH")
        echo "[setup_env] patch applied in $root"
    else
        echo "ERROR: the patch neither applies nor is already applied in $root." >&2
        echo "       sha256 of the files it expects (verl==0.5.0 wheel):" >&2
        echo "         $SHA_FSDP_WORKERS  verl/workers/fsdp_workers.py" >&2
        echo "         $SHA_RAY_TRAINER  verl/trainer/ppo/ray_trainer.py" >&2
        (cd "$root" && sha256sum verl/workers/fsdp_workers.py verl/trainer/ppo/ray_trainer.py) >&2 || true
        exit 1
    fi
}

if [ "${VERL_IN_PLACE:-0}" = "1" ]; then
    SITE=$(python -c "import os, verl; print(os.path.dirname(os.path.dirname(os.path.abspath(verl.__file__))))")
    case "$SITE" in
        "$CONDA_PREFIX"/*) ;;
        *) echo "ERROR: verl resolves to $SITE, which is outside this env ($CONDA_PREFIX)." >&2; exit 1 ;;
    esac
    apply_patch_in "$SITE"
else
    if [ ! -d "$VERL_SRC/verl" ]; then
        echo "[setup_env] cloning verl v0.5.0 into $VERL_SRC"
        git clone --depth 1 --branch v0.5.0 https://github.com/volcengine/verl.git "$VERL_SRC"
    fi
    apply_patch_in "$VERL_SRC"
    pip uninstall -y verl >/dev/null
    pip install --no-deps -e "$VERL_SRC"
fi

BASELINE_HASH_AFTER=$(hash_baseline)
if [ "$BASELINE_HASH_BEFORE" != "$BASELINE_HASH_AFTER" ]; then
    echo "ERROR: the verl files of '$BASE_ENV' changed while this script ran" >&2
    echo "       (before $BASELINE_HASH_BEFORE, after $BASELINE_HASH_AFTER). Stop and tell the team." >&2
    exit 1
fi

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python - <<'EOF'
import inspect

import torch
import transformers
import verl
from verl.trainer.ppo import ray_trainer
from verl.workers import fsdp_workers

assert '"sparsespec"' in inspect.getsource(fsdp_workers.ActorRolloutRefWorker._build_rollout), "worker patch missing"
assert 'startswith("sparsespec/")' in inspect.getsource(ray_trainer), "trainer patch missing"
import serve.rl.verl_rollout  # noqa: F401  (rollout class + sharding manager; needs no flashinfer)

print(f"verl {verl.__version__} (patched) | torch {torch.__version__} | transformers {transformers.__version__}")
try:
    import vllm

    print(f"vllm {vllm.__version__}")
except Exception as exc:  # noqa: BLE001
    print(f"vllm: not importable ({type(exc).__name__})")
try:
    import flashinfer  # noqa: F401
    import serve.rl.engine  # noqa: F401

    print("SparseSpec engine importable: yes (engine=sparsespec can run in this env)")
except Exception as exc:  # noqa: BLE001
    print(f"SparseSpec engine importable: NO ({type(exc).__name__}: {exc})")
    print("  -> only engine=reference works in this env until Harrison's build is installed")
EOF
echo "setup_env: OK. Activate with: conda activate $ENV_NAME"
