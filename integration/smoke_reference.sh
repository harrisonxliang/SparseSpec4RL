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
#   bash integration/smoke_reference.sh --check      # preflight + override list only; runs and deletes nothing
#   bash integration/smoke_reference.sh --help
#
# Run it in the env made by integration/setup_env.sh (conda activate verl-sparsespec-dev).
#
# Environment variables (all optional):
#   STEPS=3                    training steps
#   EXP_NAME=reference_plumbing   experiment name: log baseline/logs/$EXP_NAME.log,
#                              CSV baseline/results/$EXP_NAME.timing.csv, checkpoint dir
#                              $CKPT_ROOT/$EXP_NAME (removed before the run, see below)
#   PROMPTS=8                  prompts per step (x rollout.n=8 samples each)
#   RESPONSE_LEN=512           data.max_response_length
#   MIN_FREE_GIB=<n>           override the free-GPU-memory requirement (0 skips the check)
#   CHECK_ONLY=1               same as --check
#   MODEL_PATH, DATA_DIR, CKPT_ROOT as in baseline/config/grpo_base.sh
#
# Before the run it removes ONLY $CKPT_ROOT/$EXP_NAME and the previous
# baseline/results/$EXP_NAME.timing.csv: verl's trainer.resume_mode=auto would
# otherwise resume from an earlier attempt's checkpoint, and a stale CSV could
# pass the gate. Names of real runs (*seed<N>*, vllm_baseline*, smoke) are refused.
#
# Exit status: 0 PASS (trainer exit 0 and gate 3d PASS); otherwise the trainer's
# exit status, or 1.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
export REPO_ROOT

usage() { sed -n '2,/^set -euo pipefail/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }

case "${CHECK_ONLY:-0}" in
    1 | true | True | yes) CHECK_ONLY=1 ;;
    "" | 0 | false | False | no) CHECK_ONLY=0 ;;
    *)
        echo "smoke_reference.sh: CHECK_ONLY must be 1 or 0, got '$CHECK_ONLY'" >&2
        exit 2
        ;;
esac
while [ $# -gt 0 ]; do
    case "$1" in
        --check) CHECK_ONLY=1 ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "smoke_reference.sh: unknown argument '$1' (settings are environment variables; see --help)" >&2
            exit 2
            ;;
    esac
    shift
done

STEPS=${STEPS:-3}
EXP_NAME=${EXP_NAME:-reference_plumbing}
PROMPTS=${PROMPTS:-8}                 # prompts per step (x rollout.n=8 samples)
RESPONSE_LEN=${RESPONSE_LEN:-512}
MIN_FREE_GIB=${MIN_FREE_GIB:-}

# shellcheck source=integration/lib.sh
source "$SCRIPT_DIR/lib.sh"
ss_init smoke_reference

LOG_DIR="$REPO_ROOT/baseline/logs"
RESULTS_DIR="$REPO_ROOT/baseline/results"
LOG_FILE="$LOG_DIR/$EXP_NAME.log"
CSV_FILE="$RESULTS_DIR/$EXP_NAME.timing.csv"
GATE_DIR="$RESULTS_DIR/sparsespec_checks"
TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/smoke_reference.XXXXXX")
FINAL_DONE=0
on_exit() {
    local status=$?
    rm -rf "$TMP_DIR" 2>/dev/null || true
    if [ "$FINAL_DONE" != 1 ] && [ "$status" -ne 0 ]; then
        printf '\n[smoke_reference] FAIL (exit status %s) in step %d: %s; the trainer %s. Log: %s\n' \
            "$status" "$SS_STEP_NO" "$SS_STEP_TITLE" "${TRAINER_RAN:-was not started}" "${SS_LOG:-not saved}"
    fi
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

N_FAIL=0
N_WARN=0
ok() { printf '[smoke_reference]   ok    %s\n' "$*"; }
info() { printf '[smoke_reference]   info  %s\n' "$*"; }
warn() {
    N_WARN=$((N_WARN + 1))
    printf '[smoke_reference]   WARN  %s\n' "$1"
    if [ $# -ge 2 ] && [ -n "$2" ]; then printf '[smoke_reference]         fix: %s\n' "$2"; fi
}
fail() {
    N_FAIL=$((N_FAIL + 1))
    printf '[smoke_reference]   FAIL  %s\n' "$1"
    if [ $# -ge 2 ] && [ -n "$2" ]; then printf '[smoke_reference]         fix: %s\n' "$2"; fi
}
indent() { sed 's/^/[smoke_reference]         | /'; }
is_pos_int() { case "$1" in '' | *[!0-9]* | 0*) return 1 ;; *) return 0 ;; esac; }
nearest_existing_dir() {
    local d=$1
    while [ ! -d "$d" ]; do d=$(dirname "$d"); done
    printf '%s\n' "$d"
}

# ================================================================ 1. inputs
ss_step "inputs"
is_pos_int "$STEPS" || ss_die "STEPS must be a positive integer, got '$STEPS'"
is_pos_int "$PROMPTS" || ss_die "PROMPTS must be a positive integer, got '$PROMPTS'"
is_pos_int "$RESPONSE_LEN" || ss_die "RESPONSE_LEN must be a positive integer, got '$RESPONSE_LEN'"
case "$EXP_NAME" in
    "" | *[!A-Za-z0-9._-]* | .*) ss_die "EXP_NAME must be a plain name of letters, digits, '.', '_', '-', got '$EXP_NAME'" ;;
esac
case "$EXP_NAME" in
    *seed[0-9]* | vllm_baseline* | smoke)
        ss_die "EXP_NAME=$EXP_NAME looks like a real run; this script deletes \$CKPT_ROOT/\$EXP_NAME before it starts. Use another name (default reference_plumbing)"
        ;;
esac
if [ -n "$MIN_FREE_GIB" ] && ! { [ "$MIN_FREE_GIB" = 0 ] || is_pos_int "$MIN_FREE_GIB"; }; then
    ss_die "MIN_FREE_GIB must be a whole number of GiB, got '$MIN_FREE_GIB'"
fi
ss_log "STEPS=$STEPS EXP_NAME=$EXP_NAME PROMPTS=$PROMPTS RESPONSE_LEN=$RESPONSE_LEN check_only=$CHECK_ONLY"

# ================================================================ 2. overrides
ss_step "compose the hydra overrides"
# CKPT_ROOT / DATA_DIR / MODEL_PATH defaults, exactly as the run script sees them
# shellcheck source=baseline/config/grpo_base.sh
source "$REPO_ROOT/baseline/config/grpo_base.sh"
# The exact override list of the real run script, engine switched to reference.
if ! DRY_RUN=1 SPARSESPEC_ENGINE=reference STEPS="$STEPS" EXP_NAME="$EXP_NAME" \
    bash "$REPO_ROOT/baseline/run_seed_sparsespec.sh" >"$TMP_DIR/overrides" 2>"$TMP_DIR/overrides.err"; then
    indent <"$TMP_DIR/overrides.err"
    ss_die "DRY_RUN=1 bash baseline/run_seed_sparsespec.sh failed (its stderr is above)"
fi
ARGS=()
while IFS= read -r line || [ -n "$line" ]; do
    if [ -n "$line" ]; then ARGS+=("$line"); fi
done <"$TMP_DIR/overrides"
[ "${#ARGS[@]}" -gt 0 ] || ss_die "DRY_RUN=1 bash baseline/run_seed_sparsespec.sh printed no overrides"

# Reductions (later overrides win). verl's worker normalises ppo_mini_batch_size
# to ppo_mini_batch_size * rollout.n / world_size sequences (fsdp_workers.py), so
# PROMPTS here gives PROMPTS*8 sequences: exactly the step's batch, one mini-batch
# (one optimizer step) per training step, as in the full-size config (64 / 64).
ARGS+=(
    data.train_batch_size="$PROMPTS"
    actor_rollout_ref.actor.ppo_mini_batch_size="$PROMPTS"
    data.max_response_length="$RESPONSE_LEN"
    trainer.save_freq=-1
)

arg_value() {  # last value given for a key (with or without a leading '+'); empty if none
    local key=$1 item value=""
    for item in "${ARGS[@]}"; do
        case "$item" in
            "$key="* | "+$key="*) value=${item#*=} ;;
        esac
    done
    printf '%s\n' "$value"
}
N_SAMPLES=$(arg_value actor_rollout_ref.rollout.n)
GPU_SHARE=$(arg_value actor_rollout_ref.rollout.gpu_memory_utilization)
TRAIN_FILE=$(arg_value data.train_files)
VAL_FILE=$(arg_value data.val_files)
CKPT_DIR=$(arg_value trainer.default_local_dir)
MODEL=$(arg_value actor_rollout_ref.model.path)
MAX_PROMPT=$(arg_value data.max_prompt_length)
ENGINE=$(arg_value actor_rollout_ref.rollout.sparsespec.engine)
[ "$ENGINE" = reference ] || ss_die "the override list sets rollout.sparsespec.engine='$ENGINE', expected 'reference'"
is_pos_int "$N_SAMPLES" || ss_die "cannot read actor_rollout_ref.rollout.n from the override list (got '$N_SAMPLES')"

ss_log "full override list passed to python3 -m verl.trainer.main_ppo (${#ARGS[@]} entries; later entries win):"
i=0
for item in "${ARGS[@]}"; do
    i=$((i + 1))
    printf '[smoke_reference]   %3d  %s\n' "$i" "$item"
done
ss_log "batch: $PROMPTS prompts x n=$N_SAMPLES = $((PROMPTS * N_SAMPLES)) sequences per step; ppo_mini_batch_size=$PROMPTS -> $((PROMPTS * N_SAMPLES)) sequences after verl's normalisation: 1 mini-batch per step"
ss_log "responses up to $RESPONSE_LEN tokens; prompts up to $MAX_PROMPT; no checkpoint is saved (trainer.save_freq=-1)"

# ================================================================ 3. preflight
ss_step "preflight (read-only)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
info "conda env: ${CONDA_DEFAULT_ENV:-none active} (${CONDA_PREFIX:-})"
if ! command -v python3 >/dev/null 2>&1; then
    fail "python3 is not on PATH" "conda activate verl-sparsespec-dev (the env made by integration/setup_env.sh)"
else
    PY3=$(command -v python3)
    info "python3: $PY3"
    if [ -n "${CONDA_PREFIX:-}" ]; then
        case "$PY3" in
            "$CONDA_PREFIX"/*) ;;
            *) warn "python3 ($PY3) is not inside the active env ($CONDA_PREFIX)" "check PATH; the trainer and the Ray workers run this python3" ;;
        esac
    fi
    case "${CONDA_DEFAULT_ENV:-}" in
        verl-baseline) fail "the active env is verl-baseline: its verl is stock (no sparsespec rollout)" "conda activate verl-sparsespec-dev" ;;
    esac
    cat >"$TMP_DIR/preflight.py" <<'EOF'
import importlib.metadata as md
import inspect
import os
import sys
import traceback

failed = 0


def first_line(exc):
    lines = str(exc).strip().splitlines()
    text = f"{type(exc).__name__}: {lines[0]}" if lines else type(exc).__name__
    here = os.path.abspath(__file__)
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if not f.filename.startswith("<") and os.path.abspath(f.filename) != here]
    if frames:
        text += f" (at {frames[-1].filename}:{frames[-1].lineno})"
    return text


def report(good, text):
    global failed
    failed += 0 if good else 1
    print(("ok   " if good else "FAIL ") + text, flush=True)


try:
    import verl

    where = os.path.dirname(os.path.realpath(verl.__file__))
    report(verl.__version__ == "0.5.0", f"verl {verl.__version__} imports from {where} (needs 0.5.0)")
    try:
        from verl.workers import fsdp_workers

        report('"sparsespec"' in inspect.getsource(fsdp_workers.ActorRolloutRefWorker._build_rollout),
               "patch hunk 1: fsdp_workers.py builds rollout name 'sparsespec'")  # fmt: skip
    except Exception as exc:  # noqa: BLE001
        report(False, "verl.workers.fsdp_workers does not import: " + first_line(exc))
    try:
        from verl.trainer.ppo import ray_trainer

        report('startswith("sparsespec/")' in inspect.getsource(ray_trainer),
               "patch hunk 2: ray_trainer.py logs sparsespec/* as step metrics")  # fmt: skip
    except Exception as exc:  # noqa: BLE001
        report(False, "verl.trainer.ppo.ray_trainer does not import: " + first_line(exc))
except Exception as exc:  # noqa: BLE001
    report(False, "verl does not import: " + first_line(exc))
for module in ("serve.rl.verl_rollout", "serve.rl.reference_engine"):
    try:
        __import__(module)
        report(True, f"{module} imports")
    except Exception as exc:  # noqa: BLE001
        report(False, f"{module} does not import: " + first_line(exc))
for name in ("torch", "transformers", "ray"):
    try:
        print(f"info {name} {md.version(name)}")
    except Exception:  # noqa: BLE001
        print(f"info {name} not installed")
try:
    import torch

    print(f"info torch.cuda.is_available() = {torch.cuda.is_available()}")
except Exception as exc:  # noqa: BLE001
    print("info torch does not import: " + first_line(exc))
model = os.environ.get("SS_MODEL", "")
if model and not os.path.isdir(model):
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(model, "config.json")
        cached = isinstance(cached, str)
    except Exception:  # noqa: BLE001
        cached = False
    print(f"info model {model}: " + ("in the local HF cache" if cached else "NOT in the local HF cache: the workers download it (needs network to huggingface.co)"))
sys.exit(1 if failed else 0)
EOF
    if SS_MODEL="$MODEL" python3 -B "$TMP_DIR/preflight.py" >"$TMP_DIR/preflight.out" 2>&1; then
        PY_OK=1
    else
        PY_OK=0
    fi
    FAILS_BEFORE=$N_FAIL
    while IFS= read -r line; do
        case "$line" in
            "ok   "*) ok "${line#ok   }" ;;
            "FAIL "*) fail "${line#FAIL }" ;;
            "info "*) info "${line#info }" ;;
            *) printf '[smoke_reference]         | %s\n' "$line" ;;
        esac
    done <"$TMP_DIR/preflight.out"
    if [ "$PY_OK" != 1 ] && [ "$N_FAIL" = "$FAILS_BEFORE" ]; then
        fail "the import preflight ($PY3) crashed (output above)"
    fi
    if [ "$PY_OK" != 1 ]; then
        printf '[smoke_reference]         fix: %s\n' \
            "activate the env made by integration/setup_env.sh (conda activate verl-sparsespec-dev) and check it with: bash integration/setup_env.sh <that env> --check; for serve.rl.* errors run from a checkout of branch sparsespec-verl-backend"
    fi
fi

# data: verl builds the train AND the validation dataset even with test_freq=-1
for f in "$TRAIN_FILE" "$VAL_FILE"; do
    if [ -f "$f" ]; then
        ok "data: $f"
    else
        fail "data file $f does not exist (data.train_files / data.val_files; verl loads both even though validation is off)" \
            "create both (downloads MATH from the HF hub): conda run -n verl-baseline python $REPO_ROOT/baseline/prepare_data.py --out-dir $(dirname "$f")"
    fi
done
case "$MODEL" in
    /* | ./* | ../*)
        if [ -f "$MODEL/config.json" ]; then ok "model: $MODEL"; else fail "model path $MODEL has no config.json" "set MODEL_PATH to a HF model directory or a hub id"; fi
        ;;
esac

for d in "$LOG_DIR" "$RESULTS_DIR"; do
    parent=$(nearest_existing_dir "$d")
    if [ -w "$parent" ]; then ok "writable: $d"; else fail "cannot write $d (nearest existing: $parent) as $(id -un)" "fix the permissions of the checkout"; fi
done

# GPU memory
if [ "$MIN_FREE_GIB" = 0 ]; then
    warn "MIN_FREE_GIB=0: free-GPU-memory check skipped"
elif ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi not found: no NVIDIA driver on this machine?" "run on the A100 box"
elif ! nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits >"$TMP_DIR/gpus" 2>"$TMP_DIR/gpus.err"; then
    fail "nvidia-smi --query-gpu failed:" "check the driver (nvidia-smi by hand); MIN_FREE_GIB=0 skips this check"
    indent <"$TMP_DIR/gpus.err"
else
    nvidia-smi 2>&1 | indent || true
    WANT=${CUDA_VISIBLE_DEVICES:-}
    WANT=${WANT%%,*}
    GPU_LINE=""
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        idx=$(printf '%s' "$line" | cut -d, -f1 | tr -d ' ')
        uuid=$(printf '%s' "$line" | cut -d, -f2 | tr -d ' ')
        if [ -z "$WANT" ] || [ "$WANT" = "$idx" ] || [ "$WANT" = "$uuid" ]; then
            GPU_LINE=$line
            break
        fi
    done <"$TMP_DIR/gpus"
    if [ -z "$GPU_LINE" ]; then
        fail "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} selects no GPU that nvidia-smi lists" "unset CUDA_VISIBLE_DEVICES or set it to an index nvidia-smi shows"
    else
        GPU_NAME=$(printf '%s' "$GPU_LINE" | cut -d, -f3 | sed 's/^ *//; s/ *$//')
        TOTAL_MIB=$(printf '%s' "$GPU_LINE" | cut -d, -f4 | tr -d ' ')
        FREE_MIB=$(printf '%s' "$GPU_LINE" | cut -d, -f6 | tr -d ' ')
        if ! is_pos_int "$TOTAL_MIB" || ! { [ "$FREE_MIB" = 0 ] || is_pos_int "$FREE_MIB"; }; then
            fail "cannot read total / free memory from nvidia-smi ('$GPU_LINE')" "MIN_FREE_GIB=0 skips this check"
        else
            # reference engine: at most GPU_SHARE x total. serve/rl/reference_engine.py sizes each
            # generation micro-batch so that weights + KV cache (x2 for HF DynamicCache re-allocation)
            # + 2 GiB scratch stay inside rollout.gpu_memory_utilization x total; with 64 sequences of
            # up to 1024 + 512 tokens the share is what binds. FSDP actor: fp32 weights + grads + two
            # Adam moments, 16 B/param x 0.6B = 9 GiB, resident for the whole run. 3 GiB for CUDA
            # contexts, NCCL, the bf16 ref model during its log-prob pass and allocator slack. The
            # update phase (20480-token micro-batches, unchanged by the reduction) fits the same total.
            if [ -n "$MIN_FREE_GIB" ]; then
                NEED_MIB=$((MIN_FREE_GIB * 1024))
                NEED_WHY="MIN_FREE_GIB=$MIN_FREE_GIB"
            else
                SHARE_MIB=$(awk -v t="$TOTAL_MIB" -v f="${GPU_SHARE:-0.45}" 'BEGIN { printf "%d", t * f }')
                NEED_MIB=$((SHARE_MIB + 12 * 1024))
                NEED_WHY="reference engine <= ${GPU_SHARE:-0.45} x $TOTAL_MIB MiB = $SHARE_MIB MiB (rollout.gpu_memory_utilization; reference_engine.py keeps weights + KV + 2 GiB scratch inside it) + 9 GiB FSDP actor (fp32 weights, grads, Adam: 16 B/param x 0.6B) + 3 GiB CUDA contexts / NCCL / ref model pass / slack; an estimate, MIN_FREE_GIB=<n> overrides it"
            fi
            info "GPU ${idx}: $GPU_NAME, $TOTAL_MIB MiB total, $FREE_MIB MiB free"
            info "needs about $NEED_MIB MiB free: $NEED_WHY"
            if [ "$FREE_MIB" -ge "$NEED_MIB" ]; then
                ok "GPU ${idx} has $FREE_MIB MiB free >= $NEED_MIB MiB"
            else
                fail "GPU ${idx} has only $FREE_MIB MiB free, this run needs about $NEED_MIB MiB" \
                    "wait until the card is free (nvidia-smi; someone else's job is on it), or MIN_FREE_GIB=<n> if you know it fits"
            fi
        fi
    fi
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null | indent || true
fi
if command -v free >/dev/null 2>&1; then
    info "host memory:"
    free -g 2>&1 | indent || true
fi
if [ -n "${RAY_ADDRESS:-}" ]; then
    warn "RAY_ADDRESS=$RAY_ADDRESS is set: verl will join that Ray cluster instead of starting its own" "unset RAY_ADDRESS unless that is intended"
fi

# what will be removed
case "$CKPT_DIR" in
    "$CKPT_ROOT/$EXP_NAME") ;;
    *) fail "the override list writes checkpoints to '$CKPT_DIR', not \$CKPT_ROOT/\$EXP_NAME = '$CKPT_ROOT/$EXP_NAME'" "baseline/run_seed_sparsespec.sh changed: re-check this script against it" ;;
esac
if [ -e "$CKPT_DIR" ]; then
    info "checkpoint dir of an earlier attempt: $CKPT_DIR ($(ls "$CKPT_DIR" 2>/dev/null | tr '\n' ' ')): it will be removed (verl would resume from it)"
else
    info "no checkpoint dir $CKPT_DIR (nothing to remove)"
fi
[ -e "$CSV_FILE" ] && info "the previous $CSV_FILE will be removed before the run"

printf '[smoke_reference] preflight: %d failed, %d warning(s)\n' "$N_FAIL" "$N_WARN"
if [ "$N_FAIL" -gt 0 ]; then
    if [ "$CHECK_ONLY" = 1 ]; then
        FINAL_DONE=1
        printf '\n[smoke_reference] CHECK FAILED: %d problem(s) above (each with a fix); nothing was run or removed.%s\n' "$N_FAIL" "${SS_LOG:+ Log: $SS_LOG}"
        exit 1
    fi
    ss_die "$N_FAIL preflight check(s) failed (each listed above with a fix); nothing was run or removed"
fi
if [ "$CHECK_ONLY" = 1 ]; then
    FINAL_DONE=1
    printf '\n[smoke_reference] CHECK OK: ready to run (%d warning(s)); nothing was run or removed. The run would remove %s and write %s and %s.%s\n' \
        "$N_WARN" "$CKPT_DIR" "$LOG_FILE" "$CSV_FILE" "${SS_LOG:+ Log: $SS_LOG}"
    exit 0
fi

# ================================================================ 4. clean slate
ss_step "remove this smoke's own checkpoint dir and stale CSV"
case "$CKPT_DIR" in
    "" | / | "$HOME" | "$CKPT_ROOT" | "$CKPT_ROOT/") ss_die "refusing to remove '$CKPT_DIR'" ;;
esac
[ "$(basename "$CKPT_DIR")" = "$EXP_NAME" ] || ss_die "refusing to remove '$CKPT_DIR': its name is not $EXP_NAME"
if [ -e "$CKPT_DIR" ]; then
    ss_run rm -rf "$CKPT_DIR"
    [ ! -e "$CKPT_DIR" ] || ss_die "$CKPT_DIR still exists after rm -rf"
fi
[ -e "$CSV_FILE" ] && ss_run rm -f "$CSV_FILE"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

# ================================================================ 5. trainer
ss_step "run the trainer (reference engine, $STEPS steps)"
ss_log "trainer log: $LOG_FILE"
printf '[smoke_reference] + python3 -m verl.trainer.main_ppo <the %d overrides above> 2>&1 | tee %s\n' "${#ARGS[@]}" "$LOG_FILE"
TRAINER_RAN="ran"
START_S=$(date +%s)
STATUSES=(0 0)
python3 -m verl.trainer.main_ppo "${ARGS[@]}" 2>&1 | tee "$LOG_FILE" || STATUSES=("${PIPESTATUS[@]}")
TRAINER_STATUS=${STATUSES[0]}
END_S=$(date +%s)
TRAINER_RAN="exited with status $TRAINER_STATUS"
ss_log "trainer exit status: $TRAINER_STATUS after $((END_S - START_S)) s"
[ "${STATUSES[1]:-0}" = 0 ] || ss_warn "tee exited with status ${STATUSES[1]}: $LOG_FILE may be incomplete"
if [ "$TRAINER_STATUS" -ne 0 ]; then
    ss_log "last 40 lines of $LOG_FILE:"
    tail -n 40 "$LOG_FILE" 2>/dev/null | sed 's/^/    | /' || true
    ss_log "first error-looking lines of $LOG_FILE (line numbers):"
    grep -n -m 10 -E 'Traceback|Error|error:|CUDA out of memory|OutOfMemory|Killed' "$LOG_FILE" 2>/dev/null | sed 's/^/    | /' || echo "    | (none)"
fi
STEP_LINES=$(grep -c 'step:[0-9]' "$LOG_FILE" 2>/dev/null || true)
ss_log "lines with 'step:N' metrics in the log: ${STEP_LINES:-0} (expected $STEPS)"

# ================================================================ 6. CSV
ss_step "parse the log into the timing CSV"
if ss_run python3 "$REPO_ROOT/baseline/parse_timing.py" "$LOG_FILE" -o "$CSV_FILE"; then :; else
    ss_warn "parse_timing.py exited with status $?"
fi
if [ -f "$CSV_FILE" ]; then
    ss_log "CSV: $CSV_FILE ($(($(wc -l <"$CSV_FILE") - 1)) step rows)"
else
    ss_log "no CSV was written (no step metric lines in the log)"
fi

# ================================================================ 7. gate 3d
ss_step "gate 3d check (--engine reference)"
GATE_STATUS=none
if [ -f "$CSV_FILE" ]; then
    if ss_run python3 -m integration.gates.gate_d_smoke_check --exp-name "$EXP_NAME" --steps "$STEPS" --engine reference; then
        GATE_STATUS=0
    else
        GATE_STATUS=$?
    fi
    case "$GATE_STATUS" in
        0) GATE_TEXT="gate 3d PASS" ;;
        1) GATE_TEXT="gate 3d FAIL" ;;
        *) GATE_TEXT="gate 3d CRASHED (exit status $GATE_STATUS)" ;;
    esac
else
    GATE_TEXT="gate 3d not run (no CSV)"
fi

FINAL_DONE=1
WHERE="log: $LOG_FILE | CSV: $CSV_FILE$([ -f "$CSV_FILE" ] || echo ' (not written)') | gate: $GATE_DIR/gate_d_smoke_$EXP_NAME.md | full log: ${SS_LOG:-not saved}"
if [ "$TRAINER_STATUS" -eq 0 ] && [ "$GATE_STATUS" = 0 ]; then
    printf '\n[smoke_reference] PASS: trainer exit 0, %s, %d steps | %s\n' "$GATE_TEXT" "$STEPS" "$WHERE"
    exit 0
fi
printf '\n[smoke_reference] FAIL: trainer exit %s, %s | %s\n' "$TRAINER_STATUS" "$GATE_TEXT" "$WHERE"
if [ "$TRAINER_STATUS" -ne 0 ]; then exit "$TRAINER_STATUS"; fi
exit 1
