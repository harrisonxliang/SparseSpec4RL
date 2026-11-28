#!/usr/bin/env bash
# run_all_gates.sh - every GPU gate of the SparseSpec verl backend in one command.
#
#   bash integration/run_all_gates.sh            # from anywhere; takes several hours
#   bash integration/run_all_gates.sh --help
#
# Run it inside tmux (or with nohup): it outlives an SSH session only that way.
# It can be started from any conda env or none: every command is run with
#   conda run --no-capture-output -n <env> env PYTHONPATH=<repo> ...
# (vLLM parts in BASELINE_ENV, everything else in SPARSESPEC_ENV).
#
# What it does, in order (a stage whose dependency failed is SKIPped with the
# reason; every other stage still runs; nothing stops the loop):
#   env                       versions of both envs, pip freeze, verl patch, nvidia-smi, RAM, disk, paths
#   cpu_tests                 CPU pytest suite (SPARSESPEC_ENV), without test_engine_gpu.py
#   engine_import             import serve.rl.engine
#   reference_plumbing        bash integration/smoke_reference.sh (HF reference engine, reduced size)
#   gpu_unit_tests            serve/rl/tests/test_engine_gpu.py
#   gpu_unit_tests_host_guard          ... with SPARSESPEC_TEST_ADMISSION=host_guard
#   gpu_unit_tests_offload             ... with SPARSESPEC_TEST_OFFLOAD_STRESS=1 (worst_case admission)
#   gpu_unit_tests_offload_host_guard  ... with SPARSESPEC_TEST_OFFLOAD_STRESS=1, host_guard
#   gate_3b_vllm              stock vLLM greedy reference (BASELINE_ENV)
#   gate_3b                   greedy gate, three-way with the vLLM JSON (two-way if gate_3b_vllm failed)
#   gate_3c_synthetic         reload gate on a synthetic checkpoint
#   gate_3c_step25            reload gate on $CKPT_ROOT/vllm_baseline_seed0/hf/global_step_25 (SKIP if missing)
#   gate_3c_vllm_reference    stock-vLLM reference numbers (BASELINE_ENV), step-25 ckpt or the synthetic one
#   smoke_worst_case          gate 3d: SMOKE_STEPS full-size training steps, default config   (+ _check)
#   smoke_host_guard          SPARSESPEC_ADMISSION=host_guard                                 (+ _check)
#   smoke_no_graph            SPARSESPEC_CUDA_GRAPH=False                                     (+ _check)
#   smoke_no_logprobs         SPARSESPEC_CALC_LOGPROBS=False, SMOKE_LOGPROB_STEPS steps       (+ _check)
#   smoke_compare             integration.gates.smoke_compare over the four timing CSVs
# then SUMMARY.txt (one screen) and a tarball of everything:
#   <BUNDLE_ROOT>/gates_<host>_<UTC time>/  and  <BUNDLE_ROOT>/gates_<host>_<UTC time>.tar.gz
#
# Shared-GPU rule: before starting, and again before every GPU stage, it reads
# nvidia-smi. It refuses to start (exit 2; the env record is still bundled) if
# less than the required memory is free, and before a later GPU stage it waits
# up to GPU_WAIT_S for the memory to be free, then SKIPs that stage. It NEVER
# kills, signals or renices a process it did not start itself. (Its own stage
# process is stopped only when that stage's timeout expires or when the runner
# itself is interrupted.)
#
# Environment (all optional):
#   SPARSESPEC_ENV        conda env with Harrison's build + patched verl   (verl-sparsespec)
#   BASELINE_ENV          stock vLLM 0.9.1 env, never modified             (verl-baseline)
#   STAGES                comma list: run only these stages (selecting smoke_X also selects smoke_X_check)
#   SKIP_STAGES           comma list: do not run these stages
#   MIN_FREE_GPU_GIB      free GPU memory required before every GPU stage (default: total - GPU_ALLOWANCE_GIB)
#   GPU_ALLOWANCE_GIB     what may be in use when MIN_FREE_GPU_GIB is not set (1.5)
#   GPU_INDEX             nvidia-smi index of the GPU (first entry of CUDA_VISIBLE_DEVICES, else 0)
#   GPU_WAIT_S            how long to wait for free GPU memory before a later stage (900)
#   GPU_POLL_S            re-check interval while waiting (30)
#   BUNDLE_ROOT           where the bundle goes (<repo>/baseline/results/gate_bundles)
#   CKPT_ROOT DATA_DIR MODEL_PATH   defaults from baseline/config/grpo_base.sh
#   SMOKE_STEPS           steps of each smoke run (3)
#   SMOKE_LOGPROB_STEPS   steps of the calculate_log_probs=False run (1)
#   TIMEOUT_<STAGE>       per-stage timeout in seconds, stage name upper-cased, e.g.
#                         TIMEOUT_SMOKE_WORST_CASE=14400; 0 = no timeout (defaults: --help)
#   TIMEOUT_BIN           the timeout command (auto: timeout, then gtimeout); "none" disables timeouts
#   KILL_AFTER_S          after a timeout, SIGKILL our stage this many seconds after SIGTERM (30)
#   HEARTBEAT_S           print a "still running" line this often (300)
#   SUMMARY_PYTHON        python >= 3.6 for the summary (python3 on PATH)
#   HOST_MEMINFO          meminfo file to read (/proc/meminfo; for tests)
#
# Exit status: 0 every required gate passed (3c step-25 may be absent); 1 some
# gate failed or was not run; 2 refused to start (GPU busy, no conda, bad
# option); 128+N interrupted by signal N. SUMMARY.txt and the tarball are
# written in every case except a bad option.

RAG_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

RAG_STAGES="env cpu_tests engine_import reference_plumbing gpu_unit_tests gpu_unit_tests_host_guard
gpu_unit_tests_offload gpu_unit_tests_offload_host_guard gate_3b_vllm gate_3b gate_3c_synthetic gate_3c_step25
gate_3c_vllm_reference smoke_worst_case smoke_worst_case_check smoke_host_guard smoke_host_guard_check
smoke_no_graph smoke_no_graph_check smoke_no_logprobs smoke_no_logprobs_check smoke_compare"
RAG_STAGES=$(echo $RAG_STAGES)   # one line, single spaces
RAG_SMOKE_PREFIX="gates_smoke_"
RAG_SMOKE_LABELS="worst_case host_guard no_graph no_logprobs"
# variables the runner sets per stage; values inherited from the caller's shell are dropped
RAG_PER_STAGE_VARS="SPARSESPEC_TEST_ADMISSION SPARSESPEC_TEST_OFFLOAD_STRESS SPARSESPEC_ADMISSION
SPARSESPEC_CUDA_GRAPH SPARSESPEC_CALC_LOGPROBS SPARSESPEC_ENGINE STEPS EXP_NAME SEED DRY_RUN ALLOW_FULL_SIZE_REFERENCE"

rag_default_timeout() {  # seconds; rough upper bounds, not measurements
    case $1 in
        env) echo 600 ;;                          # per command inside the env stage
        cpu_tests) echo 1800 ;;
        engine_import) echo 600 ;;
        reference_plumbing) echo 3600 ;;
        gpu_unit_tests) echo 5400 ;;              # first run compiles the triton / flashinfer kernels
        gpu_unit_tests_*) echo 3600 ;;
        gate_3b_vllm) echo 2700 ;;
        gate_3b) echo 3600 ;;
        gate_3c_synthetic | gate_3c_step25) echo 5400 ;;
        gate_3c_vllm_reference) echo 3600 ;;
        smoke_no_logprobs) echo 7200 ;;
        smoke_*_check | smoke_compare) echo 600 ;;
        smoke_*) echo 14400 ;;
        *) echo 3600 ;;
    esac
}

rag_estimate() {  # rough expected duration for the --help table: UNMEASURED estimates
    case $1 in
        env) echo "2-5 min" ;;
        cpu_tests) echo "2-5 min" ;;
        engine_import) echo "<1 min" ;;
        reference_plumbing) echo "10-30 min" ;;
        gpu_unit_tests) echo "15-40 min (kernel compile)" ;;
        gpu_unit_tests_*) echo "10-25 min" ;;
        gate_3b_vllm) echo "5-15 min" ;;
        gate_3b) echo "10-25 min" ;;
        gate_3c_synthetic | gate_3c_step25) echo "25-60 min" ;;
        gate_3c_vllm_reference) echo "15-40 min" ;;
        smoke_no_logprobs) echo "20-60 min" ;;
        smoke_*_check | smoke_compare) echo "<1 min" ;;
        smoke_*) echo "45 min-2.5 h" ;;
        *) echo "?" ;;
    esac
}

rag_stage_envkind() {  # host | sparsespec | baseline
    case $1 in
        env) echo host ;;
        gate_3b_vllm | gate_3c_vllm_reference) echo baseline ;;
        *) echo sparsespec ;;
    esac
}

rag_stage_is_gpu() {
    case $1 in
        reference_plumbing | gpu_unit_tests | gpu_unit_tests_host_guard | gpu_unit_tests_offload | \
            gpu_unit_tests_offload_host_guard | gate_3b_vllm | gate_3b | gate_3c_synthetic | gate_3c_step25 | \
            gate_3c_vllm_reference | smoke_worst_case | smoke_host_guard | smoke_no_graph | smoke_no_logprobs)
            return 0 ;;
    esac
    return 1
}

rag_stage_deps() {  # hard dependencies (must PASS, unless not selected)
    case $1 in
        gpu_unit_tests | gpu_unit_tests_host_guard | gpu_unit_tests_offload | gpu_unit_tests_offload_host_guard | \
            gate_3b | gate_3c_synthetic | gate_3c_step25 | \
            smoke_worst_case | smoke_host_guard | smoke_no_graph | smoke_no_logprobs)
            echo engine_import ;;
        *) echo "" ;;
    esac
}

rag_upper() { printf '%s' "$1" | tr '[:lower:]' '[:upper:]'; }
rag_is_uint() { case $1 in '' | *[!0-9]*) return 1 ;; esac; return 0; }
rag_is_num() { printf '%s' "$1" | grep -Eq '^[0-9]+([.][0-9]+)?$'; }
rag_now() { date -u +%s; }
rag_oneline() { printf '%s' "$*" | tr '\n\r\t' '   ' | cut -c1-600; }
rag_fmt_dur() {
    local s=$1
    if [ "$s" -lt 60 ]; then printf '%ds' "$s"
    elif [ "$s" -lt 3600 ]; then printf '%dm%02ds' $((s / 60)) $((s % 60))
    else printf '%dh%02dm' $((s / 3600)) $(((s % 3600) / 60)); fi
}
rag_mib_to_gib() { awk -v m="$1" 'BEGIN { printf "%.1f", m / 1024 }'; }
rag_in_list() {  # rag_in_list <word> <comma or space separated list>
    local w
    for w in $(printf '%s' "$2" | tr ',' ' '); do [ "$w" = "$1" ] && return 0; done
    return 1
}

rag_stage_timeout() {
    local var value
    var="TIMEOUT_$(rag_upper "$1")"
    eval "value=\${$var:-}"
    if [ -z "$value" ]; then rag_default_timeout "$1"; else echo "$value"; fi
}

usage() {
    sed -n '2,/^# written in every case/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    echo
    echo "Stages, default timeouts and rough expected durations (UNMEASURED estimates):"
    local s
    for s in $RAG_STAGES; do
        printf '  %-36s %-10s %-6s timeout %6ss  ~%s\n' "$s" "$(rag_stage_envkind "$s")" \
            "$(rag_stage_is_gpu "$s" && echo GPU || echo -)" "$(rag_default_timeout "$s")" "$(rag_estimate "$s")"
    done
}

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
rag_configure() {
    REPO_ROOT=$RAG_REPO_ROOT
    export REPO_ROOT
    SPARSESPEC_ENV=${SPARSESPEC_ENV:-verl-sparsespec}
    BASELINE_ENV=${BASELINE_ENV:-verl-baseline}
    SMOKE_STEPS=${SMOKE_STEPS:-3}
    SMOKE_LOGPROB_STEPS=${SMOKE_LOGPROB_STEPS:-1}
    GPU_ALLOWANCE_GIB=${GPU_ALLOWANCE_GIB:-1.5}
    GPU_WAIT_S=${GPU_WAIT_S:-900}
    GPU_POLL_S=${GPU_POLL_S:-30}
    HEARTBEAT_S=${HEARTBEAT_S:-300}
    KILL_AFTER_S=${KILL_AFTER_S:-30}
    RAG_POLL_S=${RAG_POLL_S:-2}
    HOST_MEMINFO=${HOST_MEMINFO:-/proc/meminfo}
    STAGES=${STAGES:-}
    SKIP_STAGES=${SKIP_STAGES:-}
    if [ -z "${GPU_INDEX:-}" ]; then
        GPU_INDEX=${CUDA_VISIBLE_DEVICES%%,*}
        rag_is_uint "$GPU_INDEX" || GPU_INDEX=0
    fi
    local bad="" v
    for v in SMOKE_STEPS SMOKE_LOGPROB_STEPS GPU_WAIT_S GPU_POLL_S HEARTBEAT_S KILL_AFTER_S GPU_INDEX; do
        eval "rag_is_uint \"\${$v}\"" || bad="$bad $v"
    done
    rag_is_num "$GPU_ALLOWANCE_GIB" || bad="$bad GPU_ALLOWANCE_GIB"
    rag_is_num "$RAG_POLL_S" || bad="$bad RAG_POLL_S"
    if [ -n "${MIN_FREE_GPU_GIB:-}" ] && ! rag_is_num "$MIN_FREE_GPU_GIB"; then bad="$bad MIN_FREE_GPU_GIB"; fi
    [ "$SMOKE_STEPS" = 0 ] && bad="$bad SMOKE_STEPS"
    [ "$SMOKE_LOGPROB_STEPS" = 0 ] && bad="$bad SMOKE_LOGPROB_STEPS"
    local s
    for s in $RAG_STAGES; do
        v=$(rag_stage_timeout "$s")
        rag_is_uint "$v" || bad="$bad TIMEOUT_$(rag_upper "$s")"
    done
    if [ -n "$bad" ]; then
        echo "[run_all_gates] ERROR: not a non-negative number:$bad" >&2
        return 2
    fi
    local unknown=""
    for s in $(printf '%s' "$STAGES,$SKIP_STAGES" | tr ',' ' '); do
        rag_in_list "$s" "$RAG_STAGES summary bundle" || unknown="$unknown $s"
    done
    if [ -n "$unknown" ]; then
        echo "[run_all_gates] ERROR: unknown stage name(s) in STAGES / SKIP_STAGES:$unknown" >&2
        echo "[run_all_gates]   valid: $RAG_STAGES (env, summary and bundle always run)" >&2
        return 2
    fi

    # CKPT_ROOT / DATA_DIR / MODEL_PATH defaults (the file uses REPO_ROOT)
    if [ ! -f "$REPO_ROOT/baseline/config/grpo_base.sh" ]; then
        echo "[run_all_gates] ERROR: $REPO_ROOT/baseline/config/grpo_base.sh not found" >&2
        return 2
    fi
    # shellcheck source=../baseline/config/grpo_base.sh
    . "$REPO_ROOT/baseline/config/grpo_base.sh"
    export CKPT_ROOT DATA_DIR MODEL_PATH
    STEP25_CKPT="$CKPT_ROOT/vllm_baseline_seed0/hf/global_step_25"
    # mirrors integration/gates/make_synthetic_ckpt.py default_synthetic_dir() with its defaults
    SYNTH_CKPT_DEFAULT="$CKPT_ROOT/synthetic/relnoise0.02_seed0"
    BASELINE_SMOKE_CSV="$REPO_ROOT/baseline/results/smoke.timing.csv"

    BUNDLE_ROOT=${BUNDLE_ROOT:-$REPO_ROOT/baseline/results/gate_bundles}
    case $BUNDLE_ROOT in /*) ;; *) BUNDLE_ROOT="$(pwd)/$BUNDLE_ROOT" ;; esac
    BUNDLE_ROOT=${BUNDLE_ROOT%/}

    # tools
    if [ "${TIMEOUT_BIN:-}" = none ]; then
        TIMEOUT_CMD=""
    elif [ -n "${TIMEOUT_BIN:-}" ]; then
        TIMEOUT_CMD=$TIMEOUT_BIN
    else
        TIMEOUT_CMD=$(command -v timeout 2>/dev/null || command -v gtimeout 2>/dev/null || true)
    fi
    CONDA_BIN=""
    local c
    for c in "${CONDA_EXE:-}" "$(command -v conda 2>/dev/null || true)" "$HOME/miniconda3/bin/conda" \
        "$HOME/anaconda3/bin/conda" "$HOME/miniforge3/bin/conda" "$HOME/mambaforge/bin/conda" /opt/conda/bin/conda; do
        if [ -n "$c" ] && [ -x "$c" ]; then CONDA_BIN=$c; break; fi
    done
    RAG_PY=""
    for c in "${SUMMARY_PYTHON:-}" python3 python; do
        [ -n "$c" ] || continue
        if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 6) else 1)' \
            >/dev/null 2>&1; then
            RAG_PY=$(command -v "$c")
            break
        fi
    done

    RAG_HOST=$( (hostname 2>/dev/null || uname -n 2>/dev/null || echo host) | cut -d. -f1 | tr -c 'A-Za-z0-9_\n-' '_')
    RAG_HOST=${RAG_HOST:-host}
    RAG_TS=$(date -u +%Y%m%dT%H%M%SZ)
    BUNDLE_NAME="gates_${RAG_HOST}_${RAG_TS}"
    BUNDLE="$BUNDLE_ROOT/$BUNDLE_NAME"
    TARBALL="$BUNDLE_ROOT/$BUNDLE_NAME.tar.gz"
    GATES="$BUNDLE/gates"
    return 0
}

rag_selected() {
    local s=$1 base
    rag_in_list "$s" "summary bundle env" && return 0
    if rag_in_list "$s" "$SKIP_STAGES"; then return 1; fi
    [ -z "$STAGES" ] && return 0
    rag_in_list "$s" "$STAGES" && return 0
    case $s in
        smoke_*_check)
            base=${s%_check}
            rag_in_list "$base" "$STAGES" && return 0 ;;
    esac
    return 1
}

# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
say() {  # message on stdout (and straight into the main log once the tee may be gone)
    printf '[run_all_gates] %s\n' "$*"
    if [ "${RAG_DIRECT_LOG:-0}" = 1 ] && [ -n "${SS_LOG:-}" ]; then
        printf '[run_all_gates] %s\n' "$*" >>"$SS_LOG" 2>/dev/null || true
    fi
}

rag_fact() {  # rag_fact key value...  (key=value lines in env/facts.txt; later lines win)
    local key=$1
    shift
    printf '%s=%s\n' "$key" "$(rag_oneline "$*")" >>"$BUNDLE/env/facts.txt"
}

rag_on_err() {  # ERR trap: an unexpected failing command inside this script (not a stage failure)
    printf 'line %s: exit %s: %s\n' "$2" "$1" "$3" >>"$BUNDLE/runner_errors.txt" 2>/dev/null || true
}

rag_env_exists() {  # rag_env_exists <name>
    [ -n "$CONDA_BIN" ] || return 1
    rag_in_list "$1" "$RAG_CONDA_ENVS"
}

rag_env_of() {  # conda env name of a stage (empty for host stages)
    case $(rag_stage_envkind "$1") in
        sparsespec) echo "$SPARSESPEC_ENV" ;;
        baseline) echo "$BASELINE_ENV" ;;
        *) echo "" ;;
    esac
}

# rag_run <log> <timeout s, 0 = none> <conda env | ""> [VAR=value ...] command [args...]
# Runs the command (inside the env through conda run, with PYTHONPATH=<repo>),
# appending stdout + stderr to <log>. In the background, so that a signal to
# the runner is handled at once. Sets RAG_RC, RAG_TIMED_OUT (0/1), RAG_SECONDS.
rag_run() {
    local log=$1 tmo=$2 envname=$3
    shift 3
    local -a argv
    if [ -n "$envname" ]; then
        argv=("$CONDA_BIN" run --no-capture-output -n "$envname" env PYTHONPATH="$REPO_ROOT"
            CKPT_ROOT="$CKPT_ROOT" DATA_DIR="$DATA_DIR" MODEL_PATH="$MODEL_PATH" REPO_ROOT="$REPO_ROOT" "$@")
    else
        argv=(env PYTHONPATH="$REPO_ROOT" "$@")
    fi
    if [ -n "$TIMEOUT_CMD" ] && [ "$tmo" -gt 0 ]; then
        argv=("$TIMEOUT_CMD" -k "$KILL_AFTER_S" "$tmo" "${argv[@]}")
    fi
    {
        printf '[run_all_gates] %s $' "$(date -u '+%H:%M:%S')"
        printf ' %q' "${argv[@]}"
        printf '\n'
    } >>"$log"
    local start now next_hb last
    start=$(rag_now)
    next_hb=$((start + HEARTBEAT_S))
    (cd "$REPO_ROOT" && exec "${argv[@]}") >>"$log" 2>&1 </dev/null 3>&- 4>&- &
    RAG_CHILD=$!
    while kill -0 "$RAG_CHILD" 2>/dev/null; do
        sleep "$RAG_POLL_S"
        now=$(rag_now)
        if [ "$HEARTBEAT_S" -gt 0 ] && [ "$now" -ge "$next_hb" ]; then
            last=$(tail -n 1 "$log" 2>/dev/null | cut -c1-140 || true)
            say "  ... ${STAGE_NAME:-?} still running ($(rag_fmt_dur $((now - start)))); last log line: $last"
            next_hb=$((now + HEARTBEAT_S))
        fi
    done
    if wait "$RAG_CHILD"; then RAG_RC=0; else RAG_RC=$?; fi
    RAG_CHILD=""
    RAG_SECONDS=$(($(rag_now) - start))
    RAG_TIMED_OUT=0
    if [ -n "$TIMEOUT_CMD" ] && [ "$tmo" -gt 0 ] && { [ "$RAG_RC" = 124 ] || [ "$RAG_RC" = 137 ]; } &&
        [ "$RAG_SECONDS" -ge $((tmo - 1)) ]; then
        RAG_TIMED_OUT=1
    fi
    printf '[run_all_gates] %s exit status %s after %ss%s\n' "$(date -u '+%H:%M:%S')" "$RAG_RC" "$RAG_SECONDS" \
        "$([ "$RAG_TIMED_OUT" = 1 ] && echo " (TIMED OUT at ${tmo}s)")" >>"$log"
    return 0
}

# Run the stage's main command and set the stage result from its exit status.
rag_exec_stage() {  # rag_exec_stage <conda env> [VAR=value ...] command...
    local envname=$1 tmo
    shift
    tmo=$(rag_stage_timeout "$STAGE_NAME")
    STAGE_CMD="$(printf '%q ' "$@")"
    STAGE_ENV=$envname
    rag_run "$STAGE_LOG" "$tmo" "$envname" "$@"
    STAGE_RC=$RAG_RC
    if [ "$RAG_TIMED_OUT" = 1 ]; then
        STAGE_STATUS=FAIL
        STAGE_REASON="timed out after ${tmo}s (TIMEOUT_$(rag_upper "$STAGE_NAME")=$tmo); the process was stopped"
        rag_gpu_snapshot >>"$STAGE_LOG" 2>&1
    elif [ "$RAG_RC" = 0 ]; then
        STAGE_STATUS=PASS
    elif [ "$RAG_RC" = 1 ]; then
        STAGE_STATUS=FAIL
        STAGE_REASON="exit 1"
    else
        STAGE_STATUS=FAIL
        STAGE_REASON="exit $RAG_RC (crashed or did not start)"
    fi
}

rag_require_output() {  # a PASS without its output file is a FAIL
    local f
    for f in "$@"; do
        if [ "$STAGE_STATUS" = PASS ] && [ ! -s "$f" ]; then
            STAGE_STATUS=FAIL
            STAGE_REASON="exit 0 but ${f#"$BUNDLE"/} was not written"
        fi
    done
}

rag_pytest_all_skipped_check() {  # a pytest run in which nothing ran is not a pass
    local line
    [ "$STAGE_STATUS" = PASS ] || return 0
    # '==== 9 passed, 1 skipped in 12.1s ====' (-v) or '9 passed, 1 skipped in 12.1s' (-q)
    line=$(grep -E '^(=+ )?(no tests ran|[0-9]+ (passed|failed|skipped|errors?|deselected|xfailed|xpassed|warnings?)).* in [0-9.]+s' \
        "$STAGE_LOG" 2>/dev/null | tail -n 1 || true)
    if [ -z "$line" ]; then
        STAGE_STATUS=FAIL
        STAGE_REASON="exit 0 but no pytest summary line in the log"
    elif ! printf '%s' "$line" | grep -q ' passed'; then
        STAGE_STATUS=FAIL
        STAGE_REASON="no test passed ($(printf '%s' "$line" | tr -d '='| sed 's/^ *//')); see the skip reasons in the log"
    fi
}

# --------------------------------------------------------------------------
# GPU
# --------------------------------------------------------------------------
rag_gpu_read() {  # sets GPU_NAME GPU_TOTAL_MIB GPU_USED_MIB GPU_FREE_MIB, or RAG_GPU_ERR and returns 1
    local out line
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        RAG_GPU_ERR="nvidia-smi not found on PATH"
        return 1
    fi
    if ! out=$(nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
        --format=csv,noheader,nounits 2>&1); then
        RAG_GPU_ERR="nvidia-smi failed: $(rag_oneline "$out" | cut -c1-200)"
        return 1
    fi
    line=$(printf '%s\n' "$out" | awk -F', *' -v i="$GPU_INDEX" '$1 == i { print; exit }')
    if [ -z "$line" ]; then
        RAG_GPU_ERR="nvidia-smi lists no GPU with index $GPU_INDEX: $(rag_oneline "$out" | cut -c1-200)"
        return 1
    fi
    GPU_NAME=$(printf '%s' "$line" | awk -F', *' '{ print $2 }')
    GPU_TOTAL_MIB=$(printf '%s' "$line" | awk -F', *' '{ print $3 }')
    GPU_USED_MIB=$(printf '%s' "$line" | awk -F', *' '{ print $4 }')
    GPU_FREE_MIB=$(printf '%s' "$line" | awk -F', *' '{ print $5 }')
    if ! rag_is_num "$GPU_TOTAL_MIB" || ! rag_is_num "$GPU_FREE_MIB"; then
        RAG_GPU_ERR="cannot parse nvidia-smi memory values: $line"
        return 1
    fi
    return 0
}

rag_gpu_enough() { awk -v f="$GPU_FREE_MIB" -v r="$RAG_REQ_MIB" 'BEGIN { exit !(f + 0 >= r + 0) }'; }

rag_gpu_snapshot() {  # nvidia-smi state for a log (read-only)
    echo "[run_all_gates] ---- nvidia-smi at $(date -u '+%Y-%m-%d %H:%M:%S UTC') ----"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi 2>&1 || true
        echo "[run_all_gates] compute processes (pid, name, memory):"
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 || true
    else
        echo "nvidia-smi not found"
    fi
}

# Wait (bounded) until the required GPU memory is free. 0 = free; 1 = not (STAGE_REASON set).
rag_gpu_wait() {
    local start now waited=0 last_note=0
    start=$(rag_now)
    while :; do
        if rag_gpu_read; then
            if rag_gpu_enough; then
                echo "[run_all_gates] GPU $GPU_INDEX: $(rag_mib_to_gib "$GPU_FREE_MIB") GiB free >= required $(rag_mib_to_gib "$RAG_REQ_MIB") GiB" >>"$STAGE_LOG"
                [ "$waited" -gt 0 ] && say "  GPU memory free again after ${waited}s"
                return 0
            fi
            STAGE_REASON="GPU busy: $(rag_mib_to_gib "$GPU_FREE_MIB") GiB free < required $(rag_mib_to_gib "$RAG_REQ_MIB") GiB"
        else
            STAGE_REASON="cannot read GPU memory ($RAG_GPU_ERR)"
        fi
        now=$(rag_now)
        waited=$((now - start))
        if [ "$waited" -ge "$GPU_WAIT_S" ]; then
            STAGE_REASON="$STAGE_REASON after waiting ${waited}s (GPU_WAIT_S=$GPU_WAIT_S); nothing was killed, process table in the log"
            rag_gpu_snapshot >>"$STAGE_LOG" 2>&1
            return 1
        fi
        if [ $((now - last_note)) -ge 60 ]; then
            say "  $STAGE_REASON; waiting up to ${GPU_WAIT_S}s (re-check every ${GPU_POLL_S}s; nothing is killed)"
            last_note=$now
        fi
        sleep "$GPU_POLL_S"
    done
}

# --------------------------------------------------------------------------
# stage bookkeeping
# --------------------------------------------------------------------------
rag_stage_index() {
    local i=0 s
    for s in $RAG_STAGES; do
        i=$((i + 1))
        if [ "$s" = "$1" ]; then printf '%02d' "$i"; return 0; fi
    done
    printf '99'
}

rag_get() { eval "printf '%s' \"\${RAG_${2}_$1:-}\""; }   # rag_get <stage> ST|SK|RS
rag_status_of() { rag_get "$1" ST; }

rag_set() {  # rag_set PASS|FAIL|SKIP <reason> [skip kind]
    STAGE_STATUS=$1
    STAGE_REASON=${2:-}
    STAGE_SKIP_KIND=${3:-}
}

rag_write_status() {  # writes the current stage's status file and remembers its result
    local name=$STAGE_NAME secs
    secs=$(($(rag_now) - STAGE_START))
    [ "$STAGE_STATUS" = SKIP ] && [ -z "$STAGE_CMD" ] && secs=0
    {
        echo "name=$name"
        echo "status=$STAGE_STATUS"
        echo "reason=$(rag_oneline "$STAGE_REASON")"
        echo "exit_status=$STAGE_RC"
        echo "seconds=$secs"
        echo "env=$STAGE_ENV"
        echo "log=logs/$(basename "$STAGE_LOG")"
        echo "command=$(rag_oneline "$STAGE_CMD")"
        echo "note=$(rag_oneline "$STAGE_NOTE")"
        echo "skip_kind=$STAGE_SKIP_KIND"
        echo "started_utc=$(date -u -r "$STAGE_START" '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || date -u -d "@$STAGE_START" '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || echo "$STAGE_START")"
    } >"$BUNDLE/status/$(rag_stage_index "$name")_$name.status"
    eval "RAG_ST_$name=\$STAGE_STATUS; RAG_SK_$name=\$STAGE_SKIP_KIND; RAG_RS_$name=\$STAGE_REASON"
    {
        echo "[run_all_gates] ---- $name: $STAGE_STATUS${STAGE_REASON:+ - $STAGE_REASON} ($(rag_fmt_dur "$secs")) ----"
    } >>"$STAGE_LOG"
    RAG_STAGE_OPEN=0
}

rag_begin_stage() {
    STAGE_NAME=$1
    STAGE_LOG="$BUNDLE/logs/$(rag_stage_index "$1")_$1.log"
    STAGE_STATUS=""
    STAGE_REASON=""
    STAGE_RC=""
    STAGE_NOTE=""
    STAGE_CMD=""
    STAGE_ENV=$(rag_env_of "$1")
    STAGE_SKIP_KIND=""
    STAGE_START=$(rag_now)
    RAG_STAGE_OPEN=1
    {
        echo "[run_all_gates] stage $1 started $(date -u '+%Y-%m-%d %H:%M:%S UTC') on $RAG_HOST"
        echo "[run_all_gates] env: ${STAGE_ENV:-host}; repo: $REPO_ROOT; timeout: $(rag_stage_timeout "$1")s${TIMEOUT_CMD:+ via $TIMEOUT_CMD}"
    } >"$STAGE_LOG"
}

# The stage loop body: selection, prerequisites, dependencies, GPU, then stage_<name>.
rag_stage() {
    local name=$1 envname dep dst dsk reason
    rag_begin_stage "$name"
    ss_step "$name"
    if ! rag_selected "$name"; then
        rag_set SKIP "not selected (STAGES / SKIP_STAGES)" not_selected
        rag_write_status
        say "$name: SKIP (not selected)"
        return 0
    fi
    envname=$(rag_env_of "$name")
    if [ -n "$envname" ] && ! rag_env_exists "$envname"; then
        if [ -z "$CONDA_BIN" ]; then reason="conda not found"; else reason="conda env '$envname' does not exist (conda env list)"; fi
        rag_set SKIP "$reason" missing_env
    fi
    if [ -z "$STAGE_STATUS" ]; then rag_precheck "$name"; fi
    if [ -z "$STAGE_STATUS" ]; then
        for dep in $(rag_stage_deps "$name"); do
            dst=$(rag_status_of "$dep")
            dsk=$(rag_get "$dep" SK)
            if [ "$dsk" = not_selected ]; then
                STAGE_NOTE="${STAGE_NOTE:+$STAGE_NOTE; }dependency $dep not selected, not checked"
            elif [ "$dst" != PASS ]; then
                rag_set SKIP "dependency $dep ${dst:-not run}${dst:+: $(rag_get "$dep" RS)}" dependency
                break
            fi
        done
    fi
    if [ -z "$STAGE_STATUS" ] && rag_stage_is_gpu "$name"; then
        rag_gpu_wait || STAGE_STATUS=SKIP
        [ "$STAGE_STATUS" = SKIP ] && STAGE_SKIP_KIND=gpu_busy
    fi
    if [ -z "$STAGE_STATUS" ]; then
        say "$name: running in ${envname:-host} (log: ${STAGE_LOG#"$BUNDLE"/}, timeout $(rag_stage_timeout "$name")s)"
        "stage_$name"
        [ -n "$STAGE_STATUS" ] || rag_set FAIL "internal: stage_$name set no result (bug in run_all_gates.sh)"
    fi
    rag_write_status
    say "$name: $STAGE_STATUS ($(rag_fmt_dur $(($(rag_now) - STAGE_START))))${STAGE_REASON:+ - $STAGE_REASON}"
    if [ "$STAGE_STATUS" = FAIL ]; then
        say "  last lines of ${STAGE_LOG#"$BUNDLE"/}:"
        tail -n 15 "$STAGE_LOG" 2>/dev/null | sed 's/^/    | /' || true
    fi
    return 0
}

rag_precheck() {  # stage-specific reasons not to run; sets STAGE_STATUS=SKIP
    case $1 in
        cpu_tests | gpu_unit_tests*)
            if [ "$RAG_PYTEST_OK" != 1 ]; then
                rag_set SKIP "pytest is not installed in $SPARSESPEC_ENV ('import pytest' failed, see the env log); this script installs nothing (RUNBOOK section 1b)" missing_pytest
            fi ;;
        gate_3c_step25)
            if [ ! -d "$STEP25_CKPT" ]; then
                rag_set SKIP "step-25 checkpoint missing: expected the HF checkpoint directory $STEP25_CKPT (CKPT_ROOT=$CKPT_ROOT); rerun with STAGES=gate_3c_step25,gate_3c_vllm_reference once it is there" missing_checkpoint
            fi ;;
        gate_3c_vllm_reference)
            rag_pick_vllm_ref_ckpt
            if [ -z "$RAG_REF_CKPT" ]; then
                rag_set SKIP "no checkpoint to measure: $STEP25_CKPT is missing and the synthetic checkpoint ($RAG_SYNTH_CKPT) does not exist (gate_3c_synthetic: $(rag_status_of gate_3c_synthetic))" missing_checkpoint
            fi ;;
        smoke_*_check)
            local run=${1%_check} exp csv
            exp="$RAG_SMOKE_PREFIX${run#smoke_}"
            csv="$REPO_ROOT/baseline/results/$exp.timing.csv"
            if [ "$(rag_get "$run" SK)" = not_selected ]; then
                rag_set SKIP "its smoke run $run was not selected" dependency
            elif [ "$(rag_status_of "$run")" = SKIP ]; then
                rag_set SKIP "smoke run $run was skipped: $(rag_get "$run" RS)" dependency
            elif [ ! -s "$BUNDLE/smoke/$exp.timing.csv" ]; then
                rag_set SKIP "smoke run $run produced no timing CSV (baseline/results/$exp.timing.csv)" dependency
            fi ;;
        smoke_compare)
            local label n=0
            for label in $RAG_SMOKE_LABELS; do
                [ -s "$BUNDLE/smoke/$RAG_SMOKE_PREFIX$label.timing.csv" ] && n=$((n + 1))
            done
            [ "$n" -gt 0 ] || rag_set SKIP "no smoke run produced a timing CSV" dependency ;;
    esac
}

# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------
rag_write_probe() {
    cat >"$BUNDLE/env/probe_env.py" <<'PROBE'
# written by integration/run_all_gates.sh: versions of one conda env (read-only)
import importlib
import os
import re
import sys

print("VERSION python %s %s" % (sys.version.split()[0], sys.executable))
for name in ("torch", "transformers", "verl", "vllm", "flashinfer", "numpy", "triton", "ray"):
    try:
        module = importlib.import_module(name)
        print("VERSION %s %s %s" % (name, getattr(module, "__version__", "?"), getattr(module, "__file__", "")))
    except BaseException as exc:  # noqa: BLE001 - report anything, including SystemExit from a broken package
        message = str(exc).splitlines()[0][:200] if str(exc) else ""
        print("VERSION %s NOT importable: %s: %s" % (name, type(exc).__name__, message))
try:
    import torch

    print("VERSION torch.version.cuda %s" % torch.version.cuda)
    print("VERSION torch.cuda.is_available %s" % torch.cuda.is_available())
except BaseException as exc:  # noqa: BLE001
    print("VERSION torch.version.cuda unknown: %s" % type(exc).__name__)
try:
    import inspect

    from verl.trainer.ppo import ray_trainer
    from verl.workers import fsdp_workers

    worker = '"sparsespec"' in inspect.getsource(fsdp_workers.ActorRolloutRefWorker._build_rollout)
    trainer = 'startswith("sparsespec/")' in inspect.getsource(ray_trainer)
    state = "applied" if worker and trainer else ("partial" if worker or trainer else "absent")
    print("PATCH %s fsdp_workers_hunk=%s ray_trainer_hunk=%s" % (state, worker, trainer))
except BaseException as exc:  # noqa: BLE001
    print("PATCH unknown %s: %s" % (type(exc).__name__, str(exc)[:200]))
secret = re.compile(r"TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL", re.I)
keep = re.compile(r"^(PATH|LD_LIBRARY_PATH|LD_PRELOAD|PYTHONPATH|PYTHONHOME|CUDA_|NCCL_|CONDA_|VLLM_|HF_|TRANSFORMERS_|"
                  r"TORCH|TRITON|FLASHINFER|RAY_|SPARSESPEC|OMP_|MKL_|CKPT_ROOT|DATA_DIR|MODEL_PATH|NVIDIA_)")
for key in sorted(os.environ):
    if keep.match(key) and not secret.search(key):
        print("ENVVAR %s=%s" % (key, os.environ[key]))
PROBE
    # one-line checks as files: old conda versions mis-quote 'conda run python -c "..."' arguments
    printf '%s\n' "import pytest" "print('pytest', pytest.__version__)" >"$BUNDLE/env/check_pytest.py"
    printf '%s\n' "import serve.rl.engine as m" "print('serve.rl.engine importable:', m.__file__)" \
        >"$BUNDLE/env/check_engine_import.py"
}

rag_host_mem() {  # sets MEM_TOTAL_KIB MEM_AVAIL_KIB (empty if unknown)
    MEM_TOTAL_KIB=""
    MEM_AVAIL_KIB=""
    if [ -r "$HOST_MEMINFO" ]; then
        MEM_TOTAL_KIB=$(awk '$1 == "MemTotal:" { print $2 }' "$HOST_MEMINFO")
        MEM_AVAIL_KIB=$(awk '$1 == "MemAvailable:" { print $2 }' "$HOST_MEMINFO")
    elif command -v free >/dev/null 2>&1; then
        MEM_TOTAL_KIB=$(free -k 2>/dev/null | awk '$1 == "Mem:" { print $2 }' || true)
        MEM_AVAIL_KIB=$(free -k 2>/dev/null | awk '$1 == "Mem:" { print $7 }' || true)
    fi
}

rag_df() {  # df of a path or of its nearest existing parent
    local p=$1
    while [ -n "$p" ] && [ ! -e "$p" ]; do p=$(dirname "$p"); done
    echo "\$ df -h $p   (for $1)"
    df -h "$p" 2>&1 || true
}

stage_env() {
    local d="$BUNDLE/env" e ok=1 notes=""
    mkdir -p "$d"
    # host
    {
        echo "date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
        echo "hostname: $(hostname 2>/dev/null || true)"
        echo "uname: $(uname -a 2>/dev/null || true)"
        echo "user: $(id -un 2>/dev/null || true)"
        echo "bash: $BASH_VERSION"
        echo "nproc: $(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo '?')"
        echo "ulimit -l: $(ulimit -l 2>/dev/null || echo '?')"
        echo "conda: ${CONDA_BIN:-NOT FOUND} $([ -n "$CONDA_BIN" ] && "$CONDA_BIN" --version 2>&1)"
        echo "timeout: ${TIMEOUT_CMD:-NONE (stages cannot be stopped when they hang)}"
        echo "python for the summary: ${RAG_PY:-NONE}"
        echo "caller's conda env: ${CONDA_DEFAULT_ENV:-none}"
        echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
        echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
        echo "PATH=$PATH"
        echo
        echo "\$ free -g"
        free -g 2>&1 || echo "(free not available)"
        echo
        echo "\$ grep -E 'MemTotal|MemAvailable|SwapTotal|Mlocked|HugePages_Total' $HOST_MEMINFO"
        grep -E 'MemTotal|MemAvailable|SwapTotal|Mlocked|HugePages_Total' "$HOST_MEMINFO" 2>&1 || true
        echo
        rag_df "$REPO_ROOT"
        rag_df "$CKPT_ROOT"
    } >"$d/host.txt" 2>&1
    rag_host_mem
    rag_fact mem_total_kib "$MEM_TOTAL_KIB"
    rag_fact mem_available_kib "$MEM_AVAIL_KIB"
    rag_fact nproc "$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo '?')"
    rag_fact ulimit_l "$(ulimit -l 2>/dev/null || echo '?')"
    # git (read-only)
    {
        echo "commit: $(git -C "$REPO_ROOT" rev-parse HEAD 2>&1 || true)"
        echo "branch: $(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>&1 || true)"
        echo
        echo "\$ git status --short"
        git -C "$REPO_ROOT" status --short 2>&1 || true
        echo
        echo "\$ git diff --stat"
        git -C "$REPO_ROOT" diff --stat 2>&1 || true
    } >"$d/git.txt" 2>&1
    rag_fact commit "$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo n/a)"
    rag_fact branch "$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
    # GPU (read-only)
    rag_gpu_snapshot >"$d/nvidia_smi.txt" 2>&1
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,persistence_mode,compute_mode \
            --format=csv >"$d/nvidia_smi_query.csv" 2>&1 || true
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >"$d/nvidia_smi_apps.csv" 2>&1 || true
        rag_fact driver_version "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 || true)"
        rag_fact cuda_version "$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: *[0-9.]*' | head -n 1 | sed 's/.*: *//' || true)"
    fi
    command -v nvcc >/dev/null 2>&1 && nvcc --version >"$d/nvcc.txt" 2>&1
    # resolved paths
    {
        echo "REPO_ROOT=$REPO_ROOT"
        echo "CKPT_ROOT=$CKPT_ROOT"
        echo "DATA_DIR=$DATA_DIR"
        echo "MODEL_PATH=$MODEL_PATH"
        echo "data parquet $DATA_DIR/math_l3to5_train.parquet: $([ -f "$DATA_DIR/math_l3to5_train.parquet" ] && echo exists || echo MISSING)"
        echo "step-25 checkpoint $STEP25_CKPT: $([ -d "$STEP25_CKPT" ] && echo exists || echo MISSING)"
        [ -d "$STEP25_CKPT" ] && ls -la "$STEP25_CKPT" 2>&1
        echo "synthetic checkpoint (default) $SYNTH_CKPT_DEFAULT: $([ -d "$SYNTH_CKPT_DEFAULT" ] && echo exists || echo 'not yet')"
        echo "baseline smoke CSV $BASELINE_SMOKE_CSV: $([ -f "$BASELINE_SMOKE_CSV" ] && echo exists || echo MISSING)"
        echo "per-stage variables dropped from the caller's environment: ${RAG_DROPPED_VARS:-none}"
        echo "other SPARSESPEC_* variables passed through:"
        env | grep '^SPARSESPEC_' | sort || true
    } >"$d/paths.txt" 2>&1
    rag_fact data_parquet "$([ -f "$DATA_DIR/math_l3to5_train.parquet" ] && echo exists || echo missing)"
    rag_fact step25_ckpt "$STEP25_CKPT $([ -d "$STEP25_CKPT" ] && echo exists || echo missing)"
    # conda envs
    if [ -z "$CONDA_BIN" ]; then
        ok=0
        notes="conda not found"
    else
        "$CONDA_BIN" env list >"$d/conda_env_list.txt" 2>&1 || true
        rag_write_probe
        for e in "$SPARSESPEC_ENV" "$BASELINE_ENV"; do
            mkdir -p "$d/$e"
            if ! rag_env_exists "$e"; then
                ok=0
                notes="${notes:+$notes; }env $e MISSING"
                echo "conda env $e does not exist" >"$d/$e/versions.txt"
                continue
            fi
            rag_run "$d/$e/versions.txt" "$(rag_stage_timeout env)" "$e" python "$BUNDLE/env/probe_env.py"
            if [ "$RAG_RC" != 0 ]; then ok=0; notes="${notes:+$notes; }probe of $e exit $RAG_RC"; fi
            rag_run "$d/$e/pip_freeze.txt" "$(rag_stage_timeout env)" "$e" python -m pip freeze
            rag_run "$d/$e/conda_list.txt" "$(rag_stage_timeout env)" "" "$CONDA_BIN" list -n "$e"
            {
                echo "== $e =="
                grep -E '^(VERSION|PATCH) ' "$d/$e/versions.txt" || echo "(no version lines; see env/$e/versions.txt)"
            } >>"$STAGE_LOG" 2>&1
            notes="${notes:+$notes; }$e patch $(awk '$1 == "PATCH" { print $2; exit }' "$d/$e/versions.txt")"
        done
        if rag_env_exists "$SPARSESPEC_ENV"; then
            rag_run "$STAGE_LOG" 300 "$SPARSESPEC_ENV" python "$BUNDLE/env/check_pytest.py"
            [ "$RAG_RC" = 0 ] && RAG_PYTEST_OK=1
            [ "$RAG_PYTEST_OK" = 1 ] || notes="${notes:+$notes; }pytest missing in $SPARSESPEC_ENV"
        fi
    fi
    rag_fact pytest_in_sparsespec_env "$RAG_PYTEST_OK"
    cat "$d/host.txt" "$d/paths.txt" >>"$STAGE_LOG" 2>&1
    STAGE_NOTE=$notes
    if [ "$ok" = 1 ]; then rag_set PASS ""; else rag_set FAIL "${notes:-incomplete env record}"; fi
}

stage_cpu_tests() {
    rag_exec_stage "$SPARSESPEC_ENV" python -m pytest serve/rl/tests integration/gates/tests integration/tests \
        -q -rfE -p no:cacheprovider --ignore=serve/rl/tests/test_engine_gpu.py
    rag_pytest_all_skipped_check
}

stage_engine_import() {
    [ -f "$BUNDLE/env/check_engine_import.py" ] || rag_write_probe
    rag_exec_stage "$SPARSESPEC_ENV" python "$BUNDLE/env/check_engine_import.py"
}

stage_reference_plumbing() {
    rag_exec_stage "$SPARSESPEC_ENV" bash integration/smoke_reference.sh
}

rag_gpu_tests() {  # rag_gpu_tests [VAR=value ...]
    rag_exec_stage "$SPARSESPEC_ENV" "$@" python -m pytest serve/rl/tests/test_engine_gpu.py -v -s --log-cli-level=INFO
    rag_pytest_all_skipped_check
}
stage_gpu_unit_tests() { rag_gpu_tests; }
stage_gpu_unit_tests_host_guard() { rag_gpu_tests SPARSESPEC_TEST_ADMISSION=host_guard; }
stage_gpu_unit_tests_offload() { rag_gpu_tests SPARSESPEC_TEST_ADMISSION=worst_case SPARSESPEC_TEST_OFFLOAD_STRESS=1; }
stage_gpu_unit_tests_offload_host_guard() { rag_gpu_tests SPARSESPEC_TEST_ADMISSION=host_guard SPARSESPEC_TEST_OFFLOAD_STRESS=1; }

stage_gate_3b_vllm() {
    rag_exec_stage "$BASELINE_ENV" python integration/gates/gate_b_vllm_greedy.py \
        --out "$GATES/gate_b_vllm_greedy.json" --out-dir "$GATES"
    rag_require_output "$GATES/gate_b_vllm_greedy.json"
}

stage_gate_3b() {
    local vst
    vst=$(rag_status_of gate_3b_vllm)
    if [ "$vst" = PASS ] && [ -s "$GATES/gate_b_vllm_greedy.json" ]; then
        STAGE_NOTE="three-way (SparseSpec / HF / stock vLLM)"
        rag_exec_stage "$SPARSESPEC_ENV" python -m integration.gates.gate_b_greedy --engine sparsespec \
            --vllm-json "$GATES/gate_b_vllm_greedy.json" --tag three_way --out-dir "$GATES"
        rag_require_output "$GATES/gate_b_greedy_three_way.json"
    else
        STAGE_NOTE="THREE-WAY NOT DONE (gate_3b_vllm ${vst:-not run}): two-way only"
        rag_exec_stage "$SPARSESPEC_ENV" python -m integration.gates.gate_b_greedy --engine sparsespec \
            --tag sparsespec --out-dir "$GATES"
        rag_require_output "$GATES/gate_b_greedy_sparsespec.json"
    fi
}

stage_gate_3c_synthetic() {
    rag_exec_stage "$SPARSESPEC_ENV" python -m integration.gates.gate_c_reload --engine sparsespec --synthetic \
        --tag sparsespec_synthetic --out-dir "$GATES"
    rag_require_output "$GATES/gate_c_reload_sparsespec_synthetic.json"
}

stage_gate_3c_step25() {
    rag_exec_stage "$SPARSESPEC_ENV" python -m integration.gates.gate_c_reload --engine sparsespec \
        --ckpt "$STEP25_CKPT" --tag sparsespec_step25 --out-dir "$GATES"
    if [ "$STAGE_STATUS" = FAIL ] && [ -n "$RAG_PY" ] &&
        [ "$("$RAG_PY" "$REPO_ROOT/integration/summarize_bundle.py" --json-get \
            "$GATES/gate_c_reload_sparsespec_step25.json" failure_class 2>/dev/null || true)" = negative_control_underpowered ]; then
        STAGE_REASON="$STAGE_REASON; failure_class negative_control_underpowered: the checkpoint is too close to the base model for the negative control (not an engine failure)"
    fi
    rag_require_output "$GATES/gate_c_reload_sparsespec_step25.json"
}

rag_pick_vllm_ref_ckpt() {  # sets RAG_REF_CKPT RAG_REF_TAG RAG_SYNTH_CKPT
    local from_json=""
    RAG_REF_CKPT=""
    RAG_REF_TAG=""
    RAG_SYNTH_CKPT=$SYNTH_CKPT_DEFAULT
    if [ -n "$RAG_PY" ] && [ -s "$GATES/gate_c_reload_sparsespec_synthetic.json" ]; then
        from_json=$("$RAG_PY" "$REPO_ROOT/integration/summarize_bundle.py" --json-get \
            "$GATES/gate_c_reload_sparsespec_synthetic.json" checkpoint.dir 2>/dev/null || true)
        [ -n "$from_json" ] && RAG_SYNTH_CKPT=$from_json
    fi
    if [ -d "$STEP25_CKPT" ]; then
        RAG_REF_CKPT=$STEP25_CKPT
        RAG_REF_TAG=step25
    elif [ -d "$RAG_SYNTH_CKPT" ]; then
        RAG_REF_CKPT=$RAG_SYNTH_CKPT
        RAG_REF_TAG=synthetic
    fi
}

stage_gate_3c_vllm_reference() {
    STAGE_NOTE="checkpoint: $RAG_REF_CKPT"
    rag_exec_stage "$BASELINE_ENV" python integration/gates/gate_c_vllm_reference.py --ckpt "$RAG_REF_CKPT" \
        --tag "$RAG_REF_TAG" --out-dir "$GATES"
    rag_require_output "$GATES/gate_c_vllm_reference_$RAG_REF_TAG.json"
}

# Remove ONLY this smoke run's own checkpoint directory (verl would resume from it).
# 0 = removed or absent; 1 = refused (message on stderr).
rag_remove_smoke_ckpt() {
    local exp=$1 root=${CKPT_ROOT:-} target
    case $exp in
        "$RAG_SMOKE_PREFIX"?*) ;;
        *) echo "[run_all_gates] REFUSED to delete: experiment name '$exp' does not start with '$RAG_SMOKE_PREFIX'" >&2; return 1 ;;
    esac
    case $exp in
        *[!A-Za-z0-9_]*) echo "[run_all_gates] REFUSED to delete: experiment name '$exp' has characters other than A-Z a-z 0-9 _" >&2; return 1 ;;
    esac
    case $root in
        /?*) ;;
        *) echo "[run_all_gates] REFUSED to delete: CKPT_ROOT '$root' is not an absolute path" >&2; return 1 ;;
    esac
    target="${root%/}/$exp"
    if [ -e "$target" ] || [ -L "$target" ]; then
        echo "[run_all_gates] removing this smoke run's own checkpoint directory (verl would resume from it): $target"
        ls -la "$target" 2>&1 | head -n 20
        rm -rf -- "$target"
        if [ -e "$target" ] || [ -L "$target" ]; then
            echo "[run_all_gates] could not remove $target" >&2
            return 1
        fi
    else
        echo "[run_all_gates] no earlier checkpoint at $target"
    fi
    return 0
}

rag_smoke() {  # rag_smoke <label> <steps> <admission> <cuda_graph> <calc_logprobs>
    local label=$1 steps=$2 exp csv log
    exp="$RAG_SMOKE_PREFIX$label"
    csv="$REPO_ROOT/baseline/results/$exp.timing.csv"
    log="$REPO_ROOT/baseline/logs/$exp.log"
    mkdir -p "$BUNDLE/smoke"
    local -a vars
    vars=(SEED=0 STEPS="$steps" EXP_NAME="$exp" SPARSESPEC_ENGINE=sparsespec SPARSESPEC_ADMISSION="$3"
        SPARSESPEC_CUDA_GRAPH="$4" SPARSESPEC_CALC_LOGPROBS="$5")
    STAGE_NOTE="$steps step(s), admission=$3 cuda_graph=$4 calculate_log_probs=$5, EXP_NAME=$exp"
    rag_run "$BUNDLE/smoke/$exp.overrides.txt" 120 "" DRY_RUN=1 "${vars[@]}" bash baseline/run_seed_sparsespec.sh
    if ! rag_remove_smoke_ckpt "$exp" >>"$STAGE_LOG" 2>&1; then
        rag_set FAIL "refused to remove the old checkpoint directory of $exp (see log); not started"
        return 0
    fi
    # An earlier attempt's log / CSV of this same experiment would look like this run's output
    # (the run script only rewrites the CSV when it parses some step): move them into the bundle.
    local f
    for f in "$csv" "$log"; do
        if [ -e "$f" ]; then
            mkdir -p "$BUNDLE/smoke/previous_attempt"
            echo "[run_all_gates] moving an earlier attempt's $(basename "$f") into the bundle (smoke/previous_attempt/)" >>"$STAGE_LOG"
            mv -f "$f" "$BUNDLE/smoke/previous_attempt/" 2>>"$STAGE_LOG" || true
        fi
    done
    rag_exec_stage "$SPARSESPEC_ENV" "${vars[@]}" bash baseline/run_seed_sparsespec.sh
    if [ -s "$csv" ]; then
        cp "$csv" "$BUNDLE/smoke/" 2>>"$STAGE_LOG" || true
    else
        echo "[run_all_gates] $csv was not written by this run" >>"$STAGE_LOG"
        if [ "$STAGE_STATUS" = PASS ]; then rag_set FAIL "exit 0 but no timing CSV written (baseline/results/$exp.timing.csv)"; fi
    fi
    if [ -f "$log" ]; then cp "$log" "$BUNDLE/smoke/" 2>>"$STAGE_LOG" || true; fi
    rag_gpu_snapshot >>"$STAGE_LOG" 2>&1
}
stage_smoke_worst_case() { rag_smoke worst_case "$SMOKE_STEPS" worst_case True True; }
stage_smoke_host_guard() { rag_smoke host_guard "$SMOKE_STEPS" host_guard True True; }
stage_smoke_no_graph() { rag_smoke no_graph "$SMOKE_STEPS" worst_case False True; }
stage_smoke_no_logprobs() { rag_smoke no_logprobs "$SMOKE_LOGPROB_STEPS" worst_case True False; }

rag_smoke_check() {  # rag_smoke_check <label> <steps>
    local exp="$RAG_SMOKE_PREFIX$1"
    local -a extra
    extra=()
    if [ -f "$BASELINE_SMOKE_CSV" ]; then extra=(--baseline-csv "$BASELINE_SMOKE_CSV"); fi
    rag_exec_stage "$SPARSESPEC_ENV" python -m integration.gates.gate_d_smoke_check --exp-name "$exp" \
        --steps "$2" --engine sparsespec ${extra[@]+"${extra[@]}"} --out-dir "$GATES"
    rag_require_output "$GATES/gate_d_smoke_$exp.json"
}
stage_smoke_worst_case_check() { rag_smoke_check worst_case "$SMOKE_STEPS"; }
stage_smoke_host_guard_check() { rag_smoke_check host_guard "$SMOKE_STEPS"; }
stage_smoke_no_graph_check() { rag_smoke_check no_graph "$SMOKE_STEPS"; }
stage_smoke_no_logprobs_check() { rag_smoke_check no_logprobs "$SMOKE_LOGPROB_STEPS"; }

stage_smoke_compare() {
    local -a args
    local label csv
    args=()
    for label in $RAG_SMOKE_LABELS; do
        csv="$BUNDLE/smoke/$RAG_SMOKE_PREFIX$label.timing.csv"
        [ -s "$csv" ] && args+=(--run "$label=$csv")
    done
    if [ -s "$BUNDLE/smoke/${RAG_SMOKE_PREFIX}worst_case.timing.csv" ] &&
        [ -s "$BUNDLE/smoke/${RAG_SMOKE_PREFIX}no_logprobs.timing.csv" ]; then
        args+=(--logprob-pair worst_case,no_logprobs)
    else
        STAGE_NOTE="no log-prob pair (worst_case or no_logprobs CSV missing)"
    fi
    if [ -f "$BASELINE_SMOKE_CSV" ]; then
        cp "$BASELINE_SMOKE_CSV" "$BUNDLE/smoke/baseline_smoke.timing.csv" 2>>"$STAGE_LOG" || true
        args+=(--baseline-csv "$BUNDLE/smoke/baseline_smoke.timing.csv")
    fi
    rag_host_mem
    [ -n "$MEM_TOTAL_KIB" ] && args+=(--host-mem-total-gb "$(awk -v k="$MEM_TOTAL_KIB" 'BEGIN { printf "%.2f", k / 1048576 }')")
    [ -n "$MEM_AVAIL_KIB" ] && args+=(--host-mem-available-gb "$(awk -v k="$MEM_AVAIL_KIB" 'BEGIN { printf "%.2f", k / 1048576 }')")
    [ -n "${RAG_GPU_TOTAL_MIB:-}" ] && args+=(--gpu-total-gb "$(awk -v m="$RAG_GPU_TOTAL_MIB" 'BEGIN { printf "%.2f", m / 1024 }')")
    rag_exec_stage "$SPARSESPEC_ENV" python -m integration.gates.smoke_compare "${args[@]}" --out-dir "$GATES"
    rag_require_output "$GATES/smoke_compare.json"
}

# --------------------------------------------------------------------------
# end of run: summary and tarball (also from the signal / exit traps)
# --------------------------------------------------------------------------
rag_mark_remaining() {
    local s
    for s in $RAG_STAGES; do
        [ -n "$(rag_status_of "$s")" ] && continue
        rag_begin_stage "$s"
        rag_set SKIP "${RAG_REMAINING_REASON:-not run}" not_run
        rag_write_status
    done
}

rag_collect_repo_outputs() {  # files the stages wrote outside the bundle (default gate out-dir, logs, runs.jsonl)
    local f rel n=0
    [ -f "$BUNDLE/.run_start" ] || return 0
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        case $(basename "$f") in "$RAG_SMOKE_PREFIX"*) continue ;; esac
        rel=${f#"$REPO_ROOT"/}
        mkdir -p "$BUNDLE/repo_outputs/$(dirname "$rel")" && cp -p "$f" "$BUNDLE/repo_outputs/$rel" 2>/dev/null && n=$((n + 1))
    done <<EOF
$(find "$REPO_ROOT/baseline/results" "$REPO_ROOT/baseline/logs" "$REPO_ROOT/integration/logs" \
        -path "$BUNDLE_ROOT" -prune -o -type f -newer "$BUNDLE/.run_start" -size -204800k -print 2>/dev/null || true)
EOF
    [ "$n" -gt 0 ] && say "copied $n file(s) written by the stages outside the bundle into repo_outputs/"
    return 0
}

rag_bash_summary() {  # used only when no python >= 3.6 is available
    local f
    echo "SparseSpec gate run on $RAG_HOST ($BUNDLE_NAME) - minimal summary (no python3 for summarize_bundle.py)"
    for f in "$BUNDLE"/status/*.status; do
        [ -f "$f" ] || continue
        printf '%-34s %-5s %s\n' "$(sed -n 's/^name=//p' "$f")" "$(sed -n 's/^status=//p' "$f")" \
            "$(sed -n 's/^reason=//p' "$f" | cut -c1-68)"
    done
    echo "send this file back: $TARBALL"
}

rag_finalize() {
    [ "${RAG_FINALIZED:-0}" = 1 ] && return 0
    RAG_FINALIZED=1
    trap '' INT TERM HUP
    [ -d "$BUNDLE" ] || return 0
    if [ "${RAG_STAGE_OPEN:-0}" = 1 ] && [ -n "${STAGE_NAME:-}" ] && [ -z "$(rag_status_of "$STAGE_NAME")" ]; then
        [ -n "$STAGE_STATUS" ] || rag_set FAIL "${RAG_OPEN_STAGE_REASON:-runner stopped while this stage was running}"
        rag_write_status
    fi
    rag_mark_remaining
    rag_fact end_epoch "$(rag_now)"
    rag_collect_repo_outputs
    say "summary:"
    local summarized=0
    if [ -n "$RAG_PY" ] && "$RAG_PY" "$REPO_ROOT/integration/summarize_bundle.py" "$BUNDLE" --tarball "$TARBALL" \
        --out "$BUNDLE/SUMMARY.txt" >"$BUNDLE/.summary_stdout" 2>"$BUNDLE/summary_errors.txt"; then
        summarized=1
    elif [ -n "$CONDA_BIN" ] && rag_env_exists "$BASELINE_ENV" &&
        "$CONDA_BIN" run --no-capture-output -n "$BASELINE_ENV" python "$REPO_ROOT/integration/summarize_bundle.py" \
            "$BUNDLE" --tarball "$TARBALL" --out "$BUNDLE/SUMMARY.txt" >"$BUNDLE/.summary_stdout" 2>>"$BUNDLE/summary_errors.txt"; then
        summarized=1
    fi
    if [ "$summarized" != 1 ] || [ ! -s "$BUNDLE/SUMMARY.txt" ]; then
        rag_bash_summary >"$BUNDLE/SUMMARY.txt"
        echo FAIL >"$BUNDLE/verdict.txt"
    fi
    [ -s "$BUNDLE/summary_errors.txt" ] || rm -f "$BUNDLE/summary_errors.txt"
    rm -f "$BUNDLE/.summary_stdout"
    echo
    cat "$BUNDLE/SUMMARY.txt"
    if [ "${RAG_DIRECT_LOG:-0}" = 1 ] && [ -n "${SS_LOG:-}" ]; then cat "$BUNDLE/SUMMARY.txt" >>"$SS_LOG" 2>/dev/null || true; fi
    echo
    # the tarball, next to the bundle directory
    local rc=0 bytes
    COPYFILE_DISABLE=1 tar czf "$TARBALL.partial" -C "$BUNDLE_ROOT" "$BUNDLE_NAME" 2>"$BUNDLE_ROOT/.$BUNDLE_NAME.tar_errors" || rc=$?
    if [ -s "$TARBALL.partial" ] && [ "$rc" -le 1 ]; then
        mv -f "$TARBALL.partial" "$TARBALL"
        [ "$rc" = 1 ] && say "tar warning (a log was still being written): $(head -n 2 "$BUNDLE_ROOT/.$BUNDLE_NAME.tar_errors" | tr '\n' ' ')"
        rm -f "$BUNDLE_ROOT/.$BUNDLE_NAME.tar_errors"
        bytes=$(wc -c <"$TARBALL" | tr -d ' ')
        say "bundle directory: $BUNDLE"
        say "send this file back: $TARBALL"
        say "size: $(awk -v b="$bytes" 'BEGIN { printf "%.1f MiB (%d bytes)", b / 1048576, b }')"
    else
        say "ERROR: could not create $TARBALL (tar exit $rc): $(head -n 3 "$BUNDLE_ROOT/.$BUNDLE_NAME.tar_errors" 2>/dev/null | tr '\n' ' ')"
        say "send the directory instead: $BUNDLE"
    fi
    return 0
}

rag_on_signal() {  # rag_on_signal <INT|TERM|HUP> <number>
    trap '' INT TERM HUP PIPE
    [ "${RAG_FINALIZED:-0}" = 1 ] && return 0
    if [ "${RAG_FDS_SAVED:-0}" = 1 ]; then exec 1>&3 2>&4; fi   # the tee may have died with the signal
    RAG_DIRECT_LOG=1
    say "received SIG$1: stopping"
    local waited=0
    if [ -n "${RAG_CHILD:-}" ] && kill -0 "$RAG_CHILD" 2>/dev/null; then
        say "stopping this runner's own stage process $RAG_CHILD (${STAGE_NAME:-?}) with SIGTERM"
        kill -TERM "$RAG_CHILD" 2>/dev/null || true
        while kill -0 "$RAG_CHILD" 2>/dev/null && [ "$waited" -lt $((KILL_AFTER_S + 15)) ]; do
            sleep 1
            waited=$((waited + 1))
        done
        kill -0 "$RAG_CHILD" 2>/dev/null && say "it is still running after ${waited}s; left alone (pid $RAG_CHILD)"
    fi
    RAG_OPEN_STAGE_REASON="interrupted by SIG$1 while running"
    RAG_REMAINING_REASON="not run: the gate run was interrupted (SIG$1)"
    rag_finalize
    exit $((128 + $2))
}

rag_on_exit() {
    local rc=$?
    if [ "${RAG_FINALIZED:-0}" != 1 ] && [ -n "${BUNDLE:-}" ] && [ -d "$BUNDLE" ]; then
        if [ "${RAG_FDS_SAVED:-0}" = 1 ]; then exec 1>&3 2>&4; fi
        RAG_DIRECT_LOG=1
        say "the runner exited unexpectedly (status $rc) - probably a bug in run_all_gates.sh"
        ss_report_failure
        RAG_OPEN_STAGE_REASON="runner exited unexpectedly (status $rc)"
        RAG_REMAINING_REASON="not run: the runner exited unexpectedly (status $rc)"
        rag_finalize
        [ "$rc" = 0 ] && rc=1
    fi
    exit "$rc"
}

rag_refuse() {  # rag_refuse <message>: exit 2 with the env record bundled
    say "REFUSING TO START: $*"
    say "nothing was run, killed or signalled; the env record is in the bundle"
    rag_fact refused "$*"
    echo "$*" >"$BUNDLE/REFUSED.txt"
    RAG_REMAINING_REASON="not run: refused to start ($*)"
    rag_finalize
    exit 2
}

# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
rag_main() {
    case ${1:-} in
        -h | --help) usage; return 0 ;;
        "") ;;
        *) echo "[run_all_gates] unknown argument: $1 (options are environment variables; see --help)" >&2; return 2 ;;
    esac
    if [ -z "${BASH_VERSION:-}" ]; then echo "run with bash" >&2; return 2; fi
    RAG_DROPPED_VARS=""
    local v
    for v in $RAG_PER_STAGE_VARS; do
        if eval "[ -n \"\${$v+x}\" ]"; then RAG_DROPPED_VARS="$RAG_DROPPED_VARS $v"; unset "$v"; fi
    done
    rag_configure || return 2
    if ! mkdir -p "$BUNDLE/env" "$BUNDLE/logs" "$BUNDLE/status" "$GATES" "$BUNDLE/smoke"; then
        echo "[run_all_gates] ERROR: cannot create the bundle directory $BUNDLE" >&2
        return 2
    fi
    : >"$BUNDLE/.run_start"
    exec 3>&1 4>&2
    RAG_FDS_SAVED=1
    # shellcheck source=lib.sh
    . "$REPO_ROOT/integration/lib.sh"
    ss_init run_all_gates "$BUNDLE"
    # lib.sh's ERR trap reports the first failing command as fatal; this runner must go on
    # past failing stages, so unexpected command failures are only recorded instead.
    trap 'rag_on_err $? $LINENO "$BASH_COMMAND"' ERR
    trap 'rag_on_signal INT 2' INT
    trap 'rag_on_signal TERM 15' TERM
    trap 'rag_on_signal HUP 1' HUP
    trap 'rag_on_exit' EXIT
    cd "$REPO_ROOT" || rag_refuse "cannot cd to $REPO_ROOT"

    RAG_PYTEST_OK=0
    RAG_CONDA_ENVS=""
    rag_fact hostname "$RAG_HOST"
    rag_fact start_utc "$RAG_TS"
    rag_fact start_epoch "$(rag_now)"
    rag_fact sparsespec_env "$SPARSESPEC_ENV"
    rag_fact baseline_env "$BASELINE_ENV"
    rag_fact timeout_bin "${TIMEOUT_CMD:-none}"
    say "bundle: $BUNDLE"
    say "when it finishes, send back: $TARBALL"
    say "envs: SPARSESPEC_ENV=$SPARSESPEC_ENV BASELINE_ENV=$BASELINE_ENV; CKPT_ROOT=$CKPT_ROOT DATA_DIR=$DATA_DIR MODEL_PATH=$MODEL_PATH"
    say "STAGES=${STAGES:-all} SKIP_STAGES=${SKIP_STAGES:-none} SMOKE_STEPS=$SMOKE_STEPS SMOKE_LOGPROB_STEPS=$SMOKE_LOGPROB_STEPS"
    [ -n "$RAG_DROPPED_VARS" ] && ss_warn "ignoring these variables from your shell (the runner sets them per stage):$RAG_DROPPED_VARS"
    if [ -z "$TIMEOUT_CMD" ]; then
        ss_warn "no 'timeout' command (TIMEOUT_BIN=${TIMEOUT_BIN:-auto}): a hanging stage will NOT be stopped; Ctrl-C still writes the summary and tarball"
    fi

    # ---- preflight: conda, envs, GPU (read-only) ----
    local refuse="" missing=""
    if [ -z "$CONDA_BIN" ]; then
        refuse="conda not found (looked at \$CONDA_EXE, PATH, ~/miniconda3, ~/anaconda3, ~/miniforge3, ~/mambaforge, /opt/conda)"
    else
        RAG_CONDA_ENVS=$("$CONDA_BIN" env list 2>/dev/null | awk '!/^#/ && NF { print $1 }' | tr '\n' ' ' || true)
        if ! "$CONDA_BIN" run --help 2>&1 | grep -q -- '--no-capture-output'; then
            ss_warn "'conda run --help' does not mention --no-capture-output (conda older than 4.9?): stage commands may fail to start"
        fi
        rag_env_exists "$SPARSESPEC_ENV" || missing="$missing $SPARSESPEC_ENV (SPARSESPEC_ENV)"
        rag_env_exists "$BASELINE_ENV" || missing="$missing $BASELINE_ENV (BASELINE_ENV)"
        if [ -n "$missing" ]; then
            ss_warn "conda env(s) missing:$missing. Envs found: ${RAG_CONDA_ENVS:-none}. Their stages will be SKIPped."
        fi
        if ! rag_env_exists "$SPARSESPEC_ENV" && ! rag_env_exists "$BASELINE_ENV"; then
            refuse="neither conda env exists:$missing (envs found: ${RAG_CONDA_ENVS:-none})"
        fi
    fi
    RAG_REQ_MIB=""
    local s need_gpu=0
    for s in $RAG_STAGES; do
        if rag_selected "$s" && rag_stage_is_gpu "$s"; then need_gpu=1; fi
    done
    if rag_gpu_read; then
        RAG_GPU_TOTAL_MIB=$GPU_TOTAL_MIB
        if [ -n "${MIN_FREE_GPU_GIB:-}" ]; then
            RAG_REQ_MIB=$(awk -v g="$MIN_FREE_GPU_GIB" 'BEGIN { printf "%d", g * 1024 }')
            RAG_REQ_TEXT="MIN_FREE_GPU_GIB=$MIN_FREE_GPU_GIB"
        else
            RAG_REQ_MIB=$(awk -v t="$GPU_TOTAL_MIB" -v a="$GPU_ALLOWANCE_GIB" 'BEGIN { printf "%d", t - a * 1024 }')
            RAG_REQ_TEXT="the smoke runs need the whole card: total $(rag_mib_to_gib "$GPU_TOTAL_MIB") GiB minus GPU_ALLOWANCE_GIB=$GPU_ALLOWANCE_GIB; set MIN_FREE_GPU_GIB to override"
        fi
        rag_fact gpu_name "$GPU_NAME"
        rag_fact gpu_total_mib "$GPU_TOTAL_MIB"
        rag_fact gpu_used_mib "$GPU_USED_MIB"
        rag_fact gpu_free_mib "$GPU_FREE_MIB"
        rag_fact required_free_mib "$RAG_REQ_MIB"
        say "GPU $GPU_INDEX ($GPU_NAME): $(rag_mib_to_gib "$GPU_FREE_MIB") GiB free of $(rag_mib_to_gib "$GPU_TOTAL_MIB"); required before every GPU stage: $(rag_mib_to_gib "$RAG_REQ_MIB") GiB ($RAG_REQ_TEXT)"
        if [ "$need_gpu" = 1 ] && [ -z "$refuse" ] && ! rag_gpu_enough; then
            refuse="only $(rag_mib_to_gib "$GPU_FREE_MIB") GiB of GPU $GPU_INDEX is free, the gates need $(rag_mib_to_gib "$RAG_REQ_MIB") GiB ($RAG_REQ_TEXT). Processes on the GPU: $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ';' | cut -c1-300 || true)"
        fi
    elif [ "$need_gpu" = 1 ] && [ -z "$refuse" ]; then
        refuse="cannot read the GPU's free memory ($RAG_GPU_ERR); refusing to start GPU jobs on a shared card blind"
    fi

    # ---- the stages ----
    rag_stage env
    [ -n "$refuse" ] && rag_refuse "$refuse"
    for s in $RAG_STAGES; do
        [ "$s" = env ] && continue
        rag_stage "$s"
    done
    RAG_STAGE_OPEN=0
    rag_finalize
    if [ "$(cat "$BUNDLE/verdict.txt" 2>/dev/null)" = PASS ]; then return 0; fi
    return 1
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    rag_main "$@"
    exit $?
fi
