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
#   gate_3b                   greedy gate, three-way whenever gate_3b_vllm left its JSON in this bundle (its
#                             exit 0 or 1); two-way (engine vs HF) only without that JSON: NOT a 3b pass
#   gate_3c_synthetic         reload gate on a synthetic checkpoint
#   gate_3c_vllm_reference    stock-vLLM reference numbers (BASELINE_ENV), step-25 ckpt or the synthetic one
#   gate_3c_step25            reload gate on $CKPT_ROOT/vllm_baseline_seed0/hf/global_step_25 (SKIP if missing),
#                             with --vllm-reference from gate_3c_vllm_reference when that passed
#   smoke_worst_case          gate 3d: SMOKE_STEPS full-size training steps, default config   (+ _check)
#   smoke_host_guard          SPARSESPEC_ADMISSION=host_guard                                 (+ _check)
#   smoke_no_graph            SPARSESPEC_CUDA_GRAPH=False                                     (+ _check)
#   smoke_no_logprobs         SPARSESPEC_CALC_LOGPROBS=False, SMOKE_LOGPROB_STEPS steps       (+ _check)
#   smoke_compare             integration.gates.smoke_compare over the four timing CSVs
#   (each smoke_* run, like every run of baseline/run_seed_sparsespec.sh, appends one engine=sparsespec
#    entry named gates_smoke_<label> to baseline/results/runs.jsonl; a copy lands in repo_outputs/)
# After each smoke run its own verl checkpoint ($CKPT_ROOT/gates_smoke_<label>, about 7 GB) is
# removed (size in the stage log); a stage note says LOW DISK below 10 GiB free under CKPT_ROOT.
# A verl stage (reference_plumbing, smoke_*) writes its Ray session path(s) into its log and, unless
# it passed, copies that session's raylet / gcs / core-worker / worker-*.err logs (last 2 MiB each,
# this user's files written during the stage) to ray_logs/<stage>/ in the bundle.
# Gate 3b counts as passed only with gates/gate_b_greedy_three_way.json "three_way": true, "pass": true.
# A gate that refused (its JSON says "refused": true), stopped before writing its JSON, or crashed
# (exit 3) is reported as REFUSED / no verdict / crashed, never as a verdict; none of them passes.
# When 3c step-25 fails only on its negative control (underpowered / inconclusive), SUMMARY.txt and
# the stage log give the two commands of the step-50 rerun, to paste as they are.
# At the end: SUMMARY.txt (one screen: rows that all passed are folded when it would not fit) and a
# tarball of everything:
#   <BUNDLE_ROOT>/gates_<host>_<UTC time>/  and  <BUNDLE_ROOT>/gates_<host>_<UTC time>.tar.gz
#
# Fail fast: right after the env stage it REFUSES TO START (exit 2, nothing GPU-related run; the env
# record, REFUSED.txt with each problem and its fix, SUMMARY.txt and the tarball are written) when:
#   - SPARSESPEC_ENV does not exist             fix: bash integration/build_sparsespec_env.sh
#   - pytest is not importable in it (and a pytest stage is selected)
#                                               fix: its pytest step, or conda run -n <env> python -m pip
#                                                    install --no-deps pytest iniconfig pluggy
#   - INSTALL_PREFIX / PREFIX is set and its lib/librmm.so is missing (and a SparseSpec stage is
#     selected)                                 fix: unset INSTALL_PREFIX PREFIX
#     (set and has librmm.so: used, as by the engine, and noted; the runner never unsets them)
#   - no CUDA compiler where the engine's JIT looks ($PYTORCH_NVCC, else <torch's CUDA_HOME or
#     /usr/local/cuda>/bin/nvcc; when a stage that builds the engine is selected)
#                                               fix: paste the 'export CUDA_HOME=...' line the build printed
#   - the verl patch is not applied in it (and a verl stage is selected)
#                                               fix: bash integration/setup_env.sh <env>
#   - $DATA_DIR/math_l3to5_train.parquet or math500_test.parquet is missing (and a verl stage is
#     selected)                                 fix: export DATA_DIR, or baseline/prepare_data.py
#   - conda is missing, or the GPU is busy / unreadable while a GPU stage is selected
# A missing BASELINE_ENV only SKIPs the stock-vLLM stages (then 3b cannot pass), and a failing
# engine_import only SKIPs its dependents. Less than 24 GiB MemAvailable gives a WARNING (the engine
# page-locks 16 GiB of host memory).
#
# Shared-GPU rule: before starting, and again before every GPU stage, it reads
# nvidia-smi. It refuses to start (exit 2; the env record is still bundled) if
# less than the required memory is free, and before a later GPU stage it waits
# up to GPU_WAIT_S for the memory to be free, then SKIPs that stage. It NEVER
# kills, signals or renices a process it did not start itself. The only
# processes it ever signals are those of the process group that its own stage
# child (GNU timeout) leads: after every stage command and when the runner is
# interrupted, it lists what is left of that group, sends it SIGTERM, then
# SIGKILL after KILL_AFTER_S (GNU timeout itself waits only for its direct
# child, conda, so a stage process that handles SIGTERM could outlive it).
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
#                         TIMEOUT_SMOKE_WORST_CASE=14400; 0 = no timeout (the stage still runs under
#                         the timeout command, for its process group) (defaults: --help)
#   TIMEOUT_BIN           the timeout command (auto: timeout, then gtimeout); "none" disables timeouts
#   KILL_AFTER_S          SIGKILL a stage's own leftover processes this many seconds after SIGTERM (30)
#   HEARTBEAT_S           print a "still running" line this often (300)
#   SUMMARY_PYTHON        python >= 3.6 for the summary (python3 on PATH)
#   HOST_MEMINFO          meminfo file to read (/proc/meminfo; for tests)
#
# Exit status: 0 every required gate passed (3c step-25 may be absent); 1 some
# gate failed, was refused, crashed or was not run (a two-way 3b included);
# 2 refused to start (see above; also a bad option); 128+N interrupted by
# signal N. (The gates' own exit statuses, integration/gates/common.py
# run_main: 0 PASS, 1 FAIL verdict or a stop before the JSON, 2 refused /
# usage error, 3 crash.) SUMMARY.txt and the tarball are
# written in every case except a bad option.

RAG_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

RAG_STAGES="env cpu_tests engine_import reference_plumbing gpu_unit_tests gpu_unit_tests_host_guard
gpu_unit_tests_offload gpu_unit_tests_offload_host_guard gate_3b_vllm gate_3b gate_3c_synthetic gate_3c_vllm_reference
gate_3c_step25 smoke_worst_case smoke_worst_case_check smoke_host_guard smoke_host_guard_check
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
        gpu_unit_tests_offload*) echo "15-30 min" ;;
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
    # relative to where the user typed them (still the caller's directory here), like BUNDLE_ROOT
    case $CKPT_ROOT in /* | "") ;; *) CKPT_ROOT="$(pwd)/$CKPT_ROOT" ;; esac
    case $DATA_DIR in /* | "") ;; *) DATA_DIR="$(pwd)/$DATA_DIR" ;; esac
    export CKPT_ROOT DATA_DIR MODEL_PATH
    STEP25_CKPT="$CKPT_ROOT/vllm_baseline_seed0/hf/global_step_25"
    STEP50_CKPT="$CKPT_ROOT/vllm_baseline_seed0/hf/global_step_50"   # the 3c rerun when step 25 is too close
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
    if [ -n "$TIMEOUT_CMD" ]; then  # GNU timeout: duration 0 disables the timer, but the stage still gets its own process group
        argv=("$TIMEOUT_CMD" -k "$KILL_AFTER_S" "$tmo" "${argv[@]}")
    fi
    {
        printf '[run_all_gates] %s $' "$(date -u '+%H:%M:%S')"
        printf ' %q' "${argv[@]}"
        printf '\n'
    } >>"$log"
    local start now next_hb last probes=0
    start=$(rag_now)
    next_hb=$((start + HEARTBEAT_S))
    (cd "$REPO_ROOT" && exec "${argv[@]}") >>"$log" 2>&1 </dev/null 3>&- 4>&- &
    RAG_CHILD=$!
    RAG_STAGE_PGID=""
    RAG_REAP_NOTE=""
    while kill -0 "$RAG_CHILD" 2>/dev/null; do
        # GNU timeout makes itself the leader of a new process group (setpgid(0, 0)) that every
        # process of the stage inherits; remember it while our own child is alive and leads it
        if [ -n "$TIMEOUT_CMD" ] && [ -z "$RAG_STAGE_PGID" ] && [ "$probes" -lt 100 ]; then
            rag_probe_stage_pgid
            probes=$((probes + 1))
        fi
        # not a foreground sleep: bash drops a SIGINT that arrives while it waits for a foreground
        # child that then exits normally (e.g. still between fork and exec); `wait` runs the trap at once
        sleep "$RAG_POLL_S" </dev/null >/dev/null 2>&1 &
        wait $! || true
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
    # whatever the exit status: nothing of this command's own process group may outlive it
    rag_reap_stage_group "$log"
    return 0
}

# ---- this runner's own stage process group (never anything else) ----
# The only processes the runner ever signals are those of the process group that its own stage
# child (GNU timeout, which calls setpgid(0, 0)) created and leads. rag_probe_stage_pgid records
# that group while the child is alive, and only if it is not the runner's own group. GNU timeout
# waits for its direct child only (conda run): when conda dies at the SIGTERM, timeout exits 124
# at once and its -k SIGKILL never fires, so a workload that handles SIGTERM (Ray workers, vLLM's
# EngineCore) can outlive it. rag_reap_stage_group closes that gap. Never by name, user or GPU use.
# bash reaps a background child as soon as it exits (also during a foreground sleep), so a set
# RAG_CHILD can name a pid that is already free again: only a live process whose parent is this
# runner is still our child (a recycled pid has another parent)
rag_own_child_alive() {  # RAG_CHILD is still this runner's live child (not reaped, not a recycled pid)
    [ -n "${RAG_CHILD:-}" ] && [ "$(ps -o ppid= -p "$RAG_CHILD" 2>/dev/null | tr -d ' ' || true)" = "$$" ]
}

rag_probe_stage_pgid() {
    local g
    [ -n "${RAG_OWN_PGID:-}" ] && rag_own_child_alive || return 0
    g=$(ps -o pgid= -p "$RAG_CHILD" 2>/dev/null | tr -d ' ' || true)
    rag_is_uint "$g" || return 0
    # only a group led by our own child (pgid == its pid), never 0 / 1, never the runner's own group
    if [ "$g" -gt 1 ] && [ "$g" = "$RAG_CHILD" ] && [ "$g" != "$RAG_OWN_PGID" ]; then RAG_STAGE_PGID=$g; fi
    return 0
}

rag_pid_list() {  # stdin: rag_group_members lines; stdout: "pid,pid,..." (the first 8, then "+N more")
    awk 'NF { if (n < 8) printf "%s%s", (n ? "," : ""), $1; n++ } END { if (n > 8) printf ",+%d more", n - 8 }'
}

rag_group_members() {  # live (non-zombie) processes of process group $1: pid pgid stat elapsed command
    ps -A -o pid= -o pgid= -o stat= -o etime= -o args= 2>/dev/null |
        awk -v g="$1" '$2 == g && $3 !~ /^Z/' | cut -c1-300 || true
}

# The group's leader (pid == pgid) was our own child. Unless it is still our own live child (the
# interrupt path: rag_own_child_alive, i.e. RAG_CHILD names it and its parent is this runner), it has
# exited and been reaped, and the kernel gives a pid out again only once no process group with that
# id is left. A live process with that pid is then a NEW, unrelated process that got the recycled
# number (and leads a group of its own): not our group any more.
rag_group_recycled() {  # rag_group_recycled <pgid> <rag_group_members output>; only for use in a condition
    { ! rag_own_child_alive || [ "$RAG_CHILD" != "$1" ]; } &&
        printf '%s\n' "$2" | awk -v g="$1" '$1 == g { hit = 1 } END { exit !hit }'
}

# rag_reap_stage_group <log>: make sure no process of this stage's own process group is left:
# log them, SIGTERM the group, wait up to KILL_AFTER_S, SIGKILL the group. Sets RAG_REAP_NOTE
# (empty if nothing was left) and appends to RAG_UNKILLED what survived even SIGKILL.
rag_reap_stage_group() {
    # RAG_STAGE_PGID stays set until the group is dealt with: an interrupt in the middle (the trap
    # runs between two commands and never comes back here) then does it again from the start
    rag_reap_group "$1" "${RAG_STAGE_PGID:-}"
    RAG_STAGE_PGID=""
    return 0
}

rag_reap_group() {  # rag_reap_group <log> <process group id recorded by rag_probe_stage_pgid>
    local log=$1 g=$2 left n pids waited=0
    RAG_REAP_NOTE=""
    rag_is_uint "$g" || return 0
    if [ "$g" -le 1 ] || [ "$g" = "${RAG_OWN_PGID:-}" ]; then return 0; fi
    left=$(rag_group_members "$g")
    [ -n "$left" ] || return 0
    if rag_group_recycled "$g" "$left"; then
        {
            echo "[run_all_gates] $(date -u '+%H:%M:%S') process group id $g now belongs to a process with the same pid, started after this stage's own child exited (a recycled pid): not this runner's group, nothing is signalled:"
            printf '%s\n' "$left" | sed 's/^/    /'
        } >>"$log" 2>&1
        return 0
    fi
    n=$(printf '%s\n' "$left" | grep -c . || true)
    pids=$(printf '%s\n' "$left" | rag_pid_list)
    {
        echo "[run_all_gates] $(date -u '+%H:%M:%S') $n process(es) of this stage's own process group $g still running after its command returned (pid pgid stat elapsed command):"
        printf '%s\n' "$left" | sed 's/^/    /'
        echo "[run_all_gates] SIGTERM to process group $g (led by this runner's own stage child; nothing else is signalled); SIGKILL after ${KILL_AFTER_S}s"
    } >>"$log" 2>&1
    kill -TERM -- "-$g" 2>/dev/null || true
    while [ "$waited" -lt "$KILL_AFTER_S" ] && [ -n "$(rag_group_members "$g")" ]; do
        sleep 1
        waited=$((waited + 1))
    done
    RAG_REAP_NOTE="$n leftover process(es) of its own process group $g (pids $pids) were still running after the command returned"
    left=$(rag_group_members "$g")
    # (checked again before every signal: once our last process is gone, the id can be given out anew)
    if [ -z "$left" ] || rag_group_recycled "$g" "$left"; then
        RAG_REAP_NOTE="$RAG_REAP_NOTE; stopped with SIGTERM"
    else
        {
            echo "[run_all_gates] $(date -u '+%H:%M:%S') still running ${waited}s after SIGTERM (pid pgid stat elapsed command):"
            printf '%s\n' "$left" | sed 's/^/    /'
            echo "[run_all_gates] SIGKILL to process group $g"
        } >>"$log" 2>&1
        kill -KILL -- "-$g" 2>/dev/null || true
        waited=0
        while [ "$waited" -lt "$KILL_AFTER_S" ] && [ -n "$(rag_group_members "$g")" ]; do
            sleep 1
            waited=$((waited + 1))
        done
        left=$(rag_group_members "$g")
        if [ -z "$left" ] || rag_group_recycled "$g" "$left"; then
            RAG_REAP_NOTE="$RAG_REAP_NOTE; killed with SIGKILL after ${KILL_AFTER_S}s"
        else
            pids=$(printf '%s\n' "$left" | rag_pid_list)
            RAG_REAP_NOTE="$RAG_REAP_NOTE; STILL RUNNING after SIGKILL (stuck in the kernel / driver?): pids $pids"
            RAG_UNKILLED="${RAG_UNKILLED:+$RAG_UNKILLED; }${STAGE_NAME:-?} pids $pids"
        fi
    fi
    echo "[run_all_gates] $RAG_REAP_NOTE" >>"$log"
    return 0
}

# After a stopped GPU stage: is the card still not free enough for the next GPU stage? Whoever holds
# the memory is necessarily outside our own group (rag_reap_stage_group emptied it): only reported,
# nothing is signalled; the next GPU stage's normal wait handles it.
rag_gpu_leftover_note() {  # rag_gpu_leftover_note <log> [timed-out|interrupted]; sets RAG_GPU_LEFT_NOTE
    local log=$1 how=${2:-timed-out} apps waited=0
    RAG_GPU_LEFT_NOTE=""
    [ -n "${RAG_REQ_MIB:-}" ] || return 0
    rag_stage_is_gpu "${STAGE_NAME:-}" || return 0   # a CPU stage held no GPU memory
    while :; do
        rag_gpu_read || return 0
        if rag_gpu_enough; then return 0; fi
        [ "$waited" -ge 10 ] && break   # a just-killed process can take a few seconds to release its memory
        sleep 2
        waited=$((waited + 2))
    done
    apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | grep . || true)
    {
        echo "[run_all_gates] GPU $GPU_INDEX: only $(rag_mib_to_gib "$GPU_FREE_MIB") GiB free ${waited}s after stopping the stage (required: $(rag_mib_to_gib "$RAG_REQ_MIB") GiB); compute processes on it, none of them signalled:"
        printf '%s\n' "${apps:-(nvidia-smi lists none)}"
    } >>"$log" 2>&1
    RAG_GPU_LEFT_NOTE="GPU memory still in use after the $how stage; processes are listed in the log; nothing outside this runner's process group was signalled"
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
        STAGE_OUTCOME=timeout
        STAGE_REASON="timed out after ${tmo}s (TIMEOUT_$(rag_upper "$STAGE_NAME")=$tmo); timeout stopped its command"
        if [ -n "$RAG_REAP_NOTE" ]; then STAGE_REASON="$STAGE_REASON; $RAG_REAP_NOTE"; fi
        rag_gpu_leftover_note "$STAGE_LOG"
        if [ -n "$RAG_GPU_LEFT_NOTE" ]; then STAGE_REASON="$STAGE_REASON; $RAG_GPU_LEFT_NOTE"; fi
        rag_gpu_snapshot >>"$STAGE_LOG" 2>&1
        return 0
    fi
    if [ -n "$RAG_REAP_NOTE" ]; then STAGE_NOTE="${STAGE_NOTE:+$STAGE_NOTE; }exited, but $RAG_REAP_NOTE"; fi
    if [ "$RAG_RC" = 0 ]; then
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

rag_json_get() {  # rag_json_get <file> <dotted.key>: one scalar of a JSON file, empty if unknown
    if [ -n "$RAG_PY" ]; then
        "$RAG_PY" "$REPO_ROOT/integration/summarize_bundle.py" --json-get "$1" "$2" 2>/dev/null || true
    fi
}

# How a gate stage ended, from its exit status and its gate JSON (integration/gates/common.py
# run_main: 0 PASS, 1 FAIL verdict or a SystemExit("message") stop, 2 refused or usage error,
# 3 unexpected exception = crash, 128+N signal). Sets STAGE_OUTCOME and, unless PASS, STAGE_REASON.
rag_gate_result() {  # rag_gate_result <gate JSON>
    local json=$1 rel why refused=""
    rel=${json#"$BUNDLE"/}
    [ "$STAGE_OUTCOME" = timeout ] && return 0
    if [ -s "$json" ]; then
        refused=$(rag_json_get "$json" refused)
        if [ -z "$RAG_PY" ] && grep -q '"refused": *true' "$json" 2>/dev/null; then refused=True; fi
    fi
    if [ "$refused" = True ]; then
        why=$(rag_json_get "$json" reason)
        STAGE_STATUS=FAIL
        STAGE_OUTCOME=refused
        STAGE_REASON="REFUSED (exit $STAGE_RC, nothing was compared): ${why:-see $rel}"
        return 0
    fi
    case $STAGE_RC in
        0)
            if [ -s "$json" ]; then
                STAGE_OUTCOME=verdict
            else
                STAGE_STATUS=FAIL
                STAGE_OUTCOME=no_verdict
                STAGE_REASON="exit 0 but $rel was not written"
            fi ;;
        1)
            if [ -s "$json" ]; then
                STAGE_OUTCOME=verdict
            else
                STAGE_OUTCOME=no_verdict
                STAGE_REASON="no verdict (the gate stopped before writing its JSON; see the log)"
            fi ;;
        3)
            STAGE_OUTCOME=crashed
            STAGE_REASON="crashed (exit 3: unexpected exception, traceback in the log; no verdict)" ;;
        *)
            STAGE_OUTCOME=crashed ;;
    esac
    return 0
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
    local start now waited=0 last_note=0 busy=0
    start=$(rag_now)
    while :; do
        if rag_gpu_read; then
            if rag_gpu_enough; then
                echo "[run_all_gates] GPU $GPU_INDEX: $(rag_mib_to_gib "$GPU_FREE_MIB") GiB free >= required $(rag_mib_to_gib "$RAG_REQ_MIB") GiB" >>"$STAGE_LOG"
                if [ "$busy" = 1 ]; then
                    waited=$(($(rag_now) - start))
                    echo "[run_all_gates] waited ${waited}s for GPU memory (last reading before that: $STAGE_REASON)" >>"$STAGE_LOG"
                    say "  GPU memory free again after ${waited}s"
                fi
                STAGE_REASON=""   # the busy reading above is history, not this stage's result
                return 0
            fi
            STAGE_REASON="GPU busy: $(rag_mib_to_gib "$GPU_FREE_MIB") GiB free < required $(rag_mib_to_gib "$RAG_REQ_MIB") GiB"
        else
            STAGE_REASON="cannot read GPU memory ($RAG_GPU_ERR)"
        fi
        busy=1
        now=$(rag_now)
        waited=$((now - start))
        if [ "$waited" -ge "$GPU_WAIT_S" ]; then
            STAGE_REASON="$STAGE_REASON after waiting ${waited}s (GPU_WAIT_S=$GPU_WAIT_S); nothing was killed, process table in the log${RAG_UNKILLED:+; leftovers of this runner that survived SIGKILL: $RAG_UNKILLED}"
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

rag_get() { eval "printf '%s' \"\${RAG_${2}_$1:-}\""; }   # rag_get <stage> ST|SK|RS|RC
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
        echo "outcome=$STAGE_OUTCOME"
        # GNU date first: GNU 'date -r X' reads the modification time of a FILE named X
        echo "started_utc=$(date -u -d "@$STAGE_START" '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || date -u -r "$STAGE_START" '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || echo "$STAGE_START")"
    } >"$BUNDLE/status/$(rag_stage_index "$name")_$name.status"
    eval "RAG_ST_$name=\$STAGE_STATUS; RAG_SK_$name=\$STAGE_SKIP_KIND; RAG_RS_$name=\$STAGE_REASON; RAG_RC_$name=\$STAGE_RC"
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
    STAGE_OUTCOME=""
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
    # the compiler the FlashInfer JIT calls at every engine start (flashinfer/jit/cpp_ext.py):
    # $PYTORCH_NVCC, else <torch's CUDA_HOME or /usr/local/cuda>/bin/nvcc
    import shutil

    from torch.utils.cpp_extension import CUDA_HOME

    nvcc = os.environ.get("PYTORCH_NVCC") or os.path.join(CUDA_HOME or "/usr/local/cuda", "bin", "nvcc")
    print("NVCC %s nvcc=%s torch_CUDA_HOME=%s" % ("found" if shutil.which(nvcc) else "missing", nvcc, CUDA_HOME))
except BaseException as exc:  # noqa: BLE001
    print("NVCC unknown %s: %s" % (type(exc).__name__, str(exc)[:200]))
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
                  r"TORCH|TRITON|FLASHINFER|RAY_|SPARSESPEC|OMP_|MKL_|CKPT_ROOT|DATA_DIR|MODEL_PATH|NVIDIA_|"
                  r"INSTALL_PREFIX$|PREFIX$|MAX_JOBS$|XDG_CACHE_HOME$)")
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

rag_free_gib() {  # integer GiB available on the filesystem of a path or its nearest existing parent; empty if unknown
    local p=$1
    while [ -n "$p" ] && [ ! -e "$p" ]; do p=$(dirname "$p"); done
    df -Pk "$p" 2>/dev/null | awk 'NR == 2 && $4 ~ /^[0-9]+$/ { printf "%d", $4 / 1048576 }' || true
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
        echo "PATH=$PATH"
        echo "build / JIT variables of the caller's shell (passed to every stage unchanged):"
        for e in INSTALL_PREFIX PREFIX MAX_JOBS XDG_CACHE_HOME FLASHINFER_WORKSPACE_BASE FLASHINFER_EXTRA_LDFLAGS \
            CUDA_HOME CUDA_PATH LD_LIBRARY_PATH; do
            eval "echo \"  $e=\${$e-<unset>}\""
        done
        echo "  (INSTALL_PREFIX, else PREFIX, else CONDA_PREFIX gives the RAFT headers and librmm.so: serve/attention/backend.py)"
        [ -n "${RAG_PREFIX_NOTE:-}" ] && echo "  NOTE: $RAG_PREFIX_NOTE"
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
    rag_fact step50_ckpt "$STEP50_CKPT"
    rag_fact step50_ckpt_exists "$([ -d "$STEP50_CKPT" ] && echo yes || echo no)"
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
                grep -E '^(VERSION|PATCH|NVCC) ' "$d/$e/versions.txt" || echo "(no version lines; see env/$e/versions.txt)"
            } >>"$STAGE_LOG" 2>&1
            notes="${notes:+$notes; }$e patch $(awk '$1 == "PATCH" { print $2; exit }' "$d/$e/versions.txt")"
        done
        if rag_env_exists "$SPARSESPEC_ENV"; then
            RAG_SPARSESPEC_PATCH=$(awk '$1 == "PATCH" { print $2; exit }' "$d/$SPARSESPEC_ENV/versions.txt" 2>/dev/null || true)
            RAG_SPARSESPEC_PATCH_LINE=$(awk '$1 == "PATCH" { print; exit }' "$d/$SPARSESPEC_ENV/versions.txt" 2>/dev/null | cut -c1-200 || true)
            rag_fact sparsespec_patch "${RAG_SPARSESPEC_PATCH:-not recorded}"
            RAG_SPARSESPEC_NVCC=$(awk '$1 == "NVCC" { print $2; exit }' "$d/$SPARSESPEC_ENV/versions.txt" 2>/dev/null || true)
            RAG_SPARSESPEC_NVCC_LINE=$(awk '$1 == "NVCC" { print; exit }' "$d/$SPARSESPEC_ENV/versions.txt" 2>/dev/null | cut -c1-300 || true)
            rag_fact sparsespec_nvcc "${RAG_SPARSESPEC_NVCC:-not recorded}"
        fi
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
    # not integration/tests: those run this runner and the setup scripts against stub commands (signals,
    # fake conda); they test the scripts' logic, not this box, and would add 10+ minutes here
    rag_exec_stage "$SPARSESPEC_ENV" python -m pytest serve/rl/tests integration/gates/tests \
        -q -rfE -p no:cacheprovider --ignore=serve/rl/tests/test_engine_gpu.py --color=no
    rag_pytest_all_skipped_check
}

stage_engine_import() {
    [ -f "$BUNDLE/env/check_engine_import.py" ] || rag_write_probe
    rag_exec_stage "$SPARSESPEC_ENV" python "$BUNDLE/env/check_engine_import.py"
}

# Ray's temp root as ray resolves it when ray.init gets no _temp_dir (verl's main_ppo passes none):
# $RAY_TMPDIR/ray, else on Linux $TMPDIR/ray, else /tmp/ray.
rag_ray_root() {
    if [ -n "${RAY_TMPDIR:-}" ]; then
        printf '%s/ray' "${RAY_TMPDIR%/}"
    elif [ "$(uname -s 2>/dev/null)" = Linux ] && [ -n "${TMPDIR:-}" ]; then
        printf '%s/ray' "${TMPDIR%/}"
    else
        printf '/tmp/ray'
    fi
}

# After a verl stage: name the Ray session(s) it used in the stage log and, unless it passed, copy
# their raylet / GCS / core-worker logs and worker stderr (last 2 MiB of each) into the bundle. A
# worker that dies natively can leave its last lines only there, and the next ray.init re-points
# session_latest. Only this user's files modified since <marker> (logs/old/ included: Ray's log
# monitor moves dead workers' files there).
rag_collect_ray_logs() {  # rag_collect_ray_logs <marker file> <dest dir>
    local marker=$1 dst=$2 root f rel sess sessions="" n=0 files
    root=$(rag_ray_root)
    if [ ! -d "$root" ] || [ ! -f "$marker" ]; then
        echo "[run_all_gates] Ray session logs: no Ray temp dir at $root (RAY_TMPDIR / TMPDIR); nothing to copy" >>"$STAGE_LOG"
        return 0
    fi
    files=$(find -H "$root" -maxdepth 4 -type f -path "$root/session_*/logs/*" -user "$(id -u)" -newer "$marker" -size +0 \
        \( -name 'raylet.*' -o -name 'gcs_server.*' -o -name 'python-core-*.log' -o -name 'worker-*.err' \) -print 2>/dev/null || true)
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        rel=${f#"$root"/}
        sess=${rel%%/*}
        case " $sessions " in *" $root/$sess "*) ;; *) sessions="$sessions $root/$sess" ;; esac
        [ "$STAGE_STATUS" = PASS ] && continue
        mkdir -p "$dst/$(dirname "$rel")" 2>/dev/null || continue
        if tail -c 2097152 "$f" >"$dst/$rel" 2>/dev/null; then n=$((n + 1)); fi
    done <<EOF
$files
EOF
    if [ -z "$sessions" ]; then
        echo "[run_all_gates] Ray session logs: no session under $root was written during this stage" >>"$STAGE_LOG"
    elif [ "$STAGE_STATUS" = PASS ]; then
        echo "[run_all_gates] Ray session(s) of this stage (not copied, the stage passed):$sessions" >>"$STAGE_LOG"
    else
        echo "[run_all_gates] Ray session(s) of this stage:$sessions; $n log file(s) (last 2 MiB of each) copied to ${dst#"$BUNDLE"/}/" >>"$STAGE_LOG"
    fi
    return 0
}

stage_reference_plumbing() {
    : >"$BUNDLE/.ray_marker"
    rag_exec_stage "$SPARSESPEC_ENV" bash integration/smoke_reference.sh
    rag_collect_ray_logs "$BUNDLE/.ray_marker" "$BUNDLE/ray_logs/$STAGE_NAME"
}

rag_gpu_tests() {  # rag_gpu_tests [VAR=value ...]
    rag_exec_stage "$SPARSESPEC_ENV" "$@" python -m pytest serve/rl/tests/test_engine_gpu.py -v -s --log-cli-level=INFO --color=no
    rag_pytest_all_skipped_check
}
stage_gpu_unit_tests() { rag_gpu_tests; }
stage_gpu_unit_tests_host_guard() { rag_gpu_tests SPARSESPEC_TEST_ADMISSION=host_guard; }
stage_gpu_unit_tests_offload() { rag_gpu_tests SPARSESPEC_TEST_ADMISSION=worst_case SPARSESPEC_TEST_OFFLOAD_STRESS=1; }
stage_gpu_unit_tests_offload_host_guard() { rag_gpu_tests SPARSESPEC_TEST_ADMISSION=host_guard SPARSESPEC_TEST_OFFLOAD_STRESS=1; }

stage_gate_3b_vllm() {
    rag_exec_stage "$BASELINE_ENV" python integration/gates/gate_b_vllm_greedy.py \
        --out "$GATES/gate_b_vllm_greedy.json" --out-dir "$GATES"
    rag_gate_result "$GATES/gate_b_vllm_greedy.json"
}

# Gate 3b is the THREE-WAY gate (SparseSpec / HF / stock vLLM); only its JSON with "three_way": true
# and "pass": true passes 3b in the summary. The two-way gate (engine vs HF) runs only when there is
# no vLLM token file, as a diagnostic; it never passes 3b.
stage_gate_3b() {
    local vst vrc vjson="$GATES/gate_b_vllm_greedy.json"
    vst=$(rag_status_of gate_3b_vllm)
    vrc=$(rag_get gate_3b_vllm RC)
    # The bundle is new for every run, so this file can only come from this run's gate_3b_vllm, which
    # writes it in one go after all tokens are extracted. Its exit 1 still writes a complete token file
    # ("pass": false when a step was not greedy): the three-way gate reads it and reports that as a
    # reference-engine problem. So the file decides, not the exit status (a SystemExit("message") stop
    # also exits 1, without the file; a hang at vLLM shutdown times out with the file already written).
    if [ -s "$vjson" ]; then
        STAGE_NOTE="three-way (SparseSpec / HF / stock vLLM)"
        [ "$vst" = PASS ] || STAGE_NOTE="three-way (vLLM JSON from gate_3b_vllm ${vst:-not run}, exit ${vrc:-n/a})"
        rag_exec_stage "$SPARSESPEC_ENV" python -m integration.gates.gate_b_greedy --engine sparsespec \
            --vllm-json "$vjson" --tag three_way --out-dir "$GATES"
        rag_gate_result "$GATES/gate_b_greedy_three_way.json"
    else
        STAGE_NOTE="THREE-WAY NOT DONE (gate_3b_vllm ${vst:-not run}): two-way only"
        rag_exec_stage "$SPARSESPEC_ENV" python -m integration.gates.gate_b_greedy --engine sparsespec \
            --tag sparsespec --out-dir "$GATES"
        rag_gate_result "$GATES/gate_b_greedy_sparsespec.json"
    fi
}

stage_gate_3c_synthetic() {
    rag_exec_stage "$SPARSESPEC_ENV" python -m integration.gates.gate_c_reload --engine sparsespec --synthetic \
        --tag sparsespec_synthetic --out-dir "$GATES"
    rag_gate_result "$GATES/gate_c_reload_sparsespec_synthetic.json"
}

stage_gate_3c_step25() {
    local ref="$GATES/gate_c_vllm_reference_step25.json" rst fclass
    rst=$(rag_status_of gate_3c_vllm_reference)
    if [ "$rst" = PASS ] && [ -s "$ref" ]; then
        STAGE_NOTE="with the stock-vLLM reference ${ref#"$BUNDLE"/}"
        rag_exec_stage "$SPARSESPEC_ENV" python -m integration.gates.gate_c_reload --engine sparsespec \
            --ckpt "$STEP25_CKPT" --vllm-reference "$ref" --tag sparsespec_step25 --out-dir "$GATES"
    else
        STAGE_NOTE="NO stock-vLLM reference (gate_3c_vllm_reference ${rst:-not run}): a too-close checkpoint is reported as negative_control_inconclusive"
        rag_exec_stage "$SPARSESPEC_ENV" python -m integration.gates.gate_c_reload --engine sparsespec \
            --ckpt "$STEP25_CKPT" --tag sparsespec_step25 --out-dir "$GATES"
    fi
    rag_gate_result "$GATES/gate_c_reload_sparsespec_step25.json"
    if [ "$STAGE_STATUS" = FAIL ] && [ "$STAGE_OUTCOME" = verdict ]; then
        fclass=$(rag_json_get "$GATES/gate_c_reload_sparsespec_step25.json" failure_class)
        if [ "$fclass" = negative_control_underpowered ]; then
            STAGE_REASON="$STAGE_REASON; failure_class negative_control_underpowered: the checkpoint is too close to the base model for the negative control (consistent with a correct engine: log-prob noise within 2x stock vLLM's)"
        elif [ "$fclass" = negative_control_inconclusive ]; then
            STAGE_REASON="$STAGE_REASON; failure_class negative_control_inconclusive: only the negative control failed, no stock-vLLM reference to tell a too-close checkpoint from a noisy engine"
        fi
        case $fclass in
            negative_control_underpowered | negative_control_inconclusive)
                # the step-50 pair (stock vLLM first, then the gate), outside this bundle; also in SUMMARY.txt
                STAGE_REASON="$STAGE_REASON; rerun on the step-50 checkpoint: commands in SUMMARY.txt and in this stage's log"
                rag_step50_rerun_text >"$BUNDLE/.rerun_step50" 2>/dev/null || true
                {
                    echo "[run_all_gates] only the negative control failed; the step-50 rerun, to paste as it is:"
                    cat "$BUNDLE/.rerun_step50"
                } >>"$STAGE_LOG" 2>&1
                say "  only the negative control failed; the step-50 rerun, to paste as it is:"
                cat "$BUNDLE/.rerun_step50" 2>/dev/null || true
                rm -f "$BUNDLE/.rerun_step50" ;;
        esac
    fi
}

rag_step50_rerun_text() {  # the step-50 rerun pair, exactly as SUMMARY.txt prints it (summarize_bundle.py)
    if [ -n "$RAG_PY" ] && "$RAG_PY" "$REPO_ROOT/integration/summarize_bundle.py" --rerun-step50 "$BUNDLE" 2>/dev/null; then
        return 0
    fi
    # no python for the summary: the same commands, one (long) line each
    local envs="env PYTHONPATH=." note=""
    [ -d "$STEP50_CKPT" ] || note=" The step-50 checkpoint C does not exist yet: the baseline run saves it at step 50."
    echo "Step-50 rerun: paste the lines below as they are (stock vLLM first, then the gate; both write to the gates' default out-dir, not into this bundle).$note"
    printf 'cd %q\n' "$REPO_ROOT"
    printf 'C=%q\n' "$STEP50_CKPT"
    if [ -n "${RAG_RERUN_DATA_DIR:-}" ]; then
        printf 'D=%q\n' "$RAG_RERUN_DATA_DIR"
        envs="$envs DATA_DIR=\"\$D\""
    fi
    if [ -n "${RAG_RERUN_MODEL_PATH:-}" ]; then
        printf 'M=%q\n' "$RAG_RERUN_MODEL_PATH"
        envs="$envs MODEL_PATH=\"\$M\""
    fi
    printf '%q run --no-capture-output -n %q %s python integration/gates/gate_c_vllm_reference.py --ckpt "$C" --tag step50\n' \
        "$RAG_CONDA_CMD" "$BASELINE_ENV" "$envs"
    printf '%q run --no-capture-output -n %q %s python -m integration.gates.gate_c_reload --engine sparsespec --ckpt "$C" --tag sparsespec_step50 --vllm-reference baseline/results/sparsespec_checks/gate_c_vllm_reference_step50.json\n' \
        "$RAG_CONDA_CMD" "$SPARSESPEC_ENV" "$envs"
    return 0
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
    rag_gate_result "$GATES/gate_c_vllm_reference_$RAG_REF_TAG.json"
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
    local free_gib
    rag_df "$CKPT_ROOT" >>"$STAGE_LOG" 2>&1
    free_gib=$(rag_free_gib "$CKPT_ROOT")
    if [ -n "$free_gib" ] && [ "$free_gib" -lt 10 ]; then
        STAGE_NOTE="${STAGE_NOTE:+$STAGE_NOTE; }LOW DISK: ${free_gib} GiB free under CKPT_ROOT; verl's last-step checkpoint needs about 7 GiB"
    fi
    : >"$BUNDLE/.ray_marker"
    rag_exec_stage "$SPARSESPEC_ENV" "${vars[@]}" bash baseline/run_seed_sparsespec.sh
    if [ -s "$csv" ]; then
        cp "$csv" "$BUNDLE/smoke/" 2>>"$STAGE_LOG" || true
    else
        echo "[run_all_gates] $csv was not written by this run" >>"$STAGE_LOG"
        if [ "$STAGE_STATUS" = PASS ]; then rag_set FAIL "exit 0 but no timing CSV written (baseline/results/$exp.timing.csv)"; fi
    fi
    if [ -f "$log" ]; then cp "$log" "$BUNDLE/smoke/" 2>>"$STAGE_LOG" || true; fi
    rag_collect_ray_logs "$BUNDLE/.ray_marker" "$BUNDLE/ray_logs/$STAGE_NAME"
    rag_gpu_snapshot >>"$STAGE_LOG" 2>&1
    # verl saved a last-step checkpoint (model + optimizer, about 7 GB for Qwen3-0.6B); no later stage
    # reads it, so remove it now instead of leaving one per smoke run on the shared disk.
    if [ -d "$CKPT_ROOT/$exp" ]; then
        echo "[run_all_gates] size of this smoke run's own checkpoint before removing it: $(du -sh "$CKPT_ROOT/$exp" 2>/dev/null | awk '{ print $1 }')" >>"$STAGE_LOG"
    fi
    if ! rag_remove_smoke_ckpt "$exp" >>"$STAGE_LOG" 2>&1; then
        STAGE_NOTE="${STAGE_NOTE:+$STAGE_NOTE; }could not remove its checkpoint $CKPT_ROOT/$exp after the run (see log)"
    fi
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
    rag_gate_result "$GATES/gate_d_smoke_$exp.json"
}
stage_smoke_worst_case_check() { rag_smoke_check worst_case "$SMOKE_STEPS"; }
stage_smoke_host_guard_check() { rag_smoke_check host_guard "$SMOKE_STEPS"; }
stage_smoke_no_graph_check() { rag_smoke_check no_graph "$SMOKE_STEPS"; }
stage_smoke_no_logprobs_check() { rag_smoke_check no_logprobs "$SMOKE_LOGPROB_STEPS"; }

stage_smoke_compare() {
    local -a args
    local label csv j gpu_gb="" gpu_src="" gpu_res
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
    # --gpu-total-gb is the GPU size as torch reports it (what the engine's plan_memory uses):
    # (1) environment.gpu_total_gb that this bundle's GPU gates recorded from torch, else
    # (2) nvidia-smi memory.total - memory.reserved (memory.total alone also counts the driver /
    # firmware reserve: 40960 MiB on an A100 40GB), else (3) no flag: smoke_compare's default.
    for j in gate_b_greedy_three_way gate_b_greedy_sparsespec gate_c_reload_sparsespec_synthetic \
        gate_c_reload_sparsespec_step25; do
        [ -s "$GATES/$j.json" ] || continue
        gpu_gb=$(rag_json_get "$GATES/$j.json" environment.gpu_total_gb)
        if rag_is_num "$gpu_gb"; then gpu_src="torch total_memory from $j.json"; break; fi
        gpu_gb=""
    done
    if [ -z "$gpu_gb" ] && [ -n "${RAG_GPU_TOTAL_MIB:-}" ]; then
        gpu_res=$(nvidia-smi --query-gpu=index,memory.reserved --format=csv,noheader,nounits 2>/dev/null |
            awk -F', *' -v i="$GPU_INDEX" '$1 == i { print $2; exit }' || true)
        if rag_is_num "$gpu_res"; then
            gpu_gb=$(awk -v m="$RAG_GPU_TOTAL_MIB" -v r="$gpu_res" 'BEGIN { printf "%.2f", (m - r) / 1024 }')
            gpu_src="nvidia-smi memory.total - memory.reserved"
        fi
    fi
    if [ -n "$gpu_gb" ]; then
        args+=(--gpu-total-gb "$gpu_gb")
    else
        gpu_src="smoke_compare's default (no torch figure in the gate JSONs, no nvidia-smi memory.reserved)"
    fi
    STAGE_NOTE="${STAGE_NOTE:+$STAGE_NOTE; }GPU total: $gpu_src"
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
    rm -f "$BUNDLE/.ray_marker" "$BUNDLE/.rerun_step50"
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
    local waited=0 slog=${STAGE_LOG:-/dev/null} extra="" had_child=0
    [ -n "${RAG_CHILD:-}" ] && had_child=1
    if rag_own_child_alive; then
        [ -n "${RAG_STAGE_PGID:-}" ] || rag_probe_stage_pgid
        say "stopping this runner's own stage process $RAG_CHILD (${STAGE_NAME:-?}) with SIGTERM"
        kill -TERM "$RAG_CHILD" 2>/dev/null || true
        while rag_own_child_alive && [ "$waited" -lt $((KILL_AFTER_S + 15)) ]; do
            sleep 1
            waited=$((waited + 1))
        done
        if rag_own_child_alive; then
            say "it is still running after ${waited}s (pid $RAG_CHILD)"
            [ -n "${RAG_STAGE_PGID:-}" ] || extra="; its process $RAG_CHILD is still running (no process group of its own to stop)"
        else
            RAG_CHILD=""   # gone and collected by bash: its pid is free again (see rag_group_recycled)
        fi
    else
        RAG_CHILD=""   # already exited and collected (its pid may be someone else's by now)
    fi
    if [ "$had_child" = 1 ] && [ -z "${RAG_STAGE_PGID:-}" ] && [ -z "$TIMEOUT_CMD" ]; then
        extra="$extra; TIMEOUT_BIN=none: the stage had no process group of its own, so processes it started may still be running (nvidia-smi in the log)"
        rag_gpu_snapshot >>"$slog" 2>&1
    fi
    # the rest of the stage's own process group (a workload that handles SIGTERM can outlive timeout)
    if [ -n "${RAG_STAGE_PGID:-}" ]; then
        say "checking this stage's own process group $RAG_STAGE_PGID (SIGTERM, then SIGKILL after ${KILL_AFTER_S}s)"
        rag_reap_stage_group "$slog"
        if [ -n "$RAG_REAP_NOTE" ]; then
            say "  $RAG_REAP_NOTE"
            extra="$extra; $RAG_REAP_NOTE"
        fi
        rag_gpu_leftover_note "$slog" interrupted
        if [ -n "$RAG_GPU_LEFT_NOTE" ]; then extra="$extra; $RAG_GPU_LEFT_NOTE"; fi
    fi
    if [ "${RAG_STAGE_OPEN:-0}" = 1 ] && [ -f "$BUNDLE/.ray_marker" ]; then
        case ${STAGE_NAME:-} in
            reference_plumbing | smoke_worst_case | smoke_host_guard | smoke_no_graph | smoke_no_logprobs)
                rag_collect_ray_logs "$BUNDLE/.ray_marker" "$BUNDLE/ray_logs/$STAGE_NAME" ;;
        esac
    fi
    RAG_OPEN_STAGE_REASON="interrupted by SIG$1 while running$extra"
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

rag_add_refusal() {  # rag_add_refusal <problem and its fix>: one line of REFUSED.txt
    RAG_REFUSALS="${RAG_REFUSALS:+$RAG_REFUSALS
}$*"
}

# Refuse to start: exit 2 with the env record, REFUSED.txt (one problem and its fix per line),
# SUMMARY.txt and the tarball. Everything that can be fixed in a minute is found here, in the
# first minutes, all at once, before any GPU stage.
rag_refuse() {
    local line n
    n=$(printf '%s\n' "$RAG_REFUSALS" | grep -c . || true)
    printf '%s\n' "$RAG_REFUSALS" >"$BUNDLE/REFUSED.txt"
    say "REFUSING TO START ($n problem(s); each with its fix, also in $BUNDLE_NAME/REFUSED.txt):"
    while IFS= read -r line; do
        [ -n "$line" ] && say "  - $line"
    done <"$BUNDLE/REFUSED.txt"
    say "nothing GPU-related was run, and nothing was killed or signalled; the env record is in the bundle"
    rag_fact refused "$(head -n 1 "$BUNDLE/REFUSED.txt")"
    RAG_REMAINING_REASON="not run: refused to start (REFUSED.txt)"
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
    RAG_REFUSALS=""
    if ! cd "$REPO_ROOT"; then rag_add_refusal "cannot cd to $REPO_ROOT"; rag_refuse; fi
    RAG_OWN_PGID=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ' || true)

    RAG_PYTEST_OK=0
    RAG_SPARSESPEC_PATCH=""
    RAG_SPARSESPEC_PATCH_LINE=""
    RAG_SPARSESPEC_NVCC=""
    RAG_SPARSESPEC_NVCC_LINE=""
    RAG_CONDA_ENVS=""
    RAG_STAGE_PGID=""
    RAG_UNKILLED=""
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

    # ---- preflight: conda, envs, shell variables, host RAM, GPU (read-only) ----
    # Each problem found here that Harrison can fix in a minute is collected with its fix; after
    # the env stage (which records everything) the run refuses to start if there is any.
    local missing="" s need_gpu=0 need_sp=0 need_pytest=0 need_verl=0 need_engine=0
    for s in $RAG_STAGES; do
        [ "$s" = env ] && continue
        rag_selected "$s" || continue
        if rag_stage_is_gpu "$s"; then need_gpu=1; fi
        [ "$(rag_stage_envkind "$s")" = sparsespec ] && need_sp=1
        case $s in
            cpu_tests | gpu_unit_tests*) need_pytest=1 ;;
            reference_plumbing | smoke_worst_case | smoke_host_guard | smoke_no_graph | smoke_no_logprobs) need_verl=1 ;;
        esac
        case $s in  # stages that build the real engine (reference_plumbing uses the HF reference engine)
            gpu_unit_tests* | gate_3b | gate_3c_synthetic | gate_3c_step25 | \
                smoke_worst_case | smoke_host_guard | smoke_no_graph | smoke_no_logprobs) need_engine=1 ;;
        esac
    done
    if [ -z "$CONDA_BIN" ]; then
        rag_add_refusal "conda not found (looked at \$CONDA_EXE, PATH, ~/miniconda3, ~/anaconda3, ~/miniforge3, ~/mambaforge, /opt/conda). Fix: put conda on PATH (or export CONDA_EXE=<path to conda>) and rerun"
    else
        RAG_CONDA_ENVS=$("$CONDA_BIN" env list 2>/dev/null | awk '!/^#/ && NF { print $1 }' | tr '\n' ' ' || true)
        if ! "$CONDA_BIN" run --help 2>&1 | grep -q -- '--no-capture-output'; then
            ss_warn "'conda run --help' does not mention --no-capture-output (conda older than 4.9?): stage commands may fail to start"
        fi
        rag_env_exists "$SPARSESPEC_ENV" || missing="$missing $SPARSESPEC_ENV (SPARSESPEC_ENV)"
        rag_env_exists "$BASELINE_ENV" || missing="$missing $BASELINE_ENV (BASELINE_ENV)"
        if [ -n "$missing" ]; then
            ss_warn "conda env(s) missing:$missing. Envs found: ${RAG_CONDA_ENVS:-none}."
        fi
        # F1: without SPARSESPEC_ENV nothing of SparseSpec can run
        if [ "$need_sp" = 1 ] && ! rag_env_exists "$SPARSESPEC_ENV"; then
            rag_add_refusal "conda env '$SPARSESPEC_ENV' (SPARSESPEC_ENV) does not exist (envs found: ${RAG_CONDA_ENVS:-none}). Fix: bash integration/build_sparsespec_env.sh (HANDOFF.md section 1), or export SPARSESPEC_ENV=<the env's name>, then rerun"
        elif ! rag_env_exists "$SPARSESPEC_ENV" && ! rag_env_exists "$BASELINE_ENV"; then
            rag_add_refusal "neither conda env exists:$missing (envs found: ${RAG_CONDA_ENVS:-none}). Fix: bash integration/build_sparsespec_env.sh (HANDOFF.md section 1)"
        fi
        if ! rag_env_exists "$BASELINE_ENV"; then
            ss_warn "conda env '$BASELINE_ENV' (BASELINE_ENV) is missing: the stock-vLLM stages are SKIPped, so gate 3b cannot pass (three-way needs stock vLLM)"
        fi
    fi
    # the conda command as the printed commands use it
    RAG_CONDA_CMD=conda
    if [ -n "$CONDA_BIN" ] && [ "$(command -v conda 2>/dev/null || true)" != "$CONDA_BIN" ]; then RAG_CONDA_CMD=$CONDA_BIN; fi
    rag_fact conda_cmd "$RAG_CONDA_CMD"
    rag_fact repo_root "$REPO_ROOT"
    # what the printed rerun commands must set to see the same prompts / model as this run: only
    # what differs from the defaults of baseline/config/grpo_base.sh (the gates have the same ones)
    RAG_RERUN_DATA_DIR=""
    RAG_RERUN_MODEL_PATH=""
    if [ "$DATA_DIR" != "$HOME/data/sparsespec_baseline" ]; then RAG_RERUN_DATA_DIR=$DATA_DIR; fi
    if [ "$MODEL_PATH" != "Qwen/Qwen3-0.6B" ]; then RAG_RERUN_MODEL_PATH=$MODEL_PATH; fi
    rag_fact rerun_data_dir "$RAG_RERUN_DATA_DIR"
    rag_fact rerun_model_path "$RAG_RERUN_MODEL_PATH"
    # F3: serve/attention/backend.py takes the RAFT include and lib dir from INSTALL_PREFIX, else
    # PREFIX, else CONDA_PREFIX (the first non-empty one). The gate run must see the same shell
    # environment as the real training run, so the runner never unsets them itself.
    local pname="" pval=""
    if [ -n "${INSTALL_PREFIX:-}" ]; then pname=INSTALL_PREFIX; pval=$INSTALL_PREFIX
    elif [ -n "${PREFIX:-}" ]; then pname=PREFIX; pval=$PREFIX; fi
    RAG_PREFIX_NOTE=""
    if [ -n "$pname" ] && [ "$need_sp" = 1 ]; then   # only the SparseSpec engine reads them (not BASELINE_ENV)
        if [ -e "$pval/lib/librmm.so" ]; then
            RAG_PREFIX_NOTE="$pname=$pval is set in this shell and has lib/librmm.so: the engine takes RAFT headers and librmm.so from it, not from the env's CONDA_PREFIX (serve/attention/backend.py)"
            rag_fact prefix_override "$pname=$pval (has lib/librmm.so; used instead of CONDA_PREFIX)"
            ss_warn "$RAG_PREFIX_NOTE"
        else
            rag_add_refusal "$pname=$pval is set in this shell, and serve/attention/backend.py takes the RAFT headers and librmm.so from it (INSTALL_PREFIX, else PREFIX, else CONDA_PREFIX), but $pval/lib/librmm.so does not exist: every engine build would fail with 'librmm.so is not found'. Fix: unset INSTALL_PREFIX PREFIX, then rerun"
        fi
    fi
    # F5: verl's trainer opens data.train_files and data.val_files at start (baseline/config/grpo_base.sh)
    if [ "$need_verl" = 1 ]; then
        local pq
        for pq in "$DATA_DIR/math_l3to5_train.parquet" "$DATA_DIR/math500_test.parquet"; do
            [ -f "$pq" ] || rag_add_refusal "$pq does not exist (DATA_DIR=$DATA_DIR): reference_plumbing and every smoke run open it at start. Fix: export DATA_DIR=<the directory with the baseline's parquets>, or python baseline/prepare_data.py --out-dir \"$DATA_DIR\", then rerun"
        done
    fi
    # G1: the engine page-locks 16 GiB of host memory for its offload pool
    rag_host_mem
    if rag_is_uint "$MEM_AVAIL_KIB" && [ "$MEM_AVAIL_KIB" -lt $((24 * 1048576)) ]; then
        ss_warn "only $(awk -v k="$MEM_AVAIL_KIB" 'BEGIN { printf "%.1f", k / 1048576 }') GiB of host RAM available (MemAvailable < 24 GiB): the engine page-locks 16 GiB for its offload pool, so the engine stages may fail or the box may swap"
    fi
    RAG_REQ_MIB=""
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
        if [ "$need_gpu" = 1 ] && ! rag_gpu_enough; then
            rag_add_refusal "only $(rag_mib_to_gib "$GPU_FREE_MIB") GiB of GPU $GPU_INDEX is free, the gates need $(rag_mib_to_gib "$RAG_REQ_MIB") GiB ($RAG_REQ_TEXT). Processes on the GPU: $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ';' | cut -c1-300 || true). Fix: wait until the GPU is free (nvidia-smi), then rerun"
        fi
    elif [ "$need_gpu" = 1 ]; then
        rag_add_refusal "cannot read the GPU's free memory ($RAG_GPU_ERR); refusing to start GPU jobs on a shared card blind. Fix: make nvidia-smi work in this shell, then rerun"
    fi

    # ---- the stages ----
    rag_stage env
    # F2 / F4: what only the env probes can tell (both recorded by the env stage)
    if rag_env_exists "$SPARSESPEC_ENV"; then
        if [ "$need_pytest" = 1 ] && [ "$RAG_PYTEST_OK" != 1 ]; then
            rag_add_refusal "pytest is not importable in conda env '$SPARSESPEC_ENV' (see logs/$(basename "$STAGE_LOG")): the CPU and GPU unit-test stages, a required gate, cannot run. Fix: bash integration/build_sparsespec_env.sh (its pytest step), or: $RAG_CONDA_CMD run -n $SPARSESPEC_ENV python -m pip install --no-deps pytest iniconfig pluggy"
        fi
        if [ "$need_verl" = 1 ]; then
            case ${RAG_SPARSESPEC_PATCH:-} in
                applied) ;;
                "") ss_warn "the verl patch state of $SPARSESPEC_ENV was not recorded (see env/$SPARSESPEC_ENV/versions.txt); the verl stages may fail" ;;
                absent | partial) rag_add_refusal "the verl patch is not applied in conda env '$SPARSESPEC_ENV' (env probe: PATCH ${RAG_SPARSESPEC_PATCH}; env/$SPARSESPEC_ENV/versions.txt), so reference_plumbing and the smoke runs cannot use the SparseSpec rollout. Fix: bash integration/setup_env.sh $SPARSESPEC_ENV, then rerun" ;;
                *) rag_add_refusal "verl could not be inspected in conda env '$SPARSESPEC_ENV' (env probe: ${RAG_SPARSESPEC_PATCH_LINE:-PATCH $RAG_SPARSESPEC_PATCH}; env/$SPARSESPEC_ENV/versions.txt), so reference_plumbing and the smoke runs would fail at import. Fix: bash integration/setup_env.sh $SPARSESPEC_ENV (it reinstalls the patched verl and checks it), then rerun" ;;
            esac
        fi
    fi
    # F6: the engine JIT-compiles its kernels at every start with the compiler the env probe looked for
    if [ "$need_engine" = 1 ] && rag_env_exists "$SPARSESPEC_ENV"; then
        case ${RAG_SPARSESPEC_NVCC:-} in
            found) ;;
            missing) rag_add_refusal "no CUDA compiler for the engine's JIT in conda env '$SPARSESPEC_ENV' (env probe: ${RAG_SPARSESPEC_NVCC_LINE}): every engine start would fail compiling its kernels. Fix: in this shell (inside tmux if you use it) paste the line 'unset INSTALL_PREFIX PREFIX; export CUDA_HOME=...' that bash integration/build_sparsespec_env.sh --verify prints at its end (step 3 of 'next'), then rerun" ;;
            *) ss_warn "could not tell whether the engine's JIT finds a CUDA compiler in $SPARSESPEC_ENV (env probe: ${RAG_SPARSESPEC_NVCC_LINE:-no NVCC line}; env/$SPARSESPEC_ENV/versions.txt); the engine stages may fail at their first kernel compile" ;;
        esac
    fi
    if [ -n "$RAG_REFUSALS" ]; then rag_refuse; fi
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
