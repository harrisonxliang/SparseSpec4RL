# Shared helpers for the integration scripts (source this file; bash >= 3.2).
#
# The scripts in integration/ were written on a machine that could not run
# them (no conda, no GPU), so they are built to make a first-run failure cheap
# to diagnose: everything is copied to a log file, every phase announces
# itself, and an error prints which step failed, the command, its exit status,
# the tail of the log and what to send back.
#
#   source "$(dirname "$0")/lib.sh"
#   ss_init setup_env                 # start logging, install the error trap
#   ss_step "create the env"          # numbered banner; remembered for the error report
#   ss_run conda create ...           # echo the command, then run it
#   ss_need git "install git"         # fail early with a hint if a tool is missing
#   ss_done                           # final OK line with the log path
#
# Environment:
#   SS_LOG_DIR   where logs go (default: <repo>/integration/logs, git-ignored)
#   SS_TRACE=1   also `set -x` (xtrace goes to the log only, not the terminal)

SS_SCRIPT=${SS_SCRIPT:-script}
SS_STEP_NO=0
SS_STEP_TITLE="start"
SS_LOG=""
SS_FAILED=0

ss_timestamp() { date -u +%Y%m%dT%H%M%SZ; }

ss_log() { printf '[%s] %s\n' "$SS_SCRIPT" "$*"; }
ss_warn() { printf '[%s] WARNING: %s\n' "$SS_SCRIPT" "$*" >&2; }

ss_step() {
    SS_STEP_NO=$((SS_STEP_NO + 1))
    SS_STEP_TITLE="$*"
    printf '\n[%s] ---- step %d: %s ----\n' "$SS_SCRIPT" "$SS_STEP_NO" "$SS_STEP_TITLE"
}

# Echo a command (shell-quoted) and run it.
ss_run() {
    printf '[%s] +' "$SS_SCRIPT"
    printf ' %q' "$@"
    printf '\n'
    "$@"
}

ss_die() {
    SS_FAILED=1
    printf '\n[%s] ERROR in step %d (%s): %s\n' "$SS_SCRIPT" "$SS_STEP_NO" "$SS_STEP_TITLE" "$*" >&2
    ss_report_failure
    exit 1
}

ss_need() {  # ss_need <command> <hint>
    command -v "$1" >/dev/null 2>&1 || ss_die "'$1' not found on PATH. ${2:-}"
}

ss_report_failure() {
    {
        printf '[%s] ------------------------------------------------------------\n' "$SS_SCRIPT"
        printf '[%s] This script had never been executed before it reached the box,\n' "$SS_SCRIPT"
        printf '[%s] so a failure here may be a bug in the script itself.\n' "$SS_SCRIPT"
        printf '[%s] Send back: the lines above and the full log%s\n' "$SS_SCRIPT" "${SS_LOG:+: $SS_LOG}"
        printf '[%s] ------------------------------------------------------------\n' "$SS_SCRIPT"
    } >&2
}

ss_on_error() {  # trap handler: ss_on_error <exit status> <line> <command>
    local status=$1 line=$2 cmd=$3
    [ "$SS_FAILED" = 1 ] && return 0
    SS_FAILED=1
    {
        printf '\n[%s] FAILED in step %d (%s)\n' "$SS_SCRIPT" "$SS_STEP_NO" "$SS_STEP_TITLE"
        printf '[%s]   command: %s\n' "$SS_SCRIPT" "$cmd"
        printf '[%s]   exit status %s at line %s of %s\n' "$SS_SCRIPT" "$status" "$line" "${BASH_SOURCE[1]:-?}"
    } >&2
    ss_report_failure
}

# Start logging and install the error trap. Usage: ss_init <name> [log dir]
ss_init() {
    SS_SCRIPT=$1
    local repo_root dir
    repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
    dir=${2:-${SS_LOG_DIR:-$repo_root/integration/logs}}
    if mkdir -p "$dir" 2>/dev/null; then
        SS_LOG="$dir/${SS_SCRIPT}_$(ss_timestamp).log"
        # everything from here on goes to the terminal and to the log
        exec > >(tee -a "$SS_LOG") 2>&1
    else
        SS_LOG=""
        ss_warn "cannot create log dir $dir; output is not being saved"
    fi
    set -E
    trap 'ss_on_error $? $LINENO "$BASH_COMMAND"' ERR
    if [ "${SS_TRACE:-0}" = "1" ] && [ -n "$SS_LOG" ]; then
        exec 9>>"$SS_LOG"
        BASH_XTRACEFD=9
        set -x
    fi
    ss_log "started $(date -u '+%Y-%m-%d %H:%M:%S UTC') on $(hostname 2>/dev/null || echo '?') as $(id -un 2>/dev/null || echo '?')"
    ss_log "bash ${BASH_VERSION:-?}; cwd $(pwd)${SS_LOG:+; log $SS_LOG}"
}

ss_done() {
    printf '\n[%s] OK: %s%s\n' "$SS_SCRIPT" "${*:-finished}" "${SS_LOG:+ (log: $SS_LOG)}"
}
