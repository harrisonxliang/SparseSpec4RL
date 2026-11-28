#!/usr/bin/env bash
# Put the patched verl 0.5.0 (integration/verl-0.5.0-sparsespec.patch) into a
# conda env for the SparseSpec arm. Run on the A100 Linux box, from anywhere.
#
#   bash integration/setup_env.sh                   # env "verl-sparsespec-dev", cloned from verl-baseline if missing
#   bash integration/setup_env.sh verl-sparsespec   # Harrison's env, once it exists: only installs the patched verl
#   bash integration/setup_env.sh --check [env]     # read-only: run every check, print the plan, change nothing
#   bash integration/setup_env.sh --help
#
# What it does to the TARGET env, and nothing else:
#   1. creates it as a clone of $BASE_ENV (default verl-baseline) if it does not exist;
#   2. default: clones verl v0.5.0 into integration/.verl-0.5.0 (git-ignored), applies the patch
#      there and installs that checkout editable with --no-deps (replacing the env's stock verl);
#      VERL_IN_PLACE=1: patches the verl 0.5.0 already installed in the target env instead
#      (no network, no git); use it when the box cannot reach github.com.
#   3. self-test: one line per fact (verl version, both patch hunks, torch / transformers / vllm
#      versions, whether serve.rl.engine imports), then a final OK or FAILED line.
# It does NOT build SparseSpec / FlashInfer / RAFT / the vLLM fork (Harrison owns that build).
# It refuses to touch $BASE_ENV, "verl-baseline" and conda's "base". The baseline env's verl
# package files and its site-packages listing are hashed before and after as a tripwire.
#
# Every check runs before anything is changed; a failed check prints what is wrong and the fix.
# Re-running after any failure is safe: each step detects what an earlier run already did.
#
# Options (environment variables):
#   BASE_ENV=verl-baseline     env to clone from when the target does not exist
#   VERL_SRC=<dir>             where the editable verl checkout lives (default integration/.verl-0.5.0)
#   VERL_IN_PLACE=1            patch the target env's installed verl in place (no network / no git)
#   VERL_GIT_URL=<url>         where to clone verl from (default https://github.com/volcengine/verl.git)
#   CHECK_ONLY=1               same as --check
#   SS_LOG_DIR=<dir>           where the full log goes (default integration/logs, git-ignored)
#
# Exit status: 0 done (or --check found no problem), 1 a check or a step failed, 2 bad arguments.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

usage() {
    # the header comment above, without the leading "# "
    sed -n '2,/^set -euo pipefail/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
}

bool_var() {  # bool_var <name> <value>: prints 1 or 0, exits 2 on anything else
    case "$2" in
        1 | true | True | yes) echo 1 ;;
        "" | 0 | false | False | no) echo 0 ;;
        *)
            echo "setup_env.sh: $1 must be 1 or 0, got '$2'" >&2
            exit 2
            ;;
    esac
}

CHECK_ONLY=$(bool_var CHECK_ONLY "${CHECK_ONLY:-0}")
VERL_IN_PLACE=$(bool_var VERL_IN_PLACE "${VERL_IN_PLACE:-0}")
ENV_ARG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --check) CHECK_ONLY=1 ;;
        -h | --help)
            usage
            exit 0
            ;;
        -*)
            echo "setup_env.sh: unknown option '$1' (see: bash integration/setup_env.sh --help)" >&2
            exit 2
            ;;
        *)
            if [ -n "$ENV_ARG" ]; then
                echo "setup_env.sh: more than one env name given ('$ENV_ARG', '$1')" >&2
                exit 2
            fi
            ENV_ARG=$1
            ;;
    esac
    shift
done

ENV_NAME=${ENV_ARG:-${ENV_NAME:-verl-sparsespec-dev}}
BASE_ENV=${BASE_ENV:-verl-baseline}
PROTECTED_ENVS="verl-baseline base"   # never modified, whatever BASE_ENV says
PATCH="$REPO_ROOT/integration/verl-0.5.0-sparsespec.patch"
VERL_SRC=${VERL_SRC:-$REPO_ROOT/integration/.verl-0.5.0}
case "$VERL_SRC" in /*) ;; *) VERL_SRC="$PWD/$VERL_SRC" ;; esac
VERL_SRC=${VERL_SRC%/}
VERL_PARTIAL="$VERL_SRC.partial"      # clones land here first and are moved into place when complete
VERL_GIT_URL=${VERL_GIT_URL:-https://github.com/volcengine/verl.git}
VERL_TAG=v0.5.0
VERL_VERSION=0.5.0
FSDP_REL=verl/workers/fsdp_workers.py
TRAINER_REL=verl/trainer/ppo/ray_trainer.py
FSDP_MARKER='elif rollout_name == "sparsespec":'
TRAINER_MARKER='startswith("sparsespec/")'

# sha256 of the two files in the verl==0.5.0 PyPI wheel the patch was generated against
SHA_FSDP_WORKERS=79598ab019228a331825e315ff474322460162ae7adb436e1097ec2a32780f62
SHA_RAY_TRAINER=dc6de1e241259e6384bf8299fca4152d4162d0505e4ff99840058afbdd7c1b70

# shellcheck source=integration/lib.sh
source "$SCRIPT_DIR/lib.sh"
ss_init setup_env

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/setup_env.XXXXXX")
export GIT_TERMINAL_PROMPT=0   # git must fail, not wait for a password nobody types
# pip config `user = true` (or PIP_USER=1, or an unwritable site-packages) would install the patched
# verl into ~/.local, which verl-baseline also imports. The variable overrides every pip config file.
# PYTHONNOUSERSITE is deliberately not set: it does not stop user installs and would hide the user
# site from the checks below while the training runs still import from it.
# PIP_REQUIRE_VIRTUALENV=0: a conda env is not a virtualenv, so a pip config with
# `require-virtualenv = true` would refuse the install; where pip writes is checked below.
# PIP_TARGET / PIP_PREFIX (pip's --target / --prefix) would redirect the install: unset.
if [ -n "${PIP_TARGET:-}" ] || [ -n "${PIP_PREFIX:-}" ]; then
    ss_log "unsetting PIP_TARGET/PIP_PREFIX of the calling shell"
fi
unset PIP_TARGET PIP_PREFIX
export PIP_USER=0 PIP_REQUIRE_VIRTUALENV=0
FINAL_DONE=0
RERUN_NOTE="nothing has been changed yet"
CREATED_ENV=0          # 1 while an env this run is creating may be half made
on_exit() {
    local status=$?
    if [ "$CREATED_ENV" = 1 ]; then
        remove_half_made_env || true
    fi
    rm -rf "$TMP_DIR" 2>/dev/null || true
    if [ "$FINAL_DONE" != 1 ] && [ "$status" -ne 0 ]; then
        printf '\n[setup_env] state after the failure: %s\n' "$RERUN_NOTE"
        printf '[setup_env] FAILED (exit status %s) in step %d: %s. Log: %s\n' "$status" "$SS_STEP_NO" "$SS_STEP_TITLE" "${SS_LOG:-not saved}"
    fi
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

# ---------------------------------------------------------------- helpers
N_FAIL=0
N_WARN=0
ok() { printf '[setup_env]   ok    %s\n' "$*"; }
info() { printf '[setup_env]   info  %s\n' "$*"; }
warn() {
    N_WARN=$((N_WARN + 1))
    printf '[setup_env]   WARN  %s\n' "$1"
    if [ $# -ge 2 ] && [ -n "$2" ]; then printf '[setup_env]         fix: %s\n' "$2"; fi
}
fail() {
    N_FAIL=$((N_FAIL + 1))
    printf '[setup_env]   FAIL  %s\n' "$1"
    if [ $# -ge 2 ] && [ -n "$2" ]; then printf '[setup_env]         fix: %s\n' "$2"; fi
}
indent() { sed 's/^/[setup_env]         | /'; }
realdir() { (cd "$1" 2>/dev/null && pwd -P); }
is_protected() {
    local name
    for name in $PROTECTED_ENVS "$BASE_ENV"; do
        [ "$1" = "$name" ] && return 0
    done
    return 1
}
nearest_existing_dir() {
    local d=$1
    while [ ! -d "$d" ]; do d=$(dirname "$d"); done
    printf '%s\n' "$d"
}
with_timeout() {  # with_timeout <seconds> <cmd...>; no-op wrapper where `timeout` is missing (macOS)
    local secs=$1
    shift
    if command -v timeout >/dev/null 2>&1; then timeout "$secs" "$@"; else "$@"; fi
}
file_sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | cut -d' ' -f1
    else
        echo "?"
    fi
}
version_at_least() {  # version_at_least <version> <major> <minor>
    local major minor rest
    major=${1%%.*}
    rest=${1#*.}
    minor=${rest%%.*}
    case "$major" in '' | *[!0-9]*) return 1 ;; esac
    case "$minor" in '' | *[!0-9]*) minor=0 ;; esac
    [ "$major" -gt "$2" ] || { [ "$major" -eq "$2" ] && [ "$minor" -ge "$3" ]; }
}

# conda env name -> prefix (empty when the env is not listed); reads ENV_LIST
ENV_LIST=""
refresh_env_list() {
    ENV_LIST=$("$CONDA_BIN" env list 2>"$TMP_DIR/envlist.err") || {
        fail "'conda env list' failed:" "see the conda output above"
        indent <"$TMP_DIR/envlist.err"
        return 1
    }
}
env_prefix() { printf '%s\n' "$ENV_LIST" | awk -v n="$1" '$1 == n {print $NF; exit}'; }

# probe an env's python without activating it -> P_* globals
cat >"$TMP_DIR/probe.py" <<'EOF'
import importlib.metadata as md
import importlib.util
import os
import sys


def version(name):
    try:
        return md.version(name)
    except Exception:
        return "none"


print("python_version=" + sys.version.split()[0])
print("sys_prefix=" + os.path.realpath(sys.prefix))
print("verl_dist_version=" + version("verl"))
try:
    spec = importlib.util.find_spec("verl")  # top level only: does not import verl
    origin = spec.origin if spec is not None else None
    print("verl_dir=" + (os.path.dirname(os.path.realpath(origin)) if origin else "none"))
except Exception as exc:  # noqa: BLE001
    print("verl_dir=none")
    print("verl_find_error=%s: %s" % (type(exc).__name__, str(exc).splitlines()[0] if str(exc) else ""))
print("setuptools_version=" + version("setuptools"))
print("wheel_version=" + version("wheel"))
try:
    import site

    user_site = site.getusersitepackages() if site.ENABLE_USER_SITE else ""
except Exception:  # noqa: BLE001
    user_site = ""
print("user_site=" + (os.path.realpath(user_site) if user_site else "none"))
EOF
probe_python() {  # probe_python <python>; sets P_OK (1 = the python ran) and P_*
    P_OK=0 P_PYVER="" P_PREFIX="" P_VERL_VERSION=none P_VERL_DIR=none P_SETUPTOOLS=none P_WHEEL=none P_USER_SITE=none P_ERR=""
    local out line
    if [ ! -x "$1" ]; then
        P_ERR="$1 does not exist or is not executable"
        return 0
    fi
    if ! out=$(env -u PYTHONPATH -u PYTHONHOME "$1" -B "$TMP_DIR/probe.py" 2>&1); then
        P_ERR=$(printf '%s\n' "$out" | tail -n 5)
        return 0
    fi
    P_OK=1
    while IFS= read -r line; do
        case "$line" in
            python_version=*) P_PYVER=${line#*=} ;;
            sys_prefix=*) P_PREFIX=${line#*=} ;;
            verl_dist_version=*) P_VERL_VERSION=${line#*=} ;;
            verl_dir=*) P_VERL_DIR=${line#*=} ;;
            setuptools_version=*) P_SETUPTOOLS=${line#*=} ;;
            wheel_version=*) P_WHEEL=${line#*=} ;;
            user_site=*) P_USER_SITE=${line#*=} ;;
        esac
    done <<EOF
$out
EOF
}

# tripwire: sha256 over every file of the baseline env's verl package (path + content, no
# __pycache__) and over the sorted listing of the site-packages directory that holds it
cat >"$TMP_DIR/hash_verl.py" <<'EOF'
import hashlib
import importlib.util
import os

spec = importlib.util.find_spec("verl")
if spec is None or not spec.origin:
    print("absent no-verl-package 0 - 0 -")
    raise SystemExit(0)
pkg = os.path.dirname(os.path.realpath(spec.origin))
site = os.path.dirname(pkg)
files = hashlib.sha256()
count = 0
for root, dirs, names in os.walk(pkg):
    dirs[:] = sorted(d for d in dirs if d != "__pycache__")
    for name in sorted(names):
        if name.endswith((".pyc", ".pyo")):
            continue
        path = os.path.join(root, name)
        files.update(os.path.relpath(path, pkg).encode() + b"\0")
        with open(path, "rb") as fh:
            files.update(fh.read())
        files.update(b"\0")
        count += 1
entries = sorted(e for e in os.listdir(site) if e != "__pycache__")
listing = hashlib.sha256("\n".join(entries).encode()).hexdigest()
combined = hashlib.sha256((files.hexdigest() + listing).encode()).hexdigest()
print(combined, files.hexdigest(), count, pkg, len(entries), site)
EOF
# where `python -m pip install` of a given interpreter really writes: a throwaway 2-file wheel is
# installed with --root into a temp dir (nothing is written to the env); exit 2 = outside sys.prefix
cat >"$TMP_DIR/pip_target.py" <<'EOF'
import atexit
import base64
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

NAME = "ss_pip_target_probe"
prefix = os.path.realpath(sys.prefix)
print("sys_prefix=" + prefix)
tmp = tempfile.mkdtemp(prefix="ss_pip_target.")
atexit.register(shutil.rmtree, tmp, True)  # the probe leaves nothing behind
files = {NAME + "/__init__.py": b"",
         NAME + "-0.dist-info/METADATA": ("Metadata-Version: 2.1\nName: %s\nVersion: 0\n" % NAME).encode(),
         NAME + "-0.dist-info/WHEEL": b"Wheel-Version: 1.0\nGenerator: ss\nRoot-Is-Purelib: true\nTag: py3-none-any\n"}
rec = ["%s,sha256=%s,%d" % (k, base64.urlsafe_b64encode(hashlib.sha256(v).digest()).rstrip(b"=").decode(), len(v))
       for k, v in files.items()] + [NAME + "-0.dist-info/RECORD,,"]
files[NAME + "-0.dist-info/RECORD"] = ("\n".join(rec) + "\n").encode()
whl = os.path.join(tmp, NAME + "-0-py3-none-any.whl")
with zipfile.ZipFile(whl, "w") as z:
    for k, v in files.items():
        z.writestr(k, v)
root = os.path.join(tmp, "root")
r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", "--no-compile",
                    "--no-warn-script-location", "--root", root, whl], capture_output=True, text=True)
found = ""
for d, _, names in os.walk(root):
    if os.path.basename(d) == NAME and "__init__.py" in names:
        found = "/" + os.path.relpath(os.path.dirname(d), root)
        break
if r.returncode != 0 or not found:
    for line in (r.stdout + r.stderr).strip().splitlines()[-8:]:
        print("pip| " + line)
    print("target=")
    sys.exit(1)
target = os.path.realpath(found) if os.path.isdir(found) else found
print("target=" + target)
sys.exit(0 if (target + os.sep).startswith(prefix + os.sep) else 2)
EOF
SS_PIP_TARGET=""
pip_target() {  # pip_target <python>: 0 = pip writes inside that python's prefix; sets SS_PIP_TARGET, output in $TMP_DIR/pip_target.out
    local rc=0
    env -u PYTHONPATH -u PYTHONHOME PYTHONDONTWRITEBYTECODE=1 "$1" -B "$TMP_DIR/pip_target.py" >"$TMP_DIR/pip_target.out" 2>&1 || rc=$?
    SS_PIP_TARGET=$(sed -n 's/^target=//p' "$TMP_DIR/pip_target.out")
    return "$rc"
}

BASELINE_HASH=""
BASELINE_HASH_WHAT=""
hash_baseline() {  # sets BASELINE_HASH (combined digest) and BASELINE_HASH_WHAT (what was hashed)
    local prefix out verl_digest count pkg entries site
    prefix=$(env_prefix "$BASE_ENV")
    if [ -z "$prefix" ] || [ ! -x "$prefix/bin/python" ]; then
        BASELINE_HASH=absent
        BASELINE_HASH_WHAT="'$BASE_ENV' does not exist (or has no python): nothing to hash"
        return 0
    fi
    if ! out=$(env -u PYTHONPATH -u PYTHONHOME "$prefix/bin/python" -B "$TMP_DIR/hash_verl.py" 2>&1); then
        BASELINE_HASH=hash-error
        BASELINE_HASH_WHAT="hashing with $prefix/bin/python failed: $(printf '%s' "$out" | tail -n 1)"
        return 0
    fi
    read -r BASELINE_HASH verl_digest count pkg entries site <<EOF
$out
EOF
    if [ "$BASELINE_HASH" = "absent" ]; then
        BASELINE_HASH_WHAT="'$BASE_ENV' ($prefix) has no importable verl package: nothing to hash"
    else
        BASELINE_HASH_WHAT="sha256 over the $count files of $pkg (relative path + content, __pycache__ excluded) and over the sorted listing of the $entries entries of $site"
    fi
}

apply_patch() {  # apply_patch <dir that contains verl/>
    printf '[%s] + (cd %s && patch -p1 -s < %s)\n' "$SS_SCRIPT" "$1" "$PATCH"
    (cd "$1" && patch -p1 -s <"$PATCH")
}
patch_state() {  # patch_state <dir that contains verl/> -> applied | pristine | neither | missing
    local root=$1
    if [ ! -f "$root/$FSDP_REL" ] || [ ! -f "$root/$TRAINER_REL" ]; then
        echo missing
    elif (cd "$root" && patch -p1 --dry-run -R -s -f <"$PATCH") >/dev/null 2>&1; then
        echo applied
    elif (cd "$root" && patch -p1 --dry-run -s -f <"$PATCH") >/dev/null 2>&1; then
        echo pristine
    else
        echo neither
    fi
}
describe_patch_files() {  # what the two patched files look like, for a "neither" report
    local root=$1 rel sha expected marker
    for rel in "$FSDP_REL" "$TRAINER_REL"; do
        if [ "$rel" = "$FSDP_REL" ]; then
            expected=$SHA_FSDP_WORKERS
            marker=$FSDP_MARKER
        else
            expected=$SHA_RAY_TRAINER
            marker=$TRAINER_MARKER
        fi
        if [ ! -f "$root/$rel" ]; then
            printf '%s: MISSING\n' "$rel"
            continue
        fi
        sha=$(file_sha256 "$root/$rel")
        if [ "$sha" = "$expected" ]; then
            printf '%s: sha256 %s = the pristine verl 0.5.0 wheel file\n' "$rel" "$sha"
        else
            printf '%s: sha256 %s (pristine verl 0.5.0 wheel: %s)\n' "$rel" "$sha" "$expected"
        fi
        if grep -qF "$marker" "$root/$rel"; then
            printf '%s: contains the SparseSpec hunk (%s)\n' "$rel" "$marker"
        else
            printf '%s: does not contain the SparseSpec hunk\n' "$rel"
        fi
    done
}

remove_half_made_env() {  # only ever called for an env this run started to create
    local prefix
    CREATED_ENV=0
    is_protected "$ENV_NAME" && return 0
    refresh_env_list >/dev/null 2>&1 || return 0
    prefix=$(env_prefix "$ENV_NAME")
    [ -n "$prefix" ] || return 0
    [ "$prefix" != "$(env_prefix "$BASE_ENV")" ] || return 0
    printf '[setup_env] removing the half-made env %s (%s), created by this run\n' "$ENV_NAME" "$prefix"
    "$CONDA_BIN" remove -y -n "$ENV_NAME" --all || printf '[setup_env] WARNING: could not remove it; run: conda remove -y -n %s --all\n' "$ENV_NAME"
}

# ================================================================ 1. checks
ss_step "check every prerequisite (read-only)"
MODE_TEXT="default (editable checkout at $VERL_SRC)"
[ "$VERL_IN_PLACE" = 1 ] && MODE_TEXT="VERL_IN_PLACE=1 (patch the target env's installed verl)"
info "target env: $ENV_NAME | base env: $BASE_ENV | mode: $MODE_TEXT | check only: $CHECK_ONLY"
info "repo: $REPO_ROOT"

# ---- the target name
case "$ENV_NAME" in
    "" | */* | *" "*)
        fail "target env name '$ENV_NAME' is not a plain conda env name" "pass a name such as verl-sparsespec-dev"
        ;;
esac
if is_protected "$ENV_NAME"; then
    fail "refusing to modify '$ENV_NAME': it is the baseline env ('$BASE_ENV') or a protected env ($PROTECTED_ENVS)" \
        "pass another env name, e.g. bash integration/setup_env.sh verl-sparsespec-dev"
    ss_die "the target env is protected; nothing was changed"
fi
ok "target env '$ENV_NAME' is not the baseline ('$BASE_ENV') and not one of: $PROTECTED_ENVS"

if [ -r "$PATCH" ]; then
    ok "patch file: $PATCH"
else
    fail "patch file $PATCH is missing or unreadable" "run from an up-to-date checkout of branch sparsespec-verl-backend"
fi

# ---- conda
CONDA_BIN=""
if command -v conda >/dev/null 2>&1; then
    CONDA_BIN=$(command -v conda)
elif [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE:-}" ]; then
    CONDA_BIN=$CONDA_EXE
fi
CONDA_BASE=""
CONDA_SH=""
B_PREFIX=""
T_PREFIX=""
if [ -z "$CONDA_BIN" ]; then
    fail "conda not found (not on PATH, CONDA_EXE unset)" \
        "put conda's condabin on PATH, e.g. source <miniconda>/etc/profile.d/conda.sh, or run 'conda init bash' and open a new shell"
else
    ok "conda: $CONDA_BIN ($("$CONDA_BIN" --version 2>/dev/null || echo 'version unknown'))"
    if CONDA_BASE=$("$CONDA_BIN" info --base 2>"$TMP_DIR/info.err") && [ -n "$CONDA_BASE" ] && [ -d "$CONDA_BASE" ]; then
        ok "conda info --base: $CONDA_BASE"
        CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
        if [ ! -r "$CONDA_SH" ]; then
            fail "$CONDA_SH does not exist: conda cannot be initialised in a script" \
                "check that '$CONDA_BIN' is the conda of this box (conda info --base must name its install directory)"
            CONDA_SH=""
        else
            set +u
            if . "$CONDA_SH"; then
                set -u
                if [ "$(type -t conda 2>/dev/null || true)" = "function" ]; then
                    ok "sourced $CONDA_SH: 'conda activate' is available"
                else
                    fail "sourcing $CONDA_SH did not define the conda shell function" "check the conda installation ($CONDA_BASE)"
                fi
            else
                set -u
                fail "sourcing $CONDA_SH failed" "run: bash -c 'source $CONDA_SH && conda activate base' and fix what it reports"
            fi
        fi
    else
        fail "'conda info --base' failed or printed no directory (got '$CONDA_BASE')" "run 'conda info' by hand and fix the conda installation"
        indent <"$TMP_DIR/info.err"
        CONDA_BASE=""
    fi
fi

# ---- the envs
T_VERL_DIR="" T_SYS_PREFIX="" T_SETUPTOOLS=none T_WHEEL=none PROBE_PY="" PROBE_WHO="" P_OK=0 P_VERL_DIR=none
PIP_FLAGS="--no-deps" INSTALL_STATE=needed EXPECT_VERL_DIR=""
T_STATE=unknown    # missing | ok | nopython | noverl | wrongverl | verlelsewhere
if [ -n "$CONDA_BASE" ] && refresh_env_list; then
    B_PREFIX=$(env_prefix "$BASE_ENV")
    T_PREFIX=$(env_prefix "$ENV_NAME")
    if [ -n "$B_PREFIX" ]; then
        ok "base env '$BASE_ENV' exists: $B_PREFIX"
    elif [ -z "$T_PREFIX" ]; then
        fail "neither '$ENV_NAME' nor the base env '$BASE_ENV' exists (conda env list), so there is nothing to clone" \
            "create the baseline env first (bash baseline/setup_env.sh), or set BASE_ENV=<an env with verl 0.5.0>"
    else
        warn "base env '$BASE_ENV' does not exist; not needed because '$ENV_NAME' exists, but the baseline tripwire has nothing to hash"
    fi
    if [ -n "$T_PREFIX" ] && [ -n "$B_PREFIX" ] && [ "$(realdir "$T_PREFIX")" = "$(realdir "$B_PREFIX")" ]; then
        fail "'$ENV_NAME' and '$BASE_ENV' are the same prefix ($T_PREFIX)" "pass a different env name"
        ss_die "the target env is the baseline env; nothing was changed"
    fi

    if [ -z "$T_PREFIX" ]; then
        T_STATE=missing
        info "target env '$ENV_NAME' does not exist: it will be cloned from '$BASE_ENV'"
        PROBE_PY="${B_PREFIX:+$B_PREFIX/bin/python}"
        PROBE_WHO="'$BASE_ENV' (the clone copies it)"
    else
        info "target env '$ENV_NAME' exists: $T_PREFIX"
        PROBE_PY="$T_PREFIX/bin/python"
        PROBE_WHO="'$ENV_NAME'"
    fi

    if [ -n "$PROBE_PY" ]; then
        probe_python "$PROBE_PY"
        if [ "$P_OK" != 1 ]; then
            if [ "$T_STATE" = missing ]; then
                fail "the python of '$BASE_ENV' does not run: $P_ERR" "check the baseline env by hand (do not modify it)"
            else
                T_STATE=nopython
                fail "'$ENV_NAME' exists but its python does not run ($P_ERR); probably a clone that was interrupted" \
                    "it is not the baseline, so remove it and re-run: conda remove -y -n $ENV_NAME --all"
            fi
        else
            ok "python $P_PYVER in $PROBE_WHO"
            if [ "$P_VERL_VERSION" = "$VERL_VERSION" ]; then
                ok "$PROBE_WHO has verl $P_VERL_VERSION (the patch targets $VERL_VERSION); it imports from $P_VERL_DIR"
                [ "$T_STATE" = missing ] || T_STATE=ok
            else
                if [ "$P_VERL_VERSION" = none ]; then
                    what="has no verl installed"
                    [ "$T_STATE" = missing ] || T_STATE=noverl
                else
                    what="has verl $P_VERL_VERSION, not $VERL_VERSION"
                    [ "$T_STATE" = missing ] || T_STATE=wrongverl
                fi
                if [ "$T_STATE" = missing ]; then
                    fail "the base env '$BASE_ENV' $what, so a clone of it would too" \
                        "set BASE_ENV to an env with verl $VERL_VERSION (the baseline env must not be changed)"
                else
                    fail "'$ENV_NAME' $what (--no-deps install relies on verl $VERL_VERSION's dependencies being there)" \
                        "conda run -n $ENV_NAME python -m pip install --no-deps verl==$VERL_VERSION    (or: conda remove -y -n $ENV_NAME --all, then re-run to clone it fresh)"
                fi
            fi
            if [ "$P_VERL_DIR" != none ] && [ "$P_USER_SITE" != none ]; then
                case "$P_VERL_DIR/" in
                    "$P_USER_SITE"/*)
                        fail "in $PROBE_WHO, 'import verl' resolves to the USER site $P_VERL_DIR: it shadows the verl of every env of $(id -un), the baseline's included, and an install into '$ENV_NAME' would not be used" \
                            "ask the team before removing it (baseline runs import it too): rm -rf '$P_VERL_DIR' '$P_USER_SITE'/verl-*.dist-info"
                        ;;
                esac
            fi
            if [ -n "$P_PREFIX" ] && [ "$P_VERL_DIR" != none ]; then
                case "$P_VERL_DIR/" in
                    "$P_PREFIX"/*) ;;
                    "$P_USER_SITE"/*) ;;
                    *)
                        if [ "$VERL_IN_PLACE" = 1 ] || [ "$P_VERL_DIR" != "$(realdir "$VERL_SRC" 2>/dev/null || true)/verl" ]; then
                            [ "$T_STATE" = ok ] && T_STATE=verlelsewhere
                            warn "in $PROBE_WHO, 'import verl' resolves to $P_VERL_DIR, outside the env prefix $P_PREFIX (an editable install of another checkout?); the default mode replaces it, VERL_IN_PLACE=1 refuses it"
                        fi
                        ;;
                esac
            fi
            T_SETUPTOOLS=$P_SETUPTOOLS
            T_WHEEL=$P_WHEEL
            T_VERL_DIR=$P_VERL_DIR
            T_SYS_PREFIX=$P_PREFIX
        fi
    fi
fi

# ---- tools
if command -v patch >/dev/null 2>&1; then
    PATCH_VERSION=$(patch --version 2>&1 | head -n 1 || true)
    case "$PATCH_VERSION" in
        *GNU*) ok "patch: $(command -v patch) ($PATCH_VERSION)" ;;
        *) warn "patch at $(command -v patch) is not GNU patch ($PATCH_VERSION); the flags used here (-p1 --dry-run -R -s -f) also work with the BSD patch of macOS" ;;
    esac
else
    fail "'patch' not found on PATH" \
        "install GNU patch (apt-get install patch, or put a conda-forge 'patch' binary on PATH: conda create -y -n tools -c conda-forge patch && export PATH=\$CONDA_BASE/envs/tools/bin:\$PATH)"
fi
if command -v sha256sum >/dev/null 2>&1; then
    ok "sha256sum: $(command -v sha256sum)"
elif command -v shasum >/dev/null 2>&1; then
    ok "sha256: shasum -a 256 ($(command -v shasum)); sha256sum is not installed"
else
    warn "neither sha256sum nor shasum is installed: file hashes in error reports will show '?' (the tripwire uses python's hashlib)"
fi

# ---- the verl source
CHECKOUT_STATE=n/a     # missing | empty | pristine | applied (default mode)
INPLACE_STATE=n/a      # pristine | applied | later (in-place mode)
if [ "$VERL_IN_PLACE" = 1 ]; then
    info "VERL_IN_PLACE=1: no git clone, no pip install; the verl files inside the target env are patched"
    if [ "$T_STATE" = missing ]; then
        INPLACE_STATE=later
        if [ -n "${P_VERL_DIR:-}" ] && [ "${P_VERL_DIR:-none}" != none ] && [ -n "$B_PREFIX" ]; then
            case "$P_VERL_DIR/" in
                "$(realdir "$B_PREFIX")"/*)
                    BASE_SITE=$(dirname "$P_VERL_DIR")
                    case "$(patch_state "$BASE_SITE")" in
                        pristine) ok "the base env's verl files are the pristine 0.5.0 ones the patch expects, so the clone's will be (dry run, read-only)" ;;
                        applied) warn "the BASE env's installed verl ($P_VERL_DIR) already carries the SparseSpec patch: the baseline env is not stock" "tell the team; the baseline env must run stock verl" ;;
                        *)
                            fail "the patch does not apply to the base env's verl ($P_VERL_DIR), so it will not apply to a clone" "use the default mode (git checkout of v0.5.0) instead of VERL_IN_PLACE=1"
                            describe_patch_files "$BASE_SITE" | indent
                            ;;
                    esac
                    ;;
                *) fail "in '$BASE_ENV', verl resolves to $P_VERL_DIR, outside that env: a clone would not contain it" "use the default mode (git checkout) instead of VERL_IN_PLACE=1" ;;
            esac
        fi
    elif [ "$T_STATE" = ok ] || [ "$T_STATE" = verlelsewhere ]; then
        case "$T_VERL_DIR/" in
            "$T_SYS_PREFIX"/*)
                INPLACE_SITE=$(dirname "$T_VERL_DIR")
                INPLACE_STATE=$(patch_state "$INPLACE_SITE")
                case "$INPLACE_STATE" in
                    pristine) ok "the installed verl ($T_VERL_DIR) is pristine 0.5.0: the patch applies (dry run)" ;;
                    applied) ok "the installed verl ($T_VERL_DIR) already carries the patch (an earlier run got this far)" ;;
                    *)
                        fail "the patch neither applies to nor is already applied in $T_VERL_DIR" \
                            "restore the stock files: conda run -n $ENV_NAME python -m pip install --force-reinstall --no-deps verl==$VERL_VERSION, then re-run"
                        describe_patch_files "$INPLACE_SITE" | indent
                        ;;
                esac
                for f in "$T_VERL_DIR/workers/fsdp_workers.py" "$T_VERL_DIR/trainer/ppo/ray_trainer.py"; do
                    if [ ! -w "$f" ] || [ ! -w "$(dirname "$f")" ]; then
                        fail "no write access to $f (or its directory) as $(id -un)" "run as the owner of the env ($(ls -ld "$T_SYS_PREFIX" | awk '{print $3}'))"
                    fi
                done
                ;;
            *)
                fail "in '$ENV_NAME', verl resolves to $T_VERL_DIR, outside the env ($T_SYS_PREFIX): VERL_IN_PLACE=1 would patch that, not the env" \
                    "if it is $VERL_SRC/verl, an earlier default-mode run installed it: re-run without VERL_IN_PLACE"
                ;;
        esac
    fi
else
    # where the checkout stands
    if [ -d "$VERL_PARTIAL" ]; then
        info "$VERL_PARTIAL is left over from an interrupted clone: it will be removed"
    fi
    if [ ! -e "$VERL_SRC" ]; then
        CHECKOUT_STATE=missing
        info "no checkout at $VERL_SRC yet: verl $VERL_TAG will be cloned from $VERL_GIT_URL"
    elif [ ! -d "$VERL_SRC" ]; then
        fail "$VERL_SRC exists but is not a directory" "remove it or set VERL_SRC=<another dir>"
    elif [ -z "$(ls -A "$VERL_SRC" 2>/dev/null)" ]; then
        CHECKOUT_STATE=empty
        info "$VERL_SRC is an empty directory: verl $VERL_TAG will be cloned into it"
    elif [ ! -d "$VERL_SRC/verl" ]; then
        fail "$VERL_SRC exists but contains no verl/ package: not a finished verl checkout (an interrupted 'git clone'?)" \
            "rm -rf '$VERL_SRC' and re-run (or set VERL_SRC=<another dir>)"
    else
        SRC_VERSION=$(cat "$VERL_SRC/verl/version/version" 2>/dev/null || echo unknown)
        if [ "$SRC_VERSION" != "$VERL_VERSION" ]; then
            fail "the checkout at $VERL_SRC is verl '$SRC_VERSION' (verl/version/version), not $VERL_VERSION" "rm -rf '$VERL_SRC' and re-run"
        fi
        if command -v git >/dev/null 2>&1 && git -C "$VERL_SRC" rev-parse --git-dir >/dev/null 2>&1; then
            OTHER_CHANGES=$(git -C "$VERL_SRC" status --porcelain --untracked-files=no 2>&1 | grep -v -e " $FSDP_REL\$" -e " $TRAINER_REL\$" || true)
            if [ -n "$OTHER_CHANGES" ]; then
                fail "the checkout at $VERL_SRC has changes besides the two patched files (half-made or edited):" "rm -rf '$VERL_SRC' and re-run"
                printf '%s\n' "$OTHER_CHANGES" | head -n 20 | indent
            fi
            info "checkout commit: $(git -C "$VERL_SRC" describe --tags --always 2>/dev/null || echo '?')"
        else
            info "$VERL_SRC is not a git checkout (or git is missing): it is checked by version file and patch state only"
        fi
        CHECKOUT_STATE=$(patch_state "$VERL_SRC")
        case "$CHECKOUT_STATE" in
            pristine) ok "checkout $VERL_SRC: verl $SRC_VERSION, not patched yet" ;;
            applied) ok "checkout $VERL_SRC: verl $SRC_VERSION, already patched (an earlier run got this far)" ;;
            *)
                fail "the patch neither applies to nor is already applied in $VERL_SRC" "rm -rf '$VERL_SRC' and re-run"
                describe_patch_files "$VERL_SRC" | indent
                CHECKOUT_STATE=broken   # also when a patched file is missing: never "missing" (= clone) here
                ;;
        esac
    fi

    # git + network, only needed when there is something to clone
    if [ "$CHECKOUT_STATE" = missing ] || [ "$CHECKOUT_STATE" = empty ]; then
        if ! command -v git >/dev/null 2>&1; then
            fail "git is not installed, and there is no verl checkout at $VERL_SRC" \
                "install git, or use VERL_IN_PLACE=1 bash integration/setup_env.sh $ENV_NAME (patches the env's installed verl 0.5.0; no git, no network)"
        elif with_timeout 60 git ls-remote --exit-code "$VERL_GIT_URL" "refs/tags/$VERL_TAG" >"$TMP_DIR/lsremote.out" 2>&1; then
            ok "git $(git --version 2>/dev/null | awk '{print $3}'): $VERL_GIT_URL is reachable and has tag $VERL_TAG"
        else
            fail "cannot reach $VERL_GIT_URL or it has no tag $VERL_TAG (git ls-remote failed):" \
                "if the box has no route to github.com: VERL_IN_PLACE=1 bash integration/setup_env.sh $ENV_NAME (patches the env's installed verl in place); behind a proxy: export https_proxy=...; or clone elsewhere and copy the tree to $VERL_SRC"
            indent <"$TMP_DIR/lsremote.out"
        fi
    fi

    # pip in the env that will receive the install
    if [ -n "${PROBE_PY:-}" ] && [ "${P_OK:-0}" = 1 ]; then
        if PIP_VERSION=$(env -u PYTHONPATH -u PYTHONHOME "$PROBE_PY" -B -m pip --version 2>"$TMP_DIR/pip.err"); then
            ok "pip in $PROBE_WHO: $PIP_VERSION"
            # the clone has the same pip and the same pip config as the base, so the base answers for it
            if pip_target "$PROBE_PY"; then
                ok "pip in $PROBE_WHO installs into its own prefix: $SS_PIP_TARGET (PIP_USER=0)"
            else
                rc=$?
                if [ "$rc" = 2 ]; then
                    fail "pip in $PROBE_WHO would install into $SS_PIP_TARGET, outside the env (a pip 'user' setting or an unwritable site-packages): verl-baseline imports that user site too" \
                        "check: $PROBE_PY -m pip config list; make sure nothing in this shell sets PIP_USER after the script exports PIP_USER=0, and that $(id -un) can write to the env"
                else
                    fail "a throwaway 'pip install --no-index --root <tmp>' of a local wheel fails in $PROBE_WHO, so the install of verl would fail too:" \
                        "fix what pip reports below, then re-run"
                fi
                indent <"$TMP_DIR/pip_target.out"
            fi
        else
            fail "'python -m pip' does not work in $PROBE_WHO:" \
                "conda install -y -n $ENV_NAME pip   (after the clone, if the env does not exist yet; never in $BASE_ENV)"
            indent <"$TMP_DIR/pip.err"
        fi
        if version_at_least "${T_SETUPTOOLS:-none}" 70 1 || { version_at_least "${T_SETUPTOOLS:-none}" 64 0 && [ "${T_WHEEL:-none}" != none ]; }; then
            PIP_FLAGS="--no-deps --no-build-isolation"
            ok "setuptools ${T_SETUPTOOLS} (wheel ${T_WHEEL:-none}) in $PROBE_WHO: verl is built with --no-build-isolation, no PyPI access needed"
        else
            PIP_FLAGS="--no-deps"
            warn "setuptools ${T_SETUPTOOLS:-none} / wheel ${T_WHEEL:-none} in $PROBE_WHO cannot build an editable verl without build isolation: pip will download setuptools and wheel from PyPI (needs network to pypi.org or a mirror)" \
                "if PyPI is unreachable: VERL_IN_PLACE=1 bash integration/setup_env.sh $ENV_NAME"
        fi
    fi
    if [ "$T_STATE" = ok ] && [ "$T_VERL_DIR" = "$(realdir "$VERL_SRC" 2>/dev/null || echo "$VERL_SRC")/verl" ]; then
        INSTALL_STATE=done
        ok "'$ENV_NAME' already imports verl from $VERL_SRC/verl (editable install of an earlier run)"
    else
        INSTALL_STATE=needed
    fi

    # write access to the checkout directory
    if [ -d "$VERL_SRC" ]; then
        NOT_WRITABLE=""
        [ -w "$VERL_SRC" ] || NOT_WRITABLE="$VERL_SRC"
        if [ "$CHECKOUT_STATE" = pristine ]; then
            for f in "$VERL_SRC/$FSDP_REL" "$VERL_SRC/$TRAINER_REL"; do
                [ -w "$f" ] && [ -w "$(dirname "$f")" ] || NOT_WRITABLE="$NOT_WRITABLE $f"
            done
        fi
        if [ -z "$NOT_WRITABLE" ]; then
            ok "write access to $VERL_SRC"
        else
            fail "no write access as $(id -un) to:$NOT_WRITABLE (the checkout; pip also writes verl.egg-info there)" "fix the permissions or set VERL_SRC=<a writable dir>"
        fi
    else
        PARENT=$(nearest_existing_dir "$(dirname "$VERL_SRC")")
        if [ -w "$PARENT" ]; then
            ok "write access to $PARENT (the checkout goes to $VERL_SRC)"
        else
            fail "no write access to $PARENT, where the checkout $VERL_SRC would be created, as $(id -un)" "fix the permissions or set VERL_SRC=<a writable dir>"
        fi
    fi
fi

# ---- disk
if [ "$T_STATE" = missing ] && [ -n "$B_PREFIX" ]; then
    ENVS_DIR=$(dirname "$B_PREFIX")
    BASE_KB=$(du -sk "$B_PREFIX" 2>/dev/null | awk '{print $1}' || true)
    FREE_KB=$(df -Pk "$ENVS_DIR" 2>/dev/null | awk 'NR == 2 {print $4}' || true)
    info "disk: '$BASE_ENV' occupies $((${BASE_KB:-0} / 1024)) MiB; $ENVS_DIR has $((${FREE_KB:-0} / 1024)) MiB free (df below)"
    df -Ph "$ENVS_DIR" 2>&1 | indent || true
    if [ -n "$BASE_KB" ] && [ -n "$FREE_KB" ] && [ "$FREE_KB" -lt "$BASE_KB" ]; then
        warn "less free space in $ENVS_DIR than the size of '$BASE_ENV': the clone may not fit (conda hard-links its own packages, so a clone usually needs less; pip-installed packages are copied)" \
            "free some space, or point conda at another envs dir (conda config --add envs_dirs <dir>)"
    fi
    if [ ! -w "$ENVS_DIR" ]; then
        info "$ENVS_DIR is not writable for $(id -un): conda will put the new env into another envs_dirs entry (conda config --show envs_dirs)"
    fi
fi
case "$CHECKOUT_STATE" in
    missing | empty)
        info "disk at the checkout location (a depth-1 clone of verl is about 20 MB):"
        df -Ph "$(nearest_existing_dir "$(dirname "$VERL_SRC")")" 2>&1 | indent || true
        ;;
esac

# ---- the baseline tripwire, before anything changes
BASELINE_HASH_BEFORE=absent
if [ -n "$CONDA_BASE" ] && [ -n "$ENV_LIST" ]; then
    hash_baseline
    BASELINE_HASH_BEFORE=$BASELINE_HASH
    info "baseline tripwire ($BASE_ENV): $BASELINE_HASH_WHAT"
    info "baseline tripwire value before: $BASELINE_HASH_BEFORE"
fi

printf '[setup_env] checks: %d failed, %d warning(s)\n' "$N_FAIL" "$N_WARN"

# ================================================================ 2. plan
ss_step "plan"
if [ "$CHECK_ONLY" = 1 ]; then WOULD="would"; else WOULD="will"; fi
plan() { printf '[setup_env]   - %s\n' "$*"; }
if [ "$T_STATE" = missing ]; then
    plan "$WOULD create env '$ENV_NAME' as a clone of '$BASE_ENV' ($B_PREFIX): conda create -y -n $ENV_NAME --clone $BASE_ENV"
else
    plan "keep the existing env '$ENV_NAME' (${T_PREFIX:-?}); no env is created"
fi
if [ "$VERL_IN_PLACE" = 1 ]; then
    case "$INPLACE_STATE" in
        applied) plan "nothing to patch: the verl installed in '$ENV_NAME' already carries the patch" ;;
        *) plan "$WOULD patch the verl 0.5.0 installed in '$ENV_NAME' in place ($FSDP_REL, $TRAINER_REL under its site-packages)" ;;
    esac
    plan "no git clone and no pip install in this mode"
else
    [ -d "$VERL_PARTIAL" ] && plan "$WOULD remove the leftover $VERL_PARTIAL"
    case "$CHECKOUT_STATE" in
        missing | empty) plan "$WOULD clone verl $VERL_TAG (depth 1) from $VERL_GIT_URL into $VERL_SRC (via $VERL_PARTIAL)" ;;
        *) plan "reuse the checkout at $VERL_SRC" ;;
    esac
    case "$CHECKOUT_STATE" in
        applied) plan "nothing to patch: the checkout already carries the patch" ;;
        *) plan "$WOULD apply $PATCH in $VERL_SRC ($FSDP_REL, $TRAINER_REL)" ;;
    esac
    if [ "${INSTALL_STATE:-needed}" = done ]; then
        plan "nothing to install: '$ENV_NAME' already imports verl from $VERL_SRC/verl"
    else
        if [ "$T_STATE" = missing ]; then REPLACES="the stock verl the clone copies from '$BASE_ENV'"; else REPLACES="its verl at ${T_VERL_DIR:-?}"; fi
        plan "$WOULD run, inside '$ENV_NAME': python -m pip install ${PIP_FLAGS:---no-deps} -e $VERL_SRC (replaces $REPLACES; nothing else is installed)"
    fi
fi
plan "$WOULD check that '$BASE_ENV' is unchanged (tripwire above) and run the self-test"
plan "never written: '$BASE_ENV' (${B_PREFIX:-not found}); the only files this script writes are in '$ENV_NAME', $( [ "$VERL_IN_PLACE" = 1 ] && echo "its site-packages" || echo "$VERL_SRC" ) and the log"

if [ "$N_FAIL" -gt 0 ]; then
    if [ "$CHECK_ONLY" = 1 ]; then
        FINAL_DONE=1
        printf '\n[setup_env] CHECK FAILED: %d problem(s) above (each with a fix); nothing was changed.%s\n' "$N_FAIL" "${SS_LOG:+ Log: $SS_LOG}"
        exit 1
    fi
    ss_die "$N_FAIL check(s) failed (each is listed above with a fix); nothing was changed"
fi
if [ "$CHECK_ONLY" = 1 ]; then
    FINAL_DONE=1
    printf '\n[setup_env] CHECK OK: every prerequisite is met (%d warning(s)); nothing was changed. Run without --check to carry out the plan.%s\n' \
        "$N_WARN" "${SS_LOG:+ Log: $SS_LOG}"
    exit 0
fi

# ================================================================ 3. the env
if [ "$T_STATE" = missing ]; then
    ss_step "create '$ENV_NAME' as a clone of '$BASE_ENV' (reads '$BASE_ENV', writes only the new env)"
    CREATED_ENV=1
    RERUN_NOTE="the clone of '$ENV_NAME' failed; this run removes the half-made env, so a re-run starts the clone again"
    if ! ss_run "$CONDA_BIN" create -y -n "$ENV_NAME" --clone "$BASE_ENV"; then
        ss_die "conda create --clone failed (output above)"
    fi
    refresh_env_list || ss_die "'conda env list' failed after the clone (output above)"
    T_PREFIX=$(env_prefix "$ENV_NAME")
    [ -n "$T_PREFIX" ] || ss_die "conda create reported success but '$ENV_NAME' is not in 'conda env list'"
    probe_python "$T_PREFIX/bin/python"
    if [ "$P_OK" != 1 ] || [ "$P_VERL_VERSION" != "$VERL_VERSION" ]; then
        ss_die "the new env '$ENV_NAME' ($T_PREFIX) is not usable: python runs: $P_OK, verl: $P_VERL_VERSION ${P_ERR:+($P_ERR)}"
    fi
    CREATED_ENV=0
    ss_log "created $T_PREFIX: python $P_PYVER, verl $P_VERL_VERSION"
fi
RERUN_NOTE="'$ENV_NAME' exists and is unchanged apart from the clone; a re-run reuses it"

ss_step "activate '$ENV_NAME'"
set +u
if ! conda activate "$ENV_NAME"; then
    set -u
    ss_die "'conda activate $ENV_NAME' failed (output above); try it by hand: source $CONDA_SH && conda activate $ENV_NAME"
fi
set -u
hash -r
if [ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ] || [ "$(realdir "${CONDA_PREFIX:-/nonexistent}")" != "$(realdir "$T_PREFIX")" ]; then
    ss_die "activation did not switch to '$ENV_NAME' (CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-unset}, CONDA_PREFIX=${CONDA_PREFIX:-unset}, expected $T_PREFIX)"
fi
PYTHON=$(command -v python || true)
case "$PYTHON" in
    "$CONDA_PREFIX"/*) ss_log "active: $ENV_NAME ($CONDA_PREFIX); python = $PYTHON" ;;
    *) ss_die "after activation 'python' is '$PYTHON', not inside $CONDA_PREFIX (check PATH in ~/.bashrc)" ;;
esac
if [ -n "$B_PREFIX" ] && [ "$(realdir "$CONDA_PREFIX")" = "$(realdir "$B_PREFIX")" ]; then
    ss_die "the active env is the baseline prefix; stopping before anything is written"
fi

# ================================================================ 4. patched verl
if [ "$VERL_IN_PLACE" = 1 ]; then
    ss_step "patch the verl installed in '$ENV_NAME' in place"
    probe_python "$PYTHON"
    case "$P_VERL_DIR/" in
        "$(realdir "$CONDA_PREFIX")"/*) ;;
        *) ss_die "verl resolves to $P_VERL_DIR, outside this env ($CONDA_PREFIX); not patching it" ;;
    esac
    INPLACE_SITE=$(dirname "$P_VERL_DIR")
    case "$(patch_state "$INPLACE_SITE")" in
        applied) ss_log "the patch is already applied in $INPLACE_SITE (an earlier run got this far); nothing to do" ;;
        pristine)
            RERUN_NOTE="patching $INPLACE_SITE was interrupted; if a re-run reports 'neither applies', restore with: conda run -n $ENV_NAME python -m pip install --force-reinstall --no-deps verl==$VERL_VERSION"
            apply_patch "$INPLACE_SITE" || ss_die "patch failed in $INPLACE_SITE (output above)"
            [ "$(patch_state "$INPLACE_SITE")" = applied ] || ss_die "after patching, $INPLACE_SITE does not show the patch as applied"
            ss_log "patch applied in $INPLACE_SITE"
            ;;
        *)
            describe_patch_files "$INPLACE_SITE" | indent
            ss_die "the patch neither applies nor is already applied in $INPLACE_SITE; restore with: conda run -n $ENV_NAME python -m pip install --force-reinstall --no-deps verl==$VERL_VERSION"
            ;;
    esac
    EXPECT_VERL_DIR="$P_VERL_DIR"
    RERUN_NOTE="'$ENV_NAME' has the patched verl in place; a re-run skips the patch"
else
    # whatever the checkout state (the plan promises it): a leftover of an interrupted clone is never used
    if [ -e "$VERL_PARTIAL" ]; then
        ss_log "removing $VERL_PARTIAL (left over from an interrupted clone)"
        rm -rf "$VERL_PARTIAL"
    fi
    if [ "$CHECKOUT_STATE" = missing ] || [ "$CHECKOUT_STATE" = empty ]; then
        ss_step "clone verl $VERL_TAG into $VERL_SRC"
        RERUN_NOTE="the clone into $VERL_PARTIAL failed; a re-run removes it and clones again"
        if ! ss_run git clone --depth 1 --branch "$VERL_TAG" "$VERL_GIT_URL" "$VERL_PARTIAL"; then
            rm -rf "$VERL_PARTIAL"
            ss_die "git clone failed (output above); $VERL_PARTIAL was removed. No network? VERL_IN_PLACE=1 bash integration/setup_env.sh $ENV_NAME"
        fi
        PARTIAL_VERSION=$(cat "$VERL_PARTIAL/verl/version/version" 2>/dev/null || echo missing)
        if [ "$PARTIAL_VERSION" != "$VERL_VERSION" ]; then
            ss_die "the clone in $VERL_PARTIAL has verl/version/version '$PARTIAL_VERSION', not $VERL_VERSION (left in place for inspection; a re-run removes it)"
        fi
        if [ "$(file_sha256 "$VERL_PARTIAL/$FSDP_REL")" = "$SHA_FSDP_WORKERS" ] && [ "$(file_sha256 "$VERL_PARTIAL/$TRAINER_REL")" = "$SHA_RAY_TRAINER" ]; then
            ss_log "the two files the patch changes are byte-identical to the verl 0.5.0 wheel"
        else
            ss_warn "the two files the patch changes differ from the verl 0.5.0 wheel the patch was made against (the patch step decides whether it still applies):"
            describe_patch_files "$VERL_PARTIAL" | indent
        fi
        [ -d "$VERL_SRC" ] && rmdir "$VERL_SRC"
        mv "$VERL_PARTIAL" "$VERL_SRC"
        ss_log "checkout complete: $VERL_SRC ($(git -C "$VERL_SRC" describe --tags --always 2>/dev/null || echo '?'))"
    fi

    ss_step "apply the patch in $VERL_SRC"
    case "$(patch_state "$VERL_SRC")" in
        applied) ss_log "the patch is already applied in $VERL_SRC (an earlier run got this far); nothing to do" ;;
        pristine)
            RERUN_NOTE="patching $VERL_SRC was interrupted; if a re-run reports 'neither applies': rm -rf '$VERL_SRC' and re-run"
            apply_patch "$VERL_SRC" || ss_die "patch failed in $VERL_SRC (output above)"
            [ "$(patch_state "$VERL_SRC")" = applied ] || ss_die "after patching, $VERL_SRC does not show the patch as applied"
            ss_log "patch applied in $VERL_SRC"
            ;;
        *)
            describe_patch_files "$VERL_SRC" | indent
            ss_die "the patch neither applies nor is already applied in $VERL_SRC; rm -rf '$VERL_SRC' and re-run"
            ;;
    esac
    RERUN_NOTE="$VERL_SRC is cloned and patched; '$ENV_NAME' still has its previous verl; a re-run skips to the install"

    ss_step "install the patched checkout into '$ENV_NAME'"
    probe_python "$PYTHON"
    if [ "$P_VERL_DIR" = "$(realdir "$VERL_SRC")/verl" ] && [ "$P_VERL_VERSION" = "$VERL_VERSION" ]; then
        ss_log "'$ENV_NAME' already imports verl $P_VERL_VERSION from $VERL_SRC/verl; nothing to install"
    else
        if ! pip_target "$PYTHON"; then
            indent <"$TMP_DIR/pip_target.out"
            ss_die "pip in '$ENV_NAME' would not install into $CONDA_PREFIX (target: ${SS_PIP_TARGET:-pip failed, output above}); nothing was installed"
        fi
        ss_log "pip installs into $SS_PIP_TARGET (inside $CONDA_PREFIX)"
        ss_log "'$ENV_NAME' now has verl $P_VERL_VERSION at $P_VERL_DIR; installing $VERL_SRC editable over it"
        # No `pip uninstall` first: pip builds the editable wheel before it removes the installed verl
        # and rolls the removal back if the install fails, so a failed build leaves the env as it was.
        # shellcheck disable=SC2086  # PIP_FLAGS is a list of flags
        if ! ss_run python -m pip install ${PIP_FLAGS:---no-deps} -e "$VERL_SRC"; then
            probe_python "$PYTHON"
            ss_die "pip install failed (output above). '$ENV_NAME' now imports verl $P_VERL_VERSION from $P_VERL_DIR (pip restores the previous verl on failure). $VERL_SRC stays patched; a re-run repeats only the install"
        fi
    fi
    EXPECT_VERL_DIR="$(realdir "$VERL_SRC")/verl"
    RERUN_NOTE="'$ENV_NAME' imports the patched verl from $VERL_SRC; a re-run skips clone, patch and install"
fi

# ================================================================ 5. baseline untouched
ss_step "check that '$BASE_ENV' is unchanged"
hash_baseline
BASELINE_HASH_AFTER=$BASELINE_HASH
ss_log "what is hashed: $BASELINE_HASH_WHAT"
ss_log "before: $BASELINE_HASH_BEFORE"
ss_log "after:  $BASELINE_HASH_AFTER"
if [ "$BASELINE_HASH_BEFORE" != "$BASELINE_HASH_AFTER" ]; then
    RERUN_NOTE="the BASELINE env changed while this script ran: stop and tell the team before running anything else"
    ss_die "the verl files or the site-packages of '$BASE_ENV' changed while this script ran (before $BASELINE_HASH_BEFORE, after $BASELINE_HASH_AFTER). Stop and tell the team."
fi
ss_log "'$BASE_ENV' is unchanged"

# ================================================================ 6. self-test
ss_step "self-test in '$ENV_NAME'"
cat >"$TMP_DIR/selftest.py" <<'EOF'
import inspect
import os
import sys
import traceback

problems = []


def first_line(exc):
    lines = str(exc).strip().splitlines()
    text = f"{type(exc).__name__}: {lines[0]}" if lines else type(exc).__name__
    here = os.path.abspath(__file__)
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if not f.filename.startswith("<") and os.path.abspath(f.filename) != here]
    if frames:
        text += f" (at {frames[-1].filename}:{frames[-1].lineno})"
    return text


def line(label, text):
    print(f"[selftest] {label:<22} {text}", flush=True)


line("python", f"{sys.version.split()[0]} at {sys.executable}")
expected = os.environ.get("SS_EXPECT_VERL_DIR", "")
verl = None
try:
    import verl

    where = os.path.dirname(os.path.realpath(verl.__file__))
    note = ""
    if expected and os.path.realpath(expected) != where:
        problems.append(f"verl imports from {where}, expected {expected}")
        note = f"  <-- EXPECTED {expected}"
    line("verl", f"{verl.__version__} at {where}{note}")
    if verl.__version__ != "0.5.0":
        problems.append(f"verl is {verl.__version__}, not 0.5.0")
except Exception as exc:  # noqa: BLE001
    line("verl", "NOT importable: " + first_line(exc))
    problems.append("verl does not import")

if verl is not None:
    for label, module, check, what in (
        ("patch hunk 1 (worker)", "verl.workers.fsdp_workers",
         lambda m: '"sparsespec"' in inspect.getsource(m.ActorRolloutRefWorker._build_rollout),
         "fsdp_workers.py: rollout name 'sparsespec' in ActorRolloutRefWorker._build_rollout"),
        ("patch hunk 2 (trainer)", "verl.trainer.ppo.ray_trainer",
         lambda m: 'startswith("sparsespec/")' in inspect.getsource(m),
         "ray_trainer.py: sparsespec/* timing entries become step metrics"),
    ):  # fmt: skip
        try:
            __import__(module)
            present = check(sys.modules[module])
            line(label, ("present" if present else "MISSING") + f": {what}")
            if not present:
                problems.append(f"{label} missing")
        except Exception as exc:  # noqa: BLE001
            line(label, f"cannot check ({module} does not import): {first_line(exc)}")
            problems.append(f"{module} does not import")

try:
    import torch

    cuda = torch.version.cuda or "none"
    try:
        visible = "yes" if torch.cuda.is_available() else "no"
    except Exception:  # noqa: BLE001
        visible = "error"
    line("torch", f"{torch.__version__} (built for CUDA {cuda}; GPU visible: {visible})")
except Exception as exc:  # noqa: BLE001
    line("torch", "NOT importable: " + first_line(exc))
    problems.append("torch does not import")
for name, required in (("transformers", True), ("vllm", False), ("flashinfer", False)):
    try:
        __import__(name)
        line(name, str(getattr(sys.modules[name], "__version__", "?")))
    except Exception as exc:  # noqa: BLE001
        line(name, "NOT importable: " + first_line(exc))
        if required:
            problems.append(f"{name} does not import")

try:
    __import__("serve.rl.verl_rollout")
    line("serve.rl.verl_rollout", "imports (rollout class + sharding manager; engine=reference works)")
except Exception as exc:  # noqa: BLE001
    line("serve.rl.verl_rollout", "does NOT import: " + first_line(exc))
    problems.append("serve.rl.verl_rollout does not import")
try:
    __import__("serve.rl.engine")
    line("serve.rl.engine", "imports: engine=sparsespec can run in this env")
    engine = "yes"
except Exception as exc:  # noqa: BLE001
    line("serve.rl.engine", "does NOT import: " + first_line(exc))
    line("", "-> only engine=reference works in this env until Harrison's build is installed in it")
    engine = "no"

for problem in problems:
    line("PROBLEM", problem)
print(f"selftest_engine={engine}")
sys.exit(1 if problems else 0)
EOF
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if SS_EXPECT_VERL_DIR="$EXPECT_VERL_DIR" python -B "$TMP_DIR/selftest.py" >"$TMP_DIR/selftest.out" 2>&1; then
    SELFTEST_STATUS=0
else
    SELFTEST_STATUS=$?
fi
grep -v '^selftest_engine=' "$TMP_DIR/selftest.out" || true
ENGINE_OK=$(sed -n 's/^selftest_engine=//p' "$TMP_DIR/selftest.out")
if [ "$SELFTEST_STATUS" -ne 0 ]; then
    RERUN_NOTE="'$ENV_NAME' has the patched verl, but the self-test found the problems listed above"
    ss_die "self-test failed (exit status $SELFTEST_STATUS): see the PROBLEM lines above"
fi

FINAL_DONE=1
if [ "$ENGINE_OK" = yes ]; then
    ENGINE_TEXT="SparseSpec engine importable (engine=sparsespec and reference)"
else
    ENGINE_TEXT="SparseSpec engine NOT importable (engine=reference only)"
fi
ss_done "'$ENV_NAME' has patched verl $VERL_VERSION; $ENGINE_TEXT; '$BASE_ENV' unchanged. Activate with: conda activate $ENV_NAME"
