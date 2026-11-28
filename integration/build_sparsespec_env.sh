#!/usr/bin/env bash
# Build the conda env `verl-sparsespec` for the SparseSpec arm. Run on the A100 Linux box.
#
#   bash integration/build_sparsespec_env.sh --check     # read-only: prerequisites + the plan, changes nothing
#   bash integration/build_sparsespec_env.sh             # build (every step is restartable: just rerun it)
#   bash integration/build_sparsespec_env.sh --verify    # only the verification + package-drift report
#   bash integration/build_sparsespec_env.sh --help
#
# What the env is: a clone of verl-baseline (stock vLLM 0.9.1 + verl 0.5.0 + torch 2.7.0) plus
#   - the FlashInfer fork in 3rdparty/flashinfer (README "install customized flashinfer"),
#   - RAFT headers and rmm (librmm.so) built from 3rdparty/raft into the env prefix
#     (README "install raft header-only lib"; serve/attention/backend.py looks for them under
#     $CONDA_PREFIX/include and $CONDA_PREFIX/lib),
#   - this repo, `pip install --no-deps -e .` (why --no-deps: see step "install the repo"),
#   - pytest (+ iniconfig, pluggy, and only what `import pytest` still misses; --no-deps) if the
#     clone has none: integration/run_all_gates.sh runs the CPU and GPU test stages with it.
# NOT the vLLM fork (3rdparty/vllm): serve/ only imports vllm._custom_ops
# (serve/distribute/custom_all_reduce.py), which stock vLLM 0.9.1 has, and the fork pins torch 2.8.0,
# which would make the training path's torch differ between the two arms.
# The patched verl is installed afterwards by `bash integration/setup_env.sh verl-sparsespec`.
#
# Guarantees: verl-baseline is only read (never cloned onto, never pip-installed into; its package
# lists are fingerprinted before and after as a tripwire). Every pip install into the clone uses
# --no-deps, so no package already in the clone can change; a drift report against verl-baseline
# checks that at the end and FAILS if torch, transformers, verl, vllm, numpy or tensordict differ.
#
# Options (environment variables):
#   ENV_NAME=verl-sparsespec    target env (also the first positional argument)
#   BASE_ENV=verl-baseline      env to clone; never modified
#   BUILD_DIR=<repo>/integration/build/<env>   build tree, tools, reports (git-ignored via `build/`)
#   PARALLEL_LEVEL=<n>          parallel compile jobs for rmm/RAFT (default: min(nproc, 16); shared box)
#   FLASHINFER_EDITABLE=1       1: `pip install -e` as the README does; 0: regular install (copies
#                               the fork's headers into the env; self-contained, see packaging notes)
#   SPECGEN_INSTALL=editable    editable: `pip install --no-deps -e .`; none: skip (PYTHONPATH only)
#   SKIP_JIT_SMOKE=1            skip JIT-compiling the RAFT top-k op during verification
#   STRICT_DRIFT=1              also fail on unexpected changes of packages outside the critical six
#   CLONE_COPY=1                clone with `--copy` (no hard links into the package cache; default 1)
#   ALLOW_DIRTY_SUBMODULES=1    proceed although a submodule has modified tracked files
#   MIN_FREE_GB_ENVS / MIN_FREE_GB_BUILD / MIN_FREE_GB_HOME   disk thresholds (default 25 / 10 / 5)
#   SKIP_DU=1                   do not measure the base env size (du can be slow on network disks)
#   SKIP_NETWORK_CHECK=1        do not probe PyPI / GitHub in the prerequisite check
#   SS_TRACE=1                  xtrace into the log (integration/lib.sh)
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
# shellcheck source=integration/lib.sh
source "$REPO_ROOT/integration/lib.sh"

MODE=build
POS_ENV=""
for arg in "$@"; do
    case "$arg" in
        --check) MODE=check ;;
        --verify) MODE=verify ;;
        -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) echo "unknown option: $arg (see --help)" >&2; exit 2 ;;
        *) if [ -n "$POS_ENV" ]; then echo "only one env name may be given" >&2; exit 2; fi; POS_ENV=$arg ;;
    esac
done

ENV_NAME=${POS_ENV:-${ENV_NAME:-verl-sparsespec}}
BASE_ENV=${BASE_ENV:-verl-baseline}
BUILD_DIR=${BUILD_DIR:-$REPO_ROOT/integration/build/$ENV_NAME}
FLASHINFER_EDITABLE=${FLASHINFER_EDITABLE:-1}
SPECGEN_INSTALL=${SPECGEN_INSTALL:-editable}
SKIP_JIT_SMOKE=${SKIP_JIT_SMOKE:-0}
STRICT_DRIFT=${STRICT_DRIFT:-0}
CLONE_COPY=${CLONE_COPY:-1}
ALLOW_DIRTY_SUBMODULES=${ALLOW_DIRTY_SUBMODULES:-0}
MIN_FREE_GB_ENVS=${MIN_FREE_GB_ENVS:-25}
MIN_FREE_GB_BUILD=${MIN_FREE_GB_BUILD:-10}
MIN_FREE_GB_HOME=${MIN_FREE_GB_HOME:-5}

FI_DIR="$REPO_ROOT/3rdparty/flashinfer"
RAFT_DIR="$REPO_ROOT/3rdparty/raft"
REPORT_DIR="$BUILD_DIR/report"
TOOLS_DIR="$BUILD_DIR/tools"             # venv with cmake (and conda-pack); never a conda env
RAFT_BUILD_DIR="$BUILD_DIR/raft-build"   # LIBRAFT_BUILD_DIR (raft build.sh:56)
RAFT_STAGE_DIR="$BUILD_DIR/raft-stage"   # dry-run install to see what would land in the env
CPM_CACHE_DIR="$BUILD_DIR/cpm-cache"     # rapids-cmake / CPM downloads survive a rerun

# The six packages whose versions must be identical in both arms (training path).
CRITICAL_PKGS="torch transformers verl vllm numpy tensordict"
# Packages this script itself may add or change in the clone (everything else must stay equal).
# (flashinfer: the old name of a stock FlashInfer the clone may have had, removed in step e;
# pytest and its runtime deps: step "pytest", plus any module it names in pytest_extra.txt)
EXPECTED_PKGS="flashinfer-python flashinfer specgen setuptools ninja pytest iniconfig pluggy pygments"
PYTEST_PKGS="pytest iniconfig pluggy"
CMAKE_MIN=3.30.4          # 3rdparty/raft/cpp/CMakeLists.txt:13, cmake/RAPIDS.cmake (FATAL_ERROR)
SETUPTOOLS_MIN=77         # flashinfer pyproject: build-system requires setuptools>=77 (PEP 639 license)
SETUPTOOLS_SPEC="setuptools>=77.0.3,<80"   # the range vLLM 0.9.1 itself pins (for Python > 3.11)
GCC_MIN=9.3               # 3rdparty/raft/docs/source/build.md "GCC 9.3+"

# Nothing here should be steered by the caller's Python environment.
if [ -n "${PYTHONPATH:-}" ] || [ -n "${PYTHONHOME:-}" ] || [ -n "${PIP_TARGET:-}" ] || [ -n "${PIP_PREFIX:-}" ]; then
    echo "[build_sparsespec_env] unsetting PYTHONPATH/PYTHONHOME/PIP_TARGET/PIP_PREFIX of the calling shell (set explicitly where needed)"
fi
# PIP_TARGET / PIP_PREFIX (pip's --target / --prefix) would redirect every pip install of this build
unset PYTHONPATH PYTHONHOME PIP_TARGET PIP_PREFIX
# PIP_USER=0: a pip config with `user = true` (or PIP_USER=1, or a site-packages pip cannot write to)
# would send installs into ~/.local, which verl-baseline imports too, and would cut `pip list` down to
# user-site packages. The variable overrides every pip config file; `conda run` passes it on.
# PYTHONNOUSERSITE is deliberately NOT set: it does not stop user installs, and it would hide the
# user site from the probes and the tripwire while the training runs still import from it.
# PIP_REQUIRE_VIRTUALENV=0: a conda env is not a virtualenv, so a pip config with
# `require-virtualenv = true` would refuse every install here; where pip writes is checked below.
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1 PIP_USER=0 PIP_REQUIRE_VIRTUALENV=0

ss_init build_sparsespec_env
ss_log "mode=$MODE env=$ENV_NAME base=$BASE_ENV repo=$REPO_ROOT"
ss_log "build dir: $BUILD_DIR"

PY_DIR=$(mktemp -d "${TMPDIR:-/tmp}/ss_build_env.XXXXXX")
cleanup() { rm -rf "$PY_DIR"; }
trap cleanup EXIT

# ---------------------------------------------------------------------------------------------
# small helpers (bash 3.2: no associative arrays, no mapfile, no ${x,,})
# ---------------------------------------------------------------------------------------------
CK_BLOCK=0
CK_WARN=0
CK_BLOCKERS=""
REFUSED=0
ck_ok()   { printf '[%s]   ok    %s\n' "$SS_SCRIPT" "$*"; }
ck_info() { printf '[%s]   info  %s\n' "$SS_SCRIPT" "$*"; }
ck_warn() { CK_WARN=$((CK_WARN + 1)); printf '[%s]   WARN  %s\n' "$SS_SCRIPT" "$*"; }
ck_fail() {
    CK_BLOCK=$((CK_BLOCK + 1))
    CK_BLOCKERS="$CK_BLOCKERS
    - $*"
    printf '[%s]   FAIL  %s\n' "$SS_SCRIPT" "$*"
}

# ver_ge A B: A >= B, comparing dot-separated numeric fields (suffixes such as "+cu126" ignored)
ver_ge() {
    local a b x y
    a=$(printf '%s' "$1" | sed 's/[^0-9.].*$//')
    b=$(printf '%s' "$2" | sed 's/[^0-9.].*$//')
    while [ -n "$a" ] || [ -n "$b" ]; do
        x=${a%%.*}; y=${b%%.*}
        x=${x:-0}; y=${y:-0}
        if [ "$x" -gt "$y" ]; then return 0; fi
        if [ "$x" -lt "$y" ]; then return 1; fi
        case "$a" in *.*) a=${a#*.} ;; *) a="" ;; esac
        case "$b" in *.*) b=${b#*.} ;; *) b="" ;; esac
    done
    return 0
}
major_of() { printf '%s' "$1" | sed 's/[^0-9].*$//'; }

kv() {  # kv KEY FILE: value of the first KEY=... line
    awk -v k="$1" 'index($0, k "=") == 1 { print substr($0, length(k) + 2); exit }' "$2"
}

realdir() { (cd "$1" 2>/dev/null && pwd -P) || printf '%s' "$1"; }

free_gb() {  # free GB on the filesystem holding $1 (or its nearest existing parent)
    local d=$1
    while [ ! -d "$d" ] && [ "$d" != "/" ]; do d=$(dirname "$d"); done
    df -Pk "$d" 2>/dev/null | awk 'NR == 2 { printf "%d", $4 / 1048576 }'
}

nproc_n() { nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4; }

# ---------------------------------------------------------------------------------------------
# conda
# ---------------------------------------------------------------------------------------------
CONDA_BIN=${CONDA_EXE:-}
if [ -z "$CONDA_BIN" ] || [ ! -x "$CONDA_BIN" ]; then
    CONDA_BIN=$(command -v conda 2>/dev/null || true)
fi

env_prefix() {  # prefix of a named env, empty if it does not exist
    "$CONDA_BIN" env list 2>/dev/null | awk -v n="$1" '$1 == n { print $NF }'
}
# Run a command inside an env (activation scripts applied; CONDA_PREFIX set), echoed first.
env_run() {
    local env=$1
    shift
    ss_run "$CONDA_BIN" run --no-capture-output -n "$env" "$@"
}
# Same, quiet (for probes whose output is captured).
env_q() {
    local env=$1
    shift
    "$CONDA_BIN" run --no-capture-output -n "$env" "$@"
}

# ---------------------------------------------------------------------------------------------
# Python helpers (written to a temp dir; run with the env's own interpreter via `conda run`)
# ---------------------------------------------------------------------------------------------
write_helpers() {
cat > "$PY_DIR/env_probe.py" <<'EOF'
# ss-helper: env_probe. Read-only: package metadata and files only; imports nothing heavy.
import importlib.metadata as md
import importlib.util
import json
import os
import re
import shutil
import site
import sys

def ver(name):
    try:
        return md.version(name)
    except Exception:
        return ""

def direct_url(name):
    try:
        text = md.distribution(name).read_text("direct_url.json")
    except Exception:
        return ""
    if not text:
        return ""
    d = json.loads(text)
    return ("editable:" if d.get("dir_info", {}).get("editable") else "") + d.get("url", "")

print("python=%d.%d.%d" % sys.version_info[:3])
print("sys_prefix=" + os.path.realpath(sys.prefix))
print("conda_prefix=" + os.environ.get("CONDA_PREFIX", ""))
for n in ("torch", "transformers", "verl", "vllm", "numpy", "tensordict", "triton", "setuptools",
          "packaging", "pip", "ninja", "numba", "jinja2", "wandb", "flashinfer-python", "flashinfer",
          "specgen", "datasets", "antlr4-python3-runtime", "hydra-core", "cmake", "pytest"):
    print("ver.%s=%s" % (n, ver(n)))
for n in ("flashinfer-python", "specgen", "verl", "vllm"):
    print("url.%s=%s" % (n, direct_url(n)))
# torch.version.cuda without importing torch
cuda = ""
spec = importlib.util.find_spec("torch")
if spec and spec.submodule_search_locations:
    vf = os.path.join(list(spec.submodule_search_locations)[0], "version.py")
    if os.path.exists(vf):
        m = re.search(r"^cuda\b[^=\n]*=\s*['\"]([^'\"]*)['\"]", open(vf).read(), re.M)
        cuda = m.group(1) if m else "none"
print("torch_cuda=" + cuda)
# CUDA_HOME exactly as torch.utils.cpp_extension._find_cuda_home (torch 2.7) resolves it; the
# FlashInfer JIT uses that module's CUDA_HOME (flashinfer/jit/cpp_ext.py:15,143)
how = "CUDA_HOME/CUDA_PATH"
home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
if not home:
    nvcc = shutil.which("nvcc")
    if nvcc:
        home, how = os.path.dirname(os.path.dirname(nvcc)), "nvcc on PATH"
    elif os.path.exists("/usr/local/cuda"):
        home, how = "/usr/local/cuda", "/usr/local/cuda default"
    else:
        home, how = "", "not found"
print("cuda_home=" + home)
print("cuda_home_how=" + how)
print("which_ninja=" + (shutil.which("ninja") or ""))
print("which_cmake=" + (shutil.which("cmake") or ""))
s = importlib.util.find_spec("flashinfer")
print("flashinfer_origin=" + (s.origin or "" if s else ""))
us = site.getusersitepackages()
n_us = len(os.listdir(us)) if os.path.isdir(us) else 0
print("user_site=%s" % os.path.realpath(us))
# where each package the build may replace resolves (pip uninstalls a replaced package wherever it
# lives, the user site included, when not in a virtualenv)
for n in ("setuptools", "flashinfer-python", "flashinfer", "specgen"):
    try:
        print("loc.%s=%s" % (n, os.path.realpath(str(md.distribution(n).locate_file("")))))
    except Exception:
        print("loc.%s=" % n)
print("user_site_entries=%d" % (n_us if site.ENABLE_USER_SITE else 0))
print("env.INSTALL_PREFIX=" + os.environ.get("INSTALL_PREFIX", ""))
print("env.PREFIX=" + os.environ.get("PREFIX", ""))
EOF

cat > "$PY_DIR/drift.py" <<'EOF'
# ss-helper: drift. Compare two `pip list --format=freeze` files; pure logic.
# usage: drift.py BASE NEW CRITICAL(csv) EXPECTED(csv) OUT_MD
# exit 0: critical packages equal; 3: a critical package differs or is missing
import re
import sys

def norm(n):
    return re.sub(r"[-_.]+", "-", n).lower()

def load(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, _, v = line.partition("==")
        out[norm(name)] = v.strip()
    return out

base, new = load(sys.argv[1]), load(sys.argv[2])
critical = [norm(x) for x in sys.argv[3].split(",") if x]
expected = set(norm(x) for x in sys.argv[4].split(",") if x)
rows, bad = [], []
for n in critical:
    b, w = base.get(n, "MISSING"), new.get(n, "MISSING")
    ok = b == w and b != "MISSING"
    rows.append(("CRITICAL", n, b, w, "same" if ok else "DIFFERENT"))
    if not ok:
        bad.append(n)
others = sorted((set(base) | set(new)) - set(critical))
diffs = []
for n in others:
    b, w = base.get(n, "-"), new.get(n, "-")
    if b != w:
        kind = "expected (this script)" if n in expected else "UNEXPECTED"
        diffs.append((kind, n, b, w))
unexpected = [d for d in diffs if d[0] == "UNEXPECTED"]
lines = ["# Package drift: base vs new env", "",
         "base: `%s`  new: `%s`" % (sys.argv[1], sys.argv[2]), "",
         "| check | package | base | new | result |", "|---|---|---|---|---|"]
lines += ["| %s | %s | %s | %s | %s |" % r for r in rows]
lines += ["", "%d other difference(s), %d unexpected:" % (len(diffs), len(unexpected)), "",
          "| kind | package | base | new |", "|---|---|---|---|"]
lines += ["| %s | %s | %s | %s |" % d for d in diffs]
lines += ["", "packages: base %d, new %d" % (len(base), len(new))]
text = "\n".join(lines) + "\n"
open(sys.argv[5], "w").write(text)
print(text)
print("drift: critical_bad=%d unexpected=%d" % (len(bad), len(unexpected)))
sys.exit(3 if bad else 0)
EOF

cat > "$PY_DIR/verify_import.py" <<'EOF'
# ss-helper: verify_import. Imports flashinfer and serve.rl.engine (needs the real env).
# usage: verify_import.py REPO_ROOT FLASHINFER_SRC_DIR EXPECT_FLASHINFER_IN_SRC(0/1)
import os
import sys
import time

repo, fi_src, fi_in_src = os.path.realpath(sys.argv[1]), os.path.realpath(sys.argv[2]), sys.argv[3] == "1"
t0 = time.time()
import torch
print("torch %s cuda %s at %s" % (torch.__version__, torch.version.cuda, torch.__file__))
import triton
print("triton %s" % triton.__version__)
import flashinfer
print("flashinfer %s at %s" % (flashinfer.__version__, flashinfer.__file__))
import serve.rl.engine as engine_mod
print("serve.rl.engine at %s" % engine_mod.__file__)
import serve
print("serve.__path__ = %s" % list(serve.__path__))
problems = []
if not os.path.realpath(engine_mod.__file__).startswith(repo + os.sep):
    problems.append("serve.rl.engine was imported from outside %s" % repo)
fi_file = os.path.realpath(flashinfer.__file__)
if fi_in_src and not fi_file.startswith(fi_src + os.sep):
    problems.append("flashinfer imported from %s, expected the editable fork in %s" % (fi_file, fi_src))
if not fi_in_src and fi_file.startswith(fi_src + os.sep):
    problems.append("flashinfer imported from the source tree although FLASHINFER_EDITABLE=0")
first = list(serve.__path__)[0]
if os.path.realpath(first) != os.path.join(repo, "serve"):
    problems.append("first serve/ portion is %s, not this repo (another checkout or install shadows it)" % first)
print("imports took %.1f s" % (time.time() - t0))
for p in problems:
    print("PROBLEM: " + p)
sys.exit(1 if problems else 0)
EOF

cat > "$PY_DIR/raft_check.py" <<'EOF'
# ss-helper: raft_check. The prefix lookup of serve/attention/backend.py:304-317, then the files.
# usage: raft_check.py (environment decides, like the engine)
import glob
import os
import sys

for var in ("INSTALL_PREFIX", "PREFIX", "CONDA_PREFIX"):
    if os.environ.get(var):
        prefix, how = os.environ[var], var
        break
else:
    print("PROBLEM: none of INSTALL_PREFIX / PREFIX / CONDA_PREFIX is set (backend.py asserts)")
    sys.exit(1)
print("backend.py would use %s=%s" % (how, prefix))
if how != "CONDA_PREFIX":
    print("NOTE: %s is set in this shell and wins over CONDA_PREFIX (backend.py:305-307)" % how)
need = {
    "RAFT select_k header (serve/attention/csrc/batch_topk.cuh:7)": "include/raft/matrix/detail/select_k.cuh",
    "rmm headers": "include/rmm",
    "librmm.so (backend.py:316)": "lib/librmm.so",
}
bad = 0
for what, rel in need.items():
    p = os.path.join(prefix, rel)
    ok = os.path.exists(p)
    bad += not ok
    print("%-4s %s: %s" % ("ok" if ok else "MISSING", what, p))
for p in sorted(glob.glob(os.path.join(prefix, "lib", "librapids_logger*.so*"))):
    print("info rapids_logger library: %s" % p)
cccl = os.path.join(prefix, "include", "rapids")
print("info CCCL installed by rapids-cmake under %s: %s (the JIT adds only %s/include)"
      % (cccl, os.path.isdir(cccl), prefix))
sys.exit(1 if bad else 0)
EOF

cat > "$PY_DIR/jit_smoke.py" <<'EOF'
# ss-helper: jit_smoke. JIT-compiles and loads the RAFT top-k op exactly as BatchTopKWrapper does
# (serve/attention/backend.py:364). Needs nvcc and a visible GPU (arch detection); no GPU memory.
import os
import subprocess
import sys
import time
import traceback

from flashinfer.jit import env as jit_env

print("FlashInfer JIT cache: %s" % jit_env.FLASHINFER_JIT_DIR)
from serve.attention.backend import gen_topk_op_module

spec = gen_topk_op_module("batch_topk_csr")
so = spec.jit_library_path
t0 = time.time()
try:
    mod = spec.build_and_load()
except Exception:
    traceback.print_exc()
    print("PROBLEM: building or loading batch_topk_csr failed after %.0f s" % (time.time() - t0))
    print("ninja file: %s" % spec.ninja_path)
    if os.path.exists(so):
        print("ldd %s:" % so)
        subprocess.call(["ldd", str(so)])
    sys.exit(1)
print("batch_topk_csr built and loaded in %.0f s: %s" % (time.time() - t0, so))
print("op: %s" % mod.batch_topk)
# The load above is the check. The JIT links with -L only (no rpath), so a plain ldd in a new process
# cannot see torch/lib or $PREFIX/lib; show it with those dirs added, for information only.
import torch
from torch.utils.cpp_extension import CUDA_HOME
prefix = os.environ.get("INSTALL_PREFIX") or os.environ.get("PREFIX") or os.environ.get("CONDA_PREFIX") or ""
dirs = [os.path.join(os.path.dirname(torch.__file__), "lib"), os.path.join(prefix, "lib")]
if CUDA_HOME:
    dirs.append(os.path.join(CUDA_HOME, "lib64"))
if os.environ.get("LD_LIBRARY_PATH"):
    dirs.append(os.environ["LD_LIBRARY_PATH"])
env = dict(os.environ, LD_LIBRARY_PATH=":".join(dirs))
print("ldd with LD_LIBRARY_PATH=%s (information only: the load above is the check)" % env["LD_LIBRARY_PATH"])
try:
    r = subprocess.run(["ldd", str(so)], capture_output=True, text=True, env=env)
    print(r.stdout + r.stderr)
except OSError as e:
    print("info: could not run ldd: %s" % e)
try:
    with open("/proc/self/maps") as f:
        mapped = sorted({l.split()[-1] for l in f if l.split()[-1].startswith("/")})
except OSError:
    mapped = []
for p in mapped:
    b = os.path.basename(p)
    if any(k in b for k in ("rmm", "rapids_logger", "c10", "torch", "cudart", "batch_topk")):
        print("mapped in this process: " + p)
sys.exit(0)
EOF

cat > "$PY_DIR/pip_target.py" <<'EOF'
# ss-helper: pip_target. Where would `python -m pip install` of THIS interpreter put a package?
# Installs a throwaway 2-file wheel with --root into a temp dir (nothing is written to the env), then
# reads back the site-packages dir pip chose. A pip config `user = true`, PIP_USER=1 or a
# site-packages pip cannot write to shows up as a target outside sys.prefix (exit 2); pip failing
# (e.g. a broken pip install) exits 1 with pip's message.
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
print("PIP_USER=" + os.environ.get("PIP_USER", "<unset>"))
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
cfg = subprocess.run([sys.executable, "-m", "pip", "config", "list"], capture_output=True, text=True)
for line in cfg.stdout.splitlines():
    if line.split("=", 1)[0].strip().endswith(".user"):
        print("pip_config_user=" + line.strip())
r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", "--no-compile",
                    "--no-warn-script-location", "--root", root, whl], capture_output=True, text=True)
found = ""
for d, _, names in os.walk(root):
    if os.path.basename(d) == NAME and "__init__.py" in names:
        found = "/" + os.path.relpath(os.path.dirname(d), root)
        break
if r.returncode != 0 or not found:
    print("pip_rc=%d" % r.returncode)
    for line in (r.stdout + r.stderr).strip().splitlines()[-8:]:
        print("pip| " + line)
    print("target=")
    sys.exit(1)
target = os.path.realpath(found) if os.path.isdir(found) else found
print("target=" + target)
sys.exit(0 if (target + os.sep).startswith(prefix + os.sep) else 2)
EOF

cat > "$PY_DIR/pytest_check.py" <<'EOF'
# ss-helper: pytest_check. Does `import pytest` work? Prints missing_module=<name> when an import fails.
import sys
import traceback

try:
    import pytest
except ModuleNotFoundError as e:
    traceback.print_exc()
    print("missing_module=" + (e.name or "").split(".")[0])
    sys.exit(1)
except Exception:
    traceback.print_exc()
    print("missing_module=")
    sys.exit(1)
print("pytest %s at %s" % (pytest.__version__, pytest.__file__))
EOF

cat > "$PY_DIR/with_path.sh" <<'EOF'
#!/usr/bin/env bash
# with_path.sh DIR CMD...: prepend DIR to PATH (after conda's activation) and run CMD
PATH="$1:$PATH"
export PATH
shift
exec "$@"
EOF
}
write_helpers

# ---------------------------------------------------------------------------------------------
# state discovered by the checks (also used by the build steps)
# ---------------------------------------------------------------------------------------------
BASE_PREFIX=""
ENV_PREFIX=""
ENV_EXISTS=0
BASE_PROBE="$PY_DIR/base.probe"
ENV_PROBE="$PY_DIR/env.probe"
NEED_SUBMODULE_INIT=""        # list of submodule paths to init
CMAKE_EXE=""                  # cmake to use (empty: install into the tools venv)
NEED_SETUPTOOLS=0
NEED_NINJA=0
NEED_PYTEST=0
CUDA_HOME_FOUND=""
NVCC=""
NVCC_VER=""
GXX=""
# both files live in the submodules: re-read after they are initialised
read_submodule_facts() {
    FI_VERSION=$(cat "$FI_DIR/version.txt" 2>/dev/null || echo "?")
    RAPIDS_BRANCH=$(head -1 "$RAFT_DIR/RAPIDS_BRANCH" 2>/dev/null || echo "?")
}
read_submodule_facts

probe_env() {  # probe_env ENV OUTFILE: run env_probe.py inside the env (read-only)
    env_q "$1" env PYTHONDONTWRITEBYTECODE=1 python -B "$PY_DIR/env_probe.py" > "$2" 2>"$2.err"
}

# pip_target_check ENV: 0 when `python -m pip install` in ENV writes into ENV's own site-packages.
# Sets PIP_TARGET_MSG (one line for a FAIL / ok message); the full output is in $PY_DIR/pip_target.ENV.
PIP_TARGET_MSG=""
pip_target_check() {
    local out="$PY_DIR/pip_target.$1" rc=0 t
    env_q "$1" env PYTHONDONTWRITEBYTECODE=1 python -B "$PY_DIR/pip_target.py" > "$out" 2>&1 || rc=$?
    t=$(kv target "$out")
    case "$rc" in
        0) PIP_TARGET_MSG="pip in $1 installs into $t (PIP_USER=$(kv PIP_USER "$out"))" ;;
        2) PIP_TARGET_MSG="pip in $1 would install into $t, outside the env prefix $(kv sys_prefix "$out") (a pip 'user' setting or an unwritable site-packages; PIP_USER as pip sees it: $(kv PIP_USER "$out")). That would put packages into the user site, which $BASE_ENV imports too. Check: conda run -n $1 python -m pip config list; and that this shell does not override PIP_USER" ;;
        *) PIP_TARGET_MSG="a throwaway 'pip install --no-index --root <tmp>' of a local wheel failed in $1 (exit $rc; pip's output above), so every pip install of this build would fail too" ;;
    esac
    if [ "$rc" != 0 ]; then sed 's/^/            | /' "$out"; else grep '^pip_config_user=' "$out" | sed 's/^/            | /' || true; fi
    return "$rc"
}

submodule_state() {  # prints "<flag> <checked-out sha> <recorded sha>" for one submodule path
    local path=$1 line flag rec cur
    rec=$(git -C "$REPO_ROOT" ls-tree HEAD "$path" 2>/dev/null | awk '{ print $3 }')
    line=$(git -C "$REPO_ROOT" submodule status -- "$path" 2>/dev/null | head -1)
    flag=$(printf '%s' "$line" | cut -c1)
    if [ "$flag" = " " ]; then flag="="; fi
    cur=$(git -C "$REPO_ROOT/$path" rev-parse HEAD 2>/dev/null || echo none)
    printf '%s %s %s\n' "${flag:-?}" "$cur" "${rec:-none}"
}

check_submodule() {  # check_submodule PATH [nested-in]
    local path=$1 st flag cur rec dirty
    st=$(submodule_state "$path")
    flag=$(echo "$st" | awk '{print $1}'); cur=$(echo "$st" | awk '{print $2}'); rec=$(echo "$st" | awk '{print $3}')
    case "$flag" in
        "=")
            dirty=$(git -C "$REPO_ROOT/$path" status --porcelain --untracked-files=no 2>/dev/null | head -5)
            if [ -n "$dirty" ]; then
                if [ "$ALLOW_DIRTY_SUBMODULES" = 1 ]; then
                    ck_warn "$path at the recorded commit ${rec:0:12} but has modified tracked files (ALLOW_DIRTY_SUBMODULES=1):"
                else
                    ck_fail "$path has modified tracked files; the build would not be the recorded source. Inspect with: git -C $path status (ALLOW_DIRTY_SUBMODULES=1 to proceed)"
                fi
                printf '%s\n' "$dirty" | sed 's/^/            /'
            else
                ck_ok "$path initialised at the recorded commit ${rec:0:12}"
            fi ;;
        "-")
            ck_info "$path not initialised: the build will run git submodule update --init (recorded ${rec:0:12})"
            NEED_SUBMODULE_INIT="$NEED_SUBMODULE_INIT $path" ;;
        "+")
            ck_fail "$path is checked out at ${cur:0:12}, the superproject records ${rec:0:12}. Not touching it; fix with: git submodule update -- $path (after saving any work in it)" ;;
        "U")
            ck_fail "$path has merge conflicts (git submodule status shows U)" ;;
        *)
            ck_fail "$path: cannot read its submodule status (git submodule status -- $path)" ;;
    esac
}

run_checks() {
    local v base_py torch_cuda f
    ss_log "bash $BASH_VERSION, $(uname -s) $(uname -m)"

    # -- names -----------------------------------------------------------------------------
    if [ "$ENV_NAME" = "$BASE_ENV" ] || [ "$ENV_NAME" = "verl-baseline" ] || [ "$ENV_NAME" = "base" ] || [ -z "$ENV_NAME" ]; then
        ck_fail "refusing target env '$ENV_NAME': it must not be '$BASE_ENV', 'verl-baseline' or 'base'"
        REFUSED=1
        return 0
    fi
    ck_ok "target env name '$ENV_NAME' (base '$BASE_ENV' is only read)"

    # -- conda -----------------------------------------------------------------------------
    if [ -z "$CONDA_BIN" ]; then
        ck_fail "conda not found (neither \$CONDA_EXE nor 'conda' on PATH). Open a shell where 'conda' works."
        return 0
    fi
    v=$("$CONDA_BIN" --version 2>/dev/null | awk '{print $NF}') || v=""
    if [ -z "$v" ]; then
        ck_fail "'$CONDA_BIN --version' failed"
        return 0
    elif ver_ge "$v" 4.9; then
        ck_ok "conda $v ($CONDA_BIN); 'conda run --no-capture-output' available"
    else
        ck_fail "conda $v is older than 4.9: 'conda run --no-capture-output' is not available"
    fi

    BASE_PREFIX=$(env_prefix "$BASE_ENV")
    if [ -z "$BASE_PREFIX" ]; then
        ck_fail "base env '$BASE_ENV' not found in 'conda env list' (baseline/setup_env.sh creates it)"
        return 0
    fi
    ck_ok "base env $BASE_ENV at $BASE_PREFIX"
    if ! probe_env "$BASE_ENV" "$BASE_PROBE"; then
        ck_fail "cannot run python in $BASE_ENV: $(tail -3 "$BASE_PROBE.err" 2>/dev/null | tr '\n' ' ')"
        return 0
    fi
    base_py=$(kv python "$BASE_PROBE")
    case "$base_py" in
        3.11.*) ck_ok "$BASE_ENV: python $base_py" ;;
        *) ck_warn "$BASE_ENV: python $base_py (baseline/setup_env.sh:12 creates 3.11)" ;;
    esac
    for f in torch transformers verl vllm numpy tensordict; do
        v=$(kv "ver.$f" "$BASE_PROBE")
        if [ -z "$v" ]; then ck_fail "$BASE_ENV has no '$f' (expected a complete baseline env)"; fi
    done
    ck_info "$BASE_ENV: torch $(kv ver.torch "$BASE_PROBE") (CUDA $(kv torch_cuda "$BASE_PROBE")), vllm $(kv ver.vllm "$BASE_PROBE"), verl $(kv ver.verl "$BASE_PROBE"), transformers $(kv ver.transformers "$BASE_PROBE"), numpy $(kv ver.numpy "$BASE_PROBE"), tensordict $(kv ver.tensordict "$BASE_PROBE")"
    [ "$(kv ver.vllm "$BASE_PROBE")" = "0.9.1" ] || ck_warn "$BASE_ENV: vllm is '$(kv ver.vllm "$BASE_PROBE")', not 0.9.1 (baseline/setup_env.sh:19)"
    [ "$(kv ver.verl "$BASE_PROBE")" = "0.5.0" ] || ck_fail "$BASE_ENV: verl is '$(kv ver.verl "$BASE_PROBE")'; integration/setup_env.sh needs 0.5.0"
    case "$(kv url.verl "$BASE_PROBE")$(kv url.vllm "$BASE_PROBE")" in
        *editable:*) ck_warn "$BASE_ENV has an editable verl or vllm; the clone will point at the same source tree" ;;
    esac
    if [ -n "$(kv ver.flashinfer-python "$BASE_PROBE")$(kv ver.flashinfer "$BASE_PROBE")" ]; then
        ck_warn "$BASE_ENV has a flashinfer ($(kv ver.flashinfer-python "$BASE_PROBE")$(kv ver.flashinfer "$BASE_PROBE")); it will be replaced IN THE CLONE ONLY by the fork (README: uninstall your own version)"
    fi
    v=$(kv user_site_entries "$BASE_PROBE")
    if [ "${v:-0}" != 0 ]; then
        ck_warn "user site-packages $(kv user_site "$BASE_PROBE") has $v entries and is visible to BOTH envs (it can hide drift and shadow packages); consider PYTHONNOUSERSITE=1 for runs"
    fi
    if [ -n "$(kv env.INSTALL_PREFIX "$BASE_PROBE")$(kv env.PREFIX "$BASE_PROBE")" ]; then
        ck_warn "INSTALL_PREFIX or PREFIX is set in this shell; serve/attention/backend.py:305-307 prefers it over CONDA_PREFIX for RAFT/librmm. Unset it before running the engine."
    fi

    # -- target env --------------------------------------------------------------------------
    ENV_PREFIX=$(env_prefix "$ENV_NAME")
    if [ -n "$ENV_PREFIX" ]; then
        ENV_EXISTS=1
        if [ "$(realdir "$ENV_PREFIX")" = "$(realdir "$BASE_PREFIX")" ]; then
            ck_fail "env '$ENV_NAME' resolves to the base env's prefix $BASE_PREFIX"
            return 0
        fi
        if ! probe_env "$ENV_NAME" "$ENV_PROBE"; then
            ck_fail "env $ENV_NAME exists at $ENV_PREFIX but python does not run in it (a half-finished clone?). Remove it with: conda env remove -n $ENV_NAME"
            return 0
        fi
        if [ "$(kv ver.verl "$ENV_PROBE")" != "0.5.0" ] || [ "$(kv ver.torch "$ENV_PROBE")" != "$(kv ver.torch "$BASE_PROBE")" ]; then
            if [ "$MODE" = verify ]; then
                # let the drift report show (and fail on) the full difference
                ck_warn "env $ENV_NAME differs from $BASE_ENV in verl/torch (verl '$(kv ver.verl "$ENV_PROBE")', torch '$(kv ver.torch "$ENV_PROBE")' vs '$(kv ver.torch "$BASE_PROBE")'); see the drift report"
            else
                ck_fail "env $ENV_NAME exists but is not a clone of $BASE_ENV (verl '$(kv ver.verl "$ENV_PROBE")', torch '$(kv ver.torch "$ENV_PROBE")' vs '$(kv ver.torch "$BASE_PROBE")'). Remove it or choose another name."
            fi
        else
            ck_ok "env $ENV_NAME exists at $ENV_PREFIX (torch $(kv ver.torch "$ENV_PROBE"), verl $(kv ver.verl "$ENV_PROBE")); finished steps will be skipped"
        fi
    else
        ENV_EXISTS=0
        if [ "$MODE" = verify ]; then ck_fail "env $ENV_NAME does not exist; nothing to verify"; fi
        ck_info "env $ENV_NAME does not exist yet: it will be cloned from $BASE_ENV$(if [ "$CLONE_COPY" = 1 ]; then echo ' with --copy'; fi)"
    fi
    # the probe of the env that will be built on: the clone if it exists, otherwise the base
    if [ "$ENV_EXISTS" = 1 ]; then P="$ENV_PROBE"; else P="$BASE_PROBE"; fi

    # -- where pip installs (PIP_USER=0 is exported above; this proves it reaches pip) --------
    # The clone inherits the base env's pip and the same user/global pip config, so before the clone
    # exists the base env answers for it (pip install --root <tmp>: nothing is written into it).
    if [ "$ENV_EXISTS" = 1 ]; then f=$ENV_NAME; else f=$BASE_ENV; fi
    if pip_target_check "$f"; then ck_ok "$PIP_TARGET_MSG"; else ck_fail "$PIP_TARGET_MSG"; fi

    # -- git + submodules (flashinfer, raft; never vllm) ---------------------------------------
    if ! command -v git >/dev/null 2>&1; then
        ck_fail "git not found"
    elif [ "$(git -C "$REPO_ROOT" rev-parse --is-inside-work-tree 2>/dev/null)" != "true" ]; then
        ck_fail "$REPO_ROOT is not a git work tree (submodules cannot be checked)"
    else
        ck_info "repo at $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null) on branch $(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null)"
        check_submodule 3rdparty/flashinfer
        check_submodule 3rdparty/raft
        if [ -d "$FI_DIR/.git" ] || [ -f "$FI_DIR/.git" ]; then
            for f in 3rdparty/cutlass 3rdparty/spdlog; do
                st=$(git -C "$FI_DIR" submodule status -- "$f" 2>/dev/null | cut -c1)
                case "$st" in
                    " ") ck_ok "3rdparty/flashinfer/$f initialised" ;;
                    "-") ck_info "3rdparty/flashinfer/$f not initialised: will be (needed by the JIT: flashinfer/jit/env.py:64-68)"
                         NEED_SUBMODULE_INIT="$NEED_SUBMODULE_INIT 3rdparty/flashinfer/$f" ;;
                    "+") ck_fail "3rdparty/flashinfer/$f is not at the commit flashinfer records (git -C 3rdparty/flashinfer submodule status)" ;;
                    *) ck_fail "3rdparty/flashinfer/$f: unreadable submodule status" ;;
                esac
            done
        fi
        st=$(git -C "$REPO_ROOT" submodule status -- 3rdparty/vllm 2>/dev/null | cut -c1)
        ck_info "3rdparty/vllm (the fork) is NOT used and will not be initialised or installed (status flag '${st:-?}')"
    fi

    # -- CUDA toolkit, as the env would resolve it -------------------------------------------
    CUDA_HOME_FOUND=$(kv cuda_home "$P")
    torch_cuda=$(kv torch_cuda "$P")
    if [ -z "$CUDA_HOME_FOUND" ] || [ ! -x "$CUDA_HOME_FOUND/bin/nvcc" ]; then
        ck_fail "no nvcc for the env (CUDA_HOME='${CUDA_HOME:-}', resolution: $(kv cuda_home_how "$P") -> '${CUDA_HOME_FOUND}'). The FlashInfer JIT and RAFT both need a CUDA $(major_of "$torch_cuda").x toolkit with the standard layout (bin/nvcc, include/cuda_runtime.h, lib64/libcudart.so): export CUDA_HOME=/path/to/cuda-$torch_cuda (e.g. after 'module load cuda/$torch_cuda', or NVIDIA's runfile installed without root: 'sh cuda_${torch_cuda}.<x>_<driver>_linux.run --silent --toolkit --toolkitpath=\$HOME/cuda-$torch_cuda'), rerun --check, and keep CUDA_HOME exported in the shell that later runs integration/run_all_gates.sh. Do NOT conda-install cuda-toolkit into either env: conda's CUDA 12 packages put cuda_runtime.h under targets/x86_64-linux/include and have no lib64, and the JIT only uses \$cuda_home/include and -L\$cuda_home/lib64 (flashinfer/jit/cpp_ext.py:67,117)"
    else
        NVCC="$CUDA_HOME_FOUND/bin/nvcc"
        NVCC_VER=$("$NVCC" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9.]*\),.*/\1/p' | head -1) || NVCC_VER=""
        ck_info "nvcc $NVCC_VER at $NVCC (found via $(kv cuda_home_how "$P")); torch was built with CUDA $torch_cuda"
        if [ -z "$NVCC_VER" ]; then
            ck_fail "cannot parse '$NVCC --version'"
        elif [ "$(major_of "$NVCC_VER")" != "$(major_of "$torch_cuda")" ]; then
            ck_fail "nvcc $NVCC_VER and torch's CUDA $torch_cuda differ in the major version; JIT modules link the toolkit's cudart (flashinfer/jit/cpp_ext.py:117-123)"
        elif ! ver_ge "$NVCC_VER" 12.0; then
            ck_fail "nvcc $NVCC_VER < 12.0 (RAFT docs: CUDA Toolkit 12.0+)"
        else
            ck_ok "CUDA toolkit $NVCC_VER (major matches torch's $torch_cuda)"
            [ "$NVCC_VER" = "$torch_cuda" ] || ck_warn "nvcc $NVCC_VER != torch CUDA $torch_cuda: JIT modules are compiled against $NVCC_VER headers but run on torch's cudart $torch_cuda"
            ver_ge "$NVCC_VER" 12.9 || ck_warn "upstream used CUDA 12.9 (README.md:207). The JIT compiles RAFT/rmm 25.10 headers with -I\$CONDA_PREFIX/include only (backend.py:311,336) while rapids-cmake installs CCCL 3.0.3 under include/rapids, so the toolkit's bundled CCCL is used: older toolkits may fail to compile the top-k op. The JIT smoke step tests exactly this."
        fi
        [ -f "$CUDA_HOME_FOUND/include/cuda_runtime.h" ] || ck_fail "$CUDA_HOME_FOUND/include/cuda_runtime.h missing (CUDA headers needed by the JIT: flashinfer/jit/cpp_ext.py:68)"
    fi

    # -- host compiler -------------------------------------------------------------------------
    GXX=${CXX:-}
    [ -n "$GXX" ] || GXX=$(command -v c++ 2>/dev/null || command -v g++ 2>/dev/null || true)
    if [ -z "$GXX" ]; then
        ck_fail "no C++ compiler (CXX unset, no c++/g++ on PATH); RAFT docs: GCC 9.3+"
    else
        v=$("$GXX" -dumpfullversion -dumpversion 2>/dev/null | head -1) || v=""
        if "$GXX" --version 2>/dev/null | head -1 | grep -qi clang; then
            ck_warn "$GXX is clang ($("$GXX" --version | head -1)); RAFT and nvcc expect GCC on Linux"
        elif [ -z "$v" ]; then
            ck_fail "cannot read the version of $GXX"
        elif ! ver_ge "$v" "$GCC_MIN"; then
            ck_fail "$GXX is GCC $v < $GCC_MIN (RAFT docs/source/build.md)"
        else
            ck_ok "C++ compiler $GXX (GCC $v)"
            # nvcc's supported host GCC (NVIDIA install guide; from memory, so only a warning)
            if [ -n "$NVCC_VER" ]; then
                max=12
                if ver_ge "$NVCC_VER" 12.4; then max=13; fi
                if ver_ge "$NVCC_VER" 12.8; then max=14; fi
                [ "$(major_of "$v")" -le "$max" ] || ck_warn "GCC $v may be newer than nvcc $NVCC_VER supports (about GCC $max); nvcc would stop with 'unsupported GNU version'"
            fi
        fi
        # libstdc++ of the env vs what this compiler emits (README.md troubleshooting, GLIBCXX)
        libenv="${ENV_PREFIX:-$BASE_PREFIX}/lib/libstdc++.so.6"
        libcc=$("$GXX" -print-file-name=libstdc++.so.6 2>/dev/null || true)
        if [ -e "$libenv" ] && [ -n "$libcc" ] && [ -e "$libcc" ]; then
            ge=$({ grep -ao 'GLIBCXX_3\.4\.[0-9]*' "$libenv" 2>/dev/null || true; } | sort -u -t. -k3,3n | tail -1)
            gc=$({ grep -ao 'GLIBCXX_3\.4\.[0-9]*' "$libcc" 2>/dev/null || true; } | sort -u -t. -k3,3n | tail -1)
            if [ -n "$ge" ] && [ -n "$gc" ] && ! ver_ge "${ge#GLIBCXX_}" "${gc#GLIBCXX_}"; then
                ck_warn "the env's $libenv provides up to $ge, the compiler's libstdc++ $gc: JIT modules may fail to load with 'GLIBCXX_... not found' (README.md:281-292). Not fixed automatically: replacing the env's libstdc++ would also change what the training path loads."
            else
                ck_ok "libstdc++: env ${ge:-?}, compiler ${gc:-?}"
            fi
        fi
    fi

    # -- the JIT's own link line for cudart (flashinfer/jit/cpp_ext.py:115-135:
    #    '$cxx ... -L$cuda_home/lib64 -lcudart' + FLASHINFER_EXTRA_LDFLAGS). Blocking: without it every
    #    FlashInfer JIT module fails to link at the engine's first start. A Debian-style toolkit
    #    (CUDA_HOME=/usr, libcudart in the linker's default path) passes without a lib64.
    #    The linker is the JIT's: CXX or 'c++' as resolved INSIDE the env (cpp_ext.py:141; the env's
    #    activation scripts may set CXX, e.g. conda compilers), not this shell's $GXX.
    if [ -n "$NVCC" ]; then
        if [ "$ENV_EXISTS" = 1 ]; then jit_env=$ENV_NAME; else jit_env=$BASE_ENV; fi
        # ninja splits $cxx on spaces ("ccache g++" works), so only its first word must be on PATH
        # shellcheck disable=SC2016  # expanded by the sh inside the env
        jit_cxx=$(env_q "$jit_env" sh -c 'set -- ${CXX:-c++}; p=$(command -v "$1") || exit 1; shift; echo "ss_cxx=$p${1:+ $*}"' 2>/dev/null | sed -n 's/^ss_cxx=//p' | tail -1) || jit_cxx=""
        lt="$PY_DIR/ldprobe"; mkdir -p "$lt"
        printf 'int ss_ldprobe;\n' > "$lt/p.cc"
        if [ -z "$jit_cxx" ]; then
            ck_fail "the FlashInfer JIT links with '${CXX:-c++}' (CXX, else c++; flashinfer/jit/cpp_ext.py:141) run inside $jit_env, and that is not on PATH there. Export CXX=<path of a GCC >= $GCC_MIN g++> in this shell and in the shell that runs run_all_gates.sh, then rerun --check"
        # shellcheck disable=SC2086  # $jit_cxx (as ninja splits it) and FLASHINFER_EXTRA_LDFLAGS are word lists
        elif $jit_cxx -x c++ "$lt/p.cc" -shared -fPIC -L"$CUDA_HOME_FOUND/lib64" ${FLASHINFER_EXTRA_LDFLAGS:-} -lcudart -o "$lt/p.so" > "$lt/out.txt" 2>&1; then
            ck_ok "the JIT's link line resolves -lcudart ($jit_cxx -L$CUDA_HOME_FOUND/lib64${FLASHINFER_EXTRA_LDFLAGS:+ $FLASHINFER_EXTRA_LDFLAGS} + the linker's default paths)"
        else
            ck_fail "every FlashInfer JIT module links with '$jit_cxx ... -L$CUDA_HOME_FOUND/lib64 -lcudart' (flashinfer/jit/cpp_ext.py:115-123,141; the compiler as resolved inside $jit_env) and that link fails here: $(tail -2 "$lt/out.txt" | tr '\n' ' '). Use a toolkit with lib64/libcudart.so (see the CUDA_HOME advice: module load / NVIDIA runfile with --toolkitpath), or export FLASHINFER_EXTRA_LDFLAGS=-L<dir containing libcudart.so> in the shell for both this build and run_all_gates.sh. Do not install CUDA packages into $BASE_ENV or $ENV_NAME to fix this."
        fi
    fi

    # -- cmake (RAFT needs >= 3.30.4) and ninja -------------------------------------------------
    v=""
    f=$(command -v cmake 2>/dev/null || true)
    if [ -n "$f" ]; then v=$("$f" --version 2>/dev/null | head -1 | awk '{print $3}') || v=""; fi
    if [ -x "$TOOLS_DIR/bin/cmake" ] && ver_ge "$("$TOOLS_DIR/bin/cmake" --version | head -1 | awk '{print $3}')" "$CMAKE_MIN"; then
        CMAKE_EXE="$TOOLS_DIR/bin/cmake"
        ck_ok "cmake $("$CMAKE_EXE" --version | head -1 | awk '{print $3}') in the tools venv $TOOLS_DIR"
    elif [ -n "$v" ] && ver_ge "$v" "$CMAKE_MIN" && ! ver_ge "$v" 4.0; then
        CMAKE_EXE=$f
        ck_ok "cmake $v at $f"
    else
        CMAKE_EXE=""
        ck_info "cmake on PATH: ${v:-none}${f:+ ($f)}; RAFT needs >= $CMAKE_MIN (README.md:271-279). The build pip-installs 'cmake>=$CMAKE_MIN,<4' into the tools venv $TOOLS_DIR (no conda env is touched)"
    fi
    if [ -n "$(kv which_ninja "$P")" ]; then
        ck_ok "ninja in the env: $(kv which_ninja "$P") (RAFT's generator; the FlashInfer JIT runs 'ninja' at runtime)"
    else
        NEED_NINJA=1
        ck_warn "no ninja in the env (vLLM 0.9.1 requires it, so this is unexpected); the build will pip-install ninja --no-deps into the clone"
    fi
    v=$(kv ver.setuptools "$P")
    if [ -n "$v" ] && ver_ge "$v" "$SETUPTOOLS_MIN"; then
        ck_ok "setuptools $v (the fork's pyproject needs >= $SETUPTOOLS_MIN; --no-build-isolation uses the env's)"
    else
        NEED_SETUPTOOLS=1
        ck_info "setuptools ${v:-none} < $SETUPTOOLS_MIN: the build will install '$SETUPTOOLS_SPEC' --no-deps into the clone only (listed in the drift report)"
    fi
    v=$(kv ver.packaging "$P")
    ver_ge "${v:-0}" 24 || ck_warn "packaging ${v:-none} < 24 (the fork's build-system lists packaging>=24; not enforced with --no-build-isolation and setuptools vendors its own; left alone)"
    v=$(kv ver.pytest "$P")
    if [ -n "$v" ]; then
        ck_ok "pytest $v in the env (integration/run_all_gates.sh runs the test stages with it; 'import pytest' is checked in the build)"
    else
        NEED_PYTEST=1
        ck_info "no pytest in the env: the build installs $PYTEST_PKGS --no-deps into the clone only (the gate runner needs it and installs nothing itself)"
    fi
    # PIP_USER=0 only steers writes: outside a virtualenv pip uninstalls a package it replaces wherever
    # it is found, ~/.local included. Packages the build replaces must therefore not resolve there.
    # (pytest, iniconfig, pluggy, pygments, ninja are installed without a version spec only when
    # missing, so a user-site copy is used as it is and never replaced.)
    v=$(kv user_site "$P")
    rp_pkgs="flashinfer-python flashinfer"
    if [ "$NEED_SETUPTOOLS" = 1 ]; then rp_pkgs="setuptools $rp_pkgs"; fi
    if [ "$SPECGEN_INSTALL" = editable ]; then rp_pkgs="$rp_pkgs specgen"; fi
    for f in $rp_pkgs; do
        rp_loc=$(kv "loc.$f" "$P")
        if [ -n "$v" ] && [ -n "$rp_loc" ]; then
            case "$rp_loc/" in
                "$v"/*) ck_fail "$f resolves to the user site $rp_loc, which $BASE_ENV imports too: replacing it in the clone would make pip uninstall it from there. Move it out of the user site yourself (deciding what $BASE_ENV should import; 'python -m pip list --user' shows what is there), then rerun --check" ;;
            esac
        fi
    done

    # -- GPU -----------------------------------------------------------------------------------
    if command -v nvidia-smi >/dev/null 2>&1; then
        v=$(nvidia-smi --query-gpu=index,name,memory.used,memory.total,driver_version,compute_cap --format=csv,noheader 2>/dev/null) \
            || v=$(nvidia-smi --query-gpu=index,name,memory.used,memory.total,driver_version --format=csv,noheader 2>/dev/null) || v=""
        if [ -n "$v" ]; then
            printf '%s\n' "$v" | while IFS= read -r line; do ck_info "GPU $line"; done
            f=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1) || f=""
            if [ -n "$f" ] && [ -n "$NVCC_VER" ] && ! ver_ge "$f" "$NVCC_VER"; then
                ck_warn "driver supports CUDA $f, nvcc is $NVCC_VER"
            fi
            ck_ok "nvidia-smi works (RAFT's CMAKE_CUDA_ARCHITECTURES=NATIVE and the JIT's arch detection need a visible GPU, not free memory)"
        else
            ck_fail "nvidia-smi is installed but cannot query the GPU"
        fi
    else
        ck_fail "nvidia-smi not found: RAFT is configured for the NATIVE GPU arch (raft build.sh) and the JIT detects the arch from the device"
    fi
    if [ "${CUDA_VISIBLE_DEVICES-unset}" = "" ]; then
        ck_fail "CUDA_VISIBLE_DEVICES is set but empty: no GPU visible to the build"
    fi

    # -- disk ----------------------------------------------------------------------------------
    local envs_dir need_envs sz
    envs_dir=$(dirname "$BASE_PREFIX")
    need_envs=$MIN_FREE_GB_ENVS
    if [ "$ENV_EXISTS" = 0 ] && [ "${SKIP_DU:-0}" != 1 ]; then
        sz=$(du -sk "$BASE_PREFIX" 2>/dev/null | awk '{printf "%d", $1 / 1048576 + 1}') || sz=""
        if [ -n "$sz" ]; then
            ck_info "$BASE_ENV occupies about $sz GB; a --copy clone needs about the same"
            if [ "$((sz + 5))" -gt "$need_envs" ]; then need_envs=$((sz + 5)); fi
        fi
    fi
    if [ "$ENV_EXISTS" = 0 ]; then
        v=$(free_gb "$envs_dir")
        if [ "${v:-0}" -ge "$need_envs" ]; then ck_ok "disk: ${v} GB free for the clone in $envs_dir (need $need_envs)"
        else ck_fail "disk: ${v:-?} GB free in $envs_dir, need about $need_envs GB for the clone (conda puts new envs in the first writable envs_dirs entry; check 'conda config --show envs_dirs')"; fi
    fi
    v=$(free_gb "$BUILD_DIR")
    if [ "${v:-0}" -ge "$MIN_FREE_GB_BUILD" ]; then ck_ok "disk: ${v} GB free for $BUILD_DIR (need $MIN_FREE_GB_BUILD)"
    else ck_fail "disk: ${v:-?} GB free for $BUILD_DIR, need $MIN_FREE_GB_BUILD (set BUILD_DIR elsewhere)"; fi
    v=$(free_gb "$HOME")
    if [ "${v:-0}" -ge "$MIN_FREE_GB_HOME" ]; then ck_ok "disk: ${v} GB free in \$HOME (~/.cache/flashinfer JIT cache, pip cache)"
    else ck_warn "disk: ${v:-?} GB free in \$HOME (FlashInfer JIT cache goes to ~/.cache/flashinfer unless FLASHINFER_WORKSPACE_BASE is set)"; fi
    if command -v free >/dev/null 2>&1; then
        ck_info "host RAM: $(free -g | awk '/^Mem:/ {print $2 " GB total, " $7 " GB available"}')"
    fi

    # -- network -------------------------------------------------------------------------------
    if [ "${SKIP_NETWORK_CHECK:-0}" = 1 ]; then
        ck_info "network check skipped (SKIP_NETWORK_CHECK=1)"
    else
        if [ -n "${https_proxy:-}${HTTPS_PROXY:-}" ]; then ck_info "an https proxy is set in this shell"; fi
        net_probe "PyPI (cmake / setuptools wheels)" "https://pypi.org/simple/cmake/"
        if [ "$RAPIDS_BRANCH" = "?" ]; then
            # 3rdparty/raft not checked out yet: its RAPIDS_BRANCH file is unknown until the init step
            net_probe "github.com (rapids-cmake, rmm, CCCL, cuco, cutlass are fetched at configure time)" \
                "https://github.com/rapidsai/rapids-cmake"
            ck_info "rapids branch unknown until 3rdparty/raft is initialised; rerun --check after the build's submodule step for the exact branch probe"
        else
            net_probe "rapids-cmake $RAPIDS_BRANCH archive (raft cmake/RAPIDS.cmake downloads it at configure time)" \
                "https://github.com/rapidsai/rapids-cmake/archive/refs/heads/$RAPIDS_BRANCH.zip"
            v=$(git ls-remote --heads https://github.com/rapidsai/rmm.git "$RAPIDS_BRANCH" 2>/dev/null | awk '{print $1}' | head -1) || v=""
            if [ -n "$v" ]; then
                ck_ok "rmm $RAPIDS_BRANCH on GitHub is at ${v:0:12} (rapids-cmake fetches rmm by BRANCH, so record this commit)"
            else
                ck_fail "cannot list rmm branch $RAPIDS_BRANCH on GitHub (CPM clones rmm, rapids_logger, CCCL, cuco, cutlass from github.com during configure)"
            fi
        fi
        for f in $NEED_SUBMODULE_INIT; do
            case "$f" in 3rdparty/flashinfer/*) continue ;; esac
            v=$(git config -f "$REPO_ROOT/.gitmodules" --get "submodule.$f.url" 2>/dev/null || true)
            if [ -n "$v" ] && git ls-remote "$v" HEAD >/dev/null 2>&1; then ck_ok "submodule remote reachable: $v"
            else ck_fail "submodule remote of $f not reachable: ${v:-no url in .gitmodules}"; fi
        done
    fi
}

net_probe() {  # net_probe WHAT URL
    local code=""
    if command -v curl >/dev/null 2>&1; then
        code=$(curl -s -o /dev/null -L -I --max-time 20 -w '%{http_code}' "$2" 2>/dev/null) || code=""
    elif command -v wget >/dev/null 2>&1; then
        wget -q --spider --timeout=20 "$2" 2>/dev/null && code=200
    else
        ck_warn "neither curl nor wget: cannot check $1"
        return 0
    fi
    if [ "$code" = 200 ]; then ck_ok "network: $1"
    else ck_fail "network: $1 -> HTTP ${code:-no answer} for $2"; fi
}

print_plan() {
    local s
    ss_log "plan (each step is skipped when its result is already there; rerun the same command after a failure):"
    if [ "$ENV_EXISTS" = 1 ]; then s="skip (exists)"; else s="conda create -y -n $ENV_NAME --clone $BASE_ENV$(if [ "$CLONE_COPY" = 1 ]; then echo ' --copy'; fi)"; fi
    ss_log "  a. clone:          $s"
    s=${NEED_SUBMODULE_INIT# }
    ss_log "  b. submodules:     ${s:-none to initialise} (only flashinfer + its cutlass/spdlog, and raft; never 3rdparty/vllm)"
    if [ -n "$CMAKE_EXE" ]; then s="use $CMAKE_EXE"; else s="python -m venv $TOOLS_DIR; pip install 'cmake>=$CMAKE_MIN,<4'"; fi
    ss_log "  c. build tools:    $s"
    s="none"
    if [ "$NEED_SETUPTOOLS" = 1 ]; then s="pip install --no-deps '$SETUPTOOLS_SPEC'"; fi
    if [ "$NEED_NINJA" = 1 ]; then s="$s; pip install --no-deps ninja"; fi
    ss_log "  d. clone packages: $s"
    if [ "$FLASHINFER_EDITABLE" = 1 ]; then s="pip install --no-build-isolation --no-deps --verbose -e 3rdparty/flashinfer   (README.md:192-194, plus --no-deps)"
    else s="pip install --no-build-isolation --no-deps --verbose 3rdparty/flashinfer   (regular install, FLASHINFER_EDITABLE=0)"; fi
    if [ "$FI_VERSION" = "?" ]; then s="$s; version read after the submodule step"; else s="$s; version $FI_VERSION"; fi
    ss_log "  e. FlashInfer:     $s"
    ss_log "  f. RAFT + rmm:     3rdparty/raft/build.sh libraft -n (configure+build), staged install check, then cmake --install into \$CONDA_PREFIX of $ENV_NAME"
    ss_log "                     (README.md:196-198; INSTALL_PREFIX=<env prefix>, LIBRAFT_BUILD_DIR=$RAFT_BUILD_DIR, CPM_SOURCE_CACHE=$CPM_CACHE_DIR, rapids branch $RAPIDS_BRANCH)"
    if [ "$SPECGEN_INSTALL" = editable ]; then s="pip install --no-deps -e .   (the owner's 'pip install -e .' without its dependency pins)"; else s="skip (SPECGEN_INSTALL=none)"; fi
    ss_log "  g. this repo:      $s"
    if [ "$NEED_PYTEST" = 1 ]; then s="pip install --no-deps $PYTEST_PKGS, then any module 'import pytest' still names (one at a time, --no-deps)"
    else s="none (pytest $(kv ver.pytest "$P") present; 'import pytest' is checked)"; fi
    ss_log "  h. pytest:         $s"
    ss_log "  i. verify:         import flashinfer + serve.rl.engine; RAFT/librmm where backend.py looks;$(if [ "$SKIP_JIT_SMOKE" != 1 ]; then echo ' JIT-compile the top-k op;'; fi) import pytest; drift report vs $BASE_ENV; $BASE_ENV fingerprint unchanged"
    ss_log "  every pip install above is checked first to write into $ENV_NAME, not the user site (PIP_USER=0), and no package it replaces lives in the user site"
    ss_log "  NOT done: the vLLM fork; patched verl (next: bash integration/setup_env.sh $ENV_NAME)"
}

fingerprint_base() {  # fingerprint_base OUT: package lists of the base env (read-only)
    {
        echo "# pip list --format=freeze"
        env_q "$BASE_ENV" env PYTHONDONTWRITEBYTECODE=1 python -B -m pip list --format=freeze
        echo "# conda list --export"
        "$CONDA_BIN" list -n "$BASE_ENV" --export
    } > "$1"
}

print_next() {
    local p=${ENV_PREFIX:-<prefix of $ENV_NAME>} cuda_exports=""
    # the toolkit (and linker flags / compiler) this build was checked with; the JIT at gate time
    # resolves CUDA_HOME, FLASHINFER_EXTRA_LDFLAGS and CXX from the shell that runs the gates
    if [ -n "$CUDA_HOME_FOUND" ]; then cuda_exports="export CUDA_HOME=$(printf '%q' "$CUDA_HOME_FOUND")"; fi
    if [ -n "${FLASHINFER_EXTRA_LDFLAGS:-}" ]; then cuda_exports="${cuda_exports:-export} FLASHINFER_EXTRA_LDFLAGS=$(printf '%q' "$FLASHINFER_EXTRA_LDFLAGS")"; fi
    if [ -n "${CXX:-}" ]; then cuda_exports="${cuda_exports:-export} CXX=$(printf '%q' "$CXX")"; fi
    cat <<EOF

[$SS_SCRIPT] ---------------------------------------------------------------- next
[$SS_SCRIPT] 1) install the patched verl into the new env:
       bash integration/setup_env.sh $ENV_NAME
[$SS_SCRIPT] 2) re-run the drift report afterwards (verl becomes an editable 0.5.0; versions must still match):
       bash integration/build_sparsespec_env.sh --verify
[$SS_SCRIPT] 3) then the gate run, one command (see HANDOFF.md). Run these lines in the shell that runs it; with
[$SS_SCRIPT]    tmux, run them INSIDE the tmux session (a new tmux session does not get this shell's exports).
[$SS_SCRIPT]    backend.py:305-307 prefers INSTALL_PREFIX/PREFIX over CONDA_PREFIX, and the JIT must use the same
[$SS_SCRIPT]    CUDA toolkit as this build:
       unset INSTALL_PREFIX PREFIX${cuda_exports:+; $cuda_exports}
       bash integration/run_all_gates.sh

[$SS_SCRIPT] ---------------------------------------------------------------- packaging for the shared Drive
[$SS_SCRIPT] Recommended: conda-pack (relocatable: conda-unpack rewrites the hard-coded prefix in scripts and
[$SS_SCRIPT] metadata; a plain tarball of the env directory only works when unpacked at the identical path).
[$SS_SCRIPT] Editable installs ($(if [ "$FLASHINFER_EDITABLE" = 1 ]; then echo 'the FlashInfer fork, '; fi)specgen, and verl after setup_env.sh) are links into
[$SS_SCRIPT] this checkout; conda-pack refuses them unless told to skip them, so they are re-installed after
[$SS_SCRIPT] unpacking from a checkout at the recorded commits. FLASHINFER_EDITABLE=0 makes the fork part of
[$SS_SCRIPT] the archive instead. conda-pack goes into the tools venv, not into any conda env:
[$SS_SCRIPT] --ignore-missing-files: if the clone's setuptools was a conda package older than 77, the build replaced it with
[$SS_SCRIPT] pip (build prerequisites step); conda-pack would otherwise stop on the conda-managed setuptools files pip removed.
       $TOOLS_DIR/bin/python -m pip install conda-pack
       OUT=\$HOME/verl-sparsespec-pack; mkdir -p "\$OUT"
       $TOOLS_DIR/bin/conda-pack -p "$p" --ignore-editable-packages --ignore-missing-files -o "\$OUT/$ENV_NAME.tar.gz"
       # RAFT/rmm files were installed by cmake, not by conda or pip: confirm they are in the archive
       tar -tzf "\$OUT/$ENV_NAME.tar.gz" | grep -E '(^|/)lib/librmm\.so\$|(^|/)include/raft/matrix/detail/select_k\.cuh\$'
       # if that prints nothing, ship them separately from cmake's install manifest:
       sed "s#^$p/##" "$REPORT_DIR/raft_install_manifest.txt" > "\$OUT/raft_files.txt"
       tar -czf "\$OUT/raft-rmm-install.tar.gz" -C "$p" -T "\$OUT/raft_files.txt"
       # manifests that make the env reproducible without the archive
       conda env export -n $ENV_NAME > "\$OUT/$ENV_NAME.environment.yml"
       conda list -n $ENV_NAME --explicit > "\$OUT/$ENV_NAME.conda-explicit.txt"
       conda run -n $ENV_NAME python -m pip freeze > "\$OUT/$ENV_NAME.pip-freeze.txt"
       cp "$REPORT_DIR/drift.md" "\$OUT/"
       (cd "$REPO_ROOT" && git rev-parse HEAD && git submodule status --recursive) > "\$OUT/source-commits.txt"
       (cd "\$OUT" && sha256sum * > SHA256SUMS)
[$SS_SCRIPT] On the receiving side (any path):
       D=\$(conda info --base)/envs/$ENV_NAME; mkdir -p "\$D"; tar -xzf $ENV_NAME.tar.gz -C "\$D"
       source "\$D/bin/activate" && conda-unpack && source "\$D/bin/deactivate"
       # from a checkout at the commits in source-commits.txt:
       git submodule update --init 3rdparty/flashinfer 3rdparty/raft
       git -C 3rdparty/flashinfer submodule update --init 3rdparty/cutlass 3rdparty/spdlog
       conda run -n $ENV_NAME python -m pip install --no-build-isolation --no-deps -e 3rdparty/flashinfer
       conda run -n $ENV_NAME python -m pip install --no-deps -e .
       bash integration/setup_env.sh $ENV_NAME
       bash integration/build_sparsespec_env.sh --verify
EOF
}

# =============================================================================================
ss_step "prerequisites (read-only)"
run_checks
if [ "$REFUSED" = 0 ]; then print_plan; fi
if [ "$CK_BLOCK" -gt 0 ]; then
    ss_log "prerequisites: $CK_BLOCK blocking problem(s), $CK_WARN warning(s):$CK_BLOCKERS"
else
    ss_log "prerequisites: no blocking problem, $CK_WARN warning(s)"
fi
if [ "$MODE" = check ]; then
    if [ "$CK_BLOCK" -gt 0 ]; then
        SS_FAILED=1   # no generic failure banner for a check that found problems
        printf '\n[%s] CHECK FAILED: fix the FAIL lines above, then rerun --check.%s\n' "$SS_SCRIPT" "${SS_LOG:+ (log: $SS_LOG)}"
        exit 1
    fi
    ss_done "check passed; build with: bash integration/build_sparsespec_env.sh${POS_ENV:+ $POS_ENV}"
    exit 0
fi
[ "$CK_BLOCK" -eq 0 ] || ss_die "$CK_BLOCK blocking problem(s):$CK_BLOCKERS"

mkdir -p "$REPORT_DIR"
cp "$BASE_PROBE" "$REPORT_DIR/base.probe.txt"

ss_step "fingerprint $BASE_ENV (tripwire: it must be identical at the end)"
if [ "$MODE" = build ] || [ ! -f "$REPORT_DIR/base.fingerprint.txt" ]; then
    fingerprint_base "$REPORT_DIR/base.fingerprint.txt"
fi
[ -s "$REPORT_DIR/base.fingerprint.txt" ] || ss_die "could not list the packages of $BASE_ENV"
ss_log "$(grep -vc '^#' "$REPORT_DIR/base.fingerprint.txt") lines in $REPORT_DIR/base.fingerprint.txt"

if [ "$MODE" = build ]; then
    # -----------------------------------------------------------------------------------------
    ss_step "clone $BASE_ENV -> $ENV_NAME"
    if [ "$ENV_EXISTS" = 1 ]; then
        ss_log "exists at $ENV_PREFIX; not cloning again"
    else
        # --copy: no hard links into the package cache, so nothing written into the clone later
        # (cmake install, pip) can write through a shared inode into $BASE_ENV.
        if [ "$CLONE_COPY" = 1 ]; then
            ss_run "$CONDA_BIN" create -y -n "$ENV_NAME" --clone "$BASE_ENV" --copy
        else
            ss_run "$CONDA_BIN" create -y -n "$ENV_NAME" --clone "$BASE_ENV"
        fi
        ENV_PREFIX=$(env_prefix "$ENV_NAME")
        [ -n "$ENV_PREFIX" ] || ss_die "conda create finished but '$ENV_NAME' is not in 'conda env list'"
        ENV_EXISTS=1
    fi
    probe_env "$ENV_NAME" "$ENV_PROBE" || ss_die "python does not run in $ENV_NAME: $(tail -5 "$ENV_PROBE.err" | tr '\n' ' ')"
    [ "$(kv sys_prefix "$ENV_PROBE")" = "$(realdir "$ENV_PREFIX")" ] \
        || ss_die "conda run -n $ENV_NAME runs python from $(kv sys_prefix "$ENV_PROBE"), not $ENV_PREFIX"
    [ "$(kv conda_prefix "$ENV_PROBE")" != "" ] || ss_die "conda run does not set CONDA_PREFIX (backend.py needs it)"
    [ "$(kv ver.torch "$ENV_PROBE")" = "$(kv ver.torch "$BASE_PROBE")" ] \
        || ss_die "the clone has torch '$(kv ver.torch "$ENV_PROBE")', the base '$(kv ver.torch "$BASE_PROBE")'"
    ss_log "clone ok: $ENV_PREFIX, python $(kv python "$ENV_PROBE"), torch $(kv ver.torch "$ENV_PROBE")"

    # -----------------------------------------------------------------------------------------
    ss_step "submodules: flashinfer (+ cutlass, spdlog) and raft; NOT vllm"
    sub_top=""
    sub_fi=""
    for f in $NEED_SUBMODULE_INIT; do
        case "$f" in
            3rdparty/flashinfer/*) sub_fi="$sub_fi ${f#3rdparty/flashinfer/}" ;;
            *) sub_top="$sub_top $f" ;;
        esac
    done
    if [ -n "$sub_top" ]; then
        # shellcheck disable=SC2086  # word splitting of the path list is intended
        ss_run git -C "$REPO_ROOT" submodule update --init -- $sub_top
    fi
    # flashinfer's own submodules, also when flashinfer was only initialised just now
    for f in 3rdparty/cutlass 3rdparty/spdlog; do
        case "$(git -C "$FI_DIR" submodule status -- "$f" 2>/dev/null | cut -c1)" in
            "-") sub_fi="$sub_fi $f" ;;
        esac
    done
    # shellcheck disable=SC2086  # split the space-separated list on purpose
    sub_fi=$(printf '%s\n' $sub_fi | sort -u | tr '\n' ' ')
    if [ -n "${sub_fi// /}" ]; then
        # shellcheck disable=SC2086
        ss_run git -C "$FI_DIR" submodule update --init -- $sub_fi
    fi
    for f in 3rdparty/flashinfer 3rdparty/raft; do
        st=$(submodule_state "$f")
        [ "$(echo "$st" | awk '{print $1}')" = "=" ] || ss_die "$f is not at the recorded commit after the update: $st"
        ss_log "$f at $(echo "$st" | awk '{print $2}')"
    done
    [ -f "$FI_DIR/3rdparty/cutlass/include/cutlass/cutlass.h" ] || ss_die "flashinfer's cutlass submodule is empty"
    [ -f "$FI_DIR/3rdparty/spdlog/include/spdlog/spdlog.h" ] || ss_die "flashinfer's spdlog submodule is empty"
    [ -f "$RAFT_DIR/build.sh" ] || ss_die "3rdparty/raft/build.sh missing"
    read_submodule_facts
    [ "$FI_VERSION" != "?" ] || ss_die "3rdparty/flashinfer/version.txt missing after the submodule update"
    [ "$RAPIDS_BRANCH" != "?" ] || ss_die "3rdparty/raft/RAPIDS_BRANCH missing after the submodule update"
    ss_log "flashinfer fork version $FI_VERSION, rapids branch $RAPIDS_BRANCH"

    # -----------------------------------------------------------------------------------------
    ss_step "build tools outside every conda env (cmake >= $CMAKE_MIN)"
    if [ -z "$CMAKE_EXE" ]; then
        if [ ! -x "$TOOLS_DIR/bin/python" ]; then
            # the clone's interpreter only creates the venv; nothing is installed into the clone
            ss_run "$ENV_PREFIX/bin/python" -m venv "$TOOLS_DIR"
        fi
        ss_run "$TOOLS_DIR/bin/python" -m pip install "cmake>=$CMAKE_MIN,<4"
        CMAKE_EXE="$TOOLS_DIR/bin/cmake"
    fi
    v=$("$CMAKE_EXE" --version | head -1 | awk '{print $3}')
    ver_ge "$v" "$CMAKE_MIN" || ss_die "cmake $v at $CMAKE_EXE is older than $CMAKE_MIN"
    TOOL_BIN="$PY_DIR/toolbin"
    mkdir -p "$TOOL_BIN"
    ln -sf "$CMAKE_EXE" "$TOOL_BIN/cmake"
    if [ -x "$(dirname "$CMAKE_EXE")/ctest" ]; then ln -sf "$(dirname "$CMAKE_EXE")/ctest" "$TOOL_BIN/ctest"; fi
    ss_log "cmake $v ($CMAKE_EXE)"

    # -----------------------------------------------------------------------------------------
    ss_step "build prerequisites inside the clone (only if missing; --no-deps)"
    # before the first pip install into the clone: pip must write into the clone, not ~/.local
    pip_target_check "$ENV_NAME" || ss_die "$PIP_TARGET_MSG"
    ss_log "$PIP_TARGET_MSG"
    probe_env "$ENV_NAME" "$ENV_PROBE" || ss_die "probe of $ENV_NAME failed"
    if ! ver_ge "$(kv ver.setuptools "$ENV_PROBE")" "$SETUPTOOLS_MIN"; then
        # 3rdparty/flashinfer/pyproject.toml: license = "Apache-2.0" + license-files (PEP 639) need
        # setuptools >= 77; with --no-build-isolation pip uses the env's setuptools as is.
        env_run "$ENV_NAME" python -m pip install --no-deps "$SETUPTOOLS_SPEC"
    else
        ss_log "setuptools $(kv ver.setuptools "$ENV_PROBE") is new enough"
    fi
    if [ -z "$(kv which_ninja "$ENV_PROBE")" ]; then
        env_run "$ENV_NAME" python -m pip install --no-deps ninja
    else
        ss_log "ninja present: $(kv which_ninja "$ENV_PROBE")"
    fi

    # -----------------------------------------------------------------------------------------
    ss_step "install the FlashInfer fork ($FI_VERSION) into $ENV_NAME"
    probe_env "$ENV_NAME" "$ENV_PROBE" || ss_die "probe of $ENV_NAME failed"
    fi_real=$(realdir "$FI_DIR")
    have_url=$(kv url.flashinfer-python "$ENV_PROBE")
    have_ver=$(kv ver.flashinfer-python "$ENV_PROBE")
    if [ "$FLASHINFER_EDITABLE" = 1 ]; then want_url="editable:file://$fi_real"; else want_url="file://$fi_real"; fi
    if [ "$have_url" = "$want_url" ] && [ "$have_ver" = "$FI_VERSION" ]; then
        ss_log "already installed: flashinfer-python $have_ver ($have_url)"
    else
        for f in flashinfer-python flashinfer; do
            if [ -n "$(kv "ver.$f" "$ENV_PROBE")" ]; then
                ss_log "removing $f $(kv "ver.$f" "$ENV_PROBE") ($(kv "url.$f" "$ENV_PROBE")) from the clone"
                env_run "$ENV_NAME" python -m pip uninstall -y "$f"
            fi
        done
        # README.md:194 is `pip install --no-build-isolation --verbose --editable .`. --no-deps added:
        # the fork's install_requires (3rdparty/flashinfer/setup.py:56-65) would add
        # nvidia-cudnn-frontend and pynvml and may upgrade packaging; `import flashinfer` eagerly needs
        # only torch, jinja2, filelock, packaging, requests (static trace), all already in the clone.
        if [ "$FLASHINFER_EDITABLE" = 1 ]; then
            env_run "$ENV_NAME" python -m pip install --no-build-isolation --no-deps --verbose -e "$FI_DIR"
        else
            env_run "$ENV_NAME" python -m pip install --no-build-isolation --no-deps --verbose "$FI_DIR"
        fi
    fi

    # -----------------------------------------------------------------------------------------
    ss_step "build RAFT (header-only) + rmm and install them into $ENV_NAME"
    raft_commit=$(git -C "$RAFT_DIR" rev-parse HEAD)
    stamp="$BUILD_DIR/raft.installed"
    if [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$raft_commit $ENV_PREFIX" ] \
        && [ -f "$ENV_PREFIX/include/raft/matrix/detail/select_k.cuh" ] && [ -e "$ENV_PREFIX/lib/librmm.so" ]; then
        ss_log "already installed from raft $raft_commit (stamp $stamp); delete the stamp to rebuild"
    else
        jobs=${PARALLEL_LEVEL:-}
        if [ -z "$jobs" ]; then jobs=$(nproc_n); [ "$jobs" -gt 16 ] && jobs=16; fi
        mkdir -p "$RAFT_BUILD_DIR" "$CPM_CACHE_DIR"
        rmm_head=$(git ls-remote --heads https://github.com/rapidsai/rmm.git "$RAPIDS_BRANCH" 2>/dev/null | awk '{print $1}' | head -1) || rmm_head=""
        ss_log "rapids-cmake and rmm are fetched by branch $RAPIDS_BRANCH (not pinned): rmm head ${rmm_head:-unknown} today"
        printf 'raft %s\nrapids branch %s\nrmm head %s\nnvcc %s (%s)\ncmake %s\n' "$raft_commit" "$RAPIDS_BRANCH" \
            "${rmm_head:-unknown}" "$NVCC_VER" "$NVCC" "$v" > "$REPORT_DIR/raft_sources.txt"
        # README.md:198 is `./build.sh libraft` with INSTALL_PREFIX defaulting to $CONDA_PREFIX
        # (raft build.sh:101). Here: the same target with -n (configure + build, no install), then a
        # staged install to see the file list, then the real install. Same files, but nothing lands
        # in the env before we know it will not overwrite something there.
        env_run "$ENV_NAME" bash "$PY_DIR/with_path.sh" "$TOOL_BIN" env \
            INSTALL_PREFIX="$ENV_PREFIX" LIBRAFT_BUILD_DIR="$RAFT_BUILD_DIR" PARALLEL_LEVEL="$jobs" \
            CMAKE_GENERATOR=Ninja CPM_SOURCE_CACHE="$CPM_CACHE_DIR" \
            CUDACXX="$NVCC" CUDAToolkit_ROOT="$CUDA_HOME_FOUND" \
            bash "$RAFT_DIR/build.sh" libraft -n
        rm -rf "$RAFT_STAGE_DIR"
        # the staged install rewrites install_manifest.txt; keep the list of a previous real install
        prev_manifest="$REPORT_DIR/raft_install_manifest.txt"
        [ -f "$prev_manifest" ] || : > "$prev_manifest"
        env_run "$ENV_NAME" bash "$PY_DIR/with_path.sh" "$TOOL_BIN" cmake --install "$RAFT_BUILD_DIR" --prefix "$RAFT_STAGE_DIR"
        (cd "$RAFT_STAGE_DIR" && find . \( -type f -o -type l \) | sed 's#^\./##' | sort) > "$REPORT_DIR/raft_install_files.txt"
        n=$(wc -l < "$REPORT_DIR/raft_install_files.txt" | tr -d ' ')
        [ "$n" -gt 0 ] || ss_die "the staged RAFT install is empty ($RAFT_STAGE_DIR)"
        grep -q '^include/raft/matrix/detail/select_k\.cuh$' "$REPORT_DIR/raft_install_files.txt" \
            || ss_die "the staged install has no include/raft/matrix/detail/select_k.cuh"
        grep -q '^lib/librmm\.so$' "$REPORT_DIR/raft_install_files.txt" \
            || ss_die "the staged install has no lib/librmm.so (backend.py:316 asserts on it); see $REPORT_DIR/raft_install_files.txt"
        # Would the install overwrite files of the env that cmake did not put there itself?
        clash="$REPORT_DIR/raft_install_clashes.txt"
        : > "$clash"
        while IFS= read -r rel; do
            tgt="$ENV_PREFIX/$rel"
            [ -e "$tgt" ] || continue
            # shellcheck disable=SC2012  # hard-link count; ls -ld is the same on GNU and BSD
            links=$(ls -ld "$tgt" 2>/dev/null | awk '{print $2}')
            if [ "${links:-1}" -gt 1 ] && [ ! -d "$tgt" ]; then echo "HARDLINKED $rel" >> "$clash"; fi
            if grep -qxF "$tgt" "$prev_manifest"; then
                continue   # ours from a previous run
            fi
            echo "EXISTS $rel" >> "$clash"
        done < "$REPORT_DIR/raft_install_files.txt"
        if grep -q '^HARDLINKED' "$clash"; then
            ss_die "the RAFT install would write into hard-linked files of $ENV_NAME (shared with the conda package cache and possibly with $BASE_ENV): $(grep '^HARDLINKED' "$clash" | head -5 | tr '\n' ' ') ... full list: $clash. Recreate the clone with --copy (CLONE_COPY=1) or decide by hand."
        fi
        if grep -q '^EXISTS' "$clash"; then
            ss_warn "the RAFT install replaces $(grep -c '^EXISTS' "$clash") file(s) already in $ENV_NAME (list: $clash):"
            grep '^EXISTS' "$clash" | head -20 | sed 's/^/    /'
        fi
        ss_log "$n files to install (list: $REPORT_DIR/raft_install_files.txt)"
        env_run "$ENV_NAME" bash "$PY_DIR/with_path.sh" "$TOOL_BIN" cmake --install "$RAFT_BUILD_DIR"
        if [ -f "$RAFT_BUILD_DIR/install_manifest.txt" ]; then
            cp "$RAFT_BUILD_DIR/install_manifest.txt" "$prev_manifest"
            m=$(wc -l < "$prev_manifest" | tr -d ' ')
            # an install rule with an absolute DESTINATION ignores --prefix: the staged list would miss it
            if [ "$m" != "$n" ]; then
                ss_warn "cmake installed $m files into the env but the staged install had $n (list: $prev_manifest)"
            fi
        else
            ss_warn "no install_manifest.txt in $RAFT_BUILD_DIR after the install"
        fi
        [ -f "$ENV_PREFIX/include/raft/matrix/detail/select_k.cuh" ] || ss_die "RAFT headers not in $ENV_PREFIX/include after the install"
        [ -e "$ENV_PREFIX/lib/librmm.so" ] || ss_die "librmm.so not in $ENV_PREFIX/lib after the install"
        echo "$raft_commit $ENV_PREFIX" > "$stamp"
        rm -rf "$RAFT_STAGE_DIR"
    fi

    # -----------------------------------------------------------------------------------------
    ss_step "install the repo (specgen): pip install --no-deps -e ."
    # The owner wrote `pip install -e .`. With its dependencies it would change the training path:
    #  - datasets==3.0.1 (pyproject.toml:10) downgrades the datasets verl loads parquet with, and with
    #    it dill (<0.3.9) and fsspec (<=2024.6.1);
    #  - latex2sympy2 pins antlr4-python3-runtime==4.7.2, while hydra-core / omegaconf (verl's config
    #    system) require 4.9.*;
    #  - pre-commit, word2number, tabulate are not used by serve/rl (tabulate/datasets: serve/run.py
    #    only); scipy, wandb, numba are already in the clone (vLLM 0.9.1 / verl 0.5.0 requirements).
    # Every third-party module the engine imports (static trace from serve.rl.engine: torch,
    # flashinfer, triton, numba, numpy, jinja2, transformers, safetensors, huggingface_hub, tqdm,
    # wandb, vllm._custom_ops) is already in the clone or installed above, so --no-deps adds nothing.
    # The editable install only makes `serve` importable without PYTHONPATH (setuptools maps the
    # namespace dirs of the repo root); the run scripts export PYTHONPATH=<repo> anyway.
    if [ "$SPECGEN_INSTALL" = editable ]; then
        probe_env "$ENV_NAME" "$ENV_PROBE" || ss_die "probe of $ENV_NAME failed"
        if [ "$(kv url.specgen "$ENV_PROBE")" = "editable:file://$REPO_ROOT" ]; then
            ss_log "already installed editable from $REPO_ROOT"
        else
            env_run "$ENV_NAME" python -m pip install --no-deps -e "$REPO_ROOT"
        fi
    else
        ss_log "skipped (SPECGEN_INSTALL=$SPECGEN_INSTALL)"
    fi

    # -----------------------------------------------------------------------------------------
    ss_step "pytest for the gate runner (only if 'import pytest' fails; --no-deps)"
    # integration/run_all_gates.sh runs the CPU tests and the GPU unit tests (a required gate) with
    # this env's pytest and installs nothing itself. --no-deps, and pip leaves a package that is
    # already installed as it is, so no package already in the clone changes: only missing ones are
    # added. A module 'import pytest' still misses (pygments for pytest >= 8.4, ...) is added
    # alone, by its module name, and recorded in pytest_extra.txt for the drift report.
    pytest_extra="$REPORT_DIR/pytest_extra.txt"
    [ -f "$pytest_extra" ] || : > "$pytest_extra"
    pt_out="$PY_DIR/pytest_check.out"
    pt_tried=""
    pt_round=0
    mkdir -p "$BUILD_DIR/run"
    while :; do
        pt_rc=0
        (cd "$BUILD_DIR/run" && env_q "$ENV_NAME" env PYTHONDONTWRITEBYTECODE=1 python -B "$PY_DIR/pytest_check.py") > "$pt_out" 2>&1 || pt_rc=$?
        if [ "$pt_rc" = 0 ]; then
            ss_log "import pytest works: $(grep '^pytest ' "$pt_out" | tail -1)"
            break
        fi
        miss=$(kv missing_module "$pt_out")
        if [ "$pt_round" = 0 ] && [ "$miss" = pytest ]; then
            # shellcheck disable=SC2086  # PYTEST_PKGS is a list of names
            env_run "$ENV_NAME" python -m pip install --no-deps $PYTEST_PKGS \
                || ss_die "pip install --no-deps $PYTEST_PKGS into $ENV_NAME failed (output above; PyPI reachable? see the network check)"
        elif [ -n "$miss" ] && [ "$miss" != pytest ] && [ "$pt_round" -lt 4 ] \
            && case " $pt_tried " in *" $miss "*) false ;; *) true ;; esac; then
            ss_log "'import pytest' still fails: module '$miss' is missing; installing exactly that, --no-deps"
            env_run "$ENV_NAME" python -m pip install --no-deps "$miss" \
                || ss_die "'import pytest' in $ENV_NAME fails: module '$miss' is missing, and 'pip install --no-deps $miss' failed (output above; the package may have another name). Install the package that provides '$miss' with: conda run -n $ENV_NAME python -m pip install --no-deps <package>, then rerun"
            pt_tried="$pt_tried $miss"
            grep -qxF "$miss" "$pytest_extra" || echo "$miss" >> "$pytest_extra"
        else
            sed 's/^/    /' "$pt_out"
            if [ -n "$miss" ]; then
                ss_die "'import pytest' in $ENV_NAME still fails: module '$miss' is missing (tried:${pt_tried:- nothing}). Install it by hand with: conda run -n $ENV_NAME python -m pip install --no-deps <the package that provides $miss>, then rerun"
            fi
            ss_die "'import pytest' in $ENV_NAME fails, and not because of a missing module (traceback above)"
        fi
        pt_round=$((pt_round + 1))
    done
fi

# =============================================================================================
VERIFY_FAIL=0
ss_step "verify: import flashinfer and serve.rl.engine (PYTHONPATH=repo, cwd outside the repo)"
probe_env "$ENV_NAME" "$ENV_PROBE" || ss_die "probe of $ENV_NAME failed"
cp "$ENV_PROBE" "$REPORT_DIR/env.probe.txt"
ENV_PREFIX=$(kv sys_prefix "$ENV_PROBE")
mkdir -p "$BUILD_DIR/run"
if (cd "$BUILD_DIR/run" && env_run "$ENV_NAME" env PYTHONPATH="$REPO_ROOT" python "$PY_DIR/verify_import.py" \
        "$REPO_ROOT" "$FI_DIR" "$FLASHINFER_EDITABLE"); then
    ss_log "imports OK"
else
    VERIFY_FAIL=1
    ss_warn "IMPORT CHECK FAILED (output above). This is the real proof that the engine can run in $ENV_NAME."
fi

ss_step "verify: RAFT headers and librmm.so where serve/attention/backend.py looks"
if env_run "$ENV_NAME" python "$PY_DIR/raft_check.py"; then
    ss_log "RAFT / rmm files OK"
else
    VERIFY_FAIL=1
    ss_warn "RAFT CHECK FAILED"
fi

ss_step "verify: JIT-compile and load the RAFT top-k op (batch_topk_csr)"
if [ "$SKIP_JIT_SMOKE" = 1 ]; then
    ss_log "skipped (SKIP_JIT_SMOKE=1); the engine's first start compiles it instead"
else
    ss_log "compiles serve/attention/csrc/ops.cu with nvcc against RAFT/rmm (no GPU memory; cache under ~/.cache/flashinfer of $(id -un 2>/dev/null || echo this user))"
    if (cd "$BUILD_DIR/run" && env_run "$ENV_NAME" env PYTHONPATH="$REPO_ROOT" python "$PY_DIR/jit_smoke.py") 2>&1 \
        | tee "$REPORT_DIR/jit_smoke.txt"; then
        ss_log "top-k op compiled and loaded"
    else
        VERIFY_FAIL=1
        ss_warn "JIT SMOKE FAILED (output above, saved to $REPORT_DIR/jit_smoke.txt): building or loading batch_topk_csr raised."
        ss_warn "If the traceback names a missing librmm.so / librapids_logger.so, the dlopen inside python could not resolve it (backend.py:341 passes only -L, no rpath): report it;"
        ss_warn "if nvcc fails inside raft/rmm/cuda headers, compare the toolkit's CCCL with the one RAFT expects (include/rapids)."
    fi
fi

ss_step "verify: pytest imports in $ENV_NAME (the gate runner's test stages need it)"
if (cd "$BUILD_DIR/run" && env_run "$ENV_NAME" env PYTHONDONTWRITEBYTECODE=1 python -B "$PY_DIR/pytest_check.py"); then
    ss_log "pytest OK"
else
    VERIFY_FAIL=1
    ss_warn "PYTEST CHECK FAILED: 'import pytest' fails in $ENV_NAME (output above); run_all_gates.sh would skip the CPU and GPU test stages. A build run (without --verify) installs it."
fi

ss_step "verify: package drift $BASE_ENV -> $ENV_NAME"
env_q "$BASE_ENV" env PYTHONDONTWRITEBYTECODE=1 python -B -m pip list --format=freeze > "$REPORT_DIR/base.pip-list.txt"
env_q "$ENV_NAME" python -m pip list --format=freeze > "$REPORT_DIR/env.pip-list.txt"
env_q "$BASE_ENV" env PYTHONDONTWRITEBYTECODE=1 python -B -m pip freeze > "$REPORT_DIR/base.pip-freeze.txt" || true
env_q "$ENV_NAME" python -m pip freeze > "$REPORT_DIR/env.pip-freeze.txt" || true
"$CONDA_BIN" list -n "$BASE_ENV" --export > "$REPORT_DIR/base.conda-export.txt" 2>/dev/null || true
"$CONDA_BIN" list -n "$ENV_NAME" --export > "$REPORT_DIR/env.conda-export.txt" 2>/dev/null || true
crit=$(printf '%s' "$CRITICAL_PKGS" | tr ' ' ',')
pt_extra=""
if [ -f "$REPORT_DIR/pytest_extra.txt" ]; then pt_extra=$(tr '\n' ' ' < "$REPORT_DIR/pytest_extra.txt"); fi
# shellcheck disable=SC2086  # split the name lists on purpose
expd=$(printf '%s\n' $EXPECTED_PKGS $pt_extra | tr '\n' ',')
drift_rc=0
env_q "$ENV_NAME" python "$PY_DIR/drift.py" "$REPORT_DIR/base.pip-list.txt" "$REPORT_DIR/env.pip-list.txt" \
    "$crit" "$expd" "$REPORT_DIR/drift.md" > "$PY_DIR/drift.out" || drift_rc=$?
cat "$PY_DIR/drift.out"
if [ "$drift_rc" = 3 ]; then
    VERIFY_FAIL=1
    ss_warn "DRIFT: a critical package (one of: $CRITICAL_PKGS) differs between $BASE_ENV and $ENV_NAME. The two arms' training paths would not be comparable."
elif [ "$drift_rc" != 0 ]; then
    VERIFY_FAIL=1
    ss_warn "drift report crashed (status $drift_rc)"
fi
n_unexp=$(sed -n 's/^drift: critical_bad=[0-9]* unexpected=\([0-9]*\)$/\1/p' "$PY_DIR/drift.out")
if [ "${n_unexp:-0}" != 0 ]; then
    ss_warn "$n_unexp UNEXPECTED package difference(s) (table above; this script installs only $EXPECTED_PKGS)"
    if [ "$STRICT_DRIFT" = 1 ]; then VERIFY_FAIL=1; fi
fi
# conda-level packages (not pip): report only
# (pypi_0: installed by pip; dev_0: an egg-link editable install. When pip replaces a conda package,
# e.g. setuptools in the prerequisites step, conda drops that package's record, so it shows here.)
grep -v -e '=pypi_0$' -e '=dev_0$' "$REPORT_DIR/base.conda-export.txt" 2>/dev/null | grep -v '^#' | sort > "$PY_DIR/c1" || true
grep -v -e '=pypi_0$' -e '=dev_0$' "$REPORT_DIR/env.conda-export.txt" 2>/dev/null | grep -v '^#' | sort > "$PY_DIR/c2" || true
if ! cmp -s "$PY_DIR/c1" "$PY_DIR/c2"; then
    # diff exits 1 here by definition (the files differ); under pipefail that must not stop the script
    c_names=$({ diff "$PY_DIR/c1" "$PY_DIR/c2" || true; } | sed -n 's/^[<>] \([^=]*\)=.*/\1/p' | sort -u | tr '\n' ' ')
    c_names=${c_names% }
    c_unexp=""
    for f in $c_names; do
        case " $EXPECTED_PKGS " in *" $f "*) ;; *) c_unexp="$c_unexp $f" ;; esac
    done
    if [ -n "$c_unexp" ]; then
        ss_warn "conda packages (non-pip) differ between the envs (not only packages this script replaces:$c_unexp):"
    else
        ss_log "conda packages (non-pip): only $c_names differ, which this script replaced with pip (expected):"
    fi
    diff "$PY_DIR/c1" "$PY_DIR/c2" | sed 's/^/    /' || true
else
    ss_log "conda packages (non-pip): identical"
fi
ss_log "report: $REPORT_DIR/drift.md (freeze files next to it)"

ss_step "verify: $BASE_ENV unchanged"
fingerprint_base "$PY_DIR/base.fingerprint.after"
if cmp -s "$REPORT_DIR/base.fingerprint.txt" "$PY_DIR/base.fingerprint.after"; then
    ss_log "$BASE_ENV package lists identical to the start of the build"
else
    diff "$REPORT_DIR/base.fingerprint.txt" "$PY_DIR/base.fingerprint.after" | sed 's/^/    /' || true
    ss_die "the package lists of $BASE_ENV CHANGED while this script ran. Stop and tell the team."
fi

if [ "$VERIFY_FAIL" != 0 ]; then
    ss_die "verification failed (see the WARNING lines above and $REPORT_DIR). Send back the full log."
fi
print_next
ss_done "$ENV_NAME built and verified (report: $REPORT_DIR)"
