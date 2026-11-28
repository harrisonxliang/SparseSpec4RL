"""CPU tests of integration/setup_env.sh and integration/smoke_reference.sh.

    PYTHONPATH=. python -m pytest integration/tests/test_setup_scripts.py -q

Both scripts are written for the A100 box (conda, CUDA) and cannot really run
here, so they are run for real on a temp copy of the repo files with stand-in
commands on PATH:

* ``conda``: a bash script over a fake conda root whose "envs" are
  venv-style directories (pyvenv.cfg + a symlink to this interpreter). Each
  env's site-packages holds its own copy of the pristine verl 0.5.0 package,
  a fake ``pip`` module (``--version``; ``install --no-deps [-e DIR | verl==X]``)
  and a ``.pth`` file that makes the third-party packages of the python
  running pytest (torch, transformers, ray, ...; verl and pip excluded)
  importable. ``create --clone`` copies the directory, ``run`` execs inside
  it, and a fake ``etc/profile.d/conda.sh`` defines the ``conda activate``
  shell function.
* ``git``: ``clone`` copies the pristine verl tree into the destination and
  commits it with the real git (so ``git status`` works on it); ``ls-remote``
  answers without network. Everything else goes to the real git.
* ``nvidia-smi``, ``python3`` (smoke only: the trainer, ``python3 -m
  verl.trainer.main_ppo``, is a stub that prints a realistic verl console log;
  every other python3 call is the real interpreter, so the import preflight,
  baseline/parse_timing.py and the gate 3d checker run for real).
* The real ``patch`` (the BSD patch of macOS here, GNU patch on Linux).

PATH is built from a symlink farm of the system tool directories minus git /
patch / python / conda, so each test decides exactly which tools exist.

Pristine verl 0.5.0 source: $SPARSESPEC_VERL_SRC (an unpacked verl package
directory) if set, otherwise the verl package installed in the running python
(reverse-patched first if it carries the SparseSpec patch); the tests skip if
its two patched files do not match the hashes recorded in setup_env.sh.

There is deliberately no ``__init__.py`` next to this file (see
serve/rl/tests/conftest.py for why).
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP = "integration/setup_env.sh"
SMOKE = "integration/smoke_reference.sh"
BASH = "/bin/bash" if Path("/bin/bash").exists() else shutil.which("bash")
FSDP_REL = Path("verl/workers/fsdp_workers.py")
TRAINER_REL = Path("verl/trainer/ppo/ray_trainer.py")
PYVER = f"python{sys.version_info[0]}.{sys.version_info[1]}"
REAL_GIT = shutil.which("git")
REAL_PATCH = shutil.which("patch")
DEFAULT_ENV = "verl-sparsespec-dev"

pytestmark = pytest.mark.skipif(BASH is None or REAL_PATCH is None, reason="needs bash and patch")


# ---------------------------------------------------------------------------
# session-wide building blocks
# ---------------------------------------------------------------------------
def _write_exe(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip())
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _recorded_hashes() -> dict:
    text = (REPO_ROOT / SETUP).read_text()
    fsdp = re.search(r"^SHA_FSDP_WORKERS=([0-9a-f]{64})$", text, flags=re.M)
    trainer = re.search(r"^SHA_RAY_TRAINER=([0-9a-f]{64})$", text, flags=re.M)
    assert fsdp and trainer, "setup_env.sh must record the sha256 of the two files the patch expects"
    return {FSDP_REL: fsdp.group(1), TRAINER_REL: trainer.group(1)}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_digest(root: Path) -> str:
    """Every file below root (path + content, symlinks by target), __pycache__ excluded."""
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames) + sorted(d for d in dirnames if os.path.islink(os.path.join(dirpath, d))):
            path = os.path.join(dirpath, name)
            h.update(os.path.relpath(path, root).encode() + b"\0")
            if os.path.islink(path):
                h.update(b"link:" + os.readlink(path).encode())
            elif os.path.isfile(path):
                with open(path, "rb") as fh:
                    h.update(fh.read())
    return h.hexdigest()


def _pristine_source() -> Path | None:
    env = os.environ.get("SPARSESPEC_VERL_SRC", "")
    if env and (Path(env) / "workers" / "fsdp_workers.py").is_file():
        return Path(env)
    spec = importlib.util.find_spec("verl")
    if spec is not None and spec.origin:
        return Path(spec.origin).parent
    return None


SITE_EXCLUDE = re.compile(r"^(verl|verl-.*\.dist-info|pip|pip-.*\.dist-info|__pycache__|__editable__.*|_virtualenv.*|.*\.pth)$")

PIP_MAIN = r'''
"""Fake pip for the setup_env.sh tests: --version, install --no-deps [-e DIR | verl==X]."""
import os
import shutil
import sys

SITE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(text):
    calls = os.environ.get("FAKE_CALLS")
    if calls:
        with open(calls, "a") as fh:
            fh.write(text + "\n")


def remove_verl():
    for name in os.listdir(SITE):
        if name == "verl" or (name.startswith("verl-") and name.endswith(".dist-info")) or name.startswith("__editable__.verl"):
            path = os.path.join(SITE, name)
            shutil.rmtree(path) if os.path.isdir(path) and not os.path.islink(path) else os.remove(path)


def write_dist(version, editable=None):
    dist = os.path.join(SITE, f"verl-{version}.dist-info")
    os.makedirs(dist)
    with open(os.path.join(dist, "METADATA"), "w") as fh:
        fh.write(f"Metadata-Version: 2.1\nName: verl\nVersion: {version}\n")
    with open(os.path.join(dist, "top_level.txt"), "w") as fh:
        fh.write("verl\n")
    if editable:
        with open(os.path.join(SITE, f"__editable__.verl-{version}.pth"), "w") as fh:
            fh.write(editable + "\n")


def user_install():
    """Like real pip: a `user = true` config (FAKE_PIP_USER_CONFIG) is overridden by PIP_USER=0;
    FAKE_PIP_FORCE_USER stands for any other reason pip would pick the user site."""
    if os.environ.get("FAKE_PIP_FORCE_USER"):
        return True
    return bool(os.environ.get("FAKE_PIP_USER_CONFIG")) and os.environ.get("PIP_USER", "") not in ("0", "false", "no", "off")


def user_site():
    return os.path.join(os.environ["HOME"], ".local", "lib", f"python{sys.version_info[0]}.{sys.version_info[1]}", "site-packages")


args = sys.argv[1:]
log("pip " + " ".join(args))
if args[:1] == ["install"]:
    log("pipenv PIP_USER=" + os.environ.get("PIP_USER", "<unset>"))
if args[:1] == ["--version"]:
    print(f"pip 24.0 from {SITE}/pip (python {sys.version_info[0]}.{sys.version_info[1]})")
    sys.exit(0)
if args[:2] == ["config", "list"]:
    if os.environ.get("FAKE_PIP_USER_CONFIG"):
        print("global.user='true'")
    sys.exit(0)
if args[:1] == ["install"] and os.environ.get("FAKE_PIP_REQUIRE_VENV") and \
        os.environ.get("PIP_REQUIRE_VIRTUALENV", "") not in ("0", "false", "no", "off"):
    # like real pip with `require-virtualenv = true` in a config file: a conda env is no virtualenv
    print("ERROR: Could not find an activated virtualenv (required).", file=sys.stderr)
    sys.exit(3)
if args[:1] == ["install"] and os.environ.get("PIP_TARGET"):  # real pip reads PIP_TARGET as --target
    SITE = os.environ["PIP_TARGET"]
    if "--root" not in args:
        os.makedirs(SITE, exist_ok=True)
if args[:1] == ["install"] and "--root" in args:  # setup_env.sh's pip_target probe: a local wheel
    import zipfile

    root = args[args.index("--root") + 1]
    dest = os.path.join(root, (user_site() if user_install() else SITE).lstrip("/"))
    with zipfile.ZipFile(args[-1]) as z:
        z.extractall(dest)
    print("Successfully installed " + os.path.basename(args[-1]))
    sys.exit(0)
if args[:1] == ["install"]:
    if user_install():  # where real pip would put it: the user site, shared by every env
        SITE = user_site()
        os.makedirs(SITE, exist_ok=True)
    if os.environ.get("FAKE_PIP_FAIL"):
        print("ERROR: Could not build editable verl (fake pip: injected failure)", file=sys.stderr)
        sys.exit(1)
    touch = os.environ.get("FAKE_PIP_TOUCH")
    if touch:  # simulates a write into another env (the tripwire test)
        with open(touch, "a") as fh:
            fh.write("\n# changed by an install\n")
    editable, reqs, rest = None, [], args[1:]
    while rest:
        item = rest.pop(0)
        if item == "-e":
            editable = os.path.realpath(rest.pop(0))
        elif not item.startswith("-"):
            reqs.append(item)
    if editable:
        with open(os.path.join(editable, "verl", "version", "version")) as fh:
            version = fh.read().strip()
        remove_verl()
        write_dist(version, editable)
        print(f"Successfully installed verl-{version}")
        sys.exit(0)
    if len(reqs) == 1 and reqs[0].startswith("verl=="):
        version = reqs[0].split("==", 1)[1]
        remove_verl()
        shutil.copytree(os.environ["FAKE_PIP_VERL_SOURCE"], os.path.join(SITE, "verl"))
        write_dist(version)
        print(f"Successfully installed verl-{version}")
        sys.exit(0)
print("fake pip: unsupported command: " + " ".join(args), file=sys.stderr)
sys.exit(2)
'''


@pytest.fixture(scope="session")
def tools(tmp_path_factory):
    """Pristine verl, a patched copy, the base-env template, the system-tool farm."""
    root = tmp_path_factory.mktemp("setup_scripts")
    source = _pristine_source()
    if source is None:
        pytest.skip("no verl package to build the fake envs from (set SPARSESPEC_VERL_SRC)")
    pristine_root = root / "pristine"
    shutil.copytree(source, pristine_root / "verl", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    patch_file = REPO_ROOT / "integration" / "verl-0.5.0-sparsespec.patch"
    if 'elif rollout_name == "sparsespec":' in (pristine_root / FSDP_REL).read_text():
        with open(patch_file) as fh:
            subprocess.run([REAL_PATCH, "-p1", "-R", "-s", "-f"], cwd=pristine_root, stdin=fh, check=True)
    for rel, expected in _recorded_hashes().items():
        if _sha(pristine_root / rel) != expected:
            pytest.skip(f"{source}/{rel.relative_to('verl')} is not the pristine verl 0.5.0 file")
    pristine = pristine_root / "verl"

    patched_root = root / "patched"
    shutil.copytree(pristine, patched_root / "verl")
    with open(patch_file) as fh:
        subprocess.run([REAL_PATCH, "-p1", "-s"], cwd=patched_root, stdin=fh, check=True)

    # third-party packages of this python, minus verl and pip, as symlink farms in sys.path order
    farms = []
    for i, entry in enumerate(p for p in sys.path if p.endswith("site-packages") and os.path.isdir(p)):
        farm = root / "deps" / str(i)
        farm.mkdir(parents=True)
        for name in os.listdir(entry):
            if not SITE_EXCLUDE.match(name):
                (farm / name).symlink_to(os.path.join(entry, name))
        farms.append(str(farm))

    base_exe = Path(getattr(sys, "_base_executable", sys.executable))
    template = root / "env-template"
    (template / "bin").mkdir(parents=True)
    (template / "pyvenv.cfg").write_text(
        f"home = {base_exe.parent}\ninclude-system-site-packages = false\n"
        f"version = {sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}\n"
    )
    (template / "bin" / "python").symlink_to(base_exe)
    (template / "bin" / "python3").symlink_to("python")
    site = template / "lib" / PYVER / "site-packages"
    site.mkdir(parents=True)
    shutil.copytree(pristine, site / "verl")
    dist = site / "verl-0.5.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: verl\nVersion: 0.5.0\n")
    (dist / "top_level.txt").write_text("verl\n")
    (site / "pip").mkdir()
    (site / "pip" / "__init__.py").write_text('__version__ = "24.0"\n')
    (site / "pip" / "__main__.py").write_text(PIP_MAIN)
    (site / "_ss_deps.pth").write_text("\n".join(farms) + "\n")
    probe = subprocess.run(
        [str(template / "bin" / "python"), "-c", "import sys, verl, torch; print(sys.prefix); print(verl.__file__)"],
        capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
    )  # fmt: skip
    if probe.returncode != 0 or os.path.realpath(probe.stdout.split()[0]) != os.path.realpath(template):
        pytest.skip(f"a venv-style fake env does not work with this interpreter: {probe.stdout}{probe.stderr}")

    # system tools without git / patch / python / pip / conda / nvidia-smi / free
    sysbin = root / "sysbin"
    sysbin.mkdir()
    skip = re.compile(r"^(git.*|patch|conda|python.*|pip.*|nvidia-smi|free|pydoc.*|2to3.*|idle.*)$")
    for d in ("/usr/bin", "/bin", "/usr/sbin", "/sbin"):
        if os.path.isdir(d):
            for name in os.listdir(d):
                if not skip.match(name) and not (sysbin / name).exists():
                    (sysbin / name).symlink_to(os.path.join(d, name))
    patchbin = root / "patchbin"
    patchbin.mkdir()
    (patchbin / "patch").symlink_to(REAL_PATCH)
    return SimpleNamespace(
        root=root, pristine=pristine, pristine_root=pristine_root, patched_root=patched_root,
        template=template, sysbin=sysbin, patchbin=patchbin,
    )  # fmt: skip


# ---------------------------------------------------------------------------
# per-test sandbox
# ---------------------------------------------------------------------------
FAKE_CONDA = r"""
#!/bin/bash
# fake conda for the setup_env.sh tests
ROOT=__ROOT__
echo "conda $*" >> "${FAKE_CALLS:-/dev/null}"
name_arg() { local prev=""; for a in "$@"; do [ "$prev" = "-n" ] && { echo "$a"; return; }; prev=$a; done; }
case "${1:-}" in
    --version) echo "conda 24.1.2" ;;
    info)
        if [ "${2:-}" = "--base" ]; then
            if [ -n "${FAKE_CONDA_INFO_FAIL:-}" ]; then echo "CondaError: injected info failure" >&2; exit 1; fi
            echo "$ROOT"
        else
            echo "active environment : ${CONDA_DEFAULT_ENV:-None}"
        fi ;;
    env)
        case "${2:-}" in
            list)
                echo "# conda environments:"; echo "#"
                printf '%-26s %s\n' base "$ROOT"
                for d in "$ROOT"/envs/*; do [ -d "$d" ] && printf '%-26s %s\n' "$(basename "$d")" "$d"; done ;;
            remove) rm -rf "$ROOT/envs/$(name_arg "$@")" ;;
            *) echo "fake conda: unsupported: $*" >&2; exit 2 ;;
        esac ;;
    create)
        name=$(name_arg "$@"); src=""; prev=""
        for a in "$@"; do [ "$prev" = "--clone" ] && src=$a; prev=$a; done
        [ -n "$src" ] || { echo "fake conda: only --clone is supported" >&2; exit 2; }
        dest="$ROOT/envs/$name"
        [ -e "$dest" ] && { echo "CondaValueError: prefix already exists: $dest" >&2; exit 1; }
        if [ -n "${FAKE_CONDA_CREATE_FAIL:-}" ]; then
            mkdir -p "$dest"; cp -R "$ROOT/envs/$src/bin" "$dest/"
            echo "CondaError: injected failure while cloning (disk full?)" >&2; exit 1
        fi
        echo "Source:      $ROOT/envs/$src"; echo "Destination: $dest"
        cp -R "$ROOT/envs/$src" "$dest" ;;
    remove) rm -rf "$ROOT/envs/$(name_arg "$@")" ;;
    run)
        shift; name=""
        while [ $# -gt 0 ]; do
            case "$1" in --no-capture-output) shift ;; -n) name=$2; shift 2 ;; *) break ;; esac
        done
        p="$ROOT/envs/$name"; [ -d "$p" ] || { echo "EnvironmentLocationNotFound: $name" >&2; exit 1; }
        CONDA_PREFIX=$p CONDA_DEFAULT_ENV=$name PATH="$p/bin:$PATH" exec "$@" ;;
    *) echo "fake conda: unsupported: $*" >&2; exit 2 ;;
esac
"""

FAKE_CONDA_SH = r"""
# fake etc/profile.d/conda.sh
export CONDA_EXE='__ROOT__/bin/conda'
conda() {
    if [ "${1:-}" = "activate" ]; then
        echo "conda activate $2" >> "${FAKE_CALLS:-/dev/null}"
        if [ -n "${FAKE_CONDA_ACTIVATE_FAIL:-}" ]; then echo "fake conda activate: injected failure" >&2; return 1; fi
        if [ ! -d "__ROOT__/envs/$2" ]; then echo "EnvironmentNameNotFound: $2" >&2; return 1; fi
        export CONDA_PREFIX="__ROOT__/envs/$2" CONDA_DEFAULT_ENV="$2" PATH="__ROOT__/envs/$2/bin:$PATH"
        return 0
    fi
    "$CONDA_EXE" "$@"
}
"""

FAKE_GIT = r"""
#!/bin/bash
# fake git for the setup_env.sh tests: clone / ls-remote without network
echo "git $*" >> "${FAKE_CALLS:-/dev/null}"
REAL_GIT=__REAL_GIT__
case "${1:-}" in
    ls-remote)
        if [ -n "${FAKE_GIT_OFFLINE:-}" ]; then
            echo "fatal: unable to access 'https://github.com/volcengine/verl.git/': Could not resolve host: github.com" >&2; exit 128
        fi
        printf '0123456789abcdef0123456789abcdef01234567\trefs/tags/v0.5.0\n' ;;
    clone)
        dest=${!#}
        if [ -e "$dest" ] && [ -n "$(ls -A "$dest")" ]; then
            echo "fatal: destination path '$dest' already exists and is not an empty directory." >&2; exit 128
        fi
        echo "Cloning into '$dest'..."
        mkdir -p "$dest/verl"
        if [ -n "${FAKE_GIT_CLONE_FAIL:-}" ]; then
            cp "$FAKE_VERL_PKG/__init__.py" "$dest/verl/"
            echo "error: RPC failed; curl 56 (injected)" >&2; echo "fatal: early EOF" >&2; exit 128
        fi
        rmdir "$dest/verl"; cp -R "$FAKE_VERL_PKG" "$dest/verl"
        printf 'from setuptools import setup\nsetup(name="verl")\n' > "$dest/setup.py"
        (cd "$dest" && "$REAL_GIT" init -q && "$REAL_GIT" add -A \
            && "$REAL_GIT" -c user.name=fake -c user.email=fake@example.com -c commit.gpgsign=false commit -q -m "verl v0.5.0" \
            && "$REAL_GIT" tag v0.5.0) >/dev/null || exit 1 ;;
    *) exec "$REAL_GIT" "$@" ;;
esac
"""


class Sandbox:
    def __init__(self, tools, tmp_path: Path, *, with_envs: bool = True):
        self.tools = tools
        self.tmp = tmp_path
        self.repo = tmp_path / "repo"
        for rel in ("integration/lib.sh", SETUP, SMOKE, "integration/verl-0.5.0-sparsespec.patch",
                    "baseline/run_seed_sparsespec.sh", "baseline/config/grpo_base.sh", "baseline/parse_timing.py"):  # fmt: skip
            (self.repo / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO_ROOT / rel, self.repo / rel)
        shutil.copytree(REPO_ROOT / "integration" / "gates", self.repo / "integration" / "gates",
                        ignore=shutil.ignore_patterns("__pycache__", "tests"))  # fmt: skip
        (self.repo / "serve").symlink_to(REPO_ROOT / "serve")
        self.conda_root = tmp_path / "conda"
        _write_exe(self.conda_root / "bin" / "conda", FAKE_CONDA.replace("__ROOT__", str(self.conda_root)))
        conda_sh = self.conda_root / "etc" / "profile.d" / "conda.sh"
        conda_sh.parent.mkdir(parents=True)
        conda_sh.write_text(FAKE_CONDA_SH.replace("__ROOT__", str(self.conda_root)))
        (self.conda_root / "envs").mkdir()
        if with_envs:
            shutil.copytree(tools.template, self.env_dir("verl-baseline"), symlinks=True)
        self.gitbin = tmp_path / "gitbin"
        _write_exe(self.gitbin / "git", FAKE_GIT.replace("__REAL_GIT__", REAL_GIT or "/usr/bin/git"))
        self.calls = tmp_path / "calls.txt"
        self.home = tmp_path / "home"
        self.home.mkdir()
        self.logs = tmp_path / "logs"
        self.verl_src = self.repo / "integration" / ".verl-0.5.0"
        self.baseline_digest = self.digest("verl-baseline") if with_envs else None

    # ---- envs
    def env_dir(self, name: str) -> Path:
        return self.conda_root / "envs" / name

    def site(self, name: str) -> Path:
        return self.env_dir(name) / "lib" / PYVER / "site-packages"

    def make_env(self, name: str, *, verl_version: str | None = "0.5.0", pip: bool = True) -> Path:
        shutil.copytree(self.env_dir("verl-baseline"), self.env_dir(name), symlinks=True)
        site = self.site(name)
        if verl_version is None:
            shutil.rmtree(site / "verl")
            shutil.rmtree(site / "verl-0.5.0.dist-info")
        elif verl_version != "0.5.0":
            dist = site / "verl-0.5.0.dist-info"
            (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: verl\nVersion: {verl_version}\n")
            dist.rename(site / f"verl-{verl_version}.dist-info")
        if not pip:
            shutil.rmtree(site / "pip")
        return self.env_dir(name)

    def digest(self, name: str) -> str:
        return _tree_digest(self.env_dir(name))

    def env_names(self) -> list[str]:
        return sorted(p.name for p in (self.conda_root / "envs").iterdir())

    # ---- running
    def environ(self, *, conda=True, git=True, patch=True, extra_path=(), **overrides) -> dict:
        path = list(extra_path)
        if conda:
            path.append(str(self.conda_root / "bin"))
        if git:
            path.append(str(self.gitbin))
        if patch:
            path.append(str(self.tools.patchbin))
        path.append(str(self.tools.sysbin))
        env = {
            "PATH": os.pathsep.join(path),
            "HOME": str(self.home),
            "TMPDIR": str(self.tmp),
            "LANG": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
            "SS_LOG_DIR": str(self.logs),
            "FAKE_CALLS": str(self.calls),
            "FAKE_VERL_PKG": str(self.tools.pristine),
            "FAKE_PIP_VERL_SOURCE": str(self.tools.pristine),
        }
        env.update({k: str(v) for k, v in overrides.items() if v is not None})
        return env

    def run(self, script: str, *args, env: dict | None = None, **env_kw) -> SimpleNamespace:
        env = env if env is not None else self.environ(**env_kw)
        start = len(self.read_calls())
        proc = subprocess.run([BASH, str(self.repo / script), *args], cwd=self.tmp, env=env,
                              capture_output=True, text=True, timeout=600)  # fmt: skip
        out = proc.stdout + proc.stderr
        lines = [line for line in out.splitlines() if line.strip()]
        return SimpleNamespace(code=proc.returncode, out=out, last=lines[-1] if lines else "",
                               calls=self.read_calls()[start:])  # fmt: skip

    def read_calls(self) -> list[str]:
        return self.calls.read_text().splitlines() if self.calls.exists() else []


def changing_calls(calls: list[str]) -> list[str]:
    """Calls that change something (an env, a checkout or an install). The pip_target probe
    (`pip install ... --root <tmp> <wheel>`) writes only into a temp dir and is not one."""
    pattern = re.compile(r"^(conda (create|remove|env remove)|git clone|pip install(?!.* --root ))")
    return [c for c in calls if pattern.match(c)]


def show(result) -> str:
    return f"exit {result.code}\n{result.out[-6000:]}"


@pytest.fixture
def sandbox(tools, tmp_path):
    box = Sandbox(tools, tmp_path)
    yield box
    for d in tmp_path.rglob("*"):  # undo chmods so pytest can clean up
        if d.is_dir() and not d.is_symlink():
            try:
                d.chmod(d.stat().st_mode | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRUSR)
            except OSError:
                pass


def assert_setup_ok(box: Sandbox, result, env: str = DEFAULT_ENV):
    assert result.code == 0, show(result)
    assert result.last.startswith("[setup_env] OK: "), show(result)
    assert f"'{env}' has patched verl 0.5.0" in result.last
    assert box.digest("verl-baseline") == box.baseline_digest, "the baseline env changed"
    out = result.out
    assert re.search(r"\[selftest\] verl\s+0\.5\.0 at ", out), out
    assert re.search(r"\[selftest\] patch hunk 1 \(worker\)\s+present", out), out
    assert re.search(r"\[selftest\] patch hunk 2 \(trainer\)\s+present", out), out
    assert re.search(r"\[selftest\] torch\s+\d", out) and re.search(r"\[selftest\] transformers\s+\d", out), out
    assert re.search(r"\[selftest\] vllm\s+", out), out
    engine = re.search(r"\[selftest\] serve\.rl\.engine\s+(.*)", out)
    assert engine, out
    if "does NOT import" in engine.group(1):  # this machine: no vllm / flashinfer; the reason's first line is shown
        assert re.search(r"does NOT import: \w+(Error|Exception)\b.*: .+", engine.group(1)), engine.group(1)
        assert "engine=reference only" in result.last
    assert "baseline tripwire" in out and "sha256 over the" in out and "files of" in out
    assert re.search(r"---- step \d+: self-test", out)


# ---------------------------------------------------------------------------
# both scripts: shell dialect, --help
# ---------------------------------------------------------------------------
BASH4_ONLY = re.compile(
    r"\bmapfile\b|\breadarray\b|declare\s+-[a-zA-Z]*[An]|local\s+-[a-zA-Z]*[An]|\$\{[A-Za-z_][A-Za-z0-9_]*(,,|\^\^|,|\^)\}"
    r"|\bwait\s+-n\b|\|&|;;&|\bcoproc\b|\$EPOCH|&>>"
)


@pytest.mark.parametrize("script", [SETUP, SMOKE])
def test_scripts_are_bash_3_2_compatible(script):
    proc = subprocess.run([BASH, "-n", str(REPO_ROOT / script)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    code = [line for line in (REPO_ROOT / script).read_text().splitlines() if not line.lstrip().startswith("#")]
    offending = [line for line in code if BASH4_ONLY.search(line)]
    assert not offending, offending
    assert "integration/lib.sh" in (REPO_ROOT / script).read_text() or "lib.sh" in (REPO_ROOT / script).read_text()
    assert re.search(r"^ss_init \w+", (REPO_ROOT / script).read_text(), flags=re.M)


@pytest.mark.parametrize("script", [SETUP, SMOKE])
def test_help_prints_usage_and_does_nothing(sandbox, script):
    result = sandbox.run(script, "--help")
    assert result.code == 0, show(result)
    assert "Usage" in result.out or "bash integration/" in result.out
    assert "--check" in result.out and "set -euo" not in result.out
    assert result.calls == [] and not sandbox.logs.exists()
    bad = sandbox.run(script, "--no-such-option")
    assert bad.code == 2 and "unknown" in bad.out


# ---------------------------------------------------------------------------
# setup_env.sh
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("how", ["flag", "env", "in_place"])
def test_check_mode_changes_nothing_and_prints_the_plan(sandbox, how):
    args, kw = (["--check"], {}) if how == "flag" else ([], {"CHECK_ONLY": "1"})
    if how == "in_place":
        args, kw = ["--check"], {"VERL_IN_PLACE": "1"}
    result = sandbox.run(SETUP, *args, **kw)
    assert result.code == 0, show(result)
    assert result.last.startswith("[setup_env] CHECK OK:") and "nothing was changed" in result.last, show(result)
    assert changing_calls(result.calls) == [], result.calls
    assert sandbox.env_names() == ["verl-baseline"]
    assert not sandbox.verl_src.exists()
    assert sandbox.digest("verl-baseline") == sandbox.baseline_digest
    out = result.out
    assert f"would create env '{DEFAULT_ENV}' as a clone of 'verl-baseline'" in out
    assert "never written: 'verl-baseline'" in out
    assert re.search(r"---- step 1: check every prerequisite", out) and re.search(r"---- step 2: plan", out)
    for expected in ("ok    conda:", "ok    conda info --base:", "'conda activate' is available", "base env 'verl-baseline' exists",
                     "has verl 0.5.0", "baseline tripwire (verl-baseline): sha256 over the"):  # fmt: skip
        assert expected in out, (expected, out)
    if how == "in_place":
        assert "would patch the verl 0.5.0 installed in" in out and "no git clone and no pip install" in out
        assert "pristine 0.5.0 ones the patch expects" in out
        assert not [c for c in result.calls if c.startswith("git ")]
    else:
        assert "would clone verl v0.5.0" in out and "would apply" in out and "python -m pip install --no-deps" in out
        assert "has tag v0.5.0" in out and "ok    pip in" in out and "write access to" in out
        assert "disk: 'verl-baseline' occupies" in out
    logs = list(sandbox.logs.glob("setup_env_*.log"))
    assert len(logs) == 1 and "CHECK OK" in logs[0].read_text()


def test_default_mode_end_to_end_then_rerun_is_a_no_op(sandbox):
    first = sandbox.run(SETUP)
    assert_setup_ok(sandbox, first)
    assert sandbox.env_names() == sorted([DEFAULT_ENV, "verl-baseline"])
    assert [c.split()[0:2] for c in changing_calls(first.calls)] == [["conda", "create"], ["git", "clone"], ["pip", "install"]]
    assert any(c.startswith("pip install --no-deps") and c.endswith(f"-e {os.path.realpath(sandbox.verl_src)}") or c.endswith(f"-e {sandbox.verl_src}") for c in first.calls), first.calls
    # the checkout is patched and the env imports it
    assert 'elif rollout_name == "sparsespec":' in (sandbox.verl_src / FSDP_REL).read_text()
    assert not (sandbox.site(DEFAULT_ENV) / "verl").exists()
    assert re.search(rf"\[selftest\] verl\s+0\.5\.0 at {re.escape(os.path.realpath(sandbox.verl_src))}/verl", first.out), first.out
    assert not Path(str(sandbox.verl_src) + ".partial").exists()
    # the baseline's own verl files are untouched and still pristine
    for rel, expected in _recorded_hashes().items():
        assert _sha(sandbox.site("verl-baseline") / rel) == expected
    log = next(sandbox.logs.glob("setup_env_*.log")).read_text()
    assert "[selftest] patch hunk 1 (worker)" in log and "[setup_env] OK:" in log

    again = sandbox.run(SETUP)
    assert_setup_ok(sandbox, again)
    assert changing_calls(again.calls) == [], again.calls
    assert "already patched (an earlier run got this far)" in again.out
    assert "already imports verl 0.5.0 from" in again.out and "nothing to install" in again.out

    check = sandbox.run(SETUP, "--check")
    assert check.code == 0 and "CHECK OK" in check.last, show(check)
    assert f"keep the existing env '{DEFAULT_ENV}'" in check.out and "nothing to install" in check.out


def test_in_place_mode_end_to_end_then_rerun(sandbox):
    result = sandbox.run(SETUP, VERL_IN_PLACE="1")
    assert_setup_ok(sandbox, result)
    assert not [c for c in result.calls if c.startswith(("git ", "pip install"))], result.calls
    assert not sandbox.verl_src.exists()
    site = sandbox.site(DEFAULT_ENV)
    assert 'elif rollout_name == "sparsespec":' in (site / FSDP_REL).read_text()
    assert 'startswith("sparsespec/")' in (site / TRAINER_REL).read_text()
    for rel, expected in _recorded_hashes().items():
        assert _sha(sandbox.site("verl-baseline") / rel) == expected
    again = sandbox.run(SETUP, VERL_IN_PLACE="1")
    assert_setup_ok(sandbox, again)
    assert "already applied" in again.out and changing_calls(again.calls) == []


def test_existing_env_is_only_given_the_patched_verl(sandbox):
    """Harrison's env: exists already; no clone, only the install."""
    sandbox.make_env("verl-sparsespec")
    result = sandbox.run(SETUP, "verl-sparsespec")
    assert_setup_ok(sandbox, result, env="verl-sparsespec")
    assert not [c for c in result.calls if c.startswith("conda create")]
    assert "keep the existing env 'verl-sparsespec'" in result.out


@pytest.mark.parametrize("target, base", [("verl-baseline", None), ("base", None), ("my-baseline", "my-baseline")])
def test_the_baseline_env_is_refused(sandbox, target, base):
    if base:
        sandbox.make_env(base)
    before = {name: sandbox.digest(name) for name in sandbox.env_names()}
    result = sandbox.run(SETUP, target, BASE_ENV=base)
    assert result.code == 1, show(result)
    assert f"refusing to modify '{target}'" in result.out
    assert result.last.startswith("[setup_env] FAILED"), show(result)
    assert changing_calls(result.calls) == [] and not [c for c in result.calls if c.startswith("conda")]
    assert {name: sandbox.digest(name) for name in sandbox.env_names()} == before
    with_check = sandbox.run(SETUP, "--check", target, BASE_ENV=base)
    assert with_check.code == 1 and "refusing to modify" in with_check.out


def _prepare_precondition(box: Sandbox, case: str) -> dict:
    """Arrange one missing prerequisite; returns environ() keyword arguments."""
    if case == "no_conda":
        return {"conda": False}
    if case == "conda_info_fails":
        return {"FAKE_CONDA_INFO_FAIL": "1"}
    if case == "no_conda_sh":
        (box.conda_root / "etc" / "profile.d" / "conda.sh").rename(box.tmp / "conda.sh.away")
        return {}
    if case == "no_base_env":
        return {"BASE_ENV": "verl-baseline-typo"}
    if case == "no_git":
        return {"git": False}
    if case == "offline":
        return {"FAKE_GIT_OFFLINE": "1"}
    if case == "no_patch":
        return {"patch": False}
    if case == "pip_broken":
        box.make_env(DEFAULT_ENV, pip=False)
        return {}
    if case == "env_lacks_verl":
        box.make_env(DEFAULT_ENV, verl_version=None)
        return {}
    if case == "wrong_verl":
        box.make_env(DEFAULT_ENV, verl_version="0.4.1")
        return {}
    if case == "pip_user_target":
        return {"FAKE_PIP_FORCE_USER": "1"}
    if case == "checkout_dir_not_writable":
        ro = box.tmp / "readonly"
        ro.mkdir()
        ro.chmod(0o555)
        return {"VERL_SRC": str(ro / "verl-src")}
    if case == "half_made_checkout":
        box.verl_src.mkdir(parents=True)
        (box.verl_src / "setup.py").write_text("# an interrupted clone\n")
        return {}
    if case == "checkout_missing_file":
        subprocess.run([str(box.gitbin / "git"), "clone", "x", str(box.verl_src)], env=box.environ(), check=True,
                       capture_output=True)  # fmt: skip
        (box.verl_src / "verl" / "protocol.py").unlink()
        return {}
    if case == "checkout_edited":
        subprocess.run([str(box.gitbin / "git"), "clone", "x", str(box.verl_src)], env=box.environ(), check=True,
                       capture_output=True)  # fmt: skip
        path = box.verl_src / FSDP_REL
        path.write_text(path.read_text().replace('log_gpu_memory_usage("After building sharding manager"', 'log_gpu_memory_usage("edited"'))
        return {}
    raise AssertionError(case)


def _undo_precondition(box: Sandbox, case: str, result) -> dict:
    """Apply the fix the script printed (or the obvious one); returns the environ() kwargs for the re-run."""
    if case == "no_conda_sh":
        (box.tmp / "conda.sh.away").rename(box.conda_root / "etc" / "profile.d" / "conda.sh")
    elif case in ("pip_broken", "wrong_verl"):
        shutil.rmtree(box.env_dir(DEFAULT_ENV))  # the fix the message offers: remove it and re-run
    elif case == "env_lacks_verl":
        fix = re.search(r"fix: (conda run -n \S+ python -m pip install --no-deps verl==0\.5\.0)", result.out)
        assert fix, result.out
        proc = subprocess.run(fix.group(1), shell=True, env=box.environ(), capture_output=True, text=True, cwd=box.tmp)
        assert proc.returncode == 0, proc.stdout + proc.stderr
    elif case == "checkout_dir_not_writable":
        (box.tmp / "readonly").chmod(0o755)
        return {"VERL_SRC": str(box.tmp / "readonly" / "verl-src")}
    elif case in ("half_made_checkout", "checkout_missing_file", "checkout_edited"):
        assert f"rm -rf '{box.verl_src}'" in result.out, result.out
        shutil.rmtree(box.verl_src)
    return {}


PRECONDITIONS = {
    "no_conda": ["conda not found (not on PATH, CONDA_EXE unset)", "condabin"],
    "conda_info_fails": ["'conda info --base' failed", "CondaError: injected info failure"],
    "no_conda_sh": ["etc/profile.d/conda.sh does not exist"],
    "no_base_env": [f"neither '{DEFAULT_ENV}' nor the base env 'verl-baseline-typo' exists", "baseline/setup_env.sh"],
    "no_git": ["git is not installed", "VERL_IN_PLACE=1 bash integration/setup_env.sh"],
    "offline": ["cannot reach https://github.com/volcengine/verl.git", "Could not resolve host", "VERL_IN_PLACE=1"],
    "no_patch": ["'patch' not found on PATH", "install GNU patch"],
    "pip_broken": ["'python -m pip' does not work in", "No module named pip", f"conda install -y -n {DEFAULT_ENV} pip"],
    "env_lacks_verl": [f"'{DEFAULT_ENV}' has no verl installed",
                       f"fix: conda run -n {DEFAULT_ENV} python -m pip install --no-deps verl==0.5.0"],
    "wrong_verl": [f"'{DEFAULT_ENV}' has verl 0.4.1, not 0.5.0", f"conda run -n {DEFAULT_ENV} python -m pip install --no-deps verl==0.5.0"],
    "checkout_dir_not_writable": ["no write access to", "where the checkout", "set VERL_SRC=<a writable dir>"],
    "pip_user_target": ["would install into", "/.local/lib/", "outside the env", "pip config list"],
    "half_made_checkout": ["contains no verl/ package: not a finished verl checkout", "rm -rf"],
    "checkout_missing_file": ["has changes besides the two patched files", "verl/protocol.py"],
    "checkout_edited": ["the patch neither applies to nor is already applied in", "sha256", "(pristine verl 0.5.0 wheel: "],
}  # fmt: skip
FULL_RERUN = {"env_lacks_verl", "half_made_checkout", "checkout_dir_not_writable"}


@pytest.mark.parametrize("case", sorted(PRECONDITIONS))
def test_each_missing_prerequisite_is_named_and_changes_nothing(sandbox, case):
    kw = _prepare_precondition(sandbox, case)
    envs_before = {name: sandbox.digest(name) for name in sandbox.env_names()}
    src_before = _tree_digest(sandbox.verl_src) if sandbox.verl_src.exists() else None
    for args in ([], ["--check"]):
        result = sandbox.run(SETUP, *args, **kw)
        assert result.code == 1, show(result)
        for text in PRECONDITIONS[case]:
            assert text in result.out, (text, show(result))
        assert re.search(r"FAIL  .*\n.*fix: ", result.out), show(result)  # every failure comes with a fix line
        expected_last = "[setup_env] CHECK FAILED" if args else "[setup_env] FAILED"
        assert result.last.startswith(expected_last) or (not args and "FAILED" in result.last), show(result)
        assert "nothing was changed" in result.out
        assert changing_calls(result.calls) == [], result.calls
        assert {name: sandbox.digest(name) for name in sandbox.env_names()} == envs_before
        assert (_tree_digest(sandbox.verl_src) if sandbox.verl_src.exists() else None) == src_before
        assert not Path(str(sandbox.verl_src) + ".partial").exists()
    # with the cause removed, the next run is not blocked by anything the failed run left
    rerun_kw = _undo_precondition(sandbox, case, result)
    if case in FULL_RERUN:
        assert_setup_ok(sandbox, sandbox.run(SETUP, **rerun_kw))
    else:
        check = sandbox.run(SETUP, "--check", **rerun_kw)
        assert check.code == 0 and check.last.startswith("[setup_env] CHECK OK"), show(check)


def test_failures_during_the_run_leave_a_state_the_next_run_completes(sandbox):
    """create fails -> clone fails -> pip fails -> activation fails -> success, each run getting further."""
    partial = Path(str(sandbox.verl_src) + ".partial")

    r1 = sandbox.run(SETUP, FAKE_CONDA_CREATE_FAIL="1")
    assert r1.code == 1 and r1.last.startswith("[setup_env] FAILED"), show(r1)
    assert "conda create --clone failed" in r1.out and "removing the half-made env" in r1.out
    assert "a re-run starts the clone again" in r1.out
    assert sandbox.env_names() == ["verl-baseline"], "the half-made clone must be removed"

    r2 = sandbox.run(SETUP, FAKE_GIT_CLONE_FAIL="1")
    assert r2.code == 1 and r2.last.startswith("[setup_env] FAILED"), show(r2)
    assert "git clone failed" in r2.out and "early EOF" in r2.out and "was removed" in r2.out
    assert sandbox.env_names() == sorted([DEFAULT_ENV, "verl-baseline"]) and not partial.exists() and not sandbox.verl_src.exists()

    partial.mkdir()  # as if a clone had been killed before this script could clean up
    (partial / "junk").write_text("x")
    r3 = sandbox.run(SETUP, FAKE_PIP_FAIL="1")
    assert r3.code == 1 and r3.last.startswith("[setup_env] FAILED"), show(r3)
    assert "left over from an interrupted clone" in r3.out
    assert "pip install failed" in r3.out and "injected failure" in r3.out
    assert "stays patched; a re-run repeats only the install" in r3.out
    assert f"now imports verl 0.5.0 from {os.path.realpath(sandbox.site(DEFAULT_ENV))}/verl" in r3.out
    assert 'elif rollout_name == "sparsespec":' in (sandbox.verl_src / FSDP_REL).read_text()
    assert (sandbox.site(DEFAULT_ENV) / "verl").is_dir() and not partial.exists()

    r4 = sandbox.run(SETUP, FAKE_CONDA_ACTIVATE_FAIL="1")
    assert r4.code == 1 and r4.last.startswith("[setup_env] FAILED"), show(r4)
    assert f"'conda activate {DEFAULT_ENV}' failed" in r4.out and "injected failure" in r4.out

    r5 = sandbox.run(SETUP)
    assert_setup_ok(sandbox, r5)
    assert "already patched (an earlier run got this far)" in r5.out
    assert [c.split()[0:2] for c in changing_calls(r5.calls)] == [["pip", "install"]]


def test_pip_user_config_is_overridden_and_the_install_target_checked(sandbox):
    """A pip config `user = true` would send the patched verl into ~/.local (which verl-baseline
    imports too); the script exports PIP_USER=0 for every pip call and checks pip's real target."""
    text = (REPO_ROOT / SETUP).read_text()
    assert re.search(r"^export PIP_USER=0\b", text, flags=re.M)
    assert not re.search(r"^[^#]*PYTHONNOUSERSITE", text, flags=re.M), "PYTHONNOUSERSITE must not be set"
    result = sandbox.run(SETUP, FAKE_PIP_USER_CONFIG="1")
    assert_setup_ok(sandbox, result)
    installs = [c for c in result.calls if c.startswith("pipenv ")]
    assert len(installs) >= 2 and set(installs) == {"pipenv PIP_USER=0"}, result.calls  # probe(s) + the install
    assert "installs into its own prefix" in result.out and "pip installs into " in result.out
    assert not (sandbox.home / ".local").exists()
    # the same config without the export: what the fake pip (like real pip) would have done
    env = sandbox.environ(FAKE_PIP_USER_CONFIG="1")
    probe = subprocess.run(
        [str(sandbox.env_dir(DEFAULT_ENV) / "bin" / "python"), "-m", "pip", "install", "--no-deps", "-e", str(sandbox.verl_src)],
        env=env, capture_output=True, text=True,
    )  # fmt: skip
    assert probe.returncode == 0 and (sandbox.home / ".local").exists(), probe.stdout + probe.stderr


def test_require_virtualenv_config_and_caller_pip_target_do_not_reach_pip(sandbox):
    """A pip config `require-virtualenv = true` would refuse every install into a conda env, and a
    PIP_TARGET / PIP_PREFIX exported by the caller would redirect the install (pip reads them as
    --target / --prefix): the script exports PIP_REQUIRE_VIRTUALENV=0 and unsets the other two."""
    text = (REPO_ROOT / SETUP).read_text()
    assert re.search(r"^export PIP_USER=0 PIP_REQUIRE_VIRTUALENV=0$", text, flags=re.M)
    assert re.search(r"^unset PIP_TARGET PIP_PREFIX$", text, flags=re.M)
    assert not re.search(r"^\s*PIP_TARGET=", text, flags=re.M), "the script's own variable must not be pip's PIP_TARGET"
    elsewhere = sandbox.tmp / "elsewhere"
    result = sandbox.run(SETUP, FAKE_PIP_REQUIRE_VENV="1", PIP_REQUIRE_VIRTUALENV="1",
                         PIP_TARGET=elsewhere, PIP_PREFIX=elsewhere)  # fmt: skip
    assert_setup_ok(sandbox, result)
    assert "unsetting PIP_TARGET/PIP_PREFIX" in result.out
    assert "installs into its own prefix" in result.out and not elsewhere.exists()
    # negative controls: the fake pip, like real pip, honours both without the script
    py = str(sandbox.env_dir(DEFAULT_ENV) / "bin" / "python")
    cmd = [py, "-m", "pip", "install", "--no-deps", "-e", str(sandbox.verl_src)]
    refused = subprocess.run(cmd, env=sandbox.environ(FAKE_PIP_REQUIRE_VENV="1"), capture_output=True, text=True)
    assert refused.returncode == 3 and "activated virtualenv" in refused.stderr
    redirected = subprocess.run(cmd, env=sandbox.environ(PIP_TARGET=elsewhere), capture_output=True, text=True)
    assert redirected.returncode == 0 and elsewhere.exists(), redirected.stdout + redirected.stderr


def test_stale_partial_clone_is_removed_whatever_the_checkout_state(sandbox):
    assert_setup_ok(sandbox, sandbox.run(SETUP))
    partial = Path(str(sandbox.verl_src) + ".partial")
    partial.mkdir()
    (partial / "junk").write_text("x")
    second = sandbox.run(SETUP)
    assert_setup_ok(sandbox, second)
    assert "will remove the leftover" in second.out and f"removing {partial}" in second.out
    assert not partial.exists()
    third = sandbox.run(SETUP)
    assert_setup_ok(sandbox, third)
    assert "leftover" not in third.out and "left over" not in third.out


def test_tripwire_fires_if_the_baseline_changes(sandbox):
    victim = sandbox.site("verl-baseline") / TRAINER_REL
    result = sandbox.run(SETUP, FAKE_PIP_TOUCH=str(victim))
    assert result.code == 1, show(result)
    assert "changed while this script ran" in result.out and "Stop and tell the team" in result.out
    assert "the BASELINE env changed" in result.out
    assert re.search(r"what is hashed: sha256 over the \d+ files of .*verl-baseline/lib/.*/verl", result.out), result.out
    before, after = re.search(r"before: (\S+)", result.out).group(1), re.search(r"after:  (\S+)", result.out).group(1)
    assert before != after and len(before) == 64


def test_selftest_failure_ends_with_a_failed_line(sandbox):
    """A broken repo checkout (serve/ missing): the env is set up, the self-test says what is wrong."""
    (sandbox.repo / "serve").unlink()
    result = sandbox.run(SETUP)
    assert result.code == 1, show(result)
    assert re.search(r"\[selftest\] serve\.rl\.verl_rollout\s+does NOT import: ModuleNotFoundError: No module named 'serve'", result.out)
    assert "[selftest] PROBLEM" in result.out and result.last.startswith("[setup_env] FAILED"), show(result)


# ---------------------------------------------------------------------------
# smoke_reference.sh
# ---------------------------------------------------------------------------
STUB_TRAINER = r'''
"""Stand-in for `python3 -m verl.trainer.main_ppo`: prints a verl 0.5.0-style console log."""
import os
import pprint
import sys

argv = open(os.environ["FAKE_ARGV"]).read().splitlines()
values = {a.split("=", 1)[0].lstrip("+"): a.split("=", 1)[1] for a in argv}
ckpt = values["trainer.default_local_dir"]
with open(os.environ["FAKE_CALLS"], "a") as fh:
    fh.write(f"trainer ckpt_dir_exists={int(os.path.exists(ckpt))}\n")
steps = int(values["trainer.total_training_steps"])
mode = os.environ.get("FAKE_TRAINER_MODE", "ok")
print("[validate_config] All configuration checks passed successfully!", flush=True)
print("(TaskRunner pid=4242) Size of train dataloader: 937, Size of val dataloader: 1")
print(f"(TaskRunner pid=4242) Total training steps: {steps}")
print("(WorkerDict pid=4301) SparseSpecRollout: engine=reference, mem_fraction=0.45, max_batch_size=64")
print("(TaskRunner pid=4242) Training from scratch")
for step in range(1, steps + 1):
    if mode == "crash" and step == 2:
        print("(TaskRunner pid=4242) Traceback (most recent call last):")
        print('(TaskRunner pid=4242)   File "verl/trainer/ppo/ray_trainer.py", line 1173, in fit')
        print("(TaskRunner pid=4242) ray.exceptions.RayTaskError(OutOfMemoryError): torch.OutOfMemoryError: CUDA out of memory.")
        print("Error executing job with overrides: [...]", file=sys.stderr)
        sys.exit(1)
    data = {
        "training/global_step": step, "critic/score/mean": 0.0 if step == 1 else 0.125,
        "response_length/mean": 411.5, "actor/pg_loss": -0.0012,
        "timing_s/gen": 30.5 + step, "timing_s/generate_sequences": 29.0, "timing_s/update_actor": 8.25,
        "timing_s/step": 50.0 + step, "sparsespec/accept_len_mean": 0.0, "sparsespec/accept_rate": 0.0,
        "sparsespec/num_cycles": 0.0, "sparsespec/weight_sync_s": 0.85, "sparsespec/wake_s": 0.0,
        "sparsespec/graph_capture_s": 0.0, "sparsespec/gen_s": 28.5, "sparsespec/sleep_s": 0.02,
    }  # fmt: skip
    print(f"Training Progress: {100 * step // steps}%| {step}/{steps}")
    print("(TaskRunner pid=4242) " + " - ".join([f"step:{step}"] + [f"{k}:{pprint.pformat(v)}" for k, v in data.items()]), flush=True)
'''

FAKE_NVIDIA_SMI = r"""
#!/bin/bash
echo "nvidia-smi $*" >> "${FAKE_CALLS:-/dev/null}"
case "$*" in
    *--query-gpu=*)
        if [ -n "${FAKE_NVSMI_FAIL:-}" ]; then
            echo "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver." >&2; exit 9
        fi
        echo "0, GPU-aaaa1111, NVIDIA A100-SXM4-40GB, 40960, $((40960 - ${FAKE_GPU_FREE:-40400})), ${FAKE_GPU_FREE:-40400}"
        [ -n "${FAKE_GPU1_FREE:-}" ] && echo "1, GPU-bbbb2222, NVIDIA A100-SXM4-40GB, 40960, $((40960 - FAKE_GPU1_FREE)), $FAKE_GPU1_FREE"
        exit 0 ;;
    *--query-compute-apps=*) echo "pid, process_name, used_gpu_memory [MiB]" ;;
    *) echo "| NVIDIA-SMI 550.54.15   Driver Version: 550.54.15   CUDA Version: 12.4 |" ;;
esac
"""


class SmokeBox:
    def __init__(self, box: Sandbox):
        self.box = box
        tmp = box.tmp
        self.shim = tmp / "smokebin"
        self.argv = tmp / "trainer_argv.txt"
        stub = tmp / "stub_trainer.py"
        stub.write_text(STUB_TRAINER)
        _write_exe(self.shim / "python3", f"""
            #!/bin/bash
            echo "python3 $*" >> "${{FAKE_CALLS:-/dev/null}}"
            if [ "${{1:-}}" = "-m" ] && [ "${{2:-}}" = "verl.trainer.main_ppo" ]; then
                shift 2
                printf '%s\\n' "$@" > "$FAKE_ARGV"
                exec "{sys.executable}" "{stub}"
            fi
            exec "{sys.executable}" "$@"
            """)
        _write_exe(self.shim / "nvidia-smi", FAKE_NVIDIA_SMI)
        self.data = tmp / "data"
        self.data.mkdir()
        for name in ("math_l3to5_train.parquet", "math500_test.parquet"):
            (self.data / name).write_bytes(b"PAR1")
        self.ckpts = tmp / "ckpts"
        self.ckpt = self.ckpts / "reference_plumbing"
        (self.ckpt / "global_step_3").mkdir(parents=True)
        (self.ckpt / "latest_checkpointed_iteration.txt").write_text("3")
        (self.ckpts / "vllm_baseline_seed0" / "hf").mkdir(parents=True)
        self.results = box.repo / "baseline" / "results"
        self.csv = self.results / "reference_plumbing.timing.csv"

    def run(self, *args, pythonpath=None, nvidia=True, **kw):
        extra = [str(self.shim)] if nvidia else []
        env = self.box.environ(conda=False, git=False, patch=False, extra_path=extra, **kw)
        if not nvidia:  # python3 shim without nvidia-smi
            nv_less = self.box.tmp / "smokebin_no_nvsmi"
            if not nv_less.exists():
                nv_less.mkdir()
                (nv_less / "python3").symlink_to(self.shim / "python3")
            env["PATH"] = str(nv_less) + os.pathsep + env["PATH"]
        env.update({
            "PYTHONPATH": str(pythonpath or self.box.tools.patched_root),
            "DATA_DIR": str(self.data), "CKPT_ROOT": str(self.ckpts), "FAKE_ARGV": str(self.argv),
            "HF_HUB_OFFLINE": "1",
        })  # fmt: skip
        return self.box.run(SMOKE, *args, env=env)


@pytest.fixture
def smoke(tools, tmp_path):
    return SmokeBox(Sandbox(tools, tmp_path, with_envs=False))


def _trainer_called(result) -> bool:
    return any(c.startswith("python3 -m verl.trainer.main_ppo") for c in result.calls)


def test_smoke_happy_path(smoke):
    smoke.results.mkdir(parents=True)
    smoke.csv.write_text("step,stale\n1,1\n")  # from an earlier attempt: must not survive
    result = smoke.run()
    assert result.code == 0, show(result)
    assert result.last.startswith("[smoke_reference] PASS: trainer exit 0, gate 3d PASS, 3 steps"), show(result)
    for where in (f"log: {smoke.box.repo}/baseline/logs/reference_plumbing.log", f"CSV: {smoke.csv}", "gate_d_smoke_reference_plumbing.md"):
        assert where in result.last, (where, result.last)
    # own checkpoint dir removed BEFORE the trainer started; nothing else touched
    assert "trainer ckpt_dir_exists=0" in result.calls
    assert not smoke.ckpt.exists() and (smoke.ckpts / "vllm_baseline_seed0" / "hf").is_dir()
    assert "stale" not in smoke.csv.read_text() and smoke.csv.read_text().startswith("step,")
    # the full override list is in the log, the reductions last
    argv = smoke.argv.read_text().splitlines()
    assert argv[-4:] == ["data.train_batch_size=8", "actor_rollout_ref.actor.ppo_mini_batch_size=8",
                         "data.max_response_length=512", "trainer.save_freq=-1"]  # fmt: skip
    assert "+actor_rollout_ref.rollout.sparsespec.engine=reference" in argv
    for i, item in enumerate(argv, start=1):
        assert re.search(rf"\[smoke_reference\]\s+{i}  {re.escape(item)}$", result.out, flags=re.M), item
    assert "8 prompts x n=8 = 64 sequences per step" in result.out and "1 mini-batch per step" in result.out
    gate = [c for c in result.calls if "gate_d_smoke_check" in c]
    assert gate == ["python3 -m integration.gates.gate_d_smoke_check --exp-name reference_plumbing --steps 3 --engine reference"]
    gate_json = smoke.results / "sparsespec_checks" / "gate_d_smoke_reference_plumbing.json"
    assert '"pass": true' in gate_json.read_text()
    assert "needs about 30720 MiB free" in result.out and "ok    GPU 0 has 40400 MiB free" in result.out
    for text in ("ok    patch hunk 1", "ok    patch hunk 2", "ok    serve.rl.verl_rollout imports", "ok    data:"):
        assert text in result.out, text
    log = next(smoke.box.logs.glob("smoke_reference_*.log")).read_text()
    assert "PASS: trainer exit 0" in log and "  1  " in log


def test_smoke_check_mode_runs_and_removes_nothing(smoke):
    result = smoke.run("--check")
    assert result.code == 0, show(result)
    assert result.last.startswith("[smoke_reference] CHECK OK"), show(result)
    assert not _trainer_called(result) and smoke.ckpt.is_dir()
    assert "it will be removed (verl would resume from it)" in result.out
    assert "data.train_batch_size=8" in result.out


SMOKE_PREFLIGHT = {
    "unpatched_verl": ({"pythonpath": "pristine"}, ["FAIL  patch hunk 1", "FAIL  patch hunk 2", "bash integration/setup_env.sh"]),
    "no_serve": ({"no_serve": True}, ["FAIL  serve.rl.verl_rollout does not import: ModuleNotFoundError: No module named 'serve'"]),
    "no_data": ({"no_data": True}, ["math_l3to5_train.parquet does not exist", "math500_test.parquet does not exist",
                                    "baseline/prepare_data.py --out-dir"]),
    "gpu_busy": ({"FAKE_GPU_FREE": "12000"}, ["GPU 0 has only 12000 MiB free, this run needs about 30720 MiB", "wait until the card is free"]),
    "second_gpu_busy": ({"FAKE_GPU1_FREE": "1000", "CUDA_VISIBLE_DEVICES": "1"}, ["GPU 1 has only 1000 MiB free"]),
    "no_nvidia_smi": ({"nvidia": False}, ["nvidia-smi not found"]),
    "nvidia_smi_fails": ({"FAKE_NVSMI_FAIL": "1"}, ["nvidia-smi --query-gpu failed", "couldn't communicate with the NVIDIA driver"]),
    "baseline_env_active": ({"CONDA_DEFAULT_ENV": "verl-baseline"}, ["the active env is verl-baseline"]),
}  # fmt: skip


@pytest.mark.parametrize("case", sorted(SMOKE_PREFLIGHT))
def test_smoke_preflight_failures_stop_before_anything_is_removed(smoke, case):
    kw, texts = SMOKE_PREFLIGHT[case]
    kw = dict(kw)
    if kw.pop("pythonpath", None) == "pristine":
        kw["pythonpath"] = smoke.box.tools.pristine_root
    if kw.pop("no_serve", False):
        (smoke.box.repo / "serve").unlink()
    if kw.pop("no_data", False):
        shutil.rmtree(smoke.data)
    args = ["--check"] if case == "gpu_busy" else []  # --check reports the same failures (one case shown)
    result = smoke.run(*args, **kw)
    assert result.code == 1, show(result)
    for text in texts:
        assert text in result.out, (text, show(result))
    assert not _trainer_called(result), result.calls
    assert smoke.ckpt.is_dir(), "the checkpoint dir must only be removed right before the trainer starts"
    assert "nothing was run or removed" in result.out
    assert result.last.startswith("[smoke_reference] CHECK FAILED" if args else "[smoke_reference] FAIL"), show(result)


def test_smoke_gpu_check_can_be_overridden(smoke):
    result = smoke.run("--check", FAKE_GPU_FREE="12000", MIN_FREE_GIB="0")
    assert result.code == 0 and "free-GPU-memory check skipped" in result.out, show(result)
    result = smoke.run("--check", FAKE_GPU_FREE="12000", MIN_FREE_GIB="10")
    assert result.code == 0 and "needs about 10240 MiB free: MIN_FREE_GIB=10" in result.out, show(result)


def test_smoke_trainer_failure_shows_the_log_tail_and_fails(smoke):
    result = smoke.run(FAKE_TRAINER_MODE="crash")
    assert result.code == 1, show(result)
    assert "trainer exit status: 1" in result.out
    assert "last 40 lines of" in result.out and "CUDA out of memory" in result.out
    assert "first error-looking lines" in result.out
    assert result.last.startswith("[smoke_reference] FAIL: trainer exit 1, gate 3d FAIL"), show(result)
    assert "missing [2, 3]" in result.out  # gate 3d says which steps are missing


@pytest.mark.parametrize("kw, text", [
    ({"STEPS": "0"}, "STEPS must be a positive integer"),
    ({"PROMPTS": "eight"}, "PROMPTS must be a positive integer"),
    ({"EXP_NAME": "sparsespec_seed0"}, "looks like a real run"),
    ({"EXP_NAME": "vllm_baseline_seed0"}, "looks like a real run"),
    ({"EXP_NAME": "../x"}, "EXP_NAME must be a plain name"),
])  # fmt: skip
def test_smoke_refuses_bad_inputs(smoke, kw, text):
    result = smoke.run(**kw)
    assert result.code == 1 and text in result.out, show(result)
    assert not _trainer_called(result) and smoke.ckpt.is_dir()
    assert (smoke.ckpts / "vllm_baseline_seed0" / "hf").is_dir()


def test_smoke_overrides_are_valid_for_verl_0_5_0(smoke):
    """The reduced config composes with verl's hydra config, passes RayPPOTrainer._validate_config,
    and the actor's mini-batch normalisation gives exactly one mini-batch per step."""
    hydra = pytest.importorskip("hydra")
    pytest.importorskip("verl")
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf, open_dict

    result = smoke.run()
    assert result.code == 0, show(result)
    argv = smoke.argv.read_text().splitlines()
    GlobalHydra.instance().clear()
    try:
        with hydra.initialize_config_module("verl.trainer.config", version_base=None):
            cfg = hydra.compose("ppo_trainer", overrides=argv)
    finally:
        GlobalHydra.instance().clear()
    rollout, actor = cfg.actor_rollout_ref.rollout, cfg.actor_rollout_ref.actor
    assert cfg.data.train_batch_size == 8 and rollout.n == 8 and actor.ppo_mini_batch_size == 8
    assert cfg.data.max_response_length == 512 and rollout.response_length == 512  # interpolated
    assert rollout.prompt_length == 1024 and cfg.trainer.save_freq == -1 and cfg.trainer.resume_mode == "auto"
    assert cfg.trainer.total_training_steps == 3 and rollout.sparsespec.engine == "reference"
    assert actor.use_dynamic_bsz is True and actor.ppo_max_token_len_per_gpu >= 1024 + 512

    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    RayPPOTrainer._validate_config(SimpleNamespace(config=cfg, use_critic=False, use_reference_policy=True))

    # ActorRolloutRefWorker.__init__ (fsdp_workers.py): mini *= rollout.n; mini //= world // sp
    world, sp = cfg.trainer.n_gpus_per_node * cfg.trainer.nnodes, actor.ulysses_sequence_parallel_size
    mini = actor.ppo_mini_batch_size * rollout.n // (world // sp)
    batch = cfg.data.train_batch_size * rollout.n
    assert (mini, batch) == (64, 64)
    import torch
    from verl import DataProto

    data = DataProto.from_dict(tensors={"x": torch.zeros(batch, 2)})
    assert len(data.split(mini)) == 1  # dp_actor.update_policy: one optimizer step per training step
    assert len(list(data.make_iterator(mini_batch_size=mini, epochs=1))) == 1  # batch % mini == 0
    with open_dict(cfg):
        assert OmegaConf.select(cfg, "trainer.default_local_dir") == str(smoke.ckpt)
