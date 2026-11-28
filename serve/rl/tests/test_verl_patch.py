"""integration/verl-0.5.0-sparsespec.patch applies to verl 0.5.0 and adds
exactly the ``sparsespec`` rollout branch and the metrics statement.

A copy of the verl package is patched in a temp dir (never the installed
one). Source of the copy: $SPARSESPEC_VERL_SRC if it points at an unpacked
pristine verl==0.5.0 package directory, otherwise the installed verl package;
if that copy already carries the patch (the SparseSpec env on the box), it is
used as is.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Optional: an unpacked pristine verl==0.5.0 package directory (.../verl).
_VERL_SRC_ENV = os.environ.get("SPARSESPEC_VERL_SRC", "")
PRISTINE_WHEEL_SOURCE = Path(_VERL_SRC_ENV) if _VERL_SRC_ENV else None
FSDP_WORKERS = Path("verl") / "workers" / "fsdp_workers.py"
RAY_TRAINER = Path("verl") / "trainer" / "ppo" / "ray_trainer.py"


def verl_source_dir() -> Path:
    if PRISTINE_WHEEL_SOURCE is not None and PRISTINE_WHEEL_SOURCE.is_dir():
        return PRISTINE_WHEEL_SOURCE
    verl = pytest.importorskip("verl")
    return Path(verl.__file__).parent


def copy_verl(destination: Path) -> Path:
    """<destination>/verl = a copy of the verl package; returns <destination>."""
    shutil.copytree(verl_source_dir(), destination / "verl", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return destination


def _run(cmd, cwd: Path, stdin_path: Path | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(cwd.parent))  # never pick up an enclosing git repo
    if stdin_path is None:
        return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    with open(stdin_path) as stdin:
        return subprocess.run(cmd, cwd=cwd, env=env, stdin=stdin, capture_output=True, text=True)


class GitApply:
    name = "git"

    @staticmethod
    def check(root, patch, reverse=False):
        return (
            _run(["git", "apply", "--check"] + (["--reverse"] if reverse else []) + [str(patch)], root).returncode == 0
        )

    @staticmethod
    def apply(root, patch):
        return _run(["git", "apply", str(patch)], root)


class PatchP1:
    """The commands integration/setup_env.sh uses."""

    name = "patch"

    @staticmethod
    def check(root, patch, reverse=False):
        cmd = ["patch", "-p1", "--dry-run"] + (["-R"] if reverse else []) + ["-s", "-f"]
        return _run(cmd, root, stdin_path=patch).returncode == 0

    @staticmethod
    def apply(root, patch):
        return _run(["patch", "-p1", "-s"], root, stdin_path=patch)


TOOLS = {"git": GitApply, "patch": PatchP1}


_PATCHED: dict = {}


def _patched_with(tool_name: str, tk, tmp_path_factory) -> SimpleNamespace:
    """A verl copy with the patch applied by the given tool (built once per session)."""
    if tool_name in _PATCHED:
        return _PATCHED[tool_name]
    tool = TOOLS[tool_name]
    patch = tk.patch_path()
    assert patch.is_file()

    original = verl_source_dir().parent  # the package the copy was made from (read-only use)
    root = copy_verl(tmp_path_factory.mktemp(f"verl-patched-{tool.name}"))

    if tool.check(root, patch, reverse=True):
        state = "already applied"  # e.g. the installed verl of the SparseSpec env
        assert not tool.check(root, patch), "patch applies both forwards and backwards?"
    else:
        assert tool.check(root, patch), (
            f"the patch neither applies to nor is already applied in the verl at {verl_source_dir()} "
            "(it targets verl 0.5.0)"
        )
        result = tool.apply(root, patch)
        assert result.returncode == 0, result.stdout + result.stderr
        state = "applied"
        # idempotence checks setup_env.sh relies on: now only the reverse direction applies
        assert tool.check(root, patch, reverse=True)
        assert not tool.check(root, patch)
        assert not list(root.rglob("*.orig")) and not list(root.rglob("*.rej"))

    _PATCHED[tool_name] = SimpleNamespace(root=root, original=original, state=state, tool=tool.name, patch=patch)
    return _PATCHED[tool_name]


@pytest.fixture(scope="module", params=sorted(TOOLS))
def patched(request, tk, tmp_path_factory):
    """The patched verl copy, once per tool (``git apply`` and ``patch -p1``)."""
    if shutil.which(request.param) is None:
        pytest.skip(f"{request.param} is not installed")
    return _patched_with(request.param, tk, tmp_path_factory)


@pytest.fixture(scope="module")
def patched_once(tk, tmp_path_factory):
    """The patched verl copy made by whichever tool exists (for the one slow test)."""
    available = [name for name in sorted(TOOLS) if shutil.which(name)]
    if not available:
        pytest.skip("neither git nor patch is installed")
    return _patched_with(available[0], tk, tmp_path_factory)


def _function(tree, class_name: str, name: str) -> ast.FunctionDef:
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _rollout_branches(build_rollout: ast.FunctionDef) -> dict:
    """rollout name -> ast.If node of the ``if rollout_name == "...":`` chain."""
    first = next(
        node
        for node in build_rollout.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and getattr(node.test.left, "id", None) == "rollout_name"
    )
    branches, node = {}, first
    while True:
        branches[node.test.comparators[0].value] = node
        if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            node = node.orelse[0]
        else:
            branches["<else>"] = node.orelse
            return branches


# ---------------------------------------------------------------------------
def test_patch_touches_exactly_two_files(tk):
    text = tk.patch_path().read_text()
    targets = re.findall(r"^\+\+\+ b/(\S+)", text, flags=re.M)
    assert targets == [FSDP_WORKERS.as_posix(), RAY_TRAINER.as_posix()]
    assert re.findall(r"^--- a/(\S+)", text, flags=re.M) == targets
    removed = [ln for ln in text.splitlines() if ln.startswith("-") and not ln.startswith("--- ")]
    assert removed == []  # purely additive


def test_both_files_compile_after_patching(patched):
    for rel in (FSDP_WORKERS, RAY_TRAINER):
        path = patched.root / rel
        compile(path.read_text(), str(path), "exec")


def test_only_the_patch_lines_were_added(tk, patched):
    if patched.state == "already applied":
        pytest.skip("the verl copy came patched: there is no pristine file to diff against")
    for rel in (FSDP_WORKERS, RAY_TRAINER):
        before = (patched.original / rel).read_text().splitlines()
        after = (patched.root / rel).read_text().splitlines()
        added = tk.patch_added_lines(rel.as_posix())
        assert len(after) == len(before) + len(added)
        # one contiguous block was inserted: common prefix + inserted block + common suffix
        prefix = next((i for i, (a, b) in enumerate(zip(before, after)) if a != b), len(before))
        suffix = len(before) - prefix
        inserted = after[prefix : len(after) - suffix]
        assert after[:prefix] + after[len(after) - suffix :] == before, "the patch removed or rewrote existing lines"
        # a diff may attribute the block's blank separator line to either end: compare as a multiset
        assert sorted(inserted) == sorted(added)
        assert [ln for ln in inserted if ln.strip()] == [ln for ln in added if ln.strip()]
    # every other file of the package is byte-identical
    compared = 0
    for path in (patched.original / "verl").rglob("*.py"):
        rel = path.relative_to(patched.original)
        if rel not in (FSDP_WORKERS, RAY_TRAINER):
            assert path.read_bytes() == (patched.root / rel).read_bytes(), rel
            compared += 1
    assert compared > 100


def test_build_rollout_has_the_sparsespec_branch(tk, patched):
    source = (patched.root / FSDP_WORKERS).read_text()
    tree = ast.parse(source)
    build_rollout = _function(tree, "ActorRolloutRefWorker", "_build_rollout")
    branches = _rollout_branches(build_rollout)

    assert list(branches) == ["hf", "vllm", "sglang", "sparsespec", "<else>"]
    # unknown rollout names still raise
    assert isinstance(branches["<else>"][0], ast.Raise)
    assert isinstance(build_rollout.body[-1], ast.Return)
    assert ast.unparse(build_rollout.body[-1]) == "return (rollout, rollout_sharding_manager)"

    branch = branches["sparsespec"]
    imports = [n for n in branch.body if isinstance(n, ast.ImportFrom)]
    assert [(n.module, [a.name for a in n.names]) for n in imports] == [
        ("serve.rl.verl_rollout", ["build_sparsespec_rollout"])
    ]
    assign = next(
        n
        for n in branch.body
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and getattr(n.value.func, "id", None) == "build_sparsespec_rollout"
    )
    assert ast.unparse(assign.targets[0]) == "(rollout, rollout_sharding_manager)"
    assert {kw.arg: ast.unparse(kw.value) for kw in assign.value.keywords} == {
        "model_path": "local_path",
        "rollout_config": "self.config.rollout",
        "tokenizer": "self.tokenizer",
        "model_hf_config": "self.actor_model_config",
        "actor_module": "self.actor_module_fsdp",
        "offload_param": "self._is_offload_param",
    }
    # the string the run script's pre-flight check looks for
    assert '"sparsespec"' in ast.get_source_segment(source, build_rollout)


def test_sparsespec_branch_only_uses_names_the_vllm_branch_uses(patched):
    """No NameError / AttributeError waiting on the GPU box: every name and
    every ``self.<attr>`` of the new branch is also used by the stock vllm
    branch right above it (or is a builtin / defined in the branch itself)."""
    tree = ast.parse((patched.root / FSDP_WORKERS).read_text())
    branches = _rollout_branches(_function(tree, "ActorRolloutRefWorker", "_build_rollout"))

    def loaded_names(nodes):
        return {
            n.id for body in nodes for n in ast.walk(body) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }

    def self_attrs(nodes):
        return {
            n.attr
            for body in nodes
            for n in ast.walk(body)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self"
        }

    def defined_names(nodes):
        out = set()
        for body in nodes:
            for n in ast.walk(body):
                if isinstance(n, ast.ImportFrom | ast.Import):
                    out |= {a.asname or a.name for a in n.names}
                elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    out.add(n.id)
        return out

    ours, vllm = branches["sparsespec"].body, branches["vllm"].body
    free = loaded_names(ours) - defined_names(ours) - set(dir(builtins))
    assert free <= loaded_names(vllm) | {"rollout_name"}, free - loaded_names(vllm)
    assert self_attrs(ours) <= self_attrs(vllm), self_attrs(ours) - self_attrs(vllm)
    # and those module-level helpers really exist in the file
    module_names = {
        a.asname or a.name for n in tree.body if isinstance(n, ast.ImportFrom | ast.Import) for a in n.names
    }
    module_names |= {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    assert {"copy_to_local", "log_gpu_memory_usage", "logger"} <= free
    assert {"copy_to_local", "log_gpu_memory_usage", "logger"} <= module_names


def test_trainer_statement_sits_right_after_the_timing_merge(tk, patched):
    source = (patched.root / RAY_TRAINER).read_text()
    tree = ast.parse(source)
    fit = _function(tree, "RayPPOTrainer", "fit")
    statement = tk.trainer_metrics_statement()

    wanted = ast.unparse(ast.parse(statement).body[0])
    hits = [
        (parent, index)
        for parent in ast.walk(fit)
        for field in ("body", "orelse", "finalbody")
        if isinstance(getattr(parent, field, None), list)
        for index, node in enumerate(getattr(parent, field))
        if isinstance(node, ast.Expr) and ast.unparse(node) == wanted
    ]
    assert len(hits) == 1, "the metrics statement must appear exactly once in RayPPOTrainer.fit"
    parent, index = hits[0]
    assert (
        isinstance(parent, ast.With) and 'marked_timer("gen"' in ast.get_source_segment(source, parent).splitlines()[0]
    )
    previous = [ast.unparse(node) for node in parent.body[index - 2 : index]]
    assert previous == [
        "timing_raw.update(gen_batch_output.meta_info['timing'])",
        "gen_batch_output.meta_info.pop('timing', None)",
    ]
    # `metrics` and `timing_raw` are the trainer's own per-step dicts
    fit_source = ast.get_source_segment(source, fit)
    assert "metrics = {}" in fit_source and "timing_raw = {}" in fit_source


def test_git_apply_and_patch_produce_identical_files(tk, tmp_path_factory):
    if shutil.which("git") is None or shutil.which("patch") is None:
        pytest.skip("needs both git and patch")
    by_git = _patched_with("git", tk, tmp_path_factory)
    by_patch = _patched_with("patch", tk, tmp_path_factory)
    assert by_git.root != by_patch.root and by_git.state == by_patch.state
    for rel in (FSDP_WORKERS, RAY_TRAINER):
        assert (by_git.root / rel).read_bytes() == (by_patch.root / rel).read_bytes(), rel


def test_run_script_preflight_recognises_the_patched_verl(tk, patched_once):
    """Run the very check baseline/run_seed_sparsespec.sh runs (its python
    heredoc, verbatim) in a fresh interpreter that imports the patched copy.
    Done once: both tools produce identical files (tested above)."""
    patched = patched_once
    script = (tk.REPO_ROOT / "baseline" / "run_seed_sparsespec.sh").read_text()
    match = re.search(r"python3 - <<'EOF'\n(.*?)\nEOF\n", script, flags=re.S)
    assert match, "pre-flight heredoc not found in run_seed_sparsespec.sh"
    preflight = match.group(1)
    assert "ActorRolloutRefWorker._build_rollout" in preflight and "sys.exit(" in preflight

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(patched.root), str(tk.REPO_ROOT), env.get("PYTHONPATH", "")])
    # guard: the interpreter must really import the patched copy, then the heredoc decides the exit code
    guard = (
        "import os, verl\n"
        f"assert os.path.realpath(os.path.dirname(verl.__file__)) == {os.path.realpath(patched.root / 'verl')!r}, "
        "verl.__file__\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", guard + preflight], env=env, capture_output=True, text=True, cwd=patched.root
    )
    assert proc.returncode == 0, proc.stderr[-3000:]

    if patched.state == "applied":  # ... and it rejects the unpatched verl (same test, done on the source text)
        source = (patched.original / FSDP_WORKERS).read_text()
        build_rollout = _function(ast.parse(source), "ActorRolloutRefWorker", "_build_rollout")
        assert '"sparsespec"' not in ast.get_source_segment(source, build_rollout)


def test_setup_env_hashes_match_the_pristine_wheel(tk, patched):
    """integration/setup_env.sh prints the sha256 of the files the patch expects."""
    setup = tk.REPO_ROOT / "integration" / "setup_env.sh"
    if not setup.is_file():
        pytest.skip("integration/setup_env.sh does not exist")
    if patched.state == "already applied":
        pytest.skip("the verl copy came patched: no pristine files to hash")
    text = setup.read_text()
    expected = {
        FSDP_WORKERS: re.search(r"^SHA_FSDP_WORKERS=([0-9a-f]{64})$", text, flags=re.M),
        RAY_TRAINER: re.search(r"^SHA_RAY_TRAINER=([0-9a-f]{64})$", text, flags=re.M),
    }
    if not all(expected.values()):
        pytest.skip("setup_env.sh no longer records the file hashes")
    for rel, match in expected.items():
        assert hashlib.sha256((patched.original / rel).read_bytes()).hexdigest() == match.group(1), rel
