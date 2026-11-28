"""Stub-driven tests of integration/run_all_gates.sh and integration/summarize_bundle.py.

No conda, GPU, vLLM or verl is needed: each test copies the runner, lib.sh,
summarize_bundle.py and baseline/config/grpo_base.sh into a temp "repo" and
puts stub executables first on PATH:

    conda        records the env name of every ``conda run`` and execs the command
    nvidia-smi   realistic query / table output; free memory read from a file
    python       dispatches on its arguments: pytest, the gate CLIs, the import
                 check and pip freeze are faked; anything else runs the real python
    python3      the real python (the runner's summary step)
    timeout      a GNU-timeout stand-in (own process group, exit 124)
    free         procps-style output

The run script and smoke_reference.sh in the temp repo are stubs too. Every
stubbed command appends a JSON line (stage key, conda env, argv, selected
variables) to a calls log, which the tests read.

Run from the repo root: ``PYTHONPATH=. python -m pytest integration/tests -q``
(no ``__init__.py`` in this directory on purpose, see serve/rl/tests/conftest.py).
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import textwrap
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "integration" / "run_all_gates.sh"
SUMMARIZER = REPO / "integration" / "summarize_bundle.py"

ALL_STAGES = (
    "env cpu_tests engine_import reference_plumbing gpu_unit_tests gpu_unit_tests_host_guard gpu_unit_tests_offload "
    "gpu_unit_tests_offload_host_guard gate_3b_vllm gate_3b gate_3c_synthetic gate_3c_step25 gate_3c_vllm_reference "
    "smoke_worst_case smoke_worst_case_check smoke_host_guard smoke_host_guard_check smoke_no_graph "
    "smoke_no_graph_check smoke_no_logprobs smoke_no_logprobs_check smoke_compare"
).split()
BASELINE_STAGES = {"gate_3b_vllm", "gate_3c_vllm_reference"}

pytestmark = pytest.mark.skipif(shutil.which("bash") is None or shutil.which("tar") is None, reason="needs bash and tar")

# ---------------------------------------------------------------------------
# the stubs
# ---------------------------------------------------------------------------
STUB_IMPL = r'''
import json, os, signal, subprocess, sys, time

CONTROL = json.load(open(os.environ["STUB_CONTROL"])) if os.environ.get("STUB_CONTROL") else {}
CALLS = os.environ.get("STUB_CALLS")
REAL_PY = os.environ["STUB_REAL_PY"]
TOTAL = 40960
VARS = ("SPARSESPEC_TEST_ADMISSION", "SPARSESPEC_TEST_OFFLOAD_STRESS", "SPARSESPEC_ADMISSION", "SPARSESPEC_CUDA_GRAPH",
        "SPARSESPEC_CALC_LOGPROBS", "SPARSESPEC_ENGINE", "STEPS", "EXP_NAME", "SEED", "DRY_RUN", "PYTHONPATH",
        "CKPT_ROOT", "DATA_DIR", "MODEL_PATH")


def record(key, argv):
    if not CALLS:
        return
    entry = {"key": key, "env": os.environ.get("CONDA_DEFAULT_ENV"), "argv": list(argv), "pid": os.getpid(),
             "vars": {k: os.environ[k] for k in VARS if k in os.environ}, "cwd": os.getcwd()}
    with open(CALLS, "a") as f:
        f.write(json.dumps(entry) + "\n")


def flag(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def behave(key):
    """hang / set GPU free memory / exit status, as the control file says for this stage key."""
    gpu = CONTROL.get("gpu_free_after", {}).get(key)
    if gpu is not None:
        open(os.environ["STUB_GPU_FREE"], "w").write(str(gpu))
    if key in CONTROL.get("hang", []):
        time.sleep(600)
    return int(CONTROL.get("exit", {}).get(key, 0))


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def gpu_free():
    try:
        return int(open(os.environ["STUB_GPU_FREE"]).read().strip())
    except (KeyError, OSError, ValueError):
        return 40000


# ---------------------------------------------------------------- python
def summary_3b(num=16, identical=14):
    return {"num_prompts": num, "identical": identical, "near_tie": num - identical, "mismatch": 0,
            "max_margin": 0.31, "first_diffs": [-1] * identical + [113 + i for i in range(num - identical)]}


def fake_python(argv):
    if argv[:2] == ["-m", "pytest"]:
        if "serve/rl/tests/test_engine_gpu.py" in argv:
            key = "gpu_unit_tests"
            if os.environ.get("SPARSESPEC_TEST_OFFLOAD_STRESS") == "1":
                key += "_offload"
            if os.environ.get("SPARSESPEC_TEST_ADMISSION") == "host_guard":
                key += "_host_guard"
        else:
            key = "cpu_tests"
        record(key, argv)
        rc = behave(key)
        print("serve/rl/tests/test_engine_gpu.py::test_greedy PASSED")
        if key in CONTROL.get("all_skipped", []):
            print("=" * 20 + " 9 skipped in 0.42s " + "=" * 20)
        elif rc == 1:
            print("FAILED serve/rl/tests/test_engine_gpu.py::test_sleep_wake - AssertionError: memory not freed")
            print("=" * 20 + " 1 failed, 8 passed in 12.10s " + "=" * 20)
        elif "-q" in argv:
            print("9 passed, 1 skipped in 12.10s")  # pytest -q: no '=' decoration
        else:
            print("=" * 20 + " 9 passed, 1 skipped in 12.10s " + "=" * 20)
        return rc
    if argv and argv[0].endswith(("check_pytest.py", "check_engine_import.py")):
        code = open(argv[0]).read()
        if "import pytest" in code:
            record("pytest_check", argv)
            if CONTROL.get("pytest_missing"):
                print("ModuleNotFoundError: No module named 'pytest'", file=sys.stderr)
                return 1
            print("pytest 8.3.0")
            return 0
        if "serve.rl.engine" in code:
            record("engine_import", argv)
            rc = behave("engine_import")
            if rc:
                print("Traceback (most recent call last):\n  File \"<string>\", line 1\n"
                      "ModuleNotFoundError: No module named 'flashinfer'", file=sys.stderr)
            else:
                print("serve.rl.engine importable: /repo/serve/rl/engine.py")
            return rc
    if argv[:3] == ["-m", "pip", "freeze"]:
        record("pip", argv)
        print("torch==2.7.1\nvllm==0.9.1")
        return 0
    if argv and argv[0].endswith("probe_env.py"):
        record("probe", argv)
        env = os.environ.get("CONDA_DEFAULT_ENV")
        print("VERSION python 3.11.9 /envs/%s/bin/python" % env)
        print("VERSION torch 2.7.1+cu126 /x")
        print("VERSION vllm 0.9.1 /x")
        print("VERSION verl 0.5.0 /x")
        print("VERSION flashinfer %s" % ("0.2.6+fork /x" if env == "verl-sparsespec" else "NOT importable: ModuleNotFoundError"))
        print("PATCH %s" % ("applied" if env == "verl-sparsespec" else "absent"))
        return 0
    out_dir = flag(argv, "--out-dir")
    if argv and argv[0] == "integration/gates/gate_b_vllm_greedy.py":
        record("gate_3b_vllm", argv)
        rc = behave("gate_3b_vllm")
        if rc == 0:
            write_json(flag(argv, "--out"), {"pass": True, "summary": summary_3b()})
        print("[gate 3b vllm] %s" % ("PASS" if rc == 0 else "FAIL"))
        return rc
    if argv[:2] == ["-m", "integration.gates.gate_b_greedy"]:
        record("gate_3b", argv)
        rc = behave("gate_3b")
        tag = flag(argv, "--tag")
        if "--vllm-json" in argv:
            data = {"pass": rc == 0, "rule": "margin<=1.0", "pairs": {
                "sparsespec_vs_hf": dict(summary_3b(), **{"pass": True}),
                "vllm_vs_hf": dict(summary_3b(16, 15), **{"pass": True}),
                "sparsespec_vs_vllm": dict(summary_3b(16, 13), **{"pass": rc == 0})}}
        else:
            data = {"pass": rc == 0, "config": {"engine": "sparsespec", "tie_margin": 1.0}, "summary": summary_3b()}
        write_json(os.path.join(out_dir, "gate_b_greedy_%s.json" % tag), data)
        print("[gate 3b] %s" % ("PASS" if rc == 0 else "FAIL"))
        return rc
    if argv[:2] == ["-m", "integration.gates.gate_c_reload"]:
        tag = flag(argv, "--tag")
        key = {"sparsespec_synthetic": "gate_3c_synthetic", "sparsespec_step25": "gate_3c_step25"}[tag]
        record(key, argv)
        rc = behave(key)
        if "--synthetic" in argv:
            ckpt = os.path.join(os.environ["CKPT_ROOT"], "synthetic", "relnoise0.02_seed0")
            os.makedirs(ckpt, exist_ok=True)
        else:
            ckpt = flag(argv, "--ckpt")
        data = {"gate": "3c_reload", "pass": rc == 0, "failure_class": None, "checkpoint": {"dir": ckpt},
                "stats": {"reloaded": {"mean": 0.0081}, "base_after_roundtrip": {"mean": 0.0072},
                          "negative_control": {"mean": 0.318 if rc == 0 else 0.009}},
                "checks": [{"name": "reload1_bitwise", "ok": True}, {"name": "reload3_bitwise", "ok": True}]}
        if rc == 1 and key in CONTROL.get("underpowered", []):
            data["failure_class"] = "negative_control_underpowered"
            data["failed_checks"] = ["negative_control_clearly_larger"]
            data["rerun"] = {"next_step": 50, "next_ckpt": ckpt.replace("global_step_25", "global_step_50")}
        if rc in (0, 1):
            write_json(os.path.join(out_dir, "gate_c_reload_%s.json" % tag), data)
        if rc > 1:
            print("Traceback (most recent call last):\nRuntimeError: CUDA error: an illegal memory access", file=sys.stderr)
        print("[gate 3c] %s" % ("PASS" if rc == 0 else "FAIL"))
        return rc
    if argv and argv[0] == "integration/gates/gate_c_vllm_reference.py":
        record("gate_3c_vllm_reference", argv)
        rc = behave("gate_3c_vllm_reference")
        tag = flag(argv, "--tag")
        write_json(os.path.join(out_dir, "gate_c_vllm_reference%s.json" % ("_" + tag if tag else "")),
                   {"stats": {"engine_vs_trainer": {"mean": 0.0123, "p99": 0.09, "count": 4096}},
                    "config": {"ckpt": flag(argv, "--ckpt")}})
        return rc
    if argv[:2] == ["-m", "integration.gates.gate_d_smoke_check"]:
        exp = flag(argv, "--exp-name")
        key = "smoke_%s_check" % exp[len("gates_smoke_"):]
        record(key, argv)
        rc = behave(key)
        csv = os.path.join(os.environ["REPO_ROOT"], "baseline", "results", exp + ".timing.csv")
        assert os.path.isfile(csv), csv
        steps = sum(1 for _ in open(csv)) - 1
        write_json(os.path.join(out_dir, "gate_d_smoke_%s.json" % exp),
                   {"pass": rc == 0, "steps_found": list(range(1, steps + 1)),
                    "checks": [{"name": "steps_present", "ok": rc == 0}]})
        print("[gate 3d] %s" % ("PASS" if rc == 0 else "FAIL"))
        return rc
    if argv[:2] == ["-m", "integration.gates.smoke_compare"]:
        record("smoke_compare", argv)
        rc = behave("smoke_compare")
        # the shape integration/gates/smoke_compare.py writes (host_memory.rows, runs[].reading)
        write_json(os.path.join(out_dir, "smoke_compare.json"), {
            "kind": "smoke_compare", "pass": True,
            "runs": [{"label": "worst_case", "reading": "batch_size_mean at the admission cap with kv_util_peak 0.41: "
                                                        "the cap binds; a larger cpu_offload_ratio would help"}],
            "host_memory": {"rows": [
                {"name": "default", "cpu_offload_ratio": 1.5, "pinned_gb": 16.0, "inflight_worst_case": 27,
                 "inflight_host_guard": 41, "fits_available": True},
                {"name": "3.0", "cpu_offload_ratio": 3.0, "pinned_gb": 32.0, "inflight_worst_case": 46,
                 "inflight_host_guard": 64, "fits_available": True}]}})
        return rc
    os.execv(REAL_PY, [REAL_PY] + argv)


# ---------------------------------------------------------------- run scripts
def run_seed(argv):
    exp = os.environ.get("EXP_NAME", "")
    label = exp[len("gates_smoke_"):] if exp.startswith("gates_smoke_") else exp
    if os.environ.get("DRY_RUN") == "1":
        record("dry_run_" + label, argv)
        print("actor_rollout_ref.rollout.name=sparsespec")
        print("+actor_rollout_ref.rollout.sparsespec.admission=%s" % os.environ.get("SPARSESPEC_ADMISSION"))
        print("trainer.experiment_name=%s" % exp)
        return 0
    key = "smoke_" + label
    ckpt = os.path.join(os.environ["CKPT_ROOT"], exp)
    record(key, argv)
    with open(os.environ["STUB_CKPT_SEEN"], "a") as f:
        f.write("%s %s\n" % (exp, os.path.exists(ckpt)))
    rc = behave(key)
    repo = os.environ["REPO_ROOT"]
    steps = int(os.environ.get("STEPS", "100"))
    done = steps if rc == 0 else 1
    os.makedirs(os.path.join(repo, "baseline", "logs"), exist_ok=True)
    with open(os.path.join(repo, "baseline", "logs", exp + ".log"), "w") as f:
        f.write("step:1 - timing_s/gen:400\n")
    if key not in CONTROL.get("no_csv", []):
        gen = {"worst_case": 400.0, "host_guard": 380.0, "no_graph": 520.0, "no_logprobs": 390.0}.get(label, 1.0)
        with open(os.path.join(repo, "baseline", "results", exp + ".timing.csv"), "w") as f:
            f.write("step,timing_s/gen,timing_s/step,critic/score/mean,sparsespec/batch_size_mean,"
                    "sparsespec/kv_util_peak,sparsespec/num_offloads\n")
            for step in range(1, done + 1):
                f.write("%d,%s,%s,0.1,27.5,0.41,%d\n" % (step, gen + (30 if step == 1 else 0), gen + 300, step - 1))
    if rc == 0:
        os.makedirs(os.path.join(ckpt, "global_step_%d" % steps), exist_ok=True)
    else:
        print("Traceback (most recent call last):\ntorch.OutOfMemoryError: CUDA out of memory.", file=sys.stderr)
    return rc


def smoke_reference(argv):
    record("reference_plumbing", argv)
    rc = behave("reference_plumbing")
    print("[smoke_reference] done, exit status %d" % rc)
    return rc


# ---------------------------------------------------------------- system tools
def conda(argv):
    if argv[:1] == ["--version"]:
        print("conda 24.5.0")
        return 0
    if argv[:2] == ["env", "list"]:
        print("# conda environments:\n#\nbase                  *  /opt/conda")
        for env in os.environ.get("STUB_ENVS", "").split():
            print("%-22s   /opt/conda/envs/%s" % (env, env))
        return 0
    if argv[:2] == ["run", "--help"]:
        print("usage: conda run [-h] [--no-capture-output] [-n ENVIRONMENT] ...")
        return 0
    if argv[:1] == ["list"]:
        print("# packages in environment %s:\ntorch  2.7.1" % flag(argv, "-n"))
        return 0
    if argv[:1] == ["run"]:
        rest = argv[1:]
        assert rest[0] == "--no-capture-output", rest
        assert rest[1] == "-n", rest
        env, cmd = rest[2], rest[3:]
        os.environ["CONDA_DEFAULT_ENV"] = env
        os.execvp(cmd[0], cmd)
    print("conda stub: unsupported %r" % argv, file=sys.stderr)
    return 2


def nvidia_smi(argv):
    free = gpu_free()
    used = TOTAL - free
    busy = used > 2000
    query = [a for a in argv if a.startswith("--query-gpu=")]
    fmt = next((a.split("=", 1)[1] for a in argv if a.startswith("--format=")), "csv")
    units = "nounits" not in fmt
    header = "noheader" not in fmt
    if query:
        fields = query[0].split("=", 1)[1].split(",")
        values = {"index": "0", "uuid": "GPU-8f2c1e9a-0000", "name": "NVIDIA A100-SXM4-40GB",
                  "driver_version": "535.183.01", "memory.total": "%d%s" % (TOTAL, " MiB" if units else ""),
                  "memory.used": "%d%s" % (used, " MiB" if units else ""),
                  "memory.free": "%d%s" % (free, " MiB" if units else ""), "utilization.gpu": "0 %",
                  "temperature.gpu": "31", "persistence_mode": "Enabled", "compute_mode": "Default"}
        if header:
            print(", ".join(fields))
        print(", ".join(values.get(f, "[N/A]") for f in fields))
        return 0
    if any(a.startswith("--query-compute-apps=") for a in argv):
        if header:
            print("pid, process_name, used_gpu_memory [MiB]")
        if busy:
            print("424242, /home/teammate/envs/x/bin/python3, %d MiB" % (used - 500))
        return 0
    print("+---------------------------------------------------------------------------------------+")
    print("| NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 12.2     |")
    print("|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:07:00.0 Off |                    0 |")
    print("| N/A   31C    P0              52W / 400W |  %5dMiB / %dMiB |      0%%      Default |" % (used, TOTAL))
    print("| Processes:                                                                            |")
    if busy:
        print("|    0   N/A  N/A    424242      C   /home/teammate/envs/x/bin/python3        %dMiB |" % (used - 500))
    return 0


def free_cmd(argv):
    print("               total        used        free      shared  buff/cache   available")
    if "-k" in argv:
        print("Mem:       528280000    20000000   480000000      100000    28280000   500000000")
    else:
        print("Mem:             503          19         457           0          26         476")
    print("Swap:              0           0           0")
    return 0


def timeout_cmd(argv):
    """GNU timeout stand-in: timeout -k K SECS cmd...; child in its own process group, exit 124 on expiry."""
    assert argv[0] == "-k", argv
    kill_after, secs, cmd = float(argv[1]), float(argv[2]), argv[3:]
    child = subprocess.Popen(cmd, start_new_session=True)

    def forward(signum, _frame):
        try:
            os.killpg(child.pid, signum)
        except OSError:
            pass
        try:
            child.wait(kill_after)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, forward)
    try:
        return child.wait(secs)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(kill_after)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            return 137
        return 124


if __name__ == "__main__":
    tool, args = sys.argv[1], sys.argv[2:]
    funcs = {"python": fake_python, "run_seed": run_seed, "smoke_reference": smoke_reference, "conda": conda,
             "nvidia-smi": nvidia_smi, "free": free_cmd, "timeout": timeout_cmd}
    sys.exit(funcs[tool](args))
'''


class Sandbox:
    """A temp repo + stub bin dir; ``run()`` executes the runner in it."""

    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.repo = tmp / "repo"
        self.bin = tmp / "bin"
        self.ckpt = tmp / "ckpts"
        self.bundles = tmp / "bundles"
        self.calls = tmp / "calls.jsonl"
        self.control_path = tmp / "control.json"
        self.gpu_free = tmp / "gpu_free"
        self.ckpt_seen = tmp / "ckpt_seen.txt"
        self.control = {}
        for rel in ("integration/run_all_gates.sh", "integration/lib.sh", "integration/summarize_bundle.py",
                    "baseline/config/grpo_base.sh"):
            (self.repo / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / rel, self.repo / rel)
        for d in ("baseline/results", "integration/gates", "serve/rl/tests"):
            (self.repo / d).mkdir(parents=True, exist_ok=True)
        self.bin.mkdir()
        self.ckpt.mkdir()
        impl = tmp / "stub_impl.py"
        impl.write_text(STUB_IMPL)
        for tool in ("python", "conda", "nvidia-smi", "free", "timeout"):
            self._script(self.bin / tool, f"#!/bin/sh\nexec '{sys.executable}' '{impl}' {tool} \"$@\"\n")
        os.symlink(sys.executable, self.bin / "python3")
        self._script(self.repo / "baseline/run_seed_sparsespec.sh",
                     f"#!/bin/sh\nexec '{sys.executable}' '{impl}' run_seed \"$@\"\n")
        self._script(self.repo / "integration/smoke_reference.sh",
                     f"#!/bin/sh\nexec '{sys.executable}' '{impl}' smoke_reference \"$@\"\n")
        self.meminfo = tmp / "meminfo"
        self.meminfo.write_text("MemTotal:       528280000 kB\nMemFree:        480000000 kB\n"
                                "MemAvailable:   500000000 kB\n")
        self.set_gpu_free(40000)
        (tmp / "home").mkdir()
        (tmp / "elsewhere").mkdir()

    @staticmethod
    def _script(path: Path, text: str):
        path.write_text(text)
        path.chmod(0o755)

    def set_gpu_free(self, mib: int):
        self.gpu_free.write_text(str(mib))

    def make_step25(self):
        path = self.ckpt / "vllm_baseline_seed0" / "hf" / "global_step_25"
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}")
        return path

    def env(self, **extra):
        env = {
            "PATH": f"{self.bin}:/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": str(self.tmp / "home"),
            "LANG": "C",
            "CKPT_ROOT": str(self.ckpt),
            "DATA_DIR": str(self.tmp / "data"),
            "BUNDLE_ROOT": str(self.bundles),
            "TIMEOUT_BIN": str(self.bin / "timeout"),
            "HOST_MEMINFO": str(self.meminfo),
            "RAG_POLL_S": "0.05",
            "GPU_WAIT_S": "2",
            "GPU_POLL_S": "1",
            "KILL_AFTER_S": "2",
            "STUB_CONTROL": str(self.control_path),
            "STUB_CALLS": str(self.calls),
            "STUB_REAL_PY": sys.executable,
            "STUB_GPU_FREE": str(self.gpu_free),
            "STUB_CKPT_SEEN": str(self.ckpt_seen),
            "STUB_ENVS": "verl-sparsespec verl-baseline",
        }
        env.update({k: str(v) for k, v in extra.items()})
        return env

    def run(self, timeout=180, **extra):
        self.control_path.write_text(json.dumps(self.control))
        proc = subprocess.run(["bash", str(self.repo / "integration/run_all_gates.sh")], cwd=self.tmp / "elsewhere",
                              env=self.env(**extra), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              timeout=timeout)
        return Result(self, proc.returncode, proc.stdout)

    def start(self, **extra):
        self.control_path.write_text(json.dumps(self.control))
        return subprocess.Popen(["bash", str(self.repo / "integration/run_all_gates.sh")], cwd=self.tmp / "elsewhere",
                                env=self.env(**extra), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                start_new_session=True)

    def read_calls(self):
        if not self.calls.exists():
            return []
        return [json.loads(line) for line in self.calls.read_text().splitlines() if line.strip()]


class Result:
    def __init__(self, sandbox, rc, out):
        self.sb, self.rc, self.out = sandbox, rc, out
        bundles = sorted(p for p in sandbox.bundles.glob("gates_*") if p.is_dir())
        self.bundle = bundles[-1] if bundles else None
        self.calls = sandbox.read_calls()

    def status(self, stage):
        hits = list((self.bundle / "status").glob(f"*_{stage}.status"))
        assert len(hits) == 1, (stage, hits)
        out = {}
        for line in hits[0].read_text().splitlines():
            key, _, value = line.partition("=")
            out[key] = value
        return out

    def summary(self):
        return (self.bundle / "SUMMARY.txt").read_text()

    def tarball(self):
        return self.sb.bundles / (self.bundle.name + ".tar.gz")

    def stage_calls(self):
        return [c for c in self.calls if c["key"] in ALL_STAGES]

    def call(self, key):
        hits = [c for c in self.calls if c["key"] == key]
        assert len(hits) == 1, (key, hits)
        return hits[0]

    def explain(self):
        return f"exit {self.rc}\n--- output ---\n{self.out[-6000:]}"


@pytest.fixture
def sb(tmp_path):
    return Sandbox(tmp_path)


def assert_no_runner_errors(res):
    errors = res.bundle / "runner_errors.txt"
    assert not errors.exists(), "unexpected failing commands inside the runner:\n" + errors.read_text()


# ---------------------------------------------------------------------------
# static checks
# ---------------------------------------------------------------------------
def test_bash_syntax():
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_shellcheck():
    proc = subprocess.run(["shellcheck", "-S", "warning", "-x", str(RUNNER)], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, cwd=REPO / "integration")
    assert proc.returncode == 0, proc.stdout


def test_no_bash4_constructs():
    text = RUNNER.read_text()
    for bad in ("mapfile", "readarray", "declare -A", "local -A", "wait -n", ",,}", "^^}", "&>>", "|&", "$EPOCHSECONDS"):
        assert bad not in text, bad


def test_help_lists_stages_and_options(sb):
    proc = subprocess.run(["bash", str(sb.repo / "integration/run_all_gates.sh"), "--help"], env=sb.env(),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.returncode == 0, proc.stdout
    for word in ALL_STAGES + ["SPARSESPEC_ENV", "BASELINE_ENV", "STAGES", "SKIP_STAGES", "MIN_FREE_GPU_GIB",
                              "BUNDLE_ROOT", "TIMEOUT_", "CKPT_ROOT", "SMOKE_STEPS", "UNMEASURED"]:
        assert word in proc.stdout, word
    assert not sb.bundles.exists()


def test_bad_stage_name_refused_before_anything(sb):
    res = sb.run(STAGES="gate_3b,gate_9z")
    assert res.rc == 2
    assert "gate_9z" in res.out
    assert res.calls == []
    assert res.bundle is None


# ---------------------------------------------------------------------------
# full runs
# ---------------------------------------------------------------------------
def test_full_run_mixed_outcome(sb):
    step25 = sb.make_step25()
    # the previous smoke attempt's checkpoint (must go) and an unrelated run's (must stay)
    old = sb.ckpt / "gates_smoke_worst_case" / "global_step_3"
    old.mkdir(parents=True)
    keep = sb.ckpt / "vllm_baseline_seed0" / "hf" / "global_step_25" / "config.json"
    other = sb.ckpt / "sparsespec_seed0" / "global_step_25"
    other.mkdir(parents=True)
    sb.control = {
        "exit": {"reference_plumbing": 1, "gate_3b_vllm": 1, "gate_3c_step25": 1, "smoke_host_guard": 2},
        "underpowered": ["gate_3c_step25"],
    }
    res = sb.run()
    assert res.rc == 1, res.explain()
    assert_no_runner_errors(res)

    # every stage in order, each in the right env
    expected = [s for s in ALL_STAGES if s != "env"]
    assert [c["key"] for c in res.stage_calls()] == expected, res.explain()
    for call in res.stage_calls():
        want = "verl-baseline" if call["key"] in BASELINE_STAGES else "verl-sparsespec"
        assert call["env"] == want, call
        assert call["vars"]["PYTHONPATH"] == str(sb.repo), call
        assert call["cwd"] == str(sb.repo), call
    for key in ("probe", "pip"):
        assert {c["env"] for c in res.calls if c["key"] == key} == {"verl-sparsespec", "verl-baseline"}

    # a failing stage does not stop later independent stages
    assert res.status("reference_plumbing")["status"] == "FAIL"
    assert res.status("reference_plumbing")["exit_status"] == "1"
    assert res.status("reference_plumbing")["reason"] == "exit 1"
    for stage in ("gpu_unit_tests", "gpu_unit_tests_host_guard", "gpu_unit_tests_offload",
                  "gpu_unit_tests_offload_host_guard", "gate_3b", "gate_3c_synthetic", "gate_3c_vllm_reference",
                  "smoke_worst_case", "smoke_worst_case_check", "smoke_no_graph", "smoke_no_logprobs", "smoke_compare"):
        assert res.status(stage)["status"] == "PASS", (stage, res.status(stage))

    # GPU unit-test variants
    v = {k: res.call(k)["vars"] for k in ("gpu_unit_tests", "gpu_unit_tests_host_guard", "gpu_unit_tests_offload",
                                          "gpu_unit_tests_offload_host_guard")}
    assert "SPARSESPEC_TEST_ADMISSION" not in v["gpu_unit_tests"] and "SPARSESPEC_TEST_OFFLOAD_STRESS" not in v["gpu_unit_tests"]
    assert v["gpu_unit_tests_host_guard"]["SPARSESPEC_TEST_ADMISSION"] == "host_guard"
    assert v["gpu_unit_tests_offload"]["SPARSESPEC_TEST_OFFLOAD_STRESS"] == "1"
    assert v["gpu_unit_tests_offload"]["SPARSESPEC_TEST_ADMISSION"] == "worst_case"
    assert v["gpu_unit_tests_offload_host_guard"]["SPARSESPEC_TEST_ADMISSION"] == "host_guard"
    gpu_argv = res.call("gpu_unit_tests")["argv"]
    assert gpu_argv[:6] == ["-m", "pytest", "serve/rl/tests/test_engine_gpu.py", "-v", "-s", "--log-cli-level=INFO"]
    cpu_argv = res.call("cpu_tests")["argv"]
    assert "--ignore=serve/rl/tests/test_engine_gpu.py" in cpu_argv

    # gate_3b_vllm failed -> two-way gate, three-way reported as not done
    argv3b = res.call("gate_3b")["argv"]
    assert "--vllm-json" not in argv3b and argv3b[argv3b.index("--tag") + 1] == "sparsespec"
    assert "THREE-WAY NOT DONE" in res.status("gate_3b")["note"]

    # 3c step-25 on the real checkpoint path; vLLM reference on the same checkpoint
    argv25 = res.call("gate_3c_step25")["argv"]
    assert argv25[argv25.index("--ckpt") + 1] == str(step25)
    assert res.status("gate_3c_step25")["status"] == "FAIL"
    assert "negative_control_underpowered" in res.status("gate_3c_step25")["reason"]
    ref = res.call("gate_3c_vllm_reference")["argv"]
    assert ref[ref.index("--ckpt") + 1] == str(step25) and ref[ref.index("--tag") + 1] == "step25"

    # smoke runs: per-run settings, own checkpoint removed first, nothing else touched
    smoke = {k: res.call(k)["vars"] for k in ("smoke_worst_case", "smoke_host_guard", "smoke_no_graph", "smoke_no_logprobs")}
    assert (smoke["smoke_worst_case"]["SPARSESPEC_ADMISSION"], smoke["smoke_worst_case"]["SPARSESPEC_CUDA_GRAPH"],
            smoke["smoke_worst_case"]["SPARSESPEC_CALC_LOGPROBS"], smoke["smoke_worst_case"]["STEPS"]) == (
        "worst_case", "True", "True", "3")
    assert smoke["smoke_host_guard"]["SPARSESPEC_ADMISSION"] == "host_guard"
    assert smoke["smoke_no_graph"]["SPARSESPEC_CUDA_GRAPH"] == "False"
    assert (smoke["smoke_no_logprobs"]["SPARSESPEC_CALC_LOGPROBS"], smoke["smoke_no_logprobs"]["STEPS"]) == ("False", "1")
    for k, vars_ in smoke.items():
        assert vars_["EXP_NAME"] == "gates_smoke_" + k[len("smoke_"):]
        assert vars_["SPARSESPEC_ENGINE"] == "sparsespec" and vars_["SEED"] == "0"
    seen = dict(line.split() for line in sb.ckpt_seen.read_text().splitlines())
    assert seen == {"gates_smoke_worst_case": "False", "gates_smoke_host_guard": "False",
                    "gates_smoke_no_graph": "False", "gates_smoke_no_logprobs": "False"}
    assert keep.exists() and other.exists()
    for label in ("worst_case", "host_guard", "no_graph", "no_logprobs"):
        overrides = res.bundle / "smoke" / f"gates_smoke_{label}.overrides.txt"
        assert "trainer.experiment_name=gates_smoke_" + label in overrides.read_text()
        assert (res.bundle / "smoke" / f"gates_smoke_{label}.timing.csv").exists()
        assert (res.bundle / "smoke" / f"gates_smoke_{label}.log").exists()
    # the crashed host_guard run still left a 1-step CSV: its check ran and the run is FAIL
    assert res.status("smoke_host_guard")["status"] == "FAIL"
    assert "crashed" in res.status("smoke_host_guard")["reason"]
    assert res.status("smoke_host_guard_check")["status"] == "PASS"

    cmp_argv = res.call("smoke_compare")["argv"]
    runs = [cmp_argv[i + 1].split("=")[0] for i, a in enumerate(cmp_argv) if a == "--run"]
    assert runs == ["worst_case", "host_guard", "no_graph", "no_logprobs"]
    assert cmp_argv[cmp_argv.index("--logprob-pair") + 1] == "worst_case,no_logprobs"
    assert cmp_argv[cmp_argv.index("--host-mem-total-gb") + 1] == "503.81"
    assert cmp_argv[cmp_argv.index("--host-mem-available-gb") + 1] == "476.84"
    assert cmp_argv[cmp_argv.index("--gpu-total-gb") + 1] == "40.00"

    # env record
    env_dir = res.bundle / "env"
    for name in ("host.txt", "git.txt", "nvidia_smi.txt", "nvidia_smi_query.csv", "nvidia_smi_apps.csv", "paths.txt",
                 "facts.txt", "verl-sparsespec/versions.txt", "verl-sparsespec/pip_freeze.txt",
                 "verl-baseline/versions.txt", "verl-baseline/conda_list.txt"):
        assert (env_dir / name).exists(), name
    assert "Mem:" in (env_dir / "host.txt").read_text()
    paths = (env_dir / "paths.txt").read_text()
    assert f"{step25}: exists" in paths and "math_l3to5_train.parquet: MISSING" in paths
    assert "CUDA Version" in (env_dir / "nvidia_smi.txt").read_text()

    # summary: one screen, mixed outcome, explicit underpowered sentence
    summary = res.summary()
    lines = summary.splitlines()
    assert len(lines) <= 46, summary
    too_wide = [line for line in lines if len(line) > 110 and not line.startswith("send this file back: ")]
    assert not too_wide, too_wide  # the tarball path is never clipped
    assert "FAILED ONLY because the checkpoint is too close to the base model" in summary
    assert "Rerun on the step-50 checkpoint" in summary
    assert "VERDICT: NOT PASSED" in summary and "3c step-25 FAIL (underpowered)" in summary
    assert "3b PASS" in summary and "GPU unit tests PASS" in summary and "3d PASS" in summary
    assert "9 passed, 0 failed" in summary
    assert "reloaded 0.0081, round-trip 0.0072, neg-ctrl 0.318, bitwise ok" in summary
    assert "vLLM mean |dlogp| 0.0123 (n=4096, ckpt global_step_25)" in summary
    assert "1.5: 16 GiB pinned, 27/41 in flight, fits yes; 3: 32 GiB pinned, 46/64 in flight, fits yes" in flat(summary)
    assert "worst_case reading: batch_size_mean at the admission cap" in summary
    assert "503.8 GiB total" in summary
    assert "log-prob cost" in summary and "with 430.0 s, without 420.0 s" in summary
    assert "CUDA out of memory" in summary  # the crashed smoke's last error line
    assert "send this file back: " + str(res.tarball()) in summary
    assert "THREE-WAY NOT DONE" in summary
    assert summary.strip() in res.out  # printed too
    assert (res.bundle / "verdict.txt").read_text().strip() == "FAIL"

    # tarball next to the directory, with every stage log; path and size are the last lines
    with tarfile.open(res.tarball()) as tar:
        names = tar.getnames()
    for i, stage in enumerate(ALL_STAGES, 1):
        assert f"{res.bundle.name}/logs/{i:02d}_{stage}.log" in names, stage
        assert f"{res.bundle.name}/status/{i:02d}_{stage}.status" in names, stage
    for rel in ("SUMMARY.txt", "env/facts.txt", "gates/gate_c_reload_sparsespec_step25.json",
                "gates/gate_d_smoke_gates_smoke_worst_case.json", "smoke/gates_smoke_no_graph.timing.csv"):
        assert f"{res.bundle.name}/{rel}" in names, rel
    last = [line for line in res.out.strip().splitlines() if line.strip()][-3:]
    assert last[1] == f"[run_all_gates] send this file back: {res.tarball()}"
    assert last[2].startswith("[run_all_gates] size: ") and "MiB" in last[2]


def flat(text):
    return " ".join(text.split())


def test_all_pass_without_step25_checkpoint(sb):
    res = sb.run(SMOKE_STEPS=2)
    assert res.rc == 0, res.explain()
    assert_no_runner_errors(res)
    st = res.status("gate_3c_step25")
    assert st["status"] == "SKIP" and st["skip_kind"] == "missing_checkpoint"
    assert str(sb.ckpt / "vllm_baseline_seed0" / "hf" / "global_step_25") in st["reason"]
    assert "gate_3c_step25" not in [c["key"] for c in res.calls]
    # three-way 3b with the vLLM JSON from the bundle
    argv = res.call("gate_3b")["argv"]
    assert argv[argv.index("--vllm-json") + 1] == str(res.bundle / "gates" / "gate_b_vllm_greedy.json")
    assert argv[argv.index("--tag") + 1] == "three_way"
    # vLLM reference falls back to the synthetic checkpoint that gate_3c_synthetic created
    ref = res.call("gate_3c_vllm_reference")["argv"]
    assert ref[ref.index("--ckpt") + 1] == str(sb.ckpt / "synthetic" / "relnoise0.02_seed0")
    assert ref[ref.index("--tag") + 1] == "synthetic"
    assert res.call("smoke_worst_case")["vars"]["STEPS"] == "2"
    assert res.call("smoke_no_logprobs")["vars"]["STEPS"] == "1"
    summary = flat(res.summary())
    assert "VERDICT: ALL REQUIRED GATES PASSED" in summary
    assert "3c step-25 not run (no checkpoint)" in summary
    assert "step-25 checkpoint missing: expected" in summary
    for pair in ("sparsespec_vs_hf", "vllm_vs_hf", "sparsespec_vs_vllm"):
        assert pair in summary
    assert "earliest at 113, largest margin 0.31, rule margin<=1.0" in summary
    assert len(res.summary().splitlines()) <= 46
    assert (res.bundle / "verdict.txt").read_text().strip() == "PASS"


def test_dependents_skipped_when_engine_import_fails(sb):
    sb.control = {"exit": {"engine_import": 1}}
    res = sb.run()
    assert res.rc == 1, res.explain()
    assert_no_runner_errors(res)
    assert res.status("engine_import")["status"] == "FAIL"
    for stage in ("gpu_unit_tests", "gpu_unit_tests_host_guard", "gpu_unit_tests_offload",
                  "gpu_unit_tests_offload_host_guard", "gate_3b", "gate_3c_synthetic", "smoke_worst_case",
                  "smoke_host_guard", "smoke_no_graph", "smoke_no_logprobs"):
        st = res.status(stage)
        assert st["status"] == "SKIP" and st["reason"].startswith("dependency engine_import FAIL"), (stage, st)
    for stage in ("smoke_worst_case_check", "smoke_compare"):
        assert res.status(stage)["status"] == "SKIP"
    # independent stages still ran
    keys = [c["key"] for c in res.stage_calls()]
    assert keys == ["cpu_tests", "engine_import", "reference_plumbing", "gate_3b_vllm"], keys
    # no step-25 and no synthetic checkpoint: the vLLM reference has nothing to measure
    st = res.status("gate_3c_vllm_reference")
    assert st["status"] == "SKIP" and "synthetic checkpoint" in st["reason"]
    summary = flat(res.summary())
    assert "dependency engine_import FAIL" in summary
    assert "ModuleNotFoundError: No module named 'flashinfer'" in summary
    assert "VERDICT: NOT PASSED" in summary and "3d not run" in summary


def test_refuses_to_start_when_gpu_busy(sb):
    sb.set_gpu_free(10000)
    res = sb.run()
    assert res.rc == 2, res.explain()
    assert "REFUSING TO START" in res.out
    assert "38.5 GiB" in res.out and "MIN_FREE_GPU_GIB" in res.out and "424242" in res.out
    assert [c["key"] for c in res.stage_calls()] == []
    assert {c["key"] for c in res.calls} <= {"probe", "pip", "pytest_check"}
    for name in ("env/facts.txt", "env/nvidia_smi.txt", "env/host.txt", "env/verl-sparsespec/versions.txt",
                 "REFUSED.txt", "SUMMARY.txt"):
        assert (res.bundle / name).exists(), name
    assert res.status("env")["status"] == "PASS"
    assert res.status("gate_3b")["status"] == "SKIP" and "refused to start" in res.status("gate_3b")["reason"]
    assert res.tarball().exists()
    assert "REFUSED TO START" in res.summary()


def test_min_free_override_and_cpu_only_selection(sb):
    sb.set_gpu_free(10000)
    res = sb.run(MIN_FREE_GPU_GIB=5, STAGES="gate_3b_vllm")
    assert res.rc == 1, res.explain()  # required gates not run -> not passed, but it started
    assert [c["key"] for c in res.stage_calls()] == ["gate_3b_vllm"]
    assert res.status("gate_3b")["skip_kind"] == "not_selected"
    # a CPU-only selection does not need the GPU at all
    sb2 = Sandbox(sb.tmp / "second")
    sb2.set_gpu_free(100)
    res2 = sb2.run(STAGES="cpu_tests,engine_import")
    assert [c["key"] for c in res2.stage_calls()] == ["cpu_tests", "engine_import"], res2.explain()


def test_gpu_busy_before_a_later_stage_is_a_skip(sb):
    sb.make_step25()
    sb.control = {"gpu_free_after": {"gate_3b_vllm": 9000}}
    res = sb.run(STAGES="gate_3b_vllm,gate_3b,cpu_tests,gate_3c_step25")
    assert_no_runner_errors(res)
    assert res.status("gate_3b_vllm")["status"] == "PASS"
    for stage in ("gate_3b", "gate_3c_step25"):
        st = res.status(stage)
        assert st["status"] == "SKIP" and st["skip_kind"] == "gpu_busy", st
        assert "GPU busy" in st["reason"] and "after waiting" in st["reason"]
        log = (res.bundle / st["log"]).read_text()
        assert "424242" in log and "compute processes" in log
    assert res.status("cpu_tests")["status"] == "PASS"
    assert [c["key"] for c in res.stage_calls()] == ["cpu_tests", "gate_3b_vllm"]


def test_missing_baseline_env_and_missing_pytest(sb):
    sb.control = {"pytest_missing": True}
    res = sb.run(STUB_ENVS="verl-sparsespec", STAGES="cpu_tests,gpu_unit_tests,gate_3b_vllm,engine_import")
    assert "conda env(s) missing: verl-baseline" in res.out
    st = res.status("gate_3b_vllm")
    assert st["status"] == "SKIP" and "conda env 'verl-baseline' does not exist" in st["reason"]
    for stage in ("cpu_tests", "gpu_unit_tests"):
        st = res.status(stage)
        assert st["status"] == "SKIP" and "pytest is not installed in verl-sparsespec" in st["reason"], st
    assert [c["key"] for c in res.stage_calls()] == ["engine_import"]
    assert res.status("env")["status"] == "FAIL"  # one env could not be recorded


def test_stale_smoke_csv_from_an_earlier_attempt_is_not_used(sb):
    stale = sb.repo / "baseline" / "results" / "gates_smoke_worst_case.timing.csv"
    stale.write_text("step,timing_s/gen\n1,1.0\n2,1.0\n3,1.0\n")
    sb.control = {"no_csv": ["smoke_worst_case"]}
    res = sb.run(STAGES="smoke_worst_case")
    st = res.status("smoke_worst_case")
    assert st["status"] == "FAIL" and "no timing CSV" in st["reason"], st
    assert res.status("smoke_worst_case_check")["status"] == "SKIP"
    assert (res.bundle / "smoke" / "previous_attempt" / stale.name).read_text().startswith("step,")
    assert not stale.exists() and not (res.bundle / "smoke" / stale.name).exists()
    assert "gates_smoke_worst_case_check" not in [c["key"] for c in res.calls]


def test_all_skipped_gpu_tests_are_not_a_pass(sb):
    sb.control = {"all_skipped": ["gpu_unit_tests"]}
    res = sb.run(STAGES="gpu_unit_tests")
    st = res.status("gpu_unit_tests")
    assert st["status"] == "FAIL" and "no test passed" in st["reason"], st


def test_stage_timeout_stops_only_that_stage(sb):
    sb.control = {"hang": ["gpu_unit_tests"]}
    res = sb.run(STAGES="gpu_unit_tests,gpu_unit_tests_host_guard", TIMEOUT_GPU_UNIT_TESTS=2)
    st = res.status("gpu_unit_tests")
    assert st["status"] == "FAIL" and "timed out after 2s" in st["reason"], st
    assert res.status("gpu_unit_tests_host_guard")["status"] == "PASS"
    hung = res.call("gpu_unit_tests")["pid"]
    with pytest.raises(OSError):
        os.kill(hung, 0)


def _wait_for_call(sb, key, proc, limit=60):
    deadline = time.time() + limit
    while time.time() < deadline:
        if any(c["key"] == key for c in sb.read_calls()):
            return
        if proc.poll() is not None:
            raise AssertionError("runner exited early:\n" + proc.stdout.read())
        time.sleep(0.1)
    raise AssertionError(f"{key} never started")


@pytest.mark.parametrize("how", ["ctrl_c_to_process_group", "sigterm_no_timeout_cmd"])
def test_interrupted_run_still_writes_summary_and_tarball(sb, how):
    sb.control = {"hang": ["gate_3b"]}
    extra = {"STAGES": "gate_3b_vllm,gate_3b,gate_3c_synthetic"}
    if how == "sigterm_no_timeout_cmd":
        extra["TIMEOUT_BIN"] = "none"
    proc = sb.start(**extra)
    try:
        _wait_for_call(sb, "gate_3b", proc)
        time.sleep(0.5)
        if how == "ctrl_c_to_process_group":
            os.killpg(proc.pid, signal.SIGINT)
            expected_rc = 130
        else:
            os.kill(proc.pid, signal.SIGTERM)
            expected_rc = 143
        out, _ = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    assert proc.returncode == expected_rc, out
    res = Result(sb, proc.returncode, out)
    st = res.status("gate_3b")
    assert st["status"] == "FAIL" and "interrupted by SIG" in st["reason"], st
    assert res.status("gate_3c_synthetic")["status"] == "SKIP"
    assert "interrupted" in res.status("gate_3c_synthetic")["reason"]
    assert res.status("gate_3b_vllm")["status"] == "PASS"
    assert (res.bundle / "SUMMARY.txt").exists()
    assert "interrupted by SIG" in res.summary()
    with tarfile.open(res.tarball()) as tar:
        names = tar.getnames()
    assert f"{res.bundle.name}/SUMMARY.txt" in names
    assert sorted(n for n in names if "/logs/" in n and n.endswith(".log") and "/._" not in n) == sorted(
        f"{res.bundle.name}/logs/{i:02d}_{s}.log" for i, s in enumerate(ALL_STAGES, 1))
    hung = res.call("gate_3b")["pid"]
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            os.kill(hung, 0)
        except OSError:
            break
        time.sleep(0.2)
    else:
        raise AssertionError("the runner's own hung stage process was not stopped")
    main_logs = list(res.bundle.glob("run_all_gates_*.log"))
    assert main_logs and "received SIG" in main_logs[0].read_text()


# ---------------------------------------------------------------------------
# pieces
# ---------------------------------------------------------------------------
def _source(sb, snippet, **env):
    script = f". '{sb.repo}/integration/run_all_gates.sh'\n{snippet}\n"
    return subprocess.run(["bash", "-c", script], env=sb.env(**env), stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)


def test_deletion_guard(sb):
    victim = sb.ckpt / "sparsespec_seed0"
    victim.mkdir()
    (sb.ckpt / "gates_smoke_x").mkdir()
    outside = sb.tmp / "outside"
    outside.mkdir()
    for name in ("sparsespec_seed0", "../outside", "gates_smoke_../../outside", "gates_smoke_a/b", "gates_smoke_",
                 "", "gates_smoke_x y", "smoke_worst_case"):
        proc = _source(sb, f"rag_remove_smoke_ckpt '{name}'; echo rc=$?")
        assert "rc=1" in proc.stdout and "REFUSED" in proc.stdout, (name, proc.stdout)
    proc = _source(sb, "rag_remove_smoke_ckpt gates_smoke_x; echo rc=$?", CKPT_ROOT="relative/ckpts")
    assert "rc=1" in proc.stdout and "not an absolute path" in proc.stdout
    proc = _source(sb, "rag_remove_smoke_ckpt gates_smoke_x; echo rc=$?", CKPT_ROOT="")
    assert "rc=1" in proc.stdout
    assert victim.exists() and outside.exists() and (sb.ckpt / "gates_smoke_x").exists()
    proc = _source(sb, "rag_remove_smoke_ckpt gates_smoke_x; echo rc=$?")
    assert "rc=0" in proc.stdout and not (sb.ckpt / "gates_smoke_x").exists() and victim.exists()
    proc = _source(sb, "rag_remove_smoke_ckpt gates_smoke_never_there; echo rc=$?")
    assert "rc=0" in proc.stdout and "no earlier checkpoint" in proc.stdout


def test_probe_script_runs_on_a_real_python(sb, tmp_path):
    bundle = tmp_path / "b"
    (bundle / "env").mkdir(parents=True)
    proc = _source(sb, f"BUNDLE='{bundle}'; rag_write_probe")
    assert proc.returncode == 0, proc.stdout
    out = subprocess.run([sys.executable, str(bundle / "env" / "probe_env.py")], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, timeout=300, env={**os.environ, "HF_TOKEN": "secret"})
    assert out.returncode == 0, out.stdout
    assert "VERSION python " in out.stdout and "VERSION vllm " in out.stdout and "PATCH " in out.stdout
    assert "secret" not in out.stdout
    for helper, expect in (("check_pytest.py", "pytest "), ("check_engine_import.py", None)):
        run = subprocess.run([sys.executable, str(bundle / "env" / helper)], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, cwd=sb.tmp)
        if expect:
            assert run.returncode == 0 and expect in run.stdout, run.stdout
        else:  # no repo on the path here: must fail loudly, not pass
            assert run.returncode != 0 and "serve" in run.stdout, run.stdout


def test_summarizer_survives_missing_and_malformed_files(tmp_path):
    b = tmp_path / "bundle"
    for d in ("env", "status", "logs", "gates", "smoke"):
        (b / d).mkdir(parents=True)
    (b / "env" / "facts.txt").write_text("hostname=box\ngarbage line\ngpu_total_mib=not-a-number\n")
    (b / "status" / "01_env.status").write_text("name=env\nstatus=PASS\nseconds=abc\n")
    (b / "status" / "05_gpu_unit_tests.status").write_text("name=gpu_unit_tests\nstatus=FAIL\nreason=exit status 1\n")
    (b / "status" / "10_gate_3b.status").write_text("name=gate_3b\nstatus=PASS\n")
    (b / "status" / "11_gate_3c_synthetic.status").write_text("name=gate_3c_synthetic\nstatus=PASS\n")
    (b / "status" / "12_gate_3c_step25.status").write_text("name=gate_3c_step25\nstatus=FAIL\n")
    (b / "status" / "14_smoke_worst_case.status").write_text("name=smoke_worst_case\nstatus=PASS\n")
    (b / "status" / "99_weird.status").write_bytes(b"\xff\xfe\x00garbage")
    (b / "gates" / "gate_b_greedy_three_way.json").write_text("{not json")
    (b / "gates" / "gate_c_reload_sparsespec_synthetic.json").write_text("[1, 2, 3]")
    (b / "gates" / "gate_c_reload_sparsespec_step25.json").write_text(json.dumps(
        {"pass": False, "failure_class": "negative_control_underpowered", "stats": {"reloaded": {"mean": "x"}},
         "checks": "nope", "rerun": None}))
    (b / "gates" / "smoke_compare.json").write_text(json.dumps({"host_memory": {"rows": [None, {"pinned_gb": "x"}]},
                                                                 "runs": "bad"}))
    (b / "smoke" / "gates_smoke_worst_case.timing.csv").write_text("no,step,column\n1,2,3\n")
    (b / "smoke" / "gates_smoke_no_graph.timing.csv").write_text("step,timing_s/gen\n1,nan\nx,3\n2,\n")
    (b / "logs" / "05_gpu_unit_tests.log").write_bytes(b"\x00\xffbinary\nRuntimeError: boom\n")
    proc = subprocess.run([sys.executable, str(SUMMARIZER), str(b), "--tarball", "/x/t.tar.gz"],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.returncode == 0, proc.stdout
    text = (b / "SUMMARY.txt").read_text()
    assert "n/a" in text and "Traceback" not in text and "summary failed" not in text
    assert "FAILED ONLY because the checkpoint is too close" in text
    assert "RuntimeError: boom" in text
    assert "VERDICT: NOT PASSED" in text and "send this file back: /x/t.tar.gz" in text
    assert (b / "verdict.txt").read_text().strip() == "FAIL"
    # an empty bundle directory and a missing one
    empty = tmp_path / "empty"
    empty.mkdir()
    proc = subprocess.run([sys.executable, str(SUMMARIZER), str(empty)], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    assert proc.returncode == 0 and "VERDICT: NOT PASSED" in proc.stdout, proc.stdout
    # --json-get
    j = tmp_path / "j.json"
    j.write_text(json.dumps({"checkpoint": {"dir": "/ck/synth"}, "failure_class": None}))
    get = lambda *a: subprocess.run([sys.executable, str(SUMMARIZER), "--json-get", *a], stdout=subprocess.PIPE,
                                    text=True)
    assert get(str(j), "checkpoint.dir").stdout.strip() == "/ck/synth"
    assert get(str(j), "failure_class").returncode == 1
    assert get(str(tmp_path / "missing.json"), "x").returncode == 1


def test_pytest_counts_parsing():
    sys.path.insert(0, str(REPO / "integration"))
    try:
        import summarize_bundle as sbm
    finally:
        sys.path.pop(0)
    log = textwrap.dedent("""
        ===== 3 passed in 1.00s =====
        FAILED serve/rl/tests/test_engine_gpu.py::test_sleep - AssertionError
        ============ 1 failed, 11 passed, 2 skipped, 3 warnings in 301.12s (0:05:01) ============
    """)
    assert sbm.pytest_counts("x\n477 passed, 2 skipped in 61.20s (0:01:01)\n") == {"passed": 477, "skipped": 2}
    assert sbm.pytest_counts("====== no tests ran in 0.01s ======") == {}
    assert sbm.pytest_counts(log) == {"failed": 1, "passed": 11, "skipped": 2, "warnings": 3}
    assert sbm.pytest_text(log).startswith("11 passed, 1 failed, 2 skipped; first: test_engine_gpu.py::test_sleep")
    assert sbm.pytest_counts("no summary here") is None
