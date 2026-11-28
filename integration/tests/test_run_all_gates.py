"""Stub-driven tests of integration/run_all_gates.sh and integration/summarize_bundle.py.

No conda, GPU, vLLM or verl is needed: each test copies the runner, lib.sh,
summarize_bundle.py and baseline/config/grpo_base.sh into a temp "repo" and
puts stub executables first on PATH:

    conda        records the env name of every ``conda run`` and execs the command
    nvidia-smi   realistic query / table output; free memory read from a file
    python       dispatches on its arguments: pytest, the gate CLIs, the import
                 check and pip freeze are faked; anything else runs the real python
    python3      the real python (the runner's summary step)
    timeout      a GNU coreutils timeout stand-in: it leads a new process group that the command
                 inherits, and returns as soon as its DIRECT child exits (exit 124)
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
import tempfile
import textwrap
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "integration" / "run_all_gates.sh"
SUMMARIZER = REPO / "integration" / "summarize_bundle.py"

ALL_STAGES = (
    "env cpu_tests engine_import reference_plumbing gpu_unit_tests gpu_unit_tests_host_guard gpu_unit_tests_offload "
    "gpu_unit_tests_offload_host_guard gate_3b_vllm gate_3b gate_3c_synthetic gate_3c_vllm_reference gate_3c_step25 "
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
        "CKPT_ROOT", "DATA_DIR", "MODEL_PATH", "INSTALL_PREFIX", "PREFIX")


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
    if key in CONTROL.get("trap_term", []):  # like the Ray driver / workers and vLLM's EngineCore
        signal.signal(signal.SIGTERM, lambda *_: None)
    if key in CONTROL.get("hang", []):
        time.sleep(600)
    return int(CONTROL.get("exit", {}).get(key, 0))


def colored(line):
    """pytest's own rule: color when FORCE_COLOR / PY_COLORS ask for it, unless --color=no."""
    return line if "--color=no" in sys.argv or not (os.environ.get("FORCE_COLOR") or os.environ.get("PY_COLORS") == "1") \
        else "\x1b[32m\x1b[1m" + line + "\x1b[0m"


def crash(prefix):
    """what integration.gates.common.run_main prints for an unexpected exception (exit 3)"""
    print("%s CRASHED: unexpected RuntimeError; no verdict was computed (exit status 3); traceback below" % prefix)
    print("Traceback (most recent call last):\nRuntimeError: CUDA error: an illegal memory access", file=sys.stderr)
    return 3


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
            print(colored("=" * 20 + " 9 skipped in 0.42s " + "=" * 20))
        elif rc == 1:
            print("FAILED serve/rl/tests/test_engine_gpu.py::test_sleep_wake - AssertionError: memory not freed")
            print(colored("=" * 20 + " 1 failed, 8 passed in 12.10s " + "=" * 20))
        elif "-q" in argv:
            print(colored("9 passed, 1 skipped in 12.10s"))  # pytest -q: no '=' decoration
        else:
            print(colored("=" * 20 + " 9 passed, 1 skipped in 12.10s " + "=" * 20))
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
        state = CONTROL.get("patch_state", "applied") if env == "verl-sparsespec" else "absent"
        print("PATCH %s fsdp_workers_hunk=%s ray_trainer_hunk=%s" % (state, state == "applied", state == "applied"))
        nvcc = CONTROL.get("nvcc_state", "found")
        if nvcc == "unknown":
            print("NVCC unknown ImportError: no module named setuptools")
        elif nvcc != "none":  # "none": a probe that prints no NVCC line at all
            print("NVCC %s nvcc=/usr/local/cuda/bin/nvcc torch_CUDA_HOME=None" % nvcc)
        return 0
    out_dir = flag(argv, "--out-dir")
    if argv and argv[0] == "integration/gates/gate_b_vllm_greedy.py":
        record("gate_3b_vllm", argv)
        rc = behave("gate_3b_vllm")
        if rc == 3:
            return crash("[gate 3b vllm]")
        if rc == 0 or (rc == 1 and CONTROL.get("vllm_json_on_fail")):  # exit 1: a step was not greedy
            write_json(flag(argv, "--out"), {"pass": rc == 0, "summary": summary_3b()})
        if CONTROL.get("vllm_hang_after_json"):  # the token file is complete, then vLLM never shuts down
            sys.stdout.flush()
            time.sleep(600)
        print("[gate 3b vllm] %s" % ("PASS" if rc == 0 else "FAIL"))
        return rc
    if argv[:2] == ["-m", "integration.gates.gate_b_greedy"]:
        record("gate_3b", argv)
        rc = behave("gate_3b")
        tag = flag(argv, "--tag")
        if "gate_3b" in CONTROL.get("refuse", []):  # what gate_b_greedy.main writes on InputMismatch
            reason = ("the vLLM file %s prompts are not the prompts of this run: prompt 3 differs at token 4 "
                      "(lengths 212 vs 213; ids 151644 vs 872)" % flag(argv, "--vllm-json"))
            write_json(os.path.join(out_dir, "gate_b_greedy_%s.json" % tag),
                       {"gate": "3b_greedy", "pass": False, "refused": True, "reason": reason, "three_way": True})
            print("[gate 3b] REFUSED: " + reason)
            return 2
        if "--vllm-json" in argv:
            data = {"pass": rc == 0, "three_way": True, "rule": "margin<=1.0", "config": {"engine": "sparsespec"},
                    "pairs": {
                        "sparsespec_vs_hf": dict(summary_3b(), **{"pass": True}),
                        "vllm_vs_hf": dict(summary_3b(16, 15), **{"pass": True}),
                        "sparsespec_vs_vllm": dict(summary_3b(16, 13), **{"pass": rc == 0 and not CONTROL.get("r2_fail")})}}
            if CONTROL.get("r2_fail"):  # only R2 fails: every pair line still says PASS
                data["pairs"]["sparsespec_vs_vllm"]["pass"] = True
                data["failed_checks"] = [{"name": "R2_profile_significance", "blame": ["engine"]}]
                data["three_way_result"] = {"profile": {"rule": "significance", "alpha": 0.05, "p_value": 0.0166,
                                                        "engine": {"divergences": 5, "earliest": 40},
                                                        "vllm": {"divergences": 0, "earliest": None}}}
        else:
            data = {"pass": rc == 0, "three_way": False, "config": {"engine": "sparsespec", "tie_margin": 1.0},
                    "summary": summary_3b()}
        write_json(os.path.join(out_dir, "gate_b_greedy_%s.json" % tag), data)
        print("[gate 3b] %s" % ("PASS" if rc == 0 else "FAIL"))
        return rc
    if argv[:2] == ["-m", "integration.gates.gate_c_reload"]:
        tag = flag(argv, "--tag")
        key = {"sparsespec_synthetic": "gate_3c_synthetic", "sparsespec_step25": "gate_3c_step25"}[tag]
        record(key, argv)
        rc = behave(key)
        if rc == 3:
            return crash("[gate 3c]")
        if rc == 1 and key in CONTROL.get("no_json", []):  # a SystemExit("...") stop before any verdict
            print("[gate 3c] FAIL checkpoint not found: %s" % flag(argv, "--ckpt", "synthetic"), file=sys.stderr)
            return 1
        if "--synthetic" in argv:
            ckpt = os.path.join(os.environ["CKPT_ROOT"], "synthetic", "relnoise0.02_seed0")
            os.makedirs(ckpt, exist_ok=True)
        else:
            ckpt = flag(argv, "--ckpt")
        data = {"gate": "3c_reload", "pass": rc == 0, "failure_class": None, "checkpoint": {"dir": ckpt},
                "stats": {"reloaded": {"mean": 0.0081}, "base_after_roundtrip": {"mean": 0.0072},
                          "negative_control": {"mean": 0.318 if rc == 0 else 0.009}},
                "checks": [{"name": "reload1_bitwise", "ok": True}, {"name": "reload3_bitwise", "ok": True}]}
        if rc == 1 and key in CONTROL.get("underpowered", []) + CONTROL.get("inconclusive", []):
            data["failure_class"] = ("negative_control_underpowered" if key in CONTROL.get("underpowered", [])
                                     else "negative_control_inconclusive")
            data["failed_checks"] = ["negative_control_clearly_larger"]
            data["rerun"] = {"next_step": 50, "next_ckpt": ckpt.replace("global_step_25", "global_step_50")}
        if os.environ.get("STUB_TORCH_GB"):  # common.environment_info: torch's total_memory / GiB
            data["environment"] = {"gpu_total_gb": float(os.environ["STUB_TORCH_GB"])}
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
    if os.environ.get("RAY_TMPDIR"):  # what ray.init leaves under $RAY_TMPDIR/ray/session_<ts>/logs
        logs = os.path.join(os.environ["RAY_TMPDIR"], "ray", "session_2026_%s" % label, "logs")
        for rel in ("raylet.out", "gcs_server.out", "python-core-worker-1a.log", "worker-1a-01.err",
                    "old/worker-0f-01.err", "events/event_RAYLET.log", "dashboard.log"):
            os.makedirs(os.path.dirname(os.path.join(logs, rel)), exist_ok=True)
            with open(os.path.join(logs, rel), "w") as f:
                f.write("%s of %s\n" % (rel, label))
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
        if CONTROL.get("conda_intermediate"):
            # real conda run: a python process that runs the command as its child and dies at once on
            # SIGTERM, leaving the command behind in timeout's process group
            return subprocess.Popen(cmd).wait()
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
                  "memory.reserved": CONTROL.get("gpu_reserved", "%d%s" % (571, " MiB" if units else "")),
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
    """GNU coreutils timeout (src/timeout.c) stand-in: timeout -k K SECS cmd...

    Like the real one it calls setpgid(0, 0), so the command and everything it starts inherit
    timeout's own process group; on expiry (or on SIGTERM) it signals the direct child and the group
    (ignoring the signal itself) and arms a SIGKILL timer of K seconds. It waits for its DIRECT
    child only: once that has exited it returns (124 after a timeout), and the SIGKILL timer never
    fires, whatever is left in the group.
    """
    assert argv[0] == "-k", argv
    kill_after, secs, cmd = float(argv[1]), float(argv[2]), argv[3:]
    os.setpgid(0, 0)
    child = subprocess.Popen(cmd)
    state = {"timed_out": False, "kill_armed": False}

    def send(sig):
        try:
            os.kill(child.pid, sig)
        except OSError:
            pass
        if sig != signal.SIGKILL:
            signal.signal(sig, signal.SIG_IGN)
        os.killpg(0, sig)  # with SIGKILL this kills timeout too: exit status 137

    def cleanup(signum):
        if state["kill_armed"]:
            send(signal.SIGKILL)
            return
        state["kill_armed"] = True
        signal.setitimer(signal.ITIMER_REAL, kill_after)
        send(signal.SIGTERM)

    def on_alarm(_signum, _frame):
        if not state["kill_armed"]:
            state["timed_out"] = True
        cleanup(signal.SIGALRM)

    signal.signal(signal.SIGALRM, on_alarm)
    signal.signal(signal.SIGTERM, lambda *_: cleanup(signal.SIGTERM))
    signal.setitimer(signal.ITIMER_REAL, secs)
    rc = child.wait()
    if state["timed_out"]:
        return 124
    return rc if rc >= 0 else 128 - rc


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
        self.data = tmp / "data"  # DATA_DIR: verl opens both parquets at start (preflight F5)
        self.data.mkdir()
        for name in ("math_l3to5_train.parquet", "math500_test.parquet"):
            (self.data / name).write_text("")

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

    def run(self, timeout=600, **extra):
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


def _short_dir():
    """A fresh directory with a short path, as on the box (None if /tmp cannot run scripts).

    SUMMARY.txt prints the repo and checkpoint paths (the step-50 rerun), and its layout depends on
    how many lines they need; pytest's tmp_path is over 100 characters long on macOS."""
    try:
        root = Path(tempfile.mkdtemp(prefix="rag_", dir="/tmp")).resolve()
    except OSError:
        return None
    probe = root / "probe.sh"
    try:
        Sandbox._script(probe, "#!/bin/sh\nexit 0\n")
        ok = subprocess.run([str(probe)]).returncode == 0
        probe.unlink()
    except OSError:
        ok = False
    if not ok:
        shutil.rmtree(root, ignore_errors=True)
        return None
    return root


@pytest.fixture
def sb(tmp_path):
    root = _short_dir()
    try:
        yield Sandbox(root or tmp_path)
    finally:
        if root is not None:
            shutil.rmtree(root, ignore_errors=True)


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
    # the box runs the engine and gate suites, not the stub tests of the runner / setup scripts themselves
    assert "serve/rl/tests" in cpu_argv and "integration/gates/tests" in cpu_argv and "integration/tests" not in cpu_argv
    assert "--color=no" in gpu_argv and "--color=no" in cpu_argv

    # gate_3b_vllm failed without a JSON -> two-way gate, three-way reported as not done
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
    # the reference ran first and is handed to the step-25 gate
    assert argv25[argv25.index("--vllm-reference") + 1] == str(res.bundle / "gates" / "gate_c_vllm_reference_step25.json")

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
    # each smoke run's own checkpoint is removed after the run, its size recorded first
    left = sorted(p.name for p in sb.ckpt.iterdir() if p.name.startswith("gates_smoke_"))
    assert left == [], left
    smoke_log = (res.bundle / res.status("smoke_worst_case")["log"]).read_text()
    assert "size of this smoke run's own checkpoint before removing it" in smoke_log
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
    # no torch figure in the stub gate JSONs: nvidia-smi memory.total - memory.reserved, (40960 - 571) / 1024
    assert cmp_argv[cmp_argv.index("--gpu-total-gb") + 1] == "39.44"
    assert "GPU total: nvidia-smi memory.total - memory.reserved" in res.status("smoke_compare")["note"]

    # env record
    env_dir = res.bundle / "env"
    for name in ("host.txt", "git.txt", "nvidia_smi.txt", "nvidia_smi_query.csv", "nvidia_smi_apps.csv", "paths.txt",
                 "facts.txt", "verl-sparsespec/versions.txt", "verl-sparsespec/pip_freeze.txt",
                 "verl-baseline/versions.txt", "verl-baseline/conda_list.txt"):
        assert (env_dir / name).exists(), name
    assert "Mem:" in (env_dir / "host.txt").read_text()
    paths = (env_dir / "paths.txt").read_text()
    assert f"{step25}: exists" in paths and "math_l3to5_train.parquet: exists" in paths
    assert "CUDA Version" in (env_dir / "nvidia_smi.txt").read_text()

    # summary: one screen, mixed outcome, explicit underpowered sentence
    summary = res.summary()
    lines = summary.splitlines()
    assert len(lines) <= 46, summary
    too_wide = [line for line in lines if len(line) > 110 and not line.startswith("send this file back: ")]
    assert not too_wide, too_wide  # the tarball path is never clipped
    assert "FAILED ONLY because the checkpoint is too close to the base model" in summary
    assert "Step-50 rerun: paste the lines below as they are (stock vLLM first, then the gate" in flat(summary)
    assert "The step-50 checkpoint C does not exist yet" in flat(summary)
    verdict = flat(summary[summary.index("VERDICT:"):])
    assert "VERDICT: NOT PASSED" in verdict and "3c step-25 FAIL (underpowered)" in verdict
    # changed in round 3: a two-way 3b pass is not "3b PASS" (gate_3b_vllm failed without writing a JSON)
    assert "3b two-way only (three-way NOT DONE, gate_3b_vllm FAIL)" in verdict and "3b PASS" not in verdict
    assert "GPU unit tests PASS" in verdict and "3d PASS" in verdict
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
    # the stock-vLLM producer stopped without a verdict: said so, with the gate's own last message below
    row = lines.index(next(line for line in lines if line.startswith("gate_3b_vllm ")))
    assert lines[row].endswith("  no verdict (the gate stopped before writing its JSON)"), lines[row]
    assert lines[row + 1] == "    [gate 3b vllm] FAIL"
    assert res.status("gate_3b_vllm")["outcome"] == "no_verdict"
    # to stay on one screen, rows that all passed are folded (nothing is dropped)
    assert "gpu_unit_tests + 3 variants       PASS" in summary and "in stage order: 9/9/9/9 passed, " in summary
    assert "smoke runs, 3d check PASS         PASS" in summary and "  worst_case/no_graph/no_logprobs: " in summary
    assert "PASS, nothing to add: env " in summary
    # the step-50 rerun lines paste as they are and run outside the old bundle; the same lines are
    # in the stage log and on the console
    block = assert_rerun_pastes(sb, summary, "summary", res.bundle)
    assert len(block) == 8, block  # cd, C=, D=, two lines + three lines of commands
    assert "\n".join(block) in (res.bundle / res.status("gate_3c_step25")["log"]).read_text()
    assert "\n".join(block) in res.out
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


FAKE_CONDA = r"""import json, os, sys
with open(os.environ["RERUN_CALLS"], "a") as f:
    f.write(json.dumps({"argv": sys.argv[1:], "cwd": os.getcwd()}) + "\n")
"""


def rerun_lines(text):
    """The shell lines of a printed step-50 rerun: from 'cd <repo>' to the end of the second command."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("cd "))
    end = next(i for i in range(start, len(lines)) if lines[i].startswith(("VERDICT:", "[run_all_gates]")))
    return lines[start:end]


def assert_rerun_pastes(sb, text, tag, bundle=None):
    """Paste the printed step-50 rerun lines into bash (conda recorded, not run): the stock-vLLM
    reference on step 50 first, then the reload gate with that reference; in the repo, with this
    run's DATA_DIR, no --out-dir, nothing pointing into the old bundle. Returns the lines."""
    lines = rerun_lines(text)
    script = "\n".join(lines) + "\n"
    fake = sb.tmp / ("rerun_" + tag)
    fake.mkdir()
    (fake / "conda_impl.py").write_text(FAKE_CONDA)
    Sandbox._script(fake / "conda", f"#!/bin/sh\nexec '{sys.executable}' '{fake / 'conda_impl.py'}' \"$@\"\n")
    calls = fake / "calls.jsonl"
    proc = subprocess.run(["bash", "-c", script], env={"PATH": f"{fake}:/usr/bin:/bin", "RERUN_CALLS": str(calls)},
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=sb.tmp)
    assert proc.returncode == 0 and proc.stdout == "", (proc.stdout, script)
    got = [json.loads(line) for line in calls.read_text().splitlines()]
    ckpt50 = str(sb.ckpt / "vllm_baseline_seed0" / "hf" / "global_step_50")
    data = "DATA_DIR=" + str(sb.tmp / "data")  # the sandbox does not use the default DATA_DIR
    assert [c["cwd"] for c in got] == [os.path.realpath(sb.repo)] * 2, got
    assert got[0]["argv"] == ["run", "--no-capture-output", "-n", "verl-baseline", "env", "PYTHONPATH=.", data, "python",
                              "integration/gates/gate_c_vllm_reference.py", "--ckpt", ckpt50, "--tag", "step50"], got
    assert got[1]["argv"] == ["run", "--no-capture-output", "-n", "verl-sparsespec", "env", "PYTHONPATH=.", data, "python",
                              "-m", "integration.gates.gate_c_reload", "--engine", "sparsespec", "--ckpt", ckpt50, "--tag",
                              "sparsespec_step50", "--vllm-reference",
                              "baseline/results/sparsespec_checks/gate_c_vllm_reference_step50.json"], got
    assert "--out-dir" not in script and "MODEL_PATH" not in script  # default out-dir; the default model
    if bundle is not None:
        assert str(bundle) not in script
    return lines


def test_all_pass_without_step25_checkpoint(sb):
    sb.control["gpu_reserved"] = "[N/A]"  # old driver: no memory.reserved -> smoke_compare's own default
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
    assert "--gpu-total-gb" not in res.call("smoke_compare")["argv"]
    assert "GPU total: smoke_compare's default" in res.status("smoke_compare")["note"]
    summary = flat(res.summary())
    assert "VERDICT: ALL REQUIRED GATES PASSED" in summary and "; 3b PASS;" in summary
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


def test_missing_baseline_env_is_a_skip_and_3b_is_not_passed(sb):
    res = sb.run(STUB_ENVS="verl-sparsespec", STAGES="cpu_tests,gate_3b_vllm,engine_import,gate_3b")
    assert res.rc == 1, res.explain()
    assert_no_runner_errors(res)
    assert "conda env(s) missing: verl-baseline" in res.out
    st = res.status("gate_3b_vllm")
    assert st["status"] == "SKIP" and "conda env 'verl-baseline' does not exist" in st["reason"]
    assert [c["key"] for c in res.stage_calls()] == ["cpu_tests", "engine_import", "gate_3b"]
    assert "--vllm-json" not in res.call("gate_3b")["argv"]
    assert res.status("env")["status"] == "FAIL"  # one env could not be recorded
    verdict = flat(res.summary()[res.summary().index("VERDICT:"):])
    assert "3b two-way only (three-way NOT DONE, gate_3b_vllm SKIP)" in verdict


def assert_refused(res, *fixes):
    """exit 2 after the env stage, nothing GPU-related ran, REFUSED.txt / SUMMARY.txt / tarball written"""
    assert res.rc == 2, res.explain()
    assert_no_runner_errors(res)
    assert "REFUSING TO START" in res.out
    assert [c["key"] for c in res.stage_calls()] == []
    assert {c["key"] for c in res.calls} <= {"probe", "pip", "pytest_check"}
    refused = (res.bundle / "REFUSED.txt").read_text()
    for fix in fixes:
        assert fix in refused, refused
        assert fix in res.out
        assert fix in flat(res.summary()), res.summary()
    assert "REFUSED TO START" in res.summary() and "VERDICT: NOT PASSED" in res.summary()
    assert res.tarball().exists()
    with tarfile.open(res.tarball()) as tar:
        assert f"{res.bundle.name}/REFUSED.txt" in tar.getnames()
    assert res.status("gate_3b")["status"] == "SKIP" and "refused to start" in res.status("gate_3b")["reason"]
    assert len(res.summary().splitlines()) <= 46, res.summary()
    return refused


def test_refuses_when_sparsespec_env_is_missing(sb):  # F1
    res = sb.run(STUB_ENVS="verl-baseline")
    refused = assert_refused(res, "Fix: bash integration/build_sparsespec_env.sh")
    assert "conda env 'verl-sparsespec' (SPARSESPEC_ENV) does not exist" in refused
    # a selection that needs no SparseSpec stage still runs
    sb2 = Sandbox(sb.tmp / "second")
    res2 = sb2.run(STUB_ENVS="verl-baseline", STAGES="gate_3b_vllm")
    assert [c["key"] for c in res2.stage_calls()] == ["gate_3b_vllm"], res2.explain()


def test_refuses_when_pytest_is_missing(sb):  # F2: the pytest stages used to be SKIPped silently
    sb.control = {"pytest_missing": True}
    res = sb.run()
    refused = assert_refused(res, "conda run -n verl-sparsespec python -m pip install --no-deps pytest iniconfig pluggy",
                             "bash integration/build_sparsespec_env.sh (its pytest step)")
    assert "pytest is not importable in conda env 'verl-sparsespec'" in refused
    # without a pytest stage selected, no refusal
    sb2 = Sandbox(sb.tmp / "second")
    sb2.control = {"pytest_missing": True}
    res2 = sb2.run(STAGES="engine_import")
    assert res2.rc == 1 and [c["key"] for c in res2.stage_calls()] == ["engine_import"], res2.explain()


def test_refuses_when_install_prefix_has_no_librmm(sb):  # F3
    res = sb.run(INSTALL_PREFIX="/nonexistent/raft-build", PREFIX="/also/not/used")
    refused = assert_refused(res, "Fix: unset INSTALL_PREFIX PREFIX")
    assert "INSTALL_PREFIX=/nonexistent/raft-build is set" in refused and "librmm.so does not exist" in refused
    host = (res.bundle / "env" / "host.txt").read_text()
    assert "INSTALL_PREFIX=/nonexistent/raft-build" in host and "PREFIX=/also/not/used" in host
    for name in ("MAX_JOBS", "XDG_CACHE_HOME", "FLASHINFER_WORKSPACE_BASE", "FLASHINFER_EXTRA_LDFLAGS", "CUDA_HOME",
                 "CUDA_PATH", "LD_LIBRARY_PATH"):
        assert f"  {name}=" in host, name
    # only the SparseSpec engine reads them: a stock-vLLM-only rerun is not refused (a 0 timeout also runs)
    sb2 = Sandbox(sb.tmp / "second")
    res2 = sb2.run(INSTALL_PREFIX="/nonexistent/raft-build", STAGES="gate_3b_vllm,gate_3c_vllm_reference",
                   TIMEOUT_GATE_3B_VLLM=0)
    assert res2.rc == 1 and "REFUSING TO START" not in res2.out, res2.explain()
    assert [c["key"] for c in res2.stage_calls()] == ["gate_3b_vllm"], res2.explain()
    assert res2.status("gate_3b_vllm")["status"] == "PASS", res2.status("gate_3b_vllm")
    log = (res2.bundle / res2.status("gate_3b_vllm")["log"]).read_text()
    assert "timeout -k 2 0 " in log, log  # still under the timeout command (its own process group), no timer


def test_refuses_when_a_data_parquet_is_missing(sb):  # F5: verl opens both at start, hours into the run
    (sb.data / "math500_test.parquet").unlink()
    refused = assert_refused(sb.run(), "Fix: export DATA_DIR=")
    assert f"{sb.data}/math500_test.parquet does not exist (DATA_DIR={sb.data})" in refused, refused
    assert "math_l3to5_train.parquet" not in refused and "baseline/prepare_data.py --out-dir" in refused
    # no verl stage selected: the parquets are not needed (the 3b / 3c gates rebuild the rows from HF)
    sb2 = Sandbox(sb.tmp / "second")
    shutil.rmtree(sb2.data)
    res2 = sb2.run(STAGES="gate_3b_vllm,gate_3b")
    assert [c["key"] for c in res2.stage_calls()] == ["gate_3b_vllm", "gate_3b"], res2.explain()


def test_relative_ckpt_root_and_data_dir_are_resolved_where_typed(sb):  # used to lose every smoke run
    (sb.tmp / "elsewhere" / "data").symlink_to(sb.data)
    res = sb.run(STAGES="smoke_worst_case", CKPT_ROOT="ckpts_rel", DATA_DIR="data")
    assert_no_runner_errors(res)
    assert res.status("smoke_worst_case")["status"] == "PASS", res.explain()
    seen = res.call("smoke_worst_case")["vars"]
    where = os.path.realpath(sb.tmp / "elsewhere")
    assert seen["CKPT_ROOT"] == os.path.join(where, "ckpts_rel"), seen
    assert seen["DATA_DIR"] == os.path.join(where, "data"), seen
    assert not (sb.repo / "ckpts_rel").exists()


def test_prefix_with_librmm_is_used_not_unset(sb):  # F3, the accepted case
    prefix = sb.tmp / "raft"
    (prefix / "lib").mkdir(parents=True)
    (prefix / "lib" / "librmm.so").write_text("")
    res = sb.run(PREFIX=str(prefix), STAGES="cpu_tests")
    assert res.rc == 1, res.explain()  # required gates not run, but it started
    assert_no_runner_errors(res)
    assert res.call("cpu_tests")["vars"]["PREFIX"] == str(prefix)  # passed through unchanged
    assert f"PREFIX={prefix}" in (res.bundle / "env" / "host.txt").read_text()
    assert "NOTE: PREFIX=" in res.summary() and "has lib/librmm.so" in res.summary()
    assert "WARNING: PREFIX=" in res.out


def test_refuses_when_the_verl_patch_is_not_applied(sb):  # F4, together with F2: all problems at once
    sb.control = {"patch_state": "absent", "pytest_missing": True}
    res = sb.run()
    refused = assert_refused(res, "Fix: bash integration/setup_env.sh verl-sparsespec", "pip install --no-deps pytest")
    assert len(refused.strip().splitlines()) == 2 and "PATCH absent" in refused
    # no verl stage selected: the patch is not needed
    sb2 = Sandbox(sb.tmp / "second")
    sb2.control = {"patch_state": "absent"}
    res2 = sb2.run(STAGES="cpu_tests,engine_import")
    assert [c["key"] for c in res2.stage_calls()] == ["cpu_tests", "engine_import"], res2.explain()
    # verl could not even be imported by the probe: also refused, with what the probe said
    sb3 = Sandbox(sb.tmp / "third")
    sb3.control = {"patch_state": "unknown"}
    refused = assert_refused(sb3.run(STAGES="reference_plumbing"), "Fix: bash integration/setup_env.sh verl-sparsespec")
    assert "verl could not be inspected" in refused and "PATCH unknown fsdp_workers_hunk=False" in refused


def test_refuses_when_the_jit_has_no_cuda_compiler(sb):  # F6
    sb.control = {"nvcc_state": "missing"}
    res = sb.run()
    refused = assert_refused(res, "export CUDA_HOME=...", "bash integration/build_sparsespec_env.sh --verify")
    assert "no CUDA compiler for the engine's JIT in conda env 'verl-sparsespec'" in refused
    assert "NVCC missing nvcc=/usr/local/cuda/bin/nvcc torch_CUDA_HOME=None" in refused
    # no stage that builds the real engine is selected: not needed, not refused
    sb2 = Sandbox(sb.tmp / "second")
    sb2.control = {"nvcc_state": "missing"}
    res2 = sb2.run(STAGES="cpu_tests,engine_import,gate_3b_vllm")
    assert "REFUSING TO START" not in res2.out, res2.explain()
    assert [c["key"] for c in res2.stage_calls()] == ["cpu_tests", "engine_import", "gate_3b_vllm"], res2.explain()
    # the probe could not tell (or printed nothing): a warning, never a refusal
    for n, state in enumerate(("unknown", "none")):
        sb3 = Sandbox(sb.tmp / ("third%d" % n))
        sb3.control = {"nvcc_state": state}
        res3 = sb3.run(STAGES="gate_3b_vllm,gate_3b")
        assert "REFUSING TO START" not in res3.out, res3.explain()
        assert "could not tell whether the engine's JIT finds a CUDA compiler" in res3.out, res3.explain()
        assert [c["key"] for c in res3.stage_calls()] == ["gate_3b_vllm", "gate_3b"], res3.explain()


def test_low_host_ram_is_a_warning_not_a_refusal(sb):  # G1
    sb.meminfo.write_text("MemTotal:       33554432 kB\nMemFree:        10000000 kB\nMemAvailable:   20971520 kB\n")
    res = sb.run(STAGES="cpu_tests")
    assert res.rc == 1, res.explain()
    assert [c["key"] for c in res.stage_calls()] == ["cpu_tests"]
    assert "WARNING: only 20.0 GiB of host RAM available" in res.out
    assert "WARNING: only 20 GiB host RAM available at start (< 24 GiB)" in res.summary()


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
    assert _gone(hung)


def test_timed_out_stage_behind_conda_with_a_sigterm_handler_is_stopped(sb):
    # conda run (a short-lived intermediate) dies at timeout's SIGTERM, so GNU timeout returns 124 at
    # once and its -k SIGKILL never fires; the workload ignores SIGTERM and holds the GPU
    sb.control = {"hang": ["gpu_unit_tests"], "trap_term": ["gpu_unit_tests"], "conda_intermediate": True,
                  "gpu_free_after": {"gpu_unit_tests": 9000}}
    res = sb.run(STAGES="gpu_unit_tests,gpu_unit_tests_host_guard", TIMEOUT_GPU_UNIT_TESTS=3, KILL_AFTER_S=2)
    assert_no_runner_errors(res)
    st = res.status("gpu_unit_tests")
    assert st["status"] == "FAIL" and st["outcome"] == "timeout", st
    assert "timed out after 3s" in st["reason"] and "leftover process(es) of its own process group" in st["reason"], st
    assert ("GPU memory still in use after the timed-out stage; processes are listed in the log; "
            "nothing outside this runner's process group was signalled") in st["reason"]
    hung = res.call("gpu_unit_tests")["pid"]
    assert _gone(hung)
    assert "(pids %d) were still running after the command returned; killed with SIGKILL after 2s" % hung in st["reason"]
    log = (res.bundle / st["log"]).read_text()
    assert "SIGTERM to process group" in log and "SIGKILL to process group" in log
    assert "this stage's own process group" in log and str(hung) in log
    assert "424242" in log  # the teammate's process is listed, never signalled
    # the GPU stays busy (stub): the next stage waits, then is SKIPped as usual
    assert res.status("gpu_unit_tests_host_guard")["skip_kind"] == "gpu_busy"


def test_gpu_busy_then_free_is_a_clean_pass(sb):  # runner:4
    sb.control = {"gpu_free_after": {"gate_3b_vllm": 9000}}

    def free_after_first_busy_poll():
        deadline = time.time() + 120
        while time.time() < deadline:
            if any("waiting up to" in p.read_text() for p in sb.bundles.glob("gates_*/run_all_gates_*.log")):
                time.sleep(0.2)
                sb.set_gpu_free(40000)
                return
            time.sleep(0.05)

    threading.Thread(target=free_after_first_busy_poll, daemon=True).start()
    res = sb.run(STAGES="gate_3b_vllm,gate_3b", GPU_WAIT_S=40, GPU_POLL_S=3)
    assert_no_runner_errors(res)
    st = res.status("gate_3b")
    assert st["status"] == "PASS" and st["reason"] == "", st
    assert not any(l.startswith("[run_all_gates] gate_3b: PASS") and "GPU busy" in l for l in res.out.splitlines())
    assert "GPU memory free again after" in res.out
    assert "waited" in (res.bundle / st["log"]).read_text()


def _interrupt(sb, proc, key, sig, group=True):
    """Signal the started runner (or its whole process group, like Ctrl-C) once stage ``key`` runs."""
    try:
        _wait_for_call(sb, key, proc)
        time.sleep(0.5)
        if group:
            os.killpg(proc.pid, sig)
        else:
            os.kill(proc.pid, sig)
        out, _ = proc.communicate(timeout=120)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    return out


def _gone(pid):
    """True once a process no longer runs (a zombie that its parent has not collected yet counts as gone)."""
    stat = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], stdout=subprocess.PIPE, text=True).stdout.strip()
    return stat == "" or stat.startswith("Z")


def _wait_for_call(sb, key, proc, limit=60):
    deadline = time.time() + limit
    while time.time() < deadline:
        if any(c["key"] == key for c in sb.read_calls()):
            return
        if proc.poll() is not None:
            raise AssertionError("runner exited early:\n" + proc.stdout.read())
        time.sleep(0.1)
    raise AssertionError(f"{key} never started")


@pytest.mark.parametrize("how", ["ctrl_c_to_process_group", "sigterm_no_timeout_cmd", "ctrl_c_conda_and_trap",
                                 "ctrl_c_no_stage_timeout"])
def test_interrupted_run_still_writes_summary_and_tarball(sb, how):
    sb.control = {"hang": ["gate_3b"]}
    extra = {"STAGES": "gate_3b_vllm,gate_3b,gate_3c_synthetic"}
    if how == "sigterm_no_timeout_cmd":
        extra["TIMEOUT_BIN"] = "none"
    if how == "ctrl_c_no_stage_timeout":  # TIMEOUT_GATE_3B=0: no timer, but still its own process group
        extra["TIMEOUT_GATE_3B"] = "0"
    if how in ("ctrl_c_conda_and_trap", "ctrl_c_no_stage_timeout"):
        # the stage's workload ignores SIGTERM, behind a conda that does not
        sb.control.update({"trap_term": ["gate_3b"], "conda_intermediate": True})
    proc = sb.start(**extra)
    if how in ("ctrl_c_to_process_group", "ctrl_c_conda_and_trap"):
        out, expected_rc = _interrupt(sb, proc, "gate_3b", signal.SIGINT), 130
    else:
        out, expected_rc = _interrupt(sb, proc, "gate_3b", signal.SIGTERM, group=False), 143
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
    while not _gone(hung):
        assert time.time() < deadline, "the runner's own hung stage process was not stopped"
        time.sleep(0.2)
    main_logs = list(res.bundle.glob("run_all_gates_*.log"))
    assert main_logs and "received SIG" in main_logs[0].read_text()
    assert ("TIMEOUT_BIN=none" in st["reason"]) == (how == "sigterm_no_timeout_cmd"), st
    if how in ("ctrl_c_conda_and_trap", "ctrl_c_no_stage_timeout"):
        assert "leftover process(es) of its own process group" in st["reason"], st
        assert "killed with SIGKILL after 2s" in st["reason"], st
        log = (res.bundle / st["log"]).read_text()
        assert "SIGTERM to process group" in log and "SIGKILL to process group" in log


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
                         stderr=subprocess.STDOUT, text=True, timeout=300,
                         env={**os.environ, "HF_TOKEN": "secret", "INSTALL_PREFIX": "/x/raft", "PREFIX": "/y",
                              "MAX_JOBS": "4", "XDG_CACHE_HOME": "/c", "PREFIX_OTHER": "no", "FLASHINFER_WORKSPACE_BASE": "/w"})
    assert out.returncode == 0, out.stdout
    assert "VERSION python " in out.stdout and "VERSION vllm " in out.stdout and "PATCH " in out.stdout
    assert "secret" not in out.stdout
    for line in ("ENVVAR INSTALL_PREFIX=/x/raft", "ENVVAR PREFIX=/y", "ENVVAR MAX_JOBS=4", "ENVVAR XDG_CACHE_HOME=/c",
                 "ENVVAR FLASHINFER_WORKSPACE_BASE=/w"):
        assert line in out.stdout.splitlines(), line
    assert "PREFIX_OTHER" not in out.stdout
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


# ---------------------------------------------------------------------------
# round 3: gate 3b rule, gate outcomes, Ray logs, colors, GPU total, step-50 rerun, one-screen cap
# ---------------------------------------------------------------------------
def verdict_of(res):
    summary = res.summary()
    return flat(summary[summary.index("VERDICT:"):])


def test_two_way_fallback_is_not_a_3b_pass(sb):  # gate-3b:1 / runner:1
    # the stock-vLLM producer stops without its JSON (exit 1), crashes (exit 3), or is not run at all:
    # the two-way gate still runs and passes, but gate 3b is NOT passed and the run exits 1
    for n, (control, extra) in enumerate((({"exit": {"gate_3b_vllm": 1}}, {}), ({"exit": {"gate_3b_vllm": 3}}, {}),
                                          ({}, {"SKIP_STAGES": "gate_3b_vllm"}))):
        box = Sandbox(sb.tmp / ("case%d" % n))
        box.control = control
        res = box.run(STAGES="engine_import,gate_3b_vllm,gate_3b", **extra)
        assert res.rc == 1, res.explain()
        assert_no_runner_errors(res)
        assert res.status("gate_3b")["status"] == "PASS" and "--vllm-json" not in res.call("gate_3b")["argv"]
        vst = res.status("gate_3b_vllm")["status"]
        assert "3b two-way only (three-way NOT DONE, gate_3b_vllm %s)" % vst in verdict_of(res), verdict_of(res)
        assert "3b PASS" not in verdict_of(res)
        assert (res.bundle / "verdict.txt").read_text().strip() == "FAIL"


def test_vllm_json_decides_three_way_not_the_exit_status(sb):  # runner:1
    # exit 1 with a complete token file (a step was not greedy): three-way, which then decides 3b
    sb.control = {"exit": {"gate_3b_vllm": 1}, "vllm_json_on_fail": True}
    res = sb.run(STAGES="engine_import,gate_3b_vllm,gate_3b")
    assert_no_runner_errors(res)
    argv = res.call("gate_3b")["argv"]
    assert argv[argv.index("--vllm-json") + 1] == str(res.bundle / "gates" / "gate_b_vllm_greedy.json")
    assert argv[argv.index("--tag") + 1] == "three_way"
    assert res.status("gate_3b_vllm")["outcome"] == "verdict"
    assert res.status("gate_3b")["note"] == "three-way (vLLM JSON from gate_3b_vllm FAIL, exit 1)"
    assert "; 3b PASS;" in verdict_of(res)
    # the token file is written, then vLLM hangs at shutdown until the stage timeout: still three-way
    box = Sandbox(sb.tmp / "hang")
    box.control = {"vllm_hang_after_json": True}
    res = box.run(STAGES="engine_import,gate_3b_vllm,gate_3b", TIMEOUT_GATE_3B_VLLM=2)
    assert_no_runner_errors(res)
    assert res.status("gate_3b_vllm")["outcome"] == "timeout"
    assert "--vllm-json" in res.call("gate_3b")["argv"]
    assert res.status("gate_3b")["note"] == "three-way (vLLM JSON from gate_3b_vllm FAIL, exit 124)"
    assert "; 3b PASS;" in verdict_of(res)


def test_three_way_fail_names_the_failed_check(sb):  # gate-3b:2
    sb.control = {"exit": {"gate_3b": 1}, "r2_fail": True}
    res = sb.run(STAGES="engine_import,gate_3b_vllm,gate_3b")
    assert res.status("gate_3b")["outcome"] == "verdict"
    summary = res.summary()
    assert summary.count("PASS  diverged") == 3  # every pair line still says PASS
    assert "    3b failed: R2_profile_significance [engine]" in summary
    assert "3b R2 significance: diverging prompts sparsespec 5 / vLLM 0, earliest 40 / none, p 0.0166 < 0.05" in summary
    assert "3b FAIL" in verdict_of(res)


def test_gate_3b_refusal_is_shown_as_refused(sb):  # gate-3b:3 / runner:8
    sb.control = {"refuse": ["gate_3b"]}
    res = sb.run(STAGES="engine_import,gate_3b_vllm,gate_3b")
    assert res.rc == 1, res.explain()
    assert_no_runner_errors(res)
    st = res.status("gate_3b")
    assert st["status"] == "FAIL" and st["exit_status"] == "2" and st["outcome"] == "refused", st
    assert st["reason"].startswith("REFUSED (exit 2, nothing was compared): the vLLM file"), st
    assert "gate_3b: FAIL" in res.out and "REFUSED (exit 2, nothing was compared)" in res.out
    summary = flat(res.summary())
    assert "REFUSED (inputs do not match; nothing was compared) | exit 2" in summary, summary
    assert "REFUSED: the vLLM file gates/gate_b_vllm_greedy.json prompts are not the prompts of this run" in summary
    assert "prompt 3 differs at token 4" in summary and str(res.bundle / "gates") not in summary
    assert "(crash" not in summary and "gate JSON n/a" not in summary
    assert "3b REFUSED" in verdict_of(res) and "VERDICT: NOT PASSED" in summary
    # the gate's JSON alone is enough for the summary (a status file without "outcome")
    path = next((res.bundle / "status").glob("*_gate_3b.status"))
    path.write_text("".join(line for line in path.read_text().splitlines(True) if not line.startswith("outcome=")))
    proc = subprocess.run([sys.executable, str(SUMMARIZER), str(res.bundle), "--out", str(sb.tmp / "again.txt")],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert "REFUSED (inputs do not match; nothing was compared)" in proc.stdout and "3b REFUSED" in flat(proc.stdout)


def test_gate_without_verdict_and_crashed_gate(sb):  # runner:8 / gate-3c-smoke-gputests:4
    sb.make_step25()
    sb.control = {"exit": {"gate_3c_synthetic": 1, "gate_3c_step25": 3}, "no_json": ["gate_3c_synthetic"]}
    res = sb.run(STAGES="engine_import,gate_3c_synthetic,gate_3c_step25")
    assert res.rc == 1, res.explain()
    assert_no_runner_errors(res)
    st = res.status("gate_3c_synthetic")
    assert st["status"] == "FAIL" and st["outcome"] == "no_verdict" and st["exit_status"] == "1", st
    assert st["reason"] == "no verdict (the gate stopped before writing its JSON; see the log)"
    st = res.status("gate_3c_step25")
    assert st["status"] == "FAIL" and st["outcome"] == "crashed" and st["exit_status"] == "3", st
    assert st["reason"] == "crashed (exit 3: unexpected exception, traceback in the log; no verdict)"
    lines = res.summary().splitlines()
    row = next(i for i, line in enumerate(lines) if line.startswith("gate_3c_synthetic"))
    assert lines[row].endswith("no verdict (the gate stopped before writing its JSON)"), lines[row]
    assert lines[row + 1].startswith("    [gate 3c] FAIL checkpoint not found"), lines[row + 1]  # the gate's own message
    row = next(i for i, line in enumerate(lines) if line.startswith("gate_3c_step25"))
    assert lines[row].endswith("crashed (exit 3: unexpected exception; no verdict)"), lines[row]
    assert lines[row + 1] == "    RuntimeError: CUDA error: an illegal memory access", lines[row + 1]  # the exception line
    verdict = verdict_of(res)
    assert "3c synthetic no verdict" in verdict and "3c step-25 crashed" in verdict and "VERDICT: NOT PASSED" in verdict
    assert all(len(line) <= 110 for line in lines[:-1])


def test_inconclusive_step25_prints_the_rerun_pair(sb):  # runner:9 without a step-25 vLLM reference
    sb.make_step25()
    step50 = sb.ckpt / "vllm_baseline_seed0" / "hf" / "global_step_50"
    step50.mkdir(parents=True)
    sb.control = {"exit": {"gate_3c_step25": 1}, "inconclusive": ["gate_3c_step25"]}
    res = sb.run(STAGES="engine_import,gate_3c_step25")
    assert_no_runner_errors(res)
    st = res.status("gate_3c_step25")
    assert "--vllm-reference" not in res.call("gate_3c_step25")["argv"]
    assert st["note"].startswith("NO stock-vLLM reference (gate_3c_vllm_reference SKIP)"), st
    assert "negative_control_inconclusive" in st["reason"] and "rerun on the step-50 checkpoint" in st["reason"]
    summary = res.summary()
    assert "(negative_control_inconclusive)" in flat(summary) and "does not exist yet" not in flat(summary)
    assert "3c step-25 FAIL (inconclusive)" in verdict_of(res)
    block = assert_rerun_pastes(sb, summary, "summary", res.bundle)
    assert "\n".join(block) in res.out and "the step-50 rerun, to paste as it is" in res.out


def test_step50_rerun_text_without_python(sb):  # runner:9: the bash fallback gives the same pair
    proc = _source(sb, "rag_configure >/dev/null || exit 9; RAG_PY=''; RAG_CONDA_CMD=conda; BUNDLE=/nonexistent\n"
                       "RAG_RERUN_DATA_DIR=$DATA_DIR; RAG_RERUN_MODEL_PATH=''; rag_step50_rerun_text; echo '[run_all_gates] end'")
    assert proc.returncode == 0, proc.stdout
    assert "Step-50 rerun: paste the lines below as they are" in proc.stdout
    assert "The step-50 checkpoint C does not exist yet" in proc.stdout
    assert_rerun_pastes(sb, proc.stdout, "bash")


def test_ray_session_logs_of_a_failed_verl_stage_are_bundled(sb):  # runner:3
    ray_tmp = sb.tmp / "raytmp"
    (ray_tmp / "ray" / "session_2020_old" / "logs").mkdir(parents=True)
    (ray_tmp / "ray" / "session_2020_old" / "logs" / "raylet.out").write_text("old session\n")
    old = time.time() - 3600
    os.utime(ray_tmp / "ray" / "session_2020_old" / "logs" / "raylet.out", (old, old))
    sb.control = {"exit": {"smoke_worst_case": 1}}
    res = sb.run(STAGES="reference_plumbing,smoke_worst_case,smoke_host_guard", RAY_TMPDIR=ray_tmp)
    assert_no_runner_errors(res)
    got = sorted(str(p.relative_to(res.bundle)) for p in (res.bundle / "ray_logs").rglob("*") if p.is_file())
    base = "ray_logs/smoke_worst_case/session_2026_worst_case/logs/"
    assert got == sorted(base + r for r in ("gcs_server.out", "old/worker-0f-01.err", "python-core-worker-1a.log",
                                            "raylet.out", "worker-1a-01.err")), got
    log_fail = (res.bundle / res.status("smoke_worst_case")["log"]).read_text()
    assert "Ray session(s) of this stage: " + str(ray_tmp / "ray" / "session_2026_worst_case") in log_fail
    assert "5 log file(s) (last 2 MiB of each) copied to ray_logs/smoke_worst_case/" in log_fail
    log_pass = (res.bundle / res.status("smoke_host_guard")["log"]).read_text()
    assert "(not copied, the stage passed): " + str(ray_tmp / "ray" / "session_2026_host_guard") in log_pass
    assert "session_2020_old" not in log_fail + log_pass
    # the stub smoke_reference.sh starts no Ray session
    assert "no session under" in (res.bundle / res.status("reference_plumbing")["log"]).read_text()
    with tarfile.open(res.tarball()) as tar:
        names = tar.getnames()
    assert f"{res.bundle.name}/{base}raylet.out" in names
    assert not [n for n in names if n.endswith(".ray_marker")]


def test_colored_pytest_output_does_not_fail_a_passing_stage(sb):  # runner:5
    res = sb.run(STAGES="cpu_tests,gpu_unit_tests", FORCE_COLOR=1, PY_COLORS=1)
    for stage in ("cpu_tests", "gpu_unit_tests"):
        assert res.status(stage)["status"] == "PASS", res.status(stage)
        assert "\x1b[" not in (res.bundle / res.status(stage)["log"]).read_text()  # --color=no reached the stub
    sys.path.insert(0, str(REPO / "integration"))
    try:
        import summarize_bundle as sbm
    finally:
        sys.path.pop(0)
    # a log that is colored all the same (not produced by this runner) is still parsed
    colored = "\x1b[32m=== \x1b[32m\x1b[1m1 passed\x1b[0m, \x1b[33m1 skipped\x1b[0m\x1b[32m in 0.01s\x1b[0m ===\n"
    assert sbm.pytest_counts(colored) == {"passed": 1, "skipped": 1}
    assert sbm.pytest_text("\x1b[31mFAILED\x1b[0m t.py::x\n" + colored).startswith("1 passed, 0 failed")
    assert sbm.last_error_line("\x1b[1m\x1b[31mE   RuntimeError: boom\x1b[0m\nRuntimeError: boom\x1b[0m") == "RuntimeError: boom"


def test_gpu_total_from_the_gate_json(sb):  # runner:6 / gate-3c-smoke-gputests:3, source (1)
    res = sb.run(STAGES="engine_import,gate_3c_synthetic,smoke_worst_case,smoke_compare", STUB_TORCH_GB="39.58590316772461")
    assert_no_runner_errors(res)
    argv = res.call("smoke_compare")["argv"]
    assert argv[argv.index("--gpu-total-gb") + 1] == "39.58590316772461"
    assert "torch total_memory from gate_c_reload_sparsespec_synthetic.json" in res.status("smoke_compare")["note"]


def test_low_disk_note_and_smoke_checkpoint_removal(sb):  # runner:10
    Sandbox._script(sb.bin / "df", "#!/bin/sh\necho 'Filesystem 1024-blocks Used Available Capacity Mounted on'\n"
                                   "echo '/dev/x 104857600 99614720 5242880 95% /'\n")
    res = sb.run(STAGES="smoke_worst_case")
    assert_no_runner_errors(res)
    st = res.status("smoke_worst_case")
    assert st["status"] == "PASS", st
    assert "LOW DISK: 5 GiB free under CKPT_ROOT; verl's last-step checkpoint needs about 7 GiB" in st["note"], st
    assert not (sb.ckpt / "gates_smoke_worst_case").exists()
    proc = _source(sb, f"rag_free_gib '{sb.tmp}/not/there/yet'; echo; echo rc=$?", PATH="/usr/bin:/bin")
    assert proc.stdout.splitlines()[0].isdigit() and "rc=0" in proc.stdout, proc.stdout  # the real df


def test_interrupt_without_timeout_command_says_what_may_be_left(sb):  # TIMEOUT_BIN=none: no group of its own
    sb.control = {"hang": ["gate_3b"], "trap_term": ["gate_3b"], "conda_intermediate": True}
    proc = sb.start(STAGES="gate_3b_vllm,gate_3b", TIMEOUT_BIN="none")
    try:
        out = _interrupt(sb, proc, "gate_3b", signal.SIGINT)
        assert proc.returncode == 130, out
        res = Result(sb, proc.returncode, out)
        st = res.status("gate_3b")
        assert "interrupted by SIGINT while running; TIMEOUT_BIN=none: the stage had no process group of its own, " \
               "so processes it started may still be running (nvidia-smi in the log)" in st["reason"], st
        assert "nvidia-smi at" in (res.bundle / st["log"]).read_text()
    finally:  # the orphaned stub workload is in the test's own session
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass


def test_reaper_signals_only_the_group_of_its_own_exited_child(sb):  # runner:2, the guards
    # a process group led by a LIVE process that the runner did not start: what a recycled pid would look like
    leader = subprocess.Popen([sys.executable, "-c", "import os, time; os.setpgid(0, 0); print('up', flush=True); "
                                                     "time.sleep(120)"], stdout=subprocess.PIPE, text=True)
    log = sb.tmp / "reap.log"
    setup = ("KILL_AFTER_S=1; STAGE_NAME=x; RAG_UNKILLED=''; RAG_CHILD=''; "
             "RAG_OWN_PGID=$(ps -o pgid= -p $$ | tr -d ' '); ")
    try:
        assert leader.stdout.readline().strip() == "up"
        proc = _source(sb, setup + f"RAG_STAGE_PGID={leader.pid}; rag_reap_stage_group '{log}'; echo \"note=[$RAG_REAP_NOTE]\"")
        assert "note=[]" in proc.stdout, proc.stdout
        assert "not this runner's group, nothing is signalled" in log.read_text()
        # the runner's own group, init, and anything that is not a process group id: never touched
        for group in ("$RAG_OWN_PGID", "1", "0", "''", "abc", "-5"):
            proc = _source(sb, setup + f"RAG_STAGE_PGID={group}; rag_reap_stage_group '{log}'; "
                                       "echo \"note=[$RAG_REAP_NOTE] rc=$? left=[$RAG_STAGE_PGID]\"")
            assert "note=[] rc=0 left=[]" in proc.stdout, (group, proc.stdout)
        # the probe records a group only when the runner's own child leads it: a plain background
        # child is in the runner's own group, and another process's group is never taken over
        proc = _source(sb, setup + "sleep 5 & RAG_CHILD=$!; rag_probe_stage_pgid; echo \"probe=[$RAG_STAGE_PGID]\"; "
                                   f"kill $RAG_CHILD; RAG_CHILD={leader.pid}; RAG_OWN_PGID={leader.pid}; "
                                   "rag_probe_stage_pgid; echo \"own=[$RAG_STAGE_PGID]\"")
        assert "probe=[]" in proc.stdout and "own=[]" in proc.stdout, proc.stdout
        # bash reaps an exited child at once (even during a sleep) while RAG_CHILD still names it: a set
        # RAG_CHILD is ours only while its parent is the runner; a stale one naming a recycled pid that
        # leads a group must not let that group through
        proc = _source(sb, setup + "(exec sleep 3) & RAG_CHILD=$!; rag_own_child_alive && echo live=yes; kill $RAG_CHILD; "
                                   "(exec sleep 0.1) & RAG_CHILD=$!; sleep 1; rag_own_child_alive || echo reaped=yes; "
                                   f"RAG_CHILD={leader.pid}; rag_own_child_alive || echo foreign=yes; "
                                   f"RAG_STAGE_PGID={leader.pid}; rag_reap_stage_group '{log}'; echo \"stale=[$RAG_REAP_NOTE]\"")
        for word in ("live=yes", "reaped=yes", "foreign=yes", "stale=[]"):
            assert word in proc.stdout, (word, proc.stdout)
        time.sleep(0.3)
        assert leader.poll() is None  # still alive: nothing was signalled
    finally:
        leader.kill()
        leader.wait()


def test_interrupt_during_a_verl_stage_still_bundles_its_ray_logs(sb):  # runner:3 on the interrupt path
    ray_tmp = sb.tmp / "raytmp"
    ray_tmp.mkdir()
    sb.control = {"hang": ["smoke_worst_case"]}
    proc = sb.start(STAGES="smoke_worst_case", RAY_TMPDIR=ray_tmp)
    out = _interrupt(sb, proc, "smoke_worst_case", signal.SIGINT)
    assert proc.returncode == 130, out
    res = Result(sb, proc.returncode, out)
    assert "interrupted by SIGINT" in res.status("smoke_worst_case")["reason"]
    assert (res.bundle / "ray_logs" / "smoke_worst_case" / "session_2026_worst_case" / "logs" / "raylet.out").exists()
    with tarfile.open(res.tarball()) as tar:
        assert any(n.endswith("session_2026_worst_case/logs/worker-1a-01.err") for n in tar.getnames())


def _stage_bundle(root, stages, facts=""):
    """A hand-made bundle: stages = {name: (status, extra status lines, log text)}."""
    for d in ("env", "status", "logs", "gates", "smoke"):
        (root / d).mkdir(parents=True)
    (root / "env" / "facts.txt").write_text(facts)
    for i, (name, (status, extra, log)) in enumerate(stages.items(), 1):
        (root / "status" / ("%02d_%s.status" % (i, name))).write_text(
            "name=%s\nstatus=%s\nseconds=60\nlog=logs/%02d_%s.log\n%s" % (name, status, i, name, extra))
        (root / "logs" / ("%02d_%s.log" % (i, name))).write_text(log)
    return root


def _summarize(bundle):
    proc = subprocess.run([sys.executable, str(SUMMARIZER), str(bundle), "--tarball", "/x/t.tar.gz"],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.returncode == 0, proc.stdout
    return (bundle / "SUMMARY.txt").read_text()


def test_summary_3b_shapes(tmp_path):  # gate-3b:1/2: what counts as a 3b pass, summarizer only
    def summary(name, gate_json=None, status="PASS"):
        b = _stage_bundle(tmp_path / name, {"gate_3b": (status, "reason=exit 1\noutcome=verdict\n", "")})
        if gate_json:
            (b / "gates" / gate_json[0]).write_text(json.dumps(gate_json[1]))
        return _summarize(b)

    assert "3b two-way only" in summary("none")
    assert "3b two-way only" in summary("two", ("gate_b_greedy_sparsespec.json", {"pass": True, "three_way": False}))
    assert "3b two-way only" in summary("bad", ("gate_b_greedy_three_way.json", {"pass": False, "three_way": True}))
    assert "3b two-way only" in summary("flag", ("gate_b_greedy_three_way.json", {"pass": True}))
    assert "; 3b PASS" in flat(summary("good", ("gate_b_greedy_three_way.json", {"pass": True, "three_way": True})))
    text = summary("twofail", ("gate_b_greedy_sparsespec.json", {"pass": False, "three_way": False, "checks": [
        {"name": "engine_response_contract", "ok": False}, {"name": "R1", "ok": True}]}), status="FAIL")
    assert "3b failed: engine_response_contract" in text and "3b FAIL" in text


def test_summary_shows_what_a_gate_without_verdict_said_last(tmp_path):  # runner:8
    status = "reason=no verdict (the gate stopped before writing its JSON; see the log)\nexit_status=1\noutcome=no_verdict\n"
    conda = ("\nERROR conda.cli.main_run:execute(125): `conda run env PYTHONPATH=/r python -m integration.gates."
             "gate_c_reload --engine sparsespec` failed. (See above for error)\n")
    end = ("[run_all_gates] 12:00:09 exit status 1 after 9s\n[run_all_gates] 12:00:10 1 process(es) of this stage's own "
           "process group 77 still running after its command returned (pid pgid stat elapsed command):\n"
           "    78 77 S 00:09 python -m x\n[run_all_gates] ---- gate_3c_synthetic: FAIL ----\n")
    logs = {  # the gate's own last line, with or without a "[gate ...]" prefix; an exception line wins
        "gate_3c_synthetic": "[gate 3c] loading\ncheckpoint not found: /ck/x (pass --synthetic to create a stand-in there)\n",
        "gate_3c_vllm_reference": "[preflight] FAIL: cuda:0 (GPU 0) has 10.0 GiB free but this run needs 24.0 GiB\n",
        "gate_3c_step25": "[gate 3c] FAIL sampling: responses empty\nValueError: bad shape\nsys:1: ResourceWarning\n",
        "gate_3b_vllm": "",
    }
    b = _stage_bundle(tmp_path / "b", {n: ("FAIL", status, "[run_all_gates] stage started\n" + text + conda + end)
                                       for n, text in logs.items()})
    lines = _summarize(b).splitlines()

    def below(stage):
        row = next(i for i, line in enumerate(lines) if line.startswith(stage + " "))
        assert lines[row].endswith("  no verdict (the gate stopped before writing its JSON)"), lines[row]
        return lines[row + 1]

    assert below("gate_3c_synthetic") == "    checkpoint not found: /ck/x (pass --synthetic to create a stand-in there)"
    assert below("gate_3c_vllm_reference") == "    [preflight] FAIL: cuda:0 (GPU 0) has 10.0 GiB free but this run needs 24.0 GiB"
    assert below("gate_3c_step25") == "    ValueError: bad shape"
    assert below("gate_3b_vllm") == "    nothing printed: see logs/04_gate_3b_vllm.log"
    assert "3c synthetic no verdict" in flat("\n".join(lines))


def test_summary_stays_one_screen(tmp_path):  # G2: the cap holds by construction, the verdict always stays
    sys.path.insert(0, str(REPO / "integration"))
    try:
        import summarize_bundle as sbm
    finally:
        sys.path.pop(0)
    assert (sbm.MAX_LINES, sbm.WIDTH) == (46, 110)
    names = [s for s in ALL_STAGES if not s.endswith("_check")]
    # everything passed, three-way 3b, and step 25 failed only on its negative control (the likely
    # outcome on the box), with very long paths and two warnings
    long = "/data/" + "very_long_directory_name/" * 12
    facts = ("repo_root=%srepo\nstep50_ckpt=%sckpts/vllm_baseline_seed0/hf/global_step_50\nstep50_ckpt_exists=no\n"
             "rerun_data_dir=%sdata\nmem_available_kib=20971520\nprefix_override=PREFIX=/x (has lib/librmm.so)\n"
             "timeout_bin=/usr/bin/timeout\nsparsespec_env=verl-sparsespec\nbaseline_env=verl-baseline\n"
             % (long, long, long))
    stages = {n: ("PASS", "", "9 passed in 1.00s\n") for n in names}
    stages["gate_3c_step25"] = ("FAIL", "reason=exit 1\noutcome=verdict\n", "")
    for label in ("worst_case", "host_guard", "no_graph", "no_logprobs"):
        stages["smoke_%s_check" % label] = ("PASS", "", "")
    b = _stage_bundle(tmp_path / "likely", stages, facts)
    pair = {"num_prompts": 16, "identical": 14, "max_margin": 0.3, "first_diffs": [5], "pass": True}
    (b / "gates" / "gate_b_greedy_three_way.json").write_text(json.dumps(
        {"pass": True, "three_way": True, "pairs": {"sparsespec_vs_hf": pair, "vllm_vs_hf": pair, "sparsespec_vs_vllm": pair}}))
    (b / "gates" / "gate_c_reload_sparsespec_step25.json").write_text(json.dumps(
        {"pass": False, "failure_class": "negative_control_underpowered"}))
    (b / "gates" / "smoke_compare.json").write_text(json.dumps({"host_memory": {"rows": [
        {"cpu_offload_ratio": r, "pinned_gb": 16.0 * r, "inflight_worst_case": 27, "inflight_host_guard": 41,
         "fits_available": True} for r in (1.5, 2.0, 3.0, 4.0)]}, "runs": [{"label": "worst_case", "reading": "x" * 200}]}))
    for label in ("worst_case", "host_guard", "no_graph", "no_logprobs"):
        (b / "smoke" / ("gates_smoke_%s.timing.csv" % label)).write_text("step,timing_s/gen\n1,400\n2,380\n")
    text = _summarize(b)
    lines = text.splitlines()
    assert len(lines) <= 46 and all(len(line) <= 110 for line in lines), text
    assert lines[-1] == "send this file back: /x/t.tar.gz" and "VERDICT: NOT PASSED" in text
    assert "3c step-25 FAIL (underpowered)" in flat(text) and "; 3b PASS;" in flat(text)
    assert "gpu_unit_tests + 3 variants" in text and "in stage order: 9/9/9/9 passed, 1m00s/1m00s/1m00s/1m00s" in text
    assert "smoke runs, 3d check PASS" in text and "  all 4, in table order: 1m00s/1m00s/1m00s/1m00s" in text
    assert "PASS, nothing to add: env 1m00s, engine_import 1m00s, smoke_compare 1m00s" in text
    assert text.count("PASS  diverged 2/16") == 3  # the three 3b pair lines stay
    assert "WARNING: only 20 GiB host RAM available" in text and "NOTE: PREFIX=/x" in text
    # with paths this long the commands cannot stay on the screen: a pointer to the stage log instead
    assert "the two commands to paste (stock vLLM first, then the gate) are at the end of logs/13_gate_3c_step25.log" \
        in flat(text), text
    assert "conda run" not in text
    # with ordinary paths the same bundle shows the commands themselves
    (b / "env" / "facts.txt").write_text(facts.replace(long, "/home/h/"))
    text = _summarize(b)
    assert len(text.splitlines()) <= 46 and "the two commands to paste" not in text, text
    assert "cd /home/h/repo\nC=/home/h/ckpts/vllm_baseline_seed0/hf/global_step_50\nD=/home/h/data\n" in text
    assert text.count("conda run --no-capture-output -n ") == 2 and "worst_case reading: " in text

    # everything crashed at once and the run was refused on top: the middle is cut, never the verdict
    stages = {n: ("FAIL", "reason=crashed (exit 3: unexpected exception, traceback in the log; no verdict)\n"
                          "exit_status=3\noutcome=crashed\n", "Traceback (most recent call last):\nRuntimeError: boom %s\n" % n)
              for n in ALL_STAGES}
    b = _stage_bundle(tmp_path / "worst", stages)
    (b / "REFUSED.txt").write_text("".join("problem %d: %s Fix: do something\n" % (i, "long text " * 30) for i in range(5)))
    text = _summarize(b)
    lines = text.splitlines()
    assert len(lines) == 46 and all(len(line) <= 110 for line in lines), text
    assert "more line(s) cut to keep this one screen" in text
    assert lines[-1] == "send this file back: /x/t.tar.gz" and any(line.startswith("VERDICT: NOT PASSED") for line in lines)
    assert "RuntimeError: boom cpu_tests" in text and "REFUSED TO START" in text
    assert (b / "verdict.txt").read_text().strip() == "FAIL"
