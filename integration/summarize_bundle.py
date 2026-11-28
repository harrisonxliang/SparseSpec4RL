#!/usr/bin/env python3
"""One-screen PASS / FAIL summary of a gate bundle written by integration/run_all_gates.sh.

    python3 integration/summarize_bundle.py <bundle dir> [--tarball PATH] [--out SUMMARY.txt]
    python3 integration/summarize_bundle.py --json-get FILE dotted.key   # helper used by the runner

Pure standard library (Python >= 3.6), so it runs with whatever python3 the box
has, outside any conda env. It must never crash on a missing or malformed file:
anything it cannot read is printed as ``n/a``. Besides printing the summary it
writes ``SUMMARY.txt`` and ``verdict.txt`` (``PASS`` or ``FAIL``) into the bundle.

Bundle layout it reads (all written by the runner):

    env/facts.txt                key=value facts (host, GPU, RAM, paths)
    env/<conda env>/versions.txt "VERSION <module> <version>" / "PATCH ..." lines
    status/NN_<stage>.status     key=value: name, status, reason, exit_status, seconds, note, skip_kind
    logs/NN_<stage>.log          full stdout + stderr of the stage
    gates/*.json                 gate outputs (--out-dir of every gate)
    smoke/gates_smoke_<label>.timing.csv, smoke/baseline_smoke.timing.csv
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import sys

WIDTH = 110
NA = "n/a"

SMOKE_LABELS = ("worst_case", "host_guard", "no_graph", "no_logprobs")
SMOKE_PREFIX = "gates_smoke_"

# (label in the verdict line, stages that must all PASS)
REQUIRED = (
    ("GPU unit tests", ("gpu_unit_tests", "gpu_unit_tests_host_guard", "gpu_unit_tests_offload",
                        "gpu_unit_tests_offload_host_guard")),
    ("3b", ("gate_3b",)),
    ("3c synthetic", ("gate_3c_synthetic",)),
    ("3c step-25", ("gate_3c_step25",)),
    ("3d", ("smoke_worst_case", "smoke_worst_case_check")),
)

UNDERPOWERED = "negative_control_underpowered"
UNDERPOWERED_SENTENCE = (
    "3c step-25 FAILED ONLY because the checkpoint is too close to the base model for the negative control "
    "(failure_class negative_control_underpowered): not an engine failure."
)


# ---------------------------------------------------------------------------
# tolerant readers
# ---------------------------------------------------------------------------
def read_text(path, limit=None):
    try:
        with open(path, errors="replace") as f:
            return f.read() if limit is None else f.read(limit)
    except (OSError, ValueError):
        return None


def read_tail(path, max_bytes=400000):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            return f.read().decode("utf-8", "replace")
    except (OSError, ValueError):
        return None


def read_kv(path):
    """key=value lines; later keys win; missing file -> {}."""
    out = {}
    text = read_text(path)
    if not text:
        return out
    for line in text.splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip()
    return out


def read_json(path):
    """dict or None (missing, unreadable, malformed or not an object)."""
    text = read_text(path)
    if text is None:
        return None
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def dig(obj, *keys):
    for key in keys:
        if isinstance(obj, dict) and key in obj:
            obj = obj[key]
        elif isinstance(obj, list) and isinstance(key, int) and -len(obj) <= key < len(obj):
            obj = obj[key]
        else:
            return None
    return obj


def num(value):
    """float or None (bools, strings, NaN-free conversion)."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def fmt(value, digits=4):
    value = num(value)
    if value is None:
        return NA
    if value != 0 and (abs(value) < 10 ** -(digits - 1) or abs(value) >= 10 ** 6):
        return "%.*g" % (digits, value)
    if abs(value) >= 100:
        return "%.1f" % value
    return "%.*g" % (digits, value)


def fmt_int(value):
    value = num(value)
    return NA if value is None else str(int(round(value)))


def fmt_duration(seconds):
    seconds = num(seconds)
    if seconds is None:
        return NA
    seconds = int(round(seconds))
    if seconds < 60:
        return "%ds" % seconds
    if seconds < 3600:
        return "%dm%02ds" % (seconds // 60, seconds % 60)
    return "%dh%02dm" % (seconds // 3600, (seconds % 3600) // 60)


def gib_from_kib(value):
    value = num(value)
    return None if value is None else value / (1024.0 * 1024.0)


def gib_from_mib(value):
    value = num(value)
    return None if value is None else value / 1024.0


def clip(text, width=WIDTH):
    text = re.sub(r"[\r\n\t]+", " ", str(text)).rstrip()
    return text if len(text) <= width else text[: width - 3] + "..."


def wrap(text, width=WIDTH, indent="  ", max_lines=3):
    words = " ".join(str(text).split()).split(" ")
    lines, line = [], ""
    for word in words:
        candidate = (line + " " + word) if line else word
        if len(indent) + len(candidate) > width and line:
            lines.append(indent + line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(indent + line)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = clip(lines[-1], width - 3) + "..." if not lines[-1].endswith("...") else lines[-1]
    return lines


# ---------------------------------------------------------------------------
# bundle contents
# ---------------------------------------------------------------------------
class Bundle(object):
    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.facts = read_kv(os.path.join(self.root, "env", "facts.txt"))
        self.stages = []  # ordered status dicts
        self.by_name = {}
        for path in sorted(glob.glob(os.path.join(self.root, "status", "*.status"))):
            status = read_kv(path)
            name = status.get("name") or re.sub(r"^\d+_", "", os.path.basename(path)[: -len(".status")])
            status["name"] = name
            status["_path"] = path
            if name in self.by_name:  # a stage written twice (should not happen): keep the last
                self.stages = [s for s in self.stages if s["name"] != name]
            self.stages.append(status)
            self.by_name[name] = status

    def path(self, *parts):
        return os.path.join(self.root, *parts)

    def gate_json(self, name):
        return read_json(self.path("gates", name))

    def status(self, name):
        return (self.by_name.get(name) or {}).get("status", "")

    def log_path(self, name):
        entry = self.by_name.get(name) or {}
        log = entry.get("log")
        if log:
            return log if os.path.isabs(log) else self.path(log)
        hits = sorted(glob.glob(self.path("logs", "*_%s.log" % name)))
        return hits[-1] if hits else None

    def versions(self, env_name):
        out = {}
        text = read_text(self.path("env", env_name, "versions.txt")) if env_name else None
        for line in (text or "").splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "VERSION":
                out[parts[1]] = " ".join(parts[2:3])
            elif len(parts) >= 2 and parts[0] == "PATCH":
                out["verl_patch"] = parts[1]
        return out


# ---------------------------------------------------------------------------
# per-stage result text
# ---------------------------------------------------------------------------
# '==== 9 passed, 1 skipped in 12.1s ====' (-v) or '9 passed, 1 skipped in 12.1s' (-q)
PYTEST_SUMMARY = re.compile(
    r"^(?:=+ )?((?:no tests ran|\d+ (?:passed|failed|skipped|errors?|deselected|xfailed|xpassed|warnings?))[^=]*?)"
    r" in [0-9.]+s(?: \([0-9:.]+\))?(?: =+)?\s*$")
PYTEST_COUNT = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected|warnings?)")
ERROR_LINE = re.compile(
    r"^\s*(?:[\w.]+\.)?(?:[A-Z]\w*(?:Error|Exception|Exit|Interrupt)|AssertionError|OutOfMemoryError)\b.*"
)


def pytest_counts(log_text):
    """'12 passed, 1 skipped' style summary of the LAST pytest run in a log, or None."""
    last = None
    for line in (log_text or "").splitlines():
        match = PYTEST_SUMMARY.match(line.strip())
        if match:
            last = match.group(1)
    if last is None:
        return None
    counts = {}
    for count, kind in PYTEST_COUNT.findall(last):
        kind = {"error": "errors", "warning": "warnings"}.get(kind, kind)
        counts[kind] = counts.get(kind, 0) + int(count)
    return counts


def pytest_text(log_text):
    counts = pytest_counts(log_text)
    if counts is None:
        return "pytest summary n/a"
    text = "%d passed, %d failed" % (counts.get("passed", 0), counts.get("failed", 0))
    for kind in ("errors", "skipped"):
        if counts.get(kind):
            text += ", %d %s" % (counts[kind], kind)
    failed = re.findall(r"^(?:FAILED|ERROR) (\S+)", log_text or "", flags=re.M)
    if failed:
        text += "; first: " + failed[0].split("/")[-1]
    return text


def last_error_line(log_text):
    """The last traceback-looking line of a log (exception type + message), or None."""
    found = None
    for line in (log_text or "").splitlines()[-3000:]:
        if ERROR_LINE.match(line) or re.search(r"\bCUDA out of memory\b|\bKilled\b|\bSegmentation fault\b", line):
            found = line.strip()
    return found


def gate_3b_pairs(data):
    """[(pair name, pass, diverged, earliest, largest margin, rule)] from a gate_b_greedy JSON of any shape."""
    if not isinstance(data, dict):
        return []
    rule_default = data.get("rule") or dig(data, "verdict", "rule")
    tie = dig(data, "config", "tie_margin")
    if not rule_default and num(tie) is not None:
        rule_default = "margin<=%s" % fmt(tie)
    candidates = []
    pairs = data.get("pairs")
    if isinstance(pairs, dict):
        candidates = [(str(k), v) for k, v in pairs.items() if isinstance(v, dict)]
    elif isinstance(pairs, list):
        for i, v in enumerate(pairs):
            if isinstance(v, dict):
                candidates.append((str(v.get("name") or v.get("pair") or "pair%d" % i), v))
    if not candidates and isinstance(data.get("summary"), dict):
        engine = dig(data, "config", "engine") or "engine"
        candidates = [("%s_vs_hf" % engine, {"summary": data["summary"], "pass": data.get("pass")})]
    out = []
    for name, entry in candidates:
        summary = entry.get("summary") if isinstance(entry.get("summary"), dict) else entry
        total, identical = num(summary.get("num_prompts")), num(summary.get("identical"))
        diverged = summary.get("diverged")
        if num(diverged) is None and total is not None and identical is not None:
            diverged = total - identical
        firsts = summary.get("first_diffs")
        earliest = summary.get("earliest_divergence")
        if num(earliest) is None and isinstance(firsts, list):
            positions = [num(x) for x in firsts if num(x) is not None and num(x) >= 0]
            earliest = min(positions) if positions else None
            if not positions and firsts:
                earliest = "none"
        margin = summary.get("max_margin")
        rule = entry.get("rule") or summary.get("rule") or rule_default
        passed = entry.get("pass")
        if isinstance(rule, dict):
            rule = rule.get("name") or rule.get("text")
        out.append((name, passed, diverged, earliest, margin, rule, total))
    return out


def gate_3c_text(data):
    if data is None:
        return "gate JSON n/a"
    stats = data.get("stats") if isinstance(data.get("stats"), dict) else {}
    bitwise = NA
    checks = data.get("checks")
    if isinstance(checks, list):
        flags = [c.get("ok") for c in checks if isinstance(c, dict) and "bitwise" in str(c.get("name", ""))]
        if flags:
            bitwise = "ok" if all(f is True for f in flags) else "NOT EQUAL"
    text = "reloaded %s, round-trip %s, neg-ctrl %s, bitwise %s" % (
        fmt(dig(stats, "reloaded", "mean")),
        fmt(dig(stats, "base_after_roundtrip", "mean")),
        fmt(dig(stats, "negative_control", "mean")),
        bitwise,
    )
    if data.get("pass") is False:
        failure_class = data.get("failure_class")
        if failure_class:
            text += "; " + str(failure_class)
        else:
            failed = data.get("failed_checks")
            if isinstance(failed, list) and failed:
                text += " | failed: " + ",".join(str(x) for x in failed[:3])
    return text


def vllm_ref_text(data):
    if data is None:
        return "vLLM reference JSON n/a"
    stats = dig(data, "stats", "engine_vs_trainer") or {}
    ckpt = dig(data, "config", "ckpt") or NA
    return "vLLM mean |dlogp| %s (n=%s, ckpt %s)" % (
        fmt(stats.get("mean")), fmt_int(stats.get("count")), os.path.basename(str(ckpt)))


def smoke_check_text(data):
    if data is None:
        return "3d checker JSON n/a"
    found = data.get("steps_found")
    steps = len(found) if isinstance(found, list) else found
    text = "3d checker %s, steps found %s" % ("PASS" if data.get("pass") is True else "FAIL" if data.get("pass") is False else NA,
                                             fmt_int(steps) if not isinstance(steps, str) else steps)
    checks = data.get("checks")
    if data.get("pass") is False and isinstance(checks, list):
        bad = [str(c.get("name")) for c in checks if isinstance(c, dict) and c.get("ok") is False]
        if bad:
            text += "; failed: " + ",".join(bad[:3])
    return text


# ---------------------------------------------------------------------------
# smoke timing CSVs
# ---------------------------------------------------------------------------
def read_timing_csv(path):
    """{step: {column: float}} or None."""
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "step" not in reader.fieldnames:
                return None
            steps = {}
            for row in reader:
                step = num(row.get("step"))
                if step is None:
                    continue
                values = {}
                for key, cell in row.items():
                    if key and key != "step":
                        value = num(cell)
                        if value is not None:
                            values[key] = value
                steps.setdefault(int(step), {}).update(values)
            return steps
    except (OSError, ValueError, csv.Error):
        return None


def column(steps, name, which=None):
    values = [(s, steps[s][name]) for s in sorted(steps) if name in steps[s]]
    if which is not None:
        values = [(s, v) for s, v in values if s in which]
    return [v for _, v in values]


def smoke_row(steps):
    if not steps:
        return None
    later = [s for s in steps if s >= 2]
    gen_later = column(steps, "timing_s/gen", later) if later else []
    gen_first = column(steps, "timing_s/gen", [min(steps)])
    bsz = column(steps, "sparsespec/batch_size_mean")
    kv = column(steps, "sparsespec/kv_util_peak")
    off = column(steps, "sparsespec/num_offloads")
    return {
        "steps": len(steps),
        "gen_first": gen_first[0] if gen_first else None,
        "gen_later": sum(gen_later) / len(gen_later) if gen_later else None,
        "batch_size_mean": sum(bsz) / len(bsz) if bsz else None,
        "kv_util_peak": max(kv) if kv else None,
        "num_offloads": sum(off) if off else None,
    }


def offload_rows_text(data):
    """'1.5: 16 GiB pinned, 27/40 in flight (worst_case/host_guard), fits yes; ...' from smoke_compare's host_memory."""
    rows = dig(data, "host_memory", "rows")
    if not isinstance(rows, list) or not rows:
        return None
    parts = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        fits = row.get("fits_available")
        parts.append("%s: %s GiB pinned, %s/%s in flight, fits %s" % (
            fmt(row.get("cpu_offload_ratio"), 3), fmt(row.get("pinned_gb"), 3), fmt_int(row.get("inflight_worst_case")),
            fmt_int(row.get("inflight_host_guard")), "yes" if fits is True else "NO" if fits is False else NA))
    if not parts:
        return None
    return ("cpu_offload_ratio (in flight worst_case/host_guard; fits = pinned + margin <= available RAM): "
            + "; ".join(parts))


def run_reading(data, label):
    runs = data.get("runs") if isinstance(data, dict) else None
    for run in runs if isinstance(runs, list) else []:
        if isinstance(run, dict) and run.get("label") == label and isinstance(run.get("reading"), str):
            return run["reading"]
    return None


def find_offload_note(data):
    """First string under a key that mentions the offload ratio, anywhere in the smoke_compare JSON."""
    stack = [data]
    while stack:
        item = stack.pop(0)
        if isinstance(item, dict):
            for key, value in item.items():
                if "offload_ratio" in str(key) and isinstance(value, str) and value.strip():
                    return value
                if "offload_ratio" in str(key) and isinstance(value, dict):
                    for sub in ("note", "text", "summary", "verdict"):
                        if isinstance(value.get(sub), str):
                            return value[sub]
                stack.append(value)
        elif isinstance(item, list):
            stack.extend(item)
    return None


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def stage_result(bundle, entry):
    name, status = entry["name"], entry.get("status", "")
    reason = entry.get("reason", "")
    note = entry.get("note", "")
    if status == "SKIP":
        return reason or "skipped"
    log_text = None
    if name in ("cpu_tests",) or name.startswith("gpu_unit_tests"):
        log_text = read_tail(bundle.log_path(name) or "")
        text = pytest_text(log_text)
    elif name == "gate_3b":
        data = bundle.gate_json("gate_b_greedy_three_way.json") or bundle.gate_json("gate_b_greedy_sparsespec.json")
        pairs = gate_3b_pairs(data)
        text = (note or "mode n/a") if pairs else "%s; gate JSON n/a" % (note or "mode n/a")
    elif name == "gate_3b_vllm":
        data = bundle.gate_json("gate_b_vllm_greedy.json")
        text = "JSON written, pass=%s" % (data.get("pass") if data else NA) if data else "gate JSON n/a"
    elif name == "gate_3c_synthetic":
        text = gate_3c_text(bundle.gate_json("gate_c_reload_sparsespec_synthetic.json"))
    elif name == "gate_3c_step25":
        text = gate_3c_text(bundle.gate_json("gate_c_reload_sparsespec_step25.json"))
    elif name == "gate_3c_vllm_reference":
        hits = sorted(glob.glob(bundle.path("gates", "gate_c_vllm_reference*.json")))
        hits = [h for h in hits if not h.endswith(".samples.json")]
        text = vllm_ref_text(read_json(hits[-1]) if hits else None)
    elif name.startswith("smoke_") and name.endswith("_check"):
        label = name[len("smoke_"): -len("_check")]
        text = smoke_check_text(bundle.gate_json("gate_d_smoke_%s%s.json" % (SMOKE_PREFIX, label)))
    elif name.startswith("smoke_") and name != "smoke_compare":
        steps = read_timing_csv(bundle.path("smoke", "%s%s.timing.csv" % (SMOKE_PREFIX, name[len("smoke_"):])))
        row = smoke_row(steps)
        text = ("%d step(s) in CSV" % row["steps"]) if row else "timing CSV n/a"
    elif name == "smoke_compare":
        data = bundle.gate_json("smoke_compare.json")
        text = "smoke_compare.json written" if data else "smoke_compare JSON n/a"
    elif name == "env":
        text = note or "recorded in env/"
    else:
        text = note or ""
    if status == "FAIL":
        if log_text is None:
            log_text = read_tail(bundle.log_path(name) or "")
        error = last_error_line(log_text)
        short = re.sub(r"^exit (\d+) \(crashed or did not start\)", r"exit \1 (crash)", reason or "failed")
        specific = text and NA not in text and not name.startswith("smoke_")
        parts = [text, short] if specific else [short] + ([error] if error else []) + ([text] if text else [])
        if specific and error:
            parts.append(error)
        text = " | ".join(parts)
    return text


def env_line(bundle, env_name):
    if not env_name:
        return "n/a"
    v = bundle.versions(env_name)
    if not v:
        return "%s: versions n/a" % env_name
    parts = []
    for key, short in (("torch", "torch"), ("vllm", "vllm"), ("verl", "verl"), ("flashinfer", "fi")):
        value = v.get(key)
        parts.append("%s %s" % (short, value if value and value != "NOT" else "-"))
    if v.get("verl_patch"):
        parts.append("patch " + v["verl_patch"])
    return "%s: %s" % (env_name, " ".join(parts))


def build_summary(bundle, tarball=None):
    f = bundle.facts
    lines = []
    lines.append(clip("SparseSpec gate run  host %s  commit %s (%s)  started %s  total %s" % (
        f.get("hostname", NA), f.get("commit", NA)[:10], f.get("branch", NA), f.get("start_utc", NA),
        fmt_duration(num(f.get("end_epoch")) - num(f.get("start_epoch")) if num(f.get("end_epoch")) is not None
                     and num(f.get("start_epoch")) is not None else None))))
    total, free0 = gib_from_mib(f.get("gpu_total_mib")), gib_from_mib(f.get("gpu_free_mib"))
    lines.append(clip("GPU %s: %s GiB total, %s GiB free at start (needed %s); driver %s, CUDA %s" % (
        f.get("gpu_name", NA), fmt(total, 3), fmt(free0, 3), fmt(gib_from_mib(f.get("required_free_mib")), 3),
        f.get("driver_version", NA), f.get("cuda_version", NA))))
    mem_total, mem_avail = gib_from_kib(f.get("mem_total_kib")), gib_from_kib(f.get("mem_available_kib"))
    lines.append(clip("Host RAM %s GiB total, %s GiB available at start; nproc %s; ulimit -l %s" % (
        fmt(mem_total, 4), fmt(mem_avail, 4), f.get("nproc", NA), f.get("ulimit_l", NA))))
    lines.append(clip("env " + env_line(bundle, f.get("sparsespec_env"))))
    lines.append(clip("env " + env_line(bundle, f.get("baseline_env"))))
    if f.get("timeout_bin") in (None, "", "none"):
        lines.append("WARNING: no timeout command on this host: a hanging stage was not stopped")
    if f.get("refused"):
        lines.append(clip("REFUSED TO START: " + f.get("refused")))
    errors = read_text(bundle.path("runner_errors.txt"))
    if errors and errors.strip():
        lines.append(clip("runner: %d unexpected command failure(s) inside run_all_gates.sh, see runner_errors.txt"
                          % len(errors.strip().splitlines())))
    lines.append("-" * WIDTH)
    lines.append("%-33s %-4s %6s  %s" % ("stage", "res", "time", "result"))
    for entry in bundle.stages:
        if entry["name"].startswith("smoke_") and entry["name"].endswith("_check"):
            continue  # shown on its smoke run's line
        prefix = "%-33s %-4s %6s  " % (entry["name"][:33], entry.get("status", "?")[:4], fmt_duration(entry.get("seconds")))
        result = stage_result(bundle, entry)
        check = bundle.by_name.get(entry["name"] + "_check")
        if check is not None and entry["name"].startswith("smoke_"):
            result += " | check %s %s" % (check.get("status", "?"), stage_result(bundle, check))
        lines.append(clip(prefix + result))
        if entry["name"] == "gate_3b" and entry.get("status") != "SKIP":
            data = bundle.gate_json("gate_b_greedy_three_way.json") or bundle.gate_json("gate_b_greedy_sparsespec.json")
            for name, passed, diverged, earliest, margin, rule, count in gate_3b_pairs(data)[:3]:
                lines.append(clip("    3b %-24s %s  diverged %s/%s, earliest at %s, largest margin %s, rule %s" % (
                    name[:24], "PASS" if passed is True else "FAIL" if passed is False else "    ",
                    fmt_int(diverged), fmt_int(count), earliest if isinstance(earliest, str) else fmt_int(earliest),
                    fmt(margin), rule or NA)))
    lines.append("-" * WIDTH)

    # smoke table
    rows = []
    for label in SMOKE_LABELS:
        rows.append((label, smoke_row(read_timing_csv(bundle.path("smoke", "%s%s.timing.csv" % (SMOKE_PREFIX, label)))),
                     bundle.status("smoke_%s_check" % label)))
    base = smoke_row(read_timing_csv(bundle.path("smoke", "baseline_smoke.timing.csv")))
    if any(r[1] for r in rows) or base:
        lines.append("%-22s %6s %11s %13s %16s %13s %13s  %s" % (
            "smoke run", "steps", "gen s st.1", "gen s st.2+", "batch_size_mean", "kv_util_peak", "num_offloads", "3d"))
        for label, row, check in rows:
            if row is None:
                lines.append("%-22s %6s" % (label, NA))
                continue
            lines.append("%-22s %6d %11s %13s %16s %13s %13s  %s" % (
                label, row["steps"], fmt(row["gen_first"]), fmt(row["gen_later"]), fmt(row["batch_size_mean"]),
                fmt(row["kv_util_peak"]), fmt_int(row["num_offloads"]), check or NA))
        if base:
            lines.append("%-22s %6d %11s %13s" % ("vLLM baseline smoke", base["steps"], fmt(base["gen_first"]),
                                                    fmt(base["gen_later"])))
        on, off = rows[0][1], rows[3][1]
        if on and off and num(on["gen_first"]) and num(off["gen_first"]):
            delta = on["gen_first"] - off["gen_first"]
            lines.append(clip("log-prob cost (calculate_log_probs, step 1 vs step 1): with %s s, without %s s -> %+.1f s (%+.1f%%)"
                              % (fmt(on["gen_first"]), fmt(off["gen_first"]), delta, 100.0 * delta / off["gen_first"])))
        else:
            lines.append("log-prob cost (calculate_log_probs): n/a (needs the worst_case and no_logprobs CSVs)")
    else:
        lines.append("smoke runs: no timing CSV in this bundle")
    compare = bundle.gate_json("smoke_compare.json")
    rows_text = offload_rows_text(compare) if compare else None
    note = find_offload_note(compare) if compare and not rows_text else None
    reading = run_reading(compare, "worst_case") if compare else None
    if rows_text:
        lines.extend(wrap(rows_text, indent="", max_lines=2))
        if reading:
            lines.append(clip("worst_case reading: " + reading))
    elif note:
        lines.extend(wrap("cpu_offload_ratio: " + note, indent="", max_lines=2))
    else:
        lines.append(clip("cpu_offload_ratio note: n/a (smoke_compare JSON missing or has no note); host RAM %s GiB total, "
                          "%s GiB available" % (fmt(mem_total, 4), fmt(mem_avail, 4))))

    # underpowered negative control: say it explicitly
    step25 = bundle.gate_json("gate_c_reload_sparsespec_step25.json")
    if step25 and step25.get("pass") is False and step25.get("failure_class") == UNDERPOWERED:
        nxt, step = dig(step25, "rerun", "next_ckpt"), dig(step25, "rerun", "next_step")
        rerun = (" Rerun on the step-%s checkpoint (commands in gates/gate_c_reload_sparsespec_step25.md): %s"
                 % (step if step is not None else "50", nxt) if nxt else
                 " Rerun on a later checkpoint: see gates/gate_c_reload_sparsespec_step25.md.")
        lines.extend(wrap(UNDERPOWERED_SENTENCE + rerun, indent="", max_lines=3))

    # verdict
    verdict, parts, failed = "PASS", [], []
    for label, names in REQUIRED:
        statuses = [bundle.status(n) for n in names]
        entries = [bundle.by_name.get(n) or {} for n in names]
        if label == "3c step-25" and any(e.get("skip_kind") == "missing_checkpoint" for e in entries):
            parts.append("3c step-25 not run (no checkpoint)")
            continue
        if statuses and all(s == "PASS" for s in statuses):
            parts.append("%s PASS" % label)
            continue
        verdict = "FAIL"
        if label == "3c step-25" and step25 and step25.get("failure_class") == UNDERPOWERED:
            word = "FAIL (underpowered)"
        elif "FAIL" in statuses:
            word = "FAIL"
        elif all(s in ("", None) for s in statuses):
            word = "not run"
        else:
            word = "not run" if all(s in ("SKIP", "", None) for s in statuses) else "incomplete"
        parts.append("%s %s" % (label, word))
        failed.append(label)
    if f.get("refused"):
        verdict = "FAIL"
    head = "VERDICT: ALL REQUIRED GATES PASSED" if verdict == "PASS" else "VERDICT: NOT PASSED"
    lines.extend(wrap("%s - %s" % (head, "; ".join(parts)), indent="", max_lines=2))
    lines.append("send this file back: %s" % (tarball or NA))
    return lines, verdict


def json_get(path, dotted):
    data = read_json(path)
    if data is None:
        return 1
    value = data
    for key in dotted.split("."):
        value = dig(value, key)
    if value is None or isinstance(value, (dict, list)):
        return 1
    print(value)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle", nargs="?", help="bundle directory written by integration/run_all_gates.sh")
    parser.add_argument("--tarball", default=None, help="path printed on the 'send this file back' line")
    parser.add_argument("--out", default=None, help="where to write the summary (default: <bundle>/SUMMARY.txt)")
    parser.add_argument("--json-get", nargs=2, metavar=("FILE", "KEY"), help="print one scalar from a JSON file")
    args = parser.parse_args(argv)
    if args.json_get:
        return json_get(*args.json_get)
    if not args.bundle:
        parser.error("bundle directory required")
    bundle = Bundle(args.bundle)
    try:
        lines, verdict = build_summary(bundle, args.tarball)
    except Exception as exc:  # last resort: a summary must always come out
        lines = ["summary failed (%s: %s); status files:" % (type(exc).__name__, exc)]
        lines += ["%-34s %-5s %s" % (s["name"], s.get("status", "?"), clip(s.get("reason", ""), 60)) for s in bundle.stages]
        lines.append("send this file back: %s" % (args.tarball or NA))
        verdict = "FAIL"
    text = "\n".join(lines) + "\n"
    out = args.out or os.path.join(bundle.root, "SUMMARY.txt")
    try:
        with open(out, "w") as f:
            f.write(text)
        with open(os.path.join(bundle.root, "verdict.txt"), "w") as f:
            f.write(verdict + "\n")
    except OSError as exc:
        sys.stderr.write("cannot write %s: %s\n" % (out, exc))
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
