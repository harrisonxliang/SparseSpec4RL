"""Append a run-metadata entry to the results log.

The work split requires every run to be logged with config, git hash,
GPU type, and GPU-hours. run_seed.sh calls this automatically at the
end of each run.
"""

import argparse
import datetime
import json
import os
import subprocess


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--config-version", required=True)
    parser.add_argument("--engine", required=True, choices=["vllm", "sparsespec"])
    parser.add_argument("--started-at", required=True)
    parser.add_argument("--ended-at", required=True)
    parser.add_argument("--exit-status", type=int, required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()

    fmt = "%Y-%m-%dT%H:%M:%SZ"
    start = datetime.datetime.strptime(args.started_at, fmt)
    end = datetime.datetime.strptime(args.ended_at, fmt)
    gpu_hours = round((end - start).total_seconds() / 3600, 3)

    ckpts = []
    if os.path.isdir(args.ckpt_dir):
        ckpts = sorted(d for d in os.listdir(args.ckpt_dir) if d.startswith("global_step_"))

    entry = {
        "exp_name": args.exp_name,
        "engine": args.engine,
        "seed": args.seed,
        "steps": args.steps,
        "config_version": args.config_version,
        "git_hash": sh("git rev-parse HEAD"),
        "git_dirty": bool(sh("git status --porcelain")),
        "gpu": sh("nvidia-smi --query-gpu=name --format=csv,noheader | head -n1"),
        "started_at": args.started_at,
        "ended_at": args.ended_at,
        "gpu_hours": gpu_hours,
        "exit_status": args.exit_status,
        "checkpoints": ckpts,
        "ckpt_dir": args.ckpt_dir,
        "verl_version": sh("python3 -c 'import verl; print(verl.__version__)'"),
        "vllm_version": sh("python3 -c 'import vllm; print(vllm.__version__)'"),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"logged run to {args.out}:")
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
