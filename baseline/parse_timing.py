"""Parse a verl console log into a per-step metrics CSV.

verl's console logger prints one line per training step of the form
  step:N - timing_s/gen:12.34 - timing_s/update_actor:5.6 - critic/score/mean:0.12 - ...
This grabs every numeric key:value pair on such lines, producing a wide
CSV (one row per step, union of keys as columns). The columns we care
about for the paper:
  timing_s/gen            rollout generation time
  timing_s/update_actor   policy update time
  timing_s/step           total step time (weight sync is step - gen - update - logprob)
  critic/score/mean       mean rule-based reward
  response_length/mean    mean response length

Usage: python baseline/parse_timing.py baseline/logs/EXP.log -o baseline/results/EXP.timing.csv
"""

import argparse
import csv
import re

PAIR_RE = re.compile(r"([A-Za-z0-9_@/\.\-]+):(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?=\s|$|\s*-\s)")

SUMMARY_KEYS = ["timing_s/gen", "timing_s/update_actor", "timing_s/step",
                "critic/score/mean", "response_length/mean"]


def parse(log_path: str):
    steps = {}
    with open(log_path, errors="replace") as f:
        for line in f:
            if "step:" not in line or "timing" not in line:
                continue
            pairs = dict(PAIR_RE.findall(line))
            if "step" not in pairs:
                continue
            step = int(float(pairs.pop("step")))
            row = {k: float(v) for k, v in pairs.items()}
            steps.setdefault(step, {}).update(row)
    return steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()

    steps = parse(args.log)
    if not steps:
        print(f"WARN: no step metric lines found in {args.log}")
        return

    keys = sorted({k for row in steps.values() for k in row})
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + keys)
        for step in sorted(steps):
            writer.writerow([step] + [steps[step].get(k, "") for k in keys])

    print(f"wrote {args.out} ({len(steps)} steps, {len(keys)} metrics)")
    for key in SUMMARY_KEYS:
        vals = [row[key] for row in steps.values() if key in row]
        if vals:
            print(f"  {key}: mean {sum(vals)/len(vals):.2f} (last {vals[-1]:.2f})")


if __name__ == "__main__":
    main()
