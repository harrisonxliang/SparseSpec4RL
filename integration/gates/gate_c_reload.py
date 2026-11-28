"""Gate 3c: in-place weight reload and engine-vs-trainer log-prob agreement.

    # real gate on the A100, with the baseline's step-25 checkpoint
    python -m integration.gates.gate_c_reload --engine sparsespec \
        --ckpt $CKPT_ROOT/vllm_baseline_seed0/hf/global_step_25
    # before the baseline has trained: synthetic checkpoint (base + noise), created on the fly
    python -m integration.gates.gate_c_reload --engine sparsespec --synthetic
    # side by side with stock vLLM (run gate_c_vllm_reference.py in the baseline env first)
    python -m integration.gates.gate_c_reload --engine sparsespec --synthetic \
        --vllm-reference baseline/results/sparsespec_checks/gate_c_vllm_reference.json

ONE engine instance does everything (SparseSpecEngine allows one per process):

  build     engine from the BASE model; record every parameter's data pointer;
            greedy fingerprint F0 (a few prompts, short).
  reload 1  load_weights(checkpoint) while awake (CUDA graphs live):
            verify_weights(checkpoint) bitwise ok, tied weights tied, pointers
            unchanged, verify_weights(base) no longer ok. Greedy fingerprint F1.
            Sample --num-prompts x --n responses (each prompt n times in a
            row, as verl repeats them) at T=1; score the same sequences with
            the trainer forward on the CHECKPOINT weights   -> "reloaded".
  reload 2  load_weights(base): verify_weights(base) ok, verify_weights(checkpoint)
            NOT ok. Greedy fingerprint must equal F0. Sample again; score with
            HF(base)        -> "base_after_roundtrip" (same small range as "reloaded")
            HF(checkpoint)  -> "negative_control"     (engine has base weights, the
                               trainer has checkpoint weights: clearly larger)
  reload 3  load_weights(checkpoint): bitwise ok, greedy fingerprint F2 == F1
            exactly (nothing stale survives a round trip).

With ``--sleep-while-scoring`` (default) the engine sleeps while the trainer
forward runs and reloads 2 and 3 happen asleep, followed by wake_up(): the
order of a training step (sharding manager: load weights, wake, generate,
sleep). Reload 1 always happens awake, under the CUDA graphs captured with
the base weights.

To read the negative control the report also says how different the
checkpoint is from the base model: the fraction of bf16 weights that differ
and mean |delta log p| between HF(base) and HF(checkpoint) on the same
sequences. The mean-based control only has power when that distance is well
above the engine's bf16 noise. A real early checkpoint may not be (lr 1e-6 for
25 steps moves most weights by less than half a bf16 step); the synthetic one
is. For that case there is a second statistic, the tracking coefficient
(``common.tracking_coefficient``): the least-squares weight of
HF(checkpoint) - HF(base) in engine - HF(base), about 1 when the engine's
log-probs follow the checkpoint and about 0 when they follow the base model,
with a standard error that shrinks with the number of tokens.

Verdict: every sub-check must pass; thresholds are flags and are printed.
  * bitwise / pointer / tied / fingerprint / response-contract checks as above;
  * "reloaded" and "base_after_roundtrip" means <= --max-mean-abs-dlogp and
    within --same-range-factor of each other;
  * tracking coefficient > 0.5 after reload 1 and < 0.5 after reload 2,
    whenever its standard error is <= --tracking-max-se;
  * negative control, ``--neg-control mean`` (default): negative-control mean
    > --neg-control-factor x reloaded mean. With ``--neg-control tracking`` the
    mean rule is only reported and the tracking coefficients (which must then
    be evaluable) decide: for checkpoints too close to the base model.
Exit status 1 on FAIL. Outputs: ``<out-dir>/gate_c_reload_<tag>.json`` / ``.md``
(``<tag>`` defaults to the engine name), plus ``gate_c_reload_<tag>.samples.json``
on failure (``--save-samples``).

Failure classification (JSON ``failure_class``; the verdict and exit status
are NOT changed by it): ``null`` on PASS; ``"negative_control_underpowered"``
when the ONLY failed check is the mean-based negative control, every other
check (bitwise / pointer / tied / fingerprint / response contract / means /
same range) passed, and HF(base) vs HF(checkpoint) themselves differ by no
more than the rule demands, i.e. no engine could have satisfied it: the
checkpoint is too close to the base model. The JSON, the markdown and the
console then say so, with the numbers, what the tracking coefficients say,
and the rerun command for a later checkpoint (``.../hf/global_step_50`` when
``--ckpt`` ends in ``global_step_<N>``). ``"engine"`` for every other failure.
``failed_checks`` lists the names of the failed checks.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import math
import os
import re
import shlex
import sys
import textwrap
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

from integration.gates import common  # noqa: E402
from integration.gates import make_synthetic_ckpt as synth  # noqa: E402


def _has_weights(path: str | None) -> bool:
    return bool(path) and os.path.isdir(path) and bool(glob.glob(os.path.join(path, "*.safetensors")))


def _existing_checkpoint(path: str, args) -> dict:
    """Describe a checkpoint directory that is already there (synthetic ones carry their report)."""
    report_path = os.path.join(path, synth.REPORT_NAME)
    if not os.path.exists(report_path):
        return {"dir": path, "kind": "given"}
    with open(report_path) as f:
        report = json.load(f)
    common.log(
        f"[gate 3c] {path} is a synthetic checkpoint (rel_noise={report.get('rel_noise')}, seed={report.get('seed')}, "
        f"base {report.get('base_dir')}); reusing it"
    )
    if args.synthetic and (report.get("rel_noise") != args.rel_noise or report.get("seed") != args.synthetic_seed):
        common.log("[gate 3c] NOTE: --rel-noise / --synthetic-seed are ignored because the checkpoint already exists")
    return {"dir": path, "kind": "synthetic (reused)", "report": report}


def resolve_checkpoint(args, base_dir: str) -> tuple[str, dict]:
    """The checkpoint directory to reload: ``--ckpt`` if it exists; with
    ``--synthetic`` a synthetic one is created there (or in the default
    synthetic directory) when it is absent; without either flag the
    baseline's step-25 checkpoint."""
    explicit = os.path.abspath(os.path.expanduser(args.ckpt)) if args.ckpt else None
    if _has_weights(explicit):
        return explicit, _existing_checkpoint(explicit, args)
    if args.synthetic:
        target = explicit or synth.default_synthetic_dir(args.rel_noise, args.synthetic_seed)
        if _has_weights(target):
            return target, _existing_checkpoint(target, args)
        common.log(f"[gate 3c] creating a synthetic checkpoint in {target} (rel_noise={args.rel_noise:g})")
        report = synth.make_synthetic_checkpoint(base_dir, target, args.rel_noise, args.synthetic_seed)
        return target, {"dir": target, "kind": "synthetic (created)", "report": report}
    if explicit:
        raise SystemExit(f"checkpoint not found: {explicit} (pass --synthetic to create a stand-in there)")
    if _has_weights(common.DEFAULT_STEP25_CKPT):
        return common.DEFAULT_STEP25_CKPT, {"dir": common.DEFAULT_STEP25_CKPT, "kind": "baseline step 25"}
    raise SystemExit(
        f"no checkpoint: {common.DEFAULT_STEP25_CKPT} does not exist. Pass --ckpt <hf dir>, or --synthetic to "
        "create a stand-in (base model + noise)."
    )


def greedy_fingerprint(engine, prompts, max_new_tokens: int, eos) -> tuple[str, list[list[int]]]:
    result = engine.generate(prompts, max_new_tokens=max_new_tokens, temperature=0.0, eos_token_ids=eos)
    tokens = [list(t) for t in result.token_ids]
    return common.fingerprint(tokens), tokens


def describe_token_diff(a: list[list[int]], b: list[list[int]]) -> str:
    for i, (x, y) in enumerate(zip(a, b)):
        pos = common.first_difference(x, y)
        if pos >= 0:
            return f"first difference: prompt {i}, position {pos}"
    return "identical"


def sample(engine, prompts, args, eos, label: str, checks: common.Checks) -> dict:
    common.log(f"[gate 3c] {label}: sampling {len(prompts)} responses, max_new_tokens={args.max_new_tokens}, T={args.temperature}")
    start = time.perf_counter()
    result = engine.generate(
        prompts, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=1.0, top_k=-1, eos_token_ids=eos
    )
    seconds = time.perf_counter() - start
    num_tokens = sum(len(t) for t in result.token_ids)
    problems = common.check_responses(result, args.max_new_tokens, eos)
    checks.add(
        f"{label}_response_contract",
        not problems,
        f"{len(result.token_ids)} responses, {num_tokens} tokens, {seconds:.1f}s"
        if not problems
        else f"{len(problems)} violation(s), e.g. {problems[:3]}",
    )
    unusable = [i for i, (t, lp) in enumerate(zip(result.token_ids, result.logprobs)) if len(t) == 0 or len(t) != len(lp)]
    if unusable:
        raise SystemExit(
            f"[gate 3c] FAIL {label}: {len(unusable)} response(s) are empty or lack one log-prob per token "
            f"(e.g. index {unusable[:5]}); the log-prob comparison cannot run"
        )
    stats = dict(result.stats)
    common.log(
        f"[gate 3c] {label}: {num_tokens} tokens in {seconds:.1f}s ({num_tokens / max(seconds, 1e-9):.0f} tok/s), "
        f"stopped by EOS: {sum(r == 'stop' for r in result.finish_reasons)}/{len(result.finish_reasons)}, "
        f"accept_len_mean={stats.get('accept_len_mean', 0.0):.2f}, num_offloads={stats.get('num_offloads', 0.0):.0f}"
    )
    return {
        "responses": [list(t) for t in result.token_ids],
        "logprobs": [list(x) for x in result.logprobs],
        "finish_reasons": list(result.finish_reasons),
        "engine_stats": stats,
        "seconds": seconds,
        "num_tokens": num_tokens,
    }


def stat_line(name: str, stats: dict) -> str:
    return (
        f"{name:<24s} n={stats['count']:<8d} mean={common.fmt_num(stats['mean'])} median={common.fmt_num(stats['median'])} "
        f"p99={common.fmt_num(stats['p99'])} max={common.fmt_num(stats['max'])} max_prob_ratio={common.fmt_num(stats['max_prob_ratio'])}"
    )


# ----------------------------------------------------------------------------
# failure classification (does not change the verdict)
# ----------------------------------------------------------------------------
FAILURE_UNDERPOWERED = "negative_control_underpowered"
FAILURE_ENGINE = "engine"
# the mean-based negative-control rule(s): the only checks an underpowered checkpoint may fail
MEAN_RULE_CHECKS = ("negative_control_clearly_larger",)
# every one of these must be present AND pass before a failure may be blamed on the checkpoint
ENGINE_CHECKS = (
    "checkpoint_differs_from_base",
    "build_matches_base",
    "reload1_bitwise",
    "reload1_tied",
    "reload1_pointers_unchanged",
    "reload1_base_no_longer_matches",
    "sample_1_response_contract",
    "reload2_bitwise_base",
    "reload2_checkpoint_no_longer_matches",
    "reload2_pointers_unchanged",
    "base_fingerprint_after_roundtrip",
    "sample_2_response_contract",
    "reload3_bitwise",
    "reload3_pointers_unchanged",
    "fingerprint_F2_equals_F1",
    "reloaded_all_finite",
    "base_after_roundtrip_all_finite",
    "reloaded_mean_small",
    "base_after_roundtrip_mean_small",
    "base_after_roundtrip_same_range",
)


def classify_failure(check_items, *, neg_control_mode: str, mean_rule_ok: bool, distance: float, needed: float):
    """``(failure_class, reason)``: ``(None, ...)`` when every check passed,
    ``(FAILURE_UNDERPOWERED, ...)`` when the gate failed ONLY on the mean-based
    negative control and HF(base) vs HF(checkpoint) differ by no more than that
    rule demands (``distance <= needed``: no engine could satisfy it), and
    ``(FAILURE_ENGINE, ...)`` for every other failure. Conservative: anything
    unexpected (a missing engine check, non-finite numbers, tracking mode) is
    ``engine``."""
    failed = [c["name"] for c in check_items if not c["ok"]]
    if not failed:
        return None, "all checks passed"
    extra = [name for name in failed if name not in MEAN_RULE_CHECKS]
    if extra:
        return FAILURE_ENGINE, f"engine checks failed: {extra}"
    if neg_control_mode != "mean":
        return FAILURE_ENGINE, f"--neg-control {neg_control_mode}: the mean rule is not part of the verdict"
    present = {c["name"]: c["ok"] for c in check_items}
    missing = [name for name in ENGINE_CHECKS if name not in present]
    if missing:
        return FAILURE_ENGINE, f"engine checks missing from the run: {missing}"
    if mean_rule_ok:
        return FAILURE_ENGINE, "inconsistent: the mean rule failed as a check but holds"
    if not (isinstance(distance, (int, float)) and math.isfinite(distance) and math.isfinite(needed)):
        return FAILURE_ENGINE, f"checkpoint distance {distance!r} / required {needed!r} not finite"
    if distance > needed:
        return FAILURE_ENGINE, (
            f"HF(base) vs HF(checkpoint) differ by {common.fmt_num(distance)} > the required {common.fmt_num(needed)}: "
            "a correct engine could have satisfied the mean rule"
        )
    return FAILURE_UNDERPOWERED, (
        f"only the mean rule failed and HF(base) vs HF(checkpoint) differ by {common.fmt_num(distance)} <= the "
        f"required {common.fmt_num(needed)}"
    )


def _flag_value(argv: list[str], flag: str) -> str | None:
    value = None
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            value = argv[i + 1]
        elif arg.startswith(flag + "="):
            value = arg[len(flag) + 1 :]
    return value


def _set_flag(argv: list[str], flag: str, value: str) -> list[str]:
    """``argv`` with every ``flag VALUE`` / ``flag=VALUE`` replaced by ``flag value`` (appended if absent)."""
    out: list[str] = []
    found = False
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == flag:
            if not found:
                out += [flag, value]
            found = True
            i += 2
            continue
        if arg.startswith(flag + "="):
            if not found:
                out += [flag, value]
            found = True
            i += 1
            continue
        out.append(arg)
        i += 1
    if not found:
        out += [flag, value]
    return out


def next_checkpoint(ckpt_dir: str) -> dict | None:
    """The next checkpoint to try when ``ckpt_dir`` ends in ``global_step_<N>``:
    step 50 for N < 50, step 100 for N < 100 (verl saves every 25 steps of a
    100-step run); None otherwise."""
    path = os.path.normpath(ckpt_dir)
    match = re.fullmatch(r"global_step_(\d+)", os.path.basename(path))
    if not match:
        return None
    step = int(match.group(1))
    nxt = 50 if step < 50 else 100 if step < 100 else None
    if nxt is None:
        return None
    new_dir = os.path.join(os.path.dirname(path), f"global_step_{nxt}")
    return {"step": step, "next_step": nxt, "dir": new_dir, "exists": os.path.isdir(new_dir)}


def next_tag(tag: str, step: int, nxt: int) -> str:
    """``sparsespec_step25`` -> ``sparsespec_step50``; a tag without the step number gets ``_step<next>``."""
    pattern = re.compile(rf"(step_?){step}(?!\d)")
    if pattern.search(tag):
        return pattern.sub(lambda m: f"{m.group(1)}{nxt}", tag, count=1)
    return f"{tag}_step{nxt}"


GENERIC_RERUN_HINT = (
    "rerun the same command with --ckpt pointing at a later checkpoint of the same training run "
    "(e.g. .../hf/global_step_50 instead of .../hf/global_step_25) and a new --tag"
)


def rerun_commands(raw_argv: list[str], args, ckpt_dir: str, tag: str) -> dict:
    """Commands (repo root, PYTHONPATH=repo) that repeat this gate on a later checkpoint."""
    nxt = next_checkpoint(ckpt_dir)
    if nxt is None:
        return {"next_step": None, "next_ckpt": None, "next_ckpt_exists": None, "commands": [], "hint": GENERIC_RERUN_HINT}
    argv = _set_flag(list(raw_argv), "--ckpt", nxt["dir"])
    argv = _set_flag(argv, "--tag", next_tag(tag, nxt["step"], nxt["next_step"]))
    commands = []
    if args.vllm_reference:
        # the stock-vLLM numbers were measured on the old checkpoint: measure them again on the new one
        ref_dir = os.path.dirname(os.path.abspath(os.path.expanduser(args.vllm_reference)))
        ref_tag = f"step{nxt['next_step']}"
        ref_argv = ["python", "integration/gates/gate_c_vllm_reference.py", "--ckpt", nxt["dir"], "--tag", ref_tag,
                    "--out-dir", ref_dir]  # fmt: skip
        for flag in ("--num-prompts", "--n", "--max-new-tokens", "--prompt-seed", "--data", "--prompts-file"):
            value = _flag_value(raw_argv, flag)
            if value is not None:
                ref_argv += [flag, value]
        commands.append({"env": "verl-baseline", "argv": ref_argv, "command": shlex.join(ref_argv)})
        argv = _set_flag(argv, "--vllm-reference", os.path.join(ref_dir, f"gate_c_vllm_reference_{ref_tag}.json"))
    gate_argv = ["python", "-m", "integration.gates.gate_c_reload"] + argv
    env = "verl-sparsespec" if args.engine == "sparsespec" else "verl-sparsespec-dev or verl-sparsespec"
    commands.append({"env": env, "argv": gate_argv, "command": shlex.join(gate_argv)})
    hint = (
        f"same command with --ckpt {nxt['dir']}"
        + ("" if nxt["exists"] else " (that directory does not exist yet: wait for the training run to save it)")
    )
    return {"next_step": nxt["next_step"], "next_ckpt": nxt["dir"], "next_ckpt_exists": nxt["exists"],
            "commands": commands, "hint": hint}  # fmt: skip


def underpowered_explanation(*, stats, weights_diff, needed, distance, args, tracking, tracking_evaluable, rerun) -> str:
    """The paragraph that goes into the console, the markdown and the JSON."""
    fmt = common.fmt_num
    reloaded, negative = stats["reloaded"], stats["negative_control"]
    track_1, track_2 = tracking["sample_1"], tracking["sample_2"]
    text = (
        f"The gate FAILED only because the checkpoint is too close to the base model for the "
        f"{fmt(args.neg_control_factor)}x mean rule to have power; this is not an engine failure, and the default "
        "verdict (FAIL, exit status 1) is unchanged. Every engine check passed: bitwise reload, parameter pointers, "
        "tied weights, greedy fingerprints, response contract, the reloaded / base_after_roundtrip means and their "
        "same-range rule. The only failed check is negative_control_clearly_larger: it needs engine(base) vs "
        f"HF(checkpoint) mean |delta log p| ({fmt(negative['mean'])}) to exceed {fmt(args.neg_control_factor)} x "
        f"max(reloaded mean {fmt(reloaded['mean'])}, noise floor {args.noise_floor:g}) = {fmt(needed)}, but HF(base) "
        f"and HF(checkpoint) themselves differ by only {fmt(distance)} on the same sequences "
        f"({100 * weights_diff['fraction_changed']:.3f}% of bf16 weights changed: {weights_diff['num_changed']:,} of "
        f"{weights_diff['num_elements']:,}), so no engine, however correct, could pass it. "
        f"Tracking coefficients (0 = the engine's log-probs follow the base model, 1 = they follow the checkpoint): "
        f"{fmt(track_1['beta'])} +/- {fmt(track_1['se'])} after reload 1 (checkpoint weights) and "
        f"{fmt(track_2['beta'])} +/- {fmt(track_2['se'])} after reload 2 (base weights); "
    )
    if tracking_evaluable:
        text += (
            f"their standard errors are within --tracking-max-se={args.tracking_max_se:g}, so they were part of the "
            "verdict and passed (> 0.5 after reload 1, < 0.5 after reload 2): the engine follows whichever weights "
            "it holds. "
        )
    else:
        text += (
            f"NOT evaluable (standard errors above --tracking-max-se={args.tracking_max_se:g}, or undefined), so on "
            "these samples that statistic cannot tell the two models apart either: it neither confirms nor "
            "contradicts the reload. "
        )
    if rerun["commands"]:
        text += (
            f"Rerun the gate on the step-{rerun['next_step']} checkpoint, which is further from the base model: "
            + " ; then ".join(c["command"] for c in rerun["commands"])
            + ("" if rerun["next_ckpt_exists"] else f" ({rerun['next_ckpt']} does not exist yet)")
            + "."
        )
    else:
        text += f"To get a decisive negative control, {rerun['hint']}."
    return text


def build_markdown(out: dict) -> str:
    cfg, stats, ckpt = out["config"], out["stats"], out["checkpoint"]
    diff = out["checkpoint_vs_base"]
    track = out["tracking"]
    failure_lines: list[str] = []
    if out.get("failure_class") == FAILURE_UNDERPOWERED:
        rerun = out.get("rerun") or {}
        failure_lines = [
            "## Failure class: `negative_control_underpowered` (checkpoint too close to the base model)",
            "",
            out["failure_explanation"],
            "",
        ]
        if rerun.get("commands"):
            failure_lines += [f"Rerun on the step-{rerun['next_step']} checkpoint (repo root, `PYTHONPATH=$PWD`):", "", "```bash"]
            for command in rerun["commands"]:
                failure_lines += [f"# env: {command['env']}", command["command"]]
            failure_lines += ["```", ""]
        else:
            failure_lines += [f"Rerun: {rerun.get('hint', GENERIC_RERUN_HINT)}.", ""]
    elif out.get("failure_class"):
        failure_lines = [
            f"Failure class: `{out['failure_class']}` ({out.get('failure_reason')}). "
            f"Failed checks: {', '.join(f'`{n}`' for n in out.get('failed_checks', []))}",
            "",
        ]
    rows = [
        ("reloaded: engine(checkpoint) vs HF(checkpoint)", stats["reloaded"]),
        ("base_after_roundtrip: engine(base) vs HF(base)", stats["base_after_roundtrip"]),
        ("negative_control: engine(base) vs HF(checkpoint)", stats["negative_control"]),
        ("negative_control_mirror: engine(checkpoint) vs HF(base)", stats["negative_control_mirror"]),
        ("checkpoint distance: HF(base) vs HF(checkpoint), sequences of sample 2", stats["hf_base_vs_ckpt"]),
    ]
    lines = [
        f"# Gate 3c - weight reload: {cfg['engine']} engine",
        "",
        f"**Verdict: {'PASS' if out['pass'] else 'FAIL'}**",
        "",
        *failure_lines,
        f"- base model: `{cfg['model']}`",
        f"- checkpoint: `{ckpt['dir']}` ({ckpt['kind']})",
        f"- sampling: {cfg['num_prompts']} prompts x n={cfg['n']} = {cfg['num_prompts'] * cfg['n']} responses per sample, "
        f"T={cfg['temperature']}, max_new_tokens={cfg['max_new_tokens']} ({out['prompts']['source']})",
        f"- trainer forward: HF {cfg['trainer_dtype']}, autocast bf16 on CUDA, logits / T, float32 log-softmax "
        f"(`common.hf_trainer_logprobs`), device {cfg['device']}",
        f"- engine sleeps while the trainer forward runs: {cfg['sleep_while_scoring']}",
        "",
        "## Checks",
        "",
        out["checks_markdown"],
        "",
        "Thresholds: "
        f"`--max-mean-abs-dlogp {cfg['max_mean_abs_dlogp']}` (reloaded / base_after_roundtrip mean), "
        f"`--same-range-factor {cfg['same_range_factor']}` (the two means within this factor of each other, "
        f"means below `--noise-floor {cfg['noise_floor']}` count as equal), "
        f"`--neg-control-factor {cfg['neg_control_factor']}` (negative-control mean > factor x max(reloaded mean, noise floor)), "
        f"`--neg-control {cfg['neg_control']}`, `--tracking-max-se {cfg['tracking_max_se']}` (tracking coefficient "
        "> 0.5 after reload 1 and < 0.5 after reload 2 when its standard error is at most this).",
        "",
        "## Engine vs trainer: |delta log p| per response token",
        "",
        common.stats_table(rows),
        "",
        "`max_prob_ratio` = exp(max |delta log p|), the worst per-token probability ratio.",
        "",
        "## How different is the checkpoint from the base model?",
        "",
        f"- bf16 weights that differ: {diff['num_changed']:,} of {diff['num_elements']:,} "
        f"({100 * diff['fraction_changed']:.2f}%), in {diff['num_tensors_changed']} of {diff['num_tensors']} tensors",
        f"- mean |delta log p| between HF(base) and HF(checkpoint) on the same sequences: "
        f"{common.fmt_num(stats['hf_base_vs_ckpt']['mean'])} (this is what the negative control can show at most, "
        "up to engine noise)",
        f"- tracking coefficient (0 = engine follows the base model, 1 = follows the checkpoint): "
        f"after reload 1: {common.fmt_num(track['sample_1']['beta'])} +/- {common.fmt_num(track['sample_1']['se'])}; "
        f"after reload 2 (base): {common.fmt_num(track['sample_2']['beta'])} +/- {common.fmt_num(track['sample_2']['se'])}",
        f"- mean-based negative control ({'part of the verdict' if out['negative_control']['mode'] == 'mean' else 'reported only: --neg-control tracking'}): "
        f"{'holds' if out['negative_control']['mean_rule_ok'] else 'does NOT hold'}. {out['negative_control']['mean_rule_detail']}",
        "",
    ]
    ref = out.get("vllm_reference")
    if ref:
        lines += [
            "## Side by side with stock vLLM",
            "",
            f"Reference file: `{ref['path']}` (checkpoint `{ref.get('ckpt')}`, {ref.get('num_sequences')} sequences)",
            "",
            common.stats_table(
                [
                    (f"{cfg['engine']}: reloaded", stats["reloaded"]),
                    (f"{cfg['engine']}: base_after_roundtrip", stats["base_after_roundtrip"]),
                    ("stock vLLM vs HF trainer", ref["stats"]),
                ]
            ),
            "",
            f"Ratio of means, {cfg['engine']} reloaded / stock vLLM: {common.fmt_num(ref.get('mean_ratio'))}"
            + ("" if ref.get("same_checkpoint") else "  (NOTE: the reference was measured on a different checkpoint path)"),
            "",
        ]
    lines += [
        "## Greedy fingerprints",
        "",
        common.md_table(["fingerprint", "weights", "sha256"], [[k, v["weights"], v["sha256"]] for k, v in out["fingerprints"].items()]),
        "",
        "## Sampling",
        "",
        common.md_table(
            ["sample", "weights", "responses", "tokens", "seconds", "tok/s", "accept_len_mean", "accept_rate", "num_offloads", "peak_mem_gb"],
            [[name, s["weights"], len(s["finish_reasons"]), s["num_tokens"], s["seconds"], s["num_tokens"] / max(s["seconds"], 1e-9),
              s["engine_stats"].get("accept_len_mean"), s["engine_stats"].get("accept_rate"), s["engine_stats"].get("num_offloads"),
              s["engine_stats"].get("peak_mem_gb")] for name, s in out["samples"].items()],
        ),  # fmt: skip
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=common.DEFAULT_MODEL, help="BASE model the engine is built from")
    parser.add_argument("--ckpt", default=None, help=f"HF checkpoint dir to reload (default: {common.DEFAULT_STEP25_CKPT})")
    parser.add_argument("--synthetic", action="store_true", help="create (or reuse) a synthetic checkpoint if --ckpt is absent")
    parser.add_argument("--rel-noise", type=float, default=synth.DEFAULT_REL_NOISE, help="--synthetic: noise / tensor RMS")
    parser.add_argument("--synthetic-seed", type=int, default=0)
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--n", type=int, default=4, help="responses per prompt (prompts are repeated in a row, as verl does)")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--fingerprint-prompts", type=int, default=4)
    parser.add_argument("--fingerprint-tokens", type=int, default=64)
    parser.add_argument("--trainer-dtype", choices=sorted(common.DTYPES), default="bfloat16")
    parser.add_argument("--score-token-budget", type=int, default=8192, help="padded tokens per trainer forward batch")
    parser.add_argument(
        "--sleep-while-scoring",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="engine.sleep() while the trainer forward runs; reloads 2 and 3 then happen asleep (training order)",
    )
    thresholds = parser.add_argument_group("verdict thresholds")
    thresholds.add_argument(
        "--max-mean-abs-dlogp", type=float, default=0.1,
        help="upper bound on mean |delta log p| of 'reloaded' and 'base_after_roundtrip' (default %(default)s)",
    )  # fmt: skip
    thresholds.add_argument(
        "--same-range-factor", type=float, default=2.0,
        help="'reloaded' and 'base_after_roundtrip' means must be within this factor of each other (default %(default)s)",
    )  # fmt: skip
    thresholds.add_argument(
        "--neg-control-factor", type=float, default=3.0,
        help="negative-control mean must exceed this x max(reloaded mean, noise floor) (default %(default)s)",
    )  # fmt: skip
    thresholds.add_argument(
        "--neg-control", choices=["mean", "tracking"], default="mean",
        help="what decides the negative control: 'mean' = the mean |delta log p| rule (default); 'tracking' = the "
        "tracking coefficients, for a checkpoint too close to the base model for the mean rule to have power",
    )  # fmt: skip
    thresholds.add_argument(
        "--tracking-max-se", type=float, default=0.1,
        help="tracking coefficients with a larger standard error are not evaluated (default %(default)s)",
    )  # fmt: skip
    thresholds.add_argument(
        "--noise-floor", type=float, default=1e-4,
        help="means below this are treated as equal by the two ratio rules (default %(default)s)",
    )  # fmt: skip
    parser.add_argument("--vllm-reference", default=None, help="JSON written by gate_c_vllm_reference.py, shown side by side")
    parser.add_argument("--prompts-file", default=None, help="JSON list of token-id lists (instead of MATH prompts)")
    parser.add_argument("--data", default=None, help="train parquet (default: $DATA_DIR/math_l3to5_train.parquet)")
    parser.add_argument("--prompt-seed", type=int, default=0)
    parser.add_argument("--eos-token-ids", default=None, help="comma-separated override (default: from the base model)")
    parser.add_argument("--save-samples", choices=["never", "on-fail", "always"], default="on-fail")
    parser.add_argument("--out-dir", default=common.DEFAULT_OUT_DIR)
    parser.add_argument("--tag", default=None, help="suffix of the output files (default: the engine name)")
    common.add_engine_args(parser)
    common.add_preflight_args(parser)
    raw_argv = [str(a) for a in (sys.argv[1:] if argv is None else argv)]  # for the rerun command
    args = parser.parse_args(argv)
    tag = args.tag or args.engine
    if args.temperature <= 0:
        raise SystemExit("--temperature must be > 0 (the gate compares log-probs of sampled tokens)")

    base_dir = common.resolve_model_dir(args.model)
    preflight = common.run_preflight(args, engine_share=args.mem_fraction)
    ckpt_dir, ckpt_info = resolve_checkpoint(args, base_dir)
    eos = [int(t) for t in args.eos_token_ids.split(",")] if args.eos_token_ids else common.eos_ids_from_checkpoint(base_dir)
    if not eos:
        raise SystemExit("no EOS token ids in the base model; pass --eos-token-ids")
    prompts, prompt_meta = common.resolve_prompts(
        prompts_file=args.prompts_file, model_dir=base_dir, n=args.num_prompts, data_path=args.data, seed=args.prompt_seed
    )
    sample_prompts = common.repeat_interleave(prompts, args.n)
    fp_prompts = prompts[: args.fingerprint_prompts]
    common.log(
        f"[gate 3c] base {base_dir}\n[gate 3c] checkpoint {ckpt_dir} ({ckpt_info['kind']})\n"
        f"[gate 3c] {len(prompts)} prompts x n={args.n} ({prompt_meta['source']}), prompt lengths "
        f"{min(map(len, prompts))}-{max(map(len, prompts))}, eos={eos}"
    )

    checks = common.Checks()
    base_state = common.read_state_dict(base_dir)
    ckpt_state = common.read_state_dict(ckpt_dir)
    weights_diff = common.bf16_diff_report(base_state, ckpt_state)
    checks.add(
        "checkpoint_differs_from_base",
        weights_diff["num_changed"] > 0 and not weights_diff["shape_mismatch"],
        f"{weights_diff['num_changed']:,} of {weights_diff['num_elements']:,} bf16 weights differ "
        f"({100 * weights_diff['fraction_changed']:.2f}%)",
    )

    hf_models: dict = {}

    def trainer_logprobs(which: str, sequences: dict, label: str) -> list[list[float]]:
        if which not in hf_models:
            path = base_dir if which == "base" else ckpt_dir
            common.log(f"[gate 3c] loading the HF trainer model ({which}, {args.trainer_dtype}) from {path}")
            hf_models[which] = common.load_hf_model(path, args.device, args.trainer_dtype)
        return common.hf_trainer_logprobs(
            hf_models[which],
            sample_prompts,
            sequences["responses"],
            temperature=args.temperature,
            token_budget=args.score_token_budget,
            pad_token_id=eos[-1],
            progress=f"gate 3c {label}",
        )

    def reload(state: dict, name: str, label: str) -> dict:
        report = engine.load_weights(state)
        common.log(
            f"[gate 3c] {label}: load_weights({name}) wrote {report.num_params} tensors, "
            f"{report.num_bytes / 2**20:.0f} MiB in {report.seconds:.2f}s"
        )
        return {"num_params": report.num_params, "num_bytes": report.num_bytes, "seconds": report.seconds}

    # ---------------- build from the base model ----------------
    common.log(f"[gate 3c] building the {args.engine} engine from the base model on {args.device}")
    engine = common.build_engine(args.engine, base_dir, **common.engine_opts_from_args(args))
    pointers = engine.parameter_pointers()
    verify = engine.verify_weights(base_state)
    checks.add("build_matches_base", verify["ok"], f"{verify['num_params_checked']} tensors, {verify['num_mismatched']} mismatched")
    fingerprints: dict = {}
    f0_hash, f0_tokens = greedy_fingerprint(engine, fp_prompts, args.fingerprint_tokens, eos)
    fingerprints["F0"] = {"weights": "base (as built)", "sha256": f0_hash, "tokens": f0_tokens}
    reloads: dict = {}

    # ---------------- reload 1: checkpoint, engine awake ----------------
    reloads["reload_1"] = reload(ckpt_state, "checkpoint", "reload 1")
    verify = engine.verify_weights(ckpt_state)
    checks.add(
        "reload1_bitwise",
        verify["ok"],
        f"verify_weights(checkpoint): {verify['num_params_checked']} tensors, {verify['num_elements_checked']:,} elements, "
        f"{verify['num_mismatched']} mismatched {verify['mismatched'][:2] if verify['mismatched'] else ''}",
    )
    checks.add("reload1_tied", verify["tied_ok"], f"tied_ok={verify['tied_ok']}, shared storage={verify['tied_shared_storage']}")
    checks.add("reload1_pointers_unchanged", engine.parameter_pointers() == pointers, f"{len(pointers)} parameter pointers")
    verify_other = engine.verify_weights(base_state)
    checks.add(
        "reload1_base_no_longer_matches",
        not verify_other["ok"],
        f"verify_weights(base) reports {verify_other['num_mismatched']} mismatched tensors (must be > 0)",
    )
    f1_hash, f1_tokens = greedy_fingerprint(engine, fp_prompts, args.fingerprint_tokens, eos)
    fingerprints["F1"] = {"weights": "checkpoint (reload 1, awake)", "sha256": f1_hash, "tokens": f1_tokens}
    common.log(f"[gate 3c] greedy fingerprint F0={f0_hash[:12]} (base), F1={f1_hash[:12]} (checkpoint); differ: {f0_hash != f1_hash}")

    sample_1 = sample(engine, sample_prompts, args, eos, "sample_1", checks)
    sample_1["weights"] = "checkpoint"
    if args.sleep_while_scoring:
        engine.sleep()
    sample_1["hf_ckpt"] = trainer_logprobs("ckpt", sample_1, "sample 1 / HF(checkpoint)")
    sample_1["hf_base"] = trainer_logprobs("base", sample_1, "sample 1 / HF(base)")
    stats = {
        "reloaded": common.compare_logprobs(sample_1["logprobs"], sample_1["hf_ckpt"]),
        "negative_control_mirror": common.compare_logprobs(sample_1["logprobs"], sample_1["hf_base"]),
    }
    common.log("[gate 3c] " + stat_line("reloaded", stats["reloaded"]))

    # ---------------- reload 2: back to the base model ----------------
    reloads["reload_2"] = reload(base_state, "base", "reload 2")
    verify = engine.verify_weights(base_state)
    checks.add(
        "reload2_bitwise_base",
        verify["ok"] and verify["tied_ok"],
        f"verify_weights(base): {verify['num_mismatched']} mismatched of {verify['num_params_checked']}, tied_ok={verify['tied_ok']}",
    )
    verify_other = engine.verify_weights(ckpt_state)
    checks.add(
        "reload2_checkpoint_no_longer_matches",
        not verify_other["ok"],
        f"verify_weights(checkpoint) reports {verify_other['num_mismatched']} mismatched tensors (must be > 0)",
    )
    checks.add("reload2_pointers_unchanged", engine.parameter_pointers() == pointers, f"{len(pointers)} parameter pointers")
    engine.wake_up()
    f0b_hash, f0b_tokens = greedy_fingerprint(engine, fp_prompts, args.fingerprint_tokens, eos)
    fingerprints["F0_roundtrip"] = {"weights": "base (reload 2)", "sha256": f0b_hash, "tokens": f0b_tokens}
    checks.add(
        "base_fingerprint_after_roundtrip",
        f0b_tokens == f0_tokens,
        f"greedy tokens with base weights before / after the round trip: {describe_token_diff(f0_tokens, f0b_tokens)}",
    )

    sample_2 = sample(engine, sample_prompts, args, eos, "sample_2", checks)
    sample_2["weights"] = "base (after round trip)"
    if args.sleep_while_scoring:
        engine.sleep()
    sample_2["hf_base"] = trainer_logprobs("base", sample_2, "sample 2 / HF(base)")
    sample_2["hf_ckpt"] = trainer_logprobs("ckpt", sample_2, "sample 2 / HF(checkpoint)")
    stats["base_after_roundtrip"] = common.compare_logprobs(sample_2["logprobs"], sample_2["hf_base"])
    stats["negative_control"] = common.compare_logprobs(sample_2["logprobs"], sample_2["hf_ckpt"])
    stats["hf_base_vs_ckpt"] = common.compare_logprobs(sample_2["hf_base"], sample_2["hf_ckpt"])
    tracking = {
        "sample_1": common.tracking_coefficient(sample_1["logprobs"], sample_1["hf_base"], sample_1["hf_ckpt"]),
        "sample_2": common.tracking_coefficient(sample_2["logprobs"], sample_2["hf_base"], sample_2["hf_ckpt"]),
    }

    # ---------------- reload 3: checkpoint again ----------------
    reloads["reload_3"] = reload(ckpt_state, "checkpoint", "reload 3")
    verify = engine.verify_weights(ckpt_state)
    checks.add(
        "reload3_bitwise",
        verify["ok"] and verify["tied_ok"],
        f"verify_weights(checkpoint): {verify['num_mismatched']} mismatched of {verify['num_params_checked']}, tied_ok={verify['tied_ok']}",
    )
    checks.add("reload3_pointers_unchanged", engine.parameter_pointers() == pointers, f"{len(pointers)} parameter pointers")
    engine.wake_up()
    f2_hash, f2_tokens = greedy_fingerprint(engine, fp_prompts, args.fingerprint_tokens, eos)
    fingerprints["F2"] = {"weights": "checkpoint (reload 3)", "sha256": f2_hash, "tokens": f2_tokens}
    checks.add(
        "fingerprint_F2_equals_F1",
        f2_tokens == f1_tokens,
        f"greedy tokens with checkpoint weights after reload 1 / reload 3: {describe_token_diff(f1_tokens, f2_tokens)} "
        f"(F1={f1_hash[:12]}, F2={f2_hash[:12]})",
    )
    memory_report = engine.memory_report() if hasattr(engine, "memory_report") else {}
    engine.sleep()
    engine.close()
    del engine
    hf_models.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------------- log-prob verdicts ----------------
    reloaded, roundtrip, negative = stats["reloaded"], stats["base_after_roundtrip"], stats["negative_control"]
    floor = args.noise_floor
    for name in ("reloaded", "base_after_roundtrip"):
        s = stats[name]
        checks.add(
            f"{name}_all_finite",
            s["num_nonfinite"] == 0 and s["count"] > 0,
            f"{s['count']} tokens compared, {s['num_nonfinite']} non-finite",
        )
        checks.add(
            f"{name}_mean_small",
            s["count"] > 0 and s["mean"] <= args.max_mean_abs_dlogp,
            f"mean |delta log p| = {common.fmt_num(s['mean'])} (limit --max-mean-abs-dlogp={args.max_mean_abs_dlogp}); "
            f"median {common.fmt_num(s['median'])}, p99 {common.fmt_num(s['p99'])}, max {common.fmt_num(s['max'])}",
        )
    low, high = sorted([max(reloaded["mean"], floor), max(roundtrip["mean"], floor)]) if reloaded["count"] and roundtrip["count"] else (0.0, float("inf"))  # fmt: skip
    checks.add(
        "base_after_roundtrip_same_range",
        high <= args.same_range_factor * low,
        f"reloaded mean {common.fmt_num(reloaded['mean'])} vs base_after_roundtrip mean {common.fmt_num(roundtrip['mean'])}: "
        f"ratio {common.fmt_num(high / low if low else float('inf'))} (limit --same-range-factor={args.same_range_factor}, "
        f"noise floor {floor:g})",
    )
    # which model do the engine's log-probs follow? (0 = base, 1 = checkpoint)
    track_1, track_2 = tracking["sample_1"], tracking["sample_2"]
    tracking_evaluable = all(
        math.isfinite(t["beta"]) and math.isfinite(t["se"]) and t["se"] <= args.tracking_max_se for t in (track_1, track_2)
    )
    if tracking_evaluable:
        checks.add(
            "tracking_follows_checkpoint_after_reload1",
            track_1["beta"] > 0.5,
            f"tracking coefficient {common.fmt_num(track_1['beta'])} +/- {common.fmt_num(track_1['se'])} "
            "(1 = the engine's log-probs follow the checkpoint, 0 = the base model; must be > 0.5)",
        )
        checks.add(
            "tracking_follows_base_after_reload2",
            track_2["beta"] < 0.5,
            f"tracking coefficient {common.fmt_num(track_2['beta'])} +/- {common.fmt_num(track_2['se'])} (must be < 0.5)",
        )
    else:
        detail = (
            f"standard errors {common.fmt_num(track_1['se'])} / {common.fmt_num(track_2['se'])} exceed "
            f"--tracking-max-se={args.tracking_max_se} (or are undefined): HF(base) and HF(checkpoint) cannot be told "
            "apart on these samples"
        )
        if args.neg_control == "tracking":
            checks.add("tracking_evaluable", False, detail)
        else:
            common.log(f"[gate 3c] NOTE tracking coefficients not evaluated: {detail}")

    needed = args.neg_control_factor * max(reloaded["mean"], floor) if reloaded["count"] else float("inf")
    distance = stats["hf_base_vs_ckpt"]["mean"]
    mean_rule_ok = negative["count"] > 0 and negative["mean"] > needed
    mean_rule_detail = (
        f"engine(base) vs HF(checkpoint) mean {common.fmt_num(negative['mean'])} must exceed "
        f"--neg-control-factor={args.neg_control_factor} x max(reloaded mean {common.fmt_num(reloaded['mean'])}, "
        f"noise floor {floor:g}) = {common.fmt_num(needed)}; for scale, HF(base) vs HF(checkpoint) on the same "
        f"sequences: mean {common.fmt_num(distance)}, bf16 weights changed {100 * weights_diff['fraction_changed']:.2f}%"
    )
    if not mean_rule_ok and distance <= needed:
        mean_rule_detail += (
            ". The two trainer models themselves differ by no more than the required amount: this checkpoint is too "
            "close to the base model for the mean rule to be satisfiable by ANY engine. Use --synthetic for a "
            "decisive mean-based control, or --neg-control tracking"
        )
    if args.neg_control == "mean":
        checks.add("negative_control_clearly_larger", mean_rule_ok, mean_rule_detail)
    else:
        common.log(
            f"[gate 3c] NOTE mean-based negative control {'holds' if mean_rule_ok else 'does NOT hold'} "
            f"(reported only, --neg-control tracking): {mean_rule_detail}"
        )
    negative_control = {"mode": args.neg_control, "mean_rule_ok": mean_rule_ok, "mean_rule_detail": mean_rule_detail,
                        "tracking_evaluable": tracking_evaluable}  # fmt: skip

    # ---------------- stock-vLLM numbers side by side ----------------
    vllm_ref = None
    if args.vllm_reference:
        with open(os.path.expanduser(args.vllm_reference)) as f:
            ref = json.load(f)
        ref_stats = ref["stats"]["engine_vs_trainer"]
        ref_ckpt = ref.get("config", {}).get("ckpt")
        vllm_ref = {
            "path": os.path.abspath(os.path.expanduser(args.vllm_reference)),
            "ckpt": ref_ckpt,
            "same_checkpoint": bool(ref_ckpt) and os.path.realpath(ref_ckpt) == os.path.realpath(ckpt_dir),
            "num_sequences": ref.get("config", {}).get("num_sequences"),
            "stats": ref_stats,
            "mean_ratio": (reloaded["mean"] / ref_stats["mean"]) if ref_stats.get("mean") else None,
        }

    passed = checks.ok
    failed_checks = [item["name"] for item in checks.failed]
    failure_class, failure_reason = classify_failure(
        checks.items, neg_control_mode=args.neg_control, mean_rule_ok=mean_rule_ok, distance=distance, needed=needed
    )
    failure_explanation, rerun = None, None
    if failure_class == FAILURE_UNDERPOWERED:
        rerun = rerun_commands(raw_argv, args, ckpt_dir, tag)
        failure_explanation = underpowered_explanation(
            stats=stats, weights_diff=weights_diff, needed=needed, distance=distance, args=args, tracking=tracking,
            tracking_evaluable=tracking_evaluable, rerun=rerun,
        )  # fmt: skip
    samples_out = {}
    for name, s in (("sample_1", sample_1), ("sample_2", sample_2)):
        samples_out[name] = {k: s[k] for k in ("weights", "finish_reasons", "engine_stats", "seconds", "num_tokens")}
    output = {
        "gate": "3c_reload",
        "pass": passed,
        "failure_class": failure_class,
        "failure_reason": failure_reason,
        "failed_checks": failed_checks,
        "failure_explanation": failure_explanation,
        "rerun": rerun,
        "config": {
            "engine": args.engine,
            "model": base_dir,
            "device": args.device,
            "num_prompts": len(prompts),
            "n": args.n,
            "num_sequences": len(sample_prompts),
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "trainer_dtype": args.trainer_dtype,
            "score_token_budget": args.score_token_budget,
            "sleep_while_scoring": args.sleep_while_scoring,
            "fingerprint_prompts": len(fp_prompts),
            "fingerprint_tokens": args.fingerprint_tokens,
            "max_mean_abs_dlogp": args.max_mean_abs_dlogp,
            "same_range_factor": args.same_range_factor,
            "neg_control_factor": args.neg_control_factor,
            "noise_floor": args.noise_floor,
            "neg_control": args.neg_control,
            "tracking_max_se": args.tracking_max_se,
            "eos_token_ids": eos,
            "engine_options": common.engine_opts_from_args(args),
        },
        "checkpoint": ckpt_info,
        "checkpoint_vs_base": weights_diff,
        "prompts": prompt_meta,
        "checks": checks.items,
        "checks_markdown": checks.markdown(),
        "stats": stats,
        "tracking": tracking,
        "negative_control": negative_control,
        "reloads": reloads,
        "fingerprints": fingerprints,
        "samples": samples_out,
        "vllm_reference": vllm_ref,
        "engine_memory_report": memory_report,
        "preflight": preflight,
        "environment": common.environment_info(args.device),
    }
    json_path = common.write_json(os.path.join(args.out_dir, f"gate_c_reload_{tag}.json"), output)
    md_path = common.write_text(os.path.join(args.out_dir, f"gate_c_reload_{tag}.md"), build_markdown(output))
    samples_path = None
    if args.save_samples == "always" or (args.save_samples == "on-fail" and not passed):
        samples_path = common.write_json(
            os.path.join(args.out_dir, f"gate_c_reload_{tag}.samples.json"),
            {
                "prompts": sample_prompts,
                "sample_1": {k: sample_1[k] for k in ("weights", "responses", "logprobs", "hf_ckpt", "hf_base")},
                "sample_2": {k: sample_2[k] for k in ("weights", "responses", "logprobs", "hf_ckpt", "hf_base")},
            },
        )

    common.log("")
    common.log("[gate 3c] |delta log p| per response token, engine vs trainer forward:")
    for name in ("reloaded", "base_after_roundtrip", "negative_control", "negative_control_mirror", "hf_base_vs_ckpt"):
        common.log("[gate 3c]   " + stat_line(name, stats[name]))
    common.log(
        f"[gate 3c] checkpoint vs base: {100 * weights_diff['fraction_changed']:.2f}% of bf16 weights differ; "
        f"HF(base) vs HF(checkpoint) mean |delta log p| = {common.fmt_num(stats['hf_base_vs_ckpt']['mean'])}"
    )
    common.log(
        f"[gate 3c] tracking coefficient (0 = base, 1 = checkpoint): after reload 1 "
        f"{common.fmt_num(tracking['sample_1']['beta'])} +/- {common.fmt_num(tracking['sample_1']['se'])}, after reload 2 "
        f"{common.fmt_num(tracking['sample_2']['beta'])} +/- {common.fmt_num(tracking['sample_2']['se'])}"
    )
    if vllm_ref:
        common.log("[gate 3c] side by side with stock vLLM (" + vllm_ref["path"] + "):")
        common.log("[gate 3c]   " + stat_line(f"{args.engine}: reloaded", reloaded))
        common.log("[gate 3c]   " + stat_line("stock vLLM", vllm_ref["stats"]))
        common.log(f"[gate 3c]   ratio of means {args.engine} / vLLM: {common.fmt_num(vllm_ref['mean_ratio'])}")
        if not vllm_ref["same_checkpoint"]:
            common.log(f"[gate 3c]   NOTE: the vLLM reference was measured on {vllm_ref['ckpt']}, this run on {ckpt_dir}")
    common.log(f"[gate 3c] wrote {json_path}")
    common.log(f"[gate 3c] wrote {md_path}")
    if samples_path:
        common.log(f"[gate 3c] wrote {samples_path}")
    if passed:
        common.log(f"[gate 3c] PASS ({len(checks.items)} checks)")
        return 0
    for item in checks.failed:
        common.log(f"[gate 3c] FAIL {item['name']}: {item['detail']}")
    if failure_class == FAILURE_UNDERPOWERED:
        common.log("[gate 3c]")
        common.log("[gate 3c] FAILURE CLASS negative_control_underpowered: the checkpoint is too close to the base model")
        for line in textwrap.wrap(failure_explanation, 110, break_long_words=False, break_on_hyphens=False):
            common.log(f"[gate 3c]   {line}")
        if rerun["commands"]:
            common.log(f"[gate 3c] rerun on the step-{rerun['next_step']} checkpoint (repo root, PYTHONPATH=$PWD):")
            for command in rerun["commands"]:
                common.log(f"[gate 3c]   # env {command['env']}")
                common.log(f"[gate 3c]   {command['command']}")
        else:
            common.log(f"[gate 3c] rerun: {rerun['hint']}")
    common.log(
        f"[gate 3c] FAIL ({len(checks.failed)} of {len(checks.items)} checks failed; failure class: {failure_class})"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
