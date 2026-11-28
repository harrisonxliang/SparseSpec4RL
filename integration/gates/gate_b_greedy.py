"""Gate 3b: greedy decoding, rollout engine vs HuggingFace transformers on the same weights.

    python -m integration.gates.gate_b_greedy --engine sparsespec          # the real gate (A100)
    python -m integration.gates.gate_b_greedy --engine reference           # plumbing check
    python -m integration.gates.gate_b_greedy --engine reference --device cpu \
        --model /tmp/tiny-qwen3 --prompts-file /tmp/tiny-qwen3/prompts.json --max-new-tokens 32   # CPU smoke

What it does
  1. ``--num-prompts`` (16) MATH training prompts, built as verl builds them.
  2. The engine generates all of them in ONE call at temperature 0,
     ``--max-new-tokens`` (256) each.
  3. HuggingFace generates each prompt on its own (bfloat16, batch of one, no
     padding) with a plain argmax loop.
  4. Per prompt: the first position where the two token sequences differ
     (-1 = identical). At that position the shared prefix is teacher-forced
     through HF and the HF logit MARGIN between HF's top token and the
     engine's token is reported. bf16 arithmetic differs between the two
     implementations, so a flip at a near-tie is expected now and then; a
     flip against a large margin means the engine computed something else.

Verdict (thresholds are flags, printed with the result)
  PASS iff (a) the engine's responses obey the response contract,
           (b) neither response is a strict prefix of the other,
           (c) every divergence has HF margin <= ``--tie-margin`` logits,
           (d) at least ``--min-identical`` prompts match exactly (default 0),
           (e) only with ``--max-divergence-ratio R``: the engine-vs-HF divergence
               rate is at most R x the HF-vs-HF flip rate (see below).
  As a yardstick for (c) the report also counts HF-vs-HF flips: positions
  where HF's own argmax changes between its cached step-by-step forward and
  one uncached forward over the same tokens, with their margins.

Outputs: ``<out-dir>/gate_b_greedy_<tag>.json`` and ``.md``; exit status 1 on FAIL.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

from integration.gates import common  # noqa: E402


def bf16_ulp(value: float) -> float:
    """Spacing of bfloat16 values around ``value`` (8 significant bits)."""
    value = abs(float(value))
    if value == 0.0 or not math.isfinite(value):
        return 0.0
    return 2.0 ** (math.floor(math.log2(value)) - 7)


def analyse_prompt(model, prompt, engine_tokens, hf_tokens, tie_margin: float) -> dict:
    """Compare one engine response with HF's and measure the margin at the first difference."""
    first_diff = common.first_difference(engine_tokens, hf_tokens)
    record = {
        "prompt_len": len(prompt),
        "engine_len": len(engine_tokens),
        "hf_len": len(hf_tokens),
        "first_diff": first_diff,
        "status": "identical",
        "hf_token": None,
        "engine_token": None,
        "tf_top_token": None,
        "margin": None,
        "margin_bf16_ulps": None,
        "engine_token_rank": None,
        "p_top": None,
        "p_engine_token": None,
    }

    # One uncached forward over prompt + HF response: row j is HF's
    # distribution after the prefix both sequences share up to j.
    logits = common.hf_teacher_forced_logits(model, prompt, hf_tokens)
    hf_ids = torch.tensor(hf_tokens, dtype=torch.long, device=logits.device)
    top_values, top_ids = logits.max(dim=-1)
    own = logits.gather(-1, hf_ids.unsqueeze(-1)).squeeze(-1)
    self_flip = top_ids != hf_ids
    record["hf_self_flips"] = int(self_flip.sum().item())
    record["hf_self_flip_max_margin"] = float((top_values - own)[self_flip].max().item()) if bool(self_flip.any()) else 0.0

    if first_diff < 0:
        return record
    if first_diff >= min(len(engine_tokens), len(hf_tokens)):
        # one response is a strict prefix of the other: with identical tokens up
        # to there, one side stopped without EOS or continued past one
        record["status"] = "LENGTH_MISMATCH"
        return record

    row = logits[first_diff]
    engine_token = int(engine_tokens[first_diff])
    top_token = int(top_ids[first_diff].item())
    margin = float((row[top_token] - row[engine_token]).item())
    probs = torch.softmax(row, dim=-1)
    ulp = bf16_ulp(float(row[top_token].item()))
    record.update(
        hf_token=int(hf_tokens[first_diff]),
        engine_token=engine_token,
        tf_top_token=top_token,
        margin=margin,
        margin_bf16_ulps=(margin / ulp) if ulp > 0 else None,
        engine_token_rank=int((row > row[engine_token]).sum().item()),
        p_top=float(probs[top_token].item()),
        p_engine_token=float(probs[engine_token].item()),
        status="near_tie" if margin <= tie_margin else "MISMATCH",
    )
    return record


def build_markdown(result: dict) -> str:
    cfg, summary = result["config"], result["summary"]
    lines = [
        f"# Gate 3b - greedy decoding: {cfg['engine']} engine vs HuggingFace",
        "",
        f"**Verdict: {'PASS' if result['pass'] else 'FAIL'}**",
        "",
        f"- model: `{cfg['model']}` (HF reference: `{cfg['hf_model']}`, {cfg['hf_dtype']})",
        f"- prompts: {cfg['num_prompts']} ({result['prompts']['source']}), max_new_tokens={cfg['max_new_tokens']}, "
        f"temperature=0, device={cfg['device']}",
        f"- rule: PASS iff the engine obeys the response contract, no response is a strict prefix of the other, "
        f"every divergence has HF margin <= --tie-margin = {cfg['tie_margin']} logits, and at least "
        f"--min-identical = {cfg['min_identical']} prompts are identical"
        + (
            f", and the divergence rate is at most --max-divergence-ratio = {cfg['max_divergence_ratio']} x the HF-vs-HF flip rate"
            if cfg["max_divergence_ratio"] is not None
            else ""
        ),
        "",
        "## Summary",
        "",
        common.md_table(
            ["identical", "near-tie divergence", "MISMATCH (margin > tie-margin)", "LENGTH_MISMATCH", "max margin", "median first diff (diverging prompts)"],
            [[summary["identical"], summary["near_tie"], summary["mismatch"], summary["length_mismatch"],
              summary["max_margin"], summary["median_first_diff"]]],
        ),  # fmt: skip
        "",
        f"Yardstick (not part of the verdict): HF vs HF (cached step-by-step vs one uncached forward) flips its own "
        f"argmax at {summary['hf_self_flips']} of {summary['hf_tokens']} positions "
        f"({100 * summary['hf_self_flip_rate']:.2f}%), largest margin "
        f"{common.fmt_num(summary['hf_self_flip_max_margin'])} logits. Engine vs HF: "
        f"{summary['near_tie'] + summary['mismatch']} divergence(s) in {summary['positions_compared']} positions with a "
        f"shared prefix ({100 * summary['divergence_rate']:.2f}%). A divergence between two slightly different models "
        "also tends to happen first where the margin is small, so a divergence RATE far above HF's own, or margins "
        "far above the HF-vs-HF ones, deserve a look even when every margin is under the limit.",
        "",
        "## Checks",
        "",
        result["checks_markdown"],
        "",
        "## Per prompt",
        "",
        "`first diff` = first response position where the token ids differ (-1: none). `margin` = HF logit of "
        "HF's top token minus HF logit of the engine's token, teacher-forcing the shared prefix; `ulps` = the "
        "same in units of the bf16 spacing at the top logit; `rank` = number of tokens HF prefers to the engine's.",
        "",
        common.md_table(
            ["prompt", "prompt len", "engine len", "HF len", "first diff", "status", "HF token", "engine token",
             "margin", "ulps", "rank", "p(HF top)", "p(engine token)"],
            [[i, r["prompt_len"], r["engine_len"], r["hf_len"], r["first_diff"], r["status"], r["hf_token"],
              r["engine_token"], r["margin"], r["margin_bf16_ulps"], r["engine_token_rank"], r["p_top"],
              r["p_engine_token"]] for i, r in enumerate(result["per_prompt"])],
        ),  # fmt: skip
        "",
        "## Engine statistics",
        "",
        common.md_table(["stat", "value"], [[k, v] for k, v in result["engine_stats"].items()]),
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=common.DEFAULT_MODEL, help="weights for the engine: HF directory or hub id")
    parser.add_argument(
        "--hf-model", default=None, help="weights for the HF reference (default: --model; differ only to self-test the gate)"
    )
    parser.add_argument("--hf-dtype", choices=sorted(common.DTYPES), default="bfloat16")
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--tie-margin",
        type=float,
        default=1.0,
        help="largest HF logit margin at a divergence that still counts as a bf16 near-tie (default %(default)s)",
    )
    parser.add_argument("--min-identical", type=int, default=0, help="prompts that must match exactly (default 0)")
    parser.add_argument(
        "--max-divergence-ratio",
        type=float,
        default=None,
        help="optional extra rule: FAIL if the engine-vs-HF divergence rate exceeds this x the HF-vs-HF flip rate "
        "(floored at one flip over all HF positions); off by default",
    )
    parser.add_argument("--prompts-file", default=None, help="JSON list of token-id lists (instead of MATH prompts)")
    parser.add_argument("--data", default=None, help="train parquet (default: $DATA_DIR/math_l3to5_train.parquet)")
    parser.add_argument("--prompt-seed", type=int, default=0)
    parser.add_argument("--eos-token-ids", default=None, help="comma-separated override (default: from the checkpoint)")
    parser.add_argument("--out-dir", default=common.DEFAULT_OUT_DIR)
    parser.add_argument("--tag", default=None, help="suffix of the output files (default: the engine name)")
    common.add_engine_args(parser)
    common.add_preflight_args(parser)
    args = parser.parse_args(argv)
    tag = args.tag or args.engine

    preflight = common.run_preflight(args, engine_share=args.mem_fraction)
    model_dir = common.resolve_model_dir(args.model)
    hf_dir = common.resolve_model_dir(args.hf_model) if args.hf_model else model_dir
    eos = (
        [int(t) for t in args.eos_token_ids.split(",")] if args.eos_token_ids else common.eos_ids_from_checkpoint(model_dir)
    )
    if not eos:
        raise SystemExit("no EOS token ids in the checkpoint; pass --eos-token-ids")
    prompts, prompt_meta = common.resolve_prompts(
        prompts_file=args.prompts_file, model_dir=model_dir, n=args.num_prompts, data_path=args.data, seed=args.prompt_seed
    )
    common.log(
        f"[gate 3b] {len(prompts)} prompts ({prompt_meta['source']}), lengths {min(map(len, prompts))}-"
        f"{max(map(len, prompts))} tokens, eos={eos}"
    )

    checks = common.Checks()

    # ---- engine: all prompts in one call, temperature 0 ----
    common.log(f"[gate 3b] building the {args.engine} engine from {model_dir} on {args.device}")
    engine = common.build_engine(args.engine, model_dir, **common.engine_opts_from_args(args))
    start = time.perf_counter()
    result = engine.generate(prompts, max_new_tokens=args.max_new_tokens, temperature=0.0, eos_token_ids=eos)
    engine_seconds = time.perf_counter() - start
    engine_tokens = [list(t) for t in result.token_ids]
    engine_stats = dict(result.stats)
    problems = common.check_responses(result, args.max_new_tokens, eos, greedy=True)
    checks.add(
        "engine_response_contract",
        not problems,
        "all responses end with their first EOS or have exactly max_new_tokens tokens; greedy log-probs are 0.0"
        if not problems
        else f"{len(problems)} violation(s), e.g. {problems[:3]}",
    )
    engine.sleep()  # release the rollout-only GPU memory before the HF model runs
    engine.close()
    del engine, result
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- HF: one prompt at a time, no padding ----
    common.log(f"[gate 3b] HF greedy ({args.hf_dtype}) from {hf_dir}")
    model = common.load_hf_model(hf_dir, args.device, args.hf_dtype)
    per_prompt = []
    start = time.perf_counter()
    for i, prompt in enumerate(prompts):
        hf_tokens = common.hf_greedy(model, prompt, args.max_new_tokens, eos)
        record = analyse_prompt(model, prompt, engine_tokens[i], hf_tokens, args.tie_margin)
        record["engine_tokens"] = engine_tokens[i]
        record["hf_tokens"] = hf_tokens
        per_prompt.append(record)
        where = "identical" if record["first_diff"] < 0 else f"first diff at {record['first_diff']}"
        extra = "" if record["margin"] is None else f", HF margin {record['margin']:.4f} logits"
        common.log(
            f"[gate 3b] prompt {i:2d}: engine {record['engine_len']:4d} tok, HF {record['hf_len']:4d} tok, "
            f"{where}{extra} -> {record['status']}"
        )
    hf_seconds = time.perf_counter() - start
    del model

    # ---- summary and verdict ----
    def count(status: str) -> int:
        return sum(1 for r in per_prompt if r["status"] == status)

    margins = [r["margin"] for r in per_prompt if r["margin"] is not None]
    diverged = sorted(r["first_diff"] for r in per_prompt if r["first_diff"] >= 0)
    summary = {
        "num_prompts": len(per_prompt),
        "identical": count("identical"),
        "near_tie": count("near_tie"),
        "mismatch": count("MISMATCH"),
        "length_mismatch": count("LENGTH_MISMATCH"),
        "max_margin": max(margins) if margins else None,
        "margins_sorted": sorted(margins, reverse=True),
        "first_diffs": [r["first_diff"] for r in per_prompt],
        "median_first_diff": diverged[len(diverged) // 2] if diverged else None,
        "hf_tokens": sum(r["hf_len"] for r in per_prompt),
        "hf_self_flips": sum(r["hf_self_flips"] for r in per_prompt),
        "hf_self_flip_max_margin": max(r["hf_self_flip_max_margin"] for r in per_prompt),
        # positions where both saw the same prefix: everything up to and including the first difference
        "positions_compared": sum(
            (r["first_diff"] + 1) if 0 <= r["first_diff"] < min(r["engine_len"], r["hf_len"]) else min(r["engine_len"], r["hf_len"])
            for r in per_prompt
        ),
        "engine_seconds": engine_seconds,
        "hf_seconds": hf_seconds,
    }
    summary["divergence_rate"] = (summary["near_tie"] + summary["mismatch"]) / max(summary["positions_compared"], 1)
    summary["hf_self_flip_rate"] = summary["hf_self_flips"] / max(summary["hf_tokens"], 1)
    checks.add(
        "no_length_mismatch",
        summary["length_mismatch"] == 0,
        f"{summary['length_mismatch']} prompt(s) where one response is a strict prefix of the other",
    )
    checks.add(
        "divergences_are_near_ties",
        summary["mismatch"] == 0,
        f"{summary['mismatch']} divergence(s) with HF margin > --tie-margin={args.tie_margin} logits; "
        f"{summary['near_tie']} near-tie divergence(s), largest margin {common.fmt_num(summary['max_margin'])}; "
        f"{summary['identical']}/{summary['num_prompts']} identical",
    )
    checks.add(
        "min_identical",
        summary["identical"] >= args.min_identical,
        f"{summary['identical']} identical prompt(s), --min-identical={args.min_identical}",
    )
    if args.max_divergence_ratio is not None:
        floor_rate = max(summary["hf_self_flip_rate"], 1.0 / max(summary["hf_tokens"], 1))
        limit = args.max_divergence_ratio * floor_rate
        checks.add(
            "divergence_rate",
            summary["divergence_rate"] <= limit,
            f"engine vs HF diverges at {100 * summary['divergence_rate']:.3f}% of positions; limit "
            f"--max-divergence-ratio={args.max_divergence_ratio} x {100 * floor_rate:.3f}% (HF vs HF) = {100 * limit:.3f}%",
        )

    passed = checks.ok
    output = {
        "gate": "3b_greedy",
        "pass": passed,
        "config": {
            "engine": args.engine,
            "model": model_dir,
            "hf_model": hf_dir,
            "hf_dtype": args.hf_dtype,
            "device": args.device,
            "num_prompts": len(prompts),
            "max_new_tokens": args.max_new_tokens,
            "tie_margin": args.tie_margin,
            "min_identical": args.min_identical,
            "max_divergence_ratio": args.max_divergence_ratio,
            "eos_token_ids": eos,
            "engine_options": common.engine_opts_from_args(args),
        },
        "prompts": prompt_meta,
        "summary": summary,
        "checks": checks.items,
        "checks_markdown": checks.markdown(),
        "per_prompt": per_prompt,
        "engine_stats": engine_stats,
        "preflight": preflight,
        "environment": common.environment_info(args.device),
    }
    json_path = common.write_json(os.path.join(args.out_dir, f"gate_b_greedy_{tag}.json"), output)
    md_path = common.write_text(os.path.join(args.out_dir, f"gate_b_greedy_{tag}.md"), build_markdown(output))

    common.log("")
    common.log(
        f"[gate 3b] {summary['identical']}/{summary['num_prompts']} identical, {summary['near_tie']} near-tie, "
        f"{summary['mismatch']} MISMATCH, {summary['length_mismatch']} LENGTH_MISMATCH; "
        f"max HF margin at a divergence: {common.fmt_num(summary['max_margin'])} logits"
    )
    common.log(
        f"[gate 3b] yardstick: HF vs HF (cached vs uncached forward) flips its argmax at {summary['hf_self_flips']} "
        f"of {summary['hf_tokens']} positions ({100 * summary['hf_self_flip_rate']:.2f}%), largest margin "
        f"{common.fmt_num(summary['hf_self_flip_max_margin'])} logits; engine vs HF: "
        f"{summary['near_tie'] + summary['mismatch']} divergence(s) in {summary['positions_compared']} positions with a "
        f"shared prefix ({100 * summary['divergence_rate']:.2f}%). A far higher engine rate, or engine margins far above "
        "the HF-vs-HF ones, deserve a look even when the gate passes (not part of the verdict)."
    )
    common.log(
        f"[gate 3b] rule: contract ok AND no strict-prefix responses AND every divergence margin <= "
        f"--tie-margin={args.tie_margin} AND identical >= --min-identical={args.min_identical}"
        + (
            f" AND divergence rate <= --max-divergence-ratio={args.max_divergence_ratio} x the HF-vs-HF flip rate"
            if args.max_divergence_ratio is not None
            else " (--max-divergence-ratio not set)"
        )
    )
    common.log(f"[gate 3b] wrote {json_path}")
    common.log(f"[gate 3b] wrote {md_path}")
    if passed:
        common.log("[gate 3b] PASS")
        return 0
    for item in checks.failed:
        common.log(f"[gate 3b] FAIL {item['name']}: {item['detail']}")
    common.log("[gate 3b] FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
