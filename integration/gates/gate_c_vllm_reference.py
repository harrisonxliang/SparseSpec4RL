"""Gate 3c reference numbers: the same engine-vs-trainer measurement for STOCK vLLM.

Run in the BASELINE env (stock vLLM 0.9.1; see baseline/setup_env.sh), from the repo root:

    python integration/gates/gate_c_vllm_reference.py \
        --ckpt $CKPT_ROOT/vllm_baseline_seed0/hf/global_step_25

and hand the result to the SparseSpec gate (in the SparseSpec env):

    python -m integration.gates.gate_c_reload --engine sparsespec --ckpt <same checkpoint> \
        --vllm-reference baseline/results/sparsespec_checks/gate_c_vllm_reference.json

Use the SAME checkpoint, prompts (--num-prompts / --n / --prompt-seed / --data)
and --max-new-tokens in both, otherwise the two rows are not comparable.

Output files (under --out-dir, default baseline/results/sparsespec_checks/):
  NAME = gate_c_vllm_reference           without --tag
  NAME = gate_c_vllm_reference_<tag>     with --tag <tag> (letters, digits, '.', '_', '-')
  <NAME>.samples.json   phase 'sample' (vLLM token ids + log-probs; --samples overrides the path)
  <NAME>.json           phase 'score': top-level "pass" (true when every token has a finite
                        log-prob on both sides) and stats.engine_vs_trainer, the input of
                        gate_c_reload --vllm-reference
  <NAME>.md             phase 'score'
e.g. --tag step50 -> gate_c_vllm_reference_step50.json. Exit status: 0 PASS, 1 FAIL
(non-finite or missing log-probs), 3 an unexpected exception here or in the sampling
child (crash: no <NAME>.json is written), 128 + N if the child was killed by signal N;
unusable inputs (a samples file from another checkpoint, mismatched lengths) stop with
an error message and status 1. Phases 'all' and 'score' end with a line starting "[gate 3c vllm reference]"
that contains PASS or FAIL.

Phases
  sample  stock vLLM on the checkpoint (dtype bfloat16), ``SamplingParams(n=1,
          temperature=1.0, top_p=1.0, top_k=-1, max_tokens=2048, logprobs=0,
          detokenize=False)``; token ids and the sampled token's log-prob are
          read exactly as verl's ``vLLMRollout.generate_sequences`` reads them
          (``output.outputs[0].token_ids`` and ``logprobs[i][token].logprob``).
          Writes ``<name>.samples.json``.
  score   the HF trainer forward on the same checkpoint scores the same
          sequences (``common.hf_trainer_logprobs``) and the two sets of
          log-probs are compared. Writes ``<name>.json`` and ``<name>.md``.
  all     (default) runs ``sample`` in a CHILD python process, so that vLLM's
          GPU memory is released before the scoring model is loaded, then
          ``score`` in this process.

Temperature is fixed at 1.0: vLLM's V1 sampler (the default in 0.9.1) reports
log-probs of the RAW logits (``vllm/v1/sample/sampler.py``: "Use the original
logits (before any penalties or temperature scaling)"), which equal the
sampling distribution's only at temperature 1, the training temperature.

This file must stay importable in the baseline env: it uses torch,
transformers, ``integration.gates.common`` and (inside ``common``)
``serve.rl.base`` only; never ``serve.rl.engine``.

NOT smoke-tested with a real vLLM (none on the development machine): the
vLLM calls were written against the 0.9.1 sources (``entrypoints/llm.py``,
``sampling_params.py``, ``outputs.py``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from integration.gates import common  # noqa: E402

TEMPERATURE = 1.0
BASE_NAME = "gate_c_vllm_reference"


def output_name(tag: str | None) -> str:
    """File-name stem of the outputs: ``gate_c_vllm_reference`` or ``gate_c_vllm_reference_<tag>``."""
    return BASE_NAME + (f"_{tag}" if tag else "")


def vllm_generate(args, ckpt_dir: str, prompts: list[list[int]]) -> dict:
    """Sample one response per prompt with stock vLLM; returns token ids, log-probs, finish reasons."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=ckpt_dir,
        tokenizer=args.tokenizer or ckpt_dir,
        dtype="bfloat16",
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max(len(p) for p in prompts) + args.max_new_tokens,
        # the remaining options mirror verl's vLLMRollout with baseline/config/grpo_base.sh
        enforce_eager=True,
        disable_log_stats=True,
        max_num_batched_tokens=10240,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=TEMPERATURE,
        top_p=1.0,
        top_k=-1,
        max_tokens=args.max_new_tokens,
        logprobs=0,  # 0 = only the sampled token's log-prob
        detokenize=False,
    )
    start = time.perf_counter()
    outputs = llm.generate(
        prompts=[{"prompt_token_ids": list(p)} for p in prompts],
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    seconds = time.perf_counter() - start

    responses, logprobs, reasons = [], [], []
    for output in outputs:
        completion = output.outputs[0]
        response_ids = list(completion.token_ids)
        # verl: curr_log_prob.append(logprob[response_ids[i]].logprob)
        row = [float(logprob[response_ids[i]].logprob) for i, logprob in enumerate(completion.logprobs)]
        responses.append([int(t) for t in response_ids])
        logprobs.append(row)
        reasons.append(completion.finish_reason)
    return {"responses": responses, "logprobs": logprobs, "finish_reasons": reasons, "seconds": seconds}


def phase_sample(args, ckpt_dir: str, samples_path: str) -> int:
    prompts, prompt_meta = common.resolve_prompts(
        prompts_file=args.prompts_file,
        model_dir=args.tokenizer or ckpt_dir,
        n=args.num_prompts,
        data_path=args.data,
        seed=args.prompt_seed,
    )
    sample_prompts = common.repeat_interleave(prompts, args.n)
    common.log(
        f"[vllm reference] sampling {len(sample_prompts)} responses ({len(prompts)} prompts x n={args.n}, "
        f"{prompt_meta['source']}), max_new_tokens={args.max_new_tokens}, T={TEMPERATURE}"
    )
    generated = vllm_generate(args, ckpt_dir, sample_prompts)
    num_tokens = sum(len(r) for r in generated["responses"])
    common.log(
        f"[vllm reference] {num_tokens} tokens in {generated['seconds']:.1f}s "
        f"({num_tokens / max(generated['seconds'], 1e-9):.0f} tok/s)"
    )
    import vllm

    common.write_json(
        samples_path,
        {
            "kind": "gate_c_vllm_reference_samples",
            "ckpt": ckpt_dir,
            "vllm_version": getattr(vllm, "__version__", None),
            "temperature": TEMPERATURE,
            "max_new_tokens": args.max_new_tokens,
            "num_prompts": len(prompts),
            "n": args.n,
            "prompts_meta": prompt_meta,
            "prompts": sample_prompts,
            **generated,
        },
    )
    common.log(f"[vllm reference] wrote {samples_path}")
    return 0


def phase_score(args, ckpt_dir: str, samples_path: str, out_base: str) -> int:
    with open(samples_path) as f:
        samples = json.load(f)
    prompts, responses, logprobs = samples["prompts"], samples["responses"], samples["logprobs"]
    if not (len(prompts) == len(responses) == len(logprobs)) or not prompts:
        raise SystemExit(f"{samples_path}: prompts / responses / logprobs have different or zero lengths")
    if samples.get("ckpt") and os.path.realpath(samples["ckpt"]) != os.path.realpath(ckpt_dir):
        raise SystemExit(f"{samples_path} was sampled from {samples['ckpt']}, but --ckpt is {ckpt_dir}")
    eos = common.eos_ids_from_checkpoint(ckpt_dir)
    notes = common.check_responses(
        types.SimpleNamespace(token_ids=responses, logprobs=logprobs, finish_reasons=samples["finish_reasons"]),
        samples["max_new_tokens"],
        eos,
    )
    unusable = [i for i, (r, lp) in enumerate(zip(responses, logprobs)) if len(r) == 0 or len(r) != len(lp)]
    if unusable:
        raise SystemExit(
            f"[vllm reference] FAIL: {len(unusable)} sequence(s) are empty or have a different number of tokens "
            f"and log-probs, e.g. index {unusable[:5]}"
        )
    for note in notes[:5]:
        common.log(f"[vllm reference] NOTE (stock vLLM output vs the engines' response contract): {note}")

    common.run_preflight(args, engine_share=0.0)
    common.log(f"[vllm reference] scoring {len(prompts)} sequences with the HF trainer forward on {args.device}")
    model = common.load_hf_model(ckpt_dir, args.device, args.trainer_dtype)
    trainer = common.hf_trainer_logprobs(
        model,
        prompts,
        responses,
        temperature=samples["temperature"],
        token_budget=args.score_token_budget,
        pad_token_id=eos[-1] if eos else 0,
        progress="vllm reference",
    )
    stats = common.compare_logprobs(logprobs, trainer)
    passed = stats["count"] > 0 and stats["num_nonfinite"] == 0
    output = {
        "kind": "gate_c_vllm_reference",
        "pass": passed,
        "config": {
            "ckpt": ckpt_dir,
            "num_prompts": samples["num_prompts"],
            "n": samples["n"],
            "num_sequences": len(prompts),
            "max_new_tokens": samples["max_new_tokens"],
            "temperature": samples["temperature"],
            "trainer_dtype": args.trainer_dtype,
            "vllm_version": samples.get("vllm_version"),
            "samples": os.path.abspath(samples_path),
        },
        "prompts": samples.get("prompts_meta"),
        "stats": {"engine_vs_trainer": stats},
        "sampling": {
            "num_tokens": sum(len(r) for r in responses),
            "seconds": samples.get("seconds"),
            "stopped_by_eos": sum(r == "stop" for r in samples["finish_reasons"]),
        },
        "contract_notes": notes[:50],
        "environment": common.environment_info(args.device),
    }
    json_path = common.write_json(out_base + ".json", output)
    markdown = "\n".join(
        [
            "# Gate 3c reference - stock vLLM vs the HF trainer forward",
            "",
            f"- checkpoint: `{ckpt_dir}`",
            f"- vLLM {samples.get('vllm_version')}, dtype bfloat16, T={samples['temperature']}, top_p=1, top_k=-1, "
            f"max_tokens={samples['max_new_tokens']}",
            f"- {samples['num_prompts']} prompts x n={samples['n']} = {len(prompts)} sequences, "
            f"{output['sampling']['num_tokens']} response tokens "
            f"({(samples.get('prompts_meta') or {}).get('source')})",
            "",
            common.stats_table([("stock vLLM (sampled-token log-prob) vs HF trainer forward", stats)]),
            "",
            "`max_prob_ratio` = exp(max |delta log p|). Compare with the `reloaded` row of gate_c_reload.",
            "",
        ]
    )
    md_path = common.write_text(out_base + ".md", markdown)
    common.log(
        f"[vllm reference] |delta log p| vLLM vs trainer: n={stats['count']} mean={common.fmt_num(stats['mean'])} "
        f"median={common.fmt_num(stats['median'])} p99={common.fmt_num(stats['p99'])} max={common.fmt_num(stats['max'])} "
        f"max_prob_ratio={common.fmt_num(stats['max_prob_ratio'])} (non-finite: {stats['num_nonfinite']})"
    )
    common.log(f"[vllm reference] wrote {json_path}")
    common.log(f"[vllm reference] wrote {md_path}")
    if not passed:
        common.log("[vllm reference] FAIL: non-finite or missing log-probs")
        common.log(f"[gate 3c vllm reference] FAIL: {stats['num_nonfinite']} non-finite, {stats['count']} compared ({json_path})")
        return 1
    common.log(f"[gate 3c vllm reference] PASS: {stats['count']} tokens compared, mean |delta log p| {common.fmt_num(stats['mean'])} ({json_path})")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=["all", "sample", "score"], default="all")
    parser.add_argument("--ckpt", default=common.DEFAULT_STEP25_CKPT, help="HF checkpoint dir (weights + tokenizer)")
    parser.add_argument("--tokenizer", default=None, help="tokenizer dir (default: --ckpt)")
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45, help="vLLM's GPU share (as in training)")
    parser.add_argument("--seed", type=int, default=0, help="vLLM engine seed")
    parser.add_argument("--prompts-file", default=None, help="JSON list of token-id lists (instead of MATH prompts)")
    parser.add_argument("--data", default=None, help="train parquet (default: $DATA_DIR/math_l3to5_train.parquet)")
    parser.add_argument("--prompt-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0", help="device of the HF scoring model")
    parser.add_argument("--trainer-dtype", choices=sorted(common.DTYPES), default="bfloat16")
    parser.add_argument("--score-token-budget", type=int, default=8192)
    parser.add_argument("--samples", default=None, help="samples JSON (default: <out-dir>/<name>.samples.json)")
    parser.add_argument("--out-dir", default=common.DEFAULT_OUT_DIR)
    parser.add_argument(
        "--tag", default=None,
        help="suffix of the output files: gate_c_vllm_reference_<tag>.{json,md,samples.json} (default: no suffix, "
        "gate_c_vllm_reference.{json,md,samples.json}); letters, digits, '.', '_', '-'",
    )  # fmt: skip
    common.add_preflight_args(parser)
    args = parser.parse_args(argv)

    if args.tag is not None and not re.fullmatch(r"[A-Za-z0-9._-]+", args.tag):
        parser.error(f"--tag {args.tag!r}: use letters, digits, '.', '_' and '-' only (it becomes part of a file name)")
    ckpt_dir = os.path.abspath(os.path.expanduser(args.ckpt))
    if not os.path.isdir(ckpt_dir):
        raise SystemExit(f"checkpoint directory not found: {ckpt_dir}")
    name = output_name(args.tag)
    out_base = os.path.join(args.out_dir, name)
    samples_path = os.path.abspath(os.path.expanduser(args.samples)) if args.samples else out_base + ".samples.json"

    if args.phase == "sample":
        return phase_sample(args, ckpt_dir, samples_path)
    if args.phase == "score":
        return phase_score(args, ckpt_dir, samples_path, out_base)

    # all: nvidia-smi only here, so this process holds no GPU memory while vLLM runs in the child
    if not args.skip_preflight:
        common.gpu_preflight(args.require_free_gb, args.device, share=args.gpu_memory_utilization, extra_gb=2.0)
    child = [
        sys.executable, os.path.abspath(__file__), "--phase", "sample",
        "--ckpt", ckpt_dir, "--samples", samples_path, "--out-dir", args.out_dir,
        "--num-prompts", str(args.num_prompts), "--n", str(args.n), "--max-new-tokens", str(args.max_new_tokens),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization), "--seed", str(args.seed),
        "--prompt-seed", str(args.prompt_seed),
    ]  # fmt: skip
    for flag, value in (("--tokenizer", args.tokenizer), ("--prompts-file", args.prompts_file), ("--data", args.data)):
        if value:
            child += [flag, value]
    common.log("[vllm reference] phase 'sample' in a child process: " + " ".join(child))
    status = subprocess.run(child).returncode
    if status != 0:
        common.log(f"[vllm reference] FAIL: the sampling process exited with status {status}")
        common.log(f"[gate 3c vllm reference] FAIL: phase 'sample' (stock vLLM) exited with status {status}; no {name}.json")
        # a negative status means the child was killed by a signal: report it as a crash, not as FAIL
        return 128 - status if status < 0 else status
    return phase_score(args, ckpt_dir, samples_path, out_base)


if __name__ == "__main__":
    common.run_main(main, "[gate 3c vllm reference]")
