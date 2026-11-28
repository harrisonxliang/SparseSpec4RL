"""Gate 3b, stock-vLLM side: greedy responses of stock vLLM 0.9.1 with per-step top-2 log-probs.

Run in the BASELINE env (``verl-baseline``: stock vLLM 0.9.1), from the repo root:

    python integration/gates/gate_b_vllm_greedy.py --out baseline/results/sparsespec_checks/gate_b_vllm_greedy.json

then hand the file to the gate in the SparseSpec env:

    python -m integration.gates.gate_b_greedy --engine sparsespec --vllm-json <that file> --tag three_way

Use the SAME --model, --num-prompts, --max-new-tokens, --prompt-seed and --data
(or --prompts-file) in both; the gate refuses a file whose prompts differ.

The engine is configured as verl 0.5.0's ``vLLMRollout`` runs it for the
baseline arm (``workers/rollout/vllm_rollout/vllm_rollout_spmd.py`` with
``baseline/config/grpo_base.sh``): dtype bfloat16, ``enforce_eager=True``,
chunked prefill with ``max_num_batched_tokens=10240``, prefix caching on,
``disable_custom_all_reduce``, seed 0, ``max_model_len = 1024 + 8192``.
Greedy sampling is verl's ``do_sample=False`` setting (temperature 0,
top_p 1, top_k -1), ``n=1``, ``detokenize=False``, no extra stop tokens
(vLLM adds the checkpoint's generation-config EOS ids itself, as in training).
Not copied from verl, because they do not change the numbers:
``enable_sleep_mode`` (verl sleeps the engine between steps) and
``distributed_executor_backend="external_launcher"`` (verl runs under
torchrun; a standalone process uses vLLM's default single-process executor).

The only deviation from the baseline's sampling parameters is ``logprobs=2``
instead of ``logprobs=0``, so that every step carries the top-2 candidates.
It does not change the sampled tokens: in V1 (``v1/sample/sampler.py``,
``Sampler.forward``) the log-probs are computed into a separate tensor
(``raw_logprobs = logits.log_softmax(dim=-1, dtype=torch.float32)``) before
the float32 copy of the logits that is argmax-ed.

What the per-step numbers are (read from the 0.9.1 sources):
  * V1 (default in 0.9.1, ``VLLM_USE_V1`` unset): the log-probs are
    ``log_softmax`` of the RAW logits (the lm_head output in the model dtype,
    bfloat16, upcast to float32) "before any penalties or temperature
    scaling". V0 (``model_executor/layers/sampler.py``): the log-probs are of
    the processed logits, but at temperature < 1e-5 the temperature is
    replaced by 1.0 (``sampling_metadata.py``) and no penalties are set here,
    so they are the raw logits' log-softmax as well.
  * Hence ``margin = logprob(top-1) - logprob(top-2)`` is exactly the raw
    LOGIT margin ``logit(top-1) - logit(top-2)`` (the log-sum-exp cancels),
    up to float32 rounding (~1e-6).
  * Per step vLLM returns ``dict[token_id, Logprob(logprob, rank,
    decoded_token)]`` holding the sampled token (its true rank) and the top-2
    (ranks 1, 2; ``v1/engine/logprobs.py::_make_logprob_dict``), i.e. 2 or 3
    entries. The top-2 are taken as the two largest log-probs.

Output: ONE JSON file (``--out``, default ``<out-dir>/gate_b_vllm_greedy.json``)
in the schema of ``common.greedy_tokens_record`` (side ``vllm``): prompts and
their hash, token ids, finish reasons, per-step sampled log-prob and rank,
top-2 token ids and log-probs, margins, the engine and sampling arguments,
V0 or V1, and the vllm / torch / transformers versions. ``pass`` is true when
the output is usable and every sampled token is a top-1 token (greedy).
Exit status 0 / 1 accordingly.

This file must stay importable in the baseline env and on a CPU machine: it
imports torch, transformers, ``integration.gates.common`` (which uses
``serve.rl.base`` only); vLLM is imported inside ``run_vllm`` only, and never
``serve.rl.engine``.

NOT run with a real vLLM (none on the development machine): everything except
``run_vllm`` is exercised by CPU tests with stand-in output objects that have
the fields of the 0.9.1 ``RequestOutput`` / ``CompletionOutput`` / ``Logprob``.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from integration.gates import common  # noqa: E402

EXPECTED_VLLM_VERSION = "0.9.1"
NUM_LOGPROBS = 2
# baseline/config/grpo_base.sh: data.max_prompt_length=1024, data.max_response_length=8192 and
# actor_rollout_ref.rollout.max_num_batched_tokens=10240; verl sets max_model_len = prompt + response
BASELINE_MAX_MODEL_LEN = 1024 + 8192
BASELINE_MAX_NUM_BATCHED_TOKENS = 10240

LOGPROBS_DEFINITION = (
    "log_softmax of the raw model logits (lm_head output in bfloat16, upcast to float32), before temperature or "
    "penalties: vLLM V1 Sampler.forward computes raw_logprobs = logits.log_softmax(dim=-1, dtype=torch.float32) "
    "from the original logits; V0 replaces a temperature < 1e-5 by 1.0 before its log_softmax. margin = "
    "logprob(top-1) - logprob(top-2) is therefore the raw logit margin (log-sum-exp cancels; float32 rounding)."
)


class VllmOutputError(RuntimeError):
    """vLLM returned something this script cannot turn into a usable token file."""


def engine_kwargs(args, model_dir: str) -> dict:
    """Arguments of ``vllm.LLM``: verl's vLLMRollout with baseline/config/grpo_base.sh (see the module docstring)."""
    return dict(
        model=model_dir,
        tokenizer=model_dir,
        tensor_parallel_size=1,
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_custom_all_reduce=True,
        skip_tokenizer_init=False,
        max_model_len=args.max_model_len,
        disable_log_stats=True,
        max_num_batched_tokens=BASELINE_MAX_NUM_BATCHED_TOKENS,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        trust_remote_code=False,
        seed=args.seed,
    )


def sampling_kwargs(args) -> dict:
    """Arguments of ``vllm.SamplingParams``: verl's greedy (do_sample=False) setting plus logprobs=2."""
    return dict(
        n=1,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        ignore_eos=False,  # rollout.yaml: ignore_eos: False (copied into SamplingParams by verl)
        max_tokens=args.max_new_tokens,
        logprobs=NUM_LOGPROBS,
        detokenize=False,
    )


def run_vllm(llm_kwargs: dict, params_kwargs: dict, prompts: list[list[int]]):
    """The only function that touches vLLM. Returns ``(outputs, engine_info)``:
    the list of ``RequestOutput`` from ``LLM.generate`` and which engine ran.

    Tests replace this function with a stand-in that returns objects with the
    same fields.
    """
    import vllm
    from vllm import LLM, SamplingParams

    llm = LLM(**llm_kwargs)
    engine_cls = type(llm.llm_engine)
    info = {
        "engine_class": f"{engine_cls.__module__}.{engine_cls.__qualname__}",
        # LLMEngine.from_engine_args picks vllm.v1.engine.llm_engine.LLMEngine when envs.VLLM_USE_V1
        "engine_version": "V1" if engine_cls.__module__.startswith("vllm.v1.") else "V0",
        "VLLM_USE_V1_env": os.environ.get("VLLM_USE_V1"),
    }
    try:
        import vllm.envs as vllm_envs

        info["vllm_envs_VLLM_USE_V1"] = bool(vllm_envs.VLLM_USE_V1)
    except Exception as exc:  # recorded, never fatal
        info["vllm_envs_VLLM_USE_V1"] = f"unavailable: {exc!r}"
    try:
        info["compilation_level"] = int(llm.llm_engine.vllm_config.compilation_config.level)
    except Exception as exc:
        info["compilation_level"] = f"unavailable: {exc!r}"
    info["vllm_version_reported"] = getattr(vllm, "__version__", None)
    params = SamplingParams(**params_kwargs)
    info["sampling_params_repr"] = repr(params)
    start = time.perf_counter()
    # same call shape as vLLMRollout.generate_sequences: TokensPrompt dicts, use_tqdm=False
    outputs = llm.generate(
        prompts=[{"prompt_token_ids": list(p)} for p in prompts], sampling_params=params, use_tqdm=False
    )
    info["generate_seconds"] = time.perf_counter() - start
    return outputs, info


def _step_top2(step, token: int, where: str) -> dict:
    """Top-2 and the sampled token's numbers from one step's ``dict[token_id, Logprob]``."""
    if not isinstance(step, dict) or not step:
        raise VllmOutputError(f"{where}: no log-prob dict (got {type(step).__name__})")
    if token not in step:
        raise VllmOutputError(f"{where}: the sampled token {token} has no log-prob entry (keys {sorted(step)[:5]})")
    entries = []
    for token_id, lp in step.items():
        rank = getattr(lp, "rank", None)
        entries.append((float(lp.logprob), rank if rank is not None else sys.maxsize, int(token_id)))
    if len(entries) < NUM_LOGPROBS:
        raise VllmOutputError(f"{where}: {len(entries)} log-prob entries, expected at least {NUM_LOGPROBS} (logprobs=2)")
    entries.sort(key=lambda e: (-e[0], e[1], e[2]))
    (lp1, _, t1), (lp2, _, t2) = entries[0], entries[1]
    sampled = step[token]
    margin = lp1 - lp2 if math.isfinite(lp1) and math.isfinite(lp2) else None
    return {
        "top2": [t1, t2],
        "top2_logprobs": [lp1, lp2],
        "margin": margin,
        "sampled_logprob": float(sampled.logprob),
        "sampled_rank": getattr(sampled, "rank", None),
        # greedy: the sampled token must carry the largest log-prob (exact ties allowed)
        "greedy_ok": float(sampled.logprob) >= lp1,
    }


def extract_outputs(outputs, prompts: list[list[int]]) -> dict:
    """Token ids, finish reasons and per-step top-2 data from vLLM's ``RequestOutput`` list.

    Fields read (vLLM 0.9.1 ``outputs.py`` / ``sequence.py``): ``RequestOutput.prompt_token_ids``,
    ``.outputs[0]`` -> ``CompletionOutput.token_ids``, ``.logprobs`` (one
    ``dict[int, Logprob]`` per token), ``.finish_reason``, ``.stop_reason``;
    ``Logprob.logprob`` / ``.rank``.
    """
    outputs = list(outputs)
    if len(outputs) != len(prompts):
        raise VllmOutputError(f"vLLM returned {len(outputs)} outputs for {len(prompts)} prompts")
    result = {
        key: []
        for key in ("token_ids", "finish_reasons", "stop_reasons", "sampled_logprobs", "sampled_ranks",
                    "top2_token_ids", "top2_logprobs", "margins")
    }  # fmt: skip
    anomalies = []
    for i, (output, prompt) in enumerate(zip(outputs, prompts)):
        # LLM.generate sorts by request id, i.e. input order; checked, not assumed
        got_prompt = [int(t) for t in (output.prompt_token_ids or [])]
        if got_prompt != list(prompt):
            raise VllmOutputError(
                f"output {i} (request_id={getattr(output, 'request_id', None)!r}) is for a different prompt: "
                f"first difference at token {common.first_difference(got_prompt, list(prompt))}"
            )
        if len(output.outputs) != 1:
            raise VllmOutputError(f"output {i}: {len(output.outputs)} completions, expected 1 (n=1)")
        completion = output.outputs[0]
        tokens = [int(t) for t in completion.token_ids]
        steps = completion.logprobs
        if steps is None or len(steps) != len(tokens):
            raise VllmOutputError(
                f"output {i}: {len(tokens)} tokens but {None if steps is None else len(steps)} log-prob dicts "
                "(logprobs=2 was requested)"
            )
        if not tokens:
            raise VllmOutputError(f"output {i}: empty response")
        rows = [_step_top2(step, tok, f"output {i} step {j}") for j, (step, tok) in enumerate(zip(steps, tokens))]
        for j, row in enumerate(rows):
            if not row["greedy_ok"]:
                anomalies.append(
                    {"prompt": i, "position": j, "token": tokens[j], "sampled_logprob": row["sampled_logprob"],
                     "top1": row["top2"][0], "top1_logprob": row["top2_logprobs"][0]}
                )  # fmt: skip
        result["token_ids"].append(tokens)
        result["finish_reasons"].append(completion.finish_reason)
        stop = completion.stop_reason
        result["stop_reasons"].append(stop if stop is None or isinstance(stop, (int, str)) else str(stop))
        result["sampled_logprobs"].append([r["sampled_logprob"] for r in rows])
        result["sampled_ranks"].append([r["sampled_rank"] for r in rows])
        result["top2_token_ids"].append([r["top2"] for r in rows])
        result["top2_logprobs"].append([r["top2_logprobs"] for r in rows])
        result["margins"].append([r["margin"] for r in rows])
    result["greedy_anomalies"] = anomalies
    return result


def package_versions() -> dict:
    import torch
    import transformers

    versions = {"torch": torch.__version__, "transformers": transformers.__version__, "vllm": None}
    try:
        from importlib import metadata

        versions["vllm"] = metadata.version("vllm")
    except Exception:
        try:
            import vllm

            versions["vllm"] = getattr(vllm, "__version__", None)
        except Exception:
            versions["vllm"] = None
    return versions


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None, help="output JSON (default: <out-dir>/gate_b_vllm_greedy.json)")
    parser.add_argument("--out-dir", default=common.DEFAULT_OUT_DIR)
    parser.add_argument("--model", default=common.DEFAULT_MODEL, help="HF directory or hub id (as gate_b_greedy --model)")
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--prompts-file", default=None, help="JSON list of token-id lists (instead of MATH prompts)")
    parser.add_argument("--data", default=None, help="train parquet (default: $DATA_DIR/math_l3to5_train.parquet)")
    parser.add_argument("--prompt-seed", type=int, default=0)
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.45, help="vLLM's GPU share (rollout.gpu_memory_utilization)"
    )
    parser.add_argument("--seed", type=int, default=0, help="vLLM engine seed (verl: rollout seed 0)")
    parser.add_argument(
        "--max-model-len", type=int, default=BASELINE_MAX_MODEL_LEN, help="default: the baseline's 1024 + 8192"
    )
    common.add_preflight_args(parser)
    args = parser.parse_args(argv)
    out_path = os.path.abspath(os.path.expanduser(args.out or os.path.join(args.out_dir, "gate_b_vllm_greedy.json")))

    if args.skip_preflight:
        common.log("[preflight] skipped (--skip-preflight)")
        preflight = {"skipped": True}
    else:
        # vLLM claims gpu_memory_utilization x total and refuses to start if that much is not free
        preflight = common.gpu_preflight(
            args.require_free_gb, "cuda:0", share=args.gpu_memory_utilization, extra_gb=1.0
        )
    model_dir = common.resolve_model_dir(args.model)
    eos = common.eos_ids_from_checkpoint(model_dir)
    prompts, prompt_meta = common.resolve_prompts(
        prompts_file=args.prompts_file, model_dir=model_dir, n=args.num_prompts, data_path=args.data, seed=args.prompt_seed
    )
    longest = max(map(len, prompts))
    if longest + args.max_new_tokens > args.max_model_len:
        raise SystemExit(
            f"[gate 3b vllm] longest prompt ({longest}) + max_new_tokens ({args.max_new_tokens}) exceeds "
            f"--max-model-len {args.max_model_len}"
        )
    versions = package_versions()
    common.log(
        f"[gate 3b vllm] {len(prompts)} prompts ({prompt_meta['source']}), lengths {min(map(len, prompts))}-{longest}, "
        f"sha256 {common.prompts_sha256(prompts)[:16]}..., max_new_tokens={args.max_new_tokens}, T=0, logprobs=2; "
        f"versions {versions}"
    )
    if versions["vllm"] is not None and versions["vllm"] != EXPECTED_VLLM_VERSION:
        common.log(
            f"[gate 3b vllm] WARNING: vllm {versions['vllm']} is not the baseline's {EXPECTED_VLLM_VERSION}; "
            "is this the verl-baseline env?"
        )
    llm_kwargs = engine_kwargs(args, model_dir)
    params_kwargs = sampling_kwargs(args)
    common.log(f"[gate 3b vllm] vllm.LLM({llm_kwargs})")
    common.log(f"[gate 3b vllm] vllm.SamplingParams({params_kwargs})")

    outputs, engine_info = run_vllm(llm_kwargs, params_kwargs, prompts)
    common.log(f"[gate 3b vllm] engine: {engine_info.get('engine_version')} ({engine_info.get('engine_class')})")
    extracted = extract_outputs(outputs, prompts)
    anomalies = extracted.pop("greedy_anomalies")
    margins = [m for row in extracted["margins"] for m in row if m is not None]
    contract = common.check_responses(
        types.SimpleNamespace(
            token_ids=extracted["token_ids"],
            logprobs=extracted["sampled_logprobs"],
            finish_reasons=extracted["finish_reasons"],
        ),
        args.max_new_tokens,
        eos,
    )
    passed = not anomalies
    record = common.greedy_tokens_record(
        side="vllm",
        engine=f"vllm {versions['vllm']} {engine_info.get('engine_version')}",
        model=model_dir,
        prompts=prompts,
        prompts_meta=prompt_meta,
        token_ids=extracted["token_ids"],
        finish_reasons=extracted["finish_reasons"],
        max_new_tokens=args.max_new_tokens,
        eos_token_ids=eos,
        **{k: v for k, v in extracted.items() if k not in ("token_ids", "finish_reasons")},
        num_logprobs=NUM_LOGPROBS,
        logprobs_definition=LOGPROBS_DEFINITION,
        margin_is_logit_margin=True,
        engine_kwargs=llm_kwargs,
        sampling_kwargs=params_kwargs,
        engine_info=engine_info,
        versions=versions,
        greedy_anomalies=anomalies,
        contract_notes=contract,
        pass_rule="usable output and every sampled token has the largest log-prob of its step",
        environment=common.environment_info(None),
        preflight=preflight,
    )
    record["pass"] = passed
    common.write_json(out_path, record)

    num_tokens = sum(len(r) for r in extracted["token_ids"])
    common.log(
        f"[gate 3b vllm] {num_tokens} tokens, {sum(r == 'stop' for r in extracted['finish_reasons'])}/{len(prompts)} "
        f"stopped by EOS; smallest top-2 margin {common.fmt_num(min(margins) if margins else None)} logits; "
        f"{len(contract)} response-contract note(s)"
    )
    for note in contract[:5]:
        common.log(f"[gate 3b vllm] NOTE (vLLM output vs the engines' response contract): {note}")
    common.log(f"[gate 3b vllm] wrote {out_path}")
    if passed:
        common.log("[gate 3b vllm] PASS (token file usable; every step greedy)")
        return 0
    for item in anomalies[:5]:
        common.log(f"[gate 3b vllm] sampled token is not top-1: {item}")
    common.log(f"[gate 3b vllm] FAIL: {len(anomalies)} step(s) where the sampled token is not the top-1 token at T=0")
    return 1


if __name__ == "__main__":
    common.run_main(main, "[gate 3b vllm]")
