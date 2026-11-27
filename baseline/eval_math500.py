"""Offline MATH-500 evaluation on saved checkpoints.

Loads a weight-only HF checkpoint into STOCK vLLM (both arms are
evaluated with this same script/engine, so the eval engine is not a
confound), samples N completions per problem, and reports pass@1
averaged over the N samples (proposal: N=4).

Usage (on the A100 box):
  python baseline/eval_math500.py \
      --ckpt $CKPT_ROOT/vllm_baseline_seed0/hf/global_step_50 \
      --exp-name vllm_baseline_seed0 --step 50

Sampling defaults match the training distribution (T=1.0, top_p=1.0);
keep them identical across every checkpoint of every arm.
"""

import argparse
import json
import os

from math_grader import grade_answer

INSTRUCTION = "\nPlease think step by step and then provide only the final answer within \\boxed{}."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="HF model dir or HF hub id")
    parser.add_argument("--tokenizer", default=None, help="defaults to --ckpt")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--step", type=int, required=True, help="training step of this checkpoint (0 = base model)")
    parser.add_argument("--data", default=None, help="math500 parquet from prepare_data.py; falls back to HF hub")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    # ---- load problems ----
    if args.data and os.path.exists(args.data):
        import pandas as pd

        df = pd.read_parquet(args.data)
        problems = [
            {"problem": row["prompt"][0]["content"], "answer": row["reward_model"]["ground_truth"],
             "unique_id": row["extra_info"].get("unique_id", str(i))}
            for i, row in df.iterrows()
        ]
        already_instructed = True
    else:
        from datasets import load_dataset

        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        problems = [{"problem": ex["problem"], "answer": ex["answer"], "unique_id": ex["unique_id"]} for ex in ds]
        already_instructed = False

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.ckpt)
    prompts = []
    for p in problems:
        content = p["problem"] if already_instructed else p["problem"] + INSTRUCTION
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    llm = LLM(
        model=args.ckpt,
        tokenizer=args.tokenizer or args.ckpt,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_tokens + 1024,
    )
    sampling = SamplingParams(
        n=args.n, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens
    )
    outputs = llm.generate(prompts, sampling)

    # ---- grade ----
    os.makedirs(args.out_dir, exist_ok=True)
    detail_path = os.path.join(args.out_dir, f"{args.exp_name}.step{args.step}.math500.jsonl")
    n_correct_total, n_samples_total, n_unparseable = 0, 0, 0
    per_problem = []
    with open(detail_path, "w") as f:
        for p, out in zip(problems, outputs):
            corrects, lengths = [], []
            for comp in out.outputs:
                ok = grade_answer(comp.text, p["answer"])
                corrects.append(ok)
                lengths.append(len(comp.token_ids))
                if "\\boxed" not in comp.text:
                    n_unparseable += 1
            n_correct_total += sum(corrects)
            n_samples_total += len(corrects)
            per_problem.append(sum(corrects) / len(corrects))
            f.write(json.dumps({
                "unique_id": p["unique_id"],
                "pass_at_1": sum(corrects) / len(corrects),
                "correct": corrects,
                "response_tokens": lengths,
            }) + "\n")

    pass_at_1 = n_correct_total / n_samples_total
    summary = {
        "exp_name": args.exp_name,
        "step": args.step,
        "ckpt": args.ckpt,
        "n_problems": len(problems),
        "n_samples_per_problem": args.n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "pass_at_1": pass_at_1,
        "unparseable_rate": n_unparseable / n_samples_total,
    }
    summary_path = os.path.join(args.out_dir, "math500_summary.jsonl")
    with open(summary_path, "a") as f:
        f.write(json.dumps(summary) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"details: {detail_path}")
    print(f"summary appended to: {summary_path}")


if __name__ == "__main__":
    main()
