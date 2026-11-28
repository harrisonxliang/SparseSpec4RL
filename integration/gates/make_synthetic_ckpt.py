"""Synthetic "trained" checkpoint: base model + small random noise.

A stand-in for the real baseline checkpoint
(``$CKPT_ROOT/vllm_baseline_seed0/hf/global_step_25``) so the reload gate (3c)
can run before the baseline has trained: an HF directory with float32
safetensors (verl's actor keeps float32 master weights, so that is what the
engine receives during training), the base model's config and its tokenizer
files.

    python -m integration.gates.make_synthetic_ckpt \
        --base Qwen/Qwen3-0.6B --out $CKPT_ROOT/synthetic/relnoise0.02_seed0

The noise is Gaussian with standard deviation ``--rel-noise`` x RMS(tensor),
drawn per tensor in sorted-name order from ``torch.Generator(seed)``: the same
(base, rel-noise, seed, torch version) gives the same checkpoint. A tied
``lm_head.weight`` / ``embed_tokens.weight`` pair gets ONE noise tensor, so
the checkpoint stays tied. The script prints the fraction of weights whose
bfloat16 value changed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

from integration.gates import common  # noqa: E402

# Why the default is 2% and not something "realistically small":
# bfloat16 keeps 8 significant bits (1 implicit + 7 stored mantissa bits), so
# neighbouring bf16 values are 2^-7 = 0.78% apart (relative) at the bottom of
# a binade and 0.39% apart at the top. Round-to-nearest therefore absorbs any
# perturbation below half of that: noise far below ~0.4% relative mostly
# rounds away and the engine, which holds bf16 weights, would be loading a
# model that is bitwise almost the base model. (A real step-25 checkpoint at
# lr 1e-6 is in that regime: most of its bf16 weights equal the base model's.)
# For a reload test the checkpoint must be a clearly DIFFERENT bf16 model, so
# the default noise is several bf16 steps for a typical weight: at 2% of the
# tensor's RMS about 90% of the bf16 values change, while the model stays a
# working language model (the perturbation is comparable to ~7-bit weight
# quantization noise).
DEFAULT_REL_NOISE = 0.02

REPORT_NAME = "synthetic_ckpt_report.json"
_EMBED_SUFFIX = "embed_tokens.weight"
_HEAD_KEY = "lm_head.weight"


def default_synthetic_dir(rel_noise: float = DEFAULT_REL_NOISE, seed: int = 0) -> str:
    return os.path.join(common.DEFAULT_CKPT_ROOT, "synthetic", f"relnoise{rel_noise:g}_seed{seed}")


def perturb_state_dict(state: dict, rel_noise: float, seed: int, *, tied: bool) -> dict:
    """float32 copy of ``state`` with Gaussian noise of std ``rel_noise`` x RMS(tensor) on every float tensor."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    embed_key = next((k for k in state if k.endswith(_EMBED_SUFFIX)), None)
    out: dict[str, torch.Tensor] = {}
    for name in sorted(state):
        tensor = state[name]
        if not tensor.is_floating_point():
            out[name] = tensor.clone()
            continue
        if tied and name == _HEAD_KEY and embed_key is not None:
            continue  # filled from the embedding below: one noise tensor for the tied pair
        value = tensor.to(torch.float32)
        rms = value.pow(2).mean().sqrt()
        noise = torch.randn(value.shape, generator=generator, dtype=torch.float32)
        out[name] = value + noise * (float(rel_noise) * rms)
    if tied and _HEAD_KEY in state and embed_key is not None:
        if state[_HEAD_KEY].shape != state[embed_key].shape or not torch.equal(
            state[_HEAD_KEY].to(torch.float32), state[embed_key].to(torch.float32)
        ):
            raise ValueError("tie_word_embeddings=True but the base checkpoint's lm_head and embedding differ")
        out[_HEAD_KEY] = out[embed_key].clone()
    return out


def make_synthetic_checkpoint(
    base: str, out_dir: str, rel_noise: float = DEFAULT_REL_NOISE, seed: int = 0, *, overwrite: bool = False
) -> dict:
    """Write the synthetic checkpoint to ``out_dir`` and return its report (also saved there)."""
    from safetensors.torch import save_file

    if rel_noise <= 0:
        raise ValueError(f"rel_noise must be > 0, got {rel_noise}")
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    if os.path.isdir(out_dir) and os.listdir(out_dir) and not overwrite:
        raise FileExistsError(f"{out_dir} exists and is not empty (pass --overwrite to replace it)")
    base_dir = common.resolve_model_dir(base)
    with open(os.path.join(base_dir, "config.json")) as f:
        config = json.load(f)
    tied = bool(config.get("tie_word_embeddings", False))

    base_state = common.read_state_dict(base_dir)
    new_state = perturb_state_dict(base_state, rel_noise, seed, tied=tied)
    diff = common.bf16_diff_report(base_state, new_state)

    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(out_dir):  # only reached with --overwrite: drop stale weight shards
        if name.endswith(".safetensors") or name.endswith(".safetensors.index.json"):
            os.remove(os.path.join(out_dir, name))
    copied = common.copy_non_weight_files(base_dir, out_dir)
    for key in ("torch_dtype", "dtype"):  # transformers 4.x / 5.x spelling
        if key in config:
            config[key] = "float32"
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    save_file(new_state, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})

    # read back: float32 on disk, tied pair identical, same key set as the base
    written = common.read_state_dict(out_dir)
    if set(written) != set(base_state):
        raise RuntimeError("synthetic checkpoint does not have the base checkpoint's keys")
    bad_dtype = [k for k, v in written.items() if v.is_floating_point() and v.dtype != torch.float32]
    if bad_dtype:
        raise RuntimeError(f"non-float32 tensors in the synthetic checkpoint: {bad_dtype[:5]}")
    embed_key = next((k for k in written if k.endswith(_EMBED_SUFFIX)), None)
    tied_equal = None
    if tied and _HEAD_KEY in written and embed_key is not None:
        tied_equal = bool(torch.equal(written[_HEAD_KEY], written[embed_key]))
        if not tied_equal:
            raise RuntimeError("tied lm_head / embedding differ in the synthetic checkpoint")

    tokenizer_files = [n for n in copied if n.startswith(("tokenizer", "vocab", "merges", "special_tokens", "added_tokens", "chat_template"))]  # fmt: skip
    if not tokenizer_files:
        common.log(f"[synthetic] WARNING: {base_dir} has no tokenizer files; the checkpoint has none either")
    report = {
        "base": str(base),
        "base_dir": base_dir,
        "out_dir": out_dir,
        "rel_noise": float(rel_noise),
        "seed": int(seed),
        "tie_word_embeddings": tied,
        "tied_pair_stored_twice": tied_equal is not None,
        "num_tensors": len(written),
        "num_parameters": int(sum(v.numel() for v in written.values())),
        "bf16_fraction_changed": diff["fraction_changed"],
        "bf16_num_changed": diff["num_changed"],
        "bf16_num_elements": diff["num_elements"],
        "copied_files": copied,
        "tokenizer_files": tokenizer_files,
        "torch": torch.__version__,
    }
    common.write_json(os.path.join(out_dir, REPORT_NAME), report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=common.DEFAULT_MODEL, help="base model: HF directory or hub id")
    parser.add_argument("--out", default=None, help="output directory (default: $CKPT_ROOT/synthetic/relnoise<r>_seed<s>)")
    parser.add_argument(
        "--rel-noise",
        type=float,
        default=DEFAULT_REL_NOISE,
        help="noise std as a fraction of each tensor's RMS (default %(default)s; see the comment in this file)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    out_dir = args.out or default_synthetic_dir(args.rel_noise, args.seed)
    report = make_synthetic_checkpoint(args.base, out_dir, args.rel_noise, args.seed, overwrite=args.overwrite)
    common.log(f"[synthetic] wrote {report['out_dir']}")
    common.log(
        f"[synthetic] {report['num_tensors']} tensors, {report['num_parameters']:,} parameters, float32, "
        f"rel_noise={report['rel_noise']:g}, seed={report['seed']}"
    )
    common.log(
        f"[synthetic] bf16 values changed vs base: {report['bf16_num_changed']:,} / {report['bf16_num_elements']:,} "
        f"= {100 * report['bf16_fraction_changed']:.2f}%"
    )
    if report["bf16_fraction_changed"] < 0.5:
        common.log(
            "[synthetic] WARNING: fewer than half of the bf16 weights changed; raise --rel-noise for a checkpoint "
            "the engine can tell apart from the base model"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
