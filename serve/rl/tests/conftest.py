"""Shared fixtures for the CPU unit tests of the SparseSpec verl backend.

Run from the repository root::

    PYTHONPATH=. python -m pytest serve/rl/tests -q

Nothing here needs CUDA, flashinfer or triton, and nothing stubs them:

    test_base / test_memory / test_weights   the pure-Python / pure-torch modules
    test_reference_engine                    ReferenceEngine on CPU (tiny Qwen3)
    test_verl_rollout                        rollout class + sharding manager on verl's DataProto
    test_metrics_logging                     sparsespec/* metrics: trainer -> console log -> CSV
    test_run_script                          baseline/run_seed_sparsespec.sh vs the baseline, via hydra
    test_verl_patch                          integration/verl-0.5.0-sparsespec.patch on a verl copy
    test_gpu_sources_static                  ast-level checks of engine.py / scheduler.py

The real engine's behaviour is covered by ``test_engine_gpu.py`` (another
module in this directory, skipped without CUDA) on the A100 box.

There is deliberately no ``__init__.py`` in this directory: ``serve/`` is a
namespace package, so with one pytest would put ``serve/`` itself on
``sys.path`` and import the tests as ``rl.tests...`` (a second copy of
``serve.rl``). Helpers are therefore handed to the tests through the ``tk``
fixture instead of being imported from a helper module.

The tiny model mirrors the structure of Qwen3-0.6B that matters to the code
under test: Qwen3 architecture (q_norm / k_norm, grouped KV heads), tied
embeddings, an EOS id LIST, and a pad token that is also one of the EOS ids
(151643 ``<|endoftext|>`` for Qwen3).
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# tiny Qwen3
# ---------------------------------------------------------------------------
VOCAB = 160
HIDDEN = 32
LAYERS = 2
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 8
INTERMEDIATE = 64
MAX_POSITIONS = 512

IM_END_ID = 7  # plays <|im_end|> (151645): config.eos_token_id
ENDOFTEXT_ID = 3  # plays <|endoftext|> (151643): second EOS id AND the pad token
EOS_IDS = [IM_END_ID, ENDOFTEXT_ID]  # generation_config.eos_token_id is a list for Qwen3
PAD_ID = ENDOFTEXT_ID
FIRST_NORMAL_ID = 8  # prompts are drawn from [FIRST_NORMAL_ID, VOCAB): no pad / EOS inside a prompt

# Matrices ~ N(0, 0.2^2): next-token entropy is about 4.4 nats (of ln 160 =
# 5.1), so sampling is neither uniform nor degenerate.
_WEIGHT_STD = 0.2

_INT_VIEW = {2: torch.int16, 4: torch.int32, 8: torch.int64}


def tiny_qwen3_config(**overrides):
    from transformers import Qwen3Config

    kwargs = dict(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        intermediate_size=INTERMEDIATE,
        max_position_embeddings=MAX_POSITIONS,
        tie_word_embeddings=True,
        eos_token_id=IM_END_ID,
        pad_token_id=PAD_ID,
        bos_token_id=ENDOFTEXT_ID,
    )
    kwargs.update(overrides)
    return Qwen3Config(**kwargs)


def fill_parameters(module: nn.Module, seed: int) -> None:
    """Deterministic values that do not depend on (or touch) the global RNG.

    Matrices get N(0, 0.2^2); vectors (the RMSNorm gains) get 1 + N(0, 0.1^2)
    so that a wrong gain mapping is visible. Tied parameters are filled once.
    """
    gen = torch.Generator().manual_seed(10_000 + seed)
    with torch.no_grad():
        for _, param in module.named_parameters():  # de-duplicated: a tied lm_head appears once
            if param.ndim >= 2:
                values = torch.randn(param.shape, generator=gen) * _WEIGHT_STD
            else:
                values = 1.0 + 0.1 * torch.randn(param.shape, generator=gen)
            param.copy_(values.to(param.dtype))


def make_tiny_qwen3(seed: int = 0, dtype: torch.dtype = torch.float32):
    """A random Qwen3ForCausalLM on CPU (eval mode, parameters trainable)."""
    from transformers import Qwen3ForCausalLM

    rng_state = torch.get_rng_state()
    try:
        model = Qwen3ForCausalLM(tiny_qwen3_config())
    finally:
        torch.set_rng_state(rng_state)
    assert model.lm_head.weight is model.model.embed_tokens.weight, "HF did not tie lm_head to embed_tokens"
    fill_parameters(model, seed)
    model.generation_config.eos_token_id = list(EOS_IDS)
    model.generation_config.pad_token_id = PAD_ID
    return model.to(dtype).eval()


def make_prompts(lengths, seed: int = 0) -> list[list[int]]:
    gen = torch.Generator().manual_seed(20_000 + seed)
    return [torch.randint(FIRST_NORMAL_ID, VOCAB, (int(n),), generator=gen).tolist() for n in lengths]


# ---------------------------------------------------------------------------
# engine-layout stand-in for serve.model.model.Transformer
# ---------------------------------------------------------------------------
class _RMSNormLike(nn.Module):
    """serve.model.model.RMSNorm: one parameter, ``weight`` of shape [dim]."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))


class _MergedAttention(nn.Module):
    """serve.model.model.Attention after ``merge()``.

    ``__init__`` registers q_proj, k_proj, v_proj, o_proj, q_norm, k_norm;
    ``merge()`` adds ``wqkv = cat([q_proj, k_proj, v_proj], dim=0)`` and
    deletes q_proj / k_proj / v_proj. ``o_proj`` never has a bias.
    """

    def __init__(self, dim: int, num_qo_heads: int, num_kv_heads: int, head_dim: int, attention_bias: bool):
        super().__init__()
        self.o_proj = nn.Linear(num_qo_heads * head_dim, dim, bias=False)
        self.q_norm = _RMSNormLike(head_dim)
        self.k_norm = _RMSNormLike(head_dim)
        self.wqkv = nn.Linear(dim, (num_qo_heads + 2 * num_kv_heads) * head_dim, bias=attention_bias)


class _MergedFeedForward(nn.Module):
    """serve.model.model.FeedForward after ``merge()``:
    ``w12 = cat([gate_proj, up_proj], dim=0)``; gate_proj / up_proj deleted."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.w12 = nn.Linear(dim, 2 * hidden, bias=False)


class _Block(nn.Module):
    def __init__(self, dim, num_qo_heads, num_kv_heads, head_dim, hidden, attention_bias):
        super().__init__()
        self.input_layernorm = _RMSNormLike(dim)
        self.self_attn = _MergedAttention(dim, num_qo_heads, num_kv_heads, head_dim, attention_bias)
        self.post_attention_layernorm = _RMSNormLike(dim)
        self.mlp = _MergedFeedForward(dim, hidden)


class FakeEngineTransformer(nn.Module):
    """Has exactly the parameter names / shapes ``Transformer.from_pretrained``
    leaves behind (no forward pass; flashinfer is not involved).

    tie: "shared" -> lm_head.weight IS embed_tokens.weight (the engine after
                     ``SparseSpecEngine._retie_embeddings``);
         "copy"   -> lm_head.weight is a separate, equal tensor (what
                     ``from_pretrained`` + ``.to(device)`` leaves behind);
         "untied" -> config.tie_word_embeddings is False, independent tensors.
    """

    def __init__(
        self,
        tie: str = "shared",
        dtype: torch.dtype = torch.float32,
        attention_bias: bool = False,
        seed: int = 99,
        *,
        vocab: int = VOCAB,
        dim: int = HIDDEN,
        num_layers: int = LAYERS,
        num_qo_heads: int = HEADS,
        num_kv_heads: int = KV_HEADS,
        head_dim: int = HEAD_DIM,
        hidden: int = INTERMEDIATE,
    ):
        super().__init__()
        assert tie in ("shared", "copy", "untied")
        self.embed_tokens = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList(
            _Block(dim, num_qo_heads, num_kv_heads, head_dim, hidden, attention_bias) for _ in range(num_layers)
        )
        self.norm = _RMSNormLike(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.config = SimpleNamespace(tie_word_embeddings=tie != "untied", num_layers=num_layers)
        self.tie = tie

        fill_parameters(self, seed)
        if tie == "shared":
            self.lm_head.weight = self.embed_tokens.weight
        elif tie == "copy":
            with torch.no_grad():
                self.lm_head.weight.copy_(self.embed_tokens.weight)
        self.to(dtype)
        self.requires_grad_(False)
        if tie == "shared":
            assert self.lm_head.weight.data_ptr() == self.embed_tokens.weight.data_ptr()
        else:
            assert self.lm_head.weight.data_ptr() != self.embed_tokens.weight.data_ptr()


def expected_engine_state(hf_state: dict, dtype: torch.dtype, num_layers: int = LAYERS) -> dict:
    """Engine-layout tensors for an HF state dict, built by hand the way
    ``Transformer.from_pretrained`` builds them: strip ``model.``, cast every
    tensor (``model.to(dtype)``), then fuse with ``torch.cat(..., dim=0)``.

    Independent of serve.rl.weights: this is the oracle its output is compared to.
    """
    src = {(k[len("model.") :] if k.startswith("model.") else k): v.detach().to(dtype) for k, v in hf_state.items()}
    out = {"embed_tokens.weight": src["embed_tokens.weight"]}
    for i in range(num_layers):
        p = f"layers.{i}."
        attn = p + "self_attn."
        out[p + "input_layernorm.weight"] = src[p + "input_layernorm.weight"]
        out[attn + "o_proj.weight"] = src[attn + "o_proj.weight"]
        out[attn + "q_norm.weight"] = src[attn + "q_norm.weight"]
        out[attn + "k_norm.weight"] = src[attn + "k_norm.weight"]
        out[attn + "wqkv.weight"] = torch.cat(
            [src[attn + "q_proj.weight"], src[attn + "k_proj.weight"], src[attn + "v_proj.weight"]], dim=0
        )
        if attn + "q_proj.bias" in src:
            out[attn + "wqkv.bias"] = torch.cat(
                [src[attn + "q_proj.bias"], src[attn + "k_proj.bias"], src[attn + "v_proj.bias"]], dim=0
            )
        out[p + "post_attention_layernorm.weight"] = src[p + "post_attention_layernorm.weight"]
        out[p + "mlp.down_proj.weight"] = src[p + "mlp.down_proj.weight"]
        out[p + "mlp.w12.weight"] = torch.cat([src[p + "mlp.gate_proj.weight"], src[p + "mlp.up_proj.weight"]], dim=0)
    out["norm.weight"] = src["norm.weight"]
    out["lm_head.weight"] = src["lm_head.weight"] if "lm_head.weight" in src else src["embed_tokens.weight"]
    return out


# ---------------------------------------------------------------------------
# comparisons
# ---------------------------------------------------------------------------
def bits(tensor: torch.Tensor) -> torch.Tensor:
    """Integer view of a tensor's raw bits (bf16 has no numpy equivalent)."""
    tensor = tensor.detach().cpu().contiguous()
    return tensor.view(_INT_VIEW[tensor.element_size()]) if tensor.is_floating_point() else tensor


def bit_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(bits(a), bits(b)))


def named_params(module: nn.Module) -> dict:
    """name -> parameter, tied parameters listed under both names."""
    return dict(module.named_parameters(remove_duplicate=False))


def snapshot(module: nn.Module) -> dict:
    return {name: param.detach().clone() for name, param in named_params(module).items()}


def assert_matches_snapshot(module: nn.Module, snap: dict) -> None:
    params = named_params(module)
    assert set(params) == set(snap)
    changed = [name for name, param in params.items() if not bit_equal(param, snap[name])]
    assert not changed, f"parameters changed: {changed}"


def perturbed_state(module: nn.Module, seed: int, scale: float = 0.05) -> dict:
    """A state dict that differs from ``module`` in EVERY tensor (tied keys stay equal)."""
    gen = torch.Generator().manual_seed(30_000 + seed)
    out, by_storage = {}, {}
    for key, value in module.state_dict().items():
        ptr = value.data_ptr()
        if ptr not in by_storage:
            noise = torch.randn(value.shape, generator=gen) * scale
            by_storage[ptr] = (value.detach().float() + noise).to(value.dtype)
        out[key] = by_storage[ptr]
    return out


# ---------------------------------------------------------------------------
# independent references computed with plain HF forwards (no cache, no padding)
# ---------------------------------------------------------------------------
@torch.no_grad()
def hf_greedy(model, prompt, max_new_tokens: int, min_margin: float = 2e-3) -> list[int]:
    """Greedy continuation of ONE unpadded prompt, re-running the full
    sequence every step. EOS is ignored (see ``cut_at_eos``).

    The margin check makes a near-tie in the random test model fail loudly
    here instead of showing up as a mysterious token mismatch elsewhere: the
    engine computes the same logits through padded, cached batches, which
    differ from this pass by float32 noise (~1e-6). Every prompt / weight seed
    used with this helper has a top-2 margin of at least 4e-3.
    """
    seq = [int(t) for t in prompt]
    out: list[int] = []
    for _ in range(max_new_tokens):
        logits = model(input_ids=torch.tensor([seq], dtype=torch.long), use_cache=False).logits[0, -1].float()
        top2 = logits.topk(2).values
        assert float(top2[0] - top2[1]) > min_margin, "near-tie in the test model: pick another prompt / seed"
        token = int(logits.argmax())
        seq.append(token)
        out.append(token)
    return out


def cut_at_eos(tokens, eos_ids) -> list[int]:
    """The response contract: everything up to and including the first EOS."""
    eos = set(int(t) for t in eos_ids)
    for i, token in enumerate(tokens):
        if token in eos:
            return list(tokens[: i + 1])
    return list(tokens)


@torch.no_grad()
def teacher_forced_logprobs(model, prompt, response, temperature: float) -> list[float]:
    """log_softmax(logits / T) of every response token, from ONE uncached
    forward over prompt + response (no padding, default position ids)."""
    seq = torch.tensor([[int(t) for t in prompt] + [int(t) for t in response]], dtype=torch.long)
    logits = model(input_ids=seq, use_cache=False).logits[0].double()
    first = len(prompt) - 1
    logp = torch.log_softmax(logits[first : first + len(response)] / temperature, dim=-1)
    index = torch.tensor([int(t) for t in response], dtype=torch.long).unsqueeze(-1)
    return logp.gather(-1, index).squeeze(-1).tolist()


def make_reference_engine(model, *, dtype: str = "float32", seed: int = 0, max_batch_size: int = 8, **kwargs):
    """ReferenceEngine on CPU around a COPY of ``model`` (the engine casts and
    freezes the module it is given in place)."""
    from serve.rl.reference_engine import ReferenceEngine

    return ReferenceEngine(
        "tiny-qwen3-in-memory",
        max_batch_size=max_batch_size,
        seed=seed,
        dtype=dtype,
        device="cpu",
        model=copy.deepcopy(model),
        **kwargs,
    )


class GenerateRecorder:
    """Wraps ``engine.generate`` in place and records every call."""

    def __init__(self, engine):
        self.calls: list[SimpleNamespace] = []
        original = engine.generate

        def generate(*args, **kwargs):
            result = original(*args, **kwargs)
            self.calls.append(SimpleNamespace(args=args, kwargs=kwargs, result=copy.deepcopy(result)))
            return result

        engine.generate = generate

    @property
    def last(self) -> SimpleNamespace:
        return self.calls[-1]


# ---------------------------------------------------------------------------
# verl-side stand-ins (verl is imported lazily: test_base / test_memory /
# test_weights do not need it)
# ---------------------------------------------------------------------------
class TinyTokenizer:
    """The rollout only reads ``pad_token_id`` from the tokenizer."""

    pad_token_id = PAD_ID
    eos_token_id = IM_END_ID


def tiny_generation_config() -> SimpleNamespace:
    """What ``ActorRolloutRefWorker.generation_config`` holds for Qwen3."""
    return SimpleNamespace(eos_token_id=list(EOS_IDS), pad_token_id=PAD_ID)


def verl_rollout_yaml() -> Path:
    import verl

    return Path(verl.__file__).parent / "trainer" / "config" / "rollout" / "rollout.yaml"


def make_rollout_config(**overrides):
    """``actor_rollout_ref.rollout`` as hydra hands it to the worker: verl's
    own rollout.yaml defaults, the overrides of baseline/run_seed_sparsespec.sh
    that matter here, and tiny lengths. Dotted keys address nested values,
    e.g. ``**{"sparsespec.seed": 3, "val_kwargs.temperature": 0.5}``.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(verl_rollout_yaml())
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create(
            {
                "name": "sparsespec",
                "prompt_length": 24,
                "response_length": 16,
                "tensor_model_parallel_size": 1,
                "gpu_memory_utilization": 0.45,
                "calculate_log_probs": True,
                "sparsespec": {
                    "engine": "reference",
                    "spec_stride": 8,
                    "budget_ratio": 0.05,
                    "num_min_budget": 128,
                    "kv_cache": "pillar",
                    "max_batch_size": 64,
                    "cuda_graph": True,
                },
            }
        ),
    )
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, force_add=True)
    OmegaConf.set_struct(cfg, True)  # hydra-composed configs are in struct mode
    return cfg


def collate(rows: list[dict]) -> dict:
    """Copy of verl.utils.dataset.rl_dataset.collate_fn (l.37-65); the module
    itself imports ``datasets``, which the tests should not depend on."""
    tensors: dict = {}
    non_tensors: dict = {}
    for row in rows:
        for key, val in row.items():
            (tensors if isinstance(val, torch.Tensor) else non_tensors).setdefault(key, []).append(val)
    out = {key: torch.stack(val, dim=0) for key, val in tensors.items()}
    out.update({key: np.array(val, dtype=object) for key, val in non_tensors.items()})
    return out


def build_gen_batch(
    prompt_token_ids,
    prompt_length: int,
    *,
    n: int = 1,
    with_raw_prompt_ids: bool = True,
    global_steps: int = 1,
):
    """(batch, gen_batch) built the way verl 0.5.0 builds them.

    Rows: ``RLHFDataset.__getitem__`` (utils/dataset/rl_dataset.py l.261-309):
    tokenizer output -> ``postprocess_data`` (LEFT padded to
    data.max_prompt_length with tokenizer.pad_token_id) ->
    ``compute_position_id_with_mask``; ``raw_prompt_ids`` is the unpadded list.
    Batch: ``collate_fn`` -> ``DataProto.from_single_dict`` -> the pop / repeat
    of ``RayPPOTrainer.fit`` (trainer/ppo/ray_trainer.py l.1141-1163).

    ``with_raw_prompt_ids=False`` drops that field, which makes the rollout
    fall back to stripping the left padding itself.
    """
    import verl.utils.torch_functional as verl_F
    from verl import DataProto
    from verl.utils.model import compute_position_id_with_mask

    rows = []
    for index, ids in enumerate(prompt_token_ids):
        input_ids = torch.tensor([list(ids)], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=prompt_length,
            pad_token_id=TinyTokenizer.pad_token_id,
            left_pad=True,
            truncation="error",
        )
        position_ids = compute_position_id_with_mask(attention_mask)
        row = {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "index": index,
            "data_source": "unit-test",
        }
        if with_raw_prompt_ids:
            row["raw_prompt_ids"] = [int(t) for t in ids]
        rows.append(row)

    batch = DataProto.from_single_dict(collate(rows))
    non_tensor_keys = ["index"] + (["raw_prompt_ids"] if with_raw_prompt_ids else [])
    gen_batch = batch.pop(
        batch_keys=["input_ids", "attention_mask", "position_ids"],
        non_tensor_batch_keys=non_tensor_keys,
    )
    gen_batch.meta_info["global_steps"] = global_steps
    gen_batch = gen_batch.repeat(repeat_times=n, interleave=True)
    return batch, gen_batch


_DEFAULT = object()


def worker_generate_sequences(rollout, sharding_manager, prompts, generation_config=_DEFAULT, tokenizer=None):
    """``ActorRolloutRefWorker.generate_sequences`` (workers/fsdp_workers.py
    l.720-756) with ``self.*`` replaced by arguments and the GPU-only calls
    (device move, log_gpu_memory_usage, empty_cache) dropped. Same order of
    operations, same verl helpers.

    ``generation_config`` defaults to the tiny Qwen3-like one (EOS id list);
    pass ``generation_config=None`` to take the worker's tokenizer fallback.
    """
    from verl.utils.profiler import simple_timer
    from verl.utils.profiler.performance import reduce_timing

    if generation_config is _DEFAULT:
        generation_config = tiny_generation_config()
    if tokenizer is None:
        tokenizer = TinyTokenizer()
    prompts = prompts.to("cpu")

    meta_info = {
        "eos_token_id": generation_config.eos_token_id if generation_config is not None else tokenizer.eos_token_id,
        "pad_token_id": generation_config.pad_token_id if generation_config is not None else tokenizer.pad_token_id,
    }
    prompts.meta_info.update(meta_info)
    timing_generate = {}
    with sharding_manager:
        prompts = sharding_manager.preprocess_data(prompts)
        with simple_timer("generate_sequences", timing_generate):
            output = rollout.generate_sequences(prompts=prompts)

        output = sharding_manager.postprocess_data(output)

    timing_generate.update(sharding_manager.timing)
    # We calculate the average timing across all ranks
    # to make sure meta_info["timing"] is the same
    timing_generate = reduce_timing(timing_generate)
    output.meta_info["timing"] = timing_generate
    output = output.to("cpu")
    return output


SPARSESPEC_TIMING_KEYS = (
    # written by SparseSpecShardingManager.__enter__ / __exit__
    "sparsespec/weight_sync_s",
    "sparsespec/wake_s",
    "sparsespec/graph_capture_s",
    "sparsespec/sleep_s",
    # engine stats of the generate() call inside the context
    "sparsespec/accept_len_mean",
    "sparsespec/accept_rate",
    "sparsespec/num_cycles",
    "sparsespec/num_iters",
    "sparsespec/batch_size_mean",
    "sparsespec/num_offloads",
    "sparsespec/kv_util_peak",
    "sparsespec/peak_mem_gb",
    "sparsespec/process_peak_alloc_gb",
    "sparsespec/process_peak_reserved_gb",
    "sparsespec/gen_s",
    # what the engine keeps on the GPU while the trainer runs (SparseSpecShardingManager.__exit__)
    "sparsespec/resident_asleep_gb",
)


def patch_path() -> Path:
    return REPO_ROOT / "integration" / "verl-0.5.0-sparsespec.patch"


def patch_added_lines(target_suffix: str) -> list[str]:
    """Lines the verl patch ADDS to the file whose path ends with ``target_suffix``
    (leading ``+`` removed, indentation kept)."""
    added: list[str] = []
    in_target = False
    for line in patch_path().read_text().splitlines():
        if line.startswith("+++ "):
            in_target = line.split()[1].endswith(target_suffix)
            continue
        if line.startswith("--- "):
            in_target = False
            continue
        if in_target and line.startswith("+"):
            added.append(line[1:])
    return added


def trainer_metrics_statement() -> str:
    """The one statement the patch adds to RayPPOTrainer.fit, as source."""
    lines = [ln.strip() for ln in patch_added_lines("verl/trainer/ppo/ray_trainer.py")]
    code = [ln for ln in lines if ln and not ln.startswith("#")]
    assert len(code) == 1, f"expected exactly one added statement in ray_trainer.py, got {code}"
    return code[0]


def apply_trainer_metrics_statement(metrics: dict, timing_raw: dict) -> None:
    """Run the patch's statement on the trainer's two dicts (``metrics`` and
    ``timing_raw`` are the names it uses inside ``RayPPOTrainer.fit``)."""
    namespace = {"metrics": metrics, "timing_raw": timing_raw}
    exec(compile(trainer_metrics_statement(), "<verl-0.5.0-sparsespec.patch>", "exec"), namespace)


def sparsespec_branch_source() -> str:
    """The ``elif rollout_name == "sparsespec":`` branch the patch adds to
    ``ActorRolloutRefWorker._build_rollout``, dedented and turned into an ``if``."""
    import textwrap

    code = textwrap.dedent("\n".join(patch_added_lines("verl/workers/fsdp_workers.py"))).strip("\n")
    assert code.startswith('elif rollout_name == "sparsespec":'), code[:80]
    return code[2:]


_TOOLKIT = SimpleNamespace(
    REPO_ROOT=REPO_ROOT,
    VOCAB=VOCAB,
    HIDDEN=HIDDEN,
    LAYERS=LAYERS,
    HEADS=HEADS,
    KV_HEADS=KV_HEADS,
    HEAD_DIM=HEAD_DIM,
    INTERMEDIATE=INTERMEDIATE,
    MAX_POSITIONS=MAX_POSITIONS,
    IM_END_ID=IM_END_ID,
    ENDOFTEXT_ID=ENDOFTEXT_ID,
    EOS_IDS=EOS_IDS,
    PAD_ID=PAD_ID,
    FIRST_NORMAL_ID=FIRST_NORMAL_ID,
    SPARSESPEC_TIMING_KEYS=SPARSESPEC_TIMING_KEYS,
    tiny_qwen3_config=tiny_qwen3_config,
    fill_parameters=fill_parameters,
    make_tiny_qwen3=make_tiny_qwen3,
    make_prompts=make_prompts,
    FakeEngineTransformer=FakeEngineTransformer,
    expected_engine_state=expected_engine_state,
    bits=bits,
    bit_equal=bit_equal,
    named_params=named_params,
    snapshot=snapshot,
    assert_matches_snapshot=assert_matches_snapshot,
    perturbed_state=perturbed_state,
    hf_greedy=hf_greedy,
    cut_at_eos=cut_at_eos,
    teacher_forced_logprobs=teacher_forced_logprobs,
    make_reference_engine=make_reference_engine,
    GenerateRecorder=GenerateRecorder,
    TinyTokenizer=TinyTokenizer,
    tiny_generation_config=tiny_generation_config,
    verl_rollout_yaml=verl_rollout_yaml,
    make_rollout_config=make_rollout_config,
    collate=collate,
    build_gen_batch=build_gen_batch,
    worker_generate_sequences=worker_generate_sequences,
    patch_path=patch_path,
    patch_added_lines=patch_added_lines,
    trainer_metrics_statement=trainer_metrics_statement,
    apply_trainer_metrics_statement=apply_trainer_metrics_statement,
    sparsespec_branch_source=sparsespec_branch_source,
)


@pytest.fixture(scope="session")
def tk() -> SimpleNamespace:
    """Toolkit: constants and builders shared by the test modules."""
    return _TOOLKIT


@pytest.fixture(scope="session")
def _tiny_templates():
    return {seed: make_tiny_qwen3(seed) for seed in (0, 1)}


@pytest.fixture
def tiny_model(_tiny_templates):
    """float32 tiny Qwen3 with weights "A" (a private copy per test)."""
    return copy.deepcopy(_tiny_templates[0])


@pytest.fixture
def tiny_model_b(_tiny_templates):
    """float32 tiny Qwen3 with different weights "B"."""
    return copy.deepcopy(_tiny_templates[1])


@pytest.fixture
def hf_state(tiny_model) -> dict:
    """HF-named float32 state dict of ``tiny_model`` (keys ``model.*`` + ``lm_head.weight``)."""
    return {k: v.detach().clone() for k, v in tiny_model.state_dict().items()}


@pytest.fixture(autouse=True)
def _cpu_test_preconditions(request):
    """For the CPU test modules: every test starts from the same global seed
    (no test, and no code under test, may depend on the global torch RNG), and
    torch's process-wide defaults are what a trainer process has.

    The second check is a tripwire for the GPU engine, which may only change
    the default dtype / device inside its own calls: if test_engine_gpu.py ran
    earlier in this process and leaked bf16-on-cuda defaults, say so here
    instead of failing somewhere obscure. test_engine_gpu.py itself is left
    alone (it manages its own RNG state).
    """
    if Path(str(request.node.fspath)).name == "test_engine_gpu.py":
        yield
        return
    assert (
        torch.get_default_dtype() == torch.float32
    ), f"torch default dtype is {torch.get_default_dtype()}: something (the GPU engine?) leaked it"
    assert torch.empty(0).device.type == "cpu", "torch default device is not cpu: something leaked it"
    torch.manual_seed(1234)
    yield
