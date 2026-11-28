"""CPU tests of the correctness-gate scripts (no GPU, no flashinfer, no vllm).

    PYTHONPATH=. python -m pytest integration/gates/tests -q

The gates are run end to end with ``--engine reference`` on a tiny random
Qwen3 model written to a temp dir (float32 safetensors, tied embeddings) and
prompts given as token-id lists, which proves their control flow, statistics,
verdict rules and output files. Every verdict rule is also shown to FAIL when
it should: wrong weights for the greedy gate, deliberately broken engines for
the reload gate, broken CSVs for the smoke-run check.

There is deliberately no ``__init__.py`` next to this file: with one, pytest
would put ``integration/`` on ``sys.path`` and import ``integration.gates`` a
second time as ``gates``.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import pprint
import random
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from integration.gates import common  # noqa: E402
from integration.gates import gate_b_greedy, gate_c_reload, gate_c_vllm_reference, gate_d_smoke_check  # noqa: E402
from integration.gates import make_synthetic_ckpt as synth  # noqa: E402

VOCAB = 512
EOS_IDS = [VOCAB - 2, VOCAB - 1]  # generation_config.eos_token_id is a list for Qwen3
PAD_ID = VOCAB - 2


# ----------------------------------------------------------------------------
# tiny model, tokenizer, prompts
# ----------------------------------------------------------------------------
def _write_tiny_qwen3(path: Path, seed: int = 0) -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        tie_word_embeddings=True,
        # large init: logits have real margins (a 0.02 init gives a near-uniform model)
        initializer_range=0.5,
        eos_token_id=EOS_IDS[1],
        pad_token_id=PAD_ID,
        bos_token_id=None,
    )
    rng_state = torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        model = Qwen3ForCausalLM(config).to(torch.float32).eval()
    finally:
        torch.set_rng_state(rng_state)
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    model.generation_config.eos_token_id = list(EOS_IDS)
    model.generation_config.pad_token_id = PAD_ID
    model.save_pretrained(str(path), safe_serialization=True)


def _write_tiny_tokenizer(path: Path) -> None:
    """Whitespace word-level tokenizer with a ChatML-like chat template."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {f"w{i}": i for i in range(VOCAB - 8)}
    for offset, word in enumerate(["[UNK]", "user", "assistant", "<nl>", "<|im_start|>", "<spare>"]):
        vocab[word] = VOCAB - 8 + offset
    vocab["<|endoftext|>"] = EOS_IDS[0]
    vocab["<|im_end|>"] = EOS_IDS[1]
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", eos_token="<|im_end|>", pad_token="<|endoftext|>"
    )
    fast.chat_template = (
        "{% for m in messages %}<|im_start|> {{ m['role'] }} <nl> {{ m['content'] }} <|im_end|> <nl> {% endfor %}"
        "{% if add_generation_prompt %}<|im_start|> assistant <nl> {% endif %}"
    )
    fast.save_pretrained(str(path))


@pytest.fixture(scope="session")
def workdir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("gates")


@pytest.fixture(scope="session")
def tiny_dir(workdir) -> str:
    path = workdir / "tiny-qwen3"
    _write_tiny_qwen3(path)
    _write_tiny_tokenizer(path)
    return str(path)


@pytest.fixture(scope="session")
def prompts_file(workdir) -> str:
    rng = random.Random(0)
    prompts = [[rng.randrange(0, VOCAB - 8) for _ in range(rng.randrange(5, 40))] for _ in range(16)]
    path = workdir / "prompts.json"
    path.write_text(json.dumps(prompts))
    return str(path)


@pytest.fixture(scope="session")
def synthetic_dir(workdir, tiny_dir) -> str:
    out = workdir / "tiny-synthetic"
    report = synth.make_synthetic_checkpoint(tiny_dir, str(out), rel_noise=synth.DEFAULT_REL_NOISE, seed=0)
    assert report["bf16_fraction_changed"] > 0.5
    return str(out)


@pytest.fixture(scope="session")
def hf_tiny(tiny_dir):
    return common.load_hf_model(tiny_dir, "cpu", "float32")


def _read(path) -> dict:
    with open(path) as f:
        return json.load(f)


def _failed(result: dict) -> set[str]:
    return {c["name"] for c in result["checks"] if not c["ok"]}


# ----------------------------------------------------------------------------
# common.py: statistics
# ----------------------------------------------------------------------------
def test_compare_logprobs_statistics():
    a = [[-1.0, -2.0, -3.0], [-0.5]]
    b = [[-1.1, -2.0, -2.5], [-0.5]]
    stats = common.compare_logprobs(a, b)
    assert stats["count"] == 4 and stats["num_nonfinite"] == 0
    assert stats["mean"] == pytest.approx((0.1 + 0.0 + 0.5 + 0.0) / 4)
    assert stats["median"] == pytest.approx(0.05)
    assert stats["max"] == pytest.approx(0.5)
    assert stats["max_prob_ratio"] == pytest.approx(math.exp(0.5))
    assert stats["p99"] == pytest.approx(0.5 - 0.01 * 3 * (0.5 - 0.1))  # numpy's linear interpolation
    prob_diffs = [abs(math.exp(-1.0) - math.exp(-1.1)), 0.0, abs(math.exp(-3.0) - math.exp(-2.5)), 0.0]
    assert stats["prob_diff_max"] == pytest.approx(max(prob_diffs))  # the largest |delta log p| is not the largest |delta p|
    assert stats["prob_diff_mean"] == pytest.approx(sum(prob_diffs) / 4)


def test_compare_logprobs_counts_nonfinite_and_rejects_shape_mismatch():
    stats = common.compare_logprobs([[-1.0, float("-inf"), float("nan")]], [[-1.5, -2.0, -1.0]])
    assert stats["count"] == 1 and stats["num_nonfinite"] == 2 and stats["mean"] == pytest.approx(0.5)
    empty = common.compare_logprobs([[float("nan")]], [[-1.0]])
    assert empty["count"] == 0 and math.isnan(empty["mean"])
    with pytest.raises(ValueError):
        common.compare_logprobs([[-1.0, -2.0]], [[-1.0]])
    with pytest.raises(ValueError):
        common.compare_logprobs([[-1.0]], [[-1.0], [-2.0]])


def test_tracking_coefficient_tells_which_model_the_engine_follows():
    rng = random.Random(0)
    ref_a = [[rng.uniform(-5, 0) for _ in range(200)] for _ in range(20)]
    # B differs from A by much LESS than the engine noise: a ratio of mean |delta| could not tell them apart
    ref_b = [[x + rng.gauss(0, 0.01) for x in row] for row in ref_a]
    follows_a = [[x + rng.gauss(0, 0.05) for x in row] for row in ref_a]
    follows_b = [[x + rng.gauss(0, 0.05) for x in row] for row in ref_b]
    on_a = common.tracking_coefficient(follows_a, ref_a, ref_b)
    on_b = common.tracking_coefficient(follows_b, ref_a, ref_b)
    assert abs(on_a["beta"]) < 5 * on_a["se"] and on_a["se"] < 0.15
    assert abs(on_b["beta"] - 1.0) < 5 * on_b["se"] and on_b["se"] < 0.15
    assert on_a["count"] == 4000 and on_a["signal_rms"] == pytest.approx(0.01, rel=0.1)
    degenerate = common.tracking_coefficient(follows_a, ref_a, ref_a)
    assert math.isnan(degenerate["beta"])


# ----------------------------------------------------------------------------
# common.py: the trainer forward
# ----------------------------------------------------------------------------
@torch.no_grad()
def _independent_logprobs(model, prompt, response, temperature):
    """One unpadded sequence, default position ids, float64 log-softmax: an oracle that shares no code with common."""
    ids = torch.tensor([list(prompt) + list(response)], dtype=torch.long)
    logits = model(input_ids=ids, use_cache=False).logits[0].double()
    logp = torch.log_softmax(logits[len(prompt) - 1 : len(prompt) - 1 + len(response)] / temperature, dim=-1)
    return logp.gather(-1, torch.tensor(list(response)).unsqueeze(-1)).squeeze(-1).tolist()


@pytest.mark.parametrize("temperature", [1.0, 0.7])
@pytest.mark.parametrize("token_budget", [16, 150, 100_000])  # batch of one / several batches / one batch
def test_hf_trainer_logprobs_matches_an_unpadded_forward(hf_tiny, temperature, token_budget):
    rng = random.Random(1)
    prompts = [[rng.randrange(0, VOCAB - 8) for _ in range(n)] for n in (5, 31, 12, 20, 7)]
    responses = [[rng.randrange(0, VOCAB) for _ in range(n)] for n in (9, 1, 40, 3, 17)]
    got = common.hf_trainer_logprobs(
        hf_tiny, prompts, responses, temperature=temperature, token_budget=token_budget, pad_token_id=PAD_ID
    )
    assert [len(row) for row in got] == [len(r) for r in responses]  # caller's order and lengths
    for prompt, response, row in zip(prompts, responses, got):
        want = _independent_logprobs(hf_tiny, prompt, response, temperature)
        assert row == pytest.approx(want, abs=2e-4)
        assert all(x <= 0.0 for x in row)


def test_hf_trainer_logprobs_ignores_the_pad_token_and_validates_input(hf_tiny):
    prompts, responses = [[1, 2, 3], [4, 5, 6, 7, 8, 9]], [[10, 11], [12, 13, 14, 15, 16]]
    a = common.hf_trainer_logprobs(hf_tiny, prompts, responses, pad_token_id=0)
    b = common.hf_trainer_logprobs(hf_tiny, prompts, responses, pad_token_id=77)
    assert a == b  # right padding is masked out
    with pytest.raises(ValueError):
        common.hf_trainer_logprobs(hf_tiny, prompts, responses[:1])
    with pytest.raises(ValueError):
        common.hf_trainer_logprobs(hf_tiny, prompts, [[10], []])
    with pytest.raises(ValueError):
        common.hf_trainer_logprobs(hf_tiny, prompts, responses, temperature=0.0)


def test_load_hf_model_casts_parameters_but_not_the_rotary_buffer(tiny_dir):
    """bf16 parameters, float32 buffers: what from_pretrained(torch_dtype=bf16) gives the verl actor and vLLM.
    A Module.to(dtype=bf16) on top would round Qwen3's rotary inv_freq (a different model at long positions)."""
    model = common.load_hf_model(tiny_dir, "cpu", "bfloat16")
    assert {p.dtype for p in model.parameters()} == {torch.bfloat16}
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    buffers = {name: buf.dtype for name, buf in model.named_buffers() if buf.is_floating_point()}
    assert any(name.endswith("inv_freq") for name in buffers), buffers
    assert set(buffers.values()) == {torch.float32}, buffers
    reference = common.load_hf_model(tiny_dir, "cpu", "float32")
    for name, buf in model.named_buffers():
        if buf.is_floating_point():  # bit-identical to the float32 model's buffer, not a bf16 rounding of it
            assert torch.equal(buf, dict(reference.named_buffers())[name]), name


def test_hf_greedy_and_teacher_forced_logits_agree(hf_tiny):
    prompt = [3, 14, 15, 92, 65, 35]
    tokens = common.hf_greedy(hf_tiny, prompt, 24, EOS_IDS)
    assert 1 <= len(tokens) <= 24
    assert all(t not in EOS_IDS for t in tokens[:-1])
    assert len(tokens) == 24 or tokens[-1] in EOS_IDS
    logits = common.hf_teacher_forced_logits(hf_tiny, prompt, tokens)
    assert logits.shape == (len(tokens), VOCAB) and logits.dtype == torch.float32
    assert logits.argmax(dim=-1).tolist() == tokens  # float32: cached and uncached forwards agree
    # an EOS declared mid-way stops the loop right there, EOS kept
    cut = common.hf_greedy(hf_tiny, prompt, 24, [tokens[5]])
    assert cut == tokens[: tokens.index(tokens[5]) + 1]


# ----------------------------------------------------------------------------
# common.py: contract checker, small helpers
# ----------------------------------------------------------------------------
def _result(tokens, logprobs, reasons):
    from serve.rl.base import GenerationResult

    return GenerationResult(token_ids=tokens, logprobs=logprobs, finish_reasons=reasons)


def test_check_responses_accepts_valid_and_names_each_violation():
    eos = [9]
    ok = _result([[1, 2, 9], [1, 2, 3, 4]], [[-0.1, -0.2, -0.3], [-1.0, -1.0, -1.0, 0.0]], ["stop", "length"])
    assert common.check_responses(ok, 4, eos) == []

    def problems(tokens, logprobs, reasons, **kw):
        return common.check_responses(_result([tokens], [logprobs], [reasons]), 4, eos, **kw)

    assert any("empty" in p for p in problems([], [], "length"))
    assert any("log-probs" in p for p in problems([1, 9], [-0.1], "stop"))
    assert any("max_new_tokens" in p for p in problems([1, 2, 3, 4, 5], [-0.1] * 5, "length"))
    assert any("after the first EOS" in p for p in problems([1, 9, 3, 4], [-0.1] * 4, "length"))
    assert any("no EOS" in p for p in problems([1, 2], [-0.1] * 2, "length"))  # stopped early without EOS
    assert any("finish_reason" in p for p in problems([1, 9], [-0.1] * 2, "length"))
    assert any("finish_reason" in p for p in problems([1, 2, 3, 4], [-0.1] * 4, "stop"))
    assert any("not finite or > 0" in p for p in problems([1, 9], [-0.1, float("nan")], "stop"))
    assert any("not finite or > 0" in p for p in problems([1, 9], [-0.1, 0.25], "stop"))
    assert any("greedy" in p for p in problems([1, 9], [0.0, -0.1], "stop", greedy=True))
    assert problems([1, 9], [0.0, 0.0], "stop", greedy=True) == []


def test_small_helpers(tmp_path):
    assert common.first_difference([1, 2, 3], [1, 2, 3]) == -1
    assert common.first_difference([1, 2, 3], [1, 5, 3]) == 1
    assert common.first_difference([1, 2], [1, 2, 3]) == 2  # strict prefix
    assert common.repeat_interleave([[1], [2, 3]], 3) == [[1], [1], [1], [2, 3], [2, 3], [2, 3]]
    assert torch.equal(  # verl: gen_batch.repeat(repeat_times=n, interleave=True) -> repeat_interleave
        torch.tensor(common.repeat_interleave([[1], [2]], 2)), torch.tensor([[1], [2]]).repeat_interleave(2, dim=0)
    )
    assert common.fingerprint([[1, 2], [3]]) == common.fingerprint([(1, 2), (3,)])
    assert common.fingerprint([[1, 2], [3]]) != common.fingerprint([[1], [2, 3]])

    path = common.write_json(str(tmp_path / "sub" / "x.json"), {"nan": float("nan"), "inf": float("inf"), "t": torch.tensor([1.5]), "s": {2, 1}})
    assert _read(path) == {"nan": None, "inf": None, "t": [1.5], "s": [1, 2]}  # strict JSON
    table = common.md_table(["a|b", "c"], [[1.23456789, None], ["x|y", True]])
    assert table.splitlines()[0] == "| a\\|b | c |" and "| 1.235 | - |" in table and "x\\|y" in table


def test_prompts_file_loading(tmp_path, prompts_file, tiny_dir):
    assert len(common.load_prompts_file(prompts_file)) == 16
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(json.dumps({"prompts": [[1, 2], [3]]}))
    assert common.load_prompts_file(str(wrapped)) == [[1, 2], [3]]
    for bad in ([], [[1], []], [[1, "2"]], [[1.5]], [[True]], {"x": 1}, [1, 2]):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            common.load_prompts_file(str(path))
    prompts, meta = common.resolve_prompts(prompts_file=prompts_file, model_dir=tiny_dir, n=5)
    assert len(prompts) == 5 and meta["source"].startswith("prompts-file:")
    with pytest.raises(ValueError):
        common.resolve_prompts(prompts_file=prompts_file, model_dir=tiny_dir, n=17)


def test_model_dir_and_checkpoint_readers(tiny_dir, tmp_path):
    assert common.resolve_model_dir(tiny_dir) == os.path.abspath(tiny_dir)
    for missing in ("/no/such/dir", "./relative/missing", "a/b/c", "~/definitely-not-here-xyz"):
        with pytest.raises(FileNotFoundError):  # never mistaken for a hub id
            common.resolve_model_dir(missing)
    state = common.read_state_dict(tiny_dir)
    assert "model.embed_tokens.weight" in state and all(v.dtype == torch.float32 for v in state.values())
    with pytest.raises(FileNotFoundError):
        common.read_state_dict(str(tmp_path))
    assert common.eos_ids_from_checkpoint(tiny_dir) == sorted(EOS_IDS)


def test_bf16_diff_report(tiny_dir):
    state = common.read_state_dict(tiny_dir)
    same = common.bf16_diff_report(state, {k: v.clone() for k, v in state.items()})
    assert same["num_changed"] == 0 and same["fraction_changed"] == 0.0 and same["num_tensors"] == len(state)
    # a perturbation far below half a bf16 step is invisible, one bf16 step on one element is counted once
    tiny_noise = {k: v * (1 + 1e-5) for k, v in state.items()}
    assert common.bf16_diff_report(state, tiny_noise)["fraction_changed"] < 0.01
    bumped = {k: v.clone() for k, v in state.items()}
    key = "model.norm.weight"
    as_bf16 = bumped[key].to(torch.bfloat16)
    as_bf16.view(torch.int16)[0] += 1
    bumped[key] = as_bf16.to(torch.float32)
    report = common.bf16_diff_report(state, bumped)
    assert report["num_changed"] == 1 and report["num_tensors_changed"] == 1
    # wrapper prefixes are normalized the way the engine's loader normalizes them
    prefixed = {"_fsdp_wrapped_module." + k: v for k, v in state.items()}
    assert common.bf16_diff_report(state, prefixed)["num_tensors"] == len(state)


def test_prompt_constants_match_prepare_data():
    common.check_prompt_constants_in_sync()  # raises on drift
    theirs = common._prepare_data_constants()
    assert theirs["INSTRUCTION"] == common.INSTRUCTION
    assert theirs["TRAIN_LEVELS"] == common.TRAIN_LEVELS


# ----------------------------------------------------------------------------
# common.py: MATH prompts are built exactly as verl builds them
# ----------------------------------------------------------------------------
def _prepare_data_namespace() -> dict:
    """INSTRUCTION and make_row() of baseline/prepare_data.py, taken from its
    source without running its imports (datasets, math_grader)."""
    import ast

    source = (REPO_ROOT / "baseline" / "prepare_data.py").read_text()
    wanted = [
        node
        for node in ast.parse(source).body
        if (isinstance(node, ast.FunctionDef) and node.name == "make_row")
        or (isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "INSTRUCTION")
    ]
    namespace: dict = {}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), "prepare_data.py", "exec"), namespace)
    return namespace


def test_load_math_prompts_matches_verl_dataset(tiny_dir, tmp_path):
    try:  # verl's RLHFDataset reads the parquet through `datasets` (pyarrow)
        import pandas as pd
        import pyarrow  # noqa: F401
        from omegaconf import OmegaConf
        from verl.utils.dataset.rl_dataset import RLHFDataset
    except ImportError as exc:  # also a pyarrow / numpy version mismatch
        pytest.skip(f"needs pandas, pyarrow, datasets and verl: {type(exc).__name__}: {exc}")
    from transformers import AutoTokenizer

    ns = _prepare_data_namespace()
    rng = random.Random(3)
    rows = []
    for i in range(40):
        problem = " ".join(f"w{rng.randrange(0, 400)}" for _ in range(rng.randrange(3, 60)))
        rows.append(ns["make_row"](problem, str(i), "math_l3to5_train", "train", i, {"level": "Level 3", "type": "x"}))
    parquet = tmp_path / "math_l3to5_train.parquet"
    pd.DataFrame(rows).to_parquet(parquet)  # as baseline/prepare_data.py writes it

    tokenizer = AutoTokenizer.from_pretrained(tiny_dir)

    def verl_prompts(max_prompt_length: int) -> dict:
        """extra_info.index -> the prompt ids verl hands to the rollout engine."""
        config = OmegaConf.create(
            {
                "max_prompt_length": max_prompt_length,
                "filter_overlong_prompts": True,
                "filter_overlong_prompts_workers": 1,
                "truncation": "error",
                "cache_dir": str(tmp_path / "verl-cache"),
            }
        )
        dataset = RLHFDataset(data_files=str(parquet), tokenizer=tokenizer, config=config)
        out = {}
        for item in range(len(dataset)):
            row = dataset[item]
            ids = [int(t) for t in row["raw_prompt_ids"]]
            # what the rollout derives from the left-padded batch when raw_prompt_ids is absent
            assert row["input_ids"][row["attention_mask"].bool()].tolist() == ids
            out[row["extra_info"]["index"]] = ids
        return out

    # 1) every row: identical token ids
    by_index = verl_prompts(common.MAX_PROMPT_LENGTH)
    assert len(by_index) == len(rows)
    prompts, meta = common.load_math_prompts(tokenizer, len(rows), data_path=str(parquet), seed=0, return_meta=True)
    assert sorted(meta["rows"]) == list(range(len(rows))) and meta["rows"] != list(range(len(rows)))  # shuffled
    for row_index, ids in zip(meta["rows"], prompts):
        assert ids == by_index[row_index]
    assert ns["INSTRUCTION"].split()[0] in rows[0]["prompt"][0]["content"]  # the instruction is part of the prompt

    # 2) the length filter (data.filter_overlong_prompts): rows longer than the limit are skipped
    limit = 40
    short_rows = {i for i, ids in by_index.items() if len(ids) <= limit}
    assert 0 < len(short_rows) < len(rows)
    kept, kept_meta = common.load_math_prompts(
        tokenizer, len(short_rows), data_path=str(parquet), seed=0, max_prompt_length=limit, return_meta=True
    )
    assert set(kept_meta["rows"]) == short_rows and all(len(p) <= limit for p in kept)
    with pytest.raises(ValueError):
        common.load_math_prompts(tokenizer, len(short_rows) + 1, data_path=str(parquet), max_prompt_length=limit)
    # verl's own filter measures len(tokenizer.apply_chat_template(...)); that is a token count only where
    # apply_chat_template returns a list of ids (transformers 4.x, the target env), so compare only there
    if isinstance(tokenizer.apply_chat_template(rows[0]["prompt"], add_generation_prompt=True), list):
        assert set(verl_prompts(limit)) == short_rows

    # 3) deterministic, seed-dependent order; a given path must exist
    assert common.load_math_prompts(tokenizer, 5, data_path=str(parquet), seed=0) == prompts[:5]
    assert common.load_math_prompts(tokenizer, 5, data_path=str(parquet), seed=1) != prompts[:5]
    with pytest.raises(FileNotFoundError):
        common.load_math_prompts(tokenizer, 1, data_path=str(tmp_path / "missing.parquet"))


# ----------------------------------------------------------------------------
# common.py: shared-GPU preflight (nvidia-smi output injected)
# ----------------------------------------------------------------------------
SMI = (
    "0, GPU-aaaaaaaa-0000-0000-0000-000000000000, NVIDIA A100-PCIE-40GB, 40536, 30000, 10536\n"
    "1, GPU-bbbbbbbb-1111-1111-1111-111111111111, NVIDIA A100-PCIE-40GB, 40536, 536, 40000\n"
)


def test_gpu_preflight(monkeypatch):
    assert common.gpu_preflight(100.0, "cpu")["skipped"] is True
    gpus = common.parse_nvidia_smi(SMI)
    assert [g["index"] for g in gpus] == [0, 1] and gpus[0]["free_mib"] == 10536.0

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    report = common.gpu_preflight(8.0, "cuda:0", smi_output=SMI)
    assert report["gpu_index"] == 0 and report["free_gb"] == pytest.approx(10536 / 1024)
    with pytest.raises(SystemExit, match="10.3 GiB free"):
        common.gpu_preflight(12.0, "cuda:0", smi_output=SMI)  # refuses; nothing else happens
    assert common.gpu_preflight(30.0, "cuda:1", smi_output=SMI)["gpu_index"] == 1
    # requirement derived from the engine's share of the card
    assert common.gpu_preflight(None, "cuda:1", share=0.45, extra_gb=6.0, smi_output=SMI)["required_free_gb"] == pytest.approx(
        0.45 * 40536 / 1024 + 6.0
    )
    with pytest.raises(SystemExit):
        common.gpu_preflight(None, "cuda:0", share=0.45, extra_gb=6.0, smi_output=SMI)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")  # cuda:0 is now the second physical card
    assert common.gpu_preflight(30.0, "cuda:0", smi_output=SMI)["gpu_index"] == 1
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-bbbbbbbb,GPU-aaaaaaaa")
    assert common.gpu_preflight(30.0, "cuda:0", smi_output=SMI)["gpu_index"] == 1
    assert common.gpu_preflight(8.0, "cuda:1", smi_output=SMI)["gpu_index"] == 0
    with pytest.raises(SystemExit, match="cannot map"):
        common.gpu_preflight(1.0, "cuda:2", smi_output=SMI)


# ----------------------------------------------------------------------------
# make_synthetic_ckpt
# ----------------------------------------------------------------------------
def test_synthetic_checkpoint(tiny_dir, synthetic_dir, tmp_path):
    report = _read(os.path.join(synthetic_dir, synth.REPORT_NAME))
    assert report["rel_noise"] == synth.DEFAULT_REL_NOISE and report["tie_word_embeddings"] is True
    assert 0.8 < report["bf16_fraction_changed"] <= 1.0  # the default noise changes bf16 values clearly

    base, new = common.read_state_dict(tiny_dir), common.read_state_dict(synthetic_dir)
    assert set(base) == set(new) and all(v.dtype == torch.float32 for v in new.values())
    for key, value in base.items():  # noise std = rel_noise x the tensor's RMS (5 sigma of the sample RMS)
        relative = float((new[key] - value).pow(2).mean().sqrt() / value.pow(2).mean().sqrt())
        assert relative == pytest.approx(synth.DEFAULT_REL_NOISE, rel=5 / math.sqrt(2 * value.numel())), key
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"):
        assert os.path.exists(os.path.join(synthetic_dir, name)), name

    # loads as an HF model whose embeddings are still tied, and is a different model
    model = common.load_hf_model(synthetic_dir, "cpu", "float32")
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    assert torch.equal(model.model.embed_tokens.weight, new["model.embed_tokens.weight"])

    # deterministic by seed; refuses to overwrite silently
    again, other = tmp_path / "again", tmp_path / "other"
    synth.make_synthetic_checkpoint(tiny_dir, str(again), seed=0)
    synth.make_synthetic_checkpoint(tiny_dir, str(other), seed=1)
    assert (again / "model.safetensors").read_bytes() == Path(synthetic_dir, "model.safetensors").read_bytes()
    assert (other / "model.safetensors").read_bytes() != Path(synthetic_dir, "model.safetensors").read_bytes()
    with pytest.raises(FileExistsError):
        synth.make_synthetic_checkpoint(tiny_dir, str(again), seed=0)
    assert synth.main(["--base", tiny_dir, "--out", str(again), "--overwrite", "--rel-noise", "0.0005"]) == 0
    assert _read(again / synth.REPORT_NAME)["bf16_fraction_changed"] < 0.5  # far below a bf16 step: mostly rounds away


def test_synthetic_checkpoint_keeps_a_stored_tied_pair_tied():
    state = {"model.embed_tokens.weight": torch.randn(8, 4), "lm_head.weight": None, "model.norm.weight": torch.ones(4)}
    state["lm_head.weight"] = state["model.embed_tokens.weight"].clone()  # checkpoints that store both
    out = synth.perturb_state_dict(state, 0.05, seed=0, tied=True)
    assert torch.equal(out["lm_head.weight"], out["model.embed_tokens.weight"])
    assert not torch.equal(out["lm_head.weight"], state["lm_head.weight"])
    untied = synth.perturb_state_dict(state, 0.05, seed=0, tied=False)
    assert not torch.equal(untied["lm_head.weight"], untied["model.embed_tokens.weight"])


# ----------------------------------------------------------------------------
# gate 3b
# ----------------------------------------------------------------------------
def _gate_b(tiny_dir, prompts_file, out_dir, tag, *extra) -> tuple[int, dict]:
    status = gate_b_greedy.main(
        ["--engine", "reference", "--device", "cpu", "--model", tiny_dir, "--prompts-file", prompts_file,
         "--num-prompts", "8", "--max-new-tokens", "40", "--out-dir", str(out_dir), "--tag", tag, *extra]
    )  # fmt: skip
    return status, _read(Path(out_dir) / f"gate_b_greedy_{tag}.json")


def test_gate_b_passes_on_identical_weights(tiny_dir, prompts_file, tmp_path):
    status, result = _gate_b(tiny_dir, prompts_file, tmp_path, "same")
    assert status == 0 and result["pass"] is True and not _failed(result)
    summary = result["summary"]
    assert summary["num_prompts"] == 8 and summary["mismatch"] == 0 and summary["length_mismatch"] == 0
    assert summary["identical"] + summary["near_tie"] == 8
    assert summary["positions_compared"] == sum(
        r["engine_len"] if r["first_diff"] < 0 else r["first_diff"] + 1 for r in result["per_prompt"]
    )
    assert summary["divergence_rate"] == pytest.approx(summary["near_tie"] / summary["positions_compared"])
    assert summary["hf_self_flip_rate"] == pytest.approx(summary["hf_self_flips"] / summary["hf_tokens"])
    for record in result["per_prompt"]:
        assert record["first_diff"] == common.first_difference(record["engine_tokens"], record["hf_tokens"])
        if record["first_diff"] >= 0:  # bf16 near-tie: HF itself barely prefers its token
            assert record["status"] == "near_tie" and 0.0 <= record["margin"] <= result["config"]["tie_margin"]
            assert record["engine_token"] == record["engine_tokens"][record["first_diff"]]
    text = (tmp_path / "gate_b_greedy_same.md").read_text()
    assert "**Verdict: PASS**" in text and "--tie-margin = 1.0" in text and "| prompt |" in text


def test_gate_b_fails_on_different_weights(tiny_dir, synthetic_dir, prompts_file, workdir, tmp_path):
    very_different = workdir / "tiny-very-different"
    if not very_different.exists():
        synth.make_synthetic_checkpoint(tiny_dir, str(very_different), rel_noise=0.3, seed=0)
    status, result = _gate_b(tiny_dir, prompts_file, tmp_path, "wrong", "--hf-model", str(very_different))
    assert status == 1 and result["pass"] is False
    assert "divergences_are_near_ties" in _failed(result)
    assert result["summary"]["mismatch"] >= 4 and result["summary"]["max_margin"] > 1.0
    assert result["summary"]["divergence_rate"] > 10 * max(result["summary"]["hf_self_flip_rate"], 0.001)
    assert "**Verdict: FAIL**" in (tmp_path / "gate_b_greedy_wrong.md").read_text()
    # the thresholds are the flags: a zero tie margin turns any non-tie divergence into a failure,
    # --min-identical can demand exact matches
    status, strict = _gate_b(tiny_dir, prompts_file, tmp_path, "strict", "--min-identical", "9")
    assert status == 1 and _failed(strict) == {"min_identical"}
    # a slightly different model (2% noise) diverges mostly at small margins: the optional rate rule catches it
    status, subtle = _gate_b(
        tiny_dir, prompts_file, tmp_path, "subtle", "--hf-model", synthetic_dir, "--tie-margin", "1e9", "--max-divergence-ratio", "5"
    )
    assert status == 1 and _failed(subtle) == {"divergence_rate"}
    # same weights in float32 (no bf16 noise on any platform): nothing diverges, the rate rule passes
    status, same = _gate_b(
        tiny_dir, prompts_file, tmp_path, "same_rate", "--max-divergence-ratio", "5", "--dtype", "float32", "--hf-dtype", "float32"
    )
    assert status == 0 and "divergence_rate" in {c["name"] for c in same["checks"]}
    assert same["summary"]["identical"] == 8 and same["summary"]["divergence_rate"] == 0.0


def test_gate_b_analysis_of_one_prompt(hf_tiny):
    prompt = [3, 14, 15, 92, 65, 35]
    hf_tokens = common.hf_greedy(hf_tiny, prompt, 16, EOS_IDS)
    same = gate_b_greedy.analyse_prompt(hf_tiny, prompt, list(hf_tokens), hf_tokens, 1.0)
    assert same["status"] == "identical" and same["first_diff"] == -1 and same["margin"] is None

    logits = common.hf_teacher_forced_logits(hf_tiny, prompt, hf_tokens)
    position = 5
    runner_up, worst = int(logits[position].topk(2).indices[1]), int(logits[position].argmin())
    for wrong, rank in ((runner_up, 1), (worst, VOCAB - 1)):
        engine_tokens = hf_tokens[:position] + [wrong] + hf_tokens[position + 1 :]
        record = gate_b_greedy.analyse_prompt(hf_tiny, prompt, engine_tokens, hf_tokens, tie_margin=1e9)
        assert record["first_diff"] == position and record["engine_token"] == wrong and record["hf_token"] == hf_tokens[position]
        assert record["margin"] == pytest.approx(float(logits[position].max() - logits[position][wrong]), abs=1e-5)
        assert record["engine_token_rank"] == rank and record["status"] == "near_tie"
        assert record["p_top"] > record["p_engine_token"]
        strict = gate_b_greedy.analyse_prompt(hf_tiny, prompt, engine_tokens, hf_tokens, tie_margin=record["margin"] / 2)
        assert strict["status"] == "MISMATCH"
    # one response a strict prefix of the other is a contract violation, not a near-tie
    cut = gate_b_greedy.analyse_prompt(hf_tiny, prompt, hf_tokens[:7], hf_tokens, 1.0)
    assert cut["status"] == "LENGTH_MISMATCH" and cut["first_diff"] == 7
    assert gate_b_greedy.bf16_ulp(20.0) == 0.125 and gate_b_greedy.bf16_ulp(1.0) == 2**-7


# ----------------------------------------------------------------------------
# gate 3c
# ----------------------------------------------------------------------------
def _gate_c(tiny_dir, ckpt, prompts_file, out_dir, tag, *extra) -> tuple[int, dict]:
    status = gate_c_reload.main(
        ["--engine", "reference", "--device", "cpu", "--model", tiny_dir, "--ckpt", ckpt, "--prompts-file", prompts_file,
         "--num-prompts", "6", "--n", "4", "--max-new-tokens", "40", "--fingerprint-prompts", "3",
         "--fingerprint-tokens", "16", "--out-dir", str(out_dir), "--tag", tag, *extra]
    )  # fmt: skip
    return status, _read(Path(out_dir) / f"gate_c_reload_{tag}.json")


def test_gate_c_passes_with_a_working_engine(tiny_dir, synthetic_dir, prompts_file, tmp_path):
    status, result = _gate_c(tiny_dir, synthetic_dir, prompts_file, tmp_path, "ok")
    assert status == 0 and result["pass"] is True and not _failed(result), _failed(result)
    names = [c["name"] for c in result["checks"]]
    for expected in (
        "checkpoint_differs_from_base", "build_matches_base", "reload1_bitwise", "reload1_tied",
        "reload1_pointers_unchanged", "reload1_base_no_longer_matches", "sample_1_response_contract",
        "reload2_bitwise_base", "reload2_checkpoint_no_longer_matches", "reload2_pointers_unchanged",
        "base_fingerprint_after_roundtrip", "sample_2_response_contract", "reload3_bitwise",
        "reload3_pointers_unchanged", "fingerprint_F2_equals_F1", "reloaded_mean_small",
        "base_after_roundtrip_mean_small", "base_after_roundtrip_same_range", "negative_control_clearly_larger",
        "tracking_follows_checkpoint_after_reload1", "tracking_follows_base_after_reload2",
    ):  # fmt: skip
        assert expected in names, expected
    assert result["negative_control"] == {
        "mode": "mean", "mean_rule_ok": True, "tracking_evaluable": True,
        "mean_rule_detail": result["negative_control"]["mean_rule_detail"],
    }  # fmt: skip

    stats, cfg = result["stats"], result["config"]
    assert cfg["num_sequences"] == 24 and cfg["n"] == 4
    for name in ("reloaded", "base_after_roundtrip", "negative_control", "negative_control_mirror", "hf_base_vs_ckpt"):
        assert stats[name]["count"] > 0 and stats[name]["num_nonfinite"] == 0
        assert stats[name]["max_prob_ratio"] == pytest.approx(math.exp(stats[name]["max"]))
    assert stats["reloaded"]["count"] == result["samples"]["sample_1"]["num_tokens"]
    assert stats["base_after_roundtrip"]["count"] == stats["negative_control"]["count"] == result["samples"]["sample_2"]["num_tokens"]
    # engine noise is small, the wrong-weights controls are an order of magnitude above it
    assert stats["reloaded"]["mean"] < 0.05 and stats["base_after_roundtrip"]["mean"] < 0.05
    assert stats["negative_control"]["mean"] > 5 * stats["reloaded"]["mean"]
    assert stats["negative_control_mirror"]["mean"] > 5 * stats["reloaded"]["mean"]
    # the engine follows the checkpoint after reload 1 and the base model after reload 2
    assert result["tracking"]["sample_1"]["beta"] == pytest.approx(1.0, abs=0.1)
    assert result["tracking"]["sample_2"]["beta"] == pytest.approx(0.0, abs=0.1)
    assert 0.8 < result["checkpoint_vs_base"]["fraction_changed"] <= 1.0

    prints = result["fingerprints"]
    assert prints["F1"]["sha256"] == prints["F2"]["sha256"] != prints["F0"]["sha256"] == prints["F0_roundtrip"]["sha256"]
    text = (tmp_path / "gate_c_reload_ok.md").read_text()
    assert "**Verdict: PASS**" in text and "negative_control" in text and "--neg-control-factor 3.0" in text
    assert not (tmp_path / "gate_c_reload_ok.samples.json").exists()  # only written on failure by default


def test_gate_c_creates_and_reuses_a_synthetic_checkpoint(tiny_dir, prompts_file, tmp_path):
    target = tmp_path / "made-on-the-fly"
    missing_status = None
    try:
        _gate_c(tiny_dir, str(target), prompts_file, tmp_path, "missing")
    except SystemExit as exc:  # without --synthetic a missing checkpoint is an error
        missing_status = str(exc)
    assert missing_status and "checkpoint not found" in missing_status and not target.exists()

    status, result = _gate_c(tiny_dir, str(target), prompts_file, tmp_path, "made", "--synthetic", "--save-samples", "always")
    assert status == 0 and result["checkpoint"]["kind"] == "synthetic (created)"
    assert (target / "model.safetensors").exists() and (target / synth.REPORT_NAME).exists()
    samples = _read(tmp_path / "gate_c_reload_made.samples.json")
    assert len(samples["prompts"]) == 24 and samples["prompts"][0] == samples["prompts"][3] != samples["prompts"][4]
    for name in ("sample_1", "sample_2"):
        assert [len(r) for r in samples[name]["responses"]] == [len(r) for r in samples[name]["hf_ckpt"]]
    status, result = _gate_c(tiny_dir, str(target), prompts_file, tmp_path, "reused", "--synthetic", "--no-sleep-while-scoring")
    assert status == 0 and result["checkpoint"]["kind"] == "synthetic (reused)" and result["config"]["sleep_while_scoring"] is False
    assert result["checkpoint"]["report"]["rel_noise"] == synth.DEFAULT_REL_NOISE


def test_gate_c_fails_when_the_checkpoint_is_the_base_model(tiny_dir, prompts_file, tmp_path):
    status, result = _gate_c(tiny_dir, tiny_dir, prompts_file, tmp_path, "vacuous")
    assert status == 1 and result["pass"] is False
    assert {
        "checkpoint_differs_from_base", "reload1_base_no_longer_matches",
        "reload2_checkpoint_no_longer_matches", "negative_control_clearly_larger",
    } <= _failed(result)  # fmt: skip
    assert (tmp_path / "gate_c_reload_vacuous.samples.json").exists()  # --save-samples on-fail


class _FaultyEngine:
    """A ReferenceEngine with one deliberate defect; everything else is delegated."""

    def __init__(self, inner, fault: str):
        self._inner = inner
        self._fault = fault
        self._loads = 0
        # "stale_forward": generation keeps using the weights the engine was built with
        self._frozen = copy.deepcopy(inner.model) if fault == "stale_forward" else None

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def load_weights(self, named_tensors):
        self._loads += 1
        if self._fault == "noop_reload":
            from serve.rl.base import LoadReport

            return LoadReport(num_params=0, num_sources=0, num_bytes=0, seconds=0.0)
        report = self._inner.load_weights(named_tensors)
        if self._fault == "realloc":  # correct values, but one parameter moved to new storage
            param = next(iter(self._inner.model.parameters()))
            param.data = param.data.clone()
        if self._fault == "corrupt_one_value" and self._loads == 2:  # one weight off by one bf16 step
            param = self._inner.model.model.norm.weight
            param.data.view(torch.int16)[0] += 1
        return report

    def generate(self, prompts, max_new_tokens, temperature=1.0, top_p=1.0, top_k=-1, *, eos_token_ids=None):
        if self._fault == "stale_forward":
            live, self._inner.model = self._inner.model, self._frozen
            try:
                return self._inner.generate(prompts, max_new_tokens, temperature, top_p, top_k, eos_token_ids=eos_token_ids)
            finally:
                self._inner.model = live
        result = self._inner.generate(prompts, max_new_tokens, temperature, top_p, top_k, eos_token_ids=eos_token_ids)
        if self._fault == "wrong_logprobs" and temperature > 0:  # e.g. the draft's log-probs instead of the target's
            result.logprobs = [[1.5 * x for x in row] for row in result.logprobs]
        if self._fault == "stale_state" and temperature == 0 and self._loads >= 3:  # something survived the round trip
            result.token_ids[0][-1] = (result.token_ids[0][-1] + 1) % (VOCAB - 8)
        if self._fault == "drops_last_token" and temperature > 0:  # e.g. the EOS stripped, or one token short
            result.token_ids = [tokens[:-1] or tokens for tokens in result.token_ids]
            result.logprobs = [row[: len(tokens)] for row, tokens in zip(result.logprobs, result.token_ids)]
        return result


@pytest.mark.parametrize(
    "fault, must_fail",
    [
        ("noop_reload", {"reload1_bitwise", "reload1_base_no_longer_matches", "tracking_follows_checkpoint_after_reload1"}),
        ("realloc", {"reload1_pointers_unchanged", "reload2_pointers_unchanged", "reload3_pointers_unchanged"}),
        ("corrupt_one_value", {"reload2_bitwise_base"}),
        ("stale_forward", {"reloaded_mean_small", "base_after_roundtrip_same_range", "negative_control_clearly_larger",
                           "tracking_follows_checkpoint_after_reload1"}),
        ("wrong_logprobs", {"reloaded_mean_small", "base_after_roundtrip_mean_small"}),
        ("stale_state", {"fingerprint_F2_equals_F1"}),
        ("drops_last_token", {"sample_1_response_contract", "sample_2_response_contract"}),
    ],
)
def test_gate_c_catches_a_broken_engine(fault, must_fail, tiny_dir, synthetic_dir, prompts_file, tmp_path, monkeypatch):
    real_build = common.build_engine
    monkeypatch.setattr(common, "build_engine", lambda kind, path, **opts: _FaultyEngine(real_build(kind, path, **opts), fault))
    status, result = _gate_c(tiny_dir, synthetic_dir, prompts_file, tmp_path, fault)
    assert status == 1 and result["pass"] is False
    assert must_fail <= _failed(result), f"{fault}: failed checks {_failed(result)}"
    assert "**Verdict: FAIL**" in (tmp_path / f"gate_c_reload_{fault}.md").read_text()


def test_gate_c_negative_control_when_the_checkpoint_is_close_to_the_base_model(
    tiny_dir, prompts_file, workdir, tmp_path, monkeypatch
):
    """A real early checkpoint differs from the base model by little more than bf16 noise. The mean-based negative
    control then has no power (no engine could satisfy it); the tracking coefficient still tells the models apart."""
    close = workdir / "tiny-close"
    if not close.exists():
        synth.make_synthetic_checkpoint(tiny_dir, str(close), rel_noise=1e-4, seed=0)
    assert 0.0 < _read(close / synth.REPORT_NAME)["bf16_fraction_changed"] < 0.2  # most bf16 weights are unchanged
    # demand far more than these two models differ (and do not depend on this platform's bf16 noise level)
    factor = ["--neg-control-factor", "100", "--noise-floor", "0.01"]

    status, result = _gate_c(tiny_dir, str(close), prompts_file, tmp_path, "close_mean", *factor)
    assert status == 1 and _failed(result) == {"negative_control_clearly_larger"}
    detail = next(c["detail"] for c in result["checks"] if c["name"] == "negative_control_clearly_larger")
    assert "satisfiable by ANY engine" in detail and "--neg-control tracking" in detail
    assert result["tracking"]["sample_1"]["beta"] > 0.8 and result["tracking"]["sample_2"]["beta"] < 0.2

    status, result = _gate_c(tiny_dir, str(close), prompts_file, tmp_path, "close_tracking", *factor, "--neg-control", "tracking")
    assert status == 0 and not _failed(result)
    assert result["negative_control"]["mode"] == "tracking" and result["negative_control"]["mean_rule_ok"] is False
    assert "negative_control_clearly_larger" not in {c["name"] for c in result["checks"]}
    assert "reported only" in (tmp_path / "gate_c_reload_close_tracking.md").read_text()

    # tracking mode needs evaluable coefficients; in mean mode an unevaluable coefficient is just not checked
    status, result = _gate_c(
        tiny_dir, str(close), prompts_file, tmp_path, "close_se", *factor, "--neg-control", "tracking", "--tracking-max-se", "1e-9"
    )
    assert status == 1 and _failed(result) == {"tracking_evaluable"}
    status, result = _gate_c(tiny_dir, str(close), prompts_file, tmp_path, "close_se_mean", "--neg-control-factor", "1.5", "--tracking-max-se", "1e-9")
    assert not any(c["name"].startswith("tracking") for c in result["checks"])
    assert result["negative_control"]["tracking_evaluable"] is False

    # ... and in tracking mode an engine that keeps generating with the base weights is still caught
    real_build = common.build_engine
    monkeypatch.setattr(
        common, "build_engine", lambda kind, path, **opts: _FaultyEngine(real_build(kind, path, **opts), "stale_forward")
    )
    status, result = _gate_c(tiny_dir, str(close), prompts_file, tmp_path, "close_stale", *factor, "--neg-control", "tracking")
    assert status == 1 and "tracking_follows_checkpoint_after_reload1" in _failed(result)
    assert result["tracking"]["sample_1"]["beta"] < 0.2


# ----------------------------------------------------------------------------
# gate 3c reference (stock vLLM): only the scoring phase can run without vllm
# ----------------------------------------------------------------------------
def _stand_in_vllm_samples(synthetic_dir, prompts_file, path, num_prompts=6, n=4, max_new_tokens=40) -> None:
    """A samples file in the schema phase 'sample' writes, produced by the
    reference engine instead of vLLM (vLLM is not available on CPU)."""
    prompts = common.repeat_interleave(common.load_prompts_file(prompts_file)[:num_prompts], n)
    engine = common.build_engine("reference", synthetic_dir, dtype="bfloat16", device="cpu", seed=3)
    result = engine.generate(prompts, max_new_tokens=max_new_tokens, temperature=1.0, eos_token_ids=EOS_IDS)
    common.write_json(
        str(path),
        {
            "kind": "gate_c_vllm_reference_samples", "ckpt": synthetic_dir, "vllm_version": "stand-in",
            "temperature": 1.0, "max_new_tokens": max_new_tokens, "num_prompts": num_prompts, "n": n,
            "prompts_meta": {"source": "prompts-file"}, "prompts": prompts, "responses": result.token_ids,
            "logprobs": result.logprobs, "finish_reasons": result.finish_reasons, "seconds": 1.0,
        },
    )  # fmt: skip


def test_vllm_reference_scoring_phase_and_side_by_side(tiny_dir, synthetic_dir, prompts_file, tmp_path):
    samples = tmp_path / "gate_c_vllm_reference.samples.json"
    _stand_in_vllm_samples(synthetic_dir, prompts_file, samples)
    status = gate_c_vllm_reference.main(
        ["--phase", "score", "--ckpt", synthetic_dir, "--device", "cpu", "--out-dir", str(tmp_path)]
    )
    assert status == 0
    reference = _read(tmp_path / "gate_c_vllm_reference.json")
    stats = reference["stats"]["engine_vs_trainer"]
    assert reference["config"]["num_sequences"] == 24 and stats["count"] == reference["sampling"]["num_tokens"]
    assert 0.0 <= stats["mean"] < 0.05 and stats["num_nonfinite"] == 0
    assert "stock vLLM" in (tmp_path / "gate_c_vllm_reference.md").read_text()

    # scoring against a different checkpoint than the samples came from is refused
    with pytest.raises(SystemExit, match="was sampled from"):
        gate_c_vllm_reference.main(["--phase", "score", "--ckpt", tiny_dir, "--device", "cpu", "--out-dir", str(tmp_path)])

    status, result = _gate_c(
        tiny_dir, synthetic_dir, prompts_file, tmp_path, "with_ref", "--vllm-reference", str(tmp_path / "gate_c_vllm_reference.json")
    )
    assert status == 0
    side = result["vllm_reference"]
    assert side["same_checkpoint"] is True and side["stats"] == stats
    assert side["mean_ratio"] == pytest.approx(result["stats"]["reloaded"]["mean"] / stats["mean"])
    assert "Side by side with stock vLLM" in (tmp_path / "gate_c_reload_with_ref.md").read_text()


def test_vllm_reference_rejects_unusable_samples(synthetic_dir, prompts_file, tmp_path):
    samples = tmp_path / "gate_c_vllm_reference_bad.samples.json"
    _stand_in_vllm_samples(synthetic_dir, prompts_file, samples)
    data = _read(samples)
    data["logprobs"][2] = data["logprobs"][2][:-1]
    samples.write_text(json.dumps(data))
    with pytest.raises(SystemExit, match="different number of tokens"):
        gate_c_vllm_reference.main(
            ["--phase", "score", "--ckpt", synthetic_dir, "--device", "cpu", "--out-dir", str(tmp_path), "--tag", "bad"]
        )


@pytest.mark.skipif(importlib.util.find_spec("vllm") is not None, reason="vllm is installed: phase 'sample' would start it")
def test_vllm_reference_reports_a_failed_sampling_process(synthetic_dir, prompts_file, tmp_path):
    status = gate_c_vllm_reference.main(
        ["--ckpt", synthetic_dir, "--device", "cpu", "--prompts-file", prompts_file, "--num-prompts", "2", "--n", "1",
         "--max-new-tokens", "4", "--out-dir", str(tmp_path), "--skip-preflight"]
    )  # fmt: skip
    assert status != 0  # the child process cannot import vllm; no result file is fabricated
    assert not (tmp_path / "gate_c_vllm_reference.json").exists()


# ----------------------------------------------------------------------------
# gate 3d
# ----------------------------------------------------------------------------
def _verl_log(kind: str, steps: int = 3, *, drop_step=None, prefix="sparsespec/", zero_reward=False, accept=None, drop_column=None) -> str:
    """Console output in verl's format (utils/logger/aggregate_logger.py: concat_dict_to_str)."""
    lines = ["Training Progress:   0%|          | 0/3 [00:00<?, ?it/s]"]
    for step in range(1, steps + 1):
        if step == drop_step:
            continue
        gen_s = 40.0 + step if kind != "vllm" else 95.0 + step
        data = {
            "training/global_step": step,
            "critic/score/mean": 0.0 if (zero_reward or step == 1) else 0.1 + 0.01 * step,
            "response_length/mean": 2800.5 + step,
            "actor/pg_loss": -1.5e-05 * step,
            "timing_s/gen": gen_s + 6.5,
            "timing_s/update_actor": 60.25,
            "timing_s/step": gen_s + 80.125,
        }
        if kind != "vllm":
            speculative = kind == "sparsespec"
            engine = {
                "accept_len_mean": accept if accept is not None else (5.5 + 0.1 * step if speculative else 0.0),
                "accept_rate": 0.62 if speculative else 0.0,
                "num_cycles": 250000.0 + step if speculative else 0.0,
                "gen_s": gen_s, "weight_sync_s": 0.85, "wake_s": 3.5, "graph_capture_s": 3.25, "sleep_s": 0.05,
            }  # fmt: skip
            engine.pop(drop_column, None)
            data.update({prefix + k: v for k, v in engine.items()})
        lines.append("(TaskRunner pid=1) " + " - ".join([f"step:{step}"] + [f"{k}:{pprint.pformat(v)}" for k, v in data.items()]))
    return "\n".join(lines) + "\n"


def _timing_csv(tmp_path: Path, name: str, log_text: str) -> str:
    """Log -> CSV through the baseline's own parser, as run_seed_sparsespec.sh does."""
    log = tmp_path / f"{name}.log"
    log.write_text(log_text)
    csv_path = tmp_path / f"{name}.timing.csv"
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "baseline" / "parse_timing.py"), str(log), "-o", str(csv_path)],
        check=True, capture_output=True,
    )  # fmt: skip
    return str(csv_path)


def _gate_d(csv_path, out_dir, *extra) -> tuple[int, dict]:
    status = gate_d_smoke_check.main(["--csv", csv_path, "--out-dir", str(out_dir), *extra])
    name = os.path.basename(csv_path).replace(".timing.csv", "")
    return status, _read(Path(out_dir) / f"gate_d_smoke_{name}.json")


def test_gate_d_passes_on_good_runs_and_compares_with_the_baseline(tmp_path):
    good = _timing_csv(tmp_path, "good_sparsespec", _verl_log("sparsespec"))
    baseline = _timing_csv(tmp_path, "vllm_baseline", _verl_log("vllm", steps=10))
    status, result = _gate_d(good, tmp_path, "--steps", "3", "--engine", "sparsespec", "--baseline-csv", baseline)
    assert status == 0 and result["pass"] is True and not _failed(result)
    assert result["summary"]["timing_s/gen"]["mean"] == pytest.approx((47.5 + 48.5 + 49.5) / 3)
    assert result["summary"]["sparsespec/accept_len_mean"]["mean"] == pytest.approx(5.7)
    comparison = result["baseline_comparison"]
    assert comparison["steps"] == [1, 2, 3]
    gen = comparison["metrics"]["timing_s/gen"]
    assert gen["baseline_mean"] == pytest.approx(103.5) and gen["this_over_baseline"] == pytest.approx(48.5 / 103.5)
    text = (tmp_path / "gate_d_smoke_good_sparsespec.md").read_text()
    assert "**Verdict: PASS**" in text and "Step time vs the vLLM baseline" in text

    reference = _timing_csv(tmp_path, "good_reference", _verl_log("reference"))
    status, result = _gate_d(reference, tmp_path, "--steps", "3", "--engine", "reference")
    assert status == 0 and not _failed(result)


@pytest.mark.parametrize(
    "name, log_kwargs, args, must_fail",
    [
        ("missing_step", {"drop_step": 2}, ["--steps", "3"], {"all_steps_present", "present_finite[timing_s/gen]"}),
        ("missing_step_no_flag", {"drop_step": 2}, [], {"all_steps_present"}),
        ("run_too_short", {}, ["--steps", "5"], {"all_steps_present"}),
        ("zero_reward", {"zero_reward": True}, ["--steps", "3"], {"reward_nonzero_on_some_step"}),
        ("nothing_accepted", {"accept": 1.0}, ["--steps", "3"], {"sparsespec_rule[sparsespec/accept_len_mean]"}),
        ("trainer_hunk_missing", {"prefix": "timing_s/sparsespec/"}, ["--steps", "3"],
         {"present_finite[sparsespec/gen_s]", "present_finite[sparsespec/wake_s]", "rollout_time_contains_engine_time"}),
        ("column_missing", {"drop_column": "sleep_s"}, ["--steps", "3"], {"present_finite[sparsespec/sleep_s]"}),
        ("wrong_engine_flag", {}, ["--steps", "3", "--engine", "reference"],
         {"reference_rule[sparsespec/accept_len_mean]", "reference_rule[sparsespec/num_cycles]"}),
    ],
)  # fmt: skip
def test_gate_d_fails_on_bad_runs(name, log_kwargs, args, must_fail, tmp_path):
    csv_path = _timing_csv(tmp_path, name, _verl_log("sparsespec", **log_kwargs))
    status, result = _gate_d(csv_path, tmp_path, *(args if "--engine" in args else [*args, "--engine", "sparsespec"]))
    assert status == 1 and result["pass"] is False
    assert must_fail <= _failed(result), _failed(result)


def test_gate_d_reference_run_checked_as_sparsespec_fails_and_missing_csv_is_an_error(tmp_path):
    reference = _timing_csv(tmp_path, "reference_run", _verl_log("reference"))
    status, result = _gate_d(reference, tmp_path, "--steps", "3", "--engine", "sparsespec")
    assert status == 1 and "sparsespec_rule[sparsespec/accept_len_mean]" in _failed(result)
    assert gate_d_smoke_check.main(["--csv", str(tmp_path / "nope.timing.csv"), "--out-dir", str(tmp_path)]) == 1
    with pytest.raises(SystemExit):
        gate_d_smoke_check.main(["--out-dir", str(tmp_path)])  # neither --csv nor --exp-name


# ----------------------------------------------------------------------------
# the scripts run both as files and as modules, from any working directory
# ----------------------------------------------------------------------------
def test_scripts_start_as_file_and_as_module(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    good = _timing_csv(tmp_path, "cli", _verl_log("sparsespec"))
    as_file = subprocess.run(
        [sys.executable, str(REPO_ROOT / "integration" / "gates" / "gate_d_smoke_check.py"), "--csv", good, "--steps", "3",
         "--out-dir", str(tmp_path)],
        cwd=str(tmp_path), env=env, capture_output=True, text=True,
    )  # fmt: skip
    assert as_file.returncode == 0, as_file.stdout + as_file.stderr
    assert "[gate 3d] PASS" in as_file.stdout
    as_module = subprocess.run(
        [sys.executable, "-m", "integration.gates.gate_d_smoke_check", "--csv", good, "--steps", "4", "--out-dir", str(tmp_path)],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
    )  # fmt: skip
    assert as_module.returncode == 1 and "FAIL all_steps_present" in as_module.stdout
    for script in ("gate_b_greedy", "gate_c_reload", "gate_c_vllm_reference", "make_synthetic_ckpt"):
        helped = subprocess.run(
            [sys.executable, str(REPO_ROOT / "integration" / "gates" / f"{script}.py"), "--help"],
            cwd=str(tmp_path), env=env, capture_output=True, text=True,
        )  # fmt: skip
        assert helped.returncode == 0 and "usage:" in helped.stdout, helped.stderr


def test_gate_modules_do_not_import_gpu_only_packages(tiny_dir):
    """Importing the gates (and building the reference engine) must not pull in
    serve.rl.engine / flashinfer / vllm: the same modules run in the
    stock-vLLM baseline env and on CPU-only machines."""
    code = (
        "import sys\n"
        "from integration.gates import common, gate_b_greedy, gate_c_reload, gate_c_vllm_reference, "
        "gate_d_smoke_check, make_synthetic_ckpt\n"
        f"engine = common.build_engine('reference', {tiny_dir!r}, dtype='float32', device='cpu')\n"
        "engine.close()\n"
        "common.eos_ids_from_checkpoint(" + repr(tiny_dir) + ")\n"
        "bad = [m for m in ('serve.rl.engine', 'serve.rl.scheduler', 'flashinfer', 'vllm') if m in sys.modules]\n"
        "print('IMPORTED:' + ','.join(bad))\n"
    )
    run = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "IMPORTED:\n" in run.stdout + "\n", run.stdout
    with pytest.raises(ValueError):
        common.build_engine("vllm", tiny_dir)
