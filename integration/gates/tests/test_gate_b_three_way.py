"""CPU tests of gate 3b's three-way mode and of the stock-vLLM producer script.

    PYTHONPATH=. python -m pytest integration/gates/tests/test_gate_b_three_way.py -q

Setup: a tiny random Qwen3 (float32, tied embeddings) written to a temp dir;
``ReferenceEngine`` stands in for SparseSpec; the vLLM side is written by the
REAL ``gate_b_vllm_greedy.main`` with only ``run_vllm`` replaced by a stand-in
that runs the tiny model greedily (uncached argmax) and returns objects with
the fields of vLLM 0.9.1's ``RequestOutput`` / ``CompletionOutput`` /
``Logprob``, the per-step log-prob dicts built as ``v1/sample/sampler.py``
(``gather_logprobs``) and ``v1/engine/logprobs.py`` (``_make_logprob_dict``)
build them. Controlled perturbations flip one token, either at a genuine
near-tie or against a large margin, in the engine or in the "vLLM" file.

Not exercised here: ``gate_b_vllm_greedy.run_vllm`` itself (needs vLLM and a
GPU) and the SparseSpec engine.

No ``__init__.py`` next to this file (see test_gates_cpu.py).
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from integration.gates import common  # noqa: E402
from integration.gates import gate_b_greedy as gb  # noqa: E402
from integration.gates import gate_b_vllm_greedy as gv  # noqa: E402
from serve.rl.base import GenerationResult  # noqa: E402

VOCAB = 512
EOS_IDS = [VOCAB - 2, VOCAB - 1]
PAD_ID = VOCAB - 2
NUM_PROMPTS = 16
MAX_NEW = 40
_REAL_BUILD_ENGINE = common.build_engine  # before any test patches it


# ----------------------------------------------------------------------------
# stand-ins for vLLM 0.9.1's output objects (field names and order as in the sources)
# ----------------------------------------------------------------------------
@dataclasses.dataclass
class Logprob:  # vllm/sequence.py
    logprob: float
    rank: Optional[int] = None
    decoded_token: Optional[str] = None


@dataclasses.dataclass
class CompletionOutput:  # vllm/outputs.py
    index: int
    text: str
    token_ids: list
    cumulative_logprob: Optional[float]
    logprobs: Optional[list]
    finish_reason: Optional[str] = None
    stop_reason: Any = None
    lora_request: Any = None


class RequestOutput:  # vllm/outputs.py (constructor signature of 0.9.1)
    def __init__(self, request_id, prompt, prompt_token_ids, prompt_logprobs, outputs, finished, metrics=None,
                 lora_request=None, encoder_prompt=None, encoder_prompt_token_ids=None, num_cached_tokens=None, *,
                 multi_modal_placeholders=None, kv_transfer_params=None):  # fmt: skip
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.multi_modal_placeholders = multi_modal_placeholders or {}
        self.prompt_logprobs = prompt_logprobs
        self.outputs = outputs
        self.finished = finished
        self.metrics = metrics
        self.lora_request = lora_request
        self.encoder_prompt = encoder_prompt
        self.encoder_prompt_token_ids = encoder_prompt_token_ids
        self.num_cached_tokens = num_cached_tokens
        self.kv_transfer_params = kv_transfer_params


def vllm_step_dict(row: torch.Tensor, num_logprobs: int = 2) -> tuple[int, dict]:
    """(sampled token, per-step dict) for one float32 logits row, as vLLM V1 builds them at T=0."""
    raw = row.log_softmax(dim=-1, dtype=torch.float32)  # Sampler.compute_logprobs on the raw logits
    sampled = int(row.to(torch.float32).argmax())  # greedy_sample
    top_lp, top_ids = torch.topk(raw, num_logprobs, dim=-1)  # gather_logprobs
    token_lp = raw[sampled]
    rank = int((raw >= token_lp).sum())
    ids = [sampled] + [int(t) for t in top_ids]
    lps = [float(token_lp)] + [float(x) for x in top_lp]
    ranks = [rank] + list(range(1, num_logprobs + 1))
    # _make_logprob_dict: sampled first, then the top-k (a duplicate key keeps its first position, new value)
    return sampled, {t: Logprob(logprob=lp, rank=r, decoded_token=None) for t, lp, r in zip(ids, lps, ranks)}


@torch.no_grad()
def fake_vllm_outputs(model, prompts, max_tokens: int, eos, perturb: dict | None = None) -> list[RequestOutput]:
    """Greedy (uncached full forward per step) with optional "numerical noise" at chosen steps.

    ``perturb[i] = (position, token)``: at that step the token's logit is raised just above the top
    logit (by 0.01), as if this engine's arithmetic preferred it; vLLM then reports it as its top-1.
    """
    perturb = perturb or {}
    outputs = []
    for i, prompt in enumerate(prompts):
        tokens, steps = [], []
        reason = "length"
        for j in range(max_tokens):
            ids = torch.tensor([list(prompt) + tokens])
            row = model(input_ids=ids, use_cache=False).logits[0, -1].to(torch.float32).clone()
            if i in perturb and perturb[i][0] == j:
                row[perturb[i][1]] = row.max() + 0.01
            token, step = vllm_step_dict(row)
            tokens.append(token)
            steps.append(step)
            if token in eos:
                reason = "stop"
                break
        completion = CompletionOutput(
            index=0, text="", token_ids=tokens, cumulative_logprob=sum(s[t].logprob for s, t in zip(steps, tokens)),
            logprobs=steps, finish_reason=reason, stop_reason=None,
        )  # fmt: skip
        outputs.append(RequestOutput(str(i), None, list(prompt), None, [completion], True))
    return outputs


# ----------------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------------
def _write_tiny_qwen3(path: Path, seed: int = 0) -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, max_position_embeddings=512, tie_word_embeddings=True,
        initializer_range=0.5, eos_token_id=EOS_IDS[1], pad_token_id=PAD_ID, bos_token_id=None,
    )  # fmt: skip
    state = torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        model = Qwen3ForCausalLM(config).to(torch.float32).eval()
    finally:
        torch.set_rng_state(state)
    model.generation_config.eos_token_id = list(EOS_IDS)
    model.generation_config.pad_token_id = PAD_ID
    model.save_pretrained(str(path), safe_serialization=True)


@pytest.fixture(scope="module")
def work(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("gate_b_three_way")


@pytest.fixture(scope="module")
def tiny(work) -> str:
    path = work / "tiny-qwen3"
    _write_tiny_qwen3(path)
    return str(path)


def _prompts(seed: int = 0) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(0, VOCAB - 8) for _ in range(rng.randrange(5, 40))] for _ in range(NUM_PROMPTS)]


@pytest.fixture(scope="module")
def prompts_file(work) -> str:
    path = work / "prompts.json"
    path.write_text(json.dumps(_prompts()))
    return str(path)


@pytest.fixture(scope="module")
def hf(tiny):
    return common.load_hf_model(tiny, "cpu", "float32")


@pytest.fixture(scope="module")
def hf_greedy(hf):
    """HF's greedy responses (cached loop, as the gate) and the teacher-forced logits along them."""
    out = []
    for prompt in _prompts():
        tokens = common.hf_greedy(hf, prompt, MAX_NEW, EOS_IDS)
        out.append((tokens, common.hf_teacher_forced_logits(hf, prompt, tokens)))
    return out


def _pick(hf_greedy, *, near_tie: bool, avoid: int | None = None) -> tuple[int, int, int, float]:
    """(prompt, position, token, HF margin): the runner-up at the smallest top-2 gap, or a far token."""
    best = None
    for i, (tokens, logits) in enumerate(hf_greedy):
        if i == avoid:
            continue
        for j in range(2, len(tokens) - 2):
            values, ids = logits[j].topk(2)
            if int(ids[0]) != tokens[j]:
                continue  # HF self-flip here: not a clean test position
            gap = float(values[0] - values[1])
            if near_tie:
                if int(ids[1]) in EOS_IDS:
                    continue
                cand = (gap, i, j, int(ids[1]))
                best = cand if best is None or cand < best else best
            else:
                far = [t for t in torch.argsort(logits[j]).tolist() if t not in EOS_IDS][0]
                margin = float(values[0] - logits[j][far])
                cand = (-margin, i, j, far)
                best = cand if best is None or cand < best else best
    _, i, j, token = best
    margin = float(hf_greedy[i][1][j].max() - hf_greedy[i][1][j][token])
    return i, j, token, margin


@pytest.fixture(scope="module")
def near_tie(hf_greedy):
    i, j, token, margin = _pick(hf_greedy, near_tie=True)
    assert margin < 0.5, f"tiny model has no near-tie below 0.5 logits (smallest {margin}); the tests need one"
    return i, j, token, margin


@pytest.fixture(scope="module")
def far(hf_greedy):
    i, j, token, margin = _pick(hf_greedy, near_tie=False)
    assert margin > 5.0
    return i, j, token, margin


def _write_vllm_json(monkeypatch, hf, tiny, prompts_file, out: Path, perturb=None, *extra) -> tuple[int, dict, dict]:
    seen = {}

    def stand_in(llm_kwargs, params_kwargs, prompts):
        seen["llm_kwargs"], seen["params_kwargs"], seen["prompts"] = llm_kwargs, params_kwargs, prompts
        outputs = fake_vllm_outputs(hf, prompts, params_kwargs["max_tokens"], EOS_IDS, perturb)
        return outputs, {"engine_class": "vllm.v1.engine.llm_engine.LLMEngine", "engine_version": "V1", "stand_in": True}

    monkeypatch.setattr(gv, "run_vllm", stand_in)
    status = gv.main(
        ["--out", str(out), "--model", tiny, "--prompts-file", prompts_file, "--num-prompts", str(NUM_PROMPTS),
         "--max-new-tokens", str(MAX_NEW), "--skip-preflight", *extra]
    )  # fmt: skip
    return status, json.loads(out.read_text()), seen


@pytest.fixture(scope="module")
def vllm_files(work, hf, tiny, prompts_file, near_tie, far):
    """The synthetic vLLM files: unperturbed, near-tie flip, large-margin flip."""
    mp = pytest.MonkeyPatch()
    try:
        files = {}
        for name, perturb in (("clean", None), ("near_tie", {near_tie[0]: near_tie[1:3]}), ("far", {far[0]: far[1:3]})):
            path = work / f"vllm_{name}.json"
            status, _, _ = _write_vllm_json(mp, hf, tiny, prompts_file, path, perturb)
            assert status == 0
            files[name] = str(path)
        return files
    finally:
        mp.undo()


class PerturbedEngine:
    """ReferenceEngine whose response to one prompt takes ``token`` at ``position`` and continues greedily."""

    def __init__(self, inner, prompts, flip):
        self.inner, self.prompts, self.flip = inner, prompts, flip

    def generate(self, prompts, max_new_tokens, temperature=1.0, top_p=1.0, top_k=-1, *, eos_token_ids=None,
                 return_logprobs=True):  # fmt: skip
        res = self.inner.generate(prompts, max_new_tokens, temperature, eos_token_ids=eos_token_ids)
        tokens, reasons = [list(t) for t in res.token_ids], list(res.finish_reasons)
        for i, (pos, token) in self.flip.items():
            head = tokens[i][:pos] + [token]
            cont = self.inner.generate([list(prompts[i]) + head], max_new_tokens - pos - 1, temperature,
                                       eos_token_ids=eos_token_ids)  # fmt: skip
            tokens[i] = head + list(cont.token_ids[0])
            reasons[i] = cont.finish_reasons[0]
        return GenerationResult(tokens, [[0.0] * len(t) for t in tokens], reasons, dict(res.stats))

    def sleep(self):
        self.inner.sleep()

    def close(self):
        self.inner.close()


def _gate(tmp: Path, tiny, prompts_file, tag, *extra, monkeypatch=None, flip=None, capsys=None) -> tuple[int, dict | None, str]:
    if flip:

        def build(kind, model, **opts):
            return PerturbedEngine(_REAL_BUILD_ENGINE(kind, model, **opts), _prompts(), flip)

        monkeypatch.setattr(common, "build_engine", build)
    status = gb.main(
        ["--engine", "reference", "--device", "cpu", "--dtype", "float32", "--hf-dtype", "float32", "--model", tiny,
         "--prompts-file", prompts_file, "--num-prompts", str(NUM_PROMPTS), "--max-new-tokens", str(MAX_NEW),
         "--out-dir", str(tmp), "--tag", tag, *extra]
    )  # fmt: skip
    out = capsys.readouterr().out if capsys else ""
    path = tmp / f"gate_b_greedy_{tag}.json"
    return status, (json.loads(path.read_text()) if path.exists() else None), out


def _failed(result) -> dict:
    return {f["name"]: f["blame"] for f in result["failed_checks"]}


def _pair(result, pair) -> dict:
    return result["three_way_result"]["pairs"][pair]


# ----------------------------------------------------------------------------
# vLLM producer script
# ----------------------------------------------------------------------------
def test_vllm_step_extraction_follows_the_vllm_dict_layout():
    row = torch.tensor([0.0, 5.0, 4.5, -1.0, 4.5])
    token, step = vllm_step_dict(row)
    assert token == 1 and len(step) == 2 and list(step)[0] == 1  # sampled token in the top-2: two entries
    out = gv._step_top2(step, token, "t")
    assert out["top2"][0] == 1 and out["top2"][1] in (2, 4)
    assert out["margin"] == pytest.approx(0.5, abs=1e-6)  # difference of log-probs == logit margin
    assert out["sampled_rank"] == 1 and out["greedy_ok"]
    # a sampled token outside the top-2 (not greedy): three entries, rank from (logprobs >= lp).sum()
    raw = row.log_softmax(-1)
    step3 = {3: Logprob(float(raw[3]), 5), 1: Logprob(float(raw[1]), 1), 2: Logprob(float(raw[2]), 2)}
    bad = gv._step_top2(step3, 3, "t")
    assert bad["top2"] == [1, 2] and not bad["greedy_ok"] and bad["sampled_rank"] == 5
    with pytest.raises(gv.VllmOutputError, match="at least 2"):
        gv._step_top2({1: Logprob(-0.1, 1)}, 1, "t")
    with pytest.raises(gv.VllmOutputError, match="no log-prob entry"):
        gv._step_top2(step, 7, "t")


def test_vllm_extract_outputs_checks_structure(hf):
    prompts = _prompts()[:3]
    outputs = fake_vllm_outputs(hf, prompts, 6, EOS_IDS)
    got = gv.extract_outputs(outputs, prompts)
    assert got["greedy_anomalies"] == [] and len(got["token_ids"]) == 3
    for tokens, top2, margins, lps in zip(got["token_ids"], got["top2_token_ids"], got["margins"], got["sampled_logprobs"]):
        assert [t[0] for t in top2] == tokens  # greedy: the sampled token is the top-1
        assert all(m >= 0 for m in margins) and all(lp <= 0 for lp in lps)
    with pytest.raises(gv.VllmOutputError, match="different prompt"):
        gv.extract_outputs(outputs, [prompts[1], prompts[0], prompts[2]])
    with pytest.raises(gv.VllmOutputError, match="outputs for"):
        gv.extract_outputs(outputs[:2], prompts)
    broken = fake_vllm_outputs(hf, prompts[:1], 4, EOS_IDS)
    broken[0].outputs[0].logprobs = broken[0].outputs[0].logprobs[:-1]
    with pytest.raises(gv.VllmOutputError, match="log-prob dicts"):
        gv.extract_outputs(broken, prompts[:1])


def test_vllm_script_writes_a_complete_token_file(monkeypatch, hf, tiny, prompts_file, tmp_path, capsys):
    status, data, seen = _write_vllm_json(monkeypatch, hf, tiny, prompts_file, tmp_path / "v.json")
    out = capsys.readouterr().out
    assert status == 0 and data["pass"] is True and "[gate 3b vllm] PASS" in out
    # engine configured like the baseline arm, greedy with top-2 log-probs
    llm, params = seen["llm_kwargs"], seen["params_kwargs"]
    assert llm["dtype"] == "bfloat16" and llm["enforce_eager"] is True and llm["seed"] == 0
    assert llm["enable_prefix_caching"] and llm["enable_chunked_prefill"] and llm["max_num_batched_tokens"] == 10240
    assert llm["max_model_len"] == 1024 + 8192 and llm["model"] == os.path.abspath(tiny)
    assert params == {"n": 1, "temperature": 0.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0, "ignore_eos": False,
                      "max_tokens": MAX_NEW, "logprobs": 2, "detokenize": False}  # fmt: skip
    assert seen["prompts"] == _prompts()
    # the file: validated by the reader the gate uses, with every promised field
    checked = common.read_greedy_tokens(str(tmp_path / "v.json"), side="vllm")
    assert checked["prompts"] == _prompts() and checked["prompts_sha256"] == common.prompts_sha256(_prompts())
    for key in ("top2_token_ids", "top2_logprobs", "margins", "sampled_logprobs", "sampled_ranks", "finish_reasons",
                "stop_reasons", "engine_kwargs", "sampling_kwargs", "engine_info", "versions", "logprobs_definition"):
        assert key in data, key  # fmt: skip
    assert data["margin_is_logit_margin"] is True and "raw" in data["logprobs_definition"]
    assert data["engine_info"]["engine_version"] == "V1" and "V1" in data["engine"]
    assert set(data["versions"]) == {"vllm", "torch", "transformers"} and data["versions"]["torch"] == torch.__version__
    assert data["temperature"] == 0.0 and data["max_new_tokens"] == MAX_NEW and data["eos_token_ids"] == EOS_IDS
    assert data["greedy_anomalies"] == [] and data["contract_notes"] == []


def test_vllm_script_fails_on_a_non_greedy_step(monkeypatch, tiny, prompts_file, tmp_path, hf, capsys):
    seen = {}

    def stand_in(llm_kwargs, params_kwargs, prompts):
        outputs = fake_vllm_outputs(hf, prompts, params_kwargs["max_tokens"], EOS_IDS)
        seen["prompt"] = next(i for i, o in enumerate(outputs) if len(o.outputs[0].token_ids) >= 5)
        completion = outputs[seen["prompt"]].outputs[0]
        step = completion.logprobs[3]
        worst = min(step, key=lambda t: step[t].logprob)
        completion.logprobs[3] = {**step, 7: Logprob(step[worst].logprob - 3.0, 400)}
        completion.token_ids = completion.token_ids[:3] + [7] + completion.token_ids[4:]
        return outputs, {"engine_version": "V1"}

    monkeypatch.setattr(gv, "run_vllm", stand_in)
    status = gv.main(["--out", str(tmp_path / "v.json"), "--model", tiny, "--prompts-file", prompts_file,
                      "--max-new-tokens", str(MAX_NEW), "--skip-preflight"])  # fmt: skip
    data = json.loads((tmp_path / "v.json").read_text())
    assert status == 1 and data["pass"] is False and "[gate 3b vllm] FAIL" in capsys.readouterr().out
    assert data["greedy_anomalies"][0]["prompt"] == seen["prompt"] and data["greedy_anomalies"][0]["position"] == 3


def test_vllm_script_starts_as_a_file_and_imports_no_gpu_package(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    helped = subprocess.run(
        [sys.executable, str(REPO_ROOT / "integration" / "gates" / "gate_b_vllm_greedy.py"), "--help"],
        cwd=str(tmp_path), env=env, capture_output=True, text=True,
    )  # fmt: skip
    assert helped.returncode == 0 and "--gpu-memory-utilization" in helped.stdout, helped.stderr
    for flag in ("--out ", "--out-dir", "--model", "--num-prompts", "--max-new-tokens", "--prompts-file", "--data",
                 "--prompt-seed", "--seed", "--skip-preflight", "--require-free-gb"):
        assert flag in helped.stdout, flag  # fmt: skip
    code = (
        "import sys\nfrom integration.gates import gate_b_vllm_greedy, gate_b_greedy\n"
        "bad = [m for m in ('serve.rl.engine', 'serve.rl.scheduler', 'flashinfer', 'triton', 'vllm') if m in sys.modules]\n"
        "print('IMPORTED:' + ','.join(bad))\n"
    )
    run = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert run.returncode == 0 and "IMPORTED:\n" in run.stdout + "\n", run.stdout + run.stderr
    helped = subprocess.run(
        [sys.executable, str(REPO_ROOT / "integration" / "gates" / "gate_b_greedy.py"), "--help"],
        cwd=str(tmp_path), env=env, capture_output=True, text=True,
    )  # fmt: skip
    assert helped.returncode == 0
    for flag in ("--vllm-json", "--compare-only", "--profile-rule", "--profile-alpha", "--stored-margins"):
        assert flag in helped.stdout, flag


# ----------------------------------------------------------------------------
# gate 3b: three-way verdicts
# ----------------------------------------------------------------------------
def test_identical_triples_pass(tiny, prompts_file, vllm_files, tmp_path, capsys):
    status, result, out = _gate(tmp_path, tiny, prompts_file, "same", "--vllm-json", vllm_files["clean"], capsys=capsys)
    assert status == 0 and result["pass"] is True and result["three_way"] is True, out
    assert result["profile_rule"] == "significance" and result["config"]["profile_rule"] == "significance"
    for pair, _, _ in gb.PAIRS:
        assert _pair(result, pair)["summary"]["identical"] == NUM_PROMPTS, pair
    profile = result["three_way_result"]["profile"]
    assert profile["engine"]["divergences"] == profile["vllm"]["divergences"] == 0 and profile["p_value"] == 1.0
    assert "[gate 3b] three-way PASS (--profile-rule=significance)" in out
    assert "R2 literal:" in out and "R2 significance:" in out and "--profile-rule = significance" in out
    for suffix in (".engine.json", ".hf.json", ".md"):
        assert (tmp_path / f"gate_b_greedy_same{suffix}").exists()
    md = (tmp_path / "gate_b_greedy_same.md").read_text()
    assert "**Verdict: PASS**" in md and "R2 rule: `significance`" in md and "## reference vs vLLM" in md


def test_without_vllm_json_the_gate_says_three_way_was_not_done(tiny, prompts_file, tmp_path, capsys):
    status, result, out = _gate(tmp_path, tiny, prompts_file, "two_way", capsys=capsys)
    assert status == 0 and result["pass"] is True and result["three_way"] is False
    assert "THREE-WAY COMPARISON NOT DONE" in result["three_way_note"] and "THREE-WAY COMPARISON NOT DONE" in out
    assert "THREE-WAY COMPARISON NOT DONE" in (tmp_path / "gate_b_greedy_two_way.md").read_text()
    assert "[gate 3b] PASS (two-way only" in out
    engine = common.read_greedy_tokens(str(tmp_path / "gate_b_greedy_two_way.engine.json"), side="engine")
    hf_side = common.read_greedy_tokens(str(tmp_path / "gate_b_greedy_two_way.hf.json"), side="hf")
    assert engine["token_ids"] == [r["engine_tokens"] for r in result["per_prompt"]]
    assert hf_side["token_ids"] == [r["hf_tokens"] for r in result["per_prompt"]]
    assert hf_side["hf_self_flips"] == [r["hf_self_flips"] for r in result["per_prompt"]]


def test_near_tie_flip_in_vllm_passes(tiny, prompts_file, vllm_files, near_tie, tmp_path, capsys):
    i, j, token, margin = near_tie
    status, result, out = _gate(tmp_path, tiny, prompts_file, "vt", "--vllm-json", vllm_files["near_tie"], capsys=capsys)
    assert status == 0 and result["pass"] is True, out
    vh, sv = _pair(result, "vllm_vs_hf"), _pair(result, "engine_vs_vllm")
    assert vh["summary"]["near_tie"] == 1 and vh["summary"]["identical"] == NUM_PROMPTS - 1
    rec = vh["per_prompt"][i]
    assert rec["first_diff"] == j and rec["token_a"] == token and rec["status"] == "near_tie"
    assert rec["margin"] == pytest.approx(margin, abs=1e-4)
    assert rec["vllm_margin"] == pytest.approx(0.01, abs=1e-4)  # vLLM's own top-2 margin from its file
    srec = sv["per_prompt"][i]  # engine == HF here, so S vs V sees the same flip with the same margin
    assert srec["first_diff"] == j and srec["margin"] == pytest.approx(margin, abs=1e-4) and srec["status"] == "near_tie"
    assert srec["shortfall_a"] == pytest.approx(0.0, abs=1e-6)
    profile = result["three_way_result"]["profile"]
    assert (profile["engine"]["divergences"], profile["vllm"]["divergences"]) == (0, 1) and profile["strict_ok"]


def test_near_tie_flip_in_engine_passes_significance_and_fails_strict(
    tiny, prompts_file, vllm_files, near_tie, tmp_path, monkeypatch, capsys
):
    i, j, token, margin = near_tie
    flip = {i: (j, token)}
    status, result, out = _gate(tmp_path, tiny, prompts_file, "et", "--vllm-json", vllm_files["clean"],
                                monkeypatch=monkeypatch, flip=flip, capsys=capsys)  # fmt: skip
    assert status == 0 and result["pass"] is True, out
    profile = result["three_way_result"]["profile"]
    assert (profile["engine"]["divergences"], profile["vllm"]["divergences"]) == (1, 0)
    assert profile["significance_ok"] and not profile["strict_ok"]
    assert profile["p_value"] == pytest.approx(profile["engine_exposure_share"])  # n = 1: P(X >= 1) = share
    assert "strict rule FAILS" in out  # the literal comparison is printed even when it does not decide
    # the three-way engine-vs-HF margin is the two-way one (same teacher-forced row)
    assert _pair(result, "engine_vs_hf")["per_prompt"][i]["margin"] == pytest.approx(result["per_prompt"][i]["margin"])
    # the same run under the literal rule: FAIL, and only R2 fails
    status, strict, out = _gate(tmp_path, tiny, prompts_file, "et_strict", "--vllm-json", vllm_files["clean"],
                                "--profile-rule", "strict", monkeypatch=monkeypatch, flip=flip, capsys=capsys)  # fmt: skip
    assert status == 1 and strict["pass"] is False and strict["profile_rule"] == "strict"
    assert _failed(strict) == {"R2_profile_strict": ["engine"]}
    assert "R2 rule: `strict`" in (tmp_path / "gate_b_greedy_et_strict.md").read_text()
    assert "R2 significance:" in out and "--profile-rule = strict" in out


def test_large_margin_flip_in_the_engine_fails_r1_and_names_the_pairs(
    tiny, prompts_file, vllm_files, far, tmp_path, monkeypatch, capsys
):
    i, j, token, margin = far
    status, result, out = _gate(tmp_path, tiny, prompts_file, "ef", "--vllm-json", vllm_files["clean"],
                                monkeypatch=monkeypatch, flip={i: (j, token)}, capsys=capsys)  # fmt: skip
    assert status == 1 and result["pass"] is False
    failed = _failed(result)
    assert failed["R1_near_ties_engine_vs_hf"] == ["engine"] and failed["R1_near_ties_engine_vs_vllm"] == ["engine"]
    assert "R1_near_ties_vllm_vs_hf" not in failed
    rec = _pair(result, "engine_vs_hf")["per_prompt"][i]
    assert rec["status"] == "MISMATCH" and rec["first_diff"] == j and rec["margin"] == pytest.approx(margin, abs=1e-4)
    assert "FAIL R1_near_ties_engine_vs_hf [ENGINE PROBLEM (reference)]" in out
    assert "reference vs HF" in out and "three-way FAIL (--profile-rule=significance)" in out


def test_large_margin_flip_in_vllm_fails_as_a_reference_problem(tiny, prompts_file, vllm_files, far, tmp_path, capsys):
    i, j, token, margin = far
    status, result, out = _gate(tmp_path, tiny, prompts_file, "vf", "--vllm-json", vllm_files["far"], capsys=capsys)
    assert status == 1 and result["pass"] is False
    failed = _failed(result)
    assert failed == {"R1_near_ties_vllm_vs_hf": ["vllm"], "R1_near_ties_engine_vs_vllm": ["vllm"]}
    detail = next(c["detail"] for c in result["checks"] if c["name"] == "R1_near_ties_vllm_vs_hf")
    assert detail.startswith("REFERENCE-ENGINE PROBLEM")
    rec = _pair(result, "engine_vs_vllm")["per_prompt"][i]
    assert rec["shortfall_a"] == pytest.approx(0.0, abs=1e-6) and rec["shortfall_b"] == pytest.approx(margin, abs=1e-4)
    assert "caused by the REFERENCE engines only" in out and "does not implicate reference" in out
    # the compact top-level per-pair view
    assert {p: v["pass"] for p, v in result["pairs"].items()} == {"engine_vs_hf": True, "vllm_vs_hf": False, "engine_vs_vllm": False}
    assert result["pairs"]["vllm_vs_hf"]["summary"]["earliest_divergence"] == j


def test_mismatched_inputs_are_refused(tiny, prompts_file, vllm_files, work, tmp_path, monkeypatch, hf, capsys):
    other = _prompts()
    other[3] = other[3][:4] + [(other[3][4] + 1) % (VOCAB - 8)] + other[3][5:]
    other_file = work / "prompts_other.json"
    other_file.write_text(json.dumps(other))
    status, _, _ = _write_vllm_json(monkeypatch, hf, tiny, str(other_file), tmp_path / "other.json")
    assert status == 0
    capsys.readouterr()
    status, result, out = _gate(tmp_path, tiny, prompts_file, "refused", "--vllm-json", str(tmp_path / "other.json"),
                                capsys=capsys)  # fmt: skip
    assert status == 2 and not (tmp_path / "gate_b_greedy_refused.engine.json").exists()  # refused before the engine ran
    assert result["pass"] is False and result["refused"] is True and "prompt 3 differs at token 4" in result["reason"]
    assert "REFUSED" in out and "prompt 3 differs at token 4" in out and "--prompt-seed" in out
    # a file whose prompts were edited after writing (hash no longer matches)
    data = json.loads(Path(vllm_files["clean"]).read_text())
    data["prompts"][0][0] += 1
    (tmp_path / "edited.json").write_text(json.dumps(data))
    status, _, out = _gate(tmp_path, tiny, prompts_file, "edited", "--vllm-json", str(tmp_path / "edited.json"), capsys=capsys)
    assert status == 2 and "edited or corrupted" in out
    # a refusal overwrites an earlier result of the same tag: no stale PASS survives
    status, _, _ = _gate(tmp_path, tiny, prompts_file, "stale", "--vllm-json", vllm_files["clean"], capsys=capsys)
    assert status == 0
    status, stale, out = _gate(tmp_path, tiny, prompts_file, "stale", "--vllm-json", str(tmp_path / "edited.json"),
                               capsys=capsys)  # fmt: skip
    assert status == 2 and stale["pass"] is False and stale["refused"] is True
    assert "REFUSED" in (tmp_path / "gate_b_greedy_stale.md").read_text()
    # other settings
    status, _, out = _gate(tmp_path, tiny, prompts_file, "len", "--vllm-json", vllm_files["clean"],
                           "--max-new-tokens", "20", capsys=capsys)  # fmt: skip
    assert status == 2 and "max_new_tokens=40" in out
    data = json.loads(Path(vllm_files["clean"]).read_text())
    data["model"] = str(tmp_path / "elsewhere")
    (tmp_path / "moved.json").write_text(json.dumps(data))
    status, _, out = _gate(tmp_path, tiny, prompts_file, "model", "--vllm-json", str(tmp_path / "moved.json"), capsys=capsys)
    assert status == 2 and "--allow-model-mismatch" in out
    status, _, _ = _gate(tmp_path, tiny, prompts_file, "model_ok", "--vllm-json", str(tmp_path / "moved.json"),
                         "--allow-model-mismatch", capsys=capsys)  # fmt: skip
    assert status == 0


def test_compare_only_reproduces_the_verdict_from_files(tiny, prompts_file, vllm_files, far, tmp_path, monkeypatch, capsys):
    i, j, token, _ = far
    status, full, _ = _gate(tmp_path, tiny, prompts_file, "cmp", "--vllm-json", vllm_files["near_tie"],
                            monkeypatch=monkeypatch, flip={i: (j, token)}, capsys=capsys)  # fmt: skip
    assert status == 1
    base = ["--compare-only", "--tag", "cmp", "--out-dir", str(tmp_path), "--vllm-json", vllm_files["near_tie"],
            "--device", "cpu"]  # fmt: skip
    for extra, margins_from in ((["--stored-margins"], "stored"), (["--hf-model", tiny], "hf_model")):
        assert gb.main(base + extra) == 1
        again = json.loads((tmp_path / "gate_b_greedy_cmp_recomputed.json").read_text())
        assert again["mode"] == "compare_only" and again["margins_from"] == margins_from
        assert again["pass"] == full["pass"] and again["three_way_result"] == full["three_way_result"]
        assert [(c["name"], c["ok"]) for c in again["checks"]] == [(c["name"], c["ok"]) for c in full["checks"]]
        assert again["failed_checks"] == full["failed_checks"]
    # the strict rule can be applied afterwards from the same files
    assert gb.main(base + ["--stored-margins", "--profile-rule", "strict"]) == 1
    # files of a two-way run hold no vLLM margins: stored-only comparison is refused, the HF model computes them
    monkeypatch.undo()  # an unperturbed engine from here on
    status, _, _ = _gate(tmp_path, tiny, prompts_file, "late", capsys=capsys)
    assert status == 0
    capsys.readouterr()
    late = ["--compare-only", "--tag", "late", "--out-dir", str(tmp_path), "--vllm-json", vllm_files["near_tie"],
            "--device", "cpu"]  # fmt: skip
    assert gb.main(late + ["--stored-margins"]) == 2
    assert "not stored" in capsys.readouterr().out
    assert gb.main(late) == 0


def test_vllm_length_label_on_a_final_eos_is_not_a_contract_problem():
    """vLLM V1 checks the length cap before the EOS: an EOS as the max_tokens-th token is labelled "length"."""
    side = {"token_ids": [[1, 2, EOS_IDS[0]], [1, 2, 3], [1, EOS_IDS[1]], [1, EOS_IDS[0], 2]],
            "finish_reasons": ["length", "length", "stop", "length"], "max_new_tokens": 3,
            "sampled_logprobs": [[-0.1] * 3, [-0.1] * 3, [-0.1] * 2, [-0.1] * 3]}  # fmt: skip
    problems = gb.vllm_contract_problems(side, EOS_IDS)
    # response 3 really breaks the contract (tokens after its EOS); responses 0-2 do not
    assert problems == ["response 3: tokens after the first EOS (position 1)", "response 3: ends with EOS but finish_reason='length'"]
    assert gb.vllm_contract_problems({**side, "token_ids": side["token_ids"][:3], "finish_reasons": side["finish_reasons"][:3],
                                      "sampled_logprobs": side["sampled_logprobs"][:3]}, EOS_IDS) == []  # fmt: skip
    # the engines get no such allowance
    assert any("finish_reason='length'" in p for p in gb._contract_of(
        {**side, "logprobs": [[0.0] * len(t) for t in side["token_ids"]]}, EOS_IDS, greedy=True))  # fmt: skip


# ----------------------------------------------------------------------------
# R2 statistics
# ----------------------------------------------------------------------------
def _records(first_diffs, length=100) -> list[dict]:
    recs = []
    for d in first_diffs:
        if d < 0:
            recs.append({"status": "identical", "first_diff": -1, "exposure": length})
        else:
            recs.append({"status": "near_tie", "first_diff": d, "exposure": d + 1})
    return recs


def test_profile_rules():
    clean = [-1] * 12
    # (D_S, D_V) = (2, 2) at similar positions: both rules pass when S's earliest is not earlier
    s, v = _records([50, 80] + clean + [-1, -1]), _records([40, 90] + clean + [-1, -1])
    prof = gb.divergence_profile(s, v, rule="significance", alpha=0.05, length=100)
    assert prof["engine"]["divergences"] == prof["vllm"]["divergences"] == 2
    share = prof["engine"]["exposure"] / (prof["engine"]["exposure"] + prof["vllm"]["exposure"])
    assert prof["p_value"] == pytest.approx(sum(math.comb(4, k) * share**k * (1 - share) ** (4 - k) for k in (2, 3, 4)))
    assert prof["significance_ok"] and prof["strict_ok"] and prof["ok"]
    # same counts, S's earliest divergence earlier: strict fails literally, significance does not care
    s_early = _records([30, 80] + clean + [-1, -1])
    prof = gb.divergence_profile(s_early, v, rule="strict", alpha=0.05, length=100)
    assert not prof["strict_ok"] and prof["significance_ok"] and not prof["ok"]
    # one more divergence than vLLM: strict fails; significance passes
    prof = gb.divergence_profile(_records([50, 80, 95] + clean + [-1]), v, rule="significance", alpha=0.05, length=100)
    assert not prof["strict_ok"] and prof["significance_ok"] and prof["ok"]
    # grossly worse SparseSpec: 12 early divergences vs 1
    gross = _records([3, 5, 8, 2, 9, 4, 7, 6, 3, 5, 2, 8, -1, -1, -1, -1])
    prof = gb.divergence_profile(gross, _records([60] + [-1] * 15), rule="significance", alpha=0.05, length=100)
    assert prof["p_value"] < 1e-6 and not prof["significance_ok"] and not prof["ok"]
    with pytest.raises(ValueError):
        gb.divergence_profile(s, v, rule="literal", alpha=0.05, length=100)
    lines = gb.profile_lines(prof, "sparsespec")
    assert lines[0].startswith("literal:") and lines[1].startswith("significance:") and "--profile-alpha" in lines[1]


def test_binomial_p_value_against_independent_computations():
    for n in (0, 1, 2, 3, 5, 8, 13, 21, 32):
        for p in (0.1, 0.37, 0.5, 0.8):
            frac_p = Fraction(p)
            for k in range(0, n + 2):
                exact = sum(math.comb(n, j) * frac_p**j * (1 - frac_p) ** (n - j) for j in range(k, n + 1))
                assert gb.binomial_sf(k, n, p) == pytest.approx(float(exact), rel=1e-9, abs=1e-15)
            if n:
                ks = torch.arange(0, n + 1, dtype=torch.float64)
                pmf = torch.distributions.Binomial(total_count=n, probs=torch.tensor(p, dtype=torch.float64)).log_prob(ks).exp()
                for k in range(0, n + 1):
                    assert gb.binomial_sf(k, n, p) == pytest.approx(float(pmf[k:].sum()), rel=1e-9, abs=1e-12)
    assert gb.binomial_sf(4, 4, 0.5) == pytest.approx(1 / 16)
    assert gb.binomial_sf(8, 9, 0.5) == pytest.approx(10 / 512)  # the docstring's "8 vs 1 gives p = 0.02"


def test_strict_rule_false_fail_probability_matches_simulation_and_docstring():
    length, n = 256, 16
    rng = np.random.default_rng(0)
    for q in (0.1, 0.5):
        h = 1 - (1 - q) ** (1 / length)
        exact = gb.strict_rule_false_fail_probability(h, n, length)
        trials = 20000
        draws = []
        for _ in range(2):
            pos = rng.geometric(h, size=(trials, n)) - 1
            div = pos < length
            draws.append((div.sum(1), np.where(div, pos, length + 1).min(1)))
        (ds, ms), (dv, mv) = draws
        simulated = 1 - ((ds == 0) | ((ds <= dv) & (ms >= mv))).mean()
        assert exact == pytest.approx(simulated, abs=0.015)
    # the numbers quoted in the docstring
    for q, quoted in ((0.05, 0.43), (0.10, 0.55), (0.25, 0.64), (0.50, 0.67), (0.75, 0.68), (0.90, 0.66)):
        h = 1 - (1 - q) ** (1 / length)
        assert round(gb.strict_rule_false_fail_probability(h, n, length), 2) == quoted
        assert f"q = {q:.2f}: {quoted:.2f}" in gb.__doc__
    assert gb.strict_rule_false_fail_probability(0.0, n, length) == 0.0
