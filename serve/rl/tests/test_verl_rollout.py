"""serve/rl/verl_rollout.py against the real verl 0.5.0 ``DataProto``.

The engine is ``ReferenceEngine`` on CPU around the tiny Qwen3 (plus a
scripted engine for hand-picked edge cases). Prompts are built the way verl
builds them (``tk.build_gen_batch``), the worker is simulated with a copy of
``ActorRolloutRefWorker.generate_sequences`` (``tk.worker_generate_sequences``)
and the expected output tensors come from the tail of
``vLLMRollout.generate_sequences`` itself (``vllm_generate_sequences_tail``).
"""

from __future__ import annotations

import ast
import copy
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

verl = pytest.importorskip("verl")

from verl import DataProto  # noqa: E402
from verl.utils.fsdp_utils import fsdp_version  # noqa: E402
from verl.utils.torch_functional import get_response_mask, logprobs_from_logits, pad_2d_list_to_length  # noqa: E402

from serve.rl import verl_rollout as verl_rollout_module  # noqa: E402
from serve.rl.base import STAT_KEYS, GenerationResult, LoadReport, RolloutEngine, empty_stats  # noqa: E402
from serve.rl.reference_engine import ReferenceEngine  # noqa: E402
from serve.rl.verl_rollout import (  # noqa: E402
    METRIC_PREFIX,
    SparseSpecRollout,
    SparseSpecRolloutConfig,
    SparseSpecShardingManager,
    _pre_process_inputs,
    build_sparsespec_rollout,
)

PROMPT_LENGTH = 24
RESPONSE_LENGTH = 16
PROMPT_LENGTHS = [5, 24, 2, 11, 7, 19]  # one prompt fills prompt_length exactly (no left padding)


# ---------------------------------------------------------------------------
# reference implementations copied from verl
# ---------------------------------------------------------------------------
def vllm_generate_sequences_tail(
    self, idx, attention_mask, position_ids, eos_token_id, batch_size, non_tensor_batch, response, rollout_log_probs
):
    # VERBATIM COPY (re-indented) of the tail of vLLMRollout.generate_sequences,
    # verl 0.5.0, verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py l.341-384.
    # `response` / `rollout_log_probs` enter as the per-request lists vLLM returned.
    # test_vllm_tail_is_a_verbatim_copy checks this body against the installed file.
    # fmt: off
    response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
        idx.device
    )
    if self.config.calculate_log_probs:
        rollout_log_probs = pad_2d_list_to_length(
            rollout_log_probs, -1, max_length=self.config.response_length
        ).to(idx.device)
        rollout_log_probs = rollout_log_probs.to(torch.float32)

    seq = torch.cat([idx, response], dim=-1)

    response_length = response.size(1)
    delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
    delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
    if position_ids.dim() == 3:  # qwen2vl mrope
        delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

    # TODO(sgm): fix position_ids on right_pad
    # prompt: left pad + response: right pad
    # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
    # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
    response_position_ids = position_ids[..., -1:] + delta_position_id
    position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
    response_attention_mask = get_response_mask(
        response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
    )
    attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

    # all the tp ranks should contain the same data here. data in all ranks are valid
    batch = TensorDict(
        {
            "prompts": idx,
            "responses": response,
            "input_ids": seq,  # here input_ids become the whole sentences
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=batch_size,
    )
    if self.config.calculate_log_probs:
        # we will recompute old log prob with actor
        batch["rollout_log_probs"] = rollout_log_probs

    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
    # fmt: on


def expected_vllm_output(config, prompts_snapshot, eos_token_id, raw_result, non_tensor_batch):
    """What vLLMRollout would have returned had vLLM produced ``raw_result``."""
    self = SimpleNamespace(config=config, pad_token_id=3)
    idx = prompts_snapshot["input_ids"]
    return vllm_generate_sequences_tail(
        self,
        idx,
        prompts_snapshot["attention_mask"],
        prompts_snapshot["position_ids"],
        eos_token_id,
        idx.size(0),
        non_tensor_batch,
        [list(r) for r in raw_result.token_ids],
        [list(r) for r in raw_result.logprobs],
    )


@torch.no_grad()
def actor_old_log_probs(actor, batch, temperature: float) -> torch.Tensor:
    """DataParallelPPOActor._forward_micro_batch (verl/workers/actor/dp_actor.py
    l.239-263, the branch without remove-padding), minus the bf16 autocast:
    how verl recomputes old_log_probs from the rollout's output tensors."""
    response_length = batch["responses"].size(-1)
    output = actor(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        use_cache=False,
    )  # prevent model thinks we are generating
    logits = output.logits
    logits.div_(temperature)
    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
    return logprobs_from_logits(logits, batch["responses"])


def rollout_probs_diff_max(batch) -> float:
    """training/rollout_probs_diff_max as RayPPOTrainer.fit computes it (ray_trainer.py l.1238-1251)."""
    rollout_old_log_probs = batch["rollout_log_probs"]
    actor_old_log_probs_ = batch["old_log_probs"]
    attention_mask = batch["attention_mask"]
    responses = batch["responses"]
    response_length = responses.size(1)
    response_mask = attention_mask[:, -response_length:]

    rollout_probs = torch.exp(rollout_old_log_probs)
    actor_probs = torch.exp(actor_old_log_probs_)
    rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
    rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
    return torch.max(rollout_probs_diff).item()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class ScriptedEngine(RolloutEngine):
    """Returns canned responses, to place EOS / padding exactly where a test wants them."""

    name = "scripted"

    def __init__(self, responses, logprobs=None):
        self.responses = [list(r) for r in responses]
        self.logprobs = logprobs or [[-0.25 * (j + 1) for j in range(len(r))] for r in responses]
        self.calls = []
        self._sleeping = False

    def generate(self, prompt_token_ids, max_new_tokens, temperature=1.0, top_p=1.0, top_k=-1, *, eos_token_ids=None):
        if self._sleeping:
            raise RuntimeError("asleep")
        self.calls.append(
            SimpleNamespace(
                prompts=[list(p) for p in prompt_token_ids],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                eos_token_ids=eos_token_ids,
            )
        )
        stats = empty_stats()
        # deliberately not all plain floats: the sharding manager must normalise what it publishes
        stats.update(num_requests=len(self.responses), gen_s=0.5, num_cycles=7, accept_len_mean=np.float32(4.5))
        return GenerationResult(
            token_ids=[list(r) for r in self.responses],
            logprobs=[list(r) for r in self.logprobs],
            finish_reasons=["stop"] * len(self.responses),
            stats=stats,
        )

    def load_weights(self, named_tensors):
        return LoadReport(0, 0, 0, 0.0)

    def sleep(self):
        self._sleeping = True

    def wake_up(self):
        self._sleeping = False

    @property
    def is_sleeping(self):
        return self._sleeping


def with_worker_meta(tk, gen_batch, **extra):
    """meta_info as ActorRolloutRefWorker.generate_sequences sets it (fsdp_workers.py l.726-734)."""
    generation_config = tk.tiny_generation_config()
    gen_batch.meta_info.update(
        {"eos_token_id": generation_config.eos_token_id, "pad_token_id": generation_config.pad_token_id}
    )
    gen_batch.meta_info.update(extra)
    return gen_batch


def snapshot_prompts(gen_batch) -> dict:
    return {key: gen_batch.batch[key].clone() for key in ("input_ids", "attention_mask", "position_ids")}


def make_rollout(tk, model, *, engine=None, engine_kwargs=None, awake=True, **config_overrides):
    config = tk.make_rollout_config(prompt_length=PROMPT_LENGTH, response_length=RESPONSE_LENGTH, **config_overrides)
    if engine is None:
        engine = tk.make_reference_engine(model, **(engine_kwargs or {}))
    rollout = SparseSpecRollout(
        model_path="unused: the engine is injected",
        config=config,
        tokenizer=tk.TinyTokenizer(),
        model_hf_config=model.config,
        engine=engine,
    )
    if awake:
        engine.wake_up()  # what SparseSpecShardingManager.__enter__ does
    return rollout


def make_stack(tk, actor, *, engine_dtype="float32", engine_seed=0, engine_weights_seed=1, **config_overrides):
    """rollout + sharding manager around ``actor``; the engine starts with
    DIFFERENT weights (seed 1), so only a real sync makes it match the actor."""
    engine = tk.make_reference_engine(tk.make_tiny_qwen3(engine_weights_seed), dtype=engine_dtype, seed=engine_seed)
    rollout = make_rollout(tk, actor, engine=engine, awake=False, **config_overrides)
    manager = SparseSpecShardingManager(
        module=actor, rollout=rollout, rollout_config=rollout.config, offload_param=False
    )
    return SimpleNamespace(config=rollout.config, engine=engine, rollout=rollout, manager=manager, actor=actor)


def assert_same_dataproto_tensors(actual, expected) -> None:
    assert set(actual.batch.keys()) == set(expected.batch.keys())
    assert actual.batch.batch_size == expected.batch.batch_size
    for key in expected.batch.keys():
        a, e = actual.batch[key], expected.batch[key]
        assert a.dtype == e.dtype, f"{key}: dtype {a.dtype} != {e.dtype}"
        assert a.shape == e.shape, f"{key}: shape {tuple(a.shape)} != {tuple(e.shape)}"
        assert a.device == e.device, key
        assert torch.equal(a, e), key


def response_rows(output, eos_ids) -> list[list[int]]:
    """Unpadded responses, recovered through the response part of the attention mask."""
    responses = output.batch["responses"]
    mask = output.batch["attention_mask"][:, -responses.size(1) :]
    return [row[: int(m.sum())].tolist() for row, m in zip(responses, mask)]


# ---------------------------------------------------------------------------
# the copies above are faithful
# ---------------------------------------------------------------------------
def _vllm_rollout_source() -> str:
    path = Path(verl.__file__).parent / "workers" / "rollout" / "vllm_rollout" / "vllm_rollout_spmd.py"
    return path.read_text()


def _is_response_padding(stmt) -> bool:
    """``response = pad_2d_list_to_length(response, ...)``: where the copied tail starts."""
    return (
        isinstance(stmt, ast.Assign)
        and getattr(stmt.targets[0], "id", None) == "response"
        and "pad_2d_list_to_length(response" in ast.unparse(stmt.value)
    )


def test_vllm_tail_is_a_verbatim_copy():
    """Statement for statement (compared as ASTs, so re-wrapping by a formatter
    cannot break it): the helper IS the end of vLLMRollout.generate_sequences."""
    tree = ast.parse(_vllm_rollout_source())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "vLLMRollout")
    theirs = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "generate_sequences")
    # in verl the tail starts inside `with self.update_sampling_params(**kwargs):` and continues after it
    with_index, with_block = next(
        (i, n)
        for i, n in enumerate(theirs.body)
        if isinstance(n, ast.With) and "update_sampling_params" in ast.unparse(n.items[0])
    )
    inner = next(i for i, stmt in enumerate(with_block.body) if _is_response_padding(stmt))
    their_tail = with_block.body[inner:] + theirs.body[with_index + 1 :]
    assert isinstance(their_tail[-1], ast.Return) and theirs.body[-1] is their_tail[-1]

    ours = ast.parse(inspect.getsource(vllm_generate_sequences_tail)).body[0]
    assert _is_response_padding(ours.body[0])
    assert [ast.dump(stmt) for stmt in ours.body] == [ast.dump(stmt) for stmt in their_tail]
    assert len(ours.body) >= 14  # padding, log-probs, cat, positions, mask, TensorDict, return


def test_pre_process_inputs_is_verls():
    tree = ast.parse(_vllm_rollout_source())
    theirs = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_pre_process_inputs")
    ours = ast.parse(inspect.getsource(_pre_process_inputs)).body[0]
    assert ast.dump(theirs) == ast.dump(ours)  # comments aside, the same code


# ---------------------------------------------------------------------------
# prompts are built the way verl builds them
# ---------------------------------------------------------------------------
def test_gen_batch_looks_like_verls(tk):
    prompts = tk.make_prompts(PROMPT_LENGTHS, seed=2)
    batch, gen_batch = tk.build_gen_batch(prompts, PROMPT_LENGTH, n=2)
    assert isinstance(gen_batch, DataProto)
    assert set(gen_batch.batch.keys()) == {"input_ids", "attention_mask", "position_ids"}
    assert set(gen_batch.non_tensor_batch) == {"raw_prompt_ids", "index"}
    assert gen_batch.meta_info == {"global_steps": 1}
    assert set(batch.non_tensor_batch) == {"data_source"} and len(batch) == len(prompts)

    ids, mask, pos = (gen_batch.batch[k] for k in ("input_ids", "attention_mask", "position_ids"))
    assert ids.shape == mask.shape == pos.shape == (2 * len(prompts), PROMPT_LENGTH)
    assert ids.dtype == mask.dtype == pos.dtype == torch.int64
    for row in range(2 * len(prompts)):
        prompt = prompts[row // 2]  # interleaved repeat
        pad = PROMPT_LENGTH - len(prompt)
        assert ids[row, :pad].tolist() == [tk.PAD_ID] * pad  # LEFT padded with tokenizer.pad_token_id
        assert ids[row, pad:].tolist() == prompt
        assert mask[row].tolist() == [0] * pad + [1] * len(prompt)
        assert pos[row].tolist() == [0] * pad + list(range(len(prompt)))
        assert list(gen_batch.non_tensor_batch["raw_prompt_ids"][row]) == prompt
    assert gen_batch.non_tensor_batch["index"].tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


# ---------------------------------------------------------------------------
# output contract
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("with_raw_prompt_ids", [True, False])
@pytest.mark.parametrize("calculate_log_probs", [True, False])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_output_equals_vllm_rollout_tail(tk, tiny_model, with_raw_prompt_ids, calculate_log_probs, temperature):
    rollout = make_rollout(tk, tiny_model, calculate_log_probs=calculate_log_probs, temperature=temperature)
    recorder = tk.GenerateRecorder(rollout.engine)
    prompts = tk.make_prompts(PROMPT_LENGTHS, seed=2)
    n = 4
    bsz = n * len(prompts)

    mixed = False
    for _ in range(6):  # until the batch holds both early-stopped and full-length responses
        _, gen_batch = tk.build_gen_batch(prompts, PROMPT_LENGTH, n=n, with_raw_prompt_ids=with_raw_prompt_ids)
        gen_batch = with_worker_meta(tk, gen_batch)
        inputs = snapshot_prompts(gen_batch)
        non_tensor_in = gen_batch.non_tensor_batch
        index_before = non_tensor_in["index"].copy()

        output = rollout.generate_sequences(gen_batch)

        call = recorder.last
        raw = call.result
        # --- what the engine was asked -------------------------------------
        assert [list(p) for p in call.args[0]] == [prompts[i // n] for i in range(bsz)]  # unpadded, interleaved
        assert all(type(t) is int for p in call.args[0] for t in p)
        assert call.kwargs["max_new_tokens"] == RESPONSE_LENGTH
        assert call.kwargs["eos_token_ids"] == tk.EOS_IDS and isinstance(call.kwargs["eos_token_ids"], list)
        assert call.kwargs["temperature"] == temperature and type(call.kwargs["temperature"]) is float
        assert call.kwargs["top_p"] == 1.0 and type(call.kwargs["top_p"]) is float
        assert call.kwargs["top_k"] == -1 and type(call.kwargs["top_k"]) is int

        # --- parity with vLLMRollout's own post-processing --------------------
        expected = expected_vllm_output(rollout.config, inputs, tk.EOS_IDS, raw, {"index": index_before})
        assert type(output) is DataProto
        assert_same_dataproto_tensors(output, expected)

        # --- the contract, field by field -------------------------------------
        b = output.batch
        keys = {"prompts", "responses", "input_ids", "attention_mask", "position_ids"}
        assert set(b.keys()) == keys | ({"rollout_log_probs"} if calculate_log_probs else set())
        assert tuple(b.batch_size) == (bsz,)
        assert b["prompts"].shape == (bsz, PROMPT_LENGTH)
        assert b["responses"].shape == (bsz, RESPONSE_LENGTH)
        for key in ("input_ids", "attention_mask", "position_ids"):
            assert b[key].shape == (bsz, PROMPT_LENGTH + RESPONSE_LENGTH), key
        for key in keys:
            assert b[key].dtype == torch.int64, key
            assert b[key].device.type == "cpu" and not b[key].requires_grad and not b[key].is_inference(), key

        assert torch.equal(b["prompts"], inputs["input_ids"])  # prompts untouched
        assert torch.equal(b["input_ids"], torch.cat([b["prompts"], b["responses"]], dim=-1))
        for row, (tokens, logprobs) in enumerate(zip(raw.token_ids, raw.logprobs)):
            k = len(tokens)
            assert b["responses"][row, :k].tolist() == tokens
            assert b["responses"][row, k:].tolist() == [tk.PAD_ID] * (RESPONSE_LENGTH - k)  # RIGHT padded
            # mask: prompt mask, then 1 up to and including the first EOS, 0 afterwards
            assert torch.equal(b["attention_mask"][row, :PROMPT_LENGTH], inputs["attention_mask"][row])
            assert b["attention_mask"][row, PROMPT_LENGTH:].tolist() == [1] * k + [0] * (RESPONSE_LENGTH - k)
            # positions continue from the last prompt position, also across the padding
            last = int(inputs["position_ids"][row, -1])
            assert last == len(prompts[row // n]) - 1
            assert torch.equal(b["position_ids"][row, :PROMPT_LENGTH], inputs["position_ids"][row])
            assert b["position_ids"][row, PROMPT_LENGTH:].tolist() == list(range(last + 1, last + 1 + RESPONSE_LENGTH))
            if calculate_log_probs:
                lp = b["rollout_log_probs"][row]
                assert torch.equal(lp[:k], torch.tensor(logprobs, dtype=torch.float32))
                assert lp[k:].tolist() == [-1.0] * (RESPONSE_LENGTH - k)  # padded with -1
        assert torch.equal(
            b["attention_mask"][:, PROMPT_LENGTH:],
            get_response_mask(response_id=b["responses"], eos_token=tk.EOS_IDS, dtype=inputs["attention_mask"].dtype),
        )
        if calculate_log_probs:
            assert b["rollout_log_probs"].dtype == torch.float32
            assert b["rollout_log_probs"].shape == (bsz, RESPONSE_LENGTH)
        else:
            assert "rollout_log_probs" not in b.keys()

        # --- non-tensor data / inputs --------------------------------------------
        assert output.meta_info == {}
        assert "raw_prompt_ids" not in output.non_tensor_batch  # popped, as vLLMRollout does
        assert set(output.non_tensor_batch) == {"index"}
        assert np.array_equal(output.non_tensor_batch["index"], index_before)
        assert "raw_prompt_ids" not in non_tensor_in
        for key, value in inputs.items():  # the prompt DataProto's tensors were not modified
            assert torch.equal(gen_batch.batch[key], value), key

        lengths = {len(t) for t in raw.token_ids}
        if min(lengths) < RESPONSE_LENGTH and any(
            len(t) == RESPONSE_LENGTH and t[-1] not in tk.EOS_IDS for t in raw.token_ids
        ):
            mixed = True
            break
    assert mixed, "never saw a batch with both early-stopped and length-limited responses"
    assert rollout.last_stats == raw.stats


def test_output_contract_edge_cases_with_scripted_responses(tk, tiny_model):
    im_end, pad = tk.IM_END_ID, tk.PAD_ID
    full = list(range(100, 100 + RESPONSE_LENGTH))
    responses = [
        [im_end],  # EOS as the very first token
        [50, 51, pad],  # stopped on <|endoftext|>, which is also the pad token
        full,  # length-limited, no EOS
        full[:-1] + [im_end],  # EOS exactly at the last position
        [60, 61, 62, 63, im_end],
    ]
    engine = ScriptedEngine(responses)
    rollout = make_rollout(tk, tiny_model, engine=engine)
    prompts = tk.make_prompts([3, 24, 9, 1, 12], seed=3)
    _, gen_batch = tk.build_gen_batch(prompts, PROMPT_LENGTH)
    gen_batch = with_worker_meta(tk, gen_batch)
    inputs = snapshot_prompts(gen_batch)

    output = rollout.generate_sequences(gen_batch)

    assert engine.calls[0].prompts == prompts
    b = output.batch
    assert b["responses"].tolist() == [r + [pad] * (RESPONSE_LENGTH - len(r)) for r in responses]
    assert b["attention_mask"][:, PROMPT_LENGTH:].tolist() == [
        [1] * len(r) + [0] * (RESPONSE_LENGTH - len(r)) for r in responses
    ]
    assert b["attention_mask"][:, PROMPT_LENGTH:].sum(-1).tolist() == [1, 3, 16, 16, 5]
    for row, prompt in enumerate(prompts):
        assert b["position_ids"][row, PROMPT_LENGTH:].tolist() == list(
            range(len(prompt), len(prompt) + RESPONSE_LENGTH)
        )
    expected_lp = [lp + [-1.0] * (RESPONSE_LENGTH - len(lp)) for lp in engine.logprobs]
    assert b["rollout_log_probs"].tolist() == expected_lp and b["rollout_log_probs"].dtype == torch.float32
    assert_same_dataproto_tensors(
        output,
        expected_vllm_output(rollout.config, inputs, tk.EOS_IDS, GenerationResult(responses, engine.logprobs, []), {}),
    )


def test_all_responses_short_are_still_padded_to_response_length(tk, tiny_model):
    engine = ScriptedEngine([[40, tk.IM_END_ID], [tk.IM_END_ID]])
    rollout = make_rollout(tk, tiny_model, engine=engine)
    _, gen_batch = tk.build_gen_batch(tk.make_prompts([4, 6]), PROMPT_LENGTH)
    output = rollout.generate_sequences(with_worker_meta(tk, gen_batch))
    assert output.batch["responses"].shape == (2, RESPONSE_LENGTH)
    assert output.batch["rollout_log_probs"].shape == (2, RESPONSE_LENGTH)
    assert output.batch["input_ids"].shape == (2, PROMPT_LENGTH + RESPONSE_LENGTH)


def test_engine_response_longer_than_response_length_is_an_error(tk, tiny_model):
    engine = ScriptedEngine([list(range(100, 100 + RESPONSE_LENGTH + 1)), [tk.IM_END_ID]])
    rollout = make_rollout(tk, tiny_model, engine=engine)
    _, gen_batch = tk.build_gen_batch(tk.make_prompts([4, 6]), PROMPT_LENGTH)
    with pytest.raises(RuntimeError, match="longer than"):
        rollout.generate_sequences(with_worker_meta(tk, gen_batch))


@pytest.mark.parametrize("eos_meta", [7, [7], [7, 3]])
def test_eos_token_id_may_be_an_int_or_a_list(tk, tiny_model, eos_meta):
    """generation_config.eos_token_id is a list for Qwen3; the tokenizer fallback is an int."""
    responses = [[50, 7, 51], [52, 3]]
    engine = ScriptedEngine(responses)
    rollout = make_rollout(tk, tiny_model, engine=engine)
    _, gen_batch = tk.build_gen_batch(tk.make_prompts([4, 6]), PROMPT_LENGTH)
    gen_batch.meta_info["eos_token_id"] = eos_meta
    output = rollout.generate_sequences(gen_batch)
    assert engine.calls[0].eos_token_ids == eos_meta
    eos = [eos_meta] if isinstance(eos_meta, int) else eos_meta
    row0 = 2  # [50, 7]: the scripted engine ignores EOS, the mask must not
    row1 = 2 if 3 in eos else RESPONSE_LENGTH  # pad == 3 only counts as EOS when the list says so
    assert output.batch["attention_mask"][:, PROMPT_LENGTH:].sum(-1).tolist() == [row0, row1]


def test_raw_prompt_ids_as_2d_object_array_and_as_lists(tk, tiny_model):
    """verl's collate_fn turns equal-length prompts into a 2-D object array
    (rows are ndarrays) and ragged ones into a 1-D array of lists."""
    for lengths in ([6, 6, 6], [6, 3, 9]):
        prompts = tk.make_prompts(lengths, seed=4)
        engine = ScriptedEngine([[tk.IM_END_ID]] * 3)
        rollout = make_rollout(tk, tiny_model, engine=engine)
        _, gen_batch = tk.build_gen_batch(prompts, PROMPT_LENGTH)
        raw = gen_batch.non_tensor_batch["raw_prompt_ids"]
        assert raw.ndim == (2 if len(set(lengths)) == 1 else 1)
        rollout.generate_sequences(with_worker_meta(tk, gen_batch))
        assert engine.calls[0].prompts == prompts
        assert all(type(t) is int for p in engine.calls[0].prompts for t in p)


def test_prompts_are_recovered_from_left_padded_input_ids(tk, tiny_model):
    prompts = tk.make_prompts(PROMPT_LENGTHS, seed=5)
    engine = ScriptedEngine([[tk.IM_END_ID]] * len(prompts))
    rollout = make_rollout(tk, tiny_model, engine=engine)
    _, gen_batch = tk.build_gen_batch(prompts, PROMPT_LENGTH, with_raw_prompt_ids=False)
    assert "raw_prompt_ids" not in gen_batch.non_tensor_batch
    output = rollout.generate_sequences(with_worker_meta(tk, gen_batch))
    assert engine.calls[0].prompts == prompts
    assert "raw_prompt_ids" not in output.non_tensor_batch
    assert _pre_process_inputs(tk.PAD_ID, torch.tensor([3, 3, 9, 3, 10])) == [9, 3, 10]  # only LEFT padding goes


def test_bad_raw_prompt_ids_and_multimodal_inputs_are_rejected(tk, tiny_model):
    rollout = make_rollout(tk, tiny_model, engine=ScriptedEngine([[tk.IM_END_ID]] * 2))
    _, gen_batch = tk.build_gen_batch(tk.make_prompts([4, 6]), PROMPT_LENGTH)
    gen_batch.non_tensor_batch["raw_prompt_ids"] = np.array(["1 2 3", "4 5"], dtype=object)
    with pytest.raises(TypeError, match="list or numpy array"):
        rollout.generate_sequences(with_worker_meta(tk, gen_batch))

    _, gen_batch = tk.build_gen_batch(tk.make_prompts([4, 6]), PROMPT_LENGTH)
    gen_batch.non_tensor_batch["multi_modal_data"] = np.array([{}, {}], dtype=object)
    with pytest.raises(NotImplementedError, match="multi-modal"):
        rollout.generate_sequences(with_worker_meta(tk, gen_batch))


def test_generate_outside_the_sharding_manager_fails_loudly(tk, tiny_model):
    """free_cache_engine=True: the rollout puts the engine to sleep at
    construction (as vLLMRollout does); only the sharding manager wakes it."""
    rollout = make_rollout(tk, tiny_model, awake=False)
    assert rollout.engine.is_sleeping is True
    _, gen_batch = tk.build_gen_batch(tk.make_prompts([4, 6]), PROMPT_LENGTH)
    with pytest.raises(RuntimeError, match="asleep"):
        rollout.generate_sequences(with_worker_meta(tk, gen_batch))

    rollout = make_rollout(tk, tiny_model, awake=False, free_cache_engine=False)
    assert rollout.engine.is_sleeping is False
    _, gen_batch = tk.build_gen_batch(tk.make_prompts([4, 6]), PROMPT_LENGTH)
    assert len(rollout.generate_sequences(with_worker_meta(tk, gen_batch))) == 2


# ---------------------------------------------------------------------------
# sampling parameters
# ---------------------------------------------------------------------------
def test_do_sample_false_is_greedy(tk, tiny_model):
    rollout = make_rollout(tk, tiny_model)
    recorder = tk.GenerateRecorder(rollout.engine)
    prompts = tk.make_prompts(PROMPT_LENGTHS, seed=2)
    expected = [tk.cut_at_eos(tk.hf_greedy(tiny_model, p, RESPONSE_LENGTH), tk.EOS_IDS) for p in prompts]

    outputs = []
    for _ in range(2):
        _, gen_batch = tk.build_gen_batch(prompts, PROMPT_LENGTH)
        outputs.append(rollout.generate_sequences(with_worker_meta(tk, gen_batch, do_sample=False)))
        kwargs = recorder.last.kwargs
        assert (kwargs["temperature"], kwargs["top_p"], kwargs["top_k"]) == (0.0, 1.0, -1)
        assert type(kwargs["temperature"]) is float and type(kwargs["top_k"]) is int

    assert response_rows(outputs[0], tk.EOS_IDS) == expected  # the HF greedy continuation
    assert_same_dataproto_tensors(outputs[0], outputs[1])  # deterministic
    # greedy log-probs are reported as 0.0; padding stays -1
    mask = outputs[0].batch["attention_mask"][:, PROMPT_LENGTH:].bool()
    lp = outputs[0].batch["rollout_log_probs"]
    assert torch.all(lp[mask] == 0.0) and torch.all(lp[~mask] == -1.0)


def test_sampling_parameter_resolution(tk, tiny_model):
    """Same precedence as vLLMRollout: do_sample=False > validate > rollout.{temperature,top_p,top_k}."""
    rollout = make_rollout(
        tk,
        tiny_model,
        engine=ScriptedEngine([[tk.IM_END_ID]] * 2),
        temperature=0.7,
        **{"val_kwargs.temperature": 0.5, "val_kwargs.top_p": 1.0, "val_kwargs.top_k": 0},
    )
    engine = rollout.engine

    def call(**meta):
        _, gen_batch = tk.build_gen_batch(tk.make_prompts([4, 6]), PROMPT_LENGTH)
        rollout.generate_sequences(with_worker_meta(tk, gen_batch, **meta))
        last = engine.calls[-1]
        assert type(last.temperature) is float and type(last.top_p) is float and type(last.top_k) is int
        assert last.max_new_tokens == RESPONSE_LENGTH
        return last.temperature, last.top_p, last.top_k

    assert call() == (0.7, 1.0, -1)  # training rollout
    assert call(do_sample=True) == (0.7, 1.0, -1)
    assert call(validate=True) == (0.5, 1.0, 0)  # rollout.val_kwargs
    assert call(validate=True, do_sample=True) == (0.5, 1.0, 0)
    assert call(do_sample=False) == (0.0, 1.0, -1)  # greedy
    assert call(do_sample=False, validate=True) == (0.0, 1.0, -1)  # verl's validation default
    assert call(validate=False) == (0.7, 1.0, -1)


def test_validation_with_verl_default_val_kwargs_is_greedy(tk, tiny_model):
    rollout = make_rollout(tk, tiny_model)  # rollout.yaml: val_kwargs.temperature = 0
    recorder = tk.GenerateRecorder(rollout.engine)
    prompts = tk.make_prompts(PROMPT_LENGTHS, seed=2)
    _, gen_batch = tk.build_gen_batch(prompts, PROMPT_LENGTH)
    output = rollout.generate_sequences(with_worker_meta(tk, gen_batch, validate=True))
    assert recorder.last.kwargs["temperature"] == 0.0
    expected = [tk.cut_at_eos(tk.hf_greedy(tiny_model, p, RESPONSE_LENGTH), tk.EOS_IDS) for p in prompts]
    assert response_rows(output, tk.EOS_IDS) == expected


@pytest.mark.parametrize(
    "overrides, meta",
    [
        ({"top_p": 0.9}, {}),
        ({"top_k": 50}, {}),
        ({"val_kwargs.top_p": 0.8}, {"validate": True}),
        ({"val_kwargs.top_k": 20}, {"validate": True}),
    ],
)
def test_top_p_top_k_settings_raise_instead_of_being_ignored(tk, tiny_model, overrides, meta):
    rollout = make_rollout(tk, tiny_model, **overrides)
    _, gen_batch = tk.build_gen_batch(tk.make_prompts([4, 6]), PROMPT_LENGTH)
    with pytest.raises(NotImplementedError):
        rollout.generate_sequences(with_worker_meta(tk, gen_batch, **meta))


def test_unsupported_val_kwargs_do_not_affect_training_rollouts(tk, tiny_model):
    rollout = make_rollout(tk, tiny_model, **{"val_kwargs.top_p": 0.8})
    _, gen_batch = tk.build_gen_batch(tk.make_prompts([4, 6]), PROMPT_LENGTH)
    assert len(rollout.generate_sequences(with_worker_meta(tk, gen_batch))) == 2


def test_batch_repeated_8x_gives_distinct_samples(tk, tiny_model):
    """GRPO: rollout.n=8 -> every prompt appears 8 times (interleaved); at T=1 the 8 samples must differ."""
    n = 8
    rollout = make_rollout(tk, tiny_model)
    prompts = tk.make_prompts([9, 4, 17], seed=6)
    _, gen_batch = tk.build_gen_batch(prompts, PROMPT_LENGTH, n=n)
    output = rollout.generate_sequences(with_worker_meta(tk, gen_batch))

    assert len(output) == n * len(prompts)
    rows = [tuple(r) for r in output.batch["responses"].tolist()]
    for group in range(len(prompts)):
        chunk = slice(group * n, (group + 1) * n)
        assert torch.equal(output.batch["prompts"][chunk], output.batch["prompts"][group * n].expand(n, -1))
        assert output.batch["prompts"][group * n, PROMPT_LENGTH - len(prompts[group]) :].tolist() == prompts[group]
        assert len(set(rows[chunk])) >= n - 1, "samples of one prompt are (nearly) identical"
    assert len(set(rows)) >= len(rows) - 2
    assert output.non_tensor_batch["index"].tolist() == [g for g in range(len(prompts)) for _ in range(n)]
    # log-probs differ along with the samples
    assert len({tuple(r) for r in output.batch["rollout_log_probs"].tolist()}) >= len(rows) - 2


def test_rollout_log_probs_match_the_actor_recomputation(tk, tiny_model):
    """verl recomputes old_log_probs with the actor from the rollout's output
    tensors and logs |exp(rollout) - exp(actor)| (training/rollout_probs_diff_*).
    With the same float32 weights on both sides the difference is numerical noise,
    which also proves input_ids / attention_mask / position_ids are consistent."""
    for temperature in (1.0, 0.7):
        rollout = make_rollout(tk, tiny_model, temperature=temperature)
        prompts = tk.make_prompts(PROMPT_LENGTHS, seed=2)
        _, gen_batch = tk.build_gen_batch(prompts, PROMPT_LENGTH, n=4)
        output = rollout.generate_sequences(with_worker_meta(tk, gen_batch))
        batch = output.batch
        batch["old_log_probs"] = actor_old_log_probs(tiny_model, batch, temperature)
        assert rollout_probs_diff_max(batch) < 1e-4
        mask = batch["attention_mask"][:, -RESPONSE_LENGTH:].bool()
        assert (batch["rollout_log_probs"] - batch["old_log_probs"])[mask].abs().max().item() < 1e-4
        assert int(mask.sum()) > len(prompts)
        # a different temperature on the actor side is clearly visible
        batch["old_log_probs"] = actor_old_log_probs(tiny_model, batch, 1.7 - temperature)
        assert rollout_probs_diff_max(batch) > 1e-3


# ---------------------------------------------------------------------------
# SparseSpecRolloutConfig
# ---------------------------------------------------------------------------
def test_config_defaults(tk):
    # verl's own rollout.yaml (no sparsespec block, no seed), in struct mode as hydra delivers it
    stock = OmegaConf.load(tk.verl_rollout_yaml())
    OmegaConf.set_struct(stock, True)
    assert "sparsespec" not in stock and "seed" not in stock
    for rollout_config in (stock, OmegaConf.create({"name": "sparsespec"}), {"name": "sparsespec"}, {"sparsespec": {}}):
        cfg = SparseSpecRolloutConfig.from_rollout_config(rollout_config)
        assert cfg == SparseSpecRolloutConfig(
            engine="sparsespec",
            spec_stride=8,
            budget_ratio=0.05,
            num_min_budget=128,
            kv_cache="pillar",
            max_batch_size=64,
            cuda_graph=True,
            seed=0,
            chunk_size=256,
            cpu_offload_ratio=1.5,
            kv_capacity_tokens=None,
            transient_reserve_gb=2.0,
        )
        assert type(cfg.seed) is int


def test_config_values_and_types(tk):
    rollout_config = tk.make_rollout_config(
        **{
            "sparsespec.engine": "reference",
            "sparsespec.spec_stride": 4,
            "sparsespec.budget_ratio": 0.1,
            "sparsespec.num_min_budget": 64,
            "sparsespec.max_batch_size": 32,
            "sparsespec.cuda_graph": False,
            "sparsespec.chunk_size": 128,
            "sparsespec.cpu_offload_ratio": 2,
            "sparsespec.kv_capacity_tokens": 50000,
            "sparsespec.transient_reserve_gb": 3,
        }
    )
    cfg = SparseSpecRolloutConfig.from_rollout_config(rollout_config)
    assert (cfg.engine, cfg.spec_stride, cfg.budget_ratio, cfg.num_min_budget) == ("reference", 4, 0.1, 64)
    assert (cfg.max_batch_size, cfg.cuda_graph, cfg.chunk_size) == (32, False, 128)
    assert (cfg.cpu_offload_ratio, cfg.kv_capacity_tokens, cfg.transient_reserve_gb) == (2.0, 50000, 3.0)
    for name, kind in {
        "spec_stride": int, "budget_ratio": float, "num_min_budget": int, "max_batch_size": int,
        "cuda_graph": bool, "seed": int, "chunk_size": int, "cpu_offload_ratio": float,
        "kv_capacity_tokens": int, "transient_reserve_gb": float,
    }.items():  # fmt: skip
        assert type(getattr(cfg, name)) is kind, name


def test_config_unknown_key_raises(tk):
    rollout_config = tk.make_rollout_config(**{"sparsespec.spec_strid": 4})
    with pytest.raises(ValueError, match="spec_strid"):
        SparseSpecRolloutConfig.from_rollout_config(rollout_config)
    with pytest.raises(ValueError, match="unknown"):
        SparseSpecRolloutConfig.from_rollout_config({"sparsespec": {"temperature": 1.0}})


@pytest.mark.parametrize("engine", ["vllm", "hf", "SparseSpec", ""])
def test_config_engine_is_validated(tk, engine):
    with pytest.raises(ValueError, match="engine"):
        SparseSpecRolloutConfig.from_rollout_config(tk.make_rollout_config(**{"sparsespec.engine": engine}))


def test_config_engine_accepts_both_backends(tk):
    for engine in ("sparsespec", "reference"):
        cfg = SparseSpecRolloutConfig.from_rollout_config(tk.make_rollout_config(**{"sparsespec.engine": engine}))
        assert cfg.engine == engine


def test_config_seed_falls_back_to_rollout_seed(tk):
    parse = SparseSpecRolloutConfig.from_rollout_config
    assert parse(tk.make_rollout_config()).seed == 0  # verl's rollout.yaml has no seed: 0, like vLLMRollout
    assert parse(tk.make_rollout_config(seed=123)).seed == 123
    assert parse(tk.make_rollout_config(seed=123, **{"sparsespec.seed": 7})).seed == 7
    assert parse(tk.make_rollout_config(seed=123, **{"sparsespec.seed": 0})).seed == 0  # an explicit 0 is a value
    assert parse(tk.make_rollout_config(**{"sparsespec.seed": 9})).seed == 9
    assert parse({"seed": 5, "sparsespec": {"spec_stride": 8}}).seed == 5


def test_rollout_uses_the_parsed_config(tk, tiny_model):
    rollout = make_rollout(tk, tiny_model, seed=42, **{"sparsespec.spec_stride": 6})
    assert isinstance(rollout.ss, SparseSpecRolloutConfig)
    assert (rollout.ss.seed, rollout.ss.spec_stride, rollout.ss.engine) == (42, 6, "reference")
    assert rollout.pad_token_id == tk.PAD_ID
    with pytest.raises(ValueError, match="unknown"):
        make_rollout(tk, tiny_model, **{"sparsespec.typo": 1})


# ---------------------------------------------------------------------------
# unsupported settings
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"tensor_model_parallel_size": 2}, NotImplementedError),  # verl's default!
        ({"mode": "async"}, NotImplementedError),
        ({"ignore_eos": True}, NotImplementedError),
        ({"dtype": "float16"}, NotImplementedError),
        ({"dtype": "float32"}, NotImplementedError),
        ({"sparsespec.engine": "vllm"}, ValueError),
        ({"prompt_length": 400, "response_length": 113}, AssertionError),  # > max_position_embeddings (512)
    ],
)
def test_unsupported_settings_raise_before_an_engine_is_built(tk, tiny_model, overrides, error):
    config = tk.make_rollout_config(**overrides)
    with pytest.raises(error):
        # engine=None and a model path that does not exist: reaching the engine
        # build would fail with a different (OS / hub) error
        SparseSpecRollout(
            model_path="/nonexistent/sparsespec-test-model",
            config=config,
            tokenizer=tk.TinyTokenizer(),
            model_hf_config=tiny_model.config,
        )


def test_context_length_check(tk, tiny_model):
    hf_config = SimpleNamespace()  # no max_position_embeddings at all
    with pytest.raises(ValueError, match="max_position_embeddings"):
        SparseSpecRollout("unused", tk.make_rollout_config(), tk.TinyTokenizer(), hf_config, engine=ScriptedEngine([]))
    # exactly prompt_length + response_length is allowed (same check as vLLMRollout)
    config = tk.make_rollout_config(prompt_length=400, response_length=112)
    assert tiny_model.config.max_position_embeddings == 512
    SparseSpecRollout("unused", config, tk.TinyTokenizer(), tiny_model.config, engine=ScriptedEngine([]))


def test_supported_baseline_settings_are_accepted(tk, tiny_model):
    config = tk.make_rollout_config(
        temperature=1.0, top_p=1.0, top_k=-1, n=8, gpu_memory_utilization=0.45, free_cache_engine=True,
        enable_chunked_prefill=True, max_num_batched_tokens=10240, mode="sync", dtype="bfloat16",
    )  # fmt: skip
    engine = ScriptedEngine([])
    rollout = SparseSpecRollout("unused", config, tk.TinyTokenizer(), tiny_model.config, engine=engine)
    assert rollout.engine is engine and engine.is_sleeping is True


# ---------------------------------------------------------------------------
# SparseSpecShardingManager (plain HF module as the actor)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("engine_dtype", ["float32", "bfloat16"])
def test_enter_copies_the_actor_weights_into_the_engine(tk, tiny_model, engine_dtype):
    actor = tiny_model
    assert fsdp_version(actor) == 0  # a plain module: state_dict() is the full state dict
    stack = make_stack(tk, actor, engine_dtype=engine_dtype)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[engine_dtype]
    pointers = stack.engine.parameter_pointers()
    assert stack.engine.verify_weights(actor.state_dict())["ok"] is False  # the engine starts with other weights

    tk.fill_parameters(actor, seed=77)  # "an optimizer step": every tensor changes
    actor_before = tk.snapshot(actor)
    assert stack.engine.verify_weights(actor.state_dict())["ok"] is False

    with stack.manager:
        engine_params = dict(stack.engine.model.named_parameters(remove_duplicate=False))
        actor_state = actor.state_dict()
        assert set(engine_params) == set(actor_state)
        for name, param in engine_params.items():
            assert param.dtype == dtype
            assert tk.bit_equal(param, actor_state[name].to(dtype)), name  # bitwise, after the dtype cast
        assert stack.engine.verify_weights(actor_state)["ok"] is True
        assert stack.engine.parameter_pointers() == pointers  # written in place

    # the actor itself is not modified by the sync, and nothing is shared with the engine
    tk.assert_matches_snapshot(actor, actor_before)
    assert all(p.dtype == torch.float32 and p.requires_grad for p in actor.parameters())
    assert not {p.data_ptr() for p in actor.parameters()} & set(pointers.values())


def test_engine_is_awake_only_inside_the_context(tk, tiny_model):
    stack = make_stack(tk, tiny_model, free_cache_engine=True)
    assert stack.engine.is_sleeping is True  # asleep right after construction, like vLLMRollout
    for _ in range(2):
        with stack.manager:
            assert stack.engine.is_sleeping is False
        assert stack.engine.is_sleeping is True


def test_engine_stays_awake_without_free_cache_engine(tk, tiny_model):
    stack = make_stack(tk, tiny_model, free_cache_engine=False)
    assert stack.engine.is_sleeping is False
    for _ in range(2):
        with stack.manager:
            assert stack.engine.is_sleeping is False
            assert stack.manager.timing["sparsespec/wake_s"] == 0.0
            assert stack.manager.timing["sparsespec/graph_capture_s"] == 0.0
        assert stack.engine.is_sleeping is False
    # the weights are synced all the same
    assert stack.engine.verify_weights(tiny_model.state_dict())["ok"] is True


def test_actor_is_in_train_mode_after_exit(tk, tiny_model):
    actor = tiny_model.eval()
    stack = make_stack(tk, actor)
    with stack.manager:
        pass
    assert actor.training is True
    assert stack.engine.model.training is False  # the engine's copy stays in eval mode


def test_timing_keys_and_values(tk, tiny_model):
    stack = make_stack(tk, tiny_model)
    recorder = tk.GenerateRecorder(stack.engine)
    _, gen_batch = tk.build_gen_batch(tk.make_prompts(PROMPT_LENGTHS, seed=2), PROMPT_LENGTH, n=2)

    with stack.manager:
        # after __enter__: sync + wake timings and verl's own "reshard"
        assert set(stack.manager.timing) == {
            "reshard", "sparsespec/weight_sync_s", "sparsespec/wake_s", "sparsespec/graph_capture_s",
        }  # fmt: skip
        stack.rollout.generate_sequences(with_worker_meta(tk, gen_batch))
        assert stack.rollout.last_stats == recorder.last.result.stats

    timing = stack.manager.timing
    assert set(timing) == {"reshard"} | set(tk.SPARSESPEC_TIMING_KEYS)
    assert all(key.startswith(METRIC_PREFIX) for key in tk.SPARSESPEC_TIMING_KEYS) and METRIC_PREFIX == "sparsespec/"
    assert all(type(v) is float for v in timing.values()), {k: type(v) for k, v in timing.items()}
    assert all(v >= 0.0 for v in timing.values())
    assert timing["reshard"] >= timing["sparsespec/weight_sync_s"] > 0.0
    assert timing["reshard"] >= timing["sparsespec/wake_s"]
    assert timing["sparsespec/graph_capture_s"] == 0.0  # no CUDA graphs in the reference engine

    # the engine statistics of the generate() call made inside the context
    stats = recorder.last.result.stats
    for name in ("accept_len_mean", "accept_rate", "num_cycles", "num_iters", "batch_size_mean", "num_offloads",
                 "kv_util_peak", "peak_mem_gb", "gen_s"):  # fmt: skip
        assert name in STAT_KEYS
        assert timing["sparsespec/" + name] == stats[name], name
    assert timing["sparsespec/gen_s"] > 0.0 and timing["sparsespec/num_iters"] > 0.0
    assert timing["sparsespec/accept_len_mean"] == timing["sparsespec/num_cycles"] == 0.0  # reference engine

    # verl averages the dict across ranks through a float32 tensor (reduce_timing): must be convertible
    reduced = torch.tensor([timing[k] for k in sorted(timing)], dtype=torch.float32)
    assert torch.isfinite(reduced).all()

    assert stack.rollout.last_stats == {}  # consumed: cannot leak into the next step


def test_timing_publishes_the_engine_stats(tk, tiny_model):
    """With an engine that reports speculation statistics they arrive unchanged."""
    engine = ScriptedEngine([[tk.IM_END_ID]] * 2)
    rollout = make_rollout(tk, tiny_model, engine=engine, awake=False)
    manager = SparseSpecShardingManager(module=tiny_model, rollout=rollout, rollout_config=rollout.config)
    _, gen_batch = tk.build_gen_batch(tk.make_prompts([4, 6]), PROMPT_LENGTH)
    with manager:
        rollout.generate_sequences(with_worker_meta(tk, gen_batch))
    assert manager.timing["sparsespec/num_cycles"] == 7.0
    assert manager.timing["sparsespec/accept_len_mean"] == 4.5
    assert manager.timing["sparsespec/gen_s"] == 0.5
    # plain Python floats whatever the engine handed over (int, numpy scalar): verl's
    # console logger and reduce_timing both need real numbers
    assert all(type(v) is float for v in manager.timing.values()), {k: type(v) for k, v in manager.timing.items()}


def test_timing_is_fresh_every_step_and_stats_do_not_leak(tk, tiny_model):
    stack = make_stack(tk, tiny_model)
    _, gen_batch = tk.build_gen_batch(tk.make_prompts([4, 6]), PROMPT_LENGTH)
    with stack.manager:
        stack.rollout.generate_sequences(with_worker_meta(tk, gen_batch))
    first = dict(stack.manager.timing)
    assert first["sparsespec/gen_s"] > 0.0

    with stack.manager:  # a step in which nothing is generated
        assert "sparsespec/gen_s" not in stack.manager.timing
    second = stack.manager.timing
    assert second is not first and set(second) == set(first)
    assert second["sparsespec/gen_s"] == 0.0 and second["sparsespec/num_iters"] == 0.0
    assert all(type(v) is float for v in second.values()), {k: type(v) for k, v in second.items()}


def test_exception_inside_the_context_still_sleeps_the_engine(tk, tiny_model):
    actor = tiny_model.eval()
    stack = make_stack(tk, actor)
    with pytest.raises(ZeroDivisionError):
        with stack.manager:
            assert stack.engine.is_sleeping is False
            1 / 0
    assert stack.engine.is_sleeping is True
    assert actor.training is True
    assert set(stack.manager.timing) == {"reshard"} | set(tk.SPARSESPEC_TIMING_KEYS)
    with stack.manager:  # and the next step works
        assert stack.engine.is_sleeping is False


def test_weight_sync_failure_leaves_the_engine_asleep_and_unchanged(tk, tiny_model):
    stack = make_stack(tk, tiny_model)
    before = tk.snapshot(stack.engine.model)
    wrong = copy.deepcopy(tiny_model)
    wrong.model.layers = wrong.model.layers[:1]  # an actor with a different architecture
    manager = SparseSpecShardingManager(module=wrong, rollout=stack.rollout, rollout_config=stack.config)
    with pytest.raises(KeyError):
        manager.__enter__()
    assert stack.engine.is_sleeping is True
    tk.assert_matches_snapshot(stack.engine.model, before)


def test_manager_is_a_verl_sharding_manager(tk, tiny_model):
    from verl.workers.rollout.base import BaseRollout
    from verl.workers.sharding_manager.base import BaseShardingManager

    stack = make_stack(tk, tiny_model)
    assert isinstance(stack.manager, BaseShardingManager) and isinstance(stack.rollout, BaseRollout)
    assert stack.manager.timing == {}  # readable before the first step
    data = DataProto.from_dict(tensors={"x": torch.zeros(2, 1)})
    assert stack.manager.preprocess_data(data) is data and stack.manager.postprocess_data(data) is data


# ---------------------------------------------------------------------------
# end to end: the worker's generate_sequences, twice, with a weight change
# ---------------------------------------------------------------------------
def trainer_side(tk, batch, gen_batch_output, n):
    """What RayPPOTrainer.fit does with the worker's output (ray_trainer.py
    l.1174-1200, with the statement integration/verl-0.5.0-sparsespec.patch adds)."""
    from verl.trainer.ppo.metric_utils import compute_timing_metrics

    metrics, timing_raw = {}, {"gen": 0.75}
    timing_raw.update(gen_batch_output.meta_info["timing"])
    gen_batch_output.meta_info.pop("timing", None)
    tk.apply_trainer_metrics_statement(metrics, timing_raw)

    batch = batch.repeat(repeat_times=n, interleave=True)
    batch = batch.union(gen_batch_output)
    metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
    return batch, metrics, timing_raw


@pytest.mark.parametrize("engine_dtype", ["float32", "bfloat16"])
def test_two_training_steps_end_to_end(tk, tiny_model, engine_dtype):
    n = 4
    actor = tiny_model.train()
    stack = make_stack(tk, actor, engine_dtype=engine_dtype, n=n)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[engine_dtype]
    pointers = stack.engine.parameter_pointers()
    prompts = tk.make_prompts(PROMPT_LENGTHS, seed=2)
    bsz = n * len(prompts)

    previous_actor = None
    responses_per_step = []
    for step in (1, 2):
        batch, gen_batch = tk.build_gen_batch(prompts, PROMPT_LENGTH, n=n, global_steps=step)
        actor.eval()  # e.g. left in eval mode by compute_log_prob
        output = tk.worker_generate_sequences(stack.rollout, stack.manager, gen_batch)

        # --- state after the worker call ------------------------------------
        assert stack.engine.is_sleeping is True
        assert actor.training is True
        assert stack.engine.parameter_pointers() == pointers
        for name, param in stack.engine.model.named_parameters(remove_duplicate=False):
            assert tk.bit_equal(param, actor.state_dict()[name].to(dtype)), (step, name)

        # --- the DataProto the trainer receives -----------------------------
        assert type(output) is DataProto and len(output) == bsz
        timing = output.meta_info["timing"]
        assert set(timing) == {"generate_sequences", "reshard"} | set(tk.SPARSESPEC_TIMING_KEYS)
        assert all(type(v) is float for v in timing.values())
        assert timing["generate_sequences"] >= timing["sparsespec/gen_s"] > 0.0
        assert set(output.batch.keys()) == {
            "prompts", "responses", "input_ids", "attention_mask", "position_ids", "rollout_log_probs",
        }  # fmt: skip
        assert output.batch["input_ids"].shape == (bsz, PROMPT_LENGTH + RESPONSE_LENGTH)
        assert all(t.device.type == "cpu" for t in output.batch.values())

        # --- the trainer's next lines: metrics + union with the rest of the batch
        batch, metrics, timing_raw = trainer_side(tk, batch, output, n)
        assert "timing" not in output.meta_info
        assert not any(k.startswith("sparsespec/") for k in timing_raw)
        assert {k for k in metrics if k.startswith("sparsespec/")} == set(tk.SPARSESPEC_TIMING_KEYS)
        assert {"timing_s/gen", "timing_s/reshard", "timing_s/generate_sequences", "timing_per_token_ms/gen"} <= set(
            metrics
        )
        assert not any(k.startswith("timing_s/sparsespec") for k in metrics)
        assert len(batch) == bsz and set(batch.non_tensor_batch) == {"data_source", "index"}
        assert output.meta_info == {} and batch.meta_info == {}  # like vLLMRollout: no meta_info is carried over

        # --- the rollout used THIS step's actor weights ----------------------------
        rows = response_rows(output, tk.EOS_IDS)
        responses_per_step.append(rows)
        assert all(1 <= len(r) <= RESPONSE_LENGTH for r in rows)
        if engine_dtype == "float32":
            temperature = float(stack.config.temperature)
            batch.batch["old_log_probs"] = actor_old_log_probs(actor, batch.batch, temperature)
            assert rollout_probs_diff_max(batch.batch) < 1e-4, step
            if previous_actor is not None:  # ... and not the weights of the step before
                batch.batch["old_log_probs"] = actor_old_log_probs(previous_actor, batch.batch, temperature)
                assert rollout_probs_diff_max(batch.batch) > 1e-3

        # --- update_actor: every weight moves --------------------------------------
        previous_actor = copy.deepcopy(actor)
        with torch.no_grad():
            noise = torch.Generator().manual_seed(step)
            for param in actor.parameters():
                param.add_(torch.randn(param.shape, generator=noise) * 0.05)
        assert stack.engine.verify_weights(actor.state_dict())["ok"] is False  # stale until the next enter

    assert responses_per_step[0] != responses_per_step[1]


def test_worker_flow_with_do_sample_false_tracks_the_actor(tk, tiny_model):
    """Greedy rollouts before and after a weight change equal the HF greedy
    continuation of the actor as it is at that moment."""
    actor = tiny_model
    stack = make_stack(tk, actor)
    prompts = tk.make_prompts(PROMPT_LENGTHS, seed=2)
    seen = []
    for step in (1, 2):
        _, gen_batch = tk.build_gen_batch(prompts, PROMPT_LENGTH, global_steps=step)
        gen_batch.meta_info["do_sample"] = False
        output = tk.worker_generate_sequences(stack.rollout, stack.manager, gen_batch)
        reference = copy.deepcopy(actor).eval()
        expected = [tk.cut_at_eos(tk.hf_greedy(reference, p, RESPONSE_LENGTH), tk.EOS_IDS) for p in prompts]
        assert response_rows(output, tk.EOS_IDS) == expected, step
        seen.append(expected)
        tk.fill_parameters(actor, seed=53 + step)  # "update_actor": new weights for the next step
    assert seen[0] != seen[1]


def test_worker_flow_falls_back_to_the_tokenizer_eos(tk, tiny_model):
    """generation_config is None -> the worker passes tokenizer.eos_token_id (an int)."""
    stack = make_stack(tk, tiny_model)
    recorder = tk.GenerateRecorder(stack.engine)
    _, gen_batch = tk.build_gen_batch(tk.make_prompts(PROMPT_LENGTHS, seed=2), PROMPT_LENGTH, n=2)
    output = tk.worker_generate_sequences(
        stack.rollout, stack.manager, gen_batch, generation_config=None, tokenizer=tk.TinyTokenizer()
    )
    assert recorder.last.kwargs["eos_token_ids"] == tk.IM_END_ID
    mask = output.batch["attention_mask"][:, PROMPT_LENGTH:]
    assert torch.equal(mask, get_response_mask(output.batch["responses"], eos_token=tk.IM_END_ID, dtype=mask.dtype))
    # only <|im_end|> ends a response now: <|endoftext|> (the pad id) may appear inside one
    for tokens in recorder.last.result.token_ids:
        assert tk.IM_END_ID not in tokens[:-1]


# ---------------------------------------------------------------------------
# build_sparsespec_rollout: the entry point the verl patch calls
# ---------------------------------------------------------------------------
def test_build_sparsespec_rollout_with_the_reference_engine(tk, tiny_model, tiny_model_b, tmp_path, monkeypatch):
    """engine=reference, loaded from a checkpoint directory like on the box.

    build_engine puts the engine on the current CUDA device when there is one;
    this is the CPU suite (and the box's GPU is shared), so CUDA is hidden here.
    The GPU path is exercised by integration/smoke_reference.sh.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    checkpoint = tmp_path / "tiny-qwen3"
    tiny_model_b.save_pretrained(checkpoint)  # the engine starts from the checkpoint (weights B)
    actor = tiny_model  # the actor has moved on (weights A)

    config = tk.make_rollout_config(prompt_length=PROMPT_LENGTH, response_length=RESPONSE_LENGTH, seed=11)
    rollout, manager = build_sparsespec_rollout(
        model_path=str(checkpoint),
        rollout_config=config,
        tokenizer=tk.TinyTokenizer(),
        model_hf_config=actor.config,
        actor_module=actor,
        offload_param=False,
    )
    assert isinstance(rollout, SparseSpecRollout) and isinstance(manager, SparseSpecShardingManager)
    engine = rollout.engine
    assert isinstance(engine, ReferenceEngine) and manager.engine is engine and manager.rollout is rollout
    assert engine.is_sleeping is True
    assert engine.dtype == torch.bfloat16  # rollout.dtype
    assert engine.seed == 11  # rollout.seed
    assert engine.max_batch_size == 64 and engine.mem_fraction == 0.45
    assert engine.device.type == "cpu"
    for name, param in engine.model.named_parameters(remove_duplicate=False):
        assert tk.bit_equal(param, tiny_model_b.state_dict()[name].to(torch.bfloat16)), name

    _, gen_batch = tk.build_gen_batch(tk.make_prompts(PROMPT_LENGTHS, seed=2), PROMPT_LENGTH, n=2)
    output = tk.worker_generate_sequences(rollout, manager, gen_batch)
    assert output.batch["responses"].shape == (2 * len(PROMPT_LENGTHS), RESPONSE_LENGTH)
    assert engine.is_sleeping is True
    for name, param in engine.model.named_parameters(remove_duplicate=False):
        assert tk.bit_equal(param, actor.state_dict()[name].to(torch.bfloat16)), name
    assert set(output.meta_info["timing"]) == {"generate_sequences", "reshard"} | set(tk.SPARSESPEC_TIMING_KEYS)


def test_build_sparsespec_rollout_signature_matches_the_patch(tk):
    """The patched _build_rollout calls build_sparsespec_rollout with exactly these keywords."""
    added = tk.sparsespec_branch_source()
    tree = ast.parse(added)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "build_sparsespec_rollout"
    ]
    assert len(calls) == 1 and not calls[0].args
    passed = {kw.arg for kw in calls[0].keywords}
    signature = inspect.signature(build_sparsespec_rollout)
    assert passed == set(signature.parameters)
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in signature.parameters.values())
    assert "from serve.rl.verl_rollout import build_sparsespec_rollout" in added


# ---------------------------------------------------------------------------
# static checks of the GPU-only engine against what the glue expects of it
# (serve/rl/engine.py cannot be imported here: read it with ast)
# ---------------------------------------------------------------------------
def _engine_class(tk):
    tree = ast.parse((tk.REPO_ROOT / "serve" / "rl" / "engine.py").read_text())
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SparseSpecEngine")


def _method(cls, name):
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def test_sparsespec_engine_accepts_what_build_engine_passes(tk):
    init = _method(_engine_class(tk), "__init__")
    positional = [a.arg for a in init.args.args][1:]
    keyword_only = [a.arg for a in init.args.kwonlyargs]
    reference = list(inspect.signature(ReferenceEngine.__init__).parameters)[1:10]
    assert positional == reference  # the shared positional signature of the brief
    assert {"dtype", "device"} <= set(keyword_only)

    build = ast.parse(inspect.getsource(verl_rollout_module.build_engine)).body[0]
    common = next(
        node.value
        for node in ast.walk(build)
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "common"
    )
    passed = {kw.arg for kw in common.keywords}
    engine_call = next(
        node
        for node in ast.walk(build)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "SparseSpecEngine"
    )
    passed |= {kw.arg for kw in engine_call.keywords if kw.arg is not None}
    assert passed <= set(positional) | set(keyword_only), passed - set(positional) - set(keyword_only)
    # ... and ReferenceEngine takes the common part too
    assert {kw.arg for kw in common.keywords} <= set(inspect.signature(ReferenceEngine.__init__).parameters)
    # every sparsespec.* option reaches an engine argument (or is the engine selector itself)
    options = {f for f in SparseSpecRolloutConfig.__dataclass_fields__} - {"engine"}
    assert options <= passed, options - passed


def test_sparsespec_engine_implements_the_interface_the_glue_uses(tk):
    cls = _engine_class(tk)
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    for name in ("generate", "load_weights", "verify_weights", "parameter_pointers", "sleep", "wake_up", "close",
                 "is_sleeping", "last_wake_stats"):  # fmt: skip
        assert name in methods, name
    for name in ("is_sleeping", "last_wake_stats"):
        assert any(getattr(d, "id", None) == "property" for d in methods[name].decorator_list), name

    generate = methods["generate"]
    expected = inspect.signature(RolloutEngine.generate)
    assert [a.arg for a in generate.args.args] == [
        n for n, p in expected.parameters.items() if p.kind is not inspect.Parameter.KEYWORD_ONLY
    ]
    assert [a.arg for a in generate.args.kwonlyargs] == ["eos_token_ids"]

    # generate() fills every statistic the sharding manager publishes, under the shared names
    updates = [
        node
        for node in ast.walk(generate)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "update"
        and getattr(node.func.value, "id", None) == "stats"
    ]
    filled = {kw.arg for call in updates for kw in call.keywords}
    assert filled <= set(STAT_KEYS), filled - set(STAT_KEYS)
    assert set(verl_rollout_module._PUBLISHED_STATS) <= filled
    assert set(verl_rollout_module._PUBLISHED_STATS) <= set(STAT_KEYS)
