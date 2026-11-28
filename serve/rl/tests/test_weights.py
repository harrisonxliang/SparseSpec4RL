"""serve/rl/weights.py: HF-named tensors -> engine parameters, in place.

The engine side is ``FakeEngineTransformer`` (conftest.py), which has the
parameter layout ``serve.model.model.Transformer`` has after
``from_pretrained`` (fused ``wqkv`` / ``w12``). The first tests pin that
stand-in to the real source so it cannot silently go stale.
"""

from __future__ import annotations

import ast
import copy
import itertools
from collections import OrderedDict

import pytest
import torch

from serve.rl.base import LoadReport
from serve.rl.weights import (
    load_weights_inplace,
    normalize_hf_key,
    normalize_state_dict,
    parameter_pointers,
    unique_parameter_bytes,
    verify_weights,
)

WRAPPERS = ("model.", "module.", "_fsdp_wrapped_module.")
# every ordering of every subset of the wrapper prefixes, plus repetitions
PREFIX_NESTINGS = sorted(
    {"".join(combo) for r in range(len(WRAPPERS) + 1) for combo in itertools.permutations(WRAPPERS, r)}
    | {"model.model.", "module.module.model.", "_fsdp_wrapped_module.model._fsdp_wrapped_module.model."}
)


def bare(key: str) -> str:
    """HF key without HF's own ``model.`` prefix."""
    return key[len("model.") :] if key.startswith("model.") else key


def with_prefix(state: dict, prefix: str) -> dict:
    return {prefix + bare(k): v for k, v in state.items()}


def flip_one_bit(param: torch.Tensor, flat_index: int) -> None:
    view = param.view(-1).view({2: torch.int16, 4: torch.int32}[param.element_size()])
    view[flat_index] ^= 1


# ---------------------------------------------------------------------------
# the stand-in has the real engine's parameter layout
# ---------------------------------------------------------------------------
def test_fake_engine_parameter_names_and_shapes(tk):
    q, kv = tk.HEADS * tk.HEAD_DIM, tk.KV_HEADS * tk.HEAD_DIM
    expected = {"embed_tokens.weight": (tk.VOCAB, tk.HIDDEN), "norm.weight": (tk.HIDDEN,)}
    expected["lm_head.weight"] = (tk.VOCAB, tk.HIDDEN)
    for i in range(tk.LAYERS):
        p = f"layers.{i}."
        expected[p + "input_layernorm.weight"] = (tk.HIDDEN,)
        expected[p + "post_attention_layernorm.weight"] = (tk.HIDDEN,)
        expected[p + "self_attn.wqkv.weight"] = (q + 2 * kv, tk.HIDDEN)
        expected[p + "self_attn.o_proj.weight"] = (tk.HIDDEN, q)
        expected[p + "self_attn.q_norm.weight"] = (tk.HEAD_DIM,)
        expected[p + "self_attn.k_norm.weight"] = (tk.HEAD_DIM,)
        expected[p + "mlp.w12.weight"] = (2 * tk.INTERMEDIATE, tk.HIDDEN)
        expected[p + "mlp.down_proj.weight"] = (tk.HIDDEN, tk.INTERMEDIATE)

    for tie in ("shared", "copy", "untied"):
        fake = tk.FakeEngineTransformer(tie)
        got = {name: tuple(param.shape) for name, param in tk.named_params(fake).items()}
        assert got == expected
        assert fake.config.tie_word_embeddings is (tie != "untied")
        shared = fake.lm_head.weight.data_ptr() == fake.embed_tokens.weight.data_ptr()
        assert shared is (tie == "shared")
        if tie == "copy":
            assert tk.bit_equal(fake.lm_head.weight, fake.embed_tokens.weight)

    biased = tk.FakeEngineTransformer("shared", attention_bias=True)
    assert tuple(biased.layers[0].self_attn.wqkv.bias.shape) == (q + 2 * kv,)


def test_fake_engine_matches_the_oracle_layout(tk, hf_state):
    oracle = tk.expected_engine_state(hf_state, torch.float32)
    fake = tk.FakeEngineTransformer("copy")
    assert {n: tuple(p.shape) for n, p in tk.named_params(fake).items()} == {
        n: tuple(t.shape) for n, t in oracle.items()
    }


def _class(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)


def _method(cls, name):
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _self_attrs_assigned(func) -> list[str]:
    out = []
    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    out.append(target.attr)
    return out


def _assigned_value(func, name: str) -> str:
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name:
            return ast.unparse(node.value)
    raise AssertionError(f"{name} is not assigned in {func.name}")


def _deleted_self_attrs(func) -> set[str]:
    return {
        target.attr
        for node in ast.walk(func)
        if isinstance(node, ast.Delete)
        for target in node.targets
        if isinstance(target, ast.Attribute)
    }


def test_engine_source_still_builds_the_layout_the_fake_mimics(tk):
    """serve/model/model.py cannot be imported without flashinfer, so read it
    with ast: module attribute names, and the order in which merge() fuses."""
    tree = ast.parse((tk.REPO_ROOT / "serve" / "model" / "model.py").read_text())

    transformer = _self_attrs_assigned(_method(_class(tree, "Transformer"), "__init__"))
    assert {"embed_tokens", "layers", "norm", "lm_head"} <= set(transformer)
    block = _self_attrs_assigned(_method(_class(tree, "TransformerBlock"), "__init__"))
    assert block == ["input_layernorm", "self_attn", "post_attention_layernorm", "mlp"]
    assert _self_attrs_assigned(_method(_class(tree, "RMSNorm"), "__init__")) == ["eps", "weight"]

    attention = _class(tree, "Attention")
    assert {"q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"} <= set(
        _self_attrs_assigned(_method(attention, "__init__"))
    )
    merge = _method(attention, "merge")
    assert _assigned_value(merge, "wqkv_weight") == (
        "torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0)"
    )
    assert _assigned_value(merge, "wqkv_bias") in (
        "torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], dim=0)",
        "None",
    )
    assert "wqkv" in _self_attrs_assigned(merge)
    assert _deleted_self_attrs(merge) == {"q_proj", "k_proj", "v_proj"}

    feed_forward = _class(tree, "FeedForward")
    assert {"gate_proj", "up_proj", "down_proj"} <= set(_self_attrs_assigned(_method(feed_forward, "__init__")))
    merge = _method(feed_forward, "merge")
    assert _assigned_value(merge, "w12_weight") == "torch.cat([self.gate_proj.weight, self.up_proj.weight], dim=0)"
    assert "w12" in _self_attrs_assigned(merge)
    assert _deleted_self_attrs(merge) == {"gate_proj", "up_proj"}


# ---------------------------------------------------------------------------
# the mapping
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tie", ["shared", "copy"])
def test_fused_mapping_is_bitwise_right(tk, hf_state, tie):
    fake = tk.FakeEngineTransformer(tie)
    before = tk.snapshot(fake)
    report = load_weights_inplace(fake, hf_state)

    oracle = tk.expected_engine_state(hf_state, torch.float32)
    params = tk.named_params(fake)
    assert set(params) == set(oracle)
    for name, param in params.items():
        assert tk.bit_equal(param, oracle[name]), name
        assert not tk.bit_equal(param, before[name]), f"{name}: the load was a no-op"

    # the fusion order, spelled out against the raw HF tensors
    q, kv, inter = tk.HEADS * tk.HEAD_DIM, tk.KV_HEADS * tk.HEAD_DIM, tk.INTERMEDIATE
    for i in range(tk.LAYERS):
        hf = f"model.layers.{i}."
        wqkv = fake.layers[i].self_attn.wqkv.weight
        assert tk.bit_equal(wqkv[:q], hf_state[hf + "self_attn.q_proj.weight"])
        assert tk.bit_equal(wqkv[q : q + kv], hf_state[hf + "self_attn.k_proj.weight"])
        assert tk.bit_equal(wqkv[q + kv :], hf_state[hf + "self_attn.v_proj.weight"])
        w12 = fake.layers[i].mlp.w12.weight
        assert tk.bit_equal(w12[:inter], hf_state[hf + "mlp.gate_proj.weight"])
        assert tk.bit_equal(w12[inter:], hf_state[hf + "mlp.up_proj.weight"])
    assert tk.bit_equal(fake.embed_tokens.weight, hf_state["model.embed_tokens.weight"])
    assert tk.bit_equal(fake.lm_head.weight, hf_state["model.embed_tokens.weight"])

    assert isinstance(report, LoadReport)
    unique = len(list(fake.parameters()))  # nn.Module de-duplicates a shared lm_head
    assert report.num_params == unique == (len(params) - 1 if tie == "shared" else len(params))
    assert report.num_sources == len(hf_state)
    assert report.num_bytes == unique_parameter_bytes(fake.parameters())
    assert report.seconds >= 0.0


def test_fused_bias_mapping(tk, hf_state):
    """Models with attention biases (Qwen2-style: q/k/v have a bias, o_proj does not)."""
    gen = torch.Generator().manual_seed(7)
    q, kv = tk.HEADS * tk.HEAD_DIM, tk.KV_HEADS * tk.HEAD_DIM
    state = dict(hf_state)
    for i in range(tk.LAYERS):
        attn = f"model.layers.{i}.self_attn."
        state[attn + "q_proj.bias"] = torch.randn(q, generator=gen)
        state[attn + "k_proj.bias"] = torch.randn(kv, generator=gen)
        state[attn + "v_proj.bias"] = torch.randn(kv, generator=gen)

    fake = tk.FakeEngineTransformer("shared", attention_bias=True)
    load_weights_inplace(fake, state)
    oracle = tk.expected_engine_state(state, torch.float32)
    for name, param in tk.named_params(fake).items():
        assert tk.bit_equal(param, oracle[name]), name
    bias = fake.layers[1].self_attn.wqkv.bias
    attn = "model.layers.1.self_attn."
    assert tk.bit_equal(bias[:q], state[attn + "q_proj.bias"])
    assert tk.bit_equal(bias[q : q + kv], state[attn + "k_proj.bias"])
    assert tk.bit_equal(bias[q + kv :], state[attn + "v_proj.bias"])
    assert verify_weights(fake, state)["ok"] is True

    # an engine without biases must refuse a state dict that has them
    plain = tk.FakeEngineTransformer("shared")
    snap = tk.snapshot(plain)
    with pytest.raises(KeyError, match="no parameter for"):
        load_weights_inplace(plain, state)
    tk.assert_matches_snapshot(plain, snap)
    # and an engine with biases must refuse a state dict without them
    snap = tk.snapshot(fake)
    with pytest.raises(KeyError, match="missing"):
        load_weights_inplace(fake, hf_state)
    tk.assert_matches_snapshot(fake, snap)


@pytest.mark.parametrize("tie", ["shared", "copy"])
def test_fp32_to_bf16_cast_equals_tensor_to(tk, hf_state, tie):
    fake = tk.FakeEngineTransformer(tie, dtype=torch.bfloat16)
    source_copy = {k: v.clone() for k, v in hf_state.items()}
    load_weights_inplace(fake, hf_state)

    for i in range(tk.LAYERS):
        hf = f"model.layers.{i}."
        fused = torch.cat(
            [hf_state[hf + f"self_attn.{n}_proj.weight"] for n in ("q", "k", "v")], dim=0
        )  # float32, fused first
        assert tk.bit_equal(fake.layers[i].self_attn.wqkv.weight, fused.to(torch.bfloat16))
        fused = torch.cat([hf_state[hf + "mlp.gate_proj.weight"], hf_state[hf + "mlp.up_proj.weight"]], dim=0)
        assert tk.bit_equal(fake.layers[i].mlp.w12.weight, fused.to(torch.bfloat16))
        for name in ("input_layernorm", "post_attention_layernorm"):
            assert tk.bit_equal(
                getattr(fake.layers[i], name).weight, hf_state[hf + name + ".weight"].to(torch.bfloat16)
            )
    # ... and equals what from_pretrained does: cast every tensor, then fuse
    oracle = tk.expected_engine_state(hf_state, torch.bfloat16)
    for name, param in tk.named_params(fake).items():
        assert param.dtype == torch.bfloat16
        assert tk.bit_equal(param, oracle[name]), name

    # the cast really rounds (otherwise this test would not distinguish anything)
    embed = hf_state["model.embed_tokens.weight"]
    assert not torch.equal(fake.embed_tokens.weight.float(), embed)
    # the source tensors are left alone
    for key, value in hf_state.items():
        assert value.dtype == torch.float32 and torch.equal(value, source_copy[key])


def test_bf16_source_into_fp32_engine_is_exact(tk, hf_state):
    state = {k: v.to(torch.bfloat16) for k, v in hf_state.items()}
    fake = tk.FakeEngineTransformer("shared", dtype=torch.float32)
    load_weights_inplace(fake, state)
    oracle = tk.expected_engine_state(state, torch.float32)
    for name, param in tk.named_params(fake).items():
        assert param.dtype == torch.float32 and tk.bit_equal(param, oracle[name]), name
    assert torch.equal(fake.norm.weight, state["model.norm.weight"].float())


@pytest.mark.parametrize("tie", ["shared", "copy"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_copies_are_in_place(tk, hf_state, tie, dtype):
    fake = tk.FakeEngineTransformer(tie, dtype=dtype)
    params_before = tk.named_params(fake)
    ptrs_before = {name: param.data_ptr() for name, param in params_before.items()}
    reported_before = parameter_pointers(fake)
    assert reported_before == ptrs_before

    load_weights_inplace(fake, hf_state)
    load_weights_inplace(fake, tk.perturbed_state(tk.make_tiny_qwen3(0), seed=3))  # a second step

    params_after = tk.named_params(fake)
    assert set(params_after) == set(params_before)
    for name, param in params_after.items():
        assert param is params_before[name], f"{name} was re-assigned"
        assert param.data_ptr() == ptrs_before[name], f"{name} moved"
        assert param.dtype == dtype and not param.requires_grad
    assert parameter_pointers(fake) == reported_before
    shared = fake.lm_head.weight.data_ptr() == fake.embed_tokens.weight.data_ptr()
    assert shared is (tie == "shared")


def test_engine_keeps_copies_not_references(tk, hf_state):
    fake = tk.FakeEngineTransformer("shared")
    load_weights_inplace(fake, hf_state)
    snap = tk.snapshot(fake)
    source_ptrs = {v.data_ptr() for v in hf_state.values()}
    assert not source_ptrs & set(parameter_pointers(fake).values())
    for value in hf_state.values():
        value.add_(1.0)  # the trainer keeps updating its own tensors
    tk.assert_matches_snapshot(fake, snap)


def test_loading_inference_tensors_under_inference_mode(tk, hf_state):
    """How SparseSpecEngine uses it: parameters created in inference mode,
    load / verify called under inference mode, sources are ordinary tensors."""
    with torch.inference_mode():
        fake = tk.FakeEngineTransformer("shared", dtype=torch.bfloat16)
    assert fake.embed_tokens.weight.is_inference()
    ptrs = parameter_pointers(fake)
    with torch.inference_mode():
        load_weights_inplace(fake, hf_state)
        report = verify_weights(fake, hf_state)
    assert report["ok"] is True
    assert parameter_pointers(fake) == ptrs
    oracle = tk.expected_engine_state(hf_state, torch.bfloat16)
    for name, param in tk.named_params(fake).items():
        assert tk.bit_equal(param, oracle[name]), name


def test_sources_with_requires_grad_and_pair_iterables(tk, tiny_model):
    """The actor's live parameters (requires_grad=True) are a valid source, as
    is any iterable of (name, tensor) pairs."""
    oracle = tk.expected_engine_state(tiny_model.state_dict(), torch.float32)

    fake = tk.FakeEngineTransformer("shared")
    live = list(tiny_model.named_parameters(remove_duplicate=False))
    assert all(p.requires_grad for _, p in live)
    load_weights_inplace(fake, live)
    assert all(tk.bit_equal(p, oracle[n]) for n, p in tk.named_params(fake).items())
    assert not any(p.requires_grad for p in fake.parameters())

    fake = tk.FakeEngineTransformer("copy")
    load_weights_inplace(fake, ((k, v) for k, v in tiny_model.state_dict().items()))  # a generator
    assert all(tk.bit_equal(p, oracle[n]) for n, p in tk.named_params(fake).items())

    fake = tk.FakeEngineTransformer("copy")
    load_weights_inplace(fake, OrderedDict(reversed(list(tiny_model.state_dict().items()))))  # any key order
    assert all(tk.bit_equal(p, oracle[n]) for n, p in tk.named_params(fake).items())


# ---------------------------------------------------------------------------
# wrapper prefixes
# ---------------------------------------------------------------------------
def test_prefix_nestings_cover_every_wrapper_order():
    assert "" in PREFIX_NESTINGS
    assert "_fsdp_wrapped_module.model." in PREFIX_NESTINGS  # FSDP1 around an HF causal LM
    assert "module._fsdp_wrapped_module.model." in PREFIX_NESTINGS
    assert len(PREFIX_NESTINGS) >= 16


@pytest.mark.parametrize("prefix", PREFIX_NESTINGS)
def test_normalize_hf_key(prefix):
    for name in ("layers.0.self_attn.q_proj.weight", "embed_tokens.weight", "lm_head.weight", "norm.weight"):
        assert normalize_hf_key(prefix + name) == name


def test_normalize_hf_key_only_strips_leading_wrappers():
    assert normalize_hf_key("layers.0.mlp.model.weight") == "layers.0.mlp.model.weight"
    assert normalize_hf_key("models.weight") == "models.weight"  # "model." must match as a whole component
    assert normalize_hf_key("lm_head.module.weight") == "lm_head.module.weight"


@pytest.mark.parametrize("prefix", PREFIX_NESTINGS)
def test_prefixed_state_dicts_load(tk, hf_state, prefix):
    state = with_prefix(hf_state, prefix)
    assert all(k.startswith(prefix) for k in state)
    fake = tk.FakeEngineTransformer("shared")
    report = load_weights_inplace(fake, state)
    oracle = tk.expected_engine_state(hf_state, torch.float32)
    for name, param in tk.named_params(fake).items():
        assert tk.bit_equal(param, oracle[name]), name
    assert report.num_sources == len(state)
    assert verify_weights(fake, state)["ok"] is True


def test_mixed_prefixes_in_one_state_dict(tk, hf_state):
    """Each key with a different nesting (HF itself mixes ``model.x`` with ``lm_head.weight``)."""
    cycle = itertools.cycle(PREFIX_NESTINGS)
    state = {next(cycle) + bare(k): v for k, v in hf_state.items()}
    fake = tk.FakeEngineTransformer("copy", dtype=torch.bfloat16)
    load_weights_inplace(fake, state)
    oracle = tk.expected_engine_state(hf_state, torch.bfloat16)
    for name, param in tk.named_params(fake).items():
        assert tk.bit_equal(param, oracle[name]), name


def test_keys_that_collide_after_stripping_raise(tk, hf_state):
    state = dict(hf_state)
    state["_fsdp_wrapped_module.model.norm.weight"] = hf_state["model.norm.weight"].clone()
    with pytest.raises(ValueError, match="duplicate"):
        normalize_state_dict(state)
    fake = tk.FakeEngineTransformer("shared")
    snap = tk.snapshot(fake)
    with pytest.raises(ValueError, match="duplicate"):
        load_weights_inplace(fake, state)
    tk.assert_matches_snapshot(fake, snap)


# ---------------------------------------------------------------------------
# tied embeddings
# ---------------------------------------------------------------------------
def _without(state: dict, *keys: str) -> dict:
    return {k: v for k, v in state.items() if k not in keys}


@pytest.mark.parametrize("tie", ["shared", "copy"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("case", ["embed_only", "lm_head_only", "both_equal"])
def test_tied_state_dict_variants(tk, hf_state, tie, dtype, case):
    state = {
        "embed_only": _without(hf_state, "lm_head.weight"),
        "lm_head_only": _without(hf_state, "model.embed_tokens.weight"),
        "both_equal": dict(hf_state),
    }[case]
    assert ("lm_head.weight" in state, "model.embed_tokens.weight" in state) == {
        "embed_only": (False, True),
        "lm_head_only": (True, False),
        "both_equal": (True, True),
    }[case]
    fake = tk.FakeEngineTransformer(tie, dtype=dtype)
    report = load_weights_inplace(fake, state)

    tied_value = hf_state["model.embed_tokens.weight"].to(dtype)
    assert tk.bit_equal(fake.embed_tokens.weight, tied_value)
    assert tk.bit_equal(fake.lm_head.weight, tied_value)  # also when it is a separate tensor
    oracle = tk.expected_engine_state(hf_state, dtype)
    for name, param in tk.named_params(fake).items():
        assert tk.bit_equal(param, oracle[name]), name
    assert report.num_sources == len(state)

    report = verify_weights(fake, state)
    assert report["ok"] is True and report["tied_ok"] is True
    assert report["tied_shared_storage"] is (tie == "shared")


@pytest.mark.parametrize("tie", ["shared", "copy"])
def test_tied_but_different_tensors_raise(tk, hf_state, tie):
    state = dict(hf_state)
    state["lm_head.weight"] = hf_state["lm_head.weight"].clone()
    state["lm_head.weight"][5, 3] += 1.0e-3  # a single element apart
    fake = tk.FakeEngineTransformer(tie)
    snap = tk.snapshot(fake)
    with pytest.raises(ValueError, match="tie_word_embeddings"):
        load_weights_inplace(fake, state)
    tk.assert_matches_snapshot(fake, snap)
    with pytest.raises(ValueError, match="tie_word_embeddings"):
        verify_weights(fake, state)

    state["lm_head.weight"] = hf_state["lm_head.weight"][:-1]  # different shape
    with pytest.raises(ValueError, match="tie_word_embeddings"):
        load_weights_inplace(fake, state)
    tk.assert_matches_snapshot(fake, snap)


@pytest.mark.parametrize("tie", ["shared", "copy"])
def test_tied_model_needs_one_of_the_two(tk, hf_state, tie):
    fake = tk.FakeEngineTransformer(tie)
    snap = tk.snapshot(fake)
    with pytest.raises(KeyError, match="neither"):
        load_weights_inplace(fake, _without(hf_state, "lm_head.weight", "model.embed_tokens.weight"))
    tk.assert_matches_snapshot(fake, snap)


def test_tied_pair_given_in_mixed_dtypes_is_compared_by_value(tk, hf_state):
    """embed in float32 and lm_head in bf16 of the same values are not "equal"
    unless they agree after casting; a bf16-exact pair must be accepted."""
    exact = hf_state["model.embed_tokens.weight"].to(torch.bfloat16)
    state = dict(hf_state)
    state["model.embed_tokens.weight"] = exact.float()
    state["lm_head.weight"] = exact
    fake = tk.FakeEngineTransformer("copy", dtype=torch.bfloat16)
    load_weights_inplace(fake, state)
    assert tk.bit_equal(fake.embed_tokens.weight, exact) and tk.bit_equal(fake.lm_head.weight, exact)


def test_untied_engine_takes_both_matrices(tk, hf_state):
    state = dict(hf_state)
    state["lm_head.weight"] = hf_state["lm_head.weight"] + 0.25
    fake = tk.FakeEngineTransformer("untied")
    load_weights_inplace(fake, state)
    assert tk.bit_equal(fake.embed_tokens.weight, state["model.embed_tokens.weight"])
    assert tk.bit_equal(fake.lm_head.weight, state["lm_head.weight"])
    report = verify_weights(fake, state)
    assert report["ok"] is True and report["tied_shared_storage"] is None

    snap = tk.snapshot(fake)
    with pytest.raises(KeyError, match="missing"):
        load_weights_inplace(fake, _without(hf_state, "lm_head.weight"))
    with pytest.raises(KeyError, match="missing"):
        load_weights_inplace(fake, _without(hf_state, "model.embed_tokens.weight"))
    tk.assert_matches_snapshot(fake, snap)


# ---------------------------------------------------------------------------
# strictness: a bad state dict raises and changes nothing
# ---------------------------------------------------------------------------
def _drop(key):
    def edit(state):
        del state[key]

    return edit


def _add(key, shape):
    def edit(state):
        state[key] = torch.zeros(shape)

    return edit


def _reshape(key, shape):
    def edit(state):
        state[key] = torch.full(shape, 0.5)

    return edit


def _retype(key, dtype):
    def edit(state):
        state[key] = state[key].to(dtype)

    return edit


def _tied_reshape(shape):
    def edit(state):
        state["model.embed_tokens.weight"] = torch.full(shape, 0.5)
        state["lm_head.weight"] = state["model.embed_tokens.weight"]

    return edit


BAD_STATE_DICTS = {
    # --- missing ---------------------------------------------------------
    "missing_first_key": (_drop("model.layers.0.self_attn.q_proj.weight"), KeyError),
    "missing_plain_key": (_drop("model.layers.0.self_attn.q_norm.weight"), KeyError),
    "missing_fused_part": (_drop("model.layers.1.mlp.up_proj.weight"), KeyError),
    "missing_last_key": (_drop("model.norm.weight"), KeyError),
    # --- unexpected ------------------------------------------------------
    "unexpected_o_proj_bias": (_add("model.layers.0.self_attn.o_proj.bias", (32,)), KeyError),
    "unexpected_qkv_bias": (_add("model.layers.1.self_attn.q_proj.bias", (32,)), KeyError),
    "unexpected_top_level": (_add("model.extra_head.weight", (4, 4)), KeyError),
    "unexpected_extra_layer": (_add("model.layers.2.input_layernorm.weight", (32,)), KeyError),
    # --- wrong shape (first, middle and last parameter) --------------------
    "shape_embeddings": (_tied_reshape((161, 32)), ValueError),
    "shape_plain_matrix": (_reshape("model.layers.0.self_attn.o_proj.weight", (32, 31)), ValueError),
    "shape_fused_rows": (_reshape("model.layers.0.self_attn.v_proj.weight", (17, 32)), ValueError),
    "shape_fused_columns": (_reshape("model.layers.1.self_attn.k_proj.weight", (16, 31)), ValueError),
    "shape_fused_mlp": (_reshape("model.layers.1.mlp.gate_proj.weight", (64, 33)), ValueError),
    "shape_last_key": (_reshape("model.norm.weight", (33,)), ValueError),
    # --- not a floating point tensor ---------------------------------------
    "integer_tensor": (_retype("model.norm.weight", torch.int64), TypeError),
}


@pytest.mark.parametrize("defect", sorted(BAD_STATE_DICTS))
@pytest.mark.parametrize("tie", ["shared", "copy"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_bad_state_dict_raises_and_leaves_every_parameter_untouched(tk, tiny_model, defect, tie, dtype):
    edit, error = BAD_STATE_DICTS[defect]
    # values differ from the engine's in EVERY tensor, so a partial write cannot hide
    state = tk.perturbed_state(tiny_model, seed=1)
    fake = tk.FakeEngineTransformer(tie, dtype=dtype)
    load_weights_inplace(fake, tiny_model.state_dict())
    snap = tk.snapshot(fake)
    ptrs = parameter_pointers(fake)

    # sanity: without the defect this state dict loads and changes everything
    probe = copy.deepcopy(fake)
    load_weights_inplace(probe, state)
    assert all(not tk.bit_equal(p, snap[n]) for n, p in tk.named_params(probe).items())

    edit(state)
    with pytest.raises(error):
        load_weights_inplace(fake, state)
    tk.assert_matches_snapshot(fake, snap)
    assert parameter_pointers(fake) == ptrs
    # the engine is still usable: the next good state dict loads
    good = tk.perturbed_state(tiny_model, seed=2)
    load_weights_inplace(fake, good)
    assert verify_weights(fake, good)["ok"] is True


def test_error_messages_name_the_offending_keys(tk, hf_state):
    fake = tk.FakeEngineTransformer("shared")
    state = dict(hf_state)
    del state["model.layers.1.mlp.up_proj.weight"]
    with pytest.raises(KeyError, match=r"layers\.1\.mlp\.up_proj\.weight"):
        load_weights_inplace(fake, state)
    state = dict(hf_state)
    state["model.layers.0.self_attn.o_proj.bias"] = torch.zeros(32)
    with pytest.raises(KeyError, match=r"layers\.0\.self_attn\.o_proj\.bias"):
        load_weights_inplace(fake, state)
    state = dict(hf_state)
    state["model.norm.weight"] = torch.zeros(33)
    with pytest.raises(ValueError, match=r"norm\.weight"):
        load_weights_inplace(fake, state)


def test_qwen3_state_dict_into_engine_without_q_norm_is_refused(tk, hf_state):
    """serve/model/model.py only creates q_norm / k_norm when the model path
    contains "qwen3"; an engine built without them must not load silently."""
    fake = tk.FakeEngineTransformer("shared")
    for layer in fake.layers:
        del layer.self_attn.q_norm
        del layer.self_attn.k_norm
    snap = tk.snapshot(fake)
    with pytest.raises(KeyError, match="q_norm"):
        load_weights_inplace(fake, hf_state)
    tk.assert_matches_snapshot(fake, snap)


# ---------------------------------------------------------------------------
# verify_weights
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tie", ["shared", "copy"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_verify_reports_ok_after_a_load(tk, hf_state, tie, dtype):
    fake = tk.FakeEngineTransformer(tie, dtype=dtype)
    stale = verify_weights(fake, hf_state)
    assert stale["ok"] is False
    assert stale["num_mismatched"] == stale["num_params_checked"] == len(tk.named_params(fake))

    load_weights_inplace(fake, hf_state)
    snap = tk.snapshot(fake)
    report = verify_weights(fake, hf_state)
    assert report["ok"] is True
    assert report["num_mismatched"] == 0 and report["mismatched"] == []
    assert report["num_params_checked"] == len(tk.named_params(fake))
    assert report["num_elements_checked"] == sum(p.numel() for p in tk.named_params(fake).values())
    assert report["tied_ok"] is True
    assert report["tied_shared_storage"] is (tie == "shared")
    tk.assert_matches_snapshot(fake, snap)  # verifying never writes


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "target, flat_index",
    [
        ("layers.0.self_attn.wqkv.weight", 0),  # first element, q part
        ("layers.1.self_attn.wqkv.weight", 40 * 32 + 7),  # inside the k part
        ("layers.1.mlp.w12.weight", 127 * 32 + 31),  # last element, up part
        ("layers.0.self_attn.q_norm.weight", 3),
        ("layers.1.mlp.down_proj.weight", 1000),
        ("norm.weight", 31),
    ],
)
def test_verify_detects_a_single_flipped_bit(tk, hf_state, dtype, target, flat_index):
    fake = tk.FakeEngineTransformer("shared", dtype=dtype)
    load_weights_inplace(fake, hf_state)
    param = tk.named_params(fake)[target]
    flip_one_bit(param, flat_index)  # the smallest possible corruption: one ulp of one element

    report = verify_weights(fake, hf_state)
    assert report["ok"] is False
    assert report["num_mismatched"] == 1
    assert [m["name"] for m in report["mismatched"]] == [target]
    assert report["mismatched"][0]["num_diff"] == 1
    assert report["mismatched"][0]["max_abs_diff"] > 0.0
    assert report["tied_ok"] is True

    flip_one_bit(param, flat_index)  # restore
    assert verify_weights(fake, hf_state)["ok"] is True


def test_verify_flipped_element_in_shared_embedding_shows_under_both_names(tk, hf_state):
    fake = tk.FakeEngineTransformer("shared")
    load_weights_inplace(fake, hf_state)
    flip_one_bit(fake.embed_tokens.weight, 17)
    report = verify_weights(fake, hf_state)
    assert report["ok"] is False
    assert sorted(m["name"] for m in report["mismatched"]) == ["embed_tokens.weight", "lm_head.weight"]
    assert report["tied_ok"] is True  # still one tensor


def test_verify_is_bitwise_negative_zero_differs_from_zero(tk, hf_state):
    state = dict(hf_state)
    state["model.norm.weight"] = hf_state["model.norm.weight"].clone()
    state["model.norm.weight"][4] = 0.0
    fake = tk.FakeEngineTransformer("shared")
    load_weights_inplace(fake, state)
    assert verify_weights(fake, state)["ok"] is True
    with torch.no_grad():
        fake.norm.weight[4] = -0.0
    assert bool(fake.norm.weight[4] == 0.0)  # equal as numbers ...
    report = verify_weights(fake, state)
    assert report["ok"] is False  # ... but not the bits the trainer sent
    assert [m["name"] for m in report["mismatched"]] == ["norm.weight"]


def test_verify_detects_a_broken_tie(tk, hf_state):
    fake = tk.FakeEngineTransformer("copy")
    load_weights_inplace(fake, hf_state)
    assert verify_weights(fake, hf_state)["tied_ok"] is True

    flip_one_bit(fake.lm_head.weight, 123)  # lm_head drifts away from embed_tokens
    report = verify_weights(fake, hf_state)
    assert report["tied_ok"] is False
    assert report["ok"] is False
    assert report["tied_shared_storage"] is False
    assert [m["name"] for m in report["mismatched"]] == ["lm_head.weight"]

    # same when the two still match their own sources but not each other is impossible for a tied
    # model (one source feeds both), so breaking embed_tokens instead must also be caught
    fake = tk.FakeEngineTransformer("copy")
    load_weights_inplace(fake, hf_state)
    flip_one_bit(fake.embed_tokens.weight, 0)
    report = verify_weights(fake, hf_state)
    assert report["tied_ok"] is False and report["ok"] is False
    assert [m["name"] for m in report["mismatched"]] == ["embed_tokens.weight"]


def test_verify_reports_shape_mismatch_instead_of_raising(tk, hf_state):
    fake = tk.FakeEngineTransformer("shared")
    load_weights_inplace(fake, hf_state)
    state = dict(hf_state)
    state["model.norm.weight"] = torch.ones(33)
    report = verify_weights(fake, state)
    assert report["ok"] is False
    assert report["mismatched"] == [{"name": "norm.weight", "shape_engine": [32], "shape_expected": [33]}]


def test_verify_applies_the_load_mapping_not_plain_equality(tk, hf_state):
    """A bf16 engine verified against the float32 source is ok (expectation is
    cast like the load casts), and verified against other weights is not."""
    fake = tk.FakeEngineTransformer("copy", dtype=torch.bfloat16)
    load_weights_inplace(fake, hf_state)
    assert verify_weights(fake, with_prefix(hf_state, "_fsdp_wrapped_module.model."))["ok"] is True
    other = tk.perturbed_state(tk.make_tiny_qwen3(0), seed=5)
    report = verify_weights(fake, other)
    assert report["ok"] is False and report["num_mismatched"] == report["num_params_checked"]
    assert len(report["mismatched"]) <= 20  # the report is capped


# ---------------------------------------------------------------------------
# a real HF module (what ReferenceEngine holds): names carry the model. prefix
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("prefix", ["", "_fsdp_wrapped_module.", "module."])
def test_real_hf_module(tk, tiny_model, tiny_model_b, dtype, prefix):
    target = tiny_model_b.to(dtype).requires_grad_(False)
    source = {prefix + k: v for k, v in tiny_model.state_dict().items()}  # model.* and lm_head.weight
    assert any(k.startswith(prefix + "model.layers.") for k in source)

    ptrs = parameter_pointers(target)
    assert "lm_head.weight" in ptrs and "embed_tokens.weight" in ptrs  # normalized names
    assert "layers.0.self_attn.q_proj.weight" in ptrs
    assert ptrs["lm_head.weight"] == ptrs["embed_tokens.weight"]
    assert verify_weights(target, source)["ok"] is False

    with torch.no_grad():
        report = load_weights_inplace(target, source)

    for name, param in target.named_parameters(remove_duplicate=False):
        assert tk.bit_equal(param, tiny_model.state_dict()[name].to(dtype)), name
    assert parameter_pointers(target) == ptrs
    assert target.lm_head.weight is target.model.embed_tokens.weight
    assert report.num_params == len(list(target.parameters()))
    assert report.num_sources == len(source)
    assert report.num_bytes == unique_parameter_bytes(target.parameters())

    result = verify_weights(target, source)
    assert result["ok"] is True and result["tied_ok"] is True and result["tied_shared_storage"] is True
    assert result["num_params_checked"] == len(list(target.named_parameters(remove_duplicate=False)))

    if dtype == torch.float32:  # same weights -> the very same function
        ids = torch.tensor([tk.make_prompts([12], seed=4)[0]])
        with torch.no_grad():
            assert torch.equal(target(input_ids=ids).logits, tiny_model(input_ids=ids).logits)


def test_real_hf_module_strictness(tk, tiny_model, tiny_model_b):
    target = tiny_model_b.requires_grad_(False)
    snap = tk.snapshot(target)
    state = tk.perturbed_state(tiny_model, seed=9)
    del state["model.layers.1.self_attn.v_proj.weight"]
    with torch.no_grad(), pytest.raises(KeyError, match="missing"):
        load_weights_inplace(target, state)
    tk.assert_matches_snapshot(target, snap)

    state = tk.perturbed_state(tiny_model, seed=9)
    state["model.layers.1.mlp.down_proj.weight"] = torch.zeros(32, 65)
    with torch.no_grad(), pytest.raises(ValueError, match="shape mismatch"):
        load_weights_inplace(target, state)
    tk.assert_matches_snapshot(target, snap)


def test_unique_parameter_bytes_counts_tied_storage_once(tk, tiny_model):
    fake_shared = tk.FakeEngineTransformer("shared", dtype=torch.bfloat16)
    fake_copy = tk.FakeEngineTransformer("copy", dtype=torch.bfloat16)
    embed_bytes = tk.VOCAB * tk.HIDDEN * 2
    shared = unique_parameter_bytes(p for _, p in fake_shared.named_parameters(remove_duplicate=False))
    separate = unique_parameter_bytes(p for _, p in fake_copy.named_parameters(remove_duplicate=False))
    assert separate - shared == embed_bytes
    assert unique_parameter_bytes(tiny_model.parameters()) == sum(p.numel() * 4 for p in tiny_model.parameters())
