"""In-place weight reload: HuggingFace-named tensors -> engine parameters.

Pure torch (no flashinfer), so it is unit-tested on CPU against a module that
mimics the parameter layout of ``serve.model.model.Transformer``.

Name mapping mirrors ``Transformer.from_pretrained``:

* wrapper prefixes are stripped (``model.``, ``module.``, plus FSDP's
  ``_fsdp_wrapped_module.``);
* ``from_pretrained`` then FUSES projections (``Attention.merge`` /
  ``FeedForward.merge``), so the engine holds

    layers.N.self_attn.wqkv.weight = cat([q_proj, k_proj, v_proj].weight, dim=0)
    layers.N.mlp.w12.weight        = cat([gate_proj, up_proj].weight,   dim=0)

  (and the matching ``.bias`` when the model has attention biases);
* with ``tie_word_embeddings`` the state dict may carry ``embed_tokens.weight``,
  ``lm_head.weight`` or both.

Reloading is strict and two-phase: every engine parameter must be covered,
every provided tensor must be used, and shapes must match BEFORE the first
byte is written, so a bad state dict cannot leave the engine half updated.
Values are written with ``Tensor.copy_`` into the existing storage; no tensor
is ever reassigned (captured CUDA graphs hold raw pointers to them).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import torch

from serve.rl.base import LoadReport, NamedTensors

# Stripped repeatedly from the front of every key. No engine parameter name
# starts with any of these (engine names start with embed_tokens / layers /
# norm / lm_head), so stripping is unambiguous.
_WRAPPER_PREFIXES = ("_fsdp_wrapped_module.", "module.", "model.")

# engine suffix -> HF suffixes concatenated along dim 0, in this order
_FUSED = {
    ".wqkv.weight": (".q_proj.weight", ".k_proj.weight", ".v_proj.weight"),
    ".wqkv.bias": (".q_proj.bias", ".k_proj.bias", ".v_proj.bias"),
    ".w12.weight": (".gate_proj.weight", ".up_proj.weight"),
    ".w12.bias": (".gate_proj.bias", ".up_proj.bias"),
}

EMBED_KEY = "embed_tokens.weight"
LM_HEAD_KEY = "lm_head.weight"

_INT_VIEW = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def normalize_hf_key(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in _WRAPPER_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def normalize_state_dict(named_tensors: NamedTensors) -> dict[str, torch.Tensor]:
    items = named_tensors.items() if hasattr(named_tensors, "items") else named_tensors
    out: dict[str, torch.Tensor] = {}
    for key, value in items:
        norm = normalize_hf_key(key)
        if norm in out:
            raise ValueError(f"duplicate parameter after prefix stripping: {key!r} -> {norm!r}")
        out[norm] = value
    return out


@dataclass
class _Target:
    name: str  # engine parameter name
    param: torch.Tensor  # the live engine parameter (written in place)
    sources: tuple[str, ...]  # normalized HF keys, concatenated along dim 0


def _source_keys(engine_name: str) -> tuple[str, ...]:
    for suffix, parts in _FUSED.items():
        if engine_name.endswith(suffix):
            stem = engine_name[: -len(suffix)]
            return tuple(stem + p for p in parts)
    return (engine_name,)


def _is_tied(model) -> bool:
    return bool(getattr(getattr(model, "config", None), "tie_word_embeddings", False))


def _same_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.data_ptr() == b.data_ptr() and a.shape == b.shape


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise equality (distinguishes -0.0 from 0.0, treats equal NaNs as equal)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.is_floating_point():
        view = _INT_VIEW[a.element_size()]
        return bool(torch.equal(a.contiguous().view(view), b.contiguous().view(view)))
    return bool(torch.equal(a, b))


def _all_named_parameters(model) -> list[tuple[str, torch.Tensor]]:
    """(normalized name, parameter) for every parameter of ``model``.

    Names are normalized like state-dict keys, so the same code serves the
    SparseSpec ``Transformer`` (``layers.0...``) and a HuggingFace causal LM
    (``model.layers.0...``, used by the reference engine).

    remove_duplicate=False: a tied lm_head/embed pair must show up under both
    names so that both are checked and (if they are separate storages holding
    equal values) both are written.
    """
    out = []
    seen: set[str] = set()
    for name, param in model.named_parameters(remove_duplicate=False):
        norm = normalize_hf_key(name)
        if norm in seen:
            raise ValueError(f"two parameters normalize to the same name {norm!r}")
        seen.add(norm)
        out.append((norm, param))
    return out


def _build_targets(model) -> list[_Target]:
    return [_Target(name, param, _source_keys(name)) for name, param in _all_named_parameters(model)]


def _resolve_sources(model, state: dict[str, torch.Tensor]) -> tuple[list[_Target], set[str]]:
    """Check coverage and return the targets plus the set of consumed keys."""
    targets = _build_targets(model)
    tied = _is_tied(model)

    if tied:
        has_embed, has_head = EMBED_KEY in state, LM_HEAD_KEY in state
        if not has_embed and not has_head:
            raise KeyError(
                f"state dict has neither {EMBED_KEY!r} nor {LM_HEAD_KEY!r} for a model with tie_word_embeddings=True"
            )
        if has_embed and has_head:
            a, b = state[EMBED_KEY], state[LM_HEAD_KEY]
            if a.shape != b.shape or not bool(torch.equal(a.to(device=b.device, dtype=b.dtype), b)):
                raise ValueError(
                    "tie_word_embeddings=True but the provided embed_tokens.weight and lm_head.weight differ; "
                    "the engine keeps a single tied matrix and cannot represent untied weights"
                )
        tied_source = EMBED_KEY if has_embed else LM_HEAD_KEY

    used: set[str] = set()
    missing: list[str] = []
    for target in targets:
        if tied and target.name in (EMBED_KEY, LM_HEAD_KEY):
            target.sources = (tied_source,)
            used.update(k for k in (EMBED_KEY, LM_HEAD_KEY) if k in state)
            continue
        absent = [k for k in target.sources if k not in state]
        if absent:
            missing.extend(absent)
        else:
            used.update(target.sources)

    if missing:
        raise KeyError(
            f"state dict is missing {len(missing)} tensor(s) required by the engine, e.g. {sorted(missing)[:6]}"
        )
    unexpected = sorted(set(state) - used)
    if unexpected:
        raise KeyError(
            f"state dict has {len(unexpected)} tensor(s) the engine has no parameter for, e.g. {unexpected[:6]}. "
            "For Qwen3 this usually means the engine was built without q_norm/k_norm "
            "(serve/model/model.py only creates them when the model path contains 'qwen3')."
        )
    return targets, used


def _assemble(target: _Target, state: dict[str, torch.Tensor]) -> torch.Tensor:
    """Source value for one engine parameter, on its device and in its dtype.

    Each part is cast before concatenation; casting is elementwise, so this
    equals casting the concatenation. fp32 -> bf16 uses round-to-nearest-even,
    the same rounding FSDP mixed precision and vLLM's weight loader apply.
    """
    param = target.param
    parts = [state[k].detach().to(device=param.device, dtype=param.dtype) for k in target.sources]
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)


def _expected_part_rows(model, target: _Target) -> tuple[int, ...] | None:
    """Rows each fused source must contribute, read from the module that owns
    the fused parameter (None when the layout cannot be determined)."""
    if len(target.sources) == 1:
        return None
    owner = None
    for name, module in model.named_modules():
        for attr in ("wqkv", "w12"):
            fused = getattr(module, attr, None)
            if isinstance(fused, torch.nn.Module) and any(p is target.param for p in fused.parameters(recurse=False)):
                owner, kind = module, attr
    if owner is None:
        return None
    if kind == "w12":  # FeedForward.forward: silu_and_mul splits the output in two halves
        return (target.param.shape[0] // 2,) * 2
    heads = [getattr(owner, a, None) for a in ("num_qo_heads", "num_kv_heads", "head_dim")]
    if any(h is None for h in heads):
        return None
    qo, kv, dim = heads  # Attention.forward: split([qo * dim, kv * dim, kv * dim])
    return (qo * dim, kv * dim, kv * dim)


def _expected_shape(target: _Target, state: dict[str, torch.Tensor], model=None) -> tuple[int, ...]:
    shapes = [tuple(state[k].shape) for k in target.sources]
    if len(shapes) == 1:
        return shapes[0]
    tail = shapes[0][1:]
    for key, shape in zip(target.sources, shapes):
        if shape[1:] != tail:
            raise ValueError(f"cannot fuse {target.sources}: {key!r} has shape {shape}, expected (*, {tail})")
    rows = _expected_part_rows(model, target) if model is not None else None
    if rows is not None and tuple(s[0] for s in shapes) != rows:
        raise ValueError(
            f"cannot fuse {target.sources} into {target.name}: the parts have {tuple(s[0] for s in shapes)} rows "
            f"but the engine splits the fused tensor as {rows}"
        )
    return (sum(s[0] for s in shapes),) + tail


def load_weights_inplace(model, named_tensors: NamedTensors) -> LoadReport:
    """Overwrite every parameter of ``model`` in place. See module docstring.

    Call under ``torch.inference_mode()`` when the parameters were created in
    inference mode (the SparseSpec engine's are); ``no_grad`` otherwise.
    """
    start = time.perf_counter()
    state = normalize_state_dict(named_tensors)
    targets, used = _resolve_sources(model, state)

    # Phase 1: validate everything before touching a single parameter.
    for target in targets:
        expected = _expected_shape(target, state, model)
        if tuple(target.param.shape) != expected:
            raise ValueError(
                f"shape mismatch for {target.name}: engine {tuple(target.param.shape)} vs "
                f"state dict {expected} (from {target.sources})"
            )
        for key in target.sources:
            if not state[key].is_floating_point():
                raise TypeError(f"{key!r} is not a floating point tensor ({state[key].dtype})")

    # Phase 2: in-place copies. Shared storages (a structurally tied
    # lm_head/embed pair) are written once.
    written: list[torch.Tensor] = []
    num_bytes = 0
    for target in targets:
        param = target.param
        if any(_same_storage(param, done) for done in written if done.shape == param.shape):
            continue
        param.copy_(_assemble(target, state))
        written.append(param)
        num_bytes += param.numel() * param.element_size()

    if _is_tied(model):
        params = dict(_all_named_parameters(model))
        if EMBED_KEY in params and LM_HEAD_KEY in params:
            if not _bits_equal(params[EMBED_KEY], params[LM_HEAD_KEY]):
                raise RuntimeError("lm_head.weight and embed_tokens.weight are no longer tied after reload")

    return LoadReport(
        num_params=len(written),
        num_sources=len(used),
        num_bytes=num_bytes,
        seconds=time.perf_counter() - start,
    )


def verify_weights(model, named_tensors: NamedTensors) -> dict:
    """Bitwise comparison of the live parameters against a state dict.

    The expectation is the state dict mapped exactly the way
    ``load_weights_inplace`` maps it (prefix stripping, fusion, cast to the
    parameter dtype). Returns a report; ``report["ok"]`` is True only if every
    parameter matches bit for bit and tied weights are still tied.
    """
    state = normalize_state_dict(named_tensors)
    targets, _ = _resolve_sources(model, state)
    mismatched: list[dict] = []
    checked_elems = 0
    for target in targets:
        expected = _assemble(target, state)
        actual = target.param
        checked_elems += actual.numel()
        if not _bits_equal(actual, expected):
            if actual.shape == expected.shape:
                diff = (actual.float() - expected.float()).abs()
                detail = {"num_diff": int((diff != 0).sum().item()), "max_abs_diff": float(diff.max().item())}
            else:
                detail = {"shape_engine": list(actual.shape), "shape_expected": list(expected.shape)}
            mismatched.append({"name": target.name, **detail})

    tied_ok = True
    tied_shared_storage = None
    if _is_tied(model):
        params = dict(_all_named_parameters(model))
        if EMBED_KEY in params and LM_HEAD_KEY in params:
            tied_ok = _bits_equal(params[EMBED_KEY], params[LM_HEAD_KEY])
            tied_shared_storage = _same_storage(params[EMBED_KEY], params[LM_HEAD_KEY])

    return {
        "ok": not mismatched and tied_ok,
        "num_params_checked": len(targets),
        "num_elements_checked": checked_elems,
        "num_mismatched": len(mismatched),
        "mismatched": mismatched[:20],
        "tied_ok": tied_ok,
        "tied_shared_storage": tied_shared_storage,
    }


def parameter_pointers(model) -> dict[str, int]:
    """data_ptr of every parameter; unchanged pointers prove a reload was in place."""
    return {name: param.data_ptr() for name, param in _all_named_parameters(model)}


def unique_parameter_bytes(params: Iterable[torch.Tensor]) -> int:
    seen: set[int] = set()
    total = 0
    for param in params:
        ptr = param.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        total += param.numel() * param.element_size()
    return total
