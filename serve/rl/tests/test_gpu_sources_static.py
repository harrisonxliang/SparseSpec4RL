"""Static checks of the GPU-only modules, runnable without a GPU.

``serve/rl/engine.py`` and ``serve/rl/scheduler.py`` import flashinfer / triton
and cannot be imported here (and nothing is stubbed), so their first real run
is on the A100 box. These tests read the sources with ``ast`` and compare them
with the upstream ``serve/`` code they call into: imported names exist,
overridden methods keep the upstream signature, every keyword passed to an
upstream callable exists, every job argument the upstream schedulers read is
supplied, and every attribute read on an upstream object is defined there.

They prove nothing about behaviour (that is test_engine_gpu.py's job); they
catch the ImportError / TypeError / KeyError / AttributeError class of mistakes
before GPU time is spent on them.
"""

from __future__ import annotations

import ast
import inspect
import re

import pytest

from serve.rl import base, memory, weights

ENGINE = "serve/rl/engine.py"
RL_SCHEDULER = "serve/rl/scheduler.py"
BASE_SCHEDULER = "serve/scheduler/scheduler.py"
SPEC_SCHEDULER = "serve/scheduler/spec_scheduler.py"
REQUEST = "serve/request/request.py"
MODEL = "serve/model/model.py"
GRAPH_MODEL = "serve/model/graph_model.py"
CONFIG = "serve/model/config.py"
CACHE_BASE = "serve/request/kv_cache_ptr/base.py"
CACHE_STREAMING = "serve/request/kv_cache_ptr/streaming.py"
CACHE_PILLAR = "serve/request/kv_cache_ptr/pillar.py"
REJECTION_SAMPLER = "serve/sampling/rejection_sampler.py"
TP = "serve/distribute/tp.py"


@pytest.fixture(scope="module")
def src(tk):
    cache: dict = {}

    def load(rel: str):
        if rel not in cache:
            text = (tk.REPO_ROOT / rel).read_text()
            cache[rel] = (text, ast.parse(text))
        return cache[rel]

    return load


def _class(tree, name: str) -> ast.ClassDef:
    return next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name)


def _functions(node) -> dict:
    return {n.name: n for n in node.body if isinstance(n, ast.FunctionDef)}


def _params(func: ast.FunctionDef) -> list[str]:
    """Names a caller may pass (positional-or-keyword and keyword-only), without self / cls."""
    names = [a.arg for a in func.args.args + func.args.kwonlyargs]
    return names[1:] if names and names[0] in ("self", "cls") else names


def _required(func: ast.FunctionDef) -> set[str]:
    positional = [a.arg for a in func.args.args]
    required = positional[: len(positional) - len(func.args.defaults)]
    required += [a.arg for a, d in zip(func.args.kwonlyargs, func.args.kw_defaults) if d is None]
    return set(required) - {"self", "cls"}


def _calls(node, dotted: str) -> list[ast.Call]:
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and ast.unparse(n.func) == dotted]


def _keywords(call: ast.Call) -> set[str]:
    return {kw.arg for kw in call.keywords if kw.arg is not None}


def _defined_members(cls: ast.ClassDef) -> set[str]:
    """Everything an instance of the class can be asked for: methods / properties,
    class-level names (incl. dataclass fields) and ``self.x = ...`` assignments."""
    names = set(_functions(cls))
    for stmt in cls.body:
        if isinstance(stmt, ast.Assign):
            names |= {t.id for t in stmt.targets if isinstance(t, ast.Name)}
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            names.add(stmt.target.id)
        elif isinstance(stmt, ast.ClassDef):
            names.add(stmt.name)
    for node in ast.walk(cls):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Store)
            and isinstance(node.value, ast.Name)
            and node.value.id in ("self", "cls")
        ):
            names.add(node.attr)
    return names


def _string_keys_read(node, container: str) -> set[str]:
    """Keys read as ``<container>["key"]`` (loads only), e.g. container="kwargs" or "self._args"."""
    keys = set()
    for n in ast.walk(node):
        if (
            isinstance(n, ast.Subscript)
            and isinstance(n.ctx, ast.Load)
            and ast.unparse(n.value) == container
            and isinstance(n.slice, ast.Constant)
            and isinstance(n.slice.value, str)
        ):
            keys.add(n.slice.value)
    return keys


def _dict_literal(node, target: str) -> ast.Dict:
    """The dict literal assigned to ``target`` (e.g. "args" or "self._spec_args")."""
    return next(
        n.value
        for n in ast.walk(node)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == target and isinstance(n.value, ast.Dict)
    )


def _dict_keys(d: ast.Dict) -> set[str]:
    return {k.value for k in d.keys if isinstance(k, ast.Constant)}


# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rel", [ENGINE, RL_SCHEDULER])
def test_gpu_modules_parse_and_their_serve_imports_resolve(tk, src, rel):
    _, tree = src(rel)
    checked = 0
    for node in tree.body:
        if not (isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("serve.")):
            continue
        path = tk.REPO_ROOT / (node.module.replace(".", "/") + ".py")
        assert path.is_file(), f"{rel} imports from {node.module}, which is not a module file"
        target = ast.parse(path.read_text())
        defined = set()
        for stmt in ast.walk(target):
            if isinstance(stmt, ast.FunctionDef | ast.ClassDef):
                defined.add(stmt.name)
            elif isinstance(stmt, ast.Assign):
                defined |= {t.id for t in stmt.targets if isinstance(t, ast.Name)}
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                defined.add(stmt.target.id)
            elif isinstance(stmt, ast.Import | ast.ImportFrom):
                defined |= {(a.asname or a.name).split(".")[0] for a in stmt.names}
        for alias in node.names:
            assert alias.name in defined, f"{rel}: `from {node.module} import {alias.name}` does not resolve"
            checked += 1
    assert checked >= 4


def test_cpu_importable_names_used_by_the_engine_exist():
    """The serve.rl.* names the engine imports, checked against the real modules."""
    for module, names in (
        (base, ["GenerationResult", "LoadReport", "NamedTensors", "RolloutEngine", "empty_stats",
                "finalize_response", "normalize_eos_ids", "validate_prompts", "validate_sampling_params",
                "FINISH_LENGTH", "FINISH_STOP", "first_eos_index"]),
        (memory, ["GIB", "EngineLimits", "MemoryPlan", "check_workload_fits", "plan_memory", "request_token_need"]),
        (weights, ["load_weights_inplace", "parameter_pointers", "unique_parameter_bytes", "verify_weights"]),
    ):  # fmt: skip
        for name in names:
            assert hasattr(module, name), f"{module.__name__}.{name}"


def test_rl_scheduler_overrides_keep_the_upstream_signatures(src):
    rl = _class(src(RL_SCHEDULER)[1], "RLSpecScheduler")
    assert [ast.unparse(b) for b in rl.bases] == ["SpecScheduler"]
    spec = _class(src(SPEC_SCHEDULER)[1], "SpecScheduler")
    assert [ast.unparse(b) for b in spec.bases] == ["BaseScheduler"]
    upstream = {**_functions(_class(src(BASE_SCHEDULER)[1], "BaseScheduler")), **_functions(spec)}

    overrides = {name: fn for name, fn in _functions(rl).items() if not name.startswith("_")}
    assert set(overrides) == {
        "admit_requests",
        "add_request",
        "is_request_complete",
        "verify_outputs",
        "update_requests",
    }
    for name, ours in overrides.items():
        assert name in upstream, f"RLSpecScheduler.{name} overrides nothing"
        theirs = upstream[name]
        assert [a.arg for a in ours.args.args] == [a.arg for a in theirs.args.args], name
        assert (ours.args.kwarg is None) == (theirs.args.kwarg is None), name
        assert len(ours.args.defaults) == len(theirs.args.defaults), name
        # every override still runs the upstream implementation
        assert _calls(ours, f"super().{name}") or name == "is_request_complete", name

    # __init__(kv_pool, model, *, total_token_capacity, **job_args) on top of upstream's (kv_pool, model, **kwargs)
    init = _functions(rl)["__init__"]
    assert [a.arg for a in init.args.args] == ["self", "kv_pool", "model"]
    assert [a.arg for a in init.args.kwonlyargs] == ["total_token_capacity"] and init.args.kwarg is not None
    assert [a.arg for a in upstream["__init__"].args.args] == ["self", "kv_pool", "model"]
    assert upstream["__init__"].args.kwarg is not None
    # the driver loop the engine calls
    assert _params(upstream["run"]) == ["pending_requests"]


def test_engine_supplies_every_job_argument_the_schedulers_read(src):
    engine = _class(src(ENGINE)[1], "SparseSpecEngine")
    methods = _functions(engine)
    job_args = _dict_literal(methods["_job_args"], "args")
    spec_args = _dict_literal(methods["__init__"], "self._spec_args")
    supplied = _dict_keys(job_args) | _dict_keys(spec_args)
    assert _calls(methods["_job_args"], "args.update")  # ... merged into the job arguments

    required = set()
    for rel, name in (
        (BASE_SCHEDULER, "BaseScheduler"),
        (SPEC_SCHEDULER, "SpecScheduler"),
        (RL_SCHEDULER, "RLSpecScheduler"),
    ):
        cls = _class(src(rel)[1], name)
        required |= _string_keys_read(cls, "kwargs") | _string_keys_read(cls, "self._args")
    assert {"max_batch_size", "eos_token_id", "admit_policy", "async_cpu", "tp_ranks", "temperature",
            "chunk_size", "kv_cache", "spec_stride", "batch_policy"} <= required  # fmt: skip
    assert required <= supplied, f"job arguments read upstream but not supplied: {sorted(required - supplied)}"

    # the modes RLSpecScheduler insists on (it raises NotImplementedError otherwise)
    literal = {
        k.value: ast.literal_eval(v) for k, v in zip(job_args.keys, job_args.values) if isinstance(v, ast.Constant)
    }
    assert literal["admit_policy"] == "offloading"
    assert literal["async_cpu"] is False
    assert literal["tp_ranks"] == 1
    # the scheduler is given the job arguments and the admission budget
    call = _calls(methods["generate"], "RLSpecScheduler")
    assert len(call) == 1 and len(call[0].args) == 2
    assert _keywords(call[0]) == {"total_token_capacity"}
    assert any(kw.arg is None and "_job_args" in ast.unparse(kw.value) for kw in call[0].keywords)


def test_cache_class_initialisation_gets_what_it_reads(src):
    engine_init = _functions(_class(src(ENGINE)[1], "SparseSpecEngine"))["__init__"]
    call = _calls(engine_init, "PillarCachePtr.initialize_class_metadata")
    assert len(call) == 1 and not call[0].args
    passed = _keywords(call[0])
    spec_keys = _dict_keys(_dict_literal(engine_init, "self._spec_args"))

    read_kwargs, read_spec = set(), set()
    for rel, name in (
        (CACHE_BASE, "FullCachePtr"),
        (CACHE_STREAMING, "StreamingCachePtr"),
        (CACHE_PILLAR, "PillarCachePtr"),
    ):
        init = _functions(_class(src(rel)[1], name)).get("initialize_class_metadata")
        assert init is not None, name
        read_kwargs |= _string_keys_read(init, "kwargs")
        read_spec |= _string_keys_read(init, "spec_cache_args")
    assert {
        "spec_cache_args",
        "num_qo_heads",
        "num_kv_heads",
        "num_layers",
        "dtype",
        "head_dim",
        "device",
    } <= read_kwargs
    assert (
        read_kwargs <= passed
    ), f"initialize_class_metadata reads {sorted(read_kwargs - passed)} which the engine does not pass"
    assert {"spec_stride", "budget_ratio", "num_min_budget", "max_batch_size"} <= read_spec
    assert read_spec <= spec_keys, sorted(read_spec - spec_keys)


def test_engine_calls_only_pass_keywords_the_callees_accept(src):
    engine = _class(src(ENGINE)[1], "SparseSpecEngine")

    def check(dotted: str, callee: ast.FunctionDef, count: int = 1, positional: int = 0):
        calls = _calls(engine, dotted)
        assert len(calls) == count, f"{dotted}: expected {count} call(s), found {len(calls)}"
        for call in calls:
            params = _params(callee)
            assert len(call.args) == positional, dotted
            extra = _keywords(call) - set(params)
            assert not extra or callee.args.kwarg is not None, f"{dotted}: unknown keyword(s) {sorted(extra)}"
            given = set(params[:positional]) | _keywords(call)
            missing = _required(callee) - given
            assert not missing, f"{dotted}: missing required argument(s) {sorted(missing)}"

    model_tree = src(MODEL)[1]
    check("KVPool", _functions(_class(model_tree, "KVPool"))["__init__"])
    assert "cpu_capacity" in _keywords(_calls(engine, "KVPool")[0])  # the host pool is sized by the memory plan
    check("Transformer.from_pretrained", _functions(_class(model_tree, "Transformer"))["from_pretrained"])
    check("self.model.attach_kv_pool", _functions(_class(model_tree, "Transformer"))["attach_kv_pool"], positional=1)
    wrapper = _functions(_class(src(GRAPH_MODEL)[1], "CUDAGraphModelWrapper"))
    check("CUDAGraphModelWrapper", wrapper["__init__"], positional=1)
    check("runner.eager_cuda_graph_mode", wrapper["eager_cuda_graph_mode"])
    check("Request", _functions(_class(src(REQUEST)[1], "Request"))["__init__"])
    tp = next(n for n in src(TP)[1].body if isinstance(n, ast.FunctionDef) and n.name == "setup_tp_worker")
    check("setup_tp_worker", tp)

    # serve.rl.memory is importable: compare with the real signatures
    for dotted, fn in (("plan_memory", memory.plan_memory), ("check_workload_fits", memory.check_workload_fits)):
        (call,) = _calls(engine, dotted)
        signature = inspect.signature(fn)
        assert not call.args and _keywords(call) <= set(signature.parameters), dotted
        required = {n for n, p in signature.parameters.items() if p.default is inspect.Parameter.empty}
        assert required <= _keywords(call), f"{dotted}: missing {sorted(required - _keywords(call))}"
    (limits,) = _calls(engine, "EngineLimits")
    assert _keywords(limits) == set(memory.EngineLimits.__dataclass_fields__)  # every limit read from the real classes
    for call in _calls(engine, "admission_capacity_tokens"):
        inspect.signature(memory.admission_capacity_tokens).bind(*call.args, **{k: None for k in _keywords(call)})
    for call in _calls(engine, "finalize_response"):
        assert len(call.args) == len(inspect.signature(base.finalize_response).parameters)
    for dotted, fn in (
        ("load_weights_inplace", weights.load_weights_inplace),
        ("verify_weights", weights.verify_weights),
    ):
        for call in _calls(engine, dotted):
            assert len(call.args) == len(inspect.signature(fn).parameters), dotted


def test_attributes_read_on_upstream_objects_exist(src):
    engine_text, engine_tree = src(ENGINE)
    sched_text, sched_tree = src(RL_SCHEDULER)

    # --- KVPool ------------------------------------------------------------
    kv_pool = _defined_members(_class(src(MODEL)[1], "KVPool"))
    used = set(re.findall(r"\bkv_pool\.([A-Za-z_]\w*)", engine_text + sched_text))
    used |= set(re.findall(r"\bpool\.([A-Za-z_]\w*)", engine_text))  # `pool = self.kv_pool` in _check_pool_clean
    assert {"_physical_mem_gpu", "capacity", "num_free_pages", "page_size", "phase_idx", "_free_gpu", "_free_cpu",
            "_MAX_CPU_CAPACITY"} <= used  # fmt: skip
    assert used <= kv_pool, f"not defined on KVPool: {sorted(used - kv_pool)}"

    # --- ModelArgs (the engine's self.config) ---------------------------------
    model_args = _defined_members(_class(src(CONFIG)[1], "ModelArgs"))
    used = set(re.findall(r"\bself\.config\.([A-Za-z_]\w*)", engine_text))
    assert {"num_layers", "num_kv_heads", "num_qo_heads", "head_dim", "vocab_size", "dim"} <= used
    assert used <= model_args, f"not defined on ModelArgs: {sorted(used - model_args)}"
    for name in re.findall(r'getattr\(self\.config, "(\w+)"', engine_text):  # optional attributes still exist today
        assert name in model_args, name

    # --- Transformer ------------------------------------------------------------
    transformer = _defined_members(_class(src(MODEL)[1], "Transformer")) | {"parameters", "named_parameters"}
    used = set(re.findall(r"\bself\.model\.([A-Za-z_]\w*)", engine_text))
    assert used <= transformer, sorted(used - transformer)

    # --- kernel buffer constants read into EngineLimits -----------------------------
    cache_members = set()
    for rel, name in (
        (CACHE_BASE, "FullCachePtr"),
        (CACHE_STREAMING, "StreamingCachePtr"),
        (CACHE_PILLAR, "PillarCachePtr"),
    ):
        cache_members |= _defined_members(_class(src(rel)[1], name))
    used = set(re.findall(r"\bPillarCachePtr\.([A-Za-z_]\w*)", engine_text))
    assert {"MAX_TOTAL_DRAFT_KV_LEN", "MAX_TOTAL_VERIFY_KV_LEN", "MAX_LEN_SELECTED_KV_CACHE", "_INITIALIZED"} <= used
    assert used <= cache_members, sorted(used - cache_members)
    sampler = _defined_members(_class(src(REJECTION_SAMPLER)[1], "RejectionSampler"))
    assert set(re.findall(r"\bRejectionSampler\.([A-Za-z_]\w*)", engine_text)) <= sampler

    # --- Request ---------------------------------------------------------------------
    request_cls = _class(src(REQUEST)[1], "Request")
    request = _defined_members(request_cls)
    exec_types = _defined_members(_class(request_cls, "ReqExecType"))
    assert set(re.findall(r"Request\.ReqExecType\.(\w+)", sched_text)) <= exec_types
    # attributes other code attaches to requests: the upstream driver loop and RLSpecScheduler itself
    attached = set(re.findall(r"\brequest\.(\w+) = ", src(BASE_SCHEDULER)[0]))
    assert "request_id" in attached
    attached |= set(re.findall(r"\b(?:request|head)\.(\w+) = ", sched_text))
    assert {"rl_logprobs", "rl_finish_reason", "_rl_need", "_rl_last_emitted", "_rl_verify_row"} <= attached
    # variables holding a Request (not the `serve.request.request` module path)
    used = set(re.findall(r"(?<![\w.])(?:request|head|r)\.([A-Za-z_]\w*)", sched_text))
    generate = ast.get_source_segment(engine_text, _functions(_class(engine_tree, "SparseSpecEngine"))["generate"])
    used |= set(re.findall(r"(?<![\w.])request\.([A-Za-z_]\w*)", generate))
    assert {"len_input", "len_output", "len_token_ids", "len_kv_cache", "len_oracle_ctx", "prefilled", "status",
            "token_ids", "len_draft_probs"} <= used  # fmt: skip
    assert used <= request | attached, f"not defined on Request: {sorted(used - request - attached)}"

    # --- the scheduler's own self.* --------------------------------------------------
    rl = _class(sched_tree, "RLSpecScheduler")
    upstream = _defined_members(_class(src(BASE_SCHEDULER)[1], "BaseScheduler")) | _defined_members(
        _class(src(SPEC_SCHEDULER)[1], "SpecScheduler")
    )
    used = set(re.findall(r"\bself\.([A-Za-z_]\w*)", ast.get_source_segment(sched_text, rl)))
    assert {"async_on", "admit_policy", "eos_token_id", "active_requests", "offload_requests", "kv_pool",
            "spec_stride", "_args"} <= used  # fmt: skip
    undefined = used - _defined_members(rl) - upstream
    assert not undefined, f"RLSpecScheduler reads self.{sorted(undefined)} which nothing defines"
