"""Gate 3b: greedy decoding, rollout engine vs HuggingFace transformers (and stock vLLM) on the same weights.

    python -m integration.gates.gate_b_greedy --engine sparsespec          # engine vs HF only (two-way)
    python -m integration.gates.gate_b_greedy --engine sparsespec \
        --vllm-json baseline/results/sparsespec_checks/gate_b_vllm_greedy.json --tag three_way   # the real gate
    python -m integration.gates.gate_b_greedy --compare-only --tag three_way --vllm-json <file> [--stored-margins]
    python -m integration.gates.gate_b_greedy --engine reference           # plumbing check
    python -m integration.gates.gate_b_greedy --engine reference --device cpu \
        --model /tmp/tiny-qwen3 --prompts-file /tmp/tiny-qwen3/prompts.json --max-new-tokens 32   # CPU smoke

The vLLM file comes from ``integration/gates/gate_b_vllm_greedy.py``, run in the
baseline env (stock vLLM 0.9.1, enforce_eager) with the same --model,
--num-prompts, --max-new-tokens, --prompt-seed and --data / --prompts-file.

What it does
  1. ``--num-prompts`` (16) MATH training prompts, built as verl builds them.
  2. The engine generates all of them in ONE call at temperature 0,
     ``--max-new-tokens`` (256) each.
  3. HuggingFace generates each prompt on its own (bfloat16, batch of one, no
     padding) with a plain argmax loop.
  4. Per prompt: the first position where the two token sequences differ
     (-1 = identical). At that position the shared prefix is teacher-forced
     through HF and the HF logit MARGIN between HF's top token and the
     engine's token is reported. bf16 arithmetic differs between the two
     implementations, so a flip at a near-tie is expected now and then; a
     flip against a large margin means the engine computed something else.
  5. The engine's and HF's responses are written to
     ``gate_b_greedy_<tag>.engine.json`` / ``.hf.json`` (the schema of the
     vLLM file, ``common.read_greedy_tokens``), with the HF margins the
     comparison used, so every verdict can be recomputed from files
     (``--compare-only``).

Two-way verdict (no ``--vllm-json``; thresholds are flags, printed with the result)
  PASS iff (a) the engine's responses obey the response contract,
           (b) neither response is a strict prefix of the other,
           (c) every divergence has HF margin <= ``--tie-margin`` logits,
           (d) at least ``--min-identical`` prompts match exactly (default 0),
           (e) only with ``--max-divergence-ratio R``: the engine-vs-HF divergence
               rate is at most R x the HF-vs-HF flip rate (see below).
  As a yardstick for (c) the report also counts HF-vs-HF flips: positions
  where HF's own argmax changes between its cached step-by-step forward and
  one uncached forward over the same tokens, with their margins. The output
  says prominently that the three-way comparison was NOT done
  (``"three_way": false``).

Three-way verdict (``--vllm-json``): SparseSpec (S) vs HF, vLLM (V) vs HF, S vs V
  Pairs and margins. HF is the common yardstick: for each pair and prompt, d =
  first position where the two responses differ, and the pair's shared prefix
  is teacher-forced through HF. For X vs HF the margin is HF's top logit
  minus HF's logit of X's token (as in the two-way gate); for S vs V it is
  the larger of the two tokens' shortfalls from HF's top logit. vLLM's own
  top-1 minus top-2 logit margin at that step (from its file) is reported
  next to it whenever vLLM's response shares that prefix.
  Contract: both engines obey the response contract, and no response of any
  pair is a strict prefix of the other (impossible between two responses
  that obey the contract with the same EOS ids; the longer one is blamed).
  R1: every divergence of every pair is a near-tie (margin <= --tie-margin).
      A vLLM-vs-HF divergence that is not a near-tie also FAILS the gate, but
      is labelled a problem of the REFERENCE engines (the comparison is not
      usable), not of SparseSpec; an S-vs-V non-tie names the side(s) whose
      token is far from HF's top.
  R2: "S's divergence profile against HF is no worse than V's against HF",
      by ``--profile-rule`` (default ``significance``):
      * significance: first divergences are events of a constant per-token
        hazard. D_S, D_V = number of prompts where S / V diverge from HF;
        E_S, E_V = exposure, the positions compared up to and including the
        first divergence (the whole response if none; a strict-prefix pair is
        censored at the shorter length). Under equal hazards, D_S given
        D_S + D_V is Binomial(D_S + D_V, E_S / (E_S + E_V)); FAIL if the exact
        one-sided p-value P(X >= D_S) < ``--profile-alpha`` (0.05).
      * strict: FAIL unless D_S <= D_V and S's earliest divergence is not
        earlier than V's earliest.
      Both are always computed and printed (counts, earliest positions,
      per-token rates, p-value); the header and the JSON name the one that
      decides.
  Why ``strict`` is not the default. With 16 prompts the counts are tiny and
  the two engines' divergences are random (bf16 near-ties): for two engines
  with IDENTICAL per-token hazards the literal rule fails a correct engine
  with probability (exact, ``strict_rule_false_fail_probability``, 16
  prompts x 256 tokens, q = chance that one response diverges at all):
      q = 0.05: 0.43    q = 0.10: 0.55    q = 0.25: 0.64
      q = 0.50: 0.67    q = 0.75: 0.68    q = 0.90: 0.66
  (checked against a 40,000-trial simulation, agreement within 0.003), i.e.
  a coin flip or worse as soon as divergences happen at all; it passes
  reliably only when nobody diverges. The significance rule's false-FAIL
  rate for equal hazards stays below ``--profile-alpha`` (simulated in the
  same model at alpha 0.05: 0.011 / 0.026 / 0.030 / 0.036 for q = 0.10 /
  0.25 / 0.50 / 0.75; the exact test on small counts is conservative);
  its power: it flags SparseSpec when its rate is clearly higher, e.g. 8 vs
  1 divergences at equal exposure gives p = 0.02.
  The output also prints the strict rule's false-fail probability at the
  observed pooled hazard.
  PASS iff contract, R1 and R2 hold (and (d) / (e) above if set).

Outputs: ``<out-dir>/gate_b_greedy_<tag>.json`` and ``.md`` (``--compare-only``:
``gate_b_greedy_<tag>_recomputed.json`` / ``.md``), plus the two token files.
Exit status 0 PASS, 1 FAIL, 2 REFUSED (inputs do not match: different
prompts, max_new_tokens, EOS ids or model in the vLLM file; or margins missing
for ``--stored-margins``; the result JSON then holds ``"pass": false,
"refused": true`` and the reason, so no stale result survives), anything
else = crashed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

from integration.gates import common  # noqa: E402

# (pair name, side a, side b); the first two compare against the HF yardstick
PAIRS = (("engine_vs_hf", "engine", "hf"), ("vllm_vs_hf", "vllm", "hf"), ("engine_vs_vllm", "engine", "vllm"))
PROFILE_RULES = ("significance", "strict")
DIVERGED = ("near_tie", "MISMATCH")


class InputMismatch(Exception):
    """The inputs cannot be compared (different prompts / settings, missing margins). Exit status 2."""


def bf16_ulp(value: float) -> float:
    """Spacing of bfloat16 values around ``value`` (8 significant bits)."""
    value = abs(float(value))
    if value == 0.0 or not math.isfinite(value):
        return 0.0
    return 2.0 ** (math.floor(math.log2(value)) - 7)


def analyse_prompt(model, prompt, engine_tokens, hf_tokens, tie_margin: float, *, logits=None) -> dict:
    """Compare one engine response with HF's and measure the margin at the first difference.

    ``logits``: ``common.hf_teacher_forced_logits(model, prompt, hf_tokens)`` if already computed.
    """
    first_diff = common.first_difference(engine_tokens, hf_tokens)
    record = {
        "prompt_len": len(prompt),
        "engine_len": len(engine_tokens),
        "hf_len": len(hf_tokens),
        "first_diff": first_diff,
        "status": "identical",
        "hf_token": None,
        "engine_token": None,
        "tf_top_token": None,
        "margin": None,
        "margin_bf16_ulps": None,
        "engine_token_rank": None,
        "p_top": None,
        "p_engine_token": None,
    }

    # One uncached forward over prompt + HF response: row j is HF's
    # distribution after the prefix both sequences share up to j.
    if logits is None:
        logits = common.hf_teacher_forced_logits(model, prompt, hf_tokens)
    hf_ids = torch.tensor(hf_tokens, dtype=torch.long, device=logits.device)
    top_values, top_ids = logits.max(dim=-1)
    own = logits.gather(-1, hf_ids.unsqueeze(-1)).squeeze(-1)
    self_flip = top_ids != hf_ids
    record["hf_self_flips"] = int(self_flip.sum().item())
    record["hf_self_flip_max_margin"] = float((top_values - own)[self_flip].max().item()) if bool(self_flip.any()) else 0.0

    if first_diff < 0:
        return record
    if first_diff >= min(len(engine_tokens), len(hf_tokens)):
        # one response is a strict prefix of the other: with identical tokens up
        # to there, one side stopped without EOS or continued past one
        record["status"] = "LENGTH_MISMATCH"
        return record

    row = logits[first_diff]
    engine_token = int(engine_tokens[first_diff])
    top_token = int(top_ids[first_diff].item())
    margin = float((row[top_token] - row[engine_token]).item())
    probs = torch.softmax(row, dim=-1)
    ulp = bf16_ulp(float(row[top_token].item()))
    record.update(
        hf_token=int(hf_tokens[first_diff]),
        engine_token=engine_token,
        tf_top_token=top_token,
        margin=margin,
        margin_bf16_ulps=(margin / ulp) if ulp > 0 else None,
        engine_token_rank=int((row > row[engine_token]).sum().item()),
        p_top=float(probs[top_token].item()),
        p_engine_token=float(probs[engine_token].item()),
        status="near_tie" if margin <= tie_margin else "MISMATCH",
    )
    return record


# ----------------------------------------------------------------------------
# three-way comparison: HF margins (computed once, stored, looked up)
# ----------------------------------------------------------------------------
def margin_key(prompt_index: int, prefix) -> str:
    """Cache key of HF's next-token row after ``prompt + prefix``."""
    digest = hashlib.sha256(json.dumps([int(t) for t in prefix], separators=(",", ":")).encode()).hexdigest()
    return f"{prompt_index}:{len(prefix)}:{digest[:20]}"


def margin_requests(sides: dict, i: int) -> list[dict]:
    """The HF rows the three-way comparison of prompt ``i`` needs: one per diverging pair.

    ``sides`` maps "engine" / "hf" / "vllm" to lists of responses (vllm may be None).
    The prefix is taken from HF's response for X-vs-HF pairs, from the engine's for S-vs-V.
    """
    requests = []
    for pair, a, b in PAIRS:
        if sides.get(a) is None or sides.get(b) is None:
            continue
        seq_a, seq_b = sides[a][i], sides[b][i]
        d = common.first_difference(seq_a, seq_b)
        if d < 0 or d >= min(len(seq_a), len(seq_b)):
            continue
        source = "hf" if b == "hf" else a
        requests.append(
            {"pair": pair, "source": source, "position": d, "tokens": sorted({int(seq_a[d]), int(seq_b[d])}),
             "key": margin_key(i, sides[source][i][:d])}
        )  # fmt: skip
    return requests


def _margin_entry(i: int, position: int, source: str, row: torch.Tensor) -> dict:
    values, ids = row.topk(2)
    return {
        "prompt": i,
        "position": position,
        "source": source,
        "top_token": int(ids[0]),
        "top_logit": float(values[0]),
        "second_token": int(ids[1]),
        "second_logit": float(values[1]),
        "logits": {},
    }


@torch.no_grad()
def fill_margins(model, i: int, prompt, sides: dict, cache: dict, *, hf_logits=None) -> int:
    """Compute (into ``cache``) every HF row prompt ``i`` needs that is not there yet; returns how many.

    A row whose prefix is a prefix of HF's own response is read from
    ``hf_logits`` (the teacher-forced logits along HF's response, the same
    tensor the two-way analysis uses); any other prefix is teacher-forced
    along the response it comes from.
    """
    added = 0
    forwards: dict = {}
    hf_seq = sides["hf"][i]
    for req in margin_requests(sides, i):
        entry = cache.get(req["key"])
        if entry is not None and all(str(t) in entry["logits"] for t in req["tokens"]):
            continue
        pos, seq = req["position"], sides[req["source"]][i]
        if pos < len(hf_seq) and list(hf_seq[:pos]) == list(seq[:pos]):
            source = "hf"
            if hf_logits is None:
                hf_logits = common.hf_teacher_forced_logits(model, prompt, hf_seq)
            row = hf_logits[pos]
        else:
            source = req["source"]
            if source not in forwards:
                forwards[source] = common.hf_teacher_forced_logits(model, prompt, seq)
            row = forwards[source][pos]
        row = row.to(torch.float32)
        if entry is None:
            entry = cache[req["key"]] = _margin_entry(i, pos, source, row)
        for t in req["tokens"]:
            entry["logits"][str(t)] = float(row[t])
        added += 1
    return added


def _vllm_step(vllm_side: dict | None, i: int, d: int, prefix) -> tuple:
    """vLLM's own (margin, top-2 ids) at step d of prompt i, if vLLM's response shares ``prefix``."""
    if vllm_side is None:
        return None, None
    seq = vllm_side["token_ids"][i]
    if d >= len(seq) or list(seq[:d]) != list(prefix):
        return None, None
    margins, top2 = vllm_side.get("margins"), vllm_side.get("top2_token_ids")
    return (margins[i][d] if margins else None), (top2[i][d] if top2 else None)


def analyse_three_way(sides: dict, vllm_side: dict, cache: dict, tie_margin: float) -> dict:
    """Per pair and prompt: first difference, status, HF margin, the two tokens, vLLM's own margin.

    Pure function of the three token lists and the margin cache; raises
    ``InputMismatch`` listing every margin the cache does not hold.
    """
    n = len(sides["hf"])
    pairs = {}
    missing = []
    for pair, a, b in PAIRS:
        records = []
        for i in range(n):
            seq_a, seq_b = sides[a][i], sides[b][i]
            d = common.first_difference(seq_a, seq_b)
            rec = {
                "prompt": i, "len_a": len(seq_a), "len_b": len(seq_b), "first_diff": d, "status": "identical",
                "exposure": len(seq_a), "token_a": None, "token_b": None, "hf_top_token": None, "hf_top_logit": None,
                "shortfall_a": None, "shortfall_b": None, "margin": None, "blame": [], "vllm_margin": None,
                "vllm_top2": None,
            }  # fmt: skip
            if d >= 0 and d >= min(len(seq_a), len(seq_b)):
                rec.update(status="LENGTH_MISMATCH", exposure=min(len(seq_a), len(seq_b)))
                rec["blame"] = [a if len(seq_a) > len(seq_b) else b]
            elif d >= 0:
                source = "hf" if b == "hf" else a
                prefix = sides[source][i][:d]
                token_a, token_b = int(seq_a[d]), int(seq_b[d])
                entry = cache.get(margin_key(i, prefix))
                if entry is None or str(token_a) not in entry["logits"] or str(token_b) not in entry["logits"]:
                    missing.append(f"{pair} prompt {i} position {d}")
                    records.append(rec)
                    continue
                top = entry["top_logit"]
                short_a = top - entry["logits"][str(token_a)]
                short_b = top - entry["logits"][str(token_b)]
                margin = short_a if b == "hf" else max(short_a, short_b)
                vllm_margin, vllm_top2 = _vllm_step(vllm_side, i, d, prefix)
                rec.update(
                    exposure=d + 1, token_a=token_a, token_b=token_b, hf_top_token=entry["top_token"], hf_top_logit=top,
                    shortfall_a=short_a, shortfall_b=short_b, margin=margin, vllm_margin=vllm_margin,
                    vllm_top2=vllm_top2, status="near_tie" if margin <= tie_margin else "MISMATCH",
                )  # fmt: skip
                if rec["status"] == "MISMATCH":
                    if b == "hf":
                        rec["blame"] = [a]
                    else:
                        rec["blame"] = [s for s, short in ((a, short_a), (b, short_b)) if short > tie_margin]
            records.append(rec)
        pairs[pair] = {"a": a, "b": b, "per_prompt": records, "summary": _pair_summary(records)}
    if missing:
        raise InputMismatch(
            f"{len(missing)} HF margin(s) needed by the comparison are not stored ({', '.join(missing[:6])}"
            f"{', ...' if len(missing) > 6 else ''}); run --compare-only without --stored-margins so that the HF "
            "model computes them"
        )
    return pairs


def _pair_summary(records: list[dict]) -> dict:
    def count(status):
        return sum(1 for r in records if r["status"] == status)

    diverged = [r for r in records if r["status"] in DIVERGED]
    margins = [r["margin"] for r in diverged if r["margin"] is not None]
    return {
        "num_prompts": len(records),
        "identical": count("identical"),
        "near_tie": count("near_tie"),
        "mismatch": count("MISMATCH"),
        "length_mismatch": count("LENGTH_MISMATCH"),
        "divergences": len(diverged),
        "exposure": sum(r["exposure"] for r in records),
        "earliest_divergence": min((r["first_diff"] for r in diverged), default=None),
        "first_diffs": [r["first_diff"] for r in records],
        "max_margin": max(margins) if margins else None,
    }


# ----------------------------------------------------------------------------
# R2: divergence profile
# ----------------------------------------------------------------------------
def binomial_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), by exact summation (n is at most twice the prompt count)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    total = sum(math.comb(n, j) * p**j * (1.0 - p) ** (n - j) for j in range(k, n + 1))
    return min(1.0, max(0.0, total))


def strict_rule_false_fail_probability(hazard: float, num_prompts: int, length: int) -> float:
    """P(the strict rule fails S) when S and V have the SAME per-token hazard ``hazard``.

    Model: ``num_prompts`` responses of ``length`` tokens each, the first
    divergence of a response at position j with probability (1-h)^j h, none
    with s = (1-h)^length, independently for the two engines. Strict passes
    iff D_S = 0, or 1 <= D_S <= D_V and min_S >= min_V. Exact, by
    P(D = k, min >= m) = C(n, k) G(m)^k s^(n-k) with G(m) = (1-h)^m - s.
    """
    n, length = int(num_prompts), int(length)
    if hazard <= 0.0 or n <= 0 or length <= 0:
        return 0.0
    if hazard >= 1.0:
        return 0.0  # everyone diverges at position 0: D_S == D_V and min_S == min_V
    q = 1.0 - hazard
    s = q**length
    survive = [q**m - s for m in range(length + 1)]  # G(m) = P(first divergence at a position >= m)
    passed = s**n
    for k1 in range(1, n + 1):
        c1 = math.comb(n, k1) * s ** (n - k1)
        for m in range(length):
            p_s = c1 * (survive[m] ** k1 - survive[m + 1] ** k1)  # P_S(D = k1, min = m)
            if p_s <= 0.0:
                continue
            # P_V(D >= k1, min <= m)
            p_v = sum(math.comb(n, k2) * s ** (n - k2) * ((1.0 - s) ** k2 - survive[m + 1] ** k2) for k2 in range(k1, n + 1))
            passed += p_s * p_v
    return min(1.0, max(0.0, 1.0 - passed))


def divergence_profile(engine_vs_hf: list[dict], vllm_vs_hf: list[dict], *, rule: str, alpha: float, length: int) -> dict:
    """R2 numbers and both rules' outcomes (``rule`` decides)."""
    if rule not in PROFILE_RULES:
        raise ValueError(f"profile rule must be one of {PROFILE_RULES}, got {rule!r}")

    def side(records):
        diverged = [r for r in records if r["status"] in DIVERGED]
        exposure = sum(r["exposure"] for r in records)
        return {
            "divergences": len(diverged),
            "exposure": exposure,
            "rate": len(diverged) / exposure if exposure else 0.0,
            "earliest": min((r["first_diff"] for r in diverged), default=None),
        }

    s, v = side(engine_vs_hf), side(vllm_vs_hf)
    n = s["divergences"] + v["divergences"]
    total = s["exposure"] + v["exposure"]
    share = s["exposure"] / total if total else 0.5
    p_value = binomial_sf(s["divergences"], n, share) if n else 1.0
    significance_ok = p_value >= alpha
    strict_ok = s["divergences"] <= v["divergences"] and (
        s["earliest"] is None or (v["earliest"] is not None and s["earliest"] >= v["earliest"])
    )
    pooled = n / total if total else 0.0
    return {
        "rule": rule,
        "alpha": alpha,
        "engine": s,
        "vllm": v,
        "trials": n,
        "engine_exposure_share": share,
        "p_value": p_value,
        "significance_ok": significance_ok,
        "strict_ok": strict_ok,
        "ok": significance_ok if rule == "significance" else strict_ok,
        "pooled_hazard": pooled,
        "length_for_strict_estimate": int(length),
        "strict_false_fail_at_pooled_hazard": strict_rule_false_fail_probability(
            pooled, len(engine_vs_hf), length
        ),
    }


def profile_lines(profile: dict, engine_name: str) -> list[str]:
    s, v = profile["engine"], profile["vllm"]
    return [
        f"literal: {engine_name} diverges from HF on {s['divergences']} prompt(s), vLLM on {v['divergences']}; earliest "
        f"divergence {common.fmt_num(s['earliest'])} vs {common.fmt_num(v['earliest'])}; per-token rate "
        f"{s['divergences']}/{s['exposure']} = {s['rate']:.3g} vs {v['divergences']}/{v['exposure']} = {v['rate']:.3g} "
        f"-> strict rule {'ok' if profile['strict_ok'] else 'FAILS'} (D_S <= D_V and earliest_S >= earliest_V)",
        f"significance: exact one-sided conditional binomial, H0 equal hazards: P(X >= {s['divergences']}) for "
        f"X ~ Binomial({profile['trials']}, {profile['engine_exposure_share']:.3f}) = {profile['p_value']:.4g} "
        f"-> {'ok' if profile['significance_ok'] else 'FAILS'} (--profile-alpha = {profile['alpha']})",
        f"context: at the pooled hazard {profile['pooled_hazard']:.3g}/token ({len_note(profile)}) the strict rule "
        f"would fail an equally good engine with probability {profile['strict_false_fail_at_pooled_hazard']:.2f}",
        f"deciding rule: --profile-rule = {profile['rule']} -> R2 {'ok' if profile['ok'] else 'FAILS'}",
    ]


def len_note(profile: dict) -> str:
    return f"responses of ~{profile['length_for_strict_estimate']} tokens"


# ----------------------------------------------------------------------------
# input checks for the three-way mode
# ----------------------------------------------------------------------------
def load_side(path: str, side: str) -> dict:
    try:
        return common.read_greedy_tokens(path, side=side)
    except common.GreedyFileError as exc:
        raise InputMismatch(str(exc)) from exc


def check_sides_match(reference: dict, other: dict, *, label: str, allow_model_mismatch: bool) -> None:
    """Refuse (``InputMismatch``) unless ``other`` was produced for the same prompts, settings and model."""
    problem = common.describe_prompt_mismatch(reference["prompts"], other["prompts"])
    if problem is not None:
        raise InputMismatch(
            f"the {label} prompts are not the prompts of this run: {problem}. Regenerate it with the same --model, "
            "--num-prompts, --prompt-seed and --data / --prompts-file (prompts of this run: "
            f"sha256 {reference['prompts_sha256'][:16]}..., source {(reference.get('prompts_meta') or {}).get('source')}; "
            f"{label}: source {(other.get('prompts_meta') or {}).get('source')})"
        )
    if int(other["max_new_tokens"]) != int(reference["max_new_tokens"]):
        raise InputMismatch(
            f"the {label} was generated with max_new_tokens={other['max_new_tokens']}, this run uses "
            f"{reference['max_new_tokens']}"
        )
    if float(other.get("temperature", 0.0)) != 0.0:
        raise InputMismatch(f"the {label} is not greedy (temperature={other.get('temperature')})")
    if sorted(other["eos_token_ids"]) != sorted(reference["eos_token_ids"]):
        raise InputMismatch(
            f"the {label} stops on EOS ids {sorted(other['eos_token_ids'])}, this run on {sorted(reference['eos_token_ids'])}"
        )
    ours, theirs = reference.get("model"), other.get("model")
    if ours and theirs and os.path.realpath(ours) != os.path.realpath(theirs) and not allow_model_mismatch:
        raise InputMismatch(
            f"the {label} was generated from model {theirs}, this run uses {ours} (pass --allow-model-mismatch if they "
            "are known to hold the same weights)"
        )
    if other.get("side") == "vllm":
        if int(other.get("num_logprobs") or 0) < 2 or "margins" not in other or "top2_token_ids" not in other:
            raise InputMismatch(f"the {label} has no per-step top-2 data (needs logprobs=2; rerun gate_b_vllm_greedy.py)")


def _contract_of(side: dict, eos, *, greedy: bool) -> list[str]:
    logprobs = side.get("logprobs") or side.get("sampled_logprobs") or [[0.0] * len(t) for t in side["token_ids"]]
    return common.check_responses(
        types.SimpleNamespace(token_ids=side["token_ids"], logprobs=logprobs, finish_reasons=side["finish_reasons"]),
        int(side["max_new_tokens"]),
        eos,
        greedy=greedy,
    )


def vllm_contract_problems(side: dict, eos) -> list[str]:
    """``check_responses`` for the vLLM side, minus one documented vLLM convention.

    vLLM V1 (``v1/core/sched/utils.py::check_stop``) tests the length cap
    BEFORE the EOS, so a response whose ``max_tokens``-th token is an EOS is
    reported with ``finish_reason="length"``. The token ids are those of the
    contract (the EOS is the last token), so only the label differs; it is
    read as "stop" here. Everything else is checked as for the engines.
    """
    eos_set = {int(t) for t in eos}
    limit = int(side["max_new_tokens"])
    reasons = [
        "stop" if (r == "length" and len(t) == limit and t and t[-1] in eos_set) else r
        for t, r in zip(side["token_ids"], side["finish_reasons"])
    ]
    return _contract_of({**side, "finish_reasons": reasons}, eos, greedy=False)


# ----------------------------------------------------------------------------
# verdicts
# ----------------------------------------------------------------------------
def two_way_checks(summary: dict, problems: list[str], args) -> common.Checks:
    checks = common.Checks()
    checks.add(
        "engine_response_contract",
        not problems,
        "all responses end with their first EOS or have exactly max_new_tokens tokens; greedy log-probs are 0.0"
        if not problems
        else f"{len(problems)} violation(s), e.g. {problems[:3]}",
    )
    checks.add(
        "no_length_mismatch",
        summary["length_mismatch"] == 0,
        f"{summary['length_mismatch']} prompt(s) where one response is a strict prefix of the other",
    )
    checks.add(
        "divergences_are_near_ties",
        summary["mismatch"] == 0,
        f"{summary['mismatch']} divergence(s) with HF margin > --tie-margin={args.tie_margin} logits; "
        f"{summary['near_tie']} near-tie divergence(s), largest margin {common.fmt_num(summary['max_margin'])}; "
        f"{summary['identical']}/{summary['num_prompts']} identical",
    )
    _optional_checks(checks, summary["identical"], summary["divergence_rate"], summary["hf_self_flip_rate"],
                     summary["hf_tokens"], args)  # fmt: skip
    return checks


def _optional_checks(checks, identical: int, divergence_rate: float, hf_flip_rate: float, hf_tokens: int, args) -> None:
    checks.add(
        "min_identical",
        identical >= args.min_identical,
        f"{identical} identical prompt(s), --min-identical={args.min_identical}",
    )
    if args.max_divergence_ratio is not None:
        floor_rate = max(hf_flip_rate, 1.0 / max(hf_tokens, 1))
        limit = args.max_divergence_ratio * floor_rate
        checks.add(
            "divergence_rate",
            divergence_rate <= limit,
            f"engine vs HF diverges at {100 * divergence_rate:.3f}% of positions; limit "
            f"--max-divergence-ratio={args.max_divergence_ratio} x {100 * floor_rate:.3f}% (HF vs HF) = {100 * limit:.3f}%",
        )


def side_label(side: str, engine_name: str) -> str:
    return {"engine": engine_name, "hf": "HF", "vllm": "vLLM"}[side]


def pair_label(pair: str, engine_name: str) -> str:
    _, a, b = next(p for p in PAIRS if p[0] == pair)
    return f"{side_label(a, engine_name)} vs {side_label(b, engine_name)}"


def blame_text(blame: list[str], engine_name: str) -> str:
    parts = []
    if "engine" in blame:
        parts.append(f"ENGINE PROBLEM ({engine_name})")
    if "vllm" in blame or "hf" in blame:
        parts.append(f"REFERENCE-ENGINE PROBLEM (vLLM vs HF; not a {engine_name} problem)")
    return " + ".join(parts) or "-"


def three_way_checks(three: dict, engine_problems, vllm_problems, vllm_anomalies, hf_self_flip_rate, hf_tokens, args,
                     engine_name: str) -> tuple[common.Checks, dict]:  # fmt: skip
    """Contract, R1 per pair, R2, optional rules; plus who is to blame for each failed check."""
    checks = common.Checks()
    blame: dict[str, list[str]] = {}

    def add(name, ok, detail, who):
        checks.add(name, ok, detail)
        if not ok:
            blame[name] = who

    add(
        "engine_response_contract",
        not engine_problems,
        "all responses end with their first EOS or have exactly max_new_tokens tokens; greedy log-probs are 0.0"
        if not engine_problems
        else f"{len(engine_problems)} violation(s), e.g. {engine_problems[:3]}",
        ["engine"],
    )
    vllm_bad = list(vllm_problems) + [f"step {a['position']} of prompt {a['prompt']}: sampled token not top-1"
                                      for a in vllm_anomalies]  # fmt: skip
    add(
        "vllm_response_contract",
        not vllm_bad,
        "vLLM file: responses obey the contract and every step is greedy"
        if not vllm_bad
        else f"REFERENCE-ENGINE PROBLEM (stock vLLM output; not SparseSpec): {len(vllm_bad)} issue(s), e.g. {vllm_bad[:3]}",
        ["vllm"],
    )
    pairs = three["pairs"]
    for pair, a, b in PAIRS:
        recs = pairs[pair]["per_prompt"]
        label = pair_label(pair, engine_name)
        prefixes = [r for r in recs if r["status"] == "LENGTH_MISMATCH"]
        who = sorted({s for r in prefixes for s in r["blame"]})
        add(
            f"no_strict_prefix_{pair}",
            not prefixes,
            f"{label}: no response is a strict prefix of the other"
            if not prefixes
            else f"{label}: {len(prefixes)} prompt(s) where one response is a strict prefix of the other (prompts "
            f"{[r['prompt'] for r in prefixes]}; the longer one continued past an EOS or max_new_tokens): "
            f"{blame_text(who, engine_name)}",
            who,
        )
    for pair, a, b in PAIRS:
        recs = pairs[pair]["per_prompt"]
        summary = pairs[pair]["summary"]
        label = pair_label(pair, engine_name)
        bad = [r for r in recs if r["status"] == "MISMATCH"]
        who = sorted({s for r in bad for s in r["blame"]})
        name = f"R1_near_ties_{pair}"
        if bad:
            items = "; ".join(
                f"prompt {r['prompt']} at {r['first_diff']}: margin {r['margin']:.3g} (HF shortfall of "
                f"{side_label(a, engine_name)}'s token {r['shortfall_a']:.3g}, of {side_label(b, engine_name)}'s "
                f"{r['shortfall_b']:.3g})"
                for r in bad[:4]
            )
            detail = (
                f"{blame_text(who, engine_name)}: {label}: {len(bad)} divergence(s) with HF margin > --tie-margin="
                f"{args.tie_margin}: {items}{' ...' if len(bad) > 4 else ''}"
            )
        else:
            detail = (
                f"{label}: {summary['divergences']} divergence(s), all near-ties (largest margin "
                f"{common.fmt_num(summary['max_margin'])} <= {args.tie_margin}); {summary['identical']}/"
                f"{summary['num_prompts']} identical"
            )
        add(name, not bad, detail, who)
    profile = three["profile"]
    add(
        f"R2_profile_{profile['rule']}",
        profile["ok"],
        f"{engine_name} vs HF no worse than vLLM vs HF (--profile-rule={profile['rule']}): "
        + " | ".join(profile_lines(profile, engine_name)[:2]),
        ["engine"],
    )
    ev = pairs["engine_vs_hf"]["summary"]
    rate = ev["divergences"] / max(ev["exposure"], 1)
    before = len(checks.items)
    _optional_checks(checks, ev["identical"], rate, hf_self_flip_rate, hf_tokens, args)
    for item in checks.items[before:]:
        if not item["ok"]:
            blame[item["name"]] = ["engine"]
    return checks, blame


# ----------------------------------------------------------------------------
# reports
# ----------------------------------------------------------------------------
NOT_THREE_WAY = (
    "THREE-WAY COMPARISON NOT DONE: no --vllm-json was given, so this verdict compares the engine with HF only "
    "(vLLM vs HF and engine vs vLLM were not checked)."
)


def build_markdown(result: dict) -> str:
    cfg, summary = result["config"], result["summary"]
    lines = [
        f"# Gate 3b - greedy decoding: {cfg['engine']} engine vs HuggingFace",
        "",
        f"**Verdict: {'PASS' if result['pass'] else 'FAIL'}**",
        "",
        f"**{NOT_THREE_WAY}**",
        "",
        f"- model: `{cfg['model']}` (HF reference: `{cfg['hf_model']}`, {cfg['hf_dtype']})",
        f"- prompts: {cfg['num_prompts']} ({result['prompts']['source']}), max_new_tokens={cfg['max_new_tokens']}, "
        f"temperature=0, device={cfg['device']}",
        f"- rule: PASS iff the engine obeys the response contract, no response is a strict prefix of the other, "
        f"every divergence has HF margin <= --tie-margin = {cfg['tie_margin']} logits, and at least "
        f"--min-identical = {cfg['min_identical']} prompts are identical"
        + (
            f", and the divergence rate is at most --max-divergence-ratio = {cfg['max_divergence_ratio']} x the HF-vs-HF flip rate"
            if cfg["max_divergence_ratio"] is not None
            else ""
        ),
        "",
        "## Summary",
        "",
        common.md_table(
            ["identical", "near-tie divergence", "MISMATCH (margin > tie-margin)", "LENGTH_MISMATCH", "max margin", "median first diff (diverging prompts)"],
            [[summary["identical"], summary["near_tie"], summary["mismatch"], summary["length_mismatch"],
              summary["max_margin"], summary["median_first_diff"]]],
        ),  # fmt: skip
        "",
        _yardstick_text(summary),
        "",
        "## Checks",
        "",
        result["checks_markdown"],
        "",
        "## Per prompt",
        "",
        "`first diff` = first response position where the token ids differ (-1: none). `margin` = HF logit of "
        "HF's top token minus HF logit of the engine's token, teacher-forcing the shared prefix; `ulps` = the "
        "same in units of the bf16 spacing at the top logit; `rank` = number of tokens HF prefers to the engine's.",
        "",
        _per_prompt_table(result["per_prompt"]),
        "",
        "## Engine statistics",
        "",
        common.md_table(["stat", "value"], [[k, v] for k, v in result["engine_stats"].items()]),
        "",
    ]
    return "\n".join(lines)


def _yardstick_text(summary: dict) -> str:
    return (
        f"Yardstick (not part of the verdict): HF vs HF (cached step-by-step vs one uncached forward) flips its own "
        f"argmax at {summary['hf_self_flips']} of {summary['hf_tokens']} positions "
        f"({100 * summary['hf_self_flip_rate']:.2f}%), largest margin "
        f"{common.fmt_num(summary['hf_self_flip_max_margin'])} logits. Engine vs HF: "
        f"{summary['near_tie'] + summary['mismatch']} divergence(s) in {summary['positions_compared']} positions with a "
        f"shared prefix ({100 * summary['divergence_rate']:.2f}%). A divergence between two slightly different models "
        "also tends to happen first where the margin is small, so a divergence RATE far above HF's own, or margins "
        "far above the HF-vs-HF ones, deserve a look even when every margin is under the limit."
    )


def _per_prompt_table(per_prompt: list[dict]) -> str:
    return common.md_table(
        ["prompt", "prompt len", "engine len", "HF len", "first diff", "status", "HF token", "engine token",
         "margin", "ulps", "rank", "p(HF top)", "p(engine token)"],
        [[i, r["prompt_len"], r["engine_len"], r["hf_len"], r["first_diff"], r["status"], r["hf_token"],
          r["engine_token"], r["margin"], r["margin_bf16_ulps"], r["engine_token_rank"], r["p_top"],
          r["p_engine_token"]] for i, r in enumerate(per_prompt)],
    )  # fmt: skip


def summary_block(result: dict) -> list[str]:
    """The one-screen summary of a three-way result (console and markdown)."""
    cfg, three = result["config"], result["three_way_result"]
    engine_name = cfg["engine"]
    lines = [
        f"three-way greedy: {pair_label('engine_vs_hf', engine_name)}, vLLM vs HF, {pair_label('engine_vs_vllm', engine_name)}; "
        f"{cfg['num_prompts']} prompts, T=0, {cfg['max_new_tokens']} tokens; vLLM: {cfg.get('vllm_engine')}",
        f"rules: R1 every divergence margin <= --tie-margin={cfg['tie_margin']} logits; R2 --profile-rule="
        f"{cfg['profile_rule']} (--profile-alpha={cfg['profile_alpha']}); contract: no strict-prefix pairs",
        f"{'pair':<26}{'identical':>10}{'near-tie':>10}{'MISMATCH':>10}{'prefix':>8}{'earliest':>10}{'max margin':>12}",
    ]
    for pair, _, _ in PAIRS:
        s = three["pairs"][pair]["summary"]
        lines.append(
            f"{pair_label(pair, engine_name):<26}{s['identical']:>10}{s['near_tie']:>10}{s['mismatch']:>10}"
            f"{s['length_mismatch']:>8}{common.fmt_num(s['earliest_divergence']):>10}{common.fmt_num(s['max_margin']):>12}"
        )
    lines += ["R2 " + line for line in profile_lines(three["profile"], engine_name)]
    return lines


def build_markdown_three_way(result: dict) -> str:
    cfg, three = result["config"], result["three_way_result"]
    engine_name = cfg["engine"]
    lines = [
        f"# Gate 3b - three-way greedy: {engine_name} vs HF, vLLM vs HF, {engine_name} vs vLLM",
        "",
        f"**Verdict: {'PASS' if result['pass'] else 'FAIL'}** (R2 rule: `{cfg['profile_rule']}`, alpha "
        f"{cfg['profile_alpha']}; --tie-margin {cfg['tie_margin']})",
        "",
    ]
    if result.get("failed_checks"):
        lines += [
            "Failed: " + "; ".join(f"`{f['name']}` ({blame_text(f['blame'], engine_name)})" for f in result["failed_checks"]),
            "",
        ]
    lines += [
        f"- model: `{cfg['model']}` (HF reference: `{cfg['hf_model']}`, {cfg['hf_dtype']}); vLLM file: "
        f"`{cfg['vllm_json']}` ({cfg.get('vllm_engine')})",
        f"- prompts: {cfg['num_prompts']} ({(result.get('prompts') or {}).get('source')}), sha256 "
        f"`{cfg['prompts_sha256'][:16]}...`, max_new_tokens={cfg['max_new_tokens']}, temperature=0",
        "",
        "## Summary",
        "",
        "```",
        *summary_block(result),
        "```",
        "",
        "## Checks",
        "",
        result["checks_markdown"],
        "",
        "Margins: HF is the yardstick; the pair's shared prefix is teacher-forced through HF. X vs HF: HF top logit "
        "minus HF logit of X's token. "
        f"{engine_name} vs vLLM: the larger of the two tokens' shortfalls from HF's top logit. `vLLM margin` = vLLM's "
        "own top-1 minus top-2 logit margin at that step (from its file), when vLLM's response shares the prefix.",
        "",
    ]
    for pair, a, b in PAIRS:
        la, lb = side_label(a, engine_name), side_label(b, engine_name)
        lines += [
            f"## {pair_label(pair, engine_name)}",
            "",
            common.md_table(
                ["prompt", "first diff", "status", "margin", f"{la} token", f"{lb} token", "HF top", f"{la} shortfall",
                 f"{lb} shortfall", "vLLM margin", "blame"],
                [[r["prompt"], r["first_diff"], r["status"], r["margin"], r["token_a"], r["token_b"], r["hf_top_token"],
                  r["shortfall_a"], r["shortfall_b"], r["vllm_margin"], blame_text(r["blame"], engine_name) if r["blame"] else ""]
                 for r in three["pairs"][pair]["per_prompt"]],
            ),
            "",
        ]  # fmt: skip
    if result.get("summary"):
        lines += ["## HF yardstick", "", _yardstick_text(result["summary"]), ""]
    if result.get("engine_stats"):
        lines += ["## Engine statistics", "", common.md_table(["stat", "value"], [[k, v] for k, v in result["engine_stats"].items()]), ""]
    return "\n".join(lines)


def three_way_output(*, args, engine_name, engine_side, hf_side, vllm_side, vllm_path, cache) -> tuple[dict, common.Checks]:
    """Three-way comparison + verdict from the three sides (dicts in the token-file schema) and a margin cache."""
    sides = {"engine": engine_side["token_ids"], "hf": hf_side["token_ids"], "vllm": vllm_side["token_ids"]}
    pairs = analyse_three_way(sides, vllm_side, cache, args.tie_margin)
    lengths = [len(t) for t in hf_side["token_ids"]]
    profile = divergence_profile(
        pairs["engine_vs_hf"]["per_prompt"], pairs["vllm_vs_hf"]["per_prompt"], rule=args.profile_rule,
        alpha=args.profile_alpha, length=round(sum(lengths) / len(lengths)),
    )  # fmt: skip
    three = {"pairs": pairs, "profile": profile}
    eos = engine_side["eos_token_ids"]
    hf_tokens = sum(lengths)
    hf_flips = sum(hf_side.get("hf_self_flips") or [0])
    checks, blame = three_way_checks(
        three, _contract_of(engine_side, eos, greedy=True), vllm_contract_problems(vllm_side, eos),
        vllm_side.get("greedy_anomalies") or [], hf_flips / max(hf_tokens, 1), hf_tokens, args, engine_name,
    )  # fmt: skip
    output = {
        "gate": "3b_greedy",
        "three_way": True,
        "pass": checks.ok,
        "profile_rule": args.profile_rule,
        "profile_alpha": args.profile_alpha,
        "config": {
            "engine": engine_name,
            "model": engine_side.get("model"),
            "hf_model": hf_side.get("model"),
            "hf_dtype": hf_side.get("hf_dtype"),
            "num_prompts": len(sides["hf"]),
            "max_new_tokens": engine_side["max_new_tokens"],
            "eos_token_ids": eos,
            "tie_margin": args.tie_margin,
            "min_identical": args.min_identical,
            "max_divergence_ratio": args.max_divergence_ratio,
            "profile_rule": args.profile_rule,
            "profile_alpha": args.profile_alpha,
            "prompts_sha256": engine_side["prompts_sha256"],
            "vllm_json": os.path.abspath(vllm_path),
            "vllm_engine": vllm_side.get("engine"),
            "vllm_engine_kwargs": vllm_side.get("engine_kwargs"),
            "vllm_versions": vllm_side.get("versions"),
        },
        "prompts": engine_side.get("prompts_meta"),
        "three_way_result": three,
        # compact per-pair view (also what integration/summarize_bundle.py reads)
        "pairs": {
            pair: {
                "label": pair_label(pair, engine_name),
                "summary": {k: v for k, v in pairs[pair]["summary"].items()},
                "pass": all(c["ok"] for c in checks.items if c["name"] in (f"R1_near_ties_{pair}", f"no_strict_prefix_{pair}")),
                "rule": f"every divergence margin <= {args.tie_margin} and no strict prefix",
            }
            for pair, _, _ in PAIRS
        },
        "checks": checks.items,
        "checks_markdown": checks.markdown(),
        "failed_checks": [{"name": item["name"], "blame": blame.get(item["name"], [])} for item in checks.failed],
    }
    return output, checks


def report_three_way(output: dict, checks: common.Checks, json_path: str, md_path: str) -> int:
    engine_name = output["config"]["engine"]
    common.log("")
    for line in summary_block(output):
        common.log(f"[gate 3b] {line}")
    common.log(f"[gate 3b] wrote {json_path}")
    common.log(f"[gate 3b] wrote {md_path}")
    if output["pass"]:
        common.log(f"[gate 3b] three-way PASS (--profile-rule={output['profile_rule']})")
        return 0
    for item in output["failed_checks"]:
        detail = next(c["detail"] for c in checks.items if c["name"] == item["name"])
        common.log(f"[gate 3b] FAIL {item['name']} [{blame_text(item['blame'], engine_name)}]: {detail}")
    who = sorted({s for item in output["failed_checks"] for s in item["blame"]})
    if who and "engine" not in who:
        common.log(
            "[gate 3b] three-way FAIL caused by the REFERENCE engines only (vLLM vs HF): this does not implicate "
            f"{engine_name}, but the comparison is not usable until vLLM and HF agree"
        )
    else:
        common.log(f"[gate 3b] three-way FAIL (--profile-rule={output['profile_rule']})")
    return 1


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=common.DEFAULT_MODEL, help="weights for the engine: HF directory or hub id")
    parser.add_argument(
        "--hf-model", default=None, help="weights for the HF reference (default: --model; differ only to self-test the gate)"
    )
    parser.add_argument("--hf-dtype", choices=sorted(common.DTYPES), default="bfloat16")
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--tie-margin",
        type=float,
        default=1.0,
        help="largest HF logit margin at a divergence that still counts as a bf16 near-tie (default %(default)s)",
    )
    parser.add_argument("--min-identical", type=int, default=0, help="prompts that must match exactly (default 0)")
    parser.add_argument(
        "--max-divergence-ratio",
        type=float,
        default=None,
        help="optional extra rule: FAIL if the engine-vs-HF divergence rate exceeds this x the HF-vs-HF flip rate "
        "(floored at one flip over all HF positions); off by default",
    )
    parser.add_argument("--prompts-file", default=None, help="JSON list of token-id lists (instead of MATH prompts)")
    parser.add_argument("--data", default=None, help="train parquet (default: $DATA_DIR/math_l3to5_train.parquet)")
    parser.add_argument("--prompt-seed", type=int, default=0)
    parser.add_argument("--eos-token-ids", default=None, help="comma-separated override (default: from the checkpoint)")
    parser.add_argument("--out-dir", default=common.DEFAULT_OUT_DIR)
    parser.add_argument("--tag", default=None, help="suffix of the output files (default: the engine name)")
    three = parser.add_argument_group("three-way comparison with stock vLLM")
    three.add_argument("--vllm-json", default=None, help="token file written by integration/gates/gate_b_vllm_greedy.py")
    three.add_argument("--profile-rule", choices=PROFILE_RULES, default="significance", help="R2 rule (default %(default)s)")
    three.add_argument("--profile-alpha", type=float, default=0.05, help="R2 significance level (default %(default)s)")
    three.add_argument(
        "--allow-model-mismatch", action="store_true", help="compare even if the vLLM file names another model path"
    )
    three.add_argument(
        "--compare-only",
        action="store_true",
        help="recompute the three-way verdict from the token files (no engine run); needs --vllm-json",
    )
    three.add_argument("--engine-json", default=None, help="--compare-only: default <out-dir>/gate_b_greedy_<tag>.engine.json")
    three.add_argument("--hf-json", default=None, help="--compare-only: default <out-dir>/gate_b_greedy_<tag>.hf.json")
    three.add_argument(
        "--stored-margins",
        action="store_true",
        help="--compare-only: use the HF margins stored in the HF file instead of loading the HF model",
    )
    common.add_engine_args(parser)
    common.add_preflight_args(parser)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.compare_only:
            return compare_only(args)
        return full_run(args)
    except InputMismatch as exc:
        # overwrite the result files, so that a stale PASS of an earlier run cannot be mistaken for this one
        tag = args.tag or args.engine
        name = f"gate_b_greedy_{tag}" + ("_recomputed" if args.compare_only else "")
        json_path = common.write_json(
            os.path.join(args.out_dir, name + ".json"),
            {"gate": "3b_greedy", "pass": False, "refused": True, "reason": str(exc), "three_way": bool(args.vllm_json),
             "mode": "compare_only" if args.compare_only else "full_run", "argv": list(sys.argv if argv is None else argv)},
        )  # fmt: skip
        common.write_text(
            os.path.join(args.out_dir, name + ".md"),
            f"# Gate 3b - REFUSED\n\n**Verdict: none (inputs do not match; nothing was compared)**\n\n{exc}\n",
        )
        common.log(f"[gate 3b] REFUSED: {exc}")
        common.log(f"[gate 3b] nothing was compared; wrote the refusal to {json_path} (\"pass\": false); exit status 2")
        return 2


def _side_paths(args, tag: str) -> tuple[str, str]:
    base = os.path.join(args.out_dir, f"gate_b_greedy_{tag}")
    return (args.engine_json or base + ".engine.json"), (args.hf_json or base + ".hf.json")


def full_run(args) -> int:
    tag = args.tag or args.engine
    vllm_side = load_side(args.vllm_json, "vllm") if args.vllm_json else None

    preflight = common.run_preflight(args, engine_share=args.mem_fraction)
    model_dir = common.resolve_model_dir(args.model)
    hf_dir = common.resolve_model_dir(args.hf_model) if args.hf_model else model_dir
    eos = (
        [int(t) for t in args.eos_token_ids.split(",")] if args.eos_token_ids else common.eos_ids_from_checkpoint(model_dir)
    )
    if not eos:
        raise SystemExit("no EOS token ids in the checkpoint; pass --eos-token-ids")
    prompts, prompt_meta = common.resolve_prompts(
        prompts_file=args.prompts_file, model_dir=model_dir, n=args.num_prompts, data_path=args.data, seed=args.prompt_seed
    )
    common.log(
        f"[gate 3b] {len(prompts)} prompts ({prompt_meta['source']}), lengths {min(map(len, prompts))}-"
        f"{max(map(len, prompts))} tokens, sha256 {common.prompts_sha256(prompts)[:16]}..., eos={eos}"
    )
    if vllm_side is None:
        common.log(f"[gate 3b] NOTE: {NOT_THREE_WAY}")
    else:
        # refuse BEFORE spending GPU time on the engine
        ours = {"prompts": prompts, "prompts_sha256": common.prompts_sha256(prompts), "prompts_meta": prompt_meta,
                "max_new_tokens": args.max_new_tokens, "eos_token_ids": eos, "model": model_dir}  # fmt: skip
        check_sides_match(ours, vllm_side, label=f"vLLM file {args.vllm_json}", allow_model_mismatch=args.allow_model_mismatch)
        common.log(f"[gate 3b] vLLM file {args.vllm_json}: same prompts, settings and model ({vllm_side.get('engine')})")

    # ---- engine: all prompts in one call, temperature 0 ----
    common.log(f"[gate 3b] building the {args.engine} engine from {model_dir} on {args.device}")
    engine = common.build_engine(args.engine, model_dir, **common.engine_opts_from_args(args))
    start = time.perf_counter()
    result = engine.generate(prompts, max_new_tokens=args.max_new_tokens, temperature=0.0, eos_token_ids=eos)
    engine_seconds = time.perf_counter() - start
    engine_tokens = [list(t) for t in result.token_ids]
    engine_stats = dict(result.stats)
    problems = common.check_responses(result, args.max_new_tokens, eos, greedy=True)
    engine_side = common.greedy_tokens_record(
        side="engine", engine=args.engine, model=model_dir, prompts=prompts, prompts_meta=prompt_meta,
        token_ids=engine_tokens, finish_reasons=list(result.finish_reasons), max_new_tokens=args.max_new_tokens,
        eos_token_ids=eos, logprobs=[list(map(float, lp)) for lp in result.logprobs],
        engine_options=common.engine_opts_from_args(args), engine_stats=engine_stats, seconds=engine_seconds,
    )  # fmt: skip
    engine_path, hf_path = _side_paths(args, tag)
    common.write_json(engine_path, engine_side)
    common.log(f"[gate 3b] wrote {engine_path}")
    engine.sleep()  # release the rollout-only GPU memory before the HF model runs
    engine.close()
    del engine, result
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- HF: one prompt at a time, no padding ----
    common.log(f"[gate 3b] HF greedy ({args.hf_dtype}) from {hf_dir}")
    model = common.load_hf_model(hf_dir, args.device, args.hf_dtype)
    per_prompt = []
    hf_all: list[list[int]] = []
    cache: dict = {}
    vllm_tokens = vllm_side["token_ids"] if vllm_side else None
    start = time.perf_counter()
    for i, prompt in enumerate(prompts):
        hf_tokens = common.hf_greedy(model, prompt, args.max_new_tokens, eos)
        hf_all.append(hf_tokens)
        logits = common.hf_teacher_forced_logits(model, prompt, hf_tokens)
        record = analyse_prompt(model, prompt, engine_tokens[i], hf_tokens, args.tie_margin, logits=logits)
        sides_i = {"engine": engine_tokens, "hf": hf_all, "vllm": vllm_tokens}  # hf_all[i] is this prompt
        fill_margins(model, i, prompt, sides_i, cache, hf_logits=logits)
        del logits
        record["engine_tokens"] = engine_tokens[i]
        record["hf_tokens"] = hf_tokens
        per_prompt.append(record)
        where = "identical" if record["first_diff"] < 0 else f"first diff at {record['first_diff']}"
        extra = "" if record["margin"] is None else f", HF margin {record['margin']:.4f} logits"
        vllm_note = ""
        if vllm_tokens is not None:
            d_v = common.first_difference(vllm_tokens[i], hf_tokens)
            vllm_note = f"; vLLM vs HF: {'identical' if d_v < 0 else f'first diff at {d_v}'}"
        common.log(
            f"[gate 3b] prompt {i:2d}: engine {record['engine_len']:4d} tok, HF {record['hf_len']:4d} tok, "
            f"{where}{extra} -> {record['status']}{vllm_note}"
        )
    hf_seconds = time.perf_counter() - start
    del model
    hf_side = common.greedy_tokens_record(
        side="hf", engine="hf", model=hf_dir, prompts=prompts, prompts_meta=prompt_meta, token_ids=hf_all,
        finish_reasons=["stop" if t and t[-1] in set(eos) else "length" for t in hf_all],
        max_new_tokens=args.max_new_tokens, eos_token_ids=eos, hf_dtype=args.hf_dtype, device=args.device,
        hf_self_flips=[r["hf_self_flips"] for r in per_prompt],
        hf_self_flip_max_margin=[r["hf_self_flip_max_margin"] for r in per_prompt],
        margin_cache=cache, seconds=hf_seconds,
    )  # fmt: skip
    common.write_json(hf_path, hf_side)
    common.log(f"[gate 3b] wrote {hf_path}")

    # ---- two-way summary (always) ----
    def count(status: str) -> int:
        return sum(1 for r in per_prompt if r["status"] == status)

    margins = [r["margin"] for r in per_prompt if r["margin"] is not None]
    diverged = sorted(r["first_diff"] for r in per_prompt if r["first_diff"] >= 0)
    summary = {
        "num_prompts": len(per_prompt),
        "identical": count("identical"),
        "near_tie": count("near_tie"),
        "mismatch": count("MISMATCH"),
        "length_mismatch": count("LENGTH_MISMATCH"),
        "max_margin": max(margins) if margins else None,
        "margins_sorted": sorted(margins, reverse=True),
        "first_diffs": [r["first_diff"] for r in per_prompt],
        "median_first_diff": diverged[len(diverged) // 2] if diverged else None,
        "hf_tokens": sum(r["hf_len"] for r in per_prompt),
        "hf_self_flips": sum(r["hf_self_flips"] for r in per_prompt),
        "hf_self_flip_max_margin": max(r["hf_self_flip_max_margin"] for r in per_prompt),
        # positions where both saw the same prefix: everything up to and including the first difference
        "positions_compared": sum(
            (r["first_diff"] + 1) if 0 <= r["first_diff"] < min(r["engine_len"], r["hf_len"]) else min(r["engine_len"], r["hf_len"])
            for r in per_prompt
        ),
        "engine_seconds": engine_seconds,
        "hf_seconds": hf_seconds,
    }
    summary["divergence_rate"] = (summary["near_tie"] + summary["mismatch"]) / max(summary["positions_compared"], 1)
    summary["hf_self_flip_rate"] = summary["hf_self_flips"] / max(summary["hf_tokens"], 1)
    common.log(
        f"[gate 3b] {args.engine} vs HF: {summary['identical']}/{summary['num_prompts']} identical, {summary['near_tie']} "
        f"near-tie, {summary['mismatch']} MISMATCH, {summary['length_mismatch']} LENGTH_MISMATCH; "
        f"max HF margin at a divergence: {common.fmt_num(summary['max_margin'])} logits"
    )
    common.log(
        f"[gate 3b] yardstick: HF vs HF (cached vs uncached forward) flips its argmax at {summary['hf_self_flips']} "
        f"of {summary['hf_tokens']} positions ({100 * summary['hf_self_flip_rate']:.2f}%), largest margin "
        f"{common.fmt_num(summary['hf_self_flip_max_margin'])} logits; engine vs HF: "
        f"{summary['near_tie'] + summary['mismatch']} divergence(s) in {summary['positions_compared']} positions with a "
        f"shared prefix ({100 * summary['divergence_rate']:.2f}%). A far higher engine rate, or engine margins far above "
        "the HF-vs-HF ones, deserve a look even when the gate passes (not part of the verdict)."
    )
    base_config = {
        "engine": args.engine,
        "model": model_dir,
        "hf_model": hf_dir,
        "hf_dtype": args.hf_dtype,
        "device": args.device,
        "num_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "tie_margin": args.tie_margin,
        "min_identical": args.min_identical,
        "max_divergence_ratio": args.max_divergence_ratio,
        "eos_token_ids": eos,
        "engine_options": common.engine_opts_from_args(args),
        "engine_json": os.path.abspath(engine_path),
        "hf_json": os.path.abspath(hf_path),
    }
    json_path = os.path.join(args.out_dir, f"gate_b_greedy_{tag}.json")
    md_path = os.path.join(args.out_dir, f"gate_b_greedy_{tag}.md")

    if vllm_side is not None:
        output, checks = three_way_output(
            args=args, engine_name=args.engine, engine_side=engine_side, hf_side=hf_side, vllm_side=vllm_side,
            vllm_path=args.vllm_json, cache=cache,
        )  # fmt: skip
        output["config"] = {**base_config, **output["config"]}
        output.update(
            summary=summary, per_prompt=per_prompt, engine_stats=engine_stats, preflight=preflight,
            environment=common.environment_info(args.device), mode="full_run",
        )  # fmt: skip
        common.write_json(json_path, output)
        common.write_text(md_path, build_markdown_three_way(output))
        return report_three_way(output, checks, json_path, md_path)

    checks = two_way_checks(summary, problems, args)
    passed = checks.ok
    output = {
        "gate": "3b_greedy",
        "pass": passed,
        "three_way": False,
        "three_way_note": NOT_THREE_WAY,
        "config": base_config,
        "prompts": prompt_meta,
        "summary": summary,
        "checks": checks.items,
        "checks_markdown": checks.markdown(),
        "per_prompt": per_prompt,
        "engine_stats": engine_stats,
        "preflight": preflight,
        "environment": common.environment_info(args.device),
    }
    common.write_json(json_path, output)
    common.write_text(md_path, build_markdown(output))
    common.log("")
    common.log(
        f"[gate 3b] rule: contract ok AND no strict-prefix responses AND every divergence margin <= "
        f"--tie-margin={args.tie_margin} AND identical >= --min-identical={args.min_identical}"
        + (
            f" AND divergence rate <= --max-divergence-ratio={args.max_divergence_ratio} x the HF-vs-HF flip rate"
            if args.max_divergence_ratio is not None
            else " (--max-divergence-ratio not set)"
        )
    )
    common.log(f"[gate 3b] wrote {json_path}")
    common.log(f"[gate 3b] wrote {md_path}")
    common.log(f"[gate 3b] NOTE: {NOT_THREE_WAY}")
    if passed:
        common.log("[gate 3b] PASS (two-way only: engine vs HF)")
        return 0
    for item in checks.failed:
        common.log(f"[gate 3b] FAIL {item['name']}: {item['detail']}")
    common.log("[gate 3b] FAIL (two-way only: engine vs HF)")
    return 1


def compare_only(args) -> int:
    if not args.vllm_json:
        raise SystemExit("--compare-only needs --vllm-json")
    tag = args.tag or args.engine
    engine_path, hf_path = _side_paths(args, tag)
    engine_side = load_side(engine_path, "engine")
    hf_side = load_side(hf_path, "hf")
    vllm_side = load_side(args.vllm_json, "vllm")
    check_sides_match(engine_side, hf_side, label=f"HF file {hf_path}", allow_model_mismatch=True)
    check_sides_match(engine_side, vllm_side, label=f"vLLM file {args.vllm_json}", allow_model_mismatch=args.allow_model_mismatch)
    engine_name = engine_side.get("engine") or args.engine
    common.log(
        f"[gate 3b] compare-only from files: {engine_path}, {hf_path}, {args.vllm_json} "
        f"({len(engine_side['prompts'])} prompts, sha256 {engine_side['prompts_sha256'][:16]}...)"
    )
    sides = {"engine": engine_side["token_ids"], "hf": hf_side["token_ids"], "vllm": vllm_side["token_ids"]}
    preflight = {"skipped": True}
    if args.stored_margins:
        cache = dict(hf_side.get("margin_cache") or {})
        common.log(f"[gate 3b] HF margins: {len(cache)} stored in {hf_path} (--stored-margins; no HF model loaded)")
    else:
        hf_dir = common.resolve_model_dir(args.hf_model or hf_side["model"])
        hf_dtype = hf_side.get("hf_dtype") or args.hf_dtype
        preflight = common.run_preflight(args, engine_share=0.0)
        common.log(f"[gate 3b] HF margins: recomputed with the HF model {hf_dir} ({hf_dtype}) on {args.device}")
        model = common.load_hf_model(hf_dir, args.device, hf_dtype)
        cache = {}
        for i, prompt in enumerate(engine_side["prompts"]):
            fill_margins(model, i, prompt, sides, cache)
        del model
    output, checks = three_way_output(
        args=args, engine_name=engine_name, engine_side=engine_side, hf_side=hf_side, vllm_side=vllm_side,
        vllm_path=args.vllm_json, cache=cache,
    )  # fmt: skip
    output.update(
        mode="compare_only",
        margins_from="stored" if args.stored_margins else "hf_model",
        engine_json=os.path.abspath(engine_path),
        hf_json=os.path.abspath(hf_path),
        engine_stats=engine_side.get("engine_stats") or {},
        preflight=preflight,
        environment=common.environment_info(args.device),
    )
    json_path = os.path.join(args.out_dir, f"gate_b_greedy_{tag}_recomputed.json")
    md_path = os.path.join(args.out_dir, f"gate_b_greedy_{tag}_recomputed.md")
    common.write_json(json_path, output)
    common.write_text(md_path, build_markdown_three_way(output))
    return report_three_way(output, checks, json_path, md_path)


if __name__ == "__main__":
    sys.exit(main())
