# Gate 3a: review of the speculative sampler

Written 2026-11-27 from the working tree of branch `sparsespec-verl-backend`.

**Method.** Code reading. This machine has no GPU and cannot import triton or flashinfer, so none of the
kernels discussed here was executed. Every statement about what the engine does is by code reading.
The only things run are CPU transcriptions of the kernels' arithmetic (Appendix A, and the row sums
quoted under F2) and a numpy evaluation of one expression (R3). They test the algorithm as written,
not the compiled kernels.

**Scope.** The path the verl rollout backend uses: `RLSpecScheduler` on top of `SpecScheduler`,
synchronous scheduling (`async_cpu=False`), PillarAttn cache, `spec_stride` 8, temperature 1.0,
top_p 1.0, top_k -1. Asynchronous scheduling pairs a verify pass with tensors from the previous
iteration through separate bookkeeping. `RLSpecScheduler` refuses it (`serve/rl/scheduler.py:79-84`)
and it is not reviewed here.

**References.** `file:line` refers to files under the repo root. `serve/rl/engine.py` was being edited
while this was written, so it is cited by function name.

## Answers in brief

| # | Question | Answer |
|---|---|---|
| 1 | Standard speculative sampling? | Yes. The sampler is a port of vLLM 0.9.1's `vllm/v1/sample/rejection_sampler.py`. Acceptance test, residual resampling, bonus token and discard all match Leviathan et al., Algorithm 1. |
| 2 | What are p and q? | Both are the float32 softmax of `float32(bf16 logits) * (1/T)`, computed by one kernel in one call: p on the rows of the full-attention verify pass, q on the row of the sparse-attention draft pass. The stored q row is bit for bit the row the draft token was sampled from. |
| 3 | top_p / top_k | Not implemented on the path that runs. The RL wrapper raises on anything other than top_p = 1.0 and top_k in {-1, 0}. The project config asks for neither. |
| 4 | Greedy (T = 0) | p and q become one-hot. Acceptance reduces to "draft token equals the target argmax", except when the uniform is exactly 0.0 (about 2^-53 per draw). |
| 5 | Row alignment | Draft position j uses target row s + j and draft row j. The recovered token at j uses the same two rows. The bonus token is the regular sampler's draw from row s + n. No off-by-one found. |
| 6 | Deviations | None that changes the per-token law beyond floating point in this configuration. Upstream EOS handling is wrong but overridden for RL. A pointer-offset overflow is latent beyond 3,533 batch rows (the project reaches at most 823). |
| 7 | Conclusion | By code reading, emitted tokens follow the full-attention target up to floating point. Gates 3b and 3c can falsify the "p is the right row of the right model" half. No gate in `integration/gates/` tests the sampling law itself at the time of writing; section 7 proposes one. |

## 0. Names, and how one cycle runs

| Name | Meaning |
|---|---|
| k | `spec_stride`, the draft length. 8 in the project config. |
| n | Number of draft tokens in the current cycle. n = k, except in a request's first cycle, where n = k - `initial_idx` and 1 <= n <= k (section 6.1). |
| t0 | The last confirmed token before the cycle starts. |
| d_0 .. d_{n-1} | The draft tokens, 0-indexed like `pos` in the kernels. |
| q_j | The probability row d_j was sampled from (draft pass j). |
| p_j | The probability row of the verify pass for the same position. p_n is the row after d_{n-1}. |
| s | Index of the request's first row in the verify pass's flat batch (`token_indptr[idx]`). |
| V | Vocabulary size (151,936 for Qwen3-0.6B). |
| E | Exp(1) noise. |

Two names in the code are easy to misread. In `sampler.py` and in `sample_recovered_tokens*` the tensor
called `q` is the Exp(1) noise, not the draft distribution. In `SpecScheduler.verify_outputs`,
`accepted_tokens` is the number of tokens the pass emitted (accepted drafts plus one).

Each scheduler iteration is one batched forward over all active requests
(`BaseScheduler.run`, `serve/scheduler/scheduler.py:542-671`): build inputs, run the model, run
`Sampler.forward` on every output row, then `update_requests`. Requests in different states share the
batch. The model returns one logits row per fed token, in order (`serve/model/model.py:323-331`).

| Iteration | Status | Tokens fed for the request | Rows it gets | What the scheduler keeps |
|---|---|---|---|---|
| last prefill chunk | NORMAL | prompt chunk | one per token | The token sampled from the last row. It is the first response token (`spec_scheduler.py:460-466`). |
| draft pass j, j = 0 .. n-1 | DRAFT | 1 token: t0 if j = 0, else d_{j-1} | 1 | The probability row, copied to `request.draft_probs[j]` (this is q_j), and the token sampled from that row, appended as d_j. |
| verify pass | VERIFY | n + 1 tokens: t0, d_0, .., d_{n-1} | n + 1: rows s .. s + n | Rows s .. s + n - 1 (p_0 .. p_{n-1}) and the token sampled from row s + n (the bonus candidate). |

After the verify pass the request goes back to DRAFT with `len_draft_probs = 0` and
`cur_spec_idx = initial_idx = 0` (`spec_scheduler.py:569-582`), so every later cycle has n = k.

## 1. Is this standard speculative sampling?

Yes. The textbook rule (Leviathan et al. 2022, arXiv 2211.17192, Algorithm 1; Chen et al. 2023,
arXiv 2302.01318, Algorithm 2 is the same rule) is: draw drafts x_j from q_j; accept x_j with
probability min(1, p_j(x_j) / q_j(x_j)); at the first rejection emit one token from
normalize(max(0, p_j - q_j)) and stop; if nothing is rejected, emit one more token from p_n.

`serve/sampling/rejection_sampler.py` is a port of vLLM 0.9.1's v1 rejection sampler. Compared with the
vLLM source, the acceptance line is identical and the residual resampling is the same computation done
in blocks of 16,384 vocabulary entries. The port drops per-request generators and vLLM's separate greedy
kernel (greedy goes through the same kernel with one-hot probabilities, section 4).

### (a) Acceptance

One uniform per draft token, float64, in [0, 1), drawn on the device:

```python
# serve/sampling/rejection_sampler.py:249-253
uniform_probs = torch.rand(
    (num_tokens,),
    dtype=torch.float64,
    device=device,
)
```

Lines 254-262 then redraw each request's slice with `.uniform_()`. That is a leftover of vLLM's
per-request seeding and replaces i.i.d. uniforms with i.i.d. uniforms.

The test, for request `req_idx` and draft position `pos`:

```python
# serve/sampling/rejection_sampler.py:331-350 (abridged)
draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
draft_prob = tl.load(draft_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id)
target_prob = tl.load(target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id)
uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
# NOTE(woosuk): While the draft probability should never be 0,
# we check it to avoid NaNs. If it happens to be 0, we reject.
if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
    # Accept.
    token_id = draft_token_id
else:
    # Reject. Use recovered token.
    rejected = True
    token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)
```

- **What is compared.** The float32 quotient p_j(d_j) / q_j(d_j) of two float32 probabilities against
  the float64 uniform. Under Triton's promotion rule the quotient is widened to float64 for the
  comparison (not executable here, Appendix B).
- **min(1, .)** is never computed and does not need to be: u < 1, so any quotient >= 1 is accepted.
- **q(x) == 0.** The token is rejected and the recovered token is used. The branch cannot be reached
  in practice. A draft token is the argmax of `probs / noise` (section 2), a token with probability 0
  has score 0, and the row always contains positive scores. In greedy mode q(d_j) = 1. The check only
  prevents a 0/0.
- **`>=` versus `>`.** Leviathan's algorithm rejects when r > p/q, so it accepts when r <= p/q. That is
  this code. Chen et al. accept when r < min(1, p/q). The two differ only when u equals the quotient
  exactly, which has probability at most 2^-53 per draw. The one visible consequence of `>=`: if u is
  exactly 0.0, a draft token with p(x) = 0 is accepted. That needs both a zero target probability
  (float32 underflow, or greedy mode) and u == 0.0, so it is a 2^-53 event. The source comment at
  lines 244-248 says the uniforms are float64 for this reason. In the CPU transcription, replacing
  `>=` by `>` gives the same law (Appendix A).

### (b) The recovered token

```python
# serve/sampling/rejection_sampler.py:279-287
# NOTE(woosuk): Create only one distribution for each request.
batch_size = len(num_draft_tokens)
vocab_size = target_probs.shape[-1]
q = torch.empty(
    (batch_size, vocab_size),
    dtype=torch.float32,
    device=device,
)
q.exponential_()
```

```python
# serve/sampling/rejection_sampler.py:402-428 (abridged; one program per (request, position))
draft_prob = tl.load(draft_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset, ...)
target_prob = tl.load(target_probs_ptr + (start_idx + pos) * vocab_size + vocab_offset, ...)
prob = tl.maximum(target_prob - draft_prob, 0)
q = tl.load(q_ptr + req_idx * vocab_size + vocab_offset, ...)
scores = prob / q
local_val = tl.max(scores, axis=-1)
local_idx = tl.argmax(scores, axis=-1) + off
better = local_val > max_val
...
tl.store(output_token_ids_ptr + start_idx + pos, max_idx)
```

**The trick.** For weights w_i >= 0 and i.i.d. E_i ~ Exp(1), argmax_i (w_i / E_i) equals
argmin_i (E_i / w_i). E_i / w_i is exponential with rate w_i, and the minimum of independent
exponentials falls on index i with probability w_i / sum(w). So the argmax is a sample from
w / sum(w). Scaling w by a constant does not change the argmax, which is why the residual
max(p - q, 0) is never normalized. Entries with zero residual have score 0 and cannot win while any
residual is positive. The regular sampler uses the same trick (section 2).

**One noise vector per what.** One Exp(1) vector of length V per verifying request per verify pass,
shared by all of that request's draft positions. The kernel computes a recovered token for every
position (grid `(batch_size, max_spec_len)`, line 290) from the same noise row
(`q_ptr + req_idx * vocab_size`).

**Does the sharing bias anything?** No.

- Within a request, recovered tokens at different positions are correlated, but at most one of them is
  ever read: the acceptance kernel sets `rejected = True` at the first rejection and reads nothing after
  it (lines 328-353).
- Which position is rejected is a function of the draft tokens, p, q and the uniforms. The noise is
  drawn afterwards, in its own call, and is independent of all of them. Conditioning on "the first
  rejection is at j" therefore leaves the noise Exp(1), and the token read is a sample from
  normalize(max(p_j - q_j, 0)).
- Across requests the noise rows are independent. Across verify passes the noise is redrawn. The
  noise is also separate from the noise the regular sampler used to draw the draft tokens. That
  separation matters: the transcription variant that reuses the draft pass's noise for the recovered
  token fails (Appendix A).

### (c) The bonus token

It is not sampled in the rejection sampler. The verify pass is an ordinary forward, the regular sampler
draws a token from every one of its rows, and the scheduler picks the draw from the request's last row:

```python
# serve/scheduler/spec_scheduler.py:365-374 (abridged)
start_idx, end_idx = token_indptr[idx], token_indptr[idx + 1]
num_draft_tokens = end_idx - start_idx - 1  # excluding bonus token
...
bonus_token_ids_host.append(gen_token_ids_cpu[end_idx - 1])
```

```python
# serve/sampling/rejection_sampler.py:355-361
if not rejected:
    # If all tokens are accepted, append the bonus token.
    bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
    tl.store(
        output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
        bonus_token_id,
    )
```

Row `end_idx - 1` is row s + n: the full-attention, post-temperature distribution given t0 and all n
drafts, which is p_n. The draw is made before the acceptance outcome is known and with its own noise
row, so using it only when every draft was accepted does not bias it.

### (d) Tokens after the first rejection

Discarded, in three places.

- The output buffer is pre-filled with `PLACEHOLDER_TOKEN_ID = -1` (lines 172-177) and the kernel
  writes nothing once `rejected` is set (`if not rejected:` at line 330).
- `RejectionSampler.parse_output` drops the placeholders (lines 131-136).
- `SpecScheduler.verify_outputs` removes all n drafts from `token_ids`, appends the kernel's output
  and retracts the KV entries of the rejected positions:

```python
# serve/scheduler/spec_scheduler.py:278-290 (abridged)
spec_tokens_count = request.len_draft_probs
accepted_tokens = len(output_token_ids)
num_retracted_kv_cache_tokens = spec_tokens_count + 1 - accepted_tokens
...
del request.token_ids[-spec_tokens_count:]
request.token_ids.extend(output_token_ids)
# retract rejected kv-cache tokens
request.kv_cache_ptr.retract_tokens(num_retracted_kv_cache_tokens)
```

With m accepted drafts and a recovered token, the pass emits m + 1 tokens and retracts n - m KV
entries. That leaves the entries of t0, d_0 .. d_{m-1} and the KV length one below the token count,
which is the state the next draft pass expects (it feeds the one token not yet in the cache).

## 2. What exactly are p and q?

Both come out of the same function, called once per iteration on all rows of the batch:

```python
# serve/scheduler/scheduler.py:612-616
gen_token_ids_tensor, gen_token_probs_tensor = self.sampler(
    logits=gen_padded_token_logits_tensor,
    sampling_metadata=self.sampling_metadata,
    clip_length=input_metadata_tensors["token_ids"].shape[0],
)
```

`Sampler.forward` (`serve/sampling/sampler.py:142-173`) draws one Exp(1) noise value per (row,
vocabulary entry) and calls `triton_fused_sampling`, which per row computes the probabilities and the
sampled token together:

```python
# serve/sampling/fast_fused_op.py:138-141, 158, 161-162, 168 (second pass of the online softmax)
# cast xblk to float32
xblk = tl.cast(xblk, tl.float32)
xblk = xblk * inv_temperature
yblk = tl.exp(xblk - m) / d  # probs
...
scores = yblk / qblk
...
local_val = tl.max(scores, axis=-1)
local_idx = tl.argmax(scores, axis=-1) + start
...
tl.store(token_ids + row_id, max_idx)
```

| | q (draft) | p (target) |
|---|---|---|
| Forward pass | A DRAFT pass. One token fed, one row. | The VERIFY pass. n + 1 tokens fed, n + 1 rows. |
| Attention | Sparse. The kernel reads the request's `flatten_indices`: the selected "pillar" tokens followed by the recent tokens. | Full. The kernel reads all of the request's KV pages. |
| Probabilities | `softmax(float32(logits) * (1/T))`, float32 | The same expression, the same kernel, the same call. |

**Which attention.** The request's status is passed to the attention kernel and selects the index
list:

```cpp
// serve/attention/backend.py:40-47
// NORMAL(0), DRAFT(1), VERIFY(2)
if (request_type == 1){
    indices = params.flatten_indices + kv_head_idx * {{MAX_TOTAL_DRAFT_KV_LEN}};
}else{
    indices = params.kv_indices;
}
```

The KV length follows the status too (`serve/request/request.py:97-106`): all pages for NORMAL and
VERIFY, `len_loading_indices` (budget plus recent tokens) for DRAFT. In a request's first cycle the
draft is dense, not sparse: `PillarCachePtr.from_kv_cache` sets
`len_loading_indices = source_cache.len_kv_cache` and no selected tokens
(`serve/request/kv_cache_ptr/pillar.py:106-109`). So in first cycles q and p differ only by the
numerics of two different passes over the same context (the logits are bf16, so this is not
negligible), and acceptance should be clearly higher than in later cycles. That is a prediction a GPU
run can check.

**Post-temperature, identically.** Yes. Temperature enters in one place, `xblk * inv_temperature`,
with `inv_temperature = 1 / temperature` (`fast_fused_op.py:53`) taken from the scheduler-wide
`SamplingMetadata` (`scheduler.py:52-56`). Draft rows and verify rows of different requests are in the
same batch and go through the same kernel launch, so p and q cannot be given different temperatures.
At T = 1.0 the factor is exactly 1.0 and the multiplication is exact.

**Precision.** The logits are in the model dtype (bf16 in the project). The kernel casts each logit to
float32 and does the whole softmax in float32. The probabilities are written to a float32 tensor
`[rows, V]` (`fast_fused_op.py:35-39`). Rows are stored and moved as float32 without arithmetic:

| Buffer | Shape, dtype | Filled by |
|---|---|---|
| `gen_token_probs_tensor` | `[rows of the pass, V]` float32 | the sampler kernel |
| `request.draft_probs` | `[MAX_SPEC_LEN, V]` float32, per request (`request.py:53-57`); 10 x 151,936 x 4 bytes = 6.1 MB | `scatter_token_probs`, one row per draft pass |
| `draft_probs_tensor`, `target_probs_tensor` | `[total drafts being verified, V]` float32 (`spec_scheduler.py:386-395`) | `gather_token_probs` and `copy_token_probs` |

All three helpers use `batch_copy_tensor_kernel` (`serve/request/kv_cache_ptr/base.py:48-74`), which
loads and stores float32 values unchanged.

**Is q the distribution the draft token was sampled from?** Yes, the same tensor and the same row. In
the kernel, `yblk` is both stored as the probability and used for the score that picks the token.
Section 5 shows the scheduler copies that row to `draft_probs[j]` and appends that row's token as d_j.
One nuance: the race samples from the stored row divided by its sum, and the stored row sums to 1 only
up to float32 rounding (section 6.2, F2).

**Why the verify rows are the full-attention conditional even though draft passes wrote KV entries.**
A draft pass appends a KV entry computed from sparse-attention hidden states. The verify pass feeds
t0 and all n drafts again, and the append kernel writes the last n + 1 positions of the sequence
(`flashinfer.get_batch_indices_positions` returns the rightmost positions,
`3rdparty/flashinfer/flashinfer/page.py:207-239`; KV is appended before attention,
`serve/model/model.py:651-655`). Every draft-time entry is therefore overwritten with a full-attention
entry before any verify query reads it, attention inside the pass is causal (`base.py:278`), and
entries of rejected positions are retracted. By induction every KV entry a verify query attends to was
produced by a full-attention pass over confirmed tokens.

"Target" here means this engine's own full-attention forward (bf16 weights and activations, flashinfer
kernels) followed by a float32 softmax. Whether that forward matches HuggingFace's to numerical noise
is the business of gates 3b and 3c, not of the sampler.

## 3. top_p and top_k

They are not implemented on the path that runs.

- The scheduler calls `self.sampler(...)`, which is `Sampler.forward`. Its only sampling inputs are the
  two fields of `SamplingMetadata`, `temperature` and `is_greedy_sampling`
  (`serve/sampling/sampler.py:37-44`).
- `Sampler.sample` and `TopKTopPSampler` contain top-k / top-p code, but nothing calls
  `Sampler.sample`, and the path is closed twice over: `forward_native` starts with
  `assert k is None and p is None  # Consider probs w/ k and p` (line 69), and the top-k-only branch
  calls `apply_top_k_only` (line 86), which is neither defined nor imported anywhere in `serve/`.
- Upstream `serve/run.py` has no top_p or top_k argument at all.

**If they were requested.** Through the RL engine, `validate_sampling_params`
(`serve/rl/base.py:89-108`, called at the top of `SparseSpecEngine.generate` and
`ReferenceEngine.generate`) raises `NotImplementedError` for top_p != 1.0 or top_k outside {-1, 0},
before any GPU work. They cannot be silently ignored. If someone wired them into the regular sampler
only, the result would be wrong, not merely unsupported. The first token and bonus tokens would come
from the truncated target, while accepted and recovered tokens would be judged against the untruncated
p and q, and the output would follow neither distribution. The upstream docstring describes that
arrangement as a feature (`rejection_sampler.py:49-53`); it is not one. A correct implementation has
to truncate and renormalize the p rows and the q rows identically before both the draft sampling and
the acceptance test.

**Consequence for the project config** (T = 1.0, top_p = 1.0, top_k = -1): it asks for plain
temperature sampling, the one mode the sampler implements, so nothing is lost or approximated.
Validation in verl with `do_sample=False` maps to temperature 0 (`serve/rl/verl_rollout.py:194-195`),
which is the greedy mode of section 4. The baseline config disables validation anyway
(`baseline/config/grpo_base.sh:72-73`).

## 4. Greedy mode (temperature 0)

`is_greedy_sampling` is `temperature == 0.0` (`scheduler.py:55`). The sampler then returns the argmax
and a one-hot row, for draft rows and verify rows alike:

```python
# serve/sampling/sampler.py:153-157
if sampling_metadata.is_greedy_sampling:
    greedy_sampled = triton_strided_argmax(logits, self.tp_ranks).long()
    probs = torch.zeros_like(logits, dtype=torch.float32)
    probs.scatter_(dim=-1, index=greedy_sampled.unsqueeze(-1), value=1.0)
    return greedy_sampled[:clip_length], probs[:clip_length]
```

So q_j is one-hot at the draft argmax d_j, and p_j is one-hot at the target argmax a_j. The same
rejection kernel then gives:

| Case | q(d_j) | p(d_j) | Test `p/q >= u` | Result |
|---|---|---|---|---|
| d_j == a_j | 1 | 1 | 1 >= u, always true | accept |
| d_j != a_j | 1 | 0 | 0 >= u, true only if u == 0.0 | reject; the residual max(p - q, 0) is one-hot at a_j, so the recovered token is a_j |

The bonus token is the argmax of row s + n. Acceptance therefore reduces to exact match with the
target argmax, and the output is the full-attention greedy sequence, except on the event u == 0.0
(about 2^-53 per draft token), where a mismatching draft is accepted. Ties in the bf16 logits are
broken toward the lowest token id by the argmax kernel (`fast_fused_op.py:229-235`: strict `>` across
blocks, and Triton's default leftmost tie-break within a block, Appendix B), the same way in draft and
verify passes.

## 5. Row and position alignment

Claim: for a verify pass that feeds n + 1 tokens and gets rows s .. s + n,

| Output position | Token | Target row used | Draft row used |
|---|---|---|---|
| j (0 <= j < n), accepted | d_j | s + j | `draft_probs[j]` |
| j, first rejection | recovered token | s + j | `draft_probs[j]` |
| n, nothing rejected | bonus token | s + n (sampled from it by the regular sampler) | none |

and `draft_probs[j]` is the row d_j was sampled from.

**Draft side.** A DRAFT request contributes exactly one row, and the scheduler records where that row
is and where it must go:

```python
# serve/scheduler/spec_scheduler.py:172-182
if request.status == Request.ReqExecType.DRAFT:
    assert request.draft_probs is not None
    assert request.len_draft_probs < request.draft_probs.shape[0]
    assert num_tokens == 1

    probs_src_start_pos.append(token_indptr[-1])
    probs_token_num_list.append(num_tokens)
    probs_tgt_tensor_ptrs.append(
        request.draft_probs[request.len_draft_probs].data_ptr()
    )
    request.len_draft_probs += num_tokens  # clear after verify
```

1. `token_indptr[-1]` at this point is the number of rows of the requests before this one, which is
   this request's row r (its own entry is appended later, at line 197).
2. The destination is row `len_draft_probs` of the request's buffer. `len_draft_probs` is 0 at the
   start of every cycle (asserted at line 543 for the first cycle, reset at line 581 after each
   verify) and grows by one per draft pass, so draft pass j writes row j.
3. After the forward, `scatter_token_probs` copies V float32 values from row r of
   `gen_token_probs_gpu` to that destination (`spec_scheduler.py:333-338`, `base.py:409-456`).
4. The token appended for this request is `gen_token_ids_cpu[token_indptr[idx + 1] - 1]`
   (line 464-466). With one row, `token_indptr[idx + 1] - 1 == token_indptr[idx] == r`.
5. Row r's token and row r's probabilities come from the same kernel program (section 2).
6. The order of `active_requests` is the same in `prepare_forward_inputs` and `update_requests`:
   admission and offloading, which reorder the list, run before `prepare_forward_inputs`
   (`scheduler.py:545`, `557`), and nothing between it and `update_requests` touches the list.

Hence `draft_probs[j]` = q_j bit for bit, and d_j was sampled from q_j.

**Verify side.**

```python
# serve/scheduler/spec_scheduler.py:158-161
if request.status == Request.ReqExecType.VERIFY:
    # include bonus token
    num_tokens = self.spec_stride + 1 - request.initial_idx
    request.kv_cache_ptr.allocate_tokens(1)
```

1. The request feeds `token_ids[len_kv_cache - num_tokens : len_kv_cache]` (lines 192-196). Before the
   pass the last draft is the only token not in the cache, so after `allocate_tokens(1)` the KV length
   equals the token count and the slice is the last n + 1 tokens: t0, d_0, .., d_{n-1}.
   `num_tokens - 1 = k - initial_idx` equals the number of draft passes of the cycle (asserted at
   lines 562-565 and again at line 369).
2. The model returns one row per fed token and attention is causal, so row s + i is the next-token
   distribution after fed token i, given everything before it. Row s is the distribution of the token
   after t0, which is the position of d_0. Row s + j is the position of d_j. Row s + n follows
   d_{n-1} and is the bonus position.
3. `update_requests` takes the drafts from `request.token_ids[-num_draft_tokens:]`, the draft rows
   from `request.draft_probs.data_ptr()` (row 0 onward) and the target rows from `start_idx` onward,
   n rows each (lines 373-376, 401-416). All three land at the same flat offsets c .. c + n - 1, where
   c is the number of drafts of the verifying requests before this one.
4. Both kernels index all three arrays with `start_idx + pos` (`rejection_sampler.py:331-341` and
   `402-411`), where `start_idx` is c. Position j therefore pairs d_j with `target_probs[c + j]`,
   which is row s + j, and with `draft_probs[c + j]`, which is `request.draft_probs[j]`. The recovered
   token for position j is computed from the same two rows.
5. The kernel writes position j to output column j and the bonus to column n. `parse_output` keeps the
   order, and `verify_outputs` appends the list where the drafts were.

**Things that look like off-by-ones and are not.**

- `num_draft_tokens = end_idx - start_idx - 1`: the verify pass has n + 1 rows, and the last one has
  no draft.
- `num_retracted_kv_cache_tokens = spec_tokens_count + 1 - accepted_tokens`: `accepted_tokens` counts
  emitted tokens (section 1d).
- The first cycle has fewer than k drafts. Every length above is derived from `initial_idx` and
  `len_draft_probs`, not from k.
- `RejectionSampler.MAX_SPEC_LEN` (10) is passed as `max_spec_len` (line 432). It only sets the width
  of the output buffer and the kernel grid; the kernels take n from `cu_num_draft_tokens`.
- The bump of `MAX_SPEC_LEN` in `serve/run.py:354-356` and in `SparseSpecEngine.__init__` triggers on
  `spec_stride + 1 >= MAX_SPEC_LEN` although `==` would fit. It wastes rows and breaks nothing.

**Check of `serve/rl/scheduler.py`'s docstring (lines 10-27).** Its conclusion agrees with the
derivation above: output token j of a verify pass has target probability `probs[start + j, token_j]`,
and the first response token is read from row `token_indptr[idx + 1] - 1` of the NORMAL pass. The
implementation (`update_requests` lines 206-213, `_rl_collect_logprobs` lines 230-267) uses exactly
those rows and takes the count from `len(out_tokens)`, so it is correct for any n. Two sentences of
the docstring are looser than the code:

- "a verify pass feeds `k + 1` tokens" and "`m == k`" hold from a request's second cycle on. The first
  cycle feeds n + 1 = k + 1 - `initial_idx` tokens.
- "sampled from, or accepted against, a row of the full-attention probability tensor" does not
  literally cover the recovered token, which is sampled from normalize(max(p - q, 0)). Its marginal
  law is still row `start + m` (that is the speculative-sampling theorem), so reporting
  `probs[start + m, token]` as its target probability is right.

The docstring's description of upstream EOS handling is accurate (section 6.3, R1).

## 6. Deviations from the textbook algorithm, and hazards

### 6.1 No effect on the output distribution

| # | Observation | Where | Why it does not matter |
|---|---|---|---|
| N1 | A request's first cycle has n = k - `initial_idx` drafts, 1 <= n <= k. `initial_idx` is chosen at admission to stagger verify passes across requests. | `spec_scheduler.py:66-92`, `160`, `567` | Speculative sampling is exact for any draft length chosen before the cycle's random draws. All buffers are sized by n (asserts at 369 and 562-565). In the transcription, first cycles of 1, 2, 3 and k drafts all match the target law. |
| N2 | The first cycle drafts with dense attention (cold start). | `pillar.py:106-109` | q only affects how often drafts are accepted. |
| N3 | One Exp(1) vector per request, shared across its positions. | `rejection_sampler.py:279-287` | Section 1b. |
| N4 | The verify pass samples a token from each of its n + 1 rows, and recovered tokens are computed for all positions. Only one of each can be used. | `scheduler.py:612-616`, `spec_scheduler.py:374` | Unused draws, independent of what is kept. |
| N5 | The `draft_prob > 0` guard. | `rejection_sampler.py:342-344` | Unreachable (section 1a). |
| N6 | Uniforms are drawn twice. | `rejection_sampler.py:254-262` | i.i.d. replaced by i.i.d. |
| N7 | `MAX_SPEC_LEN = 10`. | `rejection_sampler.py:63`, `request.py:53-57`, `spec_scheduler.py:48`, `174` | k + 1 = 9 <= 10. The draft buffer has 10 rows for at most 8 drafts, the output buffer 11 columns for at most 9 tokens. A larger k raises the constant before any request exists, and the scheduler asserts it. |
| N8 | Quality of the draft: sparsity budget, stale or even wrong pillar indices. | `pillar.py` | Exactness needs three things only: d_j was sampled from q_j, the same q_j is used in the test and the residual, and p_j is the true target row. Nothing is required of q itself. |
| N9 | Drafting continues past a drafted EOS; tokens after the first EOS of a verify output are dropped. | `serve/rl/scheduler.py:169-177` | Stopping at the first EOS is the target process's own stopping rule. |
| N10 | CUDA-graph padding: the sampler runs on padded rows and returns the first `clip_length`. | `graph_model.py:334-339`, `sampler.py:148-173` | Only the amount of random numbers consumed changes. |
| N11 | Dead code: `Sampler.sample`, `TopKTopPSampler`, the `NO_DRAFT_PROBS` branches. | `sampler.py:47-124`, `175-203` | Never executed. |

### 6.2 Affects the output distribution, at floating-point level only

| # | Effect | Size per token | Note |
|---|---|---|---|
| F1 | The target is the float32 softmax of bf16 logits. | Definitional. | It applies to every token, including the non-speculative first one. |
| F2 | The stored rows do not sum to exactly 1. | Total variation at most \|sum(p) - sum(q)\|, about 1e-7. | See below. |
| F3 | The quotient p/q is rounded to float32. | Relative error 6e-8 in the acceptance probability. | |
| F4 | `>=` with u exactly 0.0 accepts a token with p(x) = 0. | About 2^-53. | The RL scheduler would then report log p = -inf for that token. |
| F5 | The Exp(1) noise has finite resolution. | Bounded by the probability mass on tokens whose share of the race is below about 6e-8. Not measured. | Shared with the non-speculative path and with stock vLLM. See below. |
| F6 | An all-zero residual makes the recovered token id 0. | At most about 3e-7, and only when p and q agree to rounding on the whole vocabulary. | See below. |

**F2.** Drafts are sampled from q / S_q (the race normalizes implicitly) but accepted with
min(1, p(x)/q(x)) on the stored values, and the recovered token is sampled from the normalized
residual. Writing S_p and S_q for the row sums and A for the sum of min(p, q), the emitted law is
[min(p, q)(x) + c * max(p - q, 0)(x)] / S_q with c = (S_q - A) / (S_p - A). For S_p = S_q this is
exactly p / S_p. Otherwise its total-variation distance from p / S_p is A * |S_p - S_q| / (S_p * S_q),
which is at most about |S_p - S_q|. A CPU transcription of the kernel's blocked float32 softmax
(`fast_fused_op.py:89-116` and `138-141`) on bf16-rounded Gaussian logits with V = 151,936 and
standard deviations 1, 3, 6 and 12 gave |sum - 1| <= 1.4e-7 over 20 rows. The GPU's reduction order
may differ, so that number is indicative.

**F5.** Both races divide by float32 noise from `Tensor.exponential_()`. Torch's CUDA transform computes
-log(v) from a uniform v in (0, 1] and clamps the result so it is never below 2^-24, about 6e-8
(`ATen/core/TransformationHelper.h`, function `exponential`, read in the local torch 2.15 wheel; the
box has torch ~2.7). For a float32 tensor v is, as far as I can tell from memory of torch's CUDA
distribution kernel, a float32, and float32 values just below 1 are 2^-24 apart. The noise then cannot
resolve events rarer than about 6e-8: a token whose share of the race weights is far below that wins
less often than it should, increasingly so as the share shrinks. The bias is bounded by the mass the
model puts on such tokens, which I could not measure here. It is not specific to speculation. The
engine's first response token uses the same race, and so do both samplers in stock vLLM 0.9.1
(`vllm/v1/sample/ops/topk_topp_sampler.py:242-254`, `vllm/model_executor/layers/sampler.py:528-545`),
so it does not separate the two arms of the experiment. Appendix B has a ten-second GPU check of the
noise grid.

**F6.** If max(p_j - q_j, 0) is zero everywhere, every score is 0, and the online argmax returns index
0 (first block, leftmost tie). The token is only read after a rejection. A rejection needs
p(x) < q(x) for the drafted x, and an all-zero residual needs p <= q everywhere, which with both sums
within 1.4e-7 of 1 forces q(x) - p(x) <= 3e-7. The probability of drafting such an x and rejecting it
is then at most about 3e-7. When the two rows are bitwise equal, nothing is ever rejected.

### 6.3 Real deviations, none active in the RL configuration

**R1. Upstream EOS handling after a verify pass changes where a response ends.**

```python
# serve/scheduler/spec_scheduler.py:122-129
# Check for EOS token in last spec_stride+1 tokens
for i in range(self.spec_stride + 1, 0, -1):
    if len(request.token_ids) - request.len_input < i:
        break
    if request.token_ids[-i] in self.eos_token_id:
        # clear tokens after the EOS token
        request.token_ids = request.token_ids[:-i]
        return True
```

- `[:-i]` removes the EOS itself, not only what follows it.
- The loop starts at i = k + 1, and the guard is `break`, not `continue`. While a request has fewer
  than k + 1 response tokens, nothing is scanned. An EOS emitted then can be missed for good. With
  k = 8: the first verify emits 2 tokens, the second of them EOS, so the response has 3 tokens and the
  scan is skipped. If the next verify emits 9 tokens, the scan covers only those 9, and generation
  continues past the EOS.
- In NORMAL mode the check reads `token_ids[-1]` (lines 101-106), which is a prompt token while the
  prompt is still being prefilled in chunks.

None of this changes the per-token conditional law, but it changes the sequence that is returned.
`RLSpecScheduler.is_request_complete` (`serve/rl/scheduler.py:150-182`) replaces the method: it scans
exactly the tokens the pass emitted, earliest first, keeps the EOS, deletes what follows, and ignores
unfinished prefill. By code reading that fixes all three points for the RL path.

**R2. top_p and top_k do not exist** (section 3). The RL wrapper refuses them.

**R3. Pointer offsets for the probability copies are computed in int32.**

```python
# serve/request/kv_cache_ptr/base.py:432-437 (same pattern at 482-487 and 542-553)
src_tensor_ptrs_host = np.full_like(
    src_start_pos, src_start_addr, dtype=np.int64
)
src_tensor_ptrs_host += (
    src_start_pos * dtype_to_bytes(src_tensor.dtype) * n_stride
)
```

`src_start_pos` is an int32 array, so the product is evaluated in int32 before it is added to the
int64 address. With V = 151,936 it wraps at row index 2^31 / (4 * V) = 3533.5. Run on numpy 1.26.4:
row 3534 gives -2,147,200,000 instead of 2,147,767,296. The copy kernels would then read or write a
wrong address, which is memory corruption, not a statistical bias. The project cannot reach it: one
request prefills at a time (at most `chunk_size` = 256 rows) and the other 63 contribute at most 9 rows
each, 823 rows in total, and the cumulative draft count is at most 64 * 8 = 512. Several hundred
concurrent requests, or a chunk size in the thousands, would. Not changed (latent, in a file this note
does not own, and not runnable here).

**R4. Asynchronous scheduling is outside this review.** Every row derivation in section 5 is for
`async_cpu=False`.

## 7. Conclusion

For T = 1.0, top_p = 1.0, top_k = -1 and synchronous scheduling, by code reading: the first response
token is sampled directly from the full-attention row, and every later token is produced by textbook
speculative sampling in which the draft token was sampled from exactly the stored q row, the
acceptance test and the residual use that same row and the matching full-attention row, the bonus
token is sampled from the full-attention row after the last draft, and the full-attention rows are
computed over a KV cache that verify passes have rewritten with full attention. The emitted tokens are
therefore distributed as samples from the engine's full-attention model, softmax of
`float32(bf16 logits)`, up to the floating-point effects F1-F6, of which only the noise resolution F5
is not obviously below 1e-6 and that one is shared with the non-speculative sampler and with stock
vLLM. This is a statement about the code as written. It has not been observed on a GPU.

**What would falsify it on a GPU.** The claim has two halves, and the gates listed in
`integration/gates/__init__.py` at the time of writing cover only the first.

| Half of the claim | Falsified by | Gate |
|---|---|---|
| The rows called p are the full-attention conditionals of the right positions (alignment, KV integrity, weights). | Engine-reported log-probs that differ from the trainer forward on the same sampled sequences by more than bf16 noise, especially at fixed positions of the verify cycle (every recovered token, every bonus token). | 3c, `gate_c_reload.py`, at T = 1. During training, `training/rollout_probs_diff_*` shows the same quantity every step (`run_seed_sparsespec.sh` sets `calculate_log_probs=True`). |
| Same, for the greedy path. | Greedy output that leaves HuggingFace's greedy sequence at a position where HF's logit margin is large. | 3b, `gate_b_greedy.py`. |
| Tokens are drawn from those rows with the right probabilities (acceptance rule, residual resampling, pairing of q rows with draft tokens). | A sampling-law test. | None yet. |

Gates 3b and 3c cannot see an error in the second half. In greedy mode acceptance is exact match and
the residual is one-hot, so a wrong inequality or a wrong residual changes nothing. In 3c the engine
and the trainer both evaluate p at whatever token was emitted, so they agree even if that token was
drawn from the wrong law. The transcription's negative controls "recover from p" and "recover with
the draft's noise" (Appendix A) are bugs of exactly this kind.

**Proposed measurement for the second half.** It needs no extra GPU pass beyond what 3c already runs.
If x_t is drawn from p_t, then -log p_t(x_t) has mean H(p_t), the entropy of the row, and variance
equal to the row's varentropy. So

z = sum_t [-log p_t(x_t) - H(p_t)] / sqrt(sum_t varentropy(p_t))

is approximately standard normal over many tokens, and the sum is a martingale, so dependence along a
sequence does not matter. The scoring pass of 3c (`integration/gates/common.py:hf_trainer_logprobs`)
already holds the full log-softmax row at every response position, so H and the varentropy are two
more lines there. A sampler that over-accepts drafts emits tokens closer to q, and its mean surprisal
moves from H(p) to the cross-entropy of its actual law against p. With 1e5 tokens the test resolves
about 0.01 nat per token. The engine's forward and the trainer's forward differ by bf16 noise, which
gives z a small offset of its own; the first reading below measures it. Three readings make the test
decisive:

- the same statistic on stock-vLLM samples (`gate_c_vllm_reference.py`) and on `ReferenceEngine`
  samples, as the calibration of the test itself;
- invariance: z must stay within noise when `spec_stride` or `budget_ratio` changes, because those
  may change speed and acceptance but not the law;
- first cycles: acceptance clearly higher than in later cycles while the draft is dense (section 2).
  A mispaired q row would show up there first.

The blind spot of this test is a wrong law whose cross-entropy against p happens to equal H(p). A
two-sample test on the first few response tokens of one prompt, speculative against non-speculative,
closes it at the cost of many more samples.

## Appendix A: CPU transcription of the algorithm

This is not the kernel. It restates in torch, line by line, what the cited source lines compute, on a
toy order-2 Markov model (V = 6, k = 4) whose target law over the first four response tokens is known
exactly. The draft model is a perturbed copy of the target with some q(x) = 0 where p(x) > 0 and some
p(x) = 0 where q(x) > 0, so the residual path and the always-reject path are exercised. Each
configuration is compared with the exact law by a chi-square test over all 4-token outcomes (cells
with an expected count below 10 are pooled).

Run with torch 2.15.0 on CPU, 8 replicates of 2.5 million sequences for the first six rows, one
replicate for the negative controls:

```
V=6 spec_stride=4 L=4 replicates=8 batch=2500000 torch=2.15.0
row sums: p in [0.999999881, 1.000000119]  q in [0.999999881, 1.000000119]  mean sum_x min(p, q) = 0.643
```

| Variant | What it is | First-cycle drafts | Sequences | chi-square | df | p-value | Samples in zero-probability cells | p-value of each replicate |
|---|---|---|---|---|---|---|---|---|
| `plain` | direct sampling from the target, no speculation | - | 20,000,000 | 980.2 | 940 | 0.1764 | 0 | 0.02 0.65 0.47 0.42 0.09 0.14 0.51 0.72 |
| `code` | the sampler as written | 4 | 20,000,000 | 925.6 | 940 | 0.6250 | 0 | 0.55 0.37 0.62 0.95 0.81 0.54 0.48 0.72 |
| `code` | the sampler as written | 3 | 20,000,000 | 893.8 | 940 | 0.8573 | 0 | 0.90 0.69 0.05 0.87 0.67 0.84 0.90 0.54 |
| `code` | the sampler as written | 2 | 20,000,000 | 901.6 | 940 | 0.8115 | 0 | 0.64 0.55 0.98 0.01 0.17 0.07 0.62 0.52 |
| `code` | the sampler as written | 1 | 20,000,000 | 945.8 | 940 | 0.4408 | 0 | 0.22 0.68 0.68 0.73 0.75 0.52 0.18 1.00 |
| `strict` | as written, with `>` instead of `>=` | 4 | 20,000,000 | 928.3 | 940 | 0.6006 | 0 | 0.54 0.65 0.47 0.95 0.89 0.52 0.36 0.30 |
| `target_row_plus_one` | control: target row s + j + 1 used for position j | 4 | 2,500,000 | 345,912,993.1 | 916 | 0.0000 | 34815 | (one replicate) |
| `recover_with_draft_noise` | control: recovered token uses the draft pass's noise | 4 | 2,500,000 | 26,297.2 | 916 | 0.0000 | 0 | (one replicate) |
| `recover_from_p` | control: recovered token sampled from p, not the residual | 4 | 2,500,000 | 1,053,430.2 | 916 | 0.0000 | 0 | (one replicate) |

Reading: the sampler as written is indistinguishable from the target at 20 million sequences for
every first-cycle length, and so is the `>` variant. `plain` is direct sampling from the target
through the same race, and shows what an exact sampler scores on this test, including the occasional
small p-value in a single replicate (0.02 for `plain` itself). The three negative
controls, one for each kind of error this review looked for (wrong target row, noise shared with the
draft sampling, wrong recovery distribution), fail by a wide margin, so the test would have seen them.

<details>
<summary>Script (scratch copy outside the repo; reproduced here in full)</summary>

```python
"""CPU transcription of the SparseSpec speculative-sampling arithmetic (gate 3a).

NOT the real kernels (they need triton + CUDA). Each step restates in torch
what the cited source line computes, on a toy order-2 Markov "model" for which
the exact target law of the first L response tokens is known.

  sampler.py:160-171, fast_fused_op.py:141-168   token = argmax(probs / Exp(1)), one noise row per row
  rejection_sampler.py:249-253                   u ~ U[0,1) float64, one per draft token
  rejection_sampler.py:282-287                   ONE Exp(1) vector per request, shared by its positions
  rejection_sampler.py:412-428                   recovered_j = argmax(max(p_j - q_j, 0) / noise)
  rejection_sampler.py:344                       accept iff q(x) > 0 and float32(p(x) / q(x)) >= u
  rejection_sampler.py:355-361                   bonus appended iff nothing was rejected
  spec_scheduler.py:374                          bonus = token the regular sampler drew from the last verify row
  spec_scheduler.py:160, 567                     first cycle has spec_stride - initial_idx drafts

Variant "plain" samples every token directly from the target (no speculation):
it shows what an exact sampler scores. "strict" replaces >= by >. The other
variants are negative controls that must FAIL.
Usage: python sim_spec_sampler.py [replicates] [batch]
"""

import sys

import torch

V, K, L = 6, 4, 4  # vocab, spec_stride, number of leading response tokens compared
PROMPT = (0, 1)  # fixed two-token "prompt" (the order-2 context)
EXP_MIN = 2.0**-24  # CUDA exponential_ is bounded away from 0 (ATen TransformationHelper.h)


def make_tables(seed):
    g = torch.Generator().manual_seed(seed)
    zp = torch.randn(V, V, V, generator=g) * 1.2
    zq = zp + torch.randn(V, V, V, generator=g) * 0.9  # "sparse" draft model = perturbed target
    zq[:, 0::2, 0] = -float("inf")  # q(x) == 0 < p(x): reachable only as recovered / bonus token
    zp[:, 1::3, 1] = -float("inf")  # p(x) == 0 < q(x): such a draft must always be rejected
    return torch.softmax(zp.float(), -1), torch.softmax(zq.float(), -1)  # float32, like the kernel


def race(weights):
    """argmax(w / E), E ~ Exp(1) i.i.d.: index i with probability w_i / sum(w)."""
    noise = torch.empty_like(weights).exponential_().clamp_min_(EXP_MIN)
    return (weights / noise).argmax(-1), noise


def simulate(P, Q, batch, n_first, variant="code"):
    cap = 2 + L
    seq = torch.zeros(batch, cap, dtype=torch.long)
    seq[:, 0], seq[:, 1] = PROMPT
    length = torch.full((batch,), 2, dtype=torch.long)
    rows = torch.arange(batch)

    def append(tok, mask):
        mask = mask & (length < cap)
        seq[rows[mask], length[mask]] = tok[mask]
        length[mask] += 1

    # NORMAL pass: first response token sampled from the target row (spec_scheduler.py:464-466)
    append(race(P[seq[:, 0], seq[:, 1]])[0], torch.ones(batch, dtype=torch.bool))

    n = n_first
    while bool((length < cap).any()):
        idx = length.clamp(max=cap)
        if variant == "plain":  # no speculation at all: what an exact sampler scores on this test
            append(race(P[seq[rows, idx - 2], seq[rows, idx - 1]])[0], torch.ones(batch, dtype=torch.bool))
            continue
        ca, cb = seq[rows, idx - 2], seq[rows, idx - 1]  # (.., t0): last two confirmed tokens

        # n DRAFT passes: draft_probs row j is the row draft token j was sampled from
        qs, ds, draft_noise = [], [], []
        xa, xb = ca, cb
        for _ in range(n):
            q = Q[xa, xb]
            d, e = race(q)
            qs.append(q), ds.append(d), draft_noise.append(e)
            xa, xb = xb, d

        # VERIFY pass feeds [t0, d_0..d_{n-1}]: row j conditions on t0, d_0..d_{j-1}  (n + 1 rows)
        ps = []
        xa, xb = ca, cb
        for j in range(n + 1):
            ps.append(P[xa, xb])
            if j < n:
                xa, xb = xb, ds[j]
        bonus = race(ps[n])[0]  # regular sampler on the last verify row

        u = torch.rand(batch, n, dtype=torch.float64)
        shared = torch.empty(batch, V).exponential_().clamp_min_(EXP_MIN)  # one vector per request

        rejected = torch.zeros(batch, dtype=torch.bool)
        for j in range(n):
            p_row = ps[j + 1] if variant == "target_row_plus_one" else ps[j]
            noise = draft_noise[j] if variant == "recover_with_draft_noise" else shared
            recovered = (torch.clamp(p_row - qs[j], min=0) / noise).argmax(-1)
            if variant == "recover_from_p":  # residual replaced by p itself
                recovered = race(p_row)[0]
            pd = p_row.gather(1, ds[j][:, None])[:, 0]
            qd = qs[j].gather(1, ds[j][:, None])[:, 0]
            ratio = (pd / qd).double()  # float32 division, compared in float64
            accept = (qd > 0) & ((ratio > u[:, j]) if variant == "strict" else (ratio >= u[:, j]))
            live = ~rejected
            append(torch.where(accept, ds[j], recovered), live)
            rejected |= live & ~accept
        append(bonus, ~rejected)
        n = K
    return seq[:, 2:]


def exact_law(P):
    """Joint law of the first L tokens under the target (rows renormalised in float64)."""
    Pn = P.double() / P.double().sum(-1, keepdim=True)
    grid = torch.cartesian_prod(*[torch.arange(V)] * L)
    prob = torch.ones(len(grid), dtype=torch.float64)
    a = torch.full((len(grid),), PROMPT[0])
    b = torch.full((len(grid),), PROMPT[1])
    for i in range(L):
        prob *= Pn[a, b, grid[:, i]]
        a, b = b, grid[:, i]
    return prob


def counts(samples):
    code = torch.zeros(samples.shape[0], dtype=torch.long)
    for i in range(L):
        code = code * V + samples[:, i]
    return torch.bincount(code, minlength=V**L).double()


def chi_square(obs, prob, min_expected=10.0):
    n = obs.sum()
    exp = prob * n
    big = exp >= min_expected  # cells with a small expectation are pooled into one
    o = torch.cat([obs[big], obs[~big & (exp > 0)].sum()[None]])
    e = torch.cat([exp[big], exp[~big & (exp > 0)].sum()[None]])
    o, e = o[e > 0], e[e > 0]
    stat, df = float(((o - e) ** 2 / e).sum()), len(e) - 1
    half = lambda x: torch.tensor(x / 2.0, dtype=torch.float64)  # noqa: E731
    pval = float(torch.special.gammaincc(half(df), half(stat)))
    return stat, df, pval, float(0.5 * (obs / n - prob).abs().sum()), int(obs[exp == 0].sum())


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    batch = int(float(sys.argv[2])) if len(sys.argv) > 2 else 2_500_000
    P, Q = make_tables(seed=7)
    prob = exact_law(P)
    print(f"V={V} spec_stride={K} L={L} replicates={reps} batch={batch} torch={torch.__version__}")
    print(f"row sums: p in [{float(P.sum(-1).min()):.9f}, {float(P.sum(-1).max()):.9f}]  "
          f"q in [{float(Q.sum(-1).min()):.9f}, {float(Q.sum(-1).max()):.9f}]  "
          f"mean sum_x min(p, q) = {float(torch.minimum(P, Q).sum(-1).mean()):.3f}")
    runs = [("plain", K), ("code", K), ("code", 3), ("code", 2), ("code", 1), ("strict", K),
            ("target_row_plus_one", K), ("recover_with_draft_noise", K), ("recover_from_p", K)]
    for number, (variant, n_first) in enumerate(runs):
        total = torch.zeros(V**L, dtype=torch.float64)
        per_rep = []
        for r in range(reps if variant in ("plain", "code", "strict") else 1):  # negative controls: one is plenty
            torch.manual_seed(1000 * number + r)
            c = counts(simulate(P, Q, batch, n_first, variant))
            per_rep.append(chi_square(c, prob)[2])
            total += c
        stat, df, pval, tv, impossible = chi_square(total, prob)
        print(f"{variant:25s} first-cycle drafts={n_first}  N={int(total.sum()):>9d}  chi2={stat:12.1f}  df={df}  "
              f"p={pval:.4f}  TV={tv:.5f}  zero-probability samples={impossible}  "
              f"replicate p: {' '.join(f'{x:.2f}' for x in per_rep)}", flush=True)
```

</details>

## Appendix B: what could not be verified without a GPU

| Assumption or quantity | Why it matters | How to check on the box |
|---|---|---|
| Triton passes a Python `float` argument as fp32 and promotes an fp32/fp64 comparison to fp64. | The acceptance comparison (section 1a) and `inv_temperature`. At T = 1.0 the second is irrelevant. | Covered indirectly by the sampling-law test of section 7. |
| `tl.argmax` breaks ties toward the lowest index. | Greedy ties, F6. | Gate 3b. |
| `Tensor.exponential_()` on CUDA is never 0 and has a 2^-24 grid near 0 (F5). Read from the torch 2.15 header, not torch 2.7. | A zero would produce 0/0 in the races. | `e = torch.empty(2**26, device="cuda").exponential_(); print(e.min().item(), (e[e < 1e-6].unique()[:8] * 2**24).tolist())`. Expect a minimum of 5.96e-08 and values close to 1, 2, 3, ... |
| The flashinfer kernels do what the variant declarations say: sparse indices for DRAFT, all pages for VERIFY, causal masking inside the verify pass, append before attention. | p is the full-attention conditional (section 2). | Gates 3b and 3c. |
| Row sums of the GPU softmax are within about 1e-7 of 1 (F2). | Size of F2 and F6. | `probs.double().sum(-1)` on one pass. |
| Mass on tokens with probability below 2^-24 for Qwen3-0.6B at T = 1 (F5). | Size of F5. | From the trainer forward in gate 3c: mean over positions of the sum of p(x) over tokens with p(x) < 2^-24. |
| Acceptance rate, and higher acceptance in first (dense) cycles. | Diagnostic for the pairing of q rows with draft tokens. | Engine stats; a per-cycle breakdown would have to be added. |
| Everything about asynchronous scheduling. | Out of scope. | Not used by the RL engine. |
