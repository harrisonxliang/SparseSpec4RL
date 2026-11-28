"""Correctness gates for the SparseSpec verl rollout backend.

Run on the A100 box, from the repo root, AFTER the CPU unit tests pass:

    3a  baseline/results/sparsespec_checks/3a_sampler_review.md    sampler review (written, nothing to run)
        python -m pytest serve/rl/tests/test_engine_gpu.py -v -s   engine unit tests (GPU), before 3b
    3b  python -m integration.gates.gate_b_greedy                  engine greedy == HF greedy
    3c  python -m integration.gates.gate_c_reload                  weight reload + log-prob agreement
        python integration/gates/gate_c_vllm_reference.py          same measurement for stock vLLM (baseline env)
    3d  python -m integration.gates.gate_d_smoke_check             post-check of a short training run

    python -m integration.gates.make_synthetic_ckpt                stand-in checkpoint for 3c (base + noise)
    python -m pytest integration/gates/tests -q                    CPU tests of the gate scripts themselves

Every gate takes ``--engine {sparsespec,reference}``, writes JSON + markdown
under ``baseline/results/sparsespec_checks/`` and exits non-zero when a check
fails; 3b / 3c also take ``--device`` and model / checkpoint paths and look at
free GPU memory (nvidia-smi) before allocating on the shared card.
``integration.gates.common`` does not import ``serve.rl.engine`` (flashinfer /
triton / CUDA); the SparseSpec engine is imported lazily, only for
``--engine sparsespec``.
"""
