#!/usr/bin/env bash
# =============================================================================
# CANONICAL SHARED CONFIG — vLLM baseline arm AND SparseSpec arm.
#
# CONFIG_VERSION must be bumped on ANY change, and the change announced to the
# group: the two arms are only comparable if they share every value here and
# differ only in the rollout engine. Do not override these ad hoc in run
# scripts.
#
# Proposal (section 3.1): Qwen3-0.6B, bf16 full fine-tune, thinking mode,
# GRPO, 64 prompts/step x G=8, temperature 1.0, max response 8192 (drop to
# 4096 only if the step budget forces it — that is a CONFIG_VERSION bump).
# =============================================================================
export CONFIG_VERSION=v1

export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-0.6B}
export DATA_DIR=${DATA_DIR:-$HOME/data/sparsespec_baseline}
export CKPT_ROOT=${CKPT_ROOT:-$HOME/ckpts/sparsespec_baseline}
export PROJECT_NAME=sparsespec4rl

BASE_ARGS=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False

    data.train_files="$DATA_DIR/math_l3to5_train.parquet"
    data.val_files="$DATA_DIR/math500_test.parquet"
    data.train_batch_size=64
    data.max_prompt_length=1024
    data.max_response_length=8192
    data.filter_overlong_prompts=True
    data.truncation=error
    data.shuffle=True

    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True

    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=64
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0

    # Rollout engine — the ONLY block the SparseSpec arm may change.
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.n=8
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=10240
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480
    actor_rollout_ref.rollout.free_cache_engine=True

    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=20480
    actor_rollout_ref.ref.fsdp_config.param_offload=True

    custom_reward_function.path="$REPO_ROOT/baseline/reward.py"
    custom_reward_function.name=compute_score

    trainer.project_name="$PROJECT_NAME"
    trainer.n_gpus_per_node=1
    trainer.nnodes=1
    trainer.critic_warmup=0
    trainer.save_freq=25
    trainer.test_freq=-1
    trainer.val_before_train=False
    trainer.logger=['console']
)
