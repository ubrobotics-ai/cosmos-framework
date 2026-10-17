#!/usr/bin/env bash
# ONE training iteration of action_policy_libero_edge_value on the 16 GB 4060 Ti (smoke test, not the real recipe):
# Verified 2026-10-17 on an RTX 4060 Ti 16 GB (WSL2, torch 2.10+cu130): iteration 1 loss 15.66, 8.46M trainable
# params (action heads), DCP checkpoint saved. NOT the real recipe -- see the TOML/launcher for that.
# EMA off, bf16 weights, full activation checkpointing, no compile, 1 sample, ONLY the action heads train
# (they carry the new value column). DRY=1 parses the config and exits without touching the GPU.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.13/site-packages/nvidia/cu13/lib
export LIBERO_ROOT=${LIBERO_ROOT:?set LIBERO_ROOT to the local libero_10 LeRobot dir} BASE_CHECKPOINT_PATH=examples/checkpoints/Cosmos3-Edge
export NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo MASTER_ADDR=127.0.0.1  # mirrored WSL: NCCL otherwise picks the /32 Tailscale-mirrored eth0 and hangs connecting to itself
export WAN_VAE_PATH=examples/checkpoints/wan22_vae/Wan2.2_VAE.pth IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-outputs/smoke} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OPTS=(trainer.max_iter=1 trainer.logging_iter=1 job.wandb_mode=disabled
  model.config.parallelism.data_parallel_shard_degree=1 model.config.parallelism.data_parallel_replicate_degree=1
  model.config.ema.enabled=false model.config.parallelism.fsdp_master_dtype=bfloat16
  model.config.activation_checkpointing.mode=full model.config.compile.enabled=false
  dataloader_train.max_samples_per_batch=1 dataloader_train.dataloader.num_workers=1 dataloader_train.dataloader.prefetch_factor=2
  dataloader_train.dataloader.persistent_workers=false
  "optimizer.keys_to_select=[action2llm,llm2action,action_modality_embed]" "scheduler.warm_up_steps=[0]" "scheduler.cycle_lengths=[1]"
  trainer.callbacks.manual_gc.warm_up=0)
if [[ -n "${DRY:-}" ]]; then
  CUDA_VISIBLE_DEVICES= .venv/bin/python -m cosmos_framework.scripts.train --sft-toml=examples/toml/sft_config/action_policy_libero_10_edge_value.toml --dryrun -- "${OPTS[@]}"
else
  .venv/bin/torchrun --nproc_per_node=1 --master_port=50012 -m cosmos_framework.scripts.train \
    --sft-toml=examples/toml/sft_config/action_policy_libero_10_edge_value.toml -- "${OPTS[@]}"
fi
