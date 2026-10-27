"""Entry script for FSDP training of OctWorld Wan FramePack.

Build the experiment folder once::

    python -m octworld.training.wan.init_training_folder \\
        --yaml octworld/training/wan/config/train_base.yaml \\
        --output_root experiments/ \\
        --source model_weights/Wan2.1-I2V-14B-480P

Then train::

    torchrun --nproc_per_node=8 octworld/training/wan/train.py \\
        --exp_dir experiments/dl3dv-octworld-wan-b64

By default the repo's ``octworld/training/wan/config/train_base.yaml`` is used
as the source of truth and copied into ``<exp_dir>/<tag>/config.yaml``. Pass
``--config <path>`` to use a different yaml.

The init resources (``init/model.pt``, ``pipe_config.json``,
``negative_prompt.pt``) stay at the top of ``<exp_dir>`` and are shared by every
run. Each run's own outputs (``config.yaml``, ``ckpts``, ``training_logs``,
``tb_logs``, ``debug_profile``) are written under ``<exp_dir>/<tag>/`` so the
same init folder can host multiple experiments. ``--tag`` defaults to
``default``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from octworld.training.wan.config.config import TrainConfig
from octworld.training.wan.octworld_wan_trainer import OctWorldWanTrainer

DEFAULT_CONFIG = os.path.join(
    PROJECT_ROOT, "octworld", "training", "wan", "config", "train_base.yaml")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", required=True,
                   help="Experiment dir produced by init_training_folder.py")
    p.add_argument("--tag", default="default",
                   help="Run tag. This run's outputs (config.yaml, ckpts, "
                        "training_logs, tb_logs, debug_profile) are written "
                        "under <exp_dir>/<tag>/, allowing multiple runs to "
                        "share one init folder. Defaults to 'default'.")
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help="Path to the training yaml. This file is copied into "
                        "<exp_dir>/<tag>/config.yaml (overwriting any existing "
                        "one) and used as the source of truth for training "
                        "settings.")
    p.add_argument("--debug", action="store_true",
                   help="Run a short profiled debug loop (torch.profiler) "
                        "instead of full training. Writes a timing/memory "
                        "report + chrome trace to <exp_dir>/debug_profile/.")
    p.add_argument("--debug_steps", type=int, default=10,
                   help="Number of steps to run in --debug mode (default 10).")
    p.add_argument("--activation_memory_budget", type=float, default=None,
                   help="Override model.activation_memory_budget (compile-only "
                        "Memory Budget API, 0=plain AC .. 1=default compile). "
                        "When set, also forces compile_blocks=True. Useful for "
                        "budget sweeps without editing the yaml.")
    p.add_argument("--use_selective_checkpointing", action="store_true",
                   help="Debug/sweep override: use selective activation "
                        "checkpointing (SAC) instead of torch.compile. Forces "
                        "compile_blocks=False and use_gradient_checkpointing="
                        "True. Mutually exclusive with "
                        "--activation_memory_budget.")
    p.add_argument("--mesh_mode", choices=["1d", "2d"], default=None,
                   help="Override basic.fsdp.mesh_mode (1d full-shard / 2d "
                        "HSDP). Useful for sweeping FSDP mesh forms in debug.")
    return p.parse_args()


def main():
    args = parse_args()

    use_dist = "RANK" in os.environ
    if use_dist:
        # Bind this process to its GPU *before* init_process_group so NCCL does
        # not default every rank to cuda:0 (which hangs/OOMs on multi-GPU and
        # multi-node). Pass device_id so the PG is created on the right device.
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
        else:
            dist.init_process_group(backend="gloo")

    is_main = (not use_dist) or dist.get_rank() == 0

    # Per-run outputs live under <exp_dir>/<tag>/ so the same init folder can
    # host multiple experiments without clobbering each other.
    run_dir = os.path.join(args.exp_dir, args.tag)

    # The provided yaml (default: train_base.yaml in the repo) is the source of
    # truth. Only the global rank 0 creates the run dir and copies the yaml into
    # <run_dir>/config.yaml so that multi-node runs do not concurrently write
    # the same shared file. Everyone else waits on the barrier before reading.
    exp_config_path = os.path.join(run_dir, "config.yaml")
    if is_main:
        os.makedirs(run_dir, exist_ok=True)
        shutil.copyfile(args.config, exp_config_path)
    if use_dist:
        dist.barrier()

    raw_cfg = OmegaConf.load(exp_config_path)
    if is_main:
        print(
            "Training config (%s):\n%s"
            % (exp_config_path, OmegaConf.to_yaml(raw_cfg)))
    cfg = TrainConfig(**OmegaConf.to_container(raw_cfg, resolve=True))
    # Inject runtime paths (not part of the static yaml). The run's outputs
    # (ckpts/logs/tb_logs/debug_profile) are derived from ``basic.exp_dir`` by
    # the trainer, so point it at the tag dir. Init resources stay at the
    # top-level exp dir and are referenced explicitly below.
    cfg.basic.exp_dir = run_dir
    cfg.basic.init_ckpt_path = os.path.join(args.exp_dir, "init", "model.pt")
    # The negative-prompt embed lives at the top-level init folder. The trainer
    # resolves it relative to ``basic.exp_dir`` (now the tag dir), so make it
    # absolute here to keep pointing at the shared top-level copy.
    if not os.path.isabs(cfg.model.neg_text_embed_path):
        cfg.model.neg_text_embed_path = os.path.join(
            args.exp_dir, cfg.model.neg_text_embed_path)
    with open(os.path.join(args.exp_dir, "pipe_config.json")) as f:
        pipe_config = json.load(f)

    if args.activation_memory_budget is not None:
        # Budget sweep override: the compile-only Memory Budget API needs
        # compile_blocks on, so enable it alongside the budget.
        cfg.model.compile_blocks = True
        cfg.model.activation_memory_budget = args.activation_memory_budget
        if is_main:
            print(
                f"[cfg] activation_memory_budget override: "
                f"{cfg.model.activation_memory_budget} (compile_blocks=True)"
            )

    if args.use_selective_checkpointing:
        # SAC sweep override: manual (selective) activation checkpointing
        # instead of torch.compile. Disable compile so the two recompute paths
        # do not stack, and force manual AC on.
        if args.activation_memory_budget is not None:
            raise ValueError(
                "--use_selective_checkpointing and --activation_memory_budget "
                "are mutually exclusive (SAC vs compile-budget recompute).")
        cfg.model.compile_blocks = False
        cfg.model.use_gradient_checkpointing = True
        cfg.model.use_selective_checkpointing = True
        if is_main:
            print("[cfg] selective activation checkpointing override "
                  "(compile_blocks=False, use_gradient_checkpointing=True)")

    if args.mesh_mode is not None:
        cfg.basic.fsdp.mesh_mode = args.mesh_mode
        if is_main:
            print(f"[cfg] mesh_mode override: {cfg.basic.fsdp.mesh_mode}")

    if args.debug:
        # Debug/profiling runs on a single 8-GPU node. Keep the configured
        # gradient-accumulation (semantics unchanged) but resize the global
        # batch to the debug world so the batch math holds:
        #   global_batch = micro * grad_accum * world  (e.g. 1 * 4 * 8 = 32).
        # Must happen before the trainer's batch-math validation in
        # ``BasicTrainer.__init__``.
        world = dist.get_world_size() if use_dist else 1
        cfg.basic.global_batch_size = (
            cfg.basic.micro_batch_size
            * cfg.basic.gradient_accumulation_steps
            * world
        )
        if is_main:
            print(
                f"[debug] batch override: global_batch="
                f"{cfg.basic.global_batch_size} "
                f"(micro={cfg.basic.micro_batch_size} x "
                f"grad_accum={cfg.basic.gradient_accumulation_steps} "
                f"x world={world})"
            )

    trainer = OctWorldWanTrainer(
        cfg=cfg,
        pipe_config=pipe_config,
    )
    if args.debug:
        trainer.fit_debug(n_steps=args.debug_steps)
    else:
        trainer.fit()

    if use_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
