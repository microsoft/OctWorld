"""OctWorldWanTrainer — concrete Wan FramePack trainer.

Inherits :class:`~octworld.training.fsdp_trainer.FSDPTrainer`. Owns
``self.pipe`` (a :class:`WanMultiAdapterPipeline`) and implements every
model-specific hook the trainer stack expects:

    build_pipe / apply_fsdp / materialize  (FSDPTrainer)
    compute_loss / prepare_dataloader      (BasicTrainer)

It is **not** an ``nn.Module`` — ``self.pipe`` is the object FSDP wraps and the
optimizer trains.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from octworld.pipelines.wan.wan_pipeline import (
    WanMultiAdapterPipeline, NOISE_AUG_NUM_LEVELS,
)
from octworld.training.basic_trainer import BasicTrainer
from octworld.training.utils import time_binned_mse
from .config.config import TrainConfig


def _build_nested_config(pipe_config: dict) -> dict:
    """Normalize a ``pipe_config.json`` manifest into the nested
    ``{dit, camera_adapter}`` form expected by ``build_training_model``."""
    dit_cfg = dict(pipe_config["dit"])
    if "camera_adapter" in pipe_config:
        cam_cfg = dict(pipe_config["camera_adapter"])
    else:
        cam_cfg = dict(
            in_dim=pipe_config.get("camera_in_dim", 6),
            out_dim=dit_cfg["dim"],
            zero_init_output_proj=True,
        )
    return dict(dit=dit_cfg, camera_adapter=cam_cfg)


class OctWorldWanTrainer(BasicTrainer):
    """Concrete Wan trainer. Owns ``self.pipe`` and implements the hooks."""

    def __init__(
        self,
        *,
        cfg : TrainConfig,
        pipe_config: dict,
    ):
        super().__init__(cfg=cfg.basic)
        self.model_cfg = cfg.model
        self.data_cfg = cfg.data
        self.pipe_config = pipe_config
        self.gen_frames = cfg.data.gen_frames
        # Classifier-free guidance training dropout.
        self.random_drop_ratio = cfg.model.random_drop_ratio
        self.neg_text_embed = None
        if self.random_drop_ratio > 0:
            neg_path = cfg.model.neg_text_embed_path
            if not os.path.isabs(neg_path):
                neg_path = os.path.join(cfg.basic.exp_dir, neg_path)
            neg_data = torch.load(neg_path, map_location="cpu")
            self.neg_text_embed = neg_data["text_embed"]  # (L, 4096)
            print(
                "[cfg] loaded negative text embed %s from %s"
                % (tuple(self.neg_text_embed.shape), neg_path)
            )
        self.pipe = WanMultiAdapterPipeline()
        self.pipe.scheduler.set_timesteps(1000, training=True, shift=cfg.model.shift)

    # ---- FSDPTrainer abstract methods ---------------------------------
    def build_pipe(self):
        """Construct DiT + camera adapter (on the active device — meta when
        called from ``init_models_and_more``)."""
        self.pipe.build_training_model(
            config=_build_nested_config(self.pipe_config),
            use_gradient_checkpointing=self.model_cfg.use_gradient_checkpointing,
            use_selective_checkpointing=getattr(
                self.model_cfg, "use_selective_checkpointing", False),
            compile_blocks=getattr(self.model_cfg, "compile_blocks", False),
            activation_memory_budget=getattr(
                self.model_cfg, "activation_memory_budget", 0.5),
        )

    # ---- BasicTrainer abstract methods --------------------------------
    def compute_loss(self, batch: dict):
        """Flow-matching SFT loss for Wan FramePack.

        Only the **generation** segment of the latents is noised; the anchor +
        history segments stay clean (conditioning context). Timestep sampling,
        noising, target, and loss weighting are delegated to ``pipe.scheduler``
        (set up via ``set_timesteps(..., training=True)`` in ``build_pipe``).
        """
        pipe = self.pipe
        latents = batch["latents"]           # (B, 16, T_total, H, W) clean
        rendered_latents = batch["rendered_latents"]  # (B, 16, gen, H, W)
        rendered_masks = batch["rendered_masks"]      # (B, 4, gen, H, W)
        text_embed = batch["text_embed"]     # list of (L, 4096)
        raymap = batch["raymap"]
        clip_features = batch["clip_features"]

        device = latents.device
        B, C, T_total, H, W = latents.shape
        n_history_total = 20  # anchor(1) + history(19)
        gen_frames = self.gen_frames
        if gen_frames is None:
            gen_frames = T_total - n_history_total
        assert gen_frames > 0 and (n_history_total + gen_frames) == T_total

        # Split clean latents into [history | gen].
        hist_latents = latents[:, :, :n_history_total]
        gen_clean = latents[:, :, n_history_total:]

        # Sample timestep from scheduler's pre-computed timesteps (full range).
        # Keep timesteps in fp32: they are integer-valued in [0, 1000] and the
        # latent dtype (bf16) cannot represent them exactly (bf16 quantizes to a
        # ~8-step grid near 1000), which would corrupt the nearest-match sigma /
        # weight lookups and the model's time embedding.
        timestep_id = torch.randint(0, len(pipe.scheduler.timesteps), (B,))
        timestep = pipe.scheduler.timesteps[timestep_id].to(
            device=device, dtype=torch.float32)

        # Noise and noisy latents via scheduler. The scheduler's sigmas are
        # fp32, so add_noise promotes gen_noisy to fp32; cast back to the latent
        # dtype since precision management is the caller's responsibility.
        noise = torch.randn_like(gen_clean)
        gen_noisy = pipe.scheduler.add_noise(gen_clean, noise, timestep).to(latents.dtype)
        target = pipe.scheduler.training_target(gen_clean, noise, timestep)

        # History-condition noise augmentation: draw a per-sample bucket level.
        if getattr(self.model_cfg, "noise_aug", True):
            noise_aug_level = torch.randint(
                0, NOISE_AUG_NUM_LEVELS, (B,), device=device)
        else:
            noise_aug_level = None

        # Classifier-free guidance dropout: with prob ``random_drop_ratio``,
        # replace a sample's conditioning (text + camera raymap) with the
        # unconditional branch (negative text embed + zeroed raymap).
        if self.random_drop_ratio > 0:
            drop = torch.rand(B, device=device) < self.random_drop_ratio
            if drop.any():
                neg = self.neg_text_embed.to(
                    device=device, dtype=text_embed[0].dtype)
                text_embed = [
                    neg if drop[i] else text_embed[i] for i in range(B)
                ]
                raymap[drop] = 0.0

        # Forward through pipeline model_fn.
        pred = pipe.model_fn(
            gen_noisy, timestep,
            history_latents=hist_latents,
            rendered_latents=rendered_latents,
            rendered_masks=rendered_masks,
            context=text_embed,
            clip_features=clip_features,
            raymap=raymap,
            noise_aug_level=noise_aug_level,
        )

        # Compute weighted loss.
        pred_f = pred.float()
        target_f = target.float()

        # Per-sample MSE.
        per_sample_loss = F.mse_loss(pred_f, target_f, reduction='none')
        per_sample_loss = per_sample_loss.mean(dim=(1, 2, 3, 4))  # (B,)

        # Training weights from scheduler.
        w = pipe.scheduler.training_weight(timestep)  # (B,)
        loss = (w * per_sample_loss).mean()

        metrics = {
            "loss": loss.detach().item(),
            "loss_unweighted": per_sample_loss.mean().detach().item(),
            "pred_std": pred_f.detach().std().item(),
            "target_std": target_f.std().item(),
            "timestep_mean": timestep.mean().item(),
            "weight_mean": w.mean().item(),
        }
        # Per-timestep-bin MSE (BasicTrainer helper), bucketed by normalized t.
        num_train_timesteps = getattr(
            pipe.scheduler, "num_train_timesteps", len(pipe.scheduler.timesteps))
        t_norm = timestep.float() / num_train_timesteps
        metrics.update(time_binned_mse(pred_f, target_f, t_norm))
        return loss, metrics

    def build_dataset(self):
        d = self.data_cfg
        if d.data_name == "wan_framepack_fake":
            from octworld.datasets import WanFramePackFakeDataset
            return WanFramePackFakeDataset(
                gen_frames=d.gen_frames,
            )
        if d.data_name == "wan_framepack_dl3dv":
            from octworld.datasets import WanFramePackDataset
            return WanFramePackDataset(
                target_path=d.target_path,
                metadata_path=d.metadata_path,
                gen_frames=d.gen_frames,
                load_raymap=d.load_raymap,
            )
        raise ValueError(f"Unknown data_name: {d.data_name!r}")
