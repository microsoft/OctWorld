"""Pydantic config schema for OctWorld Wan FramePack training.

Parsed from ``train_base.yaml`` (after OmegaConf env-resolution) by
``train.py``. Holds the distribution-agnostic loop config under ``basic``
(:class:`~octworld.training.basic_trainer.BaseTrainConfig`) alongside the
model-/data-/experiment-specific sections owned by the concrete trainer.
"""
from __future__ import annotations

from pydantic import BaseModel

from octworld.training.basic_trainer import BaseTrainConfig


class ExperimentConfig(BaseModel):
    """Metadata used to derive the experiment name."""

    data_tag: str
    # Optional suffix appended to the derived experiment name; empty means none.
    name_suffix: str = ""


class ModelConfig(BaseModel):
    """Wan FramePack model selection + flow-matching hyper-parameters."""

    variant: str = "wan14b_i2v"  # wan14b_i2v / tiny
    framepack_type: str
    camera_in_dim: int = 6
    use_gradient_checkpointing: bool = True
    # Selective activation checkpointing (SAC): with manual AC on, recompute
    # only the cheap per-block ops and keep flash-attention/matmul outputs
    # saved. Ignored when ``compile_blocks`` is set.
    use_selective_checkpointing: bool = False
    # Regionally ``torch.compile`` each DiT transformer block (opt-in). Fuses
    # the per-block LayerNorm + fp32 AdaLN modulation + residual chain into
    # fewer kernels. Applied before FSDP sharding. When enabled, the compile
    # Memory Budget API drives recompute and manual AC is disabled.
    compile_blocks: bool = False
    # Compile-only Memory Budget API budget in [0, 1] (0 = plain activation
    # checkpointing, 1 = default torch.compile). Only used with compile_blocks.
    activation_memory_budget: float = 0.5
    shift: float = 3.0  # flow-shift
    # History-condition noise augmentation. When enabled, each sample draws a
    # random bucket level in ``[0, NOISE_AUG_NUM_LEVELS)`` (see wan_pipeline) and
    # the history (all frames except the anchor/start) plus the rendered latents
    # are noised at the matching flow-match level. Disable to train without it.
    noise_aug: bool = True
    # Classifier-free guidance training dropout: probability of replacing a
    # sample's conditioning (text + camera raymap) with the unconditional
    # branch (negative text embed + zeroed raymap).
    random_drop_ratio: float = 0.1
    # Path to the negative-prompt embedding (``negative_prompt.pt``). Resolved
    # relative to ``basic.exp_dir`` when not absolute.
    neg_text_embed_path: str = "negative_prompt.pt"


class DatasetConfig(BaseModel):
    """Training dataloader config. See ``train_base.yaml`` for example."""

    # Which dataset implementation to instantiate:
    #   wan_framepack_dl3dv -> WanFramePackDataset (reads real data from disk)
    #   wan_framepack_fake  -> WanFramePackFakeDataset (random tensors)
    data_name: str = "wan_framepack_dl3dv"
    target_path: str
    metadata_path: str = "train_metadata.csv"
    gen_frames: int = 20
    load_raymap: bool = True
    # Rendered-warp variant selection has been removed: training always uses
    # BOTH ``online_kf5`` and ``offline_kf5`` together (WanFramePackDataset
    # yields one sample per (entry, variant) pair, doubling its length).


class TrainConfig(BaseModel):
    """Top-level Wan training config.

    ``basic`` carries the distribution-agnostic training-loop sections consumed
    by :class:`~octworld.training.basic_trainer.BasicTrainer`; ``experiment`` /
    ``model`` / ``data`` are the Wan-specific sections owned by the concrete
    :class:`~octworld.training.wan.octworld_wan_trainer.OctWorldWanTrainer`.
    """

    basic: BaseTrainConfig
    experiment: ExperimentConfig
    model: ModelConfig
    data: DatasetConfig
