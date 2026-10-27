"""Load a fine-tuned Wan DiT and camera adapter without training imports.

Original Wan transformer weights are not used. VAE / T5 / CLIP are loaded
separately via ``WanMultiAdapterPipeline.load_inference_modules``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..pipelines.wan.wan_pipeline import WanMultiAdapterPipeline

__all__ = ["resolve_exp_root", "load_wan_checkpoint"]


def resolve_exp_root(ckpt_path: str | Path) -> Path:
    """Find the nearest experiment directory containing ``pipe_config.json``.

    Accept a checkpoint file or DCP directory. Missing checkpoints and configs
    raise ``FileNotFoundError`` before any model is constructed or loaded.
    """
    checkpoint = Path(ckpt_path).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    start = checkpoint if checkpoint.is_dir() else checkpoint.parent
    for parent in (start, *start.parents):
        if (parent / "pipe_config.json").is_file():
            return parent
    raise FileNotFoundError(
        f"could not find pipe_config.json at or above {start} "
        f"for checkpoint {checkpoint}"
    )


def _build_nested_config(pipe_config: dict) -> dict:
    """Normalize the manifest using the trainer's constructor-config schema."""
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


def load_wan_checkpoint(
    ckpt_path: str | Path,
    device: str | torch.device,
    dtype: torch.dtype,
) -> WanMultiAdapterPipeline:
    """Return an eval/frozen Wan pipeline loaded from a fine-tuned checkpoint.

    Supports a legacy flat model state dict (``torch.save``) or a DCP directory
    containing ``.metadata`` and shards. DCP uses PyTorch 2.8's
    ``_EmptyStateDictLoadPlanner(keys={"model"})``: the prefix is resolved using
    ``planner_data``, so optimizer state is never read or allocated. No process
    group, original Wan transformer, training module, or forward is needed.

    Construct parameters on meta, load strictly with ``assign=True``, regenerate
    non-persistent tensors, then move to the requested device/dtype.
    """
    checkpoint = Path(ckpt_path).expanduser().resolve()
    exp_root = resolve_exp_root(checkpoint)
    is_dcp_dir = checkpoint.is_dir()
    if is_dcp_dir:
        if not (checkpoint / ".metadata").is_file():
            raise FileNotFoundError(f"DCP metadata not found: {checkpoint / '.metadata'}")
    elif not checkpoint.is_file():
        raise ValueError(f"Checkpoint must be a file or DCP directory: {checkpoint}")

    with (exp_root / "pipe_config.json").open(encoding="utf-8") as f:
        config = _build_nested_config(json.load(f))

    # Keep the runtime import local: the Wan package also imports WanWorldPipeline,
    # whose compatibility wrappers import this helper module.
    from ..pipelines.wan.wan_pipeline import WanMultiAdapterPipeline

    pipe = WanMultiAdapterPipeline()
    with torch.device("meta"):
        pipe.build_training_model(
            config=config,
            use_gradient_checkpointing=False,
        )

    if is_dcp_dir:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.default_planner import (
            _EmptyStateDictLoadPlanner,
        )
        from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

        full: dict = {}
        _load_state_dict(
            full,
            storage_reader=dcp.FileSystemReader(str(checkpoint)),
            planner=_EmptyStateDictLoadPlanner(keys={"model"}),
            no_dist=True,
        )
        assert set(full) == {"model"}, (
            f"Expected only 'model' in DCP checkpoint {checkpoint}, "
            f"got top-level keys: {list(full)}"
        )
        state_dict = full["model"]
    else:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)

    pipe.load_state_dict(state_dict, strict=True, assign=True)
    pipe.materialize()
    return pipe.to(device=device, dtype=dtype).eval().requires_grad_(False)