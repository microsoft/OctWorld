"""Shared image loading and fixed MoGe v1 geometry prediction."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from depth_prediction.predictors import create_predictor  # noqa: E402
from utils.io import center_crop  # noqa: E402


# Same Ruicheng/moge-vitl v1 weights, prepared by scripts/download_models.py.
MOGE_V1_PRETRAINED = str(PROJECT_ROOT / "model_weights" / "moge" / "model.pt")
MOGE_V1_RESOLUTION_LEVEL = 9


@dataclass(frozen=True)
class ImageGeometry:
    image: np.ndarray
    depth: np.ndarray
    depth_mask: np.ndarray
    intrinsics: np.ndarray


def load_image(path: str | Path, resolution: tuple[int, int]) -> np.ndarray:
    """Load RGB and resize by aspect-preserving center crop."""

    with Image.open(path) as image:
        prepared = center_crop(image.convert("RGB"), resolution)
        return np.asarray(prepared, dtype=np.uint8)


def predict_moge_v1(image: np.ndarray, device: str = "cuda") -> ImageGeometry:
    """Predict scale-invariant MoGe v1 depth, mask, and normalized intrinsics."""

    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be an HxWx3 uint8 RGB array")
    if not Path(MOGE_V1_PRETRAINED).is_file():
        raise FileNotFoundError(
            f"MoGe v1 weights not found at {MOGE_V1_PRETRAINED}. "
            "Run: python scripts/download_models.py --models moge"
        )
    predictor = create_predictor(
        "moge",
        device=device,
        version="v1",
        pretrained=MOGE_V1_PRETRAINED,
        resolution_level=MOGE_V1_RESOLUTION_LEVEL,
        apply_mask=True,
    )
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1)
    prediction = predictor.predict(tensor.float().div_(255.0).unsqueeze(0).to(device))
    return ImageGeometry(
        image=image,
        depth=prediction.depth[0].detach().cpu().numpy().astype(np.float32),
        depth_mask=prediction.depth_mask[0].detach().cpu().numpy().astype(bool),
        intrinsics=prediction.intrinsics[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
    )