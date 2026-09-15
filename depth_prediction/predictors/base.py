"""Base class and output dataclass for unified depth predictors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DepthPredictorOutput:
    """Unified output format for all depth predictors.

    All tensors use the convention:
        - depth:      (N, H, W)  — depth values (metric when available, otherwise scale-invariant).
        - intrinsics: (N, 3, 3)  — normalized camera intrinsic matrices (fx, fy, cx, cy in [0, 1]).
        - extrinsics: (N, 4, 4)  — world-to-camera transformation matrices.
                                   Identity when pose estimation is not supported (e.g. MoGe).
        - depth_mask: (N, H, W)  — boolean mask, True = valid depth pixel.
        - confidence: (N, H, W)  — per-pixel confidence scores (optional, None if unavailable).

    N is the number of input frames.
    """

    depth: torch.Tensor
    intrinsics: torch.Tensor
    extrinsics: torch.Tensor
    depth_mask: torch.Tensor
    confidence: Optional[torch.Tensor] = None

    # ---- helpers ----
    def to(self, device: torch.device | str) -> "DepthPredictorOutput":
        return DepthPredictorOutput(
            depth=self.depth.to(device),
            intrinsics=self.intrinsics.to(device),
            extrinsics=self.extrinsics.to(device),
            depth_mask=self.depth_mask.to(device),
            confidence=self.confidence.to(device) if self.confidence is not None else None,
        )


class BaseDepthPredictor(ABC):
    """Abstract base class for depth predictors.

    Subclasses must implement ``predict``.

    Convention for *images* input
    ----------------------------
    ``torch.Tensor`` of shape **(N, 3, H, W)**, dtype ``float32``, value
    range **[0, 1]**, **RGB** channel order.

    Each predictor is responsible for:
      1. Resizing / padding images to the resolution required by its backbone.
      2. Applying backbone-specific normalization (e.g. ImageNet mean/std).
      3. Resizing depth outputs back to the **original** input resolution.
    """

    @abstractmethod
    def predict(
        self,
        images: torch.Tensor,
    ) -> DepthPredictorOutput:
        """Run depth prediction.

        Args:
            images: (N, 3, H, W) tensor in [0, 1], RGB.

        Returns:
            DepthPredictorOutput
        """
        ...
