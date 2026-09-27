"""MoGe single-frame depth predictor backed by the vendored source tree."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from depth_prediction.predictors.base import BaseDepthPredictor, DepthPredictorOutput


MOGE_SOURCE = Path(__file__).resolve().parents[1] / "thirdparty" / "MoGe"


class MoGePredictor(BaseDepthPredictor):
    """Predict monocular depth and normalized intrinsics with MoGe."""

    CONFIDENCE_THRESHOLD = 0.5

    def __init__(
        self,
        pretrained: str = "Ruicheng/moge-2-vitl",
        version: str = "v2",
        resolution_level: int = 9,
        apply_mask: bool = False,
        device: str = "cuda",
    ):
        if not (MOGE_SOURCE / "moge" / "__init__.py").is_file():
            raise ImportError(
                "MoGe submodule is missing; run: "
                "git submodule update --init depth_prediction/thirdparty/MoGe"
            )
        source = str(MOGE_SOURCE)
        if source not in sys.path:
            sys.path.insert(0, source)

        from moge.model import import_model_class_by_version

        model_class = import_model_class_by_version(version)
        self.model = model_class.from_pretrained(pretrained).to(device).eval()
        self.device = torch.device(device)
        self.resolution_level = int(resolution_level)
        self.apply_mask = bool(apply_mask)
        self.version = version

    @torch.inference_mode()
    def predict(self, images: torch.Tensor) -> DepthPredictorOutput:
        images = images.to(self.device)
        frame_count, _, height, width = images.shape
        depths = []
        intrinsics = []
        masks = []

        for image in images:
            output = self.model.infer(
                image,
                resolution_level=self.resolution_level,
                apply_mask=self.apply_mask,
            )
            depth = output["depth"]
            intrinsic = output["intrinsics"]
            mask = output.get("mask")
            if depth.shape[-2:] != (height, width):
                depth = F.interpolate(
                    depth[None, None], size=(height, width), mode="nearest"
                )[0, 0]
            if mask is None:
                depth_mask = torch.ones(
                    height, width, dtype=torch.bool, device=self.device
                )
            else:
                if mask.shape[-2:] != (height, width):
                    mask = F.interpolate(
                        mask.float()[None, None],
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]
                depth_mask = mask > self.CONFIDENCE_THRESHOLD
            depth_mask &= (depth > 0) & torch.isfinite(depth)
            depths.append(depth)
            intrinsics.append(intrinsic)
            masks.append(depth_mask)

        return DepthPredictorOutput(
            depth=torch.stack(depths),
            intrinsics=torch.stack(intrinsics),
            extrinsics=torch.eye(4, device=self.device)
            .unsqueeze(0)
            .expand(frame_count, -1, -1),
            depth_mask=torch.stack(masks),
        )