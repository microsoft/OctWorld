"""Motion-gated keyframe selection via torchvision RAFT-Large optical flow.

A small, stateful selector that decides which frames of a video stream to keep
as *keyframes*, based on how much they have moved (mean optical-flow magnitude)
relative to a running *reference* keyframe. It mirrors the ``raft`` backend in
``example/keyframe_select/example_keyframe.py`` but adds three behaviours
required by the Wan AR world-generation rollout:

  1. The **last frame of every** ``select`` **block is always kept** (so each
     chunk's last frame becomes a keyframe).
  2. A **maximum interval**: if more than ``max_interval`` frames have passed
     since the last keyframe, the current frame is forced to be a keyframe.
  3. An **adjustable motion threshold**.

The selector is **stateful across** ``select`` **calls** (the reference frame
and the since-last-keyframe counter carry over), so feeding it consecutive
chunks reproduces a single continuous stream. Call :meth:`reset` to start a new
stream.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch


_DEFAULT_WEIGHT_PATH = (
    Path(__file__).resolve().parents[1]
    / "model_weights" / "raft" / "raft_large_C_T_SKHT_V2-ff5fadd5.pth"
)


class RaftKeyframeSelector:
    """Stateful motion-gated keyframe selector backed by RAFT-Large.

    Args:
        threshold: mean RAFT flow magnitude (full-res pixels) above which a
            frame becomes a keyframe.
        max_interval: force a keyframe if this many frames have passed since the
            last keyframe without one being selected (default 8).
        device: torch device for the RAFT model.
        weights_path: local weights file, overriding RAFT_WEIGHTS_PATH. Defaults
            to the release's model_weights/raft/ checkpoint when neither is set.
    """

    def __init__(
        self,
        threshold: float = 5.0,
        max_interval: int = 8,
        device: str = "cuda",
        weights_path: str | Path | None = None,
    ):
        if threshold <= 0:
            raise ValueError(f"threshold must be > 0, got {threshold}")
        if max_interval < 1:
            raise ValueError(f"max_interval must be >= 1, got {max_interval}")

        self.threshold = float(threshold)
        self.max_interval = int(max_interval)
        self.device = device
        self.weights_path = Path(
            weights_path if weights_path is not None
            else os.environ.get("RAFT_WEIGHTS_PATH", str(_DEFAULT_WEIGHT_PATH))
        ).expanduser()

        # Lazily built on first use so importing this module is cheap and does
        # not load local weights until a selector actually runs.
        self._model = None
        self._transforms = None

        # Running state (carried across ``select`` calls until ``reset``).
        # Stored as an already-preprocessed (1, 3, H, W) RAFT input on
        # ``self.device`` so the reference is never re-preprocessed per frame.
        self._ref: Optional[torch.Tensor] = None
        self._since_last: int = 0

    # ------------------------------------------------------------------
    # Model / preprocessing
    # ------------------------------------------------------------------

    def _ensure_model(self):
        if self._model is not None:
            return
        if not self.weights_path.is_file():
            raise FileNotFoundError(
                f"RAFT weights not found at: {self.weights_path}. "
                "Provide weights_path or set RAFT_WEIGHTS_PATH to a local checkpoint."
            )
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        model = raft_large(weights=None, progress=False)
        state_dict = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.requires_grad_(False)
        self._transforms = Raft_Large_Weights.DEFAULT.transforms()
        self._model = model.to(self.device)

    def _prep(self, frame_hwc: torch.Tensor) -> torch.Tensor:
        """Preprocess one ``(H, W, 3)`` frame into a normalised ``(1, 3, H, W)``
        RAFT input on ``self.device``.

        All work runs on GPU. This replaces the torchvision ``OpticalFlow``
        transform, which ran on CPU at full resolution for every frame and was
        the bottleneck under concurrent CPU load (video decode in the load
        workers). The normalisation matches that transform exactly
        (uint8/255 -> [0, 1] -> [-1, 1]), so selection results are unchanged.

        Accepts uint8 frames or float frames in ``[0, 1]``.
        """
        img = frame_hwc.to(self.device, non_blocking=True)
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        else:
            img = img.float().clamp(0.0, 1.0)
        img = img.permute(2, 0, 1).contiguous()  # (3, H, W)
        img = img * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        img = img[None]  # (1, 3, H, W)
        # RAFT requires H and W divisible by 8. Force it here by resizing down to
        # the nearest multiple of 8, so callers can pass native resolutions that
        # are not multiples of 8 (e.g. the DA3 depth resolution 294x504 -> RAFT
        # runs at 288x504). Flow magnitudes are then in this resized-pixel scale,
        # which the motion ``threshold`` is tuned against.
        _, _, h, w = img.shape
        h8, w8 = (h // 8) * 8, (w // 8) * 8
        if (h8, w8) != (h, w):
            img = torch.nn.functional.interpolate(
                img, size=(h8, w8), mode="bilinear", align_corners=False)
        return img  # (1, 3, h8, w8)

    @torch.no_grad()
    def _motion(self, cur: torch.Tensor) -> float:
        """Mean RAFT flow magnitude between the reference and ``cur``.

        Both ``self._ref`` and ``cur`` are already-preprocessed ``(1, 3, H, W)``
        tensors on ``self.device``.
        """
        flow = self._model(self._ref, cur)[-1]  # last (most refined) (1, 2, H, W)
        mag = torch.sqrt(flow[:, 0] ** 2 + flow[:, 1] ** 2)
        return mag.mean().item()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Clear the running state to start a fresh stream."""
        self._ref = None
        self._since_last = 0

    @torch.no_grad()
    def select(self, frames: torch.Tensor) -> torch.Tensor:
        """Select keyframe indices within ``frames``.

        Args:
            frames: ``(N, H, W, 3)`` RGB video block (float in ``[0, 1]`` or
                uint8). State carries over from previous calls, so the first
                frame is only auto-selected when there is no reference yet
                (stream start / after :meth:`reset`).

        Returns:
            A 1-D ``LongTensor`` of selected indices *local to this block*
            (sorted, always including the last index ``N - 1``).
        """
        n = int(frames.shape[0])
        if n == 0:
            return torch.empty(0, dtype=torch.long)

        self._ensure_model()
        # Move the whole block to GPU once; all per-frame preprocessing then
        # runs on GPU (see ``_prep``), so no full-resolution CPU work happens in
        # the per-frame loop. This is the fix for the per-frame CPU cost that
        # ballooned under concurrent video-decode load on the cluster.
        frames = frames.to(self.device, non_blocking=True)
        selected: list[int] = []

        for i in range(n):
            cur = self._prep(frames[i])
            if self._ref is None:
                # Stream start: this frame seeds the reference and is kept.
                self._ref = cur
                self._since_last = 0
                selected.append(i)
                continue

            keep = (
                self._motion(cur) > self.threshold
                or self._since_last >= self.max_interval
            )
            if keep:
                self._ref = cur
                self._since_last = 0
                selected.append(i)
            else:
                self._since_last += 1

        # Requirement 1: the last frame of every block is always a keyframe.
        if selected[-1] != n - 1:
            self._ref = self._prep(frames[n - 1])
            self._since_last = 0
            selected.append(n - 1)

        return torch.tensor(selected, dtype=torch.long)
