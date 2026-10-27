import os
import time

import numpy as np
import torch
from utils.io import write_camera_trajectory
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
from spatial_memory.representations import Representation
from torch.nn import functional as F


class BaseMapper(ABC):
    """
    Base class for spatial memory mappers.
    Subclasses implement normalize_scene, map_frames, and get_representation.
    """

    def __init__(
        self,
        resolution: Tuple[int, int] = (480, 832),
        device: str = "cuda",
        render_resolution: Optional[Tuple[int, int]] = None,):

        self.resolution = resolution
        # Resolution used when rasterising novel views. Decoupled from the
        # integration resolution (``self.resolution``): the map is a
        # resolution-independent 3D structure and intrinsics are normalised, so
        # we can integrate cheap low-res depth yet render at full resolution.
        # Defaults to the integration resolution for backward compatibility.
        self.render_resolution = render_resolution if render_resolution is not None else resolution
        self.device = device

        # Integrated 3D representation
        self._rep: Optional[Representation] = None

        # Scene normalization parameters
        self._scale: float = 1.0
        self._transform: torch.Tensor = torch.zeros(3)

        # Time logs (seconds)
        self._map_frames_time_log: List[float] = []
        self._get_representation_time_log: List[float] = []
        self._last_map_frames_time: Optional[float] = None
        self._last_get_representation_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extend_frames(
        self,
        frames: torch.Tensor,
        depths: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        depth_mask: Optional[torch.Tensor] = None,
        apply_depth_edge: bool = True,
        skip_get_representation: bool = False,
        **kwargs,
    ):
        """
        Full integration pipeline:
        1. Store / extend raw data
        2. Align scale via normalize_scene(), updating _scale and _transform
        3. Transform data into normalised space and call map_frames()
        4. Obtain representation via get_representation()

        Args:
            frames:     (N, H, W, 3) RGB images, float [0, 1]
            depths:     (N, H, W) depth maps
            extrinsics: (N, 4, 4) camera extrinsics (world-to-camera)
            intrinsics: (N, 3, 3) camera intrinsics
            **kwargs:   Additional arguments forwarded to map_frames()
        """
        
        
        if depth_mask is not None:
            depths = torch.where(depth_mask, depths, torch.zeros_like(depths))
            
        if apply_depth_edge:
            depth_mask = depths > 0
            depths[~depth_mask] = 1000.0
            non_edge = ~self.depth_edge(depths, rtol=0.03)
            depth_mask &= non_edge
            depths[~depth_mask] = 0.0
            
        # --- resize to target resolution if needed ---
        if depths.shape[1:3] != tuple(self.resolution):
            depths = torch.nn.functional.interpolate(
                depths.unsqueeze(1), size=self.resolution, mode="nearest"
            ).squeeze(1)
        if frames.shape[1:3] != tuple(self.resolution):
            frames = torch.nn.functional.interpolate(
                frames.permute(0, 3, 1, 2), size=self.resolution, mode="bilinear", align_corners=False
            ).permute(0, 2, 3, 1)

        start_time = time.perf_counter()
        self.map_frames(frames, depths, extrinsics, intrinsics, **kwargs)
        elapsed = time.perf_counter() - start_time
        self._last_map_frames_time = elapsed
        self._map_frames_time_log.append(elapsed)
        
        if not skip_get_representation:
            start_time = time.perf_counter()
            self._rep = self.get_representation()
            elapsed = time.perf_counter() - start_time
            self._last_get_representation_time = elapsed
            self._get_representation_time_log.append(elapsed)

    def render(
        self,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> dict:
        """
        Render partial observations from the integrated representation.
        Extrinsics / intrinsics are in *original* world space and will be
        aligned to the normalised scene space before rendering.

        Args:
            extrinsics: (N, 4, 4) or (4, 4)
            intrinsics: (N, 3, 3) or (3, 3)

        Returns:
            dict with "color" and "depth" tensors.
            depth is in original (un-normalised) scale.
            color is [B, 3, H, W] in [0, 1].
            depth is [B, 1, H, W].
        """
        if self._rep is None:
            raise RuntimeError("No representation available. Call integrate() first.")

        norm_extrinsics = self._transform_extrinsics(extrinsics)
        render_dict = self._rep.render(norm_extrinsics, intrinsics, self.render_resolution)

        # Convert depth back to original scale
        render_dict["depth"] = render_dict["depth"] / self._scale
        return render_dict

    def save_scene(
        self,
        output_dir: str,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
    ) -> str:
        """
        Save the current scene (representation + camera parameters) to disk.

        Saves:
          - representation file (e.g. mesh.ply) via dump_to_file
          - scene_meta.npz containing:
              - scale: float, the normalization scale
              - transform: (3,) translation applied during normalization
          - cameras.npz (if extrinsics/intrinsics provided): standardized camera trajectory

        Args:
            output_dir: directory path to save the scene
            extrinsics: optional (N, 4, 4) original-space extrinsics to save
            intrinsics: optional (N, 3, 3) intrinsics to save

        Returns:
            output_dir path
        """

        if self._rep is None:
            raise RuntimeError("No representation available. Call extend_frames() first.")

        os.makedirs(output_dir, exist_ok=True)

        # Save representation
        rep_path = os.path.join(output_dir, "mesh.ply")
        self._rep.dump_to_file(rep_path)

        # Save normalization metadata
        scale_val = self._scale.cpu().numpy() if isinstance(self._scale, torch.Tensor) else np.array(self._scale)
        np.savez(
            os.path.join(output_dir, "scene_meta.npz"),
            scale=scale_val,
            transform=self._transform.cpu().numpy(),
        )

        # Save camera trajectory in standardized format
        if extrinsics is not None and intrinsics is not None:
            ext = extrinsics.cpu().float().numpy() if isinstance(extrinsics, torch.Tensor) else np.asarray(extrinsics)
            intr = intrinsics.cpu().float().numpy() if isinstance(intrinsics, torch.Tensor) else np.asarray(intrinsics)
            write_camera_trajectory(os.path.join(output_dir, "cameras.npz"), ext, intr)

        return output_dir

    # ------------------------------------------------------------------
    # Abstract methods (to be implemented by subclasses)
    # ------------------------------------------------------------------

    @abstractmethod
    def map_frames(
        self,
        frames: torch.Tensor,
        depths: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ):
        """
        Register frames into the internal map / integrator.
        All inputs are already in normalised scene space.
        """
        ...

    @abstractmethod
    def get_representation(self) -> Representation:
        """
        Build and return the 3D Representation from the current internal state.
        """
        ...

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_normalization(
        self,
        depths: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply _scale and _transform to depths and extrinsics.
        Same logic as DynOmniScene.integrate.
        """
        depths = depths * self._scale
        extrinsics = self._transform_extrinsics(extrinsics)
        return depths, extrinsics

    def _transform_extrinsics(self, extrinsics: torch.Tensor) -> torch.Tensor:
        """
        Transform extrinsics from original world space to normalised space
        using self._scale and self._transform.
        """
        c2w = torch.inverse(extrinsics.clone().float())
        c2w[..., :3, 3] = (c2w[..., :3, 3] - self._transform.to(c2w.device)) * self._scale
        return torch.inverse(c2w)

        
    @property
    def last_map_frames_time(self) -> Optional[float]:
        """Most recent map_frames() elapsed time (seconds)."""
        return self._last_map_frames_time

    @property
    def last_get_representation_time(self) -> Optional[float]:
        """Most recent get_representation() elapsed time (seconds)."""
        return self._last_get_representation_time

    @property
    def map_frames_time_log(self) -> List[float]:
        """All map_frames() elapsed times (seconds)."""
        return self._map_frames_time_log

    @property
    def get_representation_time_log(self) -> List[float]:
        """All get_representation() elapsed times (seconds)."""
        return self._get_representation_time_log

    @staticmethod
    def depth_edge(depth: torch.Tensor, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: torch.Tensor = None) -> torch.BoolTensor:
        """
        Compute the edge mask of a depth map. The edge is defined as the pixels whose neighbors have a large difference in depth.
        
        Args:
            depth (torch.Tensor): shape (..., height, width), linear depth map
            atol (float): absolute tolerance
            rtol (float): relative tolerance

        Returns:
            edge (torch.Tensor): shape (..., height, width) of dtype torch.bool
        """
        shape = depth.shape
        depth = depth.reshape(-1, 1, *shape[-2:])
        if mask is not None:
            mask = mask.reshape(-1, 1, *shape[-2:])

        if mask is None:
            diff = (F.max_pool2d(depth, kernel_size, stride=1, padding=kernel_size // 2) + F.max_pool2d(-depth, kernel_size, stride=1, padding=kernel_size // 2))
        else:
            diff = (F.max_pool2d(torch.where(mask, depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2) + F.max_pool2d(torch.where(mask, -depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2))

        edge = torch.zeros_like(depth, dtype=torch.bool)
        if atol is not None:
            edge |= diff > atol
        if rtol is not None:
            edge |= (diff / depth).nan_to_num_() > rtol
        edge = edge.reshape(*shape)
        return edge