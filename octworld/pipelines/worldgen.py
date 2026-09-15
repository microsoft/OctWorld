"""WorldGenPipeline — Generic base class for AR world generation.

Given a single image, prompt, and camera trajectory, performs autoregressive
rollout in chunks with 3D mapper management.  Model-specific logic (FramePack,
Wan, depth normalization, etc.) is deferred to subclasses via abstract methods.
"""

import torch
import numpy as np
from abc import abstractmethod
from typing import Optional, Literal, Dict, Any, Tuple, List
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from utils.io import (
    post_process_video_for_visualize, colorize_depth,
    write_video, write_video_depth, write_colored_depth_video,
)


class WorldGenPipeline:
    """Abstract base class for AR world generation with 3D mapping.

    Subclasses must implement:
        - ``world_gen_step``   — run a single diffusion chunk
        - ``init_first_frame`` — encode the initial image/depth into internal state
    """

    def __init__(
        self,
        mapper_mode: Literal["tsdf", "octmap"] = "tsdf",
        mapper_resolution: Tuple[int, int] = (480, 832),
        mapper_device: str = "cuda",
        chunk_size: int = 16,
        camera_scale: float = 20.0,
        mapper_init_kwargs: Optional[Dict[str, Any]] = None,
        mapper_update_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            mapper_mode:  Which 3D mapper backend to use ("tsdf" or "octmap").
            mapper_resolution: (H, W) used by the mapper for rendering.
            mapper_device:  Device for the mapper.
            chunk_size:  Number of pixel frames per AR step.
            camera_scale: Scaling factor applied to camera translations when computing plucker rays.
            num_inference_steps: Diffusion steps per chunk.
        """
        self.mapper_mode = mapper_mode
        self.mapper_resolution = mapper_resolution
        self.mapper_device = mapper_device
        # Optional integration resolution, decoupled from the render/buffer
        # resolution (``mapper_resolution``). When set (e.g. the DA3 native
        # 294x504), the mapper integrates depth/frames at this resolution while
        # still rendering novel views at ``mapper_resolution``. ``None`` keeps
        # the legacy behaviour (integrate == render == mapper_resolution).
        # Set externally (e.g. by batch_world_gen) before the first ``__call__``
        # so it is picked up by the lazy ``_init_mapper``; this keeps frozen
        # pipelines (v1) unchanged.
        self.integration_resolution: Optional[Tuple[int, int]] = None
        self.chunk_size = chunk_size
        self.camera_scale = camera_scale
        self.mapper_init_kwargs = dict(mapper_init_kwargs or {})
        self.mapper_update_kwargs = dict(mapper_update_kwargs or {})

        # Mapper — created lazily in _init_mapper()
        self._mapper = None

        # AR state: accumulated pixel-space buffers
        self.all_frames: Optional[torch.Tensor] = None   # (N, H, W, 3) float [0,1]
        self.all_depths: Optional[torch.Tensor] = None   # (N, H, W) float, metric
        self.all_masks:  Optional[torch.Tensor] = None    # (N, H, W) bool

    # ------------------------------------------------------------------
    # Mapper management
    # ------------------------------------------------------------------

    def _init_mapper(self):
        """Lazily instantiate the 3D mapper.

        When ``self.integration_resolution`` is set, the mapper integrates at
        that resolution and renders at ``self.mapper_resolution`` (decoupled via
        the mapper's ``render_resolution``). Otherwise both default to
        ``mapper_resolution``.
        """
        integ_res = self.integration_resolution
        map_res = tuple(integ_res) if integ_res is not None else self.mapper_resolution
        render_res = self.mapper_resolution if integ_res is not None else None
        if self.mapper_mode == "tsdf":
            from spatial_memory.mapper.tsdf.tsdf_mapper import TSDFMapper
            # TSDFMapper does not support a decoupled render_resolution.
            if integ_res is not None:
                raise ValueError(
                    "integration_resolution is only supported for mapper_mode="
                    "'octmap' (TSDFMapper has no decoupled render_resolution).")
            self._mapper = TSDFMapper(
                resolution=map_res,
                device=self.mapper_device,
                **self.mapper_init_kwargs,
            )
        elif self.mapper_mode == "octmap":
            from spatial_memory.mapper.octmap.octmap_mapper import OctMapMapper
            self._mapper = OctMapMapper(
                resolution=map_res,
                render_resolution=render_res,
                device=self.mapper_device,
                **self.mapper_init_kwargs,
            )
        else:
            raise ValueError(f"Unknown mapper_mode: {self.mapper_mode}")

    def _update_map(
        self,
        frames: torch.Tensor,
        depths: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        depth_mask: Optional[torch.Tensor] = None,
    ):
        """Extend the 3D map with new frames/depths.

        Args:
            frames:     (N, H, W, 3) float [0,1]
            depths:     (N, H, W) float, original-scale
            extrinsics: (N, 4, 4)
            intrinsics: (N, 3, 3)
            depth_mask: (N, H, W) bool, optional
        """
        if self._mapper is None:
            self._init_mapper()
        self._mapper.extend_frames(
            frames=frames,
            depths=depths,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            depth_mask=depth_mask,
            **self.mapper_update_kwargs,
        )

    def _render_warp(
        self,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Render warped observations from the 3D map.

        Returns dict with:
            color: (N, 3, H, W) float [0,1]
            depth: (N, 1, H, W) float, original-scale
        """
        if self._mapper is None:
            raise RuntimeError("3D mapper not initialized. Call _update_map first.")
        return self._mapper.render(extrinsics, intrinsics)

    # ------------------------------------------------------------------
    # AR state management
    # ------------------------------------------------------------------

    def _append_frames(
        self,
        frames: torch.Tensor,
        depths: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ):
        """Append new pixel-space results to accumulation buffers.

        Args:
            frames: (N, H, W, 3) float [0,1]
            depths: (N, H, W) float
            masks:  (N, H, W) bool, optional
        """
        if self.all_frames is None:
            self.all_frames = frames
            self.all_depths = depths
            self.all_masks = masks
        else:
            self.all_frames = torch.cat([self.all_frames, frames], dim=0)
            self.all_depths = torch.cat([self.all_depths, depths], dim=0)
            if masks is not None and self.all_masks is not None:
                self.all_masks = torch.cat([self.all_masks, masks], dim=0)

    # ------------------------------------------------------------------
    # Shared rollout helpers (reused by subclass __call__ overrides)
    # ------------------------------------------------------------------

    def _seed_first_frame(
        self,
        image_np: torch.Tensor,
        depth: torch.Tensor,
        depth_mask: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ):
        """Seed AR buffers and the 3D map with the first frame's RGBD.

        Args:
            image_np: (H, W, 3) float [0,1] first-frame RGB.
            depth:    (H, W) float metric depth.
            depth_mask: (H, W) bool validity.
            extrinsics: (>=1, 4, 4) — only frame 0 (``[:1]``) is used.
            intrinsics: (>=1, 3, 3) — only frame 0 (``[:1]``) is used.
        """
        self._append_frames(
            image_np.unsqueeze(0), depth.unsqueeze(0), depth_mask.unsqueeze(0)
        )
        self._update_map(
            frames=image_np.unsqueeze(0),
            depths=depth.unsqueeze(0),
            extrinsics=extrinsics[:1],
            intrinsics=intrinsics[:1],
            depth_mask=depth_mask.unsqueeze(0),
        )

    def _render_chunk_warp(
        self,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Render warp color/depth/mask from the map for a camera segment.

        Returns dict with ``color`` (N,3,H,W), ``depth`` (N,1,H,W),
        ``mask`` (N,1,H,W) float.
        """
        render_dict = self._render_warp(extrinsics, intrinsics)
        warp_color = render_dict["color"]
        warp_depth = render_dict["depth"]
        warp_mask = (warp_depth > 0).float()
        return {"color": warp_color, "depth": warp_depth, "mask": warp_mask}

    def _save_warp_vis(
        self, save_dir, tag: str,
        warp_color: torch.Tensor, warp_depth: torch.Tensor, warp_mask: torch.Tensor,
        fps: int = 8,
    ):
        """Write a warp (color | colorized depth) visualization video."""
        if save_dir is None:
            return
        warp_color_vis = post_process_video_for_visualize(
            warp_color, layout='fchw', value_range=(0, 1))
        warp_depth_vis = colorize_depth(
            warp_depth.squeeze(1).cpu().numpy(),
            mask=warp_mask.squeeze(1).cpu().numpy() > 0.5)
        warp_vis = np.concatenate([warp_color_vis, warp_depth_vis], axis=2)
        write_video(str(save_dir / f"{tag}_warp.mp4"), warp_vis, fps=fps)

    def _save_accumulated_vis(self, save_dir, tag: str, fps: int = 8):
        """Write the accumulated (color | colorized depth) visualization."""
        if save_dir is None:
            return
        acc_color_vis = post_process_video_for_visualize(
            self.all_frames, layout='fhw3', value_range=(0, 1))
        acc_mask = self.all_masks.cpu().numpy() if self.all_masks is not None else None
        acc_depth_vis = colorize_depth(
            self.all_depths.cpu().numpy() if isinstance(self.all_depths, torch.Tensor) else self.all_depths,
            mask=acc_mask)
        acc_vis = np.concatenate([acc_color_vis, acc_depth_vis], axis=2)
        write_video(str(save_dir / f"{tag}_accumulated.mp4"), acc_vis, fps=fps)

    # ------------------------------------------------------------------
    # Abstract methods — to be implemented by model-specific subclass
    # ------------------------------------------------------------------

    @abstractmethod
    def init_first_frame(
        self,
        image: Image.Image,
        depth: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        prompt: str,
    ):
        """Initialize pipeline state from the first frame.

        Should encode image/depth into internal latent buffers, set up
        prompts, and prepare anything needed for generation.
        """
        ...

    @abstractmethod
    def world_gen_step(
        self,
        warp_color: torch.Tensor,
        warp_depth: torch.Tensor,
        warp_mask: torch.Tensor,
        chunk_prompt: str,
        
    ) -> Dict[str, torch.Tensor]:
        """Generate one AR chunk given warped conditions.

        Args:
            warp_color: (N, 3, H, W) float [0,1] — rendered color from 3D map.
            warp_depth: (N, 1, H, W) float — rendered depth from 3D map (metric).
            warp_mask:  (N, 1, H, W) float — validity mask from rendering.

        Returns:
            dict with:
                frames: (N, H, W, 3) float [0,1] — generated RGB frames.
                depths: (N, H, W) float — generated depth (original scale, not renormalized).
        """
        ...

    # ------------------------------------------------------------------
    # Main rollout
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def __call__(
        self,
        image: Image.Image,
        depth: torch.Tensor,
        depth_mask: Optional[torch.Tensor],
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        prompt: str = "",
        save_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run full AR world generation.

        Args:
            image:       PIL image of the first frame.
            depth:       (H, W) tensor — initial depth map (metric).
            depth_mask:  (H, W) tensor or None — initial depth validity mask.
            extrinsics:  (T, 4, 4) — full camera trajectory (T frames).
            intrinsics:  (T, 3, 3) — full camera intrinsics.
            prompt:      Text prompt (str or list[str] for per-chunk).
            num_inference_steps: Diffusion steps per chunk.
            save_dir:    Optional directory to save intermediate results.
            verbose:     Print progress info.

        Returns:
            dict with:
                frames: (T, H, W, 3) uint8 — all generated frames.
                depths: (T, H, W) float — all generated depths (metric).
        """
        num_frames = extrinsics.shape[0]

        # Pad trajectory to be evenly divisible by chunk_size
        if (num_frames - 1) % self.chunk_size != 0:
            pad = self.chunk_size - (num_frames - 1) % self.chunk_size
            pad_idx = list(range(num_frames)) + [num_frames - 1] * pad
            extrinsics = extrinsics[pad_idx]
            intrinsics = intrinsics[pad_idx]
            num_frames = extrinsics.shape[0]

        num_chunks = (num_frames - 1) // self.chunk_size
        if verbose:
            print(f"[WorldGen] {num_frames} frames, {num_chunks} chunks "
                  f"(chunk_size={self.chunk_size})")

        # Handle per-chunk prompts
        if isinstance(prompt, list):
            if len(prompt) != num_chunks:
                prompt = [prompt[i * len(prompt) // num_chunks] for i in range(num_chunks)]
            chunk_prompts = prompt
        else:
            chunk_prompts = [prompt] * num_chunks

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        # ── Step 0: Initialize from first frame ─────────────────────
        image_np = torch.tensor(
            __import__("numpy").array(image), dtype=torch.float32
        ) / 255.0  # (H, W, 3)
        if depth_mask is None:
            depth_mask = depth > 0

        self.init_first_frame(
            image=image,
            depth=depth,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            prompt=chunk_prompts[0],
            depth_mask=depth_mask,
        )

        # Seed the 3D map with the first frame
        self._seed_first_frame(
            image_np=image_np,
            depth=depth,
            depth_mask=depth_mask,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
        )

        # ── AR rollout ───────────────────────────────────────────────
        chunk_iter = range(num_chunks)
        if verbose:
            chunk_iter = tqdm(chunk_iter, desc="WorldGen chunks")

        for chunk_idx in chunk_iter:
            start = self.all_frames.shape[0]
            end = start + self.chunk_size
            chunk_extrs = extrinsics[start:end]
            chunk_intrs = intrinsics[start:end]

            # 1. Render warp from current 3D map
            warp = self._render_chunk_warp(chunk_extrs, chunk_intrs)
            warp_color = warp["color"]  # (N, 3, H, W)
            warp_depth = warp["depth"]  # (N, 1, H, W)
            warp_mask = warp["mask"]    # (N, 1, H, W)

            # Save per-chunk warp visualization
            if save_dir is not None:
                self._save_warp_vis(
                    save_dir, f"chunk_{chunk_idx:02d}",
                    warp_color, warp_depth, warp_mask)

            # 2. Generate new chunk
            gen_result = self.world_gen_step(
                warp_color=warp_color,
                warp_depth=warp_depth,
                warp_mask=warp_mask,
                chunk_prompt=chunk_prompts[chunk_idx],
            )
            gen_frames = gen_result["frames"]  # (N, H, W, 3)
            gen_depths = gen_result["depths"]  # (N, H, W)

            # 3. Update 3D map with generated frames
            gen_mask = gen_result.get("depth_mask", gen_depths > 0)
            self._update_map(
                frames=gen_frames,
                depths=gen_depths,
                extrinsics=chunk_extrs,
                intrinsics=chunk_intrs,
                depth_mask=gen_mask,
            )

            # 4. Accumulate (skip if world_gen_step already did soft_append)
            if not gen_result.get("already_appended", False):
                self._append_frames(gen_frames, gen_depths, gen_mask)

            # Save per-chunk accumulated visualization
            if save_dir is not None:
                self._save_accumulated_vis(save_dir, f"chunk_{chunk_idx:02d}")

            if verbose:
                print(f"  Chunk {chunk_idx}: generated {gen_frames.shape[0]} frames, "
                      f"total {self.all_frames.shape[0]}")

        # ── Save final output ────────────────────────────────────────
        all_frames_uint8 = (self.all_frames.clamp(0, 1) * 255).to(torch.uint8)
        if save_dir is not None:
            final_color_vis = post_process_video_for_visualize(
                self.all_frames, layout='fhw3', value_range=(0, 1))
            write_video(str(save_dir / "output.mp4"), final_color_vis, fps=8)

            final_depths_np = self.all_depths.cpu().numpy() if isinstance(self.all_depths, torch.Tensor) else self.all_depths
            final_mask_np = self.all_masks.cpu().numpy() if self.all_masks is not None else None
            write_colored_depth_video(
                str(save_dir / "output_depth.mp4"),
                final_depths_np, mask=final_mask_np, fps=8)
            write_video_depth(
                str(save_dir / "output_depth.zip"),
                final_depths_np, mask=final_mask_np)

        return {
            "frames": all_frames_uint8,
            "depths": self.all_depths,
        }
