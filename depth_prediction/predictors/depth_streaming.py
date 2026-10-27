"""Generic streaming depth estimation with SIM(3) alignment.

Works with any predictor that outputs DepthPredictorOutput (with confidence).
Generalizes da3_streaming_online.py to be predictor-agnostic.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

from depth_prediction.predictors.base import BaseDepthPredictor, DepthPredictorOutput

# Ensure da3_streaming utilities (loop_utils) are importable
_DA3_STREAMING_ROOT = str(
    Path(__file__).resolve().parent.parent / "thirdparty" / "Depth-Anything-3" / "da3_streaming"
)
if _DA3_STREAMING_ROOT not in sys.path:
    sys.path.insert(0, _DA3_STREAMING_ROOT)

from loop_utils.alignment_torch import (
    apply_sim3_direct_torch,
    depth_to_point_cloud_optimized_torch,
)
from loop_utils.sim3utils import (
    precompute_scale_chunks_with_depth,
    save_confident_pointcloud_batch,
    weighted_align_point_maps,
)


def _depth_to_point_cloud_vectorized(depth, intrinsics, extrinsics, device=None):
    """
    Compute world-space point clouds from depth, pixel-space intrinsics, and w2c extrinsics.

    Args:
        depth: [N, H, W] numpy array or torch tensor
        intrinsics: [N, 3, 3] pixel-space intrinsics
        extrinsics: [N, 3, 4] w2c (world-to-camera)

    Returns:
        point_cloud_world: [N, H, W, 3]
    """
    input_is_numpy = isinstance(depth, np.ndarray)
    if input_is_numpy:
        depth_tensor = torch.tensor(depth, dtype=torch.float32)
        intrinsics_tensor = torch.tensor(intrinsics, dtype=torch.float32)
        extrinsics_tensor = torch.tensor(extrinsics, dtype=torch.float32)
    else:
        depth_tensor = depth
        intrinsics_tensor = intrinsics
        extrinsics_tensor = extrinsics

    if device is not None:
        depth_tensor = depth_tensor.to(device)
        intrinsics_tensor = intrinsics_tensor.to(device)
        extrinsics_tensor = extrinsics_tensor.to(device)

    N, H, W = depth_tensor.shape
    dev = depth_tensor.device

    u = torch.arange(W, device=dev).float().view(1, 1, W, 1).expand(N, H, W, 1)
    v = torch.arange(H, device=dev).float().view(1, H, 1, 1).expand(N, H, W, 1)
    ones = torch.ones((N, H, W, 1), device=dev)
    pixel_coords = torch.cat([u, v, ones], dim=-1)

    intrinsics_inv = torch.inverse(intrinsics_tensor)
    camera_coords = torch.einsum("nij,nhwj->nhwi", intrinsics_inv, pixel_coords)
    camera_coords = camera_coords * depth_tensor.unsqueeze(-1)
    camera_coords_homo = torch.cat([camera_coords, ones], dim=-1)

    extrinsics_4x4 = torch.zeros(N, 4, 4, device=dev)
    extrinsics_4x4[:, :3, :4] = extrinsics_tensor
    extrinsics_4x4[:, 3, 3] = 1.0

    c2w = torch.inverse(extrinsics_4x4)
    world_coords_homo = torch.einsum("nij,nhwj->nhwi", c2w, camera_coords_homo)
    point_cloud_world = world_coords_homo[..., :3]

    if input_is_numpy:
        point_cloud_world = point_cloud_world.cpu().numpy()
    return point_cloud_world


class _InternalPredictions:
    """Mimics the raw prediction format needed by alignment routines.

    Stores pixel-space intrinsics (N,3,3), w2c extrinsics (N,3,4),
    depth (N,H,W), and confidence (N,H,W) as numpy arrays.
    """

    def __init__(self, depth: np.ndarray, conf: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray):
        self.depth = depth          # (N, H, W)
        self.conf = conf            # (N, H, W)
        self.intrinsics = intrinsics  # (N, 3, 3) pixel-space
        self.extrinsics = extrinsics  # (N, 3, 4) w2c


def _output_to_internal(output: DepthPredictorOutput) -> _InternalPredictions:
    """Inverse post-processing: convert DepthPredictorOutput to internal alignment format.

    Performs:
      - Intrinsics: normalized [0,1] → pixel-space
      - Extrinsics: w2c (N, 4, 4) → w2c (N, 3, 4)
      - Depth/confidence: torch → numpy

    Args:
        output: DepthPredictorOutput with normalized intrinsics and 4x4 extrinsics.

    Returns:
        _InternalPredictions with pixel-space intrinsics and 3x4 extrinsics.
    """
    depth = output.depth.cpu().numpy()  # (N, H, W)
    N, H, W = depth.shape

    # Confidence: use provided confidence or ones as fallback
    if output.confidence is not None:
        conf = output.confidence.cpu().numpy()
    else:
        conf = np.ones_like(depth)

    # Intrinsics: [0,1] → pixel-space
    intrinsics = output.intrinsics.cpu().numpy().copy()  # (N, 3, 3)
    intrinsics[:, 0, 0] *= W   # fx
    intrinsics[:, 0, 2] *= W   # cx
    intrinsics[:, 1, 1] *= H   # fy
    intrinsics[:, 1, 2] *= H   # cy

    # Extrinsics: (N, 4, 4) → (N, 3, 4)
    extrinsics = output.extrinsics.cpu().numpy()[:, :3, :]  # (N, 3, 4)

    return _InternalPredictions(depth=depth, conf=conf, intrinsics=intrinsics, extrinsics=extrinsics)


def _load_default_align_config() -> dict:
    """Load alignment config from DA3's base_config.yaml."""
    from depth_prediction.predictors.da3_predictor import load_da3_config
    return load_da3_config()


class DepthStreaming:
    """Generic online streaming depth estimation with SIM(3) alignment.

    Works with any predictor that produces DepthPredictorOutput (with confidence).
    The predictor outputs are in the unified coordinate system (normalized
    intrinsics, w2c 4x4), and this class handles the conversion to the internal
    format needed for SIM(3) alignment between chunks.

    Usage:
        predictor = create_predictor("pi3")
        streamer = DepthStreaming(predictor)

        # First chunk
        result_0 = streamer.add_chunk(images_0, overlap=0)

        # Subsequent chunks with overlap for alignment
        result_1 = streamer.add_chunk(images_1, overlap=30)
        result_2 = streamer.add_chunk(images_2, overlap=30)

        # Save RGBD results
        streamer.save(base_path="./output")
    """

    CONFIDENCE_THRESHOLD_COEF = 0.75

    def __init__(
        self,
        predictor: BaseDepthPredictor,
        align_config: dict | None = None,
    ):
        """
        Args:
            predictor: Any BaseDepthPredictor instance.
            align_config: Config dict controlling alignment behavior.
                Keys used: Model.align_method, Model.scale_compute_method,
                Model.Pointcloud_Save.conf_threshold_coef, Model.Pointcloud_Save.sample_ratio.
                If None, uses predictor.config (if available, e.g. DA3Predictor),
                otherwise falls back to default SIM(3) alignment config.
        """
        self.predictor = predictor
        if align_config is not None:
            self.config = align_config
        else:
            self.config = _load_default_align_config()

        # Streaming state
        self.prev_internal: _InternalPredictions | None = None
        self.prev_chunk_images: torch.Tensor | None = None  # (N, 3, H, W) cached
        self.prev_chunk_hint_ext: torch.Tensor | None = None  # (N, 4, 4) w2c cached
        self.prev_chunk_hint_int: torch.Tensor | None = None  # (N, 3, 3) norm cached
        self.sim3_cumulative: tuple[float, np.ndarray, np.ndarray] | None = None
        self.chunk_idx: int = 0
        self.all_chunk_results: list[dict] = []
        self.global_frame_idx: int = 0

        print(f"[DepthStreaming] initialized with predictor: {type(predictor).__name__}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_chunk(
        self,
        images: torch.Tensor,
        overlap: int = 0,
        hint_extrinsics: torch.Tensor | None = None,
        hint_intrinsics: torch.Tensor | None = None,
    ) -> DepthPredictorOutput:
        """Process a new chunk of frames.

        Only provide the NEW frames for this chunk. The overlap frames are
        automatically prepended from the previous chunk's cache.

        Args:
            images: (N, 3, H, W) tensor, float32, [0, 1], RGB.
                These are the NEW frames only.
            overlap: Number of frames from the tail of the previous chunk to
                     prepend for alignment. Must be 0 for the first chunk.
            hint_extrinsics: Optional camera hint for the **NEW** frames only,
                (N, 4, 4) w2c (repo convention). The overlap segment's hint is
                auto-prepended from the previous chunk's cache, mirroring how the
                overlap *images* are prepended, so DA3 receives one camera per
                frame of the full [overlap+new] stack.
            hint_intrinsics: Optional camera hint for the NEW frames only,
                (N, 3, 3) normalized (repo convention).

        Returns:
            DepthPredictorOutput with globally-aligned depth, normalized
            intrinsics, w2c extrinsics (4x4), depth_mask, and confidence.
        """
        # Validate
        if self.chunk_idx == 0 and overlap != 0:
            raise ValueError("overlap must be 0 for the first chunk.")
        if self.chunk_idx > 0 and overlap <= 0:
            raise ValueError("overlap must be > 0 for chunks after the first.")
        if self.chunk_idx > 0 and overlap > self.prev_chunk_images.shape[0]:
            raise ValueError(
                f"overlap ({overlap}) exceeds previous chunk size "
                f"({self.prev_chunk_images.shape[0]})."
            )

        use_hint = hint_extrinsics is not None and hint_intrinsics is not None

        # Build full chunk: overlap frames from cache + new frames
        if self.chunk_idx == 0:
            full_images = images
            full_hint_ext = hint_extrinsics if use_hint else None
            full_hint_int = hint_intrinsics if use_hint else None
        else:
            overlap_images = self.prev_chunk_images[-overlap:]
            full_images = torch.cat([overlap_images, images], dim=0)
            if use_hint:
                if self.prev_chunk_hint_ext is None:
                    raise ValueError(
                        "hint_extrinsics provided now but the previous chunk "
                        "had no cached hint; pass hints consistently."
                    )
                ov_ext = self.prev_chunk_hint_ext[-overlap:].to(hint_extrinsics.device)
                ov_int = self.prev_chunk_hint_int[-overlap:].to(hint_intrinsics.device)
                full_hint_ext = torch.cat([ov_ext, hint_extrinsics], dim=0)
                full_hint_int = torch.cat([ov_int, hint_intrinsics], dim=0)
            else:
                full_hint_ext = None
                full_hint_int = None

        num_new = images.shape[0]

        # Run predictor on full chunk (with camera hints if supplied)
        if use_hint:
            output = self.predictor.predict(
                full_images,
                hint_extrinsics=full_hint_ext,
                hint_intrinsics=full_hint_int,
            )
        else:
            output = self.predictor.predict(full_images)

        # Convert to internal format for alignment
        internal = _output_to_internal(output)

        # Alignment
        if self.chunk_idx == 0:
            s_cum, R_cum, t_cum = 1.0, np.eye(3), np.zeros(3)
            self.sim3_cumulative = (s_cum, R_cum, t_cum)
        else:
            s_pair, R_pair, t_pair = self._align_with_previous(internal, overlap)
            s_prev, R_prev, t_prev = self.sim3_cumulative
            s_cum = s_prev * s_pair
            R_cum = R_prev @ R_pair
            t_cum = s_prev * (R_prev @ t_pair) + t_prev
            self.sim3_cumulative = (s_cum, R_cum, t_cum)

        s_cum, R_cum, t_cum = self.sim3_cumulative

        # Extract new (non-overlap) portion
        new_depth = internal.depth[overlap:]
        new_conf = internal.conf[overlap:]
        new_intrinsics = internal.intrinsics[overlap:]
        new_extrinsics = internal.extrinsics[overlap:]

        # World points in global frame
        if self.chunk_idx == 0:
            world_points = _depth_to_point_cloud_vectorized(
                new_depth, new_intrinsics, new_extrinsics
            )
        else:
            world_points = depth_to_point_cloud_optimized_torch(
                new_depth, new_intrinsics, new_extrinsics
            )
            world_points = apply_sim3_direct_torch(world_points, s_cum, R_cum, t_cum)

        # Global c2w poses
        extrinsics_global = self._compute_global_c2w(new_extrinsics, s_cum, R_cum, t_cum)

        # Aligned depth
        aligned_depth = new_depth * s_cum

        # Assign global frame indices
        frame_start = self.global_frame_idx
        self.global_frame_idx += num_new

        # Store raw result for save()
        new_images_np = (full_images[overlap:].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        result = {
            "chunk_idx": self.chunk_idx,
            "frame_start": frame_start,
            "frame_end": self.global_frame_idx,
            "depth": aligned_depth,
            "conf": new_conf,
            "world_points": world_points,
            "images": new_images_np,
            "intrinsics": new_intrinsics,
            "extrinsics_global": extrinsics_global,
            "sim3_cumulative": self.sim3_cumulative,
        }
        self.all_chunk_results.append(result)

        # Cache for next chunk's overlap
        self.prev_chunk_images = full_images.cpu()
        self.prev_internal = internal
        if use_hint:
            self.prev_chunk_hint_ext = full_hint_ext.detach().cpu()
            self.prev_chunk_hint_int = full_hint_int.detach().cpu()
        else:
            self.prev_chunk_hint_ext = None
            self.prev_chunk_hint_int = None

        self.chunk_idx += 1

        # Convert to DepthPredictorOutput
        return self._to_output(result, output.depth.device)

    def _to_output(self, result: dict, device: torch.device | str) -> DepthPredictorOutput:
        """Convert raw chunk result to DepthPredictorOutput."""
        depth_np = result["depth"]
        conf_np = result["conf"]
        N, H, W = depth_np.shape

        depth = torch.from_numpy(depth_np).float().to(device)
        conf = torch.from_numpy(conf_np).float().to(device)

        # Intrinsics: pixel-space → normalized [0, 1]
        intrinsics = torch.from_numpy(result["intrinsics"].copy()).float().to(device)
        intrinsics[:, 0, 0] /= W
        intrinsics[:, 0, 2] /= W
        intrinsics[:, 1, 1] /= H
        intrinsics[:, 1, 2] /= H

        # Extrinsics: c2w list → w2c (N, 4, 4)
        c2w_list = result["extrinsics_global"]
        w2c_list = []
        for c2w in c2w_list:
            w2c = np.linalg.inv(c2w)
            w2c_list.append(w2c)
        extrinsics = torch.from_numpy(np.stack(w2c_list, axis=0)).float().to(device)

        # Depth mask
        conf_threshold = conf.mean() * self.CONFIDENCE_THRESHOLD_COEF
        depth_mask = (conf > conf_threshold) & (depth > 0) & torch.isfinite(depth)

        return DepthPredictorOutput(
            depth=depth,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            depth_mask=depth_mask,
            confidence=conf,
        )

    def save(self, base_path: str) -> None:
        """Save all accumulated RGBD results to disk.

        Directory structure:
            base_path/
                rgbd/
                    frame_000000.npz
                    ...
                camera_poses.txt
                pcd/
                    0_pcd.ply
                    ...
        """
        rgbd_dir = os.path.join(base_path, "rgbd")
        pcd_dir = os.path.join(base_path, "pcd")
        os.makedirs(rgbd_dir, exist_ok=True)
        os.makedirs(pcd_dir, exist_ok=True)

        all_poses = []

        for result in self.all_chunk_results:
            chunk_idx = result["chunk_idx"]
            frame_start = result["frame_start"]
            num_new = result["frame_end"] - result["frame_start"]

            for i in range(num_new):
                global_idx = frame_start + i
                filename = f"frame_{global_idx:06d}.npz"
                filepath = os.path.join(rgbd_dir, filename)
                np.savez_compressed(
                    filepath,
                    image=result["images"][i],
                    depth=result["depth"][i],
                    conf=result["conf"][i],
                    intrinsics=result["intrinsics"][i],
                    c2w=result["extrinsics_global"][i],
                )

            all_poses.extend(result["extrinsics_global"])

            # Save point cloud per chunk
            points = np.asarray(result["world_points"]).reshape(-1, 3)
            colors = result["images"].reshape(-1, 3).astype(np.uint8)
            confs = result["conf"].reshape(-1)
            ply_path = os.path.join(pcd_dir, f"{chunk_idx}_pcd.ply")
            save_confident_pointcloud_batch(
                points=points,
                colors=colors,
                confs=confs,
                output_path=ply_path,
                conf_threshold=np.mean(confs)
                * self.config["Model"]["Pointcloud_Save"]["conf_threshold_coef"],
                sample_ratio=self.config["Model"]["Pointcloud_Save"]["sample_ratio"],
            )

        # Save camera poses
        poses_path = os.path.join(base_path, "camera_poses.txt")
        with open(poses_path, "w") as f:
            for pose in all_poses:
                f.write(" ".join(str(x) for x in pose.flatten()) + "\n")

        print(
            f"[DepthStreaming] saved {len(all_poses)} frames to {base_path} "
            f"({len(self.all_chunk_results)} chunks)"
        )

    def get_global_poses(self) -> list[np.ndarray]:
        """Return global c2w (4x4) poses for all processed new frames."""
        all_poses = []
        for result in self.all_chunk_results:
            all_poses.extend(result["extrinsics_global"])
        return all_poses

    def reset(self):
        """Reset streaming state for a new sequence."""
        self.prev_internal = None
        self.prev_chunk_images = None
        self.prev_chunk_hint_ext = None
        self.prev_chunk_hint_int = None
        self.sim3_cumulative = None
        self.chunk_idx = 0
        self.all_chunk_results = []
        self.global_frame_idx = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _align_with_previous(self, cur_internal: _InternalPredictions, overlap: int):
        """Align current chunk to previous chunk via the overlap region.

        Returns pairwise SIM(3) (s, R, t): cur -> prev.
        """
        prev = self.prev_internal

        # Previous chunk: last `overlap` frames
        point_map1 = _depth_to_point_cloud_vectorized(
            prev.depth[-overlap:], prev.intrinsics[-overlap:], prev.extrinsics[-overlap:]
        )
        conf1 = prev.conf[-overlap:]

        # Current chunk: first `overlap` frames
        point_map2 = _depth_to_point_cloud_vectorized(
            cur_internal.depth[:overlap],
            cur_internal.intrinsics[:overlap],
            cur_internal.extrinsics[:overlap],
        )
        conf2 = cur_internal.conf[:overlap]

        conf_threshold = min(np.median(conf1), np.median(conf2)) * 0.1

        scale_factor = None
        if self.config["Model"]["align_method"] == "scale+se3":
            scale_factor, quality, method = precompute_scale_chunks_with_depth(
                np.squeeze(prev.depth[-overlap:]),
                np.squeeze(prev.conf[-overlap:]),
                np.squeeze(cur_internal.depth[:overlap]),
                np.squeeze(cur_internal.conf[:overlap]),
                method=self.config["Model"]["scale_compute_method"],
            )
            print(f"[align] scale precompute: {scale_factor}, quality: {quality}, method: {method}")

        s, R, t = weighted_align_point_maps(
            point_map1, conf1, point_map2, conf2,
            conf_threshold=conf_threshold,
            config=self.config,
            precompute_scale=scale_factor,
        )
        print(f"[align chunk {self.chunk_idx - 1} -> {self.chunk_idx}] s={s:.6f}")
        return s, R, t

    def _compute_global_c2w(self, extrinsics_w2c: np.ndarray, s: float, R: np.ndarray, t: np.ndarray):
        """Convert chunk-local w2c (N, 3, 4) to global c2w (list of 4x4)."""
        N = extrinsics_w2c.shape[0]
        S = np.eye(4)
        S[:3, :3] = s * R
        S[:3, 3] = t

        global_poses = []
        for i in range(N):
            w2c_4x4 = np.eye(4)
            w2c_4x4[:3, :] = extrinsics_w2c[i]
            c2w = np.linalg.inv(w2c_4x4)
            c2w_global = S @ c2w
            c2w_global[:3, :3] /= s  # remove scale from rotation
            global_poses.append(c2w_global)
        return global_poses
