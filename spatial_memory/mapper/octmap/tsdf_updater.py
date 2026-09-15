from .octree.dynamic_octree import DynOctree
from .camera import PinholeCamera
from .frustum import build_frustum_planes
import os
import torch
from typing import Tuple, List, Optional


_DEBUG_MEM = os.environ.get("OCTMAP_DEBUG_MEM", "0") == "1"


def _mem_stage(label: str, device):
    if not _DEBUG_MEM or not torch.cuda.is_available():
        return
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    cur = torch.cuda.memory_allocated(device)
    print(f"      [updater-stage] {label:<32s} cur={cur / (1024**2):8.1f} MB  peak={peak / (1024**2):8.1f} MB")
    torch.cuda.reset_peak_memory_stats(device)


class TSDFUpdater:
    '''
    Point-based TSDF updater. Integrates a batch of depth maps + cameras into a
    DynOctree representation. Camera views are processed in fixed-size batches;
    for each batch we build the union of view frustums, fetch only the octree
    leaves intersecting that union, and per-depth iteratively fuse them.

    Args:
        representation: The voxel based representation to be updated.
        sdf_truncate_margin_range: Min/max clamp range for the TSDF truncation margin.
        trunc_margin_ratio: The truncation margin ratio relative to the voxel size.
        max_num_points: Per-view chunk size when projecting/fusing corner points.
        frustum_batch_size: Number of camera views to batch when computing frustum planes.
    '''
    def __init__(
        self,
        representation: DynOctree,
        sdf_truncate_margin_range: Tuple[float, float] = (0, 1),
        trunc_margin_ratio: float = 5.0,
        max_num_points: int = 2_000_000,
        frustum_batch_size: int = 20,
    ):
        self.representation = representation
        self.sdf_truncate_margin_range = sdf_truncate_margin_range
        self.trunc_margin_ratio = trunc_margin_ratio
        self.max_num_points = int(max_num_points)
        self.frustum_batch_size = int(frustum_batch_size)

    @staticmethod
    def calc_points_vals(pts, frame, depth, camera: PinholeCamera, truncate_margin):
        """
        Memory-tight projection + visibility/TSDF-band filtering.

        Combines all three masks (in-image, depth>0, depth_diff>=-margin) into a
        single ``nonzero`` call, releases unused intermediates as soon as possible,
        and avoids the extra float->long roundtrip allocation.

        ``truncate_margin`` may be either a scalar (uniform per voxel for a single
        depth) or a (N,) tensor (per-point), and we avoid materializing a per-point
        tensor in the scalar case.
        """
        H, W = depth.shape[-2:]
        scalar_margin = not torch.is_tensor(truncate_margin)

        # Project all points
        pixel_uv, linear_depths, in_front_mask = camera.project_points(pts)
        # uv [0,1] -> integer pixel coords; do float->long once, no intermediate copy
        px = (pixel_uv[:, 0] * W + 0.5).long()
        py = (pixel_uv[:, 1] * H + 0.5).long()
        del pixel_uv

        in_img_mask = in_front_mask & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        del in_front_mask

        # First filter: only keep points that hit a valid pixel
        first_idx = torch.nonzero(in_img_mask, as_tuple=True)[0]
        del in_img_mask
        if first_idx.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=pts.device)
            return empty, torch.empty(0, device=pts.device), torch.empty(0, 3, device=pts.device)

        px = px[first_idx]
        py = py[first_idx]
        ld = linear_depths[first_idx]
        del linear_depths
        if scalar_margin:
            tm = truncate_margin
        else:
            tm = truncate_margin[first_idx]

        gt_depth = depth[py, px]
        depth_diff = gt_depth - ld
        del ld

        # Second filter: gt>0 AND inside TSDF truncation band, single nonzero
        keep = (gt_depth > 0) & (depth_diff >= -tm)
        del gt_depth
        keep_idx = torch.nonzero(keep, as_tuple=True)[0]
        del keep
        if keep_idx.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=pts.device)
            return empty, torch.empty(0, device=pts.device), torch.empty(0, 3, device=pts.device)

        valid_indices = first_idx[keep_idx]
        px = px[keep_idx]
        py = py[keep_idx]
        depth_diff = depth_diff[keep_idx]
        if scalar_margin:
            depth_diff.clamp_(max=tm)
        else:
            tm = tm[keep_idx]
            depth_diff = torch.minimum(depth_diff, tm)
            del tm

        sample_colors = frame[py, px]
        return valid_indices, depth_diff, sample_colors

    def _update_fusion_inplace(self, batch_features, batch_weights, valid_indices,
                               tsdf_new, color_new, obs_weight: float = 1.0):
        """In-place TSDF/color fusion that writes back into ``batch_features``/``batch_weights``
        at positions ``valid_indices`` without allocating fused ``feat_new`` / ``w_new`` /
        ``tsdf_updated`` / ``color_updated`` intermediates as full tensors.
        """
        # Pull the rows we need (copy is unavoidable for advanced indexing read).
        feat_old = batch_features.index_select(0, valid_indices)  # (N', 4)
        w_old = batch_weights.index_select(0, valid_indices)      # (N',)

        w_new = w_old + obs_weight
        inv_w = w_new.reciprocal()

        # TSDF (channel 0): (tsdf_old * w_old + obs_weight * tsdf_new) / w_new
        tsdf_chan = feat_old[:, 0]
        tsdf_chan.mul_(w_old).add_(tsdf_new, alpha=obs_weight).mul_(inv_w)

        # Color (channels 1-3): per-channel weighted blend, in place.
        color_chan = feat_old[:, 1:4]
        color_chan.mul_(w_old.unsqueeze(-1)).add_(color_new, alpha=obs_weight).mul_(inv_w.unsqueeze(-1))

        # Write back.
        batch_features.index_copy_(0, valid_indices, feat_old)
        batch_weights.index_copy_(0, valid_indices, w_new)

    def _run_single_view(self, frame, depth, camera: PinholeCamera,
                         features, weights, pts, truncate_margin):
        """
        Run TSDF update for a single view on pre-fetched points.

        ``truncate_margin`` may be either a scalar (per-depth uniform) or a per-point
        tensor; ``features``/``weights`` are modified in-place via index_copy_.
        """
        N = len(pts)
        step = self.max_num_points
        for i in range(0, N, step):
            end = min(i + step, N)
            batch_features = features[i:end]
            batch_weights = weights[i:end]
            batch_pts = pts[i:end]
            if torch.is_tensor(truncate_margin):
                batch_tm = truncate_margin[i:end]
            else:
                batch_tm = truncate_margin

            valid_indices, tsdf_val, color_val = self.calc_points_vals(
                batch_pts, frame, depth, camera, batch_tm)

            if valid_indices.numel() == 0:
                del valid_indices, tsdf_val, color_val
                continue

            self._update_fusion_inplace(
                batch_features, batch_weights, valid_indices, tsdf_val, color_val)

            # Release per-chunk transients eagerly so the next chunk can reuse the memory.
            del valid_indices, tsdf_val, color_val

    def _scalar_truncate_margin(self, voxel_size_scalar: float) -> float:
        return float(
            min(
                max(voxel_size_scalar * self.trunc_margin_ratio,
                    self.sdf_truncate_margin_range[0]),
                self.sdf_truncate_margin_range[1],
            )
        )

    @staticmethod
    def _fetch_single_depth_frustum_leaves(rep: DynOctree, d: int, idx: torch.Tensor):
        """Fetch leaf buffers for a single octree depth given pre-frustum-culled
        node indices ``idx``. Returns ``None`` if there are no leaves.

        Materialising one depth at a time keeps peak memory bounded by the largest
        single depth's per-corner buffers instead of the sum over all depths.
        """
        if idx.numel() == 0:
            return None
        leaf_mask = rep.children[d][idx] == -1
        idx = idx[leaf_mask]
        del leaf_mask
        if idx.numel() == 0:
            return None
        fd = rep.features[d][idx].reshape(-1, rep.feature_dims)
        wd = rep.weights[d][idx].reshape(-1)
        pd = rep.xyz_to_points(rep.xyz[d][idx], d).reshape(-1, 3)
        return (d, idx, fd, wd, pd, rep.get_voxel_size(d))

    @staticmethod
    def _writeback_single_depth(rep: DynOctree, buf):
        d, idx, fd, wd, _pd, _vs = buf
        feat_dims = rep.feature_dims
        rep.features[d][idx] = fd.reshape(-1, 8, feat_dims)
        rep.weights[d][idx] = wd.reshape(-1, 8)

    def run(
        self,
        frames: List[torch.Tensor],
        depths: List[torch.Tensor],
        cameras: List[PinholeCamera],
        start_depth: Optional[object] = None,
    ):
        """
        Integrate a batch of depth frames into the representation, using view
        frustum culling.

        Processes cameras in batches of ``self.frustum_batch_size``:
        1. Build frustum planes for all views in the batch.
        2. Fetch active node indices intersecting ANY frustum in the batch.
        3. For each octree depth: fetch its leaf buffers, fuse every view in the
           batch into them, write back, free.

        Args:
            frames: List of (H, W, 3) RGB images.
            depths: List of (H, W) depth maps.
            cameras: List of PinholeCamera objects (one per frame).
            start_depth: Minimum octree depth to start frustum traversal from.
                Accepts either:
                  - ``None`` or an ``int``: a single global start depth applied
                    to every frustum batch (the online / single-chunk case), or
                  - a per-frame sequence (list/tuple, length == ``len(frames)``):
                    for each frustum batch the shallowest (``min``) start depth
                    over that batch's frames is used. This keeps offline
                    all-frame re-updates from fusing coarse leaves into frames
                    that were only allocated at deeper levels (large-face bug).
                Nodes shallower than the resolved depth are skipped.
        """
        assert len(frames) == len(depths) == len(cameras), \
            "frames, depths, cameras must have the same length"

        num_views = len(frames)
        device = self.representation.device
        rep = self.representation

        per_frame_start_depth = isinstance(start_depth, (list, tuple))
        if per_frame_start_depth:
            assert len(start_depth) == num_views, (
                f"start_depth list length ({len(start_depth)}) must match number "
                f"of frames ({num_views})"
            )

        for batch_start in range(0, num_views, self.frustum_batch_size):
            batch_end = min(batch_start + self.frustum_batch_size, num_views)
            batch_cameras = cameras[batch_start:batch_end]

            # Resolve this batch's start depth: shallowest over the batch's frames
            # when a per-frame list is given, otherwise the single global value.
            if per_frame_start_depth:
                batch_start_depth = min(start_depth[batch_start:batch_end])
            else:
                batch_start_depth = start_depth

            # Move ONLY this batch's RGB/depth to GPU (no-op if already there);
            # freed at the end of the batch so raw-input residency is bounded by
            # frustum_batch_size frames rather than the total frame count.
            batch_frames = [frames[i].to(device) for i in range(batch_start, batch_end)]
            batch_depths = [depths[i].to(device) for i in range(batch_start, batch_end)]

            _mem_stage(f"batch[{batch_start}:{batch_end}] start", device)

            # Build frustum planes for this batch of cameras
            batch_intrinsics = torch.stack([c.intrinsics for c in batch_cameras])  # (B, 3, 3)
            batch_extrinsics = torch.stack([c.extrinsics for c in batch_cameras])  # (B, 4, 4)

            plane_normals, plane_offsets = build_frustum_planes(
                batch_intrinsics, batch_extrinsics
            )
            _mem_stage("build_frustum_planes", device)

            # Fetch active node indices within the frustum (per-depth index lists only).
            per_depth_indices = rep.fetch_active_nodes_by_frustum(
                plane_normals, plane_offsets, start_depth=batch_start_depth,
            )
            del plane_normals, plane_offsets
            _mem_stage("fetch_active_nodes_by_frustum", device)

            # Process one depth at a time: fetch -> run all views -> write back -> free.
            # This caps peak memory at a single depth's per-corner buffer instead of the
            # sum across all depths.
            for d, d_idx in enumerate(per_depth_indices):
                if d > rep.cur_depth or d_idx.numel() == 0:
                    continue
                buf = self._fetch_single_depth_frustum_leaves(rep, d, d_idx)
                if buf is None:
                    continue
                d_id, idx_d, fd, wd, pd, vs_scalar = buf
                _mem_stage(f"depth {d_id} fetch N={pd.shape[0]}", device)
                tm = self._scalar_truncate_margin(vs_scalar)
                for bi, i in enumerate(range(batch_start, batch_end)):
                    self._run_single_view(
                        batch_frames[bi], batch_depths[bi], cameras[i], fd, wd, pd, tm,
                    )
                self._writeback_single_depth(rep, buf)
                # Free buffers eagerly before moving to the next depth.
                del buf, idx_d, fd, wd, pd
                _mem_stage(f"depth {d_id} done", device)
            del per_depth_indices
            del batch_frames, batch_depths
