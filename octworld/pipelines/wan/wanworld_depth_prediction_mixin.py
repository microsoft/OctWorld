"""Depth-prediction logic for :class:`WanWorldPipeline`, factored into a mixin.

This decouples *what depth backend / strategy is used* from the Wan AR rollout
in :mod:`octworld.pipelines.wan.wanworld_pipeline`. The pipeline mixes in
:class:`WanWorldDepthPredictionMixin`, which:

- builds + holds the depth predictor (and, for ``standalone``, the streamer),
- owns the depth-mask policy and the user-trajectory scale state,
- owns all the shared depth helpers (resize / mask / first-frame predict / …),
- exposes the per-chunk methods ``predict_depth_standalone`` /
    ``predict_depth_align_with_user`` / ``predict_depth_scale_alignment`` and
    binds the active one to
  ``self.predict_depth_fn`` (selected by ``depth_method``).

Each ``process_chunk_*`` only **predicts depth + chooses cameras** — it takes
frames (+ user cameras) and returns depth + the cameras to integrate. The
pipeline's ``__call__`` owns frame assembly, the trajectory rescale, the 3D
**mapper** (``_update_map`` / ``_reset_mapper``) and the AR **buffers** — they
are not touched here. See
``docs/code_plan/wanworld_depth_prediction_decouple_plan.md``.

``process_chunk_*`` returns a dict with:

- ``depths``          (N, H, W)    — metric depth at mapper resolution.
- ``masks``           (N, H, W)    — bool depth mask.
- ``map_extrinsics``  (N, 4, 4)    — cameras to integrate into the map (w2c).
- ``map_intrinsics``  (N, 3, 3)    — normalized intrinsics for integration.

The pipeline owns frame assembly: on the first chunk it prepends the seed frame
so depth is predicted over ``[frame0 | gen0]`` (81 frames), resets the mapper,
and rebuilds the seed buffers; later chunks predict the 80 generated frames and
append. Both depth methods share this first-chunk handling. The pipeline also
owns the trajectory rescale: for ``standalone`` it refines ``s_user`` + rebases
the user trajectory onto DA3's frame-0 from the chunk-0 result.

Three depth methods:

- ``"standalone"`` — DA3 **streaming**; map integrated with DA3-predicted poses.
- ``"align_with_user"`` — strictly **non-streaming**; map integrated with the
  **user camera**; the predictor only provides depth (aligned to the user
  trajectory via ``align_to_input=True``).
- ``"scale_alignment"`` — DA3 ``predict`` only (no streamer): from chunk 1 on,
    run one joint predict over ``[prev_overlap_rgb | cur_rgb]`` and use overlap
    depth to estimate a scalar that aligns the new chunk depth to the previous
    chunk depth scale.

The first-frame depth path is **shared** by both methods: a single-frame
``predictor.predict`` (no hint), integrated with the user camera by ``__call__``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from utils.align import align


# ----------------------------------------------------------------------------
# Scale alignment (anchor depth → predicted depth), module-level so verification
# scripts can exercise the exact same code path as the pipeline.
# ----------------------------------------------------------------------------

def compute_depth_scale(
    anchor_depth: torch.Tensor,
    da3_depth: torch.Tensor,
    anchor_mask: Optional[torch.Tensor] = None,
    da3_mask: Optional[torch.Tensor] = None,
    max_points: int = 50000,
) -> float:
    """Estimate scalar ``s`` such that ``s * anchor_depth ≈ da3_depth``.

    Converts the anchor (user-trajectory) depth scale into the predictor's
    metric scale, using an inverse-depth-weighted L1 fit over valid, overlapping
    pixels. Raises ``ValueError`` if there are too few valid overlapping pixels.

    Args:
        anchor_depth: (H, W) caller-provided first-frame depth (user scale).
        da3_depth:    (H, W) predicted first-frame depth (metric scale).
        anchor_mask / da3_mask: optional (H, W) bool validity masks.
        max_points:   subsample cap for the fit.
    """
    anchor_depth = anchor_depth.float()
    da3_depth = da3_depth.float()
    m = (
        torch.isfinite(anchor_depth) & torch.isfinite(da3_depth)
        & (anchor_depth > 0) & (da3_depth > 0)
    )
    if anchor_mask is not None:
        m &= anchor_mask.bool()
    if da3_mask is not None:
        m &= da3_mask.bool()
    src = anchor_depth[m].flatten()
    tgt = da3_depth[m].flatten()
    if src.numel() < 100:
        raise ValueError(
            f"[compute_depth_scale] too few valid overlapping pixels "
            f"({src.numel()} < 100) between anchor_depth and da3_depth; "
            "cannot estimate trajectory scale. Check the anchor (moge) depth "
            "and its validity mask."
        )
    if src.numel() > max_points:
        idx = torch.randperm(src.numel(), device=src.device)[:max_points]
        src, tgt = src[idx], tgt[idx]
    w = 1.0 / tgt.clamp_min(1e-3)
    # Weighted-L1 scale fit (trunc=None branch) from World-vdt align_utils.
    a, _, _ = align(src, tgt, w)
    return float(a.item())


class WanWorldDepthPredictionMixin:
    """Depth backend + per-chunk depth strategy for ``WanWorldPipeline``.

    The mixin reads pipeline state directly (``self.all_*``,
    ``self.mapper_resolution``, ``self.device`` …), so it must be mixed into a
    :class:`WorldGenPipeline` subclass. It never mutates the mapper or AR
    buffers — that is the pipeline's job.
    """

    # ------------------------------------------------------------------
    # Backend construction / reset
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_depth_config(cfg) -> tuple[str, str]:
        depth_method = str(cfg.depth_method)
        supported_methods = {
            "standalone", "align_with_user", "scale_alignment",
        }
        if depth_method not in supported_methods:
            raise ValueError(
                f"unknown depth_method {depth_method!r}; expected one of "
                f"{sorted(supported_methods)}."
            )
        if depth_method == "standalone":
            raise ValueError(
                "depth_method='standalone' is deprecated and disabled; use "
                "'align_with_user' (default, with automatic scale-alignment "
                "fallback) or 'scale_alignment'."
            )

        depth_predictor_name = str(cfg.depth_predictor_name)
        if depth_predictor_name != "da3":
            raise ValueError(
                "depth_method in {'align_with_user','scale_alignment'} requires "
                "depth_predictor_name='da3' (camera-hint or DA3-style overlap "
                "scale transfer); got "
                f"{depth_predictor_name!r}."
            )
        return depth_method, depth_predictor_name

    def _init_depth_backend(self, cfg, device: str):
        """Build the depth predictor (+ streamer for ``standalone``), cache the
        depth-mask policy + trajectory-scale state, and bind the active
        per-chunk method to ``self.predict_depth_fn``. Validates ``depth_method``
        / ``depth_predictor_name`` combinations.
        """
        from depth_prediction.predictors import create_predictor, DepthStreaming

        self.depth_method, self.depth_predictor_name = self._validate_depth_config(cfg)
        predict_depth_fns = {
            "standalone": self.predict_depth_standalone,
            "align_with_user": self.predict_depth_align_with_user,
            "scale_alignment": self.predict_depth_scale_alignment,
        }

        # One predictor instance, shared by the (optional) streamer.
        self._depth_predictor = create_predictor(
            self.depth_predictor_name, device=device)
        # The streamer is only used by ``standalone``; ``align_with_user`` is
        # strictly non-streaming and must never touch a streamer.
        self._depth_streamer = (
            DepthStreaming(self._depth_predictor)
            if self.depth_method == "standalone" else None
        )

        # Depth-mask policy (mirrors datapipe/wan_pipes/octmap_integration.py).
        self.conf_percentile = float(cfg.conf_percentile)
        self.apply_sky_mask = bool(cfg.apply_sky_mask)
        self.sky_threshold = int(cfg.sky_threshold)

        # Trajectory-scale state (set per call).
        self._s_user: float = 1.0
        self._user_extrinsics_raw: Optional[torch.Tensor] = None
        self._da3_ext0: Optional[torch.Tensor] = None

        # Per-chunk debug masks (separate confidence / sky masks + confidence),
        # stashed by ``_da3_depth_and_mask`` for optional persistence.
        self._dbg_conf_mask: Optional[torch.Tensor] = None
        self._dbg_sky_mask: Optional[torch.Tensor] = None
        self._dbg_confidence: Optional[torch.Tensor] = None

        # ``scale_alignment`` chunk-bridge state (tail overlap from previous
        # chunk, already in the running target depth scale).
        self._scale_align_prev_rgb: Optional[torch.Tensor] = None
        self._scale_align_prev_depth: Optional[torch.Tensor] = None
        self._scale_align_prev_mask: Optional[torch.Tensor] = None
        self._scale_align_prev_extrs: Optional[torch.Tensor] = None
        self._scale_align_prev_intrs: Optional[torch.Tensor] = None

        # Active per-chunk depth method (selected by depth_method).
        self.predict_depth_fn = predict_depth_fns[self.depth_method]

    def _reset_depth_state(self):
        """Reset streaming + trajectory-scale state at the start of a rollout."""
        if self._depth_streamer is not None:
            self._depth_streamer.reset()
        self._s_user = 1.0
        self._user_extrinsics_raw = None
        self._da3_ext0 = None

        self._scale_align_prev_rgb = None
        self._scale_align_prev_depth = None
        self._scale_align_prev_mask = None
        self._scale_align_prev_extrs = None
        self._scale_align_prev_intrs = None

    # ------------------------------------------------------------------
    # Per-chunk depth methods (one per depth_method; pick via predict_depth_fn).
    # Each ONLY predicts depth + chooses cameras and returns the unified dict;
    # it never touches the mapper or AR buffers.
    # ------------------------------------------------------------------

    def predict_depth_standalone(
        self,
        *,
        frames: torch.Tensor,
        chunk_extrs: torch.Tensor,
        chunk_intrs: torch.Tensor,
        chunk_idx: int,
    ) -> Dict[str, Any]:
        """DA3 **streaming** depth; map integrated with DA3's predicted poses.

        On chunk 0 the pipeline passes the 81-frame ``[frame0 | gen0]`` stack;
        we reset the streamer and predict with ``overlap=0`` (the bootstrap).
        Later chunks stream the 80 generated frames with all previous frames as overlap.
        Mirrors the offline datapipe. The chunk-0 trajectory rescale lives in ``__call__``.
        """
        if chunk_idx == 0:
            self._depth_streamer.reset()
        # On chunk 0, use no overlap (bootstrap); from chunk 1 on, use all previous frames
        overlap = 0 if chunk_idx == 0 else self._depth_streamer.prev_chunk_images.shape[0]
        out = self._depth_streamer.add_chunk(
            self._frames_to_chw(frames), overlap=overlap)
        depth_r, mask_r = self._da3_depth_and_mask(out, frames)
        return dict(
            depths=depth_r, masks=mask_r,
            map_extrinsics=out.extrinsics, map_intrinsics=out.intrinsics,
            conf_mask=self._dbg_conf_mask, sky_mask=self._dbg_sky_mask,
            confidence=self._dbg_confidence,
        )

    def predict_depth_align_with_user(
        self,
        *,
        frames: torch.Tensor,
        chunk_extrs: torch.Tensor,
        chunk_intrs: torch.Tensor,
        chunk_idx: int,
    ) -> Dict[str, Any]:
        """Non-streaming depth aligned to the user camera, with an automatic
        **scale-alignment fallback** when DA3's Umeyama pose alignment fails.

        Primary path: a single ``predictor.predict`` over ``frames`` with the
        **user camera as a hint** and ``align_to_input=True``, so the predictor
        *provides depth* aligned to the user trajectory (depth in the
        user-camera metric scale). The map is integrated with the **user
        camera**; the predictor's returned poses are not used. On chunk 0 the
        pipeline passes the 81-frame ``[frame0 | gen0]`` stack (and matching user
        cameras).

        Fallback: ``align_to_input`` runs an Umeyama Sim(3) fit
        (``align_poses_umeyama``) that raises ``evo``'s ``GeometryException``
        (*Degenerate covariance rank*) on near-static / degenerate camera
        trajectories. When that happens for **this** chunk, we fall back to the
        scale-alignment path (:meth:`_scale_align_predict`): a **no-Umeyama**
        DA3 predict whose scale is recovered from overlap with the previous chunk
        (or the seed frame on chunk 0). The fallback is per-chunk; later chunks
        retry the Umeyama path.

        On the primary (success) path we also cache this chunk so a subsequent
        chunk that falls back has a user-scale reference to align to.
        """
        try:
            depth_r, mask_r = self._da3_chunk_depth_with_hint(
                frames, chunk_extrs, chunk_intrs)
        except Exception as e:  # noqa: BLE001 — narrowed by _is_umeyama_failure
            if not self._is_umeyama_failure(e):
                raise
            print(
                f"[WanWorldDepth] Umeyama pose-alignment failed on chunk "
                f"{chunk_idx} ({type(e).__name__}: {e}); falling back to "
                f"scale-alignment for this chunk."
            )
            return self._scale_align_predict(
                frames=frames, chunk_extrs=chunk_extrs,
                chunk_intrs=chunk_intrs, chunk_idx=chunk_idx,
                cache_keep_last=1,
            )
        # Success: cache this chunk (user scale) so a later fallback can align to
        # it. Keep only the last frame — a single overlap view is enough for a
        # robust scalar and keeps the common (no-fallback) path cheap.
        self._cache_scale_alignment_overlap(
            frames, depth_r, mask_r, chunk_extrs, chunk_intrs, keep_last=1)
        return dict(
            depths=depth_r, masks=mask_r,
            map_extrinsics=chunk_extrs, map_intrinsics=chunk_intrs,
            conf_mask=self._dbg_conf_mask, sky_mask=self._dbg_sky_mask,
            confidence=self._dbg_confidence,
        )

    def predict_depth_scale_alignment(
        self,
        *,
        frames: torch.Tensor,
        chunk_extrs: torch.Tensor,
        chunk_intrs: torch.Tensor,
        chunk_idx: int,
    ) -> Dict[str, Any]:
        """DA3 ``predict`` (**no Umeyama**) + overlap-based per-chunk scale
        transfer.

        Camera hints always come from the user trajectory and are fed to DA3 as
        conditioning, but the unstable Umeyama pose alignment is **never** run.
        The scale is recovered from chunk-to-chunk overlap: on chunk 0 the
        overlap is the **seed frame** (whose depth ``depth0`` defined
        ``s_user``), seeded into the cache by the pipeline; from chunk 1 on it is
        the previous chunk. Map integration always uses the user cameras. See
        :meth:`_scale_align_predict`.
        """
        return self._scale_align_predict(
            frames=frames, chunk_extrs=chunk_extrs,
            chunk_intrs=chunk_intrs, chunk_idx=chunk_idx,
            cache_keep_last=None,
        )

    # ------------------------------------------------------------------
    # Shared scale-alignment path (scale_alignment mode + align_with_user
    # Umeyama fallback). No Umeyama; scale recovered from overlap.
    # ------------------------------------------------------------------

    def _scale_align_predict(
        self,
        *,
        frames: torch.Tensor,
        chunk_extrs: torch.Tensor,
        chunk_intrs: torch.Tensor,
        chunk_idx: int,
        cache_keep_last: Optional[int] = None,
    ) -> Dict[str, Any]:
        """No-Umeyama depth + overlap-based scale alignment to the running scale.

        Runs a single joint DA3 predict over ``[prev_overlap | frames]`` (the
        camera hint conditions the model, but no Umeyama runs), then rescales the
        new-chunk depth so its overlap matches the previous (world-scale) depth.
        ``prev_overlap`` is the cached previous chunk, or — on chunk 0 — the seed
        frame (see :meth:`_seed_scale_align_cache`). The map is integrated with
        the user cameras. ``cache_keep_last`` controls how many trailing frames
        of this chunk are cached for the next overlap (``None`` = all).
        """
        have_cache = (
            self._scale_align_prev_rgb is not None
            and self._scale_align_prev_depth is not None
            and self._scale_align_prev_extrs is not None
            and self._scale_align_prev_intrs is not None
        )
        if not have_cache:
            # No reference to align to (seeding missing): best-effort native
            # scale, then cache for the next chunk.
            depth_r, mask_r = self._da3_chunk_depth_no_umeyama(
                frames, chunk_extrs, chunk_intrs)
            self._cache_scale_alignment_overlap(
                frames, depth_r, mask_r, chunk_extrs, chunk_intrs,
                keep_last=cache_keep_last)
            return self._scale_align_result(depth_r, mask_r, chunk_extrs, chunk_intrs)

        prev_rgb = self._scale_align_prev_rgb
        prev_depth = self._scale_align_prev_depth
        prev_mask = self._scale_align_prev_mask
        prev_extrs = self._scale_align_prev_extrs
        prev_intrs = self._scale_align_prev_intrs
        over_n = int(min(prev_rgb.shape[0], prev_depth.shape[0]))

        pred_frames = torch.cat([prev_rgb[-over_n:].to(frames), frames], dim=0)
        pred_extrs = torch.cat(
            [prev_extrs[-over_n:].to(chunk_extrs), chunk_extrs], dim=0)
        pred_intrs = torch.cat(
            [prev_intrs[-over_n:].to(chunk_intrs), chunk_intrs], dim=0)
        depth_all, mask_all = self._da3_chunk_depth_no_umeyama(
            pred_frames, pred_extrs, pred_intrs)
        over_depth = depth_all[:over_n]
        new_depth = depth_all[over_n:]
        over_mask = mask_all[:over_n]
        new_mask = mask_all[over_n:]
        self._slice_debug_masks(over_n)

        ref_depth = prev_depth[-over_n:].to(over_depth)
        ref_mask = prev_mask[-over_n:].to(over_mask) if prev_mask is not None else None
        depth_scale = self._compute_overlap_scale(
            ref_depth=ref_depth,
            pred_overlap_depth=over_depth,
            ref_mask=ref_mask,
            pred_overlap_mask=over_mask,
        )
        new_depth = new_depth / depth_scale

        self._cache_scale_alignment_overlap(
            frames, new_depth, new_mask, chunk_extrs, chunk_intrs,
            keep_last=cache_keep_last)
        return self._scale_align_result(new_depth, new_mask, chunk_extrs, chunk_intrs)

    def _scale_align_result(
        self,
        depth_r: torch.Tensor,
        mask_r: torch.Tensor,
        chunk_extrs: torch.Tensor,
        chunk_intrs: torch.Tensor,
    ) -> Dict[str, Any]:
        """Standard per-chunk result dict for the scale-alignment path (map
        integrated with the user cameras)."""
        return dict(
            depths=depth_r,
            masks=mask_r,
            map_extrinsics=chunk_extrs,
            map_intrinsics=chunk_intrs,
            conf_mask=self._dbg_conf_mask,
            sky_mask=self._dbg_sky_mask,
            confidence=self._dbg_confidence,
        )

    @staticmethod
    def _is_umeyama_failure(exc: BaseException) -> bool:
        """True iff ``exc`` (or a chained cause/context) is DA3's Umeyama
        degeneracy — ``evo``'s ``GeometryException`` / *Degenerate covariance* /
        *Umeyama* — which the scale-alignment fallback recovers from. Any other
        error propagates unchanged."""
        seen: set[int] = set()
        e: Optional[BaseException] = exc
        while e is not None and id(e) not in seen:
            seen.add(id(e))
            if type(e).__name__ == "GeometryException":
                return True
            msg = str(e).lower()
            if "umeyama" in msg or "degenerate covariance" in msg:
                return True
            e = e.__cause__ or e.__context__
        return False

    def _seed_scale_align_cache(
        self,
        *,
        image_np: torch.Tensor,
        depth0: torch.Tensor,
        mask0: Optional[torch.Tensor],
        extrinsics0: torch.Tensor,
        intrinsics0: torch.Tensor,
    ) -> None:
        """Seed the scale-alignment overlap cache with the first (seed) frame.

        This lets the **first** chunk recover its scale by aligning to ``depth0``
        — the frame that defined ``s_user`` — instead of Umeyama, and is what
        makes both the ``scale_alignment`` mode and the ``align_with_user``
        fallback work on chunk 0. Shared by both.

        Args:
            image_np: (H, W, 3) float [0, 1] seed RGB (mapper resolution).
            depth0:   (H, W) seed depth at the mapper resolution.
            mask0:    (H, W) bool seed depth mask, or ``None``.
            extrinsics0: (1, 4, 4) seed user camera (w2c, already s_user-scaled).
            intrinsics0: (1, 3, 3) seed normalized intrinsics.
        """
        self._cache_scale_alignment_overlap(
            image_np.unsqueeze(0),
            depth0.unsqueeze(0),
            None if mask0 is None else mask0.unsqueeze(0),
            extrinsics0,
            intrinsics0,
            keep_last=1,
        )


    # ------------------------------------------------------------------
    # Trajectory scale + pose alignment (user trajectory → predictor scale)
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_traj_scale(
        extrinsics_w2c: torch.Tensor, s: float,
        da3_ext0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Map a rebased (frame-0 = identity) w2c user trajectory into the
        predictor's world frame.

        Two steps:
          1. Scale camera centers by ``s``. For w2c ``[R | t]`` the camera
             center is ``C = -R^T t``; ``t' = s t`` gives ``C' = s C`` with the
             rotation unchanged.
          2. Right-multiply by ``da3_ext0`` (DA3 frame-0 w2c) to align the
             user trajectory's origin to DA3's actual frame-0 pose:
             ``E_render_i = E_scaled_i @ da3_ext0``. At i=0 (identity) this
             yields exactly ``da3_ext0``. When ``da3_ext0`` is None, no pose
             alignment is applied (frame-0 assumed identity), i.e. the user
             camera frame is kept — the shared first-frame + ``align_with_user``
             behaviour.
        """
        ext = extrinsics_w2c.clone()
        ext[:, :3, 3] *= s
        if da3_ext0 is not None:
            ext = ext @ da3_ext0.to(ext)
        return ext

    def _rescale_user_trajectory(
        self,
        anchor_depth: torch.Tensor,
        da3_depth0: torch.Tensor,
        anchor_mask: Optional[torch.Tensor],
        da3_mask0: Optional[torch.Tensor],
        da3_ext0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Recompute ``s_user`` from ``(anchor, da3_depth0)``, cache the
        frame-0 pose, and return the scaled (+ optionally pose-aligned) user
        trajectory (derived from the unscaled rebased trajectory)."""
        self._s_user = compute_depth_scale(
            anchor_depth, da3_depth0, anchor_mask=anchor_mask, da3_mask=da3_mask0,
        )
        if da3_ext0 is not None:
            self._da3_ext0 = da3_ext0.detach().cpu().float()
        print(f"[WanWorldDepth] s_user = {self._s_user:.6f}")
        return self._apply_traj_scale(
            self._user_extrinsics_raw, self._s_user, self._da3_ext0,
        )

    # ------------------------------------------------------------------
    # Resize / format helpers
    # ------------------------------------------------------------------

    def _nearest_resize_to_mapper(self, arr_nhw: torch.Tensor) -> torch.Tensor:
        """Nearest-resize a (N, h, w) float tensor to the mapper resolution."""
        import cv2

        H, W = self.mapper_resolution
        if tuple(arr_nhw.shape[-2:]) == (H, W):
            return arr_nhw
        a = arr_nhw.detach().cpu().numpy().astype(np.float32)
        N = a.shape[0]
        out = np.zeros((N, H, W), dtype=np.float32)
        for i in range(N):
            out[i] = cv2.resize(a[i], (W, H), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(out).to(arr_nhw.device)

    @staticmethod
    def _frames_to_chw(frames_fhw3: torch.Tensor) -> torch.Tensor:
        """(F,H,W,3) [0,1] → (F,3,H,W) float for the depth predictor."""
        return frames_fhw3.permute(0, 3, 1, 2).contiguous().float()

    def _da3_depth_and_mask(self, out, frames_fhw3: torch.Tensor):
        """Resize predictor depth + confidence to the mapper resolution and
        build a depth mask matching the offline datapipe
        (``datapipe/wan_pipes/octmap_integration.py``):

          - per-chunk confidence percentile threshold (``self.conf_percentile``),
            computed over the valid (finite, positive-depth) confidence values
            of the frames returned by this chunk;
          - optional sky masking (``self.apply_sky_mask`` / ``self.sky_threshold``)
            using the ONNX sky segmenter, masking out sky pixels.

        Normalized intrinsics are resolution-independent and need no adjustment.

        Args:
            out: ``DepthPredictorOutput`` (depth/confidence at predictor res).
            frames_fhw3: (N, H, W, 3) float [0,1] RGB at the mapper resolution,
                used for sky segmentation.

        Returns:
            (depth, mask): depth (N, H, W) float, mask (N, H, W) bool, both at
            the mapper resolution.

        Side effect (for debugging): the separate confidence mask, sky mask,
        and resized confidence map are stashed on ``self._dbg_conf_mask`` /
        ``self._dbg_sky_mask`` / ``self._dbg_confidence`` (CPU tensors at the
        mapper resolution) so the pipeline can persist them per chunk.
        """
        depth_r = self._nearest_resize_to_mapper(out.depth)
        conf_r = (
            self._nearest_resize_to_mapper(out.confidence)
            if out.confidence is not None else None
        )

        depth_np = depth_r.detach().cpu().numpy()
        valid = np.isfinite(depth_np) & (depth_np > 0)

        # Confidence mask: validity (finite, positive depth + finite conf) AND,
        # if requested, the per-chunk confidence percentile threshold.
        conf_mask = valid.copy()
        conf_np = None
        if conf_r is not None:
            conf_np = conf_r.detach().cpu().numpy()
            valid &= np.isfinite(conf_np)
            conf_mask = valid.copy()
            if self.conf_percentile > 0 and valid.any():
                threshold = float(np.percentile(conf_np[valid], self.conf_percentile))
                conf_mask = valid & (conf_np >= threshold)

        # Sky mask: True = keep (non-sky). All-ones when sky masking is off.
        sky_mask = np.ones_like(conf_mask)
        if self.apply_sky_mask:
            from utils.sky_segmentation import segment_non_sky_masks

            frames_uint8 = (
                frames_fhw3.clamp(0, 1).detach().cpu().numpy() * 255
            ).astype(np.uint8)
            sky_mask = segment_non_sky_masks(
                frames_uint8, threshold=self.sky_threshold, device=str(self.device),
            )

        mask = conf_mask & sky_mask

        # Stash the separate masks (+ confidence) for optional per-chunk debug
        # dumping by the pipeline.
        self._dbg_conf_mask = torch.from_numpy(conf_mask.copy())
        self._dbg_sky_mask = torch.from_numpy(np.ascontiguousarray(sky_mask))
        self._dbg_confidence = (
            None if conf_np is None else torch.from_numpy(conf_np.copy())
        )

        return depth_r, torch.from_numpy(mask).to(depth_r.device)

    def _prepare_anchor_depth(
        self, anchor_depth: torch.Tensor, anchor_mask: Optional[torch.Tensor],
    ):
        """Nearest-resize the caller-provided anchor depth (+ mask) to the
        mapper resolution and derive a validity mask if none is given."""
        import cv2

        H, W = self.mapper_resolution
        anchor_depth = torch.as_tensor(anchor_depth, dtype=torch.float32)
        if anchor_mask is None:
            anchor_mask = torch.isfinite(anchor_depth) & (anchor_depth > 0)
        else:
            anchor_mask = torch.as_tensor(anchor_mask).bool()

        if tuple(anchor_depth.shape[-2:]) != (H, W):
            d = anchor_depth.cpu().numpy()
            m = anchor_mask.cpu().numpy().astype(np.uint8)
            d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            anchor_depth = torch.from_numpy(d)
            anchor_mask = torch.from_numpy(m)
        return anchor_depth, anchor_mask

    # ------------------------------------------------------------------
    # First-frame depth (shared by both methods)
    # ------------------------------------------------------------------

    def _predict_first_frame_depth(
        self, image_hw3: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-frame depth via the raw predictor (no streamer, no hint).

        Shared by both depth methods. With a single view, DA3's pose alignment
        is degenerate, so we only consume the metric depth; the map is seeded
        with the **user camera** (see the pipeline ``__call__``). The depth mask
        follows the same datapipe policy (conf percentile + sky mask).

        Returns ``(depth0, mask0)`` at the mapper resolution.
        """
        chw = image_hw3.permute(2, 0, 1).unsqueeze(0).float()  # (1,3,H,W)
        out = self._depth_predictor.predict(chw)
        depth, mask = self._da3_depth_and_mask(out, image_hw3.unsqueeze(0))
        return depth[0], mask[0]

    # ------------------------------------------------------------------
    # align_with_user per-chunk depth (non-streaming, user-camera aligned)
    # ------------------------------------------------------------------

    def _da3_chunk_depth_with_hint(
        self,
        gen_frames: torch.Tensor,
        chunk_extrs: torch.Tensor,
        chunk_intrs: torch.Tensor,
    ):
        """Single non-streaming predict with the user camera as a hint.

        ``align_to_input=True`` makes the predictor return the user camera
        verbatim and rescale depth to the user-camera metric scale, so the
        predictor effectively only *provides depth* aligned to the user
        trajectory.
        """
        imgs = self._frames_to_chw(gen_frames)
        out = self._depth_predictor.predict(
            imgs,
            hint_extrinsics=chunk_extrs.to(self.device),
            hint_intrinsics=chunk_intrs.to(self.device),
            align_to_input=True,
        )
        return self._da3_depth_and_mask(out, gen_frames)

    def _da3_chunk_depth_no_umeyama(
        self,
        gen_frames: torch.Tensor,
        chunk_extrs: torch.Tensor,
        chunk_intrs: torch.Tensor,
    ):
        """Single non-streaming predict with the user camera as a hint but
        **without** DA3's Umeyama pose alignment (``skip_pose_align=True``).

        The camera hint still conditions the model; the returned depth is in
        DA3's native scale (no ``depth /= scale``). The scale-alignment path
        recovers the world scale from chunk-to-chunk overlap, so this avoids the
        unstable Umeyama Sim(3) fit entirely.
        """
        imgs = self._frames_to_chw(gen_frames)
        out = self._depth_predictor.predict(
            imgs,
            hint_extrinsics=chunk_extrs.to(self.device),
            hint_intrinsics=chunk_intrs.to(self.device),
            skip_pose_align=True,
        )
        return self._da3_depth_and_mask(out, gen_frames)
    def _slice_debug_masks(self, start: int):
        """Keep debug tensors aligned with returned frame range."""
        for name in ("_dbg_conf_mask", "_dbg_sky_mask", "_dbg_confidence"):
            v = getattr(self, name)
            if v is not None:
                setattr(self, name, v[start:])

    def _cache_scale_alignment_overlap(
        self,
        frames: torch.Tensor,
        depths: torch.Tensor,
        masks: Optional[torch.Tensor],
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        keep_last: Optional[int] = None,
    ):
        """Cache frames from the current chunk as the overlap reference for the
        next scale-alignment chunk.

        ``keep_last`` bounds how many trailing frames are stored: ``None`` keeps
        all frames (``scale_alignment`` mode's full-overlap behaviour), while
        ``keep_last=1`` keeps only the last frame (used by the seed cache and the
        ``align_with_user`` fallback — a single overlap view is enough for a
        robust scalar and keeps memory bounded).
        """
        if keep_last is not None:
            frames = frames[-keep_last:]
            depths = depths[-keep_last:]
            masks = None if masks is None else masks[-keep_last:]
            extrinsics = extrinsics[-keep_last:]
            intrinsics = intrinsics[-keep_last:]
        self._scale_align_prev_rgb = frames.detach().cpu().float()
        self._scale_align_prev_depth = depths.detach().cpu().float()
        self._scale_align_prev_mask = (
            None if masks is None else masks.detach().cpu().bool()
        )
        self._scale_align_prev_extrs = extrinsics.detach().cpu().float()
        self._scale_align_prev_intrs = intrinsics.detach().cpu().float()

    def _compute_overlap_scale(
        self,
        *,
        ref_depth: torch.Tensor,
        pred_overlap_depth: torch.Tensor,
        ref_mask: Optional[torch.Tensor],
        pred_overlap_mask: Optional[torch.Tensor],
    ) -> float:
        """Estimate overlap scale ``s`` with fallback when masks are too sparse.

        Returns ``s`` such that ``s * ref_depth ~= pred_overlap_depth``.
        """
        ref_flat = ref_depth.reshape(-1)
        pred_flat = pred_overlap_depth.reshape(-1)
        ref_mask_flat = None if ref_mask is None else ref_mask.reshape(-1)
        pred_mask_flat = None if pred_overlap_mask is None else pred_overlap_mask.reshape(-1)
        try:
            return compute_depth_scale(
                ref_flat,
                pred_flat,
                anchor_mask=ref_mask_flat,
                da3_mask=pred_mask_flat,
            )
        except ValueError:
            return compute_depth_scale(ref_flat, pred_flat)
