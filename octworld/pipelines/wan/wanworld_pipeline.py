"""WanWorldPipeline — AR world generation with the Wan 14B FramePack backbone.

Default inference uses non-streaming DA3 ``align_with_user`` and online OctMap:
    - Rebase the user trajectory to frame 0. Fit ``s_user * anchor_depth ≈
        da3_depth0`` from a single-frame DA3 prediction, then scale only the user
        translations. The supported modes do not refine this scale on chunk 0.
    - Select keyframes before per-chunk depth prediction. DA3 receives the scaled
        user cameras as hints; the map uses those cameras, not DA3-predicted poses.
        Degenerate Umeyama alignment falls back to overlap scale alignment for that
        chunk. ``standalone`` is retained as legacy code but disabled at startup.
    - Integrate at 294x504 and render at 480x832 (H, W). DA3's process_res=504 and
        patch14 resizing produce 294x504 for 480x832 RGB. Depth/confidence first
        return to 480x832 for masking, then the mapper resizes depth for integration.
    - Warp rendering uses the scaled user trajectory; the raymap uses its
        stride-4 cameras, rebased to the current chunk's first camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import torch
from PIL import Image

from ..worldgen import WorldGenPipeline
from .wan_pipeline import WanMultiAdapterPipeline
from .wanworld_depth_prediction_mixin import (
    WanWorldDepthPredictionMixin, compute_depth_scale,
)
from .wanworld_output_mixin import WanWorldOutputMixin
from ...diffusion.scheduler import FlowMatchScheduler
from ...models.wan_checkpoint import load_wan_checkpoint, resolve_exp_root
from utils.camera import get_plucker_ray, transpose_to_first_frames
from utils.random_state import CaseRandomStreams, seed_everything

# ``compute_depth_scale`` is re-exported from the depth mixin for backward
# compatibility (e.g. validate/wanworld_traj_render_check.py imports it here).
__all__ = ["WanWorldPipeline", "WanWorldPipeConfig", "compute_depth_scale"]


# Wan I2V-14B default negative prompt (used for classifier-free guidance).
WAN_NEGATIVE_PROMPT = (
    "镜头晃动，色调艳丽，过曝，动态，细节模糊不清，字幕，风格，作品，画作，画面，运动物体，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
    "三条腿，背景人很多，倒着走"
)

# Wan FramePack frame-count constants (latent space).
GEN_FRAMES = 20          # generation latent frames per chunk
N_HISTORY = 19           # latent history frames before generation
N_HISTORY_TOTAL = 20     # anchor(1) + history(19)
CHUNK_VIDEO_FRAMES = 80  # pixel frames per chunk (= GEN_FRAMES * 4)


@dataclass
class WanWorldPipeConfig:
    """Inference settings; dimensions are (H, W), not width/height.

    Defaults preserve the evaluated online/noise0 configuration. Model package
    architecture settings do not override these runtime values. No GPU/model
    availability is implied by constructing this dataclass.
    """

    device: str = "cuda"
    # Strict checkpoint loading replaces initialized parameters. Still seed
    # construction for buffers/third-party initialization and repeatable setup.
    # The example sets this to --seed; library callers can set it explicitly.
    init_seed: int = 0
    torch_dtype: torch.dtype = torch.bfloat16
    num_inference_steps: int = 35
    cfg_scale: float = 5.0
    cfg_merged: bool = False
    cpu_offload: bool = False
    shift: float = 3.0
    mapper_mode: str = "octmap"
    # RGB/warp render size and intermediate depth/confidence masking size;
    # this is NOT the DA3 prediction size or the OctMap integration size.
    mapper_resolution: tuple[int, int] = (480, 832)
    conf_percentile: float = 20.0
    apply_sky_mask: bool = True
    sky_threshold: int = 32
    neg_text_embed_path: Optional[str] = None
    # History AND rendered-latent augmentation bucket (0..10). Both 0 and None
    # keep conditions clean, with history_timestep=None; no extra noise is drawn.
    noise_aug_level: Optional[int] = 0
    # Depth backend selection (WanWorldDepthPredictionMixin):
    #   - "standalone":      legacy DA3 streaming; rejected before model loading.
    #   - "align_with_user": non-streaming, map uses the user camera; the
    #                         predictor only provides depth aligned to it.
    #   - "scale_alignment": non-streaming DA3 predict with overlap-based
    #                         chunk-to-chunk depth scale transfer.
    depth_method: str = "align_with_user"
    depth_predictor_name: str = "da3"
    # Fixed evaluated integration size, matching DA3 output for 480x832 RGB
    # with process_res=504 / upper_bound_resize / patch14; NOT inferred at runtime.
    # None integrates at mapper_resolution. For tsdf, explicitly set None.
    integration_resolution: Optional[tuple[int, int]] = (294, 504)
    # Online incrementally integrates each new chunk. Offline re-integrates all
    # cached frames after every chunk and is retained as an explicit override.
    offline_integration: bool = False
    octree_depth_strategy: str = "pixel_size_to_volume_size"
    # Used only by the "bigger_voxels" carve strategy.
    bigger_voxel_pixel_size_scale: float = 2.0
    # Used only by uniform selection; keyframe selection ignores this value.
    sample_ratio: int = 1
    # Frame selection for depth prediction / map integration:
    #   - "uniform":  keep every ``sample_ratio``-th frame (stride sampling).
    #   - "keyframe": RAFT-Large motion-gated keyframe selection (stateful
    #                 across chunks); see utils.keyframe.RaftKeyframeSelector.
    frame_select_method: str = "keyframe"
    # Mean flow in preprocessed RAFT pixels (480x832 at the default RGB size).
    keyframe_thresh: float = 5.0
    # Skipped-frame counter threshold, checked before increment: 8 can yield
    # a 9-frame index gap. Every selection block also keeps its last frame.
    keyframe_max_interval: int = 8


class WanWorldPipeline(WanWorldOutputMixin, WanWorldDepthPredictionMixin, WorldGenPipeline):
    """AR world generation pipeline using the Wan 14B FramePack backbone.

    Composes a :class:`WanMultiAdapterPipeline` as ``self.world_generator`` and
    drives non-streaming DA3 via :class:`WanWorldDepthPredictionMixin`
    (``align_with_user`` by default, or ``scale_alignment``). The depth
    strategy only predicts depth + chooses cameras; this pipeline owns the 3D
    mapper integration and the AR buffers (see ``__call__``).
    """

    def __init__(
        self,
        dit_ckpt_path: str = None,
        model_weights_dir: str = "model_weights",
        debug: bool = False,
        config: Optional[WanWorldPipeConfig] = None,
    ):
        cfg = config or WanWorldPipeConfig()
        self._validate_depth_config(cfg)
        self._seed_everything(cfg.init_seed)
        self._initialization_seed = int(cfg.init_seed)
        torch.backends.cudnn.benchmark = False
        device = cfg.device
        torch_dtype = cfg.torch_dtype
        neg_text_embed_path = cfg.neg_text_embed_path
        self.offline_integration = bool(cfg.offline_integration)
        self.noise_aug_level = cfg.noise_aug_level
        self.octree_depth_strategy = str(cfg.octree_depth_strategy)
        self.bigger_voxel_pixel_size_scale = float(cfg.bigger_voxel_pixel_size_scale)
        self.sample_ratio = int(cfg.sample_ratio)
        self.frame_select_method = str(cfg.frame_select_method)
        self.keyframe_thresh = float(cfg.keyframe_thresh)
        self.keyframe_max_interval = int(cfg.keyframe_max_interval)
        integration_resolution = (
            tuple(cfg.integration_resolution)
            if cfg.integration_resolution is not None else None
        )
        if integration_resolution is not None:
            if cfg.mapper_mode != "octmap":
                raise ValueError(
                    "integration_resolution is only supported for mapper_mode='octmap'"
                )
            if len(integration_resolution) != 2 or any(
                int(size) <= 0 for size in integration_resolution
            ):
                raise ValueError(
                    "integration_resolution must contain two positive dimensions, "
                    f"got {integration_resolution!r}"
                )
        if self.bigger_voxel_pixel_size_scale <= 0:
            raise ValueError(
                "bigger_voxel_pixel_size_scale must be > 0, "
                f"got {self.bigger_voxel_pixel_size_scale}"
            )
        if self.sample_ratio <= 0:
            raise ValueError(f"sample_ratio must be >= 1, got {self.sample_ratio}")
        if self.frame_select_method not in ("uniform", "keyframe"):
            raise ValueError(
                "frame_select_method must be 'uniform' or 'keyframe', "
                f"got {self.frame_select_method!r}"
            )
        if self.keyframe_thresh <= 0:
            raise ValueError(
                f"keyframe_thresh must be > 0, got {self.keyframe_thresh}")
        if self.keyframe_max_interval < 1:
            raise ValueError(
                f"keyframe_max_interval must be >= 1, got {self.keyframe_max_interval}")
        # Built lazily on first use (only when frame_select_method == 'keyframe').
        self._keyframe_selector = None

        # ── 0. Debug mode: load the tiny model for a fast fake run ───────
        # In debug mode the heavy 14B DiT (33GB, loaded from blob) is replaced
        # with the local tiny checkpoint under debugs/tiny-octworld-wan-tiny-b4
        # so the rollout plumbing can be exercised quickly.
        if debug:
            tiny_dir = Path("debugs/tiny-octworld-wan-tiny-b4")
            dit_ckpt_path = str(tiny_dir / "ckpts" / "1000" / "model_state_dict.pt")
            print(f"[WanWorldPipeline] DEBUG mode: tiny model from {tiny_dir}")
        else:
            assert dit_ckpt_path is not None, (
                "dit_ckpt_path is required when debug=False"
            )

        # ``pipe_config.json`` + ``neg_text_embed.pt`` live at the experiment
        # root, resolved by walking up from the checkpoint (see
        # :meth:`_resolve_exp_root`).
        exp_root = self._resolve_exp_root(dit_ckpt_path)
        if neg_text_embed_path is None:
            cand = exp_root / "neg_text_embed.pt"
            if cand.exists():
                neg_text_embed_path = str(cand)

        # ── 1. Build the Wan generator from a training checkpoint ────────
        self.world_generator = self._build_generator_from_ckpt(
            dit_ckpt_path, torch.device(device), torch_dtype,
        )
        # VAE / T5 / CLIP for inference (never used by training). When
        # ``cpu_offload`` is set, the VAE stays on the GPU but the T5 / CLIP
        # conditioning encoders are parked on CPU and only moved to the GPU for
        # the duration of each encode call (see WanToolsMixin).
        self.world_generator.cpu_offload = bool(cfg.cpu_offload)
        self.world_generator.load_inference_modules(model_weights_dir, device=device)
        if cfg.cpu_offload:
            self.world_generator._move_inference_module(self.world_generator.t5, "cpu")
            self.world_generator._move_inference_module(self.world_generator.clip, "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Inference uses the UniPC multistep sampler (faster than euler at the
        # same step count); training keeps its own euler scheduler.
        self.world_generator.scheduler = FlowMatchScheduler(
            template="Wan", sample_mode="unipc",
        )

        # ── 2. WorldGenPipeline infra (mapper, AR buffers) ──────────────
        WorldGenPipeline.__init__(
            self,
            mapper_mode=cfg.mapper_mode,
            mapper_resolution=cfg.mapper_resolution,
            mapper_device=device,
            chunk_size=CHUNK_VIDEO_FRAMES,
            camera_scale=1.0,  # unused: Wan raymap matches the dataset (no scale)
            mapper_init_kwargs=self._build_mapper_init_kwargs(cfg),
            mapper_update_kwargs=self._build_mapper_update_kwargs(cfg),
        )
        self.integration_resolution = integration_resolution

        # ── 3. Depth backend (non-streaming predictor) ──────────────────
        # Construction, masks, and scale state live in the depth mixin.
        # Both supported modes are non-streaming; standalone is disabled.
        self._init_depth_backend(cfg, device)

        # ── 4. Sampling config ──────────────────────────────────────────
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.num_inference_steps = int(cfg.num_inference_steps)
        self.cfg_scale = float(cfg.cfg_scale)
        self.cfg_merged = bool(cfg.cfg_merged)
        self.shift = float(cfg.shift)

        # Optional cached negative text embedding (for CFG).
        self._neg_text_embed_path = neg_text_embed_path

        # ── State buffers (set per call / per chunk) ────────────────────
        self._frame_latents: Optional[torch.Tensor] = None   # (1,16,L,H',W')
        self._pos_ctx: Optional[List[torch.Tensor]] = None
        self._neg_ctx: Optional[List[torch.Tensor]] = None
        self._clip_features: Optional[torch.Tensor] = None    # (1,n_tokens,1280)
        self._extrinsics: Optional[torch.Tensor] = None
        self._intrinsics: Optional[torch.Tensor] = None

        # Trajectory scale + pose alignment state (``_s_user``,
        # ``_user_extrinsics_raw``, ``_da3_ext0``) is initialised by
        # ``_init_depth_backend`` and managed by WanWorldDepthMixin.

    def _build_mapper_init_kwargs(self, cfg: WanWorldPipeConfig) -> Dict[str, Any]:
        if cfg.mapper_mode != "octmap":
            return {}
        from spatial_memory.mapper.octmap.octmap_mapper import IntegrationConfig

        return {
            "integration_config": IntegrationConfig(
                carve_octree_depth_strategy=cfg.octree_depth_strategy,
                carve_bigger_voxel_pixel_size_scale=cfg.bigger_voxel_pixel_size_scale,
                integration_mode="offline" if cfg.offline_integration else "online",
            )
        }

    def _build_mapper_update_kwargs(self, cfg: WanWorldPipeConfig) -> Dict[str, Any]:
        if cfg.mapper_mode != "octmap":
            return {}
        # OctMapMapper.map_frames takes ``integration_mode`` ("online"/"offline"),
        # not the old ``reintegrate`` bool (removed in the octmap refactor).
        return {"integration_mode": "offline" if cfg.offline_integration else "online"}

    def _sample_frame_indices(self, total_frames: int) -> torch.Tensor:
        stride = max(1, self.sample_ratio)
        return torch.arange(0, total_frames, stride, dtype=torch.long)

    def _keyframe_indices(self, frames: torch.Tensor) -> torch.Tensor:
        """RAFT motion-gated keyframe indices for ``frames`` (N, H, W, 3).

        The selector is stateful across chunks (reference frame + interval
        counter carry over); it is reset per rollout in ``__call__``.
        """
        # RAFT creates random parameters on its first select before loading
        # trained weights. First-use vs cached execution must not shift DA3 RNG.
        with self._auxiliary_random_scope("keyframe"):
            if self._keyframe_selector is None:
                from utils.keyframe import RaftKeyframeSelector

                self._keyframe_selector = RaftKeyframeSelector(
                    threshold=self.keyframe_thresh,
                    max_interval=self.keyframe_max_interval,
                    device=str(self.device),
                )
            return self._keyframe_selector.select(frames)

    def _select_sampled_inputs(
        self,
        frames: torch.Tensor,
        chunk_extrs: torch.Tensor,
        chunk_intrs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.frame_select_method == "keyframe":
            sample_idx = self._keyframe_indices(frames)
        else:
            sample_idx = self._sample_frame_indices(int(frames.shape[0]))
        return (
            frames.index_select(0, sample_idx),
            chunk_extrs.index_select(0, sample_idx),
            chunk_intrs.index_select(0, sample_idx),
            sample_idx,
        )

    # ------------------------------------------------------------------
    # Generator construction (compatibility wrappers for the shared loader)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_exp_root(ckpt_path: str) -> Path:
        """Delegate experiment-root lookup to the shared checkpoint helper."""
        return resolve_exp_root(ckpt_path)

    @staticmethod
    def _build_generator_from_ckpt(
        ckpt_path: str, device: torch.device, dtype: torch.dtype,
    ) -> WanMultiAdapterPipeline:
        """Delegate strict, model-only checkpoint loading to the shared helper."""
        return load_wan_checkpoint(ckpt_path, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Mapper reset (OctMap has no in-place reset → recreate)
    # ------------------------------------------------------------------

    def _reset_mapper(self):
        self._mapper = None
        self._init_mapper()

    # ------------------------------------------------------------------
    # History latents (anchor + 19), mirroring WanFramePackDataset
    # ------------------------------------------------------------------

    def _build_history_latents(self) -> torch.Tensor:
        """Build (1, 16, 20, H', W') = [anchor(1) | history(19)].

        ``self._frame_latents`` = [anchor | gen0 | gen1 | ...]. The 19 history
        frames are the latents immediately preceding the (new) generation; when
        fewer exist they are zero-padded in front (matching the dataset).
        """
        fl = self._frame_latents
        anchor = fl[:, :, :1]
        T = fl.shape[2]
        hist = fl[:, :, max(0, T - N_HISTORY):]  # up to 19 frames
        n_real = hist.shape[2]
        if n_real < N_HISTORY:
            pad = torch.zeros(
                1, fl.shape[1], N_HISTORY - n_real, fl.shape[3], fl.shape[4],
                dtype=fl.dtype,
            )
            hist = torch.cat([pad, hist], dim=2)
        return torch.cat([anchor, hist], dim=2)  # (1,16,20,H',W')

    # ------------------------------------------------------------------
    # Raymap (stride-4 cameras, relative to chunk start) — matches dataset
    # ------------------------------------------------------------------

    def _build_raymap(
        self, chunk_extrs: torch.Tensor, chunk_intrs: torch.Tensor,
    ) -> torch.Tensor:
        """Plucker raymap (1, 6, 20, H, W) from the chunk's stride-4 cameras."""
        ext = chunk_extrs[::4][:GEN_FRAMES].float()   # (20, 4, 4)
        intr = chunk_intrs[::4][:GEN_FRAMES].float()  # (20, 3, 3)
        ext = transpose_to_first_frames(ext)          # relative to chunk frame 0
        H, W = self.mapper_resolution
        rays = get_plucker_ray(ext.unsqueeze(0), intr.unsqueeze(0), H=H, W=W)
        return rays[0].permute(3, 0, 1, 2).unsqueeze(0).contiguous()  # (1,6,20,H,W)

    # ------------------------------------------------------------------
    # Abstract hooks
    # ------------------------------------------------------------------

    def init_first_frame(
        self,
        image: Image.Image,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        prompt: str,
    ):
        """Encode the anchor frame, cache prompts + CLIP features."""
        self._extrinsics = extrinsics
        self._intrinsics = intrinsics

        image_fhw3 = torch.tensor(np.array(image), dtype=torch.float32) / 255.0  # (H,W,3)

        # Anchor RGB latent (frame 0). encode_frames_to_latents wants (F,H,W,3).
        self._frame_latents = self.world_generator.encode_frames_to_latents(
            image_fhw3.unsqueeze(0)
        )  # (1, 16, 1, H', W')

        # Text + CLIP conditioning.
        self._pos_ctx = self.world_generator.encode_prompt(prompt)
        self._neg_ctx = self._load_or_encode_neg()
        self._clip_features = self.world_generator.encode_clip_features(image_fhw3)

    def _load_or_encode_neg(self) -> List[torch.Tensor]:
        if self._neg_text_embed_path is not None:
            data = torch.load(self._neg_text_embed_path, map_location="cpu")
            # Saved as a dict {"text_embed": (L,4096)} (see validate.py), or a
            # bare tensor / list of tensors.
            if isinstance(data, dict) and "text_embed" in data:
                t = data["text_embed"]
            else:
                t = data
            if isinstance(t, (list, tuple)):
                return [x.to(dtype=self.torch_dtype) for x in t]
            return [t.to(dtype=self.torch_dtype)]
        return self.world_generator.encode_prompt(WAN_NEGATIVE_PROMPT)

    @staticmethod
    def _fold_rendered_mask(
        warp_mask: torch.Tensor, h_lat: int, w_lat: int,
    ) -> torch.Tensor:
        """Fold a per-pixel-frame validity mask into the Wan I2V 4-channel ``m``.

        Mirrors ``datapipe/wan_pipes/vae_encode_rendered.py`` +
        ``WanFramePackDataset``: conservatively (min-pool) downsamples the
        (80, H, W) validity mask to the latent spatial resolution, then folds
        the 80 pixel frames into 20 latent frames x 4 pixel-frame channels.

        Returns float (1, 4, GEN_FRAMES, h_lat, w_lat).
        """
        m = warp_mask
        if m.dim() == 4:                       # (80, 1, H, W) -> (80, H, W)
            m = m.squeeze(1)
        m = (m > 0.5)
        T_, Hm, Wm = m.shape
        assert Hm % h_lat == 0 and Wm % w_lat == 0, (
            f"mask HxW {(Hm, Wm)} not divisible by latent {(h_lat, w_lat)}")
        assert T_ == GEN_FRAMES * 4, (
            f"expected {GEN_FRAMES * 4} mask frames, got {T_}")
        fh, fw = Hm // h_lat, Wm // w_lat
        # Min-pool over each fh x fw block (visible only if all pixels visible).
        m = m.reshape(T_, h_lat, fh, w_lat, fw)
        m = m.all(dim=4).all(dim=2)            # (80, h_lat, w_lat)
        # 80 pixel frames -> 20 latent frames x 4 pixel-frame channels.
        m = m.reshape(GEN_FRAMES, 4, h_lat, w_lat)   # (20, 4, h, w)
        m = m.permute(1, 0, 2, 3).contiguous()        # (4, 20, h, w)
        return m.unsqueeze(0).float()                  # (1, 4, 20, h, w)

    def world_gen_step(
        self,
        warp_color: torch.Tensor,
        warp_depth: torch.Tensor,
        warp_mask: torch.Tensor,
        chunk_prompt: str = "",
        *,
        chunk_extrs: torch.Tensor = None,
        chunk_intrs: torch.Tensor = None,
        chunk_idx: int = 0,
        case_seed: int = 0,
        sampling_generator: Optional[torch.Generator] = None,
        augmentation_generator: Optional[torch.Generator] = None,
        noise_snapshot_dir: Optional[Path] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run one Wan diffusion chunk; returns generated RGB + sampled latents.

        Depth / camera (for the map update) are handled by ``__call__`` via the
        selected non-streaming DA3 method, so this only produces RGB.
        ``noise_snapshot_dir`` is the run output directory, initialized by
        ``_begin_output_run``. Actual noise/conditions are hashed immediately
        before latent_sampling; the JSON is not a reconstruction of the RNG.
        """
        device, dtype = self.device, self.torch_dtype
        H, W = self.mapper_resolution

        # ── Encode warp RGB (prepend prev real frame → 81 → drop first) ──
        prev_frame = self.all_frames[-1:].cpu().float()              # (1,H,W,3)
        warp_color_fhw3 = warp_color.permute(0, 2, 3, 1).cpu().float()  # (80,H,W,3)
        rgb_81 = torch.cat([prev_frame, warp_color_fhw3], dim=0)     # (81,H,W,3)
        rendered_rgb_lat = self.world_generator.encode_frames_to_latents(
            rgb_81)[:, :, 1:1 + GEN_FRAMES]                          # (1,16,20,H',W')

        # ── Rendered validity mask → 4-channel ``m`` (min-pool, no VAE) ──
        # Mirror datapipe/vae_encode_rendered: conservatively (min-pool) reduce
        # the per-pixel-frame validity mask to the latent spatial resolution,
        # then fold the 80 pixel frames into 20 latent frames x 4 channels.
        H_lat, W_lat = rendered_rgb_lat.shape[3], rendered_rgb_lat.shape[4]
        rendered_masks = self._fold_rendered_mask(
            warp_mask, H_lat, W_lat).to(device=device, dtype=dtype)  # (1,4,20,H',W')

        # ── History + raymap + prompt + noise ───────────────────────────
        history = self._build_history_latents()                    # (1,16,20,H',W')
        raymap = self._build_raymap(chunk_extrs, chunk_intrs)      # (1,6,20,H,W)

        if chunk_prompt and chunk_prompt != "":
            pos_ctx = self.world_generator.encode_prompt(chunk_prompt)
        else:
            pos_ctx = self._pos_ctx

        H_lat, W_lat = self._frame_latents.shape[3], self._frame_latents.shape[4]
        noise_shape = (1, 16, GEN_FRAMES, H_lat, W_lat)
        if sampling_generator is None:
            raise ValueError("sampling_generator must be initialized by __call__")
        # Historical v2 demo contract: one continuous CPU FP32 stream per case,
        # then cast to the model dtype/device. Do not draw directly in BF16 or
        # reseed per chunk; either would change the recorded latent-noise hashes.
        noise = self.world_generator.generate_noise(
            noise_shape, generator=sampling_generator,
            rand_device="cpu", rand_torch_dtype=torch.float32,
            device=device, torch_dtype=dtype)

        # ── CFG sampling (merged: pos/neg in one batched forward) ───────
        raymap = raymap.to(device=device, dtype=dtype)
        pos_ctx_d = [c.to(device=device, dtype=dtype) for c in pos_ctx]
        neg_ctx_d = [c.to(device=device, dtype=dtype) for c in self._neg_ctx]

        # ── Pre-noise the history/rendered condition ONCE (inference) ───
        # Noise augmentation must be applied a SINGLE time per chunk and then
        # held fixed across every denoising step and both CFG branches. If we
        # instead let ``model_fn`` re-noise on each call (its training-time
        # behavior), the conditioning would jitter each step and differ between
        # the positive/negative branches, corrupting the guidance direction. So
        # we noise here via the pipeline's shared static helper and pass
        # ``prenoised_history=True`` so ``model_fn`` uses the condition as-is.
        history_d = history.to(device=device, dtype=dtype)
        rendered_d = rendered_rgb_lat.to(device=device, dtype=dtype)
        history_noise = rendered_noise = None
        if self.noise_aug_level is not None and int(self.noise_aug_level) > 0:
            if augmentation_generator is None:
                raise ValueError(
                    "augmentation_generator must be initialized by __call__")
            history_cond, rendered_cond, history_ts, history_noise, rendered_noise = (
                WanMultiAdapterPipeline.apply_history_noise_aug(
                    history_d, rendered_d, self.noise_aug_level,
                    getattr(self.world_generator.scheduler,
                            "num_train_timesteps", 1000),
                    generator=augmentation_generator,
                    return_noises=True,
                )
            )
        else:
            history_cond, rendered_cond, history_ts = history_d, rendered_d, None

        # Merged CFG batches the positive/negative conditioning along the batch
        # dim so a single forward yields both predictions. It trades VRAM
        # (doubled batch) for speed, so it is opt-in via ``cfg_merged`` and only
        # meaningful when guidance is active (cfg_scale ∉ {0, 1}); otherwise fall
        # back to the plain single-branch path.
        use_merged = self.cfg_merged and self.cfg_scale not in (0.0, 1.0)
        if use_merged:
            dup = lambda t: torch.cat([t, t], dim=0)
            input_shared = dict(
                history_latents=dup(history_cond),
                rendered_latents=dup(rendered_cond),
                rendered_masks=dup(rendered_masks),
                clip_features=dup(self._clip_features.to(device=device, dtype=dtype)),
                history_timestep=(dup(history_ts) if history_ts is not None else None),
                prenoised_history=True,
            )
            input_pos = dict(
                context=pos_ctx_d + neg_ctx_d,
                raymap=torch.cat([raymap, torch.zeros_like(raymap)], dim=0),
            )
            input_neg = {}
        else:
            input_shared = dict(
                history_latents=history_cond,
                rendered_latents=rendered_cond,
                rendered_masks=rendered_masks,
                clip_features=self._clip_features.to(device=device, dtype=dtype),
                history_timestep=history_ts,
                prenoised_history=True,
            )
            input_pos = dict(context=pos_ctx_d, raymap=raymap)
            input_neg = dict(
                context=neg_ctx_d, raymap=torch.zeros_like(raymap),
            )

        self.world_generator.scheduler.set_timesteps(
            num_inference_steps=self.num_inference_steps, shift=self.shift)
        # Capture THESE tensors, then pass the same noise object to sampling.
        # Hashing/serialization does not advance either dedicated RNG stream.
        self._record_sampling_snapshot(
            noise_snapshot_dir, chunk_idx, case_seed, noise, sampling_generator,
            augmentation_generator, history_noise, rendered_noise,
            input_shared, input_pos["raymap"],
        )
        sampled = self.world_generator.latent_sampling(
            noise, cfg_scale=self.cfg_scale, cfg_merged=use_merged,
            input_shared=input_shared, input_pos=input_pos, input_neg=input_neg,
        )  # (1,16,20,H',W')

        # ── Decode (prepend k history latents for temporal continuity) ──
        k = 1 if chunk_idx == 0 else 4
        hist_tail = self._frame_latents[:, :, -k:]
        decode_in = torch.cat([hist_tail.to(sampled), sampled], dim=2)
        decoded = self.world_generator.decode_latents_to_frames(decode_in)  # (F,H,W,3)
        new_frames = decoded[-CHUNK_VIDEO_FRAMES:].clamp(0, 1)              # (80,H,W,3)

        return {"frames": new_frames, "latents": sampled.detach().cpu().float()}

    # ------------------------------------------------------------------
    # Rollout (overrides WorldGenPipeline.__call__)
    # ------------------------------------------------------------------

    @staticmethod
    def _seed_everything(seed: int) -> None:
        """Fix all RNG seeds (torch / torch.cuda / numpy / random) for a case."""
        seed_everything(seed)

    @torch.inference_mode()
    def __call__(
        self,
        image: Image.Image,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        anchor_depth: torch.Tensor,
        anchor_mask: Optional[torch.Tensor] = None,
        prompt="",
        save_dir: Optional[str] = None,
        preview_only: bool = False,
        verbose: bool = True,
        max_chunks: Optional[int] = None,
        seed: int = 0,
    ) -> Dict[str, Any]:
        """Run AR world generation.

        Args:
            image: PIL first-frame image.
            extrinsics: (T, 4, 4) user camera trajectory (w2c). Defines the
                trajectory shape; its metric scale is set by ``anchor_depth``.
            intrinsics: (T, 3, 3) normalized intrinsics.
            anchor_depth: (H, W) first-frame depth that defines the metric
                scale of the user trajectory. The user-trajectory translations
                are scaled so this matches DA3's predicted frame-0 depth.
            anchor_mask: optional (H, W) bool validity mask for ``anchor_depth``.
            prompt: str or list[str] (per-chunk).
            save_dir: empty output directory. While running, chunks/ holds each
                chunk's output/depth/warp MP4s. Success leaves three final MP4s
                and sha256.json with actual per-chunk sampling-input hashes.
            preview_only: if True, only render the warp along the whole
                trajectory (no diffusion) and return.
            max_chunks: if set, cap the rollout to this many chunks (debug).
            seed: RNG seed fixed across ``torch`` / ``torch.cuda`` / ``numpy`` /
                ``random`` before running this case, for reproducibility.

        Returns CPU RGB/depth/masks in full frame order. Returned cameras retain
        the existing keyframe/map indexing; dense export buffers never replace
        ``all_depths`` / ``all_masks`` or the overlap cache used by AR.
        As in the historical rollout, a final chunk after chunk 0 skips depth
        prediction and integration: its output depth is zero with an all-false
        mask (black in depth.mp4). A one-chunk run retains the bootstrap.
        """
        from tqdm import tqdm

        # ── Fix all RNG seeds for this case (reproducible generation) ────
        self._seed_everything(seed)
        self._case_random_streams = CaseRandomStreams(seed)
        self._random_namespace = "inference"
        # Keep these two legacy streams byte-compatible with old noise records.
        sampling_generator = torch.Generator(device="cpu").manual_seed(int(seed))
        augmentation_generator = None
        if self.noise_aug_level is not None and int(self.noise_aug_level) > 0:
            augmentation_generator = torch.Generator(device=self.device).manual_seed(
                int(seed) + 1)

        # ── First-frame rebase (frame 0 → world origin) ─────────────────
        extrinsics = transpose_to_first_frames(extrinsics.float())
        intrinsics = intrinsics.float()

        # ── Pad trajectory to 1 + chunk_size * n ────────────────────────
        num_frames = extrinsics.shape[0]
        if (num_frames - 1) % self.chunk_size != 0:
            pad = self.chunk_size - (num_frames - 1) % self.chunk_size
            pad_idx = list(range(num_frames)) + [num_frames - 1] * pad
            extrinsics = extrinsics[pad_idx]
            intrinsics = intrinsics[pad_idx]
            num_frames = extrinsics.shape[0]
        num_chunks = (num_frames - 1) // self.chunk_size
        if max_chunks is not None:
            num_chunks = min(num_chunks, int(max_chunks))
        if num_chunks < 1:
            raise ValueError("At least one chunk is required")

        if isinstance(prompt, list):
            chunk_prompts = (prompt + [prompt[-1]] * num_chunks)[:num_chunks]
        else:
            chunk_prompts = [prompt] * num_chunks

        self._begin_output_run(save_dir, seed, num_chunks, preview=preview_only)

        # ── Reset AR + depth state ──────────────────────────────────────
        self.all_frames = self.all_depths = self.all_masks = None
        self.all_extrinsics = self.all_intrinsics = None
        self._output_depths = self._output_masks = None
        self._reset_depth_state()
        self._reset_mapper()
        if self._keyframe_selector is not None:
            self._keyframe_selector.reset()

        # ── Anchor depth → mapper resolution (nearest) ──────────────────
        anchor_depth, anchor_mask = self._prepare_anchor_depth(anchor_depth, anchor_mask)

        # ── Encode anchor / prompts ─────────────────────────────────────
        self.init_first_frame(image, extrinsics, intrinsics, chunk_prompts[0])

        # ── Seed frame 0 (shared: single-frame predict, user-camera integrate)
        # Both depth methods use the same first-frame path: predict depth with
        # the raw predictor (no streamer, no hint) and integrate with the USER
        # camera. The trajectory is scaled (scale-only, ``da3_ext0=None``) so the
        # user frame stays the world origin. ``standalone``'s chunk-0 bootstrap
        # later rebases onto DA3's frame-0 and refines the scale.
        image_np = torch.tensor(np.array(image), dtype=torch.float32) / 255.0
        depth0, mask0 = self._predict_first_frame_depth(image_np)

        self._user_extrinsics_raw = extrinsics.clone()
        extrinsics = self._rescale_user_trajectory(
            anchor_depth, depth0, anchor_mask, mask0, da3_ext0=None)

        self._seed_first_frame(
            image_np=image_np, depth=depth0, depth_mask=mask0,
            extrinsics=extrinsics, intrinsics=intrinsics,
        )
        # Seed the camera buffers with frame-0's user camera (``standalone``'s
        # chunk-0 bootstrap overwrites these with the 81-frame DA3 result).
        self.all_extrinsics = extrinsics[:1].detach().cpu().float()
        self.all_intrinsics = intrinsics[:1].detach().cpu().float()

        # Seed the scale-alignment overlap cache with frame 0 so the first chunk
        # can recover its depth scale by aligning to ``depth0`` (the frame that
        # defined ``s_user``) instead of Umeyama. This is what makes the
        # ``align_with_user`` Umeyama fallback and the ``scale_alignment`` mode
        # work on chunk 0. See WanWorldDepthPredictionMixin._seed_scale_align_cache.
        self._seed_scale_align_cache(
            image_np=image_np, depth0=depth0, mask0=mask0,
            extrinsics0=extrinsics[:1], intrinsics0=intrinsics[:1],
        )

        # ── Preview mode: render whole trajectory, no diffusion ─────────
        if preview_only:
            warp = self._render_chunk_warp(extrinsics[1:], intrinsics[1:])
            preview_frames = warp["color"].permute(0, 2, 3, 1)
            preview_depths = warp["depth"].squeeze(1)
            preview_masks = warp["mask"].squeeze(1) > 0.5
            self._save_chunk_warp(save_dir, 0, warp["color"], warp["mask"])
            self._save_chunk_outputs(save_dir, 0, preview_frames, preview_depths, preview_masks)
            self._finalize_outputs(save_dir, preview_frames, preview_depths, preview_masks)
            return {"preview": warp}

        # ── AR rollout ──────────────────────────────────────────────────
        chunk_iter = range(num_chunks)
        if verbose:
            chunk_iter = tqdm(chunk_iter, desc="WanWorld chunks")

        for c in chunk_iter:
            chunk_start = 1 + self.chunk_size * c
            chunk_extrs = extrinsics[chunk_start:chunk_start + self.chunk_size]
            chunk_intrs = intrinsics[chunk_start:chunk_start + self.chunk_size]
            skip_depth = c > 0 and c == num_chunks - 1

            # 1. render warp (scaled USER cameras)
            warp = self._render_chunk_warp(chunk_extrs, chunk_intrs)
            # Persist before diffusion so the live chunk exposes its condition.
            # Prepend only the first anchor for frame-aligned final videos.
            self._save_chunk_warp(
                save_dir, c, warp["color"], warp["mask"],
                anchor_frame=image_np if c == 0 else None,
            )

            # 2. diffuse (Wan) — raymap uses the same (scaled) user cameras
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gen = self.world_gen_step(
                warp["color"], warp["depth"], warp["mask"], chunk_prompts[c],
                chunk_extrs=chunk_extrs, chunk_intrs=chunk_intrs, chunk_idx=c,
                case_seed=seed, noise_snapshot_dir=save_dir,
                sampling_generator=sampling_generator,
                augmentation_generator=augmentation_generator,
            )
            gen_frames = gen["frames"].detach().cpu()    # (80,H,W,3)
            sampled = gen["latents"]      # (1,16,20,H',W')
            del gen, warp

            # 3. Depth prediction + map integration + AR buffers. The first chunk
            #    rebuilds the map/buffers from the 81-frame ``[frame0 | gen0]``
            #    stack (and, for ``standalone``, refines the trajectory scale);
            #    later chunks predict the 80 generated frames and append. Depth
            #    itself is delegated to ``predict_depth_fn``; everything else
            #    (mapper, buffers, trajectory) is owned here.
            if c == 0:
                full_frames = torch.cat([image_np.unsqueeze(0), gen_frames], dim=0)
                pred_frames, pred_extrs, pred_intrs, sample_idx = self._select_sampled_inputs(
                    full_frames,
                    extrinsics[:1+self.chunk_size],
                    intrinsics[:1+self.chunk_size],
                )
                res = self.predict_depth_fn(
                    frames=pred_frames,
                    chunk_extrs=pred_extrs,
                    chunk_intrs=pred_intrs,
                    chunk_idx=c,
                )
                depth_r, mask_r = res["depths"], res["masks"]
                map_extrs, map_intrs = res["map_extrinsics"], res["map_intrinsics"]
                # ``standalone`` refines s_user + rebases the user trajectory onto
                # DA3's frame-0 from the consistent 81-frame chunk-0 result, then
                # freezes the user trajectory for the rest of the rollout.
                if self.depth_method == "standalone":
                    extrinsics = self._rescale_user_trajectory(
                        anchor_depth, depth_r[0], anchor_mask, mask_r[0],
                        da3_ext0=map_extrs[0])
                # Reset the map + replace the single seed frame's buffers.
                self._reset_mapper()
                self._update_map(
                    frames=pred_frames, depths=depth_r,
                    extrinsics=map_extrs, intrinsics=map_intrs, depth_mask=mask_r)
                self.all_frames = full_frames
                self.all_depths = depth_r.cpu()
                self.all_masks = mask_r.cpu()
                self.all_extrinsics = map_extrs.detach().cpu().float()
                self.all_intrinsics = map_intrs.detach().cpu().float()
                output_frames = full_frames
                output_extrs = extrinsics[:1+self.chunk_size]
                output_intrs = intrinsics[:1+self.chunk_size]
            else:
                if skip_depth:
                    # No later warp consumes this chunk: match the old path,
                    # including skipping RAFT selection and overlap-cache updates.
                    depth_r = torch.zeros(gen_frames.shape[:3], dtype=torch.float32)
                    mask_r = torch.zeros(gen_frames.shape[:3], dtype=torch.bool)
                    map_extrs, map_intrs = chunk_extrs, chunk_intrs
                else:
                    pred_frames, pred_extrs, pred_intrs, sample_idx = self._select_sampled_inputs(
                        gen_frames, chunk_extrs, chunk_intrs)
                    res = self.predict_depth_fn(
                        frames=pred_frames, chunk_extrs=pred_extrs,
                        chunk_intrs=pred_intrs, chunk_idx=c)
                    depth_r, mask_r = res["depths"], res["masks"]
                    map_extrs, map_intrs = res["map_extrinsics"], res["map_intrinsics"]
                    self._update_map(
                        frames=pred_frames, depths=depth_r,
                        extrinsics=map_extrs, intrinsics=map_intrs, depth_mask=mask_r)
                self.all_frames = torch.cat([self.all_frames, gen_frames], dim=0)
                self.all_depths = torch.cat([self.all_depths, depth_r.cpu()], dim=0)
                self.all_masks = torch.cat([self.all_masks, mask_r.cpu()], dim=0)
                self.all_extrinsics = torch.cat(
                    [self.all_extrinsics, map_extrs.detach().cpu().float()], dim=0)
                self.all_intrinsics = torch.cat(
                    [self.all_intrinsics, map_intrs.detach().cpu().float()], dim=0)
                output_frames, output_extrs, output_intrs = gen_frames, chunk_extrs, chunk_intrs

            # accumulate latents (for history of the next chunk)
            self._frame_latents = torch.cat([self._frame_latents, sampled], dim=2)

            # Do not reintroduce final-chunk DA3 work through export-only depth.
            # Masked placeholders preserve RGB/depth indexing without pretending
            # that the unneeded final depth has been predicted.
            if skip_depth:
                dense_depth, dense_mask = depth_r, mask_r
            else:
                dense_depth, dense_mask = self._dense_output_depth(
                    output_frames, output_extrs, output_intrs, sample_idx, depth_r, mask_r)
            self._output_depths = (dense_depth if c == 0 else torch.cat(
                [self._output_depths, dense_depth], dim=0))
            self._output_masks = (dense_mask if c == 0 else torch.cat(
                [self._output_masks, dense_mask], dim=0))
            self._save_chunk_outputs(
                save_dir, c, output_frames, dense_depth, dense_mask, skip_depth=skip_depth)
            del dense_depth, dense_mask
            if verbose:
                print(f"  Chunk {c}: total {self.all_frames.shape[0]} frames")

        # ── Final output ────────────────────────────────────────────────
        self._finalize_outputs(
            save_dir, self.all_frames, self._output_depths, self._output_masks)
        all_frames_uint8 = (self.all_frames.clamp(0, 1) * 255).to(torch.uint8)

        return {
            "frames": all_frames_uint8,
            "depths": self._output_depths,
            "masks": self._output_masks,
            "sha256": self._output_manifest,
            "extrinsics": self.all_extrinsics,
            "intrinsics": self.all_intrinsics,
        }

