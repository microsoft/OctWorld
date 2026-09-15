"""WanMultiAdapterPipeline — Wan 14B FramePack model and inference components.

Loads fine-tuned weights through ``octworld.models.wan_checkpoint``. Holds:
- DiT (OctWorldWanModel) with FramePack layout
- Camera adapter (SimpleAdapter) for Plucker ray conditioning

Responsibilities:
- ``model_fn``: single forward pass with clean inputs (history_latents,
  rendered_latents/masks, etc.)
- Model state dict (dit + camera_adapter)

VAE / T5 / CLIP are loaded separately by ``load_inference_modules``. Release
inference never initializes the DiT from original Wan transformer weights;
training initialization is not supported in this release.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from ...diffusion.diffusion_pipeline import DiffusionPipeline
from ...diffusion.scheduler import FlowMatchScheduler
from ...models.wan.camera_adapter import SimpleAdapter
from ...models.wan.wan_framepack import OctWorldWanModel
from .wan_utils import WanToolsMixin


# ---------------------------------------------------------------------------
# History-condition noise augmentation
# ---------------------------------------------------------------------------
# The history condition (all history frames except the anchor/start) is
# optionally noise-augmented in flow-match ``sigma`` space. The augmentation
# level is bucketed into ``NOISE_AUG_NUM_LEVELS`` discrete steps in
# ``[0, NOISE_AUG_MAX_ALPHA]``: level ``k`` -> ``alpha = k * step`` where
# ``step = NOISE_AUG_MAX_ALPHA / (NOISE_AUG_NUM_LEVELS - 1)``. With the defaults
# (11 levels, max 0.7) the steps are 0.0, 0.07, ..., 0.7. ``alpha`` is used both
# as the flow-match noise level (``x_aug = (1-alpha) x0 + alpha eps``) and — via
# ``alpha * num_train_timesteps`` — as the timestep fed to the DiT time
# embedding for the history tokens.
NOISE_AUG_NUM_LEVELS = 11
NOISE_AUG_MAX_ALPHA = 0.7


class WanMultiAdapterPipeline(WanToolsMixin, DiffusionPipeline):
    """Wan 14B FramePack pipeline with a camera adapter.

    Components:
        - dit: OctWorldWanModel (14B DiT with FramePack)
        - camera_adapter: SimpleAdapter (Plucker rays → dim-D features)

    Release inference constructs the model architecture and loads a fine-tuned
    checkpoint via ``octworld.models.wan_checkpoint.load_wan_checkpoint``.
    """

    def __init__(
        self,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=16,
            width_division_factor=16,
            time_division_factor=4,
            time_division_remainder=1,
        )
        # Architecture is built by build_training_model; trained weights are
        # restored by load_wan_checkpoint (no original transformer loading).
        self.dit = None
        self.camera_adapter = None

        # Inference-only modules — populated by load_inference_modules(), never
        # touched by training. Plain attributes (not nn.Modules) so they do NOT
        # register as submodules → no FSDP / checkpoint / parameters() impact.
        self.vae = None
        self.t5 = None
        self.clip = None

        # When True, inference wrappers (VAE/T5/CLIP) live on CPU and are moved
        # to the GPU only for the duration of each encode/decode call. Set by
        # the inference pipeline; training leaves it False.
        self.cpu_offload = False

        self.scheduler = FlowMatchScheduler(template="Wan")

    # ------------------------------------------------------------------
    # build_training_model — construct DiT + camera adapter
    # ------------------------------------------------------------------

    def build_training_model(
        self,
        config: dict,
        *,
        use_gradient_checkpointing: bool = True,
        use_selective_checkpointing: bool = False,
        compile_blocks: bool = False,
        activation_memory_budget: float = 0.5,
    ):
        """Build DiT + camera adapter for training.

        Args:
            config: dict with 'dit' and 'camera_adapter' sub-dicts.
                - config['dit']: OctWorldWanModel constructor kwargs.
                - config['camera_adapter']: SimpleAdapter constructor kwargs.
            use_gradient_checkpointing: enable manual activation checkpointing.
                Ignored when ``compile_blocks`` is set (the memory-budget path
                drives recompute instead).
            use_selective_checkpointing: with manual AC on, recompute only the
                cheap per-block ops and keep flash-attention/matmul outputs
                saved (SAC). Ignored when ``compile_blocks`` is set.
            compile_blocks: regionally ``torch.compile`` each DiT block and use
                the compile-only Memory Budget API for selective recompute.
            activation_memory_budget: budget in [0, 1] for the compile path
                (0 = plain AC, 1 = default compile). Only used with
                ``compile_blocks``.
        """
        self.config = dict(config)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_selective_checkpointing = use_selective_checkpointing

        self.dit = OctWorldWanModel(**self.config['dit'])
        self.camera_adapter = SimpleAdapter(**self.config['camera_adapter'])

        # Defer torch.compile until AFTER the checkpoint load (see
        # ``maybe_compile_blocks``). Compiling wraps each block in an
        # ``OptimizedModule`` which inserts a ``_orig_mod.`` prefix into the
        # parameter FQNs; the init full-state-dict broadcast load matches on
        # clean (un-prefixed) keys, so compiling first breaks the load. Record
        # the intent here and apply it post-load instead.
        self._compile_blocks = bool(compile_blocks)
        self._activation_memory_budget = activation_memory_budget

        if self._compile_blocks:
            # Memory-budget-driven recompute (via the compiled min-cut
            # partitioner) replaces manual activation checkpointing: keep manual
            # AC OFF so the two do not stack (a checkpointed region forces full
            # recompute and the budget becomes moot).
            self.dit.disable_gradient_checkpointing()
        elif use_gradient_checkpointing:
            self._enable_gradient_checkpointing(
                selective=use_selective_checkpointing)

    def maybe_compile_blocks(self):
        """Apply deferred regional ``torch.compile`` to the DiT blocks.

        Called by the trainer *after* checkpoint load + materialize so the load
        sees clean parameter FQNs (compile inserts an ``_orig_mod.`` prefix).
        ``torch.compile`` is lazy (it compiles on the first forward), so
        applying it here — once the model is materialized on device — is
        functionally identical to compiling earlier, minus the state-dict key
        mismatch.
        """
        if getattr(self, "_compile_blocks", False) and hasattr(
                self.dit, "compile_blocks"):
            self.dit.compile_blocks(
                activation_memory_budget=self._activation_memory_budget)

    # ------------------------------------------------------------------
    # Gradient checkpointing
    # ------------------------------------------------------------------

    def _enable_gradient_checkpointing(self, selective: bool = False):
        """Enable gradient checkpointing on DiT blocks (optionally selective)."""
        if hasattr(self.dit, "enable_gradient_checkpointing"):
            self.dit.enable_gradient_checkpointing(selective=selective)

    # ------------------------------------------------------------------
    # FSDP2 sharding
    # ------------------------------------------------------------------

    def apply_fsdp(self, mesh, *, dtype: torch.dtype = torch.bfloat16):
        """Apply per-block FSDP2 sharding.

        Sharding granularity: each transformer block, the DiT itself, the
        camera adapter, and finally the pipeline (top-level module) itself.

        The pipeline is its own top-level ``nn.Module`` — there is no external
        wrapper — so the final ``fully_shard`` targets ``self``.

        Args:
            mesh: DeviceMesh for FSDP.
            dtype: parameter dtype for mixed-precision policy.
        """
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

        # cast_forward_inputs=False: the per-block ``freqs`` kwarg is a complex
        # RoPE tensor that must not be downcast to bf16 at the block-forward
        # boundary (that would corrupt the complex math). Token inputs are
        # already bf16 from the parent unit, and the time-modulation signal
        # ``e`` is cast to the compute dtype inside the DiT forward (the fp32
        # timestep embedding is the single fp32 island), so nothing here needs
        # FSDP's input casting.
        mp = MixedPrecisionPolicy(
            param_dtype=dtype, reduce_dtype=torch.float32,
            cast_forward_inputs=False)
        for block in self.dit.blocks:
            fully_shard(block, mesh=mesh, mp_policy=mp)
        fully_shard(self.dit, mesh=mesh, mp_policy=mp)
        fully_shard(self.camera_adapter, mesh=mesh, mp_policy=mp)
        fully_shard(self, mesh=mesh, mp_policy=mp)

    # ------------------------------------------------------------------
    # materialize — recompute non-persistent buffers after meta-init
    # ------------------------------------------------------------------

    def materialize(self):
        """Recompute deterministic non-persistent tensors after meta-device
        initialization (e.g. after ``to_empty`` / DCP broadcast load).

        Only the DiT's RoPE ``freqs`` buffer is non-persistent (it is excluded
        from every state dict), so it must be regenerated here. All trainable
        parameters — including the zero-init ``camera_adapter.proj_out`` and the
        tiled ``clean_x_embedder`` — are carried by the checkpoint and are
        restored by the checkpoint load, so they are intentionally left
        untouched (re-initializing them would wipe trained weights on resume).
        """
        self.dit.materialize()

    # ------------------------------------------------------------------
    # load_inference_modules — VAE / T5 / CLIP (inference only)
    # ------------------------------------------------------------------

    def load_inference_modules(
        self,
        model_weights_dir: str,
        device: str = "cuda",
        *,
        vae_dtype: torch.dtype = torch.float32,
        t5_dtype: torch.dtype = torch.bfloat16,
        clip_dtype: torch.dtype = torch.float16,
    ):
        """Load VAE / T5 / CLIP for inference (never called by the trainer).

        These are stored as plain attributes (``self.vae`` / ``self.t5`` /
        ``self.clip``) consumed by ``WanToolsMixin`` helpers. They are not
        ``nn.Module`` submodules of this pipeline, so they have zero impact on
        FSDP / checkpoint I/O / ``parameters()``.

        Args:
            model_weights_dir: base dir containing ``Wan2.1-I2V-14B-480P/``.
            device: device to place the modules on.
            vae_dtype / t5_dtype / clip_dtype: per-module compute dtype.
        """
        from ...models.wan_model_manager import WanModelManager

        mgr = WanModelManager(base_path=model_weights_dir)
        self.vae = mgr.get_vae_wrapper(device=device, dtype=vae_dtype)
        self.t5 = mgr.get_component("t5", torch_dtype=t5_dtype, device=device)
        self.clip = mgr.get_component("clip", torch_dtype=clip_dtype, device=device)
        return self

    # ------------------------------------------------------------------
    # model_fn — single forward with clean inputs
    # ------------------------------------------------------------------

    def model_fn(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        *,
        history_latents: torch.Tensor,
        rendered_latents: torch.Tensor,
        rendered_masks: torch.Tensor,
        context: List[torch.Tensor],
        raymap: Optional[torch.Tensor] = None,
        clip_features: Optional[torch.Tensor] = None,
        noise_aug_level: Optional[torch.Tensor] = None,
        history_timestep: Optional[torch.Tensor] = None,
        prenoised_history: bool = False,
    ) -> torch.Tensor:
        """Single denoising step with clean inputs (mirrors FramePack style).

        Rendered-warp conditioning is carried entirely by the Wan I2V channels:
        the rendered RGB latent goes into ``y`` and the rendered validity mask
        goes into the ``m`` (mask) slot. There is no separate warp adapter.

        Args:
            noisy_latents: (B, 16, T_gen, H, W) noised generation latents.
            timestep: (B,) diffusion timestep (0~1000 scale).
            history_latents: (B, 16, T_hist, H, W) clean history/anchor latents
                ordered as ``[start | history]`` (start = first/anchor frame).
            rendered_latents: (B, 16, T_gen, H, W) rendered RGB warp latents.
            rendered_masks: (B, 4, T_gen, H, W) rendered validity mask at latent
                spatial resolution. The 4 channels are the mask of the 4 pixel
                frames that make up each latent frame.
            context: list of (L, 4096) text embeddings per sample.
            raymap: (B, 6, T_gen, H_vid, W_vid) Plucker rays or None.
            clip_features: (B, n_tokens, clip_dim) CLIP features or None.
            noise_aug_level: optional history noise-augmentation bucket level(s).
                Scalar or (B,) integer tensor in ``[0, NOISE_AUG_NUM_LEVELS)``.
                ``None`` disables augmentation (history stays clean). Applied to
                every history frame **except** the anchor/start, and to the
                rendered RGB latents. Used ONLY on the training path
                (``prenoised_history=False``): the augmentation noise is freshly
                sampled on each call (correct for training, which calls this
                once per step). Ignored when ``prenoised_history=True``.
            history_timestep: optional (B,) history-condition timestep (0~1000
                scale) that modulates the (noised) history tokens. Only consumed
                when ``prenoised_history=True`` — the inference caller derives it
                once (via :meth:`apply_history_noise_aug`) alongside the noised
                condition. On the training path it is recomputed internally and
                this argument is ignored.
            prenoised_history: inference-only flag (default ``False``). When
                ``False`` (training), ``model_fn`` performs the noise
                augmentation itself. When ``True``, ``history_latents`` /
                ``rendered_latents`` are assumed to be ALREADY noised once by the
                caller and are used as-is together with the caller-provided
                ``history_timestep``; no fresh noise is drawn here. This keeps the
                conditioning fixed across denoising steps / CFG branches at
                inference. **Never set by the trainer**, so training is
                unaffected.

        Returns:
            (B, 16, T_gen, H, W) velocity prediction for generation frames.
        """
        B = noisy_latents.shape[0]
        T_hist = history_latents.shape[2]
        T_gen = noisy_latents.shape[2]
        H, W = noisy_latents.shape[3], noisy_latents.shape[4]
        device = noisy_latents.device
        dtype = noisy_latents.dtype

        # Ensure timestep is 1D and float32.
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0).expand(B)
        timestep = timestep.to(device=device, dtype=torch.float32)

        # ---- History-condition noise augmentation ----
        # Training (default, ``prenoised_history=False``): freshly sample the
        # augmentation noise on THIS call and noise the history (except the
        # anchor/start) + rendered latents via the shared static helper,
        # deriving the matching ``history_timestep``. Correct for training,
        # where each forward is an independent single denoising step.
        #
        # Inference (``prenoised_history=True``): the caller has already noised
        # ``history_latents`` / ``rendered_latents`` ONCE and supplies the
        # matching ``history_timestep``; we must NOT re-noise here (re-sampling
        # every denoising step / CFG branch would jitter the conditioning and
        # corrupt guidance). Use the inputs + timestep as-is.
        history_cond = history_latents
        rendered_cond = rendered_latents
        if prenoised_history:
            # Use caller-provided (already noised) condition + timestep as-is.
            pass
        elif noise_aug_level is not None:
            history_cond, rendered_cond, history_timestep = (
                self.apply_history_noise_aug(
                    history_latents, rendered_latents, noise_aug_level,
                    getattr(self.scheduler, "num_train_timesteps", 1000),
                )
            )
        else:
            history_timestep = None

        # Assemble full input: x = [start | history | gen], y = [start | history
        # | rendered]. The history part is shared (identical, possibly noised)
        # between x and y.
        x_in = torch.cat([history_cond, noisy_latents], dim=2)
        y = torch.cat([history_cond, rendered_cond], dim=2)

        # Mask (the ``m`` slot): all-ones on the anchor/history segment, the
        # rendered validity mask on the generation segment.
        mask_hist = torch.ones(B, 4, T_hist, H, W, device=device, dtype=dtype)
        mask = torch.cat([mask_hist, rendered_masks.to(device=device, dtype=dtype)],
                         dim=2)

        camera_features = self.camera_adapter(raymap)

        # Forward through DiT.
        pred = self.dit(
            x_in, timestep, context,
            y=y,
            mask=mask,
            clip_features=clip_features,
            camera_features=camera_features,
            history_timestep=history_timestep,
        )
        return pred

    @staticmethod
    def apply_history_noise_aug(
        history_latents: torch.Tensor,
        rendered_latents: torch.Tensor,
        noise_aug_level,
        num_train_timesteps,
        *,
        generator: Optional[torch.Generator] = None,
        return_noises: bool = False,
    ):
        """Noise-augment the history (except anchor/start) + rendered latents.

        Single source of truth for the augmentation math, shared by:
          - :meth:`model_fn` (training): called once per forward → fresh noise
            per step is the intended data augmentation.
          - the world-gen inference pipeline: called ONCE per chunk *before* the
            sampling loop, then the returned (already noised) condition is held
            fixed across all denoising steps and both CFG branches (pass
            ``prenoised_history=True`` to ``model_fn``).

        Freshly samples noise on every call. ``num_train_timesteps`` scales the
        returned ``history_timestep`` into the DiT's 0~1000 time-embedding range.

        Returns ``(history_cond, rendered_cond, history_timestep)``.
        """
        B = history_latents.shape[0]
        device = history_latents.device
        dtype = history_latents.dtype
        alpha = WanMultiAdapterPipeline._noise_aug_alpha(
            noise_aug_level, B, device)                 # (B,) f32
        a = alpha.view(B, 1, 1, 1, 1).to(dtype)
        start = history_latents[:, :, :1]
        rest = history_latents[:, :, 1:]
        history_noise = torch.randn(
            rest.shape, generator=generator, device=rest.device, dtype=rest.dtype)
        rendered_noise = torch.randn(
            rendered_latents.shape, generator=generator,
            device=rendered_latents.device, dtype=rendered_latents.dtype)
        rest = (1 - a) * rest + a * history_noise
        history_cond = torch.cat([start, rest], dim=2)
        rendered_cond = (
            (1 - a) * rendered_latents + a * rendered_noise)
        history_timestep = alpha * float(num_train_timesteps)
        if return_noises:
            return (history_cond, rendered_cond, history_timestep,
                    history_noise, rendered_noise)
        return history_cond, rendered_cond, history_timestep

    @staticmethod
    def _noise_aug_alpha(noise_aug_level, B: int, device) -> torch.Tensor:
        """Convert a scalar/(B,) bucket level to a (B,) float32 ``alpha`` tensor."""
        step = NOISE_AUG_MAX_ALPHA / (NOISE_AUG_NUM_LEVELS - 1)
        if not torch.is_tensor(noise_aug_level):
            level = torch.full((B,), float(noise_aug_level), device=device)
        else:
            level = noise_aug_level.to(device=device, dtype=torch.float32)
            if level.dim() == 0:
                level = level.expand(B)
        return (level.float() * step).clamp_(0.0, NOISE_AUG_MAX_ALPHA)

