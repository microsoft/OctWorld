# OctWorld Wan FramePack Model
# FramePack injection aligned with Hunyuan FramePack
# (octworld/models/framepack/models/hunyuan_video_packed.py).
#
# Schedule (in latent frames, history first, then generation):
#   clean_start (1) + clean_4x (16) + clean_2x (2) + clean_1x (1) + gen (20) = 40
# History segments are patchified by ``clean_x_embedder`` with kernels
# (1, 4, 2, 1); positional encoding for each segment is built at the
# generation post-patch grid then average-pooled by the same kernel so it
# matches the Conv3d output layout.
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .modules.attention import flash_attention
from .modules.model import (
    WanAttentionBlock,
    WanSelfAttention,
    MLPProj,
    Head,
    sinusoidal_embedding_1d,
    rope_params,
)

__all__ = ['OctWorldWanModel']

# ----------------------------------------------------------------------
# Selective activation checkpointing (SAC)
# ----------------------------------------------------------------------
# When manual activation checkpointing is enabled we can optionally recompute
# only the *cheap* ops in a block and keep the expensive ones saved, instead of
# recomputing the whole block. This trades a little extra activation memory for
# a large cut in recompute FLOPs. Only ops that go through ``torch_dispatch``
# can be saved.
#
# This mirrors the ``aggressive`` (blog "Policy 2") mode benchmarked in
# ``explore/bench_wan_block_ac.py`` (with ``--aggressive-include-attn``): save
# every compute-intensive aten op AND this model's *custom* flash-attn op
# (``flash_attn_3::*``), which the aten list alone misses because attention
# dispatches to a custom op rather than aten SDPA/flash. In practice, inside a
# transformer block this saves the Linear matmuls (mm/addmm) and the
# flash-attention output, recomputing LayerNorm / AdaLN / elementwise.
try:
    from torch.utils.checkpoint import (
        CheckpointPolicy,
        create_selective_checkpoint_contexts,
    )
    _SAC_AVAILABLE = True
except ImportError:  # pragma: no cover - torch < 2.4
    _SAC_AVAILABLE = False


def _build_sac_compute_intensive_ops():
    """The aggressive 'Policy 2' compute-intensive aten op set.

    Built defensively: overloads absent in this torch build are skipped. NOTE:
    this model's self/cross attention dispatches to a *custom* flash-attn op
    (``flash_attn_3::*``), not the aten SDPA/flash entries below, so those aten
    attention ops will not match here — the custom op is saved separately (see
    ``_sac_policy_fn``).
    """
    aten = torch.ops.aten
    names = [
        ("mm", "default"),
        ("convolution", "default"),
        ("convolution_backward", "default"),
        ("bmm", "default"),
        ("addmm", "default"),
        ("_scaled_dot_product_flash_attention", "default"),
        ("_scaled_dot_product_efficient_attention", "default"),
        ("_flash_attention_forward", "default"),
        ("_efficient_attention_forward", "default"),
        ("upsample_bilinear2d", "default"),
        ("_scaled_mm", "default"),
    ]
    ops = []
    for packet, overload in names:
        pkt = getattr(aten, packet, None)
        if pkt is None:
            continue
        ov = getattr(pkt, overload, None)
        if ov is not None:
            ops.append(ov)
    return set(ops)


_SAC_COMPUTE_INTENSIVE_OPS = (
    _build_sac_compute_intensive_ops() if _SAC_AVAILABLE else set())


def _sac_policy_fn(ctx, op, *args, **kwargs):
    # Save every compute-intensive aten op, plus this model's custom flash-attn
    # op (aten list misses it). Recompute everything else.
    save = op in _SAC_COMPUTE_INTENSIVE_OPS or "flash_attn" in str(op)
    if save:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def _sac_context_fn():
    """``context_fn`` for ``torch.utils.checkpoint.checkpoint`` (SAC)."""
    return create_selective_checkpoint_contexts(_sac_policy_fn)


# ----------------------------------------------------------------------
# Fixed FramePack schedule
# ----------------------------------------------------------------------
CLEAN_START_FRAMES = 1
CLEAN_4X_FRAMES = 16
CLEAN_2X_FRAMES = 2
CLEAN_1X_FRAMES = 1
N_HISTORY_FRAMES = (
    CLEAN_START_FRAMES + CLEAN_4X_FRAMES + CLEAN_2X_FRAMES + CLEAN_1X_FRAMES  # 20
)
GEN_FRAMES = 20
TOTAL_LATENT_FRAMES = N_HISTORY_FRAMES + GEN_FRAMES  # 40


def pad_for_3d_conv(x: torch.Tensor, kernel_size: Tuple[int, int, int]) -> torch.Tensor:
    """Replicate-pad x so its T/H/W are multiples of the conv kernel."""
    _, _, t, h, w = x.shape
    pt, ph, pw = kernel_size
    pad_t = (pt - (t % pt)) % pt
    pad_h = (ph - (h % ph)) % ph
    pad_w = (pw - (w % pw)) % pw
    if pad_t == 0 and pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode='replicate')


def _avg_pool3d_complex(x: torch.Tensor, kernel: int) -> torch.Tensor:
    """Average-pool a complex tensor of shape [1, C, T, H, W] by ``kernel`` on each of T,H,W."""
    real = F.avg_pool3d(x.real, kernel, stride=kernel)
    imag = F.avg_pool3d(x.imag, kernel, stride=kernel)
    return torch.complex(real, imag)


# ----------------------------------------------------------------------
# Packed (per-token) RoPE for FramePack
# ----------------------------------------------------------------------
@torch.amp.autocast('cuda', enabled=False)
def rope_apply_packed(x: torch.Tensor, freqs_packed: torch.Tensor) -> torch.Tensor:
    """Apply per-token complex RoPE.

    Args:
        x: ``[B, L, num_heads, head_dim]``
        freqs_packed: ``[B, L, head_dim/2]`` complex
    """
    b, l, n, d = x.shape
    x_complex = torch.view_as_complex(
        x.to(torch.float64).reshape(b, l, n, d // 2, 2)
    )                                          # [B, L, n, d/2]
    f = freqs_packed.unsqueeze(2)              # [B, L, 1, d/2]
    out = torch.view_as_real(x_complex * f).flatten(3)
    return out.type_as(x)


class WanFramepackSelfAttention(WanSelfAttention):
    """Self-attention that uses packed (per-token) RoPE.

    Reuses ``WanSelfAttention``'s parameters/state-dict layout; only the
    forward differs. The ``freqs`` argument is interpreted as a packed
    ``[B, L, head_dim/2]`` complex tensor and ``grid_sizes`` is ignored.
    """

    def forward(self, x, seq_lens, grid_sizes, freqs):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        # ``seq_lens`` here is constructed as a constant full-length tensor
        # (no padding: every sample spans the entire packed sequence). Passing
        # ``k_lens=None`` is numerically identical but lets ``flash_attention``
        # take its ``flatten`` fast path, avoiding the per-sample ``u[:v]``
        # slicing that forces a CUDA->CPU ``.item()`` sync on every block.
        x = flash_attention(
            q=rope_apply_packed(q, freqs),
            k=rope_apply_packed(k, freqs),
            v=v,
            k_lens=None,
            window_size=self.window_size,
        )
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanPatchEmbedForCleanLatents(nn.Module):
    """Multi-scale Conv3d patch embedder for FramePack history segments.

    ``in_channels`` must match the main ``patch_embedding`` of the model so
    weights can be tiled directly. For Wan I2V it is 36 = x(16)+mask(4)+y(16).
    """

    def __init__(self, inner_dim: int, in_channels: int,
                 patch_size: Tuple[int, int, int] = (1, 2, 2)):
        super().__init__()
        pt, ph, pw = patch_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.proj = nn.Conv3d(
            in_channels, inner_dim, kernel_size=(pt, ph, pw), stride=(pt, ph, pw))
        self.proj_2x = nn.Conv3d(
            in_channels, inner_dim,
            kernel_size=(pt * 2, ph * 2, pw * 2),
            stride=(pt * 2, ph * 2, pw * 2))
        self.proj_4x = nn.Conv3d(
            in_channels, inner_dim,
            kernel_size=(pt * 4, ph * 4, pw * 4),
            stride=(pt * 4, ph * 4, pw * 4))

    @torch.no_grad()
    def initialize_weight_from_another_conv3d(self, base_conv: nn.Conv3d):
        """Tile pretrained patch_embedding weights into proj/proj_2x/proj_4x.

        Requires ``base_conv.weight.shape[1] == self.in_channels``.
        """
        assert base_conv.weight.shape[1] == self.in_channels, (
            f"clean_x_embedder.in_channels={self.in_channels} but base_conv "
            f"has {base_conv.weight.shape[1]} input channels"
        )
        weight = base_conv.weight.detach().clone()
        bias = base_conv.bias.detach().clone() if base_conv.bias is not None else None
        sd = {
            'proj.weight': weight.clone(),
            'proj_2x.weight': repeat(
                weight, 'o c t h w -> o c (t tk) (h hk) (w wk)', tk=2, hk=2, wk=2) / 8.0,
            'proj_4x.weight': repeat(
                weight, 'o c t h w -> o c (t tk) (h hk) (w wk)', tk=4, hk=4, wk=4) / 64.0,
        }
        if bias is not None:
            sd['proj.bias'] = bias.clone()
            sd['proj_2x.bias'] = bias.clone()
            sd['proj_4x.bias'] = bias.clone()
        self.load_state_dict(sd)


class OctWorldWanModel(nn.Module):
    """Wan 14B with Hunyuan-style FramePack for autoregressive video generation.

    Differences vs base Wan 14B:
    - ``patch_embedding`` matches Wan I2V exactly: ``[x | mask | y]`` =
      ``in_dim + 4 + i2v_in_dim`` channels, so pretrained weights load 1:1.
    - Caller supplies ``y`` (i2v_in_dim ch) and ``mask`` (4 ch) explicitly at
      full T. Two conventions are supported by this model out of the box:
        - Wan-original: ``y = encode([first_frame, zeros])``, ``mask`` is 1 at
          frame 0 and 0 elsewhere.
        - OctWorld rendered: ``y = [anchor/history latent | rendered_rgb_latent]``
          and ``mask`` carries the **rendered warp validity mask** (4 channels
          = the 4 pixel frames per latent frame) on the generation segment and
          all-ones on the anchor/history segment.
    - ``clean_x_embedder`` is a multi-scale Conv3d patch embedder (kernels
      1/2/4) sharing input channels with ``patch_embedding``; weights are
      tiled from ``patch_embedding`` at init.
    - Rendered-warp conditioning is injected entirely through the Wan I2V
      channels: the rendered RGB latent lives in ``y`` and the rendered
      validity mask lives in ``mask`` (the ``m`` slot). There is no separate
      warp adapter.
    - Self-attention uses per-token packed RoPE so history segments at their
      native (frame, h, w) coordinates get correct positional encoding.
    - History-condition noise augmentation is supported via per-token time
      modulation: history tokens carry a noise timestep (``history_timestep``)
      distinct from the diffusion timestep on the generation tokens.

    The FramePack layout is fixed: (clean_start=1, clean_4x=16, clean_2x=2,
    clean_1x=1, gen=20) latent frames, total 40.
    """

    # Backward-compat sentinel kept only because init_checkpoint / training
    # module pass ``framepack_type='f1k1f16k4f2k2f1k1_g20'`` as a config arg.
    _SUPPORTED_FRAMEPACK_TYPE = 'f1k1f16k4f2k2f1k1_g20'

    def __init__(
        self,
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=5120,
        ffn_dim=13824,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=40,
        num_layers=40,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        # I2V conditioning (Wan 2.1 I2V / OctWorld rendered style):
        #   - per-frame y latent (anchor + rendered RGB) channel-concat
        #     to generation latent
        #   - CLIP image features prepended to text context via MLPProj
        i2v_in_dim: int = 16,
        clip_dim: int = 1280,
        framepack_type: str = _SUPPORTED_FRAMEPACK_TYPE,
    ):
        super().__init__()
        assert framepack_type == self._SUPPORTED_FRAMEPACK_TYPE, (
            f"OctWorldWanModel only supports framepack_type="
            f"{self._SUPPORTED_FRAMEPACK_TYPE!r}, got {framepack_type!r}"
        )

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.framepack_type = framepack_type
        self.i2v_in_dim = int(i2v_in_dim)
        assert self.i2v_in_dim > 0, "T2V mode removed; i2v_in_dim must be > 0."
        self.clip_dim = int(clip_dim)

        # Schedule constants exposed for inspection.
        self.gen_frames = GEN_FRAMES
        self.total_latent_frames = TOTAL_LATENT_FRAMES

        # ---- Core embeddings ----
        # patch_embedding matches Wan I2V exactly: [x | mask(4) | y].
        self.mask_in_dim = 4
        patch_in_channels = in_dim + self.mask_in_dim + self.i2v_in_dim
        self.patch_in_channels = patch_in_channels
        self.patch_embedding = nn.Conv3d(
            patch_in_channels,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # ---- CLIP image projection (Wan 2.1 I2V MLPProj) ----
        self.img_emb = MLPProj(self.clip_dim, dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # ---- Clean patch embedder (FramePack history segments) ----
        # Shares input channels with patch_embedding so weights can be tiled
        # directly via ``copy_weights_to_clean_patch_embeddings``.
        self.clean_x_embedder = WanPatchEmbedForCleanLatents(
            inner_dim=dim, in_channels=patch_in_channels, patch_size=patch_size)

        # ---- Transformer blocks ----
        cross_attn_type = 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # Swap each block's self-attention for the packed-rope variant.
        # State-dict keys are unchanged because we replace at the same attribute.
        for block in self.blocks:
            block.self_attn = WanFramepackSelfAttention(
                dim=dim, num_heads=num_heads, window_size=window_size,
                qk_norm=qk_norm, eps=eps,
            )

        # ---- Output head ----
        self.head = Head(dim, out_dim, patch_size, eps)

        # ---- RoPE frequency tables ----
        # ``self.freqs`` columns: [d_T | d_HW | d_HW] where d_T = (d - 4*(d//6))/2
        # and d_HW = d//6. Total = head_dim/2 complex entries per row.
        # NOTE: stored as a plain attribute (not a registered buffer) because
        # ``freqs`` is complex; registering it would let ``module.to(dtype=...)``
        # cast it to a real dtype (e.g. bf16) and break complex RoPE math.
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d_head = dim // num_heads
        self._max_seq_len = 1024
        self.freqs = torch.cat([
            rope_params(self._max_seq_len, d_head - 4 * (d_head // 6)),  # T
            rope_params(self._max_seq_len, 2 * (d_head // 6)),           # H
            rope_params(self._max_seq_len, 2 * (d_head // 6)),           # W
        ], dim=1)

        # Activation checkpointing flag (toggled via
        # ``enable_gradient_checkpointing`` / ``disable_gradient_checkpointing``).
        self.gradient_checkpointing = False
        # Selective activation checkpointing: when True (and gradient
        # checkpointing is on), recompute only the cheap ops per block and keep
        # the flash-attention output + matmuls saved (see ``_sac_context_fn``).
        self.selective_checkpointing = False

    # ================================================================
    # Gradient checkpointing
    # ================================================================

    def enable_gradient_checkpointing(self, selective: bool = False):
        """Enable activation checkpointing on transformer blocks.

        Args:
            selective: use selective activation checkpointing (SAC) — recompute
                only the cheap ops, keep flash-attention/matmul outputs saved.
                Requires torch >= 2.4; falls back to full AC with a warning if
                unavailable.
        """
        self.gradient_checkpointing = True
        if selective and not _SAC_AVAILABLE:
            print("[warn] selective activation checkpointing unavailable "
                  "(torch < 2.4); falling back to full AC")
            selective = False
        self.selective_checkpointing = selective

    def disable_gradient_checkpointing(self):
        """Disable activation checkpointing on transformer blocks."""
        self.gradient_checkpointing = False
        self.selective_checkpointing = False

    # ================================================================
    # torch.compile (regional / per-block)
    # ================================================================

    def compile_blocks(self, activation_memory_budget=None, **compile_kwargs):
        """Regionally ``torch.compile`` each transformer block (opt-in).

        Compiles per-block rather than the whole model: compile times stay low
        and each compiled block still composes with FSDP2 per-block sharding.
        Call this *before* ``apply_fsdp``.

        Regional compile fuses the per-block elementwise chain (LayerNorm +
        fp32 AdaLN modulation ``(norm(x)*(1+e)+e)`` + residual adds) into far
        fewer kernels, cutting launch overhead and the fp32 temporary churn;
        the flash-attention custom op graph-breaks cleanly (no ``fullgraph``).

        Args:
            activation_memory_budget: if given, sets the compile-only Memory
                Budget API (``torch._functorch.config.activation_memory_budget``)
                so the min-cut partitioner auto-applies pareto-optimal selective
                recompute in ``[0, 1]`` (0 = plain AC, 1 = default compile). This
                *replaces* manual activation checkpointing, which the caller must
                keep disabled to avoid stacking.
        """
        if activation_memory_budget is not None:
            try:
                import torch._functorch.config as _functorch_config
                _functorch_config.activation_memory_budget = float(
                    activation_memory_budget)
            except (ImportError, AttributeError):
                import torch._dynamo as _dynamo
                _dynamo.config.activation_memory_budget = float(
                    activation_memory_budget)
        for i in range(len(self.blocks)):
            self.blocks[i] = torch.compile(self.blocks[i], **compile_kwargs)
        self._blocks_compiled = True

    # ================================================================
    # FramePack: weight init helper
    # ================================================================

    def materialize(self):
        """Recompute RoPE frequency tables after meta-device initialization."""
        d_head = self.dim // self.num_heads
        self.freqs = torch.cat([
            rope_params(self._max_seq_len, d_head - 4 * (d_head // 6)),
            rope_params(self._max_seq_len, 2 * (d_head // 6)),
            rope_params(self._max_seq_len, 2 * (d_head // 6)),
        ], dim=1)

    def copy_weights_to_clean_patch_embeddings(self):
        """Tile pretrained ``patch_embedding`` weights into ``clean_x_embedder``."""
        self.clean_x_embedder.initialize_weight_from_another_conv3d(self.patch_embedding)

    # ================================================================
    # FramePack: latent-index construction
    # (was previously done in FramepackMultiAdapterPipeline)
    # ================================================================

    @staticmethod
    def build_latent_indices(device: torch.device):
        """Return (clean_start, clean_4x, clean_2x, clean_1x, gen) frame indices.

        Total range is ``[0, TOTAL_LATENT_FRAMES)``. ``clean_start`` and
        ``clean_1x`` are kept separate so callers can verify the layout, but
        both are processed by ``clean_x_embedder.proj`` in :meth:`forward`.
        """
        idx = torch.arange(0, TOTAL_LATENT_FRAMES, device=device)
        clean_start_idx = idx[:CLEAN_START_FRAMES]
        offset = CLEAN_START_FRAMES
        clean_4x_idx = idx[offset:offset + CLEAN_4X_FRAMES]
        offset += CLEAN_4X_FRAMES
        clean_2x_idx = idx[offset:offset + CLEAN_2X_FRAMES]
        offset += CLEAN_2X_FRAMES
        clean_1x_idx = idx[offset:offset + CLEAN_1X_FRAMES]
        offset += CLEAN_1X_FRAMES
        gen_idx = idx[offset:offset + GEN_FRAMES]
        return clean_start_idx, clean_4x_idx, clean_2x_idx, clean_1x_idx, gen_idx

    # ================================================================
    # RoPE helpers
    # ================================================================

    def _rope_dims(self) -> Tuple[int, int]:
        d_head = self.dim // self.num_heads
        d_T = (d_head - 4 * (d_head // 6)) // 2  # complex dim for T axis
        d_HW = d_head // 6                       # complex dim for each of H, W
        return d_T, d_HW

    def _build_segment_rope(
        self,
        frame_indices: torch.Tensor,
        h_grid: int,
        w_grid: int,
        downsample: int = 1,
    ) -> torch.Tensor:
        """Build per-token complex RoPE freqs for one segment.

        Frame indices are explicit (so history segments at their downsampled
        positions get correct T-axis encoding); spatial positions span the
        generation post-patch grid (h_grid x w_grid). For ``downsample > 1``,
        the [T, H, W] freq grid is average-pooled by ``downsample`` on each
        axis to match the Conv3d output of the corresponding clean head.

        Returns:
            ``[n_tokens, head_dim/2]`` complex tensor.
        """
        d_T, d_HW = self._rope_dims()
        freqs_T_table = self.freqs[:, :d_T]
        freqs_H_table = self.freqs[:, d_T:d_T + d_HW]
        freqs_W_table = self.freqs[:, d_T + d_HW:]

        f = frame_indices.shape[0]
        idx_long = frame_indices.long().to(self.freqs.device)
        fT = freqs_T_table[idx_long]                   # [f, d_T]
        fH = freqs_H_table[:h_grid]                    # [h, d_HW]
        fW = freqs_W_table[:w_grid]                    # [w, d_HW]

        fT = fT.view(f, 1, 1, d_T).expand(f, h_grid, w_grid, d_T)
        fH = fH.view(1, h_grid, 1, d_HW).expand(f, h_grid, w_grid, d_HW)
        fW = fW.view(1, 1, w_grid, d_HW).expand(f, h_grid, w_grid, d_HW)
        grid = torch.cat([fT, fH, fW], dim=-1)         # [f, h, w, head_dim/2]

        if downsample > 1:
            # [f, h, w, half] -> [1, half, f, h, w] for pooling.
            g = grid.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
            ks = (downsample, downsample, downsample)
            # pad_for_3d_conv expects shape [B, C, T, H, W].
            g = pad_for_3d_conv(g, ks)
            g = _avg_pool3d_complex(g, downsample)
            grid = g.squeeze(0).permute(1, 2, 3, 0).contiguous()

        return grid.flatten(0, 2)                      # [f*h*w, head_dim/2]

    # ================================================================
    # I2V: build full-T [x | mask | y] cat for patch embedding
    # ================================================================

    def _build_xmy(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate ``[x | mask | y]`` along channel dim (Wan I2V layout).

        The caller must supply both ``y`` (``i2v_in_dim`` channels) and
        ``mask`` (4 channels) at full T.
        """
        B, _, T, H, W = x.shape
        assert y.shape == (B, self.i2v_in_dim, T, H, W), (
            f"y shape {tuple(y.shape)} != {(B, self.i2v_in_dim, T, H, W)}"
        )
        assert mask.shape == (B, self.mask_in_dim, T, H, W), (
            f"mask shape {tuple(mask.shape)} != {(B, self.mask_in_dim, T, H, W)}"
        )
        return torch.cat([x, mask, y], dim=1)

    # ================================================================
    # Forward
    # ================================================================

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: List[torch.Tensor],
        y: torch.Tensor,
        mask: torch.Tensor,
        clip_features: torch.Tensor,
        # Camera adapter output (added to generation tokens)
        camera_features: Optional[torch.Tensor] = None,
        # History-condition noise-augmentation timestep (per sample). When
        # provided (FramePack mode), history tokens are modulated by this
        # timestep instead of the diffusion ``t`` — see the per-token ``e``
        # construction below. The anchor/start frame always uses timestep 0.
        history_timestep: Optional[torch.Tensor] = None,
    ):
        """Forward pass.

        FramePack mode is selected automatically: when ``x.shape[2] ==
        TOTAL_LATENT_FRAMES`` the input is interpreted as
        ``[history (20) | gen (20)]`` and the Hunyuan-style patchify + packed
        RoPE path is used. Otherwise the model falls back to a single
        generation segment.

        Channel layout for ``patch_embedding`` / ``clean_x_embedder`` follows
        Wan I2V: ``[x | mask(4) | y]``. Rendered-warp conditioning is carried
        entirely by ``y`` (rendered RGB latent) and ``mask`` (rendered validity
        mask); there is no separate warp adapter.

        Args:
            x: ``[B, in_dim, T, H, W]`` latents. In FramePack mode T must be
                ``TOTAL_LATENT_FRAMES`` and ``x[:, :, :N_HISTORY_FRAMES]`` is
                clean history (unaffected by the diffusion noise schedule, but
                possibly noise-augmented — see ``history_timestep``).
            t: ``[B]`` diffusion timesteps (0~1000 scale).
            context: list of text embeddings ``[L_text, text_dim]`` per sample.
            y: ``[B, i2v_in_dim, T, H, W]`` (full T).
            mask: ``[B, 4, T, H, W]`` (full T). On the generation segment the 4
                channels are the rendered validity mask of the 4 pixel frames
                per latent frame; on the anchor/history segment it is all ones.
            clip_features: ``[B, n_clip_tokens, clip_dim]`` for CLIP MLP.
            camera_features: ``[B, dim, F_gen, H_gen, W_gen]`` from
                ``SimpleAdapter``; added to gen tokens.
            history_timestep: ``[B]`` noise-augmentation timestep (0~1000 scale)
                applied to the history condition (all history frames except the
                anchor/start). ``None`` => history uses timestep 0 (clean).

        Returns:
            ``[B, out_dim, F_gen*pt, H_gen*ph, W_gen*pw]`` denoised segment.
        """
        device = self.patch_embedding.weight.device

        # ``freqs`` is a plain (non-buffer) complex attribute, so module.to()
        # does not move it; keep it co-located with the params.
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        B, _, T_total, H_lat, W_lat = x.shape
        pt, ph, pw = self.patch_size
        # FramePack is a permanent setting: the input must always carry the full
        # [history (20) | gen (20)] schedule. The non-FramePack fallback below is
        # kept for reference but is unreachable.
        assert T_total == TOTAL_LATENT_FRAMES, (
            f"FramePack mode requires T={TOTAL_LATENT_FRAMES}, got {T_total}"
        )
        use_framepack = True

        # Build full-T [x | mask | y].
        x_full = self._build_xmy(x, y=y, mask=mask)

        # ---- Build tokens + packed RoPE freqs ----
        if use_framepack:
            (clean_start_idx, clean_4x_idx,
             clean_2x_idx, clean_1x_idx, gen_idx) = self.build_latent_indices(device)

            # Split history vs generation along T (after [x|mask|y] cat).
            history_full = x_full[:, :, :N_HISTORY_FRAMES]
            gen_full = x_full[:, :, N_HISTORY_FRAMES:]

            # ---- Generation patch embedding (Wan I2V layout) ----
            gen_emb = self.patch_embedding(gen_full)
            f_gen, h_gen, w_gen = gen_emb.shape[2], gen_emb.shape[3], gen_emb.shape[4]
            gen_tokens = gen_emb.flatten(2).transpose(1, 2)

            # Camera features added to generation tokens only.
            if camera_features is not None:
                cam_tokens = camera_features.flatten(2).transpose(1, 2)
                assert cam_tokens.shape[1] == gen_tokens.shape[1], (
                    f"camera_features tokens {cam_tokens.shape[1]} != "
                    f"gen tokens {gen_tokens.shape[1]}"
                )
                gen_tokens = gen_tokens + cam_tokens
            gen_rope = self._build_segment_rope(gen_idx, h_gen, w_gen, downsample=1)

            # ---- Clean (history) patch embeddings ----
            # Slice history along T: [start (1) | 4x (16) | 2x (2) | 1x (1)].
            clean_start_x = history_full[:, :, :CLEAN_START_FRAMES]
            off = CLEAN_START_FRAMES
            clean_4x_x = history_full[:, :, off:off + CLEAN_4X_FRAMES]
            off += CLEAN_4X_FRAMES
            clean_2x_x = history_full[:, :, off:off + CLEAN_2X_FRAMES]
            off += CLEAN_2X_FRAMES
            clean_1x_x = history_full[:, :, off:off + CLEAN_1X_FRAMES]

            # Hunyuan combines start + clean_1x and runs them through proj (kernel 1).
            clean_combined_x = torch.cat([clean_start_x, clean_1x_x], dim=2)
            clean_combined_idx = torch.cat([clean_start_idx, clean_1x_idx], dim=0)

            # 4x: kernel (pt*4, ph*4, pw*4). RoPE: gen-grid with downsample=4.
            c4_in = pad_for_3d_conv(clean_4x_x, (pt * 4, ph * 4, pw * 4))
            c4_emb = self.clean_x_embedder.proj_4x(c4_in)
            c4_tokens = c4_emb.flatten(2).transpose(1, 2)
            c4_rope = self._build_segment_rope(
                clean_4x_idx, h_gen, w_gen, downsample=4)

            # 2x
            c2_in = pad_for_3d_conv(clean_2x_x, (pt * 2, ph * 2, pw * 2))
            c2_emb = self.clean_x_embedder.proj_2x(c2_in)
            c2_tokens = c2_emb.flatten(2).transpose(1, 2)
            c2_rope = self._build_segment_rope(
                clean_2x_idx, h_gen, w_gen, downsample=2)

            # 1x (start + clean_1x combined): kernel = patch_size, no spatial downsample.
            c1_in = pad_for_3d_conv(clean_combined_x, self.patch_size)
            c1_emb = self.clean_x_embedder.proj(c1_in)
            c1_tokens = c1_emb.flatten(2).transpose(1, 2)
            c1_rope = self._build_segment_rope(
                clean_combined_idx, h_gen, w_gen, downsample=1)

            # Concat order: [clean_4x, clean_2x, clean_1x_combined, gen]
            tokens = torch.cat([c4_tokens, c2_tokens, c1_tokens, gen_tokens], dim=1)
            rope = torch.cat([c4_rope, c2_rope, c1_rope, gen_rope], dim=0)

            gen_start = c4_tokens.shape[1] + c2_tokens.shape[1] + c1_tokens.shape[1]
            gen_end = gen_start + gen_tokens.shape[1]

            # Token counts per FramePack segment, used to build the per-token
            # time modulation for history noise augmentation. ``clean_combined``
            # (c1) holds [start | clean_1x]: the first ``hw`` tokens are the
            # anchor/start frame (timestep 0), the rest carry the history noise
            # timestep.
            n_c4, n_c2, n_c1 = (
                c4_tokens.shape[1], c2_tokens.shape[1], c1_tokens.shape[1])
            n_gen_tok = gen_tokens.shape[1]
            hw_tok = h_gen * w_gen

        else:
            # Fallback: treat the whole input as a single generation segment.
            gen_emb = self.patch_embedding(x_full)
            f_gen, h_gen, w_gen = gen_emb.shape[2], gen_emb.shape[3], gen_emb.shape[4]
            gen_tokens = gen_emb.flatten(2).transpose(1, 2)
            if camera_features is not None:
                cam_tokens = camera_features.flatten(2).transpose(1, 2)
                gen_tokens = gen_tokens + cam_tokens

            gen_idx = torch.arange(f_gen, device=device)
            rope = self._build_segment_rope(gen_idx, h_gen, w_gen, downsample=1)
            tokens = gen_tokens
            gen_start = 0
            gen_end = gen_tokens.shape[1]

        # Broadcast packed freqs to [B, L, head_dim/2].
        rope = rope.unsqueeze(0).expand(B, -1, -1).contiguous()

        total_len = tokens.shape[1]
        seq_lens = torch.tensor([total_len] * B, dtype=torch.long, device=device)
        # grid_sizes is unused by WanFramepackSelfAttention but the block API
        # passes it through, so supply a placeholder reflecting the gen grid.
        grid_sizes = torch.tensor(
            [[f_gen, h_gen, w_gen]] * B, dtype=torch.long, device=device)

        # ---- Time embedding (per-token in FramePack mode) ----
        # ``_time_mod`` maps a [B] timestep to (pre-projection embedding [B,dim],
        # AdaLN modulation [B,6,dim]). Generation tokens use the diffusion
        # timestep ``t``; history tokens use ``history_timestep`` (noise-aug);
        # the anchor/start frame always uses timestep 0.
        def _time_mod(tt: torch.Tensor):
            with torch.amp.autocast('cuda', dtype=torch.float32):
                emb = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, tt).float())
                e0 = self.time_projection(emb).unflatten(1, (6, self.dim))
            assert emb.dtype == torch.float32 and e0.dtype == torch.float32
            # Collapse the single fp32 island right here: return in the compute
            # dtype (``tokens``'s, i.e. bf16) so the modulation signal never
            # carries an fp32 [B, 6, L, dim] tensor downstream.
            return emb.to(tokens.dtype), e0.to(tokens.dtype)

        t = t.to(device=device, dtype=torch.float32)
        emb_gen, e0_gen = _time_mod(t)

        if use_framepack:
            if history_timestep is None:
                hist_t = torch.zeros_like(t)
            else:
                hist_t = history_timestep.to(device=device, dtype=torch.float32)
                if hist_t.dim() == 0:
                    hist_t = hist_t.expand(B)
            _, e0_hist = _time_mod(hist_t)
            _, e0_zero = _time_mod(torch.zeros_like(t))

            # Compact per-segment AdaLN modulation. The packed token sequence is
            # [c4 (hist) | c2 (hist) | c1_start (zero) | c1_rest (hist) | gen],
            # so collapsing the adjacent c4/c2 hist runs leaves four contiguous
            # same-timestep segments carrying [hist, zero, hist, gen]. Store one
            # modulation entry per segment ([B, 6, 4, dim]) plus the per-segment
            # token counts; the AttentionBlock expands this back to per-token
            # via repeat_interleave. This avoids materializing a large redundant
            # [B, 6, L, dim] tensor here and is friendlier to torch.compile.
            e0 = torch.stack(
                [e0_hist, e0_zero, e0_hist, e0_gen], dim=2)  # [B, 6, 4, dim]
            e_rep_len = (n_c4 + n_c2, hw_tok, n_c1 - hw_tok, n_gen_tok)
            assert sum(e_rep_len) == tokens.shape[1], (
                f"segment lengths {e_rep_len} sum {sum(e_rep_len)} != tokens "
                f"{tokens.shape[1]}")
        else:
            # No history segment: single shared modulation across all tokens.
            e0 = e0_gen
            e_rep_len = None

        # The output head only keeps generation tokens, so it always uses the
        # diffusion-timestep embedding (history head outputs are discarded).
        head_e = emb_gen

        # ---- Text embedding ----
        context_padded = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # ---- Prepend CLIP image tokens to text context (Wan 2.1 I2V) ----
        clip_ctx = self.img_emb(clip_features)
        context_padded = torch.cat([clip_ctx, context_padded], dim=1)

        # ---- Transformer blocks ----
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=rope,                # packed [B, L, head_dim/2] complex
            context=context_padded,
            context_lens=None,
            e_rep_len=e_rep_len,
        )
        for block in self.blocks:
            if self.gradient_checkpointing:
                if self.selective_checkpointing:
                    tokens = torch.utils.checkpoint.checkpoint(
                        block, tokens, **kwargs, use_reentrant=False,
                        context_fn=_sac_context_fn)
                else:
                    tokens = torch.utils.checkpoint.checkpoint(
                        block, tokens, **kwargs, use_reentrant=False)
            else:
                tokens = block(tokens, **kwargs)

        # ---- Head + reshape generation segment ----
        tokens = self.head(tokens, head_e)
        gen_tokens = tokens[:, gen_start:gen_end]
        output = rearrange(
            gen_tokens,
            'b (f h w) (pt ph pw c) -> b c (f pt) (h ph) (w pw)',
            f=f_gen, h=h_gen, w=w_gen,
            pt=pt, ph=ph, pw=pw, c=self.out_dim,
        )
        return output
