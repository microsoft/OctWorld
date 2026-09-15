# OctWorld Wan 14B — Model Architecture

## Overview

This directory contains the OctWorld video generation model, built on top of Wan 2.1 I2V 14B with Hunyuan-style FramePack for autoregressive long-video generation.

```
octworld/models/wan/
├── wan_framepack.py      # OctWorldWanModel (main DiT + FramePack)
├── camera_adapter.py     # SimpleAdapter (raymap → token features)
└── modules/              # Upstream Wan 2.1 components (verbatim, diffusers-free)
    ├── model.py          # WanModel, WanAttentionBlock, MLPProj, Head, RoPE
    ├── vae.py            # WanVAE (temporal causal 3D VAE)
    ├── attention.py      # flash_attention wrapper
    ├── t5.py             # T5EncoderModel (text encoder)
    ├── clip.py           # CLIPModel (image encoder)
    └── ...
```

## OctWorldWanModel (`wan_framepack.py`)

### Architecture Summary

| Parameter | Value |
|-----------|-------|
| Backbone | Wan 2.1 I2V 14B DiT |
| dim | 5120 |
| num_layers | 40 |
| num_heads | 40 |
| head_dim | 128 |
| patch_size | (1, 2, 2) |
| Parameters | 16.45B |

### Key Design Choices

1. **Pretrained-compatible `patch_embedding`**: Input channels = `in_dim(16) + mask(4) + y(16) = 36`, matching Wan 2.1 I2V exactly. Weights load 1:1 from the official checkpoint.

2. **Packed per-token RoPE** (`WanFramepackSelfAttention`): Replaces Wan's standard self-attention. Each token receives positional encoding based on its actual `(t, h, w)` coordinates in the original latent grid, allowing history segments at different temporal scales to coexist in the same attention context.

3. **Multi-scale clean history embedder** (`WanPatchEmbedForCleanLatents`): Patchifies history frames with three Conv3d heads at kernel sizes `1×`, `2×`, `4×`. Weights are tiled from `patch_embedding` at init via `copy_weights_to_clean_patch_embeddings()`.

4. **Zero-init warp adapter**: A Conv3d (`warp_in_dim=16 → dim=5120`) that injects warp-mask conditioning into generation tokens. Zero-initialized so the pretrained Wan I2V forward pass is preserved at step 0.

5. **I2V CLIP projection** (`MLPProj`): Projects CLIP image features and prepends them to the text context, matching upstream Wan 2.1 I2V.

### FramePack Schedule

Total = **40 latent frames** (history 20 + generation 20):

```
[clean_start (1)] [clean_4x (16)] [clean_2x (2)] [clean_1x (1)] [gen (20)]
|------- history (20 frames) ---------|---------- generation ----------|
```

Each history segment is patchified at its native scale:
- **clean_4x** (16 frames): Conv3d kernel `(4, 8, 8)`, stride `(4, 8, 8)` → heavily compressed
- **clean_2x** (2 frames): Conv3d kernel `(2, 4, 4)`, stride `(2, 4, 4)` → moderate compression
- **clean_1x** (start + clean_1x = 2 frames): Conv3d kernel `(1, 2, 2)` → same as generation

RoPE for each history segment is built at the generation grid resolution, then average-pooled by the corresponding kernel factor.

Token order in the attention sequence: `[clean_4x, clean_2x, clean_1x, gen]`.

### Forward Pass Modes

The model auto-selects based on `x.shape[2]`:

- **FramePack mode** (`T == 40`): Input is `[history(20) | gen(20)]`. Outputs denoised generation segment `[B, 16, 20, H, W]`.
- **Fallback mode** (`T != 40`): Input is a single generation segment (e.g., 21 frames for vanilla Wan I2V). Outputs `[B, 16, T, H, W]`.

### Channel Layout (Wan I2V Convention)

For each frame, the patch embedding sees: `[x(16) | mask(4) | y(16)]`

- **x**: Noisy latent (generation) or clean latent (history)
- **mask**: Per-frame indicator (Wan original: 1 at anchor, 0 elsewhere; Lyra: all 1s)
- **y**: Conditioning latent (Wan original: VAE-encoded first-frame+zeros; Lyra: anchor/warp-RGB)

### Warp Conditioning

Warp latents (32 ch total from the dataset) are split:
- **warp_rgb** (ch 0–15): Goes into the `y` slot alongside the anchor latent
- **warp_mask** (ch 16–31): Injected via `warp_adapter` (zero-init Conv3d, added to gen tokens)

This separation allows warp-RGB to be processed by the pretrained I2V pathway while warp-mask is a learnable addition that starts at zero.

### Sampling Conventions (3 stages)

| Stage | y | mask | x | Notes |
|-------|---|------|---|-------|
| stage1 (Wan vanilla) | `VAE.encode([first_pixel, zeros])` | 1@frame0, 0 elsewhere | noise(21) | All 21 frames denoised |
| stage2 (Lyra, no framepack) | `[anchor \| warp_rgb]` | all 1s | `[anchor_clean \| noise(20)]` | Frame 0 stays clean |
| stage3 (Lyra + framepack) | `[history \| warp_rgb]` (40 frames) | all 1s | `[history_clean(20) \| noise(20)]` | Model returns gen(20) only |

### Inference Requirements (from `docs/wan_i2v_sampling_debug.md`)

- **CFG** is mandatory: `guide_scale=5.0`, negative prompt via T5
- **Scheduler**: `FlowUniPCMultistepScheduler` (2nd-order multistep), not euler
- **Shift**: 3.0 for 480p, 5.0 for 720p
- **Latent precision**: Keep in fp32; only DiT forward runs under bf16 autocast

## Camera Adapter (`camera_adapter.py`)

`SimpleAdapter`: PixelUnshuffle + Conv2d + ResidualBlocks per frame.

- Input: raymap `[B, C_ray, T, H, W]`
- Output: `[B, 5120, T, H/2, W/2]` (matches patch-embedded latent grid)
- Added directly to generation tokens (same injection style as warp)

## Weight Loading

```python
model = OctWorldWanModel(framepack_type="f1k1f16k4f2k2f1k1_g20", warp_in_dim=16, **WAN_14B_I2V_CONFIG)
model.load_wan_pretrained(state_dict, strict=False)       # loads Wan 2.1 I2V weights
model.copy_weights_to_clean_patch_embeddings()            # tiles patch_embedding → clean_x_embedder
```

Expected missing keys after loading: 8 (`clean_x_embedder.*` × 6 + `warp_adapter.weight` + `warp_adapter.bias`).

## Pixel ↔ Latent Dimensions

- VAE temporal stride: 4 (except first frame) → 21 latent frames = 81 pixel frames
- VAE spatial stride: 8 → latent `(H/8, W/8)` = `(60, 104)` for 480×832 video
- Patch spatial stride: 2 → post-patch grid `(30, 52)` per gen frame
- Generation tokens per step: `20 × 30 × 52 = 31,200`
