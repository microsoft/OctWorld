# Wan World Generation

AR world generation with the Wan 14B FramePack backbone
([pipeline implementation](../../../octworld/pipelines/wan/wanworld_pipeline.py)).

[world_gen.py](world_gen.py) runs one case. [batch_world_gen.py](batch_world_gen.py)
loads the pipeline once and runs a `wan_standard` dataset sequentially, with
optional rank/world-size sharding.

The default single case is `scene_00000` in the bundled
[wan_standard dataset](../../../assets/wan_standard/instances.json), with seed
425 and online integration.

The dataset contains three four-chunk inputs:

- `scene_00000`: DL3DV image with its original DA3-streaming trajectory.
- `scene_00001`: the same DL3DV image with a custom smoothed trajectory.
- `scene_00002`: Minecraft image with a custom smoothed trajectory.

Each contains 321 poses and four prompt lines, so running the corresponding
README command without `--max_chunks` generates exactly four chunks.

## Input format (single canonical form)

| Input          | Format                                                              |
| -------------- | ------------------------------------------------------------------ |
| `image`        | any PIL-readable image (resized to `target_width × target_height`) |
| `trajectory`   | `.npz` with `extrinsics` (T,4,4) w2c + `intrinsics` (T,3,3)         |
| `anchor_depth` | 16-bit log-encoded depth PNG (read via `utils.io.read_depth`)      |
| `prompt`       | `.txt`, one prompt per chunk → list; single line → applied to all  |

The anchor depth defines the metric scale of the user trajectory. By default,
DA3 predicts per-chunk depth with the user cameras as hints, and OctMap uses
those user-camera poses for integration.

## Effective defaults

Dimensions below are **height x width**. These defaults match the evaluated
online/noise0 setting; they do not imply identical outputs for different seeds,
inputs, weights or environments.

| Setting | Effective value |
| --- | --- |
| CFG | 5.0, separate positive/negative forwards (`cfg_merged=False`) |
| Sampler | Wan UniPC, 35 steps, shift 3.0 |
| Condition noise | `noise_aug_level=0`: both history and rendered latents stay clean; `None` also disables it |
| Depth | DA3 `align_with_user`, non-streaming; per-chunk overlap-scale fallback if Umeyama degenerates |
| DA3 processing | longest side 504, `upper_bound_resize`, then patch14 rounding |
| DA3 output | 294x504 for 480x832 RGB input, not a fixed native output for arbitrary inputs |
| Intermediate depth/confidence | 480x832 for masking; confidence is per-call, not an accumulated output |
| OctMap integration | 294x504, online by default |
| RGB / warp rendering | 480x832 (`mapper_resolution`); separate from `integration_resolution` |
| Keyframes | RAFT mean flow >5 in preprocessed pixels; skipped-frame counter threshold 8 |
| Driver seed | 425; library `WanWorldPipeline.__call__` still defaults to 0 |
| Video export | 15 fps for every chunk/final output, depth and warp video |

### Seed control

The example passes `--seed` both to `WanWorldPipeConfig.init_seed` (before
model construction) and to the rollout. Library callers can set `init_seed`
explicitly (default 0); a later rollout seed does not reconstruct loaded models.

Diffusion's continuous CPU FP32 noise stream still uses `seed`, and condition
augmentation still uses a separate device generator with `seed + 1`. The
[`random-state helper`](../../../utils/random_state.py) additionally derives
independent, stable streams from the case seed for:

- DA3's internal random subsampling, with one isolated scope per predict call;
- scale-fitting point selection, using its own CPU generator;
- RAFT first-use construction/selection;
- export-only depth and scale, isolated from the inference streams above.

Each rollout resets these streams. Python, NumPy and Torch CPU/CUDA global RNG
states are restored after auxiliary scopes, including exceptions; export work
cannot shift later inference random draws. The output JSON records the rule
version, initial seeds and call counts under `randomness`.

This is an intentional auxiliary-RNG behavior update: old global-RNG consumption
is no longer reproduced for DA3/scale fitting. The historical diffusion-noise
hashes remain unchanged, but full videos may differ from the previous version.
The upstream pose-alignment RANSAC uses its existing fixed seed 42. cuDNN
benchmarking is disabled; exact cross-GPU/backend/kernel determinism is **not**
guaranteed. Do not run concurrent calls on the same pipeline or share global RNG
with unrelated worker threads if reproducibility is required.

Changing export fps does not resample the trajectory, alter the frame count or
change diffusion steps; 641 frames now play for approximately 42.733 seconds.
Previously saved 8fps artifacts are not rewritten.

The depth path intentionally retains the evaluated resizing sequence:
DA3 294x504 → nearest 480x832 for confidence/sky masking → depth-edge filtering
→ nearest 294x504 for integration. RGB is bilinearly resized for integration.
This is **not** direct integration of DA3's untouched output.

The default keyframe branch ignores `--sample_ratio`; it applies only to
`--frame_select_method uniform`. The interval counter is checked before
incrementing, so `--keyframe_max_interval 8` may produce a 9-frame index gap
on static input; every selection block also keeps its last frame.

`--bigger_voxel_pixel_size_scale` only affects the `bigger_voxels` carve strategy.
The legacy `--depth_method standalone` spelling is retained for compatibility,
but is disabled and fails before model loading. DA3 streaming YAML options such
as chunk size, overlap and loop closure are not active in `align_with_user`.

Library callers can change existing `WanWorldPipeConfig` fields; no extra CLI
flags are required. `integration_resolution=None` integrates at
`mapper_resolution`; it must be set to `None` when selecting the TSDF mapper.

OctMap integration is **online** by default: each map update allocates and
integrates only the newly selected frames. `--offline_integration` retains the
octree structure but clears its observation weights and re-integrates all
cached frames after each update. Offline mode therefore does not rebuild the
octree from scratch, but its update cost grows with the accumulated history.
Online mode still caches history and is unrelated to DA3 streaming. Both modes
share the initial seed/first-generated-chunk map reset. A final chunk after
chunk 0 skips **both depth prediction and map integration** because no next
chunk consumes its map; a single-chunk rollout still performs its full
first-chunk depth prediction and integration.

### DA3 backend dependency

**xformers is required for DA3's `SwiGLUFFNFused`**, not an optional attention
speedup: without it, upstream DA3 silently switches the FFN backend, changing
depth/confidence and potentially the warp. The predictor now checks the backend
before loading DA3, logs `backend_info`, and fails on a missing dependency or a
previously cached fallback (restart after fixing the environment).

The verified stack is **PyTorch 2.8.0 / CUDA 12.6, xformers 0.0.32.post2,
flash-attn 2.8.3**. For only this tuple, the
[preflight helper](../../../depth_prediction/predictors/da3_backend.py) temporarily
sets an otherwise absent `XFORMERS_IGNORE_FLASH_VERSION_CHECK=1` while importing
`xformers.ops`, then restores the environment even on failure; user-set values
(including `0`) are preserved. This only handles an eagerly imported optional
FMHA adapter's version check; it does **not** call xformers attention. DA3 still
uses Torch SDPA and Wan still uses its existing FlashAttention path.

The orange-region recovery was directly verified only in local seed-map probes,
with P20 confidence, sky masking and the 1000-element mesh filter retained. This
does not establish full first-chunk recovery or exact historical same-GPU output
reproduction.

## Run the real 14B case: first chunk, then full rollout

Run from the project root with the published model package and inference
dependencies installed. These commands use the **real 14B checkpoint**, not a
tiny model. Do **not** add `--debug` or `--preview_only` to these runs.

Use a **new or empty `--save_dir` for each run**. Existing nonempty directories
are refused rather than overwritten; symlinked output paths are also refused.
Keep logs and unrelated files outside it so successful finalization can leave
exactly the four output files described below.

### 1. First-chunk run

One chunk generates 80 new frames plus the input frame: **81 aligned
RGB/depth/warp frames**, at **480x832 (H x W), 15 fps**.

```bash
python example/worldgen/wan/world_gen.py \
    --dit_ckpt model_weights/octworld-1.5/model \
    --model_weights_dir model_weights \
    --max_chunks 1 \
    --save_dir outputs/dl3dv_gt_first_chunk/
```

### 2. Complete rollout

After inspecting the first chunk, start a new run **without `--max_chunks`**.
This reruns the case from the beginning; it does not resume the first command.
The supplied four prompts and 321 cameras define **4 chunks and 321
RGB/depth/warp frames** (1 input + 4 x 80 generated), at **480x832 (H x W),
15 fps** (approximately **21.4 seconds**).

```bash
python example/worldgen/wan/world_gen.py \
    --dit_ckpt model_weights/octworld-1.5/model \
    --model_weights_dir model_weights \
    --save_dir outputs/dl3dv_gt_4chunks/
```

The published checkpoint is **v2 / run1 / step7000**.
[pipe_config.json](../../../model_weights/octworld-1.5/pipe_config.json) and
[neg_text_embed.pt](../../../model_weights/octworld-1.5/neg_text_embed.pt) are
auto-resolved from its OctWorld model root, above `--dit_ckpt`.

Omitting the four input options selects `assets/wan_standard/scenes/scene_00000`.
Pass explicit paths from `scene_00001` or `scene_00002` to run a custom case.
Omitting `--offline_integration` selects the evaluated default online behavior.

## Batch generation

The batch runner uses the same checkpoint default, `WanWorldPipeConfig` fields,
seed 425, online integration, keyframe settings, and output contract as
`world_gen.py`. Relative paths in `instances.json` resolve from the dataset root.

```bash
python example/worldgen/wan/batch_world_gen.py \
    --instances assets/wan_standard/instances.json \
    --base_folder outputs/batch \
    --dit_ckpt model_weights/octworld-1.5/model \
    --model_weights_dir model_weights
```

The pipeline is built once. Results are written under
`outputs/batch/instances/<sample_id>/`; each sample also receives `info.json`
and either `SUCCESS.txt` or `FAILED.txt`. Use `--rank` and `--world_size` for
single-GPU worker sharding, with `CUDA_VISIBLE_DEVICES` set externally.

`--debug` selects a compatible local tiny checkpoint plus the other inference
models; it is neither the 14B first-chunk validation nor a weight-free smoke.
`--preview_only` skips diffusion, but still constructs the pipeline and requires
its weights, first-frame depth prediction and renderer.

## Output lifecycle

The [output implementation](../../../octworld/pipelines/wan/wanworld_output_mixin.py)
owns only files created for this run:

1. **While running:** each chunk has its own directory, starting at
    chunks/chunk_0000/ under `--save_dir`. Its warp.mp4 is written **before
    diffusion**, using the actual renderer RGB; output.mp4 and depth.mp4 are
    written after that chunk's generation and required depth work. The root
    sha256.json is atomically updated as warp, sampling and chunk records arrive.
2. **On failure:** already written per-chunk diagnostics are retained, including
    the current warp if diffusion fails. These files are for inspection, not a
    resume checkpoint; retry into a fresh empty directory.
3. **On success:** the root contains **exactly output.mp4, depth.mp4, warp.mp4
    and sha256.json**. Final videos are encoded from raw buffers, not decoded
    per-chunk MP4s. After writing the final manifest, the pipeline automatically
    removes only its own chunks tree. Unknown or replaced files cause cleanup
    to fail rather than deleting unrelated data.

There is **no depth.zip and no separate mask/debug MP4 export** in this contract.
The three final videos share the same frame indices, dimensions and 15 fps rate.
Chunk 0 prepends the input anchor to its warp video as well as its RGB/depth
sequence: 81 frames for chunk 0, 80 for each later chunk, and **1 + 80 × N**
frames in each final video (81 / 161 / 321 / 641 for 1 / 2 / 4 / 8 chunks).

### Depth is a color visualization

**depth.mp4 is a Spectral inverse-depth color visualization, not metric depth
and not a metric-depth ZIP. Do not decode it with `read_video_depth` in
[utils/io.py](../../../utils/io.py) or use its RGB values as depth measurements.**
Invalid/skipped pixels are black. Each chunk uses its own valid inverse-depth
range; the final video uses one range across all valid output frames, so colors
can differ between the per-chunk and final videos.

For chunks that need depth, selected keyframe depths/masks are retained exactly.
Export-only DA3 fills missing frames using the nearest original selected frame
as a scale anchor, in batches of at most **8 missing frames plus 1 anchor**.
This work isolates RNG/transient masks and does not update the AR map or replace
its keyframe/overlap buffers.

The final chunk when **`c > 0 and c == num_chunks - 1`** skips frame selection,
DA3 depth prediction, map integration **and dense export prediction**. Its
internal depth is zero with an all-false mask, producing a **black final
80-frame depth tail**, not measured zero depth (indices 81–160 for 2 chunks;
241–320 for the bundled four-chunk full case; 561–640 for an eight-chunk
rollout). Chunk 0, including a single-chunk run, keeps its complete bootstrap.
Library return values `frames`, `depths`, and `masks` remain
frame-aligned; returned cameras retain sparse map buffers followed by final-chunk
user cameras. The input anchor depth sets trajectory scale; it is not the
exported depth sequence.

### Sampling and condition hashes

The root sha256.json records **actual tensors immediately before
`latent_sampling`**: the hashed initial noise is the **same tensor object**
passed to that call, not regenerated from the seed. Hashing uses contiguous raw
tensor bytes without a dtype conversion or additional random draw.

Each chunk records the case seed (`base_seed`), AR chunk index, capture point,
generator/stream seed and device, original sampling dtype/device, and the actual
tensor's SHA-256, shape, dtype and device. Sampling uses one continuous CPU
float32 generator per case, then casts to the model dtype/device; it is not
reseeded per chunk. The same JSON also hashes the prepared `history_latents`,
`rendered_latents`, `rendered_masks`, `clip_features` and positive `raymap`, plus
raw warp RGB/mask snapshots and encoded-video file hashes.

With `noise_aug_level > 0`, it additionally records the **actual history and
rendered augmentation noise** hashes and metadata from the independent
`case_seed + 1` generator. Augmentation happens once per chunk and is reused
throughout denoising/CFG. At the default level 0 (or `None`) it is disabled.

These hashes aid reproducibility diagnosis; matching initial noise alone does
not establish matching conditions, denoised latents or pixel-identical videos.

