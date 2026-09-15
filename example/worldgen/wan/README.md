# Wan World Generation — single-case driver

AR world generation with the Wan 14B FramePack backbone
([`WanWorldPipeline`](../../../octworld/pipelines/wan/wanworld_pipeline.py)).

[`world_gen.py`](world_gen.py) runs **one** case. To run **many** cases on a
single GPU (with sharding, fail isolation, and resume), use the batch driver
under [`eval/wan/`](../../../eval/wan/README.md).

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

The default geometry setting matches the finalized evaluation setting:
`align_with_user` depth prediction, clean history (`noise_aug_level=0`), RAFT
keyframe selection (threshold 5, maximum interval 8), and OctMap integration at
DA3's native 294x504 resolution while rendering at 480x832.

OctMap integration is **online** by default: each map update allocates and
integrates only the newly selected frames. `--offline_integration` retains the
octree structure but clears its observation weights and re-integrates all
cached frames after each update. Offline mode therefore does not rebuild the
octree from scratch, but its update cost grows with the accumulated history.

## Run

```bash
python example/worldgen/wan/world_gen.py \
    --image assets/ParkingLot/input_image.png \
    --trajectory assets/ParkingLot/camera_trajectory.npz \
    --anchor_depth assets/ParkingLot/moge_depth.png \
    --prompt_file assets/ParkingLot/prompt.txt \
    --dit_ckpt model_weights/octworld-1.5/model \
    --model_weights_dir model_weights \
    --save_dir tmp/wan_world_test/
```

`pipe_config.json` and `neg_text_embed.pt` are auto-resolved from the
OctWorld model root that holds the checkpoint (the parent directory containing
`pipe_config.json` above `--dit_ckpt`).

All arguments default to the ParkingLot case. Add `--debug --max_chunks 1` for a
fast tiny-model smoke run, or `--preview_only` to render the warp without
diffusion.

