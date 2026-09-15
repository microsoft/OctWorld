# OctWorld

**OctWorld: Long-Range World-Consistent Video Generation with Octree-based 3D Mapping.**

This source snapshot contains the Wan-based world-generation pipeline, Depth
Anything 3 depth prediction, and OctMap spatial memory. Generation is conditioned
on a first-frame image, camera trajectory, anchor depth, and text prompt.

## Source and models

- [example/worldgen/wan/world_gen.py](example/worldgen/wan/world_gen.py): single-case inference entry point.
- [octworld](octworld): models, pipelines, and training utilities.
- [depth_prediction](depth_prediction): depth prediction wrappers and DA3 source.
- [spatial_memory](spatial_memory): mapping and rendering.
- [assets/ParkingLot](assets/ParkingLot): example inputs.

Use a CUDA-capable environment with the project dependencies installed. This
snapshot does not yet provide a complete environment installation script.
Model weights are not included in Git. Place the required local files under
`model_weights/`:

- `octworld-1.5/`: `model/`, `pipe_config.json`, and `neg_text_embed.pt`.
- `Wan2.1-I2V-14B-480P/`: Wan VAE, T5, CLIP, and tokenizers.
- `da3/`: `config.json` and `model.safetensors`.
- `raft/` and `skyseg/`: RAFT and sky-segmentation weights.

## Example

Run from the repository root, using the prepared Python environment:

```bash
python example/worldgen/wan/world_gen.py \
    --image assets/ParkingLot/input_image.png \
    --trajectory assets/ParkingLot/camera_trajectory.npz \
    --anchor_depth assets/ParkingLot/moge_depth.png \
    --prompt_file assets/ParkingLot/prompt.txt \
    --dit_ckpt model_weights/octworld-1.5/model \
    --model_weights_dir model_weights \
    --depth_method align_with_user \
    --save_dir outputs/ParkingLot
```

The generated video is saved as `outputs/ParkingLot/output.mp4`. Use `--help`
to list the available options.

## Depth Anything 3

DA3 is located at
[depth_prediction/thirdparty/Depth-Anything-3](depth_prediction/thirdparty/Depth-Anything-3),
based on [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3).

The planned Git repository will track this directory as a pinned submodule,
preserving the snapshot's lazy export-import change. Once the submodule is
registered, clone the repository with `git clone --recurse-submodules <REPO_URL>`;
for an existing checkout, run `git submodule update --init --recursive`.