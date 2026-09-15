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

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
