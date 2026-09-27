# Customize World Generation

This example turns one initial image and a user-designed camera trajectory into a clean `wan_standard` input for OctWorld. The intended workflow is:

1. Choose a reconstruction-friendly image.
2. Design WorldScore-style camera turns and use Qwen3-VL to write one prompt per turn.
3. Convert the turns into hard camera keyframes and preview the trajectory.
4. Optionally replace linear interpolation with a smooth C2 camera path.
5. Freeze the trajectory and build `wan_standard`.
6. Validate the final dataset before world generation.

The one-shot workflow writes one `wan_standard/` directory. A preview video is
optional and is useful only while iterating on camera turns.

## Prerequisites

Run commands from the repository root in the `base-pt` Conda environment. The workflow expects:

- A CUDA device.
- The vendored MoGe source under `depth_prediction/thirdparty/MoGe` and its
    `Ruicheng/moge-vitl` v1 weights, stored locally at `model_weights/moge/model.pt`.
- `model_weights/Qwen3-VL-8B-Instruct`, or another local Qwen3-VL model passed with `--model-dir`.
- `transformers>=4.57,<4.58`, `tokenizers>=0.22,<=0.23`, and `accelerate` for
    Qwen3-VL model loading.
- A DL3DV dataset already in `wan_standard` format. Its prompts are read as style examples for Qwen.

No image downloader or selector is included. Supplying an appropriate initial image is part of using this example.

Prepare both custom-input models with
`python scripts/download_models.py --models moge qwen` from the repository root.
The [downloader](../../scripts/download_models.py) pins revisions and verifies
hashes. MoGe is loaded from the local file above; missing weights produce an
actionable error rather than starting an implicit Hugging Face download.
See the [installation notes](../../docs/installation.md) for the shared runtime.

## Choose an Initial Image

Prefer a mostly static scene with clear depth ordering, recognizable structural edges, and enough surface texture for monocular reconstruction. A useful image usually has visible foreground, midground, and background content, and supports local navigation without requiring the model to invent a different room or distant region.

Avoid images dominated by:

- People, faces, crowds, human-like figures, or mannequins. These introduce potential object motion and ambiguity between scene motion and camera motion.
- Animals, moving vehicles, or screens showing people when they are visually prominent.
- Dense, repetitive, or homogeneous content that is difficult to reconstruct, such as retail shelves filled with nearly identical products.
- Mirrors, transparent walls, strong reflections, large textureless surfaces, repeated grids, or visually ambiguous corridors.
- Heavy clutter, many thin overlapping objects, severe occlusion, motion blur, or compression artifacts.
- A desired trajectory that would need to reveal a large unseen room, a distant area, or a semantic scene transition.

These are usage guidelines rather than hard validation rules. Judge the selected image together with its trajectory preview.

## Camera Movement Format

A camera movement file contains one camera turn per non-empty line. Comments beginning with `#` are ignored.

```text
orbit_left 1.0 45.0
pan_right 45.0
pull_out 1.0
```

Each turn becomes one world-generation chunk and one line in `prompt.txt`.

| Action | Arguments | Effect |
| --- | --- | --- |
| `move_left` | `magnitude` | Translate left |
| `move_right` | `magnitude` | Translate right |
| `push_in` | `magnitude` | Translate forward |
| `pull_out` | `magnitude` | Translate backward |
| `pan_left` | `angle_deg` | Rotate view left |
| `pan_right` | `angle_deg` | Rotate view right |
| `orbit_left` | `magnitude angle_deg` | Orbit around visible content to the left |
| `orbit_right` | `magnitude angle_deg` | Orbit around visible content to the right |
| `pull_left` | `magnitude angle_deg` | Move backward and left while turning |
| `pull_right` | `magnitude angle_deg` | Move backward and right while turning |
| `push_left` | `magnitude angle_deg` | Move forward and left while turning |
| `push_right` | `magnitude angle_deg` | Move forward and right while turning |

Translation is scaled by `--camera-speed`; angles are in degrees and are not scaled. Omitted arguments use action defaults, though explicit values are recommended for reusable trajectories.

The implementation supports any non-empty trajectory length and any valid action combination. For practical generation quality, prompt consistency, and runtime, keep most trajectories to approximately **8-16 turns**.

Three starting templates are provided:

- `trajectories/trajectory_1.txt`: 8 turns using orbit, pan, and pull-out actions.
- `trajectories/trajectory_2.txt`: 10 turns using pan, lateral movement, and pull-out actions.
- `assets/raw/DL3DVHotelRoom/camera_turns.txt` and
    `assets/raw/MinecraftCanyon/camera_turns.txt`: the two bundled four-turn inputs.

## Three-Stage Custom Trajectory Workflow

### 1. Plan turns and prompts

Write one WorldScore-style action per line, then generate one conservative scene
prompt per action with Qwen3-VL:

```bash
python example/customize_world_gen/generate_text_prompts.py \
        --image path/to/input_image.png \
        --camera-movement path/to/camera_movement.txt \
        --dl3dv-source path/to/DL3DV_test_train/wan_standard \
        --model-dir model_weights/Qwen3-VL-8B-Instruct \
        --output-prompt outputs/my_case/prompt.txt
```

Qwen sees the image, ordered action names, action semantics, and representative
DL3DV prompts. It does not receive movement magnitudes, angles, speed, or frame
counts. This keeps geometry decisions separate from conservative scene-content
planning.

### 2. Generate hard turns

Each movement line first creates one hard camera keyframe. Without
`--smooth-camera`, translation is interpolated linearly and rotation uses SLERP
between consecutive keyframes. Use this mode while adjusting turn order and
amplitudes because each segment follows its two endpoint turns directly.

### 3. Smooth the path (optional)

After the hard turns are approved, pass `--smooth-camera` to both preview and
build commands. Camera centers then use a natural cubic spline and rotations use
`RotationSpline`; every turn-boundary keyframe is preserved. The result has the
same `1 + 80N` frame count and prompt-to-chunk alignment, but removes velocity
discontinuities at turn boundaries.

## Preview the Camera Trajectory

Preview is optional. Use it when editing a trajectory to estimate MoGe v1 depth,
build a point cloud from the initial RGB-D frame, and inspect the warped camera
path before running the one-shot builder.

```bash
conda run -n base-pt --no-capture-output \
    python example/customize_world_gen/preview_camera_trajectory.py \
    --image path/to/input_image.png \
    --camera-movement example/customize_world_gen/trajectories/trajectory_1.txt \
    --smooth-camera \
    --output outputs/my_case/camera_trajectory_preview.mp4
```

Preview uses 16 frames per camera turn. For `N` turns, it renders:

$$
T_{preview} = 1 + 16N
$$

The first pose corresponds to the given image. Black regions are disocclusions: areas not observed by the initial RGB-D point cloud. Large or rapidly growing black regions usually mean the trajectory reveals too much unseen content. Reduce translation magnitude or rotation angle, simplify the turn order, and regenerate the preview.

The preview is a geometry diagnostic, not a prediction of final video quality. Keep iterating until the trajectory type, order, and amplitudes are acceptable. Only then freeze the movement file and proceed to prompt generation.

The preview command persists only the requested MP4. Depth, point cloud, intrinsics, and rendered depth are not written as side artifacts.

## Build `wan_standard`

Run the integrated builder directly for a one-shot conversion. If `--preview`
is provided, the builder verifies that the reviewed file exists; it does not
require one. The command generates prompts, predicts MoGe v1 anchor depth,
generates the production trajectory, validates staging, and atomically publishes
the dataset.

```bash
conda run -n base-pt --no-capture-output \
    python example/customize_world_gen/build_wan_standard.py \
    --image path/to/input_image.png \
    --camera-movement example/customize_world_gen/trajectories/trajectory_1.txt \
    --dl3dv-source path/to/DL3DV_test_train/wan_standard \
    --smooth-camera \
    --output outputs/my_case/wan_standard
```

The final trajectory uses 80 frames per camera turn:

$$
T_{final} = 1 + 80N
$$

The initial given image is the first conditioning pose. It is not part of the 80 generated frames assigned to the first turn.

The builder fixes anchor geometry to:

```text
predictor=moge
version=v1
pretrained=model_weights/moge/model.pt  # pinned Ruicheng/moge-vitl checkpoint
resolution_level=9
apply_mask=true
```

The Qwen prompt planner receives the initial image, ordered action names, action semantics, and source DL3DV prompt examples. It deliberately does not receive movement magnitudes, angles, camera speed, or frame counts. It generates exactly one concise prompt per turn, conservatively continuing the same scene instead of inventing new rooms, people, events, or unsupported objects.

Pass `--preview outputs/my_case/camera_trajectory_preview.mp4` after an optional
review. Use `--overwrite` to replace an existing output directory.

To rebuild the complete bundled dataset from the two raw cases while preserving
the original DL3DV GT scene, run:

```bash
python example/customize_world_gen/build_example_dataset.py \
    --raw-root assets/raw \
    --source-dataset assets/wan_standard \
    --output assets/wan_standard \
    --model-dir model_weights/Qwen3-VL-8B-Instruct \
    --smooth-camera \
    --overwrite
```

This loads Qwen once, then loads MoGe once for both custom images. The
`--reuse-prompts-from` option is validation-only: it isolates geometry testing
by reusing an existing dataset's prompts and must not be described as a Qwen run.
The bundled dataset was rebuilt end to end with Qwen3-VL-8B-Instruct and the
vendored MoGe v1 source; no preview file was supplied.

## Output Contract

The normal persistent output is:

```text
outputs/my_case/
├── camera_trajectory_preview.mp4
└── wan_standard/
    ├── instances.json
    └── scenes/
        └── scene_00000/
            ├── image.png
            ├── trajectory.npz
            ├── anchor_depth.png
            └── prompt.txt
```

`trajectory.npz` contains float32 OpenCV world-to-camera `extrinsics` with shape `(1 + 80N, 4, 4)` and normalized `intrinsics` with shape `(1 + 80N, 3, 3)`. Frame-zero extrinsic is identity. `prompt.txt` contains exactly `N` non-empty lines.

The builder uses a hidden sibling staging directory such as `.wan_standard.staging`. It is removed after successful publication. If a build fails, staging is retained for diagnosis; remove it before abandoning the case.

## Validate

The builder validates before publication. The validator can also be run independently without loading Qwen or MoGe:

```bash
conda run -n base-pt --no-capture-output \
    python example/customize_world_gen/validate_wan_standard.py \
    outputs/my_case/wan_standard
```

Validation checks the exact directory contents, image size, 2D anchor depth,
camera matrix shapes and finiteness, `1 + 80N` trajectory length, one prompt per
turn, valid anchor depth, and consistency between `instances.json` and scene
directories. Add `--require-identity` for a custom-only dataset; the original
DL3DV DA3 trajectory intentionally keeps its source frame-zero pose.

## Bundled Assets

The release separates two editable raw cases from one generated dataset:

| Dataset scene | Initial image | Trajectory |
| --- | --- | --- |
| `scene_00000` | DL3DV hotel room | Original DA3-streaming trajectory prefix |
| `scene_00001` | Same DL3DV image | Four custom turns with spline smoothing |
| `scene_00002` | Minecraft canyon | Four custom turns with spline smoothing |

Only `scene_00001` and `scene_00002` have editable source material under
`assets/raw/`, each containing exactly `image.png` and `camera_turns.txt`.
All generated image/depth/camera/prompt files live under `assets/wan_standard/`.

## Merge Multiple Cases

When several images or trajectories were built separately, merge their validated outputs into one clean dataset:

```bash
conda run -n base-pt --no-capture-output \
    python example/customize_world_gen/merge_wan_standard.py \
    --input path/to/case_1/wan_standard path/to/case_2/wan_standard \
    --output path/to/combined/wan_standard
```

Inputs are kept in command-line order and renumbered as `scene_00000`, `scene_00001`, and so on. The merger validates every input, writes through hidden staging, and validates the combined dataset before atomic publication.

## Optional Prompt-Only Debugging

The integrated builder is the recommended path because it keeps outputs clean. To inspect prompt behavior separately, run:

```bash
conda run -n base-pt --no-capture-output \
    python example/customize_world_gen/generate_text_prompts.py \
    --image path/to/input_image.png \
    --camera-movement path/to/camera_movement.txt \
    --dl3dv-source path/to/DL3DV_test_train/wan_standard \
    --output-prompt tmp/prompt.txt
```

This command writes only `prompt.txt`. Treat it as temporary debugging output; it is not required by the normal preview-and-build workflow.

## Use with World Generation

The resulting `instances.json` is directly consumable by the batch world-generation
runner. Each prompt line controls its corresponding camera turn automatically:

```bash
python example/worldgen/wan/batch_world_gen.py \
    --instances outputs/my_case/wan_standard/instances.json \
    --base_folder path/to/world_gen_outputs \
    --dit_ckpt path/to/checkpoint.pt \
    --model_weights_dir model_weights
```