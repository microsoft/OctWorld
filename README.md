# OctWorld: Long-Range World-Consistent Video Generation with Octree-based 3D Mapping

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2609.03919)
[![Project Page](https://img.shields.io/badge/Project-Website-blue)](https://microsoft.github.io/OctWorld/)
[![Code](https://img.shields.io/badge/GitHub-microsoft%2FOctWorld-181717?logo=github)](https://github.com/microsoft/OctWorld)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Official code repository:** [microsoft/OctWorld](https://github.com/microsoft/OctWorld).
**Model weights:** [OctWorld-1.5 on Hugging Face](https://huggingface.co/Maxtir/OctWorld-1.5).

**OctMap-Guided Generation with Precise Camera Control**

https://github.com/user-attachments/assets/3929178d-df5f-4e41-9d98-7d607948eb26

**Inside OctMap Online Integration**

https://github.com/user-attachments/assets/dd285158-6a82-48eb-ae01-b89f706cc874


**OctWorld** is a video generation framework with persistent **3D spatial memory**.
Starting from a single image, it generates explorable scenes along user-specified
camera trajectories. Its spatially adaptive octree map, **OctMap**, accumulates
geometry and appearance from generated RGB-D observations and renders spatial
conditions for subsequent chunks, helping preserve world consistency over long
camera paths and revisits.

This release provides the **Wan-based inference pipeline, OctMap, depth prediction,
and custom-input examples**. The training-data pipeline and training code are not
released yet.

## ✨ Features

### 1. Long-Range, World-Consistent Generation

Autoregressive video generation backed by a persistent 3D map, with spatial
conditioning from previously observed regions when the camera returns to them.

### 2. Adaptive Octree Spatial Memory

OctMap progressively fuses RGB-D observations into a sparse octree. Spatial
resolution adapts to image evidence, retaining useful geometry and appearance
without a uniformly dense scene representation.

### 3. User-Controlled Camera Trajectories

Condition generation on a first-frame image, camera poses, anchor depth, and text.
The customization tools turn camera-movement instructions into trajectories,
with optional smooth interpolation, MoGe anchor depth, and Qwen3-VL prompt planning.

## 🗺️ TODO / Release Checklist

- [x] Release the paper and project page.
- [x] Release single-case and batch world-generation inference code.
- [x] Release OctMap spatial memory and depth-prediction integration.
- [x] Release example inputs and custom camera-trajectory tools.
- [x] Release **OctWorld-1.5** weights on [Hugging Face](https://huggingface.co/Maxtir/OctWorld-1.5).
- [ ] Release **datapipe** — training-data processing and preparation (**not released yet**).
- [ ] Release **training code** — end-to-end training and fine-tuning (**not released yet**).

The custom-input builder below prepares **inference inputs**; it is not the
training-data pipeline listed above.

## 🛠️ Installation

### Prerequisites

- **System:** Linux x86_64, Python **3.10** (tested with 3.10.21).
- **Hardware:** an NVIDIA Ampere-or-newer GPU with **at least 70 GB of GPU memory**.
    Tested on an **A100 80 GB**.
- **Toolchain:** **CUDA Toolkit 12.6**, a compatible NVIDIA driver, a C++ compiler,
  Git, and FFmpeg. The toolkit is needed for native rendering extensions, not just
  the CUDA runtime bundled with PyTorch.
- **Runtime:** **PyTorch 2.8.0+cu126**, **torchvision 0.23.0+cu126**,
  **xformers 0.0.32.post2**, and **flash-attn 2.8.3**.

### Installation Steps

1. Clone the official repository, including the pinned depth-model submodules:

    ```bash
    git clone --recurse-submodules https://github.com/microsoft/OctWorld.git
    cd OctWorld
    ```

    For an existing checkout, initialize the submodules with
    `git submodule update --init --recursive`.

2. Activate a Python 3.10 environment. If needed, create it first:

    ```bash
    conda create -n octworld python=3.10.21 -y
    conda activate octworld
    ```

    Skip the creation command if the environment already exists.

3. Install the pinned inference and custom-input dependencies:

    ```bash
    export CUDA_HOME=/usr/local/cuda-12.6
    bash setup.sh
    ```

    [setup.sh](setup.sh) installs into the selected environment using its Python;
    it does not create environments, install system packages, or download weights.
    It uses [requirements.txt](requirements.txt), protects the CUDA stack with
    [constraints.txt](constraints.txt), and pins native/Git dependencies to the
    tested revisions.

4. Check an existing environment without reinstalling anything:

    ```bash
    python scripts/check_environment.py
    ```

    This checks imports, DA3's fused FFN, GPU capacity, and small FlashAttention /
    nvdiffrast CUDA operations without loading any model weights.

## 📦 Pretrained Models

### OctWorld-1.5

**OctWorld-1.5** is the Wan 14B-based world-generation model, available at
[Maxtir/OctWorld-1.5](https://huggingface.co/Maxtir/OctWorld-1.5) on Hugging Face.
See the [inference guide](example/worldgen/wan/README.md) for model loading and
generation options.

Download the complete model package from the repository root after installation:

```bash
hf download Maxtir/OctWorld-1.5 --local-dir model_weights/octworld-1.5
```

The `hf` CLI is included in the installed `huggingface-hub` dependency. This
downloads the checkpoint, pipeline configuration, and negative-text embedding
into the layout shown below. Generation requires **both OctWorld-1.5 and the
auxiliary models**; the auxiliary downloader does not download OctWorld-1.5.

### Download All Auxiliary Models

```bash
python scripts/download_models.py
```

The [downloader](scripts/download_models.py) prepares **all six auxiliary components**
for inference and custom-input preparation. Every Hugging Face revision and file
hash is pinned in the [model manifest](scripts/model_manifest.json).

| Component | Upstream source | Used for |
| :--- | :--- | :--- |
| Wan VAE / T5 / CLIP + tokenizers | [Wan2.1-I2V-14B-480P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P) | Image, text, and video encoding/decoding; the original Wan DiT is **not** downloaded. |
| DA3 Nested Giant-Large 1.1 | [DA3NESTED-GIANT-LARGE-1.1](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1) | Per-chunk depth aligned to user cameras. |
| RAFT Large | [Torchvision checkpoint](https://download.pytorch.org/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth) | Keyframe selection. |
| Sky segmentation | [skyseg](https://huggingface.co/JianyuanWang/skyseg) | Sky masking before integration. |
| MoGe v1 | [moge-vitl](https://huggingface.co/Ruicheng/moge-vitl) | Anchor depth and intrinsics for custom inputs. |
| Qwen3-VL-8B-Instruct | [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | Prompt planning for custom inputs. |

These auxiliary files occupy approximately **42.44 GB (39.53 GiB)**, excluding
OctWorld-1.5, temporary downloads, and generated outputs. Matching local files
are verified and reused; available matching Hugging Face cache files are copied
into the expected layout. Mismatched existing files are reported, never silently
overwritten. Hugging Face transfers support resuming incomplete downloads.

List the plan without network access or file changes, or verify all downloaded
files offline (including their hashes):

```bash
python scripts/download_models.py --list
python scripts/download_models.py --check
```

Use `--models wan da3 raft skyseg` for bundled-case inference only, or
`--models moge qwen` for custom-input preparation only. The default downloads both
groups.

The default output is the repository's local model directory, independent of the
shell's working directory. Model weights are **not stored in Git**:

```text
model_weights/
├── octworld-1.5/
│   ├── model/
│   ├── pipe_config.json
│   └── neg_text_embed.pt
├── Wan2.1-I2V-14B-480P/    # Wan VAE, T5, CLIP, and tokenizers
├── da3/                   # config.json and model.safetensors
├── raft/                  # RAFT keyframe-selection weights
├── skyseg/                # Sky-segmentation weights
├── moge/
│   └── model.pt            # MoGe v1, loaded locally (no implicit download)
└── Qwen3-VL-8B-Instruct/   # Only needed for custom prompt planning
```

The OctWorld checkpoint uses the **Wan 14B backbone**. Its pipeline configuration
and negative-text embedding are resolved from the model root above the DiT
checkpoint directory. MoGe v1 anchor-depth estimation is needed only when building
new inputs; the bundled inference examples already contain anchor depth and prompts,
so they do not require MoGe or Qwen3-VL to be run again.

## 🚀 Usage

Run all commands from the repository root, with the environment and model weights
prepared. Use a **new or empty output directory** for every generation run;
existing results are not overwritten.

### 1. Image-to-World Generation

The [single-case example](example/worldgen/wan/world_gen.py) defaults to the
bundled DL3DV scene with its original camera trajectory:

```bash
python example/worldgen/wan/world_gen.py \
        --dit_ckpt model_weights/octworld-1.5/model \
        --model_weights_dir model_weights \
        --save_dir outputs/dl3dv_gt
```

This example generates **4 chunks / 321 frames** at **480 × 832 (H × W), 15 fps**.

Successful single-case runs contain:

| Output | Description |
| :--- | :--- |
| `output.mp4` | Generated RGB video. |
| `depth.mp4` | Colorized inverse-depth visualization, **not metric depth**. |
| `warp.mp4` | Rendered spatial conditions used during generation. |
| `sha256.json` | Seed information and sampling/conditioning tensor hashes. |

The final chunk of a multi-chunk run does not update the map, so its depth-video
segment is black. The default uses **online OctMap integration**, DA3 depth
aligned to user cameras, and seed 425. See the
[inference guide](example/worldgen/wan/README.md) for output semantics, backend
requirements, and reproducibility limits; use `--help` for CLI options.

### 2. Batch Generation

Run all three [bundled cases](assets/README.md) with one model load:

```bash
python example/worldgen/wan/batch_world_gen.py \
    --instances assets/standard/instances.json \
        --dit_ckpt model_weights/octworld-1.5/model \
        --model_weights_dir model_weights \
        --base_folder outputs/batch
```

Results are written under `outputs/batch/instances/<sample_id>/`, together with
per-case metadata and success/failure markers. For multiple single-GPU workers,
the runner supports `--rank` and `--world_size`; select each worker's GPU with
`CUDA_VISIBLE_DEVICES`.

### 3. Custom Images and Camera Paths

The [customization example](example/customize_world_gen/README.md) converts an
image and a camera-turn file into the same `standard` dataset format used by batch
inference. It plans prompts with Qwen3-VL, estimates anchor depth with MoGe v1,
and validates the generated dataset before publishing it.

Build a custom case from the bundled editable Minecraft input without modifying
the original assets:

```bash
python example/customize_world_gen/build_standard.py \
        --image assets/raw/MinecraftCanyon/image.png \
        --camera-movement assets/raw/MinecraftCanyon/camera_turns.txt \
    --dl3dv-source assets/standard \
        --model-dir model_weights/Qwen3-VL-8B-Instruct \
        --smooth-camera \
    --output outputs/my_case/standard
```

Then generate its video:

```bash
python example/worldgen/wan/batch_world_gen.py \
    --instances outputs/my_case/standard/instances.json \
        --base_folder outputs/my_case/generated
```

Replace the image and camera-turn paths with your own to explore a new scene.
Each turn adds 80 generated frames; optional smoothing preserves turn-boundary
keyframes. The [customization guide](example/customize_world_gen/README.md)
documents supported movements, trajectory preview, dataset validation, and
rebuilding the complete bundled dataset.

## 🧩 Components

- **[OctWorld](octworld):** Wan-based models and autoregressive inference pipelines.
- **[Spatial Memory](spatial_memory):** OctMap mapping and spatial-condition rendering.
- **[Depth Prediction](depth_prediction):** depth-model wrappers and camera alignment.
- **[Examples and Assets](assets/README.md):** ready-to-run inputs and editable custom cases.

## ⚖️ License

The code is released under the [MIT License](LICENSE). Third-party code, model
weights, and datasets remain subject to their respective licenses and usage terms.

## Acknowledgement

We thank the following open-source projects that this work builds upon:

- [Wan2.1](https://github.com/Wan-Video/Wan2.1)
- [Lyra-2](https://github.com/nv-tlabs/lyra/tree/main/Lyra-2)
- [FramePack](https://github.com/lllyasviel/FramePack)
- [Depth Anything 3](https://github.com/ByteDance-Seed/depth-anything-3)
- [MoGe](https://github.com/microsoft/MoGe)

The [example assets](assets/README.md) include cases from
[WorldScore](https://github.com/haoyi-duan/WorldScore) and
[DL3DV-10K](https://github.com/DL3DV-10K/Dataset).

## 📚 Citation

If you find OctWorld useful for your research, please cite our paper:

```bibtex
@article{lv2026octworld,
    title   = {OctWorld: Long-Range World-Consistent Video Generation
                         with Octree-based 3D Mapping},
    author  = {Lv, Zelong and Xu, Sicheng and Xiang, Jianfeng and
                         Wang, Ruicheng and Dong, Yue and Deng, Yu and
                         Sun, Guangzhong and Yang, Jiaolong},
    journal = {arXiv preprint arXiv:2609.03919},
    year    = {2026}
}
```

## Trademarks

This project may contain trademarks or logos for projects, products, or services.
Authorized use of Microsoft trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must
not cause confusion or imply Microsoft sponsorship. Any use of third-party
trademarks or logos are subject to those third-party's policies.