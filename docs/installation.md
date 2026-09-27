# Installation Notes

This recipe follows the current **Python 3.10.21 / Linux x86_64 / A100 80 GB**
environment. World generation requires **at least 70 GB GPU memory** on an
Ampere-or-newer NVIDIA GPU. The measured environment uses driver **580.173.02**;
this is a reference version, not a claim that it is the minimum supported driver.

## System prerequisites

Install CUDA Toolkit **12.6** and a compatible NVIDIA driver before running the
Python installer. CUDA runtime wheels alone do not provide the compiler needed
by native extensions. Set `CUDA_HOME` to the toolkit directory.

On Ubuntu/Debian, the non-CUDA system prerequisites can be installed with:

```bash
sudo apt-get update
sudo apt-get install -y git build-essential ffmpeg libgl1 libegl1 libglib2.0-0
```

These are administrator commands for the user; the installer never invokes
`sudo` or changes system packages. Use your distribution's equivalent packages
on other systems. The root [README](../README.md) contains the clone and Conda
activation steps. Only Python is provisioned through Conda; project dependencies
are installed with the selected interpreter's pip.

## Installer behavior

[setup.sh](../setup.sh) operates on the activated Python 3.10 environment. It
supports an explicit interpreter and a non-installing verification mode:

```bash
PYTHON="$CONDA_PREFIX/bin/python" bash setup.sh
PYTHON="$CONDA_PREFIX/bin/python" bash setup.sh --check
```

The installation order is:

1. Build tools: setuptools, wheel, packaging, Ninja.
2. PyTorch **2.8.0+cu126**, torchvision **0.23.0+cu126**, xformers **0.0.32.post2**
   from the official PyTorch CUDA 12.6 index.
3. [Pinned runtime dependencies](../requirements.txt), constrained by
   [constraints.txt](../constraints.txt).
4. The upstream FlashAttention **2.8.3** wheel for **CPython 3.10 / CUDA 12 /
   Torch 2.8 / CXX11 ABI TRUE / Linux x86_64**. Other Python/ABI combinations need
   their own tested build; this recipe does not silently substitute a wheel.
5. nvdiffrast, utils3d, and utils3d-moge from the exact Git commits embedded in
   the installer. Native builds use the already installed Torch without build
   isolation. `MAX_JOBS` defaults to 4 and may be lowered on low-memory hosts.
6. Runtime import, GPU-capacity, and tiny CUDA-kernel checks.

No model weights are needed for installation. The recipe is derived from the
working environment and deliberately excludes unrelated packages from its full
freeze. The environment check is not a full world-generation benchmark.

## Important dependency choices

- **DA3 requires xformers SwiGLU.** Do not omit xformers or globally set a bypass
  environment variable. The existing DA3 helper handles the verified optional
  FlashAttention adapter import compatibility in a scoped manner.
- **MoviePy is pinned to 1.0.3:** DA3 imports `moviepy.editor`, which MoviePy 2
  removed.
- **NumPy is pinned to 2.2.6**, as in the working environment. The vendored DA3
  metadata requests NumPy <2, whereas current MoGe metadata requests NumPy >=2
  and includes additional v3/demo/training packages. OctWorld's wrappers load
  the pinned DA3 and MoGe sources directly; **do not install their full
  pyproject dependency sets** for this inference recipe.
- **MoGe v1** needs utils3d-moge, but not the v3 FlexGEMM, Gradio demo, or training
  pipeline dependencies. OctWorld's renderer separately uses the pinned utils3d.
- **OpenCV:** install the headless 4.10.0.84 distribution only. The existing shared
  environment contains both headless and GUI distributions; that collision is
  intentionally not reproduced in the installation recipe.
- **Torchaudio is not used** by these workflows. The shared environment's unrelated
  torchaudio installation is not copied into this recipe.
- DA3's optional 3DGS/gsplat exports and streaming loop-closure dependencies are
  not used by the released `align_with_user` inference path.

On a host without a GPU, an imports-only check is available:

```bash
python scripts/check_environment.py --skip-gpu
```

It cannot establish GPU readiness, rendering-kernel compatibility, or sufficient
VRAM. The default check requires the GPU and tests FlashAttention and nvdiffrast.

## Prepare weights

```bash
python scripts/download_models.py
python scripts/download_models.py --check
```

The downloader uses public upstream sources and verifies exact file contents.
It never imports Torch or loads pickle/checkpoint payloads. It skips only the
unpublished **OctWorld-1.5** package, not DA3, MoGe, Qwen, or other auxiliaries.
The default local layout is shown in the [README](../README.md). Keep that default
layout for the example commands: `--output-dir` is useful for staging downloads,
but does not automatically reconfigure all runtime loaders.

When using a separately supplied OctWorld-1.5 package, place its model directory,
pipeline configuration, and negative-text embedding together as documented.
Downloading all auxiliary models does **not** substitute for that package.