#!/usr/bin/env python3
"""Verify runtime imports/ABI and tiny CUDA kernels; never load model weights."""

from __future__ import annotations

import argparse
import importlib
from importlib import metadata
import platform
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-gpu", action="store_true", help="Imports only; does not establish GPU readiness")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 10) or platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Supported recipe: Linux x86_64, Python 3.10")
    print(f"Python: {sys.executable} ({platform.python_version()})", flush=True)
    for package, expected in {
        "torch": "2.8.0+cu126", "torchvision": "0.23.0+cu126",
        "xformers": "0.0.32.post2", "flash-attn": "2.8.3",
        "transformers": "4.57.6", "tokenizers": "0.22.2", "numpy": "2.2.6",
    }.items():
        actual = metadata.version(package)
        if actual != expected:
            raise RuntimeError(f"{package}: expected {expected}, got {actual}")
        print(f"OK {package}=={actual}", flush=True)

    sys.path.insert(0, str(PROJECT_ROOT))
    for path in (
        PROJECT_ROOT / "depth_prediction/thirdparty/Depth-Anything-3/src",
        PROJECT_ROOT / "depth_prediction/thirdparty/MoGe",
        PROJECT_ROOT / "example/customize_world_gen",
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"Missing source directory: {path}")
        sys.path.insert(0, str(path))

    import torch
    if torch.version.cuda != "12.6" or not torch.compiled_with_cxx11_abi():
        raise RuntimeError("Expected torch cu126 with CXX11 ABI enabled")
    # This guard must run before DA3 imports can cache the silent fallback FFN.
    from depth_prediction.predictors.da3_backend import require_da3_swiglu_backend

    print(f"DA3 backend: {require_da3_swiglu_backend()}", flush=True)
    modules = (
        "torchvision", "flash_attn", "nvdiffrast.torch", "utils3d", "utils3d_moge",
        "moviepy.editor", "proglog", "evo", "e3nn", "plyfile", "pillow_heif",
        "pycolmap", "av", "imageio", "imageio_ffmpeg", "onnxruntime",
        "cv2", "scipy", "safetensors", "omegaconf", "transformers", "accelerate",
        "depth_anything_3.api", "moge.model.v1", "octworld.models.wan_model_manager",
        "example.customize_world_gen.prompt_policy",
    )
    for name in modules:
        importlib.import_module(name)
        print(f"OK import {name}", flush=True)
    from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: F401

    if args.skip_gpu:
        print("PASS imports only; GPU capability and kernels were NOT checked.")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device unavailable")
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {props.name}, {props.total_memory / 2**30:.1f} GiB", flush=True)
    if props.total_memory < 70 * 10**9:
        raise RuntimeError("World generation requires at least 70 GB GPU memory")
    if props.major < 8:
        raise RuntimeError("The FlashAttention 2 recipe requires Ampere or newer")
    from flash_attn import flash_attn_func
    import nvdiffrast.torch as dr

    q = torch.zeros((1, 16, 2, 64), dtype=torch.float16, device="cuda")
    if not torch.isfinite(flash_attn_func(q, q, q)).all():
        raise RuntimeError("FlashAttention CUDA smoke failed")
    context = dr.RasterizeCudaContext(device=device)
    vertices = torch.tensor([[[-0.5, -0.5, 0, 1], [0.5, -0.5, 0, 1], [0, 0.5, 0, 1]]],
                            dtype=torch.float32, device="cuda")
    triangles = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
    raster, _ = dr.rasterize(context, vertices, triangles, resolution=[16, 16])
    if not (raster[..., 3] > 0).any():
        raise RuntimeError("nvdiffrast CUDA smoke failed")
    importlib.import_module("octworld.pipelines.wan.wanworld_pipeline")
    torch.cuda.synchronize()
    print("PASS runtime imports, DA3 fused FFN, FlashAttention and nvdiffrast CUDA smoke.\n"
          "No model weights were loaded; this is not an end-to-end generation test.")


if __name__ == "__main__":
    main()