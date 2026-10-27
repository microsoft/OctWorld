"""Keep DA3's SwiGLU implementation from silently changing with dependencies."""

from importlib import import_module, metadata
import os
from threading import RLock


_IMPORT_LOCK = RLock()
_FLASH_VERSION_FLAG = "XFORMERS_IGNORE_FLASH_VERSION_CHECK"


def require_da3_swiglu_backend():
    """Require the xformers FFN used by DA3, not its silent PyTorch fallback.

    xformers 0.0.32.post2 eagerly imports its optional FMHA adapters when
    importing SwiGLU. That import rejects flash-attn 2.8.3 even though SwiGLU
    does not use FlashAttention at all. For the specifically verified Torch
    2.8 / CUDA12.6 + FA2.8.3 stack, bypass ONLY that optional adapter's version
    check during import and immediately restore the environment. DA3 attention
    still uses torch SDPA; Wan still calls flash_attn directly. No dependency
    versions are spoofed, and other version combinations get normal validation.
    """
    try:
        xformers_version = metadata.version("xformers")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "DA3 requires xformers SwiGLU. Missing xformers would silently change "
            "depth/confidence and the generated warp. Install a build matching "
            "your PyTorch/CUDA (verified: xformers 0.0.32.post2 for Torch 2.8/cu126)."
        ) from exc
    import torch

    try:
        flash_version = metadata.version("flash-attn")
    except metadata.PackageNotFoundError:
        flash_version = None
    verified_ffn_stack = (
        xformers_version == "0.0.32.post2"
        and torch.__version__.split("+")[0] == "2.8.0"
        and torch.version.cuda == "12.6"
        and flash_version == "2.8.3"
    )
    with _IMPORT_LOCK:
        previous = os.environ.get(_FLASH_VERSION_FLAG)
        override = verified_ffn_stack and previous is None
        try:
            if override:
                os.environ[_FLASH_VERSION_FLAG] = "1"
            ops = import_module("xformers.ops")
        except ImportError as exc:
            raise RuntimeError(
                "xformers SwiGLU could not be imported; refusing DA3's silent "
                "fallback. Check the installed PyTorch/CUDA/xformers compatibility."
            ) from exc
        finally:
            if override:
                os.environ.pop(_FLASH_VERSION_FLAG, None)

        layers = import_module("depth_anything_3.model.dinov2.layers.swiglu_ffn")
        if not layers.XFORMERS_AVAILABLE or not issubclass(layers.SwiGLUFFNFused, ops.SwiGLU):
            raise RuntimeError(
                "DA3 SwiGLU was already imported with the fallback backend. "
                "Restart the process after fixing xformers; do not reuse this model."
            )
    return {
        "swiglu": "xformers.ops.SwiGLU", "xformers": xformers_version,
        "torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
        "flash_attn": flash_version,
        "optional_fmha_import_compatibility": verified_ffn_stack,
    }