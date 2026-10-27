"""utils — Shared utility modules (lazy-loaded)."""

import importlib as _importlib

_LAZY_MODULES = {
    "camera",
    "common",
    "contracts",
    "depth",
    "encode",
    "io",
    "keyframe",
    "load",
    "vision",
}


def __getattr__(name: str):
    if name in _LAZY_MODULES:
        return _importlib.import_module(f"utils.{name}")
    raise AttributeError(f"module 'utils' has no attribute {name!r}")
