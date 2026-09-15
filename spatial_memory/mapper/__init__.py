"""
spatial_memory.mapper — Unified mapper registry with lazy imports.

Provides `build_mapper(mapper_type, **kwargs)` to instantiate any registered
mapper. Mapper implementations are imported on-demand so that heavyweight
backends (e.g. TSDF's CUDA extensions) are NOT compiled/loaded unless the
corresponding mapper type is actually requested.

The mapper classes are also available as attributes (e.g.
`spatial_memory.mapper.TSDFMapper`) via PEP 562 ``__getattr__``; accessing
them triggers the same lazy import.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from .base_mapper import BaseMapper

# Maps mapper_type -> (submodule path, class name).
# Submodules are imported only when needed.
_MAPPER_LOCATIONS = {
    "octmap": (".octmap.octmap_mapper", "OctMapMapper"),
    "tsdf": (".tsdf.tsdf_mapper", "TSDFMapper"),
    "pointcloud": (".pointcloud.pointcloud_mapper", "PointCloudMapper"),
}

# class name -> mapper_type, for attribute-style lazy access.
_CLASS_NAME_TO_TYPE = {cls: mt for mt, (_, cls) in _MAPPER_LOCATIONS.items()}

# Cache of already-resolved classes.
_RESOLVED_CLASSES: dict[str, type] = {}


def _resolve(mapper_type: str) -> type:
    """Import and cache the mapper class for ``mapper_type``."""
    if mapper_type in _RESOLVED_CLASSES:
        return _RESOLVED_CLASSES[mapper_type]
    if mapper_type not in _MAPPER_LOCATIONS:
        raise ValueError(
            f"Unknown mapper type '{mapper_type}'. "
            f"Available: {list(_MAPPER_LOCATIONS.keys())}"
        )
    module_path, class_name = _MAPPER_LOCATIONS[mapper_type]
    module = import_module(module_path, package=__name__)
    cls = getattr(module, class_name)
    _RESOLVED_CLASSES[mapper_type] = cls
    return cls


class _LazyMapperRegistry:
    """Dict-like view over the mapper registry that resolves classes on access."""

    def __contains__(self, key: str) -> bool:
        return key in _MAPPER_LOCATIONS

    def __getitem__(self, key: str) -> type:
        return _resolve(key)

    def get(self, key: str, default=None):
        if key in _MAPPER_LOCATIONS:
            return _resolve(key)
        return default

    def keys(self):
        return _MAPPER_LOCATIONS.keys()

    def __iter__(self):
        return iter(_MAPPER_LOCATIONS)

    def __len__(self) -> int:
        return len(_MAPPER_LOCATIONS)

    def items(self):
        for k in _MAPPER_LOCATIONS:
            yield k, _resolve(k)

    def values(self):
        for k in _MAPPER_LOCATIONS:
            yield _resolve(k)


MAPPER_REGISTRY = _LazyMapperRegistry()


def build_mapper(mapper_type: str, **kwargs) -> BaseMapper:
    """Instantiate a mapper by type name.

    Args:
        mapper_type: One of 'octmap', 'tsdf', 'pointcloud'.
        **kwargs: Forwarded to the mapper constructor.

    Returns:
        An instance of the requested mapper.
    """
    return _resolve(mapper_type)(**kwargs)


def __getattr__(name: str):
    """PEP 562: lazily resolve mapper class attributes on the package."""
    if name in _CLASS_NAME_TO_TYPE:
        return _resolve(_CLASS_NAME_TO_TYPE[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_CLASS_NAME_TO_TYPE.keys()))


if TYPE_CHECKING:
    # For type checkers / IDEs only — no runtime import cost.
    from .octmap.octmap_mapper import OctMapMapper
    from .tsdf.tsdf_mapper import TSDFMapper
    from .pointcloud.pointcloud_mapper import PointCloudMapper


__all__ = [
    "BaseMapper",
    "OctMapMapper",
    "TSDFMapper",
    "PointCloudMapper",
    "MAPPER_REGISTRY",
    "build_mapper",
]
