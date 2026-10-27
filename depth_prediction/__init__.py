import importlib

from depth_prediction.predictors import create_predictor

__all__ = ["create_predictor", "DA3Predictor", "DepthStreaming", "MoGePredictor", "Pi3Predictor", "VGGTPredictor"]


def __getattr__(name: str):
    """Lazily forward predictor classes from .predictors (PEP 562)."""
    if name in __all__:
        return getattr(importlib.import_module("depth_prediction.predictors"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
