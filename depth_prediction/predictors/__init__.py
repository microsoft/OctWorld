import importlib

from depth_prediction.predictors.base import DepthPredictorOutput

# Predictor class -> defining module. Imported lazily so one predictor's optional
# deps (e.g. da3_streaming's loop_utils) never break the others.
_PREDICTOR_MODULES = {
    "DA3Predictor": "depth_prediction.predictors.da3_predictor",
    "DepthStreaming": "depth_prediction.predictors.depth_streaming",
    "LingBotMapPredictor": "depth_prediction.predictors.lingbot_map_predictor",
    "LingBotMapStreaming": "depth_prediction.predictors.lingbot_map_predictor",
    "MoGePredictor": "depth_prediction.predictors.moge_predictor",
    "Pi3Predictor": "depth_prediction.predictors.pi3_predictor",
    "VGGTPredictor": "depth_prediction.predictors.vggt_predictor",
    "VGGTOmegaPredictor": "depth_prediction.predictors.vggt_omega_predictor",
}

PREDICTOR_REGISTRY = {
    "da3": "DA3Predictor",
    "lingbot_map": "LingBotMapPredictor",
    "moge": "MoGePredictor",
    "pi3": "Pi3Predictor",
    "vggt": "VGGTPredictor",
    "vggt_omega": "VGGTOmegaPredictor",
}


def __getattr__(name: str):
    """Lazily import a predictor class on first access (PEP 562)."""
    module_path = _PREDICTOR_MODULES.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_path), name)


def create_predictor(name: str, **kwargs):
    """Factory function to create a depth predictor by name.

    Args:
        name: One of "moge", "pi3", "vggt".
        **kwargs: Forwarded to the predictor constructor.

    Returns:
        A depth predictor instance.
    """
    name = name.lower()
    if name not in PREDICTOR_REGISTRY:
        raise ValueError(f"Unknown predictor '{name}'. Available: {list(PREDICTOR_REGISTRY.keys())}")
    return __getattr__(PREDICTOR_REGISTRY[name])(**kwargs)


__all__ = [
    "create_predictor",
    "DA3Predictor",
    "DepthPredictorOutput",
    "DepthStreaming",
    "LingBotMapPredictor",
    "LingBotMapStreaming",
    "MoGePredictor",
    "Pi3Predictor",
    "VGGTOmegaPredictor",
    "VGGTPredictor",
]
