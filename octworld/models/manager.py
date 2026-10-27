import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Base model manager with common behavior:
    - Lazy caching
    - dtype / device / freeze / eval application
    - Subclass to define specific loading logic
    """

    def __init__(self, base_path: str = "./model_weights"):
        self.base_path = Path(base_path).resolve()
        self._cache: Dict[str, Any] = {}

    # ==================== Public API ====================

    def load_components(
        self,
        components: List[str],
        torch_dtype: Optional[torch.dtype] = None,
        device: str = "cpu",
        freeze: bool = False,
    ) -> Dict[str, Any]:
        """Batch load components by name."""
        loaded = {}
        for name in components:
            loaded[name] = self.get_component(name, torch_dtype, device, freeze)
        return loaded

    def get_component(
        self,
        name: str,
        torch_dtype: Optional[torch.dtype] = None,
        device: str = "cpu",
        freeze: bool = False,
    ) -> Any:
        """Load a single component (cached)."""
        if name in self._cache:
            return self._cache[name]

        if name not in self.available_components():
            raise ValueError(
                f"Component '{name}' is not available. "
                f"Available: {list(self.available_components())}"
            )

        logger.info(f"Loading component '{name}' ...")
        model = self._load_component(name)
        model = self._apply_common_settings(model, torch_dtype, device, freeze)
        self._cache[name] = model
        logger.info(f"Component '{name}' loaded and cached.")
        return model

    def clear_cache(self, name: Optional[str] = None):
        """Release cached models."""
        if name is None:
            self._cache.clear()
        elif name in self._cache:
            del self._cache[name]
        torch.cuda.empty_cache()

    def available_components(self) -> Dict[str, str]:
        """Return dict of {component_name: description}. Override in subclass."""
        raise NotImplementedError

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__} (base_path={self.base_path})"]
        lines.append(f"  {'Name':<35} {'Status'}")
        lines.append(f"  {'─' * 35} {'─' * 10}")
        for name in self.available_components():
            status = "loaded" if name in self._cache else "not loaded"
            lines.append(f"  {name:<35} {status}")
        return "\n".join(lines)

    # ==================== Internal ====================

    def _load_component(self, name: str) -> Any:
        """Override in subclass to implement loading logic."""
        raise NotImplementedError

    def _apply_common_settings(
        self, model: Any, dtype: Optional[torch.dtype], device: str, freeze: bool
    ) -> Any:
        """Apply dtype, device, eval, freeze to nn.Module. Non-modules pass through."""
        if not isinstance(model, nn.Module):
            return model

        if dtype is not None:
            model = model.to(dtype=dtype)
        model = model.to(device=device)
        model.eval()

        if freeze:
            model.requires_grad_(False)

        return model