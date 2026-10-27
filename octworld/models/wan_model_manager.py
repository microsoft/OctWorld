"""
WanModelManager: Manages loading of Wan 2.1 I2V-14B-480P pretrained components.

Components:
- vae: WanVAE (z_dim=16, vae_stride=(4,8,8))
- t5: T5EncoderModel (umt5-xxl text encoder)
- clip: CLIPModel (xlm-roberta ViT-H/14, I2V image encoder)

Local weights and tokenizers: model_weights/Wan2.1-I2V-14B-480P/.
Missing assets raise FileNotFoundError; this manager does not download them.
"""
import logging
from typing import Any, Dict

import torch
import torch.nn as nn

from .manager import ModelManager

logger = logging.getLogger(__name__)

_AVAILABLE_COMPONENTS = {
    "vae": "Wan 2.1 VAE (z_dim=16, stride 4x8x8)",
    "t5": "UMT5-XXL text encoder (bf16)",
    "clip": "XLM-Roberta CLIP ViT-H/14 image encoder (Wan I2V, fp16)",
}


class WanModelManager(ModelManager):
    """
    Model manager for local Wan 2.1 I2V-14B-480P VAE / T5 / CLIP components.

    Usage:
        manager = WanModelManager(base_path="./model_weights")

        # Load VAE
        vae = manager.get_component("vae", device="cuda")

        # Load T5 text encoder
        t5 = manager.get_component("t5", torch_dtype=torch.bfloat16, device="cuda")

        # Load CLIP image encoder
        clip = manager.get_component("clip", torch_dtype=torch.float16, device="cuda")
    """

    def __init__(
        self,
        base_path: str = "./model_weights",
        model_dir: str = "Wan2.1-I2V-14B-480P",
    ):
        super().__init__(base_path=base_path)
        self.model_dir = self.base_path / model_dir

    def available_components(self) -> Dict[str, str]:
        return _AVAILABLE_COMPONENTS.copy()

    def _load_component(self, name: str) -> Any:
        if name == "vae":
            return self._load_vae()
        elif name == "t5":
            return self._load_t5()
        elif name == "clip":
            return self._load_clip()
        else:
            raise ValueError(f"Unknown component: {name}")

    def _apply_common_settings(self, model: Any, dtype, device: str, freeze: bool) -> Any:
        """Override base: handle T5EncoderModel (not nn.Module) specially."""
        if isinstance(model, nn.Module):
            return super()._apply_common_settings(model, dtype, device, freeze)

        # T5EncoderModel wraps self.model (nn.Module) and self.device
        if hasattr(model, 'model') and isinstance(model.model, nn.Module):
            if dtype is not None:
                model.model = model.model.to(dtype=dtype)
            model.model = model.model.to(device=device)
            model.model.eval()
            if freeze:
                model.model.requires_grad_(False)
            # Update wrapper's device reference
            if hasattr(model, 'device'):
                model.device = torch.device(device)
            return model

        return model

    # ==================== VAE ====================

    def _load_vae(self) -> Any:
        """Load Wan 2.1 VAE."""
        from .wan.modules.vae import WanVAE

        vae_path = self.model_dir / "Wan2.1_VAE.pth"
        if not vae_path.is_file():
            raise FileNotFoundError(f"VAE weights not found at: {vae_path}")

        logger.info(f"Loading VAE from: {vae_path}")
        vae = WanVAE(vae_pth=str(vae_path), device="cpu")
        return vae.model  # Return the inner nn.Module for standard handling

    # ==================== T5 ====================

    def _load_t5(self) -> Any:
        """Load UMT5-XXL text encoder."""
        from .wan.modules.t5 import T5EncoderModel

        t5_checkpoint = self.model_dir / "models_t5_umt5-xxl-enc-bf16.pth"
        tokenizer_path = self.model_dir / "google" / "umt5-xxl"

        if not t5_checkpoint.is_file():
            raise FileNotFoundError(f"T5 weights not found at: {t5_checkpoint}")
        if not tokenizer_path.is_dir():
            raise FileNotFoundError(f"T5 tokenizer not found at: {tokenizer_path}")

        logger.info(f"Loading T5 from: {t5_checkpoint}")
        t5 = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            checkpoint_path=str(t5_checkpoint),
            tokenizer_path=str(tokenizer_path),
        )
        return t5

    # ==================== CLIP ====================

    def _load_clip(self) -> Any:
        """Load Wan 2.1 I2V CLIP image encoder (XLM-Roberta ViT-H/14)."""
        from .wan.modules.clip import CLIPModel

        clip_checkpoint = self.model_dir / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
        tokenizer_path = self.model_dir / "xlm-roberta-large"
        if not clip_checkpoint.is_file():
            raise FileNotFoundError(f"CLIP weights not found at: {clip_checkpoint}")
        if not tokenizer_path.is_dir():
            raise FileNotFoundError(f"CLIP tokenizer not found at: {tokenizer_path}")

        print(f"Loading CLIP from: {clip_checkpoint}")
        clip = CLIPModel(
            dtype=torch.float16,
            device=torch.device("cpu"),
            checkpoint_path=str(clip_checkpoint),
            tokenizer_path=str(tokenizer_path),
        )
        return clip

    # ==================== Utilities ====================

    def get_vae_wrapper(self, device: str = "cuda", dtype: torch.dtype = torch.float32):
        """
        Get the full WanVAE wrapper (with encode/decode and normalization).
        Unlike get_component("vae") which returns the raw nn.Module,
        this returns the WanVAE class with mean/std normalization.
        """
        from .wan.modules.vae import WanVAE

        vae_path = self.model_dir / "Wan2.1_VAE.pth"
        if not vae_path.is_file():
            raise FileNotFoundError(f"VAE weights not found at: {vae_path}")

        return WanVAE(vae_pth=str(vae_path), device=device, dtype=dtype)
