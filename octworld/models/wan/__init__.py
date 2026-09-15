# OctWorld Wan 14B FramePack Models
#
# Layout:
# - modules/ : upstream Wan2.1 components (WanModel, WanVAE, T5EncoderModel,
#              CLIPModel, etc.). Copied verbatim from Wan-AI/Wan2.1, with the
#              only edit being removal of `diffusers` ConfigMixin/ModelMixin
#              dependencies.
# - wan_framepack.py : OctWorldWanModel (Wan 14B DiT with FramePack
#                     autoregressive generation) + framepack-specific
#                     attention / patch-embedding subclasses.
# - camera_adapter.py : SimpleAdapter camera/raymap encoder (OctWorld-custom).

from .wan_framepack import OctWorldWanModel
from .camera_adapter import SimpleAdapter
from .modules.model import (
    WanModel,
    WanAttentionBlock,
    Head,
    sinusoidal_embedding_1d,
    rope_params,
    rope_apply,
)
from .modules.vae import WanVAE
from .modules.t5 import T5EncoderModel
from .modules.clip import CLIPModel

# Back-compat: WanModelManager now lives at octworld.models.wan_model_manager
from ..wan_model_manager import WanModelManager

__all__ = [
    'OctWorldWanModel',
    'WanModel',
    'SimpleAdapter',
    'WanVAE',
    'T5EncoderModel',
    'CLIPModel',
    'WanModelManager',
    'WanAttentionBlock',
    'Head',
    'sinusoidal_embedding_1d',
    'rope_params',
    'rope_apply',
]
