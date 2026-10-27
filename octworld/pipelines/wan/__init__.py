from .wan_pipeline import WanMultiAdapterPipeline
from .wan_utils import WanToolsMixin

__all__ = ["WanMultiAdapterPipeline", "WanToolsMixin", "WanWorldPipeline", "WanWorldPipeConfig"]


def __getattr__(name):
	# Loading model weights must not import camera/mapping/rendering code.
	if name in {"WanWorldPipeline", "WanWorldPipeConfig"}:
		from . import wanworld_pipeline
		return getattr(wanworld_pipeline, name)
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
