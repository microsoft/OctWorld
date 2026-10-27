import torch
from utils.vision import VisionPipelineMixin
from utils.load import load_state_dict, GeneralLoRALoader
from typing import Any, Dict
from abc import abstractmethod, ABC
from .scheduler import FlowMatchScheduler


class DiffusionPipeline(torch.nn.Module, VisionPipelineMixin, ABC):

    def __init__(
        self,
        device="cuda", torch_dtype=torch.float16,
        height_division_factor=64, width_division_factor=64,
        time_division_factor=None, time_division_remainder=None,
    ):
        super().__init__()
        # The device and torch_dtype is used for the storage of intermediate variables, not models.
        self.device = device
        self.torch_dtype = torch_dtype
        # The following parameters are used for shape check.
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.scheduler : FlowMatchScheduler = None
        
        
    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        super().to(*args, **kwargs)
        return self

    def check_resize_height_width(self, height, width, num_frames=None, verbose=1):
        # Shape check
        if height % self.height_division_factor != 0:
            height = (height + self.height_division_factor - 1) // self.height_division_factor * self.height_division_factor
            if verbose > 0:
                print(f"height % {self.height_division_factor} != 0. We round it up to {height}.")
        if width % self.width_division_factor != 0:
            width = (width + self.width_division_factor - 1) // self.width_division_factor * self.width_division_factor
            if verbose > 0:
                print(f"width % {self.width_division_factor} != 0. We round it up to {width}.")
        if num_frames is None:
            return height, width
        else:
            if num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames = (num_frames + self.time_division_factor - 1) // self.time_division_factor * self.time_division_factor + self.time_division_remainder
                if verbose > 0:
                    print(f"num_frames % {self.time_division_factor} != {self.time_division_remainder}. We round it up to {num_frames}.")
            return height, width, num_frames


    def generate_noise(self, shape, seed=None, generator=None, rand_device="cpu", rand_torch_dtype=torch.float32, device=None, torch_dtype=None):
        # Initialize Gaussian noise
        if seed is not None and generator is not None:
            raise ValueError("Pass either seed or generator, not both")
        if generator is None and seed is not None:
            generator = torch.Generator(rand_device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=rand_device, dtype=rand_torch_dtype)
        noise = noise.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        return noise
    
    def get_module(self, model, name):
        if "." in name:
            name, suffix = name[:name.index(".")], name[name.index(".") + 1:]
            if name.isdigit():
                return self.get_module(model[int(name)], suffix)
            else:
                return self.get_module(getattr(model, name), suffix)
        else:
            return getattr(model, name)
    
    def freeze_except(self, model_names):
        self.eval()
        self.requires_grad_(False)
        for name in model_names:
            module = self.get_module(self, name)
            if module is None:
                print(f"No {name} models in the pipeline. We cannot enable training on the model. If this occurs during the data processing stage, it is normal.")
                continue
            module.train()
            module.requires_grad_(True)
                
    
    def load_lora(
        self,
        module: torch.nn.Module,
        lora_config: str = None,
        alpha=1,
        state_dict=None,
    ):
        # notice this lora loading pipeline will directly fuse the LoRA weights to the base model
        # so this is not suitable for training
        if state_dict is None:
            lora = load_state_dict(lora_config, torch_dtype=self.torch_dtype, device=self.device)
        else:
            lora = state_dict
        lora_loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
        lora = lora_loader.convert_state_dict(lora)
        lora_loader.fuse_lora_to_base_model(module, lora, alpha=alpha)

    def compile_pipeline(self, mode: str = "default", dynamic: bool = True, fullgraph: bool = False, compile_models: list = None, **kwargs):
        """
        compile the pipeline with torch.compile. The models that will be compiled are determined by the `compilable_models` attribute of the pipeline.
        If a model has `_repeated_blocks` attribute, we will compile these blocks with regional compilation. Otherwise, we will compile the whole model.
        See https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile for details about compilation arguments.
        Args:
            mode: The compilation mode, which will be passed to `torch.compile`, options are "default", "reduce-overhead", "max-autotune" and "max-autotune-no-cudagraphs. Default to "default".
            dynamic: Whether to enable dynamic graph compilation to support dynamic input shapes, which will be passed to `torch.compile`. Default to True (recommended).
            fullgraph: Whether to use full graph compilation, which will be passed to `torch.compile`. Default to False (recommended).
            compile_models: The list of model names to be compiled. If None, we will compile the models in `pipeline.compilable_models`. Default to None.
            **kwargs: Other arguments for `torch.compile`.
        """
        compile_models = compile_models or getattr(self, "compilable_models", [])
        if len(compile_models) == 0:
            print("No compilable models in the pipeline. Skip compilation.")
            return
        for name in compile_models:
            model = getattr(self, name, None)
            if model is None:
                print(f"Model '{name}' not found in the pipeline.")
                continue
            repeated_blocks = getattr(model, "_repeated_blocks", None)
            # regional compilation for repeated blocks.
            if repeated_blocks is not None:
                for submod in model.modules():
                    if submod.__class__.__name__ in repeated_blocks:
                        submod.compile(mode=mode, dynamic=dynamic, fullgraph=fullgraph, **kwargs)
            # compile the whole model.
            else:
                model.compile(mode=mode, dynamic=dynamic, fullgraph=fullgraph, **kwargs)
            print(f"{name} is compiled with mode={mode}, dynamic={dynamic}, fullgraph={fullgraph}.")
    
    @torch.no_grad()
    def latent_sampling(
        self,
        noisy_latents: torch.Tensor,
        cfg_scale: float,
        cfg_merged: bool,
        input_shared: Dict[str, Any],
        input_pos: Dict[str, Any],
        input_neg: Dict[str, Any],
    ) -> torch.Tensor:
        """Iterative flow-match sampling loop (euler / scheduler ``step_fn``).

        Generic across pipelines: it only relies on ``self.scheduler.timesteps``,
        ``self.cfg_forward`` and ``self.scheduler.step_fn``. Call
        ``self.scheduler.set_timesteps(...)`` before invoking this. The
        per-step model evaluation (and its conditioning inputs) is delegated to
        the subclass ``model_fn`` via ``cfg_forward``.
        """
        for timestep in self.scheduler.timesteps:
            # The scheduler step (sample + v * (sigma_-sigma)) uses fp32 sigmas,
            # which promotes the latent back to fp32 between steps. Cast it to
            # the pipeline compute dtype so the model's conv/linear inputs match
            # the weights.
            noisy_latents = noisy_latents.to(dtype=self.torch_dtype)
            model_output = self.cfg_forward(
                noisy_latents=noisy_latents,
                timestep=timestep,
                cfg_scale=cfg_scale,
                cfg_merged=cfg_merged,
                input_shared=input_shared,
                input_pos=input_pos,
                input_neg=input_neg,
            )
            noisy_latents = self.scheduler.step_fn(model_output, timestep, noisy_latents)
        return noisy_latents

    @abstractmethod
    def model_fn(self, noisy_latents, timestep, *args, **kwargs):
        """The model function that will be passed to the sampler. The arguments and return values should be designed according to the requirements of the sampler."""
        pass
    
    @torch.inference_mode()
    def cfg_forward(self, 
        noisy_latents : torch.Tensor, 
        timestep : torch.Tensor, 
        cfg_scale : float,
        cfg_merged : bool = False,
        input_shared : Dict[str, Any] = {},
        input_pos : Dict[str, Any] = {},
        input_neg : Dict[str, Any] = {},):
        """Forward pass with classifier-free guidance."""
        if cfg_scale == 0.0:
            return self.model_fn(noisy_latents, timestep, **input_shared, **input_neg)
        if cfg_scale == 1.0:
            return self.model_fn(noisy_latents, timestep, **input_shared, **input_pos)
        
        if cfg_merged:
            # Merged CFG batches the positive/negative conditioning along the
            # batch dim, so replicate the (batch-1) sampling state to match
            # before the single forward, then split the output back.
            n = noisy_latents.shape[0]
            merged_latents = torch.cat([noisy_latents, noisy_latents], dim=0)
            merged_timestep = timestep
            if isinstance(timestep, torch.Tensor) and timestep.dim() > 0:
                merged_timestep = torch.cat([timestep, timestep], dim=0)
            pred = self.model_fn(merged_latents, merged_timestep, **input_shared, **input_pos)
            pred_pos, pred_neg = pred[:n], pred[n:]
        else:
            pred_pos = self.model_fn(noisy_latents, timestep, **input_shared, **input_pos)
            pred_neg = self.model_fn(noisy_latents, timestep, **input_shared, **input_neg)
        return pred_neg + cfg_scale * (pred_pos - pred_neg)