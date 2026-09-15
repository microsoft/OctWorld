from safetensors import safe_open
import torch, warnings


def load_state_dict(file_path, torch_dtype=None, device="cpu", pin_memory=False, verbose=0):
    if isinstance(file_path, list):
        state_dict = {}
        for file_path_ in file_path:
            state_dict.update(load_state_dict(file_path_, torch_dtype, device, pin_memory=pin_memory, verbose=verbose))
    else:
        if verbose >= 1:
            print(f"Loading file [started]: {file_path}")
        if file_path.endswith(".safetensors"):
            state_dict = load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype, device=device)
        else:
            state_dict = load_state_dict_from_bin(file_path, torch_dtype=torch_dtype, device=device)
        # If load state dict in CPU memory, `pin_memory=True` will make `model.to("cuda")` faster.
        if pin_memory:
            for i in state_dict:
                state_dict[i] = state_dict[i].pin_memory()
        if verbose >= 1:
            print(f"Loading file [done]: {file_path}")
    return state_dict


def load_state_dict_from_safetensors(file_path, torch_dtype=None, device="cpu"):
    state_dict = {}
    with safe_open(file_path, framework="pt", device=str(device)) as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
            if torch_dtype is not None:
                state_dict[k] = state_dict[k].to(torch_dtype)
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None, device="cpu"):
    state_dict = torch.load(file_path, map_location=device, weights_only=True)
    if len(state_dict) == 1:
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "module" in state_dict:
            state_dict = state_dict["module"]
        elif "model_state" in state_dict:
            state_dict = state_dict["model_state"]
    if torch_dtype is not None:
        for i in state_dict:
            if isinstance(state_dict[i], torch.Tensor):
                state_dict[i] = state_dict[i].to(torch_dtype)
    return state_dict


class GeneralLoRALoader:
    def __init__(self, device="cpu", torch_dtype=torch.float32):
        self.device = device
        self.torch_dtype = torch_dtype
    
    
    def get_name_dict(self, lora_state_dict):
        lora_name_dict = {}
        for key in lora_state_dict:
            if ".lora_up." in key:
                lora_A_key = "lora_down"
                lora_B_key = "lora_up"
            else:
                lora_A_key = "lora_A"
                lora_B_key = "lora_B"
            if lora_B_key not in key:
                continue
            keys = key.split(".")
            if len(keys) > keys.index(lora_B_key) + 2:
                keys.pop(keys.index(lora_B_key) + 1)
            keys.pop(keys.index(lora_B_key))
            if keys[0] == "diffusion_model":
                keys.pop(0)
            keys.pop(-1)
            target_name = ".".join(keys)
            # Alpha: Deprecated but retained for compatibility.
            key_alpha = key.replace(lora_B_key + ".weight", "alpha").replace(lora_B_key + ".default.weight", "alpha")
            if key_alpha == key or key_alpha not in lora_state_dict:
                key_alpha = None
            lora_name_dict[target_name] = (key, key.replace(lora_B_key, lora_A_key), key_alpha)
        return lora_name_dict
    
    
    def convert_state_dict(self, state_dict, suffix=".weight"):
        name_dict = self.get_name_dict(state_dict)
        state_dict_ = {}
        for name in name_dict:
            weight_up = state_dict[name_dict[name][0]]
            weight_down = state_dict[name_dict[name][1]]
            if name_dict[name][2] is not None:
                warnings.warn("Alpha detected in the LoRA file. This may be a LoRA model not trained by DiffSynth-Studio. To ensure compatibility, the LoRA weights will be converted to weight * alpha / rank.")
                alpha = state_dict[name_dict[name][2]] / weight_down.shape[0]
                weight_down = weight_down * alpha
            state_dict_[name + f".lora_B{suffix}"] = weight_up
            state_dict_[name + f".lora_A{suffix}"] = weight_down
        return state_dict_


    def fuse_lora_to_base_model(self, model: torch.nn.Module, state_dict, alpha=1.0):
        updated_num = 0
        state_dict = self.convert_state_dict(state_dict)
        lora_layer_names = set([i.replace(".lora_B.weight", "") for i in state_dict if i.endswith(".lora_B.weight")])
        for name, module in model.named_modules():
            if name in lora_layer_names:
                weight_up = state_dict[name + ".lora_B.weight"].to(device=self.device, dtype=self.torch_dtype)
                weight_down = state_dict[name + ".lora_A.weight"].to(device=self.device, dtype=self.torch_dtype)
                if len(weight_up.shape) == 4:
                    weight_up = weight_up.squeeze(3).squeeze(2)
                    weight_down = weight_down.squeeze(3).squeeze(2)
                    weight_lora = alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
                else:
                    weight_lora = alpha * torch.mm(weight_up, weight_down)
                state_dict_base = module.state_dict()
                state_dict_base["weight"] = state_dict_base["weight"].to(device=self.device, dtype=self.torch_dtype) + weight_lora
                module.load_state_dict(state_dict_base)
                updated_num += 1
        print(f"{updated_num} tensors are fused by LoRA. Fused LoRA layers cannot be cleared by `pipe.clear_lora()`.")
