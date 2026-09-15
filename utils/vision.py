import torch
import numpy as np
from einops import repeat, reduce
from PIL import Image

class VisionPipelineMixin:
    """基本功能：提供通用的视觉数据处理工具函数（如 VAE 输出转换为图像），以及一些视觉特有的输入输出参数规范。"""
    
    def preprocess_image(self, image, torch_dtype=None, device=None, pattern="B C H W", min_value=-1, max_value=1):
        # Transform a PIL.Image to torch.Tensor
        image = torch.Tensor(np.array(image, dtype=np.float32))
        image = image.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        image = image * ((max_value - min_value) / 255) + min_value
        image = repeat(image, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {}))
        return image


    def preprocess_video(self, video, torch_dtype=None, device=None, pattern="B C T H W", min_value=-1, max_value=1):
        # Transform a list of PIL.Image to torch.Tensor
        video = [self.preprocess_image(image, torch_dtype=torch_dtype, device=device, min_value=min_value, max_value=max_value) for image in video]
        video = torch.stack(video, dim=pattern.index("T") // 2)
        return video


    def vae_output_to_image(self, vae_output, pattern="B C H W", min_value=-1, max_value=1):
        # Transform a torch.Tensor to PIL.Image
        if pattern != "H W C":
            vae_output = reduce(vae_output, f"{pattern} -> H W C", reduction="mean")
        image = ((vae_output - min_value) * (255 / (max_value - min_value))).clip(0, 255)
        image = image.to(device="cpu", dtype=torch.uint8)
        image = Image.fromarray(image.numpy())
        return image


    def vae_output_to_video(self, vae_output, pattern="B C T H W", min_value=-1, max_value=1):
        # Transform a torch.Tensor to list of PIL.Image
        if pattern != "T H W C":
            vae_output = reduce(vae_output, f"{pattern} -> T H W C", reduction="mean")
        video = [self.vae_output_to_image(image, pattern="H W C", min_value=min_value, max_value=max_value) for image in vae_output]
        return video