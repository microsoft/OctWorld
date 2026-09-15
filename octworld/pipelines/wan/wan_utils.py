import torch
from contextlib import contextmanager
from typing import List


class WanToolsMixin:
    """Reusable utility methods for Wan-based pipelines.

    Expects the host class to have:
        - self.t5       (T5EncoderModel)
        - self.vae      (WanVAE wrapper)
        - self.clip     (CLIPModel, only for encode_clip_features)
        - self.device
        - self.torch_dtype

    Optional:
        - self.cpu_offload (bool): when True, the text/image conditioning
          encoders (T5 / CLIP) are kept on CPU and only moved to ``self.device``
          for the duration of the encode call, then moved back to CPU. The VAE
          is unaffected. Defaults to False when the attribute is absent (e.g.
          during training).
    """

    # ------------------------------------------------------------------
    # CPU offload helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _move_inference_module(wrapper, device):
        """Move a T5 / CLIP wrapper (and its ``device`` attribute) to
        ``device`` in place."""
        if wrapper is None:
            return
        device = torch.device(device)
        if getattr(wrapper, "model", None) is not None:
            wrapper.model.to(device)
        if hasattr(wrapper, "device"):
            wrapper.device = device

    @contextmanager
    def _offload_ctx(self, wrapper):
        """Context manager that, when ``self.cpu_offload`` is True, moves
        ``wrapper`` to ``self.device`` on enter and back to CPU on exit."""
        if not getattr(self, "cpu_offload", False) or wrapper is None:
            yield
            return
        self._move_inference_module(wrapper, self.device)
        try:
            yield
        finally:
            self._move_inference_module(wrapper, "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    def encode_prompt(self, prompt: str) -> List[torch.Tensor]:
        """Encode text prompt via T5 (UMT5-XXL).

        Args:
            prompt: Text string (or list of strings) to encode.

        Returns:
            List of [L_text, 4096] text embeddings (one per sample).
            Each tensor is variable-length (trimmed to actual token count).
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        with self._offload_ctx(self.t5):
            t5_device = self.t5.device if hasattr(self.t5, 'device') else self.device
            context = self.t5(prompt, t5_device)
        return [c.to(dtype=self.torch_dtype) for c in context]

    # ------------------------------------------------------------------
    # VAE encode / decode
    # ------------------------------------------------------------------

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video tensor to latents using Wan VAE.

        Args:
            video: [B, C, T, H, W] or [C, T, H, W] float tensor in [-1, 1].
                   Or list of [C, T, H, W] tensors.

        Returns:
            Latents [B, 16, T/4, H/8, W/8] float32.
        """
        if isinstance(video, list):
            latents = self.vae.encode(video)
            return torch.stack(latents, dim=0)
        elif video.dim() == 5:
            latents = self.vae.encode([video[i] for i in range(video.shape[0])])
            return torch.stack(latents, dim=0)
        else:
            latents = self.vae.encode([video])
            return latents[0].unsqueeze(0)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to video using Wan VAE.

        Args:
            latents: [B, 16, T/4, H/8, W/8] or [16, T/4, H/8, W/8] float tensor.

        Returns:
            Video [B, C, T, H, W] float in [-1, 1].
        """
        if latents.dim() == 5:
            videos = self.vae.decode([latents[i] for i in range(latents.shape[0])])
            return torch.stack(videos, dim=0)
        else:
            videos = self.vae.decode([latents])
            return videos[0].unsqueeze(0)

    def encode_frames_to_latents(self, frames_fhw3: torch.Tensor) -> torch.Tensor:
        """Encode pixel frames (F, H, W, 3) [0,1] to latents (1, 16, F/4, H/8, W/8).

        Handles pixel-to-VAE-input conversion (normalize to [-1, 1]).
        """
        # (F, H, W, 3) → (C, F, H, W) in [-1, 1]
        video = frames_fhw3.permute(3, 0, 1, 2).float() * 2.0 - 1.0
        latents = self.vae.encode([video.to(self.vae.device)])
        return latents[0].unsqueeze(0).cpu().float()

    def decode_latents_to_frames(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents (1, 16, T, H', W') to pixel frames (F, H, W, 3) [0,1]."""
        videos = self.vae.decode([latents[0].to(self.vae.device)])
        # videos[0]: [C, T*4, H*8, W*8] in [-1, 1]
        video = videos[0].cpu().float()
        frames = (video.clamp(-1, 1) + 1.0) / 2.0  # [C, F, H, W]
        frames = frames.permute(1, 2, 3, 0)  # [F, H, W, 3]
        return frames

    def encode_depth_to_latents(self, depth_fhw: torch.Tensor) -> torch.Tensor:
        """Encode (F, H, W) depth [0,1] to VAE latents by triplicating channels."""
        x = depth_fhw.unsqueeze(-1).repeat(1, 1, 1, 3)  # (F, H, W, 3)
        return self.encode_frames_to_latents(x)

    def decode_latents_to_depth(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents → (F, H, W) depth by averaging RGB channels."""
        frames = self.decode_latents_to_frames(latents)  # (F, H, W, 3)
        return frames.mean(dim=-1)

    # ------------------------------------------------------------------
    # CLIP image features (Wan I2V conditioning)
    # ------------------------------------------------------------------

    def encode_clip_features(self, image_hw3: torch.Tensor) -> torch.Tensor:
        """Encode a single first-frame image to Wan I2V CLIP features.

        Mirrors ``datapipe/wan_pipes/extract_clip_features.py``: the image is
        turned into a ``(3, 1, H, W)`` tensor in [-1, 1] and run through the
        CLIP vision tower (which handles resize + normalization internally).

        Args:
            image_hw3: (H, W, 3) float tensor in [0, 1], RGB.

        Returns:
            (1, n_tokens, 1280) float32 CLIP features.
        """
        # (H, W, 3) [0,1] → (3, 1, H, W) [-1, 1]
        t = image_hw3.permute(2, 0, 1).unsqueeze(1).float() * 2.0 - 1.0
        with self._offload_ctx(self.clip):
            clip_device = self.clip.device if hasattr(self.clip, "device") else self.device
            with torch.no_grad():
                feats = self.clip.visual([t.to(clip_device)])  # (1, n_tokens, 1280)
        return feats.float().cpu()
