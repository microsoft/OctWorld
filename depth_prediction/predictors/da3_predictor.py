"""Depth Anything 3 depth predictor — multi-view metric depth estimation.

Requires DA3 source code in ``depth_prediction/thirdparty/Depth-Anything-3``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_model

from depth_prediction.predictors.base import BaseDepthPredictor, DepthPredictorOutput

# DA3 paths
_DA3_ROOT = Path(__file__).resolve().parent.parent / "thirdparty" / "Depth-Anything-3"
_DA3_SRC = str(_DA3_ROOT / "src")
_DA3_STREAMING_ROOT = _DA3_ROOT / "da3_streaming"
_DA3_DEFAULT_CONFIG = str(_DA3_STREAMING_ROOT / "configs" / "base_config.yaml")
_DA3_WEIGHTS_ROOT = Path(__file__).resolve().parents[2] / "model_weights" / "da3"

if _DA3_SRC not in sys.path:
    sys.path.insert(0, _DA3_SRC)


def load_da3_config(config_path: str | None = None) -> dict:
    """Load DA3 streaming config (YAML), defaulting to base_config.yaml.

    The default config uses DA3 weights and model config from the release's
    ``model_weights/da3/`` directory. Custom YAML paths retain their original
    semantics: relative weight paths resolve against ``da3_streaming/``.
    """
    _streaming_root = str(_DA3_STREAMING_ROOT)
    if _streaming_root not in sys.path:
        sys.path.insert(0, _streaming_root)
    from loop_utils.config_utils import load_config

    config = load_config(config_path or _DA3_DEFAULT_CONFIG)
    if not config_path:
        weights = config.setdefault("Weights", {})
        weights["DA3"] = str(_DA3_WEIGHTS_ROOT / "model.safetensors")
        weights["DA3_CONFIG"] = str(_DA3_WEIGHTS_ROOT / "config.json")

    # Resolve relative weight paths against da3_streaming root
    for key in ("DA3", "DA3_CONFIG", "SALAD"):
        val = config.get("Weights", {}).get(key)
        if val and not Path(val).is_absolute():
            config["Weights"][key] = str(_DA3_STREAMING_ROOT / val)
    return config


class DA3Predictor(BaseDepthPredictor):
    """Wrapper around Depth Anything 3 for multi-view depth estimation.

    Args:
        config: DA3 streaming config dict. If ``None``, loads the default
            ``da3_streaming/configs/base_config.yaml``.
        device: Torch device string.
    """

    def __init__(
        self,
        config: dict | None = None,
        device: str = "cuda",
    ):
        from depth_anything_3.api import DepthAnything3

        self.config = config or load_da3_config()
        self.device = torch.device(device)
        self.dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )
        self.ref_view_strategy = self.config["Model"]["ref_view_strategy"]

        config_path = self.config["Weights"]["DA3_CONFIG"]
        weights_path = self.config["Weights"]["DA3"]

        with open(config_path) as f:
            model_config = json.load(f)
        self.model = DepthAnything3(**model_config)
        # DualDPT shares auxiliary LayerNorm parameters across pyramid levels.
        # safetensors omits their duplicate names; load_model validates shared
        # storage aliases while still rejecting genuinely missing/extra keys.
        load_model(self.model, weights_path, strict=True, device="cpu")
        self.model.eval()
        self.model.requires_grad_(False)
        self.model = self.model.to(self.device)

        print("[DA3Predictor] initialized.")

    # ------------------------------------------------------------------
    # Low-level API (used by DA3StreamingOnline)
    # ------------------------------------------------------------------

    def infer(
        self,
        frames: list[np.ndarray],
        ref_view_strategy: str | None = None,
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        align_to_input: bool = False,
        skip_pose_align: bool = False,
    ):
        """Run raw DA3 inference on numpy RGB frames.

        Args:
            frames: List of (H, W, 3) uint8 RGB numpy arrays.
            ref_view_strategy: Override default strategy for this call.
            extrinsics: Optional camera hint, (N, 4, 4) **w2c**, DA3 native
                convention. Fed to the model as a conditioning ``cam_token``.
            intrinsics: Optional camera hint, (N, 3, 3) **pixel-space**.
            align_to_input: Forwarded to DA3 as ``align_to_input_ext_scale``.
                Only meaningful when ``extrinsics``/``intrinsics`` are supplied.
                - ``False`` (default): return DA3's *predicted* poses
                  (Umeyama-aligned to the hint frame); depth in DA3's own scale.
                - ``True``: return the **input** extrinsics/intrinsics verbatim
                  and rescale depth to the input-camera metric scale
                  (``depth /= scale``), i.e. the prediction is aligned to the
                  user camera.
            skip_pose_align: When ``True``, run DA3 with the camera hint fed to
                the model as usual, but **skip the Umeyama pose-alignment
                post-processing entirely** (see ``_inference_no_umeyama``). The
                camera hint still conditions the model; the returned
                extrinsics/intrinsics are the **input** cameras verbatim and the
                depth is kept in DA3's native scale (no ``depth /= scale``).
                Used by the scale-alignment path, which recovers the scale from
                chunk-to-chunk overlap instead of Umeyama. ``align_to_input`` is
                ignored when this is set.

        Returns:
            DA3 ``Prediction`` object with fields: ``depth`` (N, H, W),
            ``conf`` (N, H, W), ``intrinsics`` (N, 3, 3),
            ``extrinsics`` (N, 3/4, 4), ``processed_images`` (N, H, W, 3).
            Depth is squeezed and conf is shifted by -1.0.
        """
        strategy = ref_view_strategy or self.ref_view_strategy

        torch.cuda.empty_cache()
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                if skip_pose_align:
                    predictions = self._inference_no_umeyama(
                        frames, extrinsics=extrinsics, intrinsics=intrinsics,
                        ref_view_strategy=strategy,
                    )
                else:
                    predictions = self.model.inference(
                        frames,
                        extrinsics=extrinsics,
                        intrinsics=intrinsics,
                        align_to_input_ext_scale=align_to_input,
                        ref_view_strategy=strategy,
                    )
                predictions.depth = np.squeeze(predictions.depth)
                predictions.conf -= 1.0
        torch.cuda.empty_cache()
        return predictions

    def _inference_no_umeyama(
        self,
        frames: list[np.ndarray],
        extrinsics: np.ndarray | None,
        intrinsics: np.ndarray | None,
        ref_view_strategy: str,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
    ):
        """DA3 ``inference`` **without** the Umeyama pose-alignment step.

        This reproduces the body of ``DepthAnything3.inference`` (calling the
        official model's own preprocessing / forward / post-processing helpers,
        so we do **not** modify the vendored thirdparty code), but replaces
        ``_align_to_input_extrinsics_intrinsics`` — the only place Umeyama runs —
        with a direct assignment of the **input** cameras. Depth is left in DA3's
        native scale; the caller (scale-alignment) recovers the scale from
        chunk-to-chunk overlap.

        The camera hint is still fed to the model forward (as ``cam_token``
        conditioning) exactly as in the official path, so DA3 still uses the user
        camera as a hint — only the unstable Umeyama Sim(3) fit is skipped.
        """
        m = self.model

        # Preprocess images (+ resize cameras to the processing resolution).
        imgs_cpu, ext, intr = m._preprocess_inputs(
            frames, extrinsics, intrinsics, process_res, process_res_method)

        # Prepare model-device tensors and normalize extrinsics for the forward.
        imgs, ex_t, in_t = m._prepare_model_inputs(imgs_cpu, ext, intr)
        ex_t_norm = m._normalize_extrinsics(ex_t.clone() if ex_t is not None else None)

        # Forward pass (camera hint conditions the model here, as usual).
        raw_output = m._run_model_forward(
            imgs, ex_t_norm, in_t, [], False, False, ref_view_strategy)
        prediction = m._convert_to_prediction(raw_output)

        # Replacement for _align_to_input_extrinsics_intrinsics: use the input
        # cameras verbatim, keep DA3-native depth scale, NO Umeyama.
        if ext is not None:
            prediction.extrinsics = ext[..., :3, :].cpu().numpy()
            if intr is not None:
                prediction.intrinsics = intr.cpu().numpy()

        prediction = m._add_processed_images(prediction, imgs_cpu)
        return prediction

    # ------------------------------------------------------------------
    # Standard BaseDepthPredictor interface
    # ------------------------------------------------------------------

    CONFIDENCE_THRESHOLD_COEF = 0.75  # DA3 standard: threshold = mean(conf) * coef

    @torch.inference_mode()
    def predict(
        self,
        images: torch.Tensor,
        hint_extrinsics: torch.Tensor | None = None,
        hint_intrinsics: torch.Tensor | None = None,
        align_to_input: bool = False,
        skip_pose_align: bool = False,
    ) -> DepthPredictorOutput:
        """Predict depth for a set of views.

        Args:
            images: (N, 3, H, W), float32, [0, 1], RGB.
            hint_extrinsics: Optional camera hint in the **repo convention**
                (N, 4, 4) **w2c**. Passed to DA3 as a conditioning cam_token.
            hint_intrinsics: Optional camera hint in the **repo convention**
                (N, 3, 3) **normalized** ([0, 1]). Converted to DA3's
                pixel-space convention internally.
            align_to_input: When ``True`` (and hints are supplied), DA3 returns
                the **input** camera poses/intrinsics verbatim and rescales
                depth to the input-camera metric scale. The returned
                ``extrinsics``/``intrinsics`` then equal ``hint_extrinsics``/
                ``hint_intrinsics``, so the output is aligned to the user camera.
            skip_pose_align: When ``True``, feed the camera hint to the model as
                usual but **skip the Umeyama pose-alignment** post-processing
                (see ``infer`` / ``_inference_no_umeyama``). Returns the input
                cameras verbatim with depth in DA3's native scale. Used by the
                scale-alignment path. ``align_to_input`` is ignored when set.

        Returns:
            DepthPredictorOutput with w2c extrinsics (4x4) and
            normalized intrinsics ([0, 1]).
        """
        # Convert to list of HWC uint8 numpy for DA3
        images_np = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        frames = [images_np[i] for i in range(images_np.shape[0])]

        N = len(frames)
        _, _, H_img, W_img = images.shape

        # Build camera hints in DA3's native convention (w2c 4x4 + pixel-space
        # intrinsics). hint_intrinsics arrive normalized → denormalize to pixels.
        hint_ext_np = None
        hint_int_np = None
        if hint_extrinsics is not None and hint_intrinsics is not None:
            hint_ext_np = hint_extrinsics.detach().cpu().numpy().astype(np.float32)
            k = hint_intrinsics.detach().cpu().numpy().astype(np.float32).copy()
            k[:, 0, 0] *= W_img  # fx
            k[:, 0, 2] *= W_img  # cx
            k[:, 1, 1] *= H_img  # fy
            k[:, 1, 2] *= H_img  # cy
            hint_int_np = k

        predictions = self.infer(
            frames, extrinsics=hint_ext_np, intrinsics=hint_int_np,
            align_to_input=align_to_input, skip_pose_align=skip_pose_align,
        )

        depth = torch.from_numpy(predictions.depth).float()
        conf = torch.from_numpy(predictions.conf).float()

        # infer() squeezes depth/conf, which drops the batch dim when N == 1.
        # Restore the leading view dimension so downstream shapes are (N, H, W).
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)
        if conf.ndim == 2:
            conf = conf.unsqueeze(0)

        # Intrinsics: normalize pixel units → [0, 1]
        intrinsics = torch.from_numpy(predictions.intrinsics).float()
        _, H, W = depth.shape
        intrinsics_norm = intrinsics.clone()
        intrinsics_norm[:, 0, 0] /= W  # fx
        intrinsics_norm[:, 0, 2] /= W  # cx
        intrinsics_norm[:, 1, 1] /= H  # fy
        intrinsics_norm[:, 1, 2] /= H  # cy

        # Extrinsics: ensure (N, 4, 4)
        ext = predictions.extrinsics
        if ext.shape[-2] == 3:
            ext_4x4 = np.zeros((N, 4, 4), dtype=ext.dtype)
            ext_4x4[:, :3, :] = ext
            ext_4x4[:, 3, 3] = 1.0
        else:
            ext_4x4 = ext
        extrinsics = torch.from_numpy(ext_4x4).float()

        # Depth mask: DA3 standard relative thresholding (mean * coef)
        conf_threshold = conf.mean() * self.CONFIDENCE_THRESHOLD_COEF
        depth_mask = (conf > conf_threshold) & (depth > 0) & torch.isfinite(depth)

        return DepthPredictorOutput(
            depth=depth,
            intrinsics=intrinsics_norm,
            extrinsics=extrinsics,
            depth_mask=depth_mask,
            confidence=conf,
        )
