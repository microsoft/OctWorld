"""
Sky segmentation utility for the datapipe.

Wraps the ONNX-based sky segmentation model from
https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing

Used by octmap_integration to mask out sky pixels before depth integration,
in addition to the per-chunk confidence mask.

The ONNX weights are looked up in this order:
  1. $SKYSEG_ONNX_PATH (if set)
  2. <PROJECT_ROOT>/model_weights/skyseg/skyseg.onnx
  3. ./skyseg.onnx (CWD, legacy)

An explicit environment path must be a local file. Missing weights raise
FileNotFoundError; no weights are downloaded.
"""

import copy
import os
import threading

import cv2
import numpy as np
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_WEIGHT_PATH = os.path.join(_PROJECT_ROOT, "model_weights", "skyseg", "skyseg.onnx")

_sessions: dict = {}
_session_lock = threading.Lock()


def _resolve_weight_path() -> str:
    env_path = os.environ.get("SKYSEG_ONNX_PATH")
    if env_path is not None:
        if not os.path.isfile(env_path):
            raise FileNotFoundError(f"SKYSEG_ONNX_PATH is not a local file: {env_path!r}")
        return env_path
    if os.path.isfile(_DEFAULT_WEIGHT_PATH):
        return _DEFAULT_WEIGHT_PATH
    legacy = os.path.abspath("skyseg.onnx")
    if os.path.isfile(legacy):
        return legacy
    raise FileNotFoundError(
        f"Sky segmentation weights not found at {_DEFAULT_WEIGHT_PATH} or {legacy}. "
        "Set SKYSEG_ONNX_PATH to a local ONNX weights file."
    )


def _get_session(device: str = "cpu"):
    """Return a cached InferenceSession for the requested device.

    device: "cpu" or "cuda" (also accepts "cuda:0"-style strings).
    Falls back to CPU if CUDA provider is unavailable.
    """
    dev = (device or "cpu").lower()
    use_cuda = dev.startswith("cuda")
    key = "cuda" if use_cuda else "cpu"
    sess = _sessions.get(key)
    if sess is not None:
        return sess
    with _session_lock:
        sess = _sessions.get(key)
        if sess is not None:
            return sess

        weight_path = _resolve_weight_path()
        import onnxruntime

        if use_cuda:
            available = onnxruntime.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = [
                    ("CUDAExecutionProvider", {"device_id": 0}),
                    "CPUExecutionProvider",
                ]
            else:
                print("[sky_segmentation] CUDAExecutionProvider unavailable; "
                      "falling back to CPU")
                providers = ["CPUExecutionProvider"]
                key = "cpu"
        else:
            providers = ["CPUExecutionProvider"]
        sess = onnxruntime.InferenceSession(weight_path, providers=providers)
        _sessions[key] = sess
    return sess


def _run_skyseg(image_rgb: np.ndarray, input_size=(320, 320),
                device: str = "cpu") -> np.ndarray:
    """Run ONNX sky segmentation on a single RGB image, return a normalized uint8 map.

    After min-max normalization, low scores indicate non-sky regions and high
    scores indicate sky regions. Output is resized back to the input image
    resolution.
    """
    session = _get_session(device)
    H, W = image_rgb.shape[:2]

    resize_image = cv2.resize(copy.deepcopy(image_rgb), dsize=input_size)
    x = np.asarray(resize_image, dtype=np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x / 255.0 - mean) / std
    x = x.transpose(2, 0, 1).reshape(1, 3, input_size[1], input_size[0]).astype("float32")

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    onnx_result = session.run([output_name], {input_name: x})

    onnx_result = np.array(onnx_result).squeeze()
    mn, mx = float(np.min(onnx_result)), float(np.max(onnx_result))
    if mx > mn:
        onnx_result = (onnx_result - mn) / (mx - mn)
    else:
        onnx_result = np.zeros_like(onnx_result)
    onnx_result = (onnx_result * 255).astype("uint8")

    return cv2.resize(onnx_result, (W, H))


def segment_non_sky_mask(image_rgb: np.ndarray, threshold: int = 32,
                         device: str = "cpu") -> np.ndarray:
    """Return a boolean mask where True = non-sky (keep), False = sky (drop).

    Args:
        image_rgb: uint8 RGB image, shape (H, W, 3).
        threshold: pixels whose normalized score is at or above this are treated as sky.
        device: "cpu" or "cuda" for the ONNX runtime backend.

    Returns:
        Boolean mask of shape (H, W). True = keep (non-sky).
    """
    score = _run_skyseg(image_rgb, device=device)
    # Low scores -> non-sky (keep True), high scores -> sky (False)
    return score < threshold


def _run_skyseg_batch(images_rgb: np.ndarray, input_size=(320, 320),
                      device: str = "cpu", batch_size: int = 16) -> np.ndarray:
    """Run ONNX sky segmentation on a batch of RGB images.

    Args:
        images_rgb: uint8 (F, H, W, 3)
        batch_size: number of frames per ONNX run.

    Returns:
        uint8 (F, H, W), per-frame min-max normalized scores at original resolution.
    """
    session = _get_session(device)
    F, H, W = images_rgb.shape[:3]
    in_w, in_h = input_size
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Pre-resize all frames once (cheap on CPU).
    resized = np.empty((F, in_h, in_w, 3), dtype=np.float32)
    for i in range(F):
        resized[i] = cv2.resize(images_rgb[i], dsize=(in_w, in_h))
    x = (resized / 255.0 - mean) / std
    x = x.transpose(0, 3, 1, 2).astype(np.float32)  # (F, 3, in_h, in_w)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Determine the model's expected batch size from its input shape. The
    # public skyseg.onnx weights are exported with a fixed batch dim of 1,
    # so we must run one frame at a time. If the model has a dynamic batch
    # dim (str or None), we honor the requested batch_size.
    model_batch = session.get_inputs()[0].shape[0]
    if isinstance(model_batch, int) and model_batch > 0:
        effective_batch = model_batch
    else:
        effective_batch = max(1, int(batch_size))

    out = np.empty((F, H, W), dtype=np.uint8)
    for start in tqdm(range(0, F, effective_batch), desc="Sky segmentation",
                      unit="batch"):
        end = min(start + effective_batch, F)
        batch_x = x[start:end]
        # Pad the final batch if the model requires a fixed batch size.
        actual = end - start
        if isinstance(model_batch, int) and model_batch > 0 and actual < model_batch:
            pad = np.repeat(batch_x[-1:], model_batch - actual, axis=0)
            batch_x = np.concatenate([batch_x, pad], axis=0)
        res = session.run([output_name], {input_name: batch_x})[0]
        # res shape: (B, ...) — squeeze channel dim if present
        res = np.asarray(res)
        if res.ndim == 4 and res.shape[1] == 1:
            res = res[:, 0]
        res = res[:actual]
        # Per-frame min-max normalize.
        for i in range(res.shape[0]):
            r = res[i].astype(np.float32)
            mn, mx = float(r.min()), float(r.max())
            if mx > mn:
                r = (r - mn) / (mx - mn)
            else:
                r = np.zeros_like(r)
            r = (r * 255).astype(np.uint8)
            out[start + i] = cv2.resize(r, (W, H))
    return out


def segment_non_sky_masks(images_rgb: np.ndarray, threshold: int = 32,
                          device: str = "cpu",
                          batch_size: int = 16) -> np.ndarray:
    """Batched variant of segment_non_sky_mask.

    Args:
        images_rgb: uint8 RGB images, shape (F, H, W, 3).
        device: "cpu" or "cuda" for the ONNX runtime backend.
        batch_size: ONNX batch size (only used on GPU path; CPU path uses 1).

    Returns:
        Boolean mask of shape (F, H, W). True = keep (non-sky).
    """
    use_cuda = (device or "cpu").lower().startswith("cuda")
    if use_cuda:
        scores = _run_skyseg_batch(images_rgb, device=device,
                                   batch_size=batch_size)
        return scores < threshold

    F = images_rgb.shape[0]
    out = np.empty(images_rgb.shape[:3], dtype=bool)
    for i in range(F):
        out[i] = segment_non_sky_mask(images_rgb[i], threshold=threshold,
                                      device=device)
    return out
