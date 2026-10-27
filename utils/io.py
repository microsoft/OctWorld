"""
utils.io — Standardized I/O Protocol

All data I/O in this repo should go through utils.io. Import as:

    from utils.io import (
        read_depth, write_depth,
        read_video_depth, write_video_depth,
        read_image, write_image,
        read_video, write_video,
        read_camera_trajectory, write_camera_trajectory,
        colorize_depth,
        write_colored_depth, write_colored_depth_video,
    )

Data Types & Formats
--------------------
1. Depth Map
   Format: 16-bit PNG with logarithmic encoding
   Shape: (H, W), float32. NaN = unknown, Inf = infinity.
   Metadata: near, far stored in PNG text chunks.

2. Depth Video
   Format: ZIP archive of per-frame 16-bit PNGs
   Shape: (F, H, W), float32.

3. RGB Image
   Format: Standard image formats (PNG, JPG, etc.) via imageio
   Shape: (H, W, 3), uint8.

4. RGB Video
   Format: Standard video formats (MP4, etc.) via imageio/pyav
   Shape: (F, H, W, 3), uint8.

5. Camera Trajectory
   Format: .npz file with keys 'extrinsics' and 'intrinsics'
   extrinsics: (N, 4, 4), float32, world-to-camera matrices.
   intrinsics: (N, 3, 3), float32, normalized intrinsic matrices (fx, fy, cx, cy in [0, 1]).

6. Colored Depth Map / Video
   Depth colorized using 'Spectral' colormap on inverse depth (disparity),
   then saved as standard RGB image/video.
"""

import numpy as np
import matplotlib
from PIL import Image, PngImagePlugin
import imageio.v3 as iio
import io
import zipfile
from typing import Union, IO, Tuple, Callable, Optional
import os
import shutil
import tempfile
from pathlib import Path


def _atomic_write(path: Union[str, os.PathLike], write_fn: Callable[[str], None]) -> None:
    """Write to a local scratch file, then copy it into place.

    Steps:
    1. ``write_fn`` writes the full file to a local temporary path.
    2. ``shutil.copyfile`` from the local scratch path to ``final_path``.
    """
    final_path = os.fspath(path)
    final_dir = os.path.dirname(final_path)
    if final_dir:
        os.makedirs(final_dir, exist_ok=True)
    suffix = os.path.splitext(final_path)[1]

    fd, local_tmp = tempfile.mkstemp(prefix=".write_tmp_", suffix=suffix)
    os.close(fd)

    try:
        write_fn(local_tmp)
        shutil.copyfile(local_tmp, final_path)
    finally:
        if os.path.exists(local_tmp):
            try:
                os.remove(local_tmp)
            except OSError:
                pass


# =============================================================================
# Video / Image Preprocessing
# =============================================================================


def center_crop(data, target_resolution: Tuple[int, int]):
    """
    Center-crop to the target aspect ratio, then resize to exact target resolution.

    Crops the larger dimension to match the target aspect ratio, then
    resizes to the exact target resolution using bilinear interpolation.

    Supports:
    - PIL Image
    - np.ndarray: (H, W), (H, W, C), or (F, H, W, C)
    - torch.Tensor: (H, W), (H, W, C), (F, H, W), or (F, H, W, C)

    ### Parameters:
    - `data`: input image/depth/video in any of the above formats.
    - `target_resolution`: (target_H, target_W).

    ### Returns:
    - Center-cropped and resized data in the same type and format as input.
    """
    target_H, target_W = target_resolution
    target_ratio = target_W / target_H

    if isinstance(data, Image.Image):
        img_w, img_h = data.size
        current_ratio = img_w / img_h
        if current_ratio > target_ratio:
            new_w = int(img_h * target_ratio)
            left = (img_w - new_w) // 2
            data = data.crop((left, 0, left + new_w, img_h))
        elif current_ratio < target_ratio:
            new_h = int(img_w / target_ratio)
            top = (img_h - new_h) // 2
            data = data.crop((0, top, img_w, top + new_h))
        if data.size != (target_W, target_H):
            data = data.resize((target_W, target_H), Image.BILINEAR)
        return data

    if isinstance(data, np.ndarray):
        if data.ndim == 2:
            H, W = data.shape
            current_ratio = W / H
            if current_ratio > target_ratio:
                new_W = int(H * target_ratio)
                left = (W - new_W) // 2
                data = data[:, left:left + new_W]
            elif current_ratio < target_ratio:
                new_H = int(W / target_ratio)
                top = (H - new_H) // 2
                data = data[top:top + new_H, :]
            if data.shape != (target_H, target_W):
                pil_img = Image.fromarray(data)
                pil_img = pil_img.resize((target_W, target_H), Image.BILINEAR)
                data = np.array(pil_img)
            return data
        elif data.ndim == 3:
            H, W = data.shape[0], data.shape[1]
            current_ratio = W / H
            if current_ratio > target_ratio:
                new_W = int(H * target_ratio)
                left = (W - new_W) // 2
                data = data[:, left:left + new_W, :]
            elif current_ratio < target_ratio:
                new_H = int(W / target_ratio)
                top = (H - new_H) // 2
                data = data[top:top + new_H, :, :]
            if data.shape[0:2] != (target_H, target_W):
                pil_img = Image.fromarray(data)
                pil_img = pil_img.resize((target_W, target_H), Image.BILINEAR)
                data = np.array(pil_img)
            return data
        elif data.ndim == 4:
            F, H, W, C = data.shape
            current_ratio = W / H
            if current_ratio > target_ratio:
                new_W = int(H * target_ratio)
                left = (W - new_W) // 2
                data = data[:, :, left:left + new_W, :]
            elif current_ratio < target_ratio:
                new_H = int(W / target_ratio)
                top = (H - new_H) // 2
                data = data[:, top:top + new_H, :, :]
            if data.shape[1:3] != (target_H, target_W):
                resized = []
                for i in range(data.shape[0]):
                    frame = Image.fromarray(data[i])
                    frame = frame.resize((target_W, target_H), Image.BILINEAR)
                    resized.append(np.array(frame))
                data = np.stack(resized, axis=0)
            return data

    import torch
    if isinstance(data, torch.Tensor):
        if data.ndim == 2:
            H, W = data.shape
            current_ratio = W / H
            if current_ratio > target_ratio:
                new_W = int(H * target_ratio)
                left = (W - new_W) // 2
                data = data[:, left:left + new_W]
            elif current_ratio < target_ratio:
                new_H = int(W / target_ratio)
                top = (H - new_H) // 2
                data = data[top:top + new_H, :]
            if data.shape != (target_H, target_W):
                data = torch.nn.functional.interpolate(
                    data[None, None].float(), size=(target_H, target_W), mode='bilinear', align_corners=False
                )[0, 0].to(data.dtype)
            return data
        elif data.ndim == 3:
            H, W = data.shape[0], data.shape[1]
            current_ratio = W / H
            if current_ratio > target_ratio:
                new_W = int(H * target_ratio)
                left = (W - new_W) // 2
                data = data[:, left:left + new_W]
            elif current_ratio < target_ratio:
                new_H = int(W / target_ratio)
                top = (H - new_H) // 2
                data = data[top:top + new_H, :]
            if data.shape[0:2] != (target_H, target_W):
                # (H, W, C) -> (1, C, H, W) for interpolate
                if data.ndim == 3:
                    data = data.permute(2, 0, 1)[None]
                    data = torch.nn.functional.interpolate(
                        data.float(), size=(target_H, target_W), mode='bilinear', align_corners=False
                    ).to(data.dtype)
                    data = data[0].permute(1, 2, 0)
            return data
        elif data.ndim == 4:
            F, H, W = data.shape[0], data.shape[1], data.shape[2]
            current_ratio = W / H
            if current_ratio > target_ratio:
                new_W = int(H * target_ratio)
                left = (W - new_W) // 2
                data = data[:, :, left:left + new_W]
            elif current_ratio < target_ratio:
                new_H = int(W / target_ratio)
                top = (H - new_H) // 2
                data = data[:, top:top + new_H, :]
            if data.shape[1:3] != (target_H, target_W):
                # (F, H, W, C) -> treat as batch
                C = data.shape[3] if data.shape[-1] < data.shape[1] else 1
                if data.ndim == 4 and data.shape[3] > 0:
                    data = data.permute(0, 3, 1, 2)  # (F, C, H, W)
                    data = torch.nn.functional.interpolate(
                        data.float(), size=(target_H, target_W), mode='bilinear', align_corners=False
                    ).to(data.dtype)
                    data = data.permute(0, 2, 3, 1)  # (F, H, W, C)
            return data

    raise ValueError(f"Unsupported data type/shape: {type(data)}, shape={getattr(data, 'shape', None)}")


def center_crop_video(
    video: np.ndarray,
    target_resolution: Tuple[int, int],
) -> np.ndarray:
    """
    Center-crop a video to the target aspect ratio, then resize.

    Crops the larger dimension to match the target aspect ratio, then
    resizes to the exact target resolution using bilinear interpolation.

    ### Parameters:
    - `video`: uint8 array of shape (F, H, W, 3).
    - `target_resolution`: (target_H, target_W).

    ### Returns:
    - uint8 array of shape (F, target_H, target_W, 3).
    """
    F, H, W, C = video.shape
    target_H, target_W = target_resolution
    target_ratio = target_W / target_H
    current_ratio = W / H

    if current_ratio > target_ratio:
        # Image is too wide → crop width
        new_W = int(H * target_ratio)
        left = (W - new_W) // 2
        video = video[:, :, left:left + new_W, :]
    elif current_ratio < target_ratio:
        # Image is too tall → crop height
        new_H = int(W / target_ratio)
        top = (H - new_H) // 2
        video = video[:, top:top + new_H, :, :]

    # Resize to exact target resolution
    if video.shape[1:3] != (target_H, target_W):
        from PIL import Image as PILImage
        resized = []
        for i in range(F):
            frame = PILImage.fromarray(video[i])
            frame = frame.resize((target_W, target_H), PILImage.BILINEAR)
            resized.append(np.array(frame))
        video = np.stack(resized, axis=0)

    return video


# =============================================================================
# Depth Map I/O (16-bit PNG with logarithmic encoding)
# =============================================================================


def read_depth(path: Union[str, os.PathLike, IO]) -> np.ndarray:
    """
    Read a depth image, return float32 depth array of shape (H, W).
    """
    if isinstance(path, (str, os.PathLike)):
        data = Path(path).read_bytes()
    else:
        data = path.read()
    pil_image = Image.open(io.BytesIO(data))
    raw = np.array(pil_image)
    mask_nan, mask_inf = raw == 0, raw == 65535
    # Special case: an "empty" depth map has no finite values (entirely
    # masked-out). It is stored with the `empty` metadata flag and contains
    # only 0 (NaN) and/or 65535 (inf) entries.
    if pil_image.info.get('empty') == '1':
        depth = np.full(raw.shape, np.nan, dtype=np.float32)
        depth[mask_inf] = np.inf
        return depth
    near = float(pil_image.info.get('near'))
    far = float(pil_image.info.get('far'))
    depth = (raw.astype(np.float32) - 1) / 65533
    depth = near ** (1 - depth) * far ** depth
    depth[mask_nan] = np.nan
    depth[mask_inf] = np.inf
    if 'unit' in pil_image.info:
        unit = float(pil_image.info.get('unit')) 
        depth *= unit
    return depth


def write_depth(
    path: Union[str, os.PathLike, IO[bytes]], 
    depth: np.ndarray, 
    mask: np.ndarray = None,
    max_range: float = 1e5,
    compression_level: int = 7,
):
    """
    Encode and write a depth image as 16-bit PNG format.
    ### Parameters:
    - `path: Union[str, os.PathLike, IO]`
        The file path or file object to write to.
    - `depth: np.ndarray`
        The depth array, float32 array of shape (H, W).
    - `mask: np.ndarray = None`
        Optional bool array of shape (H, W). Where False, depth is treated as invalid (stored as NaN/unknown).
    - `max_range: float = 1e5`
        Maximum depth range ratio (far/near).
    - `compression_level: int = 7`
        PNG compression level.
    
    Depth values are encoded as follows:
    - 0: unknown
    - 1 ~ 65534: depth values in logarithmic
    - 65535: infinity
    
    metadata is stored in the PNG file as text fields:
    - `near`: the minimum depth value
    - `far`: the maximum depth value
    - `unit`: the unit of the depth values (optional)
    """
    depth = depth.copy()
    if mask is not None:
        depth[~mask] = np.nan
    mask_values, mask_nan, mask_inf = np.isfinite(depth), np.isnan(depth),np.isinf(depth)

    depth = depth.astype(np.float32)

    # Special case: no finite values at all (e.g. a fully masked-out frame
    # of a video depth sequence). Skip the log encoding and emit an "empty"
    # depth map containing only 0 (NaN) and/or 65535 (inf) entries.
    if not mask_values.any():
        encoded = np.zeros(depth.shape, dtype=np.uint16)
        encoded[mask_inf] = 65535
        pil_image = Image.fromarray(encoded)
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text('empty', '1')
        pil_image.save(path, format="PNG", pnginfo=pnginfo, compress_level=compression_level)
        return

    mask_finite = depth
    near = max(depth[mask_values].min(), 1e-5)
    far = max(near * 1.1, min(depth[mask_values].max(), near * max_range))
    depth = 1 + np.round((np.log(np.nan_to_num(depth, nan=0).clip(near, far) / near) / np.log(far / near)).clip(0, 1) * 65533).astype(np.uint16) # 1~65534
    depth[mask_nan] = 0
    depth[mask_inf] = 65535

    pil_image = Image.fromarray(depth)
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text('near', str(near))
    pnginfo.add_text('far', str(far))
    pil_image.save(path, format="PNG", pnginfo=pnginfo, compress_level=compression_level)


def write_video_depth(
    path: Union[str, os.PathLike],
    depth: np.ndarray,
    mask: np.ndarray = None,
    max_range: float = 1e5,
    compression_level: int = 7,
):
    """
    Write a video depth sequence as a zip of per-frame 16-bit PNGs.

    ### Parameters:
    - `path`: Output .zip file path.
    - `depth`: float32 array of shape (F, H, W).
    - `mask`: Optional bool array of shape (F, H, W). Where False, depth is stored as invalid.
    """
    assert str(path).endswith('.zip'), "path must end with .zip"
    assert depth.ndim == 3, "depth must be (F, H, W)"

    def _write(tmp_path: str) -> None:
        with zipfile.ZipFile(tmp_path, 'w', compression=zipfile.ZIP_STORED) as zf:
            for i in range(depth.shape[0]):
                buf = io.BytesIO()
                frame_mask = mask[i] if mask is not None else None
                write_depth(buf, depth[i], mask=frame_mask, max_range=max_range, compression_level=compression_level)
                zf.writestr(f'{i:06d}.png', buf.getvalue())

    _atomic_write(path, _write)


def read_video_depth(path: Union[str, os.PathLike]) -> np.ndarray:
    """
    Read a video depth sequence from a zip of per-frame 16-bit PNGs.

    ### Parameters:
    - `path`: Input .zip file path.

    ### Returns:
    - float32 array of shape (F, H, W).
    """
    assert str(path).endswith('.zip'), "path must end with .zip"

    frames = []
    with zipfile.ZipFile(path, 'r') as zf:
        names = sorted(n for n in zf.namelist() if n.endswith('.png'))
        for name in names:
            buf = io.BytesIO(zf.read(name))
            frames.append(read_depth(buf))
    return np.stack(frames, axis=0)


# =============================================================================
# Confidence Map I/O (16-bit PNG with linear encoding)
# =============================================================================


def read_conf(path: Union[str, os.PathLike, IO]) -> np.ndarray:
    """
    Read a confidence image, return float32 array of shape (H, W).

    Inverse of `write_conf`: linearly decodes uint16 [0, 65535] back to
    [conf_min, conf_max] using metadata stored in the PNG.
    """
    if isinstance(path, (str, os.PathLike)):
        data = Path(path).read_bytes()
    else:
        data = path.read()
    pil_image = Image.open(io.BytesIO(data))
    raw = np.array(pil_image).astype(np.float32)
    conf_min = float(pil_image.info.get('conf_min'))
    conf_max = float(pil_image.info.get('conf_max'))
    if conf_max == conf_min:
        return np.full(raw.shape, conf_min, dtype=np.float32)
    conf = raw / 65535.0 * (conf_max - conf_min) + conf_min
    return conf


def write_conf(
    path: Union[str, os.PathLike, IO[bytes]],
    conf: np.ndarray,
    compression_level: int = 7,
):
    """
    Encode and write a confidence map as 16-bit PNG.

    Confidence values are linearly mapped from [conf_min, conf_max] to
    uint16 [0, 65535]. The min and max are stored as PNG text metadata.

    ### Parameters:
    - `path`: Output file path or file-like object.
    - `conf`: float32 array of shape (H, W).
    - `compression_level`: PNG compression level.
    """
    conf = conf.astype(np.float32)
    finite = np.isfinite(conf)
    if not finite.any():
        conf_min, conf_max = 0.0, 0.0
        encoded = np.zeros(conf.shape, dtype=np.uint16)
    else:
        conf_min = float(conf[finite].min())
        conf_max = float(conf[finite].max())
        if conf_max == conf_min:
            encoded = np.zeros(conf.shape, dtype=np.uint16)
        else:
            normalized = (np.nan_to_num(conf, nan=conf_min) - conf_min) / (conf_max - conf_min)
            normalized = np.clip(normalized, 0.0, 1.0)
            encoded = np.round(normalized * 65535).astype(np.uint16)

    pil_image = Image.fromarray(encoded)
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text('conf_min', str(conf_min))
    pnginfo.add_text('conf_max', str(conf_max))
    pil_image.save(path, format="PNG", pnginfo=pnginfo, compress_level=compression_level)


def write_video_conf(
    path: Union[str, os.PathLike],
    conf: np.ndarray,
    compression_level: int = 7,
):
    """
    Write a video confidence sequence as a zip of per-frame 16-bit PNGs.

    ### Parameters:
    - `path`: Output .zip file path.
    - `conf`: float32 array of shape (F, H, W).
    """
    assert str(path).endswith('.zip'), "path must end with .zip"
    assert conf.ndim == 3, "conf must be (F, H, W)"

    def _write(tmp_path: str) -> None:
        with zipfile.ZipFile(tmp_path, 'w', compression=zipfile.ZIP_STORED) as zf:
            for i in range(conf.shape[0]):
                buf = io.BytesIO()
                write_conf(buf, conf[i], compression_level=compression_level)
                zf.writestr(f'{i:06d}.png', buf.getvalue())

    _atomic_write(path, _write)


def read_video_conf(path: Union[str, os.PathLike]) -> np.ndarray:
    """
    Read a video confidence sequence from a zip of per-frame 16-bit PNGs.

    ### Returns:
    - float32 array of shape (F, H, W).
    """
    assert str(path).endswith('.zip'), "path must end with .zip"

    frames = []
    with zipfile.ZipFile(path, 'r') as zf:
        names = sorted(n for n in zf.namelist() if n.endswith('.png'))
        for name in names:
            buf = io.BytesIO(zf.read(name))
            frames.append(read_conf(buf))
    return np.stack(frames, axis=0)


# =============================================================================
# RGB Image I/O
# =============================================================================


def read_image(path: Union[str, os.PathLike]) -> np.ndarray:
    """
    Read an RGB image.

    ### Returns:
    - uint8 array of shape (H, W, 3).
    """
    img = iio.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 4:
        img = img[..., :3]
    return img


def write_image(path: Union[str, os.PathLike], image: np.ndarray) -> None:
    """
    Write an RGB image.

    ### Parameters:
    - `path`: Output file path (e.g. .png, .jpg).
    - `image`: uint8 array of shape (H, W, 3).
    """
    assert image.ndim == 3 and image.shape[2] == 3, "image must be (H, W, 3)"
    assert image.dtype == np.uint8, "image must be uint8"
    iio.imwrite(path, image)


# =============================================================================
# RGB Video I/O
# =============================================================================


def read_video(
    path: Union[str, os.PathLike],
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> np.ndarray:
    """
    Read an RGB video.

    ### Parameters:
    - `path`: Input video file path.
    - `start`: Optional start frame index (inclusive). Default reads from the
      beginning.
    - `end`: Optional end frame index (exclusive). Default reads to the end.

    ### Returns:
    - uint8 array of shape (F, H, W, 3).
    """
    if start is None and end is None:
        return iio.imread(path, plugin="pyav")

    s = 0 if start is None else int(start)
    e = None if end is None else int(end)
    assert s >= 0, f"start must be >= 0, got {s}"
    assert e is None or e >= s, f"end ({e}) must be >= start ({s})"

    frames = []
    for i, frame in enumerate(iio.imiter(path, plugin="pyav")):
        if i < s:
            continue
        if e is not None and i >= e:
            break
        frames.append(frame)
    if len(frames) == 0:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)
    return np.stack(frames, axis=0)


def write_video(
    path: Union[str, os.PathLike],
    video: np.ndarray,
    fps: int = 30,
) -> None:
    """
    Write an RGB video.

    ### Parameters:
    - `path`: Output file path (e.g. .mp4).
    - `video`: uint8 array of shape (F, H, W, 3).
    - `fps`: Frames per second. Default: 30.
    """
    assert video.ndim == 4 and video.shape[3] == 3, "video must be (F, H, W, 3)"
    assert video.dtype == np.uint8, "video must be uint8"
    iio.imwrite(path, video, fps=fps, codec="h264", plugin="pyav")


# =============================================================================
# Camera Trajectory I/O
# =============================================================================


def read_camera_trajectory(path: Union[str, os.PathLike]) -> tuple[np.ndarray, np.ndarray]:
    """
    Read a camera trajectory from .npz file.

    ### Returns:
    - `extrinsics`: float32 array of shape (N, 4, 4), camera-to-world matrices.
    - `intrinsics`: float32 array of shape (N, 3, 3), intrinsic matrices.
    """
    data = np.load(path)
    extrinsics = data['extrinsics'].astype(np.float32)
    intrinsics = data['intrinsics'].astype(np.float32)
    assert extrinsics.ndim == 3 and extrinsics.shape[1:] == (4, 4), \
        f"extrinsics must be (N, 4, 4), got {extrinsics.shape}"
    assert intrinsics.ndim == 3 and intrinsics.shape[1:] == (3, 3), \
        f"intrinsics must be (N, 3, 3), got {intrinsics.shape}"
    return extrinsics, intrinsics


def write_camera_trajectory(
    path: Union[str, os.PathLike],
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
) -> None:
    """
    Write a camera trajectory to .npz file.

    ### Parameters:
    - `path`: Output .npz file path.
    - `extrinsics`: float32 array of shape (N, 4, 4), camera-to-world matrices.
    - `intrinsics`: float32 array of shape (N, 3, 3), intrinsic matrices.
    """
    extrinsics = np.asarray(extrinsics, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    assert extrinsics.ndim == 3 and extrinsics.shape[1:] == (4, 4), \
        f"extrinsics must be (N, 4, 4), got {extrinsics.shape}"
    assert intrinsics.ndim == 3 and intrinsics.shape[1:] == (3, 3), \
        f"intrinsics must be (N, 3, 3), got {intrinsics.shape}"
    _atomic_write(path, lambda tmp: np.savez(tmp, extrinsics=extrinsics, intrinsics=intrinsics))


# =============================================================================
# Depth Colorization
# =============================================================================


def colorize_depth(
    depth: np.ndarray,
    mask: np.ndarray = None,
    normalize: bool = True,
    cmap: str = 'Spectral',
) -> np.ndarray:
    """
    Convert a depth map to a colorized RGB image.

    Uses global normalization across all frames (for video input) to ensure
    consistent color scale without per-frame jumps.

    ### Parameters:
    - `depth`: float32 array of shape (H, W) or (F, H, W).
    - `mask`: optional bool array, same shape as depth. Only colorize where mask is True.
    - `normalize`: whether to normalize disparity to [0, 1] using quantiles.
    - `cmap`: matplotlib colormap name.

    ### Returns:
    - uint8 array of shape (H, W, 3) or (F, H, W, 3).
    """
    if mask is None:
        depth = np.where(depth > 0, depth, np.nan)
    else:
        depth = np.where((depth > 0) & mask, depth, np.nan)
    disp = 1 / depth
    if normalize:
        min_disp, max_disp = np.nanquantile(disp, 0.001), np.nanquantile(disp, 0.99)
        disp = (disp - min_disp) / (max_disp - min_disp)
    colored = np.nan_to_num(matplotlib.colormaps[cmap](1.0 - disp)[..., :3], 0)
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    return colored


# =============================================================================
# Colored Depth Image/Video I/O (convenience wrappers)
# =============================================================================


def write_colored_depth(
    path: Union[str, os.PathLike],
    depth: np.ndarray,
    mask: np.ndarray = None,
    normalize: bool = True,
    cmap: str = 'Spectral',
) -> None:
    """
    Colorize a depth map and write as RGB image.

    ### Parameters:
    - `path`: Output image file path (e.g. .png, .jpg).
    - `depth`: float32 array of shape (H, W).
    """
    colored = colorize_depth(depth, mask=mask, normalize=normalize, cmap=cmap)
    write_image(path, colored)


def write_colored_depth_video(
    path: Union[str, os.PathLike],
    depth: np.ndarray,
    mask: np.ndarray = None,
    normalize: bool = True,
    cmap: str = 'Spectral',
    fps: int = 30,
) -> None:
    """
    Colorize a depth video and write as RGB video.

    ### Parameters:
    - `path`: Output video file path (e.g. .mp4).
    - `depth`: float32 array of shape (F, H, W).
    - `fps`: Frames per second. Default: 30.
    """
    colored = colorize_depth(depth, mask=mask, normalize=normalize, cmap=cmap)
    write_video(path, colored, fps=fps)


# =============================================================================
# Video Post-Processing (merged from utils/post.py)
# =============================================================================


def post_process_video_for_visualize(
    video: Union["torch.Tensor", np.ndarray],
    layout: "Optional[Literal['fchw', 'fhw3']]" = None,
    value_range: "Optional[Tuple[float, float]]" = None,
    verbose: bool = False,
    return_type: "Literal['numpy', 'pil']" = 'numpy',
):
    """
    Convert an RGB video tensor/array to uint8 (F, H, W, 3) for visualization.

    Only supports color (3-channel) video. For depth visualization, use
    ``colorize_depth`` instead.

    ### Parameters:
    - `video`: torch.Tensor or numpy.ndarray.
    - `layout`: 'fchw' or 'fhw3'. None = auto-infer.
    - `value_range`: (lo, hi) e.g. (0,1), (-1,1), (0,255). None = auto-infer.
    - `verbose`: Print debug info.
    - `return_type`: 'numpy' returns (F,H,W,3) uint8; 'pil' returns list[PIL.Image].

    ### Returns:
    - uint8 array (F, H, W, 3) or list[PIL.Image].
    """
    import torch as _torch

    input_type = type(video).__name__
    if isinstance(video, _torch.Tensor):
        video = video.detach().cpu().float().numpy()
    elif isinstance(video, np.ndarray):
        video = video.astype(np.float64) if video.dtype == np.float16 else video.copy()
    else:
        raise TypeError(f"Input must be torch.Tensor or numpy.ndarray, got {input_type}")

    if verbose:
        print(f"[visualize] input type={input_type}, dtype={video.dtype}, shape={video.shape}, "
              f"min={video.min():.4f}, max={video.max():.4f}")

    # Layout → (F, H, W, 3)
    ndim = video.ndim
    if layout is not None:
        if layout == 'fchw':
            assert ndim == 4 and video.shape[1] == 3, \
                f"layout='fchw' requires shape (F,3,H,W), got {video.shape}"
            video = np.transpose(video, (0, 2, 3, 1))
        elif layout == 'fhw3':
            assert ndim == 4 and video.shape[-1] == 3, \
                f"layout='fhw3' requires shape (F,H,W,3), got {video.shape}"
        else:
            raise ValueError(f"Unsupported layout: {layout}, must be 'fchw' or 'fhw3'")
    else:
        if ndim == 4:
            if video.shape[-1] == 3:
                pass
            elif video.shape[1] == 3:
                video = np.transpose(video, (0, 2, 3, 1))
            elif video.shape[0] == 3:
                video = np.transpose(video, (1, 2, 3, 0))
            else:
                raise ValueError(f"Cannot auto-infer layout, shape={video.shape}")
        else:
            raise ValueError(f"Input must be 4D with 3 channels, got shape={video.shape}")

    assert video.shape[-1] == 3, f"Expected 3 channels, got {video.shape[-1]}"

    # Infer value_range
    vmin_data, vmax_data = float(video.min()), float(video.max())
    if value_range is None:
        if vmin_data >= 0 and vmax_data <= 1.0 + 1e-3:
            value_range = (0, 1)
        elif vmin_data >= -1.0 - 1e-3 and vmax_data <= 1.0 + 1e-3:
            value_range = (-1, 1)
        elif vmin_data >= 0 and vmax_data <= 255.0 + 1e-3:
            value_range = (0, 255)
        else:
            raise ValueError(
                f"Cannot auto-infer value_range (min={vmin_data:.4f}, max={vmax_data:.4f})")

    if verbose:
        print(f"[visualize] after layout: shape={video.shape}, value_range={value_range}")

    lo, hi = value_range
    video = (video - lo) / (hi - lo + 1e-8)
    rgb = np.clip(video * 255.0, 0, 255).astype(np.uint8)

    if verbose:
        print(f"[visualize] output: shape={rgb.shape}, dtype={rgb.dtype}, "
              f"min={rgb.min()}, max={rgb.max()}")

    if return_type == 'numpy':
        return rgb
    return [Image.fromarray(rgb[i]) for i in range(rgb.shape[0])]


def post_process_auto_to_rgb(video, value_range=None):
    """
    Convert any torch.Tensor or numpy.ndarray video to (F, H, W, 3) uint8 RGB.

    Supports shapes: (F,H,W,3), (F,3,H,W), (3,F,H,W).
    For single-channel/depth, use ``colorize_depth`` instead.

    ### Parameters:
    - `video`: torch.Tensor or numpy.ndarray.
    - `value_range`: (lo, hi). None = auto-infer from (0,1), (-1,1), (0,255).

    ### Returns:
    - uint8 array of shape (F, H, W, 3).
    """
    import torch as _torch

    if isinstance(video, _torch.Tensor):
        video = video.detach().cpu().numpy()
    elif not isinstance(video, np.ndarray):
        raise TypeError("Input must be torch.Tensor or numpy.ndarray")

    if value_range is None:
        if video.min() >= 0 and video.max() <= 1:
            value_range = (0, 1)
        elif video.min() >= -1 and video.max() <= 1:
            value_range = (-1, 1)
        elif video.min() >= 0 and video.max() <= 255:
            value_range = (0, 255)
        else:
            raise ValueError("Cannot auto-infer value_range, please specify manually")

    if video.ndim == 4:
        if video.shape[-1] == 3:
            pass
        elif video.shape[1] == 3:
            video = np.transpose(video, (0, 2, 3, 1))
        elif video.shape[0] == 3:
            video = np.transpose(video, (1, 2, 3, 0))
        else:
            raise ValueError(f"Unrecognized video shape: {video.shape}")
    else:
        raise ValueError(f"Input must be 4D, got shape={video.shape}")

    video = (video - value_range[0]) / (value_range[1] - value_range[0])
    video = np.clip(video * 255.0, 0, 255).astype(np.uint8)
    return video


def post_process_frames(frames, depth_mode=False, value_range=(0, 1)):
    """
    Post-process frames for visualization.
    Supports shapes: fhw3, fhw, fhw1, (B,C,F,H,W), (F,C,H,W).

    For depth_mode, uses ``colorize_depth`` for colorization.
    """
    import torch as _torch

    if isinstance(frames, _torch.Tensor):
        frames = frames.float().cpu().numpy()
    else:
        assert isinstance(frames, np.ndarray), "frames should be a numpy array or torch tensor"
    if frames.ndim == 5:  # (B, C, F, H, W)
        assert frames.shape[0] == 1
        frames = frames.squeeze(0).transpose(1, 2, 3, 0)
    elif frames.ndim == 4:  # (F, H, W, C) / (F, C, H, W)
        if frames.shape[1] == 3:
            frames = frames.transpose(0, 2, 3, 1)
    frames = (frames - value_range[0]) / (value_range[1] - value_range[0])
    frames = frames.clip(0, 1)
    if depth_mode:
        if frames.ndim == 4:
            frames = frames.mean(axis=-1, keepdims=False)
        return colorize_depth(frames)
    else:
        frames = (frames * 255).astype(np.uint8)
    return frames


def post_process_warp_result(gt_images, warp_rgb, warp_depth, warp_mask, gt_depth=None, white_bg=True):
    """Post-process warp results into a side-by-side visualization video."""
    import torch as _torch

    masks = warp_mask > 0.5
    depth_rgb = colorize_depth(
        warp_depth.clone().squeeze(1).cpu().numpy(),
        mask=masks.squeeze(1).cpu().numpy(),
    )
    if gt_depth is not None:
        gt_depth_rgb = colorize_depth(gt_depth)
        depth_rgb = np.concatenate([gt_depth_rgb, depth_rgb], axis=2)
    colored_depth = depth_rgb
    warp_rgb = warp_rgb[:, :3].permute(0, 2, 3, 1).clamp(0, 1)

    if white_bg:
        masks_bhw = masks.squeeze(1)
        warp_rgb[~masks_bhw] = 1.0
    reproject_color = (warp_rgb * 255).cpu().numpy().astype(np.uint8)
    if gt_images is None:
        result_video = np.concatenate([reproject_color, colored_depth], axis=2)
        return result_video
    if isinstance(gt_images, np.ndarray):
        gt_images = _torch.from_numpy(gt_images).to(warp_rgb.device) / 255.0
    elif gt_images.min() < 0:
        gt_images = (gt_images + 1.0) / 2.0
        gt_images = gt_images.permute(0, 2, 3, 1)
    gt_color = (gt_images * 255).cpu().numpy().astype(np.uint8)

    result_video = np.concatenate([gt_color, reproject_color, colored_depth], axis=2)
    return result_video


def post_process_video_snapshot(
    mp4_files,
    max_frames_per_video=10,
    resize_shape=(160, 90),
    cols=10,
    fps=10,
    clip_horizontal_range=None,
    clip_vertical_range=None,
    depth_mode=False,
):
    """
    Combine multiple videos into a mosaic grid video.

    Args:
        mp4_files: list of file paths (str) or numpy arrays (F,H,W,3).
        max_frames_per_video: frames to extract per video.
        resize_shape: (W, H) for each cell.
        cols: number of columns in the grid.
        clip_horizontal_range: (left_ratio, right_ratio) for horizontal crop.
        clip_vertical_range: (top_ratio, bottom_ratio) for vertical crop.
        depth_mode: if True, load .npz depth files and colorize.

    Returns:
        list of numpy arrays (canvas frames), or None if empty.
    """
    import cv2
    import math

    videos_frames = []

    for video_file in mp4_files:
        if isinstance(video_file, np.ndarray):
            frame_befor_process = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in video_file]
        else:
            if depth_mode:
                depth = np.load(video_file)["depths"]
                depth = depth.squeeze(1)
                depth_mask = np.isfinite(depth)
                depth = depth / depth[depth_mask].max() * 0.9
                depth[~depth_mask] = 1.0
                frame_befor_process = colorize_depth(depth)
            else:
                cap = cv2.VideoCapture(video_file)
                count = 0
                success = True
                frame_befor_process = []
                while success and count < max_frames_per_video:
                    success, frame = cap.read()
                    if not success:
                        break
                    frame_befor_process.append(frame)
                    count += 1
                cap.release()

        frames_this_video = []
        for frame in frame_befor_process:
            if isinstance(frame, np.ndarray) and frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            h, w, _ = frame.shape
            if clip_horizontal_range:
                left = math.floor(w * clip_horizontal_range[0])
                right = math.ceil(w * clip_horizontal_range[1])
                frame = frame[:, left:right, :]
            if clip_vertical_range:
                top = math.floor(h * clip_vertical_range[0])
                bottom = math.ceil(h * clip_vertical_range[1])
                frame = frame[top:bottom, :, :]
            resized = cv2.resize(frame, resize_shape)
            if not depth_mode:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            frames_this_video.append(resized)

        while len(frames_this_video) < max_frames_per_video:
            black = np.zeros((resize_shape[1], resize_shape[0], 3), dtype=np.uint8)
            frames_this_video.append(black)

        videos_frames.append(frames_this_video)

    if not videos_frames:
        print("No frames extracted!")
        return None

    video_count = len(videos_frames)
    rows = int(np.ceil(video_count / cols))
    total_frames = max_frames_per_video
    frame_h, frame_w = resize_shape[1], resize_shape[0]

    output_frames = []
    for i in range(total_frames):
        canvas = np.ones((rows * frame_h, cols * frame_w, 3), dtype=np.uint8) * 255
        for vid_idx, frames in enumerate(videos_frames):
            r = vid_idx // cols
            c = vid_idx % cols
            y1, y2 = r * frame_h, (r + 1) * frame_h
            x1, x2 = c * frame_w, (c + 1) * frame_w
            canvas[y1:y2, x1:x2] = frames[i]
        output_frames.append(canvas)

    return output_frames
