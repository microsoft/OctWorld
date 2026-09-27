"""Frame-aligned Wan exports, separate from keyframe-only AR/map state."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import stat

import numpy as np
import torch

from utils.io import colorize_depth, write_video


class WanWorldOutputMixin:
    @contextmanager
    def _isolated_output_prediction(self):
        """Protect global RNG and transient masks, including on prediction errors.

        The raw no-Umeyama helper does not touch overlap caches. Its mask helper
        does assign diagnostic masks; restore these (and any _last_* mask state)
        rather than allowing an export batch to replace the AR chunk's state.
        Neither dedicated sampling nor augmentation generator is passed here.
        """
        np_state, py_state = np.random.get_state(), random.getstate()
        prefixes = ("_dbg_", "_last_")
        mask_state = {k: v for k, v in vars(self).items() if k.startswith(prefixes)}
        previous_namespace = getattr(self, "_random_namespace", None)
        self._random_namespace = "export"
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        try:
            with torch.random.fork_rng(devices=devices):
                yield
        finally:
            if previous_namespace is None:
                del self._random_namespace
            else:
                self._random_namespace = previous_namespace
            np.random.set_state(np_state)
            random.setstate(py_state)
            for name in list(vars(self)):
                if name.startswith(prefixes) and name not in mask_state:
                    delattr(self, name)
            for name, value in mask_state.items():
                setattr(self, name, value)
            if devices:
                torch.cuda.empty_cache()

    @torch.inference_mode()
    def _dense_output_depth(
        self, frames, extrinsics, intrinsics, sample_idx, depths, masks, batch_size=8,
    ):
        """Keep keyframes verbatim; predict every missing frame with a local anchor.

        Each missing frame uses its nearest selected frame (ties go to the first
        selected frame). Up to eight missing frames sharing that anchor are
        predicted jointly with it, with camera hints and skip_pose_align=True.
        Divide by the overlap scale fitted against the ORIGINAL chunk depth/mask,
        not an earlier export batch, so scale cannot drift through export batches.
        Full output storage and per-batch results live on CPU; no depth resampling
        is applied to selected frames and no placeholder frames are emitted.
        """
        frames = frames.detach().cpu()
        extrinsics, intrinsics = extrinsics.detach().cpu(), intrinsics.detach().cpu()
        sample_idx = sample_idx.detach().cpu().long()
        depths, masks = depths.detach().cpu(), masks.detach().cpu()
        n, h, w = frames.shape[:3]
        if (batch_size < 1 or sample_idx.ndim != 1 or sample_idx.numel() == 0
                or sample_idx.unique().numel() != sample_idx.numel()
                or (sample_idx < 0).any() or (sample_idx >= n).any()
                or tuple(depths.shape) != (sample_idx.numel(), h, w)
                or masks.shape != depths.shape):
            raise ValueError("Dense output requires unique keyframes with original HxW depth/masks")

        dense_depth = depths.new_empty((n, h, w))
        dense_mask = torch.empty((n, h, w), dtype=torch.bool)
        dense_depth[sample_idx], dense_mask[sample_idx] = depths, masks
        missing = torch.ones(n, dtype=torch.bool)
        missing[sample_idx] = False
        missing_idx = missing.nonzero(as_tuple=True)[0]
        if missing_idx.numel() == 0:
            return dense_depth, dense_mask

        nearest = (missing_idx[:, None] - sample_idx[None, :]).abs().argmin(dim=1)
        with self._isolated_output_prediction():
            for anchor_pos in nearest.unique().tolist():
                group = missing_idx[nearest == anchor_pos]
                for batch in group.split(batch_size):
                    indices = torch.cat([sample_idx[anchor_pos:anchor_pos + 1], batch])
                    batch_depth, batch_mask = self._da3_chunk_depth_no_umeyama(
                        frames[indices], extrinsics[indices], intrinsics[indices])
                    batch_depth = batch_depth.detach().cpu()
                    batch_mask = batch_mask.detach().cpu()
                    scale = self._compute_overlap_scale(
                        ref_depth=depths[anchor_pos:anchor_pos + 1],
                        pred_overlap_depth=batch_depth[:1],
                        ref_mask=masks[anchor_pos:anchor_pos + 1],
                        pred_overlap_mask=batch_mask[:1],
                    )
                    if not np.isfinite(scale) or scale <= 0:
                        raise ValueError(f"Invalid dense-output anchor scale: {scale}")
                    new_depth = batch_depth[1:] / scale
                    dense_depth[batch] = new_depth
                    dense_mask[batch] = (
                        batch_mask[1:] & torch.isfinite(new_depth) & (new_depth > 0))
                    del batch_depth, batch_mask, new_depth
        return dense_depth, dense_mask

    def _begin_output_run(self, save_dir, seed, num_chunks, preview=False):
        """Start an explicitly owned run; reject nonempty or symlinked paths.

        All other output methods require this call, including standalone
        world_gen_step tests. None means manifest/buffers only, with no I/O.
        Returns the live manifest, not a copy.
        """
        self._owned_output_dir = None
        self._output_owned_dirs = {}
        self._output_owned_files = {}
        self._output_warps = []
        self._output_manifest = {
            "schema_version": 1, "status": "running",
            "seed": None if seed is None else int(seed),
            "num_chunks": None if num_chunks is None else int(num_chunks),
            "preview": bool(preview), "fps": 15, "chunks": [], "files": {},
            "filenames": {
                "output.mp4": "Generated RGB; renderer RGB in preview mode.",
                "depth.mp4": "Spectral inverse-depth color; invalid/skipped pixels black.",
                "warp.mp4": "Actual renderer conditioning RGB, with chunk-0 anchor if supplied.",
                "sha256.json": "Run manifest, actual sampling/input snapshots and file hashes.",
                "chunks/chunk_{index:04d}/": "Live output/depth/warp videos; removed on successful finalization.",
            },
            "tensor_hash_encoding": "contiguous raw bytes viewed as uint8; no dtype conversion",
            "frame_ranges": "Zero-based, end-exclusive; size is [width, height].",
            "chunk_files_retained": save_dir is not None,
        }
        if save_dir is not None:
            requested = Path(save_dir).absolute()
            if any(p.is_symlink() for p in (requested, *requested.parents)):
                raise ValueError("Output directory must not contain symlinks")
            requested.mkdir(parents=True, exist_ok=True)
            root = requested.resolve()
            if any(root.iterdir()):
                raise FileExistsError(f"Refusing nonempty output directory: {root}")
            self._owned_output_dir = root
            self._output_owned_dirs[root] = self._output_identity(root)
            chunks = root / "chunks"
            chunks.mkdir()
            self._output_owned_dirs[chunks] = self._output_identity(chunks)
            if set(root.iterdir()) != {chunks}:
                raise FileExistsError("Output directory changed during initialization")
            self._flush_output_manifest()
        return self._output_manifest

    @staticmethod
    def _tensor_snapshot(tensor):
        """Hash the actual tensor's bytes, including BF16, without any RNG use."""
        raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
        return {
            "shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "device": str(tensor.device), "sha256": hashlib.sha256(raw).hexdigest(),
        }

    @staticmethod
    def _output_identity(path):
        info = path.lstat()
        return info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode)

    def _check_output_path(self, path, allow_new=False):
        """Only access unchanged run-owned parents/files, never a symlink."""
        root = self._owned_output_dir
        if root is None or not path.is_relative_to(root):
            raise RuntimeError("Path is outside this output run")
        if any(p.is_symlink() for p in (root, *root.parents)):
            raise RuntimeError("Output directory was replaced with a symlink")
        parents = [root] if path == root else [root / p for p in reversed(path.relative_to(root).parents)]
        for directory in parents:
            identity = self._output_identity(directory)
            if (identity != self._output_owned_dirs.get(directory)
                    or identity[2] != stat.S_IFDIR):
                raise RuntimeError(f"Unowned or replaced output directory: {directory}")
        expected = self._output_owned_dirs.get(path, self._output_owned_files.get(path))
        try:
            identity = self._output_identity(path)
        except FileNotFoundError:
            if allow_new and expected is None:
                return
            raise
        if identity != expected or identity[2] not in (stat.S_IFDIR, stat.S_IFREG):
            raise RuntimeError(f"Refusing unowned or replaced output path: {path}")

    def _require_output_run(self, save_dir):
        if not hasattr(self, "_output_manifest"):
            raise RuntimeError("Call _begin_output_run before recording/saving outputs")
        if self._output_manifest["status"] == "completed":
            raise RuntimeError("Output run is complete; call _begin_output_run for a new run")
        root = self._owned_output_dir
        if save_dir is None:
            if root is not None:
                raise ValueError("save_dir must match _begin_output_run")
        else:
            requested = Path(save_dir).absolute()
            if (root is None or any(p.is_symlink() for p in (requested, *requested.parents))
                    or requested.resolve() != root):
                raise ValueError("save_dir must match the non-symlink run-owned directory")
            self._check_output_path(root)
        return root

    def _output_chunk(self, chunk_idx):
        index = int(chunk_idx)
        if index < 0:
            raise ValueError("chunk_idx must be nonnegative")
        for entry in self._output_manifest["chunks"]:
            if entry["index"] == index:
                return entry
        entry = {"index": index, "status": "pending", "files": {}}
        self._output_manifest["chunks"].append(entry)
        self._output_manifest["chunks"].sort(key=lambda item: item["index"])
        return entry

    def _output_chunk_dir(self, chunk_idx):
        path = self._owned_output_dir / "chunks" / f"chunk_{int(chunk_idx):04d}"
        self._check_output_path(path, allow_new=True)
        if path not in self._output_owned_dirs:
            path.mkdir()
            self._output_owned_dirs[path] = self._output_identity(path)
        return path

    def _atomic_output_write(self, path, writer):
        """Keep temporary files in the owned directory, with an encoder suffix."""
        self._check_output_path(path, allow_new=True)
        temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
        self._check_output_path(temporary, allow_new=True)
        # Exclusive creation also refuses unknown/stale temporary files.
        with temporary.open("xb"):
            pass
        self._output_owned_files[temporary] = self._output_identity(temporary)
        try:
            writer(temporary)
            self._check_output_path(temporary)
            self._check_output_path(path, allow_new=True)
            os.replace(temporary, path)
            self._output_owned_files[path] = self._output_owned_files.pop(temporary)
        finally:
            if temporary in self._output_owned_files:
                self._check_output_path(temporary)
                temporary.unlink()
                del self._output_owned_files[temporary]

    def _flush_output_manifest(self):
        # Validate even an in-memory run: never permit JSON NaN/Infinity.
        streams = getattr(self, "_case_random_streams", None)
        if streams is not None:
            self._output_manifest["randomness"] = {
                **streams.describe(),
                "initialization_seed": getattr(self, "_initialization_seed", None),
                "sampling_seed": streams.seed,
                "augmentation_seed": streams.seed + 1,
                "augmentation_enabled": bool(getattr(self, "noise_aug_level", 0)),
                "fixed_upstream_pose_ransac_seed": 42,
                "cuda_bitwise_determinism_guaranteed": False,
            }
        text = json.dumps(self._output_manifest, indent=2, allow_nan=False) + "\n"
        if self._owned_output_dir is not None:
            self._atomic_output_write(
                self._owned_output_dir / "sha256.json",
                lambda temporary: temporary.write_text(text, encoding="utf-8"))

    def _write_output_video(self, path, video):
        fps = self._output_manifest["fps"]
        self._atomic_output_write(path, lambda temporary: write_video(str(temporary), video, fps=fps))
        self._check_output_path(path)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return {
            "path": path.relative_to(self._owned_output_dir).as_posix(),
            "sha256": digest.hexdigest(), "bytes": path.stat().st_size,
            "frames": int(len(video)), "fps": fps,
            "size": [int(video.shape[2]), int(video.shape[1])],
        }

    def _record_sampling_snapshot(
        self, save_dir, chunk_idx, case_seed, noise, sampling_generator,
        augmentation_generator, history_noise, rendered_noise, input_shared, raymap,
    ):
        """Call immediately before latent_sampling with its prepared arguments.

        raymap must be input_pos['raymap'], including merged-CFG expansion, not
        an earlier unbatched raymap. No tensor/noise is reconstructed here.
        Returns the live chunk entry and preserves already-recorded warp data.
        """
        self._require_output_run(save_dir)
        if sampling_generator is None:
            raise ValueError("The actual sampling_generator is required")
        level = getattr(self, "noise_aug_level", None)
        augmented = history_noise is not None or rendered_noise is not None
        augmentation = {"enabled": augmented or (level is not None and int(level) > 0)}
        if augmentation["enabled"]:
            augmentation.update({
                "level": None if level is None else int(level),
                "generator": "torch.Generator" if augmentation_generator is not None else None,
                "stream_seed": (int(augmentation_generator.initial_seed())
                                if augmentation_generator is not None else None),
                "generator_device": (str(augmentation_generator.device)
                                     if augmentation_generator is not None else None),
                "noises_captured": history_noise is not None and rendered_noise is not None,
            })
            for name, tensor in (("history", history_noise), ("rendered", rendered_noise)):
                if tensor is not None:
                    augmentation[name] = self._tensor_snapshot(tensor)
                    augmentation[f"{name}_sha256"] = augmentation[name]["sha256"]
        sampler_inputs = {
            name: (self._tensor_snapshot(input_shared[name])
                   if input_shared.get(name) is not None else None)
            for name in ("history_latents", "rendered_latents", "rendered_masks", "clip_features")
        }
        sampler_inputs["raymap"] = self._tensor_snapshot(raymap) if raymap is not None else None
        entry = self._output_chunk(chunk_idx)
        entry.update({
            "status": "sampling", "sampling_step": int(chunk_idx),
            "sampling_step_kind": "ar_chunk", "base_seed": int(case_seed),
            "capture_point": "before_latent_sampling",
            "capture_note": "Actual noise and prepared sampler tensors; not regenerated/reconstructed.",
            "sampling": {
                "generator": "torch.Generator", "stream_seed": int(sampling_generator.initial_seed()),
                "generator_device": str(sampling_generator.device),
                "rand_device": "cpu", "rand_dtype": "torch.float32",
                **self._tensor_snapshot(noise),
            },
            "noise_augmentation": augmentation, "sampler_inputs": sampler_inputs,
        })
        self._flush_output_manifest()
        return entry

    @staticmethod
    def _output_rgb_video(frames):
        if frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) == 0:
            raise ValueError("RGB must be nonempty (F,H,W,3), float [0,1] or uint8")
        video = np.empty(tuple(frames.shape), dtype=np.uint8)
        for i, frame in enumerate(frames):
            frame = torch.as_tensor(frame).detach().cpu()
            if frame.dtype != torch.uint8:
                frame = (frame.float().clamp(0, 1) * 255).to(torch.uint8)
            video[i] = frame.numpy()
        return video

    def _save_chunk_warp(self, save_dir, chunk_idx, warp_color, warp_mask, anchor_frame=None):
        """Save the unmodified renderer output before the conditioning consumes it."""
        root = self._require_output_run(save_dir)
        if int(chunk_idx) != len(self._output_warps):
            raise ValueError("Save each chunk warp once, in consecutive index order")
        if warp_color.ndim != 4 or warp_color.shape[1] != 3:
            raise ValueError("warp_color must have shape (F,3,H,W)")
        expected_mask = (warp_color.shape[0], *warp_color.shape[2:])
        if tuple(warp_mask.shape) not in (expected_mask, (expected_mask[0], 1, *expected_mask[1:])):
            raise ValueError("Warp color and mask must be frame-aligned")
        warp = {
            "status": "ready", "capture_point": "before_warp_condition_consumption",
            "color": self._tensor_snapshot(warp_color), "mask": self._tensor_snapshot(warp_mask),
            "anchor_prepended": anchor_frame is not None,
            "frame0": "anchor_frame" if anchor_frame is not None else "renderer_frame_0",
        }
        video = self._output_rgb_video(warp_color.permute(0, 2, 3, 1))
        if anchor_frame is not None:
            if int(chunk_idx) != 0:
                raise ValueError("Only the first chunk can prepend an anchor")
            anchor = torch.as_tensor(anchor_frame)
            warp["anchor"] = self._tensor_snapshot(anchor)
            if anchor.ndim == 3:
                if (tuple(anchor.shape) != tuple(video.shape[1:])
                        and tuple(anchor.shape) == (3, *video.shape[1:3])):
                    anchor = anchor.permute(1, 2, 0)
                anchor = anchor.unsqueeze(0)
            if tuple(anchor.shape) != (1, *video.shape[1:]):
                raise ValueError("Anchor must be one RGB frame with the renderer resolution")
            video = np.concatenate([self._output_rgb_video(anchor), video], axis=0)
        warp["frames"] = int(len(video))
        entry = self._output_chunk(chunk_idx)
        entry["warp"] = warp
        self._output_warps.append(video)  # Retain raw CPU uint8, never decode a chunk MP4.
        self._flush_output_manifest()
        if root is not None:
            entry["files"]["warp.mp4"] = self._write_output_video(
                self._output_chunk_dir(chunk_idx) / "warp.mp4", video)
            warp["status"] = "saved"
        self._flush_output_manifest()
        return entry

    @staticmethod
    def _check_output_alignment(frames, depths, masks):
        if (frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) == 0
                or tuple(frames.shape[:3]) != tuple(depths.shape)
                or tuple(depths.shape) != tuple(masks.shape)):
            raise ValueError("RGB, depth and mask outputs must be nonempty and frame-aligned")

    @staticmethod
    def _output_disparity(depth, mask, skip_depth=False):
        # Only a single frame is converted to float64, never a video colormap.
        depth = torch.as_tensor(depth).detach().cpu().double().numpy()
        valid = torch.as_tensor(mask).detach().cpu().bool().numpy().copy()
        valid &= np.isfinite(depth) & (depth > 0)
        if skip_depth:
            valid[:] = False
        disparity = np.zeros_like(depth)
        with np.errstate(over="ignore", under="ignore"):
            np.divide(1.0, depth, out=disparity, where=valid)
        valid &= np.isfinite(disparity)
        return disparity, valid

    def _output_depth_summary(self, depths, masks, skip_depth=False):
        lo = hi = None
        count = 0
        for depth, mask in zip(depths, masks):
            disparity, valid = self._output_disparity(depth, mask, skip_depth)
            count += int(valid.sum())
            if valid.any():
                values = disparity[valid]
                lower, upper = float(values.min()), float(values.max())
                lo = lower if lo is None else min(lo, lower)
                hi = upper if hi is None else max(hi, upper)
        total = int(np.prod(depths.shape))
        return {
            "valid_pixels": count, "total_pixels": total,
            "valid_ratio": count / total if total else 0.0, "all_invalid": count == 0,
            "disparity_range": [lo, hi] if count else None,
            "normalization": "global_minmax_disparity", "constant_disparity_value": 0.5,
            "colormap": "Spectral", "invalid_color": [0, 0, 0],
        }

    def _output_depth_video(self, depths, masks, summary, skip_depth=False):
        video = np.zeros((*depths.shape, 3), dtype=np.uint8)
        if summary["all_invalid"]:
            return video
        lo, hi = summary["disparity_range"]
        for i, (depth, mask) in enumerate(zip(depths, masks)):
            disparity, valid = self._output_disparity(depth, mask, skip_depth)
            if not valid.any():
                continue
            normalized = np.zeros_like(disparity)
            if hi > lo:
                np.subtract(disparity, lo, out=normalized, where=valid)
                np.divide(normalized, hi - lo, out=normalized, where=valid)
            else:
                normalized[valid] = 0.5
            np.clip(normalized, 0, 1, out=normalized)
            # colorize_depth(normalize=False) still takes 1/depth internally.
            # Supply reciprocal normalized disparity; zero maps exactly to inf.
            color_depth = np.full_like(normalized, np.inf)
            with np.errstate(over="ignore", under="ignore"):
                np.divide(1.0, normalized, out=color_depth, where=valid & (normalized > 0))
            colored = colorize_depth(color_depth, mask=valid, normalize=False)
            colored[~valid] = 0
            video[i] = colored
        return video

    def _save_chunk_outputs(self, save_dir, chunk_idx, frames, depths, masks, skip_depth=False):
        """Save chunk 0 including its anchor, then only new frames (81/80 normally)."""
        root = self._require_output_run(save_dir)
        self._check_output_alignment(frames, depths, masks)
        entry = self._output_chunk(chunk_idx)
        if entry["status"] == "completed":
            raise ValueError("Chunk outputs have already been completed")
        if "warp" not in entry or entry["warp"]["frames"] != len(frames):
            raise ValueError("Save a frame-aligned chunk warp before chunk outputs")
        if tuple(self._output_warps[int(chunk_idx)].shape) != tuple(frames.shape):
            raise ValueError("Chunk warp and output RGB resolutions must match")
        if not self._output_manifest["preview"]:
            expected = int(getattr(self, "chunk_size", 80)) + int(int(chunk_idx) == 0)
            if len(frames) != expected:
                raise ValueError(f"Chunk {chunk_idx} requires {expected} frames, got {len(frames)}")
        summary = self._output_depth_summary(depths, masks, skip_depth)
        entry.update({
            "status": "saving", "frame_count": int(len(frames)),
            "depth_skipped": bool(skip_depth), "valid_ratio": summary["valid_ratio"],
            "rgb": self._tensor_snapshot(torch.as_tensor(frames)),
            "depth": self._tensor_snapshot(torch.as_tensor(depths)),
            "mask": self._tensor_snapshot(torch.as_tensor(masks)), "depth_summary": summary,
        })
        self._flush_output_manifest()
        if root is not None:
            directory = self._output_chunk_dir(chunk_idx)
            entry["files"]["output.mp4"] = self._write_output_video(
                directory / "output.mp4", self._output_rgb_video(frames))
            entry["files"]["depth.mp4"] = self._write_output_video(
                directory / "depth.mp4", self._output_depth_video(depths, masks, summary, skip_depth))
        entry["status"] = "completed"
        self._flush_output_manifest()
        return entry

    def _cleanup_output_chunks(self):
        """Preflight the entire owned tree, then remove only its known paths."""
        root = self._owned_output_dir
        chunks = root / "chunks"
        final_paths = {root / name for name in ("output.mp4", "depth.mp4", "warp.mp4", "sha256.json")}
        self._check_output_path(root)
        if set(root.iterdir()) != {chunks, *final_paths}:
            raise RuntimeError("Unknown files in output directory; retaining chunk diagnostics")
        for path in final_paths:
            self._check_output_path(path)
        directories = [p for p in self._output_owned_dirs if p.is_relative_to(chunks)]
        files = [p for p in self._output_owned_files if p.is_relative_to(chunks)]
        for path in [*directories, *files]:
            self._check_output_path(path)
        for directory in directories:
            expected = {p for p in [*directories, *files] if p.parent == directory}
            if set(directory.iterdir()) != expected:
                raise RuntimeError(f"Unknown chunk files; refusing cleanup: {directory}")
        for path in files:
            self._check_output_path(path)
            path.unlink()
            del self._output_owned_files[path]
        for path in sorted(directories, key=lambda p: len(p.parts), reverse=True):
            self._check_output_path(path)
            path.rmdir()
            del self._output_owned_dirs[path]

    def _finalize_outputs(self, save_dir, frames, depths, masks):
        """Encode three final videos from raw buffers; retain chunks on failure.

        The final depth range is global across all valid frames. A skipped tail
        remains black, without predicting any additional depth. Returns manifest.
        """
        root = self._require_output_run(save_dir)
        manifest = self._output_manifest
        try:
            self._check_output_alignment(frames, depths, masks)
            chunks = manifest["chunks"]
            if (not chunks or any(entry["status"] != "completed" for entry in chunks)
                    or [entry["index"] for entry in chunks] != list(range(len(chunks)))
                    or len(self._output_warps) != len(chunks)):
                raise ValueError("Finalization requires consecutive completed chunks and their raw warps")
            if (not manifest["preview"] and manifest["num_chunks"] is not None
                    and len(chunks) != manifest["num_chunks"]):
                raise ValueError("Completed chunk count does not match num_chunks")
            offset, skipped = 0, []
            for entry, warp in zip(chunks, self._output_warps):
                end = offset + entry["frame_count"]
                if tuple(warp.shape) != (entry["frame_count"], *frames.shape[1:]):
                    raise ValueError("Raw chunk warps must match RGB frame counts and resolution")
                entry["frame_range"] = [offset, end]
                if entry["depth_skipped"]:
                    skipped.append([offset, end])
                offset = end
            if offset != len(frames):
                raise ValueError("Final RGB/depth/warp frame counts must match")
            # The caller's masks are never changed, even for an explicitly skipped tail.
            final_masks = masks
            if skipped:
                final_masks = torch.as_tensor(masks).detach().cpu().bool().clone()
                for start, end in skipped:
                    final_masks[start:end] = False
            summary = self._output_depth_summary(depths, final_masks)
            manifest.update({
                "status": "finalizing", "frame_count": int(len(frames)),
                "size": [int(frames.shape[2]), int(frames.shape[1])],
                "depth_skipped_ranges": skipped,
                "depth_tail_skipped_range": (skipped[-1] if skipped and skipped[-1][1] == len(frames) else None),
                "depth_summary": summary,
            })
            manifest.pop("error", None)
            self._flush_output_manifest()
            if root is not None:
                manifest["files"]["output.mp4"] = self._write_output_video(
                    root / "output.mp4", self._output_rgb_video(frames))
                manifest["files"]["depth.mp4"] = self._write_output_video(
                    root / "depth.mp4", self._output_depth_video(depths, final_masks, summary))
                manifest["files"]["warp.mp4"] = self._write_output_video(
                    root / "warp.mp4", np.concatenate(self._output_warps, axis=0))
            manifest.update(status="completed", chunk_files_retained=False)
            # Commit the final JSON before cleanup: JSON/encoding failure leaves all chunks.
            self._flush_output_manifest()
            if root is not None:
                self._cleanup_output_chunks()
            self._output_warps.clear()
        except Exception as exc:
            manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            manifest["chunk_files_retained"] = root is not None and (root / "chunks").exists()
            try:
                self._flush_output_manifest()
            except (OSError, RuntimeError, ValueError):
                pass  # Do not bypass ownership checks or hide the original failure.
            raise
        return manifest