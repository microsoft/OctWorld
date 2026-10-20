"""OctWorld Generation driver using WanWorldPipeline with image + trajectory.

User inputs (single canonical format each):
  1. a first-frame image (any PIL-readable image),
  2. a camera trajectory ``.npz`` with ``extrinsics`` (T,4,4) and
     ``intrinsics`` (T,3,3),
  3. a first-frame anchor depth as a 16-bit log-encoded PNG (read via
      ``utils.io.read_depth``); defines the metric scale of the user
      trajectory. Defaults to the bundled DL3DV GT example.
  4. a prompt ``.txt`` file (one prompt per chunk → list; single line → str).

Per-chunk depth comes from DA3 aligned to the supplied user cameras; only the
first-frame anchor depth is supplied to anchor the trajectory scale.

Usage:
    python example/worldgen/wan/world_gen.py \
        --image path/to/first.png \
        --trajectory path/to/traj.npz \
        --anchor_depth assets/standard/scenes/scene_00000/anchor_depth.png \
        --prompt_file assets/standard/scenes/scene_00000/prompt.txt \
        --dit_ckpt <ckpt.pt> \
        --model_weights_dir model_weights \
        --save_dir outputs/dl3dv_gt/ \
        [--preview_only]
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Published OctWorld 1.5 model package.
DEFAULT_DIT_CKPT = os.path.join(
    PROJECT_ROOT, "model_weights", "octworld-1.5", "model")

DEFAULT_CASE_DIR = os.path.join(
    PROJECT_ROOT, "assets", "standard", "scenes", "scene_00000")
DEFAULT_IMAGE = os.path.join(DEFAULT_CASE_DIR, "image.png")
DEFAULT_TRAJECTORY = os.path.join(DEFAULT_CASE_DIR, "trajectory.npz")
DEFAULT_ANCHOR_DEPTH = os.path.join(DEFAULT_CASE_DIR, "anchor_depth.png")
DEFAULT_PROMPT_FILE = os.path.join(DEFAULT_CASE_DIR, "prompt.txt")


def parse_args():
    p = argparse.ArgumentParser(description="OctWorld Generation driver")
    # User inputs default to the bundled DL3DV GT case in assets/standard.
    p.add_argument("--image", default=DEFAULT_IMAGE, help="First-frame image path.")
    p.add_argument("--trajectory", default=DEFAULT_TRAJECTORY,
                   help=".npz with 'extrinsics' (T,4,4) and 'intrinsics' (T,3,3).")
    p.add_argument("--anchor_depth", default=DEFAULT_ANCHOR_DEPTH,
                   help="First-frame anchor depth (16-bit log-encoded PNG). Defines "
                        "the metric scale of the user trajectory. Defaults to the "
                        "bundled DL3DV GT anchor depth.")
    p.add_argument("--prompt", default=None,
                   help="Single text prompt override (applied to all chunks). "
                        "If unset, --prompt_file is used.")
    p.add_argument(
        "--prompt_file",
        default=None,
        help=(
            ".txt with one prompt per chunk (single line → str). "
            "Defaults to the bundled DL3DV GT prompts."
        ),
    )
    # Model / weights
    p.add_argument("--dit_ckpt", default=DEFAULT_DIT_CKPT,
                   help="OctWorld DiT checkpoint file or DCP directory. Defaults "
                        "to model_weights/octworld-1.5/model. Optional in --debug.")
    p.add_argument("--model_weights_dir", default=os.path.join(PROJECT_ROOT, "model_weights"),
                   help="Dir containing Wan2.1-I2V-14B-480P/ (VAE/T5/CLIP).")
    # Run control
    p.add_argument("--save_dir", default="outputs/dl3dv_gt/")
    p.add_argument("--preview_only", action="store_true",
                    help="Render the warp without diffusion; still constructs the "
                        "pipeline and requires model weights, depth prediction and a renderer.")
    p.add_argument("--debug", action="store_true",
                   help="Load the local tiny model (debugs/tiny-octworld-wan-tiny-b4) "
                        "instead of the 14B checkpoint. Requires a compatible local "
                        "tiny checkpoint and the other inference models.")
    p.add_argument("--max_chunks", type=int, default=None,
                   help="Stop after this many real 80-frame chunks; use 1 for the first-chunk smoke test.")
    p.add_argument("--depth_method", default="align_with_user",
                   choices=["standalone", "align_with_user", "scale_alignment"],
                    help="DA3 depth aligned to user cameras (default: align_with_user, "
                        "with per-chunk scale-alignment fallback). scale_alignment "
                        "always skips Umeyama; legacy standalone is disabled.")
    integration_group = p.add_mutually_exclusive_group()
    integration_group.add_argument(
        "--offline_integration", dest="offline_integration", action="store_true",
        help="Allocate new frames, then reset observations and re-integrate all "
             "cached frames in the retained octree.")
    integration_group.add_argument(
        "--online_integration", dest="offline_integration", action="store_false",
        help="Incrementally integrate only newly selected frames (default); "
             "history is still cached. This is not DA3 streaming.")
    p.set_defaults(offline_integration=False)
    p.add_argument("--octree_depth_strategy", default="pixel_size_to_volume_size",
                   choices=["pixel_size_to_volume_size", "max_depth", "bigger_voxels"],
                    help="OctMap carve depth strategy. bigger_voxels uses "
                        "bigger_voxel_pixel_size_scale (default 2.0) before voxel-depth mapping.")
    p.add_argument("--bigger_voxel_pixel_size_scale", type=float, default=2.0,
                   help="Pixel-size scale factor used when --octree_depth_strategy="
                        "bigger_voxels.")
    p.add_argument("--sample_ratio", type=int, default=1,
                    help="Uniform mode only: use every Nth frame for depth prediction "
                        "and map integration. Ignored by keyframe selection.")
    p.add_argument("--frame_select_method", default="keyframe",
                   choices=["uniform", "keyframe"],
                   help="Select frames for depth prediction and map integration "
                        "with RAFT keyframes (default) or uniform stride sampling.")
    p.add_argument("--keyframe_thresh", type=float, default=5.0,
                    help="Keyframe mode only: keep frames with mean RAFT flow > this "
                        "threshold, in preprocessed RAFT pixels (default input 480x832).")
    p.add_argument("--keyframe_max_interval", type=int, default=8,
                    help="Keyframe mode only: skipped-frame counter threshold. "
                        "Default 8 permits a 9-frame index gap; block ends are always kept.")
    p.add_argument("--seed", type=int, default=425,
                    help="Case RNG seed (default 425). Use the matching experiment seed "
                        "when reproducing an evaluation run.")
    args = p.parse_args()
    if args.max_chunks is not None and args.max_chunks < 1:
        p.error("--max_chunks must be at least 1")
    return args


def main():
    args = parse_args()
    if os.path.isdir(args.save_dir) and os.listdir(args.save_dir):
        raise FileExistsError(f"Use an empty --save_dir; existing results are preserved: {args.save_dir}")
    from octworld.pipelines.wan import WanWorldPipeConfig

    pipe_cfg = WanWorldPipeConfig()
    pipe_cfg.init_seed = int(args.seed)
    pipe_cfg.depth_method = args.depth_method
    pipe_cfg.offline_integration = bool(args.offline_integration)
    pipe_cfg.octree_depth_strategy = str(args.octree_depth_strategy)
    pipe_cfg.bigger_voxel_pixel_size_scale = float(args.bigger_voxel_pixel_size_scale)
    pipe_cfg.sample_ratio = int(args.sample_ratio)
    pipe_cfg.frame_select_method = str(args.frame_select_method)
    pipe_cfg.keyframe_thresh = float(args.keyframe_thresh)
    pipe_cfg.keyframe_max_interval = int(args.keyframe_max_interval)
    target_height, target_width = pipe_cfg.mapper_resolution

    # ── Load user inputs (single canonical format) ────────────────
    image = Image.open(args.image).convert("RGB").resize(
        (target_width, target_height))

    # Trajectory: .npz with 'extrinsics' (T,4,4) + 'intrinsics' (T,3,3).
    with np.load(args.trajectory) as traj:
        extrinsics = torch.from_numpy(np.asarray(traj["extrinsics"], dtype=np.float32))
        intrinsics = torch.from_numpy(np.asarray(traj["intrinsics"], dtype=np.float32))
    if (extrinsics.ndim != 3 or tuple(extrinsics.shape[1:]) != (4, 4)
            or extrinsics.shape[0] < 2
            or tuple(intrinsics.shape) != (extrinsics.shape[0], 3, 3)):
        raise ValueError("Trajectory requires matching (T,4,4)/(T,3,3) cameras with T >= 2")
    if not torch.isfinite(extrinsics).all() or not torch.isfinite(intrinsics).all():
        raise ValueError("Trajectory contains non-finite camera values")

    # Anchor depth: 16-bit log-encoded PNG (utils.io.read_depth).
    from utils.io import read_depth
    anchor_depth = torch.from_numpy(read_depth(args.anchor_depth).astype(np.float32))

    # Prompt: --prompt override, else a .txt file (one prompt per chunk → list;
    # single line → str).
    if args.prompt is not None:
        prompt = args.prompt
    else:
        with open(args.prompt_file or DEFAULT_PROMPT_FILE) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        prompt = lines if len(lines) > 1 else (lines[0] if lines else "")

    print(
        f"[Driver] seed={args.seed}, offline_integration={args.offline_integration}, "
        f"image={image.size}, traj={tuple(extrinsics.shape)}, "
        f"prompt={'list[%d]' % len(prompt) if isinstance(prompt, list) else repr(prompt)}"
    )

    # ── Build pipeline ──────────────────────────────────────────────────
    from octworld.pipelines.wan import WanWorldPipeline

    print("[Driver] Building WanWorldPipeline...")
    pipe = WanWorldPipeline(
        dit_ckpt_path=args.dit_ckpt,
        model_weights_dir=args.model_weights_dir,
        debug=args.debug,
        config=pipe_cfg,
    )

    # ── Run ──────────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    result = pipe(
        image=image,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        anchor_depth=anchor_depth,
        prompt=prompt,
        save_dir=args.save_dir,
        preview_only=args.preview_only,
        verbose=True,
        max_chunks=args.max_chunks,
        seed=int(args.seed),
    )

    if args.preview_only:
        print(f"[Driver] Preview rendered → {args.save_dir}/: output.mp4, depth.mp4, warp.mp4, sha256.json")
        return

    out_frames = result["frames"]
    print(f"[Driver] Done. frames={tuple(out_frames.shape)}, "
          f"depths={tuple(result['depths'].shape)}, "
          f"saved → {args.save_dir}/: output.mp4, depth.mp4, warp.mp4, sha256.json")


if __name__ == "__main__":
    main()
