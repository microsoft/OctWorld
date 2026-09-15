"""Driver for WanWorldPipeline — AR world generation from image + trajectory.

User inputs (single canonical format each):
  1. a first-frame image (any PIL-readable image),
  2. a camera trajectory ``.npz`` with ``extrinsics`` (T,4,4) and
     ``intrinsics`` (T,3,3),
  3. a first-frame anchor depth as a 16-bit log-encoded PNG (read via
     ``utils.io.read_depth``); defines the metric scale of the user
     trajectory. Defaults to the ParkingLot MoGe depth.
  4. a prompt ``.txt`` file (one prompt per chunk → list; single line → str).

Per-chunk depth comes from DA3 aligned to the supplied user cameras; only the
first-frame anchor depth is supplied to anchor the trajectory scale.

Usage:
    python example/worldgen/wan/world_gen.py \
        --image path/to/first.png \
        --trajectory path/to/traj.npz \
        --anchor_depth assets/ParkingLot/moge_depth.png \
        --prompt_file assets/ParkingLot/prompt.txt \
        --dit_ckpt <ckpt.pt> \
        --model_weights_dir model_weights \
        --save_dir debugs/wan_world_test/ \
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

DEFAULT_CASE_DIR = os.path.join(PROJECT_ROOT, "assets/ParkingLot")
DEFAULT_IMAGE = os.path.join(DEFAULT_CASE_DIR, "input_image.png")
DEFAULT_TRAJECTORY = os.path.join(DEFAULT_CASE_DIR, "camera_trajectory.npz")
DEFAULT_ANCHOR_DEPTH = os.path.join(DEFAULT_CASE_DIR, "moge_depth.png")
DEFAULT_PROMPT_FILE = os.path.join(DEFAULT_CASE_DIR, "prompt.txt")


def parse_args():
    p = argparse.ArgumentParser(description="WanWorldPipeline driver")
    # User inputs (default = ParkingLot case)
    p.add_argument("--image", default=DEFAULT_IMAGE, help="First-frame image path.")
    p.add_argument("--trajectory", default=DEFAULT_TRAJECTORY,
                   help=".npz with 'extrinsics' (T,4,4) and 'intrinsics' (T,3,3).")
    p.add_argument("--anchor_depth", default=DEFAULT_ANCHOR_DEPTH,
                   help="First-frame anchor depth (16-bit log-encoded PNG). Defines "
                        "the metric scale of the user trajectory. Defaults to the "
                        "ParkingLot MoGe depth.")
    p.add_argument("--prompt", default=None,
                   help="Single text prompt override (applied to all chunks). "
                        "If unset, --prompt_file is used.")
    p.add_argument("--prompt_file", default=None,
                   help=".txt with one prompt per chunk (single line → str). "
                        "Defaults to the ParkingLot prompt.")
    # Model / weights
    p.add_argument("--dit_ckpt", default=DEFAULT_DIT_CKPT,
                   help="OctWorld DiT checkpoint file or DCP directory. Defaults "
                        "to model_weights/octworld-1.5/model. Optional in --debug.")
    p.add_argument("--model_weights_dir", default="model_weights",
                   help="Dir containing Wan2.1-I2V-14B-480P/ (VAE/T5/CLIP).")
    # Run control
    p.add_argument("--save_dir", default="debugs/wan_world_parkinglot/")
    p.add_argument("--preview_only", action="store_true",
                   help="Only render the warp along the trajectory (no diffusion).")
    p.add_argument("--debug", action="store_true",
                   help="Load the local tiny model (debugs/tiny-octworld-wan-tiny-b4) "
                        "instead of the 14B checkpoint for a fast fake run.")
    p.add_argument("--max_chunks", type=int, default=None,
                   help="Cap the rollout to this many chunks (debug fake run).")
    p.add_argument("--depth_method", default="align_with_user",
                   choices=["standalone", "align_with_user", "scale_alignment"],
                   help="Depth prediction method (default: align_with_user).")
    integration_group = p.add_mutually_exclusive_group()
    integration_group.add_argument(
        "--offline_integration", dest="offline_integration", action="store_true",
        help="Re-integrate all cached frames after each OctMap update.")
    integration_group.add_argument(
        "--online_integration", dest="offline_integration", action="store_false",
        help="Incrementally integrate only new frames (default).")
    p.set_defaults(offline_integration=False)
    p.add_argument("--octree_depth_strategy", default="pixel_size_to_volume_size",
                   choices=["pixel_size_to_volume_size", "max_depth", "bigger_voxels"],
                   help="OctMap carve depth strategy. 'bigger_voxels' doubles pixel size before voxel-depth mapping.")
    p.add_argument("--bigger_voxel_pixel_size_scale", type=float, default=2.0,
                   help="Pixel-size scale factor used when --octree_depth_strategy="
                        "bigger_voxels.")
    p.add_argument("--sample_ratio", type=int, default=1,
                   help="Only use every Nth frame for depth prediction + 3D map integration.")
    p.add_argument("--frame_select_method", default="keyframe",
                   choices=["uniform", "keyframe"],
                   help="Select frames for depth prediction and map integration "
                        "with RAFT keyframes (default) or uniform stride sampling.")
    p.add_argument("--keyframe_thresh", type=float, default=5.0,
                   help="Mean RAFT flow magnitude threshold for a keyframe.")
    p.add_argument("--keyframe_max_interval", type=int, default=8,
                   help="Force a keyframe after this many frames without one.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed fixed before running the case.")
    return p.parse_args()


def main():
    args = parse_args()
    from octworld.pipelines.wan import WanWorldPipeConfig

    pipe_cfg = WanWorldPipeConfig()
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
    traj = np.load(args.trajectory)
    extrinsics = torch.from_numpy(np.asarray(traj["extrinsics"], dtype=np.float32))
    intrinsics = torch.from_numpy(np.asarray(traj["intrinsics"], dtype=np.float32))

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
    print(f"[Driver] image={image.size}, traj={tuple(extrinsics.shape)}, "
          f"prompt={'list[%d]' % len(prompt) if isinstance(prompt, list) else repr(prompt)}")

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
        print(f"[Driver] Preview rendered → {args.save_dir}/preview_warp.mp4")
        return

    out_frames = result["frames"]
    print(f"[Driver] Done. frames={tuple(out_frames.shape)}, "
          f"saved → {args.save_dir}/output.mp4")


if __name__ == "__main__":
    main()
