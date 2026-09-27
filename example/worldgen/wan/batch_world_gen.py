"""Batch driver for WanWorldPipeline -- run a wan_standard dataset.

Builds the pipeline **once** and runs a list of instances sequentially. Each
instance specifies its own inputs (image, trajectory, anchor_depth, prompt).
Runtime settings and defaults mirror the adjacent single-case ``world_gen.py``.

Outputs are written to::

    <base_folder> / <instances_json_basename> / <sample_id> /
        info.json                 # resolved paths + runtime config
        output.mp4 / depth.mp4 / warp.mp4 / sha256.json
        SUCCESS.txt   or   FAILED.txt

``sample_id`` is the instance's 0-based position in the instances list (global,
unaffected by rank), so different ranks never collide.

Sharding: pass ``--rank`` / ``--world_size`` to split instances across single-GPU
workers (``instances[rank::world_size]``, mirroring datapipe). Pin each worker to
a GPU externally via ``CUDA_VISIBLE_DEVICES``.

Instances JSON format (a list of dicts)::

    [
      {"image": "...", "trajectory": "...npz",
       "anchor_depth": "...png",   # optional → defaults to ParkingLot moge
       "prompt_file": "...txt"},   # optional → defaults to ParkingLot prompt
      ...
    ]

Usage::

    python example/worldgen/wan/batch_world_gen.py \
        --instances assets/wan_standard/instances.json \
        --base_folder tmp/batch_out \
        --dit_ckpt <ckpt.pt> \
        --model_weights_dir model_weights \
        [--rank 0 --world_size 1] \
        [--debug]
"""

import argparse
from dataclasses import asdict
import json
import os
import sys
import traceback
from typing import Any

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Defaults mirror world_gen.py.
DEFAULT_DIT_CKPT = os.path.join(
    PROJECT_ROOT, "model_weights", "octworld-1.5", "model")
DEFAULT_INSTANCES = os.path.join(
    PROJECT_ROOT, "assets", "wan_standard", "instances.json")


def parse_args():
    p = argparse.ArgumentParser(description="WanWorldPipeline batch driver")
    # Instances + outputs
    p.add_argument("--instances", default=DEFAULT_INSTANCES,
                   help="JSON file: a list of instance dicts (image/trajectory/"
                        "anchor_depth/prompt_file).")
    p.add_argument("--base_folder", required=True,
                   help="Output root. Results go to "
                        "<base_folder>/<exp_name>/<sample_id>/.")
    p.add_argument("--exp_name", default=None,
                   help="Output subdir name under base_folder. Defaults to the "
                        "instances JSON basename.")
    # Sharding (single-GPU workers; pin GPUs via CUDA_VISIBLE_DEVICES)
    p.add_argument("--rank", type=int, default=0, help="Current rank.")
    p.add_argument("--world_size", type=int, default=1, help="Total ranks.")
    # Model / weights (paths are not fixed → always CLI)
    p.add_argument("--dit_ckpt", default=DEFAULT_DIT_CKPT,
                   help="Trained Wan DiT checkpoint (.pt). Optional in --debug.")
    p.add_argument("--model_weights_dir", default=os.path.join(PROJECT_ROOT, "model_weights"),
                   help="Dir containing Wan2.1-I2V-14B-480P/ (VAE/T5/CLIP).")
    # Debug
    p.add_argument("--debug", action="store_true",
                   help="Load the local tiny model for a fast fake run.")
    # Depth backend
    p.add_argument("--depth_method", default="align_with_user",
                   choices=["standalone", "align_with_user", "scale_alignment"],
                   help="Depth prediction method (see WanWorldDepthMixin): "
                        "'standalone' = DA3 streaming with DA3 poses; "
                        "'align_with_user' = non-streaming, map uses the user "
                        "camera and the predictor only provides depth; "
                        "'scale_alignment' = non-streaming DA3 predict + "
                        "overlap-based chunk scale alignment.")
    integration_group = p.add_mutually_exclusive_group()
    integration_group.add_argument(
        "--offline_integration", dest="offline_integration", action="store_true",
        help="Re-integrate all cached observations after each map update.")
    integration_group.add_argument(
        "--online_integration", dest="offline_integration", action="store_false",
        help="Use incremental online OctMap integration.")
    p.set_defaults(offline_integration=False)
    p.add_argument("--octree_depth_strategy", default="pixel_size_to_volume_size",
                   choices=["pixel_size_to_volume_size", "max_depth", "bigger_voxels"],
                   help="OctMap carve depth strategy. 'bigger_voxels' doubles the "
                        "effective pixel size before mapping to voxel depth.")
    p.add_argument("--bigger_voxel_pixel_size_scale", type=float, default=2.0,
                   help="Pixel-size scale factor used when --octree_depth_strategy="
                        "bigger_voxels.")
    p.add_argument("--sample_ratio", type=int, default=1,
                   help="Only use every Nth frame for depth prediction and 3D map "
                        "integration. 1 keeps the original dense behavior.")
    p.add_argument("--frame_select_method", default="keyframe",
                   choices=["uniform", "keyframe"],
                   help="How to pick frames for depth prediction / map integration. "
                        "'uniform' = every Nth frame (--sample_ratio); 'keyframe' = "
                        "RAFT-Large motion-gated keyframe selection (default).")
    p.add_argument("--keyframe_thresh", type=float, default=5.0,
                   help="Mean RAFT flow magnitude threshold for a keyframe "
                        "(only used when --frame_select_method=keyframe).")
    p.add_argument("--keyframe_max_interval", type=int, default=8,
                   help="Force a keyframe after this many frames without one "
                        "(only used when --frame_select_method=keyframe).")
    p.add_argument("--seed", type=int, default=425,
                   help="Case RNG seed (default 425, matching world_gen.py).")
    p.add_argument("--max_chunks", type=int, default=None,
                   help="Cap the AR rollout to this many chunks (debug). "
                        "E.g. 1 generates only the first chunk.")
    args = p.parse_args()
    if args.rank < 0 or args.world_size < 1 or args.rank >= args.world_size:
        p.error("require 0 <= --rank < --world_size")
    if args.max_chunks is not None and args.max_chunks < 1:
        p.error("--max_chunks must be at least 1")
    return args


def _load_instance_prompt(inst, base_dir):
    def _resolve(path):
        if not path:
            return path
        return path if os.path.isabs(path) or base_dir is None else os.path.join(base_dir, path)

    prompt_path = _resolve(inst["prompt_file"])
    with open(prompt_path) as f:
        lines = [line.strip() for line in f if line.strip()]
    prompt = lines if len(lines) > 1 else (lines[0] if lines else "")
    return prompt, os.path.abspath(prompt_path)


def load_instance_inputs(inst, target_height, target_width, base_dir=None):
    """Load one instance's inputs in the single canonical format.

    Relative input paths are resolved against ``base_dir``
    (the directory containing the instances JSON), so a dataset's own
    ``instances.json`` with relative paths works without rewriting. ``resolved``
    is the dict of absolute paths actually used; ``meta`` records the camera
    trajectory length.
    """
    from utils.io import read_depth
    def _resolve(p):
        if not p:
            return p
        return p if os.path.isabs(p) or base_dir is None else os.path.join(base_dir, p)

    image_path = _resolve(inst["image"])
    traj_path = _resolve(inst["trajectory"])
    anchor_path = _resolve(inst["anchor_depth"])
    prompt_path = _resolve(inst["prompt_file"])

    # Image.
    image = Image.open(image_path).convert("RGB").resize(
        (target_width, target_height))

    # Trajectory: .npz with 'extrinsics' (T,4,4) + 'intrinsics' (T,3,3).
    traj = np.load(traj_path)
    extrinsics = torch.from_numpy(np.asarray(traj["extrinsics"], dtype=np.float32))
    intrinsics = torch.from_numpy(np.asarray(traj["intrinsics"], dtype=np.float32))
    if (extrinsics.ndim != 3 or tuple(extrinsics.shape[1:]) != (4, 4)
            or extrinsics.shape[0] < 2
            or tuple(intrinsics.shape) != (extrinsics.shape[0], 3, 3)):
        raise ValueError("Trajectory requires matching (T,4,4)/(T,3,3) cameras with T >= 2")
    if not torch.isfinite(extrinsics).all() or not torch.isfinite(intrinsics).all():
        raise ValueError("Trajectory contains non-finite camera values")

    # Anchor depth: 16-bit log-encoded PNG.
    anchor_depth = torch.from_numpy(read_depth(anchor_path).astype(np.float32))

    prompt, prompt_source = _load_instance_prompt(inst, base_dir)

    resolved = {
        "image": os.path.abspath(image_path),
        "trajectory": os.path.abspath(traj_path),
        "anchor_depth": os.path.abspath(anchor_path),
        "prompt_file": os.path.abspath(prompt_path),
        "prompt_source": prompt_source,
    }
    meta = {
        "trajectory_length": int(extrinsics.shape[0]),
    }
    return image, extrinsics, intrinsics, anchor_depth, prompt, resolved, meta


def _serialize_runtime_config(pipe_cfg) -> dict[str, Any]:
    cfg = asdict(pipe_cfg)
    cfg["torch_dtype"] = str(pipe_cfg.torch_dtype)
    return cfg


def main():
    args = parse_args()
    from octworld.pipelines.wan import WanWorldPipeline, WanWorldPipeConfig

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

    # ── Load instances list ──────────────────────────────────────────
    with open(args.instances) as f:
        instances = json.load(f)
    assert isinstance(instances, list), "instances JSON must be a list of dicts"

    instances_name = os.path.splitext(os.path.basename(args.instances))[0]
    instances_dir = os.path.dirname(os.path.abspath(args.instances))
    base_name = args.exp_name or instances_name
    out_root = os.path.join(args.base_folder, base_name)

    # ── Build pipeline once ──────────────────────────────────────────
    print(f"[Batch] rank={args.rank}/{args.world_size} building WanWorldPipeline...")
    pipe = WanWorldPipeline(
        dit_ckpt_path=args.dit_ckpt,
        model_weights_dir=args.model_weights_dir,
        debug=args.debug,
        config=pipe_cfg,
    )
    # ── Run sharded instances ────────────────────────────────────────
    n_done = n_fail = n_skip = 0
    for i, inst in enumerate(instances):
        # Shard: this rank only handles its slice (global sample_id = i).
        if i % args.world_size != args.rank:
            continue

        save_dir = os.path.join(out_root, str(i))
        os.makedirs(save_dir, exist_ok=True)
        success_tag = os.path.join(save_dir, "SUCCESS.txt")
        fail_tag = os.path.join(save_dir, "FAILED.txt")

        if os.path.exists(success_tag):
            print(f"[Batch] sample {i}: already done → skip")
            n_skip += 1
            continue

        # Clear a stale FAILED.txt from a previous run before retrying.
        if os.path.exists(fail_tag):
            os.remove(fail_tag)

        try:
            (image, extrinsics, intrinsics, anchor_depth,
             prompt, resolved, meta) = load_instance_inputs(
                inst, target_height, target_width, base_dir=instances_dir)

            print(f"[Batch] sample {i}: image={image.size}, "
                  f"traj={tuple(extrinsics.shape)} → {save_dir}")
            pipe(
                image=image,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                anchor_depth=anchor_depth,
                prompt=prompt,
                save_dir=save_dir,
                preview_only=False,
                verbose=True,
                max_chunks=args.max_chunks,
                seed=int(args.seed),
            )

            with open(os.path.join(save_dir, "info.json"), "w") as f:
                json.dump({
                    "sample_id": i,
                    "instances_file": os.path.abspath(args.instances),
                    "inputs": resolved,
                    "prompt": prompt,
                    "trajectory_length": meta["trajectory_length"],
                    "seed": int(args.seed),
                    "pipe_runtime_config": _serialize_runtime_config(pipe_cfg),
                }, f, indent=2, ensure_ascii=False)

            with open(success_tag, "w") as f:
                f.write("ok\n")
            n_done += 1
            print(f"[Batch] sample {i}: SUCCESS")
        except Exception:
            tb = traceback.format_exc()
            with open(fail_tag, "w") as f:
                f.write(tb)
            n_fail += 1
            print(f"[Batch] sample {i}: FAILED\n{tb}")

    print(f"[Batch] rank={args.rank}/{args.world_size} done. "
          f"success={n_done} failed={n_fail} skipped={n_skip}")


if __name__ == "__main__":
    main()
