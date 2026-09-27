"""Build one clean wan_standard case from an approved camera trajectory."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from camera_trajectory import (
    CameraTrajectoryGenerator,
    parse_camera_movement,
    save_camera_trajectory,
)
from generate_text_prompts import generate_prompt_file
from geometry import load_image, predict_moge_v1
from validate_wan_standard import validate_wan_standard


FINAL_FRAMES_PER_TURN = 80
FINAL_RESOLUTION = (480, 832)


def _instance() -> dict[str, str]:
    prefix = "scenes/scene_00000"
    return {
        "image": f"{prefix}/image.png",
        "trajectory": f"{prefix}/trajectory.npz",
        "anchor_depth": f"{prefix}/anchor_depth.png",
        "prompt_file": f"{prefix}/prompt.txt",
    }


def build(args: argparse.Namespace) -> dict[str, int]:
    output = args.output.expanduser().resolve()
    staging = output.with_name(f".{output.name}.staging")
    if args.preview is not None and not args.preview.expanduser().is_file():
        raise FileNotFoundError(f"preview not found: {args.preview}")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {output}; pass --overwrite")

    commands = parse_camera_movement(args.camera_movement)
    if staging.exists():
        shutil.rmtree(staging)
    staging_scene = staging / "scenes" / "scene_00000"
    staging_scene.mkdir(parents=True)

    try:
        image = load_image(args.image, FINAL_RESOLUTION)
        Image.fromarray(image).save(staging_scene / "image.png")

        prompts = generate_prompt_file(
            image_path=staging_scene / "image.png",
            movement_path=args.camera_movement.resolve(),
            output_path=staging_scene / "prompt.txt",
            dl3dv_source=args.dl3dv_source.expanduser().resolve(),
            model_dir=args.model_dir,
            device=args.device,
            dtype=args.dtype,
            attn_impl=args.attn_impl,
            max_new_tokens=args.max_new_tokens,
            max_attempts=args.max_attempts,
        )
        if len(prompts) != len(commands):
            raise AssertionError("prompt count changed after prompt generation")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        geometry = predict_moge_v1(image, device=args.device)
        extrinsics = CameraTrajectoryGenerator(
            camera_speed=args.camera_speed,
            frames_per_turn=FINAL_FRAMES_PER_TURN,
            smooth=args.smooth_camera,
        ).generate(commands)
        expected_frames = 1 + FINAL_FRAMES_PER_TURN * len(commands)
        if extrinsics.shape != (expected_frames, 4, 4):
            raise AssertionError(
                f"trajectory has {extrinsics.shape[0]} frames, expected {expected_frames}"
            )
        save_camera_trajectory(
            staging_scene / "trajectory.npz", extrinsics, geometry.intrinsics
        )

        from utils.io import write_depth

        write_depth(
            staging_scene / "anchor_depth.png",
            geometry.depth,
            mask=geometry.depth_mask,
        )
        (staging / "instances.json").write_text(
            json.dumps([_instance()], indent=2) + "\n", encoding="utf-8"
        )
        summary = validate_wan_standard(staging, require_identity=True)
        if output.exists():
            shutil.rmtree(output)
        staging.replace(output)
        return summary
    except BaseException:
        print(f"Build failed; temporary staging retained at {staging}")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--camera-movement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--preview",
        type=Path,
        help="Optional reviewed camera_trajectory_preview.mp4",
    )
    parser.add_argument("--dl3dv-source", type=Path, required=True)
    parser.add_argument(
        "--model-dir", default="model_weights/Qwen3-VL-8B-Instruct"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument(
        "--attn-impl",
        choices=("flash_attention_2", "sdpa", "eager"),
        default="flash_attention_2",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--camera-speed", type=float, default=1.0)
    parser.add_argument(
        "--smooth-camera",
        action="store_true",
        help="Use C2 spline interpolation between hard camera-turn keyframes",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"image not found: {args.image}")
    if not args.camera_movement.is_file():
        raise FileNotFoundError(
            f"camera movement not found: {args.camera_movement}"
        )
    summary = build(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()