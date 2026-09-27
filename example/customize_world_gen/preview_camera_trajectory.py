"""Render one camera trajectory preview from an initial image and action file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera_trajectory import CameraTrajectoryGenerator, parse_camera_movement  # noqa: E402
from geometry import load_image, predict_moge_v1  # noqa: E402
from spatial_memory.mapper import build_mapper  # noqa: E402
from utils.io import post_process_video_for_visualize, write_video  # noqa: E402


PREVIEW_FRAMES_PER_TURN = 16


def render_preview(
    image_path: Path,
    movement_path: Path,
    output_path: Path,
    *,
    resolution: tuple[int, int],
    fps: int,
    camera_speed: float,
    smooth_camera: bool,
    device: str,
) -> None:
    commands = parse_camera_movement(movement_path)
    image = load_image(image_path, resolution)
    geometry = predict_moge_v1(image, device=device)
    extrinsics = CameraTrajectoryGenerator(
        camera_speed=camera_speed,
        frames_per_turn=PREVIEW_FRAMES_PER_TURN,
        smooth=smooth_camera,
    ).generate(commands)
    intrinsics = np.broadcast_to(
        geometry.intrinsics, (extrinsics.shape[0], 3, 3)
    ).copy()

    mapper = build_mapper("pointcloud", resolution=resolution, device=device)
    frames = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0).to(device)
    depths = torch.from_numpy(geometry.depth).unsqueeze(0).to(device)
    masks = torch.from_numpy(geometry.depth_mask).unsqueeze(0).to(device)
    first_extrinsic = torch.from_numpy(extrinsics[:1]).to(device)
    first_intrinsic = torch.from_numpy(intrinsics[:1]).to(device)
    mapper.extend_frames(
        frames,
        depths,
        first_extrinsic,
        first_intrinsic,
        depth_mask=masks,
    )
    render = mapper.render(
        torch.from_numpy(extrinsics).to(device),
        torch.from_numpy(intrinsics).to(device),
    )
    video = post_process_video_for_visualize(
        render["color"], layout="fchw", value_range=(0.0, 1.0)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_video(output_path, video, fps=fps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--camera-movement", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("camera_trajectory_preview.mp4"),
    )
    parser.add_argument("--resolution", nargs=2, type=int, default=(240, 416))
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--camera-speed", type=float, default=1.0)
    parser.add_argument(
        "--smooth-camera",
        action="store_true",
        help="Use C2 spline interpolation between hard camera-turn keyframes",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"image not found: {args.image}")
    if not args.camera_movement.is_file():
        raise FileNotFoundError(
            f"camera movement not found: {args.camera_movement}"
        )
    render_preview(
        args.image.resolve(),
        args.camera_movement.resolve(),
        args.output.expanduser().resolve(),
        resolution=tuple(args.resolution),
        fps=args.fps,
        camera_speed=args.camera_speed,
        smooth_camera=args.smooth_camera,
        device=args.device,
    )
    print(f"Wrote camera trajectory preview: {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()