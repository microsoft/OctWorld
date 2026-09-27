"""Build the bundled three-case wan_standard dataset in one pass."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera_trajectory import CameraTrajectoryGenerator, parse_camera_movement
from geometry import load_image
from prompt_policy import Qwen3VLPromptPlanner, load_dl3dv_prompt_examples
from utils.io import read_depth, write_depth


FRAMES_PER_TURN = 80
RESOLUTION = (480, 832)
SCENE_FILES = {"image.png", "trajectory.npz", "anchor_depth.png", "prompt.txt"}
CUSTOM_CASES = (
    ("scene_00001", "DL3DVHotelRoom"),
    ("scene_00002", "MinecraftCanyon"),
)


def _instance(scene_name: str) -> dict[str, str]:
    prefix = f"scenes/{scene_name}"
    return {
        "image": f"{prefix}/image.png",
        "trajectory": f"{prefix}/trajectory.npz",
        "anchor_depth": f"{prefix}/anchor_depth.png",
        "prompt_file": f"{prefix}/prompt.txt",
    }


def _read_prompts(path: Path, expected_count: int) -> list[str]:
    prompts = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(prompts) != expected_count:
        raise ValueError(
            f"expected {expected_count} prompts in {path}, got {len(prompts)}"
        )
    return prompts


def _validate_dataset(root: Path) -> dict[str, int]:
    instances = json.loads((root / "instances.json").read_text())
    if len(instances) != 3:
        raise ValueError(f"expected 3 instances, got {len(instances)}")
    for index, instance in enumerate(instances):
        paths = {key: root / value for key, value in instance.items()}
        scene_dir = paths["image"].parent
        if {path.name for path in scene_dir.iterdir()} != SCENE_FILES:
            raise ValueError(f"scene {index} does not contain exactly four files")
        if not all(path.is_file() for path in paths.values()):
            raise FileNotFoundError(f"scene {index} has missing files")
        with Image.open(paths["image"]) as image:
            if image.size != (832, 480):
                raise ValueError(f"scene {index} image size is {image.size}")
        with np.load(paths["trajectory"]) as trajectory:
            extrinsics = np.asarray(trajectory["extrinsics"])
            intrinsics = np.asarray(trajectory["intrinsics"])
        if extrinsics.shape != (321, 4, 4) or intrinsics.shape != (321, 3, 3):
            raise ValueError(f"scene {index} has invalid camera shapes")
        if not np.isfinite(extrinsics).all() or not np.isfinite(intrinsics).all():
            raise ValueError(f"scene {index} has non-finite cameras")
        depth = read_depth(paths["anchor_depth"])
        if depth.ndim != 2 or int((np.isfinite(depth) & (depth > 0)).sum()) < 100:
            raise ValueError(f"scene {index} has invalid anchor depth")
        _read_prompts(paths["prompt_file"], 4)
    return {"cases": 3, "camera_turns": 12}


def build(args: argparse.Namespace) -> dict[str, int]:
    raw_root = args.raw_root.expanduser().resolve()
    source_dataset = args.source_dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    staging = output.with_name(f".{output.name}.staging")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output}; pass --overwrite")
    if staging.exists():
        shutil.rmtree(staging)
    scenes_root = staging / "scenes"
    scenes_root.mkdir(parents=True)

    source_gt = source_dataset / "scenes" / "scene_00000"
    if {path.name for path in source_gt.iterdir()} != SCENE_FILES:
        raise ValueError(f"invalid source GT scene: {source_gt}")
    shutil.copytree(source_gt, scenes_root / "scene_00000")

    case_inputs = []
    for scene_name, raw_name in CUSTOM_CASES:
        case_root = raw_root / raw_name
        if {path.name for path in case_root.iterdir()} != {
            "image.png",
            "camera_turns.txt",
        }:
            raise ValueError(f"raw case must contain only image and turns: {case_root}")
        commands = parse_camera_movement(case_root / "camera_turns.txt")
        if len(commands) != 4:
            raise ValueError(f"bundled case must contain exactly four turns: {case_root}")
        image = load_image(case_root / "image.png", RESOLUTION)
        case_inputs.append((scene_name, case_root, commands, image))

    if args.reuse_prompts_from is not None:
        prompt_source = args.reuse_prompts_from.expanduser().resolve()
        prompt_sets = {
            scene_name: _read_prompts(
                prompt_source / "scenes" / scene_name / "prompt.txt", len(commands)
            )
            for scene_name, _, commands, _ in case_inputs
        }
    else:
        style_examples = load_dl3dv_prompt_examples(source_dataset)
        planner = Qwen3VLPromptPlanner(
            args.model_dir,
            device=args.device,
            dtype=args.dtype,
            attn_impl=args.attn_impl,
            max_new_tokens=args.max_new_tokens,
        )
        prompt_sets = {}
        for scene_name, _, commands, image in case_inputs:
            prompt_sets[scene_name] = planner.generate(
                Image.fromarray(image),
                commands,
                style_examples,
                max_attempts=args.max_attempts,
            )
        del planner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    from depth_prediction.predictors import create_predictor

    predictor = create_predictor(
        "moge",
        device=args.device,
        version="v1",
        pretrained=args.moge_model,
        resolution_level=9,
        apply_mask=True,
    )
    for scene_name, _, commands, image in case_inputs:
        destination = scenes_root / scene_name
        destination.mkdir()
        Image.fromarray(image).save(destination / "image.png")
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1)
        geometry = predictor.predict(
            tensor.float().div_(255.0).unsqueeze(0).to(args.device)
        )
        depth = geometry.depth[0].detach().cpu().numpy().astype(np.float32)
        depth_mask = geometry.depth_mask[0].detach().cpu().numpy().astype(bool)
        intrinsic = geometry.intrinsics[0].detach().cpu().numpy().astype(np.float32)
        write_depth(destination / "anchor_depth.png", depth, mask=depth_mask)

        extrinsics = CameraTrajectoryGenerator(
            camera_speed=args.camera_speed,
            frames_per_turn=FRAMES_PER_TURN,
            smooth=args.smooth_camera,
        ).generate(commands)
        intrinsics = np.broadcast_to(intrinsic, (len(extrinsics), 3, 3)).copy()
        np.savez_compressed(
            destination / "trajectory.npz",
            extrinsics=extrinsics.astype(np.float32),
            intrinsics=intrinsics.astype(np.float32),
        )
        (destination / "prompt.txt").write_text(
            "\n".join(prompt_sets[scene_name]) + "\n", encoding="utf-8"
        )

    instances = [_instance(f"scene_{index:05d}") for index in range(3)]
    (staging / "instances.json").write_text(
        json.dumps(instances, indent=2) + "\n", encoding="utf-8"
    )
    summary = _validate_dataset(staging)
    if output.exists():
        shutil.rmtree(output)
    staging.replace(output)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "assets/raw")
    parser.add_argument(
        "--source-dataset",
        type=Path,
        default=PROJECT_ROOT / "assets/wan_standard",
        help="Dataset providing the unchanged DL3DV GT case and prompt examples",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "assets/wan_standard"
    )
    parser.add_argument("--model-dir", default="model_weights/Qwen3-VL-8B-Instruct")
    parser.add_argument("--moge-model", default="Ruicheng/moge-vitl")
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
    parser.add_argument("--smooth-camera", action="store_true")
    parser.add_argument(
        "--reuse-prompts-from",
        type=Path,
        help="Validation-only: reuse existing prompts instead of loading Qwen",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = build(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()