"""Validate a strict customize-world-generation wan_standard dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.io import read_depth  # noqa: E402


FRAMES_PER_TURN = 80
ROOT_ENTRIES = {"instances.json", "scenes"}
SCENE_FILES = {"image.png", "trajectory.npz", "anchor_depth.png", "prompt.txt"}
INSTANCE_KEYS = {"image", "trajectory", "anchor_depth", "prompt_file"}


def _resolve_inside(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"instance path escapes dataset root: {relative_path}") from error
    return path


def validate_wan_standard(
    root: str | Path, *, require_identity: bool = False
) -> dict[str, int]:
    """Validate files, geometry, prompt counts, and clean output structure."""

    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"wan_standard root not found: {root}")
    root_entries = {path.name for path in root.iterdir()}
    if root_entries != ROOT_ENTRIES:
        raise ValueError(
            f"wan_standard root must contain only {sorted(ROOT_ENTRIES)}, "
            f"got {sorted(root_entries)}"
        )

    instances = json.loads((root / "instances.json").read_text(encoding="utf-8"))
    if not isinstance(instances, list) or not instances:
        raise ValueError("instances.json must contain a non-empty list")

    referenced_scene_dirs = set()
    total_turns = 0
    for index, instance in enumerate(instances):
        if not isinstance(instance, dict) or set(instance) != INSTANCE_KEYS:
            keys = sorted(instance) if isinstance(instance, dict) else type(instance).__name__
            raise ValueError(f"instance {index} has unexpected keys/type: {keys}")
        paths = {
            key: _resolve_inside(root, value) for key, value in instance.items()
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"instance {index} has missing files: {missing}")

        scene_dir = paths["image"].parent
        if any(path.parent != scene_dir for path in paths.values()):
            raise ValueError(f"instance {index} files must share one scene directory")
        scene_files = {path.name for path in scene_dir.iterdir()}
        if scene_files != SCENE_FILES:
            raise ValueError(
                f"instance {index} scene must contain only {sorted(SCENE_FILES)}, "
                f"got {sorted(scene_files)}"
            )
        referenced_scene_dirs.add(scene_dir)

        with Image.open(paths["image"]) as image:
            if image.size != (832, 480):
                raise ValueError(
                    f"instance {index} image must be 832x480, got {image.size}"
                )
            image.convert("RGB")

        with np.load(paths["trajectory"]) as trajectory:
            if set(trajectory.files) != {"extrinsics", "intrinsics"}:
                raise ValueError(
                    f"instance {index} trajectory must contain extrinsics/intrinsics"
                )
            extrinsics = np.asarray(trajectory["extrinsics"])
            intrinsics = np.asarray(trajectory["intrinsics"])
        if extrinsics.ndim != 3 or extrinsics.shape[1:] != (4, 4):
            raise ValueError(f"instance {index} has invalid extrinsics shape")
        if intrinsics.shape != (extrinsics.shape[0], 3, 3):
            raise ValueError(f"instance {index} has invalid intrinsics shape")
        if not np.isfinite(extrinsics).all() or not np.isfinite(intrinsics).all():
            raise ValueError(f"instance {index} trajectory contains non-finite values")
        if require_identity and not np.allclose(extrinsics[0], np.eye(4), atol=1e-5):
            raise ValueError(f"instance {index} frame-zero extrinsic is not identity")
        if (extrinsics.shape[0] - 1) % FRAMES_PER_TURN:
            raise ValueError(f"instance {index} trajectory is not 1+80N frames")

        prompts = [
            line.strip()
            for line in paths["prompt_file"].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        expected_turns = (extrinsics.shape[0] - 1) // FRAMES_PER_TURN
        if expected_turns < 1:
            raise ValueError(f"instance {index} must contain at least one camera turn")
        if len(prompts) != expected_turns:
            raise ValueError(
                f"instance {index} has {len(prompts)} prompts for "
                f"{expected_turns} camera turns"
            )

        depth = read_depth(paths["anchor_depth"])
        if depth.ndim != 2:
            raise ValueError(f"instance {index} depth must be 2D, got {depth.shape}")
        valid_depth = np.isfinite(depth) & (depth > 0)
        if int(valid_depth.sum()) < 100:
            raise ValueError(f"instance {index} has insufficient valid anchor depth")
        total_turns += expected_turns

    scenes_root = root / "scenes"
    actual_scene_dirs = {path.resolve() for path in scenes_root.iterdir() if path.is_dir()}
    non_directories = [path.name for path in scenes_root.iterdir() if not path.is_dir()]
    if non_directories:
        raise ValueError(f"scenes root contains non-directories: {non_directories}")
    if actual_scene_dirs != referenced_scene_dirs:
        raise ValueError("scenes root and instances.json references do not match")
    return {"cases": len(instances), "camera_turns": total_turns}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--require-identity",
        action="store_true",
        help="Require every trajectory to be rebased to an identity first pose",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_wan_standard(
        args.root, require_identity=args.require_identity
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()