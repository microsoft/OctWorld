"""Merge validated wan_standard datasets into one clean dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from validate_wan_standard import INSTANCE_KEYS, validate_wan_standard


def merge_wan_standard(
    inputs: list[Path], output: Path, *, overwrite: bool = False
) -> dict[str, int]:
    """Validate, renumber, and atomically publish multiple datasets."""

    if not inputs:
        raise ValueError("at least one input dataset is required")
    inputs = [path.expanduser().resolve() for path in inputs]
    output = output.expanduser().resolve()
    staging = output.with_name(f".{output.name}.staging")
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output}; pass --overwrite")
    if staging.exists():
        shutil.rmtree(staging)
    scenes_root = staging / "scenes"
    scenes_root.mkdir(parents=True)

    merged_instances = []
    try:
        for source_root in inputs:
            validate_wan_standard(source_root)
            source_instances = json.loads(
                (source_root / "instances.json").read_text(encoding="utf-8")
            )
            for source_instance in source_instances:
                scene_name = f"scene_{len(merged_instances):05d}"
                destination = scenes_root / scene_name
                source_paths = {
                    key: source_root / source_instance[key] for key in INSTANCE_KEYS
                }
                source_scene = source_paths["image"].parent
                shutil.copytree(source_scene, destination, copy_function=shutil.copy2)
                prefix = f"scenes/{scene_name}"
                merged_instances.append(
                    {
                        "image": f"{prefix}/image.png",
                        "trajectory": f"{prefix}/trajectory.npz",
                        "anchor_depth": f"{prefix}/anchor_depth.png",
                        "prompt_file": f"{prefix}/prompt.txt",
                    }
                )

        (staging / "instances.json").write_text(
            json.dumps(merged_instances, indent=2) + "\n", encoding="utf-8"
        )
        summary = validate_wan_standard(staging)
        if output.exists():
            shutil.rmtree(output)
        staging.replace(output)
        return summary
    except BaseException:
        print(f"Merge failed; temporary staging retained at {staging}")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = merge_wan_standard(args.input, args.output, overwrite=args.overwrite)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()