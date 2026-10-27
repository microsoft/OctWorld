from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import batch_world_gen
import world_gen


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class BatchWorldGenTest(unittest.TestCase):
    def test_defaults_match_single_case_driver(self) -> None:
        with patch.object(sys, "argv", ["world_gen.py"]):
            single = world_gen.parse_args()
        with patch.object(
            sys,
            "argv",
            ["batch_world_gen.py", "--base_folder", "/tmp/batch-output"],
        ):
            batch = batch_world_gen.parse_args()

        shared = (
            "dit_ckpt",
            "model_weights_dir",
            "debug",
            "depth_method",
            "offline_integration",
            "octree_depth_strategy",
            "bigger_voxel_pixel_size_scale",
            "sample_ratio",
            "frame_select_method",
            "keyframe_thresh",
            "keyframe_max_interval",
            "seed",
            "max_chunks",
        )
        for name in shared:
            self.assertEqual(getattr(batch, name), getattr(single, name), name)

        dataset = PROJECT_ROOT / "assets" / "standard"
        scene = dataset / "scenes" / "scene_00000"
        self.assertEqual(Path(batch.instances), dataset / "instances.json")
        self.assertEqual(Path(single.image), scene / "image.png")
        self.assertEqual(Path(single.trajectory), scene / "trajectory.npz")
        self.assertEqual(Path(single.anchor_depth), scene / "anchor_depth.png")
        self.assertEqual(Path(world_gen.DEFAULT_PROMPT_FILE), scene / "prompt.txt")
        self.assertEqual(Path(single.save_dir), Path("outputs/dl3dv_gt"))

    def test_bundled_dataset_loads(self) -> None:
        dataset = PROJECT_ROOT / "assets" / "standard"
        instances = json.loads((dataset / "instances.json").read_text())
        self.assertEqual(len(instances), 3)
        for instance in instances:
            inputs = batch_world_gen.load_instance_inputs(
                instance, 480, 832, base_dir=str(dataset)
            )
            image, extrinsics, intrinsics, _, prompts, _, meta = inputs
            self.assertEqual(image.size, (832, 480))
            self.assertEqual(tuple(extrinsics.shape), (321, 4, 4))
            self.assertEqual(tuple(intrinsics.shape), (321, 3, 3))
            self.assertEqual(len(prompts), 4)
            self.assertEqual(meta["trajectory_length"], 321)


if __name__ == "__main__":
    unittest.main()