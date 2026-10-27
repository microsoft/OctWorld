"""CPU regressions for standard dataset entry points and bundled inputs."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import build_example_dataset
import build_standard
from merge_standard import merge_standard
from prompt_policy import load_dl3dv_prompt_examples
from validate_standard import validate_standard


DATASET = Path(__file__).resolve().parents[2] / "assets" / "standard"


class StandardDatasetTest(unittest.TestCase):
    def test_bundled_dataset_validates(self) -> None:
        self.assertEqual(validate_standard(DATASET), {"cases": 3, "camera_turns": 12})
        self.assertEqual(len(load_dl3dv_prompt_examples(DATASET)), 3)

    def test_bundled_builder_defaults(self) -> None:
        with patch.object(sys, "argv", ["build_example_dataset.py"]):
            args = build_example_dataset.parse_args()
        self.assertEqual(args.source_dataset, DATASET)
        self.assertEqual(args.output, DATASET)

    def test_single_builder_uses_standard_validator(self) -> None:
        self.assertIs(build_standard.validate_standard, validate_standard)
        with patch.object(
            sys,
            "argv",
            [
                "build_standard.py",
                "--image", "image.png",
                "--camera-movement", "camera_turns.txt",
                "--dl3dv-source", str(DATASET),
                "--output", "outputs/my_case/standard",
            ],
        ):
            args = build_standard.parse_args()
        self.assertEqual(args.dl3dv_source, DATASET)
        self.assertEqual(args.output, Path("outputs/my_case/standard"))

    def test_merge_standard_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "standard"
            summary = merge_standard([DATASET, DATASET], output)
            self.assertEqual(summary, {"cases": 6, "camera_turns": 24})
            self.assertEqual(validate_standard(output), summary)
            self.assertFalse(output.with_name(".standard.staging").exists())
            instances = json.loads((output / "instances.json").read_text())
            self.assertEqual(
                instances[-1]["image"], "scenes/scene_00005/image.png"
            )
        self.assertEqual(validate_standard(DATASET), {"cases": 3, "camera_turns": 12})


if __name__ == "__main__":
    unittest.main()