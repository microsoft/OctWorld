from pathlib import Path
import unittest

import numpy as np

from camera_trajectory import CameraTrajectoryGenerator, parse_camera_movement


HERE = Path(__file__).resolve().parent


class CameraTrajectoryTest(unittest.TestCase):
    def test_smooth_trajectory_preserves_hard_turn_keyframes(self) -> None:
        commands = parse_camera_movement(HERE / "trajectories/trajectory_1.txt")
        linear = CameraTrajectoryGenerator(frames_per_turn=80).generate(commands)
        smooth = CameraTrajectoryGenerator(
            frames_per_turn=80, smooth=True
        ).generate(commands)

        self.assertEqual(linear.shape, (641, 4, 4))
        self.assertEqual(smooth.shape, linear.shape)
        np.testing.assert_allclose(smooth[::80], linear[::80], atol=1e-5)
        np.testing.assert_allclose(smooth[0], np.eye(4), atol=1e-6)
        self.assertTrue(np.isfinite(smooth).all())


if __name__ == "__main__":
    unittest.main()