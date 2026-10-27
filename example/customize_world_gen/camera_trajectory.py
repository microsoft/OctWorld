"""Convert text camera actions into OctWorld OpenCV trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline, Slerp


ACTION_SPECS = {
    "move_right": (1, (1.0,)),
    "move_left": (1, (1.0,)),
    "push_in": (1, (1.0,)),
    "pull_out": (1, (1.0,)),
    "pan_right": (1, (30.0,)),
    "pan_left": (1, (30.0,)),
    "orbit_right": (2, (1.0, 30.0)),
    "orbit_left": (2, (1.0, 30.0)),
    "pull_right": (2, (1.0, 30.0)),
    "pull_left": (2, (1.0, 30.0)),
    "push_right": (2, (1.0, 30.0)),
    "push_left": (2, (1.0, 30.0)),
}

Command = tuple[str, tuple[float, ...]]


@dataclass
class _Camera:
    rt: np.ndarray

    @classmethod
    def identity(cls) -> "_Camera":
        return cls(np.eye(4, dtype=np.float64))

    def copy(self) -> "_Camera":
        return _Camera(self.rt.copy())


def _rotation_y(angle_degrees: float) -> np.ndarray:
    theta = np.deg2rad(angle_degrees)
    cosine, sine = np.cos(theta), np.sin(theta)
    return np.array(
        [[cosine, 0.0, -sine], [0.0, 1.0, 0.0], [sine, 0.0, cosine]],
        dtype=np.float64,
    )


def resolve_commands(commands: Sequence[Command]) -> list[Command]:
    """Validate camera actions and fill omitted arguments with defaults."""

    resolved = []
    for action, raw_args in commands:
        if action not in ACTION_SPECS:
            supported = ", ".join(ACTION_SPECS)
            raise ValueError(f"unknown camera action {action!r}; supported: {supported}")
        expected_count, defaults = ACTION_SPECS[action]
        args = tuple(float(value) for value in raw_args) if raw_args else defaults
        if len(args) != expected_count:
            raise ValueError(
                f"{action!r} expects {expected_count} argument(s), got {len(args)}"
            )
        if expected_count and args[0] <= 0:
            raise ValueError(f"{action!r} magnitude/angle must be positive")
        if expected_count == 2 and args[1] <= 0:
            raise ValueError(f"{action!r} angle must be positive")
        resolved.append((action, args))
    if not resolved:
        raise ValueError("at least one camera command is required")
    return resolved


def parse_camera_movement(path: str | Path) -> list[Command]:
    """Read ``<action> [arg ...]`` commands with comments and blank lines."""

    commands: list[Command] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        content = line.partition("#")[0].strip()
        if not content:
            continue
        parts = content.split()
        try:
            args = tuple(float(value) for value in parts[1:])
        except ValueError as error:
            raise ValueError(
                f"invalid number on line {line_number}: {line!r}"
            ) from error
        commands.append((parts[0], args))
    return resolve_commands(commands)


def smooth_camera_keyframes(
    key_extrinsics: np.ndarray, frames_per_turn: int
) -> np.ndarray:
    """Interpolate OpenCV world-to-camera keyframes with C2 splines."""

    key_extrinsics = np.asarray(key_extrinsics)
    if key_extrinsics.ndim != 3 or key_extrinsics.shape[1:] != (4, 4):
        raise ValueError(
            f"key_extrinsics must have shape (N, 4, 4), got {key_extrinsics.shape}"
        )
    if key_extrinsics.shape[0] < 2:
        raise ValueError("at least two camera keyframes are required")
    if frames_per_turn < 1:
        raise ValueError("frames_per_turn must be at least 1")

    key_cameras_to_world = np.linalg.inv(key_extrinsics)
    key_times = np.arange(len(key_extrinsics), dtype=np.float64)
    sample_times = np.arange(
        (len(key_extrinsics) - 1) * frames_per_turn + 1,
        dtype=np.float64,
    ) / frames_per_turn
    center_spline = CubicSpline(
        key_times,
        key_cameras_to_world[:, :3, 3],
        axis=0,
        bc_type="natural",
    )
    rotation_spline = RotationSpline(
        key_times,
        Rotation.from_matrix(key_cameras_to_world[:, :3, :3]),
    )

    cameras_to_world = np.broadcast_to(
        np.eye(4, dtype=np.float64), (len(sample_times), 4, 4)
    ).copy()
    cameras_to_world[:, :3, :3] = rotation_spline(sample_times).as_matrix()
    cameras_to_world[:, :3, 3] = center_spline(sample_times)
    return np.linalg.inv(cameras_to_world).astype(np.float32)


class CameraTrajectoryGenerator:
    """Generate normalized OpenCV world-to-camera poses from camera actions."""

    def __init__(
        self,
        camera_speed: float = 1.0,
        frames_per_turn: int = 80,
        smooth: bool = False,
    ):
        if camera_speed <= 0:
            raise ValueError("camera_speed must be positive")
        if frames_per_turn < 1:
            raise ValueError("frames_per_turn must be at least 1")
        self.camera_speed = float(camera_speed)
        self.frames_per_turn = int(frames_per_turn)
        self.smooth = bool(smooth)

    def generate(self, commands: Sequence[Command]) -> np.ndarray:
        resolved = resolve_commands(commands)
        keyframes = [_Camera.identity()]
        for action, args in resolved:
            rebase = np.linalg.inv(keyframes[-1].rt)
            keyframes = [_Camera(camera.rt @ rebase) for camera in keyframes]
            keyframes.append(self._apply(action, args, keyframes[-1]))

        first_inverse = np.linalg.inv(keyframes[0].rt)
        keyframes = [_Camera(camera.rt @ first_inverse) for camera in keyframes]

        basis = np.diag([-1.0, -1.0, 1.0, 1.0])
        if self.smooth:
            key_extrinsics = np.stack(
                [basis @ camera.rt @ basis for camera in keyframes]
            )
            key_extrinsics = key_extrinsics @ np.linalg.inv(key_extrinsics[0])
            return smooth_camera_keyframes(key_extrinsics, self.frames_per_turn)

        interpolated = self._interpolate_keyframes(keyframes)
        extrinsics = np.stack([basis @ camera.rt @ basis for camera in interpolated])
        extrinsics = extrinsics @ np.linalg.inv(extrinsics[0])
        return extrinsics.astype(np.float32)

    def _apply(self, action: str, args: tuple[float, ...], camera: _Camera) -> _Camera:
        if action == "move_right":
            return self._move_right(args[0], camera)
        if action == "move_left":
            return self._move_right(-args[0], camera)
        if action == "push_in":
            return self._push_in(args[0], camera)
        if action == "pull_out":
            return self._push_in(-args[0], camera)
        if action == "pan_right":
            return self._pan_right(args[0], camera)
        if action == "pan_left":
            return self._pan_right(-args[0], camera)
        if action == "orbit_right":
            return self._orbit_right(args[0], args[1], camera)
        if action == "orbit_left":
            return self._orbit_right(args[0], -args[1], camera)
        if action == "pull_right":
            return self._pull_right(args[0], args[1], camera)
        if action == "pull_left":
            return self._pull_right(-args[0], -args[1], camera)
        if action == "push_right":
            return self._push_right(args[0], args[1], camera)
        if action == "push_left":
            return self._push_right(-args[0], -args[1], camera)
        raise AssertionError(f"unhandled action: {action}")

    def _move_right(self, magnitude: float, camera: _Camera) -> _Camera:
        result = camera.copy()
        result.rt[0, 3] += self.camera_speed * magnitude
        return result

    def _push_in(self, magnitude: float, camera: _Camera) -> _Camera:
        result = camera.copy()
        result.rt[2, 3] -= self.camera_speed * magnitude
        return result

    @staticmethod
    def _pan_right(angle_degrees: float, camera: _Camera) -> _Camera:
        result = camera.copy()
        result.rt[:3, :3] = result.rt[:3, :3] @ _rotation_y(-angle_degrees)
        return result

    def _orbit_right(
        self, magnitude: float, angle_degrees: float, camera: _Camera
    ) -> _Camera:
        radius = self.camera_speed * magnitude
        z_translation = camera.rt[2, 3]
        negative = camera.copy()
        positive = camera.copy()
        negative.rt[2, 3] = radius - z_translation
        positive.rt[2, 3] = z_translation - radius
        rotation = np.eye(4, dtype=np.float64)
        rotation[:3, :3] = _rotation_y(angle_degrees)
        return _Camera(negative.rt @ rotation @ positive.rt)

    def _pull_right(
        self, magnitude: float, angle_degrees: float, camera: _Camera
    ) -> _Camera:
        result = self._move_right(magnitude, camera)
        result = self._push_in(-magnitude, result)
        return self._pan_right(angle_degrees, result)

    def _push_right(
        self, magnitude: float, angle_degrees: float, camera: _Camera
    ) -> _Camera:
        result = self._move_right(magnitude, camera)
        result = self._push_in(magnitude, result)
        return self._pan_right(angle_degrees, result)

    def _interpolate_keyframes(self, keyframes: Sequence[_Camera]) -> list[_Camera]:
        cameras = [keyframes[0].copy()]
        sample_times = np.arange(1, self.frames_per_turn) / self.frames_per_turn
        for start, end in zip(keyframes[:-1], keyframes[1:]):
            if sample_times.size:
                rotations = Rotation.from_matrix(
                    np.stack([start.rt[:3, :3], end.rt[:3, :3]])
                )
                interpolator = Slerp([0.0, 1.0], rotations)
                interpolated_rotations = interpolator(sample_times).as_matrix()
                for time, rotation in zip(sample_times, interpolated_rotations):
                    translation = start.rt[:3, 3] + time * (
                        end.rt[:3, 3] - start.rt[:3, 3]
                    )
                    rt = np.eye(4, dtype=np.float64)
                    rt[:3, :3] = rotation
                    rt[:3, 3] = translation
                    cameras.append(_Camera(rt))
            cameras.append(end.copy())
        return cameras


def save_camera_trajectory(
    path: str | Path, extrinsics: np.ndarray, intrinsics: np.ndarray
) -> None:
    """Save a validated trajectory in the canonical NPZ format."""

    extrinsics = np.asarray(extrinsics, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    if intrinsics.shape == (3, 3):
        intrinsics = np.broadcast_to(
            intrinsics, (extrinsics.shape[0], 3, 3)
        ).copy()
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (4, 4):
        raise ValueError(f"extrinsics must have shape (T, 4, 4), got {extrinsics.shape}")
    if intrinsics.shape != (extrinsics.shape[0], 3, 3):
        raise ValueError(
            f"intrinsics must have shape ({extrinsics.shape[0]}, 3, 3), "
            f"got {intrinsics.shape}"
        )
    if not np.isfinite(extrinsics).all() or not np.isfinite(intrinsics).all():
        raise ValueError("trajectory contains non-finite values")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, extrinsics=extrinsics, intrinsics=intrinsics)