from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from MotionMatching.MotionMatchingConfig import (
    MM_ACTION_LABELS,
    MM_FOOT_POSITION_JOINTS,
    MM_FUTURE_SAMPLE_OFFSETS,
    MM_INITIAL_FEATURE_GROUP_WEIGHTS,
    MM_VELOCITY_JOINTS,
)
from genoview.utils import quat


@dataclass(frozen=True)
class FeatureLayout:
    group_names: tuple[str, ...]
    group_starts: tuple[int, ...]
    group_stops: tuple[int, ...]
    group_weights: tuple[float, ...]

    @property
    def feature_dim(self) -> int:
        return int(self.group_stops[-1]) if self.group_stops else 0

    def group_slice(self, group_name: str) -> slice:
        if group_name not in self.group_names:
            raise KeyError(f"Unknown feature group: {group_name}")
        index = self.group_names.index(group_name)
        return slice(self.group_starts[index], self.group_stops[index])

    def weight_vector(self) -> np.ndarray:
        weights = np.ones(self.feature_dim, dtype=np.float32)
        for name, start, stop, weight in zip(
            self.group_names,
            self.group_starts,
            self.group_stops,
            self.group_weights,
        ):
            weights[int(start):int(stop)] = float(weight)
        return weights.astype(np.float32)

    def to_npz_metadata(self) -> dict:
        return {
            "feature_group_names": np.asarray(self.group_names),
            "feature_group_starts": np.asarray(self.group_starts, dtype=np.int32),
            "feature_group_stops": np.asarray(self.group_stops, dtype=np.int32),
            "feature_group_weights": np.asarray(self.group_weights, dtype=np.float32),
        }

    @classmethod
    def from_npz(cls, data) -> "FeatureLayout":
        return cls(
            group_names=tuple(str(name) for name in data["feature_group_names"].tolist()),
            group_starts=tuple(int(value) for value in data["feature_group_starts"].tolist()),
            group_stops=tuple(int(value) for value in data["feature_group_stops"].tolist()),
            group_weights=tuple(float(value) for value in data["feature_group_weights"].tolist()),
        )


def resolve_joint_indices(joint_names, selected_joint_names) -> np.ndarray:
    joint_name_to_index = {str(joint_name): index for index, joint_name in enumerate(joint_names)}
    missing = [joint_name for joint_name in selected_joint_names if joint_name not in joint_name_to_index]
    if missing:
        raise ValueError(f"Missing joints in skeleton: {missing}")
    return np.asarray([joint_name_to_index[joint_name] for joint_name in selected_joint_names], dtype=np.int32)


def build_default_feature_layout(
    foot_joint_count: int = len(MM_FOOT_POSITION_JOINTS),
    velocity_joint_count: int = len(MM_VELOCITY_JOINTS),
    future_sample_count: int = len(MM_FUTURE_SAMPLE_OFFSETS),
    action_count: int = len(MM_ACTION_LABELS),
    group_weights: dict[str, float] | None = None,
) -> FeatureLayout:
    group_weights = dict(MM_INITIAL_FEATURE_GROUP_WEIGHTS if group_weights is None else group_weights)
    groups = (
        ("foot_positions", foot_joint_count * 3),
        ("joint_velocities", velocity_joint_count * 3),
        ("future_positions", future_sample_count * 2),
        ("future_directions", future_sample_count * 2),
        ("future_velocities", future_sample_count * 2),
        ("action", action_count),
    )

    starts = []
    stops = []
    names = []
    weights = []
    cursor = 0
    for name, size in groups:
        names.append(name)
        starts.append(cursor)
        cursor += int(size)
        stops.append(cursor)
        weights.append(float(group_weights.get(name, 1.0)))

    return FeatureLayout(
        group_names=tuple(names),
        group_starts=tuple(starts),
        group_stops=tuple(stops),
        group_weights=tuple(weights),
    )


def make_action_weights(action_label: str, action_labels=MM_ACTION_LABELS) -> np.ndarray:
    if action_label not in action_labels:
        raise ValueError(f"Unsupported action label: {action_label}")
    weights = np.zeros(len(action_labels), dtype=np.float32)
    weights[tuple(action_labels).index(action_label)] = 1.0
    return weights


def normalize_action_weights(action_weights: np.ndarray, action_count: int | None = None) -> np.ndarray:
    weights = np.asarray(action_weights, dtype=np.float32)
    if weights.ndim == 1:
        weights = weights[np.newaxis, :]
        squeeze = True
    elif weights.ndim == 2:
        squeeze = False
    else:
        raise ValueError(f"action_weights must be 1D or 2D, got {weights.shape}.")

    if action_count is not None and weights.shape[-1] != int(action_count):
        raise ValueError(f"action_weights dim {weights.shape[-1]} does not match {action_count}.")
    if not np.all(np.isfinite(weights)):
        raise ValueError("action_weights contains non-finite values.")
    if np.any(weights < 0.0):
        raise ValueError("action_weights must be non-negative.")

    sums = np.sum(weights, axis=-1, keepdims=True, dtype=np.float32)
    if np.any(sums <= 1e-8):
        raise ValueError("action_weights rows must have positive sums.")
    normalized = (weights / sums).astype(np.float32)
    return normalized[0] if squeeze else normalized


def infer_clip_action_label(clip_name: str) -> str:
    clip_name = str(clip_name).lower()
    if "jump" in clip_name:
        return "jump"
    if "run" in clip_name or "sprint" in clip_name:
        return "run"
    if "walk" in clip_name:
        return "walk"
    return "idle"


def compute_local_pose_arrays(pose_source, root_trajectory_source) -> dict[str, np.ndarray]:
    root_positions = np.asarray(root_trajectory_source["positions"], dtype=np.float32)
    root_rotations = np.asarray(root_trajectory_source["rotations"], dtype=np.float32)
    root_velocities = np.asarray(root_trajectory_source["velocities"], dtype=np.float32)
    global_positions = np.asarray(pose_source["global_positions"], dtype=np.float32)
    global_rotations = np.asarray(pose_source["global_rotations"], dtype=np.float32)
    global_velocities = np.asarray(pose_source["global_velocities"], dtype=np.float32)
    global_angular_velocities = np.asarray(pose_source["global_angular_velocities"], dtype=np.float32)

    root_rotations_for_joints = root_rotations[:, np.newaxis, :]
    local_positions = quat.inv_mul_vec(
        root_rotations_for_joints,
        global_positions - root_positions[:, np.newaxis, :],
    ).astype(np.float32)
    local_rotations = quat.inv_mul(root_rotations_for_joints, global_rotations).astype(np.float32)
    local_velocities = quat.inv_mul_vec(root_rotations_for_joints, global_velocities).astype(np.float32)
    local_angular_velocities = quat.inv_mul_vec(
        root_rotations_for_joints,
        global_angular_velocities,
    ).astype(np.float32)

    if "root_angular_velocities" in pose_source:
        root_world_angular_velocities = np.asarray(pose_source["root_angular_velocities"], dtype=np.float32)
    else:
        root_world_angular_velocities = np.zeros_like(root_velocities, dtype=np.float32)

    return {
        "local_positions": local_positions,
        "local_rotations": local_rotations,
        "local_velocities": local_velocities,
        "local_angular_velocities": local_angular_velocities,
        "root_positions": root_positions,
        "root_rotations": root_rotations,
        "root_velocities": root_velocities,
        "root_angular_velocities": root_world_angular_velocities,
        "root_local_velocities": quat.inv_mul_vec(root_rotations, root_velocities).astype(np.float32),
        "root_local_angular_velocities": quat.inv_mul_vec(
            root_rotations,
            root_world_angular_velocities,
        ).astype(np.float32),
    }


def build_future_trajectory_features(
    root_trajectory_source,
    frame_indices: np.ndarray,
    future_sample_offsets=MM_FUTURE_SAMPLE_OFFSETS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_indices = np.asarray(frame_indices, dtype=np.int32)
    future_sample_offsets = np.asarray(future_sample_offsets, dtype=np.int32)
    sample_frames = frame_indices[:, np.newaxis] + future_sample_offsets[np.newaxis, :]

    root_positions = np.asarray(root_trajectory_source["positions"], dtype=np.float32)
    root_rotations = np.asarray(root_trajectory_source["rotations"], dtype=np.float32)
    root_directions = np.asarray(root_trajectory_source["directions"], dtype=np.float32)
    root_velocities = np.asarray(root_trajectory_source["velocities"], dtype=np.float32)

    current_positions = root_positions[frame_indices]
    current_rotations = root_rotations[frame_indices]

    local_positions = quat.inv_mul_vec(
        current_rotations[:, np.newaxis, :],
        root_positions[sample_frames] - current_positions[:, np.newaxis, :],
    ).astype(np.float32)
    local_directions = quat.inv_mul_vec(
        current_rotations[:, np.newaxis, :],
        root_directions[sample_frames],
    ).astype(np.float32)
    local_velocities = quat.inv_mul_vec(
        current_rotations[:, np.newaxis, :],
        root_velocities[sample_frames],
    ).astype(np.float32)
    return local_positions, local_directions, local_velocities


def build_raw_feature_matrix(
    local_pose_arrays: dict[str, np.ndarray],
    root_trajectory_source,
    frame_indices: np.ndarray,
    foot_joint_indices: np.ndarray,
    velocity_joint_indices: np.ndarray,
    action_weights: np.ndarray,
    future_sample_offsets=MM_FUTURE_SAMPLE_OFFSETS,
) -> np.ndarray:
    frame_indices = np.asarray(frame_indices, dtype=np.int32)
    frame_action_weights = normalize_action_weights(
        np.asarray(action_weights, dtype=np.float32)[frame_indices],
    )
    future_positions, future_directions, future_velocities = build_future_trajectory_features(
        root_trajectory_source,
        frame_indices,
        future_sample_offsets=future_sample_offsets,
    )

    foot_positions = local_pose_arrays["local_positions"][frame_indices][:, foot_joint_indices].reshape(len(frame_indices), -1)
    joint_velocities = local_pose_arrays["local_velocities"][frame_indices][:, velocity_joint_indices].reshape(len(frame_indices), -1)
    future_positions_xz = future_positions[:, :, [0, 2]].reshape(len(frame_indices), -1)
    future_directions_xz = future_directions[:, :, [0, 2]].reshape(len(frame_indices), -1)
    future_velocities_xz = future_velocities[:, :, [0, 2]].reshape(len(frame_indices), -1)

    return np.concatenate(
        [
            foot_positions,
            joint_velocities,
            future_positions_xz,
            future_directions_xz,
            future_velocities_xz,
            frame_action_weights,
        ],
        axis=-1,
    ).astype(np.float32)


def safe_feature_std(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.where(values < 1e-6, 1.0, values).astype(np.float32)


def compute_feature_stats(raw_features: np.ndarray, layout: FeatureLayout) -> tuple[np.ndarray, np.ndarray]:
    raw_features = np.asarray(raw_features, dtype=np.float32)
    feature_mean = np.mean(raw_features, axis=0, dtype=np.float32).astype(np.float32)
    feature_std = safe_feature_std(np.std(raw_features, axis=0, dtype=np.float32))

    # Keep action hints as direct one-hot penalties rather than data-normalized values.
    action_slice = layout.group_slice("action")
    feature_mean[action_slice] = 0.0
    feature_std[action_slice] = 1.0
    return feature_mean.astype(np.float32), feature_std.astype(np.float32)


def normalize_and_weight_features(
    raw_features: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    layout: FeatureLayout,
) -> np.ndarray:
    normalized = (
        (np.asarray(raw_features, dtype=np.float32) - np.asarray(feature_mean, dtype=np.float32))
        / np.asarray(feature_std, dtype=np.float32)
    ).astype(np.float32)
    return (normalized * layout.weight_vector()[np.newaxis, :]).astype(np.float32)
