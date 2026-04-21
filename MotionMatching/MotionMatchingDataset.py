from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from MotionMatching.MotionMatchingConfig import MM_DATABASE_STAGE
from MotionMatching.MotionMatchingFeatures import FeatureLayout


@dataclass(frozen=True)
class MotionMatchingDataSpec:
    stage: str
    dt: float
    sample_count: int
    joint_count: int
    feature_dim: int
    raw_feature_dim: int
    action_labels: tuple[str, ...]
    future_sample_offsets: tuple[int, ...]
    feature_layout: FeatureLayout

    @classmethod
    def from_npz(cls, data) -> "MotionMatchingDataSpec":
        stage = str(data["stage"].item())
        future_sample_offsets = tuple(int(value) for value in data["future_sample_offsets"].tolist())
        action_labels = tuple(str(label) for label in data["action_labels"].tolist())
        feature_layout = FeatureLayout.from_npz(data)
        features = data["features"]
        raw_features = data["raw_features"]
        local_positions = data["local_positions"]
        joint_count = int(local_positions.shape[1])
        if "joint_names" in data.files and len(data["joint_names"]) != joint_count:
            raise ValueError(
                "joint_names length does not match local pose joint count: "
                f"{len(data['joint_names'])} vs {joint_count}."
            )
        if "parents" in data.files and len(data["parents"]) != joint_count:
            raise ValueError(
                "parents length does not match local pose joint count: "
                f"{len(data['parents'])} vs {joint_count}."
            )
        return cls(
            stage=stage,
            dt=float(data["dt"].item()),
            sample_count=int(features.shape[0]),
            joint_count=joint_count,
            feature_dim=int(features.shape[1]),
            raw_feature_dim=int(raw_features.shape[1]),
            action_labels=action_labels,
            future_sample_offsets=future_sample_offsets,
            feature_layout=feature_layout,
        )


class MotionMatchingDataset:
    def __init__(self, database_path):
        self.database_path = Path(database_path)
        self.data = np.load(self.database_path, allow_pickle=False)
        self.spec = MotionMatchingDataSpec.from_npz(self.data)
        self._validate()

    def _validate(self) -> None:
        if self.spec.stage != MM_DATABASE_STAGE:
            raise ValueError(f"Unsupported Motion Matching database stage: {self.spec.stage}")
        if self.spec.feature_dim != self.spec.feature_layout.feature_dim:
            raise ValueError(
                "Feature dimension does not match layout: "
                f"{self.spec.feature_dim} vs {self.spec.feature_layout.feature_dim}."
            )
        if self.spec.raw_feature_dim != self.spec.feature_layout.feature_dim:
            raise ValueError(
                "Raw feature dimension does not match layout: "
                f"{self.spec.raw_feature_dim} vs {self.spec.feature_layout.feature_dim}."
            )

        sample_count = self.spec.sample_count
        sample_arrays = (
            "raw_features",
            "features",
            "local_positions",
            "local_rotations",
            "local_velocities",
            "local_angular_velocities",
            "root_positions",
            "root_rotations",
            "root_velocities",
            "root_angular_velocities",
            "root_local_velocities",
            "root_local_angular_velocities",
            "action_weights",
            "action_ids",
            "frame_indices",
            "clip_names",
            "variant_names",
            "mirror_flags",
        )
        for key in sample_arrays:
            if self.data[key].shape[0] != sample_count:
                raise ValueError(f"{key} sample count mismatch: {self.data[key].shape[0]} vs {sample_count}.")

        action_weights = self.data["action_weights"].astype(np.float32)
        if action_weights.shape != (sample_count, len(self.spec.action_labels)):
            raise ValueError(
                "action_weights shape mismatch: "
                f"{action_weights.shape} vs {(sample_count, len(self.spec.action_labels))}."
            )
        if not np.all(np.isfinite(action_weights)):
            raise ValueError("action_weights contains non-finite values.")
        if np.any(action_weights < 0.0):
            raise ValueError("action_weights must be non-negative.")
        if np.any(np.sum(action_weights, axis=1) <= 1e-6):
            raise ValueError("action_weights rows must have positive sums.")

        range_starts = self.data["range_starts"].astype(np.int32)
        range_stops = self.data["range_stops"].astype(np.int32)
        if len(range_starts) != len(range_stops):
            raise ValueError("range_starts and range_stops length mismatch.")
        if np.any(range_starts < 0) or np.any(range_stops > sample_count) or np.any(range_starts >= range_stops):
            raise ValueError("Invalid Motion Matching range bounds.")
        if "joint_names" not in self.data.files:
            raise ValueError("Motion Matching database is missing joint_names metadata.")
        if "parents" not in self.data.files:
            raise ValueError("Motion Matching database is missing parents metadata.")

    @property
    def features(self):
        return self.data["features"].astype(np.float32)

    @property
    def raw_features(self):
        return self.data["raw_features"].astype(np.float32)

    @property
    def action_ids(self):
        return self.data["action_ids"].astype(np.int32)

    @property
    def action_weights(self):
        weights = self.data["action_weights"].astype(np.float32)
        sums = np.sum(weights, axis=1, keepdims=True, dtype=np.float32)
        return (weights / np.maximum(sums, 1e-8)).astype(np.float32)

    @property
    def action_labels(self):
        return self.spec.action_labels

    @property
    def range_starts(self):
        return self.data["range_starts"].astype(np.int32)

    @property
    def range_stops(self):
        return self.data["range_stops"].astype(np.int32)

    @property
    def range_names(self):
        return self.data["range_names"]

    @property
    def feature_mean(self):
        return self.data["feature_mean"].astype(np.float32)

    @property
    def feature_std(self):
        return self.data["feature_std"].astype(np.float32)

    @property
    def joint_names(self):
        return tuple(str(name) for name in self.data["joint_names"].tolist())

    @property
    def parents(self):
        return self.data["parents"].astype(np.int32)

    def close(self):
        self.data.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def summarize_database(database: MotionMatchingDataset) -> dict:
    action_counts = np.bincount(
        database.action_ids,
        minlength=len(database.action_labels),
    )
    clip_names, clip_counts = np.unique(database.data["clip_names"], return_counts=True)
    variant_names, variant_counts = np.unique(database.data["variant_names"], return_counts=True)

    return {
        "path": str(database.database_path),
        "stage": database.spec.stage,
        "dt": database.spec.dt,
        "samples": database.spec.sample_count,
        "joints": database.spec.joint_count,
        "feature_dim": database.spec.feature_dim,
        "future_sample_offsets": database.spec.future_sample_offsets,
        "actions": dict(zip(database.action_labels, action_counts.tolist())),
        "clips": dict(zip(clip_names.tolist(), clip_counts.tolist())),
        "variants": dict(zip(variant_names.tolist(), variant_counts.tolist())),
        "ranges": [
            {
                "name": str(name),
                "start": int(start),
                "stop": int(stop),
                "length": int(stop - start),
            }
            for name, start, stop in zip(database.range_names, database.range_starts, database.range_stops)
        ],
    }
