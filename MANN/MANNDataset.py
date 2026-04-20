from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class MANNDataSpec:
    stage: str
    x_main_dim: int
    x_gate_dim: int
    y_dim: int
    y_pose_slice: slice
    y_root_slice: slice
    y_future_slice: slice
    x_main_action_slice: slice
    x_gate_action_slice: slice
    action_labels: Tuple[str, ...]

    @staticmethod
    def from_npz(data):
        stage = str(data["stage"].item())
        x_main_pose_dim = int(data["x_main_pose_dim"].item())
        x_main_traj_dim = int(data["x_main_traj_dim"].item())
        x_main_speed_dim = int(data["x_main_speed_dim"].item())
        x_main_action_dim = int(data["x_main_action_dim"].item())
        x_gate_vel_dim = int(data["x_gate_vel_dim"].item())
        x_gate_action_dim = int(data["x_gate_action_dim"].item())
        x_gate_speed_dim = int(data["x_gate_speed_dim"].item())
        y_pose_dim = int(data["y_pose_dim"].item())
        y_root_dim = int(data["y_root_dim"].item())
        y_future_dim = int(data["y_future_dim"].item())
        if stage != "stage2" or y_future_dim <= 0:
            raise ValueError("MANN datasets must be exported with the stage2 database builder.")

        x_main_dim = x_main_pose_dim + x_main_traj_dim + x_main_speed_dim + x_main_action_dim
        x_gate_dim = x_gate_vel_dim + x_gate_action_dim + x_gate_speed_dim
        y_dim = y_pose_dim + y_root_dim + y_future_dim

        y_pose_slice = slice(0, y_pose_dim)
        y_root_slice = slice(y_pose_dim, y_pose_dim + y_root_dim)
        y_future_slice = slice(y_pose_dim + y_root_dim, y_dim)
        x_main_action_slice = slice(x_main_pose_dim + x_main_traj_dim + x_main_speed_dim, x_main_dim)
        x_gate_action_slice = slice(x_gate_vel_dim, x_gate_vel_dim + x_gate_action_dim)

        action_labels = tuple(str(label) for label in data["action_labels"].tolist())
        return MANNDataSpec(
            stage=stage,
            x_main_dim=x_main_dim,
            x_gate_dim=x_gate_dim,
            y_dim=y_dim,
            y_pose_slice=y_pose_slice,
            y_root_slice=y_root_slice,
            y_future_slice=y_future_slice,
            x_main_action_slice=x_main_action_slice,
            x_gate_action_slice=x_gate_action_slice,
            action_labels=action_labels,
        )


@dataclass
class MANNFeatureStats:
    x_main_mean: np.ndarray
    x_main_std: np.ndarray
    x_gate_mean: np.ndarray
    x_gate_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray

    @staticmethod
    def _safe_std(values):
        values = np.asarray(values, dtype=np.float32)
        return np.where(values < 1e-6, 1.0, values).astype(np.float32)

    @classmethod
    def from_arrays(cls, x_main, x_gate, y, spec):
        x_main_mean = np.mean(x_main, axis=0, dtype=np.float32)
        x_main_std = cls._safe_std(np.std(x_main, axis=0, dtype=np.float32))
        x_gate_mean = np.mean(x_gate, axis=0, dtype=np.float32)
        x_gate_std = cls._safe_std(np.std(x_gate, axis=0, dtype=np.float32))
        y_mean = np.mean(y, axis=0, dtype=np.float32)
        y_std = cls._safe_std(np.std(y, axis=0, dtype=np.float32))

        x_main_mean[spec.x_main_action_slice] = 0.0
        x_main_std[spec.x_main_action_slice] = 1.0
        x_gate_mean[spec.x_gate_action_slice] = 0.0
        x_gate_std[spec.x_gate_action_slice] = 1.0

        return cls(
            x_main_mean=x_main_mean.astype(np.float32),
            x_main_std=x_main_std.astype(np.float32),
            x_gate_mean=x_gate_mean.astype(np.float32),
            x_gate_std=x_gate_std.astype(np.float32),
            y_mean=y_mean.astype(np.float32),
            y_std=y_std.astype(np.float32),
        )

    def normalize_x_main(self, x_main):
        return ((x_main - self.x_main_mean) / self.x_main_std).astype(np.float32)

    def normalize_x_gate(self, x_gate):
        return ((x_gate - self.x_gate_mean) / self.x_gate_std).astype(np.float32)

    def normalize_y(self, y):
        return ((y - self.y_mean) / self.y_std).astype(np.float32)

    def denormalize_y(self, y):
        return (y * self.y_std + self.y_mean).astype(np.float32)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            x_main_mean=self.x_main_mean,
            x_main_std=self.x_main_std,
            x_gate_mean=self.x_gate_mean,
            x_gate_std=self.x_gate_std,
            y_mean=self.y_mean,
            y_std=self.y_std,
        )

    @classmethod
    def load(cls, path):
        data = np.load(Path(path), allow_pickle=False)
        return cls(
            x_main_mean=data["x_main_mean"].astype(np.float32),
            x_main_std=data["x_main_std"].astype(np.float32),
            x_gate_mean=data["x_gate_mean"].astype(np.float32),
            x_gate_std=data["x_gate_std"].astype(np.float32),
            y_mean=data["y_mean"].astype(np.float32),
            y_std=data["y_std"].astype(np.float32),
        )


def build_action_splits(action_ids, train_ratio=0.8, val_ratio=0.1, seed=1234):
    action_ids = np.asarray(action_ids, dtype=np.int64)
    rng = np.random.default_rng(seed)
    split_parts = {"train": [], "val": [], "test": []}

    for action_id in np.unique(action_ids):
        indices = np.flatnonzero(action_ids == action_id).astype(np.int64)
        rng.shuffle(indices)

        train_count = int(round(len(indices) * float(train_ratio)))
        val_count = int(round(len(indices) * float(val_ratio)))
        train_count = min(max(train_count, 0), len(indices))
        val_count = min(max(val_count, 0), len(indices) - train_count)
        test_start = train_count + val_count

        split_parts["train"].append(indices[:train_count])
        split_parts["val"].append(indices[train_count:test_start])
        split_parts["test"].append(indices[test_start:])

    return {
        split_name: np.sort(np.concatenate(parts)).astype(np.int64) if parts else np.zeros((0,), dtype=np.int64)
        for split_name, parts in split_parts.items()
    }


class MANNDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        indices=None,
        normalize=False,
        stats=None,
    ):
        self.dataset_path = Path(dataset_path)
        self.data = np.load(self.dataset_path, allow_pickle=False)
        self.spec = MANNDataSpec.from_npz(self.data)

        self.x_main = self.data["x_main"].astype(np.float32)
        self.x_gate = self.data["x_gate"].astype(np.float32)
        self.y = self.data["y"].astype(np.float32)
        self.action_ids = self.data["action_ids"].astype(np.int64)
        self.clip_names = self.data["clip_names"]
        self.frame_indices = self.data["frame_indices"].astype(np.int64)
        self.mirror_flags = self.data["mirror_flags"].astype(np.uint8)
        self.variant_names = self.data["variant_names"]

        if indices is None:
            self.indices = np.arange(len(self.y), dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)

        self.stats = stats
        self.normalize = bool(normalize and stats is not None)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        sample_index = int(self.indices[index])
        x_main = self.x_main[sample_index]
        x_gate = self.x_gate[sample_index]
        y = self.y[sample_index]

        if self.normalize:
            x_main = self.stats.normalize_x_main(x_main)
            x_gate = self.stats.normalize_x_gate(x_gate)
            y = self.stats.normalize_y(y)

        return {
            "x_main": torch.from_numpy(np.asarray(x_main, dtype=np.float32)),
            "x_gate": torch.from_numpy(np.asarray(x_gate, dtype=np.float32)),
            "y": torch.from_numpy(np.asarray(y, dtype=np.float32)),
            "action_id": torch.tensor(self.action_ids[sample_index], dtype=torch.long),
            "frame_index": torch.tensor(self.frame_indices[sample_index], dtype=torch.long),
            "clip_name": str(self.clip_names[sample_index]),
        }


def build_mann_datasets(
    dataset_path,
    train_ratio=0.8,
    val_ratio=0.1,
    seed=1234,
    stats_path=None,
):
    dataset_path = Path(dataset_path)
    root_dataset = MANNDataset(dataset_path, normalize=False)
    splits = build_action_splits(root_dataset.action_ids, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    train_indices = splits["train"]
    val_indices = splits["val"]
    test_indices = splits["test"]

    if len(train_indices) == 0:
        raise ValueError("Train split is empty; cannot compute normalization statistics.")

    stats = MANNFeatureStats.from_arrays(
        root_dataset.x_main[train_indices],
        root_dataset.x_gate[train_indices],
        root_dataset.y[train_indices],
        root_dataset.spec,
    )
    if stats_path is not None:
        stats.save(stats_path)

    datasets = {
        "train": MANNDataset(dataset_path, indices=train_indices, normalize=True, stats=stats),
        "val": MANNDataset(dataset_path, indices=val_indices, normalize=True, stats=stats),
        "test": MANNDataset(dataset_path, indices=test_indices, normalize=True, stats=stats),
        "spec": root_dataset.spec,
        "stats": stats,
    }
    return datasets


def build_mann_dataloaders(
    dataset_path,
    batch_size,
    train_ratio=0.8,
    val_ratio=0.1,
    seed=1234,
    stats_path=None,
    num_workers=0,
    pin_memory=True,
    shuffle_train=True,
):
    datasets = build_mann_datasets(
        dataset_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        stats_path=stats_path,
    )

    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "spec": datasets["spec"],
        "stats": datasets["stats"],
    }
    return dataloaders
