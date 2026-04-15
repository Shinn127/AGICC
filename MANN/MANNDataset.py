from dataclasses import dataclass
from pathlib import Path
import json
from typing import Optional, Tuple

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
    y_future_slice: Optional[slice]
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

        x_main_dim = x_main_pose_dim + x_main_traj_dim + x_main_speed_dim + x_main_action_dim
        x_gate_dim = x_gate_vel_dim + x_gate_action_dim + x_gate_speed_dim
        y_dim = y_pose_dim + y_root_dim + y_future_dim

        y_pose_slice = slice(0, y_pose_dim)
        y_root_slice = slice(y_pose_dim, y_pose_dim + y_root_dim)
        y_future_slice = None if y_future_dim == 0 else slice(y_pose_dim + y_root_dim, y_dim)
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


def build_clip_splits(clip_names, train_ratio=0.8, val_ratio=0.1, seed=1234):
    clip_names = np.asarray([str(clip_name) for clip_name in clip_names])
    if len(clip_names) == 0:
        return {"split_type": "sample_indices", "train": [], "val": [], "test": []}

    rng = np.random.default_rng(seed)
    splits = {
        "split_type": "sample_indices",
        "train": [],
        "val": [],
        "test": [],
    }

    for clip_name in sorted(set(clip_names)):
        clip_indices = np.flatnonzero(clip_names == clip_name).astype(np.int64)
        rng.shuffle(clip_indices)

        split_counts = _allocate_split_counts(
            len(clip_indices),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        train_count = split_counts["train"]
        val_count = split_counts["val"]
        test_start = train_count + val_count

        splits["train"].extend(int(index) for index in clip_indices[:train_count])
        splits["val"].extend(int(index) for index in clip_indices[train_count:test_start])
        splits["test"].extend(int(index) for index in clip_indices[test_start:])

    for split_name in ("train", "val", "test"):
        splits[split_name] = sorted(splits[split_name])

    return splits


def _allocate_split_counts(total_count, train_ratio=0.8, val_ratio=0.1):
    total_count = int(total_count)
    if total_count <= 0:
        return {"train": 0, "val": 0, "test": 0}

    ratios = np.asarray(
        [
            max(0.0, float(train_ratio)),
            max(0.0, float(val_ratio)),
            max(0.0, 1.0 - float(train_ratio) - float(val_ratio)),
        ],
        dtype=np.float64,
    )
    if float(np.sum(ratios)) <= 0.0:
        ratios[:] = (1.0, 0.0, 0.0)
    ratios = ratios / np.sum(ratios)

    ideal = ratios * total_count
    counts = np.floor(ideal).astype(np.int64)
    remainder = int(total_count - int(np.sum(counts)))
    if remainder > 0:
        fractional_order = np.argsort(-(ideal - counts))
        for split_index in fractional_order[:remainder]:
            counts[int(split_index)] += 1

    positive_splits = ratios > 0.0
    if total_count >= int(np.sum(positive_splits)):
        min_counts = positive_splits.astype(np.int64)
        for split_index in np.where(positive_splits & (counts == 0))[0]:
            donor_candidates = np.where(counts > min_counts)[0]
            if len(donor_candidates) == 0:
                break
            donor_index = int(donor_candidates[np.argmax(counts[donor_candidates])])
            counts[donor_index] -= 1
            counts[int(split_index)] += 1

    while int(np.sum(counts)) > total_count:
        donor_index = int(np.argmax(counts))
        counts[donor_index] -= 1
    while int(np.sum(counts)) < total_count:
        receiver_index = int(np.argmax(ratios))
        counts[receiver_index] += 1

    return {
        "train": int(counts[0]),
        "val": int(counts[1]),
        "test": int(counts[2]),
    }


def _split_needs_rebuild(clip_names, splits, train_ratio=0.8, val_ratio=0.1, seed=1234):
    if splits.get("split_type") != "sample_indices":
        return True

    sample_count = len(clip_names)
    try:
        split_sets = {
            split_name: {int(index) for index in splits.get(split_name, [])}
            for split_name in ("train", "val", "test")
        }
    except (TypeError, ValueError):
        return True

    valid_indices = set(range(sample_count))
    assigned = set().union(*split_sets.values()) if split_sets else set()
    if assigned != valid_indices:
        return True
    if any(split_sets[left] & split_sets[right] for left, right in (("train", "val"), ("train", "test"), ("val", "test"))):
        return True

    expected_splits = build_clip_splits(
        clip_names,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    return any(len(split_sets[split_name]) != len(expected_splits[split_name]) for split_name in ("train", "val", "test"))


def save_clip_splits(path, splits):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(splits, file, indent=2, sort_keys=True)


def load_clip_splits(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def indices_from_split(clip_names, splits, split_name):
    if splits.get("split_type") == "sample_indices":
        return np.asarray(splits.get(split_name, []), dtype=np.int64)

    allowed = set(splits.get(split_name, []))
    return np.asarray([index for index, clip_name in enumerate(clip_names) if str(clip_name) in allowed], dtype=np.int64)


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
        self.mirror_flags = (
            self.data["mirror_flags"].astype(np.uint8)
            if "mirror_flags" in self.data.files else
            np.zeros((len(self.y),), dtype=np.uint8)
        )
        self.variant_names = (
            self.data["variant_names"]
            if "variant_names" in self.data.files else
            self.clip_names
        )

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
    split_path=None,
    train_ratio=0.8,
    val_ratio=0.1,
    seed=1234,
    normalize=True,
    stats_path=None,
):
    dataset_path = Path(dataset_path)
    root_dataset = MANNDataset(dataset_path, normalize=False)
    clip_names = root_dataset.clip_names

    if split_path is not None and Path(split_path).exists():
        splits = load_clip_splits(split_path)
        if _split_needs_rebuild(clip_names, splits, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed):
            splits = build_clip_splits(clip_names, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
            save_clip_splits(split_path, splits)
    else:
        splits = build_clip_splits(clip_names, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
        if split_path is not None:
            save_clip_splits(split_path, splits)

    train_indices = indices_from_split(clip_names, splits, "train")
    val_indices = indices_from_split(clip_names, splits, "val")
    test_indices = indices_from_split(clip_names, splits, "test")

    if normalize:
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
    else:
        stats = None
        if stats_path is not None and Path(stats_path).exists():
            stats = MANNFeatureStats.load(stats_path)

    datasets = {
        "train": MANNDataset(dataset_path, indices=train_indices, normalize=normalize, stats=stats),
        "val": MANNDataset(dataset_path, indices=val_indices, normalize=normalize, stats=stats),
        "test": MANNDataset(dataset_path, indices=test_indices, normalize=normalize, stats=stats),
        "spec": root_dataset.spec,
        "stats": stats,
        "splits": splits,
    }
    return datasets


def build_mann_dataloaders(
    dataset_path,
    batch_size,
    split_path=None,
    train_ratio=0.8,
    val_ratio=0.1,
    seed=1234,
    normalize=True,
    stats_path=None,
    num_workers=0,
    pin_memory=True,
    shuffle_train=True,
):
    datasets = build_mann_datasets(
        dataset_path,
        split_path=split_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        normalize=normalize,
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
        "splits": datasets["splits"],
    }
    return dataloaders
