from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import sys

MOTION_MATCHING_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MOTION_MATCHING_ROOT
if REPO_ROOT.name == "MotionMatching":
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from genoview.modules.BVHImporter import BVHImporter
from genoview.modules.MotionMirror import DEFAULT_MIRROR_AXIS, MirrorBVHAnimation
from genoview.modules.PoseModule import BuildPoseSource
from genoview.modules.RootModule import (
    ROOT_JOINT_INDEX,
    ROOT_TRAJECTORY_MODE_FLAT,
    BuildRootTrajectorySource,
)
from MotionMatching.MotionMatchingConfig import (
    DEFAULT_BVH_FRAME_TIME,
    DEFAULT_BVH_SCALE,
    DEFAULT_DATABASE_PATH,
    DEFAULT_DATASET_DIR,
    MM_ACTION_LABELS,
    MM_DATABASE_STAGE,
    MM_DEFAULT_CLIP_SPECS,
    MM_DEFAULT_LABEL_SOURCE,
    MM_FUTURE_SAMPLE_OFFSETS,
    MM_LABEL_SOURCE_AUTO,
    MM_LABEL_SOURCE_CLIP,
    MM_LABEL_SOURCES,
)
from MotionMatching.MotionMatchingFeatures import (
    FeatureLayout,
    build_default_feature_layout,
    build_raw_feature_matrix,
    compute_feature_stats,
    compute_local_pose_arrays,
    infer_clip_action_label,
    make_action_weights,
    normalize_action_weights,
    normalize_and_weight_features,
    resolve_joint_indices,
)


@dataclass(frozen=True)
class MotionMatchingClipDatabase:
    raw_features: np.ndarray
    local_positions: np.ndarray
    local_rotations: np.ndarray
    local_velocities: np.ndarray
    local_angular_velocities: np.ndarray
    root_positions: np.ndarray
    root_rotations: np.ndarray
    root_velocities: np.ndarray
    root_angular_velocities: np.ndarray
    root_local_velocities: np.ndarray
    root_local_angular_velocities: np.ndarray
    action_weights: np.ndarray
    action_ids: np.ndarray
    frame_indices: np.ndarray
    clip_names: np.ndarray
    variant_names: np.ndarray
    mirror_flags: np.ndarray
    range_starts: np.ndarray
    range_stops: np.ndarray
    range_names: np.ndarray
    parents: np.ndarray
    joint_names: np.ndarray


def _empty_clip_database() -> MotionMatchingClipDatabase:
    return MotionMatchingClipDatabase(
        raw_features=np.zeros((0, 0), dtype=np.float32),
        local_positions=np.zeros((0, 0, 3), dtype=np.float32),
        local_rotations=np.zeros((0, 0, 4), dtype=np.float32),
        local_velocities=np.zeros((0, 0, 3), dtype=np.float32),
        local_angular_velocities=np.zeros((0, 0, 3), dtype=np.float32),
        root_positions=np.zeros((0, 3), dtype=np.float32),
        root_rotations=np.zeros((0, 4), dtype=np.float32),
        root_velocities=np.zeros((0, 3), dtype=np.float32),
        root_angular_velocities=np.zeros((0, 3), dtype=np.float32),
        root_local_velocities=np.zeros((0, 3), dtype=np.float32),
        root_local_angular_velocities=np.zeros((0, 3), dtype=np.float32),
        action_weights=np.zeros((0, len(MM_ACTION_LABELS)), dtype=np.float32),
        action_ids=np.zeros((0,), dtype=np.int32),
        frame_indices=np.zeros((0,), dtype=np.int32),
        clip_names=np.asarray([], dtype="<U1"),
        variant_names=np.asarray([], dtype="<U1"),
        mirror_flags=np.zeros((0,), dtype=np.uint8),
        range_starts=np.zeros((0,), dtype=np.int32),
        range_stops=np.zeros((0,), dtype=np.int32),
        range_names=np.asarray([], dtype="<U1"),
        parents=np.zeros((0,), dtype=np.int32),
        joint_names=np.asarray([], dtype="<U1"),
    )


def _normalize_clip_spec(clip_spec):
    if isinstance(clip_spec, str):
        parts = clip_spec.split(":")
        if len(parts) == 1:
            return parts[0], None, None
        if len(parts) == 3:
            start = None if parts[1] == "" else int(parts[1])
            stop = None if parts[2] == "" else int(parts[2])
            return parts[0], start, stop
        raise ValueError(f"Invalid clip spec: {clip_spec}. Expected stem or stem:start:stop.")
    if len(clip_spec) != 3:
        raise ValueError("Clip specs must be stem strings or (clip_stem, frame_start, frame_end).")
    clip_stem, frame_start, frame_end = clip_spec
    return str(clip_stem), frame_start, frame_end


def list_motion_clips(dataset_dir=DEFAULT_DATASET_DIR, clip_specs=MM_DEFAULT_CLIP_SPECS):
    dataset_dir = Path(dataset_dir)
    normalized_specs = [_normalize_clip_spec(clip_spec) for clip_spec in clip_specs]
    missing = [
        clip_stem
        for clip_stem, _, _ in normalized_specs
        if not (dataset_dir / f"{clip_stem}.bvh").exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing required Motion Matching BVH clips in {dataset_dir}: {missing}")
    return [
        (dataset_dir / f"{clip_stem}.bvh", frame_start, frame_end)
        for clip_stem, frame_start, frame_end in normalized_specs
    ]


def _get_valid_frame_indices(
    frame_count: int,
    future_sample_offsets=MM_FUTURE_SAMPLE_OFFSETS,
    frame_start=None,
    frame_end=None,
) -> np.ndarray:
    future_sample_offsets = np.asarray(future_sample_offsets, dtype=np.int32)
    max_future_offset = int(np.max(future_sample_offsets)) if len(future_sample_offsets) else 0
    lower_bound = 0 if frame_start is None else max(0, int(frame_start))
    upper_bound = frame_count - 1 if frame_end is None else min(frame_count - 1, int(frame_end))
    last_valid_frame = min(upper_bound - max_future_offset, frame_count - 1 - max_future_offset)
    if last_valid_frame < lower_bound:
        return np.zeros((0,), dtype=np.int32)
    return np.arange(lower_bound, last_valid_frame + 1, dtype=np.int32)


def _label_key_from_clip_path(clip_path):
    clip_path = Path(clip_path)
    parts = [str(part) for part in clip_path.parts]
    lower_parts = [part.lower() for part in parts]
    if "bvh" in lower_parts:
        bvh_index = lower_parts.index("bvh")
        return Path(*parts[bvh_index:]).as_posix()
    return clip_path.stem


def _build_clip_action_weights(clip_path, animation, dt, label_source, min_action_weight):
    frame_count = animation.frame_count
    label_source = str(label_source).lower()
    if label_source == MM_LABEL_SOURCE_CLIP:
        action_label = infer_clip_action_label(Path(clip_path).stem)
        weights = np.repeat(
            make_action_weights(action_label)[np.newaxis, :],
            frame_count,
            axis=0,
        ).astype(np.float32)
        return weights, np.ones(frame_count, dtype=bool)

    if label_source == MM_LABEL_SOURCE_AUTO:
        from MANN.MANNDatabaseBuilder import build_label_action_weights

        return build_label_action_weights(
            _label_key_from_clip_path(clip_path),
            animation,
            dt,
            min_action_weight=min_action_weight,
        )

    raise ValueError(f"Unsupported label_source: {label_source}. Expected one of: {MM_LABEL_SOURCES}")


def _filter_valid_action_frames(frame_indices, valid_action_mask, future_sample_offsets):
    if len(frame_indices) == 0:
        return frame_indices
    frame_indices = np.asarray(frame_indices, dtype=np.int32)
    future_sample_offsets = np.asarray(future_sample_offsets, dtype=np.int32)
    sample_frames = frame_indices[:, np.newaxis] + future_sample_offsets[np.newaxis, :]
    valid_mask = valid_action_mask[frame_indices] & np.all(valid_action_mask[sample_frames], axis=1)
    return frame_indices[valid_mask].astype(np.int32)


def _build_clip_variant_database(
    animation,
    clip_path,
    clip_name,
    variant_name,
    frame_start,
    frame_end,
    label_source=MM_LABEL_SOURCE_CLIP,
    min_action_weight=1e-4,
    dt=DEFAULT_BVH_FRAME_TIME,
    future_sample_offsets=MM_FUTURE_SAMPLE_OFFSETS,
    mirrored=False,
) -> MotionMatchingClipDatabase:
    joint_names = [str(name) for name in animation.raw_data["names"]]
    foot_joint_indices = resolve_joint_indices(joint_names, ("LeftToeBase", "RightToeBase"))
    velocity_joint_indices = resolve_joint_indices(joint_names, ("LeftToeBase", "RightToeBase", "Hips"))

    root_trajectory_source = BuildRootTrajectorySource(
        animation.global_positions,
        animation.global_rotations,
        dt,
        rootIndex=ROOT_JOINT_INDEX,
        mode=ROOT_TRAJECTORY_MODE_FLAT,
        projectToGround=True,
        groundHeight=0.0,
    )
    pose_source = BuildPoseSource(
        animation.global_positions,
        animation.global_rotations,
        dt,
        rootTrajectorySource=root_trajectory_source,
    )
    local_pose_arrays = compute_local_pose_arrays(pose_source, root_trajectory_source)
    action_weights, valid_action_mask = _build_clip_action_weights(
        clip_path,
        animation,
        dt,
        label_source=label_source,
        min_action_weight=min_action_weight,
    )

    frame_indices = _get_valid_frame_indices(
        animation.frame_count,
        future_sample_offsets=future_sample_offsets,
        frame_start=frame_start,
        frame_end=frame_end,
    )
    frame_indices = _filter_valid_action_frames(frame_indices, valid_action_mask, future_sample_offsets)
    if len(frame_indices) == 0:
        return _empty_clip_database()

    raw_features = build_raw_feature_matrix(
        local_pose_arrays,
        root_trajectory_source,
        frame_indices,
        foot_joint_indices,
        velocity_joint_indices,
        action_weights,
        future_sample_offsets=future_sample_offsets,
    )
    frame_action_weights = normalize_action_weights(
        np.asarray(action_weights, dtype=np.float32)[frame_indices],
        action_count=len(MM_ACTION_LABELS),
    ).astype(np.float32)
    action_ids = np.argmax(frame_action_weights, axis=1).astype(np.int32)

    return MotionMatchingClipDatabase(
        raw_features=raw_features.astype(np.float32),
        local_positions=local_pose_arrays["local_positions"][frame_indices].astype(np.float32),
        local_rotations=local_pose_arrays["local_rotations"][frame_indices].astype(np.float32),
        local_velocities=local_pose_arrays["local_velocities"][frame_indices].astype(np.float32),
        local_angular_velocities=local_pose_arrays["local_angular_velocities"][frame_indices].astype(np.float32),
        root_positions=local_pose_arrays["root_positions"][frame_indices].astype(np.float32),
        root_rotations=local_pose_arrays["root_rotations"][frame_indices].astype(np.float32),
        root_velocities=local_pose_arrays["root_velocities"][frame_indices].astype(np.float32),
        root_angular_velocities=local_pose_arrays["root_angular_velocities"][frame_indices].astype(np.float32),
        root_local_velocities=local_pose_arrays["root_local_velocities"][frame_indices].astype(np.float32),
        root_local_angular_velocities=local_pose_arrays["root_local_angular_velocities"][frame_indices].astype(np.float32),
        action_weights=frame_action_weights,
        action_ids=action_ids,
        frame_indices=frame_indices.astype(np.int32),
        clip_names=np.repeat(str(clip_name), len(frame_indices)),
        variant_names=np.repeat(str(variant_name), len(frame_indices)),
        mirror_flags=np.repeat(1 if mirrored else 0, len(frame_indices)).astype(np.uint8),
        range_starts=np.asarray([0], dtype=np.int32),
        range_stops=np.asarray([len(frame_indices)], dtype=np.int32),
        range_names=np.asarray([str(variant_name)]),
        parents=np.asarray(animation.parents, dtype=np.int32),
        joint_names=np.asarray(joint_names),
    )


def build_clip_database(
    clip_path,
    frame_start=None,
    frame_end=None,
    scale=DEFAULT_BVH_SCALE,
    dt=DEFAULT_BVH_FRAME_TIME,
    future_sample_offsets=MM_FUTURE_SAMPLE_OFFSETS,
    label_source=MM_LABEL_SOURCE_CLIP,
    min_action_weight=1e-4,
    mirror=False,
    mirror_axis=DEFAULT_MIRROR_AXIS,
) -> MotionMatchingClipDatabase:
    clip_path = Path(clip_path)
    animation = BVHImporter.load(str(clip_path), scale=scale)
    clip_name = clip_path.stem

    databases = [
        _build_clip_variant_database(
            animation,
            clip_path,
            clip_name,
            clip_name,
            frame_start,
            frame_end,
            label_source=label_source,
            min_action_weight=min_action_weight,
            dt=dt,
            future_sample_offsets=future_sample_offsets,
            mirrored=False,
        )
    ]

    if mirror:
        mirrored_animation = MirrorBVHAnimation(animation, axis=mirror_axis)
        databases.append(
            _build_clip_variant_database(
                mirrored_animation,
                clip_path,
                clip_name,
                f"{clip_name}_mirror",
                frame_start,
                frame_end,
                label_source=label_source,
                min_action_weight=min_action_weight,
                dt=dt,
                future_sample_offsets=future_sample_offsets,
                mirrored=True,
            )
        )

    return concatenate_clip_databases(databases)


def concatenate_clip_databases(databases) -> MotionMatchingClipDatabase:
    non_empty = [database for database in databases if len(database.action_ids) > 0]
    if not non_empty:
        return _empty_clip_database()

    range_starts = []
    range_stops = []
    range_names = []
    offset = 0
    parents = np.asarray(non_empty[0].parents, dtype=np.int32)
    joint_names = np.asarray(non_empty[0].joint_names)
    for database in non_empty:
        if not np.array_equal(parents, np.asarray(database.parents, dtype=np.int32)):
            raise ValueError("Cannot concatenate Motion Matching databases with different parent arrays.")
        if not np.array_equal(joint_names, np.asarray(database.joint_names)):
            raise ValueError("Cannot concatenate Motion Matching databases with different joint names.")
        for start, stop, name in zip(database.range_starts, database.range_stops, database.range_names):
            range_starts.append(offset + int(start))
            range_stops.append(offset + int(stop))
            range_names.append(str(name))
        offset += len(database.action_ids)

    return MotionMatchingClipDatabase(
        raw_features=np.concatenate([database.raw_features for database in non_empty], axis=0).astype(np.float32),
        local_positions=np.concatenate([database.local_positions for database in non_empty], axis=0).astype(np.float32),
        local_rotations=np.concatenate([database.local_rotations for database in non_empty], axis=0).astype(np.float32),
        local_velocities=np.concatenate([database.local_velocities for database in non_empty], axis=0).astype(np.float32),
        local_angular_velocities=np.concatenate([database.local_angular_velocities for database in non_empty], axis=0).astype(np.float32),
        root_positions=np.concatenate([database.root_positions for database in non_empty], axis=0).astype(np.float32),
        root_rotations=np.concatenate([database.root_rotations for database in non_empty], axis=0).astype(np.float32),
        root_velocities=np.concatenate([database.root_velocities for database in non_empty], axis=0).astype(np.float32),
        root_angular_velocities=np.concatenate([database.root_angular_velocities for database in non_empty], axis=0).astype(np.float32),
        root_local_velocities=np.concatenate([database.root_local_velocities for database in non_empty], axis=0).astype(np.float32),
        root_local_angular_velocities=np.concatenate([database.root_local_angular_velocities for database in non_empty], axis=0).astype(np.float32),
        action_weights=np.concatenate([database.action_weights for database in non_empty], axis=0).astype(np.float32),
        action_ids=np.concatenate([database.action_ids for database in non_empty], axis=0).astype(np.int32),
        frame_indices=np.concatenate([database.frame_indices for database in non_empty], axis=0).astype(np.int32),
        clip_names=np.concatenate([database.clip_names for database in non_empty], axis=0),
        variant_names=np.concatenate([database.variant_names for database in non_empty], axis=0),
        mirror_flags=np.concatenate([database.mirror_flags for database in non_empty], axis=0).astype(np.uint8),
        range_starts=np.asarray(range_starts, dtype=np.int32),
        range_stops=np.asarray(range_stops, dtype=np.int32),
        range_names=np.asarray(range_names),
        parents=parents.astype(np.int32),
        joint_names=joint_names,
    )


def build_dataset(
    dataset_dir=DEFAULT_DATASET_DIR,
    clip_specs=MM_DEFAULT_CLIP_SPECS,
    scale=DEFAULT_BVH_SCALE,
    dt=DEFAULT_BVH_FRAME_TIME,
    future_sample_offsets=MM_FUTURE_SAMPLE_OFFSETS,
    label_source=MM_LABEL_SOURCE_CLIP,
    min_action_weight=1e-4,
    mirror=False,
    mirror_axis=DEFAULT_MIRROR_AXIS,
) -> tuple[MotionMatchingClipDatabase, FeatureLayout, np.ndarray, np.ndarray, np.ndarray]:
    clip_entries = list_motion_clips(dataset_dir=dataset_dir, clip_specs=clip_specs)
    databases = [
        build_clip_database(
            clip_path,
            frame_start=frame_start,
            frame_end=frame_end,
            scale=scale,
            dt=dt,
            future_sample_offsets=future_sample_offsets,
            label_source=label_source,
            min_action_weight=min_action_weight,
            mirror=mirror,
            mirror_axis=mirror_axis,
        )
        for clip_path, frame_start, frame_end in clip_entries
    ]
    dataset = concatenate_clip_databases(databases)
    if len(dataset.action_ids) == 0:
        raise ValueError("Motion Matching database is empty.")

    layout = build_default_feature_layout(
        future_sample_count=len(tuple(future_sample_offsets)),
        action_count=len(MM_ACTION_LABELS),
    )
    if dataset.raw_features.shape[1] != layout.feature_dim:
        raise ValueError(
            "Raw feature dimension does not match default layout: "
            f"{dataset.raw_features.shape[1]} vs {layout.feature_dim}."
        )
    feature_mean, feature_std = compute_feature_stats(dataset.raw_features, layout)
    features = normalize_and_weight_features(dataset.raw_features, feature_mean, feature_std, layout)
    return dataset, layout, feature_mean, feature_std, features


def save_dataset_npz(
    output_path,
    dataset: MotionMatchingClipDatabase,
    layout: FeatureLayout,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    features: np.ndarray,
    dt=DEFAULT_BVH_FRAME_TIME,
    future_sample_offsets=MM_FUTURE_SAMPLE_OFFSETS,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        stage=np.asarray(MM_DATABASE_STAGE),
        dt=np.asarray(float(dt), dtype=np.float32),
        future_sample_offsets=np.asarray(future_sample_offsets, dtype=np.int32),
        action_labels=np.asarray(MM_ACTION_LABELS),
        raw_features=dataset.raw_features,
        features=np.asarray(features, dtype=np.float32),
        feature_mean=np.asarray(feature_mean, dtype=np.float32),
        feature_std=np.asarray(feature_std, dtype=np.float32),
        local_positions=dataset.local_positions,
        local_rotations=dataset.local_rotations,
        local_velocities=dataset.local_velocities,
        local_angular_velocities=dataset.local_angular_velocities,
        root_positions=dataset.root_positions,
        root_rotations=dataset.root_rotations,
        root_velocities=dataset.root_velocities,
        root_angular_velocities=dataset.root_angular_velocities,
        root_local_velocities=dataset.root_local_velocities,
        root_local_angular_velocities=dataset.root_local_angular_velocities,
        action_weights=dataset.action_weights,
        action_ids=dataset.action_ids,
        frame_indices=dataset.frame_indices,
        clip_names=dataset.clip_names,
        variant_names=dataset.variant_names,
        mirror_flags=dataset.mirror_flags,
        range_starts=dataset.range_starts,
        range_stops=dataset.range_stops,
        range_names=dataset.range_names,
        parents=dataset.parents,
        joint_names=dataset.joint_names,
        **layout.to_npz_metadata(),
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Build the Motion Matching locomotion database.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR, help="Directory containing LaFAN1 BVH clips.")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATABASE_PATH, help="Destination .npz file.")
    parser.add_argument(
        "--clip",
        action="append",
        default=None,
        help="Clip stem or stem:start:stop. Can be repeated. Defaults to the MANN-aligned idle/walk/run/jump clip set.",
    )
    parser.add_argument("--scale", type=float, default=DEFAULT_BVH_SCALE, help="BVH import scale factor.")
    parser.add_argument("--mirror", action="store_true", help="Add mirrored samples.")
    parser.add_argument("--mirror-axis", choices=("x", "y", "z"), default=DEFAULT_MIRROR_AXIS, help="World axis reflected by mirror augmentation.")
    parser.add_argument("--label-source", choices=MM_LABEL_SOURCES, default=MM_DEFAULT_LABEL_SOURCE, help="Use clip-level or LabelModule action hints.")
    parser.add_argument("--min-action-weight", type=float, default=1e-4, help="Minimum auto-label action weight required for a frame.")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    clip_specs = args.clip if args.clip else MM_DEFAULT_CLIP_SPECS

    dataset, layout, feature_mean, feature_std, features = build_dataset(
        dataset_dir=args.dataset_dir,
        clip_specs=clip_specs,
        scale=args.scale,
        future_sample_offsets=MM_FUTURE_SAMPLE_OFFSETS,
        label_source=args.label_source,
        min_action_weight=args.min_action_weight,
        mirror=args.mirror,
        mirror_axis=args.mirror_axis,
    )
    save_dataset_npz(
        args.output,
        dataset,
        layout,
        feature_mean,
        feature_std,
        features,
        dt=DEFAULT_BVH_FRAME_TIME,
        future_sample_offsets=MM_FUTURE_SAMPLE_OFFSETS,
    )

    action_counts = np.bincount(dataset.action_ids, minlength=len(MM_ACTION_LABELS))
    print(f"Saved Motion Matching database to {args.output}")
    print(f"Samples: {len(dataset.action_ids)}")
    print(f"Feature dim: {features.shape[1]}")
    print(f"Future offsets: {tuple(MM_FUTURE_SAMPLE_OFFSETS)}")
    print(f"Actions: {dict(zip(MM_ACTION_LABELS, action_counts.tolist()))}")
    print(f"Ranges: {len(dataset.range_starts)}")


if __name__ == "__main__":
    main()
