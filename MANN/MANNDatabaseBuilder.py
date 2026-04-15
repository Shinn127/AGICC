from dataclasses import dataclass
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import os
import sys

MANN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MANN_ROOT
if REPO_ROOT.name == "MANN":
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable

from genoview.modules.BVHImporter import BVHImporter
from genoview.modules.ContactModule import BuildContactData
from genoview.modules.LabelModule import (
    LABEL_TO_INDEX as LABEL_MODULE_LABEL_TO_INDEX,
    BuildAutoFrameLabels,
    LoadLabelAnnotations,
)
from genoview.modules.MotionMirror import DEFAULT_MIRROR_AXIS, MirrorBVHAnimation
from MANN.HumanoidLocomotionConfig import (
    HUMANOID_LOCOMOTION_ACTION_LABELS,
    HUMANOID_LOCOMOTION_ACTION_PREFIX_TO_LABEL,
    HUMANOID_LOCOMOTION_GATING_JOINTS,
    HUMANOID_LOCOMOTION_PREDICTION_JOINTS,
    HUMANOID_LOCOMOTION_TRAJECTORY_CURRENT_SAMPLE_INDEX,
    HUMANOID_LOCOMOTION_TRAJECTORY_FUTURE_SAMPLE_INDICES,
    HUMANOID_LOCOMOTION_TRAJECTORY_SAMPLE_OFFSETS,
)
from genoview.modules.PoseModule import BuildLocalPose, BuildPoseSource
from genoview.modules.RootModule import (
    DEFAULT_BVH_FRAME_TIME,
    ROOT_TRAJECTORY_MODE_FLAT,
    BuildRootLocalTrajectory,
    BuildRootTrajectorySource,
)


DEFAULT_OUTPUT_DIR = Path("MANN/output/mann")
DEFAULT_LAFAN1_DIR = Path("../bvh/lafan1")
DEFAULT_LOCOMOTION_CLIP_SPECS = (
    ("walk1_subject5", None, None, None),
    ("run1_subject5", None, None, None),
    ("jumps1_subject1", None, None, None),
    ("pushAndStumble1_subject5", "idle", 260, 700),
)
DEFAULT_LOCOMOTION_CLIP_STEMS = tuple(
    clip_spec if isinstance(clip_spec, str) else clip_spec[0]
    for clip_spec in DEFAULT_LOCOMOTION_CLIP_SPECS
)
DEFAULT_MIN_MANN_ACTION_WEIGHT = 1e-4


@dataclass(frozen=True)
class MANNClipDatabase:
    x_main: np.ndarray
    x_gate: np.ndarray
    y: np.ndarray
    action_ids: np.ndarray
    clip_names: np.ndarray
    frame_indices: np.ndarray
    mirror_flags: np.ndarray
    variant_names: np.ndarray


def _default_worker_count():
    return max(1, os.cpu_count() or 1)


def _resolve_worker_count(num_workers):
    if num_workers is None or int(num_workers) <= 0:
        return _default_worker_count()
    return max(1, int(num_workers))


def _run_parallel_jobs(job_fn, job_specs, num_workers, show_progress, progress_desc):
    progress_bar = tqdm(total=len(job_specs), desc=progress_desc, leave=True) if show_progress else None

    try:
        executor_cls = ProcessPoolExecutor
        with executor_cls(max_workers=num_workers) as executor:
            futures = {
                executor.submit(job_fn, *job_spec): index
                for index, job_spec in enumerate(job_specs)
            }
            results = [None] * len(job_specs)
            for future in as_completed(futures):
                results[futures[future]] = future.result()
                if progress_bar is not None:
                    progress_bar.update(1)
    except (PermissionError, OSError):
        if progress_bar is not None:
            progress_bar.close()
        progress_bar = tqdm(total=len(job_specs), desc=f"{progress_desc} [threads]", leave=True) if show_progress else None
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(job_fn, *job_spec): index
                for index, job_spec in enumerate(job_specs)
            }
            results = [None] * len(job_specs)
            for future in as_completed(futures):
                results[futures[future]] = future.result()
                if progress_bar is not None:
                    progress_bar.update(1)

    if progress_bar is not None:
        progress_bar.close()

    return results


def _resolve_joint_indices(joint_names, selected_joint_names):
    joint_name_to_index = {joint_name: index for index, joint_name in enumerate(joint_names)}
    missing = [joint_name for joint_name in selected_joint_names if joint_name not in joint_name_to_index]
    if missing:
        raise ValueError(f"Missing joints in clip skeleton: {missing}")
    return np.asarray([joint_name_to_index[joint_name] for joint_name in selected_joint_names], dtype=np.int32)


def resolve_locomotion_action_label(clip_path):
    clip_name = Path(clip_path).stem.lower()
    for prefix, action_label in HUMANOID_LOCOMOTION_ACTION_PREFIX_TO_LABEL:
        if clip_name.startswith(str(prefix).lower()):
            return action_label
    return None


def _label_key_from_clip_path(clip_path):
    clip_path = Path(clip_path)
    parts = [str(part) for part in clip_path.parts]
    lower_parts = [part.lower() for part in parts]
    if "bvh" in lower_parts:
        bvh_index = lower_parts.index("bvh")
        return Path(*parts[bvh_index:]).as_posix()
    return clip_path.stem


def _normalize_locomotion_clip_spec(clip_spec):
    if isinstance(clip_spec, str):
        return clip_spec, None, None, None
    if len(clip_spec) != 4:
        raise ValueError(
            "Locomotion clip specs must be either a clip stem string or "
            "(clip_stem, action_label, frame_start, frame_end)."
        )
    clip_stem, action_label, frame_start, frame_end = clip_spec
    if frame_start is not None and frame_end is not None and int(frame_end) < int(frame_start):
        raise ValueError(f"Invalid frame range for {clip_stem}: {frame_start}-{frame_end}.")
    return str(clip_stem), action_label, frame_start, frame_end


def list_locomotion_clips(dataset_dir=DEFAULT_LAFAN1_DIR):
    dataset_dir = Path(dataset_dir)
    clip_specs = []
    normalized_clip_specs = [
        _normalize_locomotion_clip_spec(clip_spec)
        for clip_spec in DEFAULT_LOCOMOTION_CLIP_SPECS
    ]
    missing_clip_stems = [
        clip_stem
        for clip_stem, _, _, _ in normalized_clip_specs
        if not (dataset_dir / f"{clip_stem}.bvh").exists()
    ]
    if missing_clip_stems:
        raise FileNotFoundError(
            "Missing required locomotion BVH clips in "
            f"{dataset_dir}: {missing_clip_stems}"
        )

    for clip_stem, action_label_override, frame_start, frame_end in normalized_clip_specs:
        clip_path = dataset_dir / f"{clip_stem}.bvh"
        action_label = action_label_override or resolve_locomotion_action_label(clip_path)
        if action_label is None:
            continue
        clip_specs.append((clip_path, action_label, frame_start, frame_end))
    return clip_specs


def make_action_one_hot(action_label):
    action_index = HUMANOID_LOCOMOTION_ACTION_LABELS.index(action_label)
    one_hot = np.zeros(len(HUMANOID_LOCOMOTION_ACTION_LABELS), dtype=np.float32)
    one_hot[action_index] = 1.0
    return one_hot, action_index


def build_constant_action_weights(action_label, frame_count):
    action_one_hot, _ = make_action_one_hot(action_label)
    return np.repeat(action_one_hot[np.newaxis, :], int(frame_count), axis=0).astype(np.float32), np.ones(
        (int(frame_count),),
        dtype=bool,
    )


def _project_label_module_action_weights(label_result, min_action_weight=DEFAULT_MIN_MANN_ACTION_WEIGHT):
    if label_result.soft_weights is None:
        raise ValueError("LabelModule result did not produce soft action weights.")

    missing_labels = [
        label
        for label in HUMANOID_LOCOMOTION_ACTION_LABELS
        if label not in LABEL_MODULE_LABEL_TO_INDEX
    ]
    if missing_labels:
        raise ValueError(f"LabelModule is missing required MANN action labels: {missing_labels}")

    label_weights = np.asarray(label_result.soft_weights, dtype=np.float32)
    label_indices = [LABEL_MODULE_LABEL_TO_INDEX[label] for label in HUMANOID_LOCOMOTION_ACTION_LABELS]
    action_weights = label_weights[:, label_indices].astype(np.float32)
    action_sums = np.sum(action_weights, axis=1, keepdims=True).astype(np.float32)
    valid_mask = action_sums[:, 0] >= float(min_action_weight)
    action_weights = np.divide(
        action_weights,
        np.maximum(action_sums, 1e-8),
        out=np.zeros_like(action_weights, dtype=np.float32),
    ).astype(np.float32)
    return action_weights, valid_mask.astype(bool)


def build_label_action_weights(
    clip_name_or_path,
    animation,
    dt,
    action_label_fallback,
    use_label_actions=True,
    min_action_weight=DEFAULT_MIN_MANN_ACTION_WEIGHT,
):
    if not use_label_actions:
        return build_constant_action_weights(action_label_fallback, animation.frame_count)

    joint_names = animation.raw_data["names"]
    label_root_trajectory_source = BuildRootTrajectorySource(
        animation.global_positions,
        animation.global_rotations,
        dt,
        mode=ROOT_TRAJECTORY_MODE_FLAT,
        projectToGround=True,
        groundHeight=0.0,
    )
    label_pose_source = BuildPoseSource(
        animation.global_positions,
        animation.global_rotations,
        dt,
        rootTrajectorySource=label_root_trajectory_source,
    )
    contact_data = BuildContactData(
        animation.global_positions,
        label_pose_source["global_velocities"],
        joint_names,
        bootstrap=True,
    )
    label_result = BuildAutoFrameLabels(
        clip_name_or_path,
        animation.global_positions,
        label_pose_source,
        label_root_trajectory_source,
        contactData=contact_data,
        terrainProvider=None,
        jointNames=joint_names,
    )
    LoadLabelAnnotations(label_result, clip_name_or_path)
    return _project_label_module_action_weights(
        label_result,
        min_action_weight=min_action_weight,
    )


def flatten_pose_feature(local_pose, joint_indices):
    return np.concatenate(
        [
            local_pose["local_positions"][joint_indices].reshape(-1),
            local_pose["local_rotations_6d"][joint_indices].reshape(-1),
            local_pose["local_velocities"][joint_indices].reshape(-1),
        ]
    ).astype(np.float32)


def flatten_traj_feature(root_local_trajectory):
    return np.concatenate(
        [
            root_local_trajectory["local_positions"][:, [0, 2]].reshape(-1),
            root_local_trajectory["local_directions"][:, [0, 2]].reshape(-1),
            root_local_trajectory["local_velocities"][:, [0, 2]].reshape(-1),
        ]
    ).astype(np.float32)


def flatten_future_traj_feature(
    root_local_trajectory,
    future_sample_indices=HUMANOID_LOCOMOTION_TRAJECTORY_FUTURE_SAMPLE_INDICES,
):
    future_sample_indices = np.asarray(future_sample_indices, dtype=np.int32)
    return np.concatenate(
        [
            root_local_trajectory["local_positions"][future_sample_indices][:, [0, 2]].reshape(-1),
            root_local_trajectory["local_directions"][future_sample_indices][:, [0, 2]].reshape(-1),
            root_local_trajectory["local_velocities"][future_sample_indices][:, [0, 2]].reshape(-1),
        ]
    ).astype(np.float32)


def flatten_gating_feature(local_pose, joint_indices):
    return np.concatenate(
        [
            local_pose["local_positions"][joint_indices].reshape(-1),
            local_pose["local_velocities"][joint_indices].reshape(-1),
        ]
    ).astype(np.float32)


def build_speed_horizon(root_local_trajectory):
    local_velocities_xz = root_local_trajectory["local_velocities"][:, [0, 2]]
    return np.linalg.norm(local_velocities_xz, axis=-1).astype(np.float32)


def build_action_horizon(action_weights, current_frame, sample_offsets):
    sample_frames = int(current_frame) + np.asarray(sample_offsets, dtype=np.int32)
    return np.asarray(action_weights[sample_frames], dtype=np.float32).reshape(-1)


def build_root_delta_target(local_pose):
    return np.asarray(
        [
            local_pose["root_local_velocity"][0],
            local_pose["root_local_velocity"][2],
            local_pose["root_local_angular_velocity"][1],
        ],
        dtype=np.float32,
    )


def build_target_vector(local_pose, prediction_joint_indices, include_future=False, future_feature=None):
    feature_parts = [
        flatten_pose_feature(local_pose, prediction_joint_indices),
        build_root_delta_target(local_pose),
    ]
    if include_future:
        if future_feature is None:
            raise ValueError("future_feature must be provided when include_future=True.")
        feature_parts.append(np.asarray(future_feature, dtype=np.float32).reshape(-1))
    return np.concatenate(feature_parts).astype(np.float32)


def get_valid_frame_range(
    frame_count,
    sample_offsets,
    require_next_frame=False,
    frame_start=None,
    frame_end=None,
):
    sample_offsets = np.asarray(sample_offsets, dtype=np.int32)
    lower_bound = 0 if frame_start is None else max(0, int(frame_start))
    upper_bound = frame_count - 1 if frame_end is None else min(frame_count - 1, int(frame_end))
    first_valid_frame = max(lower_bound + 1, lower_bound + int(-np.min(sample_offsets)))
    max_frame = upper_bound - 1 if require_next_frame else upper_bound
    last_valid_frame = min(max_frame, max_frame - int(np.max(sample_offsets)))
    return first_valid_frame, last_valid_frame


def _empty_clip_database():
    return MANNClipDatabase(
        x_main=np.zeros((0, 0), dtype=np.float32),
        x_gate=np.zeros((0, 0), dtype=np.float32),
        y=np.zeros((0, 0), dtype=np.float32),
        action_ids=np.zeros((0,), dtype=np.int32),
        clip_names=np.asarray([]),
        frame_indices=np.zeros((0,), dtype=np.int32),
        mirror_flags=np.zeros((0,), dtype=np.uint8),
        variant_names=np.asarray([]),
    )


def _build_motion_sample_database(
    animation,
    clip_name,
    variant_name,
    action_weights,
    valid_action_mask,
    frame_start,
    frame_end,
    stage,
    dt=DEFAULT_BVH_FRAME_TIME,
    sample_offsets=HUMANOID_LOCOMOTION_TRAJECTORY_SAMPLE_OFFSETS,
    future_sample_indices=HUMANOID_LOCOMOTION_TRAJECTORY_FUTURE_SAMPLE_INDICES,
    mirrored=False,
    show_progress=False,
):
    include_future = stage == "stage2"
    joint_names = animation.raw_data["names"]
    action_weights = np.asarray(action_weights, dtype=np.float32)
    valid_action_mask = np.asarray(valid_action_mask, dtype=bool)

    prediction_joint_indices = _resolve_joint_indices(joint_names, HUMANOID_LOCOMOTION_PREDICTION_JOINTS)
    gating_joint_indices = _resolve_joint_indices(joint_names, HUMANOID_LOCOMOTION_GATING_JOINTS)

    root_trajectory_source = BuildRootTrajectorySource(
        animation.global_positions,
        animation.global_rotations,
        dt,
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

    first_valid_frame, last_valid_frame = get_valid_frame_range(
        animation.frame_count,
        sample_offsets,
        require_next_frame=include_future,
        frame_start=frame_start,
        frame_end=frame_end,
    )

    if first_valid_frame > last_valid_frame:
        return _empty_clip_database()

    x_main_rows = []
    x_gate_rows = []
    y_rows = []
    action_ids = []
    clip_names = []
    frame_indices = []
    mirror_flags = []
    variant_names = []

    frame_iter = range(first_valid_frame, last_valid_frame + 1)
    if show_progress:
        frame_iter = tqdm(frame_iter, desc=f"{stage} {variant_name}", leave=False)

    for current_frame in frame_iter:
        previous_frame = current_frame - 1
        sample_frames = int(current_frame) + np.asarray(sample_offsets, dtype=np.int32)
        if (
            not valid_action_mask[current_frame]
            or not valid_action_mask[previous_frame]
            or not np.all(valid_action_mask[sample_frames])
        ):
            continue

        previous_local_pose = BuildLocalPose(pose_source, root_trajectory_source, previous_frame, dt=dt)
        current_local_pose = BuildLocalPose(pose_source, root_trajectory_source, current_frame, dt=dt)
        current_root_local_trajectory = BuildRootLocalTrajectory(
            root_trajectory_source,
            current_frame,
            sampleOffsets=sample_offsets,
        )

        speed_horizon = build_speed_horizon(current_root_local_trajectory)
        action_horizon = build_action_horizon(action_weights, current_frame, sample_offsets)
        current_action_weights = action_weights[current_frame].astype(np.float32)

        x_main_rows.append(
            np.concatenate(
                [
                    flatten_pose_feature(previous_local_pose, prediction_joint_indices),
                    flatten_traj_feature(current_root_local_trajectory),
                    speed_horizon,
                    action_horizon,
                ]
            ).astype(np.float32)
        )
        x_gate_rows.append(
            np.concatenate(
                [
                    flatten_gating_feature(previous_local_pose, gating_joint_indices),
                    current_action_weights,
                    np.asarray([speed_horizon[HUMANOID_LOCOMOTION_TRAJECTORY_CURRENT_SAMPLE_INDEX]], dtype=np.float32),
                ]
            ).astype(np.float32)
        )

        future_feature = None
        if include_future:
            next_root_local_trajectory = BuildRootLocalTrajectory(
                root_trajectory_source,
                current_frame + 1,
                sampleOffsets=sample_offsets,
            )
            future_feature = flatten_future_traj_feature(
                next_root_local_trajectory,
                future_sample_indices=future_sample_indices,
            )

        y_rows.append(
            build_target_vector(
                current_local_pose,
                prediction_joint_indices,
                include_future=include_future,
                future_feature=future_feature,
            )
        )
        action_ids.append(int(np.argmax(current_action_weights)))
        clip_names.append(clip_name)
        frame_indices.append(current_frame)
        mirror_flags.append(1 if mirrored else 0)
        variant_names.append(variant_name)

    if not x_main_rows:
        x_main = np.zeros((0, 0), dtype=np.float32)
        x_gate = np.zeros((0, 0), dtype=np.float32)
        y = np.zeros((0, 0), dtype=np.float32)
    else:
        x_main = np.stack(x_main_rows).astype(np.float32)
        x_gate = np.stack(x_gate_rows).astype(np.float32)
        y = np.stack(y_rows).astype(np.float32)

    return MANNClipDatabase(
        x_main=x_main,
        x_gate=x_gate,
        y=y,
        action_ids=np.asarray(action_ids, dtype=np.int32),
        clip_names=np.asarray(clip_names),
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
        mirror_flags=np.asarray(mirror_flags, dtype=np.uint8),
        variant_names=np.asarray(variant_names),
    )


def _build_clip_database(
    clip_path,
    action_label,
    frame_start,
    frame_end,
    stage,
    scale=0.01,
    dt=DEFAULT_BVH_FRAME_TIME,
    sample_offsets=HUMANOID_LOCOMOTION_TRAJECTORY_SAMPLE_OFFSETS,
    future_sample_indices=HUMANOID_LOCOMOTION_TRAJECTORY_FUTURE_SAMPLE_INDICES,
    use_label_actions=True,
    min_action_weight=DEFAULT_MIN_MANN_ACTION_WEIGHT,
    mirror=False,
    mirror_axis=DEFAULT_MIRROR_AXIS,
    show_progress=False,
):
    animation = BVHImporter.load(str(clip_path), scale=scale)
    clip_name = Path(clip_path).stem
    label_key = _label_key_from_clip_path(clip_path)
    action_weights, valid_action_mask = build_label_action_weights(
        label_key,
        animation,
        dt,
        action_label,
        use_label_actions=use_label_actions,
        min_action_weight=min_action_weight,
    )

    databases = [
        _build_motion_sample_database(
            animation,
            clip_name,
            clip_name,
            action_weights,
            valid_action_mask,
            frame_start,
            frame_end,
            stage,
            dt=dt,
            sample_offsets=sample_offsets,
            future_sample_indices=future_sample_indices,
            mirrored=False,
            show_progress=show_progress,
        )
    ]

    if mirror:
        mirrored_animation = MirrorBVHAnimation(animation, axis=mirror_axis)
        databases.append(
            _build_motion_sample_database(
                mirrored_animation,
                clip_name,
                f"{clip_name}_mirror",
                action_weights,
                valid_action_mask,
                frame_start,
                frame_end,
                stage,
                dt=dt,
                sample_offsets=sample_offsets,
                future_sample_indices=future_sample_indices,
                mirrored=True,
                show_progress=show_progress,
            )
        )

    return concatenate_clip_databases(databases)


def build_stage1_clip_database(
    clip_path,
    action_label,
    frame_start=None,
    frame_end=None,
    scale=0.01,
    dt=DEFAULT_BVH_FRAME_TIME,
    sample_offsets=HUMANOID_LOCOMOTION_TRAJECTORY_SAMPLE_OFFSETS,
    use_label_actions=True,
    min_action_weight=DEFAULT_MIN_MANN_ACTION_WEIGHT,
    mirror=False,
    mirror_axis=DEFAULT_MIRROR_AXIS,
    show_progress=False,
):
    return _build_clip_database(
        clip_path,
        action_label,
        frame_start,
        frame_end,
        "stage1",
        scale=scale,
        dt=dt,
        sample_offsets=sample_offsets,
        use_label_actions=use_label_actions,
        min_action_weight=min_action_weight,
        mirror=mirror,
        mirror_axis=mirror_axis,
        show_progress=show_progress,
    )


def build_stage2_clip_database(
    clip_path,
    action_label,
    frame_start=None,
    frame_end=None,
    scale=0.01,
    dt=DEFAULT_BVH_FRAME_TIME,
    sample_offsets=HUMANOID_LOCOMOTION_TRAJECTORY_SAMPLE_OFFSETS,
    future_sample_indices=HUMANOID_LOCOMOTION_TRAJECTORY_FUTURE_SAMPLE_INDICES,
    use_label_actions=True,
    min_action_weight=DEFAULT_MIN_MANN_ACTION_WEIGHT,
    mirror=False,
    mirror_axis=DEFAULT_MIRROR_AXIS,
    show_progress=False,
):
    return _build_clip_database(
        clip_path,
        action_label,
        frame_start,
        frame_end,
        "stage2",
        scale=scale,
        dt=dt,
        sample_offsets=sample_offsets,
        future_sample_indices=future_sample_indices,
        use_label_actions=use_label_actions,
        min_action_weight=min_action_weight,
        mirror=mirror,
        mirror_axis=mirror_axis,
        show_progress=show_progress,
    )


def concatenate_clip_databases(databases):
    non_empty = [database for database in databases if len(database.action_ids) > 0]
    if not non_empty:
        return _empty_clip_database()

    return MANNClipDatabase(
        x_main=np.concatenate([database.x_main for database in non_empty], axis=0).astype(np.float32),
        x_gate=np.concatenate([database.x_gate for database in non_empty], axis=0).astype(np.float32),
        y=np.concatenate([database.y for database in non_empty], axis=0).astype(np.float32),
        action_ids=np.concatenate([database.action_ids for database in non_empty], axis=0).astype(np.int32),
        clip_names=np.concatenate([database.clip_names for database in non_empty], axis=0),
        frame_indices=np.concatenate([database.frame_indices for database in non_empty], axis=0).astype(np.int32),
        mirror_flags=np.concatenate([database.mirror_flags for database in non_empty], axis=0).astype(np.uint8),
        variant_names=np.concatenate([database.variant_names for database in non_empty], axis=0),
    )


def _build_dataset(
    stage,
    dataset_dir=DEFAULT_LAFAN1_DIR,
    scale=0.01,
    dt=DEFAULT_BVH_FRAME_TIME,
    sample_offsets=HUMANOID_LOCOMOTION_TRAJECTORY_SAMPLE_OFFSETS,
    future_sample_indices=HUMANOID_LOCOMOTION_TRAJECTORY_FUTURE_SAMPLE_INDICES,
    use_label_actions=True,
    min_action_weight=DEFAULT_MIN_MANN_ACTION_WEIGHT,
    mirror=False,
    mirror_axis=DEFAULT_MIRROR_AXIS,
    show_progress=False,
    num_workers=1,
):
    clip_specs = list_locomotion_clips(dataset_dir)
    num_workers = min(_resolve_worker_count(num_workers), max(1, len(clip_specs)))
    build_fn = build_stage2_clip_database if stage == "stage2" else build_stage1_clip_database

    if num_workers == 1:
        clip_iter = clip_specs
        if show_progress:
            clip_iter = tqdm(clip_iter, desc=f"{stage} clips", leave=True)

        databases = []
        for clip_path, action_label, frame_start, frame_end in clip_iter:
            if stage == "stage2":
                databases.append(
                    build_fn(
                        clip_path,
                        action_label,
                        frame_start=frame_start,
                        frame_end=frame_end,
                        scale=scale,
                        dt=dt,
                        sample_offsets=sample_offsets,
                        future_sample_indices=future_sample_indices,
                        use_label_actions=use_label_actions,
                        min_action_weight=min_action_weight,
                        mirror=mirror,
                        mirror_axis=mirror_axis,
                        show_progress=show_progress,
                    )
                )
            else:
                databases.append(
                    build_fn(
                        clip_path,
                        action_label,
                        frame_start=frame_start,
                        frame_end=frame_end,
                        scale=scale,
                        dt=dt,
                        sample_offsets=sample_offsets,
                        use_label_actions=use_label_actions,
                        min_action_weight=min_action_weight,
                        mirror=mirror,
                        mirror_axis=mirror_axis,
                        show_progress=show_progress,
                    )
                )
        return concatenate_clip_databases(databases)

    if stage == "stage2":
        job_specs = [
            (
                clip_path,
                action_label,
                frame_start,
                frame_end,
                scale,
                dt,
                sample_offsets,
                future_sample_indices,
                use_label_actions,
                min_action_weight,
                mirror,
                mirror_axis,
                False,
            )
            for clip_path, action_label, frame_start, frame_end in clip_specs
        ]
    else:
        job_specs = [
            (
                clip_path,
                action_label,
                frame_start,
                frame_end,
                scale,
                dt,
                sample_offsets,
                use_label_actions,
                min_action_weight,
                mirror,
                mirror_axis,
                False,
            )
            for clip_path, action_label, frame_start, frame_end in clip_specs
        ]

    databases = _run_parallel_jobs(
        build_fn,
        job_specs,
        num_workers=num_workers,
        show_progress=show_progress,
        progress_desc=f"{stage} clips ({num_workers} workers)",
    )
    return concatenate_clip_databases(databases)


def build_stage1_dataset(
    dataset_dir=DEFAULT_LAFAN1_DIR,
    scale=0.01,
    dt=DEFAULT_BVH_FRAME_TIME,
    sample_offsets=HUMANOID_LOCOMOTION_TRAJECTORY_SAMPLE_OFFSETS,
    use_label_actions=True,
    min_action_weight=DEFAULT_MIN_MANN_ACTION_WEIGHT,
    mirror=False,
    mirror_axis=DEFAULT_MIRROR_AXIS,
    show_progress=False,
    num_workers=1,
):
    return _build_dataset(
        "stage1",
        dataset_dir=dataset_dir,
        scale=scale,
        dt=dt,
        sample_offsets=sample_offsets,
        use_label_actions=use_label_actions,
        min_action_weight=min_action_weight,
        mirror=mirror,
        mirror_axis=mirror_axis,
        show_progress=show_progress,
        num_workers=num_workers,
    )


def build_stage2_dataset(
    dataset_dir=DEFAULT_LAFAN1_DIR,
    scale=0.01,
    dt=DEFAULT_BVH_FRAME_TIME,
    sample_offsets=HUMANOID_LOCOMOTION_TRAJECTORY_SAMPLE_OFFSETS,
    future_sample_indices=HUMANOID_LOCOMOTION_TRAJECTORY_FUTURE_SAMPLE_INDICES,
    use_label_actions=True,
    min_action_weight=DEFAULT_MIN_MANN_ACTION_WEIGHT,
    mirror=False,
    mirror_axis=DEFAULT_MIRROR_AXIS,
    show_progress=False,
    num_workers=1,
):
    return _build_dataset(
        "stage2",
        dataset_dir=dataset_dir,
        scale=scale,
        dt=dt,
        sample_offsets=sample_offsets,
        future_sample_indices=future_sample_indices,
        use_label_actions=use_label_actions,
        min_action_weight=min_action_weight,
        mirror=mirror,
        mirror_axis=mirror_axis,
        show_progress=show_progress,
        num_workers=num_workers,
    )


def build_database_metadata(stage):
    action_dim = len(HUMANOID_LOCOMOTION_ACTION_LABELS)
    prediction_joint_count = len(HUMANOID_LOCOMOTION_PREDICTION_JOINTS)
    sample_count = len(HUMANOID_LOCOMOTION_TRAJECTORY_SAMPLE_OFFSETS)
    future_count = len(HUMANOID_LOCOMOTION_TRAJECTORY_FUTURE_SAMPLE_INDICES)

    x_main_pose_dim = prediction_joint_count * (3 + 6 + 3)
    x_main_traj_dim = sample_count * (2 + 2 + 2)
    x_main_speed_dim = sample_count
    x_main_action_dim = sample_count * action_dim
    # Historical metadata name kept for loader compatibility; this block now
    # stores gating joint local positions followed by local velocities.
    x_gate_vel_dim = len(HUMANOID_LOCOMOTION_GATING_JOINTS) * (3 + 3)
    x_gate_action_dim = action_dim
    x_gate_speed_dim = 1
    y_pose_dim = prediction_joint_count * (3 + 6 + 3)
    y_root_dim = 3
    y_future_dim = future_count * (2 + 2 + 2) if stage == "stage2" else 0

    return {
        "stage": np.asarray(stage),
        "x_main_pose_dim": np.asarray(x_main_pose_dim, dtype=np.int32),
        "x_main_traj_dim": np.asarray(x_main_traj_dim, dtype=np.int32),
        "x_main_speed_dim": np.asarray(x_main_speed_dim, dtype=np.int32),
        "x_main_action_dim": np.asarray(x_main_action_dim, dtype=np.int32),
        "x_gate_vel_dim": np.asarray(x_gate_vel_dim, dtype=np.int32),
        "x_gate_action_dim": np.asarray(x_gate_action_dim, dtype=np.int32),
        "x_gate_speed_dim": np.asarray(x_gate_speed_dim, dtype=np.int32),
        "y_pose_dim": np.asarray(y_pose_dim, dtype=np.int32),
        "y_root_dim": np.asarray(y_root_dim, dtype=np.int32),
        "y_future_dim": np.asarray(y_future_dim, dtype=np.int32),
    }


def save_dataset_npz(output_path, dataset, stage):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = build_database_metadata(stage)
    np.savez_compressed(
        output_path,
        x_main=dataset.x_main,
        x_gate=dataset.x_gate,
        y=dataset.y,
        action_ids=dataset.action_ids,
        clip_names=dataset.clip_names,
        frame_indices=dataset.frame_indices,
        mirror_flags=dataset.mirror_flags,
        variant_names=dataset.variant_names,
        action_labels=np.asarray(HUMANOID_LOCOMOTION_ACTION_LABELS),
        trajectory_sample_offsets=np.asarray(HUMANOID_LOCOMOTION_TRAJECTORY_SAMPLE_OFFSETS, dtype=np.int32),
        trajectory_future_sample_indices=np.asarray(HUMANOID_LOCOMOTION_TRAJECTORY_FUTURE_SAMPLE_INDICES, dtype=np.int32),
        **metadata,
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Build the MANN locomotion database.")
    parser.add_argument("--stage", choices=("stage1", "stage2"), default="stage2", help="Database target set to build.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_LAFAN1_DIR, help="Directory containing LaFAN1 BVH clips.")
    parser.add_argument("--output", type=Path, default=None, help="Destination .npz file.")
    parser.add_argument("--scale", type=float, default=0.01, help="BVH import scale factor.")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes for clip-level export. Use 0 for auto.")
    parser.add_argument("--mirror", action="store_true", help="Add mirrored motion samples for each clip.")
    parser.add_argument("--mirror-axis", choices=("x", "y", "z"), default=DEFAULT_MIRROR_AXIS, help="World axis reflected by mirror augmentation.")
    parser.add_argument("--no-label-actions", action="store_true", help="Use clip-level hard action labels instead of LabelModule frame labels.")
    parser.add_argument("--min-action-weight", type=float, default=DEFAULT_MIN_MANN_ACTION_WEIGHT, help="Minimum projected 4-class LabelModule action weight required for a frame.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars during database export.")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    output_path = args.output or DEFAULT_OUTPUT_DIR / f"{args.stage}_locomotion_database.npz"
    show_progress = not args.no_progress

    if args.stage == "stage2":
        dataset = build_stage2_dataset(
            dataset_dir=args.dataset_dir,
            scale=args.scale,
            use_label_actions=not args.no_label_actions,
            min_action_weight=args.min_action_weight,
            mirror=args.mirror,
            mirror_axis=args.mirror_axis,
            show_progress=show_progress,
            num_workers=args.workers,
        )
    else:
        dataset = build_stage1_dataset(
            dataset_dir=args.dataset_dir,
            scale=args.scale,
            use_label_actions=not args.no_label_actions,
            min_action_weight=args.min_action_weight,
            mirror=args.mirror,
            mirror_axis=args.mirror_axis,
            show_progress=show_progress,
            num_workers=args.workers,
        )

    save_dataset_npz(output_path, dataset, stage=args.stage)

    print(f"Saved database to {output_path}")
    print(f"Samples: {len(dataset.action_ids)}")
    print(f"x_main shape: {dataset.x_main.shape}")
    print(f"x_gate shape: {dataset.x_gate.shape}")
    print(f"y shape: {dataset.y.shape}")


if __name__ == "__main__":
    main()
