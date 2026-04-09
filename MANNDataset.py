from dataclasses import dataclass
from pathlib import Path
import argparse

import numpy as np

from BVHImporter import BVHImporter
from HumanoidLocomotionConfig import (
    HUMANOID_LOCOMOTION_ACTION_LABELS,
    HUMANOID_LOCOMOTION_ACTION_PREFIX_TO_LABEL,
    HUMANOID_LOCOMOTION_GATING_JOINTS,
    HUMANOID_LOCOMOTION_PREDICTION_JOINTS,
    HUMANOID_LOCOMOTION_TRAJECTORY_CURRENT_SAMPLE_INDEX,
    HUMANOID_LOCOMOTION_TRAJECTORY_SAMPLE_OFFSETS,
)
from PoseModule import BuildLocalPose, BuildPoseSource
from RootModule import (
    DEFAULT_BVH_FRAME_TIME,
    ROOT_TRAJECTORY_MODE_FLAT,
    BuildRootLocalTrajectory,
    BuildRootTrajectorySource,
)


DEFAULT_LAFAN1_DIR = Path("resources/bvh/lafan1")


@dataclass(frozen=True)
class Stage1ClipDataset:
    x_main: np.ndarray
    x_gate: np.ndarray
    y_pose: np.ndarray
    y_root: np.ndarray
    action_ids: np.ndarray
    clip_names: np.ndarray
    frame_indices: np.ndarray


def _resolve_joint_indices(joint_names, selected_joint_names):
    joint_name_to_index = {joint_name: index for index, joint_name in enumerate(joint_names)}
    missing = [joint_name for joint_name in selected_joint_names if joint_name not in joint_name_to_index]
    if missing:
        raise ValueError(f"Missing joints in clip skeleton: {missing}")
    return np.asarray([joint_name_to_index[joint_name] for joint_name in selected_joint_names], dtype=np.int32)


def resolve_locomotion_action_label(clip_path):
    clip_name = Path(clip_path).stem
    for prefix, action_label in HUMANOID_LOCOMOTION_ACTION_PREFIX_TO_LABEL:
        if clip_name.startswith(prefix):
            return action_label
    return None


def list_locomotion_clips(dataset_dir=DEFAULT_LAFAN1_DIR):
    dataset_dir = Path(dataset_dir)
    clips = []
    for clip_path in sorted(dataset_dir.glob("*.bvh")):
        action_label = resolve_locomotion_action_label(clip_path)
        if action_label is None:
            continue
        clips.append((clip_path, action_label))
    return clips


def make_action_one_hot(action_label):
    action_index = HUMANOID_LOCOMOTION_ACTION_LABELS.index(action_label)
    one_hot = np.zeros(len(HUMANOID_LOCOMOTION_ACTION_LABELS), dtype=np.float32)
    one_hot[action_index] = 1.0
    return one_hot, action_index


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


def build_speed_horizon(root_local_trajectory):
    local_velocities_xz = root_local_trajectory["local_velocities"][:, [0, 2]]
    return np.linalg.norm(local_velocities_xz, axis=-1).astype(np.float32)


def build_action_horizon(action_one_hot, sample_count):
    tiled = np.repeat(action_one_hot[np.newaxis, :], sample_count, axis=0)
    return tiled.reshape(-1).astype(np.float32)


def build_root_delta_target(local_pose):
    return np.asarray(
        [
            local_pose["root_local_velocity"][0],
            local_pose["root_local_velocity"][2],
            local_pose["root_local_angular_velocity"][1],
        ],
        dtype=np.float32,
    )


def get_valid_frame_range(frame_count, sample_offsets):
    sample_offsets = np.asarray(sample_offsets, dtype=np.int32)
    first_valid_frame = max(1, int(-np.min(sample_offsets)))
    last_valid_frame = min(frame_count - 1, frame_count - 1 - int(np.max(sample_offsets)))
    return first_valid_frame, last_valid_frame


def build_stage1_clip_dataset(
    clip_path,
    action_label,
    scale=0.01,
    dt=DEFAULT_BVH_FRAME_TIME,
    sample_offsets=HUMANOID_LOCOMOTION_TRAJECTORY_SAMPLE_OFFSETS,
):
    animation = BVHImporter.load(str(clip_path), scale=scale)
    joint_names = animation.raw_data["names"]

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

    action_one_hot, action_id = make_action_one_hot(action_label)
    first_valid_frame, last_valid_frame = get_valid_frame_range(animation.frame_count, sample_offsets)

    x_main_rows = []
    x_gate_rows = []
    y_pose_rows = []
    y_root_rows = []
    action_ids = []
    clip_names = []
    frame_indices = []

    for current_frame in range(first_valid_frame, last_valid_frame + 1):
        previous_frame = current_frame - 1

        previous_local_pose = BuildLocalPose(
            pose_source,
            root_trajectory_source,
            previous_frame,
            dt=dt,
        )
        current_local_pose = BuildLocalPose(
            pose_source,
            root_trajectory_source,
            current_frame,
            dt=dt,
        )
        root_local_trajectory = BuildRootLocalTrajectory(
            root_trajectory_source,
            current_frame,
            sampleOffsets=sample_offsets,
        )

        speed_horizon = build_speed_horizon(root_local_trajectory)
        action_horizon = build_action_horizon(action_one_hot, len(sample_offsets))

        x_main_rows.append(
            np.concatenate(
                [
                    flatten_pose_feature(previous_local_pose, prediction_joint_indices),
                    flatten_traj_feature(root_local_trajectory),
                    speed_horizon,
                    action_horizon,
                ]
            ).astype(np.float32)
        )
        x_gate_rows.append(
            np.concatenate(
                [
                    previous_local_pose["local_velocities"][gating_joint_indices].reshape(-1).astype(np.float32),
                    action_one_hot,
                    np.asarray([speed_horizon[HUMANOID_LOCOMOTION_TRAJECTORY_CURRENT_SAMPLE_INDEX]], dtype=np.float32),
                ]
            ).astype(np.float32)
        )
        y_pose_rows.append(flatten_pose_feature(current_local_pose, prediction_joint_indices))
        y_root_rows.append(build_root_delta_target(current_local_pose))
        action_ids.append(action_id)
        clip_names.append(Path(clip_path).stem)
        frame_indices.append(current_frame)

    if not x_main_rows:
        x_main = np.zeros((0, 0), dtype=np.float32)
        x_gate = np.zeros((0, 0), dtype=np.float32)
        y_pose = np.zeros((0, 0), dtype=np.float32)
        y_root = np.zeros((0, 0), dtype=np.float32)
    else:
        x_main = np.stack(x_main_rows).astype(np.float32)
        x_gate = np.stack(x_gate_rows).astype(np.float32)
        y_pose = np.stack(y_pose_rows).astype(np.float32)
        y_root = np.stack(y_root_rows).astype(np.float32)

    return Stage1ClipDataset(
        x_main=x_main,
        x_gate=x_gate,
        y_pose=y_pose,
        y_root=y_root,
        action_ids=np.asarray(action_ids, dtype=np.int32),
        clip_names=np.asarray(clip_names),
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
    )


def concatenate_stage1_datasets(datasets):
    non_empty_datasets = [dataset for dataset in datasets if len(dataset.action_ids) > 0]
    if not non_empty_datasets:
        return Stage1ClipDataset(
            x_main=np.zeros((0, 0), dtype=np.float32),
            x_gate=np.zeros((0, 0), dtype=np.float32),
            y_pose=np.zeros((0, 0), dtype=np.float32),
            y_root=np.zeros((0, 0), dtype=np.float32),
            action_ids=np.zeros((0,), dtype=np.int32),
            clip_names=np.asarray([]),
            frame_indices=np.zeros((0,), dtype=np.int32),
        )

    return Stage1ClipDataset(
        x_main=np.concatenate([dataset.x_main for dataset in non_empty_datasets], axis=0).astype(np.float32),
        x_gate=np.concatenate([dataset.x_gate for dataset in non_empty_datasets], axis=0).astype(np.float32),
        y_pose=np.concatenate([dataset.y_pose for dataset in non_empty_datasets], axis=0).astype(np.float32),
        y_root=np.concatenate([dataset.y_root for dataset in non_empty_datasets], axis=0).astype(np.float32),
        action_ids=np.concatenate([dataset.action_ids for dataset in non_empty_datasets], axis=0).astype(np.int32),
        clip_names=np.concatenate([dataset.clip_names for dataset in non_empty_datasets], axis=0),
        frame_indices=np.concatenate([dataset.frame_indices for dataset in non_empty_datasets], axis=0).astype(np.int32),
    )


def build_stage1_dataset(
    dataset_dir=DEFAULT_LAFAN1_DIR,
    scale=0.01,
    dt=DEFAULT_BVH_FRAME_TIME,
    sample_offsets=HUMANOID_LOCOMOTION_TRAJECTORY_SAMPLE_OFFSETS,
):
    clip_datasets = []
    for clip_path, action_label in list_locomotion_clips(dataset_dir):
        clip_datasets.append(
            build_stage1_clip_dataset(
                clip_path,
                action_label,
                scale=scale,
                dt=dt,
                sample_offsets=sample_offsets,
            )
        )
    return concatenate_stage1_datasets(clip_datasets)


def save_stage1_dataset_npz(output_path, dataset):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        x_main=dataset.x_main,
        x_gate=dataset.x_gate,
        y_pose=dataset.y_pose,
        y_root=dataset.y_root,
        action_ids=dataset.action_ids,
        clip_names=dataset.clip_names,
        frame_indices=dataset.frame_indices,
        action_labels=np.asarray(HUMANOID_LOCOMOTION_ACTION_LABELS),
        trajectory_sample_offsets=np.asarray(HUMANOID_LOCOMOTION_TRAJECTORY_SAMPLE_OFFSETS, dtype=np.int32),
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Build the stage-1 MANN locomotion dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_LAFAN1_DIR,
        help="Directory containing LaFAN1 BVH clips.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/mann/stage1_locomotion_dataset.npz"),
        help="Destination .npz file.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.01,
        help="BVH import scale factor.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset = build_stage1_dataset(
        dataset_dir=args.dataset_dir,
        scale=args.scale,
    )
    save_stage1_dataset_npz(args.output, dataset)

    print(f"Saved dataset to {args.output}")
    print(f"Samples: {len(dataset.action_ids)}")
    print(f"x_main shape: {dataset.x_main.shape}")
    print(f"x_gate shape: {dataset.x_gate.shape}")
    print(f"y_pose shape: {dataset.y_pose.shape}")
    print(f"y_root shape: {dataset.y_root.shape}")


if __name__ == "__main__":
    main()
