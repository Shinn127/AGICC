from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import sys
from types import SimpleNamespace

MANN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MANN_ROOT
if REPO_ROOT.name == "MANN":
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from raylib import *
from raylib.defines import *
from pyray import Vector2, Vector3, Color, Rectangle

from genoview.modules.BVHImporter import BVHImporter
from genoview.modules.CameraController import Camera
from genoview.modules.CharacterModel import (
    LoadCharacterModel,
    LoadSceneResources,
    UpdateModelPoseFromNumpyArrays,
)
from genoview.utils.DebugDraw import DrawRootTrajectoryDebug
from MANN.HumanoidLocomotionConfig import (
    HUMANOID_LOCOMOTION_ACTION_LABELS,
    HUMANOID_LOCOMOTION_GATING_JOINTS,
    HUMANOID_LOCOMOTION_PREDICTION_JOINTS,
    HUMANOID_LOCOMOTION_TRAJECTORY_CURRENT_SAMPLE_INDEX,
    HUMANOID_LOCOMOTION_TRAJECTORY_FUTURE_SAMPLE_INDICES,
    HUMANOID_LOCOMOTION_TRAJECTORY_SAMPLE_OFFSETS,
)
from MANN.MANNDataset import MANNDataSpec, MANNFeatureStats
from MANN.MANNModel import MANN, MANNModelConfig
from genoview.modules.PoseModule import BuildLocalPose, BuildPoseSource, ReconstructPoseWorldSpace
from genoview.modules.RootModule import (
    ROOT_JOINT_INDEX,
    ROOT_TRAJECTORY_MODE_FLAT,
    DEFAULT_BVH_FRAME_TIME,
    BuildRootTrajectorySource,
)
from genoview.utils import quat
from genoview.modules.RenderModule import (
    BeginGBuffer,
    BeginShadowMap,
    CreateRenderResources,
    EndGBuffer,
    EndShadowMap,
    LoadShaderResources,
    SetShaderValueShadowMap,
    UnloadGBuffer,
    UnloadShadowMap,
    _draw_model_with_shader,
    _make_float_ptr,
    _render_final_pass,
    _render_ssao_and_blur_pass,
    ffi,
)


DEFAULT_MANN_OUTPUT_DIR = MANN_ROOT / "output" / "mann"
DEFAULT_DATABASE_PATH = DEFAULT_MANN_OUTPUT_DIR / "stage2_locomotion_database.npz"
DEFAULT_CHECKPOINT_PATH = DEFAULT_MANN_OUTPUT_DIR / "checkpoints_stage2" / "best.pt"
DEFAULT_STATS_PATH = DEFAULT_MANN_OUTPUT_DIR / "checkpoints_stage2" / "stats.npz"
DEFAULT_VIEWER_CLIP = Path("bvh/Geno_stance.bvh")
DEFAULT_INITIAL_FRAME = 0
DEFAULT_SCREEN_WIDTH = 1280
DEFAULT_SCREEN_HEIGHT = 720
DEFAULT_WALK_SPEED = 1.5
DEFAULT_RUN_SPEED = 3.0
DEFAULT_MOVE_HALFLIFE = 0.2
DEFAULT_ROTATION_HALFLIFE = 0.15
DEFAULT_GAMEPAD_ID = 0
DEFAULT_GAMEPAD_DEADZONE = 0.2
DEFAULT_TRAJECTORY_BUFFER_BLEND = 0.35
DEFAULT_Y_FUTURE_BLEND = 0.5
RESOURCES_DIR = REPO_ROOT / "resources"


def resource_path(*parts, as_bytes=False):
    path = RESOURCES_DIR.joinpath(*parts)
    return str(path).encode("utf-8") if as_bytes else str(path)


@dataclass(frozen=True)
class RuntimePaths:
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH
    stats_path: Path = DEFAULT_STATS_PATH
    database_path: Path = DEFAULT_DATABASE_PATH


@dataclass(frozen=True)
class RuntimePrediction:
    y: np.ndarray
    y_pose: np.ndarray
    y_root: np.ndarray
    y_future: np.ndarray | None
    expert_weights: np.ndarray | None = None


@dataclass(frozen=True)
class ViewerConfig:
    clip_path: Path = DEFAULT_VIEWER_CLIP
    initial_frame: int = DEFAULT_INITIAL_FRAME
    screen_width: int = DEFAULT_SCREEN_WIDTH
    screen_height: int = DEFAULT_SCREEN_HEIGHT
    gamepad_id: int = DEFAULT_GAMEPAD_ID


@dataclass
class RawInputState:
    input_source: str = "keyboard"
    move_2d: np.ndarray | None = None
    look_2d: np.ndarray | None = None
    look_active: bool = False
    run_pressed: bool = False
    desired_strafe: bool = False
    jump_pressed: bool = False
    reset_pressed: bool = False

    def __post_init__(self) -> None:
        if self.move_2d is None:
            self.move_2d = np.zeros(2, dtype=np.float32)
        if self.look_2d is None:
            self.look_2d = np.zeros(2, dtype=np.float32)


@dataclass
class ControlIntent:
    desired_velocity_world: np.ndarray
    desired_facing_world: np.ndarray
    desired_rotation: np.ndarray
    action_label: str
    desired_strafe: bool
    move_magnitude: float


@dataclass
class RuntimeState:
    root_position: np.ndarray
    root_rotation: np.ndarray
    root_velocity: np.ndarray
    root_acceleration: np.ndarray
    root_angular_velocity: np.ndarray
    previous_local_pose: dict
    action_label: str
    action_one_hot: np.ndarray
    desired_strafe: bool
    trajectory_positions_world: np.ndarray
    trajectory_directions_world: np.ndarray
    trajectory_velocities_world: np.ndarray
    sample_offsets: np.ndarray
    current_sample_index: int
    dt: float
    history_positions: list
    history_rotations: list
    history_velocities: list


@dataclass(frozen=True)
class FeatureInputs:
    x_main: np.ndarray
    x_gate: np.ndarray
    root_local_trajectory: dict
    speed_horizon: np.ndarray
    action_horizon: np.ndarray


@dataclass(frozen=True)
class RuntimeDebugPrediction:
    y: np.ndarray
    y_pose: np.ndarray
    y_root: np.ndarray
    y_future: np.ndarray | None
    predicted_local_pose: dict
    predicted_local_positions_offset: np.ndarray


class FeatureBuilder:
    def __init__(self, joint_names, parents, sample_offsets=HUMANOID_LOCOMOTION_TRAJECTORY_SAMPLE_OFFSETS):
        self.joint_names = list(joint_names)
        self.parents = np.asarray(parents, dtype=np.int32)
        self.sample_offsets = np.asarray(sample_offsets, dtype=np.int32)
        self.prediction_joint_indices = self._resolve_joint_indices(HUMANOID_LOCOMOTION_PREDICTION_JOINTS)
        self.gating_joint_indices = self._resolve_joint_indices(HUMANOID_LOCOMOTION_GATING_JOINTS)
        self.prediction_joint_mask = np.zeros(len(self.joint_names), dtype=bool)
        self.prediction_joint_mask[self.prediction_joint_indices] = True
        self.closest_predicted_ancestors = self._compute_closest_predicted_ancestors()

    def _resolve_joint_indices(self, selected_joint_names):
        joint_name_to_index = {joint_name: index for index, joint_name in enumerate(self.joint_names)}
        missing = [joint_name for joint_name in selected_joint_names if joint_name not in joint_name_to_index]
        if missing:
            raise ValueError(f"Missing joints in clip skeleton: {missing}")
        return np.asarray([joint_name_to_index[joint_name] for joint_name in selected_joint_names], dtype=np.int32)

    def _compute_closest_predicted_ancestors(self):
        closest_ancestors = np.full(len(self.joint_names), -1, dtype=np.int32)
        for joint_index in range(len(self.joint_names)):
            parent_index = int(self.parents[joint_index])
            while parent_index >= 0:
                if self.prediction_joint_mask[parent_index]:
                    closest_ancestors[joint_index] = parent_index
                    break
                parent_index = int(self.parents[parent_index])
        return closest_ancestors

    @staticmethod
    def flatten_pose_feature(local_pose, joint_indices):
        return np.concatenate(
            [
                local_pose["local_positions"][joint_indices].reshape(-1),
                local_pose["local_rotations_6d"][joint_indices].reshape(-1),
                local_pose["local_velocities"][joint_indices].reshape(-1),
            ]
        ).astype(np.float32)

    @staticmethod
    def flatten_traj_feature(root_local_trajectory):
        return np.concatenate(
            [
                root_local_trajectory["local_positions"][:, [0, 2]].reshape(-1),
                root_local_trajectory["local_directions"][:, [0, 2]].reshape(-1),
                root_local_trajectory["local_velocities"][:, [0, 2]].reshape(-1),
            ]
        ).astype(np.float32)

    @staticmethod
    def build_speed_horizon(root_local_trajectory):
        local_velocities_xz = root_local_trajectory["local_velocities"][:, [0, 2]]
        return np.linalg.norm(local_velocities_xz, axis=-1).astype(np.float32)

    @staticmethod
    def build_action_horizon(action_one_hot, sample_count):
        tiled = np.repeat(action_one_hot[np.newaxis, :], sample_count, axis=0)
        return tiled.reshape(-1).astype(np.float32)

    def build_root_local_trajectory(self, runtime_state: RuntimeState):
        root_position = runtime_state.root_position.astype(np.float32)
        root_rotation = runtime_state.root_rotation.astype(np.float32)
        local_positions = quat.inv_mul_vec(
            root_rotation,
            runtime_state.trajectory_positions_world - root_position[np.newaxis, :],
        ).astype(np.float32)
        local_directions = quat.inv_mul_vec(
            root_rotation,
            runtime_state.trajectory_directions_world,
        ).astype(np.float32)
        local_directions = np.asarray([_normalize_xz(direction) for direction in local_directions], dtype=np.float32)
        local_velocities = quat.inv_mul_vec(
            root_rotation,
            runtime_state.trajectory_velocities_world,
        ).astype(np.float32)

        return {
            "sample_offsets": runtime_state.sample_offsets.copy(),
            "current_root_position": root_position,
            "current_root_rotation": root_rotation,
            "local_positions": local_positions,
            "local_directions": local_directions,
            "local_velocities": local_velocities,
        }

    def build_inputs(self, runtime_state: RuntimeState) -> FeatureInputs:
        previous_local_pose = runtime_state.previous_local_pose
        root_local_trajectory = self.build_root_local_trajectory(runtime_state)
        speed_horizon = self.build_speed_horizon(root_local_trajectory)
        action_horizon = self.build_action_horizon(runtime_state.action_one_hot, len(runtime_state.sample_offsets))

        x_main = np.concatenate(
            [
                self.flatten_pose_feature(previous_local_pose, self.prediction_joint_indices),
                self.flatten_traj_feature(root_local_trajectory),
                speed_horizon,
                action_horizon,
            ]
        ).astype(np.float32)
        x_gate = np.concatenate(
            [
                previous_local_pose["local_velocities"][self.gating_joint_indices].reshape(-1).astype(np.float32),
                runtime_state.action_one_hot.astype(np.float32),
                np.asarray([speed_horizon[runtime_state.current_sample_index]], dtype=np.float32),
            ]
        ).astype(np.float32)

        return FeatureInputs(
            x_main=x_main,
            x_gate=x_gate,
            root_local_trajectory=root_local_trajectory,
            speed_horizon=speed_horizon,
            action_horizon=action_horizon,
        )


def _unpack_predicted_local_pose(
    runtime_state: RuntimeState,
    y_pose: np.ndarray,
    y_root: np.ndarray,
    feature_builder: FeatureBuilder,
) -> dict:
    joint_count = len(feature_builder.prediction_joint_indices)
    y_pose = np.asarray(y_pose, dtype=np.float32).reshape(-1)

    positions_dim = joint_count * 3
    rotations_dim = joint_count * 6
    velocities_dim = joint_count * 3

    local_positions_subset = y_pose[:positions_dim].reshape(joint_count, 3)
    local_rotations_6d_subset = y_pose[positions_dim:positions_dim + rotations_dim].reshape(joint_count, 6)
    local_velocities_subset = y_pose[positions_dim + rotations_dim:positions_dim + rotations_dim + velocities_dim].reshape(joint_count, 3)

    source_pose = runtime_state.previous_local_pose
    local_positions = np.asarray(source_pose["local_positions"], dtype=np.float32).copy()
    local_rotations_6d = np.asarray(source_pose["local_rotations_6d"], dtype=np.float32).copy()
    local_velocities = np.asarray(source_pose["local_velocities"], dtype=np.float32).copy()
    local_angular_velocities = np.asarray(source_pose["local_angular_velocities"], dtype=np.float32).copy()
    source_local_rotations = np.asarray(source_pose["local_rotations"], dtype=np.float32)

    local_positions[feature_builder.prediction_joint_indices] = local_positions_subset
    local_rotations_6d[feature_builder.prediction_joint_indices] = local_rotations_6d_subset
    local_velocities[feature_builder.prediction_joint_indices] = local_velocities_subset

    local_rotations = quat.from_xform_xy(local_rotations_6d.reshape(local_rotations_6d.shape[0], 3, 2)).astype(np.float32)

    for joint_index in range(len(local_positions)):
        if feature_builder.prediction_joint_mask[joint_index]:
            continue
        ancestor_index = int(feature_builder.closest_predicted_ancestors[joint_index])
        if ancestor_index < 0:
            continue

        delta_rotation = quat.mul(
            local_rotations[ancestor_index],
            quat.inv(source_local_rotations[ancestor_index]),
        ).astype(np.float32)
        source_offset = source_pose["local_positions"][joint_index] - source_pose["local_positions"][ancestor_index]
        local_positions[joint_index] = (
            quat.mul_vec(delta_rotation, source_offset) + local_positions[ancestor_index]
        ).astype(np.float32)
        local_rotations[joint_index] = quat.mul(delta_rotation, source_local_rotations[joint_index]).astype(np.float32)
        local_velocities[joint_index] = quat.mul_vec(delta_rotation, source_pose["local_velocities"][joint_index]).astype(np.float32)
        local_angular_velocities[joint_index] = quat.mul_vec(
            delta_rotation,
            source_pose["local_angular_velocities"][joint_index],
        ).astype(np.float32)

    local_rotations_6d = quat.to_xform_xy(local_rotations).reshape(local_rotations.shape[0], 6).astype(np.float32)

    predicted_local_pose = {
        "current_root_position": runtime_state.root_position.astype(np.float32),
        "current_root_rotation": runtime_state.root_rotation.astype(np.float32),
        "local_positions": local_positions.astype(np.float32),
        "local_rotations": local_rotations.astype(np.float32),
        "local_rotations_6d": local_rotations_6d.astype(np.float32),
        "local_velocities": local_velocities.astype(np.float32),
        "local_angular_velocities": local_angular_velocities.astype(np.float32),
        "root_local_velocity": np.asarray([y_root[0], 0.0, y_root[1]], dtype=np.float32),
        "root_local_angular_velocity": np.asarray([0.0, y_root[2], 0.0], dtype=np.float32),
    }
    return predicted_local_pose


def _run_debug_runtime_prediction(app) -> RuntimeDebugPrediction | None:
    if app.runtime is None:
        return None

    prediction = app.runtime.predict(
        app.current_features.x_main,
        app.current_features.x_gate,
        return_aux=False,
    )
    predicted_local_pose = _unpack_predicted_local_pose(
        app.runtime_state,
        prediction.y_pose,
        prediction.y_root,
        app.feature_builder,
    )
    local_debug_origin = app.local_debug_origin
    predicted_local_positions_offset = predicted_local_pose["local_positions"] + np.asarray(
        [local_debug_origin.x, local_debug_origin.y, local_debug_origin.z],
        dtype=np.float32,
    )

    return RuntimeDebugPrediction(
        y=prediction.y,
        y_pose=prediction.y_pose,
        y_root=prediction.y_root,
        y_future=prediction.y_future,
        predicted_local_pose=predicted_local_pose,
        predicted_local_positions_offset=predicted_local_positions_offset.astype(np.float32),
    )


def _apply_future_trajectory_correction(
    runtime_state: RuntimeState,
    debug_prediction: RuntimeDebugPrediction,
    blend: float = DEFAULT_Y_FUTURE_BLEND,
) -> None:
    if debug_prediction.y_future is None:
        return

    future_indices = np.asarray(HUMANOID_LOCOMOTION_TRAJECTORY_FUTURE_SAMPLE_INDICES, dtype=np.int32)
    future_feature = np.asarray(debug_prediction.y_future, dtype=np.float32).reshape(-1)
    future_count = len(future_indices)
    positions_xz = future_feature[: future_count * 2].reshape(future_count, 2)
    directions_xz = future_feature[future_count * 2 : future_count * 4].reshape(future_count, 2)
    velocities_xz = future_feature[future_count * 4 : future_count * 6].reshape(future_count, 2)

    local_positions = np.zeros((future_count, 3), dtype=np.float32)
    local_positions[:, 0] = positions_xz[:, 0]
    local_positions[:, 2] = positions_xz[:, 1]

    local_directions = np.zeros((future_count, 3), dtype=np.float32)
    local_directions[:, 0] = directions_xz[:, 0]
    local_directions[:, 2] = directions_xz[:, 1]
    local_directions = np.asarray([_normalize_xz(direction) for direction in local_directions], dtype=np.float32)

    local_velocities = np.zeros((future_count, 3), dtype=np.float32)
    local_velocities[:, 0] = velocities_xz[:, 0]
    local_velocities[:, 2] = velocities_xz[:, 1]

    root_local_velocity = np.asarray([debug_prediction.y_root[0], 0.0, debug_prediction.y_root[1]], dtype=np.float32)
    root_local_angular_velocity = np.asarray([0.0, debug_prediction.y_root[2], 0.0], dtype=np.float32)
    next_root_position, next_root_rotation = _integrate_root_motion_step(
        runtime_state.root_position,
        runtime_state.root_rotation,
        root_local_velocity,
        root_local_angular_velocity,
        runtime_state.dt,
    )

    corrected_positions_world = (
        quat.mul_vec(next_root_rotation, local_positions) + next_root_position[np.newaxis, :]
    ).astype(np.float32)
    corrected_directions_world = np.asarray(
        [_normalize_xz(quat.mul_vec(next_root_rotation, direction)) for direction in local_directions],
        dtype=np.float32,
    )
    corrected_velocities_world = quat.mul_vec(next_root_rotation, local_velocities).astype(np.float32)

    runtime_state.trajectory_positions_world[future_indices] = (
        (1.0 - blend) * runtime_state.trajectory_positions_world[future_indices]
        + blend * corrected_positions_world
    ).astype(np.float32)
    runtime_state.trajectory_directions_world[future_indices] = np.asarray(
        [
            _normalize_xz(
                (1.0 - blend) * runtime_state.trajectory_directions_world[trajectory_index]
                + blend * corrected_directions_world[i]
            )
            for i, trajectory_index in enumerate(future_indices)
        ],
        dtype=np.float32,
    )
    runtime_state.trajectory_velocities_world[future_indices] = (
        (1.0 - blend) * runtime_state.trajectory_velocities_world[future_indices]
        + blend * corrected_velocities_world
    ).astype(np.float32)
    runtime_state.trajectory_positions_world[:, 1] = 0.0
    runtime_state.trajectory_velocities_world[:, 1] = 0.0


def _apply_integrated_root_motion(
    runtime_state: RuntimeState,
    debug_prediction: RuntimeDebugPrediction,
) -> None:
    dt = max(1e-6, float(runtime_state.dt))
    previous_root_position = runtime_state.root_position.copy()
    previous_root_rotation = runtime_state.root_rotation.copy()
    previous_root_velocity = runtime_state.root_velocity.copy()

    root_local_velocity = np.asarray([debug_prediction.y_root[0], 0.0, debug_prediction.y_root[1]], dtype=np.float32)
    root_local_angular_velocity = np.asarray([0.0, debug_prediction.y_root[2], 0.0], dtype=np.float32)
    root_world_velocity = quat.mul_vec(previous_root_rotation, root_local_velocity).astype(np.float32)
    root_world_angular_velocity = quat.mul_vec(previous_root_rotation, root_local_angular_velocity).astype(np.float32)
    next_root_position, next_root_rotation = _integrate_root_motion_step(
        previous_root_position,
        previous_root_rotation,
        root_local_velocity,
        root_local_angular_velocity,
        dt,
    )

    runtime_state.root_position = next_root_position.astype(np.float32)
    runtime_state.root_rotation = next_root_rotation.astype(np.float32)
    runtime_state.root_velocity = root_world_velocity.astype(np.float32)
    runtime_state.root_velocity[1] = 0.0
    runtime_state.root_acceleration = ((runtime_state.root_velocity - previous_root_velocity) / dt).astype(np.float32)
    runtime_state.root_acceleration[1] = 0.0
    runtime_state.root_angular_velocity = root_world_angular_velocity.astype(np.float32)
    runtime_state.root_angular_velocity[0] = 0.0
    runtime_state.root_angular_velocity[2] = 0.0

    runtime_state.previous_local_pose["current_root_position"] = runtime_state.root_position.copy()
    runtime_state.previous_local_pose["current_root_rotation"] = runtime_state.root_rotation.copy()


def _apply_debug_runtime_feedback(runtime_state: RuntimeState, debug_prediction: RuntimeDebugPrediction | None) -> None:
    if debug_prediction is None:
        return
    runtime_state.previous_local_pose = {
        key: np.asarray(value, dtype=np.float32).copy() if isinstance(value, np.ndarray) else value
        for key, value in debug_prediction.predicted_local_pose.items()
    }
    _apply_future_trajectory_correction(runtime_state, debug_prediction)
    _apply_integrated_root_motion(runtime_state, debug_prediction)


class MANNRuntime:
    """Minimal runtime wrapper for loading and evaluating a trained MANN."""

    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
        stats_path: Path = DEFAULT_STATS_PATH,
        database_path: Path = DEFAULT_DATABASE_PATH,
        device: str | torch.device | None = None,
    ) -> None:
        self.paths = RuntimePaths(
            checkpoint_path=Path(checkpoint_path),
            stats_path=Path(stats_path),
            database_path=Path(database_path),
        )
        self.device = torch.device(device or "cpu")
        self.spec = self._load_spec(self.paths.database_path)
        self.stats = MANNFeatureStats.load(self.paths.stats_path)
        self.model, self.model_config, self.checkpoint = self._load_model(
            self.paths.checkpoint_path,
            self.device,
        )
        self._validate_loaded_artifacts()

    @staticmethod
    def _load_spec(database_path: Path) -> MANNDataSpec:
        with np.load(database_path, allow_pickle=False) as data:
            return MANNDataSpec.from_npz(data)

    @staticmethod
    def _load_model(
        checkpoint_path: Path,
        device: torch.device,
    ) -> tuple[MANN, MANNModelConfig, dict]:
        # The checkpoint is local/trusted and stores non-tensor metadata, so we
        # intentionally disable the PyTorch 2.6 weights_only default here.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        model_config = MANNModelConfig(**checkpoint["model_config"])
        model = MANN(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model, model_config, checkpoint

    def _validate_loaded_artifacts(self) -> None:
        if self.stats.x_main_mean.shape[0] != self.spec.x_main_dim:
            raise ValueError(
                "Stats x_main dimension does not match database spec: "
                f"{self.stats.x_main_mean.shape[0]} vs {self.spec.x_main_dim}."
            )
        if self.stats.x_gate_mean.shape[0] != self.spec.x_gate_dim:
            raise ValueError(
                "Stats x_gate dimension does not match database spec: "
                f"{self.stats.x_gate_mean.shape[0]} vs {self.spec.x_gate_dim}."
            )
        if self.stats.y_mean.shape[0] != self.spec.y_dim:
            raise ValueError(
                "Stats y dimension does not match database spec: "
                f"{self.stats.y_mean.shape[0]} vs {self.spec.y_dim}."
            )

        if self.model_config.x_main_dim != self.spec.x_main_dim:
            raise ValueError(
                "Checkpoint x_main dimension does not match database spec: "
                f"{self.model_config.x_main_dim} vs {self.spec.x_main_dim}."
            )
        if self.model_config.x_gate_dim != self.spec.x_gate_dim:
            raise ValueError(
                "Checkpoint x_gate dimension does not match database spec: "
                f"{self.model_config.x_gate_dim} vs {self.spec.x_gate_dim}."
            )
        if self.model_config.y_dim != self.spec.y_dim:
            raise ValueError(
                "Checkpoint y dimension does not match database spec: "
                f"{self.model_config.y_dim} vs {self.spec.y_dim}."
            )

    @property
    def action_labels(self) -> tuple[str, ...]:
        return self.spec.action_labels

    def normalize_inputs(
        self,
        x_main: np.ndarray,
        x_gate: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.stats.normalize_x_main(x_main),
            self.stats.normalize_x_gate(x_gate),
        )

    def denormalize_output(self, y: np.ndarray) -> np.ndarray:
        return self.stats.denormalize_y(y)

    def _ensure_batch(self, array: np.ndarray, expected_dim: int, name: str) -> tuple[np.ndarray, bool]:
        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 1:
            if array.shape[0] != expected_dim:
                raise ValueError(f"{name} has dim {array.shape[0]}, expected {expected_dim}.")
            return array[np.newaxis, :], True
        if array.ndim != 2 or array.shape[1] != expected_dim:
            raise ValueError(f"{name} has shape {array.shape}, expected (?, {expected_dim}).")
        return array, False

    def predict(
        self,
        x_main: np.ndarray,
        x_gate: np.ndarray,
        normalize_inputs: bool = True,
        denormalize_output: bool = True,
        return_aux: bool = False,
    ) -> RuntimePrediction:
        x_main_batch, squeeze_output = self._ensure_batch(x_main, self.spec.x_main_dim, "x_main")
        x_gate_batch, squeeze_gate = self._ensure_batch(x_gate, self.spec.x_gate_dim, "x_gate")
        if squeeze_output != squeeze_gate:
            raise ValueError("x_main and x_gate batch ranks do not match.")

        if normalize_inputs:
            x_main_batch, x_gate_batch = self.normalize_inputs(x_main_batch, x_gate_batch)

        x_main_tensor = torch.from_numpy(x_main_batch).to(self.device)
        x_gate_tensor = torch.from_numpy(x_gate_batch).to(self.device)

        with torch.no_grad():
            model_output = self.model(x_main_tensor, x_gate_tensor, return_aux=return_aux)

        if return_aux:
            y_pred_tensor = model_output["y_pred"]
            expert_weights = model_output["expert_weights"].detach().cpu().numpy().astype(np.float32)
        else:
            y_pred_tensor = model_output
            expert_weights = None

        y_pred = y_pred_tensor.detach().cpu().numpy().astype(np.float32)
        if denormalize_output:
            y_pred = self.denormalize_output(y_pred)

        y_pred_tensor_for_split = torch.from_numpy(y_pred)
        split_outputs = self.model.split_prediction(y_pred_tensor_for_split)

        y_pose = split_outputs["y_pose"].detach().cpu().numpy().astype(np.float32)
        y_root = split_outputs["y_root"].detach().cpu().numpy().astype(np.float32)
        y_future_tensor = split_outputs.get("y_future")
        y_future = None if y_future_tensor is None else y_future_tensor.detach().cpu().numpy().astype(np.float32)

        if squeeze_output:
            y_pred = y_pred[0]
            y_pose = y_pose[0]
            y_root = y_root[0]
            if y_future is not None:
                y_future = y_future[0]
            if expert_weights is not None:
                expert_weights = expert_weights[0]

        return RuntimePrediction(
            y=y_pred,
            y_pose=y_pose,
            y_root=y_root,
            y_future=y_future,
            expert_weights=expert_weights,
        )


def _halflife_to_damping(halflife: float, eps: float = 1e-5) -> float:
    return (4.0 * 0.69314718056) / (float(halflife) + eps)


def _normalize_xz(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).copy()
    vector[1] = 0.0
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        if fallback is None:
            return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        fallback = np.asarray(fallback, dtype=np.float32).copy()
        fallback[1] = 0.0
        fallback_norm = float(np.linalg.norm(fallback))
        if fallback_norm < 1e-6:
            return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        return (fallback / fallback_norm).astype(np.float32)
    return (vector / norm).astype(np.float32)


def _yaw_rotation_from_direction(direction: np.ndarray, fallback_rotation: np.ndarray) -> np.ndarray:
    direction = _normalize_xz(
        direction,
        fallback=quat.mul_vec(fallback_rotation, np.asarray([0.0, 0.0, 1.0], dtype=np.float32)),
    )
    yaw = float(np.arctan2(direction[0], direction[2]))
    return quat.from_angle_axis(yaw, np.asarray([0.0, 1.0, 0.0], dtype=np.float32)).astype(np.float32)


def _make_action_one_hot(action_label: str) -> np.ndarray:
    if action_label not in HUMANOID_LOCOMOTION_ACTION_LABELS:
        raise ValueError(f"Unsupported action label: {action_label}")
    action_one_hot = np.zeros(len(HUMANOID_LOCOMOTION_ACTION_LABELS), dtype=np.float32)
    action_one_hot[HUMANOID_LOCOMOTION_ACTION_LABELS.index(action_label)] = 1.0
    return action_one_hot


def _shape_gamepad_stick(x: float, y: float, deadzone: float = DEFAULT_GAMEPAD_DEADZONE) -> np.ndarray:
    stick = np.asarray([x, y], dtype=np.float32)
    magnitude = float(np.linalg.norm(stick))
    if magnitude <= deadzone:
        return np.zeros(2, dtype=np.float32)

    direction = stick / magnitude
    shaped_magnitude = min(magnitude * magnitude, 1.0)
    return (direction * shaped_magnitude).astype(np.float32)


def _read_gamepad_stick(gamepad_id: int, left: bool, deadzone: float = DEFAULT_GAMEPAD_DEADZONE) -> np.ndarray:
    axis_x = GAMEPAD_AXIS_LEFT_X if left else GAMEPAD_AXIS_RIGHT_X
    axis_y = GAMEPAD_AXIS_LEFT_Y if left else GAMEPAD_AXIS_RIGHT_Y
    return _shape_gamepad_stick(
        GetGamepadAxisMovement(gamepad_id, axis_x),
        GetGamepadAxisMovement(gamepad_id, axis_y),
        deadzone=deadzone,
    )


def _read_gamepad_input(app) -> RawInputState:
    gamepad_id = getattr(app, "gamepad_id", DEFAULT_GAMEPAD_ID)
    if not IsGamepadAvailable(gamepad_id):
        return RawInputState(
            input_source="gamepad:none",
            reset_pressed=bool(IsKeyPressed(KEY_R)),
        )

    left_stick = _read_gamepad_stick(gamepad_id, left=True)
    right_stick = _read_gamepad_stick(gamepad_id, left=False)
    right_trigger = float(GetGamepadAxisMovement(gamepad_id, GAMEPAD_AXIS_RIGHT_TRIGGER))
    left_trigger = float(GetGamepadAxisMovement(gamepad_id, GAMEPAD_AXIS_LEFT_TRIGGER))

    return RawInputState(
        input_source=f"gamepad:{gamepad_id}",
        move_2d=left_stick,
        look_2d=right_stick,
        look_active=bool(np.linalg.norm(right_stick) > 1e-3),
        run_pressed=bool(right_trigger > 0.5 or IsGamepadButtonDown(gamepad_id, GAMEPAD_BUTTON_RIGHT_TRIGGER_2)),
        desired_strafe=bool(left_trigger > 0.5 or IsGamepadButtonDown(gamepad_id, GAMEPAD_BUTTON_LEFT_TRIGGER_2)),
        jump_pressed=bool(IsGamepadButtonPressed(gamepad_id, GAMEPAD_BUTTON_RIGHT_FACE_DOWN)),
        reset_pressed=bool(IsGamepadButtonPressed(gamepad_id, GAMEPAD_BUTTON_MIDDLE_RIGHT) or IsKeyPressed(KEY_R)),
    )


def _read_keyboard_input(app) -> RawInputState:
    move_x = float(IsKeyDown(KEY_D) - IsKeyDown(KEY_A))
    move_y = float(IsKeyDown(KEY_W) - IsKeyDown(KEY_S))
    move_2d = np.asarray([move_x, move_y], dtype=np.float32)
    move_norm = float(np.linalg.norm(move_2d))
    if move_norm > 1.0:
        move_2d /= move_norm

    if IsMouseButtonDown(MOUSE_BUTTON_RIGHT):
        mouse_position = np.asarray([GetMousePosition().x, GetMousePosition().y], dtype=np.float32)
        if app.mouse_look_anchor is None:
            app.mouse_look_anchor = mouse_position
        else:
            momentum = 0.01
            app.mouse_look_anchor = (1.0 - momentum) * app.mouse_look_anchor + momentum * mouse_position
        look_2d = np.asarray(
            [
                mouse_position[0] - app.mouse_look_anchor[0],
                app.mouse_look_anchor[1] - mouse_position[1],
            ],
            dtype=np.float32,
        )
        look_norm = float(np.linalg.norm(look_2d))
        if look_norm > 1.0:
            look_2d /= look_norm
        look_active = look_norm > 1e-3
    else:
        app.mouse_look_anchor = None
        look_2d = np.zeros(2, dtype=np.float32)
        look_active = False

    return RawInputState(
        input_source="keyboard",
        move_2d=move_2d,
        look_2d=look_2d,
        look_active=look_active,
        run_pressed=bool(IsKeyDown(KEY_LEFT_SHIFT) or IsKeyDown(KEY_RIGHT_SHIFT)),
        desired_strafe=bool(IsKeyDown(KEY_LEFT_CONTROL) or IsKeyDown(KEY_RIGHT_CONTROL)),
        jump_pressed=bool(IsKeyPressed(KEY_SPACE)),
        reset_pressed=bool(IsKeyPressed(KEY_R)),
    )


def _derive_control_intent(input_state: RawInputState, camera: Camera, root_rotation: np.ndarray) -> ControlIntent:
    camera_rotation = quat.from_angle_axis(
        camera.azimuth,
        np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    ).astype(np.float32)

    move_world = quat.mul_vec(
        camera_rotation,
        np.asarray([input_state.move_2d[0], 0.0, input_state.move_2d[1]], dtype=np.float32),
    ).astype(np.float32)
    look_world = quat.mul_vec(
        camera_rotation,
        np.asarray([input_state.look_2d[0], 0.0, input_state.look_2d[1]], dtype=np.float32),
    ).astype(np.float32)

    move_magnitude = float(np.linalg.norm(input_state.move_2d))
    speed = DEFAULT_RUN_SPEED if input_state.run_pressed else DEFAULT_WALK_SPEED
    desired_velocity_world = (move_world * speed).astype(np.float32)

    current_forward = quat.mul_vec(root_rotation, np.asarray([0.0, 0.0, 1.0], dtype=np.float32)).astype(np.float32)
    if input_state.look_active:
        desired_facing_world = _normalize_xz(look_world, fallback=current_forward)
    elif input_state.desired_strafe:
        desired_facing_world = _normalize_xz(current_forward)
    elif np.linalg.norm(desired_velocity_world) > 1e-3:
        desired_facing_world = _normalize_xz(desired_velocity_world, fallback=current_forward)
    else:
        desired_facing_world = _normalize_xz(current_forward)

    desired_rotation = _yaw_rotation_from_direction(desired_facing_world, fallback_rotation=root_rotation)
    if input_state.jump_pressed and "jump" in HUMANOID_LOCOMOTION_ACTION_LABELS:
        action_label = "jump"
    elif input_state.run_pressed and "run" in HUMANOID_LOCOMOTION_ACTION_LABELS:
        action_label = "run"
    else:
        action_label = "walk"

    return ControlIntent(
        desired_velocity_world=desired_velocity_world,
        desired_facing_world=desired_facing_world,
        desired_rotation=desired_rotation,
        action_label=action_label,
        desired_strafe=bool(input_state.desired_strafe),
        move_magnitude=move_magnitude,
    )


def _simulation_positions_update(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    desired_velocity: np.ndarray,
    halflife: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = _halflife_to_damping(halflife) / 2.0
    j0 = velocity - desired_velocity
    j1 = acceleration + j0 * y
    eydt = np.exp(-y * dt)

    next_position = (
        eydt * (((-j1) / (y * y)) + ((-j0 - j1 * dt) / y))
        + (j1 / (y * y))
        + j0 / y
        + desired_velocity * dt
        + position
    ).astype(np.float32)
    next_velocity = (eydt * (j0 + j1 * dt) + desired_velocity).astype(np.float32)
    next_acceleration = (eydt * (acceleration - j1 * y * dt)).astype(np.float32)
    next_position[1] = 0.0
    next_velocity[1] = 0.0
    next_acceleration[1] = 0.0
    return next_position, next_velocity, next_acceleration


def _simulation_rotations_update(
    rotation: np.ndarray,
    angular_velocity: np.ndarray,
    desired_rotation: np.ndarray,
    halflife: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    y = _halflife_to_damping(halflife) / 2.0
    j0 = quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotation, desired_rotation))).astype(np.float32)
    j1 = angular_velocity + j0 * y
    eydt = float(np.exp(-y * dt))

    next_rotation = quat.mul(
        quat.from_scaled_angle_axis(eydt * (j0 + j1 * dt)).astype(np.float32),
        desired_rotation,
    ).astype(np.float32)
    next_rotation = quat.normalize(next_rotation).astype(np.float32)
    next_angular_velocity = (eydt * (angular_velocity - j1 * y * dt)).astype(np.float32)
    next_angular_velocity[0] = 0.0
    next_angular_velocity[2] = 0.0
    return next_rotation, next_angular_velocity


def _integrate_root_motion_step(
    root_position: np.ndarray,
    root_rotation: np.ndarray,
    root_local_velocity: np.ndarray,
    root_local_angular_velocity: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    root_world_velocity = quat.mul_vec(root_rotation, root_local_velocity).astype(np.float32)
    root_world_angular_velocity = quat.mul_vec(root_rotation, root_local_angular_velocity).astype(np.float32)
    next_root_position = (root_position + root_world_velocity * dt).astype(np.float32)
    next_root_rotation = quat.normalize(
        quat.mul(
            quat.from_scaled_angle_axis(root_world_angular_velocity * dt),
            root_rotation,
        )
    ).astype(np.float32)
    next_root_position[1] = 0.0
    return next_root_position, next_root_rotation


def _make_runtime_state(
    initial_local_pose: dict,
    initial_root_position: np.ndarray,
    initial_root_rotation: np.ndarray,
    sample_offsets: np.ndarray,
    dt: float,
) -> RuntimeState:
    initial_root_position = np.asarray(initial_root_position, dtype=np.float32)
    initial_root_rotation = quat.normalize(np.asarray(initial_root_rotation, dtype=np.float32))
    sample_offsets = np.asarray(sample_offsets, dtype=np.int32)
    sample_count = len(sample_offsets)
    return RuntimeState(
        root_position=initial_root_position.copy(),
        root_rotation=initial_root_rotation.copy(),
        root_velocity=np.zeros(3, dtype=np.float32),
        root_acceleration=np.zeros(3, dtype=np.float32),
        root_angular_velocity=np.zeros(3, dtype=np.float32),
        previous_local_pose={
            key: np.asarray(value, dtype=np.float32).copy()
            if isinstance(value, np.ndarray) else value
            for key, value in initial_local_pose.items()
        },
        action_label="walk",
        action_one_hot=_make_action_one_hot("walk"),
        desired_strafe=False,
        trajectory_positions_world=np.repeat(initial_root_position[np.newaxis, :], sample_count, axis=0).astype(np.float32),
        trajectory_directions_world=np.repeat(
            quat.mul_vec(initial_root_rotation, np.asarray([0.0, 0.0, 1.0], dtype=np.float32))[np.newaxis, :],
            sample_count,
            axis=0,
        ).astype(np.float32),
        trajectory_velocities_world=np.zeros((sample_count, 3), dtype=np.float32),
        sample_offsets=sample_offsets.copy(),
        current_sample_index=HUMANOID_LOCOMOTION_TRAJECTORY_CURRENT_SAMPLE_INDEX,
        dt=float(dt),
        history_positions=[initial_root_position.copy()],
        history_rotations=[initial_root_rotation.copy()],
        history_velocities=[np.zeros(3, dtype=np.float32)],
    )


def _append_runtime_history(runtime_state: RuntimeState) -> None:
    runtime_state.history_positions.append(runtime_state.root_position.copy())
    runtime_state.history_rotations.append(runtime_state.root_rotation.copy())
    runtime_state.history_velocities.append(runtime_state.root_velocity.copy())
    max_history = int(-min(HUMANOID_LOCOMOTION_TRAJECTORY_SAMPLE_OFFSETS)) + 1
    if len(runtime_state.history_positions) > max_history:
        runtime_state.history_positions = runtime_state.history_positions[-max_history:]
        runtime_state.history_rotations = runtime_state.history_rotations[-max_history:]
        runtime_state.history_velocities = runtime_state.history_velocities[-max_history:]


def _predict_future_trajectory(runtime_state: RuntimeState, intent: ControlIntent, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_offsets = np.asarray(HUMANOID_LOCOMOTION_TRAJECTORY_SAMPLE_OFFSETS, dtype=np.int32)
    positions = np.zeros((len(sample_offsets), 3), dtype=np.float32)
    rotations = np.zeros((len(sample_offsets), 4), dtype=np.float32)
    velocities = np.zeros((len(sample_offsets), 3), dtype=np.float32)

    current_index = HUMANOID_LOCOMOTION_TRAJECTORY_CURRENT_SAMPLE_INDEX
    positions[current_index] = runtime_state.root_position
    rotations[current_index] = runtime_state.root_rotation
    velocities[current_index] = runtime_state.root_velocity

    for i, offset in enumerate(sample_offsets):
        if offset >= 0:
            continue
        history_index = max(0, len(runtime_state.history_positions) - 1 + int(offset))
        positions[i] = runtime_state.history_positions[history_index]
        rotations[i] = runtime_state.history_rotations[history_index]
        velocities[i] = runtime_state.history_velocities[history_index]

    future_position = runtime_state.root_position.copy()
    future_rotation = runtime_state.root_rotation.copy()
    future_velocity = runtime_state.root_velocity.copy()
    future_acceleration = runtime_state.root_acceleration.copy()
    future_angular_velocity = runtime_state.root_angular_velocity.copy()

    future_lookup = {}
    for step in range(1, int(max(sample_offsets)) + 1):
        future_position, future_velocity, future_acceleration = _simulation_positions_update(
            future_position,
            future_velocity,
            future_acceleration,
            intent.desired_velocity_world,
            DEFAULT_MOVE_HALFLIFE,
            dt,
        )
        future_rotation, future_angular_velocity = _simulation_rotations_update(
            future_rotation,
            future_angular_velocity,
            intent.desired_rotation,
            DEFAULT_ROTATION_HALFLIFE,
            dt,
        )
        future_lookup[step] = (
            future_position.copy(),
            future_rotation.copy(),
            future_velocity.copy(),
        )

    for i, offset in enumerate(sample_offsets):
        if offset <= 0:
            continue
        positions[i], rotations[i], velocities[i] = future_lookup[int(offset)]

    directions = np.asarray(
        [quat.mul_vec(rotation, np.asarray([0.0, 0.0, 1.0], dtype=np.float32)) for rotation in rotations],
        dtype=np.float32,
    )
    future_mask = sample_offsets > 0
    if runtime_state.trajectory_positions_world.shape == positions.shape:
        positions[future_mask] = (
            (1.0 - DEFAULT_TRAJECTORY_BUFFER_BLEND) * positions[future_mask]
            + DEFAULT_TRAJECTORY_BUFFER_BLEND * runtime_state.trajectory_positions_world[future_mask]
        ).astype(np.float32)
    if runtime_state.trajectory_velocities_world.shape == velocities.shape:
        velocities[future_mask] = (
            (1.0 - DEFAULT_TRAJECTORY_BUFFER_BLEND) * velocities[future_mask]
            + DEFAULT_TRAJECTORY_BUFFER_BLEND * runtime_state.trajectory_velocities_world[future_mask]
        ).astype(np.float32)
    if runtime_state.trajectory_directions_world.shape == directions.shape:
        directions[future_mask] = (
            (1.0 - DEFAULT_TRAJECTORY_BUFFER_BLEND) * directions[future_mask]
            + DEFAULT_TRAJECTORY_BUFFER_BLEND * runtime_state.trajectory_directions_world[future_mask]
        ).astype(np.float32)
    directions = np.asarray([_normalize_xz(direction) for direction in directions], dtype=np.float32)
    positions[:, 1] = 0.0
    velocities[:, 1] = 0.0
    return positions, directions, velocities


def _apply_root_transform(
    base_positions: np.ndarray,
    base_rotations: np.ndarray,
    base_root_position: np.ndarray,
    base_root_rotation: np.ndarray,
    root_position: np.ndarray,
    root_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    local_positions = quat.inv_mul_vec(base_root_rotation, base_positions - base_root_position).astype(np.float32)
    local_rotations = quat.inv_mul(base_root_rotation, base_rotations).astype(np.float32)
    world_positions = (quat.mul_vec(root_rotation, local_positions) + root_position).astype(np.float32)
    world_rotations = quat.mul(root_rotation, local_rotations).astype(np.float32)
    return world_positions, quat.normalize(world_rotations).astype(np.float32)


def _build_local_debug_trajectory(feature_inputs: FeatureInputs | None, local_debug_origin: Vector3):
    if feature_inputs is None:
        return None

    root_local_trajectory = feature_inputs.root_local_trajectory
    origin = np.asarray([local_debug_origin.x, local_debug_origin.y, local_debug_origin.z], dtype=np.float32)
    local_positions = np.asarray(root_local_trajectory["local_positions"], dtype=np.float32) + origin[np.newaxis, :]
    local_directions = np.asarray(root_local_trajectory["local_directions"], dtype=np.float32)
    local_velocities = np.asarray(root_local_trajectory["local_velocities"], dtype=np.float32)
    return local_positions, local_directions, local_velocities


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Static viewer and runtime loader for the Geno MANN controller.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH, help="Path to the trained MANN checkpoint.")
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS_PATH, help="Path to the saved normalization stats.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH, help="Path to the exported MANN database.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device used for inference.")
    parser.add_argument("--self-test", action="store_true", help="Run the runtime loading self-test instead of opening the static viewer.")
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Database sample index used by the runtime self-test.",
    )
    parser.add_argument("--clip", type=Path, default=DEFAULT_VIEWER_CLIP, help="BVH clip relative to resources used by the static viewer.")
    parser.add_argument("--frame", type=int, default=DEFAULT_INITIAL_FRAME, help="Frame index used as the initial static pose.")
    parser.add_argument("--screen-width", type=int, default=DEFAULT_SCREEN_WIDTH, help="Viewer window width.")
    parser.add_argument("--screen-height", type=int, default=DEFAULT_SCREEN_HEIGHT, help="Viewer window height.")
    parser.add_argument("--gamepad-id", type=int, default=DEFAULT_GAMEPAD_ID, help="Gamepad id used by the viewer controls.")
    return parser


def _run_runtime_self_test(args) -> None:
    runtime = MANNRuntime(
        checkpoint_path=args.checkpoint,
        stats_path=args.stats,
        database_path=args.database,
        device=args.device,
    )

    with np.load(args.database, allow_pickle=False) as data:
        sample_index = int(np.clip(args.sample_index, 0, len(data["x_main"]) - 1))
        x_main = data["x_main"][sample_index].astype(np.float32)
        x_gate = data["x_gate"][sample_index].astype(np.float32)

    prediction = runtime.predict(x_main, x_gate, return_aux=True)

    print("Loaded Geno MANN runtime")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  stats:      {args.stats}")
    print(f"  database:   {args.database}")
    print(f"  device:     {runtime.device}")
    print(f"  stage:      {runtime.spec.stage}")
    print(f"  x_main_dim: {runtime.spec.x_main_dim}")
    print(f"  x_gate_dim: {runtime.spec.x_gate_dim}")
    print(f"  y_dim:      {runtime.spec.y_dim}")
    print(f"  actions:    {runtime.action_labels}")
    print(f"  sample:     {sample_index}")
    print(f"  y_pose:     {prediction.y_pose.shape}")
    print(f"  y_root:     {prediction.y_root.shape}")
    print(f"  y_future:   {None if prediction.y_future is None else prediction.y_future.shape}")
    print(
        "  experts:    "
        f"{None if prediction.expert_weights is None else prediction.expert_weights.shape}"
    )


def _load_static_motion_resources(scene, clip_path: Path, initial_frame: int):
    resolved_clip = resource_path(*clip_path.parts)
    bvh_animation = BVHImporter.load(resolved_clip, scale=0.01)

    if not (0 <= int(initial_frame) < bvh_animation.frame_count):
        raise ValueError(
            f"Initial frame {initial_frame} is out of range for {clip_path} "
            f"(frame_count={bvh_animation.frame_count})."
        )

    global_positions = bvh_animation.global_positions
    global_rotations = bvh_animation.global_rotations
    root_trajectory_source = BuildRootTrajectorySource(
        global_positions,
        global_rotations,
        DEFAULT_BVH_FRAME_TIME,
        mode=ROOT_TRAJECTORY_MODE_FLAT,
        projectToGround=True,
        groundHeight=0.0,
    )
    pose_source = BuildPoseSource(
        global_positions,
        global_rotations,
        DEFAULT_BVH_FRAME_TIME,
        rootTrajectorySource=root_trajectory_source,
    )

    frame_index = int(initial_frame)
    base_positions = global_positions[frame_index].astype(np.float32)
    base_rotations = global_rotations[frame_index].astype(np.float32)
    base_root_position = base_positions[ROOT_JOINT_INDEX].astype(np.float32)
    base_root_rotation = base_rotations[ROOT_JOINT_INDEX].astype(np.float32)
    initial_local_pose = BuildLocalPose(
        pose_source,
        root_trajectory_source,
        frame_index,
        dt=DEFAULT_BVH_FRAME_TIME,
    )

    UpdateModelPoseFromNumpyArrays(
        scene.geno_model,
        scene.bind_pos,
        scene.bind_rot,
        base_positions,
        base_rotations,
    )
    UpdateModelPoseFromNumpyArrays(
        scene.pose_model,
        scene.bind_pos,
        scene.bind_rot,
        base_positions,
        base_rotations,
    )

    return SimpleNamespace(
        clip_path=clip_path,
        bvh_animation=bvh_animation,
        frame_index=frame_index,
        joint_names=list(bvh_animation.raw_data["names"]),
        global_positions=global_positions,
        global_rotations=global_rotations,
        base_positions=base_positions,
        base_rotations=base_rotations,
        base_root_position=base_root_position,
        base_root_rotation=base_root_rotation,
        root_trajectory_source=root_trajectory_source,
        pose_source=pose_source,
        initial_local_pose=initial_local_pose,
    )


def _load_mann_scene_resources():
    scene = LoadSceneResources(resource_path)
    scene.pose_model = LoadCharacterModel(resource_path("Geno.bin", as_bytes=True))
    return scene


def _create_viewer_state(config: ViewerConfig):
    scene = _load_mann_scene_resources()
    motion = _load_static_motion_resources(scene, config.clip_path, config.initial_frame)
    feature_builder = FeatureBuilder(motion.joint_names, motion.bvh_animation.parents)
    runtime = MANNRuntime()
    camera = Camera()
    input_state = RawInputState()
    runtime_state = _make_runtime_state(
        motion.initial_local_pose,
        motion.base_root_position,
        motion.base_root_rotation,
        np.asarray(HUMANOID_LOCOMOTION_TRAJECTORY_SAMPLE_OFFSETS, dtype=np.int32),
        DEFAULT_BVH_FRAME_TIME,
    )
    control_intent = _derive_control_intent(input_state, camera, runtime_state.root_rotation)
    trajectory_positions, trajectory_directions, trajectory_velocities = _predict_future_trajectory(
        runtime_state,
        control_intent,
        runtime_state.dt,
    )
    runtime_state.trajectory_positions_world = trajectory_positions.copy()
    runtime_state.trajectory_directions_world = trajectory_directions.copy()
    runtime_state.trajectory_velocities_world = trajectory_velocities.copy()
    current_features = feature_builder.build_inputs(runtime_state)

    return SimpleNamespace(
        screen_width=int(config.screen_width),
        screen_height=int(config.screen_height),
        scene=scene,
        motion=motion,
        feature_builder=feature_builder,
        runtime=runtime,
        shaders=LoadShaderResources(resource_path),
        render=CreateRenderResources(config.screen_width, config.screen_height),
        camera=camera,
        show_flat_ground=True,
        show_pose_model=True,
        input_state=input_state,
        control_intent=control_intent,
        runtime_state=runtime_state,
        current_features=current_features,
        debug_prediction=None,
        local_debug_origin=Vector3(-2.0, 0.0, 0.0),
        gamepad_id=int(config.gamepad_id),
        mouse_look_anchor=None,
    )


def _update_static_tracking(app):
    dt = max(1e-6, float(GetFrameTime()))
    app.input_state = _read_gamepad_input(app)
    if app.input_state.reset_pressed:
        app.runtime_state = _make_runtime_state(
            app.motion.initial_local_pose,
            app.motion.base_root_position,
            app.motion.base_root_rotation,
            np.asarray(HUMANOID_LOCOMOTION_TRAJECTORY_SAMPLE_OFFSETS, dtype=np.int32),
            DEFAULT_BVH_FRAME_TIME,
        )

    app.runtime_state.dt = dt
    app.control_intent = _derive_control_intent(app.input_state, app.camera, app.runtime_state.root_rotation)
    app.runtime_state.action_label = app.control_intent.action_label
    app.runtime_state.action_one_hot = _make_action_one_hot(app.control_intent.action_label)
    app.runtime_state.desired_strafe = app.control_intent.desired_strafe

    trajectory_positions, trajectory_directions, trajectory_velocities = _predict_future_trajectory(
        app.runtime_state,
        app.control_intent,
        dt,
    )
    app.runtime_state.trajectory_positions_world = trajectory_positions.copy()
    app.runtime_state.trajectory_directions_world = trajectory_directions.copy()
    app.runtime_state.trajectory_velocities_world = trajectory_velocities.copy()
    app.current_features = app.feature_builder.build_inputs(app.runtime_state)
    app.debug_prediction = _run_debug_runtime_prediction(app)
    _apply_debug_runtime_feedback(app.runtime_state, app.debug_prediction)
    _append_runtime_history(app.runtime_state)
    app.current_features = app.feature_builder.build_inputs(app.runtime_state)

    if app.debug_prediction is not None:
        reconstructed_pose = ReconstructPoseWorldSpace(
            app.runtime_state.previous_local_pose,
            rootPosition=app.runtime_state.root_position,
            rootRotation=app.runtime_state.root_rotation,
            integrateRootMotion=False,
            dt=dt,
        )
        world_positions = reconstructed_pose["world_positions"]
        world_rotations = reconstructed_pose["world_rotations"]
    else:
        world_positions, world_rotations = _apply_root_transform(
            app.motion.base_positions,
            app.motion.base_rotations,
            app.motion.base_root_position,
            app.motion.base_root_rotation,
            app.runtime_state.root_position,
            app.runtime_state.root_rotation,
        )
    UpdateModelPoseFromNumpyArrays(
        app.scene.geno_model,
        app.scene.bind_pos,
        app.scene.bind_rot,
        world_positions,
        world_rotations,
    )
    if app.show_pose_model:
        if app.debug_prediction is not None:
            UpdateModelPoseFromNumpyArrays(
                app.scene.pose_model,
                app.scene.bind_pos,
                app.scene.bind_rot,
                app.debug_prediction.predicted_local_positions_offset,
                app.debug_prediction.predicted_local_pose["local_rotations"],
            )
        else:
            UpdateModelPoseFromNumpyArrays(
                app.scene.pose_model,
                app.scene.bind_pos,
                app.scene.bind_rot,
                world_positions,
                world_rotations,
            )

    hip_position = Vector3(*world_positions[ROOT_JOINT_INDEX])

    app.render.shadow_light.target = Vector3(hip_position.x, 0.0, hip_position.z)
    app.render.shadow_light.position = Vector3Add(
        app.render.shadow_light.target,
        Vector3Scale(app.render.light_dir, -5.0),
    )

    app.camera.update(
        Vector3(hip_position.x, 0.75, hip_position.z),
        GetMouseDelta().x if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(0) else 0.0,
        GetMouseDelta().y if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(0) else 0.0,
        GetMouseDelta().x if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(1) else 0.0,
        GetMouseDelta().y if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(1) else 0.0,
        GetMouseWheelMove(),
        GetFrameTime(),
    )

    return SimpleNamespace(
        hip_position=hip_position,
        clip_label=str(app.motion.clip_path),
        frame_index=app.motion.frame_index,
        action_label=app.runtime_state.action_label,
        input_state=app.input_state,
        control_intent=app.control_intent,
        feature_inputs=app.current_features,
    )


def _render_static_shadow_pass(app):
    render = app.render
    shaders = app.shaders
    scene = app.scene

    BeginShadowMap(render.shadow_map, render.shadow_light)

    light_view_proj = MatrixMultiply(rlGetMatrixModelview(), rlGetMatrixProjection())
    light_clip_near_ptr = _make_float_ptr(rlGetCullDistanceNear())
    light_clip_far_ptr = _make_float_ptr(rlGetCullDistanceFar())

    SetShaderValue(shaders.shadow.program, shaders.shadow.light_clip_near, light_clip_near_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.shadow.program, shaders.shadow.light_clip_far, light_clip_far_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(
        shaders.skinned_shadow.program,
        shaders.skinned_shadow.light_clip_near,
        light_clip_near_ptr,
        SHADER_UNIFORM_FLOAT,
    )
    SetShaderValue(
        shaders.skinned_shadow.program,
        shaders.skinned_shadow.light_clip_far,
        light_clip_far_ptr,
        SHADER_UNIFORM_FLOAT,
    )

    if app.show_flat_ground:
        _draw_model_with_shader(scene.ground_model, shaders.shadow.program, scene.ground_position, WHITE)
    _draw_model_with_shader(scene.geno_model, shaders.skinned_shadow.program, scene.geno_position, WHITE)
    if app.show_pose_model:
        _draw_model_with_shader(scene.pose_model, shaders.skinned_shadow.program, scene.geno_position, WHITE)

    EndShadowMap()

    return SimpleNamespace(
        view_proj=light_view_proj,
        clip_near_ptr=light_clip_near_ptr,
        clip_far_ptr=light_clip_far_ptr,
    )


def _render_static_gbuffer_pass(app):
    render = app.render
    shaders = app.shaders
    scene = app.scene

    BeginGBuffer(render.gbuffer, app.camera.cam3d)

    cam_view = rlGetMatrixModelview()
    cam_proj = rlGetMatrixProjection()
    cam_inv_proj = MatrixInvert(cam_proj)
    cam_inv_view_proj = MatrixInvert(MatrixMultiply(cam_view, cam_proj))
    cam_clip_near_ptr = _make_float_ptr(rlGetCullDistanceNear())
    cam_clip_far_ptr = _make_float_ptr(rlGetCullDistanceFar())
    specularity_ptr = _make_float_ptr(0.5)
    glossiness_ptr = _make_float_ptr(10.0)

    SetShaderValue(shaders.basic.program, shaders.basic.specularity, specularity_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.basic.program, shaders.basic.glossiness, glossiness_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.basic.program, shaders.basic.cam_clip_near, cam_clip_near_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.basic.program, shaders.basic.cam_clip_far, cam_clip_far_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.skinned_basic.program, shaders.skinned_basic.specularity, specularity_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.skinned_basic.program, shaders.skinned_basic.glossiness, glossiness_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(
        shaders.skinned_basic.program,
        shaders.skinned_basic.cam_clip_near,
        cam_clip_near_ptr,
        SHADER_UNIFORM_FLOAT,
    )
    SetShaderValue(
        shaders.skinned_basic.program,
        shaders.skinned_basic.cam_clip_far,
        cam_clip_far_ptr,
        SHADER_UNIFORM_FLOAT,
    )

    if app.show_flat_ground:
        _draw_model_with_shader(scene.ground_model, shaders.basic.program, scene.ground_position, Color(190, 190, 190, 255))
    _draw_model_with_shader(scene.geno_model, shaders.skinned_basic.program, scene.geno_position, Color(170, 220, 255, 255))
    if app.show_pose_model:
        _draw_model_with_shader(scene.pose_model, shaders.skinned_basic.program, scene.geno_position, Color(110, 190, 255, 255))

    EndGBuffer(app.screen_width, app.screen_height)

    return SimpleNamespace(
        view=cam_view,
        proj=cam_proj,
        inv_proj=cam_inv_proj,
        inv_view_proj=cam_inv_view_proj,
        clip_near_ptr=cam_clip_near_ptr,
        clip_far_ptr=cam_clip_far_ptr,
    )


def _render_static_lighting_pass(app, camera_pass):
    render = app.render
    shaders = app.shaders

    BeginTextureMode(render.lighted)
    BeginShaderMode(shaders.lighting.program)

    sun_color = Vector3(253.0 / 255.0, 255.0 / 255.0, 232.0 / 255.0)
    sun_strength_ptr = _make_float_ptr(0.25)
    sky_color = Vector3(174.0 / 255.0, 183.0 / 255.0, 190.0 / 255.0)
    sky_strength_ptr = _make_float_ptr(0.15)
    ground_strength_ptr = _make_float_ptr(0.1)
    ambient_strength_ptr = _make_float_ptr(1.0)
    exposure_ptr = _make_float_ptr(0.9)

    SetShaderValueTexture(shaders.lighting.program, shaders.lighting.gbuffer_color, render.gbuffer.color)
    SetShaderValueTexture(shaders.lighting.program, shaders.lighting.gbuffer_normal, render.gbuffer.normal)
    SetShaderValueTexture(shaders.lighting.program, shaders.lighting.gbuffer_depth, render.gbuffer.depth)
    SetShaderValueTexture(shaders.lighting.program, shaders.lighting.ssao, render.ssao_front.texture)
    SetShaderValue(
        shaders.lighting.program,
        shaders.lighting.cam_pos,
        ffi.addressof(app.camera.cam3d.position),
        SHADER_UNIFORM_VEC3,
    )
    SetShaderValueMatrix(shaders.lighting.program, shaders.lighting.cam_inv_view_proj, camera_pass.inv_view_proj)
    SetShaderValue(shaders.lighting.program, shaders.lighting.light_dir, ffi.addressof(render.light_dir), SHADER_UNIFORM_VEC3)
    SetShaderValue(shaders.lighting.program, shaders.lighting.sun_color, ffi.addressof(sun_color), SHADER_UNIFORM_VEC3)
    SetShaderValue(shaders.lighting.program, shaders.lighting.sun_strength, sun_strength_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.lighting.program, shaders.lighting.sky_color, ffi.addressof(sky_color), SHADER_UNIFORM_VEC3)
    SetShaderValue(shaders.lighting.program, shaders.lighting.sky_strength, sky_strength_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.lighting.program, shaders.lighting.ground_strength, ground_strength_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.lighting.program, shaders.lighting.ambient_strength, ambient_strength_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.lighting.program, shaders.lighting.exposure, exposure_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.lighting.program, shaders.lighting.cam_clip_near, camera_pass.clip_near_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.lighting.program, shaders.lighting.cam_clip_far, camera_pass.clip_far_ptr, SHADER_UNIFORM_FLOAT)

    ClearBackground(RAYWHITE)
    DrawTextureRec(
        render.gbuffer.color,
        Rectangle(0, 0, render.gbuffer.color.width, -render.gbuffer.color.height),
        Vector2(0, 0),
        WHITE,
    )

    EndShaderMode()

    BeginMode3D(app.camera.cam3d)
    DrawRootTrajectoryDebug(
        app.runtime_state.trajectory_positions_world,
        app.runtime_state.trajectory_directions_world,
        app.runtime_state.trajectory_velocities_world,
        app.runtime_state.sample_offsets,
        drawDirection=True,
        drawVelocity=True,
        directionScale=0.2,
        velocityScale=0.15,
    )
    local_debug_trajectory = _build_local_debug_trajectory(app.current_features, app.local_debug_origin)
    if local_debug_trajectory is not None:
        local_positions, local_directions, local_velocities = local_debug_trajectory
        DrawRootTrajectoryDebug(
            local_positions,
            local_directions,
            local_velocities,
            app.runtime_state.sample_offsets,
            drawDirection=True,
            drawVelocity=True,
            directionScale=0.2,
            velocityScale=0.15,
        )
    EndMode3D()

    EndTextureMode()


def _draw_static_ui(app, frame_state):
    GuiGroupBox(Rectangle(20, 10, 420, 210), b"Geno MANN Viewer")
    GuiLabel(Rectangle(30, 30, 340, 20), b"Ctrl + Left Click - Rotate")
    GuiLabel(Rectangle(30, 50, 340, 20), b"Ctrl + Right Click - Pan")
    GuiLabel(Rectangle(30, 70, 340, 20), b"Mouse Scroll - Zoom")
    GuiLabel(Rectangle(30, 90, 340, 20), b"Left Stick move, Right Stick face")
    GuiLabel(Rectangle(30, 110, 340, 20), b"RT run, LT strafe, A jump, Start reset")
    GuiLabel(Rectangle(30, 130, 340, 20), f"Clip: {frame_state.clip_label}".encode("utf-8"))
    GuiLabel(Rectangle(30, 150, 340, 20), f"Frame: {frame_state.frame_index}  Action: {frame_state.action_label}".encode("utf-8"))
    GuiLabel(Rectangle(30, 170, 340, 20), b"Blue Geno = model local prediction")
    GuiLabel(
        Rectangle(30, 190, 380, 20),
        (
            b"Prediction: off"
            if app.debug_prediction is None else
            (
                f"y_root: [{app.debug_prediction.y_root[0]: .2f} "
                f"{app.debug_prediction.y_root[1]: .2f} "
                f"{app.debug_prediction.y_root[2]: .2f}]"
            ).encode("utf-8")
        ),
    )

    GuiGroupBox(Rectangle(20, 230, 420, 240), b"Input Debug")
    GuiLabel(
        Rectangle(30, 250, 340, 20),
        (
            f"Source: {frame_state.input_state.input_source}  "
            f"Move: [{frame_state.input_state.move_2d[0]: .2f} "
            f"{frame_state.input_state.move_2d[1]: .2f}]"
        ).encode("utf-8"),
    )
    GuiLabel(
        Rectangle(30, 270, 340, 20),
        (
            f"Look: [{frame_state.input_state.look_2d[0]: .2f} "
            f"{frame_state.input_state.look_2d[1]: .2f}]"
        ).encode("utf-8"),
    )
    GuiLabel(
        Rectangle(30, 290, 340, 20),
        (
            f"Run: {'on' if frame_state.input_state.run_pressed else 'off'}  "
            f"Strafe: {'on' if frame_state.control_intent.desired_strafe else 'off'}  "
            f"Jump: {'yes' if frame_state.input_state.jump_pressed else 'no'}  "
            f"Look: {'on' if frame_state.input_state.look_active else 'off'}"
        ).encode("utf-8"),
    )
    GuiLabel(
        Rectangle(30, 310, 340, 20),
        (
            f"Desired Vel: [{frame_state.control_intent.desired_velocity_world[0]: .2f} "
            f"{frame_state.control_intent.desired_velocity_world[2]: .2f}]"
        ).encode("utf-8"),
    )
    GuiLabel(
        Rectangle(30, 330, 340, 20),
        (
            f"Facing Dir: [{frame_state.control_intent.desired_facing_world[0]: .2f} "
            f"{frame_state.control_intent.desired_facing_world[2]: .2f}]"
        ).encode("utf-8"),
    )
    GuiLabel(
        Rectangle(30, 350, 340, 20),
        (
            f"Root Vel: [{app.runtime_state.root_velocity[0]: .2f} "
            f"{app.runtime_state.root_velocity[2]: .2f}]"
        ).encode("utf-8"),
    )
    GuiLabel(
        Rectangle(30, 370, 340, 20),
        (
            f"x_main: {frame_state.feature_inputs.x_main.shape[0]}  "
            f"x_gate: {frame_state.feature_inputs.x_gate.shape[0]}"
        ).encode("utf-8"),
    )
    GuiLabel(
        Rectangle(30, 390, 340, 20),
        (
            f"speed@curr: "
            f"{frame_state.feature_inputs.speed_horizon[HUMANOID_LOCOMOTION_TRAJECTORY_CURRENT_SAMPLE_INDEX]: .2f}"
        ).encode("utf-8"),
    )
    GuiLabel(
        Rectangle(30, 410, 380, 20),
        (
            f"x_main norm: {float(np.linalg.norm(frame_state.feature_inputs.x_main)): .2f}  "
            f"x_gate norm: {float(np.linalg.norm(frame_state.feature_inputs.x_gate)): .2f}"
        ).encode("utf-8"),
    )


def _unload_static_viewer_resources(app):
    if app is None:
        return

    UnloadRenderTexture(app.render.lighted)
    UnloadRenderTexture(app.render.ssao_back)
    UnloadRenderTexture(app.render.ssao_front)
    UnloadGBuffer(app.render.gbuffer)
    UnloadShadowMap(app.render.shadow_map)

    UnloadModel(app.scene.pose_model)
    UnloadModel(app.scene.geno_model)
    UnloadModel(app.scene.ground_model)

    UnloadShader(app.shaders.fxaa.program)
    UnloadShader(app.shaders.blur.program)
    UnloadShader(app.shaders.ssao.program)
    UnloadShader(app.shaders.lighting.program)
    UnloadShader(app.shaders.basic.program)
    UnloadShader(app.shaders.skinned_basic.program)
    UnloadShader(app.shaders.skinned_shadow.program)
    UnloadShader(app.shaders.shadow.program)


def _run_static_viewer(args) -> None:
    config = ViewerConfig(
        clip_path=args.clip,
        initial_frame=args.frame,
        screen_width=args.screen_width,
        screen_height=args.screen_height,
        gamepad_id=args.gamepad_id,
    )

    SetConfigFlags(FLAG_VSYNC_HINT)
    InitWindow(config.screen_width, config.screen_height, b"Geno MANN Controller")
    SetTargetFPS(60)

    app = None
    try:
        app = _create_viewer_state(config)
        rlSetClipPlanes(0.01, 50.0)

        while not WindowShouldClose():
            frame_state = _update_static_tracking(app)

            rlDisableColorBlend()
            BeginDrawing()

            shadow_pass = _render_static_shadow_pass(app)
            camera_pass = _render_static_gbuffer_pass(app)
            _render_ssao_and_blur_pass(app, shadow_pass, camera_pass)
            _render_static_lighting_pass(app, camera_pass)
            _render_final_pass(app)

            rlEnableColorBlend()
            _draw_static_ui(app, frame_state)
            EndDrawing()
    finally:
        _unload_static_viewer_resources(app)
        CloseWindow()


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.self_test:
        _run_runtime_self_test(args)
    else:
        _run_static_viewer(args)


if __name__ == "__main__":
    main()
