from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from MotionMatching.MotionMatchingConfig import (
    DEFAULT_BVH_FRAME_TIME,
    MM_ACTION_BLEND_HALFLIFE,
    MM_FOOT_POSITION_JOINTS,
    MM_INERTIALIZATION_HALFLIFE,
    MM_SEARCH_INTERVAL,
    MM_VELOCITY_JOINTS,
)
from MotionMatching.MotionMatchingDataset import MotionMatchingDataset
from MotionMatching.MotionMatchingFeatures import (
    make_action_weights,
    normalize_action_weights,
    normalize_and_weight_features,
    resolve_joint_indices,
)
from MotionMatching.MotionMatchingInertialization import (
    InertializationState,
    begin_transition_from_output,
    make_inertialization_state,
    update_inertialized_pose,
)
from MotionMatching.MotionMatchingSearch import MotionMatchingSearchIndex, SearchConfig, SearchResult
from genoview.utils import quat


DEFAULT_WALK_SPEED = 1.5
DEFAULT_RUN_SPEED = 3.0
DEFAULT_MOVE_HALFLIFE = 0.2
DEFAULT_ROTATION_HALFLIFE = 0.15


@dataclass(frozen=True)
class RuntimeConfig:
    search_interval: float = MM_SEARCH_INTERVAL
    move_halflife: float = DEFAULT_MOVE_HALFLIFE
    rotation_halflife: float = DEFAULT_ROTATION_HALFLIFE
    action_blend_halflife: float = MM_ACTION_BLEND_HALFLIFE
    inertialization_halflife: float = MM_INERTIALIZATION_HALFLIFE
    transition_cooldown: float = 0.18
    search_config: SearchConfig = SearchConfig()


@dataclass(frozen=True)
class ControlIntent:
    desired_velocity_world: np.ndarray
    desired_facing_world: np.ndarray
    desired_rotation: np.ndarray
    action_label: str
    action_weights: np.ndarray
    move_magnitude: float
    desired_strafe: bool = False


@dataclass
class RuntimeState:
    current_index: int
    root_position: np.ndarray
    root_rotation: np.ndarray
    root_velocity: np.ndarray
    root_acceleration: np.ndarray
    root_angular_velocity: np.ndarray
    local_positions: np.ndarray
    local_rotations: np.ndarray
    local_velocities: np.ndarray
    local_angular_velocities: np.ndarray
    inertialization: InertializationState
    action_weights: np.ndarray
    search_timer: float = 0.0
    transition_cooldown_timer: float = 0.0
    playback_accumulator: float = 0.0
    time: float = 0.0
    transition_count: int = 0
    last_search_result: SearchResult | None = None
    last_query_distance: float = 0.0


@dataclass(frozen=True)
class RuntimeFrame:
    time: float
    current_index: int
    range_index: int
    range_name: str
    action_id: int
    action_label: str
    action_weights: np.ndarray
    root_position: np.ndarray
    root_rotation: np.ndarray
    root_velocity: np.ndarray
    world_positions: np.ndarray
    world_rotations: np.ndarray
    transitioned: bool
    transition_count: int
    transition_cooldown: float
    query_distance: float
    search_result: SearchResult | None


def _halflife_to_damping(halflife: float, eps: float = 1e-5) -> float:
    return (4.0 * 0.69314718056) / (float(halflife) + eps)


def _blend_halflife(current: np.ndarray, target: np.ndarray, halflife: float, dt: float) -> np.ndarray:
    if halflife <= 0.0:
        return np.asarray(target, dtype=np.float32)
    alpha = float(1.0 - np.exp(-0.69314718056 * float(dt) / float(halflife)))
    return ((1.0 - alpha) * current + alpha * target).astype(np.float32)


def _normalize_xz(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).copy()
    vector[1] = 0.0
    norm = float(np.linalg.norm(vector))
    if norm > 1e-6:
        return (vector / norm).astype(np.float32)

    if fallback is None:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    fallback = np.asarray(fallback, dtype=np.float32).copy()
    fallback[1] = 0.0
    fallback_norm = float(np.linalg.norm(fallback))
    if fallback_norm < 1e-6:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    return (fallback / fallback_norm).astype(np.float32)


def _yaw_rotation_from_direction(direction: np.ndarray, fallback_rotation: np.ndarray) -> np.ndarray:
    fallback_forward = quat.mul_vec(
        fallback_rotation,
        np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    ).astype(np.float32)
    direction = _normalize_xz(direction, fallback=fallback_forward)
    yaw = float(np.arctan2(direction[0], direction[2]))
    return quat.normalize(
        quat.from_angle_axis(yaw, np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
    ).astype(np.float32)


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
    next_position[..., 1] = 0.0
    next_velocity[..., 1] = 0.0
    next_acceleration[..., 1] = 0.0
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
    next_rotation = quat.normalize(
        quat.mul(
            quat.from_scaled_angle_axis(eydt * (j0 + j1 * dt)).astype(np.float32),
            desired_rotation,
        )
    ).astype(np.float32)
    next_angular_velocity = (eydt * (angular_velocity - j1 * y * dt)).astype(np.float32)
    next_angular_velocity[0] = 0.0
    next_angular_velocity[2] = 0.0
    return next_rotation, next_angular_velocity


def _simulation_rotations_predict_samples(
    rotation: np.ndarray,
    angular_velocity: np.ndarray,
    desired_rotation: np.ndarray,
    halflife: float,
    sample_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y = _halflife_to_damping(halflife) / 2.0
    sample_times = np.asarray(sample_times, dtype=np.float32).reshape(-1, 1)
    j0 = quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotation, desired_rotation))).astype(np.float32)
    j1 = angular_velocity + j0 * y
    eydt = np.exp(-y * sample_times).astype(np.float32)
    rotations = quat.normalize(
        quat.mul(
            quat.from_scaled_angle_axis(eydt * (j0[np.newaxis, :] + j1[np.newaxis, :] * sample_times)).astype(np.float32),
            desired_rotation,
        )
    ).astype(np.float32)
    angular_velocities = (
        eydt * (angular_velocity[np.newaxis, :] - j1[np.newaxis, :] * y * sample_times)
    ).astype(np.float32)
    angular_velocities[:, 0] = 0.0
    angular_velocities[:, 2] = 0.0
    return rotations, angular_velocities


class MotionMatchingRuntime:
    def __init__(
        self,
        database: MotionMatchingDataset,
        config: RuntimeConfig | None = None,
        initial_action: str = "idle",
        initial_index: int | None = None,
    ) -> None:
        self.database = database
        self.config = config or RuntimeConfig()
        self.search_index = MotionMatchingSearchIndex.from_dataset(
            database,
            config=self.config.search_config,
        )
        self.features = self.search_index.features
        self.feature_mean = database.feature_mean
        self.feature_std = database.feature_std
        self.feature_layout = database.spec.feature_layout
        self.local_positions = database.data["local_positions"].astype(np.float32)
        self.local_rotations = database.data["local_rotations"].astype(np.float32)
        self.local_velocities = database.data["local_velocities"].astype(np.float32)
        self.local_angular_velocities = database.data["local_angular_velocities"].astype(np.float32)
        self.root_local_velocities = database.data["root_local_velocities"].astype(np.float32)
        self.root_local_angular_velocities = database.data["root_local_angular_velocities"].astype(np.float32)
        self.joint_names = database.joint_names
        self.parents = database.parents
        self.foot_joint_indices = resolve_joint_indices(self.joint_names, MM_FOOT_POSITION_JOINTS)
        self.velocity_joint_indices = resolve_joint_indices(self.joint_names, MM_VELOCITY_JOINTS)
        self.db_dt = float(database.spec.dt or DEFAULT_BVH_FRAME_TIME)

        if initial_index is None:
            initial_action_id = self.search_index.resolve_action_id(initial_action)
            initial_bucket = self.search_index.action_buckets.get(int(initial_action_id), np.zeros((0,), dtype=np.int32))
            if len(initial_bucket) == 0:
                initial_index = 0
            else:
                initial_index = int(initial_bucket[0])

        self.state = self._make_initial_state(int(initial_index), initial_action)

    def _make_initial_state(self, initial_index: int, initial_action: str | None = None) -> RuntimeState:
        root_position = np.zeros(3, dtype=np.float32)
        root_rotation = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        if initial_action is None:
            action_weights = self.database.action_weights[initial_index]
        else:
            action_weights = make_action_weights(initial_action, action_labels=self.database.action_labels)
        return RuntimeState(
            current_index=initial_index,
            root_position=root_position,
            root_rotation=root_rotation,
            root_velocity=np.zeros(3, dtype=np.float32),
            root_acceleration=np.zeros(3, dtype=np.float32),
            root_angular_velocity=np.zeros(3, dtype=np.float32),
            local_positions=self.local_positions[initial_index].copy(),
            local_rotations=self.local_rotations[initial_index].copy(),
            local_velocities=self.local_velocities[initial_index].copy(),
            local_angular_velocities=self.local_angular_velocities[initial_index].copy(),
            inertialization=make_inertialization_state(self.local_positions.shape[1]),
            action_weights=normalize_action_weights(action_weights, action_count=len(self.database.action_labels)),
            search_timer=0.0,
            transition_cooldown_timer=0.0,
        )

    def reset(self, initial_action: str = "idle", initial_index: int | None = None) -> None:
        if initial_index is None:
            initial_action_id = self.search_index.resolve_action_id(initial_action)
            initial_bucket = self.search_index.action_buckets.get(
                int(initial_action_id),
                np.zeros((0,), dtype=np.int32),
            )
            initial_index = int(initial_bucket[0]) if len(initial_bucket) > 0 else 0
        self.state = self._make_initial_state(int(initial_index), initial_action)

    def make_locomotion_intent(
        self,
        move_direction_world: np.ndarray,
        speed: float,
        action_label: str,
        facing_direction_world: np.ndarray | None = None,
        desired_strafe: bool = False,
    ) -> ControlIntent:
        move_direction_world = np.asarray(move_direction_world, dtype=np.float32)
        move_magnitude = float(np.linalg.norm(move_direction_world[[0, 2]]))
        if move_magnitude > 1e-6:
            desired_move_world = _normalize_xz(move_direction_world)
        else:
            desired_move_world = np.zeros(3, dtype=np.float32)

        current_forward = quat.mul_vec(
            self.state.root_rotation,
            np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        ).astype(np.float32)
        if facing_direction_world is not None:
            desired_facing_world = _normalize_xz(facing_direction_world, fallback=current_forward)
        elif desired_strafe:
            desired_facing_world = _normalize_xz(current_forward)
        elif move_magnitude > 1e-6:
            desired_facing_world = _normalize_xz(move_direction_world, fallback=current_forward)
        else:
            desired_facing_world = _normalize_xz(current_forward)

        desired_velocity_world = (desired_move_world * float(speed) * min(move_magnitude, 1.0)).astype(np.float32)
        desired_rotation = _yaw_rotation_from_direction(desired_facing_world, self.state.root_rotation)
        return ControlIntent(
            desired_velocity_world=desired_velocity_world,
            desired_facing_world=desired_facing_world,
            desired_rotation=desired_rotation,
            action_label=str(action_label),
            action_weights=make_action_weights(action_label, action_labels=self.database.action_labels),
            move_magnitude=move_magnitude,
            desired_strafe=bool(desired_strafe),
        )

    def _update_action_weights(self, intent: ControlIntent, dt: float) -> None:
        blended = _blend_halflife(
            self.state.action_weights,
            intent.action_weights,
            self.config.action_blend_halflife,
            dt,
        )
        self.state.action_weights = normalize_action_weights(
            blended,
            action_count=len(self.database.action_labels),
        )

    def predict_future_trajectory(self, intent: ControlIntent) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        offsets = np.asarray(self.database.spec.future_sample_offsets, dtype=np.int32)
        if len(offsets) == 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
            )

        sample_times = (offsets.astype(np.float32) * self.db_dt).reshape(-1, 1)
        positions, velocities, _ = _simulation_positions_update(
            self.state.root_position,
            self.state.root_velocity,
            self.state.root_acceleration,
            intent.desired_velocity_world,
            self.config.move_halflife,
            sample_times,
        )
        rotations, _ = _simulation_rotations_predict_samples(
            self.state.root_rotation,
            self.state.root_angular_velocity,
            intent.desired_rotation,
            self.config.rotation_halflife,
            sample_times,
        )

        directions = quat.mul_vec(
            rotations,
            np.repeat(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), len(offsets), axis=0),
        ).astype(np.float32)
        directions = np.asarray([_normalize_xz(direction) for direction in directions], dtype=np.float32)
        return positions, directions, velocities

    def build_query_feature(self, intent: ControlIntent) -> np.ndarray:
        future_positions, future_directions, future_velocities = self.predict_future_trajectory(intent)
        local_future_positions = quat.inv_mul_vec(
            self.state.root_rotation,
            future_positions - self.state.root_position[np.newaxis, :],
        ).astype(np.float32)
        local_future_directions = quat.inv_mul_vec(
            self.state.root_rotation,
            future_directions,
        ).astype(np.float32)
        local_future_velocities = quat.inv_mul_vec(
            self.state.root_rotation,
            future_velocities,
        ).astype(np.float32)

        raw_feature = np.concatenate(
            [
                self.state.local_positions[self.foot_joint_indices].reshape(-1),
                self.state.local_velocities[self.velocity_joint_indices].reshape(-1),
                local_future_positions[:, [0, 2]].reshape(-1),
                local_future_directions[:, [0, 2]].reshape(-1),
                local_future_velocities[:, [0, 2]].reshape(-1),
                self.state.action_weights,
            ]
        ).astype(np.float32)
        return normalize_and_weight_features(
            raw_feature[np.newaxis, :],
            self.feature_mean,
            self.feature_std,
            self.feature_layout,
        )[0]

    def _search_if_needed(self, intent: ControlIntent, dt: float) -> tuple[bool, SearchResult | None, float]:
        self.state.search_timer -= float(dt)
        self.state.transition_cooldown_timer = max(
            0.0,
            float(self.state.transition_cooldown_timer) - float(dt),
        )
        if self.state.search_timer > 0.0:
            return False, self.state.last_search_result, float(self.state.last_query_distance)

        query_feature = self.build_query_feature(intent)
        current_distance = float(np.linalg.norm(query_feature - self.features[self.state.current_index]))
        self.state.last_query_distance = current_distance

        result = self.search_index.search(
            query_feature,
            current_index=self.state.current_index,
        )
        self.state.search_timer = float(self.config.search_interval)
        self.state.last_search_result = result

        transitioned = False
        if (
            result.index != self.state.current_index
            and self.state.transition_cooldown_timer <= 0.0
            and self.search_index.should_transition(current_distance, result)
        ):
            self._begin_inertialized_transition(int(result.index))
            self.state.current_index = int(result.index)
            self.state.transition_count += 1
            self.state.transition_cooldown_timer = float(self.config.transition_cooldown)
            transitioned = True

        return transitioned, result, current_distance

    def _begin_inertialized_transition(self, target_index: int) -> None:
        begin_transition_from_output(
            self.state.inertialization,
            self.state.local_positions,
            self.state.local_rotations,
            self.state.local_velocities,
            self.state.local_angular_velocities,
            self.local_positions[target_index],
            self.local_rotations[target_index],
            self.local_velocities[target_index],
            self.local_angular_velocities[target_index],
        )

    def _update_output_pose_from_database(self, dt: float) -> None:
        index = self.state.current_index
        (
            self.state.local_positions,
            self.state.local_rotations,
            self.state.local_velocities,
            self.state.local_angular_velocities,
        ) = update_inertialized_pose(
            self.state.inertialization,
            self.local_positions[index],
            self.local_rotations[index],
            self.local_velocities[index],
            self.local_angular_velocities[index],
            self.config.inertialization_halflife,
            dt,
        )

    def _advance_frame(self, dt: float) -> None:
        self.state.playback_accumulator += float(dt) / max(self.db_dt, 1e-8)
        while self.state.playback_accumulator >= 1.0:
            next_index = self.search_index.get_next_index(self.state.current_index)
            if next_index is None:
                self.state.search_timer = 0.0
                self.state.playback_accumulator = 0.0
                break
            self.state.current_index = int(next_index)
            self.state.playback_accumulator -= 1.0

    def _integrate_root_motion(self, dt: float) -> None:
        index = self.state.current_index
        previous_velocity = self.state.root_velocity.copy()
        root_local_velocity = self.root_local_velocities[index]
        root_local_angular_velocity = self.root_local_angular_velocities[index]
        root_world_velocity = quat.mul_vec(self.state.root_rotation, root_local_velocity).astype(np.float32)
        root_world_angular_velocity = quat.mul_vec(self.state.root_rotation, root_local_angular_velocity).astype(np.float32)

        self.state.root_position = (self.state.root_position + root_world_velocity * float(dt)).astype(np.float32)
        self.state.root_rotation = quat.normalize(
            quat.mul(
                quat.from_scaled_angle_axis(root_world_angular_velocity * float(dt)).astype(np.float32),
                self.state.root_rotation,
            )
        ).astype(np.float32)
        self.state.root_position[1] = 0.0
        self.state.root_velocity = root_world_velocity.astype(np.float32)
        self.state.root_velocity[1] = 0.0
        self.state.root_acceleration = ((self.state.root_velocity - previous_velocity) / max(float(dt), 1e-6)).astype(np.float32)
        self.state.root_acceleration[1] = 0.0
        self.state.root_angular_velocity = root_world_angular_velocity.astype(np.float32)
        self.state.root_angular_velocity[0] = 0.0
        self.state.root_angular_velocity[2] = 0.0

    def _build_world_pose(self) -> tuple[np.ndarray, np.ndarray]:
        world_positions = (
            quat.mul_vec(self.state.root_rotation, self.state.local_positions)
            + self.state.root_position[np.newaxis, :]
        ).astype(np.float32)
        world_rotations = quat.normalize(
            quat.mul(self.state.root_rotation, self.state.local_rotations)
        ).astype(np.float32)
        return world_positions, world_rotations

    def update(self, intent: ControlIntent, dt: float) -> RuntimeFrame:
        self._update_action_weights(intent, dt)
        transitioned, search_result, query_distance = self._search_if_needed(intent, dt)
        self._advance_frame(dt)
        self._update_output_pose_from_database(dt)
        self._integrate_root_motion(dt)
        self.state.time += float(dt)

        world_positions, world_rotations = self._build_world_pose()
        range_index = self.search_index.get_range_index(self.state.current_index)
        action_id = int(self.search_index.action_ids[self.state.current_index])
        return RuntimeFrame(
            time=float(self.state.time),
            current_index=int(self.state.current_index),
            range_index=int(range_index),
            range_name=self.search_index.range_names[range_index],
            action_id=action_id,
            action_label=self.search_index.action_labels[action_id],
            action_weights=self.state.action_weights.copy(),
            root_position=self.state.root_position.copy(),
            root_rotation=self.state.root_rotation.copy(),
            root_velocity=self.state.root_velocity.copy(),
            world_positions=world_positions,
            world_rotations=world_rotations,
            transitioned=bool(transitioned),
            transition_count=int(self.state.transition_count),
            transition_cooldown=float(self.state.transition_cooldown_timer),
            query_distance=float(query_distance),
            search_result=search_result,
        )
