from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from MotionMatching.MotionMatchingConfig import (
    DEFAULT_BVH_FRAME_TIME,
    MM_ACTION_BLEND_HALFLIFE,
    MM_FORCE_SEARCH_COOLDOWN,
    MM_FORCE_SEARCH_ENABLED,
    MM_FORCE_SEARCH_ROTATION_THRESHOLD,
    MM_FORCE_SEARCH_VELOCITY_THRESHOLD,
    MM_FOOT_POSITION_JOINTS,
    MM_INERTIALIZATION_HALFLIFE,
    MM_JUMP_ACTION_LABEL,
    MM_JUMP_CANDIDATE_RANGE_KEYWORD,
    MM_JUMP_ENTER_LOCK_TIME,
    MM_JUMP_ENTRY_WINDOW_FRAMES,
    MM_JUMP_EXIT_GRACE_TIME,
    MM_JUMP_MIN_SEGMENT_FRAMES,
    MM_ROOT_ADJUSTMENT_BY_VELOCITY,
    MM_ROOT_ADJUSTMENT_ENABLED,
    MM_ROOT_ADJUSTMENT_POSITION_HALFLIFE,
    MM_ROOT_ADJUSTMENT_POSITION_MAX_RATIO,
    MM_ROOT_ADJUSTMENT_ROTATION_HALFLIFE,
    MM_ROOT_ADJUSTMENT_ROTATION_MAX_RATIO,
    MM_ROOT_CLAMPING_ENABLED,
    MM_ROOT_CLAMPING_MAX_ANGLE,
    MM_ROOT_CLAMPING_MAX_DISTANCE,
    MM_ROOT_SYNCHRONIZATION_DATA_FACTOR,
    MM_ROOT_SYNCHRONIZATION_ENABLED,
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
ACTION_PHASE_LOCOMOTION = "locomotion"
ACTION_PHASE_JUMP_ENTER = "jump_enter"
ACTION_PHASE_JUMP_HOLD = "jump_hold"
ACTION_PHASE_JUMP_EXIT = "jump_exit"
CANDIDATE_MODE_DEFAULT = "default"
CANDIDATE_MODE_LOCOMOTION = "locomotion"
CANDIDATE_MODE_JUMP_ENTRY = "jump_entry"
CANDIDATE_MODE_JUMP_HOLD = "jump_hold"


@dataclass(frozen=True)
class RuntimeConfig:
    search_interval: float = MM_SEARCH_INTERVAL
    move_halflife: float = DEFAULT_MOVE_HALFLIFE
    rotation_halflife: float = DEFAULT_ROTATION_HALFLIFE
    action_blend_halflife: float = MM_ACTION_BLEND_HALFLIFE
    inertialization_halflife: float = MM_INERTIALIZATION_HALFLIFE
    transition_cooldown: float = 0.18
    jump_entry_window_frames: int = MM_JUMP_ENTRY_WINDOW_FRAMES
    jump_min_segment_frames: int = MM_JUMP_MIN_SEGMENT_FRAMES
    jump_enter_lock_time: float = MM_JUMP_ENTER_LOCK_TIME
    jump_exit_grace_time: float = MM_JUMP_EXIT_GRACE_TIME
    force_search_enabled: bool = MM_FORCE_SEARCH_ENABLED
    force_search_velocity_threshold: float = MM_FORCE_SEARCH_VELOCITY_THRESHOLD
    force_search_rotation_threshold: float = MM_FORCE_SEARCH_ROTATION_THRESHOLD
    force_search_cooldown: float = MM_FORCE_SEARCH_COOLDOWN
    root_adjustment_enabled: bool = MM_ROOT_ADJUSTMENT_ENABLED
    root_adjustment_by_velocity: bool = MM_ROOT_ADJUSTMENT_BY_VELOCITY
    root_adjustment_position_halflife: float = MM_ROOT_ADJUSTMENT_POSITION_HALFLIFE
    root_adjustment_rotation_halflife: float = MM_ROOT_ADJUSTMENT_ROTATION_HALFLIFE
    root_adjustment_position_max_ratio: float = MM_ROOT_ADJUSTMENT_POSITION_MAX_RATIO
    root_adjustment_rotation_max_ratio: float = MM_ROOT_ADJUSTMENT_ROTATION_MAX_RATIO
    root_clamping_enabled: bool = MM_ROOT_CLAMPING_ENABLED
    root_clamping_max_distance: float = MM_ROOT_CLAMPING_MAX_DISTANCE
    root_clamping_max_angle: float = MM_ROOT_CLAMPING_MAX_ANGLE
    root_synchronization_enabled: bool = MM_ROOT_SYNCHRONIZATION_ENABLED
    root_synchronization_data_factor: float = MM_ROOT_SYNCHRONIZATION_DATA_FACTOR
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
    jump_down: bool = False
    jump_pressed: bool = False
    jump_released: bool = False


@dataclass
class RuntimeState:
    current_index: int
    root_position: np.ndarray
    root_rotation: np.ndarray
    root_velocity: np.ndarray
    root_acceleration: np.ndarray
    root_angular_velocity: np.ndarray
    simulation_position: np.ndarray
    simulation_rotation: np.ndarray
    simulation_velocity: np.ndarray
    simulation_acceleration: np.ndarray
    simulation_angular_velocity: np.ndarray
    local_positions: np.ndarray
    local_rotations: np.ndarray
    local_velocities: np.ndarray
    local_angular_velocities: np.ndarray
    inertialization: InertializationState
    action_weights: np.ndarray
    previous_desired_velocity: np.ndarray
    previous_desired_rotation: np.ndarray
    desired_velocity_change_prev: np.ndarray
    desired_velocity_change_curr: np.ndarray
    desired_rotation_change_prev: np.ndarray
    desired_rotation_change_curr: np.ndarray
    search_timer: float = 0.0
    force_search_timer: float = 0.0
    transition_cooldown_timer: float = 0.0
    playback_accumulator: float = 0.0
    time: float = 0.0
    transition_count: int = 0
    last_search_result: SearchResult | None = None
    last_query_distance: float = 0.0
    last_current_score: float = 0.0
    last_force_search: bool = False
    last_search_reason: str = "none"
    active_action: str | None = None
    action_phase: str = ACTION_PHASE_LOCOMOTION
    action_lock_timer: float = 0.0
    pending_action_exit: bool = False
    last_candidate_mode: str = CANDIDATE_MODE_DEFAULT
    last_candidate_count: int = 0
    last_force_transition: bool = False


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
    simulation_position: np.ndarray
    simulation_rotation: np.ndarray
    simulation_velocity: np.ndarray
    root_position_error: float
    root_rotation_error: float
    world_positions: np.ndarray
    world_rotations: np.ndarray
    transitioned: bool
    transition_count: int
    transition_cooldown: float
    query_distance: float
    current_score: float
    forced_search: bool
    search_reason: str
    search_result: SearchResult | None
    active_action: str | None = None
    action_phase: str = ACTION_PHASE_LOCOMOTION
    pending_action_exit: bool = False
    candidate_mode: str = CANDIDATE_MODE_DEFAULT
    candidate_count: int = 0
    force_transition: bool = False


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


def _yaw_from_rotation(rotation: np.ndarray) -> float:
    forward = quat.mul_vec(
        np.asarray(rotation, dtype=np.float32),
        np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    ).astype(np.float32)
    return float(np.arctan2(float(forward[0]), float(forward[2])))


def _yaw_rotation(yaw: float) -> np.ndarray:
    return quat.normalize(
        quat.from_angle_axis(float(yaw), np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
    ).astype(np.float32)


def _wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(float(angle)), np.cos(float(angle))))


def _rotation_angle_error(current: np.ndarray, target: np.ndarray) -> float:
    return abs(_wrap_angle(_yaw_from_rotation(current) - _yaw_from_rotation(target)))


def _clamp_vector_length(vector: np.ndarray, max_length: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    max_length = float(max_length)
    length = float(np.linalg.norm(vector[[0, 2]]))
    if length <= max_length or length <= 1e-8:
        return vector.astype(np.float32)
    return (vector * (max_length / length)).astype(np.float32)


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
        self.jump_action_id = self.search_index.resolve_action_id(MM_JUMP_ACTION_LABEL)
        self.jump_all_candidates = self._build_jump_all_candidates()
        self.jump_entry_candidates = self._build_jump_entry_candidates(
            self.jump_all_candidates,
            int(self.config.jump_entry_window_frames),
            int(self.config.jump_min_segment_frames),
        )
        self.locomotion_candidates = self._build_locomotion_candidates()

        if initial_index is None:
            initial_action_id = self.search_index.resolve_action_id(initial_action)
            initial_bucket = self.search_index.action_buckets.get(int(initial_action_id), np.zeros((0,), dtype=np.int32))
            if len(initial_bucket) == 0:
                initial_index = 0
            else:
                initial_index = int(initial_bucket[0])

        self.state = self._make_initial_state(int(initial_index), initial_action)

    def _build_locomotion_candidates(self) -> np.ndarray:
        mask = self.search_index.action_ids != int(self.jump_action_id)
        candidates = self.search_index.all_indices[mask].astype(np.int32)
        return candidates if len(candidates) > 0 else self.search_index.all_indices

    def _build_jump_all_candidates(self) -> np.ndarray:
        candidates = self.search_index.action_buckets.get(
            int(self.jump_action_id),
            np.zeros((0,), dtype=np.int32),
        ).astype(np.int32)
        if len(candidates) == 0:
            return candidates

        keyword = str(MM_JUMP_CANDIDATE_RANGE_KEYWORD).lower()
        if not keyword:
            return candidates
        range_names = np.asarray(self.search_index.range_names, dtype=object)
        range_mask = np.asarray(
            [keyword in str(range_names[self.search_index.range_ids[index]]).lower() for index in candidates],
            dtype=bool,
        )
        filtered = candidates[range_mask].astype(np.int32)
        return filtered if len(filtered) > 0 else candidates

    @staticmethod
    def _build_jump_entry_candidates(
        jump_candidates: np.ndarray,
        entry_window_frames: int,
        min_segment_frames: int,
    ) -> np.ndarray:
        jump_candidates = np.asarray(jump_candidates, dtype=np.int32).reshape(-1)
        if len(jump_candidates) == 0:
            return jump_candidates

        entry_window_frames = max(1, int(entry_window_frames))
        min_segment_frames = max(1, int(min_segment_frames))
        entries = []
        segment_start = 0
        for index in range(1, len(jump_candidates) + 1):
            at_segment_end = index == len(jump_candidates) or int(jump_candidates[index]) != int(jump_candidates[index - 1]) + 1
            if not at_segment_end:
                continue
            segment = jump_candidates[segment_start:index]
            if len(segment) >= min_segment_frames:
                entries.append(segment[:entry_window_frames])
            segment_start = index

        if not entries:
            return jump_candidates
        return np.concatenate(entries, axis=0).astype(np.int32)

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
            simulation_position=root_position.copy(),
            simulation_rotation=root_rotation.copy(),
            simulation_velocity=np.zeros(3, dtype=np.float32),
            simulation_acceleration=np.zeros(3, dtype=np.float32),
            simulation_angular_velocity=np.zeros(3, dtype=np.float32),
            local_positions=self.local_positions[initial_index].copy(),
            local_rotations=self.local_rotations[initial_index].copy(),
            local_velocities=self.local_velocities[initial_index].copy(),
            local_angular_velocities=self.local_angular_velocities[initial_index].copy(),
            inertialization=make_inertialization_state(self.local_positions.shape[1]),
            action_weights=normalize_action_weights(action_weights, action_count=len(self.database.action_labels)),
            previous_desired_velocity=np.zeros(3, dtype=np.float32),
            previous_desired_rotation=root_rotation.copy(),
            desired_velocity_change_prev=np.zeros(3, dtype=np.float32),
            desired_velocity_change_curr=np.zeros(3, dtype=np.float32),
            desired_rotation_change_prev=np.zeros(3, dtype=np.float32),
            desired_rotation_change_curr=np.zeros(3, dtype=np.float32),
            search_timer=0.0,
            force_search_timer=0.0,
            transition_cooldown_timer=0.0,
            active_action=None,
            action_phase=ACTION_PHASE_LOCOMOTION,
            action_lock_timer=0.0,
            pending_action_exit=False,
            last_candidate_mode=CANDIDATE_MODE_LOCOMOTION,
            last_candidate_count=len(self.locomotion_candidates),
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
        jump_down: bool = False,
        jump_pressed: bool = False,
        jump_released: bool = False,
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
            jump_down=bool(jump_down),
            jump_pressed=bool(jump_pressed),
            jump_released=bool(jump_released),
        )

    def _update_action_state(self, intent: ControlIntent, dt: float) -> tuple[bool, bool]:
        self.state.action_lock_timer = max(0.0, float(self.state.action_lock_timer) - float(dt))
        self.state.last_force_transition = False

        wants_jump_enter = bool(intent.jump_pressed or (intent.jump_down and self.state.active_action is None))
        if wants_jump_enter and len(self.jump_entry_candidates) > 0:
            self.state.active_action = MM_JUMP_ACTION_LABEL
            self.state.action_phase = ACTION_PHASE_JUMP_ENTER
            self.state.pending_action_exit = False
            self.state.action_lock_timer = max(
                float(self.state.action_lock_timer),
                float(self.config.jump_enter_lock_time),
            )
            self.state.search_timer = 0.0
            self.state.last_force_transition = True
            return True, True

        if self.state.active_action == MM_JUMP_ACTION_LABEL:
            if bool(intent.jump_released or not intent.jump_down):
                self.state.pending_action_exit = True

            if self.state.pending_action_exit:
                if self.state.action_lock_timer <= 0.0:
                    self.state.active_action = None
                    self.state.action_phase = ACTION_PHASE_LOCOMOTION
                    self.state.pending_action_exit = False
                    self.state.action_lock_timer = float(self.config.jump_exit_grace_time)
                    self.state.search_timer = 0.0
                    self.state.last_force_transition = True
                    return True, True
                self.state.action_phase = ACTION_PHASE_JUMP_EXIT
            elif self.state.action_phase == ACTION_PHASE_JUMP_ENTER and self.state.action_lock_timer <= 0.0:
                self.state.action_phase = ACTION_PHASE_JUMP_HOLD
            elif self.state.action_phase not in (ACTION_PHASE_JUMP_ENTER, ACTION_PHASE_JUMP_EXIT):
                self.state.action_phase = ACTION_PHASE_JUMP_HOLD

        return False, False

    def _candidate_policy(self) -> tuple[np.ndarray | None, str]:
        if self.state.active_action == MM_JUMP_ACTION_LABEL:
            if self.state.action_phase == ACTION_PHASE_JUMP_ENTER:
                return self.jump_entry_candidates, CANDIDATE_MODE_JUMP_ENTRY
            return self.jump_all_candidates, CANDIDATE_MODE_JUMP_HOLD

        if self.state.action_phase == ACTION_PHASE_LOCOMOTION:
            return self.locomotion_candidates, CANDIDATE_MODE_LOCOMOTION
        return None, CANDIDATE_MODE_DEFAULT

    def _update_action_weights(self, intent: ControlIntent, dt: float) -> None:
        if self.state.active_action == MM_JUMP_ACTION_LABEL:
            self.state.action_weights = make_action_weights(
                MM_JUMP_ACTION_LABEL,
                action_labels=self.database.action_labels,
            )
            return

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

    def _update_simulation(self, intent: ControlIntent, dt: float) -> None:
        (
            self.state.simulation_position,
            self.state.simulation_velocity,
            self.state.simulation_acceleration,
        ) = _simulation_positions_update(
            self.state.simulation_position,
            self.state.simulation_velocity,
            self.state.simulation_acceleration,
            intent.desired_velocity_world,
            self.config.move_halflife,
            dt,
        )
        (
            self.state.simulation_rotation,
            self.state.simulation_angular_velocity,
        ) = _simulation_rotations_update(
            self.state.simulation_rotation,
            self.state.simulation_angular_velocity,
            intent.desired_rotation,
            self.config.rotation_halflife,
            dt,
        )

    def _update_force_search_state(self, intent: ControlIntent, dt: float) -> bool:
        if not self.config.force_search_enabled:
            self.state.last_force_search = False
            return False

        dt_safe = max(float(dt), 1e-6)
        velocity_change_prev = self.state.desired_velocity_change_curr.copy()
        velocity_change_curr = (
            np.asarray(intent.desired_velocity_world, dtype=np.float32)
            - self.state.previous_desired_velocity
        ) / dt_safe

        rotation_delta = quat.abs(
            quat.mul_inv(
                np.asarray(intent.desired_rotation, dtype=np.float32),
                self.state.previous_desired_rotation,
            )
        ).astype(np.float32)
        rotation_change_prev = self.state.desired_rotation_change_curr.copy()
        rotation_change_curr = quat.to_scaled_angle_axis(rotation_delta).astype(np.float32) / dt_safe
        rotation_change_curr[0] = 0.0
        rotation_change_curr[2] = 0.0

        self.state.previous_desired_velocity = np.asarray(intent.desired_velocity_world, dtype=np.float32).copy()
        self.state.previous_desired_rotation = np.asarray(intent.desired_rotation, dtype=np.float32).copy()
        self.state.desired_velocity_change_prev = velocity_change_prev
        self.state.desired_velocity_change_curr = velocity_change_curr.astype(np.float32)
        self.state.desired_rotation_change_prev = rotation_change_prev
        self.state.desired_rotation_change_curr = rotation_change_curr.astype(np.float32)

        self.state.force_search_timer = max(0.0, float(self.state.force_search_timer) - float(dt))
        if self.state.force_search_timer > 0.0:
            self.state.last_force_search = False
            return False

        velocity_settled = (
            float(np.linalg.norm(velocity_change_prev[[0, 2]])) >= float(self.config.force_search_velocity_threshold)
            and float(np.linalg.norm(velocity_change_curr[[0, 2]])) < float(self.config.force_search_velocity_threshold)
        )
        rotation_settled = (
            float(np.linalg.norm(rotation_change_prev)) >= float(self.config.force_search_rotation_threshold)
            and float(np.linalg.norm(rotation_change_curr)) < float(self.config.force_search_rotation_threshold)
        )
        force_search = bool(velocity_settled or rotation_settled)
        if force_search:
            self.state.force_search_timer = float(self.config.force_search_cooldown)
        self.state.last_force_search = force_search
        return force_search

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
            self.state.simulation_position,
            self.state.simulation_velocity,
            self.state.simulation_acceleration,
            intent.desired_velocity_world,
            self.config.move_halflife,
            sample_times,
        )
        rotations, _ = _simulation_rotations_predict_samples(
            self.state.simulation_rotation,
            self.state.simulation_angular_velocity,
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

    def _search_if_needed(
        self,
        intent: ControlIntent,
        dt: float,
        force_search: bool = False,
        force_transition: bool = False,
        candidate_indices: np.ndarray | None = None,
        candidate_mode: str = CANDIDATE_MODE_DEFAULT,
    ) -> tuple[bool, SearchResult | None, float]:
        self.state.search_timer -= float(dt)
        self.state.transition_cooldown_timer = max(
            0.0,
            float(self.state.transition_cooldown_timer) - float(dt),
        )
        candidate_count = 0 if candidate_indices is None else int(len(candidate_indices))
        self.state.last_candidate_mode = str(candidate_mode)
        self.state.last_candidate_count = candidate_count
        frames_until_end = self.search_index.get_frames_until_range_end(self.state.current_index)
        range_end_search = frames_until_end <= max(1, int(self.config.search_config.ignore_range_end_frames))
        should_search = bool(force_search or range_end_search or self.state.search_timer <= 0.0)
        if not should_search:
            self.state.last_search_reason = "none"
            return False, self.state.last_search_result, float(self.state.last_query_distance)

        query_feature = self.build_query_feature(intent)
        current_distance = float(np.linalg.norm(query_feature - self.features[self.state.current_index]))
        current_score = self.search_index.score_candidate(
            query_feature,
            self.state.current_index,
            action_weights=self.state.action_weights,
        )
        self.state.last_query_distance = current_distance
        self.state.last_current_score = current_score

        result = self.search_index.search(
            query_feature,
            current_index=self.state.current_index,
            action_weights=self.state.action_weights,
            candidate_indices=candidate_indices,
        )
        self.state.search_timer = float(self.config.search_interval)
        self.state.last_search_result = result
        if force_transition:
            self.state.last_search_reason = f"{candidate_mode}:force_transition"
        elif force_search:
            self.state.last_search_reason = "force"
        elif range_end_search:
            self.state.last_search_reason = "range_end"
        else:
            self.state.last_search_reason = "interval"

        transitioned = False
        can_transition = bool(
            result.index != self.state.current_index
            and (
                force_transition
                or (
                    self.state.transition_cooldown_timer <= 0.0
                    and self.search_index.should_transition(current_score, result)
                )
            )
        )
        if can_transition:
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

    def _apply_root_synchronization(self) -> bool:
        if not self.config.root_synchronization_enabled:
            return False

        data_factor = float(np.clip(self.config.root_synchronization_data_factor, 0.0, 1.0))
        target_position = (
            (1.0 - data_factor) * self.state.simulation_position
            + data_factor * self.state.root_position
        ).astype(np.float32)
        target_position[1] = 0.0

        simulation_yaw = _yaw_from_rotation(self.state.simulation_rotation)
        root_yaw = _yaw_from_rotation(self.state.root_rotation)
        target_yaw = simulation_yaw + _wrap_angle(root_yaw - simulation_yaw) * data_factor
        target_rotation = _yaw_rotation(target_yaw)

        self.state.simulation_position = target_position.copy()
        self.state.simulation_rotation = target_rotation.copy()
        self.state.root_position = target_position.copy()
        self.state.root_rotation = target_rotation.copy()
        return True

    def _apply_root_adjustment(self, dt: float) -> None:
        if not self.config.root_adjustment_enabled:
            return

        position_delta = (self.state.simulation_position - self.state.root_position).astype(np.float32)
        position_delta[1] = 0.0
        position_correction = (
            position_delta
            - _blend_halflife(
                position_delta,
                np.zeros(3, dtype=np.float32),
                self.config.root_adjustment_position_halflife,
                dt,
            )
        ).astype(np.float32)

        if self.config.root_adjustment_by_velocity:
            speed = max(
                float(np.linalg.norm(self.state.root_velocity[[0, 2]])),
                float(np.linalg.norm(self.state.simulation_velocity[[0, 2]])),
                0.2,
            )
            max_length = (
                float(self.config.root_adjustment_position_max_ratio)
                * speed
                * max(float(dt), 0.0)
            )
            position_correction = _clamp_vector_length(position_correction, max_length)

        self.state.root_position = (self.state.root_position + position_correction).astype(np.float32)
        self.state.root_position[1] = 0.0

        root_yaw = _yaw_from_rotation(self.state.root_rotation)
        simulation_yaw = _yaw_from_rotation(self.state.simulation_rotation)
        yaw_error = _wrap_angle(simulation_yaw - root_yaw)
        yaw_correction = yaw_error - _blend_halflife(
            np.asarray([yaw_error], dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            self.config.root_adjustment_rotation_halflife,
            dt,
        )[0]

        if self.config.root_adjustment_by_velocity:
            angular_speed = max(
                abs(float(self.state.root_angular_velocity[1])),
                abs(float(self.state.simulation_angular_velocity[1])),
                0.5,
            )
            max_angle = (
                float(self.config.root_adjustment_rotation_max_ratio)
                * angular_speed
                * max(float(dt), 0.0)
            )
            yaw_correction = float(np.clip(yaw_correction, -max_angle, max_angle))

        self.state.root_rotation = _yaw_rotation(root_yaw + yaw_correction)

    def _apply_root_clamping(self) -> None:
        if not self.config.root_clamping_enabled:
            return

        max_distance = float(self.config.root_clamping_max_distance)
        if max_distance >= 0.0:
            offset = (self.state.root_position - self.state.simulation_position).astype(np.float32)
            offset[1] = 0.0
            distance = float(np.linalg.norm(offset[[0, 2]]))
            if distance > max_distance and distance > 1e-8:
                self.state.root_position = (
                    self.state.simulation_position + offset * (max_distance / distance)
                ).astype(np.float32)
                self.state.root_position[1] = 0.0

        max_angle = float(self.config.root_clamping_max_angle)
        if max_angle >= 0.0:
            simulation_yaw = _yaw_from_rotation(self.state.simulation_rotation)
            root_yaw = _yaw_from_rotation(self.state.root_rotation)
            yaw_from_simulation = _wrap_angle(root_yaw - simulation_yaw)
            clamped_yaw = simulation_yaw + float(np.clip(yaw_from_simulation, -max_angle, max_angle))
            self.state.root_rotation = _yaw_rotation(clamped_yaw)

    def _apply_root_quality_layer(self, dt: float) -> None:
        if self._apply_root_synchronization():
            return
        self._apply_root_adjustment(dt)
        self._apply_root_clamping()

    def _refresh_root_derivatives(
        self,
        previous_root_position: np.ndarray,
        previous_root_rotation: np.ndarray,
        previous_root_velocity: np.ndarray,
        dt: float,
    ) -> None:
        dt_safe = max(float(dt), 1e-6)
        next_velocity = ((self.state.root_position - previous_root_position) / dt_safe).astype(np.float32)
        next_velocity[1] = 0.0
        self.state.root_acceleration = ((next_velocity - previous_root_velocity) / dt_safe).astype(np.float32)
        self.state.root_acceleration[1] = 0.0
        self.state.root_velocity = next_velocity

        rotation_delta = quat.abs(
            quat.mul_inv(self.state.root_rotation, np.asarray(previous_root_rotation, dtype=np.float32))
        ).astype(np.float32)
        root_angular_velocity = (quat.to_scaled_angle_axis(rotation_delta) / dt_safe).astype(np.float32)
        root_angular_velocity[0] = 0.0
        root_angular_velocity[2] = 0.0
        self.state.root_angular_velocity = root_angular_velocity

    def _root_position_error(self) -> float:
        delta = (self.state.root_position - self.state.simulation_position).astype(np.float32)
        delta[1] = 0.0
        return float(np.linalg.norm(delta[[0, 2]]))

    def _root_rotation_error(self) -> float:
        return _rotation_angle_error(self.state.root_rotation, self.state.simulation_rotation)

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
        action_force_search, action_force_transition = self._update_action_state(intent, dt)
        self._update_action_weights(intent, dt)
        self._update_simulation(intent, dt)
        force_search = self._update_force_search_state(intent, dt)
        candidate_indices, candidate_mode = self._candidate_policy()
        transitioned, search_result, query_distance = self._search_if_needed(
            intent,
            dt,
            force_search=bool(force_search or action_force_search),
            force_transition=bool(action_force_transition),
            candidate_indices=candidate_indices,
            candidate_mode=candidate_mode,
        )
        self._advance_frame(dt)
        self._update_output_pose_from_database(dt)
        previous_root_position = self.state.root_position.copy()
        previous_root_rotation = self.state.root_rotation.copy()
        previous_root_velocity = self.state.root_velocity.copy()
        self._integrate_root_motion(dt)
        self._apply_root_quality_layer(dt)
        self._refresh_root_derivatives(
            previous_root_position,
            previous_root_rotation,
            previous_root_velocity,
            dt,
        )
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
            simulation_position=self.state.simulation_position.copy(),
            simulation_rotation=self.state.simulation_rotation.copy(),
            simulation_velocity=self.state.simulation_velocity.copy(),
            root_position_error=self._root_position_error(),
            root_rotation_error=self._root_rotation_error(),
            world_positions=world_positions,
            world_rotations=world_rotations,
            transitioned=bool(transitioned),
            transition_count=int(self.state.transition_count),
            transition_cooldown=float(self.state.transition_cooldown_timer),
            query_distance=float(query_distance),
            current_score=float(self.state.last_current_score),
            forced_search=bool(force_search or action_force_search),
            search_reason=str(self.state.last_search_reason),
            search_result=search_result,
            active_action=self.state.active_action,
            action_phase=str(self.state.action_phase),
            pending_action_exit=bool(self.state.pending_action_exit),
            candidate_mode=str(self.state.last_candidate_mode),
            candidate_count=int(self.state.last_candidate_count),
            force_transition=bool(self.state.last_force_transition),
        )
