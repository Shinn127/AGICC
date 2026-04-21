from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from genoview.utils import quat


@dataclass
class InertializationState:
    position_offsets: np.ndarray
    rotation_offsets: np.ndarray
    velocity_offsets: np.ndarray
    angular_velocity_offsets: np.ndarray


def make_inertialization_state(joint_count: int) -> InertializationState:
    joint_count = int(joint_count)
    return InertializationState(
        position_offsets=np.zeros((joint_count, 3), dtype=np.float32),
        rotation_offsets=quat.eye([joint_count]).astype(np.float32),
        velocity_offsets=np.zeros((joint_count, 3), dtype=np.float32),
        angular_velocity_offsets=np.zeros((joint_count, 3), dtype=np.float32),
    )


def reset_inertialization_state(state: InertializationState) -> None:
    state.position_offsets[:] = 0.0
    state.rotation_offsets[:] = quat.eye([len(state.rotation_offsets)]).astype(np.float32)
    state.velocity_offsets[:] = 0.0
    state.angular_velocity_offsets[:] = 0.0


def begin_transition_from_output(
    state: InertializationState,
    output_positions: np.ndarray,
    output_rotations: np.ndarray,
    output_velocities: np.ndarray,
    output_angular_velocities: np.ndarray,
    target_positions: np.ndarray,
    target_rotations: np.ndarray,
    target_velocities: np.ndarray,
    target_angular_velocities: np.ndarray,
) -> None:
    state.position_offsets[:] = (
        np.asarray(output_positions, dtype=np.float32)
        - np.asarray(target_positions, dtype=np.float32)
    ).astype(np.float32)
    state.velocity_offsets[:] = (
        np.asarray(output_velocities, dtype=np.float32)
        - np.asarray(target_velocities, dtype=np.float32)
    ).astype(np.float32)
    state.rotation_offsets[:] = quat.abs(
        quat.mul_inv(
            np.asarray(output_rotations, dtype=np.float32),
            np.asarray(target_rotations, dtype=np.float32),
        )
    ).astype(np.float32)
    state.angular_velocity_offsets[:] = (
        np.asarray(output_angular_velocities, dtype=np.float32)
        - np.asarray(target_angular_velocities, dtype=np.float32)
    ).astype(np.float32)


def _halflife_to_damping(halflife: float, eps: float = 1e-5) -> float:
    return (4.0 * 0.69314718056) / (float(halflife) + eps)


def decay_spring_damper_position(
    position_offset: np.ndarray,
    velocity_offset: np.ndarray,
    halflife: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    y = _halflife_to_damping(halflife) / 2.0
    j1 = velocity_offset + position_offset * y
    eydt = np.exp(-y * float(dt))
    return (
        (eydt * (position_offset + j1 * float(dt))).astype(np.float32),
        (eydt * (velocity_offset - j1 * y * float(dt))).astype(np.float32),
    )


def decay_spring_damper_rotation(
    rotation_offset: np.ndarray,
    angular_velocity_offset: np.ndarray,
    halflife: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    y = _halflife_to_damping(halflife) / 2.0
    j0 = quat.to_scaled_angle_axis(quat.abs(rotation_offset)).astype(np.float32)
    j1 = angular_velocity_offset + j0 * y
    eydt = np.exp(-y * float(dt))
    return (
        quat.normalize(
            quat.from_scaled_angle_axis(eydt * (j0 + j1 * float(dt))).astype(np.float32)
        ).astype(np.float32),
        (eydt * (angular_velocity_offset - j1 * y * float(dt))).astype(np.float32),
    )


def update_inertialized_pose(
    state: InertializationState,
    target_positions: np.ndarray,
    target_rotations: np.ndarray,
    target_velocities: np.ndarray,
    target_angular_velocities: np.ndarray,
    halflife: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if float(halflife) <= 0.0:
        reset_inertialization_state(state)
        return (
            np.asarray(target_positions, dtype=np.float32).copy(),
            np.asarray(target_rotations, dtype=np.float32).copy(),
            np.asarray(target_velocities, dtype=np.float32).copy(),
            np.asarray(target_angular_velocities, dtype=np.float32).copy(),
        )

    state.position_offsets[:], state.velocity_offsets[:] = decay_spring_damper_position(
        state.position_offsets,
        state.velocity_offsets,
        halflife,
        dt,
    )
    state.rotation_offsets[:], state.angular_velocity_offsets[:] = decay_spring_damper_rotation(
        state.rotation_offsets,
        state.angular_velocity_offsets,
        halflife,
        dt,
    )

    output_positions = (np.asarray(target_positions, dtype=np.float32) + state.position_offsets).astype(np.float32)
    output_rotations = quat.normalize(
        quat.mul(state.rotation_offsets, np.asarray(target_rotations, dtype=np.float32))
    ).astype(np.float32)
    output_velocities = (np.asarray(target_velocities, dtype=np.float32) + state.velocity_offsets).astype(np.float32)
    output_angular_velocities = (
        np.asarray(target_angular_velocities, dtype=np.float32)
        + state.angular_velocity_offsets
    ).astype(np.float32)
    return output_positions, output_rotations, output_velocities, output_angular_velocities
