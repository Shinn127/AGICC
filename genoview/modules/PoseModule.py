import numpy as np

from genoview.utils import quat
from genoview.modules.RootModule import DEFAULT_BVH_FRAME_TIME, ComputeFiniteDifferenceVelocities

def _compute_pose_angular_velocities(globalRotations, dt=DEFAULT_BVH_FRAME_TIME):
    globalRotations = np.asarray(globalRotations, dtype=np.float32)
    angularVelocities = np.zeros(globalRotations.shape[:-1] + (3,), dtype=np.float32)

    if len(globalRotations) == 0:
        return angularVelocities
    if len(globalRotations) == 1:
        return angularVelocities
    if len(globalRotations) == 2:
        singleStepAngularVelocity = (
            quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(globalRotations[1], globalRotations[0]))) / dt
        ).astype(np.float32)
        angularVelocities[:] = singleStepAngularVelocity
        return angularVelocities

    angularVelocities[1:-1] = (
        0.5 * quat.to_scaled_angle_axis(
            quat.abs(quat.mul_inv(globalRotations[2:], globalRotations[1:-1]))
        ) / dt +
        0.5 * quat.to_scaled_angle_axis(
            quat.abs(quat.mul_inv(globalRotations[1:-1], globalRotations[:-2]))
        ) / dt
    ).astype(np.float32)

    if len(globalRotations) >= 4:
        angularVelocities[0] = angularVelocities[1] - (angularVelocities[3] - angularVelocities[2])
        angularVelocities[-1] = angularVelocities[-2] + (angularVelocities[-2] - angularVelocities[-3])
    else:
        angularVelocities[0] = angularVelocities[1]
        angularVelocities[-1] = angularVelocities[-2]

    return angularVelocities.astype(np.float32)


def _convert_local_rotations_to_6d(localRotations):
    rotationsXY = quat.to_xform_xy(localRotations).astype(np.float32)
    return rotationsXY.reshape(rotationsXY.shape[:-2] + (6,))


def BuildPoseSource(
    globalPositions,
    globalRotations,
    dt=DEFAULT_BVH_FRAME_TIME,
    rootTrajectorySource=None):

    poseSource = {
        "global_positions": np.asarray(globalPositions, dtype=np.float32),
        "global_rotations": np.asarray(globalRotations, dtype=np.float32),
    }
    poseSource["global_velocities"] = ComputeFiniteDifferenceVelocities(poseSource["global_positions"], dt)
    poseSource["global_angular_velocities"] = _compute_pose_angular_velocities(
        poseSource["global_rotations"],
        dt,
    )

    if rootTrajectorySource is not None:
        poseSource["root_angular_velocities"] = _compute_pose_angular_velocities(
            np.asarray(rootTrajectorySource["rotations"], dtype=np.float32),
            dt,
        )

    return poseSource


def BuildLocalPose(
    poseSource,
    rootTrajectorySource,
    currentFrame,
    dt=DEFAULT_BVH_FRAME_TIME):

    rootPosition = rootTrajectorySource["positions"][currentFrame].astype(np.float32)
    rootRotation = rootTrajectorySource["rotations"][currentFrame].astype(np.float32)
    rootVelocity = rootTrajectorySource["velocities"][currentFrame].astype(np.float32)

    if "root_angular_velocities" in poseSource:
        rootAngularVelocity = poseSource["root_angular_velocities"][currentFrame].astype(np.float32)
    else:
        rootAngularVelocity = _compute_pose_angular_velocities(
            np.asarray(rootTrajectorySource["rotations"], dtype=np.float32),
            dt,
        )[currentFrame]

    globalPositions = poseSource["global_positions"][currentFrame]
    globalRotations = poseSource["global_rotations"][currentFrame]
    globalVelocities = poseSource["global_velocities"][currentFrame]
    globalAngularVelocities = poseSource["global_angular_velocities"][currentFrame]

    localPositions = quat.inv_mul_vec(rootRotation, globalPositions - rootPosition).astype(np.float32)
    localRotations = quat.inv_mul(rootRotation, globalRotations).astype(np.float32)
    localVelocities = quat.inv_mul_vec(rootRotation, globalVelocities).astype(np.float32)
    localAngularVelocities = quat.inv_mul_vec(rootRotation, globalAngularVelocities).astype(np.float32)
    rootLocalVelocity = quat.inv_mul_vec(rootRotation, rootVelocity).astype(np.float32)
    rootLocalAngularVelocity = quat.inv_mul_vec(rootRotation, rootAngularVelocity).astype(np.float32)

    return {
        "current_root_position": rootPosition,
        "current_root_rotation": rootRotation,
        "local_positions": localPositions,
        "local_rotations": localRotations,
        "local_rotations_6d": _convert_local_rotations_to_6d(localRotations),
        "local_velocities": localVelocities,
        "local_angular_velocities": localAngularVelocities,
        "root_local_velocity": rootLocalVelocity,
        "root_local_angular_velocity": rootLocalAngularVelocity,
    }


def _convert_6d_rotations_to_local_rotations(localRotations6D):
    rotationsXY = np.asarray(localRotations6D, dtype=np.float32).reshape(
        localRotations6D.shape[:-1] + (3, 2)
    )
    return quat.from_xform_xy(rotationsXY).astype(np.float32)


def _integrate_root_motion(
    rootPosition,
    rootRotation,
    rootLocalVelocity,
    rootLocalAngularVelocity,
    dt=DEFAULT_BVH_FRAME_TIME):

    rootWorldVelocity = quat.mul_vec(rootRotation, rootLocalVelocity).astype(np.float32)
    rootWorldAngularVelocity = quat.mul_vec(rootRotation, rootLocalAngularVelocity).astype(np.float32)
    nextRootPosition = (rootPosition + rootWorldVelocity * dt).astype(np.float32)
    nextRootRotation = quat.normalize(
        quat.mul(
            quat.from_scaled_angle_axis(rootWorldAngularVelocity * dt),
            rootRotation,
        )
    ).astype(np.float32)

    return {
        "root_position": nextRootPosition,
        "root_rotation": nextRootRotation,
        "root_world_velocity": rootWorldVelocity,
        "root_world_angular_velocity": rootWorldAngularVelocity,
    }


def ReconstructPoseWorldSpace(
    localPose,
    rootPosition=None,
    rootRotation=None,
    integrateRootMotion=False,
    dt=DEFAULT_BVH_FRAME_TIME):

    rootPosition = (
        localPose["current_root_position"].astype(np.float32)
        if rootPosition is None else
        np.asarray(rootPosition, dtype=np.float32)
    )
    rootRotation = (
        localPose["current_root_rotation"].astype(np.float32)
        if rootRotation is None else
        np.asarray(rootRotation, dtype=np.float32)
    )

    if integrateRootMotion:
        rootMotion = _integrate_root_motion(
            rootPosition,
            rootRotation,
            localPose["root_local_velocity"],
            localPose["root_local_angular_velocity"],
            dt,
        )
        rootPosition = rootMotion["root_position"]
        rootRotation = rootMotion["root_rotation"]
    else:
        rootMotion = {
            "root_position": rootPosition,
            "root_rotation": rootRotation,
            "root_world_velocity": quat.mul_vec(rootRotation, localPose["root_local_velocity"]).astype(np.float32),
            "root_world_angular_velocity": quat.mul_vec(
                rootRotation,
                localPose["root_local_angular_velocity"],
            ).astype(np.float32),
        }

    localRotations = localPose.get("local_rotations")
    if localRotations is None:
        localRotations = _convert_6d_rotations_to_local_rotations(localPose["local_rotations_6d"])

    worldPositions = (
        quat.mul_vec(rootRotation, localPose["local_positions"]) + rootPosition
    ).astype(np.float32)
    worldRotations = quat.normalize(quat.mul(rootRotation, localRotations)).astype(np.float32)
    worldVelocities = quat.mul_vec(rootRotation, localPose["local_velocities"]).astype(np.float32)

    result = {
        "root_position": rootPosition.astype(np.float32),
        "root_rotation": rootRotation.astype(np.float32),
        "root_world_velocity": rootMotion["root_world_velocity"],
        "root_world_angular_velocity": rootMotion["root_world_angular_velocity"],
        "world_positions": worldPositions,
        "world_rotations": worldRotations,
        "world_velocities": worldVelocities,
    }

    if "local_angular_velocities" in localPose:
        result["world_angular_velocities"] = quat.mul_vec(
            rootRotation,
            localPose["local_angular_velocities"],
        ).astype(np.float32)

    return result


def ComputePosePositionError(originalPositions, reconstructedPositions):
    positionError = np.linalg.norm(reconstructedPositions - originalPositions, axis=-1)
    return float(positionError.mean()), float(positionError.max())
