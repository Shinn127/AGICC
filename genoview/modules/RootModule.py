import numpy as np
from scipy import signal as scipy_signal

from genoview.utils import quat


ROOT_JOINT_INDEX = 0
ROOT_FORWARD_AXIS = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
ROOT_UP_AXIS = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
ROOT_TRAJECTORY_MODE_FLAT = "flat"
ROOT_TRAJECTORY_MODE_HEIGHT_3D = "height_3d"
DEFAULT_TRAJECTORY_SAMPLE_OFFSETS = np.arange(-60, 61, 10, dtype=np.int32)
DEFAULT_BVH_FRAME_TIME = 1.0 / 60.0
TRAJECTORY_POSITION_SMOOTH_WINDOW = 31
TRAJECTORY_DIRECTION_SMOOTH_WINDOW = 61
TRAJECTORY_SMOOTH_POLYORDER = 3
MIN_TERRAIN_UP_DOT = 0.6
MIN_TERRAIN_UP_CONTINUITY_DOT = 0.3
TERRAIN_UP_SMOOTH_ALPHA = 0.15
TERRAIN_FORWARD_SMOOTH_ALPHA = 0.3

def ClampFrameIndex(frameIndex, frameCount):
    if frameCount <= 0:
        return 0
    return int(max(0, min(int(frameIndex), int(frameCount) - 1)))


def ComputeFiniteDifferenceVelocities(samples, dt):
    samples = np.asarray(samples, dtype=np.float32)
    velocities = np.zeros_like(samples, dtype=np.float32)
    if len(samples) <= 1:
        return velocities
    velocities[1:-1] = (samples[2:] - samples[:-2]) / (2.0 * dt)
    velocities[0] = (samples[1] - samples[0]) / dt
    velocities[-1] = (samples[-1] - samples[-2]) / dt
    return velocities.astype(np.float32)


def GetRootTrajectorySampleOffsets(step=10, radius=60):
    return np.arange(-radius, radius + 1, step, dtype=np.int32)


def _normalize_direction(direction, fallback=ROOT_FORWARD_AXIS):
    directionNorm = np.linalg.norm(direction)
    if directionNorm < 1e-8:
        return fallback.copy()
    return (direction / directionNorm).astype(np.float32)


def _normalize_directions(directions, fallback=ROOT_FORWARD_AXIS):
    return np.asarray([_normalize_direction(direction, fallback=fallback) for direction in directions], dtype=np.float32)


def _get_savgol_window_length(sampleCount, preferredWindow, polyorder):
    if sampleCount <= polyorder:
        return None

    windowLength = min(preferredWindow, sampleCount)
    if windowLength % 2 == 0:
        windowLength -= 1
    minWindow = polyorder + 1 if (polyorder + 1) % 2 == 1 else polyorder + 2
    if windowLength < minWindow:
        windowLength = minWindow
    if windowLength > sampleCount:
        windowLength = sampleCount if sampleCount % 2 == 1 else sampleCount - 1
    if windowLength <= polyorder or windowLength < 3:
        return None

    return int(windowLength)


def _apply_savgol_filter(data, preferredWindow, polyorder):
    data = np.asarray(data, dtype=np.float32)
    windowLength = _get_savgol_window_length(len(data), preferredWindow, polyorder)
    if windowLength is None:
        return data.copy()
    return scipy_signal.savgol_filter(
        data,
        windowLength,
        polyorder,
        axis=0,
        mode="interp",
    ).astype(np.float32)


def _project_trajectory_to_ground(vectors, groundHeight=0.0):
    projected = np.asarray(vectors, dtype=np.float32).copy()
    projected[:, 1] = groundHeight
    return projected


def _resolve_root_trajectory_mode(mode=None, projectToGround=True):
    if mode is None:
        return ROOT_TRAJECTORY_MODE_FLAT if projectToGround else ROOT_TRAJECTORY_MODE_HEIGHT_3D
    if mode not in (ROOT_TRAJECTORY_MODE_FLAT, ROOT_TRAJECTORY_MODE_HEIGHT_3D):
        raise ValueError(
            f'Unsupported root trajectory mode "{mode}". '
            f"Expected one of: {ROOT_TRAJECTORY_MODE_FLAT}, {ROOT_TRAJECTORY_MODE_HEIGHT_3D}."
        )
    return mode


def _project_vector_to_plane(vector, planeNormal, fallback=ROOT_FORWARD_AXIS):
    vector = np.asarray(vector, dtype=np.float32)
    planeNormal = _normalize_direction(planeNormal, fallback=ROOT_UP_AXIS)
    projected = vector - np.dot(vector, planeNormal) * planeNormal
    projectedNorm = np.linalg.norm(projected)
    if projectedNorm < 1e-8:
        fallback = np.asarray(fallback, dtype=np.float32)
        projected = fallback - np.dot(fallback, planeNormal) * planeNormal
        projectedNorm = np.linalg.norm(projected)
        if projectedNorm < 1e-8:
            alternate = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            if abs(np.dot(alternate, planeNormal)) > 0.9:
                alternate = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
            projected = alternate - np.dot(alternate, planeNormal) * planeNormal
            projectedNorm = np.linalg.norm(projected)
    return (projected / max(projectedNorm, 1e-8)).astype(np.float32)


def _remove_vector_component_along_normal(vector, planeNormal):
    vector = np.asarray(vector, dtype=np.float32)
    planeNormal = _normalize_direction(planeNormal, fallback=ROOT_UP_AXIS)
    return (vector - np.dot(vector, planeNormal) * planeNormal).astype(np.float32)


def _build_rotation_from_up_forward(upDirection, forwardDirection):
    upDirection = _normalize_direction(upDirection, fallback=ROOT_UP_AXIS)
    forwardDirection = _project_vector_to_plane(forwardDirection, upDirection, fallback=ROOT_FORWARD_AXIS)
    rightDirection = np.cross(upDirection, forwardDirection).astype(np.float32)
    rightDirection = _normalize_direction(rightDirection, fallback=np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    forwardDirection = _normalize_direction(
        np.cross(rightDirection, upDirection).astype(np.float32),
        fallback=forwardDirection,
    )
    xform = np.stack([rightDirection, upDirection, forwardDirection], axis=-1).astype(np.float32)
    return quat.normalize(quat.from_xform(xform)).astype(np.float32)


def _build_heading_rotations_from_directions(directions):
    horizontalDirections = _normalize_directions(
        _project_trajectory_to_ground(directions, groundHeight=0.0),
        fallback=ROOT_FORWARD_AXIS,
    )
    return np.asarray(
        [_build_rotation_from_up_forward(ROOT_UP_AXIS, direction) for direction in horizontalDirections],
        dtype=np.float32,
    )


def _build_motion_root_trajectory_source_from_positions_and_directions(
    rootPositions,
    rootDirections,
    dt,
    mode=ROOT_TRAJECTORY_MODE_FLAT,
    groundHeight=0.0,
):
    mode = _resolve_root_trajectory_mode(mode, projectToGround=(mode == ROOT_TRAJECTORY_MODE_FLAT))
    rootPositions = np.asarray(rootPositions, dtype=np.float32)
    rootDirections = np.asarray(rootDirections, dtype=np.float32)

    motionPositions = rootPositions.copy()
    if mode == ROOT_TRAJECTORY_MODE_FLAT:
        motionPositions = _project_trajectory_to_ground(motionPositions, groundHeight=groundHeight)

    horizontalDirections = _normalize_directions(
        _project_trajectory_to_ground(rootDirections, groundHeight=0.0),
        fallback=ROOT_FORWARD_AXIS,
    )
    smoothedPositions = _apply_savgol_filter(
        motionPositions,
        TRAJECTORY_POSITION_SMOOTH_WINDOW,
        TRAJECTORY_SMOOTH_POLYORDER,
    )
    smoothedDirections = _normalize_directions(
        _apply_savgol_filter(
            horizontalDirections,
            TRAJECTORY_DIRECTION_SMOOTH_WINDOW,
            TRAJECTORY_SMOOTH_POLYORDER,
        ),
        fallback=ROOT_FORWARD_AXIS,
    )
    smoothedRotations = _build_heading_rotations_from_directions(smoothedDirections)
    smoothedVelocities = ComputeFiniteDifferenceVelocities(smoothedPositions, dt)

    return {
        "positions": smoothedPositions.astype(np.float32),
        "directions": smoothedDirections.astype(np.float32),
        "rotations": smoothedRotations.astype(np.float32),
        "velocities": smoothedVelocities.astype(np.float32),
        "dt": float(dt),
        "mode": mode,
    }


def _stabilize_terrain_frame_series(candidateUps, candidateForwards):
    candidateUps = np.asarray(candidateUps, dtype=np.float32)
    candidateForwards = np.asarray(candidateForwards, dtype=np.float32)

    if len(candidateUps) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
        )

    stableUps = np.zeros_like(candidateUps, dtype=np.float32)
    stableForwards = np.zeros_like(candidateForwards, dtype=np.float32)
    stableRotations = np.zeros((len(candidateUps), 4), dtype=np.float32)

    previousUp = ROOT_UP_AXIS.copy()
    previousForward = ROOT_FORWARD_AXIS.copy()

    for i in range(len(candidateUps)):
        upCandidate = _normalize_direction(candidateUps[i], fallback=previousUp)
        if (
            upCandidate[1] < MIN_TERRAIN_UP_DOT or
            np.dot(upCandidate, previousUp) < MIN_TERRAIN_UP_CONTINUITY_DOT
        ):
            upCandidate = previousUp.copy()

        stableUp = _normalize_direction(
            (1.0 - TERRAIN_UP_SMOOTH_ALPHA) * previousUp +
            TERRAIN_UP_SMOOTH_ALPHA * upCandidate,
            fallback=previousUp,
        )

        forwardCandidate = _project_vector_to_plane(
            candidateForwards[i],
            stableUp,
            fallback=previousForward,
        )
        if np.dot(forwardCandidate, previousForward) < 0.0:
            forwardCandidate = -forwardCandidate

        stableForward = _normalize_direction(
            (1.0 - TERRAIN_FORWARD_SMOOTH_ALPHA) * previousForward +
            TERRAIN_FORWARD_SMOOTH_ALPHA * forwardCandidate,
            fallback=previousForward,
        )

        stableUps[i] = stableUp
        stableForwards[i] = stableForward
        stableRotations[i] = _build_rotation_from_up_forward(stableUp, stableForward)

        previousUp = stableUp
        previousForward = stableForward

    return (
        stableUps.astype(np.float32),
        stableForwards.astype(np.float32),
        stableRotations.astype(np.float32),
    )


def _compute_root_world_forward_series(globalRotations, rootIndex=ROOT_JOINT_INDEX):
    return quat.mul_vec(globalRotations[:, rootIndex], ROOT_FORWARD_AXIS).astype(np.float32)


def _build_smoothed_root_trajectory_source(
    globalPositions,
    globalRotations,
    dt,
    rootIndex=ROOT_JOINT_INDEX,
    mode=None,
    projectToGround=True,
    groundHeight=0.0):

    rootPositions = np.asarray(globalPositions[:, rootIndex], dtype=np.float32)
    rootDirections = _compute_root_world_forward_series(globalRotations, rootIndex=rootIndex)
    mode = _resolve_root_trajectory_mode(mode, projectToGround=projectToGround)
    return _build_motion_root_trajectory_source_from_positions_and_directions(
        rootPositions,
        rootDirections,
        dt,
        mode=mode,
        groundHeight=groundHeight,
    )


def AdaptRootTrajectoryToTerrain(
    rootTrajectory,
    terrainProvider,
    alignPositionsToTerrain=False,
):
    rootPositions = np.asarray(rootTrajectory["positions"], dtype=np.float32)
    rootDirections = np.asarray(rootTrajectory["directions"], dtype=np.float32)
    rootRotations = np.asarray(rootTrajectory["rotations"], dtype=np.float32)
    rootVelocities = np.asarray(rootTrajectory["velocities"], dtype=np.float32)
    dt = float(rootTrajectory.get("dt", DEFAULT_BVH_FRAME_TIME))

    terrainHeights = terrainProvider.sample_heights(rootPositions).astype(np.float32)
    terrainNormals = _normalize_directions(
        terrainProvider.sample_normals(rootPositions).astype(np.float32),
        fallback=ROOT_UP_AXIS,
    )
    smoothedNormalCandidates = _normalize_directions(
        _apply_savgol_filter(
            terrainNormals,
            TRAJECTORY_DIRECTION_SMOOTH_WINDOW,
            TRAJECTORY_SMOOTH_POLYORDER,
        ),
        fallback=ROOT_UP_AXIS,
    )
    smoothedNormals, _, terrainAlignedRotations = _stabilize_terrain_frame_series(
        smoothedNormalCandidates,
        rootDirections,
    )

    adaptedPositions = rootPositions.copy()
    adaptedVelocities = rootVelocities.copy()
    if alignPositionsToTerrain:
        adaptedPositions[:, 1] = terrainHeights
        adaptedVelocities = ComputeFiniteDifferenceVelocities(adaptedPositions, dt)

    tiltRotations = quat.normalize(
        quat.mul_inv(terrainAlignedRotations, rootRotations)
    ).astype(np.float32)

    return {
        "positions": adaptedPositions.astype(np.float32),
        "directions": rootDirections.astype(np.float32),
        "rotations": rootRotations.astype(np.float32),
        "velocities": adaptedVelocities.astype(np.float32),
        "dt": dt,
        "mode": rootTrajectory.get("mode"),
        "terrain_heights": terrainHeights.astype(np.float32),
        "terrain_normals": smoothedNormals.astype(np.float32),
        "tilt_rotations": tiltRotations.astype(np.float32),
        "terrain_aligned_rotations": terrainAlignedRotations.astype(np.float32),
    }


def BuildRootTrajectorySource(
    globalPositions,
    globalRotations,
    dt,
    rootIndex=ROOT_JOINT_INDEX,
    mode=None,
    projectToGround=True,
    groundHeight=0.0):
    mode = _resolve_root_trajectory_mode(mode, projectToGround=projectToGround)
    return _build_smoothed_root_trajectory_source(
        globalPositions,
        globalRotations,
        dt,
        rootIndex=rootIndex,
        mode=mode,
        projectToGround=projectToGround,
        groundHeight=groundHeight,
    )


def BuildRootLocalTrajectory(
    trajectorySource,
    currentFrame,
    sampleOffsets=DEFAULT_TRAJECTORY_SAMPLE_OFFSETS):

    trajectoryPositions = trajectorySource["positions"]
    trajectoryDirections = trajectorySource["directions"]
    trajectoryRotations = trajectorySource["rotations"]
    trajectoryVelocities = trajectorySource["velocities"]
    frameCount = len(trajectoryPositions)
    sampleOffsets = np.asarray(sampleOffsets, dtype=np.int32)
    sampleFrames = np.asarray(
        [ClampFrameIndex(currentFrame + offset, frameCount) for offset in sampleOffsets],
        dtype=np.int32,
    )

    currentRootPosition = trajectoryPositions[currentFrame].astype(np.float32)
    currentRootRotation = trajectoryRotations[currentFrame].astype(np.float32)

    localPositions = np.zeros((len(sampleFrames), 3), dtype=np.float32)
    localDirections = np.zeros((len(sampleFrames), 3), dtype=np.float32)
    localVelocities = np.zeros((len(sampleFrames), 3), dtype=np.float32)

    for i, sampleFrame in enumerate(sampleFrames):
        sampleRootPosition = trajectoryPositions[sampleFrame]
        sampleWorldForward = trajectoryDirections[sampleFrame]
        sampleWorldVelocity = trajectoryVelocities[sampleFrame]

        localPositions[i] = quat.inv_mul_vec(
            currentRootRotation,
            sampleRootPosition - currentRootPosition,
        ).astype(np.float32)
        localDirections[i] = _normalize_direction(quat.inv_mul_vec(currentRootRotation, sampleWorldForward))
        localVelocities[i] = quat.inv_mul_vec(currentRootRotation, sampleWorldVelocity).astype(np.float32)

    return {
        "sample_offsets": sampleOffsets,
        "sample_frames": sampleFrames,
        "current_root_position": currentRootPosition,
        "current_root_rotation": currentRootRotation,
        "local_positions": localPositions,
        "local_directions": localDirections,
        "local_velocities": localVelocities,
    }


def BuildTerrainAdaptedRootTrajectoryDisplay(
    rootTrajectory,
    terrainTrajectorySource,
    heightOffset=0.01,
    alignDirectionsToTerrain=True,
    alignVelocitiesToTerrain=True,
):
    sampleFrames = np.asarray(rootTrajectory["sample_frames"], dtype=np.int32)
    worldPositions = np.asarray(
        terrainTrajectorySource["positions"][sampleFrames],
        dtype=np.float32,
    ).copy()
    worldVelocities = np.asarray(
        terrainTrajectorySource["velocities"][sampleFrames],
        dtype=np.float32,
    ).copy()

    if "terrain_aligned_rotations" in terrainTrajectorySource:
        terrainAlignedRotations = np.asarray(
            terrainTrajectorySource["terrain_aligned_rotations"][sampleFrames],
            dtype=np.float32,
        )
        worldDirections = quat.mul_vec(
            terrainAlignedRotations,
            np.repeat(ROOT_FORWARD_AXIS[np.newaxis, :], len(sampleFrames), axis=0),
        ).astype(np.float32)
    else:
        worldDirections = np.asarray(
            terrainTrajectorySource["directions"][sampleFrames],
            dtype=np.float32,
        ).copy()

    if "terrain_heights" in terrainTrajectorySource:
        worldPositions[:, 1] = terrainTrajectorySource["terrain_heights"][sampleFrames] + heightOffset
    else:
        worldPositions[:, 1] += heightOffset

    if "terrain_normals" in terrainTrajectorySource:
        terrainNormals = np.asarray(
            terrainTrajectorySource["terrain_normals"][sampleFrames],
            dtype=np.float32,
        )
        if alignDirectionsToTerrain:
            worldDirections = np.asarray(
                [
                    _project_vector_to_plane(direction, normal, fallback=ROOT_FORWARD_AXIS)
                    for direction, normal in zip(worldDirections, terrainNormals)
                ],
                dtype=np.float32,
            )
        if alignVelocitiesToTerrain:
            worldVelocities = np.asarray(
                [
                    _remove_vector_component_along_normal(velocity, normal)
                    for velocity, normal in zip(worldVelocities, terrainNormals)
                ],
                dtype=np.float32,
            )

    return {
        "world_positions": worldPositions.astype(np.float32),
        "world_directions": _normalize_directions(worldDirections, fallback=ROOT_FORWARD_AXIS),
        "world_velocities": worldVelocities.astype(np.float32),
    }
