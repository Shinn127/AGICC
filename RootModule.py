import numpy as np

import quat

try:
    from scipy import signal as scipy_signal
except ImportError:
    scipy_signal = None


ROOT_JOINT_INDEX = 0
ROOT_FORWARD_AXIS = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
DEFAULT_TRAJECTORY_SAMPLE_OFFSETS = np.arange(-60, 61, 10, dtype=np.int32)
DEFAULT_BVH_FRAME_TIME = 1.0 / 60.0
TRAJECTORY_POSITION_SMOOTH_WINDOW = 31
TRAJECTORY_DIRECTION_SMOOTH_WINDOW = 61
TRAJECTORY_SMOOTH_POLYORDER = 3


def GetRootTrajectorySampleOffsets(step=10, radius=60):
    return np.arange(-radius, radius + 1, step, dtype=np.int32)


def ClampFrameIndex(frameIndex, frameCount):
    return int(np.clip(frameIndex, 0, frameCount - 1))


def NormalizeDirection(direction, fallback=ROOT_FORWARD_AXIS):
    directionNorm = np.linalg.norm(direction)
    if directionNorm < 1e-8:
        return fallback.copy()
    return (direction / directionNorm).astype(np.float32)


def NormalizeDirections(directions, fallback=ROOT_FORWARD_AXIS):
    return np.asarray([NormalizeDirection(direction, fallback=fallback) for direction in directions], dtype=np.float32)


def GetSavgolWindowLength(sampleCount, preferredWindow, polyorder):
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


def ApplySavgolFilterFallback(data, windowLength, polyorder):
    sampleCount = len(data)
    halfWindow = windowLength // 2
    output = np.empty_like(data, dtype=np.float32)

    centerX = np.arange(-halfWindow, halfWindow + 1, dtype=np.float32)
    centerDesign = np.vander(centerX, polyorder + 1, increasing=True)
    centerWeights = np.linalg.pinv(centerDesign)[0].astype(np.float32)
    leftX = np.arange(windowLength, dtype=np.float32)
    leftDesign = np.vander(leftX, polyorder + 1, increasing=True)
    leftPseudoInverse = np.linalg.pinv(leftDesign).astype(np.float32)

    for i in range(halfWindow, sampleCount - halfWindow):
        output[i] = centerWeights @ data[i - halfWindow:i + halfWindow + 1]

    leftWindow = data[:windowLength]
    leftCoefficients = leftPseudoInverse @ leftWindow
    for i in range(halfWindow):
        output[i] = (
            np.vander(np.asarray([i], dtype=np.float32), polyorder + 1, increasing=True) @ leftCoefficients
        )[0]

    rightWindow = data[-windowLength:]
    rightCoefficients = leftPseudoInverse @ rightWindow
    rightStart = sampleCount - windowLength
    for i in range(sampleCount - halfWindow, sampleCount):
        localIndex = i - rightStart
        output[i] = (
            np.vander(np.asarray([localIndex], dtype=np.float32), polyorder + 1, increasing=True) @ rightCoefficients
        )[0]

    return output.astype(np.float32)


def ApplySavgolFilter(data, preferredWindow, polyorder):
    data = np.asarray(data, dtype=np.float32)
    windowLength = GetSavgolWindowLength(len(data), preferredWindow, polyorder)
    if windowLength is None:
        return data.copy()
    if scipy_signal is not None:
        return scipy_signal.savgol_filter(
            data,
            windowLength,
            polyorder,
            axis=0,
            mode="interp",
        ).astype(np.float32)
    return ApplySavgolFilterFallback(data, windowLength, polyorder)


def ProjectTrajectoryToGround(vectors, groundHeight=0.0):
    projected = np.asarray(vectors, dtype=np.float32).copy()
    projected[:, 1] = groundHeight
    return projected


def ComputeRootWorldForwardSeries(globalRotations, rootIndex=ROOT_JOINT_INDEX):
    return quat.mul_vec(globalRotations[:, rootIndex], ROOT_FORWARD_AXIS).astype(np.float32)


def ComputeTrajectoryVelocities(positions, dt):
    velocities = np.zeros_like(positions, dtype=np.float32)
    if len(positions) == 0:
        return velocities
    if len(positions) == 1:
        return velocities
    if len(positions) == 2:
        singleStepVelocity = ((positions[1] - positions[0]) / dt).astype(np.float32)
        velocities[:] = singleStepVelocity
        return velocities

    velocities[1:-1] = (
        0.5 * (positions[2:] - positions[1:-1]) / dt +
        0.5 * (positions[1:-1] - positions[:-2]) / dt
    ).astype(np.float32)

    if len(positions) >= 4:
        velocities[0] = velocities[1] - (velocities[3] - velocities[2])
        velocities[-1] = velocities[-2] + (velocities[-2] - velocities[-3])
    else:
        velocities[0] = velocities[1]
        velocities[-1] = velocities[-2]

    return velocities.astype(np.float32)


def BuildSmoothedRootTrajectorySource(
    globalPositions,
    globalRotations,
    dt,
    rootIndex=ROOT_JOINT_INDEX,
    projectToGround=True,
    groundHeight=0.0):

    rootPositions = np.asarray(globalPositions[:, rootIndex], dtype=np.float32)
    rootDirections = ComputeRootWorldForwardSeries(globalRotations, rootIndex=rootIndex)

    if projectToGround:
        rootPositions = ProjectTrajectoryToGround(rootPositions, groundHeight=groundHeight)
        rootDirections = ProjectTrajectoryToGround(rootDirections, groundHeight=0.0)

    rootDirections = NormalizeDirections(rootDirections)
    smoothedPositions = ApplySavgolFilter(
        rootPositions,
        TRAJECTORY_POSITION_SMOOTH_WINDOW,
        TRAJECTORY_SMOOTH_POLYORDER,
    )
    smoothedDirections = ApplySavgolFilter(
        rootDirections,
        TRAJECTORY_DIRECTION_SMOOTH_WINDOW,
        TRAJECTORY_SMOOTH_POLYORDER,
    )
    smoothedDirections = NormalizeDirections(smoothedDirections)
    smoothedRotations = quat.normalize(
        quat.between(
            np.repeat(ROOT_FORWARD_AXIS[np.newaxis, :], len(smoothedDirections), axis=0),
            smoothedDirections,
        )
    ).astype(np.float32)
    smoothedVelocities = ComputeTrajectoryVelocities(smoothedPositions, dt)

    return {
        "positions": smoothedPositions.astype(np.float32),
        "directions": smoothedDirections.astype(np.float32),
        "rotations": smoothedRotations.astype(np.float32),
        "velocities": smoothedVelocities.astype(np.float32),
    }


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
        localDirections[i] = NormalizeDirection(quat.inv_mul_vec(currentRootRotation, sampleWorldForward))
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


def ReconstructRootTrajectoryWorldSpace(
    localPositions,
    localDirections,
    localVelocities,
    currentRootPosition,
    currentRootRotation):

    trajectoryCount = len(localPositions)
    repeatedRotation = np.repeat(currentRootRotation[np.newaxis, :], trajectoryCount, axis=0)
    worldPositions = quat.mul_vec(repeatedRotation, localPositions) + currentRootPosition[np.newaxis, :]
    worldDirections = np.asarray(
        [NormalizeDirection(direction) for direction in quat.mul_vec(repeatedRotation, localDirections)],
        dtype=np.float32,
    )
    worldVelocities = quat.mul_vec(repeatedRotation, localVelocities).astype(np.float32)

    return {
        "world_positions": worldPositions.astype(np.float32),
        "world_directions": worldDirections,
        "world_velocities": worldVelocities,
    }


def ApplyTrajectoryGroundProjection(
    worldPositions,
    worldDirections,
    worldVelocities,
    groundHeight=0.0,
    projectToGround=True,
    heightOffset=0.01):

    projectedPositions = worldPositions.copy()
    projectedDirections = worldDirections.copy()
    projectedVelocities = worldVelocities.copy()

    if projectToGround:
        projectedPositions[:, 1] = groundHeight + heightOffset
        projectedDirections[:, 1] = 0.0
        projectedVelocities[:, 1] = 0.0
        projectedDirections = np.asarray(
            [NormalizeDirection(direction) for direction in projectedDirections],
            dtype=np.float32,
        )

    return {
        "world_positions": projectedPositions.astype(np.float32),
        "world_directions": projectedDirections.astype(np.float32),
        "world_velocities": projectedVelocities.astype(np.float32),
    }


def BuildRootTrajectoryDisplay(
    rootTrajectory,
    groundHeight=0.0,
    projectToGround=True,
    heightOffset=0.01):

    rootTrajectoryWorld = ReconstructRootTrajectoryWorldSpace(
        rootTrajectory["local_positions"],
        rootTrajectory["local_directions"],
        rootTrajectory["local_velocities"],
        rootTrajectory["current_root_position"],
        rootTrajectory["current_root_rotation"],
    )

    return ApplyTrajectoryGroundProjection(
        rootTrajectoryWorld["world_positions"],
        rootTrajectoryWorld["world_directions"],
        rootTrajectoryWorld["world_velocities"],
        groundHeight=groundHeight,
        projectToGround=projectToGround,
        heightOffset=heightOffset,
    )
