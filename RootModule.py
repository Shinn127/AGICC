import numpy as np

import quat
from Utils import ClampFrameIndex, ComputeFiniteDifferenceVelocities

try:
    from scipy import signal as scipy_signal
except ImportError:
    scipy_signal = None


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


def GetRootTrajectorySampleOffsets(step=10, radius=60):
    return np.arange(-radius, radius + 1, step, dtype=np.int32)


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


def ResolveRootTrajectoryMode(mode=None, projectToGround=True):
    if mode is None:
        return ROOT_TRAJECTORY_MODE_FLAT if projectToGround else ROOT_TRAJECTORY_MODE_HEIGHT_3D
    if mode not in (ROOT_TRAJECTORY_MODE_FLAT, ROOT_TRAJECTORY_MODE_HEIGHT_3D):
        raise ValueError(
            f'Unsupported root trajectory mode "{mode}". '
            f"Expected one of: {ROOT_TRAJECTORY_MODE_FLAT}, {ROOT_TRAJECTORY_MODE_HEIGHT_3D}."
        )
    return mode


def ProjectVectorToPlane(vector, planeNormal, fallback=ROOT_FORWARD_AXIS):
    vector = np.asarray(vector, dtype=np.float32)
    planeNormal = NormalizeDirection(planeNormal, fallback=ROOT_UP_AXIS)
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


def RemoveVectorComponentAlongNormal(vector, planeNormal):
    vector = np.asarray(vector, dtype=np.float32)
    planeNormal = NormalizeDirection(planeNormal, fallback=ROOT_UP_AXIS)
    return (vector - np.dot(vector, planeNormal) * planeNormal).astype(np.float32)


def BuildRotationFromUpForward(upDirection, forwardDirection):
    upDirection = NormalizeDirection(upDirection, fallback=ROOT_UP_AXIS)
    forwardDirection = ProjectVectorToPlane(forwardDirection, upDirection, fallback=ROOT_FORWARD_AXIS)
    rightDirection = np.cross(upDirection, forwardDirection).astype(np.float32)
    rightDirection = NormalizeDirection(rightDirection, fallback=np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    forwardDirection = NormalizeDirection(
        np.cross(rightDirection, upDirection).astype(np.float32),
        fallback=forwardDirection,
    )
    xform = np.stack([rightDirection, upDirection, forwardDirection], axis=-1).astype(np.float32)
    return quat.normalize(quat.from_xform(xform)).astype(np.float32)


def BuildHeadingRotationsFromDirections(directions):
    horizontalDirections = NormalizeDirections(
        ProjectTrajectoryToGround(directions, groundHeight=0.0),
        fallback=ROOT_FORWARD_AXIS,
    )
    return np.asarray(
        [BuildRotationFromUpForward(ROOT_UP_AXIS, direction) for direction in horizontalDirections],
        dtype=np.float32,
    )


def BuildMotionRootTrajectorySourceFromPositionsAndDirections(
    rootPositions,
    rootDirections,
    dt,
    mode=ROOT_TRAJECTORY_MODE_FLAT,
    groundHeight=0.0,
):
    mode = ResolveRootTrajectoryMode(mode, projectToGround=(mode == ROOT_TRAJECTORY_MODE_FLAT))
    rootPositions = np.asarray(rootPositions, dtype=np.float32)
    rootDirections = np.asarray(rootDirections, dtype=np.float32)

    motionPositions = rootPositions.copy()
    if mode == ROOT_TRAJECTORY_MODE_FLAT:
        motionPositions = ProjectTrajectoryToGround(motionPositions, groundHeight=groundHeight)

    horizontalDirections = NormalizeDirections(
        ProjectTrajectoryToGround(rootDirections, groundHeight=0.0),
        fallback=ROOT_FORWARD_AXIS,
    )
    smoothedPositions = ApplySavgolFilter(
        motionPositions,
        TRAJECTORY_POSITION_SMOOTH_WINDOW,
        TRAJECTORY_SMOOTH_POLYORDER,
    )
    smoothedDirections = NormalizeDirections(
        ApplySavgolFilter(
            horizontalDirections,
            TRAJECTORY_DIRECTION_SMOOTH_WINDOW,
            TRAJECTORY_SMOOTH_POLYORDER,
        ),
        fallback=ROOT_FORWARD_AXIS,
    )
    smoothedRotations = BuildHeadingRotationsFromDirections(smoothedDirections)
    smoothedVelocities = ComputeFiniteDifferenceVelocities(smoothedPositions, dt)

    return {
        "positions": smoothedPositions.astype(np.float32),
        "directions": smoothedDirections.astype(np.float32),
        "rotations": smoothedRotations.astype(np.float32),
        "velocities": smoothedVelocities.astype(np.float32),
        "dt": float(dt),
        "mode": mode,
    }


def StabilizeTerrainFrameSeries(candidateUps, candidateForwards):
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
        upCandidate = NormalizeDirection(candidateUps[i], fallback=previousUp)
        if (
            upCandidate[1] < MIN_TERRAIN_UP_DOT or
            np.dot(upCandidate, previousUp) < MIN_TERRAIN_UP_CONTINUITY_DOT
        ):
            upCandidate = previousUp.copy()

        stableUp = NormalizeDirection(
            (1.0 - TERRAIN_UP_SMOOTH_ALPHA) * previousUp +
            TERRAIN_UP_SMOOTH_ALPHA * upCandidate,
            fallback=previousUp,
        )

        forwardCandidate = ProjectVectorToPlane(
            candidateForwards[i],
            stableUp,
            fallback=previousForward,
        )
        if np.dot(forwardCandidate, previousForward) < 0.0:
            forwardCandidate = -forwardCandidate

        stableForward = NormalizeDirection(
            (1.0 - TERRAIN_FORWARD_SMOOTH_ALPHA) * previousForward +
            TERRAIN_FORWARD_SMOOTH_ALPHA * forwardCandidate,
            fallback=previousForward,
        )

        stableUps[i] = stableUp
        stableForwards[i] = stableForward
        stableRotations[i] = BuildRotationFromUpForward(stableUp, stableForward)

        previousUp = stableUp
        previousForward = stableForward

    return (
        stableUps.astype(np.float32),
        stableForwards.astype(np.float32),
        stableRotations.astype(np.float32),
    )


def ComputeRootWorldForwardSeries(globalRotations, rootIndex=ROOT_JOINT_INDEX):
    return quat.mul_vec(globalRotations[:, rootIndex], ROOT_FORWARD_AXIS).astype(np.float32)


def BuildSmoothedRootTrajectorySource(
    globalPositions,
    globalRotations,
    dt,
    rootIndex=ROOT_JOINT_INDEX,
    mode=None,
    projectToGround=True,
    groundHeight=0.0):

    rootPositions = np.asarray(globalPositions[:, rootIndex], dtype=np.float32)
    rootDirections = ComputeRootWorldForwardSeries(globalRotations, rootIndex=rootIndex)
    mode = ResolveRootTrajectoryMode(mode, projectToGround=projectToGround)
    return BuildMotionRootTrajectorySourceFromPositionsAndDirections(
        rootPositions,
        rootDirections,
        dt,
        mode=mode,
        groundHeight=groundHeight,
    )


def BuildFlatRootTrajectorySource(
    globalPositions,
    globalRotations,
    dt,
    rootIndex=ROOT_JOINT_INDEX,
    groundHeight=0.0,
):
    return BuildSmoothedRootTrajectorySource(
        globalPositions,
        globalRotations,
        dt,
        rootIndex=rootIndex,
        mode=ROOT_TRAJECTORY_MODE_FLAT,
        projectToGround=True,
        groundHeight=groundHeight,
    )


def BuildHeightRootTrajectorySource(
    globalPositions,
    globalRotations,
    dt,
    rootIndex=ROOT_JOINT_INDEX,
):
    return BuildSmoothedRootTrajectorySource(
        globalPositions,
        globalRotations,
        dt,
        rootIndex=rootIndex,
        mode=ROOT_TRAJECTORY_MODE_HEIGHT_3D,
        projectToGround=False,
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
    terrainNormals = NormalizeDirections(
        terrainProvider.sample_normals(rootPositions).astype(np.float32),
        fallback=ROOT_UP_AXIS,
    )
    smoothedNormalCandidates = NormalizeDirections(
        ApplySavgolFilter(
            terrainNormals,
            TRAJECTORY_DIRECTION_SMOOTH_WINDOW,
            TRAJECTORY_SMOOTH_POLYORDER,
        ),
        fallback=ROOT_UP_AXIS,
    )
    smoothedNormals, _, displayRotations = StabilizeTerrainFrameSeries(
        smoothedNormalCandidates,
        rootDirections,
    )

    adaptedPositions = rootPositions.copy()
    adaptedVelocities = rootVelocities.copy()
    if alignPositionsToTerrain:
        adaptedPositions[:, 1] = terrainHeights
        adaptedVelocities = ComputeFiniteDifferenceVelocities(adaptedPositions, dt)

    tiltRotations = quat.normalize(
        quat.mul_inv(displayRotations, rootRotations)
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
        "display_rotations": displayRotations.astype(np.float32),
    }


def BuildRootTrajectorySource(
    globalPositions,
    globalRotations,
    dt,
    rootIndex=ROOT_JOINT_INDEX,
    mode=None,
    projectToGround=True,
    groundHeight=0.0,
    terrainProvider=None,
    alignPositionsToTerrain=False):
    mode = ResolveRootTrajectoryMode(mode, projectToGround=projectToGround)

    if terrainProvider is not None:
        return BuildTerrainAwareRootTrajectorySource(
            globalPositions,
            globalRotations,
            terrainProvider,
            dt,
            rootIndex=rootIndex,
            mode=mode,
            groundHeight=groundHeight,
            alignPositionsToTerrain=alignPositionsToTerrain,
        )

    return BuildSmoothedRootTrajectorySource(
        globalPositions,
        globalRotations,
        dt,
        rootIndex=rootIndex,
        mode=mode,
        projectToGround=projectToGround,
        groundHeight=groundHeight,
    )


def BuildTerrainAwareRootTrajectorySource(
    globalPositions,
    globalRotations,
    terrainProvider,
    dt,
    rootIndex=ROOT_JOINT_INDEX,
    mode=None,
    groundHeight=0.0,
    alignPositionsToTerrain=False):
    mode = ResolveRootTrajectoryMode(mode, projectToGround=(mode == ROOT_TRAJECTORY_MODE_FLAT))
    motionTrajectory = BuildSmoothedRootTrajectorySource(
        globalPositions,
        globalRotations,
        dt,
        rootIndex=rootIndex,
        mode=mode,
        projectToGround=(mode == ROOT_TRAJECTORY_MODE_FLAT),
        groundHeight=groundHeight,
    )
    return AdaptRootTrajectoryToTerrain(
        motionTrajectory,
        terrainProvider,
        alignPositionsToTerrain=alignPositionsToTerrain,
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


def ApplyTrajectoryTerrainProjection(
    worldPositions,
    worldDirections,
    worldVelocities,
    terrainProvider,
    projectToTerrain=True,
    heightOffset=0.01):

    projectedPositions = np.asarray(worldPositions, dtype=np.float32).copy()
    projectedDirections = np.asarray(worldDirections, dtype=np.float32).copy()
    projectedVelocities = np.asarray(worldVelocities, dtype=np.float32).copy()

    if projectToTerrain and terrainProvider is not None and len(projectedPositions) > 0:
        projectedPositions[:, 1] = terrainProvider.sample_heights(projectedPositions) + heightOffset

    return {
        "world_positions": projectedPositions.astype(np.float32),
        "world_directions": projectedDirections.astype(np.float32),
        "world_velocities": projectedVelocities.astype(np.float32),
    }


def BuildRootTrajectoryDisplay(
    rootTrajectory,
    groundHeight=0.0,
    projectToGround=True,
    heightOffset=0.01,
    terrainProvider=None,
    projectToTerrain=False):

    rootTrajectoryWorld = ReconstructRootTrajectoryWorldSpace(
        rootTrajectory["local_positions"],
        rootTrajectory["local_directions"],
        rootTrajectory["local_velocities"],
        rootTrajectory["current_root_position"],
        rootTrajectory["current_root_rotation"],
    )

    if terrainProvider is not None and projectToTerrain:
        return ApplyTrajectoryTerrainProjection(
            rootTrajectoryWorld["world_positions"],
            rootTrajectoryWorld["world_directions"],
            rootTrajectoryWorld["world_velocities"],
            terrainProvider,
            projectToTerrain=projectToTerrain,
            heightOffset=heightOffset,
        )

    return ApplyTrajectoryGroundProjection(
        rootTrajectoryWorld["world_positions"],
        rootTrajectoryWorld["world_directions"],
        rootTrajectoryWorld["world_velocities"],
        groundHeight=groundHeight,
        projectToGround=projectToGround,
        heightOffset=heightOffset,
    )


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

    if "display_rotations" in terrainTrajectorySource:
        displayRotations = np.asarray(
            terrainTrajectorySource["display_rotations"][sampleFrames],
            dtype=np.float32,
        )
        worldDirections = quat.mul_vec(
            displayRotations,
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
                    ProjectVectorToPlane(direction, normal, fallback=ROOT_FORWARD_AXIS)
                    for direction, normal in zip(worldDirections, terrainNormals)
                ],
                dtype=np.float32,
            )
        if alignVelocitiesToTerrain:
            worldVelocities = np.asarray(
                [
                    RemoveVectorComponentAlongNormal(velocity, normal)
                    for velocity, normal in zip(worldVelocities, terrainNormals)
                ],
                dtype=np.float32,
            )

    return {
        "world_positions": worldPositions.astype(np.float32),
        "world_directions": NormalizeDirections(worldDirections, fallback=ROOT_FORWARD_AXIS),
        "world_velocities": worldVelocities.astype(np.float32),
    }
