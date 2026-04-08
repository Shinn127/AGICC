import numpy as np


DEFAULT_PROXY_SEGMENT_SPACING = 0.12
DEFAULT_PROXY_CLEARANCE_MARGIN = 0.005


def EstimateJointProxyRadius(jointName):
    jointName = str(jointName)

    if "Hips" in jointName:
        return 0.12
    if "Spine" in jointName:
        return 0.11
    if "Neck" in jointName:
        return 0.06
    if "Head" in jointName:
        return 0.09
    if "Shoulder" in jointName:
        return 0.05
    if "UpLeg" in jointName:
        return 0.09
    if "Leg" in jointName:
        return 0.07
    if "Foot" in jointName:
        return 0.055
    if "ToeBase" in jointName:
        return 0.035
    if "Arm" in jointName:
        return 0.06
    if "ForeArm" in jointName:
        return 0.05
    if "Hand" in jointName:
        return 0.035
    if "Thumb" in jointName or "Index" in jointName or "Middle" in jointName or "Ring" in jointName or "Pinky" in jointName:
        return 0.02
    if "End" in jointName:
        return 0.02
    return 0.04


def _build_segment_samples(referencePositions, parents, jointNames, segmentSpacing=DEFAULT_PROXY_SEGMENT_SPACING):
    parentIndices = []
    childIndices = []
    tValues = []
    radii = []

    for childIndex, parentIndex in enumerate(parents):
        if parentIndex < 0:
            continue

        parentPos = referencePositions[parentIndex]
        childPos = referencePositions[childIndex]
        segmentLength = float(np.linalg.norm(childPos - parentPos))
        sampleCount = max(1, int(np.ceil(segmentLength / max(float(segmentSpacing), 1e-4))))
        sampleTs = np.linspace(0.0, 1.0, sampleCount + 2, dtype=np.float32)[1:-1]
        radius = EstimateJointProxyRadius(jointNames[childIndex])

        for sampleT in sampleTs:
            parentIndices.append(parentIndex)
            childIndices.append(childIndex)
            tValues.append(float(sampleT))
            radii.append(radius)

    return {
        "parent_indices": np.asarray(parentIndices, dtype=np.int32),
        "child_indices": np.asarray(childIndices, dtype=np.int32),
        "t_values": np.asarray(tValues, dtype=np.float32),
        "radii": np.asarray(radii, dtype=np.float32),
    }


def BuildBodyProxyLayout(referencePositions, parents, jointNames, segmentSpacing=DEFAULT_PROXY_SEGMENT_SPACING):
    referencePositions = np.asarray(referencePositions, dtype=np.float32)
    jointRadii = np.asarray([EstimateJointProxyRadius(name) for name in jointNames], dtype=np.float32)
    segmentLayout = _build_segment_samples(referencePositions, parents, jointNames, segmentSpacing=segmentSpacing)

    return {
        "joint_indices": np.arange(len(jointNames), dtype=np.int32),
        "joint_radii": jointRadii,
        "segment_parent_indices": segmentLayout["parent_indices"],
        "segment_child_indices": segmentLayout["child_indices"],
        "segment_t_values": segmentLayout["t_values"],
        "segment_radii": segmentLayout["radii"],
        "joint_names": list(jointNames),
    }


def BuildBodyProxyData(globalPositions, parents, jointNames, segmentSpacing=DEFAULT_PROXY_SEGMENT_SPACING):
    globalPositions = np.asarray(globalPositions, dtype=np.float32)
    layout = BuildBodyProxyLayout(globalPositions[0], parents, jointNames, segmentSpacing=segmentSpacing)

    jointPositions = globalPositions[:, layout["joint_indices"]]

    if len(layout["segment_t_values"]) > 0:
        parentPositions = globalPositions[:, layout["segment_parent_indices"]]
        childPositions = globalPositions[:, layout["segment_child_indices"]]
        segmentPositions = (
            (1.0 - layout["segment_t_values"][np.newaxis, :, np.newaxis]) * parentPositions +
            layout["segment_t_values"][np.newaxis, :, np.newaxis] * childPositions
        ).astype(np.float32)
    else:
        segmentPositions = np.zeros((len(globalPositions), 0, 3), dtype=np.float32)

    allPositions = np.concatenate([jointPositions, segmentPositions], axis=1).astype(np.float32)
    allRadii = np.concatenate([layout["joint_radii"], layout["segment_radii"]]).astype(np.float32)

    return {
        "layout": layout,
        "positions": allPositions,
        "radii": allRadii,
        "joint_count": len(layout["joint_indices"]),
        "segment_sample_count": len(layout["segment_t_values"]),
    }


def BuildBodyProxyFrame(globalPositionsFrame, bodyProxyLayout):
    globalPositionsFrame = np.asarray(globalPositionsFrame, dtype=np.float32)
    layout = bodyProxyLayout

    jointPositions = globalPositionsFrame[layout["joint_indices"]]

    if len(layout["segment_t_values"]) > 0:
        parentPositions = globalPositionsFrame[layout["segment_parent_indices"]]
        childPositions = globalPositionsFrame[layout["segment_child_indices"]]
        segmentPositions = (
            (1.0 - layout["segment_t_values"][:, np.newaxis]) * parentPositions +
            layout["segment_t_values"][:, np.newaxis] * childPositions
        ).astype(np.float32)
    else:
        segmentPositions = np.zeros((0, 3), dtype=np.float32)

    allPositions = np.concatenate([jointPositions, segmentPositions], axis=0).astype(np.float32)
    allRadii = np.concatenate([layout["joint_radii"], layout["segment_radii"]]).astype(np.float32)

    return {
        "layout": layout,
        "positions": allPositions,
        "radii": allRadii,
        "joint_count": len(layout["joint_indices"]),
        "segment_sample_count": len(layout["segment_t_values"]),
    }


def ComputeTerrainPenetrationData(bodyProxyData, terrainProvider, clearanceMargin=DEFAULT_PROXY_CLEARANCE_MARGIN):
    proxyPositions = np.asarray(bodyProxyData["positions"], dtype=np.float32)
    proxyRadii = np.asarray(bodyProxyData["radii"], dtype=np.float32)

    terrainHeights = terrainProvider.sample_heights(proxyPositions.reshape(-1, 3)).reshape(proxyPositions.shape[:2]).astype(np.float32)
    bottomHeights = (proxyPositions[..., 1] - proxyRadii[np.newaxis, :] - float(clearanceMargin)).astype(np.float32)
    penetrationDepths = (terrainHeights - bottomHeights).astype(np.float32)
    penetrationMask = (penetrationDepths > 0.0).astype(np.uint8)

    terrainPoints = proxyPositions.copy()
    terrainPoints[..., 1] = terrainHeights
    bottomPoints = proxyPositions.copy()
    bottomPoints[..., 1] = bottomHeights

    return {
        "terrain_heights": terrainHeights,
        "bottom_heights": bottomHeights,
        "penetration_depths": penetrationDepths,
        "penetration_mask": penetrationMask,
        "terrain_points": terrainPoints.astype(np.float32),
        "bottom_points": bottomPoints.astype(np.float32),
        "penetration_count_per_frame": np.sum(penetrationMask, axis=1).astype(np.int32),
        "max_penetration_per_frame": np.maximum(np.max(penetrationDepths, axis=1), 0.0).astype(np.float32),
    }


def ComputeTerrainPenetrationFrame(bodyProxyFrame, terrainProvider, clearanceMargin=DEFAULT_PROXY_CLEARANCE_MARGIN):
    proxyPositions = np.asarray(bodyProxyFrame["positions"], dtype=np.float32)
    proxyRadii = np.asarray(bodyProxyFrame["radii"], dtype=np.float32)

    terrainHeights = terrainProvider.sample_heights(proxyPositions).astype(np.float32)
    bottomHeights = (proxyPositions[:, 1] - proxyRadii - float(clearanceMargin)).astype(np.float32)
    penetrationDepths = (terrainHeights - bottomHeights).astype(np.float32)
    penetrationMask = (penetrationDepths > 0.0).astype(np.uint8)

    terrainPoints = proxyPositions.copy()
    terrainPoints[:, 1] = terrainHeights
    bottomPoints = proxyPositions.copy()
    bottomPoints[:, 1] = bottomHeights

    return {
        "terrain_heights": terrainHeights,
        "bottom_heights": bottomHeights,
        "penetration_depths": penetrationDepths,
        "penetration_mask": penetrationMask,
        "terrain_points": terrainPoints.astype(np.float32),
        "bottom_points": bottomPoints.astype(np.float32),
        "penetration_count": int(np.sum(penetrationMask)),
        "max_penetration": float(max(float(np.max(penetrationDepths)) if len(penetrationDepths) > 0 else 0.0, 0.0)),
    }
