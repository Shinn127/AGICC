import numpy as np


DEFAULT_CONTACT_CONFIGS = [
    ("LeftFoot", 0.08, 0.15),
    ("LeftToeBase", 0.06, 0.15),
    ("RightFoot", 0.08, 0.15),
    ("RightToeBase", 0.06, 0.15),
]
DEFAULT_BOOTSTRAP_ENVELOPE_WINDOW = 31
DEFAULT_PROXY_SEGMENT_SPACING = 0.12
DEFAULT_PROXY_CLEARANCE_MARGIN = 0.005
JOINT_PROXY_RADIUS_RULES = (
    ("Hips", 0.12),
    ("Spine", 0.11),
    ("Neck", 0.06),
    ("Head", 0.09),
    ("Shoulder", 0.05),
    ("UpLeg", 0.09),
    ("Leg", 0.07),
    ("Foot", 0.055),
    ("ToeBase", 0.035),
    ("Arm", 0.06),
    ("ForeArm", 0.05),
    ("Hand", 0.035),
    (("Thumb", "Index", "Middle", "Ring", "Pinky"), 0.02),
    ("End", 0.02),
)

# Public contact builders.

def _resolve_contact_joint_indices(jointNames, contactConfigs):
    jointIndices = []
    resolvedConfigs = []

    for config in contactConfigs:
        jointName, heightThreshold, velocityThreshold = config
        if jointName not in jointNames:
            raise ValueError(f'Contact joint "{jointName}" was not found in joint names.')
        jointIndices.append(jointNames.index(jointName))
        resolvedConfigs.append((jointName, float(heightThreshold), float(velocityThreshold)))

    return np.asarray(jointIndices, dtype=np.int32), resolvedConfigs


def BuildContactSource(globalPositions, globalVelocities, jointNames, contactConfigs=DEFAULT_CONTACT_CONFIGS):
    globalPositions = np.asarray(globalPositions, dtype=np.float32)
    globalVelocities = np.asarray(globalVelocities, dtype=np.float32)
    jointIndices, resolvedConfigs = _resolve_contact_joint_indices(jointNames, contactConfigs)

    contactPositions = globalPositions[:, jointIndices]
    contactVelocities = globalVelocities[:, jointIndices]
    contactSpeeds = np.linalg.norm(contactVelocities, axis=-1).astype(np.float32)

    return {
        "joint_indices": jointIndices,
        "joint_names": [config[0] for config in resolvedConfigs],
        "height_thresholds": np.asarray([config[1] for config in resolvedConfigs], dtype=np.float32),
        "velocity_thresholds": np.asarray([config[2] for config in resolvedConfigs], dtype=np.float32),
        "positions": contactPositions.astype(np.float32),
        "velocities": contactVelocities.astype(np.float32),
        "speeds": contactSpeeds,
    }


def _compute_raw_contacts(contactSource):
    positions = contactSource["positions"]
    speeds = contactSource["speeds"]
    heightThresholds = contactSource["height_thresholds"]
    velocityThresholds = contactSource["velocity_thresholds"]

    heightCriterion = positions[..., 1] < heightThresholds[np.newaxis, :]
    velocityCriterion = speeds < velocityThresholds[np.newaxis, :]
    return (heightCriterion & velocityCriterion).astype(np.uint8)


def _filter_contacts(contacts, filterSize=6):
    contacts = np.asarray(contacts, dtype=np.uint8)
    if filterSize <= 1 or len(contacts) == 0:
        return contacts.copy()

    radius = filterSize // 2
    filteredContacts = np.empty_like(contacts)

    for contactIndex in range(contacts.shape[1]):
        padded = np.pad(contacts[:, contactIndex], (radius, radius), mode="edge")
        for frameIndex in range(contacts.shape[0]):
            window = padded[frameIndex:frameIndex + filterSize]
            filteredContacts[frameIndex, contactIndex] = 1 if np.median(window) >= 0.5 else 0

    return filteredContacts.astype(np.uint8)


def _compute_height_lower_envelope(heights, windowSize=DEFAULT_BOOTSTRAP_ENVELOPE_WINDOW):
    heights = np.asarray(heights, dtype=np.float32)
    if windowSize <= 1 or len(heights) == 0:
        return heights.copy()

    radius = windowSize // 2
    lowerEnvelope = np.empty_like(heights, dtype=np.float32)

    for contactIndex in range(heights.shape[1]):
        padded = np.pad(heights[:, contactIndex], (radius, radius), mode="edge")
        for frameIndex in range(heights.shape[0]):
            window = padded[frameIndex:frameIndex + windowSize]
            lowerEnvelope[frameIndex, contactIndex] = float(np.min(window))

    return lowerEnvelope.astype(np.float32)


def _compute_bootstrap_contacts(
    contactSource,
    envelopeWindow=DEFAULT_BOOTSTRAP_ENVELOPE_WINDOW):

    positions = contactSource["positions"]
    speeds = contactSource["speeds"]
    heightThresholds = contactSource["height_thresholds"]
    velocityThresholds = contactSource["velocity_thresholds"]

    heights = positions[..., 1].astype(np.float32)
    lowerEnvelope = _compute_height_lower_envelope(heights, windowSize=envelopeWindow)
    relativeHeights = (heights - lowerEnvelope).astype(np.float32)

    heightCriterion = relativeHeights < heightThresholds[np.newaxis, :]
    velocityCriterion = speeds < velocityThresholds[np.newaxis, :]
    contactsRaw = (heightCriterion & velocityCriterion).astype(np.uint8)

    return {
        "contacts_raw": contactsRaw,
        "lower_envelope": lowerEnvelope,
        "relative_heights": relativeHeights,
    }


def _build_contact_data_result(contactSource, contactsRaw, contactsFiltered, **extraFields):
    return {
        "joint_indices": contactSource["joint_indices"],
        "joint_names": contactSource["joint_names"],
        "positions": contactSource["positions"],
        "velocities": contactSource["velocities"],
        "speeds": contactSource["speeds"],
        "height_thresholds": contactSource["height_thresholds"],
        "velocity_thresholds": contactSource["velocity_thresholds"],
        **extraFields,
        "contacts_raw": contactsRaw,
        "contacts_filtered": contactsFiltered,
    }


def BuildContactData(
    globalPositions,
    globalVelocities,
    jointNames,
    contactConfigs=DEFAULT_CONTACT_CONFIGS,
    applyTemporalFilter=True,
    filterSize=6,
    terrainProvider=None,
    bootstrap=False,
    envelopeWindow=DEFAULT_BOOTSTRAP_ENVELOPE_WINDOW):

    if bootstrap:
        return BuildBootstrapContactData(
            globalPositions,
            globalVelocities,
            jointNames,
            contactConfigs=contactConfigs,
            applyTemporalFilter=applyTemporalFilter,
            filterSize=filterSize,
            envelopeWindow=envelopeWindow,
        )

    if terrainProvider is not None:
        return BuildTerrainAwareContactData(
            globalPositions,
            globalVelocities,
            jointNames,
            terrainProvider,
            contactConfigs=contactConfigs,
            applyTemporalFilter=applyTemporalFilter,
            filterSize=filterSize,
        )

    contactSource = BuildContactSource(
        globalPositions,
        globalVelocities,
        jointNames,
        contactConfigs=contactConfigs,
    )
    contactsRaw = _compute_raw_contacts(contactSource)
    contactsFiltered = (
        _filter_contacts(contactsRaw, filterSize=filterSize)
        if applyTemporalFilter else
        contactsRaw.copy()
    )

    return _build_contact_data_result(contactSource, contactsRaw, contactsFiltered)


# Mode-specific contact adapters.


def BuildBootstrapContactData(
    globalPositions,
    globalVelocities,
    jointNames,
    contactConfigs=DEFAULT_CONTACT_CONFIGS,
    applyTemporalFilter=True,
    filterSize=6,
    envelopeWindow=DEFAULT_BOOTSTRAP_ENVELOPE_WINDOW):

    contactSource = BuildContactSource(
        globalPositions,
        globalVelocities,
        jointNames,
        contactConfigs=contactConfigs,
    )
    bootstrapResult = _compute_bootstrap_contacts(
        contactSource,
        envelopeWindow=envelopeWindow,
    )
    contactsRaw = bootstrapResult["contacts_raw"]
    contactsFiltered = (
        _filter_contacts(contactsRaw, filterSize=filterSize)
        if applyTemporalFilter else
        contactsRaw.copy()
    )

    return _build_contact_data_result(
        contactSource,
        contactsRaw,
        contactsFiltered,
        height_lower_envelope=bootstrapResult["lower_envelope"],
        relative_heights=bootstrapResult["relative_heights"],
    )


def _compute_terrain_aware_contacts(contactSource, terrainProvider):
    positions = contactSource["positions"]
    speeds = contactSource["speeds"]
    heightThresholds = contactSource["height_thresholds"]
    velocityThresholds = contactSource["velocity_thresholds"]

    terrainHeights = terrainProvider.sample_heights(positions.reshape(-1, 3)).reshape(positions.shape[:2]).astype(np.float32)
    distancesToGround = (positions[..., 1] - terrainHeights).astype(np.float32)

    heightCriterion = distancesToGround < heightThresholds[np.newaxis, :]
    velocityCriterion = speeds < velocityThresholds[np.newaxis, :]
    contactsRaw = (heightCriterion & velocityCriterion).astype(np.uint8)

    return {
        "contacts_raw": contactsRaw,
        "terrain_heights": terrainHeights,
        "distances_to_ground": distancesToGround,
    }


def BuildTerrainAwareContactData(
    globalPositions,
    globalVelocities,
    jointNames,
    terrainProvider,
    contactConfigs=DEFAULT_CONTACT_CONFIGS,
    applyTemporalFilter=True,
    filterSize=6):

    contactSource = BuildContactSource(
        globalPositions,
        globalVelocities,
        jointNames,
        contactConfigs=contactConfigs,
    )
    terrainAwareResult = _compute_terrain_aware_contacts(contactSource, terrainProvider)
    contactsRaw = terrainAwareResult["contacts_raw"]
    contactsFiltered = (
        _filter_contacts(contactsRaw, filterSize=filterSize)
        if applyTemporalFilter else
        contactsRaw.copy()
    )

    return _build_contact_data_result(
        contactSource,
        contactsRaw,
        contactsFiltered,
        terrain_heights=terrainAwareResult["terrain_heights"],
        distances_to_ground=terrainAwareResult["distances_to_ground"],
    )


def EstimateJointProxyRadius(jointName):
    jointName = str(jointName)
    for patterns, radius in JOINT_PROXY_RADIUS_RULES:
        if isinstance(patterns, str):
            patterns = (patterns,)
        if any(pattern in jointName for pattern in patterns):
            return radius
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
