import numpy as np


DEFAULT_CONTACT_CONFIGS = [
    ("LeftFoot", 0.08, 0.15),
    ("LeftToeBase", 0.06, 0.15),
    ("RightFoot", 0.08, 0.15),
    ("RightToeBase", 0.06, 0.15),
]
DEFAULT_BOOTSTRAP_ENVELOPE_WINDOW = 31

__all__ = [
    "DEFAULT_CONTACT_CONFIGS",
    "DEFAULT_BOOTSTRAP_ENVELOPE_WINDOW",
    "BuildContactSource",
    "BuildContactData",
    "BuildBootstrapContactData",
    "BuildTerrainAwareContactData",
    "GetFrameContacts",
]


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


def GetFrameContacts(contactData, frameIndex, filtered=True):
    key = "contacts_filtered" if filtered else "contacts_raw"
    return contactData[key][frameIndex]
