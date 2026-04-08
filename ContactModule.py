import numpy as np


DEFAULT_CONTACT_CONFIGS = [
    ("LeftFoot", 0.08, 0.15),
    ("LeftToeBase", 0.06, 0.15),
    ("RightFoot", 0.08, 0.15),
    ("RightToeBase", 0.06, 0.15),
]


def ResolveContactJointIndices(jointNames, contactConfigs):
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
    jointIndices, resolvedConfigs = ResolveContactJointIndices(jointNames, contactConfigs)

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


def ComputeRawContacts(contactSource):
    positions = contactSource["positions"]
    speeds = contactSource["speeds"]
    heightThresholds = contactSource["height_thresholds"]
    velocityThresholds = contactSource["velocity_thresholds"]

    heightCriterion = positions[..., 1] < heightThresholds[np.newaxis, :]
    velocityCriterion = speeds < velocityThresholds[np.newaxis, :]
    return (heightCriterion & velocityCriterion).astype(np.uint8)


def FilterContacts(contacts, filterSize=6):
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


def BuildContactData(
    globalPositions,
    globalVelocities,
    jointNames,
    contactConfigs=DEFAULT_CONTACT_CONFIGS,
    applyTemporalFilter=True,
    filterSize=6):

    contactSource = BuildContactSource(
        globalPositions,
        globalVelocities,
        jointNames,
        contactConfigs=contactConfigs,
    )
    contactsRaw = ComputeRawContacts(contactSource)
    contactsFiltered = (
        FilterContacts(contactsRaw, filterSize=filterSize)
        if applyTemporalFilter else
        contactsRaw.copy()
    )

    return {
        "joint_indices": contactSource["joint_indices"],
        "joint_names": contactSource["joint_names"],
        "positions": contactSource["positions"],
        "velocities": contactSource["velocities"],
        "speeds": contactSource["speeds"],
        "height_thresholds": contactSource["height_thresholds"],
        "velocity_thresholds": contactSource["velocity_thresholds"],
        "contacts_raw": contactsRaw,
        "contacts_filtered": contactsFiltered,
    }


def GetFrameContacts(contactData, frameIndex, filtered=True):
    key = "contacts_filtered" if filtered else "contacts_raw"
    return contactData[key][frameIndex]
