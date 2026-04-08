import numpy as np


class TerrainProvider:

    def __init__(self, samplePositions, fallbackHeight=0.0, kNearest=8):
        self.sample_positions = np.asarray(samplePositions, dtype=np.float32).reshape((-1, 3))
        self.fallback_height = float(fallbackHeight)
        self.k_nearest = max(1, int(kNearest))

    def sample_height(self, worldPosition):
        worldPosition = np.asarray(worldPosition, dtype=np.float32)
        if len(self.sample_positions) == 0:
            return self.fallback_height

        nearestIndices, nearestDistancesSquared = self._get_nearest_indices(worldPosition)
        nearestHeights = self.sample_positions[nearestIndices, 1]

        if np.min(nearestDistancesSquared) < 1e-8:
            return float(nearestHeights[np.argmin(nearestDistancesSquared)])

        weights = 1.0 / np.maximum(nearestDistancesSquared, 1e-6)
        return float(np.sum(weights * nearestHeights) / np.sum(weights))

    def sample_heights(self, worldPositions):
        worldPositions = np.asarray(worldPositions, dtype=np.float32)
        if worldPositions.ndim == 1:
            return np.asarray(self.sample_height(worldPositions), dtype=np.float32)
        return np.asarray([self.sample_height(position) for position in worldPositions], dtype=np.float32)

    def sample_normal(self, worldPosition):
        worldPosition = np.asarray(worldPosition, dtype=np.float32)
        if len(self.sample_positions) < 3:
            return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

        nearestIndices, _ = self._get_nearest_indices(worldPosition, min_required=3)
        nearestPositions = self.sample_positions[nearestIndices]
        centered = nearestPositions - np.mean(nearestPositions, axis=0, keepdims=True)

        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

        normal = vh[-1].astype(np.float32)
        normalNorm = np.linalg.norm(normal)
        if normalNorm < 1e-8:
            return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

        normal = normal / normalNorm
        if normal[1] < 0.0:
            normal = -normal
        return normal.astype(np.float32)

    def sample_normals(self, worldPositions):
        worldPositions = np.asarray(worldPositions, dtype=np.float32)
        if worldPositions.ndim == 1:
            return np.asarray(self.sample_normal(worldPositions), dtype=np.float32)
        return np.asarray([self.sample_normal(position) for position in worldPositions], dtype=np.float32)

    def _get_nearest_indices(self, worldPosition, min_required=1):
        queryXZ = worldPosition[[0, 2]]
        sampleXZ = self.sample_positions[:, [0, 2]]
        distancesSquared = np.sum((sampleXZ - queryXZ[np.newaxis, :]) ** 2, axis=-1)
        nearestCount = min(max(min_required, self.k_nearest), len(self.sample_positions))
        nearestIndices = np.argpartition(distancesSquared, nearestCount - 1)[:nearestCount]
        nearestDistancesSquared = distancesSquared[nearestIndices]
        return nearestIndices, nearestDistancesSquared


def BuildTerrainSamplesFromContactData(
    contactData,
    filtered=True,
    cellSize=0.05):

    positions = np.asarray(contactData["positions"], dtype=np.float32)
    contacts = np.asarray(
        contactData["contacts_filtered" if filtered else "contacts_raw"],
        dtype=np.uint8,
    )

    if len(positions) == 0 or contacts.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    cellSize = max(float(cellSize), 1e-4)
    cellBuckets = {}

    for frameIndex in range(contacts.shape[0]):
        for contactIndex in range(contacts.shape[1]):
            if not contacts[frameIndex, contactIndex]:
                continue
            position = positions[frameIndex, contactIndex]
            cellKey = (
                int(np.round(position[0] / cellSize)),
                int(np.round(position[2] / cellSize)),
            )
            cellBuckets.setdefault(cellKey, []).append(position)

    if not cellBuckets:
        return np.zeros((0, 3), dtype=np.float32)

    terrainSamples = np.asarray(
        [np.mean(np.asarray(bucket, dtype=np.float32), axis=0) for bucket in cellBuckets.values()],
        dtype=np.float32,
    )
    return terrainSamples


def BuildTerrainProviderFromContactData(
    contactData,
    filtered=True,
    fallbackHeight=0.0,
    cellSize=0.05,
    kNearest=8):

    samplePositions = BuildTerrainSamplesFromContactData(
        contactData,
        filtered=filtered,
        cellSize=cellSize,
    )
    return TerrainProvider(
        samplePositions,
        fallbackHeight=fallbackHeight,
        kNearest=kNearest,
    )
