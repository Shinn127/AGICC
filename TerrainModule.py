import cffi
import numpy as np

from pyray import Matrix, Mesh
from raylib import *


ffi = cffi.FFI()

DEFAULT_TERRAIN_CELL_SIZE = 0.1
DEFAULT_TERRAIN_PADDING = 0.5
MAX_TERRAIN_VERTEX_COUNT = 60000

__all__ = [
    "TerrainProvider",
    "BuildTerrainProvider",
    "BuildTerrainSamplesFromContactData",
    "BuildTerrainProviderFromContactData",
    "BuildTerrainHeightGrid",
    "LoadTerrainModelFromProvider",
]


# Core terrain query object.

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


# Public terrain constructors.

def BuildTerrainProvider(
    samplePositions,
    fallbackHeight=0.0,
    kNearest=8):
    return TerrainProvider(
        samplePositions,
        fallbackHeight=fallbackHeight,
        kNearest=kNearest,
    )


# Integration adapters from contact observations into terrain data.

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
    return BuildTerrainProvider(
        samplePositions,
        fallbackHeight=fallbackHeight,
        kNearest=kNearest,
    )


# Internal helpers for height-grid and mesh generation.

def _compute_terrain_bounds(samplePositions, padding=DEFAULT_TERRAIN_PADDING, minExtent=1.0):
    samplePositions = np.asarray(samplePositions, dtype=np.float32).reshape((-1, 3))
    padding = float(padding)
    minExtent = float(minExtent)

    if len(samplePositions) == 0:
        halfExtent = 0.5 * minExtent
        return (-halfExtent, halfExtent, -halfExtent, halfExtent)

    minX = float(np.min(samplePositions[:, 0]) - padding)
    maxX = float(np.max(samplePositions[:, 0]) + padding)
    minZ = float(np.min(samplePositions[:, 2]) - padding)
    maxZ = float(np.max(samplePositions[:, 2]) + padding)

    if maxX - minX < minExtent:
        centerX = 0.5 * (minX + maxX)
        minX = centerX - 0.5 * minExtent
        maxX = centerX + 0.5 * minExtent

    if maxZ - minZ < minExtent:
        centerZ = 0.5 * (minZ + maxZ)
        minZ = centerZ - 0.5 * minExtent
        maxZ = centerZ + 0.5 * minExtent

    return (minX, maxX, minZ, maxZ)


def _choose_terrain_cell_size(bounds, desiredCellSize=DEFAULT_TERRAIN_CELL_SIZE, maxVertexCount=MAX_TERRAIN_VERTEX_COUNT):
    minX, maxX, minZ, maxZ = bounds
    cellSize = max(float(desiredCellSize), 1e-3)

    while True:
        numX = int(np.ceil((maxX - minX) / cellSize)) + 1
        numZ = int(np.ceil((maxZ - minZ) / cellSize)) + 1
        if numX * numZ <= maxVertexCount:
            return cellSize, numX, numZ
        cellSize *= 1.25


def BuildTerrainHeightGrid(
    terrainProvider,
    samplePositions,
    cellSize=DEFAULT_TERRAIN_CELL_SIZE,
    padding=DEFAULT_TERRAIN_PADDING,
    maxVertexCount=MAX_TERRAIN_VERTEX_COUNT):

    bounds = _compute_terrain_bounds(samplePositions, padding=padding)
    cellSize, numX, numZ = _choose_terrain_cell_size(
        bounds,
        desiredCellSize=cellSize,
        maxVertexCount=maxVertexCount,
    )

    minX, maxX, minZ, maxZ = bounds
    xs = np.linspace(minX, maxX, numX, dtype=np.float32)
    zs = np.linspace(minZ, maxZ, numZ, dtype=np.float32)

    gridPositions = np.zeros((numZ, numX, 3), dtype=np.float32)
    for zIndex, zValue in enumerate(zs):
        gridPositions[zIndex, :, 0] = xs
        gridPositions[zIndex, :, 2] = zValue

    heights = terrainProvider.sample_heights(gridPositions.reshape(-1, 3)).reshape((numZ, numX)).astype(np.float32)
    gridPositions[:, :, 1] = heights

    return {
        "bounds": bounds,
        "cell_size": float(cellSize),
        "num_x": int(numX),
        "num_z": int(numZ),
        "xs": xs,
        "zs": zs,
        "positions": gridPositions,
        "heights": heights,
    }


def _compute_terrain_grid_normals(gridPositions):
    gridPositions = np.asarray(gridPositions, dtype=np.float32)
    numZ, numX, _ = gridPositions.shape
    normals = np.zeros_like(gridPositions, dtype=np.float32)

    for zIndex in range(numZ):
        for xIndex in range(numX):
            xPrev = max(xIndex - 1, 0)
            xNext = min(xIndex + 1, numX - 1)
            zPrev = max(zIndex - 1, 0)
            zNext = min(zIndex + 1, numZ - 1)

            tangentX = gridPositions[zIndex, xNext] - gridPositions[zIndex, xPrev]
            tangentZ = gridPositions[zNext, xIndex] - gridPositions[zPrev, xIndex]
            normal = np.cross(tangentZ, tangentX).astype(np.float32)
            normalNorm = np.linalg.norm(normal)

            if normalNorm < 1e-8:
                normals[zIndex, xIndex] = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
            else:
                normal = normal / normalNorm
                if normal[1] < 0.0:
                    normal = -normal
                normals[zIndex, xIndex] = normal.astype(np.float32)

    return normals.astype(np.float32)


def _build_terrain_mesh_arrays(heightGrid):
    gridPositions = np.asarray(heightGrid["positions"], dtype=np.float32)
    normalsGrid = _compute_terrain_grid_normals(gridPositions)
    numZ, numX, _ = gridPositions.shape

    vertices = gridPositions.reshape((-1, 3)).astype(np.float32)
    normals = normalsGrid.reshape((-1, 3)).astype(np.float32)

    texcoords = np.zeros((numZ, numX, 2), dtype=np.float32)
    if numX > 1:
        texcoords[:, :, 0] = np.linspace(0.0, 1.0, numX, dtype=np.float32)[np.newaxis, :]
    if numZ > 1:
        texcoords[:, :, 1] = np.linspace(0.0, 1.0, numZ, dtype=np.float32)[:, np.newaxis]
    texcoords = texcoords.reshape((-1, 2)).astype(np.float32)

    triangleCount = (numX - 1) * (numZ - 1) * 2
    indices = np.zeros((triangleCount, 3), dtype=np.uint16)
    triangleIndex = 0

    for zIndex in range(numZ - 1):
        for xIndex in range(numX - 1):
            i00 = zIndex * numX + xIndex
            i10 = i00 + 1
            i01 = (zIndex + 1) * numX + xIndex
            i11 = i01 + 1

            indices[triangleIndex] = (i00, i01, i10)
            indices[triangleIndex + 1] = (i10, i01, i11)
            triangleIndex += 2

    return {
        "vertices": vertices,
        "normals": normals,
        "texcoords": texcoords,
        "indices": indices.reshape(-1),
    }


# Render adapter: converts the terrain height field into a raylib model.

def LoadTerrainModelFromProvider(
    terrainProvider,
    samplePositions,
    cellSize=DEFAULT_TERRAIN_CELL_SIZE,
    padding=DEFAULT_TERRAIN_PADDING,
    maxVertexCount=MAX_TERRAIN_VERTEX_COUNT):

    heightGrid = BuildTerrainHeightGrid(
        terrainProvider,
        samplePositions,
        cellSize=cellSize,
        padding=padding,
        maxVertexCount=maxVertexCount,
    )
    meshArrays = _build_terrain_mesh_arrays(heightGrid)

    mesh = Mesh()
    mesh.vertexCount = int(len(meshArrays["vertices"]))
    mesh.triangleCount = int(len(meshArrays["indices"]) // 3)
    mesh.vertices = MemAlloc(mesh.vertexCount * 3 * ffi.sizeof("float"))
    mesh.texcoords = MemAlloc(mesh.vertexCount * 2 * ffi.sizeof("float"))
    mesh.normals = MemAlloc(mesh.vertexCount * 3 * ffi.sizeof("float"))
    mesh.indices = MemAlloc(mesh.triangleCount * 3 * ffi.sizeof("unsigned short"))

    ffi.memmove(mesh.vertices, meshArrays["vertices"].reshape(-1).astype(np.float32).tobytes(), mesh.vertexCount * 3 * ffi.sizeof("float"))
    ffi.memmove(mesh.texcoords, meshArrays["texcoords"].reshape(-1).astype(np.float32).tobytes(), mesh.vertexCount * 2 * ffi.sizeof("float"))
    ffi.memmove(mesh.normals, meshArrays["normals"].reshape(-1).astype(np.float32).tobytes(), mesh.vertexCount * 3 * ffi.sizeof("float"))
    ffi.memmove(mesh.indices, meshArrays["indices"].astype(np.uint16).tobytes(), mesh.triangleCount * 3 * ffi.sizeof("unsigned short"))

    UploadMesh(ffi.addressof(mesh), False)
    model = LoadModelFromMesh(mesh)
    model.transform = MatrixIdentity()

    return model, heightGrid
