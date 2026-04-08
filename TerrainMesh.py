import cffi
import numpy as np

from pyray import Matrix, Mesh
from raylib import *


ffi = cffi.FFI()

DEFAULT_TERRAIN_CELL_SIZE = 0.1
DEFAULT_TERRAIN_PADDING = 0.5
MAX_TERRAIN_VERTEX_COUNT = 60000


def ComputeTerrainBounds(samplePositions, padding=DEFAULT_TERRAIN_PADDING, minExtent=1.0):
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


def ChooseTerrainCellSize(bounds, desiredCellSize=DEFAULT_TERRAIN_CELL_SIZE, maxVertexCount=MAX_TERRAIN_VERTEX_COUNT):
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

    bounds = ComputeTerrainBounds(samplePositions, padding=padding)
    cellSize, numX, numZ = ChooseTerrainCellSize(
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


def ComputeTerrainGridNormals(gridPositions):
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


def BuildTerrainMeshArrays(heightGrid):
    gridPositions = np.asarray(heightGrid["positions"], dtype=np.float32)
    normalsGrid = ComputeTerrainGridNormals(gridPositions)
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
    meshArrays = BuildTerrainMeshArrays(heightGrid)

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
