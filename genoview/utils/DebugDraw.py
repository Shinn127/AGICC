import numpy as np

from pyray import Color, Vector3, Vector4
from raylib import *

# Internal drawing helpers.

def _draw_transform(position, rotation, scale):
    rotMatrix = QuaternionToMatrix(Vector4(*rotation))

    DrawLine3D(
        Vector3(*position),
        Vector3Add(Vector3(*position), Vector3(scale * rotMatrix.m0, scale * rotMatrix.m1, scale * rotMatrix.m2)),
        RED)

    DrawLine3D(
        Vector3(*position),
        Vector3Add(Vector3(*position), Vector3(scale * rotMatrix.m4, scale * rotMatrix.m5, scale * rotMatrix.m6)),
        GREEN)

    DrawLine3D(
        Vector3(*position),
        Vector3Add(Vector3(*position), Vector3(scale * rotMatrix.m8, scale * rotMatrix.m9, scale * rotMatrix.m10)),
        BLUE)


# Public debug-space helpers and draw API.

def DrawSkeleton(positions, rotations, parents, color):
    for i in range(len(positions)):

        DrawSphereWires(
            Vector3(*positions[i]),
            0.01,
            4,
            6,
            color)

        _draw_transform(positions[i], rotations[i], 0.1)

        if parents[i] != -1:

            DrawLine3D(
                Vector3(*positions[i]),
                Vector3(*positions[parents[i]]),
                color)


def OffsetPositions(positions, offset):
    return positions + np.asarray([offset.x, offset.y, offset.z], dtype=np.float32)


def DrawPoseReconstructionError(originalPositions, reconstructedPositions, color):
    for i in range(len(originalPositions)):
        DrawLine3D(
            Vector3(*originalPositions[i]),
            Vector3(*reconstructedPositions[i]),
            color)


def DrawContactStates(
    contactPositions,
    contacts,
    activeColor=Color(90, 90, 90, 255),
    inactiveColor=Color(190, 190, 190, 255),
    activeSize=0.05,
    inactiveSize=0.05,
    drawInactive=True):

    for i in range(len(contactPositions)):
        isActive = bool(contacts[i])
        if isActive or drawInactive:
            DrawSphereWires(
                Vector3(*contactPositions[i]),
                activeSize if isActive else inactiveSize,
                4,
                10,
                activeColor if isActive else inactiveColor)


def DrawTerrainSamples(
    samplePositions,
    color=Color(120, 160, 90, 255),
    radius=0.03,
    rings=4,
    slices=8):

    for position in np.asarray(samplePositions, dtype=np.float32):
        DrawSphereWires(
            Vector3(*position),
            radius,
            rings,
            slices,
            color)


def DrawTerrainNormals(
    samplePositions,
    sampleNormals,
    color=Color(70, 120, 70, 255),
    scale=0.12):

    for position, normal in zip(
        np.asarray(samplePositions, dtype=np.float32),
        np.asarray(sampleNormals, dtype=np.float32),
    ):
        DrawLine3D(
            Vector3(*position),
            Vector3(*(position + scale * normal)),
            color)


def DrawBodyProxyFrame(
    proxyPositions,
    proxyRadii,
    color=Color(120, 140, 180, 180),
    rings=4,
    slices=8):

    for position, radius in zip(
        np.asarray(proxyPositions, dtype=np.float32),
        np.asarray(proxyRadii, dtype=np.float32),
    ):
        DrawSphereWires(
            Vector3(*position),
            float(radius),
            rings,
            slices,
            color,
        )


def DrawTerrainPenetrationFrame(
    bodyProxyPositions,
    penetrationFrame,
    pointColor=Color(200, 60, 60, 255),
    lineColor=Color(220, 120, 80, 255),
    pointRadius=0.02):

    penetrationMask = penetrationFrame["penetration_mask"]
    proxyPositions = np.asarray(bodyProxyPositions, dtype=np.float32)
    terrainPoints = penetrationFrame["terrain_points"]
    bottomPoints = penetrationFrame["bottom_points"]

    for isPenetrating, proxyPosition, terrainPoint, bottomPoint in zip(
        penetrationMask,
        proxyPositions,
        terrainPoints,
        bottomPoints,
    ):
        if not isPenetrating:
            continue
        DrawSphere(
            Vector3(*proxyPosition),
            pointRadius,
            pointColor,
        )
        DrawLine3D(
            Vector3(*bottomPoint),
            Vector3(*terrainPoint),
            lineColor,
        )


def _get_trajectory_debug_color(sampleOffset):
    if sampleOffset < 0:
        return SKYBLUE
    if sampleOffset > 0:
        return RED
    return ORANGE


def DrawRootTrajectoryDebug(
    worldPositions,
    worldDirections,
    worldVelocities,
    sampleOffsets,
    drawDirection=True,
    drawVelocity=True,
    directionScale=0.2,
    velocityScale=1.0,
    capsuleRadius=0.01,
    capsuleSlices=5,
    capsuleRings=7):

    for i, sampleOffset in enumerate(sampleOffsets):
        color = _get_trajectory_debug_color(sampleOffset)
        radius = 0.07 if sampleOffset == 0 else 0.05
        position = worldPositions[i]

        DrawSphereWires(
            Vector3(*position),
            radius,
            4,
            10,
            color)

        if drawDirection:
            DrawCapsule(
                Vector3(*position),
                Vector3(*(position + directionScale * worldDirections[i])),
                capsuleRadius,
                capsuleSlices,
                capsuleRings,
                color)

        if drawVelocity:
            DrawCapsule(
                Vector3(*position),
                Vector3(*(position + velocityScale * worldVelocities[i])),
                capsuleRadius,
                capsuleSlices,
                capsuleRings,
                GREEN)

        if i > 0:
            DrawLine3D(
                Vector3(*worldPositions[i - 1]),
                Vector3(*position),
                color)
