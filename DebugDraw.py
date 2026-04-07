import numpy as np

from pyray import Vector3, Vector4
from raylib import *


def DrawTransform(position, rotation, scale):
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


def DrawSkeleton(positions, rotations, parents, color):
    for i in range(len(positions)):

        DrawSphereWires(
            Vector3(*positions[i]),
            0.01,
            4,
            6,
            color)

        DrawTransform(positions[i], rotations[i], 0.1)

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


def GetTrajectoryDebugColor(sampleOffset):
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
        color = GetTrajectoryDebugColor(sampleOffset)
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
