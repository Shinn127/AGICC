from pyray import Camera3D, Vector3
from raylib import *
from raylib.defines import *


class Camera:

    def __init__(self):
        self.cam3d = Camera3D()
        self.cam3d.position = Vector3(2.0, 3.0, 5.0)
        self.cam3d.target = Vector3(-0.5, 1.0, 0.0)
        self.cam3d.up = Vector3(0.0, 1.0, 0.0)
        self.cam3d.fovy = 45.0
        self.cam3d.projection = CAMERA_PERSPECTIVE
        self.azimuth = 0.0
        self.altitude = 0.4
        self.distance = 4.0
        self.offset = Vector3Zero()

    def update(
        self,
        target,
        azimuthDelta,
        altitudeDelta,
        offsetDeltaX,
        offsetDeltaY,
        mouseWheel,
        dt):

        self.azimuth = self.azimuth + 1.0 * dt * -azimuthDelta
        self.altitude = Clamp(self.altitude + 1.0 * dt * altitudeDelta, 0.0, 0.4 * PI)
        self.distance = Clamp(self.distance + 20.0 * dt * -mouseWheel, 0.1, 100.0)

        rotationAzimuth = QuaternionFromAxisAngle(Vector3(0, 1, 0), self.azimuth)
        position = Vector3RotateByQuaternion(Vector3(0, 0, self.distance), rotationAzimuth)
        axis = Vector3Normalize(Vector3CrossProduct(position, Vector3(0, 1, 0)))

        rotationAltitude = QuaternionFromAxisAngle(axis, self.altitude)

        localOffset = Vector3(dt * offsetDeltaX, dt * -offsetDeltaY, 0.0)
        localOffset = Vector3RotateByQuaternion(localOffset, rotationAzimuth)

        self.offset = Vector3Add(self.offset, Vector3RotateByQuaternion(localOffset, rotationAltitude))

        cameraTarget = Vector3Add(self.offset, target)
        eye = Vector3Add(cameraTarget, Vector3RotateByQuaternion(position, rotationAltitude))

        self.cam3d.target = cameraTarget
        self.cam3d.position = eye
