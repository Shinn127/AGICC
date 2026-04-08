import numpy as np


def ClampFrameIndex(frameIndex, frameCount):
    if frameCount <= 0:
        raise ValueError("frameCount must be positive.")
    return int(np.clip(frameIndex, 0, frameCount - 1))


def ComputeFiniteDifferenceVelocities(samples, dt):
    samples = np.asarray(samples, dtype=np.float32)
    velocities = np.zeros_like(samples, dtype=np.float32)

    if len(samples) == 0:
        return velocities
    if len(samples) == 1:
        return velocities
    if len(samples) == 2:
        singleStepVelocity = ((samples[1] - samples[0]) / dt).astype(np.float32)
        velocities[:] = singleStepVelocity
        return velocities

    velocities[1:-1] = (
        0.5 * (samples[2:] - samples[1:-1]) / dt +
        0.5 * (samples[1:-1] - samples[:-2]) / dt
    ).astype(np.float32)

    if len(samples) >= 4:
        velocities[0] = velocities[1] - (velocities[3] - velocities[2])
        velocities[-1] = velocities[-2] + (velocities[-2] - velocities[-3])
    else:
        velocities[0] = velocities[1]
        velocities[-1] = velocities[-2]

    return velocities.astype(np.float32)
