from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Optional

import numpy as np


LABEL_IDLE = "idle"
LABEL_WALK = "walk"
LABEL_RUN = "run"
LABEL_JUMP = "jump"
LABEL_CROUCH = "crouch"
LABEL_TRANSITION = "transition"
LABEL_OTHER = "other"
LABEL_FALL = "fall"
LABEL_GROUND = "ground"
LABEL_GET_UP = "get_up"

TARGET_ACTION_LABELS = (
    LABEL_IDLE,
    LABEL_WALK,
    LABEL_RUN,
    LABEL_JUMP,
    LABEL_CROUCH,
    LABEL_OTHER,
)

ACTION_LABELS = (
    *TARGET_ACTION_LABELS,
    LABEL_TRANSITION,
)

LABEL_TO_INDEX = {label: index for index, label in enumerate(ACTION_LABELS)}
TARGET_LABEL_TO_INDEX = {label: index for index, label in enumerate(TARGET_ACTION_LABELS)}

LEGACY_LABEL_MAP = {
    LABEL_FALL: LABEL_OTHER,
    LABEL_GROUND: LABEL_OTHER,
    LABEL_GET_UP: LABEL_OTHER,
}

DEFAULT_TRANSITION_FRAMES = 8
IDLE_SPEED_THRESHOLD = 0.135
WALK_SPEED_LOW = 0.15
WALK_SPEED_HIGH = 1.60
RUN_SPEED_THRESHOLD = 1.50
RUN_SPEED_SCALE = 0.85
MAX_LABEL_HISTORY = 64
DEFAULT_FLIGHT_RATIO_WINDOW = 15
GET_UP_CONTEXT_LOOKBACK = 18
GET_UP_SPEED_THRESHOLD = 0.9
GET_UP_SPEED_SCALE = 0.6
DEFAULT_TRANSITION_MIN_SEGMENT_LENGTH = 8
DEFAULT_TRANSITION_MAX_SCORE_MARGIN = 0.35
DEFAULT_IDLE_MIN_SECONDS = 0.25
DEFAULT_WALK_MIN_SECONDS = 0.20
DEFAULT_RUN_MIN_SECONDS = 0.20
DEFAULT_JUMP_MIN_SECONDS = 0.12
DEFAULT_CROUCH_MIN_SECONDS = 0.25
DEFAULT_OTHER_MIN_SECONDS = 0.08
DEFAULT_CROUCH_GAP_SECONDS = 0.10
DEFAULT_JUMP_GAP_SECONDS = 0.05
DEFAULT_JUMP_SCAN_SECONDS = 0.18
DEFAULT_TRANSITION_MIN_SECONDS = 0.12
DEFAULT_LEADING_CALIBRATION_SETTLE_SECONDS = 0.42


@dataclass
class LabelAutoParams:
    run_speed_threshold: float = RUN_SPEED_THRESHOLD
    walk_min_speed_percentile: float = 20.0
    run_flight_threshold: float = 0.10
    run_low_contact_max: float = 0.25
    run_speed_margin_ratio: float = 0.10
    turn_yaw_rate_threshold: float = 1.20
    turn_speed_min: float = 0.25
    turn_context_seconds: float = 0.35
    crouch_deep_torso_ratio: float = 0.87
    crouch_locomotion_torso_ratio: float = 0.93
    crouch_knee_flexion_min: float = 50.0
    crouch_speed_max: float = 1.30
    crouch_min_seconds: float = DEFAULT_CROUCH_MIN_SECONDS
    crouch_gap_seconds: float = DEFAULT_CROUCH_GAP_SECONDS
    jump_vy_body_ratio: float = 0.80
    jump_lift_body_ratio: float = 0.08
    jump_min_air_seconds: float = 0.08
    jump_pre_seconds: float = 0.12
    jump_post_seconds: float = 0.18
    jump_scan_seconds: float = DEFAULT_JUMP_SCAN_SECONDS
    jump_gap_seconds: float = DEFAULT_JUMP_GAP_SECONDS
    smoothing_window: int = 7
    transition_frames: int = DEFAULT_TRANSITION_FRAMES
    transition_max_score_margin: float = DEFAULT_TRANSITION_MAX_SCORE_MARGIN
    transition_min_seconds: float = DEFAULT_TRANSITION_MIN_SECONDS


def CreateDefaultLabelAutoParams() -> LabelAutoParams:
    return LabelAutoParams()


def CoerceLabelAutoParams(params=None) -> LabelAutoParams:
    if params is None:
        return CreateDefaultLabelAutoParams()
    if isinstance(params, LabelAutoParams):
        return params
    if isinstance(params, dict):
        result = CreateDefaultLabelAutoParams()
        for key, value in params.items():
            if hasattr(result, key):
                setattr(result, key, value)
        return result
    raise TypeError(f"Unsupported label auto params type: {type(params)!r}")


@dataclass(frozen=True)
class LabelSegment:
    start_frame: int
    end_frame: int
    label: str
    source: str
    transition_in: int = 0
    transition_out: int = 0

    def __post_init__(self):
        if self.label not in ACTION_LABELS:
            raise ValueError(f'Unsupported action label "{self.label}".')
        if self.start_frame < 0:
            raise ValueError("start_frame must be non-negative.")
        if self.end_frame < self.start_frame:
            raise ValueError("end_frame must be greater than or equal to start_frame.")
        if self.transition_in < 0 or self.transition_out < 0:
            raise ValueError("Transition widths must be non-negative.")


@dataclass
class LabelModuleResult:
    clip_name: str
    feature_source: Optional[dict] = None
    auto_params: LabelAutoParams = field(default_factory=CreateDefaultLabelAutoParams)
    auto_scores: Optional[np.ndarray] = None
    auto_labels: Optional[np.ndarray] = None
    auto_confidence: Optional[np.ndarray] = None
    auto_segments: list[LabelSegment] = field(default_factory=list)
    manual_labels: Optional[np.ndarray] = None
    final_labels: Optional[np.ndarray] = None
    final_segments: list[LabelSegment] = field(default_factory=list)
    soft_weights: Optional[np.ndarray] = None
    transition_overrides: list[dict] = field(default_factory=list)
    annotation_path: Optional[str] = None
    annotation_loaded: bool = False
    undo_stack: list[dict] = field(default_factory=list)
    redo_stack: list[dict] = field(default_factory=list)


def NormalizeClipName(clipNameOrPath: str) -> str:
    clipName = Path(str(clipNameOrPath)).stem
    return clipName.strip()


def _remap_legacy_label(label):
    return LEGACY_LABEL_MAP.get(str(label), str(label))


def BuildLabelModuleResult(
    clipNameOrPath: str,
    featureSource=None,
    autoParams=None,
    autoScores=None,
    autoLabels=None,
    autoConfidence=None,
    autoSegments=None,
    manualLabels=None,
    finalLabels=None,
    finalSegments=None,
    softWeights=None,
    transitionOverrides=None,
    annotationPath=None,
    annotationLoaded=False,
) -> LabelModuleResult:
    clipName = NormalizeClipName(clipNameOrPath)

    return LabelModuleResult(
        clip_name=clipName,
        feature_source=featureSource,
        auto_params=CoerceLabelAutoParams(autoParams),
        auto_scores=None if autoScores is None else np.asarray(autoScores, dtype=np.float32),
        auto_labels=None if autoLabels is None else np.asarray(autoLabels),
        auto_confidence=None if autoConfidence is None else np.asarray(autoConfidence, dtype=np.float32),
        auto_segments=[] if autoSegments is None else list(autoSegments),
        manual_labels=None if manualLabels is None else np.asarray(manualLabels, dtype=object),
        final_labels=None if finalLabels is None else np.asarray(finalLabels),
        final_segments=[] if finalSegments is None else list(finalSegments),
        soft_weights=None if softWeights is None else np.asarray(softWeights),
        transition_overrides=[] if transitionOverrides is None else list(transitionOverrides),
        annotation_path=None if annotationPath is None else str(annotationPath),
        annotation_loaded=bool(annotationLoaded),
    )


def CreateEmptySoftWeights(frameCount: int, fillLabel: str = LABEL_OTHER) -> np.ndarray:
    if fillLabel not in LABEL_TO_INDEX:
        raise ValueError(f'Unsupported action label "{fillLabel}".')
    if frameCount < 0:
        raise ValueError("frameCount must be non-negative.")

    weights = np.zeros((frameCount, len(ACTION_LABELS)), dtype=np.float32)
    if frameCount > 0:
        weights[:, LABEL_TO_INDEX[fillLabel]] = 1.0
    return weights


def _safe_normalize(values, eps=1e-6):
    values = np.asarray(values, dtype=np.float32)
    scale = float(np.max(np.abs(values))) if values.size > 0 else 1.0
    scale = max(scale, eps)
    return values / scale


def _score_range(values, low, high):
    values = np.asarray(values, dtype=np.float32)
    center = 0.5 * (float(low) + float(high))
    half_width = max(0.5 * (float(high) - float(low)), 1e-6)
    return np.clip(1.0 - np.abs(values - center) / half_width, 0.0, 1.0).astype(np.float32)


def _score_greater(values, threshold, scale):
    values = np.asarray(values, dtype=np.float32)
    return np.clip((values - float(threshold)) / max(float(scale), 1e-6), 0.0, 1.0).astype(np.float32)


def _score_less(values, threshold, scale):
    values = np.asarray(values, dtype=np.float32)
    return np.clip((float(threshold) - values) / max(float(scale), 1e-6), 0.0, 1.0).astype(np.float32)


def _lookback_max(values, windowSize):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return np.zeros((0,), dtype=np.float32)
    if windowSize <= 1:
        return values.copy()

    result = np.zeros_like(values, dtype=np.float32)
    windowSize = max(1, int(windowSize))
    for frameIndex in range(len(values)):
        start = max(0, frameIndex - windowSize + 1)
        result[frameIndex] = float(np.max(values[start:frameIndex + 1]))
    return result.astype(np.float32)


def _score_margin(scores):
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D array.")
    if scores.shape[1] < 2:
        return np.full((scores.shape[0],), np.inf, dtype=np.float32)

    top2 = np.partition(scores, scores.shape[1] - 2, axis=1)[:, -2:]
    return (top2[:, 1] - top2[:, 0]).astype(np.float32)


def _sliding_mean(values, windowSize):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) == 0:
        return np.zeros((0,), dtype=np.float32)
    if windowSize <= 1:
        return values.copy()

    windowSize = max(1, int(windowSize))
    radius = windowSize // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    kernel = np.ones((windowSize,), dtype=np.float32) / float(windowSize)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _keep_mask_segments_longer_than(mask, minFrames):
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0 or minFrames <= 1:
        return mask.copy()

    result = mask.copy()
    start = None
    for frameIndex, active in enumerate(mask):
        if active and start is None:
            start = frameIndex
        elif not active and start is not None:
            if frameIndex - start < int(minFrames):
                result[start:frameIndex] = False
            start = None
    if start is not None and len(mask) - start < int(minFrames):
        result[start:] = False
    return result


def _seconds_to_frames(dt, seconds, minimum=1):
    dt = max(float(dt), 1e-6)
    return max(int(minimum), int(round(float(seconds) / dt)))


def _close_mask_gaps(mask, maxGapFrames):
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0 or maxGapFrames <= 0:
        return mask.copy()

    result = mask.copy()
    falseStart = None
    seenTrue = False
    for frameIndex, active in enumerate(mask):
        if active:
            if falseStart is not None and seenTrue and (frameIndex - falseStart) <= int(maxGapFrames):
                result[falseStart:frameIndex] = True
            falseStart = None
            seenTrue = True
        elif falseStart is None:
            falseStart = frameIndex
    return result.astype(bool)


def _vector_angle_degrees(vectorA, vectorB):
    vectorA = np.asarray(vectorA, dtype=np.float32)
    vectorB = np.asarray(vectorB, dtype=np.float32)
    vectorA = vectorA / np.maximum(np.linalg.norm(vectorA, axis=-1, keepdims=True), 1e-8)
    vectorB = vectorB / np.maximum(np.linalg.norm(vectorB, axis=-1, keepdims=True), 1e-8)
    cosine = np.clip(np.sum(vectorA * vectorB, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(cosine)).astype(np.float32)


def _joint_angle_degrees(positionA, positionB, positionC):
    return _vector_angle_degrees(positionA - positionB, positionC - positionB)


def _joint_height_series(globalPositions, jointNames, jointName):
    jointIndex = _find_joint_index(jointNames, jointName)
    if jointIndex is None:
        return np.zeros((len(globalPositions),), dtype=np.float32)
    return np.asarray(globalPositions[:, jointIndex, 1], dtype=np.float32)


def _estimate_standing_height(rootHeightAboveGround):
    rootHeightAboveGround = np.asarray(rootHeightAboveGround, dtype=np.float32)
    if rootHeightAboveGround.size == 0:
        return 1.0
    return float(np.percentile(rootHeightAboveGround, 90))


def _estimate_low_height_threshold(standingHeight):
    return 0.6 * max(float(standingHeight), 1e-3)


def _compute_root_yaw_rate(rootDirections, dt):
    rootDirections = np.asarray(rootDirections, dtype=np.float32)
    if len(rootDirections) == 0:
        return np.zeros((0,), dtype=np.float32)

    yaw = np.unwrap(np.arctan2(rootDirections[:, 0], rootDirections[:, 2])).astype(np.float32)
    yawRate = np.zeros_like(yaw, dtype=np.float32)

    if len(yaw) == 1:
        return yawRate
    if len(yaw) == 2:
        yawRate[:] = (yaw[1] - yaw[0]) / dt
        return yawRate.astype(np.float32)

    yawRate[1:-1] = 0.5 * (yaw[2:] - yaw[1:-1]) / dt + 0.5 * (yaw[1:-1] - yaw[:-2]) / dt
    yawRate[0] = yawRate[1]
    yawRate[-1] = yawRate[-2]
    return yawRate.astype(np.float32)


def _resolve_contact_masks(contactData):
    if contactData is None:
        return {
            "left_contact": None,
            "right_contact": None,
            "contact_fraction": None,
        }

    jointNames = list(contactData.get("joint_names", []))
    if "contacts_filtered" in contactData:
        contacts = np.asarray(contactData["contacts_filtered"], dtype=np.float32)
    else:
        contacts = np.asarray(contactData["contacts_raw"], dtype=np.float32)

    if contacts.size == 0:
        frameCount = int(np.asarray(contactData.get("positions", np.zeros((0, 0, 3), dtype=np.float32))).shape[0])
        return {
            "left_contact": np.zeros((frameCount,), dtype=np.float32),
            "right_contact": np.zeros((frameCount,), dtype=np.float32),
            "contact_fraction": np.zeros((frameCount,), dtype=np.float32),
        }

    leftIndices = [index for index, name in enumerate(jointNames) if "left" in name.lower()]
    rightIndices = [index for index, name in enumerate(jointNames) if "right" in name.lower()]

    if not leftIndices:
        leftIndices = [0] if contacts.shape[1] > 0 else []
    if not rightIndices:
        rightIndices = [contacts.shape[1] - 1] if contacts.shape[1] > 0 else []

    leftContact = np.max(contacts[:, leftIndices], axis=1).astype(np.float32) if leftIndices else np.zeros((contacts.shape[0],), dtype=np.float32)
    rightContact = np.max(contacts[:, rightIndices], axis=1).astype(np.float32) if rightIndices else np.zeros((contacts.shape[0],), dtype=np.float32)
    contactFraction = np.mean(contacts, axis=1).astype(np.float32)

    return {
        "left_contact": leftContact,
        "right_contact": rightContact,
        "contact_fraction": contactFraction,
    }


def _compute_motion_energy(poseSource, jointNames=None):
    globalVelocities = np.asarray(poseSource["global_velocities"], dtype=np.float32)
    globalAngularVelocities = np.asarray(poseSource["global_angular_velocities"], dtype=np.float32)
    if len(globalVelocities) == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    linearEnergy = np.mean(np.linalg.norm(globalVelocities, axis=-1), axis=1).astype(np.float32)
    angularEnergy = np.mean(np.linalg.norm(globalAngularVelocities, axis=-1), axis=1).astype(np.float32)
    motionEnergy = (linearEnergy + 0.25 * angularEnergy).astype(np.float32)

    if not jointNames:
        return motionEnergy, motionEnergy.copy()

    upperBodyIndices = [
        index for index, name in enumerate(jointNames)
        if any(token in name.lower() for token in ("spine", "neck", "head", "shoulder", "arm", "hand"))
    ]
    if not upperBodyIndices:
        return motionEnergy, motionEnergy.copy()

    upperLinear = np.mean(np.linalg.norm(globalVelocities[:, upperBodyIndices], axis=-1), axis=1).astype(np.float32)
    upperAngular = np.mean(np.linalg.norm(globalAngularVelocities[:, upperBodyIndices], axis=-1), axis=1).astype(np.float32)
    upperBodyEnergy = (upperLinear + 0.25 * upperAngular).astype(np.float32)
    return motionEnergy, upperBodyEnergy


def _find_joint_index(jointNames, jointName):
    if not jointNames:
        return None

    targetName = str(jointName).lower()
    for index, name in enumerate(jointNames):
        if str(name).lower() == targetName:
            return index
    return None



def _compute_torso_height(globalPositions, jointNames=None):
    globalPositions = np.asarray(globalPositions, dtype=np.float32)
    frameCount = int(len(globalPositions))
    if frameCount == 0 or globalPositions.ndim != 3 or not jointNames:
        return np.zeros((frameCount,), dtype=np.float32)

    hipsIndex = _find_joint_index(jointNames, "Hips")
    headIndex = _find_joint_index(jointNames, "Head")
    if hipsIndex is None or headIndex is None:
        return np.zeros((frameCount,), dtype=np.float32)

    torsoHeight = globalPositions[:, headIndex, 1] - globalPositions[:, hipsIndex, 1]
    return torsoHeight.astype(np.float32)


def _select_side_joint_candidates(jointNames, candidates):
    jointIndices = []
    for jointName in candidates:
        jointIndex = _find_joint_index(jointNames, jointName)
        if jointIndex is not None:
            jointIndices.append(jointIndex)
    return jointIndices


def _compute_side_contact_features(globalPositions, globalVelocities, terrainProvider, jointNames, sidePrefix):
    frameCount = int(len(globalPositions))
    if frameCount == 0:
        return {
            "height": np.zeros((0,), dtype=np.float32),
            "speed_xy": np.zeros((0,), dtype=np.float32),
            "position": np.zeros((0, 3), dtype=np.float32),
        }

    candidates = _select_side_joint_candidates(
        jointNames,
        (
            f"{sidePrefix}ToeBase",
            f"{sidePrefix}Foot",
        ),
    )
    if not candidates:
        return {
            "height": np.zeros((frameCount,), dtype=np.float32),
            "speed_xy": np.zeros((frameCount,), dtype=np.float32),
            "position": np.zeros((frameCount, 3), dtype=np.float32),
        }

    positions = np.asarray(globalPositions[:, candidates], dtype=np.float32)
    velocities = np.asarray(globalVelocities[:, candidates], dtype=np.float32)
    horizontalSpeeds = np.linalg.norm(velocities[..., [0, 2]], axis=-1).astype(np.float32)

    if terrainProvider is not None:
        terrainHeights = terrainProvider.sample_heights(positions.reshape(-1, 3)).reshape(positions.shape[:2]).astype(np.float32)
    else:
        terrainHeights = np.zeros(positions.shape[:2], dtype=np.float32)

    heights = (positions[..., 1] - terrainHeights).astype(np.float32)
    supportIndices = np.argmin(heights, axis=1)
    frameIndices = np.arange(frameCount, dtype=np.int32)
    supportPositions = positions[frameIndices, supportIndices].astype(np.float32)
    supportHeights = heights[frameIndices, supportIndices].astype(np.float32)
    supportSpeeds = horizontalSpeeds[frameIndices, supportIndices].astype(np.float32)

    return {
        "height": supportHeights,
        "speed_xy": supportSpeeds,
        "position": supportPositions,
    }


def _estimate_ground_height(leftFootHeights, rightFootHeights):
    leftFootHeights = np.asarray(leftFootHeights, dtype=np.float32)
    rightFootHeights = np.asarray(rightFootHeights, dtype=np.float32)
    if leftFootHeights.size == 0 and rightFootHeights.size == 0:
        return 0.0
    stacked = np.minimum(leftFootHeights, rightFootHeights) if leftFootHeights.size and rightFootHeights.size else (
        leftFootHeights if leftFootHeights.size else rightFootHeights
    )
    return float(np.percentile(stacked, 1))


def _estimate_body_height(globalPositions, terrainHeights, rootHeightAboveGround, jointNames=None):
    globalPositions = np.asarray(globalPositions, dtype=np.float32)
    terrainHeights = np.asarray(terrainHeights, dtype=np.float32)
    rootHeightAboveGround = np.asarray(rootHeightAboveGround, dtype=np.float32)
    frameCount = int(len(globalPositions))
    if frameCount == 0:
        return 1.0

    headIndex = _find_joint_index(jointNames, "Head")
    headHeightAboveGround = None
    if headIndex is not None:
        headHeightAboveGround = (globalPositions[:, headIndex, 1] - terrainHeights).astype(np.float32)
        if np.any(np.isfinite(headHeightAboveGround)):
            estimate = float(np.percentile(headHeightAboveGround, 90))
            if estimate > 1e-3:
                return estimate

    if rootHeightAboveGround.size > 0:
        fallback = float(np.percentile(rootHeightAboveGround, 90)) / 0.55
        if fallback > 1e-3:
            return fallback
    return 1.0


def _estimate_standing_hips_height(rootHeightAboveGround, groundedMask, rootHorizontalSpeed, motionEnergy):
    rootHeightAboveGround = np.asarray(rootHeightAboveGround, dtype=np.float32)
    groundedMask = np.asarray(groundedMask, dtype=bool)
    rootHorizontalSpeed = np.asarray(rootHorizontalSpeed, dtype=np.float32)
    motionEnergy = np.asarray(motionEnergy, dtype=np.float32)

    if rootHeightAboveGround.size == 0:
        return 1.0

    if np.any(groundedMask):
        groundedSpeeds = rootHorizontalSpeed[groundedMask]
        groundedEnergy = motionEnergy[groundedMask]
        speedThreshold = float(np.percentile(groundedSpeeds, 25)) if groundedSpeeds.size > 0 else float(np.percentile(rootHorizontalSpeed, 25))
        energyThreshold = float(np.percentile(groundedEnergy, 35)) if groundedEnergy.size > 0 else float(np.percentile(motionEnergy, 35))
        candidates = groundedMask & (rootHorizontalSpeed <= speedThreshold) & (motionEnergy <= energyThreshold)
        if np.sum(candidates) >= 12:
            return float(np.percentile(rootHeightAboveGround[candidates], 80))

    return float(np.percentile(rootHeightAboveGround, 85))


def _compute_calibration_pose_masks(
    globalPositions,
    jointNames,
    bodyHeight,
    rootHorizontalSpeed,
    motionEnergy,
    upperBodyEnergy,
):
    frameCount = int(len(globalPositions))
    emptyMask = np.zeros((frameCount,), dtype=np.float32)
    if frameCount == 0 or globalPositions.ndim != 3 or not jointNames:
        return {
            "t_pose_mask": emptyMask,
            "a_pose_mask": emptyMask,
            "calibration_pose_mask": emptyMask,
            "left_elbow_angle": emptyMask,
            "right_elbow_angle": emptyMask,
            "left_arm_abduction": emptyMask,
            "right_arm_abduction": emptyMask,
        }

    leftShoulderIndex = _find_joint_index(jointNames, "LeftShoulder")
    leftArmIndex = _find_joint_index(jointNames, "LeftArm")
    leftForeArmIndex = _find_joint_index(jointNames, "LeftForeArm")
    leftHandIndex = _find_joint_index(jointNames, "LeftHand")
    rightShoulderIndex = _find_joint_index(jointNames, "RightShoulder")
    rightArmIndex = _find_joint_index(jointNames, "RightArm")
    rightForeArmIndex = _find_joint_index(jointNames, "RightForeArm")
    rightHandIndex = _find_joint_index(jointNames, "RightHand")
    hipsIndex = _find_joint_index(jointNames, "Hips")
    if None in (
        leftShoulderIndex,
        leftArmIndex,
        leftForeArmIndex,
        leftHandIndex,
        rightShoulderIndex,
        rightArmIndex,
        rightForeArmIndex,
        rightHandIndex,
        hipsIndex,
    ):
        return {
            "t_pose_mask": emptyMask,
            "a_pose_mask": emptyMask,
            "calibration_pose_mask": emptyMask,
            "left_elbow_angle": emptyMask,
            "right_elbow_angle": emptyMask,
            "left_arm_abduction": emptyMask,
            "right_arm_abduction": emptyMask,
        }

    leftShoulder = globalPositions[:, leftShoulderIndex]
    leftArm = globalPositions[:, leftArmIndex]
    leftForeArm = globalPositions[:, leftForeArmIndex]
    leftHand = globalPositions[:, leftHandIndex]
    rightShoulder = globalPositions[:, rightShoulderIndex]
    rightArm = globalPositions[:, rightArmIndex]
    rightForeArm = globalPositions[:, rightForeArmIndex]
    rightHand = globalPositions[:, rightHandIndex]
    hips = globalPositions[:, hipsIndex]

    leftElbowAngle = _joint_angle_degrees(leftArm, leftForeArm, leftHand)
    rightElbowAngle = _joint_angle_degrees(rightArm, rightForeArm, rightHand)
    downAxis = np.zeros((frameCount, 3), dtype=np.float32)
    downAxis[:, 1] = -1.0
    leftArmAbduction = _vector_angle_degrees(leftHand - leftShoulder, downAxis)
    rightArmAbduction = _vector_angle_degrees(rightHand - rightShoulder, downAxis)

    bodyHeight = max(float(bodyHeight), 1e-3)
    staticBody = (
        (rootHorizontalSpeed < 0.05 * bodyHeight) &
        (motionEnergy < 0.50 * bodyHeight) &
        (upperBodyEnergy < 0.35 * bodyHeight)
    )
    armsStraight = (leftElbowAngle > 135.0) & (rightElbowAngle > 135.0)
    armsSymmetric = np.abs(leftArmAbduction - rightArmAbduction) < 25.0
    leftWristNearShoulder = np.abs(leftHand[:, 1] - leftShoulder[:, 1]) < 0.10 * bodyHeight
    rightWristNearShoulder = np.abs(rightHand[:, 1] - rightShoulder[:, 1]) < 0.10 * bodyHeight
    leftWristBelowShoulder = leftHand[:, 1] < leftShoulder[:, 1] - 0.05 * bodyHeight
    rightWristBelowShoulder = rightHand[:, 1] < rightShoulder[:, 1] - 0.05 * bodyHeight
    leftWristAboveHips = leftHand[:, 1] > hips[:, 1] - 0.15 * bodyHeight
    rightWristAboveHips = rightHand[:, 1] > hips[:, 1] - 0.15 * bodyHeight

    tPoseMask = (
        staticBody &
        armsStraight &
        armsSymmetric &
        (leftArmAbduction > 60.0) &
        (rightArmAbduction > 60.0) &
        leftWristNearShoulder &
        rightWristNearShoulder
    )
    aPoseMask = (
        staticBody &
        armsStraight &
        armsSymmetric &
        (leftArmAbduction > 18.0) &
        (rightArmAbduction > 18.0) &
        (leftArmAbduction < 80.0) &
        (rightArmAbduction < 80.0) &
        leftWristBelowShoulder &
        rightWristBelowShoulder &
        leftWristAboveHips &
        rightWristAboveHips
    )

    minFrames = max(3, int(round(0.10 * 60.0)))
    tPoseMask = _keep_mask_segments_longer_than(tPoseMask, minFrames)
    aPoseMask = _keep_mask_segments_longer_than(aPoseMask, minFrames)
    calibrationPoseMask = (tPoseMask | aPoseMask).astype(np.float32)
    calibrationPoseMask = calibrationPoseMask.astype(bool)
    calibrationPoseMask = _dilate_mask(calibrationPoseMask, preFrames=6, postFrames=6)
    calibrationPoseMask = calibrationPoseMask.astype(np.float32)

    return {
        "t_pose_mask": tPoseMask.astype(np.float32),
        "a_pose_mask": aPoseMask.astype(np.float32),
        "calibration_pose_mask": calibrationPoseMask.astype(np.float32),
        "left_elbow_angle": leftElbowAngle.astype(np.float32),
        "right_elbow_angle": rightElbowAngle.astype(np.float32),
        "left_arm_abduction": leftArmAbduction.astype(np.float32),
        "right_arm_abduction": rightArmAbduction.astype(np.float32),
    }


def BuildLabelFeatureSource(
    clipNameOrPath: str,
    globalPositions,
    poseSource,
    rootTrajectorySource,
    contactData=None,
    terrainProvider=None,
    jointNames=None,
):
    clipName = NormalizeClipName(clipNameOrPath)

    globalPositions = np.asarray(globalPositions, dtype=np.float32)
    rootPositions = np.asarray(rootTrajectorySource["positions"], dtype=np.float32)
    rootVelocities = np.asarray(rootTrajectorySource["velocities"], dtype=np.float32)
    rootDirections = np.asarray(rootTrajectorySource["directions"], dtype=np.float32)
    globalVelocities = np.asarray(poseSource["global_velocities"], dtype=np.float32)
    dt = float(rootTrajectorySource.get("dt", 1.0 / 60.0))

    frameCount = int(len(rootPositions))
    rootHorizontalSpeed = np.linalg.norm(rootVelocities[:, [0, 2]], axis=-1).astype(np.float32)
    rootVerticalSpeed = rootVelocities[:, 1].astype(np.float32)
    rootYawRate = _compute_root_yaw_rate(rootDirections, dt)

    if terrainProvider is not None and frameCount > 0:
        terrainHeights = terrainProvider.sample_heights(rootPositions).astype(np.float32)
    else:
        terrainHeights = np.zeros((frameCount,), dtype=np.float32)

    if frameCount > 0:
        referenceRootPositions = globalPositions[:, 0] if globalPositions.shape[1] > 0 else rootPositions
        rootHeightAboveGround = (referenceRootPositions[:, 1] - terrainHeights).astype(np.float32)
    else:
        rootHeightAboveGround = np.zeros((0,), dtype=np.float32)
    hipsY = rootHeightAboveGround.copy()

    contactMasks = _resolve_contact_masks(contactData)
    if contactMasks["contact_fraction"] is None:
        contactFraction = np.zeros((frameCount,), dtype=np.float32)
        leftContact = np.zeros((frameCount,), dtype=np.float32)
        rightContact = np.zeros((frameCount,), dtype=np.float32)
    else:
        contactFraction = np.asarray(contactMasks["contact_fraction"], dtype=np.float32)
        leftContact = np.asarray(contactMasks["left_contact"], dtype=np.float32)
        rightContact = np.asarray(contactMasks["right_contact"], dtype=np.float32)

    motionEnergy, upperBodyEnergy = _compute_motion_energy(poseSource, jointNames=jointNames)
    leftFootFeatures = _compute_side_contact_features(globalPositions, globalVelocities, terrainProvider, jointNames, "Left")
    rightFootFeatures = _compute_side_contact_features(globalPositions, globalVelocities, terrainProvider, jointNames, "Right")
    groundedMask = np.maximum(leftContact, rightContact) > 0.5
    airborneMask = ~groundedMask
    localFlightRatio = _sliding_mean(airborneMask.astype(np.float32), DEFAULT_FLIGHT_RATIO_WINDOW)
    bodyHeight = _estimate_body_height(globalPositions, terrainHeights, rootHeightAboveGround, jointNames=jointNames)
    standingHipsHeight = _estimate_standing_hips_height(
        rootHeightAboveGround,
        groundedMask,
        rootHorizontalSpeed,
        motionEnergy,
    )
    groundY = _estimate_ground_height(leftFootFeatures["height"], rightFootFeatures["height"])
    torsoHeight = _compute_torso_height(globalPositions, jointNames=jointNames)
    standingHeight = _estimate_standing_height(rootHeightAboveGround)
    lowHeightThreshold = _estimate_low_height_threshold(standingHeight)
    standingTorsoHeight = _estimate_standing_height(torsoHeight)
    calibrationFeatures = _compute_calibration_pose_masks(
        globalPositions,
        jointNames,
        bodyHeight,
        rootHorizontalSpeed,
        motionEnergy,
        upperBodyEnergy,
    )

    leftHipIndex = _find_joint_index(jointNames, "LeftUpLeg")
    leftKneeIndex = _find_joint_index(jointNames, "LeftLeg")
    leftAnkleIndex = _find_joint_index(jointNames, "LeftFoot")
    rightHipIndex = _find_joint_index(jointNames, "RightUpLeg")
    rightKneeIndex = _find_joint_index(jointNames, "RightLeg")
    rightAnkleIndex = _find_joint_index(jointNames, "RightFoot")
    if None in (leftHipIndex, leftKneeIndex, leftAnkleIndex):
        leftKneeAngle = np.full((frameCount,), 180.0, dtype=np.float32)
    else:
        leftKneeAngle = _joint_angle_degrees(
            globalPositions[:, leftHipIndex],
            globalPositions[:, leftKneeIndex],
            globalPositions[:, leftAnkleIndex],
        ).astype(np.float32)
    if None in (rightHipIndex, rightKneeIndex, rightAnkleIndex):
        rightKneeAngle = np.full((frameCount,), 180.0, dtype=np.float32)
    else:
        rightKneeAngle = _joint_angle_degrees(
            globalPositions[:, rightHipIndex],
            globalPositions[:, rightKneeIndex],
            globalPositions[:, rightAnkleIndex],
        ).astype(np.float32)
    kneeFlexion = (180.0 - 0.5 * (leftKneeAngle + rightKneeAngle)).astype(np.float32)

    return {
        "clip_name": clipName,
        "frame_count": frameCount,
        "dt": dt,
        "root_positions": rootPositions,
        "root_velocities": rootVelocities,
        "root_directions": rootDirections,
        "root_horizontal_speed": rootHorizontalSpeed,
        "root_vertical_speed": rootVerticalSpeed,
        "root_yaw_rate": rootYawRate,
        "body_height": float(bodyHeight),
        "ground_y": float(groundY),
        "hips_y": hipsY,
        "standing_hips_height": float(standingHipsHeight),
        "root_height_above_ground": rootHeightAboveGround,
        "standing_height": float(standingHeight),
        "low_height_threshold": float(lowHeightThreshold),
        "left_contact": leftContact,
        "right_contact": rightContact,
        "contact_fraction": contactFraction,
        "grounded": groundedMask.astype(np.float32),
        "airborne": airborneMask.astype(np.float32),
        "local_flight_ratio": localFlightRatio.astype(np.float32),
        "left_foot_height": leftFootFeatures["height"],
        "right_foot_height": rightFootFeatures["height"],
        "left_foot_speed_xy": leftFootFeatures["speed_xy"],
        "right_foot_speed_xy": rightFootFeatures["speed_xy"],
        "motion_energy": motionEnergy,
        "upper_body_energy": upperBodyEnergy,
        "t_pose_mask": calibrationFeatures["t_pose_mask"],
        "a_pose_mask": calibrationFeatures["a_pose_mask"],
        "calibration_pose_mask": calibrationFeatures["calibration_pose_mask"],
        "left_elbow_angle": calibrationFeatures["left_elbow_angle"],
        "right_elbow_angle": calibrationFeatures["right_elbow_angle"],
        "left_arm_abduction": calibrationFeatures["left_arm_abduction"],
        "right_arm_abduction": calibrationFeatures["right_arm_abduction"],
        "left_knee_angle": leftKneeAngle,
        "right_knee_angle": rightKneeAngle,
        "knee_flexion": kneeFlexion,
        "torso_height": torsoHeight,
        "standing_torso_height": float(standingTorsoHeight),
        "terrain_heights": terrainHeights,
    }


def _iter_true_segments(mask):
    mask = np.asarray(mask, dtype=bool)
    start = None
    for frameIndex, active in enumerate(mask):
        if active and start is None:
            start = frameIndex
        elif not active and start is not None:
            yield start, frameIndex - 1
            start = None
    if start is not None:
        yield start, len(mask) - 1


def _dilate_mask(mask, preFrames=0, postFrames=0):
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return mask.copy()

    result = np.zeros_like(mask, dtype=bool)
    for startFrame, endFrame in _iter_true_segments(mask):
        startFrame = max(0, int(startFrame) - int(preFrames))
        endFrame = min(len(mask) - 1, int(endFrame) + int(postFrames))
        result[startFrame:endFrame + 1] = True
    return result


def _estimate_run_speed_threshold(speedValues, defaultThreshold=RUN_SPEED_THRESHOLD):
    speedValues = np.asarray(speedValues, dtype=np.float32).reshape(-1)
    speedValues = speedValues[np.isfinite(speedValues)]
    if speedValues.size < 32:
        return float(defaultThreshold)

    lowCenter = float(np.percentile(speedValues, 35))
    highCenter = float(np.percentile(speedValues, 75))
    if highCenter - lowCenter < 1e-4:
        return float(max(defaultThreshold, np.percentile(speedValues, 75)))

    for _ in range(8):
        lowMask = np.abs(speedValues - lowCenter) <= np.abs(speedValues - highCenter)
        highMask = ~lowMask
        if not np.any(lowMask) or not np.any(highMask):
            break
        lowCenter = float(np.mean(speedValues[lowMask]))
        highCenter = float(np.mean(speedValues[highMask]))

    if highCenter < lowCenter:
        lowCenter, highCenter = highCenter, lowCenter
    if highCenter - lowCenter < 0.10:
        return float(max(defaultThreshold, np.percentile(speedValues, 80)))
    return float(max(defaultThreshold, 0.5 * (lowCenter + highCenter)))


def _compute_step_like_signal(leftContact, rightContact):
    leftContact = np.asarray(leftContact, dtype=np.float32)
    rightContact = np.asarray(rightContact, dtype=np.float32)
    contactBalance = np.abs(leftContact - rightContact).astype(np.float32)
    return _sliding_mean(contactBalance, windowSize=9)



def _build_segment_support(labels):
    labels = np.asarray(labels, dtype=object)
    support = np.zeros((len(labels),), dtype=np.float32)
    if len(labels) == 0:
        return support

    for segment in _labels_to_segments(labels, source="auto"):
        segmentLength = segment.end_frame - segment.start_frame + 1
        segmentSupport = np.clip((float(segmentLength) - 3.0) / 24.0, 0.0, 1.0)
        support[segment.start_frame:segment.end_frame + 1] = segmentSupport
    return support.astype(np.float32)


def _postprocess_auto_confidence(labels, rawLabels, rawConfidence, scoreMargins):
    labels = np.asarray(labels, dtype=object)
    rawLabels = np.asarray(rawLabels, dtype=object)
    rawConfidence = np.asarray(rawConfidence, dtype=np.float32).reshape(-1)
    if len(labels) == 0:
        return np.zeros((0,), dtype=np.float32)

    scoreMargins = np.asarray(scoreMargins, dtype=np.float32).reshape(-1)
    segmentSupport = _build_segment_support(labels)
    marginSupport = np.ones((len(labels),), dtype=np.float32)
    finiteMask = np.isfinite(scoreMargins)
    if np.any(finiteMask):
        marginSupport[finiteMask] = np.clip(
            1.0 - scoreMargins[finiteMask] / max(float(DEFAULT_TRANSITION_MAX_SCORE_MARGIN), 1e-6),
            0.0,
            1.0,
        )

    confidence = np.zeros((len(labels),), dtype=np.float32)
    for frameIndex, label in enumerate(labels):
        if label == rawLabels[frameIndex]:
            confidence[frameIndex] = float(rawConfidence[frameIndex])
            continue

        if label == LABEL_TRANSITION:
            confidence[frameIndex] = (
                0.35 +
                0.20 * float(segmentSupport[frameIndex]) +
                0.25 * float(marginSupport[frameIndex])
            )
        else:
            confidence[frameIndex] = (
                0.42 +
                0.18 * float(segmentSupport[frameIndex]) +
                0.20 * float(marginSupport[frameIndex])
            )

    return np.clip(confidence, 0.0, 1.0).astype(np.float32)


def _score_idle(featureSource):
    frameCount = int(featureSource["frame_count"])
    if frameCount == 0:
        return np.zeros((0,), dtype=np.float32)

    dt = float(featureSource.get("dt", 1.0 / 60.0))
    grounded = np.asarray(featureSource.get("grounded", np.zeros((frameCount,), dtype=np.float32)), dtype=np.float32)
    speed = np.asarray(featureSource["root_horizontal_speed"], dtype=np.float32)
    verticalSpeed = np.asarray(featureSource["root_vertical_speed"], dtype=np.float32)
    hipsY = np.asarray(featureSource.get("hips_y", featureSource["root_height_above_ground"]), dtype=np.float32)
    standingHipsHeight = max(float(featureSource.get("standing_hips_height", featureSource.get("standing_height", 1.0))), 1e-3)
    bodyHeight = max(float(featureSource.get("body_height", 1.0)), 1e-3)
    torsoHeight = np.asarray(featureSource.get("torso_height", np.full((frameCount,), 1.0, dtype=np.float32)), dtype=np.float32)
    standingTorsoHeight = max(float(featureSource.get("standing_torso_height", 1.0)), 1e-3)
    calibrationPoseMask = np.asarray(featureSource.get("calibration_pose_mask", np.zeros((frameCount,), dtype=np.float32)), dtype=np.float32)

    groundWindow = _seconds_to_frames(dt, 0.25, minimum=3)
    smoothedGrounded = _sliding_mean(grounded, groundWindow)
    groundedScore = np.clip(smoothedGrounded / 0.7, 0.0, 1.0)
    idleSpeedComfort = max(float(IDLE_SPEED_THRESHOLD), 0.18 * bodyHeight)
    idleSpeedFadeEnd = max(idleSpeedComfort + 1e-6, 0.32 * bodyHeight)
    speedFade = np.clip((speed - idleSpeedComfort) / max(idleSpeedFadeEnd - idleSpeedComfort, 1e-6), 0.0, 1.0)
    speedFade = speedFade * speedFade * (3.0 - 2.0 * speedFade)
    speedScore = 1.0 - speedFade
    heightScore = np.clip((hipsY / max(standingHipsHeight, 1e-6) - 0.75) / 0.25, 0.0, 1.0)
    torsoScore = np.clip((torsoHeight / max(standingTorsoHeight, 1e-6) - 0.82) / 0.18, 0.0, 1.0)
    postureScore = heightScore * torsoScore
    verticalStabilityScore = 1.0 - np.clip(np.abs(verticalSpeed) / max(0.25 * bodyHeight, 1e-6), 0.0, 1.0)
    calibrationPenalty = 1.0 - 0.15 * np.clip(calibrationPoseMask, 0.0, 1.0)

    return (groundedScore * speedScore * postureScore * verticalStabilityScore * calibrationPenalty).astype(np.float32)


def _score_walk(featureSource):
    frameCount = int(featureSource["frame_count"])
    if frameCount == 0:
        return np.zeros((0,), dtype=np.float32)

    speed = np.asarray(featureSource["root_horizontal_speed"], dtype=np.float32)
    leftContact = np.asarray(featureSource["left_contact"], dtype=np.float32)
    rightContact = np.asarray(featureSource["right_contact"], dtype=np.float32)
    contactFraction = np.asarray(featureSource["contact_fraction"], dtype=np.float32)
    localFlightRatio = np.asarray(featureSource.get("local_flight_ratio", np.zeros((frameCount,), dtype=np.float32)), dtype=np.float32)

    stepSignal = _compute_step_like_signal(leftContact, rightContact)
    walkMinSpeed = max(0.12, float(np.percentile(speed, 20.0)))

    speedGate = np.clip((speed - 0.2 * walkMinSpeed) / max(walkMinSpeed, 1e-6), 0.0, 1.0)
    stepGate = np.clip((stepSignal - 0.08) / 0.20, 0.0, 1.0)

    locomotionMask = (speed >= walkMinSpeed) & (stepSignal >= 0.18)
    runThreshold = _estimate_run_speed_threshold(speed[locomotionMask], defaultThreshold=RUN_SPEED_THRESHOLD)

    walkBand = np.clip((runThreshold - speed) / max(runThreshold - walkMinSpeed, 1e-6), 0.0, 1.0)
    contactScore = np.clip((contactFraction - 0.15) / 0.35, 0.0, 1.0)
    flightPenalty = 1.0 - np.clip((localFlightRatio - 0.05) / 0.15, 0.0, 1.0)

    locomotionScore = speedGate * stepGate
    return (
        0.35 * locomotionScore + 0.35 * walkBand + 0.15 * contactScore + 0.15 * flightPenalty
    ).astype(np.float32)


def _score_run(featureSource):
    frameCount = int(featureSource["frame_count"])
    if frameCount == 0:
        return np.zeros((0,), dtype=np.float32)

    speed = np.asarray(featureSource["root_horizontal_speed"], dtype=np.float32)
    leftContact = np.asarray(featureSource["left_contact"], dtype=np.float32)
    rightContact = np.asarray(featureSource["right_contact"], dtype=np.float32)
    contactFraction = np.asarray(featureSource["contact_fraction"], dtype=np.float32)
    localFlightRatio = np.asarray(featureSource.get("local_flight_ratio", np.zeros((frameCount,), dtype=np.float32)), dtype=np.float32)
    motionEnergy = np.asarray(featureSource["motion_energy"], dtype=np.float32)

    stepSignal = _compute_step_like_signal(leftContact, rightContact)
    walkMinSpeed = max(0.12, float(np.percentile(speed, 20.0)))

    speedGate = np.clip((speed - 0.2 * walkMinSpeed) / max(walkMinSpeed, 1e-6), 0.0, 1.0)
    stepGate = np.clip((stepSignal - 0.08) / 0.20, 0.0, 1.0)

    locomotionMask = (speed >= walkMinSpeed) & (stepSignal >= 0.18)
    runThreshold = _estimate_run_speed_threshold(speed[locomotionMask], defaultThreshold=RUN_SPEED_THRESHOLD)

    runDominantMotion = (
        float(np.percentile(speed, 65)) >= RUN_SPEED_THRESHOLD + 0.20
        and float(np.percentile(localFlightRatio, 60)) >= 0.12
        and float(np.percentile(contactFraction, 50)) <= 0.50
    )
    if runDominantMotion:
        runThreshold = min(runThreshold, max(1.10, float(np.percentile(speed, 25))))

    runSpeedMargin = max(0.10, 0.10 * runThreshold)
    runBand = np.clip((speed - runThreshold) / max(runSpeedMargin, 1e-6), 0.0, 1.0)

    flightScore = np.clip((localFlightRatio - 0.05) / 0.20, 0.0, 1.0)
    lowContactScore = np.clip((0.40 - contactFraction) / 0.30, 0.0, 1.0)
    runStyleScore = np.maximum(flightScore, lowContactScore)

    motionEnergyMid = float(np.percentile(motionEnergy, 50))
    motionEnergyHigh = float(np.percentile(motionEnergy, 85))
    energyScore = np.clip(
        (motionEnergy - motionEnergyMid) / max(motionEnergyHigh - motionEnergyMid, 1e-6),
        0.0,
        1.0,
    )

    locomotionScore = speedGate * stepGate
    energyFactor = np.maximum(runStyleScore, 0.3 * energyScore)
    return (
        0.30 * locomotionScore + 0.30 * runBand + 0.25 * runStyleScore + 0.15 * energyScore
    ).astype(np.float32)


def _score_jump(featureSource):
    frameCount = int(featureSource["frame_count"])
    if frameCount == 0:
        return np.zeros((0,), dtype=np.float32)

    bodyHeight = max(float(featureSource.get("body_height", 1.0)), 1e-3)
    contactFraction = np.asarray(featureSource["contact_fraction"], dtype=np.float32)
    localFlightRatio = np.asarray(featureSource.get("local_flight_ratio", np.zeros((frameCount,), dtype=np.float32)), dtype=np.float32)
    verticalSpeed = np.asarray(featureSource["root_vertical_speed"], dtype=np.float32)
    hipsY = np.asarray(featureSource.get("hips_y", featureSource["root_height_above_ground"]), dtype=np.float32)
    standingHeight = max(float(featureSource.get("standing_height", 1.0)), 1e-3)
    speed = np.asarray(featureSource["root_horizontal_speed"], dtype=np.float32)

    flightScore = np.clip(localFlightRatio / 0.08, 0.0, 1.0)
    lowContactScore = np.clip((0.40 - contactFraction) / 0.30, 0.0, 1.0)
    airborneScore = 0.70 * flightScore + 0.30 * lowContactScore

    vyUpScore = np.clip(verticalSpeed / max(0.50 * bodyHeight, 1e-6), 0.0, 1.0)
    vyDownScore = np.clip(np.abs(verticalSpeed) / max(0.80 * bodyHeight, 1e-6), 0.0, 1.0)
    liftScore = np.clip((hipsY - standingHeight) / max(0.15 * bodyHeight, 1e-6), 0.0, 1.0)
    verticalScore = np.maximum(vyUpScore, np.maximum(vyDownScore * 0.4, liftScore))

    fastRunPenalty = np.where(
        (speed >= RUN_SPEED_THRESHOLD + 0.30) & (liftScore < 0.30) & (vyUpScore < 0.50),
        0.15,
        1.0,
    )

    rawScore = 0.50 * airborneScore + 0.35 * verticalScore + 0.15 * flightScore
    return (rawScore * fastRunPenalty).astype(np.float32)


def _score_crouch(featureSource):
    frameCount = int(featureSource["frame_count"])
    if frameCount == 0:
        return np.zeros((0,), dtype=np.float32)

    contactFraction = np.asarray(featureSource["contact_fraction"], dtype=np.float32)
    localFlightRatio = np.asarray(featureSource.get("local_flight_ratio", np.zeros((frameCount,), dtype=np.float32)), dtype=np.float32)
    verticalSpeed = np.asarray(featureSource["root_vertical_speed"], dtype=np.float32)
    speed = np.asarray(featureSource["root_horizontal_speed"], dtype=np.float32)
    torsoHeight = np.asarray(featureSource.get("torso_height", np.zeros((frameCount,), dtype=np.float32)), dtype=np.float32)
    standingTorsoHeight = max(float(featureSource.get("standing_torso_height", 1.0)), 1e-3)
    kneeFlexion = np.asarray(featureSource.get("knee_flexion", np.zeros((frameCount,), dtype=np.float32)), dtype=np.float32)

    torsoRatio = torsoHeight / max(standingTorsoHeight, 1e-6)

    deepTorsoScore = np.clip((0.83 - torsoRatio) / 0.13, 0.0, 1.0)
    locoTorsoScore = np.clip((0.90 - torsoRatio) / 0.08, 0.0, 1.0)
    kneeScore = np.clip((kneeFlexion - 40.0) / 35.0, 0.0, 1.0)
    torsoGate = np.maximum(deepTorsoScore, locoTorsoScore * kneeScore)

    groundedScore = np.clip(contactFraction / 0.50, 0.0, 1.0)
    stableScore = 1.0 - np.clip(np.abs(verticalSpeed) / 0.30, 0.0, 1.0)
    lowFlightScore = 1.0 - np.clip(localFlightRatio / 0.08, 0.0, 1.0)
    speedGate = 1.0 - np.clip(speed / 1.30, 0.0, 1.0)

    crouchBase = 0.35 * groundedScore + 0.25 * stableScore + 0.20 * lowFlightScore
    return (torsoGate * speedGate * crouchBase).astype(np.float32)


def _build_leading_calibration_mask(featureSource, postSeconds=DEFAULT_LEADING_CALIBRATION_SETTLE_SECONDS):
    frameCount = int(featureSource["frame_count"])
    result = np.zeros((frameCount,), dtype=bool)
    if frameCount == 0:
        return result

    calibrationMask = np.asarray(
        featureSource.get("calibration_pose_mask", np.zeros((frameCount,), dtype=np.float32)),
        dtype=np.float32,
    ) > 0.5
    if not calibrationMask[0]:
        return result

    endFrame = 0
    while endFrame + 1 < frameCount and calibrationMask[endFrame + 1]:
        endFrame += 1

    dt = float(featureSource.get("dt", 1.0 / 60.0))
    endFrame = min(frameCount - 1, endFrame + _seconds_to_frames(dt, postSeconds, minimum=0))
    result[:endFrame + 1] = True
    return result


def _fuse_parallel_scores(idleScore, walkScore, runScore, jumpScore, crouchScore, forceOtherMask=None):
    frameCount = len(idleScore)
    actionScores = np.column_stack([idleScore, walkScore, runScore, jumpScore, crouchScore])
    otherScore = np.full(frameCount, 0.10, dtype=np.float32)

    scores = np.zeros((frameCount, len(ACTION_LABELS)), dtype=np.float32)
    scores[:, LABEL_TO_INDEX[LABEL_IDLE]] = idleScore
    scores[:, LABEL_TO_INDEX[LABEL_WALK]] = walkScore
    scores[:, LABEL_TO_INDEX[LABEL_RUN]] = runScore
    scores[:, LABEL_TO_INDEX[LABEL_JUMP]] = jumpScore
    scores[:, LABEL_TO_INDEX[LABEL_CROUCH]] = crouchScore
    scores[:, LABEL_TO_INDEX[LABEL_OTHER]] = otherScore

    if forceOtherMask is not None:
        forceOtherMask = np.asarray(forceOtherMask, dtype=bool)
        scores[forceOtherMask, :] = 0.0
        scores[forceOtherMask, LABEL_TO_INDEX[LABEL_OTHER]] = 0.95

    labels = _labels_from_scores(scores)
    confidence = np.max(scores, axis=1)
    return labels.astype(object), scores.astype(np.float32), confidence.astype(np.float32)


def BuildAutoLabelsFromMotion(featureSource, params=None):
    params = CoerceLabelAutoParams(params)
    frameCount = int(featureSource["frame_count"])
    if frameCount == 0:
        labels = np.full((0,), LABEL_OTHER, dtype=object)
        scores = np.zeros((0, len(ACTION_LABELS)), dtype=np.float32)
        confidence = np.full((0,), 0.55, dtype=np.float32)
        return labels, scores, confidence

    idleScore = _score_idle(featureSource)
    walkScore = _score_walk(featureSource)
    runScore = _score_run(featureSource)
    jumpScore = _score_jump(featureSource)
    crouchScore = _score_crouch(featureSource)
    leadingCalibrationMask = _build_leading_calibration_mask(featureSource)
    featureSource["leading_calibration_mask"] = leadingCalibrationMask.astype(np.float32)

    return _fuse_parallel_scores(
        idleScore,
        walkScore,
        runScore,
        jumpScore,
        crouchScore,
        forceOtherMask=leadingCalibrationMask,
    )


def BuildAutoLabelScores(featureSource, params=None) -> np.ndarray:
    params = CoerceLabelAutoParams(params)
    labels, scores, confidence = BuildAutoLabelsFromMotion(featureSource, params=params)
    featureSource["motion_rule_labels"] = np.asarray(labels, dtype=object)
    featureSource["auto_confidence"] = np.asarray(confidence, dtype=np.float32)
    featureSource["auto_params"] = params
    return scores.astype(np.float32)


def _labels_from_scores(scores):
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D array.")
    return np.asarray([ACTION_LABELS[index] for index in np.argmax(scores, axis=1)], dtype=object)


def _majority_filter_labels(labels, windowSize=7):
    labels = np.asarray(labels, dtype=object)
    if len(labels) == 0 or windowSize <= 1:
        return labels.copy()

    radius = max(0, int(windowSize) // 2)
    filtered = labels.copy()

    for frameIndex in range(len(labels)):
        start = max(0, frameIndex - radius)
        end = min(len(labels), frameIndex + radius + 1)
        window = labels[start:end]
        bestLabel = filtered[frameIndex]
        bestCount = -1
        for label in ACTION_LABELS:
            count = int(np.sum(window == label))
            if count > bestCount:
                bestCount = count
                bestLabel = label
        filtered[frameIndex] = bestLabel

    return filtered


def _labels_to_segments(labels, source):
    labels = np.asarray(labels, dtype=object)
    if len(labels) == 0:
        return []

    segments = []
    currentLabel = None if labels[0] is None else str(labels[0])
    startFrame = 0 if currentLabel is not None else None

    for frameIndex, label in enumerate(labels[1:], start=1):
        label = None if label is None else str(label)
        if label == currentLabel:
            continue

        if currentLabel is not None and startFrame is not None:
            segments.append(LabelSegment(startFrame, frameIndex - 1, currentLabel, source=source))

        currentLabel = label
        startFrame = frameIndex if label is not None else None

    if currentLabel is not None and startFrame is not None:
        segments.append(LabelSegment(startFrame, len(labels) - 1, currentLabel, source=source))

    return segments


def GetDefaultAnnotationPath(clipNameOrPath: str, annotationsRoot="resources/annotations") -> Path:
    clipPath = Path(str(clipNameOrPath))
    clipStem = clipPath.stem
    clipParents = list(clipPath.parent.parts)
    if clipParents and clipParents[0] == "bvh":
        clipParents = clipParents[1:]
    return Path(annotationsRoot).joinpath(*clipParents, f"{clipStem}.json")


def _min_segment_length_for_label(label, defaultMinSegmentLength, dt=1.0 / 60.0):
    label = str(label)
    perLabelMinimums = {
        LABEL_IDLE: _seconds_to_frames(dt, DEFAULT_IDLE_MIN_SECONDS),
        LABEL_WALK: _seconds_to_frames(dt, DEFAULT_WALK_MIN_SECONDS),
        LABEL_RUN: _seconds_to_frames(dt, DEFAULT_RUN_MIN_SECONDS),
        LABEL_JUMP: _seconds_to_frames(dt, DEFAULT_JUMP_MIN_SECONDS),
        LABEL_CROUCH: _seconds_to_frames(dt, DEFAULT_CROUCH_MIN_SECONDS),
        LABEL_OTHER: _seconds_to_frames(dt, DEFAULT_OTHER_MIN_SECONDS),
        LABEL_FALL: _seconds_to_frames(dt, DEFAULT_OTHER_MIN_SECONDS),
        LABEL_GROUND: _seconds_to_frames(dt, DEFAULT_OTHER_MIN_SECONDS),
        LABEL_GET_UP: _seconds_to_frames(dt, DEFAULT_OTHER_MIN_SECONDS),
    }
    return int(max(int(defaultMinSegmentLength), perLabelMinimums.get(label, int(defaultMinSegmentLength))))


def _merge_short_segments(labels, minSegmentLength=6, dt=1.0 / 60.0):
    labels = np.asarray(labels, dtype=object)
    if len(labels) == 0 or minSegmentLength <= 1:
        return labels.copy()

    merged = labels.copy()
    changed = True

    while changed:
        changed = False
        segments = _labels_to_segments(merged, source="auto")
        for segmentIndex, segment in enumerate(segments):
            segmentLength = segment.end_frame - segment.start_frame + 1
            requiredLength = _min_segment_length_for_label(segment.label, minSegmentLength, dt=dt)
            if segmentLength >= requiredLength:
                continue

            leftLabel = segments[segmentIndex - 1].label if segmentIndex > 0 else None
            rightLabel = segments[segmentIndex + 1].label if segmentIndex + 1 < len(segments) else None

            if leftLabel is None and rightLabel is None:
                continue
            if leftLabel == rightLabel and leftLabel is not None:
                targetLabel = leftLabel
            elif rightLabel is None:
                targetLabel = leftLabel
            elif leftLabel is None:
                targetLabel = rightLabel
            else:
                leftLength = segments[segmentIndex - 1].end_frame - segments[segmentIndex - 1].start_frame + 1
                rightLength = segments[segmentIndex + 1].end_frame - segments[segmentIndex + 1].start_frame + 1
                targetLabel = leftLabel if leftLength >= rightLength else rightLabel

            if targetLabel is None:
                continue

            merged[segment.start_frame:segment.end_frame + 1] = targetLabel
            changed = True
            break

    return merged


def _insert_transition_labels(
    labels,
    transitionFrames=DEFAULT_TRANSITION_FRAMES,
    scoreMargins=None,
    minSegmentLength=DEFAULT_TRANSITION_MIN_SEGMENT_LENGTH,
    maxScoreMargin=DEFAULT_TRANSITION_MAX_SCORE_MARGIN,
):
    labels = np.asarray(labels, dtype=object)
    if len(labels) == 0 or transitionFrames <= 0:
        return labels.copy()

    result = labels.copy()
    halfWidth = max(1, int(transitionFrames) // 2)
    segments = _labels_to_segments(labels, source="auto")
    scoreMargins = None if scoreMargins is None else np.asarray(scoreMargins, dtype=np.float32)

    for segmentIndex in range(len(segments) - 1):
        currentSegment = segments[segmentIndex]
        nextSegment = segments[segmentIndex + 1]
        if currentSegment.label == nextSegment.label:
            continue
        if LABEL_TRANSITION in (currentSegment.label, nextSegment.label):
            continue
        if LABEL_OTHER in (currentSegment.label, nextSegment.label):
            continue

        leftBudget = max(0, currentSegment.end_frame - currentSegment.start_frame + 1)
        rightBudget = max(0, nextSegment.end_frame - nextSegment.start_frame + 1)
        if leftBudget < int(minSegmentLength) or rightBudget < int(minSegmentLength):
            continue

        if scoreMargins is not None:
            boundaryMargin = min(
                float(scoreMargins[currentSegment.end_frame]),
                float(scoreMargins[nextSegment.start_frame]),
            )
            if boundaryMargin > float(maxScoreMargin):
                continue

        leftCount = min(halfWidth, max(1, leftBudget // 3))
        rightCount = min(halfWidth, max(1, rightBudget // 3))

        startFrame = max(currentSegment.start_frame, currentSegment.end_frame - leftCount + 1)
        endFrame = min(nextSegment.end_frame, nextSegment.start_frame + rightCount - 1)
        result[startFrame:endFrame + 1] = LABEL_TRANSITION

    return result


def _ensure_manual_labels(labelResult: LabelModuleResult) -> np.ndarray:
    if labelResult.auto_labels is None:
        raise ValueError("auto_labels must be available before manual editing.")

    frameCount = len(labelResult.auto_labels)
    if labelResult.manual_labels is None or len(labelResult.manual_labels) != frameCount:
        labelResult.manual_labels = np.full((frameCount,), None, dtype=object)
    return labelResult.manual_labels


def _clone_transition_overrides(transitionOverrides):
    return [
        {
            "start_frame": int(override["start_frame"]),
            "end_frame": int(override["end_frame"]),
            "width": int(override["width"]),
        }
        for override in transitionOverrides
    ]


def _snapshot_annotation_state(labelResult: LabelModuleResult) -> dict:
    manualLabels = _ensure_manual_labels(labelResult)
    return {
        "manual_labels": manualLabels.copy(),
        "transition_overrides": _clone_transition_overrides(labelResult.transition_overrides),
    }


def _annotation_state_equal(leftSnapshot, rightSnapshot) -> bool:
    return (
        np.array_equal(
            np.asarray(leftSnapshot["manual_labels"], dtype=object),
            np.asarray(rightSnapshot["manual_labels"], dtype=object),
        ) and
        list(leftSnapshot["transition_overrides"]) == list(rightSnapshot["transition_overrides"])
    )


def _push_annotation_history(labelResult: LabelModuleResult) -> None:
    snapshot = _snapshot_annotation_state(labelResult)
    if labelResult.undo_stack and _annotation_state_equal(labelResult.undo_stack[-1], snapshot):
        labelResult.redo_stack.clear()
        return

    labelResult.undo_stack.append(snapshot)
    if len(labelResult.undo_stack) > MAX_LABEL_HISTORY:
        labelResult.undo_stack = labelResult.undo_stack[-MAX_LABEL_HISTORY:]
    labelResult.redo_stack.clear()


def _restore_annotation_state(labelResult: LabelModuleResult, snapshot) -> LabelModuleResult:
    manualLabels = _ensure_manual_labels(labelResult)
    manualLabels[:] = np.asarray(snapshot["manual_labels"], dtype=object)
    labelResult.transition_overrides = _clone_transition_overrides(snapshot["transition_overrides"])
    return _rebuild_final_labels(labelResult)


def CanUndoLabelEdit(labelResult: LabelModuleResult) -> bool:
    return bool(labelResult.undo_stack)


def CanRedoLabelEdit(labelResult: LabelModuleResult) -> bool:
    return bool(labelResult.redo_stack)


def UndoLabelEdit(labelResult: LabelModuleResult) -> bool:
    if not labelResult.undo_stack:
        return False

    currentSnapshot = _snapshot_annotation_state(labelResult)
    targetSnapshot = labelResult.undo_stack.pop()
    labelResult.redo_stack.append(currentSnapshot)
    if len(labelResult.redo_stack) > MAX_LABEL_HISTORY:
        labelResult.redo_stack = labelResult.redo_stack[-MAX_LABEL_HISTORY:]
    _restore_annotation_state(labelResult, targetSnapshot)
    return True


def RedoLabelEdit(labelResult: LabelModuleResult) -> bool:
    if not labelResult.redo_stack:
        return False

    currentSnapshot = _snapshot_annotation_state(labelResult)
    targetSnapshot = labelResult.redo_stack.pop()
    labelResult.undo_stack.append(currentSnapshot)
    if len(labelResult.undo_stack) > MAX_LABEL_HISTORY:
        labelResult.undo_stack = labelResult.undo_stack[-MAX_LABEL_HISTORY:]
    _restore_annotation_state(labelResult, targetSnapshot)
    return True


def _ease_cosine(alpha):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return 0.5 - 0.5 * np.cos(np.pi * alpha)


def _assign_soft_weight_row(weights, frameIndex, weightMap):
    row = np.zeros((len(ACTION_LABELS),), dtype=np.float32)
    for label, value in weightMap.items():
        if label not in LABEL_TO_INDEX:
            continue
        row[LABEL_TO_INDEX[label]] = max(0.0, float(value))

    rowSum = float(np.sum(row))
    if rowSum <= 1e-8:
        row[LABEL_TO_INDEX[LABEL_OTHER]] = 1.0
    else:
        row /= rowSum

    weights[frameIndex] = row


def _find_neighbor_segment(segments, startIndex, step):
    index = startIndex + step
    while 0 <= index < len(segments):
        if segments[index].label != LABEL_TRANSITION:
            return segments[index]
        index += step
    return None


def _normalize_transition_override(startFrame, endFrame, width):
    startFrame = int(min(startFrame, endFrame))
    endFrame = int(max(startFrame, endFrame))
    width = max(0, int(width))
    return {
        "start_frame": startFrame,
        "end_frame": endFrame,
        "width": width,
    }


def _transition_width_for_range(transitionOverrides, startFrame, endFrame, defaultWidth=DEFAULT_TRANSITION_FRAMES):
    width = int(defaultWidth)
    if not transitionOverrides:
        return width

    startFrame = int(startFrame)
    endFrame = int(endFrame)
    for override in transitionOverrides:
        overrideStart = int(override["start_frame"])
        overrideEnd = int(override["end_frame"])
        if overrideEnd < startFrame or overrideStart > endFrame:
            continue
        width = int(override["width"])
    return max(0, width)


def _build_soft_weights_from_labels_with_overrides(labels, transitionOverrides=None):
    labels = np.asarray(labels, dtype=object)
    frameCount = len(labels)
    weights = CreateEmptySoftWeights(frameCount, fillLabel=LABEL_OTHER)
    if frameCount == 0:
        return weights

    weights[:] = 0.0
    hardLabelIndices = np.asarray([LABEL_TO_INDEX[str(label)] for label in labels], dtype=np.int32)
    weights[np.arange(frameCount), hardLabelIndices] = 1.0

    segments = _labels_to_segments(labels, source="compiled")

    for segmentIndex, segment in enumerate(segments):
        if segment.label != LABEL_TRANSITION:
            continue

        transitionWidth = _transition_width_for_range(
            transitionOverrides,
            segment.start_frame,
            segment.end_frame,
            defaultWidth=max(1, segment.end_frame - segment.start_frame + 1),
        )
        if transitionWidth <= 0:
            continue

        previousSegment = _find_neighbor_segment(segments, segmentIndex, -1)
        nextSegment = _find_neighbor_segment(segments, segmentIndex, 1)
        segmentLength = segment.end_frame - segment.start_frame + 1

        for localIndex, frameIndex in enumerate(range(segment.start_frame, segment.end_frame + 1)):
            alpha = 0.5 if segmentLength <= 1 else localIndex / max(segmentLength - 1, 1)
            eased = _ease_cosine(alpha)
            transitionStrength = 0.45 + 0.55 * np.sin(np.pi * alpha)

            if previousSegment is not None and nextSegment is not None:
                sideStrength = max(0.0, 1.0 - transitionStrength)
                _assign_soft_weight_row(
                    weights,
                    frameIndex,
                    {
                        previousSegment.label: (1.0 - eased) * sideStrength,
                        nextSegment.label: eased * sideStrength,
                        LABEL_TRANSITION: transitionStrength,
                    },
                )

    for segmentIndex in range(len(segments) - 1):
        leftSegment = segments[segmentIndex]
        rightSegment = segments[segmentIndex + 1]
        if LABEL_TRANSITION in (leftSegment.label, rightSegment.label):
            continue
        if leftSegment.label == rightSegment.label:
            continue

        boundaryStart = leftSegment.end_frame
        boundaryEnd = rightSegment.start_frame
        transitionWidth = _transition_width_for_range(
            transitionOverrides,
            boundaryStart,
            boundaryEnd,
            defaultWidth=DEFAULT_TRANSITION_FRAMES,
        )
        blendFrames = max(0, int(transitionWidth) // 2)
        blendFrames = min(blendFrames, leftSegment.end_frame - leftSegment.start_frame + 1, rightSegment.end_frame - rightSegment.start_frame + 1)
        if blendFrames <= 0:
            continue

        for offset in range(blendFrames):
            alpha = _ease_cosine((offset + 1) / (blendFrames + 1))
            leftFrame = leftSegment.end_frame - blendFrames + 1 + offset
            rightFrame = rightSegment.start_frame + offset

            _assign_soft_weight_row(
                weights,
                leftFrame,
                {
                    leftSegment.label: 1.0 - 0.5 * alpha,
                    rightSegment.label: 0.5 * alpha,
                },
            )
            _assign_soft_weight_row(
                weights,
                rightFrame,
                {
                    leftSegment.label: 0.5 * (1.0 - alpha),
                    rightSegment.label: 0.5 + 0.5 * alpha,
                },
            )

    return weights.astype(np.float32)


def _rebuild_final_labels(labelResult: LabelModuleResult) -> LabelModuleResult:
    if labelResult.auto_labels is None:
        raise ValueError("auto_labels must be available to build final labels.")

    manualLabels = _ensure_manual_labels(labelResult)
    finalLabels = np.asarray(labelResult.auto_labels, dtype=object).copy()
    manualMask = np.asarray([label is not None for label in manualLabels], dtype=bool)
    finalLabels[manualMask] = manualLabels[manualMask]

    labelResult.final_labels = finalLabels
    labelResult.final_segments = _labels_to_segments(finalLabels, source="final")
    labelResult.soft_weights = _build_soft_weights_from_labels_with_overrides(
        finalLabels,
        transitionOverrides=labelResult.transition_overrides,
    )
    return labelResult


def _segments_to_export_array(segments):
    segmentDtype = np.dtype([
        ("start_frame", np.int32),
        ("end_frame", np.int32),
        ("label", "U32"),
    ])
    exportArray = np.zeros((len(segments),), dtype=segmentDtype)
    for segmentIndex, segment in enumerate(segments):
        exportArray[segmentIndex] = (
            int(segment.start_frame),
            int(segment.end_frame),
            str(segment.label),
        )
    return exportArray


def _extract_public_soft_weights(softWeights):
    softWeights = np.asarray(softWeights, dtype=np.float32)
    if softWeights.ndim != 2:
        raise ValueError("softWeights must be a 2D array.")
    if softWeights.shape[0] == 0:
        return np.zeros((0, len(TARGET_ACTION_LABELS)), dtype=np.float32)

    targetIndices = np.asarray([LABEL_TO_INDEX[label] for label in TARGET_ACTION_LABELS], dtype=np.int32)
    publicWeights = np.asarray(softWeights[:, targetIndices], dtype=np.float32).copy()
    rowSums = np.sum(publicWeights, axis=1, keepdims=True)
    zeroMask = rowSums[:, 0] <= 1e-8
    if np.any(zeroMask):
        publicWeights[zeroMask, :] = 0.0
        publicWeights[zeroMask, TARGET_LABEL_TO_INDEX[LABEL_OTHER]] = 1.0
        rowSums = np.sum(publicWeights, axis=1, keepdims=True)
    publicWeights /= np.maximum(rowSums, 1e-8)
    return publicWeights.astype(np.float32)


def _collapse_labels_to_public(labels, internalSoftWeights=None):
    labels = np.asarray(labels, dtype=object)
    if len(labels) == 0:
        return np.asarray([], dtype=object)

    if internalSoftWeights is not None:
        publicWeights = _extract_public_soft_weights(internalSoftWeights)
        bestIndices = np.argmax(publicWeights, axis=1)
        fallbackLabels = np.asarray([TARGET_ACTION_LABELS[index] for index in bestIndices], dtype=object)
    else:
        fallbackLabels = np.full((len(labels),), LABEL_OTHER, dtype=object)

    publicLabels = fallbackLabels.copy()
    for frameIndex, label in enumerate(labels):
        label = _remap_legacy_label(label)
        if label in TARGET_ACTION_LABELS:
            publicLabels[frameIndex] = label
    return np.asarray(publicLabels, dtype=object)


def _label_auto_params_to_arrays(params):
    params = CoerceLabelAutoParams(params)
    names = []
    values = []
    for name in params.__dataclass_fields__:
        value = getattr(params, name)
        if isinstance(value, (int, float, np.integer, np.floating)):
            names.append(str(name))
            values.append(float(value))
    return np.asarray(names, dtype=np.str_), np.asarray(values, dtype=np.float32)


def ApplyManualLabelRange(labelResult: LabelModuleResult, startFrame: int, endFrame: int, label: str) -> LabelModuleResult:
    label = _remap_legacy_label(label)
    if label not in TARGET_ACTION_LABELS:
        raise ValueError(f'Unsupported action label "{label}".')

    manualLabels = _ensure_manual_labels(labelResult)
    startFrame = max(0, int(min(startFrame, endFrame)))
    endFrame = min(len(manualLabels) - 1, int(max(startFrame, endFrame)))
    if np.all(manualLabels[startFrame:endFrame + 1] == label):
        return labelResult
    _push_annotation_history(labelResult)
    manualLabels[startFrame:endFrame + 1] = label
    return _rebuild_final_labels(labelResult)


def ClearManualLabelRange(labelResult: LabelModuleResult, startFrame: int, endFrame: int) -> LabelModuleResult:
    manualLabels = _ensure_manual_labels(labelResult)
    startFrame = max(0, int(min(startFrame, endFrame)))
    endFrame = min(len(manualLabels) - 1, int(max(startFrame, endFrame)))
    if np.all(manualLabels[startFrame:endFrame + 1] == None):
        return labelResult
    _push_annotation_history(labelResult)
    manualLabels[startFrame:endFrame + 1] = None
    return _rebuild_final_labels(labelResult)


def ResetManualLabels(labelResult: LabelModuleResult) -> LabelModuleResult:
    manualLabels = _ensure_manual_labels(labelResult)
    if np.all(manualLabels == None) and not labelResult.transition_overrides:
        return labelResult
    _push_annotation_history(labelResult)
    manualLabels[:] = None
    labelResult.transition_overrides = []
    return _rebuild_final_labels(labelResult)


def ApplyTransitionWidthRange(labelResult: LabelModuleResult, startFrame: int, endFrame: int, width: int) -> LabelModuleResult:
    if labelResult.auto_labels is None:
        raise ValueError("auto_labels must be available before transition editing.")
    frameCount = len(labelResult.auto_labels)
    startFrame = max(0, min(int(startFrame), frameCount - 1))
    endFrame = max(0, min(int(endFrame), frameCount - 1))
    _push_annotation_history(labelResult)
    labelResult.transition_overrides.append(_normalize_transition_override(startFrame, endFrame, width))
    return _rebuild_final_labels(labelResult)


def ClearTransitionWidthRange(labelResult: LabelModuleResult, startFrame: int, endFrame: int) -> LabelModuleResult:
    startFrame = int(min(startFrame, endFrame))
    endFrame = int(max(startFrame, endFrame))
    retainedOverrides = [
        override for override in labelResult.transition_overrides
        if int(override["end_frame"]) < startFrame or int(override["start_frame"]) > endFrame
    ]
    if len(retainedOverrides) == len(labelResult.transition_overrides):
        return labelResult
    _push_annotation_history(labelResult)
    labelResult.transition_overrides = retainedOverrides
    return _rebuild_final_labels(labelResult)


def SaveLabelAnnotations(labelResult: LabelModuleResult, clipNameOrPath: str, annotationPath: Optional[str] = None) -> str:
    manualLabels = _ensure_manual_labels(labelResult)
    annotationFile = Path(annotationPath) if annotationPath is not None else GetDefaultAnnotationPath(clipNameOrPath)
    annotationFile.parent.mkdir(parents=True, exist_ok=True)

    manualSegments = _labels_to_segments(manualLabels, source="manual")
    payload = {
        "version": 1,
        "clip_name": NormalizeClipName(clipNameOrPath),
        "manual_overrides": [
            {
                "start_frame": int(segment.start_frame),
                "end_frame": int(segment.end_frame),
                "label": segment.label,
            }
            for segment in manualSegments
        ],
        "transition_overrides": [
            {
                "start_frame": int(override["start_frame"]),
                "end_frame": int(override["end_frame"]),
                "width": int(override["width"]),
            }
            for override in labelResult.transition_overrides
        ],
    }

    annotationFile.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    labelResult.annotation_path = str(annotationFile)
    labelResult.annotation_loaded = True
    return str(annotationFile)


def LoadLabelAnnotations(
    labelResult: LabelModuleResult,
    clipNameOrPath: str,
    annotationPath: Optional[str] = None,
    recordHistory: bool = False,
) -> bool:
    annotationFile = Path(annotationPath) if annotationPath is not None else GetDefaultAnnotationPath(clipNameOrPath)
    labelResult.annotation_path = str(annotationFile)

    if not annotationFile.is_file():
        labelResult.annotation_loaded = False
        return False

    payload = json.loads(annotationFile.read_text(encoding="utf-8"))
    overrides = list(payload.get("manual_overrides", []))
    transitionOverrides = list(payload.get("transition_overrides", []))
    if recordHistory:
        _push_annotation_history(labelResult)

    manualLabels = _ensure_manual_labels(labelResult)
    manualLabels[:] = None

    for override in overrides:
        label = _remap_legacy_label(override["label"])
        if label not in TARGET_ACTION_LABELS:
            continue
        startFrame = max(0, int(override["start_frame"]))
        endFrame = min(len(manualLabels) - 1, int(override["end_frame"]))
        if endFrame < startFrame:
            continue
        manualLabels[startFrame:endFrame + 1] = label

    labelResult.transition_overrides = []
    for override in transitionOverrides:
        labelResult.transition_overrides.append(
            _normalize_transition_override(
                int(override["start_frame"]),
                int(override["end_frame"]),
                int(override["width"]),
            )
        )

    _rebuild_final_labels(labelResult)
    labelResult.annotation_loaded = True
    return True


def GetDefaultExportPath(clipNameOrPath: str, exportsRoot="resources/exports") -> Path:
    clipPath = Path(str(clipNameOrPath))
    clipStem = clipPath.stem
    clipParents = list(clipPath.parent.parts)
    if clipParents and clipParents[0] == "bvh":
        clipParents = clipParents[1:]
    return Path(exportsRoot).joinpath(*clipParents, f"{clipStem}_labels.npz")


def ExportCompiledLabels(labelResult: LabelModuleResult, clipNameOrPath: str, exportPath: Optional[str] = None) -> str:
    if labelResult.final_labels is None or labelResult.soft_weights is None:
        _rebuild_final_labels(labelResult)

    exportFile = Path(exportPath) if exportPath is not None else GetDefaultExportPath(clipNameOrPath)
    exportFile.parent.mkdir(parents=True, exist_ok=True)

    internalFinalLabels = np.asarray(labelResult.final_labels, dtype=np.str_)
    internalFinalWeights = np.asarray(labelResult.soft_weights, dtype=np.float32)
    internalLabelIds = np.asarray([LABEL_TO_INDEX[str(label)] for label in internalFinalLabels], dtype=np.int32)

    internalAutoLabels = (
        np.asarray(labelResult.auto_labels, dtype=np.str_)
        if labelResult.auto_labels is not None else
        np.asarray([], dtype=np.str_)
    )
    internalAutoWeights = (
        _build_soft_weights_from_labels_with_overrides(internalAutoLabels)
        if internalAutoLabels.size > 0 else
        CreateEmptySoftWeights(0, fillLabel=LABEL_OTHER)
    )
    autoConfidence = (
        np.asarray(labelResult.auto_confidence, dtype=np.float32)
        if labelResult.auto_confidence is not None else
        np.asarray([], dtype=np.float32)
    )
    motionRuleLabels = (
        np.asarray(labelResult.feature_source.get("motion_rule_labels", []), dtype=np.str_)
        if isinstance(labelResult.feature_source, dict) else
        np.asarray([], dtype=np.str_)
    )

    publicAutoLabels = np.asarray(_collapse_labels_to_public(internalAutoLabels, internalAutoWeights), dtype=np.str_)
    publicFinalLabels = np.asarray(_collapse_labels_to_public(internalFinalLabels, internalFinalWeights), dtype=np.str_)
    publicSoftWeights = _extract_public_soft_weights(internalFinalWeights)
    publicLabelIds = np.asarray([TARGET_LABEL_TO_INDEX[str(label)] for label in publicFinalLabels], dtype=np.int32)
    publicSegments = _labels_to_segments(publicFinalLabels, source="public")
    autoParamNames, autoParamValues = _label_auto_params_to_arrays(labelResult.auto_params)

    np.savez_compressed(
        exportFile,
        clip_name=NormalizeClipName(clipNameOrPath),
        labels=np.asarray(TARGET_ACTION_LABELS, dtype=np.str_),
        target_labels=np.asarray(TARGET_ACTION_LABELS, dtype=np.str_),
        label_ids=publicLabelIds,
        auto_labels=publicAutoLabels,
        auto_confidence=autoConfidence,
        auto_param_names=autoParamNames,
        auto_param_values=autoParamValues,
        motion_rule_labels=motionRuleLabels,
        final_labels=publicFinalLabels,
        soft_weights=publicSoftWeights,
        segments=_segments_to_export_array(publicSegments),
        internal_labels=np.asarray(ACTION_LABELS, dtype=np.str_),
        internal_label_ids=internalLabelIds,
        internal_auto_labels=internalAutoLabels,
        internal_final_labels=internalFinalLabels,
        internal_soft_weights=internalFinalWeights,
        internal_segments=_segments_to_export_array(labelResult.final_segments),
    )
    return str(exportFile)


def _build_processed_auto_labels_from_feature_source(
    featureSource,
    params=None,
    minSegmentLength=6,
):
    params = CoerceLabelAutoParams(params)
    scores = BuildAutoLabelScores(featureSource, params=params)
    rawLabels = np.asarray(featureSource.get("motion_rule_labels", _labels_from_scores(scores)), dtype=object)
    rawConfidence = np.asarray(
        featureSource.get("auto_confidence", np.ones((featureSource["frame_count"],), dtype=np.float32)),
        dtype=np.float32,
    )
    scoreMargins = _score_margin(scores)
    dt = float(featureSource["dt"])
    mergeMinFrames = max(int(minSegmentLength), _seconds_to_frames(dt, 0.10))
    cleanupMinFrames = max(2, _seconds_to_frames(dt, 0.05))
    transitionMinFrames = max(2, _seconds_to_frames(dt, params.transition_min_seconds))
    labels = rawLabels.copy()
    labels = _majority_filter_labels(labels, windowSize=max(1, int(params.smoothing_window)))
    labels = _merge_short_segments(labels, minSegmentLength=mergeMinFrames, dt=dt)
    labels = _insert_transition_labels(
        labels,
        transitionFrames=max(0, int(params.transition_frames) // 2),
        scoreMargins=scoreMargins,
        minSegmentLength=transitionMinFrames,
        maxScoreMargin=float(params.transition_max_score_margin),
    )
    labels = _merge_short_segments(labels, minSegmentLength=cleanupMinFrames, dt=dt)
    autoConfidence = _postprocess_auto_confidence(labels, rawLabels, rawConfidence, scoreMargins)
    return scores, labels, autoConfidence


def RebuildAutoLabelsWithParams(labelResult: LabelModuleResult, params=None, minSegmentLength=6) -> LabelModuleResult:
    if labelResult.feature_source is None:
        raise ValueError("feature_source must be available to rebuild auto labels.")

    params = CoerceLabelAutoParams(params)
    scores, labels, autoConfidence = _build_processed_auto_labels_from_feature_source(
        labelResult.feature_source,
        params=params,
        minSegmentLength=minSegmentLength,
    )
    labelResult.auto_params = params
    labelResult.auto_scores = np.asarray(scores, dtype=np.float32)
    labelResult.auto_labels = np.asarray(labels, dtype=object)
    labelResult.auto_confidence = np.asarray(autoConfidence, dtype=np.float32)
    labelResult.auto_segments = _labels_to_segments(labelResult.auto_labels, source="auto")
    return _rebuild_final_labels(labelResult)


def BuildAutoFrameLabels(
    clipNameOrPath: str,
    globalPositions,
    poseSource,
    rootTrajectorySource,
    contactData=None,
    terrainProvider=None,
    jointNames=None,
    smoothingWindow=7,
    minSegmentLength=6,
    transitionFrames=DEFAULT_TRANSITION_FRAMES,
    labelParams=None,
):
    params = CoerceLabelAutoParams(labelParams)
    if labelParams is None:
        params.smoothing_window = int(smoothingWindow)
        params.transition_frames = int(transitionFrames)

    featureSource = BuildLabelFeatureSource(
        clipNameOrPath,
        globalPositions,
        poseSource,
        rootTrajectorySource,
        contactData=contactData,
        terrainProvider=terrainProvider,
        jointNames=jointNames,
    )
    scores, labels, autoConfidence = _build_processed_auto_labels_from_feature_source(
        featureSource,
        params=params,
        minSegmentLength=minSegmentLength,
    )

    autoSegments = _labels_to_segments(labels, source="auto")
    result = BuildLabelModuleResult(
        clipNameOrPath,
        featureSource=featureSource,
        autoParams=params,
        autoScores=scores,
        autoLabels=np.asarray(labels, dtype=object),
        autoConfidence=autoConfidence,
        autoSegments=autoSegments,
        annotationPath=str(GetDefaultAnnotationPath(clipNameOrPath)),
    )
    return _rebuild_final_labels(result)
