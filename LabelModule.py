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
LABEL_FALL = "fall"
LABEL_GROUND = "ground"
LABEL_GET_UP = "get_up"
LABEL_TRANSITION = "transition"
LABEL_OTHER = "other"

ACTION_LABELS = (
    LABEL_IDLE,
    LABEL_WALK,
    LABEL_RUN,
    LABEL_JUMP,
    LABEL_FALL,
    LABEL_GROUND,
    LABEL_GET_UP,
    LABEL_TRANSITION,
    LABEL_OTHER,
)

LABEL_TO_INDEX = {label: index for index, label in enumerate(ACTION_LABELS)}

CLIP_PRIOR_WALK = "walk"
CLIP_PRIOR_RUN = "run"
CLIP_PRIOR_JUMP = "jump"
CLIP_PRIOR_FALL_RECOVERY = "fall_recovery"
CLIP_PRIOR_GROUND = "ground"
CLIP_PRIOR_MIXED = "mixed"
CLIP_PRIOR_OTHER = "other"

CLIP_PRIORS = (
    CLIP_PRIOR_WALK,
    CLIP_PRIOR_RUN,
    CLIP_PRIOR_JUMP,
    CLIP_PRIOR_FALL_RECOVERY,
    CLIP_PRIOR_GROUND,
    CLIP_PRIOR_MIXED,
    CLIP_PRIOR_OTHER,
)

DEFAULT_TRANSITION_FRAMES = 8

_CLIP_PRIOR_PREFIX_MAP = (
    ("walk", CLIP_PRIOR_WALK),
    ("run", CLIP_PRIOR_RUN),
    ("sprint", CLIP_PRIOR_RUN),
    ("jumps", CLIP_PRIOR_JUMP),
    ("fallandgetup", CLIP_PRIOR_FALL_RECOVERY),
    ("pushandfall", CLIP_PRIOR_FALL_RECOVERY),
    ("ground", CLIP_PRIOR_GROUND),
    ("multipleactions", CLIP_PRIOR_MIXED),
    ("aiming", CLIP_PRIOR_OTHER),
    ("dance", CLIP_PRIOR_OTHER),
    ("fightandsports", CLIP_PRIOR_OTHER),
    ("fight", CLIP_PRIOR_OTHER),
    ("obstacles", CLIP_PRIOR_OTHER),
    ("pushandstumble", CLIP_PRIOR_OTHER),
    ("push", CLIP_PRIOR_OTHER),
)

_CLIP_PRIOR_CANDIDATE_LABELS = {
    CLIP_PRIOR_WALK: (
        LABEL_IDLE,
        LABEL_WALK,
        LABEL_RUN,
        LABEL_JUMP,
        LABEL_TRANSITION,
        LABEL_OTHER,
    ),
    CLIP_PRIOR_RUN: (
        LABEL_IDLE,
        LABEL_WALK,
        LABEL_RUN,
        LABEL_JUMP,
        LABEL_TRANSITION,
        LABEL_OTHER,
    ),
    CLIP_PRIOR_JUMP: (
        LABEL_IDLE,
        LABEL_WALK,
        LABEL_RUN,
        LABEL_JUMP,
        LABEL_TRANSITION,
        LABEL_OTHER,
    ),
    CLIP_PRIOR_FALL_RECOVERY: (
        LABEL_FALL,
        LABEL_GROUND,
        LABEL_GET_UP,
        LABEL_TRANSITION,
        LABEL_OTHER,
    ),
    CLIP_PRIOR_GROUND: (
        LABEL_FALL,
        LABEL_GROUND,
        LABEL_GET_UP,
        LABEL_TRANSITION,
        LABEL_OTHER,
    ),
    CLIP_PRIOR_MIXED: ACTION_LABELS,
    CLIP_PRIOR_OTHER: (
        LABEL_IDLE,
        LABEL_WALK,
        LABEL_RUN,
        LABEL_JUMP,
        LABEL_FALL,
        LABEL_GROUND,
        LABEL_GET_UP,
        LABEL_TRANSITION,
        LABEL_OTHER,
    ),
}


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
    clip_prior: str
    candidate_labels: tuple[str, ...]
    feature_source: Optional[dict] = None
    auto_scores: Optional[np.ndarray] = None
    auto_labels: Optional[np.ndarray] = None
    auto_segments: list[LabelSegment] = field(default_factory=list)
    manual_labels: Optional[np.ndarray] = None
    final_labels: Optional[np.ndarray] = None
    final_segments: list[LabelSegment] = field(default_factory=list)
    soft_weights: Optional[np.ndarray] = None
    transition_overrides: list[dict] = field(default_factory=list)
    annotation_path: Optional[str] = None
    annotation_loaded: bool = False


def NormalizeClipName(clipNameOrPath: str) -> str:
    clipName = Path(str(clipNameOrPath)).stem
    return clipName.strip()


def _canonicalize_clip_key(clipNameOrPath: str) -> str:
    clipName = NormalizeClipName(clipNameOrPath).lower()
    if "_subject" in clipName:
        clipName = clipName.split("_subject", 1)[0]
    return "".join(character for character in clipName if character.isalnum())


def GetClipPriorFromName(clipNameOrPath: str) -> str:
    clipKey = _canonicalize_clip_key(clipNameOrPath)

    for prefix, clipPrior in _CLIP_PRIOR_PREFIX_MAP:
        if clipKey.startswith(prefix):
            return clipPrior

    return CLIP_PRIOR_OTHER


def GetCandidateLabelsFromPrior(clipPrior: str) -> tuple[str, ...]:
    if clipPrior not in _CLIP_PRIOR_CANDIDATE_LABELS:
        raise ValueError(f'Unsupported clip prior "{clipPrior}".')
    return _CLIP_PRIOR_CANDIDATE_LABELS[clipPrior]


def BuildLabelModuleResult(
    clipNameOrPath: str,
    featureSource=None,
    autoScores=None,
    autoLabels=None,
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
    clipPrior = GetClipPriorFromName(clipName)
    candidateLabels = GetCandidateLabelsFromPrior(clipPrior)

    return LabelModuleResult(
        clip_name=clipName,
        clip_prior=clipPrior,
        candidate_labels=candidateLabels,
        feature_source=featureSource,
        auto_scores=None if autoScores is None else np.asarray(autoScores, dtype=np.float32),
        auto_labels=None if autoLabels is None else np.asarray(autoLabels),
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
    clipPrior = GetClipPriorFromName(clipName)
    candidateLabels = GetCandidateLabelsFromPrior(clipPrior)

    globalPositions = np.asarray(globalPositions, dtype=np.float32)
    rootPositions = np.asarray(rootTrajectorySource["positions"], dtype=np.float32)
    rootVelocities = np.asarray(rootTrajectorySource["velocities"], dtype=np.float32)
    rootDirections = np.asarray(rootTrajectorySource["directions"], dtype=np.float32)
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
    standingHeight = _estimate_standing_height(rootHeightAboveGround)
    lowHeightThreshold = _estimate_low_height_threshold(standingHeight)

    return {
        "clip_name": clipName,
        "clip_prior": clipPrior,
        "candidate_labels": candidateLabels,
        "frame_count": frameCount,
        "dt": dt,
        "root_positions": rootPositions,
        "root_velocities": rootVelocities,
        "root_directions": rootDirections,
        "root_horizontal_speed": rootHorizontalSpeed,
        "root_vertical_speed": rootVerticalSpeed,
        "root_yaw_rate": rootYawRate,
        "root_height_above_ground": rootHeightAboveGround,
        "standing_height": float(standingHeight),
        "low_height_threshold": float(lowHeightThreshold),
        "left_contact": leftContact,
        "right_contact": rightContact,
        "contact_fraction": contactFraction,
        "motion_energy": motionEnergy,
        "upper_body_energy": upperBodyEnergy,
        "terrain_heights": terrainHeights,
    }


def BuildAutoLabelScores(featureSource) -> np.ndarray:
    frameCount = int(featureSource["frame_count"])
    scores = np.zeros((frameCount, len(ACTION_LABELS)), dtype=np.float32)
    if frameCount == 0:
        return scores

    clipPrior = featureSource["clip_prior"]
    candidateLabels = set(featureSource["candidate_labels"])
    speed = np.asarray(featureSource["root_horizontal_speed"], dtype=np.float32)
    verticalSpeed = np.asarray(featureSource["root_vertical_speed"], dtype=np.float32)
    height = np.asarray(featureSource["root_height_above_ground"], dtype=np.float32)
    lowHeightThreshold = float(featureSource["low_height_threshold"])
    standingHeight = float(featureSource["standing_height"])
    contactFraction = np.asarray(featureSource["contact_fraction"], dtype=np.float32)
    leftContact = np.asarray(featureSource["left_contact"], dtype=np.float32)
    rightContact = np.asarray(featureSource["right_contact"], dtype=np.float32)
    motionEnergy = np.asarray(featureSource["motion_energy"], dtype=np.float32)
    upperBodyEnergy = np.asarray(featureSource["upper_body_energy"], dtype=np.float32)

    noContact = 1.0 - np.clip(np.maximum(leftContact, rightContact), 0.0, 1.0)
    doubleContact = np.minimum(leftContact, rightContact)
    lowHeight = _score_less(height, lowHeightThreshold, max(0.1 * standingHeight, 0.05))
    highHeight = _score_greater(height, 0.9 * standingHeight, max(0.1 * standingHeight, 0.05))
    rising = _score_greater(verticalSpeed, 0.15, 0.35)
    fallingVelocity = _score_less(verticalSpeed, -0.2, 0.4)
    lowVerticalMotion = _score_less(np.abs(verticalSpeed), 0.16, 0.24)
    idleSpeed = _score_less(speed, 0.12, 0.12)
    walkSpeed = _score_range(speed, 0.15, 1.15)
    runSpeed = _score_greater(speed, 1.0, 0.9)
    lowGroundSpeed = _score_less(speed, 0.8, 0.7)
    lowEnergy = _score_less(motionEnergy, 0.55, 0.35)
    upperBodyDominance = np.clip(_safe_normalize(upperBodyEnergy) - 0.5 * _safe_normalize(speed), 0.0, 1.0)

    scores[:, LABEL_TO_INDEX[LABEL_IDLE]] = (
        1.4 * idleSpeed +
        0.8 * lowEnergy +
        0.6 * doubleContact
    )
    scores[:, LABEL_TO_INDEX[LABEL_WALK]] = (
        1.4 * walkSpeed +
        0.7 * contactFraction +
        0.2 * _score_less(np.abs(verticalSpeed), 0.2, 0.2)
    )
    scores[:, LABEL_TO_INDEX[LABEL_RUN]] = (
        1.6 * runSpeed +
        0.4 * _score_less(contactFraction, 0.8, 0.5) +
        0.3 * _score_greater(motionEnergy, 0.8, 0.6)
    )
    scores[:, LABEL_TO_INDEX[LABEL_JUMP]] = (
        1.5 * noContact +
        0.9 * np.maximum(rising, highHeight) +
        0.3 * _score_greater(np.abs(verticalSpeed), 0.25, 0.35)
    )
    scores[:, LABEL_TO_INDEX[LABEL_FALL]] = (
        1.4 * lowHeight +
        1.0 * fallingVelocity +
        0.3 * noContact
    )
    scores[:, LABEL_TO_INDEX[LABEL_GROUND]] = (
        1.5 * lowHeight +
        1.1 * lowVerticalMotion +
        0.4 * contactFraction +
        0.3 * lowGroundSpeed
    )
    scores[:, LABEL_TO_INDEX[LABEL_GET_UP]] = (
        1.4 * lowHeight +
        1.0 * rising +
        0.5 * contactFraction
    )
    scores[:, LABEL_TO_INDEX[LABEL_OTHER]] = (
        0.4 +
        0.8 * upperBodyDominance +
        0.2 * _score_greater(motionEnergy, 0.9, 0.7)
    )
    scores[:, LABEL_TO_INDEX[LABEL_TRANSITION]] = 0.1

    if clipPrior == CLIP_PRIOR_WALK:
        scores[:, LABEL_TO_INDEX[LABEL_WALK]] += 0.35
        scores[:, LABEL_TO_INDEX[LABEL_OTHER]] -= 0.15
    elif clipPrior == CLIP_PRIOR_RUN:
        scores[:, LABEL_TO_INDEX[LABEL_RUN]] += 0.45
        scores[:, LABEL_TO_INDEX[LABEL_WALK]] += 0.15
    elif clipPrior == CLIP_PRIOR_JUMP:
        scores[:, LABEL_TO_INDEX[LABEL_JUMP]] += 0.6
    elif clipPrior == CLIP_PRIOR_FALL_RECOVERY:
        scores[:, LABEL_TO_INDEX[LABEL_FALL]] += 0.55
        scores[:, LABEL_TO_INDEX[LABEL_GROUND]] += 0.35
        scores[:, LABEL_TO_INDEX[LABEL_GET_UP]] += 0.55
        scores[:, LABEL_TO_INDEX[LABEL_IDLE]] -= 0.2
        scores[:, LABEL_TO_INDEX[LABEL_WALK]] -= 0.2
        scores[:, LABEL_TO_INDEX[LABEL_RUN]] -= 0.2
    elif clipPrior == CLIP_PRIOR_GROUND:
        scores[:, LABEL_TO_INDEX[LABEL_GROUND]] += 0.75
        scores[:, LABEL_TO_INDEX[LABEL_FALL]] += 0.15
        scores[:, LABEL_TO_INDEX[LABEL_GET_UP]] += 0.15
    elif clipPrior == CLIP_PRIOR_OTHER:
        scores[:, LABEL_TO_INDEX[LABEL_OTHER]] += 0.5

    for label, index in LABEL_TO_INDEX.items():
        if label not in candidateLabels and label != LABEL_TRANSITION:
            scores[:, index] -= 0.75

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
    startFrame = 0
    currentLabel = str(labels[0])

    for frameIndex in range(1, len(labels)):
        if labels[frameIndex] == currentLabel:
            continue
        segments.append(LabelSegment(startFrame, frameIndex - 1, currentLabel, source=source))
        startFrame = frameIndex
        currentLabel = str(labels[frameIndex])

    segments.append(LabelSegment(startFrame, len(labels) - 1, currentLabel, source=source))
    return segments


def _manual_labels_to_segments(manualLabels):
    manualLabels = np.asarray(manualLabels, dtype=object)
    if len(manualLabels) == 0:
        return []

    segments = []
    currentLabel = None
    startFrame = None

    for frameIndex, label in enumerate(manualLabels):
        label = None if label is None else str(label)
        if label == currentLabel:
            continue

        if currentLabel is not None and startFrame is not None:
            segments.append(LabelSegment(startFrame, frameIndex - 1, currentLabel, source="manual"))

        currentLabel = label
        startFrame = frameIndex if label is not None else None

    if currentLabel is not None and startFrame is not None:
        segments.append(LabelSegment(startFrame, len(manualLabels) - 1, currentLabel, source="manual"))

    return segments


def GetDefaultAnnotationPath(clipNameOrPath: str, annotationsRoot="resources/annotations") -> Path:
    clipPath = Path(str(clipNameOrPath))
    clipStem = clipPath.stem
    clipParents = list(clipPath.parent.parts)
    if clipParents and clipParents[0] == "bvh":
        clipParents = clipParents[1:]
    return Path(annotationsRoot).joinpath(*clipParents, f"{clipStem}.json")


def _merge_short_segments(labels, minSegmentLength=6):
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
            if segmentLength >= minSegmentLength:
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


def _insert_transition_labels(labels, transitionFrames=DEFAULT_TRANSITION_FRAMES):
    labels = np.asarray(labels, dtype=object)
    if len(labels) == 0 or transitionFrames <= 0:
        return labels.copy()

    result = labels.copy()
    halfWidth = max(1, int(transitionFrames) // 2)
    segments = _labels_to_segments(labels, source="auto")

    for segmentIndex in range(len(segments) - 1):
        currentSegment = segments[segmentIndex]
        nextSegment = segments[segmentIndex + 1]
        if currentSegment.label == nextSegment.label:
            continue
        if LABEL_TRANSITION in (currentSegment.label, nextSegment.label):
            continue

        leftBudget = max(0, currentSegment.end_frame - currentSegment.start_frame + 1)
        rightBudget = max(0, nextSegment.end_frame - nextSegment.start_frame + 1)
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


def _build_soft_weights_from_labels(labels, transitionFrames=DEFAULT_TRANSITION_FRAMES):
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

        previousSegment = _find_neighbor_segment(segments, segmentIndex, -1)
        nextSegment = _find_neighbor_segment(segments, segmentIndex, 1)
        segmentLength = segment.end_frame - segment.start_frame + 1

        for localIndex, frameIndex in enumerate(range(segment.start_frame, segment.end_frame + 1)):
            alpha = 0.5 if segmentLength <= 1 else localIndex / max(segmentLength - 1, 1)
            eased = _ease_cosine(alpha)
            transitionStrength = 0.45 + 0.55 * np.sin(np.pi * alpha)

            if previousSegment is not None and nextSegment is not None:
                if previousSegment.label == nextSegment.label:
                    sharedStrength = max(0.0, 1.0 - transitionStrength)
                    _assign_soft_weight_row(
                        weights,
                        frameIndex,
                        {
                            previousSegment.label: sharedStrength,
                            LABEL_TRANSITION: transitionStrength,
                        },
                    )
                else:
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
            elif previousSegment is not None:
                _assign_soft_weight_row(
                    weights,
                    frameIndex,
                    {
                        previousSegment.label: max(0.0, 1.0 - eased),
                        LABEL_TRANSITION: max(0.0, eased),
                    },
                )
            elif nextSegment is not None:
                _assign_soft_weight_row(
                    weights,
                    frameIndex,
                    {
                        LABEL_TRANSITION: max(0.0, 1.0 - eased),
                        nextSegment.label: max(0.0, eased),
                    },
                )

    implicitBlendFrames = max(1, int(transitionFrames) // 3)
    for segmentIndex in range(len(segments) - 1):
        leftSegment = segments[segmentIndex]
        rightSegment = segments[segmentIndex + 1]
        if LABEL_TRANSITION in (leftSegment.label, rightSegment.label):
            continue
        if leftSegment.label == rightSegment.label:
            continue

        leftLength = leftSegment.end_frame - leftSegment.start_frame + 1
        rightLength = rightSegment.end_frame - rightSegment.start_frame + 1
        blendFrames = min(implicitBlendFrames, leftLength, rightLength)
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


def ApplyManualLabelRange(labelResult: LabelModuleResult, startFrame: int, endFrame: int, label: str) -> LabelModuleResult:
    if label not in LABEL_TO_INDEX:
        raise ValueError(f'Unsupported action label "{label}".')

    manualLabels = _ensure_manual_labels(labelResult)
    startFrame = max(0, int(min(startFrame, endFrame)))
    endFrame = min(len(manualLabels) - 1, int(max(startFrame, endFrame)))
    manualLabels[startFrame:endFrame + 1] = label
    return _rebuild_final_labels(labelResult)


def ClearManualLabelRange(labelResult: LabelModuleResult, startFrame: int, endFrame: int) -> LabelModuleResult:
    manualLabels = _ensure_manual_labels(labelResult)
    startFrame = max(0, int(min(startFrame, endFrame)))
    endFrame = min(len(manualLabels) - 1, int(max(startFrame, endFrame)))
    manualLabels[startFrame:endFrame + 1] = None
    return _rebuild_final_labels(labelResult)


def ResetManualLabels(labelResult: LabelModuleResult) -> LabelModuleResult:
    manualLabels = _ensure_manual_labels(labelResult)
    manualLabels[:] = None
    return _rebuild_final_labels(labelResult)


def ApplyTransitionWidthRange(labelResult: LabelModuleResult, startFrame: int, endFrame: int, width: int) -> LabelModuleResult:
    if labelResult.auto_labels is None:
        raise ValueError("auto_labels must be available before transition editing.")
    frameCount = len(labelResult.auto_labels)
    startFrame = max(0, min(int(startFrame), frameCount - 1))
    endFrame = max(0, min(int(endFrame), frameCount - 1))
    labelResult.transition_overrides.append(_normalize_transition_override(startFrame, endFrame, width))
    return _rebuild_final_labels(labelResult)


def ClearTransitionWidthRange(labelResult: LabelModuleResult, startFrame: int, endFrame: int) -> LabelModuleResult:
    startFrame = int(min(startFrame, endFrame))
    endFrame = int(max(startFrame, endFrame))
    labelResult.transition_overrides = [
        override for override in labelResult.transition_overrides
        if int(override["end_frame"]) < startFrame or int(override["start_frame"]) > endFrame
    ]
    return _rebuild_final_labels(labelResult)


def SaveLabelAnnotations(labelResult: LabelModuleResult, clipNameOrPath: str, annotationPath: Optional[str] = None) -> str:
    manualLabels = _ensure_manual_labels(labelResult)
    annotationFile = Path(annotationPath) if annotationPath is not None else GetDefaultAnnotationPath(clipNameOrPath)
    annotationFile.parent.mkdir(parents=True, exist_ok=True)

    manualSegments = _manual_labels_to_segments(manualLabels)
    payload = {
        "version": 1,
        "clip_name": NormalizeClipName(clipNameOrPath),
        "clip_prior": labelResult.clip_prior,
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


def LoadLabelAnnotations(labelResult: LabelModuleResult, clipNameOrPath: str, annotationPath: Optional[str] = None) -> bool:
    annotationFile = Path(annotationPath) if annotationPath is not None else GetDefaultAnnotationPath(clipNameOrPath)
    labelResult.annotation_path = str(annotationFile)

    if not annotationFile.is_file():
        labelResult.annotation_loaded = False
        return False

    payload = json.loads(annotationFile.read_text(encoding="utf-8"))
    overrides = list(payload.get("manual_overrides", []))
    transitionOverrides = list(payload.get("transition_overrides", []))

    manualLabels = _ensure_manual_labels(labelResult)
    manualLabels[:] = None

    for override in overrides:
        label = str(override["label"])
        if label not in LABEL_TO_INDEX:
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

    labelIds = np.asarray([LABEL_TO_INDEX[str(label)] for label in labelResult.final_labels], dtype=np.int32)
    np.savez_compressed(
        exportFile,
        clip_name=NormalizeClipName(clipNameOrPath),
        labels=np.asarray(ACTION_LABELS, dtype=object),
        label_ids=labelIds,
        final_labels=np.asarray(labelResult.final_labels, dtype=object),
        soft_weights=np.asarray(labelResult.soft_weights, dtype=np.float32),
        segments=np.asarray(
            [
                (segment.start_frame, segment.end_frame, segment.label)
                for segment in labelResult.final_segments
            ],
            dtype=object,
        ),
    )
    return str(exportFile)


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
):
    featureSource = BuildLabelFeatureSource(
        clipNameOrPath,
        globalPositions,
        poseSource,
        rootTrajectorySource,
        contactData=contactData,
        terrainProvider=terrainProvider,
        jointNames=jointNames,
    )
    scores = BuildAutoLabelScores(featureSource)
    labels = _labels_from_scores(scores)
    labels = _majority_filter_labels(labels, windowSize=smoothingWindow)
    labels = _merge_short_segments(labels, minSegmentLength=minSegmentLength)
    labels = _insert_transition_labels(labels, transitionFrames=transitionFrames)
    labels = _merge_short_segments(labels, minSegmentLength=max(2, minSegmentLength // 2))

    autoSegments = _labels_to_segments(labels, source="auto")
    softWeights = CreateEmptySoftWeights(len(labels), fillLabel=LABEL_OTHER)
    if len(labels) > 0:
        softWeights[:] = 0.0
        labelIndices = np.asarray([LABEL_TO_INDEX[str(label)] for label in labels], dtype=np.int32)
        softWeights[np.arange(len(labels)), labelIndices] = 1.0

    result = BuildLabelModuleResult(
        clipNameOrPath,
        featureSource=featureSource,
        autoScores=scores,
        autoLabels=np.asarray(labels, dtype=object),
        autoSegments=autoSegments,
        softWeights=softWeights,
        annotationPath=str(GetDefaultAnnotationPath(clipNameOrPath)),
    )
    return _rebuild_final_labels(result)


__all__ = [
    "LABEL_IDLE",
    "LABEL_WALK",
    "LABEL_RUN",
    "LABEL_JUMP",
    "LABEL_FALL",
    "LABEL_GROUND",
    "LABEL_GET_UP",
    "LABEL_TRANSITION",
    "LABEL_OTHER",
    "ACTION_LABELS",
    "LABEL_TO_INDEX",
    "CLIP_PRIOR_WALK",
    "CLIP_PRIOR_RUN",
    "CLIP_PRIOR_JUMP",
    "CLIP_PRIOR_FALL_RECOVERY",
    "CLIP_PRIOR_GROUND",
    "CLIP_PRIOR_MIXED",
    "CLIP_PRIOR_OTHER",
    "CLIP_PRIORS",
    "DEFAULT_TRANSITION_FRAMES",
    "LabelSegment",
    "LabelModuleResult",
    "NormalizeClipName",
    "GetClipPriorFromName",
    "GetCandidateLabelsFromPrior",
    "BuildLabelFeatureSource",
    "BuildAutoLabelScores",
    "BuildAutoFrameLabels",
    "ApplyManualLabelRange",
    "ClearManualLabelRange",
    "ResetManualLabels",
    "ApplyTransitionWidthRange",
    "ClearTransitionWidthRange",
    "GetDefaultAnnotationPath",
    "SaveLabelAnnotations",
    "LoadLabelAnnotations",
    "GetDefaultExportPath",
    "ExportCompiledLabels",
    "BuildLabelModuleResult",
    "CreateEmptySoftWeights",
]
