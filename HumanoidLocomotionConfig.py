"""Joint-group constants for humanoid locomotion tasks.

This file intentionally captures a locomotion-oriented subset of the
available humanoid rig. It excludes fingers and most terminal helper joints
so the feature space stays focused on body motion control.
"""

HUMANOID_LOCOMOTION_TRAJECTORY_SAMPLE_OFFSETS = (
    -60,
    -50,
    -40,
    -30,
    -20,
    -10,
    0,
    10,
    20,
    30,
    40,
    50,
    60,
)
HUMANOID_LOCOMOTION_TRAJECTORY_PAST_SAMPLE_COUNT = 6
HUMANOID_LOCOMOTION_TRAJECTORY_CURRENT_SAMPLE_INDEX = 6
HUMANOID_LOCOMOTION_TRAJECTORY_FUTURE_SAMPLE_COUNT = 6
HUMANOID_LOCOMOTION_TRAJECTORY_SAMPLE_COUNT = len(HUMANOID_LOCOMOTION_TRAJECTORY_SAMPLE_OFFSETS)
HUMANOID_LOCOMOTION_TRAJECTORY_FUTURE_SAMPLE_INDICES = (
    7,
    8,
    9,
    10,
    11,
    12,
)

HUMANOID_LOCOMOTION_ROOT_JOINT = "Hips"

# The trajectory frame is also anchored at the pelvis for locomotion.
HUMANOID_LOCOMOTION_TRAJECTORY_JOINT = HUMANOID_LOCOMOTION_ROOT_JOINT

# Core body joints used for pose/state prediction in locomotion models.
HUMANOID_LOCOMOTION_PREDICTION_JOINTS = (
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Spine3",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
)

# End-effectors that best capture gait phase and foot-ground interaction.
HUMANOID_LOCOMOTION_CONTACT_JOINTS = (
    "LeftFoot",
    "LeftToeBase",
    "RightFoot",
    "RightToeBase",
)

# For locomotion, the gating network can reuse the same foot end-effectors.
HUMANOID_LOCOMOTION_GATING_JOINTS = HUMANOID_LOCOMOTION_CONTACT_JOINTS

HUMANOID_LOCOMOTION_CONTACT_CONFIGS = (
    ("LeftFoot", 0.08, 0.15),
    ("LeftToeBase", 0.06, 0.15),
    ("RightFoot", 0.08, 0.15),
    ("RightToeBase", 0.06, 0.15),
)

HUMANOID_LOCOMOTION_ACTION_LABELS = (
    "walk",
    "run",
    "jump",
)

# LaFAN1 clip-name prefixes used in the stage-1 locomotion dataset.
HUMANOID_LOCOMOTION_ACTION_PREFIX_TO_LABEL = (
    ("walk", "walk"),
    ("run", "run"),
    ("sprint", "run"),
    ("jumps", "jump"),
)

# These joints are usually not worth feeding into a locomotion controller.
HUMANOID_LOCOMOTION_EXCLUDED_JOINTS = (
    "HeadEnd",
    "RightHandThumb1",
    "RightHandThumb2",
    "RightHandThumb3",
    "RightHandThumb4",
    "RightHandIndex1",
    "RightHandIndex2",
    "RightHandIndex3",
    "RightHandIndex4",
    "RightHandMiddle1",
    "RightHandMiddle2",
    "RightHandMiddle3",
    "RightHandMiddle4",
    "RightHandRing1",
    "RightHandRing2",
    "RightHandRing3",
    "RightHandRing4",
    "RightHandPinky1",
    "RightHandPinky2",
    "RightHandPinky3",
    "RightHandPinky4",
    "RightForeArmEnd",
    "RightArmEnd",
    "LeftHandThumb1",
    "LeftHandThumb2",
    "LeftHandThumb3",
    "LeftHandThumb4",
    "LeftHandIndex1",
    "LeftHandIndex2",
    "LeftHandIndex3",
    "LeftHandIndex4",
    "LeftHandMiddle1",
    "LeftHandMiddle2",
    "LeftHandMiddle3",
    "LeftHandMiddle4",
    "LeftHandRing1",
    "LeftHandRing2",
    "LeftHandRing3",
    "LeftHandRing4",
    "LeftHandPinky1",
    "LeftHandPinky2",
    "LeftHandPinky3",
    "LeftHandPinky4",
    "LeftForeArmEnd",
    "LeftArmEnd",
    "RightToeBaseEnd",
    "RightLegEnd",
    "RightUpLegEnd",
    "LeftToeBaseEnd",
    "LeftLegEnd",
    "LeftUpLegEnd",
)
