from __future__ import annotations

from pathlib import Path


MOTION_MATCHING_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MOTION_MATCHING_ROOT
if REPO_ROOT.name == "MotionMatching":
    REPO_ROOT = REPO_ROOT.parent

DEFAULT_DATASET_DIR = REPO_ROOT.parent / "bvh" / "lafan1"
DEFAULT_OUTPUT_DIR = MOTION_MATCHING_ROOT / "output"
DEFAULT_DATABASE_PATH = DEFAULT_OUTPUT_DIR / "motion_matching_locomotion_v2.npz"

DEFAULT_BVH_SCALE = 0.01
DEFAULT_BVH_FRAME_TIME = 1.0 / 60.0

MM_DATABASE_STAGE = "phase1_mvp"

# Phase 0 MVP: keep the future trajectory compact and easy to tune.
MM_FUTURE_SAMPLE_OFFSETS = (20, 40, 60)

MM_ACTION_LABELS = (
    "idle",
    "walk",
    "run",
    "jump",
)

MM_FOOT_POSITION_JOINTS = (
    "LeftToeBase",
    "RightToeBase",
)

MM_VELOCITY_JOINTS = (
    "LeftToeBase",
    "RightToeBase",
    "Hips",
)

MM_DEFAULT_CLIP_SPECS = (
    ("walk1_subject5", None, None),
    ("run1_subject5", None, None),
    ("jumps1_subject1", None, None),
    ("pushAndStumble1_subject5", 260, 700),
)

MM_INITIAL_FEATURE_GROUP_WEIGHTS = {
    "foot_positions": 0.8,
    "joint_velocities": 1.0,
    "future_positions": 1.2,
    "future_directions": 1.0,
    "future_velocities": 0.8,
    "action": 0.3,
}

MM_SEARCH_INTERVAL = 0.12
MM_INERTIALIZATION_HALFLIFE = 0.075
MM_ACTION_BLEND_HALFLIFE = 0.08
MM_CURRENT_FRAME_BIAS = 0.01
MM_MIN_IMPROVEMENT = 0.02

MM_SEARCH_BACKEND_EXACT = "exact"
MM_SEARCH_BACKEND_KDTREE = "kdtree"
MM_SEARCH_BACKEND_AUTO = "auto"
MM_SEARCH_BACKENDS = (
    MM_SEARCH_BACKEND_EXACT,
    MM_SEARCH_BACKEND_KDTREE,
    MM_SEARCH_BACKEND_AUTO,
)
MM_DEFAULT_SEARCH_BACKEND = MM_SEARCH_BACKEND_AUTO
MM_KDTREE_MIN_SAMPLES = 50000
MM_KDTREE_LEAF_SIZE = 32
MM_KDTREE_QUERY_OVERSAMPLE = 8
MM_KDTREE_EPS = 0.0

MM_LABEL_SOURCE_CLIP = "clip"
MM_LABEL_SOURCE_AUTO = "auto"
MM_LABEL_SOURCES = (
    MM_LABEL_SOURCE_CLIP,
    MM_LABEL_SOURCE_AUTO,
)
MM_DEFAULT_LABEL_SOURCE = MM_LABEL_SOURCE_AUTO
