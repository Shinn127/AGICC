from __future__ import annotations

from pathlib import Path
import argparse
import sys

MOTION_MATCHING_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MOTION_MATCHING_ROOT
if REPO_ROOT.name == "MotionMatching":
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from MotionMatching.MotionMatchingConfig import DEFAULT_DATABASE_PATH
from MotionMatching.MotionMatchingDataset import MotionMatchingDataset, summarize_database


def _print_group_layout(database: MotionMatchingDataset) -> None:
    layout = database.spec.feature_layout
    print("Feature groups:")
    for name, start, stop, weight in zip(
        layout.group_names,
        layout.group_starts,
        layout.group_stops,
        layout.group_weights,
    ):
        print(f"  {name}: [{start}:{stop}] dim={stop - start} weight={weight:.3f}")


def _check_finite(database: MotionMatchingDataset) -> None:
    keys = (
        "raw_features",
        "features",
        "local_positions",
        "local_rotations",
        "local_velocities",
        "root_positions",
        "root_rotations",
        "root_local_velocities",
        "root_local_angular_velocities",
    )
    for key in keys:
        values = database.data[key]
        finite = bool(np.all(np.isfinite(values)))
        print(f"{key}: finite={finite} shape={values.shape}")


def _parse_min_action_samples(values: list[str] | None) -> dict[str, int]:
    minimums: dict[str, int] = {}
    for value in values or []:
        if "=" not in str(value):
            raise ValueError(f"Invalid --min-action-samples value {value!r}. Expected action=count.")
        action, count = str(value).split("=", 1)
        action = action.strip()
        if not action:
            raise ValueError(f"Invalid --min-action-samples value {value!r}: empty action name.")
        minimums[action] = int(count)
    return minimums


def _check_required_actions(database: MotionMatchingDataset, required_actions: list[str] | None, minimums: dict[str, int]) -> None:
    if not required_actions and not minimums:
        return

    action_counts = np.bincount(
        database.action_ids,
        minlength=len(database.action_labels),
    )
    counts_by_label = dict(zip(database.action_labels, action_counts.tolist()))
    for action in required_actions or []:
        if action not in counts_by_label:
            raise AssertionError(f"Required action {action!r} is not in database labels {database.action_labels}.")
        if int(counts_by_label[action]) <= 0:
            raise AssertionError(f"Required action {action!r} has no samples.")
    for action, minimum in minimums.items():
        if action not in counts_by_label:
            raise AssertionError(f"Minimum requested for unknown action {action!r}.")
        if int(counts_by_label[action]) < int(minimum):
            raise AssertionError(
                f"Action {action!r} has {counts_by_label[action]} samples, "
                f"below required minimum {minimum}."
            )
    print("Required actions: passed")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Inspect a Motion Matching database.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH, help="Path to motion_matching_database.npz.")
    parser.add_argument("--finite", action="store_true", help="Check core arrays for finite values.")
    parser.add_argument("--require-actions", nargs="*", default=None, help="Require these action labels to have at least one sample.")
    parser.add_argument(
        "--min-action-samples",
        action="append",
        default=None,
        help="Require a minimum sample count, formatted as action=count. Can be repeated.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    with MotionMatchingDataset(args.database) as database:
        summary = summarize_database(database)
        print(f"Database: {summary['path']}")
        print(f"Stage: {summary['stage']}")
        print(f"dt: {summary['dt']:.6f}")
        print(f"Samples: {summary['samples']}")
        print(f"Joints: {summary['joints']}")
        print(f"Feature dim: {summary['feature_dim']}")
        print(f"Future offsets: {summary['future_sample_offsets']}")
        print(f"Actions: {summary['actions']}")
        print(f"Clips: {summary['clips']}")
        print(f"Variants: {summary['variants']}")
        print("Ranges:")
        for range_info in summary["ranges"]:
            print(
                f"  {range_info['name']}: "
                f"[{range_info['start']}:{range_info['stop']}] "
                f"len={range_info['length']}"
            )
        _print_group_layout(database)
        if args.finite:
            _check_finite(database)
        _check_required_actions(
            database,
            args.require_actions,
            _parse_min_action_samples(args.min_action_samples),
        )


if __name__ == "__main__":
    main()
