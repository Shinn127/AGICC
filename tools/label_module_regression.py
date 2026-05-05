from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from genoview.modules.BVHImporter import LoadMotionResources
from genoview.modules.FeatureModule import EnsureLabelResources
from genoview.modules.LabelModule import LoadLabelAnnotations


DEFAULT_CLIPS = (
    "lafan1/walk1_subject5.bvh",
    "lafan1/walk3_subject5.bvh",
    "lafan1/jumps1_subject1.bvh",
)


def _build_scene_stub():
    return SimpleNamespace(ground_position=SimpleNamespace(y=0.0))


def _bvh_path_builder(project_root: Path):
    bvh_root = project_root.parent / "bvh"

    def bvh_path(*parts):
        return str(bvh_root.joinpath(*parts))

    return bvh_path


def _format_counter(counter):
    if not counter:
        return "{}"
    return "{" + ", ".join(f"{key}: {value}" for key, value in sorted(counter.items())) + "}"


def _summarize_clip(project_root: Path, clip_resource: str, low_conf_threshold: float) -> dict:
    scene = _build_scene_stub()
    motion = LoadMotionResources(_bvh_path_builder(project_root), clip_resource)
    label_result = EnsureLabelResources(scene, motion)

    auto_labels = np.asarray(label_result.auto_labels, dtype=object)
    auto_confidence = np.asarray(label_result.auto_confidence, dtype=np.float32)
    auto_counts = Counter({str(label): int(count) for label, count in zip(*np.unique(auto_labels, return_counts=True))})

    summary = {
        "clip": clip_resource,
        "auto_counts": auto_counts,
        "low_conf_frames": int(np.sum(auto_confidence < float(low_conf_threshold))),
        "conf_mean": float(np.mean(auto_confidence)) if auto_confidence.size > 0 else 0.0,
        "annotated_frames": 0,
        "auto_vs_manual_agree": None,
        "manual_breakdown": {},
    }

    if not LoadLabelAnnotations(label_result, clip_resource):
        return summary

    manual_labels = np.asarray(label_result.manual_labels, dtype=object)
    final_labels = np.asarray(label_result.final_labels, dtype=object)
    annotated_mask = manual_labels != None
    if not np.any(annotated_mask):
        return summary

    summary["annotated_frames"] = int(np.sum(annotated_mask))
    summary["auto_vs_manual_agree"] = float(np.mean(auto_labels[annotated_mask] == final_labels[annotated_mask]))

    for target_label in sorted({str(label) for label in final_labels[annotated_mask]}):
        label_mask = annotated_mask & (final_labels == target_label)
        label_counts = Counter(
            {
                str(label): int(count)
                for label, count in zip(*np.unique(auto_labels[label_mask], return_counts=True))
            }
        )
        summary["manual_breakdown"][target_label] = label_counts
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run local LabelModule regression summaries.")
    parser.add_argument(
        "--clip",
        dest="clips",
        action="append",
        help="Relative BVH clip path such as lafan1/walk3_subject5.bvh. Can be passed multiple times.",
    )
    parser.add_argument(
        "--low-conf-threshold",
        type=float,
        default=0.60,
        help="Confidence threshold used for the low-confidence frame count summary.",
    )
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    clips = tuple(args.clips) if args.clips else DEFAULT_CLIPS

    print(f"Project root: {project_root}")
    print(f"Low-confidence threshold: {float(args.low_conf_threshold):.2f}")
    print()

    for clip_resource in clips:
        summary = _summarize_clip(project_root, clip_resource, args.low_conf_threshold)
        print(summary["clip"])
        print(f"  auto_counts: {_format_counter(summary['auto_counts'])}")
        print(
            "  confidence: "
            f"mean={summary['conf_mean']:.4f} "
            f"low_frames={summary['low_conf_frames']}"
        )
        if summary["annotated_frames"] <= 0:
            print("  manual: no saved annotation")
            print()
            continue

        print(
            "  manual: "
            f"annotated_frames={summary['annotated_frames']} "
            f"auto_vs_manual_agree={summary['auto_vs_manual_agree']:.4f}"
        )
        for target_label, label_counts in summary["manual_breakdown"].items():
            print(f"    {target_label}: {_format_counter(label_counts)}")
        print()


if __name__ == "__main__":
    main()
