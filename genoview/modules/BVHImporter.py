from dataclasses import dataclass
from pathlib import Path

import numpy as np

from genoview.State import MotionResources
from genoview.modules.MotionMirror import BuildMotionVariantKey, DEFAULT_MIRROR_AXIS, MirrorBVHAnimation
from genoview.modules.RootModule import DEFAULT_BVH_FRAME_TIME, GetRootTrajectorySampleOffsets
from genoview.utils import bvh, quat


DEFAULT_BVH_CLIP = "lafan1/jumps1_subject1.bvh"
DEFAULT_BVH_DIR = "lafan1"


@dataclass(frozen=True)
class BVHAnimation:
    raw_data: dict
    parents: np.ndarray
    local_positions: np.ndarray
    local_rotations: np.ndarray
    global_positions: np.ndarray
    global_rotations: np.ndarray

    @property
    def frame_count(self):
        return self.local_positions.shape[0]


class BVHImporter:

    @staticmethod
    def load(file_name, scale=1.0):
        bvh_data = bvh.load(str(Path(file_name)))
        return BVHImporter.from_bvh_data(bvh_data, scale=scale)

    @staticmethod
    def from_bvh_data(bvh_data, scale=1.0):
        parents = bvh_data["parents"]
        local_positions = BVHImporter._build_local_positions(bvh_data["positions"], scale)
        local_rotations = BVHImporter._build_local_rotations(
            bvh_data["rotations"],
            bvh_data["order"],
        )
        global_rotations, global_positions = quat.fk(
            local_rotations,
            local_positions,
            parents,
        )

        return BVHAnimation(
            raw_data=bvh_data,
            parents=parents,
            local_positions=local_positions,
            local_rotations=local_rotations,
            global_positions=global_positions,
            global_rotations=global_rotations,
        )

    @staticmethod
    def _build_local_positions(positions, scale):
        return scale * positions.copy().astype(np.float32)

    @staticmethod
    def _build_local_rotations(rotations, order):
        return quat.unroll(
            quat.from_euler(
                np.radians(rotations),
                order=order,
            )
        )


def DiscoverBVHClips(bvh_root, default_bvh_dir=DEFAULT_BVH_DIR, default_bvh_clip=DEFAULT_BVH_CLIP):
    bvh_dir = bvh_root / default_bvh_dir
    clips = [
        str(path.relative_to(bvh_root))
        for path in sorted(bvh_dir.glob("*.bvh"))
    ]
    return clips if clips else [default_bvh_clip]


def GetClipIndex(clip_resources, clip_resource):
    try:
        return clip_resources.index(clip_resource)
    except ValueError:
        return 0


def LoadMotionResources(
    bvh_path,
    clip_resource=DEFAULT_BVH_CLIP,
    mirrored=False,
    mirror_axis=DEFAULT_MIRROR_AXIS,
):
    bvh_animation = BVHImporter.load(bvh_path(clip_resource), scale=0.01)
    if mirrored:
        bvh_animation = MirrorBVHAnimation(bvh_animation, axis=mirror_axis)

    clip_name = Path(clip_resource).stem
    if mirrored:
        clip_name += " [mirror]"

    return MotionResources(
        clip_resource=clip_resource,
        clip_name=clip_name,
        bvh_animation=bvh_animation,
        parents=bvh_animation.parents,
        global_positions=bvh_animation.global_positions,
        global_rotations=bvh_animation.global_rotations,
        trajectory_sample_offsets=GetRootTrajectorySampleOffsets(),
        bvh_frame_time=DEFAULT_BVH_FRAME_TIME,
        motion_variant=BuildMotionVariantKey(mirrored, mirror_axis),
        mirrored=bool(mirrored),
        mirror_axis=str(mirror_axis).lower(),
    )
