from dataclasses import dataclass
from pathlib import Path

import numpy as np

import bvh
import quat


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
