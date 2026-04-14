from dataclasses import replace

import numpy as np

from genoview.utils import quat


DEFAULT_MIRROR_AXIS = "x"
MOTION_VARIANT_ORIGINAL = "original"


def BuildMotionVariantKey(mirrored=False, mirror_axis=DEFAULT_MIRROR_AXIS):
    return f"mirror:{str(mirror_axis).lower()}" if mirrored else MOTION_VARIANT_ORIGINAL


def _reflection_vector(axis=DEFAULT_MIRROR_AXIS):
    axis = str(axis).lower()
    axis_to_index = {"x": 0, "y": 1, "z": 2}
    if axis not in axis_to_index:
        raise ValueError(f'Unsupported mirror axis "{axis}". Expected one of: x, y, z.')

    reflection = np.ones(3, dtype=np.float32)
    reflection[axis_to_index[axis]] = -1.0
    return reflection


def _swap_left_right_name(name):
    name = str(name)
    if name.startswith("Left"):
        return "Right" + name[4:]
    if name.startswith("Right"):
        return "Left" + name[5:]
    if name.endswith("_L"):
        return name[:-2] + "_R"
    if name.endswith("_R"):
        return name[:-2] + "_L"
    if name.endswith(".L"):
        return name[:-2] + ".R"
    if name.endswith(".R"):
        return name[:-2] + ".L"
    return name


def BuildMirrorJointPermutation(joint_names):
    joint_names = [str(name) for name in joint_names]
    name_to_index = {name: index for index, name in enumerate(joint_names)}
    permutation = np.arange(len(joint_names), dtype=np.int32)

    for index, joint_name in enumerate(joint_names):
        partner_name = _swap_left_right_name(joint_name)
        permutation[index] = name_to_index.get(partner_name, index)

    return permutation


def _reflect_rotations(rotations, reflection):
    rotations = np.asarray(rotations, dtype=np.float32)
    xforms = quat.to_xform(rotations)
    broadcast_shape = (1,) * (xforms.ndim - 2)
    row_reflection = reflection.reshape(broadcast_shape + (3, 1))
    col_reflection = reflection.reshape(broadcast_shape + (1, 3))
    mirrored_xforms = xforms * row_reflection * col_reflection
    return quat.normalize(quat.from_xform(mirrored_xforms)).astype(np.float32)


def MirrorPoseArrays(global_positions, global_rotations, parents, joint_names, axis=DEFAULT_MIRROR_AXIS):
    reflection = _reflection_vector(axis)
    permutation = BuildMirrorJointPermutation(joint_names)

    mirrored_global_positions = (
        np.asarray(global_positions, dtype=np.float32)[:, permutation, :] * reflection
    ).astype(np.float32)
    mirrored_global_rotations = _reflect_rotations(
        np.asarray(global_rotations, dtype=np.float32)[:, permutation, :],
        reflection,
    )
    mirrored_global_rotations = quat.unroll(mirrored_global_rotations).astype(np.float32)

    mirrored_local_rotations, mirrored_local_positions = quat.ik(
        mirrored_global_rotations,
        mirrored_global_positions,
        np.asarray(parents, dtype=np.int32),
    )
    mirrored_local_rotations = quat.unroll(mirrored_local_rotations).astype(np.float32)
    mirrored_local_positions = mirrored_local_positions.astype(np.float32)

    mirrored_global_rotations, mirrored_global_positions = quat.fk(
        mirrored_local_rotations,
        mirrored_local_positions,
        np.asarray(parents, dtype=np.int32),
    )

    return {
        "local_positions": mirrored_local_positions.astype(np.float32),
        "local_rotations": mirrored_local_rotations.astype(np.float32),
        "global_positions": mirrored_global_positions.astype(np.float32),
        "global_rotations": quat.unroll(mirrored_global_rotations).astype(np.float32),
        "joint_permutation": permutation,
        "reflection": reflection,
    }


def MirrorBVHAnimation(animation, axis=DEFAULT_MIRROR_AXIS):
    raw_data = dict(animation.raw_data)
    mirrored_pose = MirrorPoseArrays(
        animation.global_positions,
        animation.global_rotations,
        animation.parents,
        raw_data.get("names", []),
        axis=axis,
    )
    raw_data["mirror_axis"] = str(axis).lower()
    raw_data["mirror_joint_permutation"] = mirrored_pose["joint_permutation"]

    return replace(
        animation,
        raw_data=raw_data,
        local_positions=mirrored_pose["local_positions"],
        local_rotations=mirrored_pose["local_rotations"],
        global_positions=mirrored_pose["global_positions"],
        global_rotations=mirrored_pose["global_rotations"],
    )


def MirrorMotionResources(motion, axis=DEFAULT_MIRROR_AXIS):
    mirrored_animation = MirrorBVHAnimation(motion.bvh_animation, axis=axis)
    return replace(
        motion,
        clip_name=motion.clip_name + " [mirror]",
        bvh_animation=mirrored_animation,
        parents=mirrored_animation.parents,
        global_positions=mirrored_animation.global_positions,
        global_rotations=mirrored_animation.global_rotations,
        motion_variant=BuildMotionVariantKey(True, axis),
        mirrored=True,
        mirror_axis=str(axis).lower(),
        base_pose_source=None,
        bootstrap_contact_data=None,
        terrain_provider=None,
        contact_data=None,
        body_proxy_layout=None,
        terrain_model=None,
        terrain_height_grid=None,
        motion_root_trajectory=None,
        terrain_adapted_root_trajectory=None,
        pose_source=None,
        label_result=None,
        terrain_sample_normals=None,
    )
