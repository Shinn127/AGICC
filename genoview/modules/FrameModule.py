from pyray import Vector3
from raylib import *
from raylib.defines import *

from genoview.State import FrameState
from genoview.modules.ContactModule import BuildBodyProxyFrame, ComputeTerrainPenetrationFrame
from genoview.modules.CharacterModel import UpdateModelPoseFromNumpyArrays
from genoview.modules.FeatureModule import EnsureClipFeature, IsClipFeatureReady
from genoview.modules.PoseModule import BuildLocalPose, ComputePosePositionError, ReconstructPoseWorldSpace
from genoview.modules.RootModule import (
    ROOT_JOINT_INDEX,
    BuildRootLocalTrajectory,
    BuildTerrainAdaptedRootTrajectoryDisplay,
)
from genoview.utils.DebugDraw import OffsetPositions


def ApplyLabelResultToFrameState(frame_state, label_result):
    frame_state.clip_prior = label_result.clip_prior
    frame_state.auto_labels = label_result.auto_labels
    frame_state.auto_segments = label_result.auto_segments
    frame_state.final_labels = label_result.final_labels
    frame_state.final_segments = label_result.final_segments
    frame_state.soft_weights = label_result.soft_weights
    frame_state.current_auto_label = (
        str(label_result.auto_labels[frame_state.animation_frame])
        if label_result.auto_labels is not None and len(label_result.auto_labels) > 0 else
        "other"
    )
    frame_state.current_final_label = (
        str(label_result.final_labels[frame_state.animation_frame])
        if label_result.final_labels is not None and len(label_result.final_labels) > 0 else
        frame_state.current_auto_label
    )
    return frame_state


def BuildBaseFrameState(app, animation_frame):
    scene = app.scene
    motion = app.motion

    UpdateModelPoseFromNumpyArrays(
        scene.geno_model,
        scene.bind_pos,
        scene.bind_rot,
        motion.global_positions[animation_frame],
        motion.global_rotations[animation_frame],
    )

    root_position = motion.global_positions[animation_frame][ROOT_JOINT_INDEX]

    frame_state = FrameState(
        animation_frame=animation_frame,
        frame_count=motion.bvh_animation.frame_count,
        clip_name=motion.clip_name,
        clip_prior="labels off",
        current_auto_label="labels off",
        current_final_label="labels off",
        auto_labels=None,
        auto_segments=None,
        final_labels=None,
        final_segments=None,
        soft_weights=None,
        hip_position=Vector3(*root_position),
    )
    if motion.label_result is not None:
        ApplyLabelResultToFrameState(frame_state, motion.label_result)
    return frame_state


def EnsureRootTrajectoryFrameState(app, frame_state):
    if frame_state.root_trajectory is not None:
        return frame_state
    if not IsClipFeatureReady(app, "motion_root_trajectory") or not IsClipFeatureReady(app, "terrain_adapted_root_trajectory"):
        return frame_state

    motion = app.motion
    root_trajectory = BuildRootLocalTrajectory(
        EnsureClipFeature(app, "motion_root_trajectory"),
        frame_state.animation_frame,
        sampleOffsets=motion.trajectory_sample_offsets,
    )
    frame_state.root_trajectory = root_trajectory
    frame_state.terrain_root_trajectory_display = BuildTerrainAdaptedRootTrajectoryDisplay(
        root_trajectory,
        EnsureClipFeature(app, "terrain_adapted_root_trajectory"),
        heightOffset=0.02,
        alignDirectionsToTerrain=True,
        alignVelocitiesToTerrain=True,
    )

    return frame_state


def EnsurePoseFrameState(app, frame_state):
    if frame_state.local_pose is not None:
        return frame_state
    if not IsClipFeatureReady(app, "pose_source") or not IsClipFeatureReady(app, "motion_root_trajectory"):
        return frame_state

    scene = app.scene
    motion = app.motion
    debug = app.debug

    local_pose = BuildLocalPose(
        EnsureClipFeature(app, "pose_source"),
        EnsureClipFeature(app, "motion_root_trajectory"),
        frame_state.animation_frame,
        dt=motion.bvh_frame_time,
    )
    reconstructed_pose_world = ReconstructPoseWorldSpace(
        local_pose,
        integrateRootMotion=debug.integrate_root_motion_ptr[0],
        dt=motion.bvh_frame_time,
    )
    local_pose_positions_offset = OffsetPositions(local_pose["local_positions"], debug.local_debug_origin)

    pose_model = EnsureClipFeature(app, "pose_model") if debug.draw_reconstructed_pose_ptr[0] else None
    if debug.draw_reconstructed_pose_ptr[0] and debug.draw_pose_model_local_ptr[0]:
        UpdateModelPoseFromNumpyArrays(
            pose_model,
            scene.bind_pos,
            scene.bind_rot,
            local_pose_positions_offset,
            local_pose["local_rotations"],
        )
    elif debug.draw_reconstructed_pose_ptr[0]:
        UpdateModelPoseFromNumpyArrays(
            pose_model,
            scene.bind_pos,
            scene.bind_rot,
            reconstructed_pose_world["world_positions"],
            reconstructed_pose_world["world_rotations"],
        )

    frame_state.local_pose = local_pose
    frame_state.local_pose_positions_offset = local_pose_positions_offset
    frame_state.reconstructed_pose_world = reconstructed_pose_world

    pose_focus_position = (
        local_pose_positions_offset[ROOT_JOINT_INDEX]
        if debug.draw_pose_model_local_ptr[0] else
        reconstructed_pose_world["world_positions"][ROOT_JOINT_INDEX]
    )
    frame_state.hip_position = Vector3(*pose_focus_position)

    return frame_state


def EnsurePoseErrorFrameState(app, frame_state):
    if frame_state.pose_comparison_positions is not None:
        return frame_state
    if not IsClipFeatureReady(app, "pose_source"):
        return frame_state

    EnsurePoseFrameState(app, frame_state)
    motion = app.motion
    debug = app.debug
    pose_comparison_frame = (
        min(frame_state.animation_frame + 1, motion.bvh_animation.frame_count - 1)
        if debug.integrate_root_motion_ptr[0] else
        frame_state.animation_frame
    )
    pose_comparison_positions = motion.global_positions[pose_comparison_frame]
    pose_position_error_mean, pose_position_error_max = ComputePosePositionError(
        pose_comparison_positions,
        frame_state.reconstructed_pose_world["world_positions"],
    )

    frame_state.pose_comparison_positions = pose_comparison_positions
    frame_state.pose_error_label = b"Pred Err(+1)" if debug.integrate_root_motion_ptr[0] else b"Recon Err"
    frame_state.pose_position_error_mean = pose_position_error_mean
    frame_state.pose_position_error_max = pose_position_error_max

    return frame_state


def EnsureContactFrameState(app, frame_state):
    if frame_state.frame_contacts is not None and (
        frame_state.pose_contact_positions is not None or
        not app.debug.draw_reconstructed_pose_ptr[0]
    ):
        return frame_state
    if not IsClipFeatureReady(app, "contact_data"):
        return frame_state

    contact_data = EnsureClipFeature(app, "contact_data")
    contact_indices = contact_data["joint_indices"]
    frame_state.frame_contacts = contact_data["contacts_filtered"][frame_state.animation_frame]
    frame_state.bvh_contact_positions = contact_data["positions"][frame_state.animation_frame]

    if app.debug.draw_reconstructed_pose_ptr[0]:
        EnsurePoseFrameState(app, frame_state)
        frame_state.pose_contact_positions = (
            frame_state.local_pose_positions_offset[contact_indices]
            if app.debug.draw_pose_model_local_ptr[0] else
            frame_state.reconstructed_pose_world["world_positions"][contact_indices]
        )

    return frame_state


def EnsureBootstrapContactFrameState(app, frame_state):
    if frame_state.bootstrap_frame_contacts is not None and (
        frame_state.bootstrap_pose_contact_positions is not None or
        not app.debug.draw_reconstructed_pose_ptr[0]
    ):
        return frame_state
    if not IsClipFeatureReady(app, "bootstrap_contacts"):
        return frame_state

    bootstrap_contact_data = EnsureClipFeature(app, "bootstrap_contacts")
    bootstrap_contact_indices = bootstrap_contact_data["joint_indices"]
    frame_state.bootstrap_frame_contacts = bootstrap_contact_data["contacts_filtered"][frame_state.animation_frame]
    frame_state.bootstrap_bvh_contact_positions = bootstrap_contact_data["positions"][frame_state.animation_frame]

    if app.debug.draw_reconstructed_pose_ptr[0]:
        EnsurePoseFrameState(app, frame_state)
        frame_state.bootstrap_pose_contact_positions = (
            frame_state.local_pose_positions_offset[bootstrap_contact_indices]
            if app.debug.draw_pose_model_local_ptr[0] else
            frame_state.reconstructed_pose_world["world_positions"][bootstrap_contact_indices]
        )

    return frame_state


def EnsureTerrainFocusFrameState(app, frame_state):
    if frame_state.terrain_height_at_focus is not None:
        return frame_state
    if not IsClipFeatureReady(app, "terrain_provider") or not IsClipFeatureReady(app, "terrain_adapted_root_trajectory"):
        return frame_state

    terrain_query_position = app.motion.global_positions[frame_state.animation_frame][ROOT_JOINT_INDEX]
    if frame_state.reconstructed_pose_world is not None:
        terrain_query_position = frame_state.reconstructed_pose_world["world_positions"][ROOT_JOINT_INDEX]
    terrain_provider = EnsureClipFeature(app, "terrain_provider")
    terrain_adapted_root_trajectory = EnsureClipFeature(app, "terrain_adapted_root_trajectory")
    frame_state.terrain_height_at_focus = terrain_provider.sample_height(terrain_query_position)
    frame_state.terrain_normal_at_focus = terrain_adapted_root_trajectory["terrain_normals"][frame_state.animation_frame]

    return frame_state


def EnsureBodyProxyFrameState(app, frame_state):
    if frame_state.body_proxy_positions is not None:
        return frame_state
    if not IsClipFeatureReady(app, "body_proxy_layout"):
        return frame_state

    body_proxy_frame = BuildBodyProxyFrame(
        app.motion.global_positions[frame_state.animation_frame],
        EnsureClipFeature(app, "body_proxy_layout"),
    )
    frame_state.body_proxy_positions = body_proxy_frame["positions"]
    frame_state.body_proxy_radii = body_proxy_frame["radii"]
    frame_state.penetration_frame = None

    return frame_state


def EnsurePenetrationFrameState(app, frame_state):
    if frame_state.penetration_frame is not None:
        return frame_state
    if not IsClipFeatureReady(app, "terrain_provider") or not IsClipFeatureReady(app, "body_proxy_layout"):
        return frame_state

    EnsureBodyProxyFrameState(app, frame_state)
    penetration_frame = ComputeTerrainPenetrationFrame(
        {
            "positions": frame_state.body_proxy_positions,
            "radii": frame_state.body_proxy_radii,
        },
        EnsureClipFeature(app, "terrain_provider"),
    )
    frame_state.penetration_frame = penetration_frame
    frame_state.penetration_count = penetration_frame["penetration_count"]
    frame_state.max_penetration_depth = penetration_frame["max_penetration"]

    return frame_state


def BuildFrameState(app, animation_frame):
    frame_state = BuildBaseFrameState(app, animation_frame)
    debug = app.debug

    if (debug.draw_reconstructed_pose_ptr[0] or debug.draw_pose_model_local_ptr[0]) and IsClipFeatureReady(app, "pose_source"):
        EnsurePoseFrameState(app, frame_state)
    if debug.draw_reconstruction_error_ptr[0] and IsClipFeatureReady(app, "pose_source"):
        EnsurePoseErrorFrameState(app, frame_state)
    if debug.draw_root_trajectory_ptr[0] and IsClipFeatureReady(app, "terrain_adapted_root_trajectory"):
        EnsureRootTrajectoryFrameState(app, frame_state)
    if debug.draw_contacts_ptr[0] and IsClipFeatureReady(app, "contact_data"):
        EnsureContactFrameState(app, frame_state)
    if debug.draw_bootstrap_contacts_ptr[0] and IsClipFeatureReady(app, "bootstrap_contacts"):
        EnsureBootstrapContactFrameState(app, frame_state)
    if debug.draw_body_proxy_ptr[0] and IsClipFeatureReady(app, "body_proxy_layout"):
        EnsureBodyProxyFrameState(app, frame_state)
    if (
        debug.draw_terrain_penetration_ptr[0] and
        IsClipFeatureReady(app, "body_proxy_layout") and
        IsClipFeatureReady(app, "terrain_provider")
    ):
        EnsurePenetrationFrameState(app, frame_state)

    return frame_state


def UpdateTracking(app, frame_state):
    render = app.render

    render.shadow_light.target = Vector3(frame_state.hip_position.x, 0.0, frame_state.hip_position.z)
    render.shadow_light.position = Vector3Add(
        render.shadow_light.target,
        Vector3Scale(render.light_dir, -5.0),
    )

    mouse_wheel = 0.0 if getattr(app.debug, "clip_dropdown_open", False) else GetMouseWheelMove()
    app.camera.update(
        Vector3(frame_state.hip_position.x, 0.75, frame_state.hip_position.z),
        GetMouseDelta().x if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(0) else 0.0,
        GetMouseDelta().y if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(0) else 0.0,
        GetMouseDelta().x if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(1) else 0.0,
        GetMouseDelta().y if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(1) else 0.0,
        mouse_wheel,
        GetFrameTime(),
    )
