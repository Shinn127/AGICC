import cffi
from pyray import Color, Rectangle, Vector3
from raylib import *

from genoview.State import DebugState
from genoview.modules.LabelModule import DEFAULT_TRANSITION_FRAMES
from genoview.modules.PlaybackController import PlaybackController
from genoview.utils.DebugDraw import (
    DrawBodyProxyFrame,
    DrawContactStates,
    DrawPoseReconstructionError,
    DrawRootTrajectoryDebug,
    DrawSkeleton,
    DrawTerrainPenetrationFrame,
    OffsetPositions,
)


ffi = cffi.FFI()


def _make_bool_ptr(value):
    ptr = ffi.new("bool*")
    ptr[0] = bool(value)
    return ptr


def CreateDebugState(frame_count, frame_time):
    return DebugState(
        label_module_ptr=_make_bool_ptr(False),
        draw_bone_transforms_ptr=_make_bool_ptr(False),
        draw_flat_ground_ptr=_make_bool_ptr(True),
        draw_terrain_mesh_ptr=_make_bool_ptr(False),
        draw_root_trajectory_ptr=_make_bool_ptr(False),
        draw_trajectory_directions_ptr=_make_bool_ptr(False),
        draw_trajectory_velocity_ptr=_make_bool_ptr(False),
        draw_contacts_ptr=_make_bool_ptr(False),
        draw_bootstrap_contacts_ptr=_make_bool_ptr(False),
        draw_terrain_samples_ptr=_make_bool_ptr(False),
        draw_terrain_normals_ptr=_make_bool_ptr(False),
        draw_body_proxy_ptr=_make_bool_ptr(False),
        draw_terrain_penetration_ptr=_make_bool_ptr(False),
        draw_reconstructed_pose_ptr=_make_bool_ptr(False),
        draw_pose_model_local_ptr=_make_bool_ptr(False),
        draw_reconstruction_error_ptr=_make_bool_ptr(False),
        integrate_root_motion_ptr=_make_bool_ptr(False),
        selected_timeline_mode="final",
        local_debug_origin=Vector3(-2.0, 0.0, 0.0),
        pose_model_color=Color(110, 190, 255, 255),
        selected_action_label="walk",
        transition_width=DEFAULT_TRANSITION_FRAMES,
        annotation_status="Auto-loaded labels",
        playback=PlaybackController(frame_count, frame_time),
    )


def _count_enabled_modules(debug):
    module_ptrs = (
        debug.label_module_ptr,
        debug.draw_terrain_mesh_ptr,
        debug.draw_root_trajectory_ptr,
        debug.draw_contacts_ptr,
        debug.draw_bootstrap_contacts_ptr,
        debug.draw_terrain_samples_ptr,
        debug.draw_terrain_normals_ptr,
        debug.draw_body_proxy_ptr,
        debug.draw_terrain_penetration_ptr,
        debug.draw_reconstructed_pose_ptr,
        debug.draw_reconstruction_error_ptr,
        debug.draw_bone_transforms_ptr,
        debug.integrate_root_motion_ptr,
    )
    return sum(1 for module_ptr in module_ptrs if module_ptr[0])


def _draw_module_checkbox(x, y, label, value_ptr):
    GuiCheckBox(Rectangle(x, y, 20, 20), label, value_ptr)


def _get_module_items(debug):
    return (
        (b"Labels / Annotation", debug.label_module_ptr),
        (b"Terrain Mesh", debug.draw_terrain_mesh_ptr),
        (b"Root Trajectory", debug.draw_root_trajectory_ptr),
        (b"Trajectory Directions", debug.draw_trajectory_directions_ptr),
        (b"Trajectory Velocity", debug.draw_trajectory_velocity_ptr),
        (b"Contacts", debug.draw_contacts_ptr),
        (b"Bootstrap Contacts", debug.draw_bootstrap_contacts_ptr),
        (b"Terrain Samples", debug.draw_terrain_samples_ptr),
        (b"Terrain Normals", debug.draw_terrain_normals_ptr),
        (b"Body Proxy", debug.draw_body_proxy_ptr),
        (b"Penetration", debug.draw_terrain_penetration_ptr),
        (b"Blue Geno", debug.draw_reconstructed_pose_ptr),
        (b"Blue Geno Local", debug.draw_pose_model_local_ptr),
        (b"Reconstruction Error", debug.draw_reconstruction_error_ptr),
        (b"Bone Transforms", debug.draw_bone_transforms_ptr),
        (b"Integrate Root Motion", debug.integrate_root_motion_ptr),
    )


def GetModuleDropdownHeight(app):
    item_gap = 24
    panel_padding = 16
    button_gap = 28
    return button_gap + panel_padding + len(_get_module_items(app.debug)) * item_gap


def DrawModuleDropdown(app, x, y, width):
    debug = app.debug
    enabled_count = _count_enabled_modules(debug)
    button_label = ("Modules: %d enabled" % enabled_count).encode("utf-8")
    if GuiButton(Rectangle(x, y, width, 24), button_label):
        debug.module_dropdown_open = not debug.module_dropdown_open

    if not debug.module_dropdown_open:
        return

    item_gap = 24
    panel_height = 16 + len(_get_module_items(debug)) * item_gap
    DrawRectangleRec(Rectangle(x, y + 28, width, panel_height), Fade(RAYWHITE, 0.96))
    DrawRectangleLinesEx(Rectangle(x, y + 28, width, panel_height), 1.0, DARKGRAY)

    item_x = x + 10
    item_y = y + 38
    for index, (label, value_ptr) in enumerate(_get_module_items(debug)):
        _draw_module_checkbox(item_x, item_y + index * item_gap, label, value_ptr)


def _draw_async_progress(app, x, y, width):
    progress_state = app.async_progress
    if not progress_state.active:
        return y

    row_height = 20
    progress = max(0.0, min(1.0, float(progress_state.progress)))
    title = progress_state.title or "Preparing"
    detail = progress_state.detail or "Working..."
    bar_y = y + row_height
    bar_height = 14

    GuiLabel(Rectangle(x, y, width, row_height), title.encode("utf-8"))
    DrawRectangleLinesEx(Rectangle(x, bar_y, width, bar_height), 1.0, GRAY)

    if progress_state.indeterminate:
        segment_width = max(24.0, width * 0.25)
        segment_x = x + (width - segment_width) * (GetTime() * 0.85 % 1.0)
        DrawRectangleRec(Rectangle(segment_x + 1, bar_y + 1, segment_width - 2, bar_height - 2), SKYBLUE)
    else:
        DrawRectangleRec(Rectangle(x + 1, bar_y + 1, max(0.0, (width - 2) * progress), bar_height - 2), SKYBLUE)

    detail_y = bar_y + bar_height + 2
    GuiLabel(Rectangle(x, detail_y, width, row_height), detail.encode("utf-8"))
    return detail_y + row_height + 4


def GetAsyncProgressHeight(app):
    return 58 if app.async_progress.active else 0


def DrawMotionDebugOverlay(
    app,
    frame_state,
    draw_terrain_debug_overlay,
    ensure_pose_frame,
    ensure_pose_error_frame,
    ensure_root_trajectory_frame,
    ensure_contact_frame,
    ensure_bootstrap_contact_frame,
    ensure_body_proxy_frame,
    ensure_penetration_frame):

    debug = app.debug
    motion = app.motion

    BeginMode3D(app.camera.cam3d)

    if debug.draw_bone_transforms_ptr[0]:
        DrawSkeleton(
            motion.global_positions[frame_state.animation_frame],
            motion.global_rotations[frame_state.animation_frame],
            motion.parents,
            GRAY,
        )

        if debug.draw_reconstructed_pose_ptr[0]:
            ensure_pose_frame(app, frame_state)
            if frame_state.reconstructed_pose_world is not None and debug.draw_pose_model_local_ptr[0]:
                DrawSkeleton(
                    frame_state.local_pose_positions_offset,
                    frame_state.local_pose["local_rotations"],
                    motion.parents,
                    debug.pose_model_color,
                )
            elif frame_state.reconstructed_pose_world is not None:
                DrawSkeleton(
                    frame_state.reconstructed_pose_world["world_positions"],
                    frame_state.reconstructed_pose_world["world_rotations"],
                    motion.parents,
                    debug.pose_model_color,
                )

    if debug.draw_root_trajectory_ptr[0]:
        ensure_root_trajectory_frame(app, frame_state)
        if frame_state.root_trajectory is not None and frame_state.terrain_root_trajectory_display is not None:
            DrawRootTrajectoryDebug(
                frame_state.terrain_root_trajectory_display["world_positions"],
                frame_state.terrain_root_trajectory_display["world_directions"],
                frame_state.terrain_root_trajectory_display["world_velocities"],
                frame_state.root_trajectory["sample_offsets"],
                drawDirection=debug.draw_trajectory_directions_ptr[0],
                drawVelocity=debug.draw_trajectory_velocity_ptr[0],
            )

        if debug.draw_reconstructed_pose_ptr[0] and debug.draw_pose_model_local_ptr[0] and frame_state.root_trajectory is not None:
            DrawRootTrajectoryDebug(
                OffsetPositions(frame_state.root_trajectory["local_positions"], debug.local_debug_origin),
                frame_state.root_trajectory["local_directions"],
                frame_state.root_trajectory["local_velocities"],
                frame_state.root_trajectory["sample_offsets"],
                drawDirection=debug.draw_trajectory_directions_ptr[0],
                drawVelocity=debug.draw_trajectory_velocity_ptr[0],
            )

    if debug.draw_contacts_ptr[0]:
        ensure_contact_frame(app, frame_state)
        if frame_state.frame_contacts is not None:
            DrawContactStates(frame_state.bvh_contact_positions, frame_state.frame_contacts)
            if debug.draw_reconstructed_pose_ptr[0] and frame_state.pose_contact_positions is not None:
                DrawContactStates(frame_state.pose_contact_positions, frame_state.frame_contacts)

    if debug.draw_bootstrap_contacts_ptr[0]:
        ensure_bootstrap_contact_frame(app, frame_state)
        if frame_state.bootstrap_frame_contacts is not None:
            DrawContactStates(
                frame_state.bootstrap_bvh_contact_positions,
                frame_state.bootstrap_frame_contacts,
                activeColor=Color(150, 110, 60, 255),
                inactiveColor=Color(210, 190, 160, 255),
                activeSize=0.04,
                inactiveSize=0.04,
            )
            if debug.draw_reconstructed_pose_ptr[0] and frame_state.bootstrap_pose_contact_positions is not None:
                DrawContactStates(
                    frame_state.bootstrap_pose_contact_positions,
                    frame_state.bootstrap_frame_contacts,
                    activeColor=Color(150, 110, 60, 255),
                    inactiveColor=Color(210, 190, 160, 255),
                    activeSize=0.04,
                    inactiveSize=0.04,
                )

    draw_terrain_debug_overlay(app)

    if debug.draw_body_proxy_ptr[0]:
        ensure_body_proxy_frame(app, frame_state)
        if frame_state.body_proxy_positions is not None:
            DrawBodyProxyFrame(frame_state.body_proxy_positions, frame_state.body_proxy_radii)

    if debug.draw_terrain_penetration_ptr[0]:
        ensure_penetration_frame(app, frame_state)
        if frame_state.penetration_frame is not None:
            DrawTerrainPenetrationFrame(frame_state.body_proxy_positions, frame_state.penetration_frame)

    if debug.draw_reconstruction_error_ptr[0]:
        ensure_pose_error_frame(app, frame_state)
        if frame_state.pose_comparison_positions is not None and frame_state.reconstructed_pose_world is not None:
            DrawPoseReconstructionError(
                frame_state.pose_comparison_positions,
                frame_state.reconstructed_pose_world["world_positions"],
                MAGENTA,
            )

    EndMode3D()


def DrawRenderingOptionsPanel(
    app,
    frame_state,
    panel_y,
    panel_height,
    draw_terrain_metrics,
    ensure_terrain_focus,
    ensure_penetration_frame,
    ensure_pose_error_frame):

    debug = app.debug
    screen_width = app.screen_width
    content_x = screen_width - 250
    content_y = panel_y + 35
    row_height = 20
    row_gap = 2

    content_height = 35 + row_height
    content_height += GetAsyncProgressHeight(app)
    terrain_metrics_enabled = (
        debug.draw_terrain_mesh_ptr[0] or
        debug.draw_root_trajectory_ptr[0] or
        debug.draw_terrain_samples_ptr[0] or
        debug.draw_terrain_normals_ptr[0] or
        debug.draw_terrain_penetration_ptr[0]
    )
    if terrain_metrics_enabled:
        content_height += 5 * row_height
    if debug.draw_terrain_penetration_ptr[0]:
        content_height += row_height + row_gap
    if debug.draw_reconstruction_error_ptr[0]:
        content_height += row_height + row_gap
    if debug.module_dropdown_open:
        content_height = max(content_height, GetModuleDropdownHeight(app))

    panel_height = max(panel_height, content_height + 20)
    GuiGroupBox(Rectangle(screen_width - 260, panel_y, 240, panel_height), b"Rendering")
    GuiLabel(Rectangle(content_x, content_y, 220, row_height), b"Flat Ground: On")
    content_y += row_height + row_gap
    content_y = _draw_async_progress(app, content_x, content_y, 220)

    if terrain_metrics_enabled:
        content_y = draw_terrain_metrics(app, frame_state, ensure_terrain_focus, content_y) + row_gap
    if debug.draw_terrain_penetration_ptr[0]:
        ensure_penetration_frame(app, frame_state)
        if frame_state.penetration_frame is not None:
            GuiLabel(
                Rectangle(content_x, content_y, 220, row_height),
                b"Pen: %d max %.4f" % (
                    frame_state.penetration_count,
                    frame_state.max_penetration_depth,
                ),
            )
            content_y += row_height + row_gap

    if debug.draw_reconstruction_error_ptr[0]:
        ensure_pose_error_frame(app, frame_state)
        if frame_state.pose_comparison_positions is not None:
            GuiLabel(
                Rectangle(content_x, content_y, 220, row_height),
                b"%s: mean %.6f max %.6f" % (
                    frame_state.pose_error_label,
                    frame_state.pose_position_error_mean,
                    frame_state.pose_position_error_max,
                ),
            )

    DrawModuleDropdown(app, content_x, panel_y + 10, 220)
