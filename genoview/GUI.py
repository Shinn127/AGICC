from pathlib import Path

import numpy as np

from pyray import Color, Rectangle, Vector2
from raylib import *
from raylib.defines import *

from genoview.modules.AsyncModule import RequestClipSwitch, RequestFeatureLoad
from genoview.modules.FeatureModule import EnsureClipFeature, IsClipFeatureReady, SyncFeatureMounts
from genoview.modules.FrameModule import (
    EnsurePenetrationFrameState,
    EnsurePoseErrorFrameState,
    EnsureTerrainFocusFrameState,
)
from genoview.modules.LabelModule import (
    ACTION_LABELS,
    ApplyManualLabelRange,
    ApplyTransitionWidthRange,
    ClearManualLabelRange,
    ClearTransitionWidthRange,
    ExportCompiledLabels,
    LoadLabelAnnotations,
    ResetManualLabels,
    SaveLabelAnnotations,
)
from genoview.modules.MotionDebugModule import DrawRenderingOptionsPanel
from genoview.modules.TerrainModule import DrawTerrainRenderPanelMetrics


def _action_label_color(label):
    if label == "idle":
        return Color(150, 150, 150, 255)
    if label == "walk":
        return Color(90, 155, 235, 255)
    if label == "run":
        return Color(235, 115, 65, 255)
    if label == "jump":
        return Color(240, 200, 70, 255)
    if label == "fall":
        return Color(185, 75, 75, 255)
    if label == "ground":
        return Color(130, 105, 75, 255)
    if label == "get_up":
        return Color(95, 180, 120, 255)
    if label == "transition":
        return Color(165, 120, 215, 255)
    return Color(120, 120, 120, 255)


def _draw_action_label_timeline(frame_count, current_frame, bounds, segments=None, selection_start=None, selection_end=None):
    frame_count = int(frame_count)
    if frame_count <= 0:
        DrawRectangleLinesEx(bounds, 1.0, DARKGRAY)
        return

    DrawRectangleRec(bounds, Fade(RAYWHITE, 0.9))
    DrawRectangleLinesEx(bounds, 1.0, DARKGRAY)

    width = max(float(bounds.width), 1.0)
    if segments is not None:
        for segment in segments:
            start_x = bounds.x + float(segment.start_frame) / max(frame_count, 1) * width
            end_x = bounds.x + float(segment.end_frame + 1) / max(frame_count, 1) * width
            segment_width = max(1.0, end_x - start_x)
            DrawRectangleRec(
                Rectangle(start_x, bounds.y, segment_width, bounds.height),
                _action_label_color(str(segment.label)),
            )

    if selection_start is not None and selection_end is not None:
        selection_a = int(min(selection_start, selection_end))
        selection_b = int(max(selection_start, selection_end))
        selection_x = bounds.x + float(selection_a) / max(frame_count, 1) * width
        selection_width = max(1.0, float(selection_b - selection_a + 1) / max(frame_count, 1) * width)
        DrawRectangleRec(
            Rectangle(selection_x, bounds.y, selection_width, bounds.height),
            Fade(WHITE, 0.28),
        )
        DrawRectangleLinesEx(
            Rectangle(selection_x, bounds.y, selection_width, bounds.height),
            1.0,
            BLACK,
        )

    current_x = bounds.x + float(current_frame) / max(frame_count - 1, 1) * width
    DrawLineEx(
        Vector2(current_x, bounds.y - 2),
        Vector2(current_x, bounds.y + bounds.height + 2),
        2.0,
        BLACK,
    )


def _draw_soft_label_timeline(frame_count, current_frame, bounds, soft_weights=None, selection_start=None, selection_end=None):
    frame_count = int(frame_count)
    if frame_count <= 0 or soft_weights is None:
        DrawRectangleLinesEx(bounds, 1.0, DARKGRAY)
        return

    soft_weights = np.asarray(soft_weights, dtype=np.float32)
    DrawRectangleRec(bounds, Fade(RAYWHITE, 0.9))
    DrawRectangleLinesEx(bounds, 1.0, DARKGRAY)

    width = max(int(bounds.width), 1)
    for pixel_index in range(width):
        frame_alpha = 0.0 if width <= 1 else pixel_index / max(width - 1, 1)
        frame_index = int(round(frame_alpha * max(frame_count - 1, 0)))
        frame_index = min(max(frame_index, 0), frame_count - 1)
        weights = soft_weights[frame_index]
        color_r = 0.0
        color_g = 0.0
        color_b = 0.0
        color_a = 0.0
        for label_index, label in enumerate(ACTION_LABELS):
            label_color = _action_label_color(label)
            weight = float(weights[label_index])
            color_r += label_color.r * weight
            color_g += label_color.g * weight
            color_b += label_color.b * weight
            color_a += label_color.a * weight
        DrawRectangle(
            int(bounds.x + pixel_index),
            int(bounds.y),
            1,
            int(bounds.height),
            Color(int(color_r), int(color_g), int(color_b), max(120, int(color_a))),
        )

    if selection_start is not None and selection_end is not None:
        selection_a = int(min(selection_start, selection_end))
        selection_b = int(max(selection_start, selection_end))
        selection_x = bounds.x + float(selection_a) / max(frame_count, 1) * float(bounds.width)
        selection_width = max(1.0, float(selection_b - selection_a + 1) / max(frame_count, 1) * float(bounds.width))
        DrawRectangleRec(
            Rectangle(selection_x, bounds.y, selection_width, bounds.height),
            Fade(WHITE, 0.24),
        )
        DrawRectangleLinesEx(
            Rectangle(selection_x, bounds.y, selection_width, bounds.height),
            1.0,
            BLACK,
        )

    current_x = bounds.x + float(current_frame) / max(frame_count - 1, 1) * float(bounds.width)
    DrawLineEx(
        Vector2(current_x, bounds.y - 2),
        Vector2(current_x, bounds.y + bounds.height + 2),
        2.0,
        BLACK,
    )


def _draw_action_label_button(bounds, label, selected=False):
    was_pressed = GuiButton(bounds, b"")
    base_color = _action_label_color(label)
    fill_color = Fade(base_color, 0.78 if selected else 0.55)
    inner_padding = 3.0
    inner_bounds = Rectangle(
        bounds.x + inner_padding,
        bounds.y + inner_padding,
        max(1.0, bounds.width - 2.0 * inner_padding),
        max(1.0, bounds.height - 2.0 * inner_padding),
    )

    try:
        text_size = int(GuiGetStyle(DEFAULT, TEXT_SIZE))
        text_color = GetColor(GuiGetStyle(BUTTON, TEXT_COLOR_NORMAL))
    except NameError:
        text_size = 10
        text_color = DARKGRAY

    text_bytes = label.encode("utf-8")
    text_width = MeasureText(text_bytes, text_size)
    text_x = int(bounds.x + max(0.0, 0.5 * (bounds.width - text_width)))
    text_y = int(bounds.y + max(0.0, 0.5 * (bounds.height - text_size)) + 1)

    DrawRectangleRec(inner_bounds, fill_color)
    if selected:
        DrawRectangleLinesEx(
            Rectangle(bounds.x + 1.0, bounds.y + 1.0, bounds.width - 2.0, bounds.height - 2.0),
            1.0,
            Fade(BLACK, 0.35),
        )
    DrawText(text_bytes, text_x, text_y, text_size, text_color)

    return was_pressed


def _clip_display_name(clip_resource):
    clip_path = Path(str(clip_resource))
    if clip_path.parent == Path("."):
        return clip_path.name
    return str(clip_path)


def _draw_clip_dropdown(app, x, y, width, request_clip_index, open_up=False):
    debug = app.debug
    clips = app.clip_resources
    if not clips:
        GuiButton(Rectangle(x, y, width, 24), b"No clips")
        return False

    row_height = 22
    visible_rows = min(8, len(clips))
    current_label = _clip_display_name(clips[app.clip_index])
    if len(current_label) > 42:
        current_label = "..." + current_label[-39:]

    if GuiButton(Rectangle(x, y, width, 24), ("Clip: %s" % current_label).encode("utf-8")):
        debug.clip_dropdown_open = not debug.clip_dropdown_open
        debug.clip_dropdown_scroll = min(
            max(app.clip_index - visible_rows // 2, 0),
            max(0, len(clips) - visible_rows),
        )

    if not debug.clip_dropdown_open:
        return False

    max_scroll = max(0, len(clips) - visible_rows)
    debug.clip_dropdown_scroll = min(max(int(debug.clip_dropdown_scroll), 0), max_scroll)

    panel_height = visible_rows * row_height
    panel_y = y - 4 - panel_height if open_up else y + 28
    panel_rect = Rectangle(x, panel_y, width, panel_height)
    button_rect = Rectangle(x, y, width, 24)
    mouse = GetMousePosition()
    hovering_panel = CheckCollisionPointRec(mouse, panel_rect)
    hovering_button = CheckCollisionPointRec(mouse, button_rect)

    if hovering_panel:
        wheel = GetMouseWheelMove()
        if wheel != 0:
            scroll_delta = -1 if wheel > 0 else 1
            debug.clip_dropdown_scroll = min(
                max(debug.clip_dropdown_scroll + scroll_delta, 0),
                max_scroll,
            )

    DrawRectangleRec(panel_rect, Fade(RAYWHITE, 0.96))
    DrawRectangleLinesEx(panel_rect, 1.0, DARKGRAY)

    start_index = debug.clip_dropdown_scroll
    end_index = min(len(clips), start_index + visible_rows)
    BeginScissorMode(int(panel_rect.x), int(panel_rect.y), int(panel_rect.width), int(panel_rect.height))
    for row_index, clip_index in enumerate(range(start_index, end_index)):
        row_rect = Rectangle(x, panel_y + row_index * row_height, width, row_height)
        is_selected = clip_index == app.clip_index
        is_hovered = CheckCollisionPointRec(mouse, row_rect)
        if is_selected:
            DrawRectangleRec(row_rect, Fade(SKYBLUE, 0.35))
        elif is_hovered:
            DrawRectangleRec(row_rect, Fade(LIGHTGRAY, 0.45))

        label = _clip_display_name(clips[clip_index])
        if len(label) > 48:
            label = "..." + label[-45:]
        DrawText(label.encode("utf-8"), int(row_rect.x + 8), int(row_rect.y + 6), 10, DARKGRAY)

        if is_hovered and IsMouseButtonPressed(0):
            debug.clip_dropdown_open = False
            if clip_index != app.clip_index:
                request_clip_index(app, clip_index)
                EndScissorMode()
                return True
    EndScissorMode()

    if not hovering_button and not hovering_panel and IsMouseButtonPressed(0):
        debug.clip_dropdown_open = False

    return False


def _draw_clip_variant_row(app, x, y, width, request_clip_index, request_mirror_toggle, open_up=False):
    mirror_width = 104
    gap = 6
    dropdown_width = max(120, width - mirror_width - gap)
    if _draw_clip_dropdown(app, x, y, dropdown_width, request_clip_index, open_up=open_up):
        return True

    mirror_label = b"Mirror: On" if app.mirror_enabled else b"Mirror: Off"
    if GuiButton(Rectangle(x + dropdown_width + gap, y, mirror_width, 24), mirror_label):
        request_mirror_toggle(app)
        return True

    return False


def HandleLabelFeatureShortcuts(app, label_result, save_annotations, load_annotations, export_labels):
    debug = app.debug
    selection_start, selection_end = debug.playback.selection_range

    number_keys = [
        globals().get("KEY_ONE", ord("1")),
        globals().get("KEY_TWO", ord("2")),
        globals().get("KEY_THREE", ord("3")),
        globals().get("KEY_FOUR", ord("4")),
        globals().get("KEY_FIVE", ord("5")),
        globals().get("KEY_SIX", ord("6")),
        globals().get("KEY_SEVEN", ord("7")),
        globals().get("KEY_EIGHT", ord("8")),
        globals().get("KEY_NINE", ord("9")),
    ]
    for label_index, key in enumerate(number_keys):
        if label_index < len(ACTION_LABELS) and IsKeyPressed(key):
            debug.selected_action_label = ACTION_LABELS[label_index]

    if IsKeyPressed(globals().get("KEY_I", ord("I"))):
        debug.playback.mark_selection_start()
    if IsKeyPressed(globals().get("KEY_O", ord("O"))):
        debug.playback.mark_selection_end()
    if IsKeyPressed(globals().get("KEY_ENTER", 257)):
        ApplyManualLabelRange(
            label_result,
            selection_start,
            selection_end,
            debug.selected_action_label,
        )
        debug.annotation_status = "Applied selection"
    if IsKeyPressed(globals().get("KEY_BACKSPACE", 259)):
        ClearManualLabelRange(
            label_result,
            selection_start,
            selection_end,
        )
        debug.annotation_status = "Cleared selection"

    control_down = (
        IsKeyDown(globals().get("KEY_LEFT_CONTROL", 341)) or
        IsKeyDown(globals().get("KEY_RIGHT_CONTROL", 345))
    )
    if control_down and IsKeyPressed(globals().get("KEY_S", ord("S"))):
        save_annotations(app)
    if control_down and IsKeyPressed(globals().get("KEY_L", ord("L"))):
        load_annotations(app)
    if control_down and IsKeyPressed(globals().get("KEY_E", ord("E"))):
        export_labels(app)


def DrawLabelFeatureUI(
    app,
    frame_state,
    playback_layout,
    label_result,
    save_annotations,
    load_annotations,
    export_labels,
    switch_clip_index,
    switch_mirror_variant):

    debug = app.debug
    selection_start, selection_end = debug.playback.selection_range
    timeline_mode = str(getattr(debug, "selected_timeline_mode", "final"))
    if timeline_mode not in ("auto", "final", "soft"):
        timeline_mode = "final"

    GuiGroupBox(Rectangle(20, 200, 360, 126), b"Labels")
    GuiLabel(Rectangle(30, 215, 340, 20), ("Clip: %s" % frame_state.clip_name).encode("utf-8"))
    GuiLabel(Rectangle(30, 235, 160, 20), ("Prior: %s" % frame_state.clip_prior).encode("utf-8"))
    GuiLabel(Rectangle(200, 235, 170, 20), b"FPS: %d" % GetFPS())
    GuiLabel(Rectangle(30, 255, 160, 20), ("Auto: %s" % frame_state.current_auto_label).encode("utf-8"))
    GuiLabel(Rectangle(200, 255, 170, 20), ("Final: %s" % frame_state.current_final_label).encode("utf-8"))
    for mode_index, (mode_key, mode_label) in enumerate((("auto", b"Auto"), ("final", b"Final"), ("soft", b"Soft"))):
        button_rect = Rectangle(30 + mode_index * 110, 276, 96, 18)
        if timeline_mode == mode_key:
            DrawRectangleRec(button_rect, Fade(SKYBLUE, 0.28))
        if GuiButton(button_rect, mode_label):
            debug.selected_timeline_mode = mode_key
            timeline_mode = mode_key
    if _draw_clip_variant_row(app, 30, 298, 330, switch_clip_index, switch_mirror_variant, open_up=True):
        return False

    GuiGroupBox(Rectangle(20, 338, 360, 260), b"Annotate")
    GuiLabel(
        Rectangle(30, 355, 160, 20),
        ("Range: %d - %d" % (selection_start, selection_end)).encode("utf-8"),
    )
    GuiLabel(
        Rectangle(200, 355, 170, 20),
        ("Brush: %s" % debug.selected_action_label).encode("utf-8"),
    )
    GuiLabel(
        Rectangle(30, 373, 340, 18),
        ("Status: %s" % debug.annotation_status).encode("utf-8"),
    )

    for label_index, label in enumerate(ACTION_LABELS):
        row = label_index // 3
        col = label_index % 3
        button_x = 30 + col * 112
        button_y = 396 + row * 34
        button_rect = Rectangle(button_x, button_y, 102, 28)
        if _draw_action_label_button(button_rect, label, selected=(debug.selected_action_label == label)):
            debug.selected_action_label = label

    if GuiButton(Rectangle(30, 505, 76, 28), b"Apply"):
        ApplyManualLabelRange(
            label_result,
            selection_start,
            selection_end,
            debug.selected_action_label,
        )
        debug.annotation_status = "Applied selection"

    if GuiButton(Rectangle(116, 505, 76, 28), b"Clear"):
        ClearManualLabelRange(
            label_result,
            selection_start,
            selection_end,
        )
        debug.annotation_status = "Cleared selection"

    if GuiButton(Rectangle(202, 505, 76, 28), b"Save"):
        save_annotations(app)

    if GuiButton(Rectangle(288, 505, 76, 28), b"Load"):
        load_annotations(app)

    GuiLabel(Rectangle(30, 539, 95, 20), b"Blend W: %d" % debug.transition_width)
    if GuiButton(Rectangle(130, 537, 46, 24), b"-"):
        debug.transition_width = max(0, int(debug.transition_width) - 2)
    if GuiButton(Rectangle(184, 537, 46, 24), b"+"):
        debug.transition_width = min(60, int(debug.transition_width) + 2)
    if GuiButton(Rectangle(238, 537, 60, 24), b"Set"):
        ApplyTransitionWidthRange(
            label_result,
            selection_start,
            selection_end,
            debug.transition_width,
        )
        debug.annotation_status = "Set blend width"
    if GuiButton(Rectangle(306, 537, 58, 24), b"Unset"):
        ClearTransitionWidthRange(
            label_result,
            selection_start,
            selection_end,
        )
        debug.annotation_status = "Cleared blend width"

    if GuiButton(Rectangle(30, 569, 100, 24), b"Export"):
        export_labels(app)

    if GuiButton(Rectangle(140, 569, 224, 24), b"Reset Manual"):
        ResetManualLabels(label_result)
        debug.annotation_status = "Reset manual overrides"

    label_rect, timeline_rect = playback_layout.label_rows[0]
    GuiLabel(label_rect, timeline_mode.capitalize().encode("utf-8"))
    if timeline_mode == "auto":
        _draw_action_label_timeline(
            frame_state.frame_count,
            frame_state.animation_frame,
            timeline_rect,
            segments=frame_state.auto_segments,
            selection_start=selection_start,
            selection_end=selection_end,
        )
    elif timeline_mode == "final":
        _draw_action_label_timeline(
            frame_state.frame_count,
            frame_state.animation_frame,
            timeline_rect,
            segments=frame_state.final_segments,
            selection_start=selection_start,
            selection_end=selection_end,
        )
    else:
        _draw_soft_label_timeline(
            frame_state.frame_count,
            frame_state.animation_frame,
            timeline_rect,
            soft_weights=frame_state.soft_weights,
            selection_start=selection_start,
            selection_end=selection_end,
        )
    GuiLabel(playback_layout.frame_label, b"Frame")

    return True


def SetAnnotationStatusFromLoad(debug, label_result):
    if label_result is None:
        debug.annotation_status = "Labels module off"
        return
    if label_result.annotation_loaded:
        debug.annotation_status = "Loaded " + Path(label_result.annotation_path).name
    else:
        debug.annotation_status = "No saved annotation"


def SaveCurrentAnnotations(app):
    if app.motion.label_result is None:
        app.debug.annotation_status = "Labels module off"
        return None
    label_result = EnsureClipFeature(app, "labels")
    annotation_path = SaveLabelAnnotations(label_result, app.motion.clip_resource)
    app.debug.annotation_status = "Saved " + Path(annotation_path).name
    return annotation_path


def LoadCurrentAnnotations(app):
    if app.motion.label_result is None:
        app.debug.annotation_status = "Labels module off"
        return False
    label_result = EnsureClipFeature(app, "labels")
    if LoadLabelAnnotations(label_result, app.motion.clip_resource):
        app.debug.annotation_status = "Loaded " + Path(label_result.annotation_path).name
        return True
    app.debug.annotation_status = "No saved annotation"
    return False


def ExportCurrentLabels(app):
    if app.motion.label_result is None:
        app.debug.annotation_status = "Labels module off"
        return None
    label_result = EnsureClipFeature(app, "labels")
    export_path = ExportCompiledLabels(label_result, app.motion.clip_resource)
    app.debug.annotation_status = "Exported " + Path(export_path).name
    return export_path


def _handle_annotation_shortcuts(app):
    if not app.debug.label_module_ptr[0]:
        return
    if not IsClipFeatureReady(app, "labels"):
        return
    label_result = EnsureClipFeature(app, "labels")
    HandleLabelFeatureShortcuts(
        app,
        label_result,
        SaveCurrentAnnotations,
        LoadCurrentAnnotations,
        ExportCurrentLabels,
    )


def DrawAppUI(app, frame_state, bvh_path):
    debug = app.debug
    screen_width = app.screen_width
    screen_height = app.screen_height
    _handle_annotation_shortcuts(app)

    playback_layout = debug.playback.get_ui_layout(screen_width, screen_height, numLabelTracks=1)
    rendering_panel_y = 10
    rendering_panel_height = max(120, int(playback_layout.panel.y - rendering_panel_y - 12))

    GuiGroupBox(Rectangle(20, 10, 190, 180), b"Camera")
    GuiLabel(Rectangle(30, 20, 150, 20), b"Ctrl + Left Click - Rotate")
    GuiLabel(Rectangle(30, 40, 150, 20), b"Ctrl + Right Click - Pan")
    GuiLabel(Rectangle(30, 60, 150, 20), b"Mouse Scroll - Zoom")
    GuiLabel(
        Rectangle(30, 80, 150, 20),
        b"Target: [% 5.3f % 5.3f % 5.3f]" % (
            app.camera.cam3d.target.x,
            app.camera.cam3d.target.y,
            app.camera.cam3d.target.z,
        ),
    )
    GuiLabel(
        Rectangle(30, 100, 150, 20),
        b"Offset: [% 5.3f % 5.3f % 5.3f]" % (
            app.camera.offset.x,
            app.camera.offset.y,
            app.camera.offset.z,
        ),
    )
    GuiLabel(Rectangle(30, 120, 150, 20), b"Azimuth: %5.3f" % app.camera.azimuth)
    GuiLabel(Rectangle(30, 140, 150, 20), b"Altitude: %5.3f" % app.camera.altitude)
    GuiLabel(Rectangle(30, 160, 150, 20), b"Distance: %5.3f" % app.camera.distance)

    def switch_clip_index(callback_app, clip_index):
        RequestClipSwitch(callback_app, clip_index, bvh_path)

    def switch_mirror_variant(callback_app):
        RequestClipSwitch(
            callback_app,
            callback_app.clip_index,
            bvh_path,
            mirrored=not callback_app.mirror_enabled,
        )

    if debug.label_module_ptr[0]:
        if IsClipFeatureReady(app, "labels"):
            label_result = EnsureClipFeature(app, "labels")
            if not DrawLabelFeatureUI(
                app,
                frame_state,
                playback_layout,
                label_result,
                SaveCurrentAnnotations,
                LoadCurrentAnnotations,
                ExportCurrentLabels,
                switch_clip_index,
                switch_mirror_variant,
            ):
                return
        else:
            GuiGroupBox(Rectangle(20, 200, 360, 80), b"Labels")
            GuiLabel(Rectangle(30, 220, 220, 20), ("Clip: %s" % frame_state.clip_name).encode("utf-8"))
            GuiLabel(Rectangle(260, 220, 100, 20), b"FPS: %d" % GetFPS())
            GuiLabel(Rectangle(30, 242, 240, 20), b"Labels: loading...")
    else:
        GuiGroupBox(Rectangle(20, 200, 360, 80), b"Clip")
        GuiLabel(Rectangle(30, 220, 220, 20), ("Clip: %s" % frame_state.clip_name).encode("utf-8"))
        GuiLabel(Rectangle(260, 220, 100, 20), b"FPS: %d" % GetFPS())
        GuiLabel(Rectangle(30, 242, 150, 20), b"Labels: off")
        if _draw_clip_variant_row(app, 30, 245, 330, switch_clip_index, switch_mirror_variant, open_up=True):
            return

    DrawRenderingOptionsPanel(
        app,
        frame_state,
        rendering_panel_y,
        rendering_panel_height,
        DrawTerrainRenderPanelMetrics,
        EnsureTerrainFocusFrameState,
        EnsurePenetrationFrameState,
        EnsurePoseErrorFrameState,
    )

    SyncFeatureMounts(app, RequestFeatureLoad)
    debug.playback.draw_ui(screen_width, screen_height)
