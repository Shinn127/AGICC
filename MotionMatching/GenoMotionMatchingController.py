from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import sys
from types import SimpleNamespace

MOTION_MATCHING_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MOTION_MATCHING_ROOT
if REPO_ROOT.name == "MotionMatching":
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from pyray import Color, Rectangle, Vector2, Vector3
from raylib import *
from raylib.defines import *

from MotionMatching.MotionMatchingConfig import (
    DEFAULT_DATABASE_PATH,
    MM_ACTION_FILTER_MODES,
    MM_ACTION_HARD_THRESHOLD,
    MM_ACTION_MIN_CANDIDATES,
    MM_ACTION_SOFT_PENALTY,
    MM_DEFAULT_ACTION_FILTER_MODE,
    MM_DEFAULT_SEARCH_BACKEND,
    MM_FORCE_SEARCH_COOLDOWN,
    MM_FORCE_SEARCH_ROTATION_THRESHOLD,
    MM_FORCE_SEARCH_VELOCITY_THRESHOLD,
    MM_IGNORE_RANGE_END_FRAMES,
    MM_IGNORE_SURROUNDING_FRAMES,
    MM_KDTREE_EPS,
    MM_KDTREE_LEAF_SIZE,
    MM_KDTREE_MIN_SAMPLES,
    MM_KDTREE_QUERY_OVERSAMPLE,
    MM_ROOT_ADJUSTMENT_POSITION_HALFLIFE,
    MM_ROOT_ADJUSTMENT_POSITION_MAX_RATIO,
    MM_ROOT_ADJUSTMENT_ROTATION_HALFLIFE,
    MM_ROOT_ADJUSTMENT_ROTATION_MAX_RATIO,
    MM_ROOT_CLAMPING_MAX_ANGLE,
    MM_ROOT_CLAMPING_MAX_DISTANCE,
    MM_ROOT_SYNCHRONIZATION_DATA_FACTOR,
    MM_SEARCH_BACKENDS,
)
from MotionMatching.MotionMatchingDataset import MotionMatchingDataset
from MotionMatching.MotionMatchingRuntime import (
    DEFAULT_RUN_SPEED,
    DEFAULT_WALK_SPEED,
    MotionMatchingRuntime,
    RuntimeConfig,
)
from MotionMatching.MotionMatchingSearch import SearchConfig
from genoview.modules.CameraController import Camera
from genoview.modules.CharacterModel import LoadSceneResources, UnloadSceneResources, UpdateModelPoseFromNumpyArrays
from genoview.modules.RenderModule import (
    BeginGBuffer,
    BeginShadowMap,
    CreateRenderResources,
    EndGBuffer,
    EndShadowMap,
    LoadShaderResources,
    SetShaderValueShadowMap,
    UnloadRenderResources,
    UnloadShaderResources,
    _draw_model_with_shader,
    _make_float_ptr,
    _render_final_pass,
    _render_ssao_and_blur_pass,
    ffi,
)
from genoview.utils.DebugDraw import DrawRootTrajectoryDebug
from genoview.utils import quat


RESOURCES_DIR = REPO_ROOT / "resources"
DEFAULT_SCREEN_WIDTH = 1280
DEFAULT_SCREEN_HEIGHT = 720
DEFAULT_GAMEPAD_ID = 0
DEFAULT_GAMEPAD_DEADZONE = 0.2
DEFAULT_TRAJECTORY_DEBUG_HALFLIFE = 0.08
DEFAULT_TRAJECTORY_DEBUG_HEIGHT_OFFSET = 0.04


def resource_path(*parts, as_bytes=False):
    path = RESOURCES_DIR.joinpath(*parts)
    return str(path).encode("utf-8") if as_bytes else str(path)


@dataclass(frozen=True)
class ViewerConfig:
    database_path: Path = DEFAULT_DATABASE_PATH
    screen_width: int = DEFAULT_SCREEN_WIDTH
    screen_height: int = DEFAULT_SCREEN_HEIGHT
    gamepad_id: int = DEFAULT_GAMEPAD_ID
    search_backend: str = MM_DEFAULT_SEARCH_BACKEND
    kd_min_samples: int = MM_KDTREE_MIN_SAMPLES
    kd_leaf_size: int = MM_KDTREE_LEAF_SIZE
    kd_query_oversample: int = MM_KDTREE_QUERY_OVERSAMPLE
    kd_eps: float = MM_KDTREE_EPS
    ignore_surrounding_frames: int = MM_IGNORE_SURROUNDING_FRAMES
    ignore_range_end_frames: int = MM_IGNORE_RANGE_END_FRAMES
    action_filter_mode: str = MM_DEFAULT_ACTION_FILTER_MODE
    action_hard_threshold: float = MM_ACTION_HARD_THRESHOLD
    action_min_candidates: int = MM_ACTION_MIN_CANDIDATES
    action_soft_penalty: float = MM_ACTION_SOFT_PENALTY
    force_search_enabled: bool = True
    force_search_velocity_threshold: float = MM_FORCE_SEARCH_VELOCITY_THRESHOLD
    force_search_rotation_threshold: float = MM_FORCE_SEARCH_ROTATION_THRESHOLD
    force_search_cooldown: float = MM_FORCE_SEARCH_COOLDOWN
    root_adjustment_enabled: bool = True
    root_adjustment_by_velocity: bool = True
    root_adjustment_position_halflife: float = MM_ROOT_ADJUSTMENT_POSITION_HALFLIFE
    root_adjustment_rotation_halflife: float = MM_ROOT_ADJUSTMENT_ROTATION_HALFLIFE
    root_adjustment_position_max_ratio: float = MM_ROOT_ADJUSTMENT_POSITION_MAX_RATIO
    root_adjustment_rotation_max_ratio: float = MM_ROOT_ADJUSTMENT_ROTATION_MAX_RATIO
    root_clamping_enabled: bool = True
    root_clamping_max_distance: float = MM_ROOT_CLAMPING_MAX_DISTANCE
    root_clamping_max_angle: float = MM_ROOT_CLAMPING_MAX_ANGLE
    root_synchronization_enabled: bool = False
    root_synchronization_data_factor: float = MM_ROOT_SYNCHRONIZATION_DATA_FACTOR


@dataclass
class RawInputState:
    input_source: str = "gamepad:none"
    move_2d: np.ndarray | None = None
    look_2d: np.ndarray | None = None
    look_active: bool = False
    run_pressed: bool = False
    desired_strafe: bool = False
    jump_down: bool = False
    jump_pressed: bool = False
    jump_released: bool = False
    reset_pressed: bool = False

    def __post_init__(self):
        if self.move_2d is None:
            self.move_2d = np.zeros(2, dtype=np.float32)
        if self.look_2d is None:
            self.look_2d = np.zeros(2, dtype=np.float32)


def _shape_gamepad_stick(x: float, y: float, deadzone: float = DEFAULT_GAMEPAD_DEADZONE) -> np.ndarray:
    stick = np.asarray([x, y], dtype=np.float32)
    magnitude = float(np.linalg.norm(stick))
    if magnitude <= deadzone:
        return np.zeros(2, dtype=np.float32)
    direction = stick / magnitude
    shaped_magnitude = min(magnitude * magnitude, 1.0)
    return (direction * shaped_magnitude).astype(np.float32)


def _read_gamepad_input(gamepad_id: int) -> RawInputState:
    if not IsGamepadAvailable(gamepad_id):
        return RawInputState(input_source="gamepad:none")
    move_2d = _shape_gamepad_stick(
        GetGamepadAxisMovement(gamepad_id, GAMEPAD_AXIS_LEFT_X),
        GetGamepadAxisMovement(gamepad_id, GAMEPAD_AXIS_LEFT_Y),
    )
    look_2d = _shape_gamepad_stick(
        GetGamepadAxisMovement(gamepad_id, GAMEPAD_AXIS_RIGHT_X),
        GetGamepadAxisMovement(gamepad_id, GAMEPAD_AXIS_RIGHT_Y),
    )
    right_trigger = float(GetGamepadAxisMovement(gamepad_id, GAMEPAD_AXIS_RIGHT_TRIGGER))
    left_trigger = float(GetGamepadAxisMovement(gamepad_id, GAMEPAD_AXIS_LEFT_TRIGGER))
    return RawInputState(
        input_source=f"gamepad:{gamepad_id}",
        move_2d=move_2d,
        look_2d=look_2d,
        look_active=bool(np.linalg.norm(look_2d) > 1e-3),
        run_pressed=bool(right_trigger > 0.5 or IsGamepadButtonDown(gamepad_id, GAMEPAD_BUTTON_RIGHT_TRIGGER_2)),
        desired_strafe=bool(left_trigger > 0.5 or IsGamepadButtonDown(gamepad_id, GAMEPAD_BUTTON_LEFT_TRIGGER_2)),
        jump_down=bool(IsGamepadButtonDown(gamepad_id, GAMEPAD_BUTTON_RIGHT_FACE_DOWN)),
        jump_pressed=bool(IsGamepadButtonPressed(gamepad_id, GAMEPAD_BUTTON_RIGHT_FACE_DOWN)),
        jump_released=bool(IsGamepadButtonReleased(gamepad_id, GAMEPAD_BUTTON_RIGHT_FACE_DOWN)),
        reset_pressed=bool(IsGamepadButtonPressed(gamepad_id, GAMEPAD_BUTTON_MIDDLE_RIGHT)),
    )


def _read_control_input(app) -> RawInputState:
    return _read_gamepad_input(app.gamepad_id)


def _build_intent_from_input(runtime: MotionMatchingRuntime, input_state: RawInputState, camera: Camera):
    camera_rotation = quat.from_angle_axis(
        camera.azimuth,
        np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    ).astype(np.float32)
    move_world = quat.mul_vec(
        camera_rotation,
        np.asarray([input_state.move_2d[0], 0.0, input_state.move_2d[1]], dtype=np.float32),
    ).astype(np.float32)
    look_world = quat.mul_vec(
        camera_rotation,
        np.asarray([input_state.look_2d[0], 0.0, input_state.look_2d[1]], dtype=np.float32),
    ).astype(np.float32)
    move_magnitude = float(np.linalg.norm(input_state.move_2d))
    is_moving = move_magnitude > 1e-3
    speed = DEFAULT_RUN_SPEED if input_state.run_pressed else DEFAULT_WALK_SPEED
    if not is_moving:
        speed = 0.0
    if input_state.jump_down:
        action_label = "jump"
    elif is_moving and input_state.run_pressed:
        action_label = "run"
    elif is_moving:
        action_label = "walk"
    else:
        action_label = "idle"
    facing_world = look_world if input_state.look_active else None
    return runtime.make_locomotion_intent(
        move_world,
        speed,
        action_label,
        facing_direction_world=facing_world,
        desired_strafe=input_state.desired_strafe,
        jump_down=input_state.jump_down,
        jump_pressed=input_state.jump_pressed,
        jump_released=input_state.jump_released,
    )


def _trajectory_smoothing_alpha(dt: float, halflife: float = DEFAULT_TRAJECTORY_DEBUG_HALFLIFE) -> float:
    if halflife <= 0.0:
        return 1.0
    return float(1.0 - np.exp(-0.69314718056 * float(dt) / float(halflife)))


def _normalize_debug_directions(directions: np.ndarray) -> np.ndarray:
    directions = np.asarray(directions, dtype=np.float32).copy()
    directions[:, 1] = 0.0
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    fallback = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    return np.divide(
        directions,
        np.maximum(norms, 1e-6),
        out=np.repeat(fallback[np.newaxis, :], len(directions), axis=0),
        where=norms > 1e-6,
    ).astype(np.float32)


def _predict_debug_trajectory(app):
    positions, directions, velocities = app.runtime.predict_future_trajectory(app.intent)
    return (
        np.asarray(positions, dtype=np.float32),
        _normalize_debug_directions(directions),
        np.asarray(velocities, dtype=np.float32),
    )


def _reset_debug_trajectory(app) -> None:
    positions, directions, velocities = _predict_debug_trajectory(app)
    app.debug_trajectory_positions = positions.copy()
    app.debug_trajectory_directions = directions.copy()
    app.debug_trajectory_velocities = velocities.copy()


def _update_debug_trajectory(app, dt: float) -> None:
    positions, directions, velocities = _predict_debug_trajectory(app)
    if app.debug_trajectory_positions is None:
        app.debug_trajectory_positions = positions.copy()
        app.debug_trajectory_directions = directions.copy()
        app.debug_trajectory_velocities = velocities.copy()
        return

    alpha = _trajectory_smoothing_alpha(dt)
    app.debug_trajectory_positions = (
        (1.0 - alpha) * app.debug_trajectory_positions + alpha * positions
    ).astype(np.float32)
    app.debug_trajectory_velocities = (
        (1.0 - alpha) * app.debug_trajectory_velocities + alpha * velocities
    ).astype(np.float32)
    app.debug_trajectory_directions = _normalize_debug_directions(
        (1.0 - alpha) * app.debug_trajectory_directions + alpha * directions
    )


def _create_viewer_state(config: ViewerConfig):
    scene = LoadSceneResources(resource_path)
    database = MotionMatchingDataset(config.database_path)
    runtime = MotionMatchingRuntime(
        database,
        config=RuntimeConfig(
            force_search_enabled=config.force_search_enabled,
            force_search_velocity_threshold=config.force_search_velocity_threshold,
            force_search_rotation_threshold=config.force_search_rotation_threshold,
            force_search_cooldown=config.force_search_cooldown,
            root_adjustment_enabled=config.root_adjustment_enabled,
            root_adjustment_by_velocity=config.root_adjustment_by_velocity,
            root_adjustment_position_halflife=config.root_adjustment_position_halflife,
            root_adjustment_rotation_halflife=config.root_adjustment_rotation_halflife,
            root_adjustment_position_max_ratio=config.root_adjustment_position_max_ratio,
            root_adjustment_rotation_max_ratio=config.root_adjustment_rotation_max_ratio,
            root_clamping_enabled=config.root_clamping_enabled,
            root_clamping_max_distance=config.root_clamping_max_distance,
            root_clamping_max_angle=config.root_clamping_max_angle,
            root_synchronization_enabled=config.root_synchronization_enabled,
            root_synchronization_data_factor=config.root_synchronization_data_factor,
            search_config=SearchConfig(
                backend=config.search_backend,
                ignore_surrounding_frames=config.ignore_surrounding_frames,
                ignore_range_end_frames=config.ignore_range_end_frames,
                action_filter_mode=config.action_filter_mode,
                action_hard_threshold=config.action_hard_threshold,
                action_min_candidates=config.action_min_candidates,
                action_soft_penalty=config.action_soft_penalty,
                kd_min_samples=config.kd_min_samples,
                kd_leaf_size=config.kd_leaf_size,
                kd_query_oversample=config.kd_query_oversample,
                kd_eps=config.kd_eps,
            ),
        ),
        initial_action="idle",
    )
    camera = Camera()
    input_state = RawInputState()
    intent = _build_intent_from_input(runtime, input_state, camera)
    initial_frame = runtime.update(intent, database.spec.dt)
    UpdateModelPoseFromNumpyArrays(
        scene.geno_model,
        scene.bind_pos,
        scene.bind_rot,
        initial_frame.world_positions,
        initial_frame.world_rotations,
    )

    app = SimpleNamespace(
        screen_width=int(config.screen_width),
        screen_height=int(config.screen_height),
        gamepad_id=int(config.gamepad_id),
        scene=scene,
        database=database,
        runtime=runtime,
        shaders=LoadShaderResources(resource_path),
        render=CreateRenderResources(config.screen_width, config.screen_height),
        camera=camera,
        input_state=input_state,
        intent=intent,
        frame=initial_frame,
        debug_trajectory_positions=None,
        debug_trajectory_directions=None,
        debug_trajectory_velocities=None,
        show_flat_ground=True,
        search_backend=config.search_backend,
    )
    _reset_debug_trajectory(app)
    return app


def _update_tracking(app):
    dt = max(1e-6, float(GetFrameTime()))
    app.input_state = _read_control_input(app)
    if app.input_state.reset_pressed:
        app.runtime.reset(initial_action="idle")
        app.debug_trajectory_positions = None

    app.intent = _build_intent_from_input(app.runtime, app.input_state, app.camera)
    app.frame = app.runtime.update(app.intent, dt)
    _update_debug_trajectory(app, dt)
    UpdateModelPoseFromNumpyArrays(
        app.scene.geno_model,
        app.scene.bind_pos,
        app.scene.bind_rot,
        app.frame.world_positions,
        app.frame.world_rotations,
    )

    hip_position = Vector3(*app.frame.world_positions[0])
    app.render.shadow_light.target = Vector3(hip_position.x, 0.0, hip_position.z)
    app.render.shadow_light.position = Vector3Add(
        app.render.shadow_light.target,
        Vector3Scale(app.render.light_dir, -5.0),
    )
    app.camera.update(
        Vector3(hip_position.x, 0.75, hip_position.z),
        GetMouseDelta().x if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(0) else 0.0,
        GetMouseDelta().y if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(0) else 0.0,
        GetMouseDelta().x if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(1) else 0.0,
        GetMouseDelta().y if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(1) else 0.0,
        GetMouseWheelMove(),
        dt,
    )
    return hip_position


def _render_shadow_pass(app):
    render = app.render
    shaders = app.shaders
    scene = app.scene
    BeginShadowMap(render.shadow_map, render.shadow_light)
    light_view_proj = MatrixMultiply(rlGetMatrixModelview(), rlGetMatrixProjection())
    light_clip_near_ptr = _make_float_ptr(rlGetCullDistanceNear())
    light_clip_far_ptr = _make_float_ptr(rlGetCullDistanceFar())
    for shadow_shader in (shaders.shadow, shaders.skinned_shadow):
        SetShaderValue(shadow_shader.program, shadow_shader.light_clip_near, light_clip_near_ptr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(shadow_shader.program, shadow_shader.light_clip_far, light_clip_far_ptr, SHADER_UNIFORM_FLOAT)
    if app.show_flat_ground:
        _draw_model_with_shader(scene.ground_model, shaders.shadow.program, scene.ground_position, WHITE)
    _draw_model_with_shader(scene.geno_model, shaders.skinned_shadow.program, scene.geno_position, WHITE)
    EndShadowMap()
    return SimpleNamespace(
        view_proj=light_view_proj,
        clip_near_ptr=light_clip_near_ptr,
        clip_far_ptr=light_clip_far_ptr,
    )


def _render_gbuffer_pass(app):
    render = app.render
    shaders = app.shaders
    scene = app.scene
    BeginGBuffer(render.gbuffer, app.camera.cam3d)
    cam_view = rlGetMatrixModelview()
    cam_proj = rlGetMatrixProjection()
    cam_inv_proj = MatrixInvert(cam_proj)
    cam_inv_view_proj = MatrixInvert(MatrixMultiply(cam_view, cam_proj))
    cam_clip_near_ptr = _make_float_ptr(rlGetCullDistanceNear())
    cam_clip_far_ptr = _make_float_ptr(rlGetCullDistanceFar())
    specularity_ptr = _make_float_ptr(0.5)
    glossiness_ptr = _make_float_ptr(10.0)
    for shader in (shaders.basic, shaders.skinned_basic):
        SetShaderValue(shader.program, shader.specularity, specularity_ptr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(shader.program, shader.glossiness, glossiness_ptr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(shader.program, shader.cam_clip_near, cam_clip_near_ptr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(shader.program, shader.cam_clip_far, cam_clip_far_ptr, SHADER_UNIFORM_FLOAT)
    if app.show_flat_ground:
        _draw_model_with_shader(scene.ground_model, shaders.basic.program, scene.ground_position, Color(190, 190, 190, 255))
    _draw_model_with_shader(scene.geno_model, shaders.skinned_basic.program, scene.geno_position, Color(180, 225, 255, 255))
    EndGBuffer(app.screen_width, app.screen_height)
    return SimpleNamespace(
        view=cam_view,
        proj=cam_proj,
        inv_proj=cam_inv_proj,
        inv_view_proj=cam_inv_view_proj,
        clip_near_ptr=cam_clip_near_ptr,
        clip_far_ptr=cam_clip_far_ptr,
    )


def _render_lighting_and_debug_pass(app, camera_pass):
    render = app.render
    shaders = app.shaders
    BeginTextureMode(render.lighted)
    BeginShaderMode(shaders.lighting.program)
    sun_color = Vector3(253.0 / 255.0, 255.0 / 255.0, 232.0 / 255.0)
    sky_color = Vector3(174.0 / 255.0, 183.0 / 255.0, 190.0 / 255.0)
    SetShaderValueTexture(shaders.lighting.program, shaders.lighting.gbuffer_color, render.gbuffer.color)
    SetShaderValueTexture(shaders.lighting.program, shaders.lighting.gbuffer_normal, render.gbuffer.normal)
    SetShaderValueTexture(shaders.lighting.program, shaders.lighting.gbuffer_depth, render.gbuffer.depth)
    SetShaderValueTexture(shaders.lighting.program, shaders.lighting.ssao, render.ssao_front.texture)
    SetShaderValue(shaders.lighting.program, shaders.lighting.cam_pos, ffi.addressof(app.camera.cam3d.position), SHADER_UNIFORM_VEC3)
    SetShaderValueMatrix(shaders.lighting.program, shaders.lighting.cam_inv_view_proj, camera_pass.inv_view_proj)
    SetShaderValue(shaders.lighting.program, shaders.lighting.light_dir, ffi.addressof(render.light_dir), SHADER_UNIFORM_VEC3)
    SetShaderValue(shaders.lighting.program, shaders.lighting.sun_color, ffi.addressof(sun_color), SHADER_UNIFORM_VEC3)
    SetShaderValue(shaders.lighting.program, shaders.lighting.sun_strength, _make_float_ptr(0.25), SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.lighting.program, shaders.lighting.sky_color, ffi.addressof(sky_color), SHADER_UNIFORM_VEC3)
    SetShaderValue(shaders.lighting.program, shaders.lighting.sky_strength, _make_float_ptr(0.15), SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.lighting.program, shaders.lighting.ground_strength, _make_float_ptr(0.1), SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.lighting.program, shaders.lighting.ambient_strength, _make_float_ptr(1.0), SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.lighting.program, shaders.lighting.exposure, _make_float_ptr(0.9), SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.lighting.program, shaders.lighting.cam_clip_near, camera_pass.clip_near_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.lighting.program, shaders.lighting.cam_clip_far, camera_pass.clip_far_ptr, SHADER_UNIFORM_FLOAT)
    ClearBackground(RAYWHITE)
    DrawTextureRec(
        render.gbuffer.color,
        Rectangle(0, 0, render.gbuffer.color.width, -render.gbuffer.color.height),
        Vector2(0.0, 0.0),
        WHITE,
    )
    EndShaderMode()

    BeginMode3D(app.camera.cam3d)
    future_positions = np.asarray(app.debug_trajectory_positions, dtype=np.float32).copy()
    future_positions[:, 1] += DEFAULT_TRAJECTORY_DEBUG_HEIGHT_OFFSET
    DrawRootTrajectoryDebug(
        future_positions,
        app.debug_trajectory_directions,
        app.debug_trajectory_velocities,
        np.asarray(app.database.spec.future_sample_offsets, dtype=np.int32),
        drawDirection=True,
        drawVelocity=True,
        directionScale=0.25,
        velocityScale=0.15,
    )
    EndMode3D()
    EndTextureMode()


def _draw_ui(app):
    frame = app.frame
    search = frame.search_result
    GuiGroupBox(Rectangle(20, 10, 520, 405), b"Geno Motion Matching")
    GuiLabel(Rectangle(30, 30, 390, 20), b"Gamepad: left stick move, right stick face, RT run")
    GuiLabel(Rectangle(30, 50, 390, 20), b"LT strafe, A jump, menu/start reset")
    GuiLabel(Rectangle(30, 70, 390, 20), b"Ctrl + mouse: camera only, wheel: zoom")
    GuiLabel(Rectangle(30, 95, 390, 20), f"Input: {app.input_state.input_source}".encode("utf-8"))
    GuiLabel(
        Rectangle(30, 115, 390, 20),
        (
            f"Move: [{app.input_state.move_2d[0]:.2f},{app.input_state.move_2d[1]:.2f}] "
            f"Look: [{app.input_state.look_2d[0]:.2f},{app.input_state.look_2d[1]:.2f}]"
        ).encode("utf-8"),
    )
    GuiLabel(
        Rectangle(30, 135, 390, 20),
        f"Intent: {app.intent.action_label}  DB Argmax: {frame.action_label}".encode("utf-8"),
    )
    GuiLabel(Rectangle(30, 155, 390, 20), f"Index: {frame.current_index}  Transitions: {frame.transition_count}".encode("utf-8"))
    GuiLabel(Rectangle(30, 175, 390, 20), f"Root: x={frame.root_position[0]:.2f} z={frame.root_position[2]:.2f}".encode("utf-8"))
    GuiLabel(
        Rectangle(30, 195, 420, 20),
        (
            f"Sim: x={frame.simulation_position[0]:.2f} z={frame.simulation_position[2]:.2f} "
            f"Err: {frame.root_position_error:.2f}m/{frame.root_rotation_error:.2f}rad"
        ).encode("utf-8"),
    )
    GuiLabel(Rectangle(30, 215, 390, 20), f"Speed: {np.linalg.norm(frame.root_velocity[[0, 2]]):.2f}  Query: {frame.query_distance:.2f}".encode("utf-8"))
    GuiLabel(
        Rectangle(30, 235, 390, 20),
        (
            f"Cooldown: {frame.transition_cooldown:.2f}s "
            f"Look: {'on' if app.input_state.look_active else 'off'} "
            f"Strafe: {'on' if app.input_state.desired_strafe else 'off'} "
            f"Jump: d={'1' if app.input_state.jump_down else '0'} "
            f"p={'1' if app.input_state.jump_pressed else '0'} "
            f"r={'1' if app.input_state.jump_released else '0'}"
        ).encode("utf-8"),
    )
    GuiLabel(
        Rectangle(30, 255, 390, 20),
        (
            f"CmdVel: [{app.intent.desired_velocity_world[0]:.2f},{app.intent.desired_velocity_world[2]:.2f}] "
            f"Face: [{app.intent.desired_facing_world[0]:.2f},{app.intent.desired_facing_world[2]:.2f}]"
        ).encode("utf-8"),
    )
    if search is not None:
        GuiLabel(
            Rectangle(30, 275, 430, 20),
            (
                f"Search: idx={search.index} action={search.action_label} "
                f"dist={search.distance:.2f} score={search.score:.2f} backend={search.backend}"
            ).encode("utf-8"),
        )
        GuiLabel(
            Rectangle(30, 295, 430, 20),
            (
                f"Candidates: {search.filtered_candidate_count}/{search.candidate_count} "
                f"affinity={search.action_affinity:.2f} mode={search.action_filter_mode}"
            ).encode("utf-8"),
        )
    GuiLabel(
        Rectangle(30, 315, 430, 20),
        (
            f"SearchMode: {app.search_backend} reason={frame.search_reason} "
            f"force={'yes' if frame.forced_search else 'no'} current={frame.current_score:.2f}"
        ).encode("utf-8"),
    )
    action_weights = frame.action_weights
    GuiLabel(
        Rectangle(30, 335, 390, 20),
        (
            f"ActionW: i={action_weights[0]:.2f} w={action_weights[1]:.2f} "
            f"r={action_weights[2]:.2f} j={action_weights[3]:.2f}"
        ).encode("utf-8"),
    )
    GuiLabel(
        Rectangle(30, 355, 480, 20),
        (
            f"ActionState: active={frame.active_action or 'none'} phase={frame.action_phase} "
            f"pendingExit={'yes' if frame.pending_action_exit else 'no'}"
        ).encode("utf-8"),
    )
    GuiLabel(
        Rectangle(30, 375, 480, 20),
        (
            f"Candidates: mode={frame.candidate_mode} count={frame.candidate_count} "
            f"forceTransition={'yes' if frame.force_transition else 'no'}"
        ).encode("utf-8"),
    )


def _render_frame(app):
    shadow_pass = _render_shadow_pass(app)
    camera_pass = _render_gbuffer_pass(app)
    _render_ssao_and_blur_pass(app, shadow_pass, camera_pass)
    _render_lighting_and_debug_pass(app, camera_pass)
    _render_final_pass(app)


def _unload_viewer_resources(app):
    if app is None:
        return
    if getattr(app, "database", None) is not None:
        app.database.close()
    UnloadRenderResources(app.render)
    UnloadSceneResources(app.scene)
    UnloadShaderResources(app.shaders)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Runtime viewer for the Geno Motion Matching controller.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH, help="Path to motion_matching_database.npz.")
    parser.add_argument("--screen-width", type=int, default=DEFAULT_SCREEN_WIDTH, help="Viewer window width.")
    parser.add_argument("--screen-height", type=int, default=DEFAULT_SCREEN_HEIGHT, help="Viewer window height.")
    parser.add_argument("--gamepad-id", type=int, default=DEFAULT_GAMEPAD_ID, help="Gamepad id used by the viewer controls.")
    parser.add_argument("--search-backend", choices=MM_SEARCH_BACKENDS, default=MM_DEFAULT_SEARCH_BACKEND, help="Motion Matching search backend.")
    parser.add_argument("--kd-min-samples", type=int, default=MM_KDTREE_MIN_SAMPLES, help="Auto backend threshold for KDTree search.")
    parser.add_argument("--kd-leaf-size", type=int, default=MM_KDTREE_LEAF_SIZE, help="scipy cKDTree leaf size.")
    parser.add_argument("--kd-query-oversample", type=int, default=MM_KDTREE_QUERY_OVERSAMPLE, help="Extra KDTree neighbors before exact rerank.")
    parser.add_argument("--kd-eps", type=float, default=MM_KDTREE_EPS, help="Approximation epsilon for cKDTree.query.")
    parser.add_argument("--ignore-surrounding-frames", type=int, default=MM_IGNORE_SURROUNDING_FRAMES, help="Exclude frames near the current sample from search.")
    parser.add_argument("--ignore-range-end-frames", type=int, default=MM_IGNORE_RANGE_END_FRAMES, help="Exclude samples too close to the end of a range.")
    parser.add_argument("--action-filter-mode", choices=MM_ACTION_FILTER_MODES, default=MM_DEFAULT_ACTION_FILTER_MODE, help="Action-aware search mode.")
    parser.add_argument("--action-hard-threshold", type=float, default=MM_ACTION_HARD_THRESHOLD, help="Minimum action affinity for hard action filtering.")
    parser.add_argument("--action-min-candidates", type=int, default=MM_ACTION_MIN_CANDIDATES, help="Minimum hard-filtered candidates before falling back.")
    parser.add_argument("--action-soft-penalty", type=float, default=MM_ACTION_SOFT_PENALTY, help="Score penalty for action mismatch.")
    parser.add_argument("--disable-force-search", action="store_true", help="Disable force-search when input changes settle.")
    parser.add_argument("--force-search-velocity-threshold", type=float, default=MM_FORCE_SEARCH_VELOCITY_THRESHOLD, help="Velocity change threshold for force-search settling.")
    parser.add_argument("--force-search-rotation-threshold", type=float, default=MM_FORCE_SEARCH_ROTATION_THRESHOLD, help="Rotation change threshold for force-search settling.")
    parser.add_argument("--force-search-cooldown", type=float, default=MM_FORCE_SEARCH_COOLDOWN, help="Cooldown between force-search triggers.")
    parser.add_argument("--disable-root-adjustment", action="store_true", help="Disable smooth root-to-simulation adjustment.")
    parser.add_argument("--disable-root-adjustment-velocity-limit", action="store_true", help="Disable velocity-limited root adjustment.")
    parser.add_argument("--root-adjustment-position-halflife", type=float, default=MM_ROOT_ADJUSTMENT_POSITION_HALFLIFE, help="Root position adjustment halflife.")
    parser.add_argument("--root-adjustment-rotation-halflife", type=float, default=MM_ROOT_ADJUSTMENT_ROTATION_HALFLIFE, help="Root rotation adjustment halflife.")
    parser.add_argument("--root-adjustment-position-max-ratio", type=float, default=MM_ROOT_ADJUSTMENT_POSITION_MAX_RATIO, help="Velocity ratio cap for position adjustment.")
    parser.add_argument("--root-adjustment-rotation-max-ratio", type=float, default=MM_ROOT_ADJUSTMENT_ROTATION_MAX_RATIO, help="Angular velocity ratio cap for rotation adjustment.")
    parser.add_argument("--disable-root-clamping", action="store_true", help="Disable hard root-to-simulation clamping.")
    parser.add_argument("--root-clamping-max-distance", type=float, default=MM_ROOT_CLAMPING_MAX_DISTANCE, help="Maximum root distance from simulation before clamping.")
    parser.add_argument("--root-clamping-max-angle", type=float, default=MM_ROOT_CLAMPING_MAX_ANGLE, help="Maximum root yaw offset from simulation before clamping.")
    parser.add_argument("--enable-root-synchronization", action="store_true", help="Synchronize root and simulation directly.")
    parser.add_argument("--root-synchronization-data-factor", type=float, default=MM_ROOT_SYNCHRONIZATION_DATA_FACTOR, help="0 follows simulation, 1 follows data root during synchronization.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = ViewerConfig(
        database_path=args.database,
        screen_width=args.screen_width,
        screen_height=args.screen_height,
        gamepad_id=args.gamepad_id,
        search_backend=args.search_backend,
        kd_min_samples=args.kd_min_samples,
        kd_leaf_size=args.kd_leaf_size,
        kd_query_oversample=args.kd_query_oversample,
        kd_eps=args.kd_eps,
        ignore_surrounding_frames=args.ignore_surrounding_frames,
        ignore_range_end_frames=args.ignore_range_end_frames,
        action_filter_mode=args.action_filter_mode,
        action_hard_threshold=args.action_hard_threshold,
        action_min_candidates=args.action_min_candidates,
        action_soft_penalty=args.action_soft_penalty,
        force_search_enabled=not args.disable_force_search,
        force_search_velocity_threshold=args.force_search_velocity_threshold,
        force_search_rotation_threshold=args.force_search_rotation_threshold,
        force_search_cooldown=args.force_search_cooldown,
        root_adjustment_enabled=not args.disable_root_adjustment,
        root_adjustment_by_velocity=not args.disable_root_adjustment_velocity_limit,
        root_adjustment_position_halflife=args.root_adjustment_position_halflife,
        root_adjustment_rotation_halflife=args.root_adjustment_rotation_halflife,
        root_adjustment_position_max_ratio=args.root_adjustment_position_max_ratio,
        root_adjustment_rotation_max_ratio=args.root_adjustment_rotation_max_ratio,
        root_clamping_enabled=not args.disable_root_clamping,
        root_clamping_max_distance=args.root_clamping_max_distance,
        root_clamping_max_angle=args.root_clamping_max_angle,
        root_synchronization_enabled=args.enable_root_synchronization,
        root_synchronization_data_factor=args.root_synchronization_data_factor,
    )
    SetConfigFlags(FLAG_VSYNC_HINT)
    InitWindow(config.screen_width, config.screen_height, b"Geno Motion Matching Controller")
    SetTargetFPS(60)
    app = None
    try:
        app = _create_viewer_state(config)
        rlSetClipPlanes(0.01, 50.0)
        while not WindowShouldClose():
            _update_tracking(app)
            rlDisableColorBlend()
            BeginDrawing()
            _render_frame(app)
            rlEnableColorBlend()
            _draw_ui(app)
            EndDrawing()
    finally:
        _unload_viewer_resources(app)
        CloseWindow()


if __name__ == "__main__":
    main()
