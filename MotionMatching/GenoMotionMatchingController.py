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
    MM_DEFAULT_SEARCH_BACKEND,
    MM_KDTREE_EPS,
    MM_KDTREE_LEAF_SIZE,
    MM_KDTREE_MIN_SAMPLES,
    MM_KDTREE_QUERY_OVERSAMPLE,
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


@dataclass
class RawInputState:
    input_source: str = "gamepad:none"
    move_2d: np.ndarray | None = None
    look_2d: np.ndarray | None = None
    look_active: bool = False
    run_pressed: bool = False
    desired_strafe: bool = False
    jump_pressed: bool = False
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
        jump_pressed=bool(IsGamepadButtonDown(gamepad_id, GAMEPAD_BUTTON_RIGHT_FACE_DOWN)),
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
    if input_state.jump_pressed:
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
            search_config=SearchConfig(
                backend=config.search_backend,
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
    GuiGroupBox(Rectangle(20, 10, 430, 305), b"Geno Motion Matching")
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
    GuiLabel(Rectangle(30, 195, 390, 20), f"Speed: {np.linalg.norm(frame.root_velocity[[0, 2]]):.2f}  Query: {frame.query_distance:.2f}".encode("utf-8"))
    GuiLabel(
        Rectangle(30, 215, 390, 20),
        (
            f"Cooldown: {frame.transition_cooldown:.2f}s "
            f"Look: {'on' if app.input_state.look_active else 'off'} "
            f"Strafe: {'on' if app.input_state.desired_strafe else 'off'} "
            f"Jump: {'yes' if app.input_state.jump_pressed else 'no'}"
        ).encode("utf-8"),
    )
    GuiLabel(
        Rectangle(30, 235, 390, 20),
        (
            f"CmdVel: [{app.intent.desired_velocity_world[0]:.2f},{app.intent.desired_velocity_world[2]:.2f}] "
            f"Face: [{app.intent.desired_facing_world[0]:.2f},{app.intent.desired_facing_world[2]:.2f}]"
        ).encode("utf-8"),
    )
    if search is not None:
        GuiLabel(
            Rectangle(30, 255, 390, 20),
            (
                f"Search: idx={search.index} action={search.action_label} "
                f"dist={search.distance:.2f} backend={search.backend}"
            ).encode("utf-8"),
        )
    GuiLabel(
        Rectangle(30, 275, 390, 20),
        f"SearchMode: {app.search_backend}".encode("utf-8"),
    )
    action_weights = frame.action_weights
    GuiLabel(
        Rectangle(30, 295, 390, 20),
        (
            f"ActionW: i={action_weights[0]:.2f} w={action_weights[1]:.2f} "
            f"r={action_weights[2]:.2f} j={action_weights[3]:.2f}"
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
