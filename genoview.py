from raylib import *
from raylib.defines import *

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from queue import SimpleQueue
from genoview.modules.CameraController import Camera
from genoview.State import (
    AppState,
    AsyncProgress,
)
from genoview.modules.MotionDebugModule import CreateDebugState
from genoview.modules.AsyncModule import (
    DrainAsyncProgress,
    PollAsyncClipLoad,
    PollAsyncFeatureLoads,
    RequestFeatureLoad,
)
from genoview.modules.BVHImporter import (
    DiscoverBVHClips,
    GetClipIndex,
    LoadMotionResources,
)
from genoview.modules.CharacterModel import LoadSceneResources, UnloadSceneResources
from genoview.modules.FeatureModule import (
    BuildFeatureRegistry,
    SyncFeatureMounts,
)
from genoview.modules.FrameModule import BuildFrameState, UpdateTracking
from genoview.modules.RenderModule import (
    CreateRenderResources,
    LoadShaderResources,
    RenderFrame,
    UnloadRenderResources,
    UnloadShaderResources,
)
from genoview.modules.PlaybackController import PlaybackController
from genoview.GUI import (
    DrawAppUI,
    SaveCurrentAnnotations,
    SetAnnotationStatusFromLoad,
)

PROJECT_ROOT = Path(__file__).resolve().parent
RESOURCES_DIR = PROJECT_ROOT / "resources"

# 手工指定启动 BVH 的入口。优先使用相对 resources/ 的路径，例如：
# "bvh/lafan1/jumps1_subject1.bvh"
STARTUP_BVH_CLIP = "bvh/lafan1/jumps1_subject1.bvh"
BVH_CLIP_DIR = "bvh/lafan1"


def resource_path(*parts, as_bytes=False):
    path = RESOURCES_DIR.joinpath(*parts)
    return str(path).encode("utf-8") if as_bytes else str(path)


def _discover_startup_clips():
    clip_resources = DiscoverBVHClips(
        RESOURCES_DIR,
        default_bvh_dir=BVH_CLIP_DIR,
        default_bvh_clip=STARTUP_BVH_CLIP,
    )
    if STARTUP_BVH_CLIP not in clip_resources:
        clip_resources.insert(0, STARTUP_BVH_CLIP)
    return clip_resources


def _create_app_state(screen_width, screen_height):
    scene = LoadSceneResources(resource_path)
    clip_resources = _discover_startup_clips()
    clip_index = GetClipIndex(clip_resources, STARTUP_BVH_CLIP)
    motion = LoadMotionResources(resource_path, clip_resources[clip_index])
    debug = CreateDebugState(motion.bvh_animation.frame_count, motion.bvh_frame_time)
    app = AppState(
        screen_width=screen_width,
        screen_height=screen_height,
        shaders=LoadShaderResources(resource_path),
        scene=scene,
        motion=motion,
        clip_resources=clip_resources,
        clip_index=clip_index,
        render=CreateRenderResources(screen_width, screen_height),
        debug=debug,
        features=BuildFeatureRegistry(resource_path),
        async_progress=AsyncProgress(),
        async_progress_events=SimpleQueue(),
        clip_load_executor=ThreadPoolExecutor(max_workers=1),
        pending_clip_load=None,
        feature_load_executor=ThreadPoolExecutor(max_workers=2),
        pending_feature_loads={},
        camera=Camera(),
    )
    SyncFeatureMounts(app, RequestFeatureLoad)
    SetAnnotationStatusFromLoad(debug, motion.label_result)
    return app


def _commit_clip_switch(app, clip_index, motion):
    SaveCurrentAnnotations(app)
    for pending in app.pending_feature_loads.values():
        pending.future.cancel()
    app.pending_feature_loads.clear()
    app.features.dispose_clip(app)

    app.clip_index = int(clip_index) % len(app.clip_resources)
    app.motion = motion
    SyncFeatureMounts(app, RequestFeatureLoad)
    app.debug.playback = PlaybackController(app.motion.bvh_animation.frame_count, app.motion.bvh_frame_time)
    SetAnnotationStatusFromLoad(app.debug, app.motion.label_result)


def _unload_app_resources(app):
    if app is None:
        return

    if app.pending_clip_load is not None:
        app.pending_clip_load.future.cancel()
        app.pending_clip_load = None
    for pending in app.pending_feature_loads.values():
        pending.future.cancel()
    app.pending_feature_loads.clear()
    app.clip_load_executor.shutdown(wait=False, cancel_futures=True)
    app.feature_load_executor.shutdown(wait=False, cancel_futures=True)
    app.features.dispose_clip(app)

    UnloadRenderResources(app.render)
    UnloadSceneResources(app.scene)
    UnloadShaderResources(app.shaders)


def main():
    screen_width = 1280
    screen_height = 720

    SetConfigFlags(FLAG_VSYNC_HINT)
    InitWindow(screen_width, screen_height, b"GenoViewPython")
    SetTargetFPS(60)

    app = None
    try:
        app = _create_app_state(screen_width, screen_height)
        rlSetClipPlanes(0.01, 50.0)

        while not WindowShouldClose():
            DrainAsyncProgress(app)
            PollAsyncClipLoad(app, _commit_clip_switch)
            PollAsyncFeatureLoads(app)
            SyncFeatureMounts(app, RequestFeatureLoad)
            animation_frame = app.debug.playback.update(GetFrameTime())
            frame_state = BuildFrameState(app, animation_frame)
            UpdateTracking(app, frame_state)

            rlDisableColorBlend()
            BeginDrawing()

            RenderFrame(app, frame_state)

            rlEnableColorBlend()
            DrawAppUI(app, frame_state, resource_path)
            EndDrawing()
    finally:
        _unload_app_resources(app)
        CloseWindow()


if __name__ == "__main__":
    main()
