from pathlib import Path
from queue import Empty

from genoview.State import AsyncProgressEvent, PendingClipLoad, PendingFeatureLoad
from genoview.modules.BVHImporter import LoadMotionResources
from genoview.modules.FeatureModule import (
    CloneMotionForFeatureLoad,
    CommitFeatureLoadResult,
    IsClipFeatureReady,
    PrepareFeatureLoad,
)


def BeginAsyncProgress(app, title, detail="", progress=0.0, indeterminate=False):
    app.async_progress.request_id += 1
    request_id = app.async_progress.request_id
    app.async_progress.active = True
    app.async_progress.title = str(title)
    app.async_progress.detail = str(detail)
    app.async_progress.progress = float(max(0.0, min(1.0, progress)))
    app.async_progress.indeterminate = bool(indeterminate)
    return request_id


def PostAsyncProgress(app, request_id, title="", detail="", progress=0.0, active=True, indeterminate=False):
    app.async_progress_events.put(AsyncProgressEvent(
        request_id=int(request_id),
        title=str(title),
        detail=str(detail),
        progress=float(max(0.0, min(1.0, progress))),
        active=bool(active),
        indeterminate=bool(indeterminate),
    ))


def DrainAsyncProgress(app):
    while True:
        try:
            event = app.async_progress_events.get_nowait()
        except Empty:
            return

        if event.request_id != app.async_progress.request_id:
            continue

        app.async_progress.active = event.active
        app.async_progress.title = event.title
        app.async_progress.detail = event.detail
        app.async_progress.progress = event.progress
        app.async_progress.indeterminate = event.indeterminate


def FinishAsyncProgress(app, request_id, title="Ready", detail="", progress=1.0):
    PostAsyncProgress(
        app,
        request_id,
        title=title,
        detail=detail,
        progress=progress,
        active=False,
    )


def RequestFeatureLoad(app, feature_id):
    if IsClipFeatureReady(app, feature_id):
        app.features.mount_clip(app, feature_id)
        return True
    if feature_id == "terrain_model" and IsClipFeatureReady(app, "terrain_height_grid"):
        app.features.mount_clip(app, feature_id)
        return True
    if feature_id == "pose_model":
        app.features.mount_clip(app, feature_id)
        return True
    if feature_id in app.pending_feature_loads:
        return False

    request_id = BeginAsyncProgress(
        app,
        "Loading module",
        feature_id.replace("_", " "),
        progress=0.0,
        indeterminate=True,
    )
    future = app.feature_load_executor.submit(
        PrepareFeatureLoad,
        feature_id,
        CloneMotionForFeatureLoad(app.motion),
        app.scene.ground_position.y,
    )
    app.pending_feature_loads[feature_id] = PendingFeatureLoad(
        request_id=request_id,
        feature_id=feature_id,
        clip_resource=app.motion.clip_resource,
        future=future,
    )
    PostAsyncProgress(
        app,
        request_id,
        title="Loading module",
        detail="Preparing " + feature_id.replace("_", " "),
        progress=0.2,
        indeterminate=True,
    )
    return False


def RequestClipSwitch(app, clip_index, resource_path):
    clip_index = int(clip_index) % len(app.clip_resources)
    clip_resource = app.clip_resources[clip_index]
    request_id = BeginAsyncProgress(
        app,
        "Loading clip",
        Path(clip_resource).name,
        progress=0.0,
        indeterminate=True,
    )

    if app.pending_clip_load is not None:
        app.pending_clip_load.future.cancel()

    future = app.clip_load_executor.submit(LoadMotionResources, resource_path, clip_resource)
    app.pending_clip_load = PendingClipLoad(
        request_id=request_id,
        clip_index=clip_index,
        clip_resource=clip_resource,
        future=future,
    )
    PostAsyncProgress(
        app,
        request_id,
        title="Loading clip",
        detail="Parsing " + Path(clip_resource).name,
        progress=0.15,
        indeterminate=True,
    )


def PollAsyncClipLoad(app, commit_clip_switch):
    pending = app.pending_clip_load
    if pending is None or not pending.future.done():
        return

    app.pending_clip_load = None
    if pending.request_id != app.async_progress.request_id:
        return

    try:
        motion = pending.future.result()
    except Exception as exc:
        PostAsyncProgress(
            app,
            pending.request_id,
            title="Clip load failed",
            detail=str(exc),
            progress=1.0,
            active=False,
        )
        return

    PostAsyncProgress(
        app,
        pending.request_id,
        title="Loading clip",
        detail="Committing " + Path(pending.clip_resource).name,
        progress=0.9,
        active=True,
    )
    DrainAsyncProgress(app)
    commit_clip_switch(app, pending.clip_index, motion)
    FinishAsyncProgress(
        app,
        pending.request_id,
        title="Clip loaded",
        detail=Path(pending.clip_resource).name,
    )


def PollAsyncFeatureLoads(app):
    completed_feature_ids = [
        feature_id
        for feature_id, pending in app.pending_feature_loads.items()
        if pending.future.done()
    ]
    for feature_id in completed_feature_ids:
        pending = app.pending_feature_loads.pop(feature_id)
        if pending.clip_resource != app.motion.clip_resource:
            continue
        try:
            result = pending.future.result()
        except Exception as exc:
            PostAsyncProgress(
                app,
                pending.request_id,
                title="Module load failed",
                detail=str(exc),
                progress=1.0,
                active=False,
            )
            continue

        PostAsyncProgress(
            app,
            pending.request_id,
            title="Loading module",
            detail="Committing " + feature_id.replace("_", " "),
            progress=0.9,
            active=True,
        )
        DrainAsyncProgress(app)
        CommitFeatureLoadResult(app, result)
        FinishAsyncProgress(
            app,
            pending.request_id,
            title="Module loaded",
            detail=feature_id.replace("_", " "),
        )
