from dataclasses import replace
from types import SimpleNamespace

from raylib import UnloadModel

from genoview.FeatureRegistry import Feature, FeatureRegistry
from genoview.State import FeatureLoadResult
from genoview.modules.CharacterModel import LoadCharacterModel
from genoview.modules.ContactModule import BuildBodyProxyLayout, BuildContactData
from genoview.modules.LabelModule import BuildAutoFrameLabels, LoadLabelAnnotations
from genoview.modules.PoseModule import BuildPoseSource
from genoview.modules.RootModule import (
    ROOT_JOINT_INDEX,
    AdaptRootTrajectoryToTerrain,
    BuildRootTrajectorySource,
)
from genoview.modules.TerrainModule import (
    BuildTerrainHeightGrid,
    BuildTerrainProviderFromContactData,
    LoadTerrainModelFromHeightGrid,
)


def EnsureBasePoseSource(motion):
    if motion.base_pose_source is None:
        motion.base_pose_source = BuildPoseSource(
            motion.global_positions,
            motion.global_rotations,
            motion.bvh_frame_time,
        )
    return motion.base_pose_source


def EnsureBootstrapContactResources(motion):
    if motion.bootstrap_contact_data is None:
        base_pose_source = EnsureBasePoseSource(motion)
        motion.bootstrap_contact_data = BuildContactData(
            motion.global_positions,
            base_pose_source["global_velocities"],
            motion.bvh_animation.raw_data["names"],
            bootstrap=True,
        )
    return motion.bootstrap_contact_data


def EnsureTerrainProviderResources(scene, motion):
    if motion.terrain_provider is None:
        bootstrap_contact_data = EnsureBootstrapContactResources(motion)
        motion.terrain_provider = BuildTerrainProviderFromContactData(
            bootstrap_contact_data,
            filtered=True,
            fallbackHeight=scene.ground_position.y,
        )
    return motion.terrain_provider


def EnsureContactResources(scene, motion):
    if motion.contact_data is None:
        base_pose_source = EnsureBasePoseSource(motion)
        terrain_provider = EnsureTerrainProviderResources(scene, motion)
        motion.contact_data = BuildContactData(
            motion.global_positions,
            base_pose_source["global_velocities"],
            motion.bvh_animation.raw_data["names"],
            terrainProvider=terrain_provider,
        )
    return motion.contact_data


def EnsureBodyProxyLayoutResources(motion):
    if motion.body_proxy_layout is None:
        motion.body_proxy_layout = BuildBodyProxyLayout(
            motion.global_positions[0],
            motion.bvh_animation.parents,
            motion.bvh_animation.raw_data["names"],
        )
    return motion.body_proxy_layout


def EnsureTerrainHeightGridResources(scene, motion):
    if motion.terrain_height_grid is None:
        terrain_provider = EnsureTerrainProviderResources(scene, motion)
        motion.terrain_height_grid = BuildTerrainHeightGrid(
            terrain_provider,
            terrain_provider.sample_positions,
            cellSize=0.1,
            padding=0.5,
        )
    return motion.terrain_height_grid


def EnsureTerrainModelResources(scene, motion):
    if motion.terrain_model is None:
        motion.terrain_model, motion.terrain_height_grid = LoadTerrainModelFromHeightGrid(
            EnsureTerrainHeightGridResources(scene, motion),
        )
    return motion.terrain_model, motion.terrain_height_grid


def EnsureMotionRootTrajectoryResources(motion):
    if motion.motion_root_trajectory is None:
        motion.motion_root_trajectory = BuildRootTrajectorySource(
            motion.global_positions,
            motion.global_rotations,
            motion.bvh_frame_time,
            rootIndex=ROOT_JOINT_INDEX,
            mode="height_3d",
        )
    return motion.motion_root_trajectory


def EnsureTerrainAdaptedRootTrajectoryResources(scene, motion):
    if motion.terrain_adapted_root_trajectory is None:
        motion.terrain_adapted_root_trajectory = AdaptRootTrajectoryToTerrain(
            EnsureMotionRootTrajectoryResources(motion),
            EnsureTerrainProviderResources(scene, motion),
            alignPositionsToTerrain=False,
        )
    return motion.terrain_adapted_root_trajectory


def EnsurePoseSourceResources(motion):
    if motion.pose_source is None:
        motion.pose_source = BuildPoseSource(
            motion.global_positions,
            motion.global_rotations,
            motion.bvh_frame_time,
            rootTrajectorySource=EnsureMotionRootTrajectoryResources(motion),
        )
    return motion.pose_source


def EnsureLabelResources(scene, motion):
    if motion.label_result is None:
        motion.label_result = BuildAutoFrameLabels(
            motion.clip_resource,
            motion.global_positions,
            EnsurePoseSourceResources(motion),
            EnsureMotionRootTrajectoryResources(motion),
            contactData=EnsureContactResources(scene, motion),
            terrainProvider=EnsureTerrainProviderResources(scene, motion),
            jointNames=motion.bvh_animation.raw_data["names"],
        )
        LoadLabelAnnotations(motion.label_result, motion.clip_resource)
    return motion.label_result


def EnsureTerrainSampleNormalsResources(scene, motion):
    if motion.terrain_sample_normals is None:
        terrain_provider = EnsureTerrainProviderResources(scene, motion)
        motion.terrain_sample_normals = terrain_provider.sample_normals(terrain_provider.sample_positions)
    return motion.terrain_sample_normals


def DisposeTerrainModelFeature(app):
    if app.motion.terrain_model is not None:
        UnloadModel(app.motion.terrain_model)
        app.motion.terrain_model = None


def EnsurePoseModelResources(scene, resource_path):
    if scene.pose_model is None:
        scene.pose_model = LoadCharacterModel(resource_path("Geno.bin", as_bytes=True))
    return scene.pose_model


def DisposePoseModelFeature(app):
    if app.scene.pose_model is not None:
        UnloadModel(app.scene.pose_model)
        app.scene.pose_model = None


def _make_feature(feature_id, ensure_clip, ready_field, dependencies=(), dispose_clip=None, ready_target="motion"):
    return Feature(
        feature_id,
        dependencies=dependencies,
        ensure_clip=ensure_clip,
        dispose_clip=dispose_clip,
        is_ready=lambda app: getattr(getattr(app, ready_target), ready_field) is not None,
    )


def BuildFeatureRegistry(resource_path):
    return FeatureRegistry({
        "base_pose_source": _make_feature(
            "base_pose_source",
            lambda app: EnsureBasePoseSource(app.motion),
            "base_pose_source",
        ),
        "bootstrap_contacts": _make_feature(
            "bootstrap_contacts",
            lambda app: EnsureBootstrapContactResources(app.motion),
            "bootstrap_contact_data",
            dependencies=("base_pose_source",),
        ),
        "terrain_provider": _make_feature(
            "terrain_provider",
            lambda app: EnsureTerrainProviderResources(app.scene, app.motion),
            "terrain_provider",
            dependencies=("bootstrap_contacts",),
        ),
        "contact_data": _make_feature(
            "contact_data",
            lambda app: EnsureContactResources(app.scene, app.motion),
            "contact_data",
            dependencies=("base_pose_source", "terrain_provider"),
        ),
        "body_proxy_layout": _make_feature(
            "body_proxy_layout",
            lambda app: EnsureBodyProxyLayoutResources(app.motion),
            "body_proxy_layout",
        ),
        "terrain_height_grid": _make_feature(
            "terrain_height_grid",
            lambda app: EnsureTerrainHeightGridResources(app.scene, app.motion),
            "terrain_height_grid",
            dependencies=("terrain_provider",),
        ),
        "terrain_model": _make_feature(
            "terrain_model",
            lambda app: EnsureTerrainModelResources(app.scene, app.motion),
            "terrain_model",
            dependencies=("terrain_height_grid",),
            dispose_clip=DisposeTerrainModelFeature,
        ),
        "motion_root_trajectory": _make_feature(
            "motion_root_trajectory",
            lambda app: EnsureMotionRootTrajectoryResources(app.motion),
            "motion_root_trajectory",
        ),
        "terrain_adapted_root_trajectory": _make_feature(
            "terrain_adapted_root_trajectory",
            lambda app: EnsureTerrainAdaptedRootTrajectoryResources(app.scene, app.motion),
            "terrain_adapted_root_trajectory",
            dependencies=("motion_root_trajectory", "terrain_provider"),
        ),
        "pose_source": _make_feature(
            "pose_source",
            lambda app: EnsurePoseSourceResources(app.motion),
            "pose_source",
            dependencies=("motion_root_trajectory",),
        ),
        "pose_model": _make_feature(
            "pose_model",
            lambda app: EnsurePoseModelResources(app.scene, resource_path),
            "pose_model",
            dispose_clip=DisposePoseModelFeature,
            ready_target="scene",
        ),
        "labels": _make_feature(
            "labels",
            lambda app: EnsureLabelResources(app.scene, app.motion),
            "label_result",
            dependencies=("pose_source", "contact_data", "terrain_provider"),
        ),
        "terrain_sample_normals": _make_feature(
            "terrain_sample_normals",
            lambda app: EnsureTerrainSampleNormalsResources(app.scene, app.motion),
            "terrain_sample_normals",
            dependencies=("terrain_provider",),
        ),
    })


def EnsureClipFeature(app, feature_id):
    return app.features.ensure_clip(app, feature_id)


def IsClipFeatureReady(app, feature_id):
    return app.features.is_clip_ready(app, feature_id)


def CloneMotionForFeatureLoad(motion):
    return replace(motion, terrain_model=None)


def _prepare_feature_resources(feature_id, scene, motion):
    loaders = {
        "base_pose_source": lambda: EnsureBasePoseSource(motion),
        "bootstrap_contacts": lambda: EnsureBootstrapContactResources(motion),
        "terrain_provider": lambda: EnsureTerrainProviderResources(scene, motion),
        "contact_data": lambda: EnsureContactResources(scene, motion),
        "body_proxy_layout": lambda: EnsureBodyProxyLayoutResources(motion),
        "terrain_height_grid": lambda: EnsureTerrainHeightGridResources(scene, motion),
        "terrain_model": lambda: EnsureTerrainHeightGridResources(scene, motion),
        "motion_root_trajectory": lambda: EnsureMotionRootTrajectoryResources(motion),
        "terrain_adapted_root_trajectory": lambda: EnsureTerrainAdaptedRootTrajectoryResources(scene, motion),
        "pose_source": lambda: EnsurePoseSourceResources(motion),
        "labels": lambda: EnsureLabelResources(scene, motion),
        "terrain_sample_normals": lambda: EnsureTerrainSampleNormalsResources(scene, motion),
    }
    loader = loaders.get(feature_id)
    if loader is not None:
        loader()


def PrepareFeatureLoad(feature_id, motion, ground_height):
    staging_motion = CloneMotionForFeatureLoad(motion)
    scene_stub = SimpleNamespace(ground_position=SimpleNamespace(y=float(ground_height)))
    _prepare_feature_resources(feature_id, scene_stub, staging_motion)

    return FeatureLoadResult(
        feature_id=feature_id,
        clip_resource=staging_motion.clip_resource,
        motion=staging_motion,
    )


def CommitFeatureLoadResult(app, result):
    if result.clip_resource != app.motion.clip_resource:
        return

    source = result.motion
    target = app.motion
    for field_name in (
        "base_pose_source",
        "bootstrap_contact_data",
        "terrain_provider",
        "contact_data",
        "body_proxy_layout",
        "terrain_height_grid",
        "motion_root_trajectory",
        "terrain_adapted_root_trajectory",
        "pose_source",
        "label_result",
        "terrain_sample_normals",
    ):
        value = getattr(source, field_name)
        if value is not None:
            setattr(target, field_name, value)

    if not IsFeatureRequested(app, result.feature_id):
        return

    if result.feature_id == "terrain_model" and target.terrain_height_grid is not None:
        app.features.mount_clip(app, "terrain_model")
    elif result.feature_id != "pose_model":
        app.features.mount_clip(app, result.feature_id)


def IsFeatureRequested(app, feature_id):
    return dict(_requested_feature_states(app.debug)).get(feature_id, False)


def _requested_feature_states(debug):
    wants_terrain_height_grid = (
        debug.draw_terrain_mesh_ptr[0] or
        debug.draw_root_trajectory_ptr[0] or
        debug.draw_terrain_samples_ptr[0] or
        debug.draw_terrain_normals_ptr[0] or
        debug.draw_terrain_penetration_ptr[0]
    )
    wants_pose_source = (
        debug.draw_reconstructed_pose_ptr[0] or
        debug.draw_pose_model_local_ptr[0] or
        debug.draw_reconstruction_error_ptr[0] or
        debug.draw_contacts_ptr[0] or
        debug.draw_bootstrap_contacts_ptr[0]
    )
    return (
        ("labels", debug.label_module_ptr[0]),
        ("terrain_model", debug.draw_terrain_mesh_ptr[0]),
        ("terrain_height_grid", wants_terrain_height_grid),
        ("terrain_adapted_root_trajectory", debug.draw_root_trajectory_ptr[0]),
        ("contact_data", debug.draw_contacts_ptr[0]),
        ("bootstrap_contacts", debug.draw_bootstrap_contacts_ptr[0]),
        ("terrain_sample_normals", debug.draw_terrain_normals_ptr[0]),
        ("body_proxy_layout", debug.draw_body_proxy_ptr[0] or debug.draw_terrain_penetration_ptr[0]),
        ("pose_source", wants_pose_source),
    )


def SyncFeatureMounts(app, request_feature_load):
    debug = app.debug
    for feature_id, should_mount in _requested_feature_states(debug):
        if should_mount:
            request_feature_load(app, feature_id)
        else:
            app.features.unmount_clip(app, feature_id)
    app.features.sync_clip_mount(app, "pose_model", debug.draw_reconstructed_pose_ptr[0])
