from pyray import (
    Vector2, Vector3,
    Color, Rectangle,
    Texture, RenderTexture)
from raylib import *
from raylib.defines import *

from types import SimpleNamespace
from pathlib import Path
import numpy as np
import cffi
from BVHImporter import BVHImporter
from CameraController import Camera
from CharacterModel import (
    LoadCharacterModel,
    GetModelBindPoseAsNumpyArrays,
    UpdateModelPoseFromNumpyArrays,
)
from BodyProxyModule import (
    BuildBodyProxyLayout,
    BuildBodyProxyFrame,
    ComputeTerrainPenetrationFrame,
)
from ContactModule import BuildContactData
from RootModule import (
    ROOT_JOINT_INDEX,
    DEFAULT_BVH_FRAME_TIME,
    GetRootTrajectorySampleOffsets,
    BuildRootTrajectorySource,
    AdaptRootTrajectoryToTerrain,
    BuildRootLocalTrajectory,
    BuildTerrainAdaptedRootTrajectoryDisplay,
)
from TerrainModule import BuildTerrainProviderFromContactData, LoadTerrainModelFromProvider
from PoseModule import (
    BuildPoseSource,
    BuildLocalPose,
    ReconstructPoseWorldSpace,
    ComputePosePositionError,
)
from DebugDraw import (
    DrawSkeleton,
    OffsetPositions,
    DrawPoseReconstructionError,
    DrawContactStates,
    DrawBodyProxyFrame,
    DrawTerrainPenetrationFrame,
    DrawTerrainSamples,
    DrawTerrainNormals,
    DrawRootTrajectoryDebug,
)
from PlaybackController import PlaybackController
from LabelModule import (
    ACTION_LABELS,
    DEFAULT_TRANSITION_FRAMES,
    BuildAutoFrameLabels,
    ApplyManualLabelRange,
    ApplyTransitionWidthRange,
    ClearManualLabelRange,
    ClearTransitionWidthRange,
    ExportCompiledLabels,
    LoadLabelAnnotations,
    ResetManualLabels,
    SaveLabelAnnotations,
)

ffi = cffi.FFI()

BASE_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = BASE_DIR / "resources"
DEFAULT_BVH_CLIP = "bvh/lafan1/jumps1_subject1.bvh"
DEFAULT_BVH_DIR = "bvh/lafan1"


def resource_path(*parts, as_bytes=False):
    path = RESOURCES_DIR.joinpath(*parts)
    return str(path).encode("utf-8") if as_bytes else str(path)


def _discover_bvh_clips():
    bvh_dir = RESOURCES_DIR / DEFAULT_BVH_DIR
    clips = [
        str(path.relative_to(RESOURCES_DIR))
        for path in sorted(bvh_dir.glob("*.bvh"))
    ]
    return clips if clips else [DEFAULT_BVH_CLIP]


def _get_clip_index(clip_resources, clip_resource):
    try:
        return clip_resources.index(clip_resource)
    except ValueError:
        return 0


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

#----------------------------------------------------------------------------------
# Shadow Maps
#----------------------------------------------------------------------------------

class ShadowLight:

    def __init__(self):

        self.target = Vector3Zero()
        self.position = Vector3Zero()
        self.up = Vector3(0.0, 1.0, 0.0)
        self.target = Vector3Zero()
        self.width = 0
        self.height = 0
        self.near = 0.0
        self.far = 1.0


def LoadShadowMap(width, height):

    target = RenderTexture()
    target.id = rlLoadFramebuffer()
    target.texture.width = width
    target.texture.height = height
    assert target.id != 0

    rlEnableFramebuffer(target.id)

    target.depth.id = rlLoadTextureDepth(width, height, False)
    target.depth.width = width
    target.depth.height = height
    target.depth.format = 19       #DEPTH_COMPONENT_24BIT?
    target.depth.mipmaps = 1
    rlFramebufferAttach(target.id, target.depth.id, RL_ATTACHMENT_DEPTH, RL_ATTACHMENT_TEXTURE2D, 0)
    assert rlFramebufferComplete(target.id)

    rlDisableFramebuffer()

    return target

def UnloadShadowMap(target):

    if target.id > 0:
        rlUnloadFramebuffer(target.id)


def BeginShadowMap(target, shadowLight):

    BeginTextureMode(target)
    ClearBackground(WHITE)

    rlDrawRenderBatchActive()      # Update and draw internal render batch

    rlMatrixMode(RL_PROJECTION)    # Switch to projection matrix
    rlPushMatrix()                 # Save previous matrix, which contains the settings for the 2d ortho projection
    rlLoadIdentity()               # Reset current matrix (projection)

    rlOrtho(
        -shadowLight.width/2, shadowLight.width/2,
        -shadowLight.height/2, shadowLight.height/2,
        shadowLight.near, shadowLight.far)

    rlMatrixMode(RL_MODELVIEW)     # Switch back to modelview matrix
    rlLoadIdentity()               # Reset current matrix (modelview)

    # Setup Camera view
    matView = MatrixLookAt(shadowLight.position, shadowLight.target, shadowLight.up)
    rlMultMatrixf(MatrixToFloatV(matView).v)      # Multiply modelview matrix by view matrix (camera)

    rlEnableDepthTest()            # Enable DEPTH_TEST for 3D


def EndShadowMap():
    rlDrawRenderBatchActive()       # Update and draw internal render batch

    rlMatrixMode(RL_PROJECTION)     # Switch to projection matrix
    rlPopMatrix()                   # Restore previous matrix (projection) from matrix stack

    rlMatrixMode(RL_MODELVIEW)      # Switch back to modelview matrix
    rlLoadIdentity()                # Reset current matrix (modelview)

    rlDisableDepthTest()            # Disable DEPTH_TEST for 2D

    EndTextureMode()

def SetShaderValueShadowMap(shader, locIndex, target):
    if locIndex > -1:
        rlEnableShader(shader.id)
        slotPtr = ffi.new('int*'); slotPtr[0] = 10  # Can be anything 0 to 15, but 0 will probably be taken up
        rlActiveTextureSlot(slotPtr[0])
        rlEnableTexture(target.depth.id)
        rlSetUniform(locIndex, slotPtr, SHADER_UNIFORM_INT, 1)

#----------------------------------------------------------------------------------
# GBuffer
#----------------------------------------------------------------------------------

class GBuffer:

    def __init__(self):
        self.id = 0              # OpenGL framebuffer object id
        self.color = Texture()   # Color buffer attachment texture
        self.normal = Texture()  # Normal buffer attachment texture
        self.depth = Texture()   # Depth buffer attachment texture


def LoadGBuffer(width, height):

    target = GBuffer()
    target.id = rlLoadFramebuffer()
    assert target.id

    rlEnableFramebuffer(target.id)

    target.color.id = rlLoadTexture(ffi.NULL, width, height, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8, 1)
    target.color.width = width
    target.color.height = height
    target.color.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
    target.color.mipmaps = 1
    rlFramebufferAttach(target.id, target.color.id, RL_ATTACHMENT_COLOR_CHANNEL0, RL_ATTACHMENT_TEXTURE2D, 0)

    target.normal.id = rlLoadTexture(ffi.NULL, width, height, PIXELFORMAT_UNCOMPRESSED_R16G16B16A16, 1)
    target.normal.width = width
    target.normal.height = height
    target.normal.format = PIXELFORMAT_UNCOMPRESSED_R16G16B16A16
    target.normal.mipmaps = 1
    rlFramebufferAttach(target.id, target.normal.id, RL_ATTACHMENT_COLOR_CHANNEL1, RL_ATTACHMENT_TEXTURE2D, 0)

    target.depth.id = rlLoadTextureDepth(width, height, False)
    target.depth.width = width
    target.depth.height = height
    target.depth.format = 19       #DEPTH_COMPONENT_24BIT?
    target.depth.mipmaps = 1
    rlFramebufferAttach(target.id, target.depth.id, RL_ATTACHMENT_DEPTH, RL_ATTACHMENT_TEXTURE2D, 0)

    assert rlFramebufferComplete(target.id)

    rlDisableFramebuffer()

    return target


def UnloadGBuffer(target):

    if target.id > 0:
        rlUnloadFramebuffer(target.id)


def BeginGBuffer(target, camera):

    rlDrawRenderBatchActive()       # Update and draw internal render batch

    rlEnableFramebuffer(target.id)  # Enable render target
    rlActiveDrawBuffers(2)

    # Set viewport and RLGL internal framebuffer size
    rlViewport(0, 0, target.color.width, target.color.height)
    rlSetFramebufferWidth(target.color.width)
    rlSetFramebufferHeight(target.color.height)

    ClearBackground(BLACK)

    rlMatrixMode(RL_PROJECTION)    # Switch to projection matrix
    rlPushMatrix()                 # Save previous matrix, which contains the settings for the 2d ortho projection
    rlLoadIdentity()               # Reset current matrix (projection)

    aspect = float(target.color.width)/float(target.color.height)

    # NOTE: zNear and zFar values are important when computing depth buffer values
    if camera.projection == CAMERA_PERSPECTIVE:

        # Setup perspective projection
        top = rlGetCullDistanceNear()*np.tan(camera.fovy*0.5*DEG2RAD)
        right = top*aspect

        rlFrustum(-right, right, -top, top, rlGetCullDistanceNear(), rlGetCullDistanceFar())

    elif camera.projection == CAMERA_ORTHOGRAPHIC:

        # Setup orthographic projection
        top = camera.fovy/2.0
        right = top*aspect

        rlOrtho(-right, right, -top,top, rlGetCullDistanceNear(), rlGetCullDistanceFar())

    rlMatrixMode(RL_MODELVIEW)     # Switch back to modelview matrix
    rlLoadIdentity()               # Reset current matrix (modelview)

    # Setup Camera view
    matView = MatrixLookAt(camera.position, camera.target, camera.up)
    rlMultMatrixf(MatrixToFloatV(matView).v)      # Multiply modelview matrix by view matrix (camera)

    rlEnableDepthTest()            # Enable DEPTH_TEST for 3D


def EndGBuffer(windowWidth, windowHeight):

    rlDrawRenderBatchActive()       # Update and draw internal render batch

    rlDisableDepthTest()            # Disable DEPTH_TEST for 2D
    rlActiveDrawBuffers(1)
    rlDisableFramebuffer()          # Disable render target (fbo)

    rlMatrixMode(RL_PROJECTION)         # Switch to projection matrix
    rlPopMatrix()                   # Restore previous matrix (projection) from matrix stack
    rlLoadIdentity()                    # Reset current matrix (projection)
    rlOrtho(0, windowWidth, windowHeight, 0, 0.0, 1.0)

    rlMatrixMode(RL_MODELVIEW)          # Switch back to modelview matrix
    rlLoadIdentity()                    # Reset current matrix (modelview)


#----------------------------------------------------------------------------------
# App
#----------------------------------------------------------------------------------

def _make_bool_ptr(value):
    ptr = ffi.new("bool*")
    ptr[0] = bool(value)
    return ptr


def _make_float_ptr(value):
    ptr = ffi.new("float*")
    ptr[0] = float(value)
    return ptr


def _draw_model_with_shader(model, shader, position, color):
    model.materials[0].shader = shader
    DrawModel(model, position, 1.0, color)


def _load_shader_resources():
    shadow_program = LoadShader(resource_path("shadow.vs", as_bytes=True), resource_path("shadow.fs", as_bytes=True))
    skinned_shadow_program = LoadShader(
        resource_path("skinnedShadow.vs", as_bytes=True),
        resource_path("shadow.fs", as_bytes=True),
    )
    skinned_basic_program = LoadShader(
        resource_path("skinnedBasic.vs", as_bytes=True),
        resource_path("basic.fs", as_bytes=True),
    )
    basic_program = LoadShader(resource_path("basic.vs", as_bytes=True), resource_path("basic.fs", as_bytes=True))
    lighting_program = LoadShader(resource_path("post.vs", as_bytes=True), resource_path("lighting.fs", as_bytes=True))
    ssao_program = LoadShader(resource_path("post.vs", as_bytes=True), resource_path("ssao.fs", as_bytes=True))
    blur_program = LoadShader(resource_path("post.vs", as_bytes=True), resource_path("blur.fs", as_bytes=True))
    fxaa_program = LoadShader(resource_path("post.vs", as_bytes=True), resource_path("fxaa.fs", as_bytes=True))

    return SimpleNamespace(
        shadow=SimpleNamespace(
            program=shadow_program,
            light_clip_near=GetShaderLocation(shadow_program, b"lightClipNear"),
            light_clip_far=GetShaderLocation(shadow_program, b"lightClipFar"),
        ),
        skinned_shadow=SimpleNamespace(
            program=skinned_shadow_program,
            light_clip_near=GetShaderLocation(skinned_shadow_program, b"lightClipNear"),
            light_clip_far=GetShaderLocation(skinned_shadow_program, b"lightClipFar"),
        ),
        skinned_basic=SimpleNamespace(
            program=skinned_basic_program,
            specularity=GetShaderLocation(skinned_basic_program, b"specularity"),
            glossiness=GetShaderLocation(skinned_basic_program, b"glossiness"),
            cam_clip_near=GetShaderLocation(skinned_basic_program, b"camClipNear"),
            cam_clip_far=GetShaderLocation(skinned_basic_program, b"camClipFar"),
        ),
        basic=SimpleNamespace(
            program=basic_program,
            specularity=GetShaderLocation(basic_program, b"specularity"),
            glossiness=GetShaderLocation(basic_program, b"glossiness"),
            cam_clip_near=GetShaderLocation(basic_program, b"camClipNear"),
            cam_clip_far=GetShaderLocation(basic_program, b"camClipFar"),
        ),
        lighting=SimpleNamespace(
            program=lighting_program,
            gbuffer_color=GetShaderLocation(lighting_program, b"gbufferColor"),
            gbuffer_normal=GetShaderLocation(lighting_program, b"gbufferNormal"),
            gbuffer_depth=GetShaderLocation(lighting_program, b"gbufferDepth"),
            ssao=GetShaderLocation(lighting_program, b"ssao"),
            cam_pos=GetShaderLocation(lighting_program, b"camPos"),
            cam_inv_view_proj=GetShaderLocation(lighting_program, b"camInvViewProj"),
            light_dir=GetShaderLocation(lighting_program, b"lightDir"),
            sun_color=GetShaderLocation(lighting_program, b"sunColor"),
            sun_strength=GetShaderLocation(lighting_program, b"sunStrength"),
            sky_color=GetShaderLocation(lighting_program, b"skyColor"),
            sky_strength=GetShaderLocation(lighting_program, b"skyStrength"),
            ground_strength=GetShaderLocation(lighting_program, b"groundStrength"),
            ambient_strength=GetShaderLocation(lighting_program, b"ambientStrength"),
            exposure=GetShaderLocation(lighting_program, b"exposure"),
            cam_clip_near=GetShaderLocation(lighting_program, b"camClipNear"),
            cam_clip_far=GetShaderLocation(lighting_program, b"camClipFar"),
        ),
        ssao=SimpleNamespace(
            program=ssao_program,
            gbuffer_normal=GetShaderLocation(ssao_program, b"gbufferNormal"),
            gbuffer_depth=GetShaderLocation(ssao_program, b"gbufferDepth"),
            cam_view=GetShaderLocation(ssao_program, b"camView"),
            cam_proj=GetShaderLocation(ssao_program, b"camProj"),
            cam_inv_proj=GetShaderLocation(ssao_program, b"camInvProj"),
            cam_inv_view_proj=GetShaderLocation(ssao_program, b"camInvViewProj"),
            light_view_proj=GetShaderLocation(ssao_program, b"lightViewProj"),
            shadow_map=GetShaderLocation(ssao_program, b"shadowMap"),
            shadow_inv_resolution=GetShaderLocation(ssao_program, b"shadowInvResolution"),
            cam_clip_near=GetShaderLocation(ssao_program, b"camClipNear"),
            cam_clip_far=GetShaderLocation(ssao_program, b"camClipFar"),
            light_clip_near=GetShaderLocation(ssao_program, b"lightClipNear"),
            light_clip_far=GetShaderLocation(ssao_program, b"lightClipFar"),
            light_dir=GetShaderLocation(ssao_program, b"lightDir"),
        ),
        blur=SimpleNamespace(
            program=blur_program,
            gbuffer_normal=GetShaderLocation(blur_program, b"gbufferNormal"),
            gbuffer_depth=GetShaderLocation(blur_program, b"gbufferDepth"),
            input_texture=GetShaderLocation(blur_program, b"inputTexture"),
            cam_inv_proj=GetShaderLocation(blur_program, b"camInvProj"),
            cam_clip_near=GetShaderLocation(blur_program, b"camClipNear"),
            cam_clip_far=GetShaderLocation(blur_program, b"camClipFar"),
            inv_texture_resolution=GetShaderLocation(blur_program, b"invTextureResolution"),
            blur_direction=GetShaderLocation(blur_program, b"blurDirection"),
        ),
        fxaa=SimpleNamespace(
            program=fxaa_program,
            input_texture=GetShaderLocation(fxaa_program, b"inputTexture"),
            inv_texture_resolution=GetShaderLocation(fxaa_program, b"invTextureResolution"),
        ),
    )


def _load_scene_resources():
    ground_model = LoadModelFromMesh(GenMeshPlane(20.0, 20.0, 10, 10))
    geno_model = LoadCharacterModel(resource_path("Geno.bin", as_bytes=True))
    pose_model = LoadCharacterModel(resource_path("Geno.bin", as_bytes=True))
    bind_pos, bind_rot = GetModelBindPoseAsNumpyArrays(geno_model)

    return SimpleNamespace(
        ground_model=ground_model,
        ground_position=Vector3(0.0, -0.01, 0.0),
        geno_model=geno_model,
        pose_model=pose_model,
        geno_position=Vector3(0.0, 0.0, 0.0),
        bind_pos=bind_pos,
        bind_rot=bind_rot,
    )


def _load_motion_resources(scene, clip_resource=DEFAULT_BVH_CLIP):
    bvh_animation = BVHImporter.load(resource_path(clip_resource), scale=0.01)
    global_positions = bvh_animation.global_positions
    global_rotations = bvh_animation.global_rotations
    bvh_frame_time = DEFAULT_BVH_FRAME_TIME
    trajectory_sample_offsets = GetRootTrajectorySampleOffsets()

    base_pose_source = BuildPoseSource(
        global_positions,
        global_rotations,
        bvh_frame_time,
    )
    bootstrap_contact_data = BuildContactData(
        global_positions,
        base_pose_source["global_velocities"],
        bvh_animation.raw_data["names"],
        bootstrap=True,
    )
    terrain_provider = BuildTerrainProviderFromContactData(
        bootstrap_contact_data,
        filtered=True,
        fallbackHeight=scene.ground_position.y,
    )
    contact_data = BuildContactData(
        global_positions,
        base_pose_source["global_velocities"],
        bvh_animation.raw_data["names"],
        terrainProvider=terrain_provider,
    )
    body_proxy_layout = BuildBodyProxyLayout(
        global_positions[0],
        bvh_animation.parents,
        bvh_animation.raw_data["names"],
    )
    terrain_model, terrain_height_grid = LoadTerrainModelFromProvider(
        terrain_provider,
        terrain_provider.sample_positions,
        cellSize=0.1,
        padding=0.5,
    )
    motion_root_trajectory = BuildRootTrajectorySource(
        global_positions,
        global_rotations,
        bvh_frame_time,
        rootIndex=ROOT_JOINT_INDEX,
        mode="height_3d",
    )
    terrain_adapted_root_trajectory = AdaptRootTrajectoryToTerrain(
        motion_root_trajectory,
        terrain_provider,
        alignPositionsToTerrain=False,
    )
    pose_source = BuildPoseSource(
        global_positions,
        global_rotations,
        bvh_frame_time,
        rootTrajectorySource=motion_root_trajectory,
    )
    label_result = BuildAutoFrameLabels(
        clip_resource,
        global_positions,
        pose_source,
        motion_root_trajectory,
        contactData=contact_data,
        terrainProvider=terrain_provider,
        jointNames=bvh_animation.raw_data["names"],
    )
    LoadLabelAnnotations(label_result, clip_resource)

    return SimpleNamespace(
        clip_resource=clip_resource,
        clip_name=Path(clip_resource).stem,
        bvh_animation=bvh_animation,
        parents=bvh_animation.parents,
        global_positions=global_positions,
        global_rotations=global_rotations,
        trajectory_sample_offsets=trajectory_sample_offsets,
        bvh_frame_time=bvh_frame_time,
        bootstrap_contact_data=bootstrap_contact_data,
        terrain_provider=terrain_provider,
        contact_data=contact_data,
        body_proxy_layout=body_proxy_layout,
        terrain_model=terrain_model,
        terrain_height_grid=terrain_height_grid,
        motion_root_trajectory=motion_root_trajectory,
        terrain_adapted_root_trajectory=terrain_adapted_root_trajectory,
        pose_source=pose_source,
        label_result=label_result,
        terrain_sample_normals=terrain_provider.sample_normals(terrain_provider.sample_positions),
    )


def _create_render_resources(screen_width, screen_height):
    light_dir = Vector3Normalize(Vector3(0.35, -1.0, -0.35))

    shadow_light = ShadowLight()
    shadow_light.target = Vector3Zero()
    shadow_light.position = Vector3Scale(light_dir, -5.0)
    shadow_light.up = Vector3(0.0, 1.0, 0.0)
    shadow_light.width = 5.0
    shadow_light.height = 5.0
    shadow_light.near = 0.01
    shadow_light.far = 10.0

    shadow_width = 1024
    shadow_height = 1024

    return SimpleNamespace(
        light_dir=light_dir,
        shadow_light=shadow_light,
        shadow_map=LoadShadowMap(shadow_width, shadow_height),
        shadow_inv_resolution=Vector2(1.0 / shadow_width, 1.0 / shadow_height),
        gbuffer=LoadGBuffer(screen_width, screen_height),
        lighted=LoadRenderTexture(screen_width, screen_height),
        ssao_front=LoadRenderTexture(screen_width, screen_height),
        ssao_back=LoadRenderTexture(screen_width, screen_height),
    )


def _create_debug_state(frame_count, frame_time):
    return SimpleNamespace(
        draw_bone_transforms_ptr=_make_bool_ptr(False),
        draw_flat_ground_ptr=_make_bool_ptr(False),
        draw_terrain_mesh_ptr=_make_bool_ptr(True),
        draw_root_trajectory_ptr=_make_bool_ptr(True),
        draw_trajectory_directions_ptr=_make_bool_ptr(True),
        draw_trajectory_velocity_ptr=_make_bool_ptr(True),
        draw_contacts_ptr=_make_bool_ptr(True),
        draw_bootstrap_contacts_ptr=_make_bool_ptr(False),
        draw_terrain_samples_ptr=_make_bool_ptr(False),
        draw_terrain_normals_ptr=_make_bool_ptr(False),
        draw_body_proxy_ptr=_make_bool_ptr(False),
        draw_terrain_penetration_ptr=_make_bool_ptr(False),
        draw_reconstructed_pose_ptr=_make_bool_ptr(True),
        draw_pose_model_local_ptr=_make_bool_ptr(False),
        draw_reconstruction_error_ptr=_make_bool_ptr(True),
        integrate_root_motion_ptr=_make_bool_ptr(False),
        selected_timeline_mode="final",
        local_debug_origin=Vector3(-2.0, 0.0, 0.0),
        pose_model_color=Color(110, 190, 255, 255),
        selected_action_label="walk",
        transition_width=DEFAULT_TRANSITION_FRAMES,
        annotation_status="Auto-loaded labels",
        playback=PlaybackController(frame_count, frame_time),
    )


def _create_app_state(screen_width, screen_height):
    scene = _load_scene_resources()
    clip_resources = _discover_bvh_clips()
    clip_index = _get_clip_index(clip_resources, DEFAULT_BVH_CLIP)
    motion = _load_motion_resources(scene, clip_resources[clip_index])
    debug = _create_debug_state(motion.bvh_animation.frame_count, motion.bvh_frame_time)
    if motion.label_result.annotation_loaded:
        debug.annotation_status = "Loaded " + Path(motion.label_result.annotation_path).name
    else:
        debug.annotation_status = "No saved annotation"
    return SimpleNamespace(
        screen_width=screen_width,
        screen_height=screen_height,
        shaders=_load_shader_resources(),
        scene=scene,
        motion=motion,
        clip_resources=clip_resources,
        clip_index=clip_index,
        render=_create_render_resources(screen_width, screen_height),
        debug=debug,
        camera=Camera(),
    )


def _set_annotation_status_from_load(debug, label_result):
    if label_result.annotation_loaded:
        debug.annotation_status = "Loaded " + Path(label_result.annotation_path).name
    else:
        debug.annotation_status = "No saved annotation"


def _save_current_annotations(app):
    annotation_path = SaveLabelAnnotations(app.motion.label_result, app.motion.clip_resource)
    app.debug.annotation_status = "Saved " + Path(annotation_path).name
    return annotation_path


def _load_current_annotations(app):
    if LoadLabelAnnotations(app.motion.label_result, app.motion.clip_resource):
        app.debug.annotation_status = "Loaded " + Path(app.motion.label_result.annotation_path).name
        return True
    app.debug.annotation_status = "No saved annotation"
    return False


def _export_current_labels(app):
    export_path = ExportCompiledLabels(app.motion.label_result, app.motion.clip_resource)
    app.debug.annotation_status = "Exported " + Path(export_path).name
    return export_path


def _switch_clip(app, clip_index):
    if not app.clip_resources:
        return

    _save_current_annotations(app)
    if hasattr(app.motion, "terrain_model"):
        UnloadModel(app.motion.terrain_model)

    app.clip_index = int(clip_index) % len(app.clip_resources)
    app.motion = _load_motion_resources(app.scene, app.clip_resources[app.clip_index])
    app.debug.playback = PlaybackController(app.motion.bvh_animation.frame_count, app.motion.bvh_frame_time)
    _set_annotation_status_from_load(app.debug, app.motion.label_result)


def _switch_clip_offset(app, offset):
    _switch_clip(app, app.clip_index + int(offset))


def _handle_annotation_shortcuts(app):
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
            app.motion.label_result,
            selection_start,
            selection_end,
            debug.selected_action_label,
        )
        debug.annotation_status = "Applied selection"
    if IsKeyPressed(globals().get("KEY_BACKSPACE", 259)):
        ClearManualLabelRange(
            app.motion.label_result,
            selection_start,
            selection_end,
        )
        debug.annotation_status = "Cleared selection"

    control_down = (
        IsKeyDown(globals().get("KEY_LEFT_CONTROL", 341)) or
        IsKeyDown(globals().get("KEY_RIGHT_CONTROL", 345))
    )
    if control_down and IsKeyPressed(globals().get("KEY_S", ord("S"))):
        _save_current_annotations(app)
    if control_down and IsKeyPressed(globals().get("KEY_L", ord("L"))):
        _load_current_annotations(app)
    if control_down and IsKeyPressed(globals().get("KEY_E", ord("E"))):
        _export_current_labels(app)


def _build_frame_state(app, animation_frame):
    scene = app.scene
    motion = app.motion
    debug = app.debug

    UpdateModelPoseFromNumpyArrays(
        scene.geno_model,
        scene.bind_pos,
        scene.bind_rot,
        motion.global_positions[animation_frame],
        motion.global_rotations[animation_frame],
    )

    root_trajectory = BuildRootLocalTrajectory(
        motion.motion_root_trajectory,
        animation_frame,
        sampleOffsets=motion.trajectory_sample_offsets,
    )
    terrain_root_trajectory_display = BuildTerrainAdaptedRootTrajectoryDisplay(
        root_trajectory,
        motion.terrain_adapted_root_trajectory,
        heightOffset=0.02,
        alignDirectionsToTerrain=True,
        alignVelocitiesToTerrain=True,
    )
    local_pose = BuildLocalPose(
        motion.pose_source,
        motion.motion_root_trajectory,
        animation_frame,
        dt=motion.bvh_frame_time,
    )
    reconstructed_pose_world = ReconstructPoseWorldSpace(
        local_pose,
        integrateRootMotion=debug.integrate_root_motion_ptr[0],
        dt=motion.bvh_frame_time,
    )

    pose_comparison_frame = (
        min(animation_frame + 1, motion.bvh_animation.frame_count - 1)
        if debug.integrate_root_motion_ptr[0] else
        animation_frame
    )
    pose_comparison_positions = motion.global_positions[pose_comparison_frame]
    pose_error_label = b"Pred Err(+1)" if debug.integrate_root_motion_ptr[0] else b"Recon Err"
    local_pose_positions_offset = OffsetPositions(local_pose["local_positions"], debug.local_debug_origin)

    if debug.draw_pose_model_local_ptr[0]:
        UpdateModelPoseFromNumpyArrays(
            scene.pose_model,
            scene.bind_pos,
            scene.bind_rot,
            local_pose_positions_offset,
            local_pose["local_rotations"],
        )
    else:
        UpdateModelPoseFromNumpyArrays(
            scene.pose_model,
            scene.bind_pos,
            scene.bind_rot,
            reconstructed_pose_world["world_positions"],
            reconstructed_pose_world["world_rotations"],
        )

    pose_position_error_mean, pose_position_error_max = ComputePosePositionError(
        pose_comparison_positions,
        reconstructed_pose_world["world_positions"],
    )

    bootstrap_contact_indices = motion.bootstrap_contact_data["joint_indices"]
    contact_indices = motion.contact_data["joint_indices"]
    bootstrap_frame_contacts = motion.bootstrap_contact_data["contacts_filtered"][animation_frame]
    frame_contacts = motion.contact_data["contacts_filtered"][animation_frame]

    bootstrap_pose_contact_positions = (
        local_pose_positions_offset[bootstrap_contact_indices]
        if debug.draw_pose_model_local_ptr[0] else
        reconstructed_pose_world["world_positions"][bootstrap_contact_indices]
    )
    pose_contact_positions = (
        local_pose_positions_offset[contact_indices]
        if debug.draw_pose_model_local_ptr[0] else
        reconstructed_pose_world["world_positions"][contact_indices]
    )
    pose_focus_position = (
        local_pose_positions_offset[ROOT_JOINT_INDEX]
        if debug.draw_pose_model_local_ptr[0] else
        reconstructed_pose_world["world_positions"][ROOT_JOINT_INDEX]
    )

    terrain_query_position = reconstructed_pose_world["world_positions"][ROOT_JOINT_INDEX]
    body_proxy_frame = BuildBodyProxyFrame(
        motion.global_positions[animation_frame],
        motion.body_proxy_layout,
    )
    penetration_frame = ComputeTerrainPenetrationFrame(
        body_proxy_frame,
        motion.terrain_provider,
    )
    auto_labels = motion.label_result.auto_labels
    auto_segments = motion.label_result.auto_segments
    final_labels = motion.label_result.final_labels
    final_segments = motion.label_result.final_segments
    soft_weights = motion.label_result.soft_weights
    current_auto_label = str(auto_labels[animation_frame]) if auto_labels is not None and len(auto_labels) > 0 else "other"
    current_final_label = str(final_labels[animation_frame]) if final_labels is not None and len(final_labels) > 0 else current_auto_label

    return SimpleNamespace(
        animation_frame=animation_frame,
        frame_count=motion.bvh_animation.frame_count,
        clip_name=motion.clip_name,
        clip_prior=motion.label_result.clip_prior,
        current_auto_label=current_auto_label,
        current_final_label=current_final_label,
        auto_labels=auto_labels,
        auto_segments=auto_segments,
        final_labels=final_labels,
        final_segments=final_segments,
        soft_weights=soft_weights,
        hip_position=Vector3(*pose_focus_position),
        root_trajectory=root_trajectory,
        terrain_root_trajectory_display=terrain_root_trajectory_display,
        local_pose=local_pose,
        local_pose_positions_offset=local_pose_positions_offset,
        reconstructed_pose_world=reconstructed_pose_world,
        pose_comparison_positions=pose_comparison_positions,
        pose_error_label=pose_error_label,
        pose_position_error_mean=pose_position_error_mean,
        pose_position_error_max=pose_position_error_max,
        bootstrap_frame_contacts=bootstrap_frame_contacts,
        bootstrap_bvh_contact_positions=motion.bootstrap_contact_data["positions"][animation_frame],
        bootstrap_pose_contact_positions=bootstrap_pose_contact_positions,
        frame_contacts=frame_contacts,
        bvh_contact_positions=motion.contact_data["positions"][animation_frame],
        pose_contact_positions=pose_contact_positions,
        terrain_height_at_focus=motion.terrain_provider.sample_height(terrain_query_position),
        terrain_normal_at_focus=motion.terrain_adapted_root_trajectory["terrain_normals"][animation_frame],
        body_proxy_positions=body_proxy_frame["positions"],
        body_proxy_radii=body_proxy_frame["radii"],
        penetration_frame=penetration_frame,
        penetration_count=penetration_frame["penetration_count"],
        max_penetration_depth=penetration_frame["max_penetration"],
    )


def _update_tracking(app, frame_state):
    render = app.render

    render.shadow_light.target = Vector3(frame_state.hip_position.x, 0.0, frame_state.hip_position.z)
    render.shadow_light.position = Vector3Add(
        render.shadow_light.target,
        Vector3Scale(render.light_dir, -5.0),
    )

    app.camera.update(
        Vector3(frame_state.hip_position.x, 0.75, frame_state.hip_position.z),
        GetMouseDelta().x if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(0) else 0.0,
        GetMouseDelta().y if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(0) else 0.0,
        GetMouseDelta().x if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(1) else 0.0,
        GetMouseDelta().y if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(1) else 0.0,
        GetMouseWheelMove(),
        GetFrameTime(),
    )


def _render_shadow_pass(app):
    render = app.render
    shaders = app.shaders
    scene = app.scene
    motion = app.motion
    debug = app.debug

    BeginShadowMap(render.shadow_map, render.shadow_light)

    light_view_proj = MatrixMultiply(rlGetMatrixModelview(), rlGetMatrixProjection())
    light_clip_near_ptr = _make_float_ptr(rlGetCullDistanceNear())
    light_clip_far_ptr = _make_float_ptr(rlGetCullDistanceFar())

    SetShaderValue(shaders.shadow.program, shaders.shadow.light_clip_near, light_clip_near_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.shadow.program, shaders.shadow.light_clip_far, light_clip_far_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(
        shaders.skinned_shadow.program,
        shaders.skinned_shadow.light_clip_near,
        light_clip_near_ptr,
        SHADER_UNIFORM_FLOAT,
    )
    SetShaderValue(
        shaders.skinned_shadow.program,
        shaders.skinned_shadow.light_clip_far,
        light_clip_far_ptr,
        SHADER_UNIFORM_FLOAT,
    )

    if debug.draw_flat_ground_ptr[0]:
        _draw_model_with_shader(scene.ground_model, shaders.shadow.program, scene.ground_position, WHITE)
    if debug.draw_terrain_mesh_ptr[0]:
        _draw_model_with_shader(motion.terrain_model, shaders.shadow.program, Vector3Zero(), WHITE)

    _draw_model_with_shader(scene.geno_model, shaders.skinned_shadow.program, scene.geno_position, WHITE)
    if debug.draw_reconstructed_pose_ptr[0]:
        _draw_model_with_shader(scene.pose_model, shaders.skinned_shadow.program, scene.geno_position, WHITE)

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
    motion = app.motion
    debug = app.debug

    BeginGBuffer(render.gbuffer, app.camera.cam3d)

    cam_view = rlGetMatrixModelview()
    cam_proj = rlGetMatrixProjection()
    cam_inv_proj = MatrixInvert(cam_proj)
    cam_inv_view_proj = MatrixInvert(MatrixMultiply(cam_view, cam_proj))
    cam_clip_near_ptr = _make_float_ptr(rlGetCullDistanceNear())
    cam_clip_far_ptr = _make_float_ptr(rlGetCullDistanceFar())
    specularity_ptr = _make_float_ptr(0.5)
    glossiness_ptr = _make_float_ptr(10.0)

    SetShaderValue(shaders.basic.program, shaders.basic.specularity, specularity_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.basic.program, shaders.basic.glossiness, glossiness_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.basic.program, shaders.basic.cam_clip_near, cam_clip_near_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.basic.program, shaders.basic.cam_clip_far, cam_clip_far_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.skinned_basic.program, shaders.skinned_basic.specularity, specularity_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.skinned_basic.program, shaders.skinned_basic.glossiness, glossiness_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(
        shaders.skinned_basic.program,
        shaders.skinned_basic.cam_clip_near,
        cam_clip_near_ptr,
        SHADER_UNIFORM_FLOAT,
    )
    SetShaderValue(
        shaders.skinned_basic.program,
        shaders.skinned_basic.cam_clip_far,
        cam_clip_far_ptr,
        SHADER_UNIFORM_FLOAT,
    )

    if debug.draw_flat_ground_ptr[0]:
        _draw_model_with_shader(scene.ground_model, shaders.basic.program, scene.ground_position, Color(190, 190, 190, 255))
    if debug.draw_terrain_mesh_ptr[0]:
        _draw_model_with_shader(motion.terrain_model, shaders.basic.program, Vector3Zero(), Color(190, 190, 190, 255))

    _draw_model_with_shader(scene.geno_model, shaders.skinned_basic.program, scene.geno_position, Color(220, 220, 220, 255))
    if debug.draw_reconstructed_pose_ptr[0]:
        _draw_model_with_shader(
            scene.pose_model,
            shaders.skinned_basic.program,
            scene.geno_position,
            debug.pose_model_color,
        )

    EndGBuffer(app.screen_width, app.screen_height)

    return SimpleNamespace(
        view=cam_view,
        proj=cam_proj,
        inv_proj=cam_inv_proj,
        inv_view_proj=cam_inv_view_proj,
        clip_near_ptr=cam_clip_near_ptr,
        clip_far_ptr=cam_clip_far_ptr,
    )


def _render_ssao_and_blur_pass(app, shadow_pass, camera_pass):
    render = app.render
    shaders = app.shaders

    BeginTextureMode(render.ssao_front)
    BeginShaderMode(shaders.ssao.program)
    SetShaderValueTexture(shaders.ssao.program, shaders.ssao.gbuffer_normal, render.gbuffer.normal)
    SetShaderValueTexture(shaders.ssao.program, shaders.ssao.gbuffer_depth, render.gbuffer.depth)
    SetShaderValueMatrix(shaders.ssao.program, shaders.ssao.cam_view, camera_pass.view)
    SetShaderValueMatrix(shaders.ssao.program, shaders.ssao.cam_proj, camera_pass.proj)
    SetShaderValueMatrix(shaders.ssao.program, shaders.ssao.cam_inv_proj, camera_pass.inv_proj)
    SetShaderValueMatrix(shaders.ssao.program, shaders.ssao.cam_inv_view_proj, camera_pass.inv_view_proj)
    SetShaderValueMatrix(shaders.ssao.program, shaders.ssao.light_view_proj, shadow_pass.view_proj)
    SetShaderValueShadowMap(shaders.ssao.program, shaders.ssao.shadow_map, render.shadow_map)
    SetShaderValue(
        shaders.ssao.program,
        shaders.ssao.shadow_inv_resolution,
        ffi.addressof(render.shadow_inv_resolution),
        SHADER_UNIFORM_VEC2,
    )
    SetShaderValue(shaders.ssao.program, shaders.ssao.cam_clip_near, camera_pass.clip_near_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.ssao.program, shaders.ssao.cam_clip_far, camera_pass.clip_far_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.ssao.program, shaders.ssao.light_clip_near, shadow_pass.clip_near_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.ssao.program, shaders.ssao.light_clip_far, shadow_pass.clip_far_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.ssao.program, shaders.ssao.light_dir, ffi.addressof(render.light_dir), SHADER_UNIFORM_VEC3)
    ClearBackground(WHITE)
    DrawTextureRec(
        render.ssao_front.texture,
        Rectangle(0, 0, render.ssao_front.texture.width, -render.ssao_front.texture.height),
        Vector2(0.0, 0.0),
        WHITE,
    )
    EndShaderMode()
    EndTextureMode()

    BeginTextureMode(render.ssao_back)
    BeginShaderMode(shaders.blur.program)
    blur_direction = Vector2(1.0, 0.0)
    blur_inv_texture_resolution = Vector2(
        1.0 / render.ssao_front.texture.width,
        1.0 / render.ssao_front.texture.height,
    )
    SetShaderValueTexture(shaders.blur.program, shaders.blur.gbuffer_normal, render.gbuffer.normal)
    SetShaderValueTexture(shaders.blur.program, shaders.blur.gbuffer_depth, render.gbuffer.depth)
    SetShaderValueTexture(shaders.blur.program, shaders.blur.input_texture, render.ssao_front.texture)
    SetShaderValueMatrix(shaders.blur.program, shaders.blur.cam_inv_proj, camera_pass.inv_proj)
    SetShaderValue(shaders.blur.program, shaders.blur.cam_clip_near, camera_pass.clip_near_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.blur.program, shaders.blur.cam_clip_far, camera_pass.clip_far_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(
        shaders.blur.program,
        shaders.blur.inv_texture_resolution,
        ffi.addressof(blur_inv_texture_resolution),
        SHADER_UNIFORM_VEC2,
    )
    SetShaderValue(shaders.blur.program, shaders.blur.blur_direction, ffi.addressof(blur_direction), SHADER_UNIFORM_VEC2)
    DrawTextureRec(
        render.ssao_back.texture,
        Rectangle(0, 0, render.ssao_back.texture.width, -render.ssao_back.texture.height),
        Vector2(0, 0),
        WHITE,
    )
    EndShaderMode()
    EndTextureMode()

    BeginTextureMode(render.ssao_front)
    BeginShaderMode(shaders.blur.program)
    blur_direction = Vector2(0.0, 1.0)
    SetShaderValueTexture(shaders.blur.program, shaders.blur.input_texture, render.ssao_back.texture)
    SetShaderValue(shaders.blur.program, shaders.blur.blur_direction, ffi.addressof(blur_direction), SHADER_UNIFORM_VEC2)
    DrawTextureRec(
        render.ssao_front.texture,
        Rectangle(0, 0, render.ssao_front.texture.width, -render.ssao_front.texture.height),
        Vector2(0, 0),
        WHITE,
    )
    EndShaderMode()
    EndTextureMode()


def _draw_debug_overlay(app, frame_state):
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
            if debug.draw_pose_model_local_ptr[0]:
                DrawSkeleton(
                    frame_state.local_pose_positions_offset,
                    frame_state.local_pose["local_rotations"],
                    motion.parents,
                    debug.pose_model_color,
                )
            else:
                DrawSkeleton(
                    frame_state.reconstructed_pose_world["world_positions"],
                    frame_state.reconstructed_pose_world["world_rotations"],
                    motion.parents,
                    debug.pose_model_color,
                )

    if debug.draw_root_trajectory_ptr[0]:
        DrawRootTrajectoryDebug(
            frame_state.terrain_root_trajectory_display["world_positions"],
            frame_state.terrain_root_trajectory_display["world_directions"],
            frame_state.terrain_root_trajectory_display["world_velocities"],
            frame_state.root_trajectory["sample_offsets"],
            drawDirection=debug.draw_trajectory_directions_ptr[0],
            drawVelocity=debug.draw_trajectory_velocity_ptr[0],
        )

        if debug.draw_reconstructed_pose_ptr[0] and debug.draw_pose_model_local_ptr[0]:
            DrawRootTrajectoryDebug(
                OffsetPositions(frame_state.root_trajectory["local_positions"], debug.local_debug_origin),
                frame_state.root_trajectory["local_directions"],
                frame_state.root_trajectory["local_velocities"],
                frame_state.root_trajectory["sample_offsets"],
                drawDirection=debug.draw_trajectory_directions_ptr[0],
                drawVelocity=debug.draw_trajectory_velocity_ptr[0],
            )

    if debug.draw_contacts_ptr[0]:
        DrawContactStates(frame_state.bvh_contact_positions, frame_state.frame_contacts)
        if debug.draw_reconstructed_pose_ptr[0]:
            DrawContactStates(frame_state.pose_contact_positions, frame_state.frame_contacts)

    if debug.draw_bootstrap_contacts_ptr[0]:
        DrawContactStates(
            frame_state.bootstrap_bvh_contact_positions,
            frame_state.bootstrap_frame_contacts,
            activeColor=Color(150, 110, 60, 255),
            inactiveColor=Color(210, 190, 160, 255),
            activeSize=0.04,
            inactiveSize=0.04,
        )
        if debug.draw_reconstructed_pose_ptr[0]:
            DrawContactStates(
                frame_state.bootstrap_pose_contact_positions,
                frame_state.bootstrap_frame_contacts,
                activeColor=Color(150, 110, 60, 255),
                inactiveColor=Color(210, 190, 160, 255),
                activeSize=0.04,
                inactiveSize=0.04,
            )

    if debug.draw_terrain_samples_ptr[0]:
        DrawTerrainSamples(motion.terrain_provider.sample_positions)

    if debug.draw_terrain_normals_ptr[0]:
        DrawTerrainNormals(
            motion.terrain_provider.sample_positions,
            motion.terrain_sample_normals,
        )

    if debug.draw_body_proxy_ptr[0]:
        DrawBodyProxyFrame(frame_state.body_proxy_positions, frame_state.body_proxy_radii)

    if debug.draw_terrain_penetration_ptr[0]:
        DrawTerrainPenetrationFrame(frame_state.body_proxy_positions, frame_state.penetration_frame)

    if debug.draw_reconstruction_error_ptr[0]:
        DrawPoseReconstructionError(
            frame_state.pose_comparison_positions,
            frame_state.reconstructed_pose_world["world_positions"],
            MAGENTA,
        )

    EndMode3D()


def _render_lighting_and_debug_pass(app, camera_pass, frame_state):
    render = app.render
    shaders = app.shaders

    BeginTextureMode(render.lighted)
    BeginShaderMode(shaders.lighting.program)

    sun_color = Vector3(253.0 / 255.0, 255.0 / 255.0, 232.0 / 255.0)
    sun_strength_ptr = _make_float_ptr(0.25)
    sky_color = Vector3(174.0 / 255.0, 183.0 / 255.0, 190.0 / 255.0)
    sky_strength_ptr = _make_float_ptr(0.15)
    ground_strength_ptr = _make_float_ptr(0.1)
    ambient_strength_ptr = _make_float_ptr(1.0)
    exposure_ptr = _make_float_ptr(0.9)

    SetShaderValueTexture(shaders.lighting.program, shaders.lighting.gbuffer_color, render.gbuffer.color)
    SetShaderValueTexture(shaders.lighting.program, shaders.lighting.gbuffer_normal, render.gbuffer.normal)
    SetShaderValueTexture(shaders.lighting.program, shaders.lighting.gbuffer_depth, render.gbuffer.depth)
    SetShaderValueTexture(shaders.lighting.program, shaders.lighting.ssao, render.ssao_front.texture)
    SetShaderValue(
        shaders.lighting.program,
        shaders.lighting.cam_pos,
        ffi.addressof(app.camera.cam3d.position),
        SHADER_UNIFORM_VEC3,
    )
    SetShaderValueMatrix(shaders.lighting.program, shaders.lighting.cam_inv_view_proj, camera_pass.inv_view_proj)
    SetShaderValue(shaders.lighting.program, shaders.lighting.light_dir, ffi.addressof(render.light_dir), SHADER_UNIFORM_VEC3)
    SetShaderValue(shaders.lighting.program, shaders.lighting.sun_color, ffi.addressof(sun_color), SHADER_UNIFORM_VEC3)
    SetShaderValue(shaders.lighting.program, shaders.lighting.sun_strength, sun_strength_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.lighting.program, shaders.lighting.sky_color, ffi.addressof(sky_color), SHADER_UNIFORM_VEC3)
    SetShaderValue(shaders.lighting.program, shaders.lighting.sky_strength, sky_strength_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.lighting.program, shaders.lighting.ground_strength, ground_strength_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.lighting.program, shaders.lighting.ambient_strength, ambient_strength_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.lighting.program, shaders.lighting.exposure, exposure_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.lighting.program, shaders.lighting.cam_clip_near, camera_pass.clip_near_ptr, SHADER_UNIFORM_FLOAT)
    SetShaderValue(shaders.lighting.program, shaders.lighting.cam_clip_far, camera_pass.clip_far_ptr, SHADER_UNIFORM_FLOAT)

    ClearBackground(RAYWHITE)
    DrawTextureRec(
        render.gbuffer.color,
        Rectangle(0, 0, render.gbuffer.color.width, -render.gbuffer.color.height),
        Vector2(0, 0),
        WHITE,
    )

    EndShaderMode()
    _draw_debug_overlay(app, frame_state)
    EndTextureMode()


def _render_final_pass(app):
    render = app.render
    fxaa_inv_texture_resolution = Vector2(
        1.0 / render.lighted.texture.width,
        1.0 / render.lighted.texture.height,
    )

    BeginShaderMode(app.shaders.fxaa.program)
    SetShaderValueTexture(app.shaders.fxaa.program, app.shaders.fxaa.input_texture, render.lighted.texture)
    SetShaderValue(
        app.shaders.fxaa.program,
        app.shaders.fxaa.inv_texture_resolution,
        ffi.addressof(fxaa_inv_texture_resolution),
        SHADER_UNIFORM_VEC2,
    )
    DrawTextureRec(
        render.lighted.texture,
        Rectangle(0, 0, render.lighted.texture.width, -render.lighted.texture.height),
        Vector2(0, 0),
        WHITE,
    )
    EndShaderMode()


def _draw_ui(app, frame_state):
    debug = app.debug
    motion = app.motion
    screen_width = app.screen_width
    screen_height = app.screen_height
    _handle_annotation_shortcuts(app)
    selection_start, selection_end = debug.playback.selection_range
    timeline_mode = str(getattr(debug, "selected_timeline_mode", "final"))
    if timeline_mode not in ("auto", "final", "soft"):
        timeline_mode = "final"

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
    GuiLabel(
        Rectangle(30, 300, 130, 20),
        b"Clip: %d / %d" % (app.clip_index + 1, len(app.clip_resources)),
    )
    if GuiButton(Rectangle(170, 298, 90, 22), b"Prev Clip"):
        _switch_clip_offset(app, -1)
        return
    if GuiButton(Rectangle(270, 298, 90, 22), b"Next Clip"):
        _switch_clip_offset(app, 1)
        return

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
            motion.label_result,
            selection_start,
            selection_end,
            debug.selected_action_label,
        )
        debug.annotation_status = "Applied selection"

    if GuiButton(Rectangle(116, 505, 76, 28), b"Clear"):
        ClearManualLabelRange(
            motion.label_result,
            selection_start,
            selection_end,
        )
        debug.annotation_status = "Cleared selection"

    if GuiButton(Rectangle(202, 505, 76, 28), b"Save"):
        _save_current_annotations(app)

    if GuiButton(Rectangle(288, 505, 76, 28), b"Load"):
        _load_current_annotations(app)

    GuiLabel(Rectangle(30, 539, 95, 20), b"Blend W: %d" % debug.transition_width)
    if GuiButton(Rectangle(130, 537, 46, 24), b"-"):
        debug.transition_width = max(0, int(debug.transition_width) - 2)
    if GuiButton(Rectangle(184, 537, 46, 24), b"+"):
        debug.transition_width = min(60, int(debug.transition_width) + 2)
    if GuiButton(Rectangle(238, 537, 60, 24), b"Set"):
        ApplyTransitionWidthRange(
            motion.label_result,
            selection_start,
            selection_end,
            debug.transition_width,
        )
        debug.annotation_status = "Set blend width"
    if GuiButton(Rectangle(306, 537, 58, 24), b"Unset"):
        ClearTransitionWidthRange(
            motion.label_result,
            selection_start,
            selection_end,
        )
        debug.annotation_status = "Cleared blend width"

    if GuiButton(Rectangle(30, 569, 100, 24), b"Export"):
        _export_current_labels(app)

    if GuiButton(Rectangle(140, 569, 224, 24), b"Reset Manual"):
        ResetManualLabels(motion.label_result)
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

    GuiGroupBox(Rectangle(screen_width - 260, rendering_panel_y, 240, rendering_panel_height), b"Rendering")
    GuiCheckBox(Rectangle(screen_width - 250, 20, 20, 20), b"Draw Transforms", debug.draw_bone_transforms_ptr)
    GuiCheckBox(Rectangle(screen_width - 250, 45, 20, 20), b"Draw Flat Ground", debug.draw_flat_ground_ptr)
    GuiCheckBox(Rectangle(screen_width - 250, 70, 20, 20), b"Draw Terrain Mesh", debug.draw_terrain_mesh_ptr)
    GuiCheckBox(Rectangle(screen_width - 250, 95, 20, 20), b"Draw Root Trajectory", debug.draw_root_trajectory_ptr)
    GuiLabel(Rectangle(screen_width - 250, 120, 220, 20), b"Terrain-Aware Root/Pose: On")
    GuiCheckBox(Rectangle(screen_width - 250, 145, 20, 20), b"Draw Directions", debug.draw_trajectory_directions_ptr)
    GuiCheckBox(Rectangle(screen_width - 250, 170, 20, 20), b"Draw Velocity", debug.draw_trajectory_velocity_ptr)
    GuiCheckBox(Rectangle(screen_width - 250, 195, 20, 20), b"Draw Contacts", debug.draw_contacts_ptr)
    GuiCheckBox(Rectangle(screen_width - 250, 220, 20, 20), b"Draw Bootstrap Contacts", debug.draw_bootstrap_contacts_ptr)
    GuiCheckBox(Rectangle(screen_width - 250, 245, 20, 20), b"Draw Terrain Samples", debug.draw_terrain_samples_ptr)
    GuiCheckBox(Rectangle(screen_width - 250, 270, 20, 20), b"Draw Terrain Normals", debug.draw_terrain_normals_ptr)
    GuiCheckBox(Rectangle(screen_width - 250, 295, 20, 20), b"Draw Body Proxy", debug.draw_body_proxy_ptr)
    GuiCheckBox(Rectangle(screen_width - 250, 320, 20, 20), b"Draw Penetration", debug.draw_terrain_penetration_ptr)
    GuiCheckBox(Rectangle(screen_width - 250, 345, 20, 20), b"Draw Blue Geno", debug.draw_reconstructed_pose_ptr)
    GuiCheckBox(Rectangle(screen_width - 250, 370, 20, 20), b"Blue Geno Local", debug.draw_pose_model_local_ptr)
    GuiCheckBox(Rectangle(screen_width - 250, 395, 20, 20), b"Draw Reconstruction Error", debug.draw_reconstruction_error_ptr)
    GuiCheckBox(Rectangle(screen_width - 250, 420, 20, 20), b"Integrate Root Motion", debug.integrate_root_motion_ptr)
    GuiLabel(Rectangle(screen_width - 250, 440, 220, 20), b"Trajectory Projection: Terrain")
    GuiLabel(
        Rectangle(screen_width - 250, 460, 220, 20),
        b"Terrain Grid: %d x %d" % (
            motion.terrain_height_grid["num_x"],
            motion.terrain_height_grid["num_z"],
        ),
    )
    GuiLabel(Rectangle(screen_width - 250, 480, 220, 20), b"Terrain H: %.4f" % frame_state.terrain_height_at_focus)
    GuiLabel(
        Rectangle(screen_width - 250, 500, 220, 20),
        b"Terrain Samples: %d" % len(motion.terrain_provider.sample_positions),
    )
    GuiLabel(
        Rectangle(screen_width - 250, 520, 220, 20),
        b"Terrain N: [% .2f % .2f % .2f]" % (
            frame_state.terrain_normal_at_focus[0],
            frame_state.terrain_normal_at_focus[1],
            frame_state.terrain_normal_at_focus[2],
        ),
    )
    GuiLabel(
        Rectangle(screen_width - 250, 540, 220, 20),
        b"Pen: %d max %.4f" % (
            frame_state.penetration_count,
            frame_state.max_penetration_depth,
        ),
    )
    GuiLabel(
        Rectangle(screen_width - 250, 560, 220, 20),
        b"%s: mean %.6f max %.6f" % (
            frame_state.pose_error_label,
            frame_state.pose_position_error_mean,
            frame_state.pose_position_error_max,
        ),
    )

    debug.playback.draw_ui(screen_width, screen_height)


def _unload_app_resources(app):
    if app is None:
        return

    UnloadRenderTexture(app.render.lighted)
    UnloadRenderTexture(app.render.ssao_back)
    UnloadRenderTexture(app.render.ssao_front)
    UnloadGBuffer(app.render.gbuffer)
    UnloadShadowMap(app.render.shadow_map)

    UnloadModel(app.motion.terrain_model)
    UnloadModel(app.scene.pose_model)
    UnloadModel(app.scene.geno_model)
    UnloadModel(app.scene.ground_model)

    UnloadShader(app.shaders.fxaa.program)
    UnloadShader(app.shaders.blur.program)
    UnloadShader(app.shaders.ssao.program)
    UnloadShader(app.shaders.lighting.program)
    UnloadShader(app.shaders.basic.program)
    UnloadShader(app.shaders.skinned_basic.program)
    UnloadShader(app.shaders.skinned_shadow.program)
    UnloadShader(app.shaders.shadow.program)


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
            animation_frame = app.debug.playback.update(GetFrameTime())
            frame_state = _build_frame_state(app, animation_frame)
            _update_tracking(app, frame_state)

            rlDisableColorBlend()
            BeginDrawing()

            shadow_pass = _render_shadow_pass(app)
            camera_pass = _render_gbuffer_pass(app)
            _render_ssao_and_blur_pass(app, shadow_pass, camera_pass)
            _render_lighting_and_debug_pass(app, camera_pass, frame_state)
            _render_final_pass(app)

            rlEnableColorBlend()
            _draw_ui(app, frame_state)
            EndDrawing()
    finally:
        _unload_app_resources(app)
        CloseWindow()


if __name__ == "__main__":
    main()
