from types import SimpleNamespace

import cffi
import numpy as np
from pyray import Color, Rectangle, Texture, RenderTexture, Vector2, Vector3
from raylib import *
from raylib.defines import *

from genoview.State import CameraPassState, RenderResources, ShadowPassState
from genoview.modules.FeatureModule import EnsureClipFeature
from genoview.modules.FrameModule import (
    EnsureBodyProxyFrameState,
    EnsureBootstrapContactFrameState,
    EnsureContactFrameState,
    EnsurePenetrationFrameState,
    EnsurePoseErrorFrameState,
    EnsurePoseFrameState,
    EnsureRootTrajectoryFrameState,
)
from genoview.modules.MotionDebugModule import DrawMotionDebugOverlay
from genoview.modules.TerrainModule import DrawTerrainDebugOverlay, DrawTerrainGBuffer, DrawTerrainShadow


ffi = cffi.FFI()


def _make_float_ptr(value):
    ptr = ffi.new("float*")
    ptr[0] = float(value)
    return ptr


def _draw_model_with_shader(model, shader, position, color):
    model.materials[0].shader = shader
    DrawModel(model, position, 1.0, color)


def _draw_texture_flipped(texture):
    DrawTextureRec(
        texture,
        Rectangle(0, 0, texture.width, -texture.height),
        Vector2(0, 0),
        WHITE,
    )


def _set_shader_float(shader, location, value_ptr):
    SetShaderValue(shader, location, value_ptr, SHADER_UNIFORM_FLOAT)


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


def LoadShaderResources(resource_path):
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


def UnloadShaderResources(shaders):
    for shader in (
        shaders.fxaa,
        shaders.blur,
        shaders.ssao,
        shaders.lighting,
        shaders.basic,
        shaders.skinned_basic,
        shaders.skinned_shadow,
        shaders.shadow,
    ):
        UnloadShader(shader.program)


def CreateRenderResources(screen_width, screen_height):
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

    return RenderResources(
        light_dir=light_dir,
        shadow_light=shadow_light,
        shadow_map=LoadShadowMap(shadow_width, shadow_height),
        shadow_inv_resolution=Vector2(1.0 / shadow_width, 1.0 / shadow_height),
        gbuffer=LoadGBuffer(screen_width, screen_height),
        lighted=LoadRenderTexture(screen_width, screen_height),
        ssao_front=LoadRenderTexture(screen_width, screen_height),
        ssao_back=LoadRenderTexture(screen_width, screen_height),
    )


def UnloadRenderResources(render):
    for target in (render.lighted, render.ssao_back, render.ssao_front):
        UnloadRenderTexture(target)
    UnloadGBuffer(render.gbuffer)
    UnloadShadowMap(render.shadow_map)


def _render_shadow_pass(app, frame_state):
    render = app.render
    shaders = app.shaders
    scene = app.scene
    debug = app.debug

    BeginShadowMap(render.shadow_map, render.shadow_light)

    light_view_proj = MatrixMultiply(rlGetMatrixModelview(), rlGetMatrixProjection())
    light_clip_near_ptr = _make_float_ptr(rlGetCullDistanceNear())
    light_clip_far_ptr = _make_float_ptr(rlGetCullDistanceFar())

    for shadow_shader in (shaders.shadow, shaders.skinned_shadow):
        _set_shader_float(shadow_shader.program, shadow_shader.light_clip_near, light_clip_near_ptr)
        _set_shader_float(shadow_shader.program, shadow_shader.light_clip_far, light_clip_far_ptr)

    if debug.draw_flat_ground_ptr[0]:
        _draw_model_with_shader(scene.ground_model, shaders.shadow.program, scene.ground_position, WHITE)
    if debug.draw_terrain_mesh_ptr[0]:
        DrawTerrainShadow(app, shaders.shadow.program)

    _draw_model_with_shader(scene.geno_model, shaders.skinned_shadow.program, scene.geno_position, WHITE)
    if debug.draw_reconstructed_pose_ptr[0]:
        EnsurePoseFrameState(app, frame_state)
        if frame_state.reconstructed_pose_world is not None:
            _draw_model_with_shader(EnsureClipFeature(app, "pose_model"), shaders.skinned_shadow.program, scene.geno_position, WHITE)

    EndShadowMap()

    return ShadowPassState(
        view_proj=light_view_proj,
        clip_near_ptr=light_clip_near_ptr,
        clip_far_ptr=light_clip_far_ptr,
    )


def _render_gbuffer_pass(app, frame_state):
    render = app.render
    shaders = app.shaders
    scene = app.scene
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

    for basic_shader in (shaders.basic, shaders.skinned_basic):
        _set_shader_float(basic_shader.program, basic_shader.specularity, specularity_ptr)
        _set_shader_float(basic_shader.program, basic_shader.glossiness, glossiness_ptr)
        _set_shader_float(basic_shader.program, basic_shader.cam_clip_near, cam_clip_near_ptr)
        _set_shader_float(basic_shader.program, basic_shader.cam_clip_far, cam_clip_far_ptr)

    if debug.draw_flat_ground_ptr[0]:
        _draw_model_with_shader(scene.ground_model, shaders.basic.program, scene.ground_position, Color(190, 190, 190, 255))
    if debug.draw_terrain_mesh_ptr[0]:
        DrawTerrainGBuffer(app, shaders.basic.program)

    _draw_model_with_shader(scene.geno_model, shaders.skinned_basic.program, scene.geno_position, Color(220, 220, 220, 255))
    if debug.draw_reconstructed_pose_ptr[0]:
        EnsurePoseFrameState(app, frame_state)
        if frame_state.reconstructed_pose_world is not None:
            _draw_model_with_shader(
                EnsureClipFeature(app, "pose_model"),
                shaders.skinned_basic.program,
                scene.geno_position,
                debug.pose_model_color,
            )

    EndGBuffer(app.screen_width, app.screen_height)

    return CameraPassState(
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
    _draw_texture_flipped(render.ssao_front.texture)
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
    _draw_texture_flipped(render.ssao_back.texture)
    EndShaderMode()
    EndTextureMode()

    BeginTextureMode(render.ssao_front)
    BeginShaderMode(shaders.blur.program)
    blur_direction = Vector2(0.0, 1.0)
    SetShaderValueTexture(shaders.blur.program, shaders.blur.input_texture, render.ssao_back.texture)
    SetShaderValue(shaders.blur.program, shaders.blur.blur_direction, ffi.addressof(blur_direction), SHADER_UNIFORM_VEC2)
    _draw_texture_flipped(render.ssao_front.texture)
    EndShaderMode()
    EndTextureMode()


def _draw_debug_overlay(app, frame_state):
    DrawMotionDebugOverlay(
        app,
        frame_state,
        DrawTerrainDebugOverlay,
        EnsurePoseFrameState,
        EnsurePoseErrorFrameState,
        EnsureRootTrajectoryFrameState,
        EnsureContactFrameState,
        EnsureBootstrapContactFrameState,
        EnsureBodyProxyFrameState,
        EnsurePenetrationFrameState,
    )


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
    _draw_texture_flipped(render.gbuffer.color)

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
    _draw_texture_flipped(render.lighted.texture)
    EndShaderMode()


def RenderFrame(app, frame_state):
    shadow_pass = _render_shadow_pass(app, frame_state)
    camera_pass = _render_gbuffer_pass(app, frame_state)
    _render_ssao_and_blur_pass(app, shadow_pass, camera_pass)
    _render_lighting_and_debug_pass(app, camera_pass, frame_state)
    _render_final_pass(app)
