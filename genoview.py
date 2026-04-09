from pyray import (
    Vector2, Vector3,
    Color, Rectangle,
    Texture, RenderTexture)
from raylib import *
from raylib.defines import *

from pathlib import Path
import quat
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
    BuildRootTrajectoryDisplay,
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

ffi = cffi.FFI()

BASE_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = BASE_DIR / "resources"


def resource_path(*parts, as_bytes=False):
    path = RESOURCES_DIR.joinpath(*parts)
    return str(path).encode("utf-8") if as_bytes else str(path)

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

if __name__ == "__main__":
    
    # Init Window
    
    screenWidth = 1280
    screenHeight = 720
    
    SetConfigFlags(FLAG_VSYNC_HINT)
    InitWindow(screenWidth, screenHeight, b"GenoViewPython")
    SetTargetFPS(60)

    # Shaders
    
    shadowShader = LoadShader(resource_path("shadow.vs", as_bytes=True), resource_path("shadow.fs", as_bytes=True))
    shadowShaderLightClipNear = GetShaderLocation(shadowShader, b"lightClipNear")
    shadowShaderLightClipFar = GetShaderLocation(shadowShader, b"lightClipFar")
    
    skinnedShadowShader = LoadShader(resource_path("skinnedShadow.vs", as_bytes=True), resource_path("shadow.fs", as_bytes=True))
    skinnedShadowShaderLightClipNear = GetShaderLocation(skinnedShadowShader, b"lightClipNear")
    skinnedShadowShaderLightClipFar = GetShaderLocation(skinnedShadowShader, b"lightClipFar")
    
    skinnedBasicShader = LoadShader(resource_path("skinnedBasic.vs", as_bytes=True), resource_path("basic.fs", as_bytes=True))
    skinnedBasicShaderSpecularity = GetShaderLocation(skinnedBasicShader, b"specularity")
    skinnedBasicShaderGlossiness = GetShaderLocation(skinnedBasicShader, b"glossiness")
    skinnedBasicShaderCamClipNear = GetShaderLocation(skinnedBasicShader, b"camClipNear")
    skinnedBasicShaderCamClipFar = GetShaderLocation(skinnedBasicShader, b"camClipFar")

    basicShader = LoadShader(resource_path("basic.vs", as_bytes=True), resource_path("basic.fs", as_bytes=True))
    basicShaderSpecularity = GetShaderLocation(basicShader, b"specularity")
    basicShaderGlossiness = GetShaderLocation(basicShader, b"glossiness")
    basicShaderCamClipNear = GetShaderLocation(basicShader, b"camClipNear")
    basicShaderCamClipFar = GetShaderLocation(basicShader, b"camClipFar")
    
    lightingShader = LoadShader(resource_path("post.vs", as_bytes=True), resource_path("lighting.fs", as_bytes=True))
    lightingShaderGBufferColor = GetShaderLocation(lightingShader, b"gbufferColor")
    lightingShaderGBufferNormal = GetShaderLocation(lightingShader, b"gbufferNormal")
    lightingShaderGBufferDepth = GetShaderLocation(lightingShader, b"gbufferDepth")
    lightingShaderSSAO = GetShaderLocation(lightingShader, b"ssao")
    lightingShaderCamPos = GetShaderLocation(lightingShader, b"camPos")
    lightingShaderCamInvViewProj = GetShaderLocation(lightingShader, b"camInvViewProj")
    lightingShaderLightDir = GetShaderLocation(lightingShader, b"lightDir")
    lightingShaderSunColor = GetShaderLocation(lightingShader, b"sunColor")
    lightingShaderSunStrength = GetShaderLocation(lightingShader, b"sunStrength")
    lightingShaderSkyColor = GetShaderLocation(lightingShader, b"skyColor")
    lightingShaderSkyStrength = GetShaderLocation(lightingShader, b"skyStrength")
    lightingShaderGroundStrength = GetShaderLocation(lightingShader, b"groundStrength")
    lightingShaderAmbientStrength = GetShaderLocation(lightingShader, b"ambientStrength")
    lightingShaderExposure = GetShaderLocation(lightingShader, b"exposure")
    lightingShaderCamClipNear = GetShaderLocation(lightingShader, b"camClipNear")
    lightingShaderCamClipFar = GetShaderLocation(lightingShader, b"camClipFar")
    
    ssaoShader = LoadShader(resource_path("post.vs", as_bytes=True), resource_path("ssao.fs", as_bytes=True))
    ssaoShaderGBufferNormal = GetShaderLocation(ssaoShader, b"gbufferNormal")
    ssaoShaderGBufferDepth = GetShaderLocation(ssaoShader, b"gbufferDepth")
    ssaoShaderCamView = GetShaderLocation(ssaoShader, b"camView")
    ssaoShaderCamProj = GetShaderLocation(ssaoShader, b"camProj")
    ssaoShaderCamInvProj = GetShaderLocation(ssaoShader, b"camInvProj")
    ssaoShaderCamInvViewProj = GetShaderLocation(ssaoShader, b"camInvViewProj")
    ssaoShaderLightViewProj = GetShaderLocation(ssaoShader, b"lightViewProj")
    ssaoShaderShadowMap = GetShaderLocation(ssaoShader, b"shadowMap")
    ssaoShaderShadowInvResolution = GetShaderLocation(ssaoShader, b"shadowInvResolution")
    ssaoShaderCamClipNear = GetShaderLocation(ssaoShader, b"camClipNear")
    ssaoShaderCamClipFar = GetShaderLocation(ssaoShader, b"camClipFar")
    ssaoShaderLightClipNear = GetShaderLocation(ssaoShader, b"lightClipNear")
    ssaoShaderLightClipFar = GetShaderLocation(ssaoShader, b"lightClipFar")
    ssaoShaderLightDir = GetShaderLocation(ssaoShader, b"lightDir")
    
    blurShader = LoadShader(resource_path("post.vs", as_bytes=True), resource_path("blur.fs", as_bytes=True))
    blurShaderGBufferNormal = GetShaderLocation(blurShader, b"gbufferNormal")
    blurShaderGBufferDepth = GetShaderLocation(blurShader, b"gbufferDepth")
    blurShaderInputTexture = GetShaderLocation(blurShader, b"inputTexture")
    blurShaderCamInvProj = GetShaderLocation(blurShader, b"camInvProj")
    blurShaderCamClipNear = GetShaderLocation(blurShader, b"camClipNear")
    blurShaderCamClipFar = GetShaderLocation(blurShader, b"camClipFar")
    blurShaderInvTextureResolution = GetShaderLocation(blurShader, b"invTextureResolution")
    blurShaderBlurDirection = GetShaderLocation(blurShader, b"blurDirection")

    fxaaShader = LoadShader(resource_path("post.vs", as_bytes=True), resource_path("fxaa.fs", as_bytes=True))
    fxaaShaderInputTexture = GetShaderLocation(fxaaShader, b"inputTexture")
    fxaaShaderInvTextureResolution = GetShaderLocation(fxaaShader, b"invTextureResolution")
    
    # Objects
    
    groundMesh = GenMeshPlane(20.0, 20.0, 10, 10)
    groundModel = LoadModelFromMesh(groundMesh)
    groundPosition = Vector3(0.0, -0.01, 0.0)
    
    genoModel = LoadCharacterModel(resource_path("Geno.bin", as_bytes=True))
    poseModel = LoadCharacterModel(resource_path("Geno.bin", as_bytes=True))
    genoPosition = Vector3(0.0, 0.0, 0.0)
    
    bindPos, bindRot = GetModelBindPoseAsNumpyArrays(genoModel)
    
    # Animation
    
    # bvhAnimation = BVHImporter.load(resource_path("ground1_subject1.bvh"), scale=0.01)
    # bvhAnimation = BVHImporter.load(resource_path("Geno_bind.bvh"), scale=0.01)
    bvhAnimation = BVHImporter.load(resource_path("bvh/lafan1/walk1_subject5.bvh"), scale=0.01)

    parents = bvhAnimation.parents
    globalRotations = bvhAnimation.global_rotations
    globalPositions = bvhAnimation.global_positions
    trajectorySampleOffsets = GetRootTrajectorySampleOffsets()
    bvhFrameTime = DEFAULT_BVH_FRAME_TIME
    basePoseSource = BuildPoseSource(
        globalPositions,
        globalRotations,
        bvhFrameTime,
    )
    bootstrapContactData = BuildContactData(
        globalPositions,
        basePoseSource["global_velocities"],
        bvhAnimation.raw_data["names"],
        bootstrap=True,
    )
    terrainProvider = BuildTerrainProviderFromContactData(
        bootstrapContactData,
        filtered=True,
        fallbackHeight=groundPosition.y,
    )
    contactData = BuildContactData(
        globalPositions,
        basePoseSource["global_velocities"],
        bvhAnimation.raw_data["names"],
        terrainProvider=terrainProvider,
    )
    bodyProxyLayout = BuildBodyProxyLayout(
        globalPositions[0],
        parents,
        bvhAnimation.raw_data["names"],
    )
    terrainModel, terrainHeightGrid = LoadTerrainModelFromProvider(
        terrainProvider,
        terrainProvider.sample_positions,
        cellSize=0.1,
        padding=0.5,
    )
    motionRootTrajectory = BuildRootTrajectorySource(
        globalPositions,
        globalRotations,
        bvhFrameTime,
        rootIndex=ROOT_JOINT_INDEX,
        mode="height_3d",
    )
    terrainAdaptedRootTrajectory = AdaptRootTrajectoryToTerrain(
        motionRootTrajectory,
        terrainProvider,
        alignPositionsToTerrain=False,
    )
    poseSource = BuildPoseSource(
        globalPositions,
        globalRotations,
        bvhFrameTime,
        rootTrajectorySource=motionRootTrajectory,
    )
    terrainSampleNormals = terrainProvider.sample_normals(terrainProvider.sample_positions)
    
    animationFrame = 0
    
    # Camera
    
    camera = Camera()
    
    rlSetClipPlanes(0.01, 50.0)
    
    # Shadows
    
    lightDir = Vector3Normalize(Vector3(0.35, -1.0, -0.35))
    
    shadowLight = ShadowLight()
    shadowLight.target = Vector3Zero()
    shadowLight.position = Vector3Scale(lightDir, -5.0)
    shadowLight.up = Vector3(0.0, 1.0, 0.0)
    shadowLight.width = 5.0
    shadowLight.height = 5.0
    shadowLight.near = 0.01
    shadowLight.far = 10.0
    
    shadowWidth = 1024
    shadowHeight = 1024
    shadowInvResolution = Vector2(1.0 / shadowWidth, 1.0 / shadowHeight)
    shadowMap = LoadShadowMap(shadowWidth, shadowHeight)    
    
    # GBuffer and Render Textures
    
    gbuffer = LoadGBuffer(screenWidth, screenHeight)
    lighted = LoadRenderTexture(screenWidth, screenHeight)
    ssaoFront = LoadRenderTexture(screenWidth, screenHeight)
    ssaoBack = LoadRenderTexture(screenWidth, screenHeight)
    
    # UI
    
    drawBoneTransformsPtr = ffi.new('bool*'); drawBoneTransformsPtr[0] = False
    drawFlatGroundPtr = ffi.new('bool*'); drawFlatGroundPtr[0] = False
    drawTerrainMeshPtr = ffi.new('bool*'); drawTerrainMeshPtr[0] = True
    drawRootTrajectoryPtr = ffi.new('bool*'); drawRootTrajectoryPtr[0] = True
    drawTrajectoryDirectionsPtr = ffi.new('bool*'); drawTrajectoryDirectionsPtr[0] = True
    drawTrajectoryVelocityPtr = ffi.new('bool*'); drawTrajectoryVelocityPtr[0] = True
    drawContactsPtr = ffi.new('bool*'); drawContactsPtr[0] = True
    drawBootstrapContactsPtr = ffi.new('bool*'); drawBootstrapContactsPtr[0] = False
    drawTerrainSamplesPtr = ffi.new('bool*'); drawTerrainSamplesPtr[0] = False
    drawTerrainNormalsPtr = ffi.new('bool*'); drawTerrainNormalsPtr[0] = False
    drawBodyProxyPtr = ffi.new('bool*'); drawBodyProxyPtr[0] = False
    drawTerrainPenetrationPtr = ffi.new('bool*'); drawTerrainPenetrationPtr[0] = False
    drawReconstructedPosePtr = ffi.new('bool*'); drawReconstructedPosePtr[0] = True
    drawPoseModelLocalPtr = ffi.new('bool*'); drawPoseModelLocalPtr[0] = False
    drawReconstructionErrorPtr = ffi.new('bool*'); drawReconstructionErrorPtr[0] = True
    integrateRootMotionPtr = ffi.new('bool*'); integrateRootMotionPtr[0] = False
    localDebugOrigin = Vector3(-2.0, 0.0, 0.0)
    poseModelColor = Color(110, 190, 255, 255)
    playback = PlaybackController(bvhAnimation.frame_count, bvhFrameTime)
    
    # Go
    
    while not WindowShouldClose():
    
        # Animation
        
        animationFrame = playback.update(GetFrameTime())
        UpdateModelPoseFromNumpyArrays(
            genoModel, bindPos, bindRot, 
            globalPositions[animationFrame], globalRotations[animationFrame])
        rootTrajectory = BuildRootLocalTrajectory(
            motionRootTrajectory,
            animationFrame,
            sampleOffsets=trajectorySampleOffsets,
        )
        rootTrajectoryDisplay = BuildRootTrajectoryDisplay(
            rootTrajectory,
            groundHeight=groundPosition.y,
            projectToGround=False,
            heightOffset=0.02,
            terrainProvider=terrainProvider,
            projectToTerrain=True,
        )
        terrainRootTrajectoryDisplay = BuildTerrainAdaptedRootTrajectoryDisplay(
            rootTrajectory,
            terrainAdaptedRootTrajectory,
            heightOffset=0.02,
            alignDirectionsToTerrain=True,
            alignVelocitiesToTerrain=True,
        )
        localPose = BuildLocalPose(
            poseSource,
            motionRootTrajectory,
            animationFrame,
            dt=bvhFrameTime,
        )
        reconstructedPoseWorld = ReconstructPoseWorldSpace(
            localPose,
            integrateRootMotion=integrateRootMotionPtr[0],
            dt=bvhFrameTime,
        )
        poseComparisonFrame = (
            min(animationFrame + 1, bvhAnimation.frame_count - 1)
            if integrateRootMotionPtr[0] else
            animationFrame
        )
        poseComparisonPositions = globalPositions[poseComparisonFrame]
        poseErrorLabel = b"Pred Err(+1)" if integrateRootMotionPtr[0] else b"Recon Err"
        if drawPoseModelLocalPtr[0]:
            UpdateModelPoseFromNumpyArrays(
                poseModel, bindPos, bindRot,
                OffsetPositions(localPose["local_positions"], localDebugOrigin),
                localPose["local_rotations"])
        else:
            UpdateModelPoseFromNumpyArrays(
                poseModel, bindPos, bindRot,
                reconstructedPoseWorld["world_positions"], reconstructedPoseWorld["world_rotations"])
        posePositionErrorMean, posePositionErrorMax = ComputePosePositionError(
            poseComparisonPositions,
            reconstructedPoseWorld["world_positions"],
        )
        bootstrapFrameContacts = bootstrapContactData["contacts_filtered"][animationFrame]
        bootstrapContactIndices = bootstrapContactData["joint_indices"]
        bootstrapBvhContactPositions = bootstrapContactData["positions"][animationFrame]
        bootstrapPoseContactPositions = (
            OffsetPositions(localPose["local_positions"][bootstrapContactIndices], localDebugOrigin)
            if drawPoseModelLocalPtr[0] else
            reconstructedPoseWorld["world_positions"][bootstrapContactIndices]
        )
        frameContacts = contactData["contacts_filtered"][animationFrame]
        contactIndices = contactData["joint_indices"]
        bvhContactPositions = contactData["positions"][animationFrame]
        poseContactPositions = (
            OffsetPositions(localPose["local_positions"][contactIndices], localDebugOrigin)
            if drawPoseModelLocalPtr[0] else
            reconstructedPoseWorld["world_positions"][contactIndices]
        )
        poseFocusPosition = (
            OffsetPositions(localPose["local_positions"], localDebugOrigin)[ROOT_JOINT_INDEX]
            if drawPoseModelLocalPtr[0] else
            reconstructedPoseWorld["world_positions"][ROOT_JOINT_INDEX]
        )
        terrainQueryPosition = reconstructedPoseWorld["world_positions"][ROOT_JOINT_INDEX]
        terrainHeightAtFocus = terrainProvider.sample_height(terrainQueryPosition)
        terrainNormalAtFocus = terrainAdaptedRootTrajectory["terrain_normals"][animationFrame]
        bodyProxyFrame = BuildBodyProxyFrame(
            globalPositions[animationFrame],
            bodyProxyLayout,
        )
        bodyProxyPositions = bodyProxyFrame["positions"]
        bodyProxyRadii = bodyProxyFrame["radii"]
        penetrationFrame = ComputeTerrainPenetrationFrame(
            bodyProxyFrame,
            terrainProvider,
        )
        penetrationCount = penetrationFrame["penetration_count"]
        maxPenetrationDepth = penetrationFrame["max_penetration"]

        # Shadow Light Tracks Character
        
        hipPosition = Vector3(*poseFocusPosition)
        
        shadowLight.target = Vector3(hipPosition.x, 0.0, hipPosition.z)
        shadowLight.position = Vector3Add(shadowLight.target, Vector3Scale(lightDir, -5.0))

        # Update Camera
        
        camera.update(
            Vector3(hipPosition.x, 0.75, hipPosition.z),
            GetMouseDelta().x if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(0) else 0.0,
            GetMouseDelta().y if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(0) else 0.0,
            GetMouseDelta().x if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(1) else 0.0,
            GetMouseDelta().y if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(1) else 0.0,
            GetMouseWheelMove(),
            GetFrameTime())
        
        # Render
        
        rlDisableColorBlend()
        
        BeginDrawing()
        
        # Render Shadow Maps
        
        BeginShadowMap(shadowMap, shadowLight)  
        
        lightViewProj = MatrixMultiply(rlGetMatrixModelview(), rlGetMatrixProjection())
        lightClipNear = rlGetCullDistanceNear()
        lightClipFar = rlGetCullDistanceFar()

        lightClipNearPtr = ffi.new("float*"); lightClipNearPtr[0] = lightClipNear
        lightClipFarPtr = ffi.new("float*"); lightClipFarPtr[0] = lightClipFar
        
        SetShaderValue(shadowShader, shadowShaderLightClipNear, lightClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(shadowShader, shadowShaderLightClipFar, lightClipFarPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(skinnedShadowShader, skinnedShadowShaderLightClipNear, lightClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(skinnedShadowShader, skinnedShadowShaderLightClipFar, lightClipFarPtr, SHADER_UNIFORM_FLOAT)
        
        if drawFlatGroundPtr[0]:
            groundModel.materials[0].shader = shadowShader
            DrawModel(groundModel, groundPosition, 1.0, WHITE)

        if drawTerrainMeshPtr[0]:
            terrainModel.materials[0].shader = shadowShader
            DrawModel(terrainModel, Vector3Zero(), 1.0, WHITE)
        
        genoModel.materials[0].shader = skinnedShadowShader
        DrawModel(genoModel, genoPosition, 1.0, WHITE)

        if drawReconstructedPosePtr[0]:
            poseModel.materials[0].shader = skinnedShadowShader
            DrawModel(poseModel, genoPosition, 1.0, WHITE)
        
        EndShadowMap()
        
        # Render GBuffer
        
        BeginGBuffer(gbuffer, camera.cam3d)
        
        camView = rlGetMatrixModelview()
        camProj = rlGetMatrixProjection()
        camInvProj = MatrixInvert(camProj)
        camInvViewProj = MatrixInvert(MatrixMultiply(camView, camProj))
        camClipNear = rlGetCullDistanceNear()
        camClipFar = rlGetCullDistanceFar()

        camClipNearPtr = ffi.new("float*"); camClipNearPtr[0] = camClipNear
        camClipFarPtr = ffi.new("float*"); camClipFarPtr[0] = camClipFar

        specularityPtr = ffi.new('float*'); specularityPtr[0] = 0.5
        glossinessPtr = ffi.new('float*'); glossinessPtr[0] = 10.0
        
        SetShaderValue(basicShader, basicShaderSpecularity, specularityPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(basicShader, basicShaderGlossiness, glossinessPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(basicShader, basicShaderCamClipNear, camClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(basicShader, basicShaderCamClipFar, camClipFarPtr, SHADER_UNIFORM_FLOAT)
        
        SetShaderValue(skinnedBasicShader, skinnedBasicShaderSpecularity, specularityPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(skinnedBasicShader, skinnedBasicShaderGlossiness, glossinessPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(skinnedBasicShader, skinnedBasicShaderCamClipNear, camClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(skinnedBasicShader, skinnedBasicShaderCamClipFar, camClipFarPtr, SHADER_UNIFORM_FLOAT)        
        
        if drawFlatGroundPtr[0]:
            groundModel.materials[0].shader = basicShader
            DrawModel(groundModel, groundPosition, 1.0, Color(190, 190, 190, 255))

        if drawTerrainMeshPtr[0]:
            terrainModel.materials[0].shader = basicShader
            DrawModel(terrainModel, Vector3Zero(), 1.0, Color(190, 190, 190, 255))
        
        genoModel.materials[0].shader = skinnedBasicShader
        DrawModel(genoModel, genoPosition, 1.0, Color(220, 220, 220, 255))

        if drawReconstructedPosePtr[0]:
            poseModel.materials[0].shader = skinnedBasicShader
            DrawModel(poseModel, genoPosition, 1.0, poseModelColor)
        
        EndGBuffer(screenWidth, screenHeight)
        
        # Render SSAO and Shadows
        
        BeginTextureMode(ssaoFront)
        
        BeginShaderMode(ssaoShader)
        
        SetShaderValueTexture(ssaoShader, ssaoShaderGBufferNormal, gbuffer.normal)
        SetShaderValueTexture(ssaoShader, ssaoShaderGBufferDepth, gbuffer.depth)
        SetShaderValueMatrix(ssaoShader, ssaoShaderCamView, camView)
        SetShaderValueMatrix(ssaoShader, ssaoShaderCamProj, camProj)
        SetShaderValueMatrix(ssaoShader, ssaoShaderCamInvProj, camInvProj)
        SetShaderValueMatrix(ssaoShader, ssaoShaderCamInvViewProj, camInvViewProj)
        SetShaderValueMatrix(ssaoShader, ssaoShaderLightViewProj, lightViewProj)
        SetShaderValueShadowMap(ssaoShader, ssaoShaderShadowMap, shadowMap)
        SetShaderValue(ssaoShader, ssaoShaderShadowInvResolution, ffi.addressof(shadowInvResolution), SHADER_UNIFORM_VEC2)
        SetShaderValue(ssaoShader, ssaoShaderCamClipNear, camClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(ssaoShader, ssaoShaderCamClipFar, camClipFarPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(ssaoShader, ssaoShaderLightClipNear, lightClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(ssaoShader, ssaoShaderLightClipFar, lightClipFarPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(ssaoShader, ssaoShaderLightDir, ffi.addressof(lightDir), SHADER_UNIFORM_VEC3)
        
        ClearBackground(WHITE)
        
        DrawTextureRec(
            ssaoFront.texture,
            Rectangle(0, 0, ssaoFront.texture.width, -ssaoFront.texture.height),
            Vector2(0.0, 0.0),
            WHITE)

        EndShaderMode()

        EndTextureMode()
        
        # Blur Horizontal
        
        BeginTextureMode(ssaoBack)
        
        BeginShaderMode(blurShader)
        
        blurDirection = Vector2(1.0, 0.0)
        blurInvTextureResolution = Vector2(1.0 / ssaoFront.texture.width, 1.0 / ssaoFront.texture.height)
        
        SetShaderValueTexture(blurShader, blurShaderGBufferNormal, gbuffer.normal)
        SetShaderValueTexture(blurShader, blurShaderGBufferDepth, gbuffer.depth)
        SetShaderValueTexture(blurShader, blurShaderInputTexture, ssaoFront.texture)
        SetShaderValueMatrix(blurShader, blurShaderCamInvProj, camInvProj)
        SetShaderValue(blurShader, blurShaderCamClipNear, camClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(blurShader, blurShaderCamClipFar, camClipFarPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(blurShader, blurShaderInvTextureResolution, ffi.addressof(blurInvTextureResolution), SHADER_UNIFORM_VEC2)
        SetShaderValue(blurShader, blurShaderBlurDirection, ffi.addressof(blurDirection), SHADER_UNIFORM_VEC2)

        DrawTextureRec(
            ssaoBack.texture,
            Rectangle(0, 0, ssaoBack.texture.width, -ssaoBack.texture.height),
            Vector2(0, 0),
            WHITE)

        EndShaderMode()

        EndTextureMode()
      
        # Blur Vertical
        
        BeginTextureMode(ssaoFront)
        
        BeginShaderMode(blurShader)
        
        blurDirection = Vector2(0.0, 1.0)
        
        SetShaderValueTexture(blurShader, blurShaderInputTexture, ssaoBack.texture)
        SetShaderValue(blurShader, blurShaderBlurDirection, ffi.addressof(blurDirection), SHADER_UNIFORM_VEC2)

        DrawTextureRec(
            ssaoFront.texture,
            Rectangle(0, 0, ssaoFront.texture.width, -ssaoFront.texture.height),
            Vector2(0, 0),
            WHITE)

        EndShaderMode()

        EndTextureMode()
      
        # Light GBuffer
        
        BeginTextureMode(lighted)
        
        BeginShaderMode(lightingShader)
        
        sunColor = Vector3(253.0 / 255.0, 255.0 / 255.0, 232.0 / 255.0)
        sunStrengthPtr = ffi.new('float*'); sunStrengthPtr[0] = 0.25
        skyColor = Vector3(174.0 / 255.0, 183.0 / 255.0, 190.0 / 255.0)
        skyStrengthPtr = ffi.new('float*'); skyStrengthPtr[0] = 0.15
        groundStrengthPtr = ffi.new('float*'); groundStrengthPtr[0] = 0.1
        ambientStrengthPtr = ffi.new('float*'); ambientStrengthPtr[0] = 1.0
        exposurePtr = ffi.new('float*'); exposurePtr[0] = 0.9
        
        SetShaderValueTexture(lightingShader, lightingShaderGBufferColor, gbuffer.color)
        SetShaderValueTexture(lightingShader, lightingShaderGBufferNormal, gbuffer.normal)
        SetShaderValueTexture(lightingShader, lightingShaderGBufferDepth, gbuffer.depth)
        SetShaderValueTexture(lightingShader, lightingShaderSSAO, ssaoFront.texture)
        SetShaderValue(lightingShader, lightingShaderCamPos, ffi.addressof(camera.cam3d.position), SHADER_UNIFORM_VEC3)
        SetShaderValueMatrix(lightingShader, lightingShaderCamInvViewProj, camInvViewProj)
        SetShaderValue(lightingShader, lightingShaderLightDir, ffi.addressof(lightDir), SHADER_UNIFORM_VEC3)
        SetShaderValue(lightingShader, lightingShaderSunColor, ffi.addressof(sunColor), SHADER_UNIFORM_VEC3)
        SetShaderValue(lightingShader, lightingShaderSunStrength, sunStrengthPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderSkyColor, ffi.addressof(skyColor), SHADER_UNIFORM_VEC3)
        SetShaderValue(lightingShader, lightingShaderSkyStrength, skyStrengthPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderGroundStrength, groundStrengthPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderAmbientStrength, ambientStrengthPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderExposure, exposurePtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderCamClipNear, camClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderCamClipFar, camClipFarPtr, SHADER_UNIFORM_FLOAT)
        
        ClearBackground(RAYWHITE)
        
        DrawTextureRec(
            gbuffer.color,
            Rectangle(0, 0, gbuffer.color.width, -gbuffer.color.height),
            Vector2(0, 0),
            WHITE)
        
        EndShaderMode()        
        
        # Debug Draw
        
        BeginMode3D(camera.cam3d)

        if drawBoneTransformsPtr[0]:
            DrawSkeleton(
                globalPositions[animationFrame], 
                globalRotations[animationFrame], 
                parents, GRAY)

            if drawReconstructedPosePtr[0]:
                if drawPoseModelLocalPtr[0]:
                    DrawSkeleton(
                        OffsetPositions(localPose["local_positions"], localDebugOrigin),
                        localPose["local_rotations"],
                        parents,
                        poseModelColor)
                else:
                    DrawSkeleton(
                        reconstructedPoseWorld["world_positions"],
                        reconstructedPoseWorld["world_rotations"],
                        parents,
                        poseModelColor)

        if drawRootTrajectoryPtr[0]:
            DrawRootTrajectoryDebug(
                terrainRootTrajectoryDisplay["world_positions"],
                terrainRootTrajectoryDisplay["world_directions"],
                terrainRootTrajectoryDisplay["world_velocities"],
                rootTrajectory["sample_offsets"],
                drawDirection=drawTrajectoryDirectionsPtr[0],
                drawVelocity=drawTrajectoryVelocityPtr[0],
            )

            if drawReconstructedPosePtr[0] and drawPoseModelLocalPtr[0]:
                DrawRootTrajectoryDebug(
                    OffsetPositions(rootTrajectory["local_positions"], localDebugOrigin),
                    rootTrajectory["local_directions"],
                    rootTrajectory["local_velocities"],
                    rootTrajectory["sample_offsets"],
                    drawDirection=drawTrajectoryDirectionsPtr[0],
                    drawVelocity=drawTrajectoryVelocityPtr[0],
                )

        if drawContactsPtr[0]:
            DrawContactStates(
                bvhContactPositions,
                frameContacts,
            )

            if drawReconstructedPosePtr[0]:
                DrawContactStates(
                    poseContactPositions,
                    frameContacts,
                )

        if drawBootstrapContactsPtr[0]:
            DrawContactStates(
                bootstrapBvhContactPositions,
                bootstrapFrameContacts,
                activeColor=Color(150, 110, 60, 255),
                inactiveColor=Color(210, 190, 160, 255),
                activeSize=0.04,
                inactiveSize=0.04,
            )

            if drawReconstructedPosePtr[0]:
                DrawContactStates(
                    bootstrapPoseContactPositions,
                    bootstrapFrameContacts,
                    activeColor=Color(150, 110, 60, 255),
                    inactiveColor=Color(210, 190, 160, 255),
                    activeSize=0.04,
                    inactiveSize=0.04,
                )

        if drawTerrainSamplesPtr[0]:
            DrawTerrainSamples(terrainProvider.sample_positions)

        if drawTerrainNormalsPtr[0]:
            DrawTerrainNormals(
                terrainProvider.sample_positions,
                terrainSampleNormals,
            )

        if drawBodyProxyPtr[0]:
            DrawBodyProxyFrame(
                bodyProxyPositions,
                bodyProxyRadii,
            )

        if drawTerrainPenetrationPtr[0]:
            DrawTerrainPenetrationFrame(
                bodyProxyPositions,
                penetrationFrame,
            )

        if drawReconstructionErrorPtr[0]:
            DrawPoseReconstructionError(
                poseComparisonPositions,
                reconstructedPoseWorld["world_positions"],
                MAGENTA)
  
        EndMode3D()

        EndTextureMode()
        
        # Render Final with FXAA
        
        BeginShaderMode(fxaaShader)

        fxaaInvTextureResolution = Vector2(1.0 / lighted.texture.width, 1.0 / lighted.texture.height)
        
        SetShaderValueTexture(fxaaShader, fxaaShaderInputTexture, lighted.texture)
        SetShaderValue(fxaaShader, fxaaShaderInvTextureResolution, ffi.addressof(fxaaInvTextureResolution), SHADER_UNIFORM_VEC2)
        
        DrawTextureRec(
            lighted.texture,
            Rectangle(0, 0, lighted.texture.width, -lighted.texture.height),
            Vector2(0, 0),
            WHITE)
        
        EndShaderMode()
  
        # UI
  
        rlEnableColorBlend()
  
        GuiGroupBox(Rectangle(20, 10, 190, 180), b"Camera")

        GuiLabel(Rectangle(30, 20, 150, 20), b"Ctrl + Left Click - Rotate")
        GuiLabel(Rectangle(30, 40, 150, 20), b"Ctrl + Right Click - Pan")
        GuiLabel(Rectangle(30, 60, 150, 20), b"Mouse Scroll - Zoom")
        GuiLabel(Rectangle(30, 80, 150, 20), b"Target: [% 5.3f % 5.3f % 5.3f]" % (camera.cam3d.target.x, camera.cam3d.target.y, camera.cam3d.target.z))
        GuiLabel(Rectangle(30, 100, 150, 20), b"Offset: [% 5.3f % 5.3f % 5.3f]" % (camera.offset.x, camera.offset.y, camera.offset.z))
        GuiLabel(Rectangle(30, 120, 150, 20), b"Azimuth: %5.3f" % camera.azimuth)
        GuiLabel(Rectangle(30, 140, 150, 20), b"Altitude: %5.3f" % camera.altitude)
        GuiLabel(Rectangle(30, 160, 150, 20), b"Distance: %5.3f" % camera.distance)
  
        GuiGroupBox(Rectangle(screenWidth - 260, 10, 240, 580), b"Rendering")

        GuiCheckBox(Rectangle(screenWidth - 250, 20, 20, 20), b"Draw Transforms", drawBoneTransformsPtr)
        GuiCheckBox(Rectangle(screenWidth - 250, 45, 20, 20), b"Draw Flat Ground", drawFlatGroundPtr)
        GuiCheckBox(Rectangle(screenWidth - 250, 70, 20, 20), b"Draw Terrain Mesh", drawTerrainMeshPtr)
        GuiCheckBox(Rectangle(screenWidth - 250, 95, 20, 20), b"Draw Root Trajectory", drawRootTrajectoryPtr)
        GuiLabel(Rectangle(screenWidth - 250, 120, 220, 20), b"Terrain-Aware Root/Pose: On")
        GuiCheckBox(Rectangle(screenWidth - 250, 145, 20, 20), b"Draw Directions", drawTrajectoryDirectionsPtr)
        GuiCheckBox(Rectangle(screenWidth - 250, 170, 20, 20), b"Draw Velocity", drawTrajectoryVelocityPtr)
        GuiCheckBox(Rectangle(screenWidth - 250, 195, 20, 20), b"Draw Contacts", drawContactsPtr)
        GuiCheckBox(Rectangle(screenWidth - 250, 220, 20, 20), b"Draw Bootstrap Contacts", drawBootstrapContactsPtr)
        GuiCheckBox(Rectangle(screenWidth - 250, 245, 20, 20), b"Draw Terrain Samples", drawTerrainSamplesPtr)
        GuiCheckBox(Rectangle(screenWidth - 250, 270, 20, 20), b"Draw Terrain Normals", drawTerrainNormalsPtr)
        GuiCheckBox(Rectangle(screenWidth - 250, 295, 20, 20), b"Draw Body Proxy", drawBodyProxyPtr)
        GuiCheckBox(Rectangle(screenWidth - 250, 320, 20, 20), b"Draw Penetration", drawTerrainPenetrationPtr)
        GuiCheckBox(Rectangle(screenWidth - 250, 345, 20, 20), b"Draw Blue Geno", drawReconstructedPosePtr)
        GuiCheckBox(Rectangle(screenWidth - 250, 370, 20, 20), b"Blue Geno Local", drawPoseModelLocalPtr)
        GuiCheckBox(Rectangle(screenWidth - 250, 395, 20, 20), b"Draw Reconstruction Error", drawReconstructionErrorPtr)
        GuiCheckBox(Rectangle(screenWidth - 250, 420, 20, 20), b"Integrate Root Motion", integrateRootMotionPtr)
        GuiLabel(Rectangle(screenWidth - 250, 440, 220, 20), b"Trajectory Projection: Terrain")
        GuiLabel(Rectangle(screenWidth - 250, 460, 220, 20), b"Terrain Grid: %d x %d" % (
            terrainHeightGrid["num_x"],
            terrainHeightGrid["num_z"],
        ))
        GuiLabel(Rectangle(screenWidth - 250, 480, 220, 20), b"Terrain H: %.4f" % terrainHeightAtFocus)
        GuiLabel(Rectangle(screenWidth - 250, 500, 220, 20), b"Terrain Samples: %d" % len(terrainProvider.sample_positions))
        GuiLabel(Rectangle(screenWidth - 250, 520, 220, 20), b"Terrain N: [% .2f % .2f % .2f]" % (
            terrainNormalAtFocus[0],
            terrainNormalAtFocus[1],
            terrainNormalAtFocus[2],
        ))
        GuiLabel(Rectangle(screenWidth - 250, 540, 220, 20), b"Pen: %d max %.4f" % (
            penetrationCount,
            maxPenetrationDepth,
        ))
        GuiLabel(Rectangle(screenWidth - 250, 560, 220, 20), b"%s: mean %.6f max %.6f" % (
            poseErrorLabel,
            posePositionErrorMean,
            posePositionErrorMax))

        playback.draw_ui(screenWidth, screenHeight)

  
        EndDrawing()

    UnloadRenderTexture(lighted)
    UnloadRenderTexture(ssaoBack)
    UnloadRenderTexture(ssaoFront)
    UnloadRenderTexture(lighted)
    UnloadGBuffer(gbuffer)

    UnloadShadowMap(shadowMap)
    
    UnloadModel(terrainModel)
    UnloadModel(poseModel)
    UnloadModel(genoModel)
    UnloadModel(groundModel)
    
    UnloadShader(fxaaShader)    
    UnloadShader(blurShader)    
    UnloadShader(ssaoShader) 
    UnloadShader(lightingShader)    
    UnloadShader(basicShader)
    UnloadShader(skinnedBasicShader)
    UnloadShader(skinnedShadowShader)
    UnloadShader(shadowShader)
    
    CloseWindow()
