import cffi
import numpy as np
import struct

from pyray import Model, Mesh, BoneInfo, Transform, Matrix, Vector3
from raylib import *

from genoview.State import SceneResources
from genoview.utils import quat


ffi = cffi.FFI()


def FileRead(out, size, f):
    ffi.memmove(out, f.read(size), size)


def LoadCharacterModel(fileName):

    model = Model()
    model.transform = MatrixIdentity()

    with open(fileName, "rb") as f:

        model.materialCount = 1
        model.materials = MemAlloc(model.materialCount * ffi.sizeof(Mesh()))
        model.materials[0] = LoadMaterialDefault()

        model.meshCount = 1
        model.meshMaterial = MemAlloc(model.meshCount * ffi.sizeof(Mesh()))
        model.meshMaterial[0] = 0

        model.meshes = MemAlloc(model.meshCount * ffi.sizeof(Mesh()))
        model.meshes[0].vertexCount = struct.unpack('I', f.read(4))[0]
        model.meshes[0].triangleCount = struct.unpack('I', f.read(4))[0]
        model.boneCount = struct.unpack('I', f.read(4))[0]

        model.meshes[0].boneCount = model.boneCount
        model.meshes[0].vertices = MemAlloc(model.meshes[0].vertexCount * 3 * ffi.sizeof("float"))
        model.meshes[0].texcoords = MemAlloc(model.meshes[0].vertexCount * 2 * ffi.sizeof("float"))
        model.meshes[0].normals = MemAlloc(model.meshes[0].vertexCount * 3 * ffi.sizeof("float"))
        model.meshes[0].boneIds = MemAlloc(model.meshes[0].vertexCount * 4 * ffi.sizeof("unsigned char"))
        model.meshes[0].boneWeights = MemAlloc(model.meshes[0].vertexCount * 4 * ffi.sizeof("float"))
        model.meshes[0].indices = MemAlloc(model.meshes[0].triangleCount * 3 * ffi.sizeof("unsigned short"))
        model.meshes[0].animVertices = MemAlloc(model.meshes[0].vertexCount * 3 * ffi.sizeof("float"))
        model.meshes[0].animNormals = MemAlloc(model.meshes[0].vertexCount * 3 * ffi.sizeof("float"))
        model.bones = MemAlloc(model.boneCount * ffi.sizeof(BoneInfo()))
        model.bindPose = MemAlloc(model.boneCount * ffi.sizeof(Transform()))

        FileRead(model.meshes[0].vertices, ffi.sizeof("float") * model.meshes[0].vertexCount * 3, f)
        FileRead(model.meshes[0].texcoords, ffi.sizeof("float") * model.meshes[0].vertexCount * 2, f)
        FileRead(model.meshes[0].normals, ffi.sizeof("float") * model.meshes[0].vertexCount * 3, f)
        FileRead(model.meshes[0].boneIds, ffi.sizeof("unsigned char") * model.meshes[0].vertexCount * 4, f)
        FileRead(model.meshes[0].boneWeights, ffi.sizeof("float") * model.meshes[0].vertexCount * 4, f)
        FileRead(model.meshes[0].indices, ffi.sizeof("unsigned short") * model.meshes[0].triangleCount * 3, f)
        ffi.memmove(model.meshes[0].animVertices, model.meshes[0].vertices, ffi.sizeof("float") * model.meshes[0].vertexCount * 3)
        ffi.memmove(model.meshes[0].animNormals, model.meshes[0].normals, ffi.sizeof("float") * model.meshes[0].vertexCount * 3)
        FileRead(model.bones, ffi.sizeof(BoneInfo()) * model.boneCount, f)
        FileRead(model.bindPose, ffi.sizeof(Transform()) * model.boneCount, f)

        model.meshes[0].boneMatrices = MemAlloc(model.boneCount * ffi.sizeof(Matrix()))
        for i in range(model.boneCount):
            model.meshes[0].boneMatrices[i] = MatrixIdentity()

    UploadMesh(ffi.addressof(model.meshes[0]), True)

    return model


def GetModelBindPoseAsNumpyArrays(model):

    bindPos = np.zeros([model.boneCount, 3])
    bindRot = np.zeros([model.boneCount, 4])

    for boneId in range(model.boneCount):
        bindTransform = model.bindPose[boneId]
        bindPos[boneId] = (bindTransform.translation.x, bindTransform.translation.y, bindTransform.translation.z)
        bindRot[boneId] = (bindTransform.rotation.w, bindTransform.rotation.x, bindTransform.rotation.y, bindTransform.rotation.z)

    return bindPos, bindRot


def UpdateModelPoseFromNumpyArrays(model, bindPos, bindRot, animPos, animRot):

    meshPos = quat.mul_vec(animRot, quat.inv_mul_vec(bindRot, -bindPos)) + animPos
    meshRot = quat.mul_inv(animRot, bindRot)

    matArray = np.frombuffer(ffi.buffer(
        model.meshes[0].boneMatrices, model.boneCount * 4 * 4 * 4),
        dtype=np.float32).reshape([model.boneCount, 4, 4])

    matArray[:,:3,:3] = quat.to_xform(meshRot)
    matArray[:,:3,3] = meshPos


def LoadSceneResources(resource_path):
    ground_model = LoadModelFromMesh(GenMeshPlane(20.0, 20.0, 10, 10))
    geno_model = LoadCharacterModel(resource_path("Geno.bin", as_bytes=True))
    bind_pos, bind_rot = GetModelBindPoseAsNumpyArrays(geno_model)

    return SceneResources(
        ground_model=ground_model,
        ground_position=Vector3(0.0, -0.01, 0.0),
        geno_model=geno_model,
        pose_model=None,
        geno_position=Vector3(0.0, 0.0, 0.0),
        bind_pos=bind_pos,
        bind_rot=bind_rot,
    )


def UnloadSceneResources(scene):
    if scene.pose_model is not None:
        UnloadModel(scene.pose_model)
    UnloadModel(scene.geno_model)
    UnloadModel(scene.ground_model)
