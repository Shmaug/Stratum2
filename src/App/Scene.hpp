#pragma once

#include <nanovdb/util/GridHandle.h>
#include <imgui/imgui.h> // materials have ImGui calls

#include "SceneGraph.hpp"

#include <Core/Mesh.hpp>
#include "Material.hpp"

namespace tinyvkpt {

struct MeshPrimitive {
	shared_ptr<Material> mMaterial;
	shared_ptr<Mesh> mMesh;
};
struct SpherePrimitive {
	shared_ptr<Material> mMaterial;
	float mRadius;
};

TransformData nodeToWorld(const SceneNode& node);

Mesh loadSerialized(CommandBuffer& commandBuffer, const filesystem::path& filename, int shape_idx = -1);
Mesh loadObj(CommandBuffer& commandBuffer, const filesystem::path& filename);

struct Camera {
	ProjectionData mProjection;
	vk::Rect2D mImageRect;

	inline ViewData view() {
		ViewData v;
		v.projection = mProjection;
		v.image_min = { mImageRect.offset.x, mImageRect.offset.y };
		v.image_max = { mImageRect.offset.x + mImageRect.extent.width, mImageRect.offset.y + mImageRect.extent.height };
		float2 extent = mProjection.back_project(float2::Constant(1)).head<2>() - mProjection.back_project(float2::Constant(-1)).head<2>();
		if (!mProjection.orthographic()) extent /= mProjection.near_plane;
		v.projection.sensor_area = abs(extent[0] * extent[1]);
		return v;
	}
};

class Scene {
public:
	struct SceneData {
		Buffer::View<byte> mAccelerationStructureBuffer;
		vk::raii::AccelerationStructureKHR mAccelerationStructure;

		unordered_map<const void* /* address of component */, pair<TransformData, uint32_t /* instance index */ >> mInstanceTransformMap;
		vector<SceneNode*> mInstanceNodes;

		Buffer::View<PackedVertexData> mVertices;
		Buffer::View<byte> mIndices;
		Buffer::View<byte> mMaterialData;
		Buffer::View<InstanceData> mInstances;
		Buffer::View<TransformData> mInstanceTransforms;
		Buffer::View<TransformData> mInstanceInverseTransforms;
		Buffer::View<TransformData> mInstanceMotionTransforms;
		Buffer::View<uint32_t> mLightInstanceMap;
		Buffer::View<float> mDistributionData;
		Buffer::View<uint32_t> mInstanceIndexMap;

		MaterialResources mResources;
		uint32_t mEnvironmentMaterialAddress;

		uint32_t mMaterialCount;
		uint32_t mEmissivePrimitiveCount;
	};

	Scene();

	void drawGui();

	void createPipelines();

	inline const shared_ptr<SceneData>& data() const { return mSceneData; }

	inline void markDirty() { mUpdateOnce = true; }
	void update(CommandBuffer& commandBuffer, const float deltaTime);

	void loadEnvironmentMap(SceneNode& root, CommandBuffer& commandBuffer, const filesystem::path& filename);
	void loadGltf(SceneNode& root, CommandBuffer& commandBuffer, const filesystem::path& filename);
	void loadMitsuba(SceneNode& root, CommandBuffer& commandBuffer, const filesystem::path& filename);
	void loadVol(SceneNode& root, CommandBuffer& commandBuffer, const filesystem::path& filename);
	void loadNvdb(SceneNode& root, CommandBuffer& commandBuffer, const filesystem::path& filename);
#ifdef STRATUM_ENABLE_ASSIMP
	void loadAssimp(SceneNode& root, CommandBuffer& commandBuffer, const filesystem::path& filename);
#endif
#ifdef STRATUM_ENABLE_OPENVDB
	void loadVdb(SceneNode& root, CommandBuffer& commandBuffer, const filesystem::path& filename);
#endif

	inline vector<string> loaderFilters() {
		return {
			"All Files", "*",
			"Environment Maps (.exr .hdr)", "*.exr *.hdr",
			"Mitsuba Scenes (.xml)", "*.xml",
			"glTF Scenes (.gltf .glb)", "*.gltf *.glb",
			"Mitsuba Volumes (.vol)" , "*.vol",
			"NVDB Volume (.nvdb)" , "*.nvdb",
	#ifdef STRATUM_ENABLE_ASSIMP
			"Autodesk (.fbx)", "*.fbx",
			"Wavefront Object Files (.obj)", "*.obj",
			"Stanford Polygon Library Files (.ply)", "*.ply",
			"Stereolithography Files (.stl)", "*.stl",
			"Blender Scenes (.blend)", "*.blend",
	#endif
	#ifdef STRATUM_ENABLE_OPENVDB
			"VDB Volumes (.vdb)", "*.vdb",
	#endif
		};
	}
	inline void load(SceneNode& root, CommandBuffer& commandBuffer, const filesystem::path& filename) {
		const string& ext = filename.extension().string();
		if (ext == ".hdr") loadEnvironmentMap(root, commandBuffer, filename);
		else if (ext == ".exr") loadEnvironmentMap(root, commandBuffer, filename);
		else if (ext == ".xml") loadMitsuba(root, commandBuffer, filename);
		else if (ext == ".gltf") loadGltf(root, commandBuffer, filename);
		else if (ext == ".glb") loadGltf(root, commandBuffer, filename);
		else if (ext == ".vol") loadVol(root, commandBuffer, filename);
		else if (ext == ".nvdb") loadNvdb(root, commandBuffer, filename);
	#ifdef STRATUM_ENABLE_ASSIMP
		else if (ext == ".fbx") loadAssimp(root, commandBuffer, filename);
		else if (ext == ".obj") loadAssimp(root, commandBuffer, filename);
		else if (ext == ".blend") loadAssimp(root, commandBuffer, filename);
		else if (ext == ".ply") loadAssimp(root, commandBuffer, filename);
		else if (ext == ".stl") loadAssimp(root, commandBuffer, filename);
	#endif
	#ifdef STRATUM_ENABLE_OPENVDB
		else if (ext == ".vdb") loadVdb(root, commandBuffer, filename)
	#endif
		else
			throw runtime_error("unknown extension:" + ext);
	}

	ImageValue1 alphaToRoughness(CommandBuffer& commandBuffer, const ImageValue1& alpha);
	ImageValue1 shininessToRoughness(CommandBuffer& commandBuffer, const ImageValue1& alpha);
	Material makeMetallicRoughnessMaterial(CommandBuffer& commandBuffer, const ImageValue3& baseColor, const ImageValue4& metallicRoughness, const ImageValue3& transmission, const float eta, const ImageValue3& emission);
	Material makeDiffuseSpecularMaterial(CommandBuffer& commandBuffer, const ImageValue3& diffuse, const ImageValue3& specular, const ImageValue1& roughness, const ImageValue3& transmission, const float eta, const ImageValue3& emission);

private:
	struct MeshBLAS {
		Buffer::View<byte> mBuffer;
		shared_ptr<vk::raii::AccelerationStructureKHR> mAccelerationStructure;
		Buffer::StrideView mIndices;
	};

	unordered_map<size_t, shared_ptr<vk::raii::AccelerationStructureKHR>> mAABBs;

	unordered_map<Mesh*, Buffer::View<PackedVertexData>> mMeshVertices;
	unordered_map<size_t, MeshBLAS> mMeshAccelerationStructures;

	shared_ptr<SceneData> mSceneData;

	shared_ptr<ComputePipeline> mCopyVerticesPipeline;

	shared_ptr<ComputePipeline> mConvertAlphaToRoughnessPipeline;
	shared_ptr<ComputePipeline> mConvertShininessToRoughnessPipeline;
	shared_ptr<ComputePipeline> mConvertPbrPipeline;
	shared_ptr<ComputePipeline> mConvertDiffuseSpecularPipeline;

	vector<string> mToLoad;

	bool mAlwaysUpdate = false;
	bool mUpdateOnce = false;
};

}