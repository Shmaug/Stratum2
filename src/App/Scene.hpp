#pragma once

#include <nanovdb/util/GridHandle.h>
#include <imgui/imgui.h> // materials have ImGui calls

#include <Core/Mesh.hpp>
#include <Core/Pipeline.hpp>

#include "SceneGraph.hpp"
#include "Material.hpp"

namespace tinyvkpt {

TransformData nodeToWorld(const Node& node);

struct MeshPrimitive {
	shared_ptr<Material> mMaterial;
	shared_ptr<Mesh> mMesh;
};

struct SpherePrimitive {
	shared_ptr<Material> mMaterial;
	float mRadius;
};

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
	struct RenderResources : public Device::Resource {
		Buffer::View<byte> mAccelerationStructureBuffer;
		shared_ptr<vk::raii::AccelerationStructureKHR> mAccelerationStructure;

		unordered_map<const void* /* address of component */, pair<TransformData, uint32_t /* instance index */ >> mInstanceTransformMap;
		vector<NodePtr> mInstanceNodes;

		Buffer::View<PackedVertexData> mVertices;
		Buffer::View<byte> mIndices;
		Buffer::View<uint32_t> mMaterialData;
		Buffer::View<InstanceData> mInstances;
		Buffer::View<TransformData> mInstanceTransforms;
		Buffer::View<TransformData> mInstanceInverseTransforms;
		Buffer::View<TransformData> mInstanceMotionTransforms;
		Buffer::View<uint32_t> mLightInstanceMap;
		Buffer::View<uint32_t> mInstanceIndexMap;

		MaterialResources mMaterialResources;
		uint32_t mEnvironmentMaterialAddress;
		uint32_t mMaterialCount;
		uint32_t mEmissivePrimitiveCount;

		RenderResources(Scene& scene, CommandBuffer& commandBuffer, const shared_ptr<RenderResources>& prevFrame = {});
		RenderResources(const RenderResources&) = default;
		RenderResources(RenderResources&&) = default;
		RenderResources& operator=(const RenderResources&) = default;
		RenderResources& operator=(RenderResources&&) = default;
	};

	Scene(Device& device);

	void createPipelines();

	inline const shared_ptr<RenderResources>& resources() const { return mResources; }
	inline const NodePtr& node() const { return mRootNode; }

	void drawGui();

	inline void markDirty() { mUpdateOnce = true; }
	void update(CommandBuffer& commandBuffer, const float deltaTime);

	NodePtr loadEnvironmentMap(CommandBuffer& commandBuffer, const filesystem::path& filename);
	NodePtr loadGltf(CommandBuffer& commandBuffer, const filesystem::path& filename);
	NodePtr loadVol(CommandBuffer& commandBuffer, const filesystem::path& filename);
	NodePtr loadNvdb(CommandBuffer& commandBuffer, const filesystem::path& filename);
#ifdef STRATUM_ENABLE_ASSIMP
	NodePtr loadAssimp(CommandBuffer& commandBuffer, const filesystem::path& filename);
#endif
#ifdef STRATUM_ENABLE_OPENVDB
	NodePtr loadVdb(CommandBuffer& commandBuffer, const filesystem::path& filename);
#endif

	inline vector<string> loaderFilters() {
		return {
			"All Files", "*",
			"Environment Maps (.exr .hdr)", "*.exr *.hdr",
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
	inline NodePtr load(CommandBuffer& commandBuffer, const filesystem::path& filename) {
		const string& ext = filename.extension().string();
		if      (ext == ".hdr") loadEnvironmentMap(commandBuffer, filename);
		else if (ext == ".exr") loadEnvironmentMap(commandBuffer, filename);
		else if (ext == ".gltf") loadGltf(commandBuffer, filename);
		else if (ext == ".glb") loadGltf(commandBuffer, filename);
		else if (ext == ".vol") loadVol(commandBuffer, filename);
		else if (ext == ".nvdb") loadNvdb(commandBuffer, filename);
	#ifdef STRATUM_ENABLE_ASSIMP
		else if (ext == ".fbx") loadAssimp(commandBuffer, filename);
		else if (ext == ".obj") loadAssimp(commandBuffer, filename);
		else if (ext == ".blend") loadAssimp(commandBuffer, filename);
		else if (ext == ".ply") loadAssimp(commandBuffer, filename);
		else if (ext == ".stl") loadAssimp(commandBuffer, filename);
	#endif
	#ifdef STRATUM_ENABLE_OPENVDB
		else if (ext == ".vdb") loadVdb(commandBuffer, filename)
	#endif
		else
			throw runtime_error("unknown extension:" + ext);
	}

	ImageValue1 alphaToRoughness(CommandBuffer& commandBuffer, const ImageValue1& alpha);
	ImageValue1 shininessToRoughness(CommandBuffer& commandBuffer, const ImageValue1& alpha);
	Material makeMetallicRoughnessMaterial(CommandBuffer& commandBuffer, const ImageValue3& baseColor, const ImageValue4& metallicRoughness, const ImageValue3& transmission, const float eta, const ImageValue3& emission);
	Material makeDiffuseSpecularMaterial(CommandBuffer& commandBuffer, const ImageValue3& diffuse, const ImageValue3& specular, const ImageValue1& roughness, const ImageValue3& transmission, const float eta, const ImageValue3& emission);

private:
	using AccelerationStructureData = pair<shared_ptr<vk::raii::AccelerationStructureKHR>, Buffer::View<byte> /* acceleration structure buffer */>;

	NodePtr mRootNode;

	unordered_map<size_t, AccelerationStructureData> mAABBs;
	unordered_map<size_t, AccelerationStructureData> mMeshAccelerationStructures;

	ResourcePool<RenderResources> mResourcePool;
	shared_ptr<RenderResources> mResources;

	ComputePipelineCache mCopyVerticesPipeline;
	ComputePipelineCache mConvertAlphaToRoughnessPipeline;
	ComputePipelineCache mConvertShininessToRoughnessPipeline;
	ComputePipelineCache mConvertPbrPipeline;
	ComputePipelineCache mConvertDiffuseSpecularPipeline;

	vector<string> mToLoad;

	bool mAlwaysUpdate = false;
	bool mUpdateOnce = false;
};

}