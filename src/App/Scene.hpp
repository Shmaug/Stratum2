#pragma once

#include <chrono>

#include <nanovdb/util/GridHandle.h>

#include <Core/Mesh.hpp>
#include <Core/Pipeline.hpp>

#include "Node.hpp"
#include "Material.hpp"

namespace tinyvkpt {

TransformData nodeToWorld(const Node& node);

struct MeshPrimitive {
	shared_ptr<Material> mMaterial;
	shared_ptr<Mesh> mMesh;

	void drawGui(Node& node);
};

struct SpherePrimitive {
	shared_ptr<Material> mMaterial;
	float mRadius;

	void drawGui(Node& node);
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

	void drawGui();
};

class Scene {
public:
	struct RenderResources : public Device::Resource {
		Buffer::View<byte> mAccelerationStructureBuffer;
		shared_ptr<vk::raii::AccelerationStructureKHR> mAccelerationStructure;

		unordered_map<const void* /* address of component */, pair<TransformData, uint32_t /* instance index */ >> mInstanceTransformMap;
		vector<weak_ptr<Node>> mInstanceNodes;

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

		RenderResources() = default;
		RenderResources(const RenderResources&) = default;
		RenderResources(RenderResources&&) = default;
		RenderResources& operator=(const RenderResources&) = default;
		RenderResources& operator=(RenderResources&&) = default;

		inline RenderResources(Device& device) : Device::Resource(device, "RenderResources"), mAccelerationStructure(nullptr) {}

		void update(Scene& scene, CommandBuffer& commandBuffer, const shared_ptr<RenderResources>& prevFrame = {});
	};

	Node& mNode;

	Scene(Node&);

	void createPipelines();

	inline const shared_ptr<RenderResources>& resources() const { return mResources; }

	void drawGui();

	inline void markDirty() { mUpdateOnce = true; }
	void update(CommandBuffer& commandBuffer, const float deltaTime);

	shared_ptr<Node> loadEnvironmentMap(CommandBuffer& commandBuffer, const filesystem::path& filename);
	shared_ptr<Node> loadGltf(CommandBuffer& commandBuffer, const filesystem::path& filename);
	shared_ptr<Node> loadVol(CommandBuffer& commandBuffer, const filesystem::path& filename);
	shared_ptr<Node> loadNvdb(CommandBuffer& commandBuffer, const filesystem::path& filename);
#ifdef STRATUM_ENABLE_ASSIMP
	shared_ptr<Node> loadAssimp(CommandBuffer& commandBuffer, const filesystem::path& filename);
#endif
#ifdef STRATUM_ENABLE_OPENVDB
	shared_ptr<Node> loadVdb(CommandBuffer& commandBuffer, const filesystem::path& filename);
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
	inline shared_ptr<Node> load(CommandBuffer& commandBuffer, const filesystem::path& filename) {
		const string& ext = filename.extension().string();
		if      (ext == ".hdr") return loadEnvironmentMap(commandBuffer, filename);
		else if (ext == ".exr") return loadEnvironmentMap(commandBuffer, filename);
		else if (ext == ".gltf") return loadGltf(commandBuffer, filename);
		else if (ext == ".glb") return loadGltf(commandBuffer, filename);
		else if (ext == ".vol") return loadVol(commandBuffer, filename);
		else if (ext == ".nvdb") return loadNvdb(commandBuffer, filename);
	#ifdef STRATUM_ENABLE_ASSIMP
		else if (ext == ".fbx") return loadAssimp(commandBuffer, filename);
		else if (ext == ".obj") return loadAssimp(commandBuffer, filename);
		else if (ext == ".blend") return loadAssimp(commandBuffer, filename);
		else if (ext == ".ply") return loadAssimp(commandBuffer, filename);
		else if (ext == ".stl") return loadAssimp(commandBuffer, filename);
	#endif
	#ifdef STRATUM_ENABLE_OPENVDB
		else if (ext == ".vdb") return loadVdb(commandBuffer, filename)
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


	unordered_map<size_t, AccelerationStructureData> mAABBs;
	unordered_map<size_t, AccelerationStructureData> mMeshAccelerationStructures;

	DeviceResourcePool<RenderResources> mResourcePool;
	shared_ptr<RenderResources> mResources;

	ComputePipelineCache mCopyVerticesPipeline;
	ComputePipelineCache mConvertAlphaToRoughnessPipeline;
	ComputePipelineCache mConvertShininessToRoughnessPipeline;
	ComputePipelineCache mConvertPbrPipeline;
	ComputePipelineCache mConvertDiffuseSpecularPipeline;

	vector<string> mToLoad;

	bool mAlwaysUpdate = false;
	bool mUpdateOnce = false;

	// dirty animation hack
	friend struct TransformData;
	float3 mAnimateTranslate = float3::Zero();
	float3 mAnimateRotate = float3::Zero();
	float3 mAnimateWiggleBase = float3::Zero();
	float3 mAnimateWiggleOffset = float3::Zero();
	float mAnimateWiggleSpeed = 1;
	float mAnimateWiggleTime = 0;
	shared_ptr<Node> mAnimatedTransform;
};

}