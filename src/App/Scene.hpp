#pragma once

#include <chrono>

#include <nanovdb/util/GridHandle.h>

#include <Core/Mesh.hpp>
#include <Core/DeviceResourcePool.hpp>

#include "Node.hpp"
#include "Material.hpp"

#include <future>

namespace stm2 {

TransformData nodeToWorld(const Node& node);

template<int N>
struct ImageValue {
	VectorType<float,N> mValue;
	Image::View mImage;

	bool drawGui(const string& label);
};

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

struct EnvironmentMap : public ImageValue<3> {
    inline void store(MaterialResources &resources) const {
        resources.mMaterialData.AppendN(mValue);
        resources.mMaterialData.Append(resources.getIndex(mImage));
    }

    void drawGui(Node &node);
};

struct Camera {
	ProjectionData mProjection;
	vk::Rect2D mImageRect;

	inline ViewData view() {
		ViewData v;
		v.mProjection = mProjection;
		v.mImageMin = { mImageRect.offset.x, mImageRect.offset.y };
		v.mImageMax = { mImageRect.offset.x + mImageRect.extent.width, mImageRect.offset.y + mImageRect.extent.height };
		float2 extent = mProjection.backProject(float2::Constant(1)).head<2>() - mProjection.backProject(float2::Constant(-1)).head<2>();
		if (!mProjection.isOrthographic()) extent /= mProjection.mNearPlane;
		v.mProjection.mSensorArea = abs(extent[0] * extent[1]);
		return v;
	}

	void drawGui();
};

class Scene {
public:
	struct FrameData {
		vector<tuple<InstanceData, shared_ptr<Material>, TransformData>> mInstances;
		vector<shared_ptr<Buffer>> mVertexBuffers;
		vector<MeshVertexInfo> mMeshVertexInfo;

		vector<VolumeInfo> mInstanceVolumeInfo;

		unordered_map<const void* /* address of component */, pair<TransformData, uint32_t /* instance index */ >> mInstanceTransformMap;
		vector<weak_ptr<Node>> mInstanceNodes;
		uint32_t mLightCount;

		DeviceResourcePool mResourcePool;

		MaterialResources mMaterialResources;
		uint32_t mEnvironmentMaterialAddress;
		uint32_t mMaterialCount;
		uint32_t mEmissivePrimitiveCount;
		float3 mAabbMin, mAabbMax;

		Descriptors mDescriptors;

		Buffer::View<byte> mAccelerationStructureBuffer;

		inline void clear() {
			mInstances.clear();
			mInstanceVolumeInfo.clear();

			mInstanceTransformMap.clear();
			mInstanceNodes.clear();
			mLightCount = 0;

			mVertexBuffers.clear();
			mMeshVertexInfo.clear();

			mMaterialResources.clear();
			mEnvironmentMaterialAddress = -1;
			mMaterialCount = 0;
			mEmissivePrimitiveCount = 0;
			mAabbMin = float3::Constant( numeric_limits<float>::infinity());
			mAabbMax = float3::Constant(-numeric_limits<float>::infinity());

			mDescriptors.clear();
			mAccelerationStructureBuffer.reset();

			mResourcePool.clean();
		}
	};

	Node& mNode;

	Scene(Node&);

	void createPipelines();

	inline const FrameData& frameData() const { return mFrameData; }

	void drawGui();

	inline void markDirty() { mUpdateOnce = true; }
	inline chrono::high_resolution_clock::time_point lastUpdate() const { return mLastUpdate; }
	void update(CommandBuffer& commandBuffer, const float deltaTime);


	shared_ptr<Node> loadEnvironmentMap(CommandBuffer& commandBuffer, const filesystem::path& filename);
	shared_ptr<Node> loadGltf(CommandBuffer& commandBuffer, const filesystem::path& filename);
	shared_ptr<Node> loadMitsuba(CommandBuffer& commandBuffer, const filesystem::path& filename);
	shared_ptr<Node> loadVol(CommandBuffer& commandBuffer, const filesystem::path& filename);
	shared_ptr<Node> loadNvdb(CommandBuffer& commandBuffer, const filesystem::path& filename);
#ifdef ENABLE_ASSIMP
	shared_ptr<Node> loadAssimp(CommandBuffer& commandBuffer, const filesystem::path& filename);
#endif
#ifdef ENABLE_OPENVDB
	shared_ptr<Node> loadVdb(CommandBuffer& commandBuffer, const filesystem::path& filename);
#endif

	inline vector<string> loaderFilters() {
		return {
			"All Files", "*",
			"Environment Maps (.exr .hdr)", "*.exr *.hdr",
			"glTF Scenes (.gltf .glb)", "*.gltf *.glb",
			"Mitsuba Volumes (.vol)" , "*.vol",
			"NVDB Volume (.nvdb)" , "*.nvdb",
	#ifdef ENABLE_ASSIMP
			"Autodesk (.fbx)", "*.fbx",
			"Wavefront Object Files (.obj)", "*.obj",
			"Stanford Polygon Library Files (.ply)", "*.ply",
			"Stereolithography Files (.stl)", "*.stl",
			"Blender Scenes (.blend)", "*.blend",
	#endif
	#ifdef ENABLE_OPENVDB
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
		else if (ext == ".xml") return loadMitsuba(commandBuffer, filename);
		else if (ext == ".vol") return loadVol(commandBuffer, filename);
		else if (ext == ".nvdb") return loadNvdb(commandBuffer, filename);
	#ifdef ENABLE_ASSIMP
		else if (ext == ".fbx") return loadAssimp(commandBuffer, filename);
		else if (ext == ".obj") return loadAssimp(commandBuffer, filename);
		else if (ext == ".blend") return loadAssimp(commandBuffer, filename);
		else if (ext == ".ply") return loadAssimp(commandBuffer, filename);
		else if (ext == ".stl") return loadAssimp(commandBuffer, filename);
	#endif
	#ifdef ENABLE_OPENVDB
		else if (ext == ".vdb") return loadVdb(commandBuffer, filename);
	#endif
		else
			throw runtime_error("unknown extension:" + ext);
	}

	ImageValue<1> alphaToRoughness        (CommandBuffer& commandBuffer, const ImageValue<1>& alpha);
	ImageValue<1> shininessToRoughness    (CommandBuffer& commandBuffer, const ImageValue<1>& alpha);
	Material makeMetallicRoughnessMaterial(CommandBuffer& commandBuffer, const ImageValue<3>& baseColor, const ImageValue<4>& metallicRoughness, const ImageValue<3>& transmission, const float eta, const ImageValue<3>& emission);
	Material makeDiffuseSpecularMaterial  (CommandBuffer& commandBuffer, const ImageValue<3>& diffuse, const ImageValue<3>& specular, const ImageValue<1>& roughness, const ImageValue<3>& transmission, const float eta, const ImageValue<3>& emission);

private:
	using AccelerationStructureData = pair<shared_ptr<vk::raii::AccelerationStructureKHR>, Buffer::View<byte> /* acceleration structure buffer */>;

	// cache aabb BLASs
	unordered_map<size_t, AccelerationStructureData> mAABBs;
	// cache mesh BLASs
	unordered_map<size_t, AccelerationStructureData> mMeshAccelerationStructures;

	FrameData mFrameData;

	void updateFrameData(CommandBuffer& commandBuffer);

	ComputePipelineCache mConvertAlphaToRoughnessPipeline;
	ComputePipelineCache mConvertShininessToRoughnessPipeline;
	ComputePipelineCache mConvertPbrPipeline;
	ComputePipelineCache mConvertDiffuseSpecularPipeline;

	vector<string> mToLoad;
	vector<future<shared_ptr<Node>>> mLoading;

	bool mAlwaysUpdate = false;
	bool mUpdateOnce = false;
	chrono::high_resolution_clock::time_point mLastUpdate;


	// HACK: animation

	friend struct TransformData;

	shared_ptr<Node> mAnimatedTransform;
	float3 mAnimateTranslate = float3::Zero();
	float3 mAnimateRotate = float3::Zero();
	float3 mAnimateWiggleBase = float3::Zero();
	float3 mAnimateWiggleOffset = float3::Zero();
	float mAnimateWiggleSpeed = 1;
	float mAnimateWiggleTime = 0;
};

}