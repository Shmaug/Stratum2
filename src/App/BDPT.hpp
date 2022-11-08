#pragma once

#include "Scene.hpp"

#include <Shaders/compat/bdpt.h>
#include <Shaders/compat/tonemap.h>

namespace tinyvkpt {

struct HashGridData;

class BDPT {
public:
	void createPipelines(Device& device);

	void drawGui();
	void update(CommandBuffer& commandBuffer, const float deltaTime);
	void render(CommandBuffer& commandBuffer, const Image::View& renderTarget, const vector<pair<ViewData,TransformData>>& views);

	inline Image::View prevResult() { return mPrevFrame ? mPrevFrame->mTonemapResult : Image::View(); }

private:
	Node& mNode;

	enum RenderPipelineIndex {
		eSamplePhotons,
		eSampleVisibility,
		eSubpathConnect,
		eTraceShadows,
		eAddLightTrace,
		ePresampleLights,
		eHashGridComputeIndices,
		eHashGridSwizzle,
		ePipelineCount
	};
	array<ComputePipelineCache, RenderPipelineIndex::ePipelineCount> mRenderPipelines;
	ComputePipelineCache mTonemapPipeline;
	ComputePipelineCache mTonemapMaxReducePipeline;

	BDPTPushConstants mPushConstants;
	PushConstants mTonemapPushConstants;

	bool mGammaCorrect = true;
	TonemapMode mTonemapMode = TonemapMode::eRaw;

	bool mHalfColorPrecision = false;
	bool mPauseRendering = false;
	bool mRandomPerFrame = true;
	bool mForceLambertian = false;
	bool mDenoise = true;
	uint32_t mSamplingFlags = 0;
	BDPTDebugMode mDebugMode = BDPTDebugMode::eNone;
	uint32_t mLightTraceQuantization = 65536;

	Buffer::View<uint32_t> mRayCount;
	vector<uint32_t> mPrevRayCount;
	vector<float> mRaysPerSecond;
	float mRayCountTimer;

	class FrameResources : public Device::Resource {
	public:
		inline FrameResources(Device& device) : Device::Resource(device, "BDPT frame resources") {}
		Descriptors mSceneDescriptors;
		shared_ptr<DescriptorSets> mDescriptors;

		shared_ptr<Scene::RenderResources> mSceneData;

		Buffer::View<ViewData> mViews;
		Buffer::View<TransformData> mViewTransforms;
		Buffer::View<TransformData> mViewInverseTransforms;
		Buffer::View<uint32_t> mViewMediumIndices;
		Image::View mRadiance;
		Image::View mAlbedo;
		Image::View mPrevUVs;
		Image::View mDebugImage;

		unordered_map<string, Buffer::View<byte>> mPathData;
		vector<shared_ptr<HashGridData>> mHashGrids;
		Buffer::View<VisibilityInfo> mSelectionData;
		bool mSelectionDataValid;

		Image::View mDenoiseResult;
		Buffer::View<uint4> mTonemapMax;
		Image::View mTonemapResult;
		uint32_t mFrameNumber;
	};

	ResourcePool<FrameResources> mFrameResourcePool;
	shared_ptr<FrameResources> mCurFrame, mPrevFrame;
};

}