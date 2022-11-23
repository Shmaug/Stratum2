#pragma once

#include <Core/Pipeline.hpp>
#include "Scene.hpp"

#include <Shaders/compat/denoiser.h>
#include <Shaders/compat/filter_type.h>

namespace tinyvkpt {

class Denoiser {
public:
	Node& mNode;

	Denoiser(Node&);

	void createPipelines(Device& device);

	void drawGui();

	Image::View denoise(CommandBuffer& commandBuffer, const Image::View& radiance, const Image::View& albedo, const Buffer::View<ViewData>& views, const Buffer::View<VisibilityInfo>& visibility, const Buffer::View<DepthInfo>& depths, const Image::View& prev_uvs);

	inline void resetAccumulation() {
		mResetAccumulation = true;
		mAccumulatedFrames = 0;
	}
	inline bool reprojection() const { return mReprojection; }
	inline bool demodulateAlbedo() const { return mDemodulateAlbedo; }
	inline void reprojection(const bool v) { mReprojection = v; }
	inline void demodulateAlbedo(const bool v) { mDemodulateAlbedo = v; }

private:
	bool mReprojection = true;
	bool mDemodulateAlbedo = true;

	uint32_t mAccumulatedFrames = 0;
	uint32_t mAtrousIterations = 0;
	uint32_t mHistoryTap = 0;
	FilterKernelType mFilterType = FilterKernelType::eBox3;
	DenoiserDebugMode mDebugMode = DenoiserDebugMode::eNone;
	bool mResetAccumulation = false;

	PushConstants mPushConstants;

	ComputePipelineCache mTemporalAccumulationPipeline;
	ComputePipelineCache mEstimateVariancePipeline;
	ComputePipelineCache mAtrousPipeline;
	ComputePipelineCache mCopyRGBPipeline;

	class FrameResources : public Device::Resource {
	public:
		inline FrameResources(Device& device) : Device::Resource(device, "Denoiser frame resources") {}

		Buffer::View<ViewData> mViews;
		Image::View mRadiance;
		Image::View mAlbedo;
		Buffer::View<VisibilityInfo> mVisibility;
		Buffer::View<DepthInfo> mDepth;
		Image::View mAccumColor;
		Image::View mAccumMoments;
		Image::View mDebugImage;
		array<Image::View, 2> mTemp;
		shared_ptr<DescriptorSets> mDescriptors;
	};

	DeviceResourcePool<FrameResources> mFrameResources;
	shared_ptr<FrameResources> mCurFrame, mPrevFrame;

};

}