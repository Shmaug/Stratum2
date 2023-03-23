#pragma once

#include <Core/Pipeline.hpp>
#include "Scene.hpp"

#include <Shaders/compat/denoiser.h>
#include <Shaders/compat/filter_type.h>

namespace stm2 {

class Denoiser {
public:
	Node& mNode;

	Denoiser(Node&);

	void createPipelines(Device& device);

	void drawGui();

	Image::View denoise(
		CommandBuffer& commandBuffer,
		const Image::View& image,
		const Image::View& albedo,
		const Image::View& prevUVs,
		const Image::View& visibility,
		const Image::View& depths,
		const Buffer::View<ViewData>& views);

	inline void resetAccumulation() {
		mResetAccumulation = true;
		mAccumulatedFrames = 0;
	}
	inline uint32_t accumulatedFrames() const { return mAccumulatedFrames; }
	inline bool reprojection() const { return mReprojection; }
	inline bool demodulateAlbedo() const { return mDemodulateAlbedo; }
	inline void reprojection(const bool v) { mReprojection = v; }
	inline void demodulateAlbedo(const bool v) { mDemodulateAlbedo = v; }

private:
	bool mReprojection = true;
	bool mDemodulateAlbedo = true;
	bool mCheckNormal = true;
	bool mCheckDepth = true;
	bool mShadowHysteresis = false;

	uint32_t mAccumulatedFrames = 0;
	uint32_t mAtrousIterations = 0;
	uint32_t mHistoryTap = 0;
	FilterKernelType mFilterType = FilterKernelType::eBox3;
	DenoiserDebugMode mDebugMode = DenoiserDebugMode::eNone;
	bool mResetAccumulation = false;

	uint32_t mHistoryLimit = 0;
	uint32_t mVarianceBoostLength = 4;
	float mSigmaLuminanceBoost = 3;
	float mShadowPreserveOffset = 0.1f;
	float mShadowPreserveScale = 1;

	ComputePipelineCache mTemporalAccumulationPipeline;
	ComputePipelineCache mEstimateVariancePipeline;
	ComputePipelineCache mAtrousPipeline;
	ComputePipelineCache mCopyRGBPipeline;

	DeviceResourcePool mResourcePool;

	Image::View mPrevVisibility;
	Image::View mPrevDepth;
};

}