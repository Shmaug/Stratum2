#ifndef DENOISER_H
#define DENOISER_H

#include "common.h"

STM_NAMESPACE_BEGIN

enum class DenoiserDebugMode {
	eNone,
	eSampleCount,
	eVariance,
	eWeightSum,
	eDebugModeCount
};

STM_NAMESPACE_END


#ifdef __cplusplus
namespace std {
inline string to_string(const stm2::DenoiserDebugMode& m) {
	switch (m) {
		default: return "Unknown";
		case stm2::DenoiserDebugMode::eNone: return "None";
		case stm2::DenoiserDebugMode::eSampleCount: return "Sample Count";
		case stm2::DenoiserDebugMode::eVariance: return "Variance";
		case stm2::DenoiserDebugMode::eWeightSum: return "Weight Sum";
	}
}
}
#endif

#ifdef __HLSL__

#include "scene.h"

struct DenoiserParameters {
	StructuredBuffer<ViewData> mViews;
	StructuredBuffer<uint> mInstanceIndexMap;
	Texture2D<float2> mVisibility;
	Texture2D<float2> mPrevVisibility;
	Texture2D<float4> mDepth;
	Texture2D<float4> mPrevDepth;
	Texture2D<float2> mPrevUVs;
	Texture2D<float4> mInput;
	Texture2D<float4> mAlbedo;
	RWTexture2D<float4> mAccumColor;
	RWTexture2D<float2> mAccumMoments;
	RWTexture2D<float4> mFilterImages[2];

	Texture2D<float4> mPrevAccumColor;
	Texture2D<float2> mPrevAccumMoments;

	SamplerState mStaticSampler;

	uint getViewIndex(const uint2 index) {
		for (uint i = 0; i < gPushConstants.mViewCount; i++)
			if (all(index >= mViews[i].mImageMin) && all(index < mViews[i].mImageMax))
				return i;
		return -1;
	}
};

#endif

#endif