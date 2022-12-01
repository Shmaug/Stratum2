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
	StructuredBuffer<ViewData> gViews;
	StructuredBuffer<uint> gInstanceIndexMap;
	StructuredBuffer<VisibilityData> gVisibility;
	StructuredBuffer<VisibilityData> gPrevVisibility;
	StructuredBuffer<DepthData> gDepth;
	StructuredBuffer<DepthData> gPrevDepth;
	Texture2D<float2> gPrevUVs;
	Texture2D<float4> gRadiance;
	Texture2D<float4> gAlbedo;
	RWTexture2D<float4> gAccumColor;
	RWTexture2D<float2> gAccumMoments;
	RWTexture2D<float4> gFilterImages[2];
	Texture2D<float4> gPrevRadiance;
	Texture2D<float4> gPrevAccumColor;
	Texture2D<float2> gPrevAccumMoments;
	SamplerState mStaticSampler;
	RWTexture2D<float4> gDebugImage;

	uint getViewIndex(const uint2 index) {
		for (uint i = 0; i < gPushConstants.mViewCount; i++)
			if (all(index >= gViews[i].mImageMin) && all(index < gViews[i].mImageMax))
				return i;
		return -1;
	}
};

#endif

#endif