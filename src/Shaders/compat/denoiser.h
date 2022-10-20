#ifndef DENOISER_H
#define DENOISER_H

#include "common.h"

#ifdef __cplusplus
namespace tinyvkpt {
#endif

enum class DenoiserDebugMode {
	eNone,
	eSampleCount,
	eVariance,
	eWeightSum,
	eDebugModeCount
};

#ifdef __cplusplus
} // namespace tinyvkpt

namespace std {
inline string to_string(const tinyvkpt::DenoiserDebugMode& m) {
switch (m) {
	default: return "Unknown";
	case tinyvkpt::DenoiserDebugMode::eNone: return "None";
	case tinyvkpt::DenoiserDebugMode::eSampleCount: return "Sample Count";
	case tinyvkpt::DenoiserDebugMode::eVariance: return "Variance";
	case tinyvkpt::DenoiserDebugMode::eWeightSum: return "Weight Sum";
}
};
}
#endif

#ifdef __HLSL__

#include "scene.h"

struct DenoiserParameters {
	StructuredBuffer<ViewData> gViews;
	StructuredBuffer<uint> gInstanceIndexMap;
	StructuredBuffer<VisibilityInfo> gVisibility;
	StructuredBuffer<VisibilityInfo> gPrevVisibility;
	StructuredBuffer<DepthInfo> gDepth;
	StructuredBuffer<DepthInfo> gPrevDepth;
	Texture2D<float2> gPrevUVs;
	Texture2D<float4> gRadiance;
	Texture2D<float4> gAlbedo;
	RWTexture2D<float4> gAccumColor;
	RWTexture2D<float2> gAccumMoments;
	RWTexture2D<float4> gFilterImages[2];
	Texture2D<float4> gPrevRadiance;
	Texture2D<float4> gPrevAccumColor;
	Texture2D<float2> gPrevAccumMoments;
	SamplerState gStaticSampler;
	RWTexture2D<float4> gDebugImage;
};

#endif

#endif