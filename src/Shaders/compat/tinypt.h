#ifndef TINYPT_H
#define TINYPT_H

#include "common.h"

STM_NAMESPACE_BEGIN

enum class TinyPTDebugMode {
	eNone = 0,
	eAlbedo,
	eDepth,
	eGeometryNormal,
	eShadingNormal,
	eTextureCoordinate,
	eMaterialAddress,
	eNumTinyPTDebugMode
};
enum class TinyPTFeatureFlagBits {
	ePerformanceCounters = 0,
	eMedia,
	eAlphaTest,
	eNormalMaps,
	eNEE,
	eNumTinyPTFeatureFlagBits
};

struct TinyPTPushConstants {
	uint mRandomSeed;
	uint mViewCount;

	uint mEnvironmentMaterialAddress;
	uint mLightCount;

	uint mMinBounces;
	uint mMaxBounces;
	uint mMaxDiffuseBounces;
};

STM_NAMESPACE_END

#ifdef __cplusplus
namespace std {
inline string to_string(const stm2::TinyPTDebugMode debugMode) {
	switch (debugMode) {
	default: return "None";
	case stm2::TinyPTDebugMode::eAlbedo: return "Albedo";
	case stm2::TinyPTDebugMode::eDepth: return "Depth";
	case stm2::TinyPTDebugMode::eGeometryNormal: return "Geometry normal";
	case stm2::TinyPTDebugMode::eShadingNormal: return "Shading normal";
	case stm2::TinyPTDebugMode::eTextureCoordinate: return "Texture coordinate";
	case stm2::TinyPTDebugMode::eMaterialAddress: return "Material address";
	}
}
inline string to_string(const stm2::TinyPTFeatureFlagBits featureFlag) {
	switch (featureFlag) {
	default: return "";
	case stm2::TinyPTFeatureFlagBits::ePerformanceCounters: return "Performance counters";
	case stm2::TinyPTFeatureFlagBits::eMedia: return "Media";
	case stm2::TinyPTFeatureFlagBits::eAlphaTest: return "Alpha test";
	case stm2::TinyPTFeatureFlagBits::eNormalMaps: return "Normal maps";
	case stm2::TinyPTFeatureFlagBits::eNEE: return "Next event estimation";
	}
}
}
#endif

#endif