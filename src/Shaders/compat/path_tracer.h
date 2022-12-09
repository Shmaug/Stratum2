#ifndef PATHTRACER_H
#define PATHTRACER_H

#include "common.h"

STM_NAMESPACE_BEGIN

struct PathTracerPushConstants {
    uint mDebugViewVertices;
    uint mDebugLightVertices;

	uint mRandomSeed;
	uint mViewCount;

    uint mEnvironmentMaterialAddress;
    float mEnvironmentSampleProbability;
	uint mLightCount;

	uint mMinBounces;
	uint mMaxBounces;
	uint mMaxDiffuseBounces;
};

enum class PathTracerFeatureFlagBits {
	ePerformanceCounters = 0,
	eMedia,
	eAlphaTest,
	eNormalMaps,
	eNee,
	eNeeMis,
	eNumPathTracerFeatureFlagBits
};
enum class PathTracerDebugMode {
	eNone = 0,
	eAlbedo,
	eDepth,
	eGeometryNormal,
	eShadingNormal,
    eTextureCoordinate,
    eMaterialAddress,
    ePrevUV,
    ePathTypeContribution,
	eNumPathTracerDebugMode
};

STM_NAMESPACE_END

#ifdef __cplusplus
namespace std {
inline string to_string(const stm2::PathTracerFeatureFlagBits featureFlag) {
	switch (featureFlag) {
	default: return "";
	case stm2::PathTracerFeatureFlagBits::ePerformanceCounters: return "Performance counters";
	case stm2::PathTracerFeatureFlagBits::eMedia: return "Media";
	case stm2::PathTracerFeatureFlagBits::eAlphaTest: return "Alpha test";
    case stm2::PathTracerFeatureFlagBits::eNormalMaps: return "Normal maps";
    case stm2::PathTracerFeatureFlagBits::eNee: return "Next event estimation";
    case stm2::PathTracerFeatureFlagBits::eNeeMis: return "Next event estimation MIS";
	}
}
inline string to_string(const stm2::PathTracerDebugMode debugMode) {
	switch (debugMode) {
	default: return "None";
	case stm2::PathTracerDebugMode::eAlbedo: return "Albedo";
	case stm2::PathTracerDebugMode::eDepth: return "Depth";
	case stm2::PathTracerDebugMode::eGeometryNormal: return "Geometry normal";
	case stm2::PathTracerDebugMode::eShadingNormal: return "Shading normal";
    case stm2::PathTracerDebugMode::eTextureCoordinate: return "Texture coordinate";
    case stm2::PathTracerDebugMode::eMaterialAddress: return "Material address";
    case stm2::PathTracerDebugMode::ePrevUV: return "Prev UV";
    case stm2::PathTracerDebugMode::ePathTypeContribution: return "Path type contribution";
	}
}
}
#endif

#endif