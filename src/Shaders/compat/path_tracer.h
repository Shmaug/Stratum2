#pragma once

#include "scene.h"

STM_NAMESPACE_BEGIN

// 80 bytes
struct VcmVertex {
    ShadingData mShadingData; // Position of the vertex
    float3 mThroughput;       // Path throughput (including emission)
    uint mPathLength;         // Number of segments between source and vertex

    float dVCM; // MIS quantity used for vertex connection and merging
    float dVC;  // MIS quantity used for vertex connection
    float dVM;  // MIS quantity used for vertex merging
    uint mLocalDirectionIn;
};


struct PathTracerPushConstants {
    float4 mSceneSphere;

    uint mEnvironmentMaterialAddress;
    float mEnvironmentSampleProbability;

	uint mViewCount;
	uint mLightCount;

	uint mMinPathLength;
    uint mMaxPathLength;

	uint mScreenPixelCount;  // Number of pixels
	uint mLightSubPathCount; // Number of light sub-paths

    uint mRandomSeed;

	float mRadiusAlpha;  // Radius reduction rate parameter
	float mRadiusFactor; // Initial radius = scene radius * radius factor
    uint mVmIteration;

	uint mHashGridCellCount;
	uint mLightImageQuantization;

    uint mDebugCameraPathLength;
    uint mDebugLightPathLength;

    float VcmMergeRadius() {
        // Setup our radius, 1st iteration has aIteration == 0, thus offset
        float radius = mRadiusFactor * mSceneSphere[3];
        radius /= pow(float(mVmIteration + 1), 0.5f * (1 - mRadiusAlpha));
        // Purely for numeric stability
        return max(radius, 1e-7f);
    }
};

enum VcmAlgorithmType {
	// unidirectional path tracing from the camera, with next event estimation
    kPathTrace = 0,

    // unidirectional path tracing from light sources
    // No MIS weights (dVCM, dVM, dVC all ignored)
    kLightTrace,

    // Camera and light vertices merged on first non-specular surface from camera.
    // Cannot handle mixed specular + non-specular materials.
    // No MIS weights (dVCM, dVM, dVC all ignored)
    kPpm,

    // Camera and light vertices merged on along full path.
    // dVCM and dVM used for MIS
    kBpm,

    // Standard bidirectional path tracing
    // dVCM and dVC used for MIS
    kBpt,

    // Vertex connection and mering
    // dVCM, dVM, and dVC used for MIS
    kVcm,

	kNumVcmAlgorithmType
};

STM_NAMESPACE_END

#ifdef __cplusplus
namespace std {
inline string to_string(const stm2::VcmAlgorithmType algorithm) {
	switch (algorithm) {
    default: return "";
    case stm2::VcmAlgorithmType::kPathTrace: return "Path tracing";
    case stm2::VcmAlgorithmType::kLightTrace: return "Light tracing";
	case stm2::VcmAlgorithmType::kPpm: return "Progressive photon mapping";
	case stm2::VcmAlgorithmType::kBpm: return "Bidirectional photon mapping";
    case stm2::VcmAlgorithmType::kBpt: return "Bidirectional path tracing";
    case stm2::VcmAlgorithmType::kVcm: return "Vertex connection and merging";
	}
}
}
#endif