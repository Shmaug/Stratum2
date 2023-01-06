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
    uint2 mOutputExtent;
	uint mScreenPixelCount;  // Number of pixels
	uint mLightSubPathCount; // Number of light sub-paths

    float4 mSceneSphere;

	uint mViewCount;
	uint mLightCount;
    uint mEnvironmentMaterialAddress;
    float mEnvironmentSampleProbability;

	uint mMinPathLength;
    uint mMaxPathLength;
	uint mLightImageQuantization;
	uint mHashGridCellCount;

    float mMergeRadius;
    float mMisVmWeightFactor;
    float mMisVcWeightFactor;
    float mVmNormalization;

    uint mRandomSeed;

    uint mDebugCameraPathLength;
    uint mDebugLightPathLength;

    uint pad;
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

    // Bidirectional path tracing with a light vertex cache
    // dVCM and dVC used for MIS
    kBptLvc,

    // Vertex connection and mering
    // dVCM, dVM, and dVC used for MIS
    kVcm,

	kNumVcmAlgorithmType
};

// Mis power
inline float Mis(float aPdf) {
    return pow2(aPdf);
}

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
    case stm2::VcmAlgorithmType::kBptLvc: return "Bidirectional path tracing (LVC)";
    case stm2::VcmAlgorithmType::kVcm: return "Vertex connection and merging";
	}
}
}
#endif