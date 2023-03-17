#pragma once

#include "scene.h"

STM_NAMESPACE_BEGIN

// Mis power
inline float Mis(float aPdf) {
    return pow2(aPdf);
}

struct VcmConstants {
    float4 mSceneSphere;
    float mMergeRadius;
    float mMisVmWeightFactor;
    float mMisVcWeightFactor;
    float mVmNormalization;
};

struct VcmPushConstants {
    uint2 mOutputExtent;
	uint mScreenPixelCount;  // Number of pixels
	uint mLightSubPathCount; // Number of light sub-paths

	uint mViewCount;
	uint mLightCount;
    uint mEnvironmentMaterialAddress;
    float mEnvironmentSampleProbability;

	uint mMinPathLength;
    uint mMaxPathLength;
	uint mLightImageQuantization;
    uint mFlags;

    uint mRandomSeed;
    uint mDebugPathLengths;
    uint pad0;
    uint pad1;

    float mDIReservoirMaxM;
    float mLVCReservoirMaxM;
    uint mDIReservoirSampleCount;
    uint mLVCReservoirSampleCount;

	SLANG_MUTATING
    void reservoirHistoryValid(const bool v) { BF_SET(mFlags, v, 0, 1); }
    bool reservoirHistoryValid() CONST_CPP { return BF_GET(mFlags, 0, 1); }

    SLANG_MUTATING
    void debugCameraPathLength(const uint v) { BF_SET(mDebugPathLengths, v, 0, 16); }
    uint debugCameraPathLength() CONST_CPP { return BF_GET(mDebugPathLengths, 0, 16); }

    SLANG_MUTATING
    void debugLightPathLength(const uint v) { BF_SET(mDebugPathLengths, v, 16, 16); }
    uint debugLightPathLength() CONST_CPP { return BF_GET(mDebugPathLengths, 16, 16); }
};

// 80 bytes
struct VcmVertex {
    ShadingData mShadingData; // Position of the vertex
    float3 mThroughput;       // Path throughput (including emission)
    uint mPackedData;         // mPathLength and mPathSamplePdfA
    float dVCM;               // MIS quantity used for vertex connection and merging
    float dVC;                // MIS quantity used for vertex connection
    float dVM;                // MIS quantity used for vertex merging
    uint mLocalDirectionIn;
};
// 48 bytes
struct PackedVcmVertex {
    float3 mLocalPosition;
    uint mInstancePrimitiveIndex;
    float3 mThroughput;       // Path throughput (including emission)
    uint mPackedData;         // mPathLength and mPathSamplePdfA
    float dVCM;               // MIS quantity used for vertex connection and merging
    float dVC;                // MIS quantity used for vertex connection
    float dVM;                // MIS quantity used for vertex merging
    uint mLocalDirectionIn;
};

// 48 bytes
struct DirectIlluminationReservoir {
    float4 mRnd;
	// source domain location
    float3 mLocalPosition;
    uint mInstancePrimitiveIndex;
    float M;
    float mIntegrationWeight;
    float mCachedTargetPdf;
	float pad;
};
// pad to 128 bytes
struct LVCReservoir {
    PackedVcmVertex mLightVertex;
    float4 pad1;
    PackedVcmVertex mCameraVertex;
    float M;
	float mIntegrationWeight;
	float mCachedTargetPdf;
    float pad;
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

enum VcmReservoirFlags {
	eNone      = 0,
	eRIS       = BIT(0),
	eReuse     = BIT(1)
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