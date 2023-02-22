#pragma once

#include "compat/scene.h"
#include "compat/material_data.h"

struct SceneParameters {
	RaytracingAccelerationStructure mAccelerationStructure;

	StructuredBuffer<InstanceData> mInstances;
	StructuredBuffer<TransformData> mInstanceTransforms;
	StructuredBuffer<TransformData> mInstanceInverseTransforms;
	StructuredBuffer<TransformData> mInstanceMotionTransforms;

    StructuredBuffer<uint> mLightInstanceMap; // light index -> instance index
    StructuredBuffer<uint> mInstanceLightMap; // instance index -> light index

	ByteAddressBuffer mMaterialData;
	StructuredBuffer<MeshVertexInfo> mMeshVertexInfo;

	RWStructuredBuffer<uint> mPerformanceCounters;
	SamplerState mStaticSampler;

	ByteAddressBuffer mVertexBuffers[gVertexBufferCount];
    Texture2D<float4> mImages[gImageCount];
    Texture2D<float2> mImage2s[gImageCount];
    Texture2D<float> mImage1s[gImageCount];
	StructuredBuffer<uint> mVolumes[gVolumeCount];
};

extension SceneParameters {
	float SampleImage1(const uint imageIndex, const float2 uv, const float uvScreenSize) {
		float lod = 0;
		if (uvScreenSize > 0) {
			float w, h;
			mImage1s[imageIndex].GetDimensions(w, h);
			lod = log2(max(uvScreenSize * max(w, h), 1e-6f));
		}
        return mImage1s[NonUniformResourceIndex(imageIndex)].SampleLevel(mStaticSampler, uv, lod);
    }
    float2 SampleImage2(const uint imageIndex, const float2 uv, const float uvScreenSize) {
        float lod = 0;
        if (uvScreenSize > 0) {
            float w, h;
            mImage2s[imageIndex].GetDimensions(w, h);
            lod = log2(max(uvScreenSize * max(w, h), 1e-6f));
        }
        return mImage2s[NonUniformResourceIndex(imageIndex)].SampleLevel(mStaticSampler, uv, lod);
    }
	float4 SampleImage4(const uint imageIndex, const float2 uv, const float uvScreenSize) {
		float lod = 0;
		if (uvScreenSize > 0) {
			float w, h;
			mImages[imageIndex].GetDimensions(w, h);
			lod = log2(max(uvScreenSize * max(w, h), 1e-6f));
		}
		return mImages[NonUniformResourceIndex(imageIndex)].SampleLevel(mStaticSampler, uv, lod);
    }
}

uint getViewIndex(const uint2 index, const uint2 extent, const uint viewCount) {
	// assume views are stacked horizontally
    // return index.x / (extent.x / gPushConstants.mViewCount);
	return 0;
}

ParameterBlock<SceneParameters> gScene;

struct ReflectanceEvalRecord {
    float3 mReflectance;
    float mFwdPdfW;
    float mRevPdfW;
};
struct DirectionSampleRecord {
    float3 mDirection;
    float3 mReflectance;
    float mFwdPdfW;
    float mRevPdfW;
    float mEta;
    float mRoughness;

    bool isSingular() { return mRoughness == 0; }
};