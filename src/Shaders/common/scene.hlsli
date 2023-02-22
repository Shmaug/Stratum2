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

	// returns emission
    float3 SampleEnvironment(const uint address, const float2 rnd, out float3 dirOut, out float2 uv, out float pdf) {
        if (address == -1) {
            pdf = 0;
            return 0;
        }

        const uint4 packedData = mMaterialData.Load<uint4>((int)address);
        float3 emission = asfloat(packedData.rgb);
        const uint environmentImage = packedData.w;

        if (environmentImage < gImageCount) {
            uv = sampleTexel(mImages[environmentImage], rnd, pdf);
            emission *= mImages[environmentImage].SampleLevel(mStaticSampler, uv, 0).rgb;
            dirOut = sphericalUvToCartesian(uv);
            pdf /= (2 * M_PI * M_PI * sqrt(1 - dirOut.y * dirOut.y));
        } else {
            dirOut = sampleUniformSphere(rnd.x, rnd.y);
            uv = cartesianToSphericalUv(dirOut);
            pdf = 1 / (4 * M_PI);
        }
		return emission;
    }

    // returns emission
    float3 EvaluateEnvironment(const uint address, const float3 direction, out float pdfW) {
        if (address == -1) {
            pdfW = 0;
            return 0;
        }

        const uint4 packedData = mMaterialData.Load<uint4>((int)address);
        float3 emission = asfloat(packedData.rgb);
        const uint environmentImage = packedData.w;

        if (environmentImage < gImageCount) {
            const float2 uv = cartesianToSphericalUv(direction);
            emission *= mImages[environmentImage].SampleLevel(mStaticSampler, uv, 0).rgb;
            pdfW = sampleTexelPdf(mImages[environmentImage], uv) / (2 * M_PI * M_PI * sqrt(1 - direction.y * direction.y));
        } else {
            pdfW = 1 / (4 * M_PI);
        }
        return emission;
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