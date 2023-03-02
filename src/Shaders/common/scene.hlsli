#pragma once

#include "compat/scene.h"
#include "compat/material_data.h"

struct SceneParameters {
#ifndef NO_SCENE_ACCELERATION_STRUCTURE
	RaytracingAccelerationStructure mAccelerationStructure;
#endif

	StructuredBuffer<InstanceData> mInstances;
	StructuredBuffer<TransformData> mInstanceTransforms;
	StructuredBuffer<TransformData> mInstanceInverseTransforms;
	StructuredBuffer<TransformData> mInstanceMotionTransforms;

    StructuredBuffer<uint> mLightInstanceMap; // light index -> instance index
    StructuredBuffer<uint> mInstanceLightMap; // instance index -> light index

	ByteAddressBuffer mMaterialData;
	StructuredBuffer<MeshVertexInfo> mMeshVertexInfo;
	StructuredBuffer<VolumeInfo> mInstanceVolumeInfo;

	RWStructuredBuffer<uint> mRayCount;
	SamplerState mStaticSampler;

	ByteAddressBuffer mVertexBuffers[gVertexBufferCount];
    Texture2D<float4> mImages[gImageCount];
    Texture2D<float2> mImage2s[gImageCount];
    Texture2D<float> mImage1s[gImageCount];
	StructuredBuffer<uint> mVolumes[gVolumeCount];

	uint GetMediumIndex(const float3 position, const uint volumeInfoCount) {
		for (uint i = 0; i < volumeInfoCount; i++) {
			const VolumeInfo info = mInstanceVolumeInfo[i];
			const float3 localPos = mInstanceInverseTransforms[info.mInstanceIndex].transformPoint(position);
			if (all(localPos >= info.mMin) && all(localPos <= info.mMax))
				return info.mInstanceIndex;
		}
		return INVALID_INSTANCE;
	}
};

uint3 LoadTriangleIndices(const ByteAddressBuffer indices, const uint offset, const uint indexStride, const uint primitiveIndex) {
    const int offsetBytes = (int)(offset + primitiveIndex * 3 * indexStride);
    uint3 tri;
    if (indexStride == 2) {
        // https://github.com/microsoft/DirectX-Graphics-Samples/blob/master/Samples/Desktop/D3D12Raytracing/src/D3D12RaytracingSimpleLighting/Raytracing.hlsl
        const int dwordAlignedOffset = offsetBytes & ~3;
        const uint2 four16BitIndices = indices.Load2(dwordAlignedOffset);
        if (dwordAlignedOffset == offsetBytes) {
            tri.x = four16BitIndices.x & 0xffff;
            tri.y = (four16BitIndices.x >> 16) & 0xffff;
            tri.z = four16BitIndices.y & 0xffff;
        } else {
            tri.x = (four16BitIndices.x >> 16) & 0xffff;
            tri.y = four16BitIndices.y & 0xffff;
            tri.z = (four16BitIndices.y >> 16) & 0xffff;
        }
    } else
        tri = indices.Load3(offsetBytes);
    return tri;
}
T LoadVertexAttribute<T>(const ByteAddressBuffer vertexBuffer, const uint offset, const uint stride, const uint index) {
    return vertexBuffer.Load<T>(int(offset + stride * index));
}
void LoadTriangleAttribute<T>(const ByteAddressBuffer vertexBuffer, const uint offset, const uint stride, const uint3 tri, out T v0, out T v1, out T v2) {
    v0 = LoadVertexAttribute<T>(vertexBuffer, offset, stride, tri[0]);
    v1 = LoadVertexAttribute<T>(vertexBuffer, offset, stride, tri[1]);
    v2 = LoadVertexAttribute<T>(vertexBuffer, offset, stride, tri[2]);
}

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

    uint3 LoadTriangleIndices(const MeshVertexInfo vertexInfo, const uint primitiveIndex) {
        return LoadTriangleIndices(mVertexBuffers[NonUniformResourceIndex(vertexInfo.indexBuffer())], vertexInfo.indexOffset(), vertexInfo.indexStride(), primitiveIndex);
    }
    uint3 LoadTriangleIndicesUniform(const MeshVertexInfo vertexInfo, const uint primitiveIndex) {
        return LoadTriangleIndices(mVertexBuffers[vertexInfo.indexBuffer()], vertexInfo.indexOffset(), vertexInfo.indexStride(), primitiveIndex);
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