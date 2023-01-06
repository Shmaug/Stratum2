#pragma once

#include "compat/scene.h"

struct SceneParameters {
	RaytracingAccelerationStructure mAccelerationStructure;

	StructuredBuffer<InstanceData> mInstances;

	StructuredBuffer<TransformData> mInstanceTransforms;
	StructuredBuffer<TransformData> mInstanceInverseTransforms;
	StructuredBuffer<TransformData> mInstanceMotionTransforms;

    StructuredBuffer<uint> mLightInstanceMap; // light index -> instance index
    StructuredBuffer<uint> mInstanceLightMap; // instance index -> light index

	ByteAddressBuffer mMaterialData;
	ByteAddressBuffer mVertexBuffers[gVertexBufferCount];
	StructuredBuffer<MeshVertexInfo> mMeshVertexInfo;

	RWStructuredBuffer<uint> mPerformanceCounters;

	Texture2D<float4> mImages[gImageCount];
	Texture2D<float> mImage1s[gImageCount];
	StructuredBuffer<uint> mVolumes[gVolumeCount];

	SamplerState mStaticSampler;
};

uint getViewIndex(const uint2 index, const uint2 extent, const uint viewCount) {
	return 0;
    // assume views are stacked horizontally
    // return index.x / (extent.x / gPushConstants.mViewCount);
}

#define SHADING_FLAG_FLIP_BITANGENT BIT(0)

extension ShadingData {
	property bool isSurface       { get { return mShapeArea > 0; } }
	property bool isMedium        { get { return mShapeArea == 0; } }
	property bool isEnvironment   { get { return mShapeArea < 0; } }
	property uint materialAddress { get { return BF_GET(mFlagsMaterialAddress, 4, 28); } }

	property bool isBitangentFlipped { get { return (bool)(mFlagsMaterialAddress & SHADING_FLAG_FLIP_BITANGENT); } }
	property int bitangentDirection  { get { return isBitangentFlipped ? -1 : 1; } }

	property float3 geometryNormal   { get { return unpackNormal(mPackedGeometryNormal); } }
	property float3 shadingNormal    { get { return unpackNormal(mPackedShadingNormal); } }
	property float3 tangent          { get { return unpackNormal(mPackedTangent); } }

	float3 toWorld(const float3 v) {
		const float3 n = shadingNormal;
		const float3 t = tangent;
		return v.x*t + v.y*cross(n, t)*bitangentDirection + v.z*n;
	}
	float3 toLocal(const float3 v) {
		const float3 n = shadingNormal;
		const float3 t = tangent;
		return float3(dot(v, t), dot(v, cross(n, t)*bitangentDirection), dot(v, n));
	}
};