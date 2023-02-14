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

ParameterBlock<SceneParameters> gScene;

uint getViewIndex(const uint2 index, const uint2 extent, const uint viewCount) {
	return 0;
    // assume views are stacked horizontally
    // return index.x / (extent.x / gPushConstants.mViewCount);
}