struct SceneParameters {
	RaytracingAccelerationStructure mAccelerationStructure;

	ByteAddressBuffer mMaterialData;

	StructuredBuffer<PackedVertexData> mVertices;
	ByteAddressBuffer mIndices;

	StructuredBuffer<InstanceData> mInstances;

	StructuredBuffer<TransformData> mInstanceTransforms;
	StructuredBuffer<TransformData> mInstanceInverseTransforms;
	StructuredBuffer<TransformData> mInstanceMotionTransforms;

	StructuredBuffer<uint> mLightInstances; // gLightInstances[light] -> instance

	RWStructuredBuffer<uint> mPerformanceCounters;

	SamplerState mStaticSampler;

	StructuredBuffer<uint> mVolumes[gVolumeCount];
	Texture2D<float4> mImages[gImageCount];
	Texture2D<float> mImage1s[gImageCount];
};