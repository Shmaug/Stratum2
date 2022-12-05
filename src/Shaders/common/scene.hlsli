struct SceneParameters {
	RaytracingAccelerationStructure mAccelerationStructure;

	StructuredBuffer<InstanceData> mInstances;

	StructuredBuffer<TransformData> mInstanceTransforms;
	StructuredBuffer<TransformData> mInstanceInverseTransforms;
	StructuredBuffer<TransformData> mInstanceMotionTransforms;

	StructuredBuffer<uint> mLightInstances; // gLightInstances[light] -> instance

	ByteAddressBuffer mMaterialData;
	ByteAddressBuffer mVertexBuffers[gVertexBufferCount];
	StructuredBuffer<MeshVertexInfo> mMeshVertexInfo;

	RWStructuredBuffer<uint> mPerformanceCounters;

	Texture2D<float4> mImages[gImageCount];
	Texture2D<float> mImage1s[gImageCount];
	StructuredBuffer<uint> mVolumes[gVolumeCount];

	SamplerState mStaticSampler;
};