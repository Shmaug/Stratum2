struct SceneParameters {
	RaytracingAccelerationStructure gAccelerationStructure;
	StructuredBuffer<PackedVertexData> gVertices;
	ByteAddressBuffer gIndices;
	StructuredBuffer<InstanceData> gInstances;
	StructuredBuffer<TransformData> gInstanceTransforms;
	StructuredBuffer<TransformData> gInstanceInverseTransforms;
	StructuredBuffer<TransformData> gInstanceMotionTransforms;
	ByteAddressBuffer gMaterialData;
	StructuredBuffer<uint> gLightInstances; // gLightInstances[light index] -> instance index
	SamplerState gStaticSampler;
	RWStructuredBuffer<uint> gPerformanceCounters;
	StructuredBuffer<uint> gVolumes[gVolumeCount];
	Texture2D<float4> gImages[gImageCount];
	Texture2D<float> gImage1s[gImageCount];
};

inline uint get_view_index(const uint2 index, StructuredBuffer<ViewData> views, const uint viewCount) {
	for (uint i = 0; i < viewCount; i++)
		if (all(index >= views[i].image_min) && all(index < views[i].image_max))
			return i;
	return -1;
}

inline uint3 load_tri_(ByteAddressBuffer indices, uint indexByteOffset, uint indexStride, uint primitiveIndex) {
	const int offsetBytes = (int)(indexByteOffset + primitiveIndex*3*indexStride);
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
inline uint3 load_tri(ByteAddressBuffer indices, const InstanceData instance, uint primitiveIndex) {
	return instance.first_vertex() + load_tri_(indices, instance.indices_byte_offset(), instance.index_stride(), primitiveIndex);
}
