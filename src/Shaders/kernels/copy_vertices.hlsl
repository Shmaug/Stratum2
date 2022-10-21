#include "../compat/scene.h"

#ifndef gVertexBufferCount
#define gVertexBufferCount 8192
#endif

struct BufferProperties {
	uint mCount;
	uint mPositionStride;
	uint mNormalStride;
	uint mTexcoordStride;
};

ByteAddressBuffer gPositions[gVertexBufferCount];
ByteAddressBuffer gNormals[gVertexBufferCount];
ByteAddressBuffer gTexcoords[gVertexBufferCount];

StructuredBuffer<BufferProperties> gBufferProperties;
RWStructuredBuffer<PackedVertexData> gVertices;


struct PushConstants {
	uint gBufferIndex;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> gPushConstants;

SLANG_SHADER("compute")
[numthreads(64,1,1)]
void main(uint3 index : SV_DispatchThreadId) {
	if (index.x >= gBufferSizes[gPushConstants.gBufferIndex]) return;

	const BufferProperties p = gBufferProperties[gPushConstants.gBufferIndex];

	PackedVertexData v;
	v.set(
		gPositions.Load<float3>(int(index.x*p.gPositionStride)),
		gNormals.Load<float3>(int(index.x*p.gNormalStride)),
		p.gTexcoordStride > 0 ? gTexcoords.Load<float2>(int(index.x*p.gTexcoordStride)) : 0 );
	gVertices[index.x] = v;
}