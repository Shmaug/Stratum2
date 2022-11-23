#include "../compat/scene.h"

#ifndef gVertexBufferCount
#define gVertexBufferCount 8192
#endif

ByteAddressBuffer gPositions[gVertexBufferCount];
ByteAddressBuffer gNormals[gVertexBufferCount];
ByteAddressBuffer gTexcoords[gVertexBufferCount];

StructuredBuffer<uint4> gInfos;
RWStructuredBuffer<PackedVertexData> gVertices;


struct PushConstants {
	uint gBufferIndex;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> gPushConstants;

SLANG_SHADER("compute")
[numthreads(64,1,1)]
void main(uint3 index : SV_DispatchThreadId) {
	const uint4 p = gInfos[gPushConstants.gBufferIndex];
	uint count = p.x;
	uint positionStride = p.y;
	uint normalStride = p.z;
	uint texcoordStride = p.w;

	if (index.x >= count) return;

	gVertices[index.x] = PackedVertexData(
		gPositions[gPushConstants.gBufferIndex].Load<float3>(int(index.x*positionStride)),
		gNormals[gPushConstants.gBufferIndex].Load<float3>(int(index.x*normalStride)),
		texcoordStride > 0 ? gTexcoords[gPushConstants.gBufferIndex].Load<float2>(int(index.x*texcoordStride)) : 0 );
}