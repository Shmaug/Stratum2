#if 0
#pragma compile slangc -profile sm_6_6 -lang slang -entry main
#endif

#include "../compat/scene.h"

RWStructuredBuffer<PackedVertexData> gVertices;
ByteAddressBuffer gPositions;
ByteAddressBuffer gNormals;
ByteAddressBuffer gTangents;
ByteAddressBuffer gTexcoords;

struct PushConstants {
	uint gCount;
	uint gPositionStride;
	uint gNormalStride;
	uint gTexcoordStride;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> gPushConstants;

SLANG_SHADER("compute")
[numthreads(64,1,1)]
void main(uint3 index : SV_DispatchThreadId) {
	if (index.x >= gPushConstants.gCount) return;
	PackedVertexData v;
	v.set(
		gPositions.Load<float3>(int(index.x*gPushConstants.gPositionStride)),
		gNormals.Load<float3>(int(index.x*gPushConstants.gNormalStride)),
		gPushConstants.gTexcoordStride > 0 ? gTexcoords.Load<float2>(int(index.x*gPushConstants.gTexcoordStride)) : 0 );
	gVertices[index.x] = v;
}