#include "compat/scene.h"

struct PushConstants {
	uint mViewCount;
	uint mRandomSeed;

	uint mMinBounces;
	uint mMaxBounces;
	uint mMaxDiffuseBounces;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> gPushConstants;

struct RenderParams {
	StructuredBuffer<ViewData> mViews;
	StructuredBuffer<TransformData> mViewTransforms;
	RWTexture2D<float4> mOutput;

	inline uint getViewIndex(const uint2 index) {
		for (uint i = 0; i < gPushConstants.mViewCount; i++)
			if (all(index >= mViews[i].mImageMin) && all(index < mViews[i].mImageMax))
				return i;
		return -1;
	}
};
ParameterBlock<RenderParams> gRenderParams;


[shader("compute")]
[numthreads(8, 8, 1)]
void render(uint3 index : SV_DispatchThreadID, uint group_index : SV_GroupIndex) {
	const uint viewIndex = gRenderParams.getViewIndex(index.xy);
	if (viewIndex == -1) return;

	float2 uv;
	const float3 localDir = gRenderParams.mViews[viewIndex].toWorld(index.xy + 0.5, /*out*/ uv);
	const TransformData t = gRenderParams.mViewTransforms[viewIndex];
	float3 rayDirection = normalize(t.transformVector(localDir));
	float3 rayOrigin = float3(t.m[0][3], t.m[1][3], t.m[2][3] );
    gRenderParams.mOutput[index.xy] = float4(rayDirection*.5+.5, 1);
}