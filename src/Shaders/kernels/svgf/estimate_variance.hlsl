struct PushConstants {
	uint gViewCount;
	float gHistoryLimit;
	float gVarianceBoostLength;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> gPushConstants;

#include "compat/denoiser.h"

ParameterBlock<DenoiserParameters> gParams;

#ifndef gDebugMode
#define gDebugMode 0
#endif

SLANG_SHADER("compute")
[numthreads(8,8,1)]
void main(uint3 index : SV_DispatchThreadId, uint3 group_index : SV_GroupThreadID) {
	const uint view_index = gParams.getViewIndex(index.xy);
	if (view_index == -1) return;

	uint2 extent;
	gParams.gFilterImages[0].GetDimensions(extent.x, extent.y);
	const uint index_1d = index.y*extent.x + index.x;

	float4 c = gParams.gAccumColor[index.xy];
	float2 m = gParams.gAccumMoments[index.xy];

	const VisibilityData vis = gParams.gVisibility[index_1d];

	const float histlen = c.a;
	if (vis.instance_index() == INVALID_INSTANCE || histlen >= gPushConstants.gHistoryLimit) {
		gParams.gFilterImages[0][index.xy] = float4(c.rgb, abs(m.y - pow2(m.x)));
		return;
	}

	const DepthData depth = gParams.gDepth[index_1d];

	float sum_w = 1;

	const int r = histlen > 1 ? 2 : 3;
	for (int yy = -r; yy <= r; yy++)
		for (int xx = -r; xx <= r; xx++) {
			if (xx == 0 && yy == 0) continue;

			const int2 p = int2(index.xy) + int2(xx, yy);
			if (!gParams.gViews[view_index].test_inside(p)) continue;

			const VisibilityData vis_p = gParams.gVisibility[p.y*extent.x + p.x];
			if (gParams.gInstanceIndexMap[vis.instance_index()] != vis_p.instance_index()) continue;

			const float w_z = abs(gParams.gDepth[p.y*extent.x + p.x].z - depth.z) / (length(depth.dz_dxy * float2(xx, yy)) + 1e-2);
			const float w_n = pow(saturate(dot(vis_p.normal(), vis.normal())), 128);
			const float w = exp(-w_z) * w_n;
			if (isnan(w) || isinf(w)) continue;

			m += gParams.gAccumMoments[p] * w;
			c.rgb += gParams.gAccumColor[p].rgb * w;
			sum_w += w;
		}

	sum_w = 1/sum_w;
	m *= sum_w;
	c.rgb *= sum_w;

	float v = abs(m.y - pow2(m.x));
	if (gPushConstants.gVarianceBoostLength > 0)
		v *= max(1, gPushConstants.gVarianceBoostLength/(1+c.a));
	gParams.gFilterImages[0][index.xy] = float4(c.rgb, v);
}