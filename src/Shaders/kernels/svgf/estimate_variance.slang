struct PushConstants {
	uint mViewCount;
	float mHistoryLimit;
	float mVarianceBoostLength;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> gPushConstants;

#include "compat/denoiser.h"

ParameterBlock<DenoiserParameters> gParams;

#ifndef gDebugMode
#define gDebugMode 0
#endif

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 index: SV_DispatchThreadId, uint3 group_index: SV_GroupThreadID) {
    uint2 extent;
    gParams.mVisibility.GetDimensions(extent.x, extent.y);
    if (any(index.xy >= extent)) return;

	const uint view_index = getViewIndex(index.xy, extent, gPushConstants.mViewCount);

	float4 c = gParams.mAccumColor[index.xy];
	float2 m = gParams.mAccumMoments[index.xy];

	const VisibilityData vis = reinterpret<VisibilityData>(gParams.mVisibility[index.xy]);

	const float histlen = c.a;
	if (vis.instanceIndex() == INVALID_INSTANCE || histlen >= gPushConstants.mHistoryLimit) {
		gParams.mFilterImages[0][index.xy] = float4(c.rgb, abs(m.y - pow2(m.x)));
		return;
	}

	const DepthData depth = reinterpret<DepthData>(gParams.mDepth[index.xy]);

	float sum_w = 1;

	const int r = histlen > 1 ? 2 : 3;
	for (int yy = -r; yy <= r; yy++)
		for (int xx = -r; xx <= r; xx++) {
			if (xx == 0 && yy == 0) continue;

			const int2 p = int2(index.xy) + int2(xx, yy);
			if (!gParams.mViews[view_index].isInside(p)) continue;

            const VisibilityData vis_p = reinterpret<VisibilityData>(gParams.mVisibility[p]);
            if (gParams.mInstanceIndexMap[vis.instanceIndex()] != vis_p.instanceIndex()) continue;

			const float w_z = abs(reinterpret<DepthData>(gParams.mDepth[p]).mDepth - depth.mDepth) / (length(depth.mDepthDerivative * float2(xx, yy)) + 1e-2);
			const float w_n = pow(saturate(dot(vis_p.normal(), vis.normal())), 128);
			const float w = exp(-w_z) * w_n;
			if (isnan(w) || isinf(w)) continue;

			m += gParams.mAccumMoments[p] * w;
			c.rgb += gParams.mAccumColor[p].rgb * w;
			sum_w += w;
		}

	sum_w = 1/sum_w;
	m *= sum_w;
	c.rgb *= sum_w;

	float v = abs(m.y - pow2(m.x));
	if (gPushConstants.mVarianceBoostLength > 0)
		v *= max(1, gPushConstants.mVarianceBoostLength/(1+c.a));
	gParams.mFilterImages[0][index.xy] = float4(c.rgb, v);
}