#ifndef gReprojection
#define gReprojection false
#endif
#ifndef gDemodulateAlbedo
#define gDemodulateAlbedo false
#endif
#ifndef gDebugMode
#define gDebugMode 0
#endif

#include "../compat/denoiser.h"

ParameterBlock<DenoiserParameters> gParams;

struct PushConstants {
	uint gViewCount;
	float gHistoryLimit;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> gPushConstants;

SLANG_SHADER("compute")
[numthreads(8,8,1)]
void main(uint3 index : SV_DispatchThreadId) {
	const uint view_index = get_view_index(index.xy, gParams.gViews, gPushConstants.gViewCount);
	if (view_index == -1) return;

	uint2 extent;
	gParams.gAccumColor.GetDimensions(extent.x, extent.y);

	const uint2 ipos = index.xy;

	float4 color_prev   = 0;
	float2 moments_prev = 0;
	float sum_w         = 0;
	if (gReprojection) {
		const VisibilityInfo vis = gParams.gVisibility[ipos.y*extent.x + ipos.x];
		const DepthInfo depth    = gParams.gDepth[ipos.y*extent.x + ipos.x];
		if (vis.instance_index() != INVALID_INSTANCE) {
			const float2 pos_prev = gParams.gViews[view_index].image_min + gParams.gPrevUVs[ipos] * float2(gParams.gViews[view_index].image_max - gParams.gViews[view_index].image_min) - 0.5;
			const int2 p = int2(pos_prev);
			const float2 w = frac(pos_prev);
			// bilinear interpolation, check each tap individually, renormalize afterwards
			for (int yy = 0; yy <= 1; yy++) {
				for (int xx = 0; xx <= 1; xx++) {
					const int2 ipos_prev = p + int2(xx, yy);
					if (!gParams.gViews[view_index].test_inside(ipos_prev)) continue;

					const VisibilityInfo prev_vis = gParams.gPrevVisibility[ipos_prev.y*extent.x + ipos_prev.x];
					if (gParams.gInstanceIndexMap[vis.instance_index()] != prev_vis.instance_index()) continue;
					if (dot(vis.normal(), prev_vis.normal()) < cos(degrees(2))) continue;
					if (abs(depth.prev_z - gParams.gPrevDepth[ipos_prev.y*extent.x + ipos_prev.x].z) >= 1.5*length(depth.dz_dxy)) continue;

					const float4 c = gParams.gPrevAccumColor[ipos_prev];

					if (c.a <= 0 || any(isnan(c)) || any(isinf(c)) || any(c != c)) continue;

					float wc = (xx == 0 ? (1 - w.x) : w.x) * (yy == 0 ? (1 - w.y) : w.y);
					//wc *= wc;
					color_prev   += c * wc;
					moments_prev += gParams.gPrevAccumMoments[ipos_prev] * wc;
					sum_w        += wc;
				}
			}
		}
	} else {
		color_prev = gParams.gPrevAccumColor[ipos];
		if (any(isnan(color_prev.rgb)) || any(isinf(color_prev.rgb)))
			color_prev = 0;
		else
			moments_prev = gParams.gPrevAccumMoments[ipos];
		sum_w = 1;
	}

	float4 color_curr = gParams.gRadiance[ipos];
	if (gDemodulateAlbedo) color_curr.rgb /= (1e-2 + gParams.gAlbedo[ipos].rgb);

	if (any(isinf(color_curr.rgb)) || any(color_curr.rgb != color_curr.rgb)) color_curr = 0;
	if (any(isinf(moments_prev)) || any(moments_prev != moments_prev)) moments_prev = 0;

	const float l = luminance(color_curr.rgb);

	if (sum_w > 0 && color_prev.a > 0) {
		const float invSum = 1/sum_w;
		color_prev   *= invSum;
		moments_prev *= invSum;

		float n = color_prev.a + color_curr.a;
		if (gPushConstants.gHistoryLimit > 0 && n > gPushConstants.gHistoryLimit)
			n = gPushConstants.gHistoryLimit;

		const float alpha = saturate(color_curr.a / n);

		gParams.gAccumColor[ipos] = float4(lerp(color_prev.rgb, color_curr.rgb, alpha), n);
		gParams.gAccumMoments[ipos] = lerp(moments_prev, float2(l, l*l), alpha);

		if ((DenoiserDebugMode)gDebugMode == DenoiserDebugMode::eSampleCount) gParams.gDebugImage[ipos] = float4(viridis_quintic(saturate(n / (gPushConstants.gHistoryLimit > 0 ? gPushConstants.gHistoryLimit : 1024))), 1);
	} else {
		gParams.gAccumColor[ipos] = color_curr;
		gParams.gAccumMoments[ipos] = float2(l, l*l);
		if ((DenoiserDebugMode)gDebugMode == DenoiserDebugMode::eSampleCount) gParams.gDebugImage[ipos] = float4(viridis_quintic(0), 1);
	}

	if ((DenoiserDebugMode)gDebugMode == DenoiserDebugMode::eWeightSum)
		gParams.gDebugImage[ipos] = float4(viridis_quintic(sum_w), 1);
	else if ((DenoiserDebugMode)gDebugMode == DenoiserDebugMode::eVariance) {
		const float2 m = gParams.gAccumMoments[ipos];
		gParams.gDebugImage[ipos] = float4(viridis_quintic(saturate(abs(m.y - pow2(m.x)))), 1);
	}
}