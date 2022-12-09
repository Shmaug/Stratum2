#ifndef gReprojection
#define gReprojection false
#endif
#ifndef gDemodulateAlbedo
#define gDemodulateAlbedo false
#endif
#ifndef gCheckNormal
#define gCheckNormal false
#endif
#ifndef gCheckDepth
#define gCheckDepth false
#endif
#ifndef gDebugMode
#define gDebugMode (DenoiserDebugMode)0
#endif

struct PushConstants {
	uint mViewCount;
	uint mHistoryLimit;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> gPushConstants;

#include "compat/denoiser.h"

ParameterBlock<DenoiserParameters> gParams;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 index: SV_DispatchThreadId) {
    uint2 extent;
    gParams.mVisibility.GetDimensions(extent.x, extent.y);
    if (any(index.xy >= extent)) return;

    const uint view_index = getViewIndex(index.xy, extent, gPushConstants.mViewCount);

	const uint2 ipos = index.xy;

	float4 color_prev   = 0;
	float2 moments_prev = 0;
	float sum_w         = 0;
	if (gReprojection) {
		const VisibilityData vis = reinterpret<VisibilityData>(gParams.mVisibility[ipos]);
		const DepthData depth    = reinterpret<DepthData>(gParams.mDepth[ipos]);
		if (vis.instanceIndex() != INVALID_INSTANCE) {
			const float2 pos_prev = gParams.mViews[view_index].mImageMin + gParams.mPrevUVs[ipos] * float2(gParams.mViews[view_index].mImageMax - gParams.mViews[view_index].mImageMin) - 0.5;
			const int2 p = int2(pos_prev);
			const float2 w = frac(pos_prev);
			// bilinear interpolation, check each tap individually, renormalize afterwards
			for (int yy = 0; yy <= 1; yy++) {
				for (int xx = 0; xx <= 1; xx++) {
					const int2 ipos_prev = p + int2(xx, yy);
					if (!gParams.mViews[view_index].isInside(ipos_prev)) continue;

                    if (gCheckNormal || gCheckDepth) {
						const VisibilityData prev_vis = reinterpret<VisibilityData>(gParams.mPrevVisibility[ipos_prev]);
						if (gParams.mInstanceIndexMap[vis.instanceIndex()] != prev_vis.instanceIndex()) continue;
						if (gCheckNormal && dot(vis.normal(), prev_vis.normal()) < cos(degrees(2))) continue;
						if (gCheckDepth && abs(depth.mPrevDepth - reinterpret<DepthData>(gParams.mPrevDepth[ipos_prev]).mDepth) >= 2 * length(depth.mDepthDerivative)) continue;
                    }

					const float4 c = gParams.mPrevAccumColor[ipos_prev];

					if (c.a <= 0 || any(isnan(c)) || any(isinf(c)) || any(c != c)) continue;

					float wc = (xx == 0 ? (1 - w.x) : w.x) * (yy == 0 ? (1 - w.y) : w.y);
					//wc *= wc;
					color_prev   += c * wc;
					moments_prev += gParams.mPrevAccumMoments[ipos_prev] * wc;
					sum_w        += wc;
				}
			}
		}
	} else {
		color_prev = gParams.mPrevAccumColor[ipos];
		if (any(isnan(color_prev.rgb)) || any(isinf(color_prev.rgb)))
			color_prev = 0;
		else
			moments_prev = gParams.mPrevAccumMoments[ipos];
		sum_w = 1;
	}

	float4 color_curr = gParams.mInput[ipos];
	if (gDemodulateAlbedo) color_curr.rgb /= (1e-2 + gParams.mAlbedo[ipos].rgb);

	if (any(isinf(color_curr.rgb)) || any(color_curr.rgb != color_curr.rgb)) color_curr = 0;
	if (any(isinf(moments_prev)) || any(moments_prev != moments_prev)) moments_prev = 0;

	const float l = luminance(color_curr.rgb);

	if (sum_w > 0 && color_prev.a > 0) {
		const float invSum = 1/sum_w;
		color_prev   *= invSum;
		moments_prev *= invSum;

		float n = color_prev.a + color_curr.a;
		if (gPushConstants.mHistoryLimit > 0 && n > gPushConstants.mHistoryLimit)
			n = gPushConstants.mHistoryLimit;

		const float alpha = saturate(color_curr.a / n);

		gParams.mAccumColor[ipos] = float4(lerp(color_prev.rgb, color_curr.rgb, alpha), n);
		gParams.mAccumMoments[ipos] = lerp(moments_prev, float2(l, l*l), alpha);

        if (gDebugMode == DenoiserDebugMode::eSampleCount) gParams.mAccumColor[ipos] = float4(viridisQuintic(saturate(n / (gPushConstants.mHistoryLimit > 0 ? gPushConstants.mHistoryLimit : 1024))), 1);
	} else {
		gParams.mAccumColor[ipos] = color_curr;
        gParams.mAccumMoments[ipos] = float2(l, l * l);

        if (gDebugMode == DenoiserDebugMode::eSampleCount) gParams.mAccumColor[ipos] = float4(viridisQuintic(0), 1);
	}

    if (gDebugMode == DenoiserDebugMode::eWeightSum)
        gParams.mAccumColor[ipos] = float4(viridisQuintic(sum_w), 1);
	else if (gDebugMode == DenoiserDebugMode::eVariance) {
        const float2 m = gParams.mAccumMoments[ipos];
        gParams.mAccumColor[ipos] = float4(viridisQuintic(saturate(abs(m.y - pow2(m.x)))), 1);
	}
}