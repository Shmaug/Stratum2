#ifndef gDebugMode
#define gDebugMode 0
#endif
#ifndef gFilterKernelType
#define gFilterKernelType 1u
#endif

struct PushConstants {
	uint mViewCount;
	float mSigmaLuminanceBoost;
	uint mIteration;
	int mStepSize;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> gPushConstants;

#include "compat/denoiser.h"
#include "compat/filter_type.h"

ParameterBlock<DenoiserParameters> gParams;


#define gInput gParams.mFilterImages[gPushConstants.mIteration%2]
#define gOutput gParams.mFilterImages[(gPushConstants.mIteration+1)%2]

struct TapData {
	uint view_index;
	int2 index;

	float3 center_normal;
	float z_center;
	float2 dz_center;
	float l_center;

	float sigma_l;

	float4 sum_color;
	float sum_weight;

	SLANG_MUTATING
	void compute_sigma_luminance() {
		const float kernel[2][2] = {
			{ 1.0 / 4.0, 1.0 / 8.0  },
			{ 1.0 / 8.0, 1.0 / 16.0 }
		};
		float s = sum_color.a*kernel[1][1];
		for (int yy = -1; yy <= 1; yy++)
			for (int xx = -1; xx <= 1; xx++) {
				if (xx == 0 && yy == 0) continue;
				const int2 p = index + int2(xx, yy);
				if (!gParams.mViews[view_index].isInside(p)) continue;
				s += gInput[p].a * kernel[abs(xx)][abs(yy)];
			}
		sigma_l = sqrt(max(s, 0))*gPushConstants.mSigmaLuminanceBoost;
	}

	SLANG_MUTATING
	void tap(const int2 offset, const float kernel_weight) {
		const int2 p = index + offset;
		if (!gParams.mViews[view_index].isInside(p)) return;

		const float4 color_p  = gInput[p];
		const float l_p = luminance(color_p.rgb);
		const float w_l = abs(l_p - l_center) / max(sigma_l, 1e-10);

		const VisibilityData vis_p = reinterpret<VisibilityData>(gParams.mVisibility[p]);
		const DepthData depth_p = reinterpret<DepthData>(gParams.mDepth[p]);
		const float w_z = abs(depth_p.mDepth - z_center) / (length(dz_center * float2(offset * gPushConstants.mStepSize)) + 1e-2);
		const float w_n = pow(max(0, dot(vis_p.normal(), center_normal)), 256);

		const float w = exp(-pow2(w_l) - w_z) * kernel_weight * w_n;
		if (isinf(w) || isnan(w)) return;

		sum_color  += color_p * float4(w, w, w, w*w);
		sum_weight += w;
	}
};

void subsampled(inout TapData t) {
	/*
	| | |x| | |
	| |x| |x| |
	|x| |x| |x|
	| |x| |x| |
	| | |x| | |
	*/

	if ((gPushConstants.mIteration & 1) == 0) {
		/*
		| | | | | |
		| |x| |x| |
		|x| |x| |x|
		| |x| |x| |
		| | | | | |
		*/
		t.tap(int2(-2,  0) * gPushConstants.mStepSize, 1.0);
		t.tap(int2( 2,  0) * gPushConstants.mStepSize, 1.0);
	} else {
		/*
		| | |x| | |
		| |x| |x| |
		| | |x| | |
		| |x| |x| |
		| | |x| | |
		*/
		t.tap(int2( 0, -2) * gPushConstants.mStepSize, 1.0);
		t.tap(int2( 0,  2) * gPushConstants.mStepSize, 1.0);
	}

	t.tap(int2(-1,  1) * gPushConstants.mStepSize, 1.0);
	t.tap(int2( 1,  1) * gPushConstants.mStepSize, 1.0);

	t.tap(int2(-1, -1) * gPushConstants.mStepSize, 1.0);
	t.tap(int2( 1, -1) * gPushConstants.mStepSize, 1.0);
}

void box3(inout TapData t) {
	const int r = 1;
	for (int yy = -r; yy <= r; yy++)
		for (int xx = -r; xx <= r; xx++)
			if (xx != 0 || yy != 0)
				t.tap(int2(xx, yy) * gPushConstants.mStepSize, 1.0);
}

void box5(inout TapData t) {
	const int r = 2;
	for(int yy = -r; yy <= r; yy++)
		for(int xx = -r; xx <= r; xx++)
			if(xx != 0 || yy != 0)
				t.tap(int2(xx, yy) * gPushConstants.mStepSize, 1.0);
}

void atrous(inout TapData t) {
	const float kernel[3] = { 1.0, 2.0 / 3.0, 1.0 / 6.0 };

	t.tap(int2( 1,  0) * gPushConstants.mStepSize, 2.0 / 3.0);
	t.tap(int2( 0,  1) * gPushConstants.mStepSize, 2.0 / 3.0);
	t.tap(int2(-1,  0) * gPushConstants.mStepSize, 2.0 / 3.0);
	t.tap(int2( 0, -1) * gPushConstants.mStepSize, 2.0 / 3.0);

	t.tap(int2( 2,  0) * gPushConstants.mStepSize, 1.0 / 6.0);
	t.tap(int2( 0,  2) * gPushConstants.mStepSize, 1.0 / 6.0);
	t.tap(int2(-2,  0) * gPushConstants.mStepSize, 1.0 / 6.0);
	t.tap(int2( 0, -2) * gPushConstants.mStepSize, 1.0 / 6.0);

	t.tap(int2( 1,  1) * gPushConstants.mStepSize, 4.0 / 9.0);
	t.tap(int2(-1,  1) * gPushConstants.mStepSize, 4.0 / 9.0);
	t.tap(int2(-1, -1) * gPushConstants.mStepSize, 4.0 / 9.0);
	t.tap(int2( 1, -1) * gPushConstants.mStepSize, 4.0 / 9.0);

	t.tap(int2( 1,  2) * gPushConstants.mStepSize, 1.0 / 9.0);
	t.tap(int2(-1,  2) * gPushConstants.mStepSize, 1.0 / 9.0);
	t.tap(int2(-1, -2) * gPushConstants.mStepSize, 1.0 / 9.0);
	t.tap(int2( 1, -2) * gPushConstants.mStepSize, 1.0 / 9.0);

	t.tap(int2( 2,  1) * gPushConstants.mStepSize, 1.0 / 9.0);
	t.tap(int2(-2,  1) * gPushConstants.mStepSize, 1.0 / 9.0);
	t.tap(int2(-2, -1) * gPushConstants.mStepSize, 1.0 / 9.0);
	t.tap(int2( 2, -1) * gPushConstants.mStepSize, 1.0 / 9.0);

	t.tap(int2( 2,  2) * gPushConstants.mStepSize, 1.0 / 36.0);
	t.tap(int2(-2,  2) * gPushConstants.mStepSize, 1.0 / 36.0);
	t.tap(int2(-2, -2) * gPushConstants.mStepSize, 1.0 / 36.0);
	t.tap(int2( 2, -2) * gPushConstants.mStepSize, 1.0 / 36.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 index: SV_DispatchThreadId) {
    uint2 extent;
    gParams.mVisibility.GetDimensions(extent.x, extent.y);
    if (any(index.xy >= extent)) return;

	TapData t;
    t.view_index = getViewIndex(index.xy, extent, gPushConstants.mViewCount);
	if (t.view_index == -1) return;

	const VisibilityData vis = reinterpret<VisibilityData>(gParams.mVisibility[index.xy]);
	const DepthData depth    = reinterpret<DepthData>(gParams.mDepth[index.xy]);
	t.index = (int2)index.xy;
	t.center_normal = vis.normal();
	t.z_center = depth.mDepth;
	t.dz_center = depth.mDepthDerivative;
	t.sum_weight = 1;
	t.sum_color = gInput[t.index];
	t.l_center = luminance(t.sum_color.rgb);

	t.compute_sigma_luminance();

	if (!isinf(t.z_center)) { // only filter foreground pixels
		switch (gFilterKernelType) {
		case FilterKernelType::eAtrous:
			atrous(t);
			break;
		default:
		case FilterKernelType::eBox3:
			box3(t);
			break;
		case FilterKernelType::eBox5:
			box5(t);
			break;
		case FilterKernelType::eSubsampled:
			subsampled(t);
			break;
		case FilterKernelType::eBox3Subsampled:
			if (gPushConstants.mStepSize == 1)
				box3(t);
			else
				subsampled(t);
			break;
		case FilterKernelType::eBox5Subsampled:
			if (gPushConstants.mStepSize == 1)
				box5(t);
			else
				subsampled(t);
			break;
		}
	}

	const float inv_w = 1/t.sum_weight;
	gOutput[t.index] = t.sum_color*float4(inv_w, inv_w, inv_w, pow2(inv_w));
}

[shader("compute")]
[numthreads(8,8,1)]
void copy_rgb(uint3 index : SV_DispatchThreadID) {
	uint2 resolution;
	gParams.mAccumColor.GetDimensions(resolution.x, resolution.y);
	if (any(index.xy >= resolution)) return;
	gParams.mAccumColor[index.xy] = float4(gParams.mFilterImages[0][index.xy].rgb, gParams.mAccumColor[index.xy].w);
}
