/*
Copyright (c) 2018, Christoph Schied
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Karlsruhe Institute of Technology nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#if 0
#pragma compile slangc -profile sm_6_6 -lang slang -entry main
#endif

#include "../compat/denoiser.h"

ParameterBlock<DenoiserParameters> gParams;

#ifndef gDebugMode
#define gDebugMode 0
#endif

struct PushConstants {
	uint gViewCount;
	float gHistoryLimit;
	float gVarianceBoostLength;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> gPushConstants;

SLANG_SHADER("compute")
[numthreads(8,8,1)]
void main(uint3 index : SV_DispatchThreadId, uint3 group_index : SV_GroupThreadID) {
	const uint view_index = get_view_index(index.xy, gParams.gViews, gPushConstants.gViewCount);
	if (view_index == -1) return;

	uint2 extent;
	gParams.gFilterImages[0].GetDimensions(extent.x, extent.y);
	const uint index_1d = index.y*extent.x + index.x;

	float4 c = gParams.gAccumColor[index.xy];
	float2 m = gParams.gAccumMoments[index.xy];

	const VisibilityInfo vis = gParams.gVisibility[index_1d];

	const float histlen = c.a;
	if (vis.instance_index() == INVALID_INSTANCE || histlen >= gPushConstants.gHistoryLimit) {
		gParams.gFilterImages[0][index.xy] = float4(c.rgb, abs(m.y - pow2(m.x)));
		return;
	}

	const DepthInfo depth = gParams.gDepth[index_1d];

	float sum_w = 1;

	const int r = histlen > 1 ? 2 : 3;
	for (int yy = -r; yy <= r; yy++)
		for (int xx = -r; xx <= r; xx++) {
			if (xx == 0 && yy == 0) continue;

			const int2 p = int2(index.xy) + int2(xx, yy);
			if (!gParams.gViews[view_index].test_inside(p)) continue;

			const VisibilityInfo vis_p = gParams.gVisibility[p.y*extent.x + p.x];
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