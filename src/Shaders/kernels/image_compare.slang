#include "compat/common.h"
#include "compat/image_compare.h"

#ifndef gMode
#define gMode 0
#endif

struct PushConstants {
    uint2 mExtent;
    float mQuantization;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> gPushConstants;

Texture2D<float4> gImage1;
Texture2D<float4> gImage2;
RWStructuredBuffer<uint> gOutput;

[shader("compute")]
[numthreads(8,8,1)]
void main(uint3 index: SV_DispatchThreadID) {
    if (any(index.xy >= gPushConstants.mExtent)) return;

	const float3 c1 = gImage1[index.xy].rgb;
	const float3 c2 = gImage2[index.xy].rgb;
    const float3 dc = abs(c1 - c2);

    float3 error3 = 0;
    if ((ImageCompareMode)gMode == ImageCompareMode::eSMAPE) {
		error3 = dc / (abs(c1) + abs(c2));
    } else if ((ImageCompareMode)gMode == ImageCompareMode::eMSE) {
		error3 = pow2(dc);
	} else {
		error3 = dc;
    }

    float error = dot(1, error3);

    error = WaveActiveSum(error);

    if (WaveIsFirstLane()) {
        const float valf = error * gPushConstants.mQuantization;
		const uint val = (uint)valf;
		uint prev;
		InterlockedAdd(gOutput[0], val, prev);
		if (0xFFFFFFFF - prev < val) {
			InterlockedOr(gOutput[1], 1);
		}
    }
}