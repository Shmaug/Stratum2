#include "compat/common.h"
#include "compat/image_compare.h"

#ifndef gMode
#define gMode 0
#endif
#ifndef gQuantization
#define gQuantization 16777216
#endif

#ifdef COPY_KERNEL
Texture2D<float4> gInput;
RWTexture2D<float4> gOutput;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 index: SV_DispatchThreadID) {
    uint2 resolution;
    gOutput.GetDimensions(resolution.x, resolution.y);
    if (any(index.xy >= resolution)) return;

	float4 color = gInput[index.xy];
#ifdef CONVERT_SRGB
	color.rgb = rgbToSrgb(color.rgb);
#endif
    gOutput[index.xy] = color;
}

#else

Texture2D<float4> gImage1;
Texture2D<float4> gImage2;
RWStructuredBuffer<uint> gOutput;

[shader("compute")]
[numthreads(8,8,1)]
void main(uint3 index : SV_DispatchThreadID) {
	uint2 resolution;
	gImage1.GetDimensions(resolution.x, resolution.y);
	if (any(index.xy >= resolution)) return;

	const float3 c1 = gImage1[index.xy].rgb;
	const float3 c2 = gImage2[index.xy].rgb;

	float error = 0;
    switch (gMode) {
    case (uint)ImageCompareMode::eSMAPE:
		error = dot(1, abs(c1 - c2) / (abs(c1) + abs(c2)));
        break;
    case (uint)ImageCompareMode::eMSE:
		error = dot(1, pow2(c1 - c2));
		break;
    default:
    case (uint)ImageCompareMode::eAverage:
		error = dot(1, c1 - c2);
		break;
	}
	error /= 3*resolution.x*resolution.y;

	error = WaveActiveSum(error);

	if (WaveIsFirstLane()) {
		uint prev;
		const float valf = error*gQuantization;
		const uint val = (uint)valf;
		InterlockedAdd(gOutput[0], val, prev);
		if (valf >= 0xFFFFFFFF || 0xFFFFFFFF - prev < val) {
			gOutput[1] = 1;
		}
    }
}
#endif