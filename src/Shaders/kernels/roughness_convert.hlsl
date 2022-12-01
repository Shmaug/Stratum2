#include "compat/common.h"

Texture2D<float> gInput;
RWTexture2D<float> gRoughnessRW;

SLANG_SHADER("compute")
[numthreads(8,8,1)]
void alpha_to_roughness(uint3 index : SV_DispatchThreadId) {
	uint2 size;
	gRoughnessRW.GetDimensions(size.x, size.y);
	if (any(index.xy >= size)) return;
	gRoughnessRW[index.xy] = saturate(sqrt(gInput[index.xy]));
}

SLANG_SHADER("compute")
[numthreads(8,8,1)]
void shininess_to_roughness(uint3 index : SV_DispatchThreadId) {
	uint2 size;
	gRoughnessRW.GetDimensions(size.x, size.y);
	if (any(index.xy >= size)) return;
	gRoughnessRW[index.xy] = saturate(sqrt(2 / (gInput[index.xy] + 2)));
}