#if 0
#pragma compile slangc -profile sm_6_6 -lang slang -entry alpha_to_roughness
#pragma compile slangc -profile sm_6_6 -lang slang -entry shininess_to_roughness
#pragma compile slangc -profile sm_6_6 -lang slang -entry from_gltf_pbr
#pragma compile slangc -profile sm_6_6 -lang slang -entry from_diffuse_specular
#endif

#include "../compat/common.h"
#include "../compat/disney_data.h"

// used by **_to_roughness kernels
Texture2D<float> gInput;
RWTexture2D<float> gRoughnessRW;

Texture2D<float4> gDiffuse; // also base color
Texture2D<float4> gSpecular;
Texture2D<float3> gTransmittance;
Texture2D<float> gRoughness;
RWTexture2D<float4> gOutput[DISNEY_DATA_N];
RWTexture2D<float> gOutputAlphaMask;
RWStructuredBuffer<uint> gOutputMinAlpha;

#ifndef gUseDiffuse
#define gUseDiffuse false
#endif
#ifndef gUseSpecular
#define gUseSpecular false
#endif
#ifndef gUseTransmittance
#define gUseTransmittance false
#endif
#ifndef gUseRoughness
#define gUseRoughness false
#endif

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

SLANG_SHADER("compute")
[numthreads(8,8,1)]
void from_gltf_pbr(uint3 index : SV_DispatchThreadId) {
	uint2 size;
	if (gUseDiffuse) gDiffuse.GetDimensions(size.x, size.y);
	else if (gUseSpecular) gDiffuse.GetDimensions(size.x, size.y);
	else if (gUseTransmittance) gTransmittance.GetDimensions(size.x, size.y);
	if (any(index.xy >= size)) return;

	const float4 diffuse = gUseDiffuse ? gDiffuse[index.xy] : 1;
	if (gUseDiffuse) {
		gOutputAlphaMask[index.xy] = diffuse.a;
		if (diffuse.a < 1)
			InterlockedMin(gOutputMinAlpha[0], uint(saturate(diffuse.a)*0xFFFFFFFF));
	}

	const float4 metallic_roughness = gUseSpecular ? gSpecular[index.xy] : 1;

	DisneyMaterialData m;
	for (uint i = 0; i < DISNEY_DATA_N; i++) m.data[i] = 1;

	m.base_color(diffuse.rgb);
	m.metallic(metallic_roughness.b);
	m.roughness(metallic_roughness.g);

	const float l = luminance(m.base_color());
	m.transmission(gUseTransmittance ? saturate(luminance(gTransmittance[index.xy].rgb)/(l > 0 ? l : 1)) : 0);

	for (uint j = 0; j < DISNEY_DATA_N; j++)
		gOutput[j][index.xy] = m.data[j];
}

SLANG_SHADER("compute")
[numthreads(8,8,1)]
void from_diffuse_specular(uint3 index : SV_DispatchThreadId) {
	uint2 size;
	if (gUseDiffuse) gDiffuse.GetDimensions(size.x, size.y);
	else if (gUseSpecular) gDiffuse.GetDimensions(size.x, size.y);
	else if (gUseTransmittance) gTransmittance.GetDimensions(size.x, size.y);
	if (any(index.xy >= size)) return;

	const float4 diffuse = gUseDiffuse ? gDiffuse[index.xy] : 0;
	if (gUseDiffuse) {
		gOutputAlphaMask[index.xy] = diffuse.a;
		if (diffuse.a < 1)
			InterlockedMin(gOutputMinAlpha[0], uint(saturate(diffuse.a)*0xFFFFFFFF));
	}

	DisneyMaterialData m;
	for (uint i = 0; i < DISNEY_DATA_N; i++) m.data[i] = 1;

	const float3 specular = gUseSpecular ? gSpecular[index.xy].rgb : 0;
	const float3 transmittance = gUseTransmittance ? gTransmittance[index.xy].rgb : 0;
	const float ld = luminance(diffuse.rgb);
	const float ls = luminance(specular);
	const float lt = luminance(transmittance);
	m.base_color((diffuse.rgb * ld + specular * ls + transmittance*lt) / (ls + ld + lt));
	if (gUseSpecular) m.metallic(saturate(ls/(ld + ls + lt)));
	if (gUseRoughness) m.roughness(gRoughness[index.xy]);
	if (gUseTransmittance) m.transmission(saturate(lt / (ld + ls + lt)));

	for (uint j = 0; j < DISNEY_DATA_N; j++) gOutput[j][index.xy] = m.data[j];
}