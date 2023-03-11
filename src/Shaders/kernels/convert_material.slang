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
#ifndef gUseEmission
#define gUseEmission false
#endif

#include "compat/common.h"
#define MATERIAL_STORE_FN
#include "common/material.hlsli"

Texture2D<float4> gDiffuse;
Texture2D<float4> gSpecular;
Texture2D<float3> gTransmittance;
Texture2D<float3> gEmission;
Texture2D<float> gRoughness;
RWTexture2D<float4> gOutput[4];
RWTexture2D<float> gOutputAlphaMask;
RWStructuredBuffer<uint> gOutputMinAlpha;


[shader("compute")]
[numthreads(8,8,1)]
void from_gltf_pbr(uint3 index : SV_DispatchThreadId) {
	uint2 size;
	if      (gUseDiffuse) gDiffuse.GetDimensions(size.x, size.y);
	else if (gUseSpecular) gSpecular.GetDimensions(size.x, size.y);
    else if (gUseTransmittance) gTransmittance.GetDimensions(size.x, size.y);
    else if (gUseEmission) gEmission.GetDimensions(size.x, size.y);
	if (any(index.xy >= size)) return;

	const float4 diffuse = gUseDiffuse ? gDiffuse[index.xy] : 1;
	if (gUseDiffuse) {
		gOutputAlphaMask[index.xy] = diffuse.a;
		if (diffuse.a < 1)
			InterlockedMin(gOutputMinAlpha[0], (uint)floor(saturate(diffuse.a)*255 + 0.5));
	}

	const float4 metallic_roughness = gUseSpecular ? gSpecular[index.xy] : 1;

    PackedMaterialData m;
    m.mPackedData = 0;
    m.mEmissionScale = 0;

    m.setBaseColor(diffuse.rgb);
    m.setEmission(gUseEmission ? gEmission[index.xy] : 1);
	m.setMetallic(metallic_roughness.b);
    m.setRoughness(metallic_roughness.g);
    m.setAnisotropic(1);
    m.setSubsurface(1);
    m.setClearcoat(1);
    m.setClearcoatGloss(1);
	const float l = luminance(m.getBaseColor());
    m.setTransmission(gUseTransmittance ? saturate(luminance(gTransmittance[index.xy].rgb) / (l > 0 ? l : 1)) : 1);
    m.setEta(1);
    m.setSpecular(1);
    m.setSheen(1);

    m.Store(gOutput, index.xy);
}

[shader("compute")]
[numthreads(8,8,1)]
void from_diffuse_specular(uint3 index : SV_DispatchThreadId) {
	uint2 size;
	if      (gUseDiffuse) gDiffuse.GetDimensions(size.x, size.y);
    else if (gUseSpecular) gSpecular.GetDimensions(size.x, size.y);
    else if (gUseTransmittance) gTransmittance.GetDimensions(size.x, size.y);
    else if (gUseEmission) gEmission.GetDimensions(size.x, size.y);
	if (any(index.xy >= size)) return;

	const float4 diffuse = gUseDiffuse ? gDiffuse[index.xy] : 0;
	if (gUseDiffuse) {
		gOutputAlphaMask[index.xy] = diffuse.a;
		if (diffuse.a < 1)
			InterlockedMin(gOutputMinAlpha[0], (uint)floor(saturate(diffuse.a)*255 + 0.5));
	}

	const float3 specular = gUseSpecular ? gSpecular[index.xy].rgb : 0;
	const float3 transmittance = gUseTransmittance ? gTransmittance[index.xy].rgb : 0;
	const float ld = luminance(diffuse.rgb);
	const float ls = luminance(specular);
    const float lt = luminance(transmittance);

	PackedMaterialData m;
    m.setBaseColor((diffuse.rgb * ld + specular * ls + transmittance * lt) / (ls + ld + lt));
    m.setEmission(gUseEmission ? gEmission[index.xy] : 1);
    m.setMetallic(gUseSpecular ? saturate(ls / (ld + ls + lt)) : 1);
    m.setRoughness(gUseRoughness ? gRoughness[index.xy] : 1);
    m.setAnisotropic(1);
    m.setSubsurface(1);
    m.setClearcoat(1);
    m.setClearcoatGloss(1);
    m.setTransmission(gUseTransmittance ? saturate(lt / (ld + ls + lt)) : 1);
    m.setEta(1);
    m.setSpecular(1);
    m.setSheen(1);

    m.Store(gOutput, index.xy);
}