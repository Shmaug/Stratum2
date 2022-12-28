#pragma once

#include "bsdf.hlsli"
#include "compat/disney_data.h"
#include "common/microfacet.hlsli"

float disneymetal_eval_pdf(const float D, const float G_in, const float cos_theta_in) {
	return D * G_in / (4 * abs(cos_theta_in));
}

float3 disneymetal_eval(const float3 baseColor, const float D, const float G, const float3 dirIn, const float h_dot_out) {
	return baseColor * schlick_fresnel3(baseColor, abs(h_dot_out)) * D * G / (4 * abs(dirIn.z));
}

struct DisneyMetalMaterial : BSDF {
    DisneyMaterialData bsdf;

    [mutating]
    void load(const uint address, const float2 uv, const float uvScreenSize, inout uint packedShadingNormal, inout uint packedTangent, const bool flipBitangent) {
        for (int i = 0; i < DisneyMaterialData::gDataCount; i++)
            bsdf.data[i] = ImageValue4(gScene.mMaterialData, address + i * ImageValue4::PackedSize).eval(uv, uvScreenSize);

        // normal map
        if (gNormalMaps) {
            const uint2 p = gScene.mMaterialData.Load<uint2>(address + (ImageValue4::PackedSize * DisneyMaterialData::gDataCount) + 4);
            ImageValue3 bump_img = { 1, p.x };
            if (bump_img.hasImage() && asfloat(p.y) > 0) {
                float3 bump = bump_img.eval(uv, uvScreenSize) * 2 - 1;
                bump = normalize(float3(bump.xy * asfloat(p.y), bump.z > 0 ? bump.z : 1));

                float3 n = unpackNormal(packedShadingNormal);
                float3 t = unpackNormal(packedTangent);

                n = normalize(t * bump.x + cross(n, t) * (flipBitangent ? -1 : 1) * bump.y + n * bump.z);
                t = normalize(t - n * dot(n, t));

                packedShadingNormal = packNormal(n);
                packedTangent = packNormal(t);
            }
        }
    }

    [mutating]
    void load(inout ShadingData sd) {
        load(sd.materialAddress, sd.mTexcoord, sd.mTexcoordScreenSize, sd.mPackedShadingNormal, sd.mPackedTangent, sd.isBitangentFlipped);
    }

    __init(uint address, const float2 uv, const float uvScreenSize, inout uint packedShadingNormal, inout uint packedTangent, const bool flipBitangent) {
        load(address, uv, uvScreenSize, packedShadingNormal, packedTangent, flipBitangent);
    }
    __init(inout ShadingData sd) {
        load(sd);
    }

    float3 emission() { return bsdf.baseColor() * bsdf.emission(); }
    float emissionPdf() { return any(bsdf.emission() > 0) ? 1 : 0; }
    float3 albedo() { return bsdf.baseColor(); }
    bool canEvaluate() { return bsdf.emission() <= 0 && any(bsdf.baseColor() > 0); }
    float continuationProb() { return saturate(luminance(bsdf.baseColor()) * (1.5 - bsdf.roughness())); }

    bool isSingular() { return bsdf.roughness() <= 1e-2; }

    MaterialEvalRecord evaluate<let Adjoint : bool>(const float3 dirIn, const float3 dirOut) {
		MaterialEvalRecord r;
		if (bsdf.emission() > 0 || dirIn.z * dirOut.z < 0) {
            r.mReflectance = 0;
            r.mFwdPdfW = r.mRevPdfW = 0;
            return r;
        }

        const float aspect = sqrt(1 - 0.9 * bsdf.anisotropic());
        const float2 alpha = max(0.0001, float2(bsdf.alpha() / aspect, bsdf.alpha() * aspect));

        const float3 h = normalize(dirIn + dirOut);

        const float D     = Dm(alpha.x, alpha.y, h);
        const float G_in  = G1(alpha.x, alpha.y, dirIn);
        const float G_out = G1(alpha.x, alpha.y, dirOut);
        r.mReflectance = disneymetal_eval(bsdf.baseColor(), D, G_in * G_out, dirIn, dot(h, dirOut));
        r.mFwdPdfW = disneymetal_eval_pdf(D, G_in, dirIn.z);
        r.mRevPdfW = disneymetal_eval_pdf(D, G_out, dirOut.z);
        return r;
    }

    MaterialSampleRecord sample<let Adjoint : bool>(const float3 rnd, const float3 dirIn) {
		MaterialSampleRecord r;
		if (bsdf.emission() > 0) {
            r.mFwdPdfW = r.mRevPdfW = 0;
            return r;
        }

        const float aspect = sqrt(1 - 0.9 * bsdf.anisotropic());
        const float2 alpha = max(0.0001, float2(bsdf.alpha() / aspect, bsdf.alpha() * aspect));

        const float3 h = sampleVisibleNormals(dirIn, alpha.x, alpha.y, rnd.xy);

        r.mDirection = reflect(-dirIn, h);
        const float D     = Dm(alpha.x, alpha.y, h);
        const float G_in  = G1(alpha.x, alpha.y, dirIn);
        const float G_out = G1(alpha.x, alpha.y, r.mDirection);
        r.mFwdPdfW = disneymetal_eval_pdf(D, G_in, dirIn.z);
        r.mRevPdfW = disneymetal_eval_pdf(D, G_out, r.mDirection.z);
        r.mEta = 0;
        r.mRoughness = bsdf.roughness();
        return r;
    }
};