#include "bsdf.hlsli"

#include "compat/image_value.h"
#include "compat/material.h"
#include "compat/scene.h"


void LoadNormalMap(inout ShadingData aoShadingData) {
    if (!gNormalMaps) return;

    const uint2 p = gScene.mMaterialData.Load<uint2>(aoShadingData.materialAddress + ImageValue4::PackedSize * MaterialData::gDataCount + 4);
    ImageValue3 bump_img = { 1, p.x };
    if (!bump_img.hasImage()) return;
    const float scale = asfloat(p.y);
    if (scale <= 0) return;

    float3 bump = bump_img.eval(aoShadingData.mTexcoord, aoShadingData.mTexcoordScreenSize);
    bump.xy = (bump.xy * 2 - 1) * scale;
    bump = normalize(bump);

    float3 n = aoShadingData.shadingNormal;
    float3 t = aoShadingData.tangent;

    n = normalize(t * bump.x + cross(n, t) * (aoShadingData.isBitangentFlipped ? -1 : 1) * bump.y + n * bump.z);
    t = normalize(t - n * dot(n, t));

    aoShadingData.mPackedShadingNormal = packNormal(n);
    aoShadingData.mPackedTangent = packNormal(t);
}

struct LambertianMaterial : BSDF {
	MaterialData bsdf;

	__init(inout ShadingData sd) {
        for (int i = 0; i < MaterialData::gDataCount; i++)
            bsdf.data[i] = ImageValue4(gScene.mMaterialData, sd.materialAddress + i*ImageValue4::PackedSize).eval(sd.mTexcoord, sd.mTexcoordScreenSize);

        LoadNormalMap(sd);
	}

    float3 emission() { return bsdf.baseColor() * bsdf.emission(); }
    float emissionPdf() { return any(bsdf.emission() > 0) ? 1 : 0; }
	float3 albedo() { return bsdf.baseColor(); }
    bool canEvaluate() { return bsdf.emission() <= 0 && any(bsdf.baseColor() > 0); }
    bool isSingular() { return false; }
    float continuationProb() { return saturate(luminance(bsdf.baseColor())); }

	MaterialEvalRecord evaluate<let Adjoint : bool>(const float3 dirIn, const float3 dirOut) {
		MaterialEvalRecord r;
		if (bsdf.emission() > 0 || dirIn.z * dirOut.z < 0) {
			r.mReflectance = 0;
			r.mFwdPdfW = r.mRevPdfW = 0;
			return r;
		}

		r.mReflectance = bsdf.baseColor() * abs(dirOut.z) / M_PI;
		r.mFwdPdfW = cosHemispherePdfW(abs(dirOut.z));
		r.mRevPdfW = cosHemispherePdfW(abs(dirIn.z));
		return r;
	}

	MaterialSampleRecord sample<let Adjoint : bool>(const float3 rnd, const float3 dirIn) {
		MaterialSampleRecord r;
		if (bsdf.emission() > 0) {
			r.mFwdPdfW = r.mRevPdfW = 0;
			return r;
		}

		// diffuse
		r.mDirection = sampleCosHemisphere(rnd.x, rnd.y);
		if (dirIn.z < 0) r.mDirection = -r.mDirection;
		r.mFwdPdfW = cosHemispherePdfW(abs(r.mDirection.z));
		r.mRevPdfW = cosHemispherePdfW(abs(dirIn.z));
		r.mRoughness = 1;
		r.mEta = 0;
		return r;
	}
};