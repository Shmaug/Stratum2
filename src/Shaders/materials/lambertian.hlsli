#include "bsdf.hlsli"

#include "compat/scene.h"
#include "compat/disney_data.h"
#include "compat/image_value.h"

struct LambertianMaterial : BSDF {
	DisneyMaterialData bsdf;

	SLANG_MUTATING
	void load(const SceneParameters scene, uint address, const float2 uv, const float uvScreenSize, inout uint packedShadingNormal, inout uint packedTangent, const bool flipBitangent) {
		for (int i = 0; i < DisneyMaterialData::gDataSize; i++)
			bsdf.data[i] = ImageValue4(address).eval(uv, uvScreenSize);

		address += 4; // skip alpha mask

		// normal map
		if (gUseNormalMaps) {
			const uint2 p = scene.mMaterialData.Load<uint2>(address);
			ImageValue3 bump_img = { 1, p.x };
			if (bump_img.hasImage() && asfloat(p.y) > 0) {
				float3 bump = bump_img.eval(uv, uvScreenSize)*2-1;
				bump = normalize(float3(bump.xy * asfloat(p.y), bump.z > 0 ? bump.z : 1));

				float3 n = unpackNormal(packedShadingNormal);
				float3 t = unpackNormal(packedTangent);

				n = normalize(t*bump.x + cross(n, t)*(flipBitangent ? -1 : 1)*bump.y + n*bump.z);
				t = normalize(t - n*dot(n, t));

				packedShadingNormal = packNormal(n);
				packedTangent        = packNormal(t);
			}
		}
	}

	SLANG_MUTATING
	void load(const SceneParameters scene, inout ShadingData sd) {
		load(scene, sd.materialAddress, sd.mTexcoord, sd.mTexcoordScreenSize, sd.mPackedShadingNormal, sd.mPackedTangent, sd.isBitangentFlipped);
	}

	__init(const SceneParameters scene, uint address, const float2 uv, const float uvScreenSize, inout uint packedShadingNormal, inout uint packedTangent, const bool flipBitangent) {
		load(scene, address, uv, uvScreenSize, packedShadingNormal, packedTangent, flipBitangent);

	}
	__init(const SceneParameters scene, inout ShadingData sd) {
		load(scene, sd);
	}


	float3 emission() { return bsdf.baseColor()*bsdf.emission(); }
	float3 albedo() { return bsdf.baseColor(); }
	bool canEvaluate() { return bsdf.emission() <= 0 && any(bsdf.baseColor() > 0); }
	bool isSingular() { return false; }

	MaterialEvalRecord evaluate(const float3 dirIn, const float3 dirOut, const bool adjoint) {
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

	MaterialSampleRecord sample(const float3 rnd, const float3 dirIn, const bool adjoint) {
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