#include "common/material.hlsli"

extension PackedMaterialData {
    float3 emission() { return getEmission(); }
    float emissionPdf() { return any(getEmission() > 0) ? 1 : 0; }
	float3 albedo() { return getBaseColor(); }
    bool canEvaluate() { return any(getBaseColor() > 0); }
    bool isSingular() { return false; }
    float continuationProb() { return saturate(luminance(getBaseColor())); }

	ReflectanceEvalRecord evaluateReflectance<let Adjoint : bool>(const float3 dirIn, const float3 dirOut) {
		ReflectanceEvalRecord r;
		if (dirIn.z * dirOut.z < 0) {
			r.mReflectance = 0;
			r.mFwdPdfW = r.mRevPdfW = 0;
			return r;
		}
		r.mReflectance = getBaseColor() * abs(dirOut.z) / M_PI;
		r.mFwdPdfW = cosHemispherePdfW(abs(dirOut.z));
		r.mRevPdfW = cosHemispherePdfW(abs(dirIn.z));
		return r;
	}

    DirectionSampleRecord sampleDirection<let Adjoint : bool>(const float3 rnd, const float3 dirIn) {
        DirectionSampleRecord r;
        r.mDirection = sampleCosHemisphere(rnd.x, rnd.y);
		if (dirIn.z < 0) r.mDirection = -r.mDirection;
        r.mReflectance = getBaseColor() * abs(r.mDirection.z) / M_PI;
		r.mFwdPdfW = cosHemispherePdfW(abs(r.mDirection.z));
		r.mRevPdfW = cosHemispherePdfW(abs(dirIn.z));
		r.mRoughness = 1;
		r.mEta = 0;
		return r;
	}
};