#include "compat/environment_image.h"
#include "bsdf.hlsli"

extension EnvironmentImage {
    __init() {
        mEmission = ImageValue3(gScene.mMaterialData, gPushConstants.mEnvironmentMaterialAddress);
    }

    float3 sample(const float2 rnd, out float3 dirOut, out float2 uv, out float pdf) {
        if (!mEmission.hasImage()) {
            uv = sampleUniformSphere(rnd.x, rnd.y);
            dirOut = sphericalUvToCartesian(uv);
            pdf = 1 / (4 * M_PI);
            return mEmission.mValue;
        } else {
            uv = sampleTexel(gScene.mImages[mEmission.mImage], rnd, pdf);
            dirOut = sphericalUvToCartesian(uv);
            pdf /= (2 * M_PI * M_PI * sqrt(1 - dirOut.y * dirOut.y));
            return mEmission.eval(uv, 0).rgb;
        }
    }

    void evaluate(const float3 dirOut, const float2 uv, out float3 emission, out float pdfW) {
        if (!mEmission.hasImage()) {
			emission = mEmission.mValue;
            pdfW = 1 / (4 * M_PI);
            return;
        }
        emission = mEmission.eval(uv, 0).rgb;

		const float pdf = sampleTexelPdf(gScene.mImages[mEmission.mImage], uv);
		pdfW = pdf / (2 * M_PI * M_PI * sqrt(1 - dirOut.y * dirOut.y));
    }
};

struct EnvironmentBSDF : BSDF {
	float3 mEmission;
    float mEmissionPdfW;

    bool canEvaluate() { return false; }
    bool isSingular() { return true; }

    float3 emission() { return mEmission; }
    float emissionPdf() { return mEmissionPdfW; }
    float3 albedo() { return 1; }
    float continuationProb() { return 0; }

    MaterialEvalRecord evaluate<let Adjoint : bool>(const float3 dirIn, const float3 dirOut) {
        return { 0, 0, 0 };
    }
    MaterialSampleRecord sample<let Adjoint : bool>(const float3 rnd, const float3 dirIn) {
		return { 0, 0, 0, 0, 0 };
	}

    __init(ShadingData shadingData) {
        if (!gHasEnvironment) {
            mEmission = 0;
            mEmissionPdfW = 0;
		} else
        	EnvironmentImage().evaluate(shadingData.mPosition, shadingData.mTexcoord, mEmission, mEmissionPdfW);
	}
};