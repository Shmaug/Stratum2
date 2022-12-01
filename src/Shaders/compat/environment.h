#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include "scene.h"
#include "image_value.h"

STM_NAMESPACE_BEGIN

struct Environment {
	ImageValue3 mEmission;

#ifdef __cplusplus
	inline void store(MaterialResources& resources) const {
		mEmission.store(resources);
	}
	inline void drawGui() {
		mEmission.drawGui("Emission");
	}
#endif

#ifdef __HLSL__
	__init(uint address) {
		mEmission = ImageValue3(address);
	}

	float3 evaluate(const float3 dirOut) {
		if (!mEmission.hasImage())
			return mEmission.mValue;
		uint w, h;
		mEmission.image().GetDimensions(w, h);
		const float2 uv = cartesianToSphericalUv(dirOut);
		return mEmission.eval(uv, 0).rgb;
	}

	float3 sample(const float2 rnd, out float3 dirOut, out float pdf) {
		if (!mEmission.hasImage()) {
			const float2 uv = sampleUniformSphere(rnd.x, rnd.y);
			dirOut = sphericalUvToCartesian(uv);
			pdf = 1/(4*M_PI);
			return mEmission.mValue;
		} else {
			uint w, h;
			mEmission.image().GetDimensions(w, h);
			const float2 uv = sampleTexel(mEmission.image(), rnd, pdf);
			dirOut = sphericalUvToCartesian(uv);
			pdf /= (2 * M_PI * M_PI * sqrt(1 - dirOut.y*dirOut.y));
			return mEmission.eval(uv, 0).rgb;
		}
	}

	float evaluatePdfW(const float3 dirOut) {
		if (!mEmission.hasImage())
			return 1/(4*M_PI);
		else {
			const float2 uv = cartesianToSphericalUv(dirOut);
			const float pdf = sampleTexelPdf(mEmission.image(), uv);
			return pdf / (2 * M_PI * M_PI * sqrt(1 - dirOut.y*dirOut.y));
		}
	}
#endif
};

STM_NAMESPACE_END

#endif