#ifndef BSDF_H
#define BSDF_H

struct MaterialEvalRecord {
	float3 mReflectance;
	float mFwdPdfW;
	float mRevPdfW;
};
struct MaterialSampleRecord {
	float3 mDirection;
	float mFwdPdfW;
	float mRevPdfW;
	float mEta;
	float mRoughness;
};

interface BSDF {
	bool canEvaluate();
	bool isSingular();

	float3 emission();
	float3 albedo();

	MaterialEvalRecord evaluate(const float3 dirIn, const float3 dirOut, const bool adjoint);
	MaterialSampleRecord sample(const float3 rnd, const float3 dirIn, const bool adjoint);
};

#endif