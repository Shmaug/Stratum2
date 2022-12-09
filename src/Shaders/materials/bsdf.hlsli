#pragma once

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

	MaterialEvalRecord evaluate<let Adjoint : bool>(const float3 dirIn, const float3 dirOut);
	MaterialSampleRecord sample<let Adjoint : bool>(const float3 rnd, const float3 dirIn);
};