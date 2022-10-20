#ifndef BSDF_H
#define BSDF_H

#ifndef Real
#define Real float
#define Vector3 float3
#define Spectrum float3
#endif

struct MaterialEvalRecord {
	Spectrum f;
	Real pdf_fwd;
	Real pdf_rev;
};
struct MaterialSampleRecord {
	Vector3 dir_out;
	Real pdf_fwd;
	Real pdf_rev;
	Real eta;
	Real roughness;
};

interface BSDF {
	Spectrum Le();
	Spectrum albedo();
	bool can_eval();
	bool is_specular();
	void eval(out MaterialEvalRecord r, const Vector3 dir_in, const Vector3 dir_out, const bool adjoint);
	Spectrum sample(out MaterialSampleRecord r, const Vector3 rnd, const Vector3 dir_in, inout Spectrum beta, const bool adjoint);
	Spectrum eval_approx(const Vector3 dir_in, const Vector3 dir_out, const bool adjoint);
};

#include "../materials/disney_material.hlsli"
#define Material DisneyMaterial
//#include "rough_plastic.hlsli"
//#define Material RoughPlastic

#endif