#include "bsdf.hlsli"
#include "../microfacet.h"

#define gMinRoughness 1e-4

struct RoughPlastic : BSDF {
	Spectrum diffuse_reflectance;
	Real alpha; // roughness^2
	Spectrum specular_reflectance;
	Real eta;
	Spectrum emission;
	Real spec_weight;

	SLANG_MUTATING
	void load(uint address, const float2 uv, const float uv_screen_size) {
		const float4 diffuse_roughness = eval_image_value4(address, uv, uv_screen_size);
		diffuse_reflectance = diffuse_roughness.rgb;
		alpha = pow2(max(gMinRoughness, diffuse_roughness.a));
		const float4 specular_transmission = eval_image_value4(address, uv, uv_screen_size);
		specular_reflectance = specular_transmission.rgb;
		//specular_transmittance = specular_transmission.a;
		emission = eval_image_value3(address, uv, uv_screen_size);
		eta = gMaterialData.Load<float>(address);

		const float lR = luminance(diffuse_reflectance);
		const float lS = luminance(specular_reflectance);
		spec_weight = lS / (lS + lR);
	}

	Spectrum Le() { return emission; }
	Spectrum albedo() { return diffuse_reflectance + 0.25*specular_reflectance; }
	bool can_eval() { return any(diffuse_reflectance > 0) || any(specular_reflectance > 0); }
	bool is_specular() { return all(diffuse_reflectance <= 1e-3) && alpha < 1e-3; }

	void eval_lambertian(out MaterialEvalRecord r, const Vector3 dir_in, const Vector3 dir_out, const bool adjoint) {
		if (dir_in.z * dir_out.z <= 0) {
			r.f = 0;
			r.pdf_fwd = 0;
			r.pdf_rev = 0;
		} else {
			r.f = diffuse_reflectance * abs(dir_out.z) / M_PI;
			r.pdf_fwd = cosine_hemisphere_pdfW(abs(dir_out.z));
			r.pdf_rev = cosine_hemisphere_pdfW(abs(dir_in.z));
		}
	}
	Spectrum sample_lambertian(out MaterialSampleRecord r, const Vector3 rnd, const Vector3 dir_in, inout Spectrum beta, const bool adjoint) {
		r.dir_out = sample_cos_hemisphere(rnd.x, rnd.y);
		r.pdf_fwd = cosine_hemisphere_pdfW(r.dir_out.z);
		r.pdf_rev = cosine_hemisphere_pdfW(abs(dir_in.z));
		if (dir_in.z < 0) r.dir_out.z = -r.dir_out.z;
		const Spectrum f = diffuse_reflectance * abs(r.dir_out.z) / M_PI;
		beta *= f / r.pdf_fwd;
		r.eta = 0;
		r.roughness = 1;
		return f;
	}

	void eval_roughplastic(out MaterialEvalRecord r, const Vector3 dir_in, const Vector3 dir_out, const Vector3 half_vector, const bool adjoint) {
		// We first account for the dielectric layer.

		// Fresnel equation determines how much light goes through,
		// and how much light is reflected for each wavelength.
		// Fresnel equation is determined by the angle between the (micro) normal and
		// both incoming and outgoing directions (dir_out & dir_in).
		// However, since they are related through the Snell-Descartes law,
		// we only need one of them.
		const Real F_i = fresnel_dielectric(dot(half_vector, dir_in ), eta);
		const Real F_o = fresnel_dielectric(dot(half_vector, dir_out), eta);
		const Real D = GTR2(half_vector.z, alpha); // "Generalized Trowbridge Reitz", GTR2 is equivalent to GGX.
		const Real G_in  = smith_masking_gtr2(dir_in, alpha);
		const Real G_out = smith_masking_gtr2(dir_out, alpha);

		const Real spec_contrib = ((G_in * G_out) * F_o * D) / (4 * dir_in.z * dir_out.z);

		// Next we account for the diffuse layer.
		// In order to reflect from the diffuse layer,
		// the photon needs to bounce through the dielectric layers twice.
		// The transmittance is computed by 1 - fresnel.
		const Real diffuse_contrib = (1 - F_o) * (1 - F_i) / M_PI;

		r.f = (specular_reflectance * spec_contrib + diffuse_reflectance * diffuse_contrib) * abs(dir_out.z);

		if (all(r.f <= 0)) {
			r.pdf_fwd = 0;
			r.pdf_rev = 0;
		} else {
			// VNDF sampling importance samples smith_masking(cos_theta_in) * GTR2(cos_theta_h, alpha) * cos_theta_out
			// (4 * cos_theta_v) is the Jacobian of the reflectiokn
			// For the diffuse lobe, we importance sample cos_theta_out
			r.pdf_fwd = lerp(cosine_hemisphere_pdfW(dir_out.z), (G_in  * D) / (4 * dir_in.z ), spec_weight);
			r.pdf_rev = lerp(cosine_hemisphere_pdfW(dir_in.z),  (G_out * D) / (4 * dir_out.z), spec_weight);
		}
	}
	Spectrum sample_roughplastic(out MaterialSampleRecord r, const Vector3 rnd, const Vector3 dir_in, inout Spectrum beta, const bool adjoint) {
		Vector3 half_vector;
		if (rnd.z <= spec_weight) {
			half_vector = sample_visible_normals(dir_in, alpha, alpha, rnd.xy);
			r.dir_out = normalize(reflect(-dir_in, half_vector));
			r.roughness = sqrt(alpha);
		} else {
			r.dir_out = sample_cos_hemisphere(rnd.x, rnd.y);
			half_vector = normalize(dir_in + r.dir_out);
			r.roughness = 1;
		}
		r.eta = 0;
		MaterialEvalRecord f;
		eval_roughplastic(f, dir_in, r.dir_out, half_vector, adjoint);
		beta *= f.f / f.pdf_fwd;
		r.pdf_fwd = f.pdf_fwd;
		r.pdf_rev = f.pdf_rev;
		return f.f;
	}

	void eval(out MaterialEvalRecord r, const Vector3 dir_in, const Vector3 dir_out, const bool adjoint = false) {
		if (any(specular_reflectance > 0))
			eval_roughplastic(r, dir_in, dir_out, normalize(dir_in + dir_out), adjoint);
		else
			eval_lambertian(r, dir_in, dir_out, adjoint);
	}
	Spectrum sample(out MaterialSampleRecord r, const Vector3 rnd, const Vector3 dir_in, inout Spectrum beta, const bool adjoint = false) {
		if (all(specular_reflectance > 0))
			return sample_roughplastic(r, rnd, dir_in, beta, adjoint);
		else
			return sample_lambertian(r, rnd, dir_in, beta, adjoint);
	}
};