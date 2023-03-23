#pragma once

#include "common/microfacet.hlsli"
#include "common/material.hlsli"

extension PackedMaterialData {
    float getEta1() { return getEta() + 1; }

    float3 eval<let Adjoint : bool>(const float3 dir_in, const float3 dir_out) {
		const float aspect = sqrt(1 - getAnisotropic() * 0.9);
		const float min_alpha = 1e-4;
		const float alpha = pow2(getRoughness());
		const float alpha_x = max(min_alpha, alpha / aspect);
        const float alpha_y = max(min_alpha, alpha * aspect);

        const float metallic = getMetallic();

		if (dir_in.z * dir_out.z > 0) {
            // Diffuse component
            const float n_dot_in  = abs(dir_in.z);
            const float n_dot_out = dir_out.z;
			float3 half_vector = normalize(dir_in + dir_out);
			// Flip half-vector if it's below surface
			if (half_vector.z * dir_in.z < 0) {
				half_vector = -half_vector;
			}
			const float n_dot_h = half_vector.z;
			const float h_dot_in  = dot(half_vector, dir_in);
			const float h_dot_out = dot(half_vector, dir_out);

			float3 contrib = 0;

			// For diffuse, metallic, sheen, and clearcoat, the light bounces
			// only at the top of the surface.
			if (dir_in.z >= 0 && dir_out.z >= 0) {
				// Diffuse

				// The base diffuse model
				const float Fd90 = float(0.5) + 2 * getRoughness() * h_dot_out * h_dot_out;
				const float schlick_n_dot_out = pow(1 - n_dot_out, float(5));
				const float schlick_n_dot_in  = pow(1 - n_dot_in , float(5));
				const float schlick_h_dot_out = pow(1 - h_dot_out, float(5));
				const float base_diffuse = (1 + (Fd90 - 1) * schlick_n_dot_out) * (1 + (Fd90 - 1) * schlick_n_dot_in);

				// The subsurface model
				// Disney's hack to increase the response at grazing angle
				const float Fss90 = h_dot_out * h_dot_out * getRoughness();
				const float Fss = (1 + (Fss90 - 1) * schlick_n_dot_out) * (1 + (Fss90 - 1) * schlick_n_dot_in);
				// Lommel-Seeliger law (modified/rescaled)
				const float ss = float(1.25) * (Fss * (1 / (n_dot_out + n_dot_in) - float(0.5)) + float(0.5));

				const float3 baseColor = getBaseColor();

                contrib += (1 - getTransmission()) * (1 - metallic) *
					baseColor * (lerp(base_diffuse, ss, getSubsurface()) / M_PI) * n_dot_out;

				// Sheen
				const float3 Ctint =
					luminance(baseColor) > 0 ? baseColor / luminance(baseColor) :
					1;
				const float3 Csheen = lerp(1, Ctint, getSheenTint());
				contrib += (1 - metallic) * getSheen() * Csheen * schlick_h_dot_out * n_dot_out;


				// Metallic
				if (n_dot_in > 0 && h_dot_out > 0 && n_dot_h > 0) {
					const float eta = getEta1(); // we're always going inside
                    const float spec_f0 = (eta - 1) * (eta - 1) / ((eta + 1) * (eta + 1));
                    const float3 spec_color = lerp(1, Ctint, getSpecularTint());
                    const float3 Cspec0 = lerp(getSpecular() * spec_f0 * spec_color, getBaseColor(), metallic);

					const float3 F    = schlick_fresnel(Cspec0, h_dot_out);
					const float D     = GTR2(half_vector, alpha_x, alpha_y);
					const float G_in  = smith_masking_gtr2(dir_in , alpha_x, alpha_y);
					const float G_out = smith_masking_gtr2(dir_out, alpha_x, alpha_y);
					const float G = G_in * G_out;

					const float spec_weight = 1 - getTransmission() * (1 - metallic);
					contrib += spec_weight * F * D * G / (4 * n_dot_in);
				}

				// Clearcoat component
				if (n_dot_in > 0 && n_dot_h > 0) {
        			const float alpha_c = lerp(0.1, 0.001, getClearcoatGloss());
					const float Fc = schlick_fresnel(float(0.04), h_dot_out);
					// Generalized Trowbridge-Reitz distribution
					const float Dc = GTR1(n_dot_h, alpha_c);
					// SmithG with fixed alpha
					const float Gc_in  = smith_masking_gtr1(dir_in);
					const float Gc_out = smith_masking_gtr1(dir_out);
					const float Gc = Gc_in * Gc_out;

					contrib += getClearcoat() * Fc * Dc * Gc / (4 * n_dot_in);
				}
			}

            // Glass
            // For glass, lights bounce at both sides of the surface.
            const float glass_w = (1 - metallic) * getTransmission();
			if (glass_w > 0) {
				const float eta = dir_in.z > 0 ? getEta1() : 1 / getEta1();
				const float Fg    = fresnel_dielectric(h_dot_in, eta);
				const float D     = GTR2(half_vector, alpha_x, alpha_y);
				const float G_in  = smith_masking_gtr2(dir_in, alpha_x, alpha_y);
				const float G_out = smith_masking_gtr2(dir_out, alpha_x, alpha_y);
				const float G = G_in * G_out;
				contrib += getBaseColor() * (glass_w * (Fg * D * G) / (4 * abs(n_dot_in)));
			}
			return contrib;
        } else {
            const float glass_w = (1 - metallic) * getTransmission();
            if (glass_w <= 0)
                return 0;

			// Only the glass component for refraction
			const float eta = dir_in.z > 0 ? getEta1() : 1 / getEta1();
			float3 half_vector = normalize(dir_in + dir_out * eta);
            // Flip half-vector if it's below surface
            if (half_vector.z * dir_in.z < 0) {
				half_vector = -half_vector;
			}

			const float eta_factor = Adjoint ? (1 / (eta * eta)) : 1;
			const float h_dot_in   = dot(half_vector, dir_in);
			const float h_dot_out  = dot(half_vector, dir_out);
			const float sqrt_denom = h_dot_in + eta * h_dot_out;

			const float Fg    = fresnel_dielectric(h_dot_in, eta);
			const float D     = GTR2(half_vector, alpha_x, alpha_y);
			const float G_in  = smith_masking_gtr2(dir_in, alpha_x, alpha_y);
			const float G_out = smith_masking_gtr2(dir_out, alpha_x, alpha_y);
			const float G = G_in * G_out;

			// Burley propose to take the square root of the base color to preserve albedo
			return sqrt(getBaseColor()) * (glass_w *
				(eta_factor * (1 - Fg) * D * G * eta * eta * abs(h_dot_out * h_dot_in)) /
				(abs(dir_in.z) * sqrt_denom * sqrt_denom));
		}
	}

    float evalPdf(const float3 dir_in, const float3 dir_out) {
		const float alpha = pow2(getRoughness());
		const float metallic = getMetallic();
		const float aspect = sqrt(1 - getAnisotropic() * float(0.9));
		const float min_alpha = float(0.0001);
		const float alpha_x = max(min_alpha, alpha / aspect);
		const float alpha_y = max(min_alpha, alpha * aspect);
		const float alpha_c = (1 - getClearcoatGloss()) * float(0.1) + getClearcoatGloss() * float(0.001);

        const float transmission = getTransmission();
		const float diffuse_weight = (1 - metallic) * (1 - transmission);
		const float metallic_weight = (1 - transmission * (1 - metallic));
		const float glass_weight = (1 - metallic) * transmission;
		const float clearcoat_weight = getClearcoat();
		const float total_weight = diffuse_weight + metallic_weight + glass_weight + clearcoat_weight;
		float diffuse_prob = diffuse_weight / total_weight;
		float metallic_prob = metallic_weight / total_weight;
		float glass_prob = glass_weight / total_weight;
		float clearcoat_prob = clearcoat_weight / total_weight;

		if (dir_in.z < 0) {
			// Our incoming ray is coming from inside,
			// so the probability of sampling the glass lobe is 1 if glass_prob is not 0.
			diffuse_prob = 0;
			metallic_prob = 0;
			clearcoat_prob = 0;
			if (glass_prob > 0) {
				glass_prob = 1;
			}
		}

		if (dir_in.z * dir_out.z > 0) {
			// For metallic: visible normal sampling -> D * G_in
			float3 half_vector = normalize(dir_in + dir_out);
			// Flip half-vector if it's below surface
			if (half_vector.z < 0) {
				half_vector = -half_vector;
			}
			const float n_dot_in  = dir_in.z;
			const float n_dot_h   = half_vector.z;
			const float h_dot_in  = dot(half_vector, dir_in);
			const float h_dot_out = dot(half_vector, dir_out);

			// For diffuse, metallic, and clearcoat, the light bounces
			// only at the top of the surface.
			if (dir_in.z >= 0 && dir_out.z >= 0) {
				diffuse_prob *= max(dir_out.z, float(0)) / M_PI;

				if (n_dot_in > 0) {
					float D    = GTR2(half_vector, alpha_x, alpha_y);
					float G_in = smith_masking_gtr2(dir_in, alpha_x, alpha_y);
					metallic_prob *= (D * G_in / (4 * n_dot_in));
				} else {
					metallic_prob = 0;
				}

				// For clearcoat: D importance sampling
				if (n_dot_h > 0 && h_dot_out > 0) {
					float Dc = GTR1(n_dot_h, alpha_c);
					clearcoat_prob *= (Dc * n_dot_h / (4 * h_dot_out));
				} else {
					clearcoat_prob = 0;
				}
			}

            // For glass: F * visible normal
            const float eta  = dir_in.z > 0 ? getEta1() : 1 / getEta1();
			const float Fg   = fresnel_dielectric(h_dot_in, eta);
			const float D    = GTR2(half_vector, alpha_x, alpha_y);
			const float G_in = smith_masking_gtr2(dir_in, alpha_x, alpha_y);
			glass_prob *= (Fg * D * G_in / (4 * abs(n_dot_in)));
		} else {
			// Only glass component for refraction
			const float eta = dir_in.z > 0 ? getEta1() : 1 / getEta1();
			float3 half_vector = normalize(dir_in + dir_out * eta);
			// Flip half-vector if it's below surface
			if (half_vector.z < 0) {
				half_vector = -half_vector;
			}
			const float h_dot_in = dot(half_vector, dir_in);
			const float h_dot_out = dot(half_vector, dir_out);
			const float D    = GTR2(half_vector, alpha_x, alpha_y);
			const float G_in = smith_masking_gtr2(dir_in, alpha_x, alpha_y);
			const float Fg   = fresnel_dielectric(h_dot_in, eta);
			const float sqrt_denom = h_dot_in + eta * h_dot_out;
			const float dh_dout = eta * eta * h_dot_out / (sqrt_denom * sqrt_denom);
			glass_prob *= (1 - Fg) * D * G_in * abs(dh_dout * h_dot_in / dir_in.z);
		}

		return diffuse_prob + metallic_prob + glass_prob + clearcoat_prob;
	}

    DirectionSampleRecord sample(const float3 dir_in, const float3 rnd) {
		float aspect = sqrt(1 - getAnisotropic() * float(0.9));
        float min_alpha = float(0.0001);
        const float alpha = pow2(getRoughness());
        float alpha_x = max(min_alpha, alpha / aspect);
        float alpha_y = max(min_alpha, alpha * aspect);
		float alpha_c = (1 - getClearcoatGloss()) * float(0.1) + getClearcoatGloss() * float(0.001);

        const float metallic = getMetallic();
        const float transmission = getTransmission();
		float diffuse_weight = (1 - metallic) * (1 - transmission);
		float metallic_weight = (1 - transmission * (1 - metallic));
		float glass_weight = (1 - metallic) * transmission;
		float clearcoat_weight = getClearcoat();

		// Two cases: 1) if we are coming from "outside" the surface,
		// sample all lobes
		if (dir_in.z >= 0) {
			float total_weight = diffuse_weight + metallic_weight + glass_weight + clearcoat_weight;
			float diffuse_prob = diffuse_weight / total_weight;
			float metallic_prob = metallic_weight / total_weight;
			float glass_prob = glass_weight / total_weight;
			// float clearcoat_prob = clearcoat_weight / total_weight;
            if (rnd.z <= diffuse_prob) {
                DirectionSampleRecord r;
                r.mDirection = sampleCosHemisphere(rnd.x, rnd.y);
                r.mEta = 0;
                r.mRoughness = 1;
                return r;
			} else if (rnd.z <= (diffuse_prob + metallic_prob)) { // metallic
				// Visible normal sampling

				// Convert the incoming direction to local coordinates
                float3 local_dir_in = dir_in;
                float3 half_vector = sample_visible_normals(local_dir_in, alpha_x, alpha_y, rnd.xy);

                // Reflect over the world space normal
                DirectionSampleRecord r;
                r.mDirection = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
                r.mEta = 0;
                r.mRoughness = getRoughness();
                return r;
			} else if (rnd.z <= (diffuse_prob + metallic_prob + glass_prob)) { // glass
				if (glass_prob <= 0) {
					// Just to be safe numerically.
					return { 0, 0 };
				}
				// Visible normal sampling

				// Convert the incoming direction to local coordinates
                float3 local_dir_in = dir_in;
                float3 half_vector = sample_visible_normals(local_dir_in, alpha_x, alpha_y, rnd.xy);

				// Now we need to decide whether to reflect or refract.
				// We do this using the Fresnel term.
				float h_dot_in = dot(half_vector, dir_in);
				float eta = dir_in.z > 0 ? getEta1() : 1 / getEta1();
				float F = fresnel_dielectric(h_dot_in, eta);
				// rescale rnd_param_w from
				// (diffuse_prob + metallic_prob, diffuse_prob + metallic_prob + glass_prob]
				// to
				// (0, 1]
				float u = (rnd.z - (diffuse_prob + metallic_prob)) / glass_prob;
				DirectionSampleRecord r;
				if (u <= F) {
					// Reflect over the world space normal
                    r.mDirection = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
                    r.mEta = 0;
					return r;
				} else {
					// Refraction
					float h_dot_out_sq = 1 - (1 - h_dot_in * h_dot_in) / (eta * eta);
					if (h_dot_out_sq <= 0) {
						return {};
					}
					// flip half_vector if needed
					if (h_dot_in < 0) {
						half_vector = -half_vector;
					}
					float h_dot_out= sqrt(h_dot_out_sq);
                    r.mDirection = -dir_in / eta + (abs(h_dot_in) / eta - h_dot_out) * half_vector;
                    r.mEta = eta;
				}
				r.mRoughness = getRoughness();
				return r;
			} else { // clearcoat
				// Only importance sampling D

				// Appendix B.2 Burley's note
				float alpha2 = alpha_c * alpha_c;
				// Equation 5
				float cos_h_elevation = sqrt(max(float(0), (1 - pow(alpha2, 1 - rnd[0])) / (1 - alpha2)));
				float sin_h_elevation = sqrt(max(1 - cos_h_elevation * cos_h_elevation, float(0)));
                float h_azimuth = 2 * M_PI * rnd[1];
                float3 local_micro_normal = float3(
					sin_h_elevation * cos(h_azimuth),
					sin_h_elevation * sin(h_azimuth),
					cos_h_elevation );

				DirectionSampleRecord r;
                r.mDirection = normalize(-dir_in + 2 * dot(dir_in, local_micro_normal) * local_micro_normal);
                r.mEta = 0;
                r.mRoughness = sqrt(alpha_c);
				return r;
			}
		} else {
			// 2) otherwise, only consider the glass lobes.

            // Convert the incoming direction to local coordinates
            float3 local_dir_in = dir_in;
			float3 local_micro_normal = sample_visible_normals(local_dir_in, alpha_x, alpha_y, rnd.xy);

			// Transform the micro normal to world space
			float3 half_vector = local_micro_normal;
			// Flip half-vector if it's below surface
			if (half_vector.z < 0) {
				half_vector = -half_vector;
			}

			// Now we need to decide whether to reflect or refract.
			// We do this using the Fresnel term.
			float h_dot_in = dot(half_vector, dir_in);
			float eta = dir_in.z > 0 ? getEta1() : 1 / getEta1();
			float F = fresnel_dielectric(h_dot_in, eta);
			float u = rnd.z;
			DirectionSampleRecord r;
			if (u <= F) {
				// Reflect over the world space normal
                r.mDirection = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
				r.mEta = 0;
			} else {
				// Refraction
				float h_dot_out_sq = 1 - (1 - h_dot_in * h_dot_in) / (eta * eta);
				if (h_dot_out_sq <= 0) {
					return { 0, 0 };
				}
				// flip half_vector if needed
				if (h_dot_in < 0) {
					half_vector = -half_vector;
				}
                float h_dot_out = sqrt(h_dot_out_sq);
                r.mDirection = -dir_in / eta + (abs(h_dot_in) / eta - h_dot_out) * half_vector;
                r.mEta = eta;
			}
			r.mRoughness = getRoughness();
			return r;
		}
	}


    float3 emission() { return getEmission(); }
    float emissionPdf() { return any(getEmission() > 0) ? 1 : 0; }
	float3 albedo() { return getBaseColor(); }
    bool canEvaluate() { return any(getBaseColor() > 0); }
    bool isSingular() { return getRoughness() < 1e-3; }
    float continuationProb() { return saturate(luminance(getBaseColor())); }


	ReflectanceEvalRecord evaluateReflectance<let Adjoint : bool>(float3 dirIn, float3 dirOut) {
        ReflectanceEvalRecord r;
		#ifdef gDebugFastBRDF
		r.mReflectance = evaluateReflectanceFast<Adjoint>(dirIn, dirOut);
		r.mFwdPdfW     = cosHemispherePdfW(abs(dirOut.z));
		r.mRevPdfW     = cosHemispherePdfW(abs(dirIn.z));
		#else
        r.mReflectance = eval<Adjoint>(dirIn, dirOut);
		r.mFwdPdfW     = evalPdf(dirIn, dirOut);
		r.mRevPdfW     = evalPdf(dirOut, dirIn);
		#endif
        return r;
    }
    float3 evaluateReflectanceFast<let Adjoint : bool>(const float3 dirIn, const float3 dirOut) {
        // phong brdf

        const float ks = saturate(1 - (1 - getTransmission()) * (1 - getMetallic()));
        const float kd = 1 - ks;

        const float n = 1 / (pow2(getRoughness()) + 0.01);
        const float alpha = max(0, dot(dirOut, normalize(-dirIn + 2 * dot(dirIn, float3(0, 0, sign(dirIn.z))) * float3(0, 0, sign(dirIn.z)))));
        return getBaseColor() * (kd / M_PI + ks * (n + 2) / (2 * M_PI) * pow(alpha, n)) * abs(dirOut.z);
    }

    DirectionSampleRecord sampleDirection<let Adjoint : bool>(const float3 rnd, float3 dirIn) {
		#ifdef gDebugFastBRDF
		DirectionSampleRecord r;
        r.mDirection = sampleCosHemisphere(rnd.x, rnd.y);
        r.mEta = 0;
        r.mRoughness = 1;
		#else
        DirectionSampleRecord r = sample(dirIn, rnd);
		#endif
        const ReflectanceEvalRecord f = evaluateReflectance<Adjoint>(dirIn, r.mDirection);
        r.mReflectance = f.mReflectance;
        r.mFwdPdfW = f.mFwdPdfW;
        r.mRevPdfW = f.mRevPdfW;
		return r;
	}
};
