#include "../common/bsdf.hlsli"
#include "../common/microfacet.hlsli"

Real Dm(const Real alpha_x, const Real alpha_y, const Vector3 h_l) {
    const Real alpha_x2 = alpha_x * alpha_x;
    const Real alpha_y2 = alpha_y * alpha_y;
    const Vector3 h_l2 = h_l * h_l;
    const Real hh = h_l2.x/alpha_x2 + h_l2.y/alpha_y2 + h_l2.z;
    return 1 / (M_PI * alpha_x * alpha_y * hh*hh);
}
Real G1(const Real alpha_x, const Real alpha_y, const Vector3 w_l) {
    const Real alpha_x2 = alpha_x * alpha_x;
    const Real alpha_y2 = alpha_y * alpha_y;
    const Vector3 w_l2 = w_l * w_l;
    const Real lambda = (sqrt(1 + (w_l2.x*alpha_x2 + w_l2.y*alpha_y2) / w_l2.z) - 1) / 2;
    return 1 / (1 + lambda);
}
Real R0(const Real eta) {
    const Real num = eta - 1;
    const Real denom = eta + 1;
    return (num*num) / (denom*denom);
}

Real Dc(const Real alpha_g, const Real h_lz) {
    const Real alpha_g2 = alpha_g * alpha_g;
    return (alpha_g2 - 1) / (M_PI * log(alpha_g2)*(1 + (alpha_g2 - 1)*h_lz*h_lz));
}
Real Gc(const Vector3 w_l) {
    const Real wx = w_l.x*0.25;
    const Real wy = w_l.y*0.25;
    const Real lambda = (sqrt(1 + (wx*wx + wy*wy)/(w_l.z*w_l.z)) - 1) / 2;
    return 1 / (1 + lambda);
}

#include "../compat/disney_data.h"
#include "disney_diffuse.hlsli"
#include "disney_metal.hlsli"
#include "disney_glass.hlsli"
#include "disney_clearcoat.hlsli"
//#include "disney_sheen.hlsli"

#include "../compat/shading_data.h"

struct DisneyMaterial : BSDF {
	DisneyMaterialData bsdf;

	SLANG_MUTATING
	void load(SceneParameters scene, uint address, const float2 uv, const float uv_screen_size, inout uint packed_shading_normal, inout uint packed_tangent, const bool flip_bitangent) {
		for (int i = 0; i < DISNEY_DATA_N; i++)
			bsdf.data[i] = eval_image_value4(address, uv, uv_screen_size);

		address += 4; // alpha mask

		// normal map
		if (gUseNormalMaps) {
			const uint2 p = scene.gMaterialData.Load<uint2>(address);
			ImageValue3 bump_img;
			bump_img.value = 1;
			bump_img.image_index = p.x;
			if (bump_img.has_image() && asfloat(p.y) > 0) {
				float3 bump = bump_img.eval(uv, uv_screen_size)*2-1;
				if (gFlipNormalMaps)
					bump.y = -bump.y;
				bump = normalize(float3(bump.xy * asfloat(p.y), bump.z > 0 ? bump.z : 1));

				float3 n = unpack_normal_octahedron(packed_shading_normal);
				float3 t = unpack_normal_octahedron(packed_tangent);

				n = normalize(t*bump.x + cross(n, t)*(flip_bitangent ? -1 : 1)*bump.y + n*bump.z);
				t = normalize(t - n*dot(n, t));

				packed_shading_normal = pack_normal_octahedron(n);
				packed_tangent        = pack_normal_octahedron(t);
			}
		}
	}

	SLANG_MUTATING
	void load(SceneParameters scene, uint address, inout ShadingData sd) {
		load(scene, address, sd.uv, sd.uv_screen_size, sd.packed_shading_normal, sd.packed_tangent, sd.flip_bitangent());
	}

	Spectrum Le() { return bsdf.base_color()*bsdf.emission(); }
	Spectrum albedo() { return bsdf.base_color(); }
	bool can_eval() { return bsdf.emission() <= 0 && any(bsdf.base_color() > 0); }
#ifdef FORCE_LAMBERTIAN
	bool is_specular() { return false; }

	Spectrum eval_approx(const Vector3 dir_in, const Vector3 dir_out, const bool adjoint) {
		if (bsdf.emission() > 0)
			return 0;
		return bsdf.base_color() * abs(dir_out.z) / M_PI;
	}
	void eval(out MaterialEvalRecord r, const Vector3 dir_in, const Vector3 dir_out, const bool adjoint) {
		if (bsdf.emission() > 0) {
			r.f = 0;
			r.pdf_fwd = r.pdf_rev = 0;
			return;
		}
		if (dir_in.z * dir_out.z <= 0) {
			r.f = 0;
			r.pdf_fwd = 0;
			r.pdf_rev = 0;
		} else {
			r.f = bsdf.base_color() * abs(dir_out.z) / M_PI;
			r.pdf_fwd = cosine_hemisphere_pdfW(abs(dir_out.z));
			r.pdf_rev = cosine_hemisphere_pdfW(abs(dir_in.z));
		}
	}
	Spectrum sample(out MaterialSampleRecord r, const Vector3 rnd, const Vector3 dir_in, inout Spectrum beta, const bool adjoint) {
		if (bsdf.emission() > 0) {
			beta = 0;
			r.pdf_fwd = r.pdf_rev = 0;
			return 0;
		}
		r.dir_out = sample_cos_hemisphere(rnd.x, rnd.y);
		r.pdf_fwd = cosine_hemisphere_pdfW(r.dir_out.z);
		r.pdf_rev = cosine_hemisphere_pdfW(abs(dir_in.z));
		if (dir_in.z < 0) r.dir_out.z = -r.dir_out.z;
		const Spectrum f = bsdf.base_color() * abs(r.dir_out.z) / M_PI;
		beta *= f / r.pdf_fwd;
		r.eta = 0;
		r.roughness = 1;
		return f;
	}
#else
	bool is_specular() { return (bsdf.metallic() > 0.999 || bsdf.transmission() > 0.999) && bsdf.roughness() <= 1e-2; }

	Spectrum eval_approx(const Vector3 dir_in, const Vector3 dir_out, const bool adjoint) {
		if (bsdf.emission() > 0) {
			return 0;
		}
		const Real shininess = 2/pow2(bsdf.roughness() + 1e-6) - 2;
		if (dir_in.z * dir_out.z <= 0) {
			const Real local_eta = dir_in.z < 0 ? 1/bsdf.eta() : bsdf.eta();
			const Real spec = pow(abs(dot(refract(-dir_in, float3(0,0,sign(dir_in.z)), 1/local_eta), dir_out)), shininess);
			return sqrt(bsdf.base_color()) * abs(dir_out.z) * spec * bsdf.transmission() * (adjoint ? 1/(local_eta*local_eta) : 1);
		} else {
			const Real spec = pow(abs(dot(reflect(-dir_in, float3(0,0,sign(dir_in.z))), dir_out)), shininess);
			return bsdf.base_color() * abs(dir_out.z) * lerp(1 / M_PI, spec, bsdf.metallic());
		}
	}
	void eval(out MaterialEvalRecord r, const Vector3 dir_in, const Vector3 dir_out, const bool adjoint) {
		if (bsdf.emission() > 0) {
			r.f = 0;
			r.pdf_fwd = r.pdf_rev = 0;
			return;
		}

		const Real one_minus_metallic = 1 - bsdf.metallic();
		Real w_diffuse = (1 - bsdf.transmission()) * one_minus_metallic;
		Real w_metal = bsdf.metallic();
		Real w_glass = bsdf.transmission() * one_minus_metallic;
		Real w_clearcoat = 0.25 * bsdf.clearcoat();

		const Real local_eta = dir_in.z < 0 ? 1/bsdf.eta() : bsdf.eta();
		const bool transmit = dir_in.z * dir_out.z < 0;
		Vector3 h = normalize(transmit ? (dir_in + dir_out * local_eta) : (dir_in + dir_out));
		if (h.z * dir_in.z < 0) h = -h;
		const Real h_dot_in = dot(h, dir_in);
		const Real h_dot_out = dot(h, dir_out);

		const Real aspect = sqrt(1 - 0.9 * bsdf.anisotropic());
		const float2 alpha = max(0.0001, float2(bsdf.alpha() / aspect, bsdf.alpha() * aspect));
		const Real D = Dm(alpha.x, alpha.y, h);
		const Real G_in  = G1(alpha.x, alpha.y, dir_in);
		const Real G_out = G1(alpha.x, alpha.y, dir_out);
		const Real F = fresnel_dielectric(h_dot_in, local_eta);

		r.f = 0;
		r.pdf_fwd = 0;
		r.pdf_rev = 0;
		if (transmit) {
			if (w_glass > 0) {
				r.f = w_glass * disneyglass_eval_refract(bsdf.base_color(), F, D, G_in * G_out, dir_in.z, h_dot_in, h_dot_out, local_eta, adjoint);
				r.pdf_fwd = w_glass * disneyglass_refract_pdf(F, D, G_in, dir_in.z, h_dot_in, h_dot_out, local_eta);
				r.pdf_rev = w_glass * disneyglass_refract_pdf(fresnel_dielectric(h_dot_out, 1/local_eta), D, G_out, dir_out.z, h_dot_out, h_dot_in, 1/local_eta);
			}
		} else {
			if (w_glass > 0) {
				r.f += w_glass * disneyglass_eval_reflect(bsdf.base_color(), F, D, G_in * G_out, dir_in.z);
				r.pdf_fwd += w_glass * disneyglass_reflect_pdf(F, D, G_in, dir_in.z);
				r.pdf_rev += w_glass * disneyglass_reflect_pdf(fresnel_dielectric(h_dot_out, local_eta), D, G_out, dir_out.z);
			}
			if (w_metal > 0) {
				r.f += w_metal * disneymetal_eval(bsdf.base_color(), D, G_in * G_out, dir_in, dot(h, dir_out));
				r.pdf_fwd += w_metal * disneymetal_eval_pdf(D, G_in, dir_in.z);
				r.pdf_rev += w_metal * disneymetal_eval_pdf(D, G_out, dir_out.z);
			}
			if (w_clearcoat > 0) {
				const Real D_c = Dc((1 - bsdf.clearcoat_gloss())*0.1 + bsdf.clearcoat_gloss()*0.001, h.z);
				r.f += w_clearcoat * disneyclearcoat_eval(D_c, dir_in, dir_out, h, h_dot_out);
				r.pdf_fwd += w_clearcoat * disneyclearcoat_eval_pdf(D_c, h, h_dot_out);
				r.pdf_rev += w_clearcoat * disneyclearcoat_eval_pdf(D_c, h, h_dot_in);
			}
			if (w_diffuse > 0) {
				r.pdf_fwd += w_diffuse * cosine_hemisphere_pdfW(abs(dir_out.z));
				r.pdf_rev += w_diffuse * cosine_hemisphere_pdfW(abs(dir_in.z));
				r.f += w_diffuse * disneydiffuse_eval(bsdf, dir_in, dir_out);
			}
		}

		if ((BDPTDebugMode)gDebugMode == BDPTDebugMode::eApproxBSDF)
			r.f = eval_approx(dir_in, dir_out, adjoint);
	}
	Spectrum sample(out MaterialSampleRecord r, const Vector3 rnd, const Vector3 dir_in, inout Spectrum beta, const bool adjoint) {
		if (bsdf.emission() > 0) {
			beta = 0;
			r.pdf_fwd = r.pdf_rev = 0;
			return 0;
		}

		const Real one_minus_metallic = 1 - bsdf.metallic();
		Real w_diffuse = (1 - bsdf.transmission()) * one_minus_metallic;
		Real w_metal = bsdf.metallic();
		Real w_glass = bsdf.transmission() * one_minus_metallic;
		Real w_clearcoat = 0.25 * bsdf.clearcoat();

		const Real aspect = sqrt(1 - 0.9 * bsdf.anisotropic());
		const float2 alpha = max(0.0001, float2(bsdf.alpha() / aspect, bsdf.alpha() * aspect));
		const Real alpha_c = (1 - bsdf.clearcoat_gloss())*0.1 + bsdf.clearcoat_gloss()*0.001;

		const Real local_eta = dir_in.z < 0 ? 1/bsdf.eta() : bsdf.eta();
		const Real G_in = G1(alpha.x, alpha.y, dir_in);

		Vector3 h;
		Real h_dot_in;
		Real D;
		Real F;

		r.eta = 0;
		r.roughness = bsdf.roughness();

		// sample direction
		if (rnd.z < w_glass + w_metal) {
			// glass/metal
			h = sample_visible_normals(dir_in, alpha.x, alpha.y, rnd.xy);
			h_dot_in = dot(h, dir_in);
			D = Dm(alpha.x, alpha.y, h);
			F = fresnel_dielectric(h_dot_in, local_eta);
			if (rnd.z < w_glass) {
				const Real h_dot_out_sq = 1 - (1 - h_dot_in * h_dot_in) / (local_eta * local_eta);
				if (h_dot_out_sq <= 0 || rnd.z/w_glass <= F) {
					// Reflection
					r.dir_out = reflect(-dir_in, h);
				} else {
					// Refraction
					r.dir_out = refract(-dir_in, h, 1/local_eta);
					r.eta = local_eta;
					const Real G_out = G1(alpha.x, alpha.y, r.dir_out);
					const Real h_dot_out = dot(h, r.dir_out);
					r.pdf_fwd = w_glass * disneyglass_refract_pdf(F, D, G_in, dir_in.z, h_dot_in, h_dot_out, local_eta);
					r.pdf_rev = w_glass * disneyglass_refract_pdf(fresnel_dielectric(h_dot_out, 1/local_eta), D, G_out, r.dir_out.z, h_dot_out, h_dot_in, 1/local_eta);
					const Spectrum f = w_glass * disneyglass_eval_refract(bsdf.base_color(), F, D, G_in * G_out, dir_in.z, h_dot_in, h_dot_out, local_eta, adjoint);
					beta *= f / r.pdf_fwd;
					return f; // other layers are all 0 when transmitting
				}
			} else {
				r.dir_out = reflect(-dir_in, h);
			}
		} else {
			if (rnd.z < w_glass + w_metal + w_clearcoat) {
				// clearcoat
				// importance sample Dc
				const Real alpha2 = alpha_c*alpha_c;
				const Real cos_phi = sqrt((1 - pow(alpha2, 1 - rnd.x)) / (1 - alpha2));
				const Real sin_phi = sqrt(1 - max(cos_phi*cos_phi, Real(0)));
				const Real theta = 2*M_PI * rnd.y;
				h = Vector3(sin_phi*cos(theta), sin_phi*sin(theta), cos_phi);
				if (dir_in.z < 0) h = -h;
				r.dir_out = reflect(-dir_in, h);
				r.roughness = alpha_c;
			} else {
				// diffuse
				r.dir_out = sample_cos_hemisphere(rnd.x, rnd.y);
				if (dir_in.z < 0) r.dir_out = -r.dir_out;
				r.roughness = 1;
				h = normalize(dir_in + r.dir_out);
			}
			h_dot_in = dot(h, dir_in);
			D = Dm(alpha.x, alpha.y, h);
			F = fresnel_dielectric(h_dot_in, local_eta);
		}

		const Real G_out = G1(alpha.x, alpha.y, r.dir_out);
		const Real h_dot_out = dot(h, r.dir_out);

		r.pdf_fwd = 0;
		r.pdf_rev = 0;
		Spectrum f = 0;

		// evaluate contribution and pdfs
		if (w_glass > 0) {
			r.pdf_fwd += w_glass * disneyglass_reflect_pdf(F, D, G_in, dir_in.z);
			r.pdf_rev += w_glass * disneyglass_reflect_pdf(fresnel_dielectric(h_dot_out, local_eta), D, G_out, r.dir_out.z);
			f += w_glass * disneyglass_eval_reflect(bsdf.base_color(), F, D, G_in * G_out, dir_in.z);
		}

		if (w_metal > 0) {
			r.pdf_fwd += w_metal * disneymetal_eval_pdf(D, G_in, dir_in.z);
			r.pdf_rev += w_metal * disneymetal_eval_pdf(D, G_out, r.dir_out.z);
			f += w_metal * disneymetal_eval(bsdf.base_color(), D, G_in * G_out, dir_in, h_dot_out);
		}

		if (w_clearcoat > 0) {
			const Real D_c = Dc(alpha_c, h.z);
			r.pdf_fwd += w_clearcoat * disneyclearcoat_eval_pdf(D_c, h, h_dot_out);
			r.pdf_rev += w_clearcoat * disneyclearcoat_eval_pdf(D_c, h, h_dot_in);
			f += w_clearcoat * disneyclearcoat_eval(D_c, dir_in, r.dir_out, h, h_dot_out);
		}

		if (w_diffuse > 0) {
			r.pdf_fwd += w_diffuse * cosine_hemisphere_pdfW(abs(r.dir_out.z));
			r.pdf_rev += w_diffuse * cosine_hemisphere_pdfW(abs(dir_in.z));
			f += w_diffuse * disneydiffuse_eval(bsdf, dir_in, r.dir_out);
		}

		if ((BDPTDebugMode)gDebugMode == BDPTDebugMode::eApproxBSDF)
			f = eval_approx(dir_in, r.dir_out, adjoint);

		beta *= f / r.pdf_fwd;
		return f;
	}
#endif
};