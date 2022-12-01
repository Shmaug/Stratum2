#include "bsdf.hlsli"
#include "common/microfacet.hlsli"

float Dm(const float alpha_x, const float alpha_y, const float3 h_l) {
    const float alpha_x2 = alpha_x * alpha_x;
    const float alpha_y2 = alpha_y * alpha_y;
    const float3 h_l2 = h_l * h_l;
    const float hh = h_l2.x/alpha_x2 + h_l2.y/alpha_y2 + h_l2.z;
    return 1 / (M_PI * alpha_x * alpha_y * hh*hh);
}
float G1(const float alpha_x, const float alpha_y, const float3 w_l) {
    const float alpha_x2 = alpha_x * alpha_x;
    const float alpha_y2 = alpha_y * alpha_y;
    const float3 w_l2 = w_l * w_l;
    const float lambda = (sqrt(1 + (w_l2.x*alpha_x2 + w_l2.y*alpha_y2) / w_l2.z) - 1) / 2;
    return 1 / (1 + lambda);
}
float R0(const float eta) {
    const float num = eta - 1;
    const float denom = eta + 1;
    return (num*num) / (denom*denom);
}

float Dc(const float alpha_g, const float h_lz) {
    const float alpha_g2 = alpha_g * alpha_g;
    return (alpha_g2 - 1) / (M_PI * log(alpha_g2)*(1 + (alpha_g2 - 1)*h_lz*h_lz));
}
float Gc(const float3 w_l) {
    const float wx = w_l.x*0.25;
    const float wy = w_l.y*0.25;
    const float lambda = (sqrt(1 + (wx*wx + wy*wy)/(w_l.z*w_l.z)) - 1) / 2;
    return 1 / (1 + lambda);
}

#include "compat/disney_data.h"
#include "disney_diffuse.hlsli"
#include "disney_metal.hlsli"
#include "disney_glass.hlsli"
#include "disney_clearcoat.hlsli"
//#include "disney_sheen.hlsli"

#include "compat/scene.h"
#include "compat/image_value.h"

struct DisneyMaterial : BSDF {
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
		load(scene, sd.materialAddress, sd.mTexcoord, sd.mTexcoordScreenSize, sd.mPackedShadingNormal, sd.mPackedTangent, sd.isBitangentFlipped());
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

	bool isSingular() { return (bsdf.metallic() > 0.999 || bsdf.transmission() > 0.999) && bsdf.roughness() <= 1e-2; }

	MaterialEvalRecord evaluate(const float3 dirIn, const float3 dirOut, const bool adjoint) {
		MaterialEvalRecord r;
		if (bsdf.emission() > 0) {
			r.mReflectance = 0;
			r.mFwdPdfW = r.mRevPdfW = 0;
			return r;
		}

		const float one_minus_metallic = 1 - bsdf.metallic();
		float w_diffuse = (1 - bsdf.transmission()) * one_minus_metallic;
		float w_metal = bsdf.metallic();
		float w_glass = bsdf.transmission() * one_minus_metallic;
		float w_clearcoat = 0.25 * bsdf.clearcoat();

		const float local_eta = dirIn.z < 0 ? 1/bsdf.eta() : bsdf.eta();
		const bool transmit = dirIn.z * dirOut.z < 0;
		float3 h = normalize(transmit ? (dirIn + dirOut * local_eta) : (dirIn + dirOut));
		if (h.z * dirIn.z < 0) h = -h;
		const float h_dot_in = dot(h, dirIn);
		const float h_dot_out = dot(h, dirOut);

		const float aspect = sqrt(1 - 0.9 * bsdf.anisotropic());
		const float2 alpha = max(0.0001, float2(bsdf.alpha() / aspect, bsdf.alpha() * aspect));
		const float D = Dm(alpha.x, alpha.y, h);
		const float G_in  = G1(alpha.x, alpha.y, dirIn);
		const float G_out = G1(alpha.x, alpha.y, dirOut);
		const float F = fresnel_dielectric(h_dot_in, local_eta);

		r.mReflectance = 0;
		r.mFwdPdfW = 0;
		r.mRevPdfW = 0;
		if (transmit) {
			if (w_glass > 0) {
				r.mReflectance = w_glass * disneyglass_eval_refract(bsdf.baseColor(), F, D, G_in * G_out, dirIn.z, h_dot_in, h_dot_out, local_eta, adjoint);
				r.mFwdPdfW     = w_glass * disneyglass_refract_pdf(F, D, G_in, dirIn.z, h_dot_in, h_dot_out, local_eta);
				r.mRevPdfW     = w_glass * disneyglass_refract_pdf(fresnel_dielectric(h_dot_out, 1/local_eta), D, G_out, dirOut.z, h_dot_out, h_dot_in, 1/local_eta);
			}
		} else {
			if (w_glass > 0) {
				r.mReflectance += w_glass * disneyglass_eval_reflect(bsdf.baseColor(), F, D, G_in * G_out, dirIn.z);
				r.mFwdPdfW     += w_glass * disneyglass_reflect_pdf(F, D, G_in, dirIn.z);
				r.mRevPdfW     += w_glass * disneyglass_reflect_pdf(fresnel_dielectric(h_dot_out, local_eta), D, G_out, dirOut.z);
			}
			if (w_metal > 0) {
				r.mReflectance += w_metal * disneymetal_eval(bsdf.baseColor(), D, G_in * G_out, dirIn, dot(h, dirOut));
				r.mFwdPdfW     += w_metal * disneymetal_eval_pdf(D, G_in, dirIn.z);
				r.mRevPdfW     += w_metal * disneymetal_eval_pdf(D, G_out, dirOut.z);
			}
			if (w_clearcoat > 0) {
				const float D_c = Dc((1 - bsdf.clearcoatGloss())*0.1 + bsdf.clearcoatGloss()*0.001, h.z);
				r.mReflectance += w_clearcoat * disneyclearcoat_eval(D_c, dirIn, dirOut, h, h_dot_out);
				r.mFwdPdfW     += w_clearcoat * disneyclearcoat_eval_pdf(D_c, h, h_dot_out);
				r.mRevPdfW     += w_clearcoat * disneyclearcoat_eval_pdf(D_c, h, h_dot_in);
			}
			if (w_diffuse > 0) {
				r.mReflectance += w_diffuse * disneydiffuse_eval(bsdf, dirIn, dirOut);
				r.mFwdPdfW     += w_diffuse * cosine_hemisphere_pdfW(abs(dirOut.z));
				r.mRevPdfW     += w_diffuse * cosine_hemisphere_pdfW(abs(dirIn.z));
			}
		}
		return r;
	}

	MaterialSampleRecord sample(const float3 rnd, const float3 dirIn, const bool adjoint) {
		MaterialSampleRecord r;
		if (bsdf.emission() > 0) {
			r.mFwdPdfW = r.mRevPdfW = 0;
			return r;
		}

		const float one_minus_metallic = 1 - bsdf.metallic();
		float w_diffuse = (1 - bsdf.transmission()) * one_minus_metallic;
		float w_metal = bsdf.metallic();
		float w_glass = bsdf.transmission() * one_minus_metallic;
		float w_clearcoat = 0.25 * bsdf.clearcoat();

		const float aspect = sqrt(1 - 0.9 * bsdf.anisotropic());
		const float2 alpha = max(0.0001, float2(bsdf.alpha() / aspect, bsdf.alpha() * aspect));
		const float alpha_c = (1 - bsdf.clearcoatGloss())*0.1 + bsdf.clearcoatGloss()*0.001;

		const float local_eta = dirIn.z < 0 ? 1/bsdf.eta() : bsdf.eta();
		const float G_in = G1(alpha.x, alpha.y, dirIn);

		float3 h;
		float h_dot_in;
		float D;
		float F;

		r.mEta = 0;
		r.mRoughness = bsdf.roughness();

		// sample direction

		if (rnd.z < w_glass + w_metal) {
			// glass or metal
			// importance sample Dm
			h = sample_visible_normals(dirIn, alpha.x, alpha.y, rnd.xy);
			h_dot_in = dot(h, dirIn);
			D = Dm(alpha.x, alpha.y, h);
			F = fresnel_dielectric(h_dot_in, local_eta);
			if (rnd.z < w_glass) {
				const float h_dot_out_sq = 1 - (1 - h_dot_in * h_dot_in) / (local_eta * local_eta);
				if (h_dot_out_sq <= 0 || rnd.z/w_glass <= F) {
					// Reflection
					r.mDirection = reflect(-dirIn, h);
				} else {
					// Refraction
					r.mDirection = refract(-dirIn, h, 1/local_eta);
					r.mEta = local_eta;
					const float G_out = G1(alpha.x, alpha.y, r.mDirection);
					const float h_dot_out = dot(h, r.mDirection);
					r.mFwdPdfW = w_glass * disneyglass_refract_pdf(F, D, G_in, dirIn.z, h_dot_in, h_dot_out, local_eta);
					r.mRevPdfW = w_glass * disneyglass_refract_pdf(fresnel_dielectric(h_dot_out, 1/local_eta), D, G_out, r.mDirection.z, h_dot_out, h_dot_in, 1/local_eta);
					//const float3 f = w_glass * disneyglass_eval_refract(bsdf.baseColor(), F, D, G_in * G_out, dirIn.z, h_dot_in, h_dot_out, local_eta, adjoint);
					return r; // other layers are all 0 when transmitting
				}
			} else {
				r.mDirection = reflect(-dirIn, h);
			}
		} else {
			// clearcoat or diffuse
			if (rnd.z < w_glass + w_metal + w_clearcoat) {
				// clearcoat
				// importance sample Dc
				const float alpha2 = alpha_c*alpha_c;
				const float cos_phi = sqrt((1 - pow(alpha2, 1 - rnd.x)) / (1 - alpha2));
				const float sin_phi = sqrt(1 - max(cos_phi*cos_phi, float(0)));
				const float theta = 2*M_PI * rnd.y;
				h = float3(sin_phi*cos(theta), sin_phi*sin(theta), cos_phi);
				if (dirIn.z < 0) h = -h;
				r.mDirection = reflect(-dirIn, h);
				r.mRoughness = alpha_c;
			} else {
				// diffuse
				r.mDirection = sample_cos_hemisphere(rnd.x, rnd.y);
				if (dirIn.z < 0) r.mDirection = -r.mDirection;
				r.mRoughness = 1;
				h = normalize(dirIn + r.mDirection);
			}
			// compute values used in pdf computations below
			h_dot_in = dot(h, dirIn);
			D = Dm(alpha.x, alpha.y, h);
			F = fresnel_dielectric(h_dot_in, local_eta);
		}

		const float G_out = G1(alpha.x, alpha.y, r.mDirection);
		const float h_dot_out = dot(h, r.mDirection);

		r.mFwdPdfW = 0;
		r.mRevPdfW = 0;

		// evaluate contribution and pdfs
		if (w_glass > 0) {
			r.mFwdPdfW += w_glass * disneyglass_reflect_pdf(F, D, G_in, dirIn.z);
			r.mRevPdfW += w_glass * disneyglass_reflect_pdf(fresnel_dielectric(h_dot_out, local_eta), D, G_out, r.mDirection.z);
		}

		if (w_metal > 0) {
			r.mFwdPdfW += w_metal * disneymetal_eval_pdf(D, G_in, dirIn.z);
			r.mRevPdfW += w_metal * disneymetal_eval_pdf(D, G_out, r.mDirection.z);
		}

		if (w_clearcoat > 0) {
			const float D_c = Dc(alpha_c, h.z);
			r.mFwdPdfW += w_clearcoat * disneyclearcoat_eval_pdf(D_c, h, h_dot_out);
			r.mRevPdfW += w_clearcoat * disneyclearcoat_eval_pdf(D_c, h, h_dot_in);
		}

		if (w_diffuse > 0) {
			r.mFwdPdfW += w_diffuse * cosine_hemisphere_pdfW(abs(r.mDirection.z));
			r.mRevPdfW += w_diffuse * cosine_hemisphere_pdfW(abs(dirIn.z));
		}

		return r;
	}
};