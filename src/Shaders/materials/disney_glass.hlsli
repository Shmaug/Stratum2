float disneyglass_reflect_pdf(const float F, const float D, const float G_in, const float cos_theta_in) {
	return (F * D * G_in) / (4 * abs(cos_theta_in));
}
float disneyglass_refract_pdf(const float F, const float D, const float G_in, const float cos_theta_in, const float h_dot_in, const float h_dot_out, const float eta) {
	const float sqrt_denom = h_dot_in + eta * h_dot_out;
	const float dh_dout = eta * eta * h_dot_out / (sqrt_denom * sqrt_denom);
	return (1 - F) * D * G_in * abs(dh_dout * h_dot_in / cos_theta_in);
}

float3 disneyglass_eval_reflect(const float3 baseColor, const float F, const float D, const float G, const float cos_theta_in) {
	return baseColor * (F * D * G) / (4 * abs(cos_theta_in));
}
float3 disneyglass_eval_refract(const float3 baseColor, const float F, const float D, const float G, const float cos_theta_in, const float h_dot_in, const float h_dot_out, const float local_eta, const bool adjoint) {
	// Snell-Descartes law predicts that the light will contract/expand
	// due to the different index of refraction. So the normal BSDF needs
	// to scale with 1/eta^2. However, the "adjoint" of the BSDF does not have
	// the eta term. This is due to the non-reciprocal nature of the index of refraction:
	// f(wi -> wo) / eta_o^2 = f(wo -> wi) / eta_i^2
	// thus f(wi -> wo) = f(wo -> wi) (eta_o / eta_i)^2
	// The adjoint of a BSDF is defined as swapping the parameter, and
	// this cancels out the eta term.
	// See Chapter 5 of Eric Veach's thesis "Robust Monte Carlo Methods for Light Transport Simulation"
	// for more details.
	const float sqrt_denom = h_dot_in + local_eta * h_dot_out;
	const float eta_factor = adjoint ? (1 / (local_eta * local_eta)) : 1;
	return sqrt(baseColor) * (eta_factor * (1 - F) * D * G * abs(h_dot_out * h_dot_in)) / (abs(cos_theta_in) * sqrt_denom * sqrt_denom);
}

MaterialEvalRecord disneyglass_eval(const DisneyMaterialData bsdf, const float3 dirIn, const float3 dirOut, const bool adjoint) {
	MaterialEvalRecord r;
	const float local_eta = dirIn.z < 0 ? 1/bsdf.eta() : bsdf.eta();
	const bool reflect = dirIn.z * dirOut.z > 0;

    float3 h = normalize(reflect ? (dirIn + dirOut) : (dirIn + dirOut * local_eta));
	if (h.z * dirIn.z < 0) h = -h;

	const float aspect = sqrt(1 - 0.9 * bsdf.anisotropic());
	const float2 alpha = max(0.0001, float2(bsdf.alpha() / aspect, bsdf.alpha() * aspect));

	const float h_dot_in = dot(h, dirIn);
	const float h_dot_out = dot(h, dirOut);
	const float F = fresnel_dielectric(h_dot_in, local_eta);
	const float D = Dm(alpha.x, alpha.y, h);
	const float G_in  = G1(alpha.x, alpha.y, dirIn);
	const float G_out = G1(alpha.x, alpha.y, dirOut);
	if (reflect) {
		r.mReflectance = disneyglass_eval_reflect(bsdf.baseColor(), F, D, G_in * G_out, dirIn.z);
		r.mFwdPdfW = disneyglass_reflect_pdf(F, D, G_in, dirIn.z);
		r.mRevPdfW = disneyglass_reflect_pdf(fresnel_dielectric(h_dot_out, local_eta), D, G_out, dirOut.z);
	} else {
		r.mReflectance = disneyglass_eval_refract(bsdf.baseColor(), F, D, G_in * G_out, dirIn.z, h_dot_in, h_dot_out, local_eta, adjoint);
		r.mFwdPdfW = disneyglass_refract_pdf(F, D, G_in, dirIn.z, h_dot_in, h_dot_out, local_eta);
		r.mRevPdfW = disneyglass_refract_pdf(fresnel_dielectric(h_dot_out, 1/local_eta), D, G_out, dirOut.z, h_dot_out, h_dot_in, 1/local_eta);
	}
	return r;
}

MaterialSampleRecord disneyglass_sample(const DisneyMaterialData bsdf, const float3 rnd, const float3 dirIn, const bool adjoint) {
	MaterialSampleRecord r;
	const float local_eta = dirIn.z < 0 ? 1/bsdf.eta() : bsdf.eta();
	const float aspect = sqrt(1 - 0.9 * bsdf.anisotropic());
	const float2 alpha = max(0.0001, float2(bsdf.alpha() / aspect, bsdf.alpha() * aspect));
	r.mRoughness = bsdf.roughness();

	const float3 h = sampleVisibleNormals(dirIn, alpha.x, alpha.y, rnd.xy);

	const float h_dot_in = dot(h, dirIn);
	const float F = fresnel_dielectric(h_dot_in, local_eta);
	const float D = Dm(alpha.x, alpha.y, h);
	const float G_in = G1(alpha.x, alpha.y, dirIn);
	const float h_dot_out_sq = 1 - (1 - h_dot_in * h_dot_in) / (local_eta * local_eta);
	if (h_dot_out_sq <= 0 || rnd.z <= F) {
		// Reflection
		r.mDirection = reflect(-dirIn, h);
		const float G_out = G1(alpha.x, alpha.y, r.mDirection);
		const float h_dot_out = dot(h, r.mDirection);
		r.mFwdPdfW = disneyglass_reflect_pdf(F, D, G_in, dirIn.z);
		r.mRevPdfW = disneyglass_reflect_pdf(fresnel_dielectric(h_dot_out, local_eta), D, G_out, r.mDirection.z);
		r.mEta = 0;
	} else {
		// Refraction
		r.mDirection = refract(-dirIn, h, 1/local_eta);
		const float G_out = G1(alpha.x, alpha.y, r.mDirection);
		const float h_dot_out = dot(h, r.mDirection);
		r.mFwdPdfW = disneyglass_refract_pdf(F, D, G_in, dirIn.z, h_dot_in, h_dot_out, local_eta);
		r.mRevPdfW = disneyglass_refract_pdf(fresnel_dielectric(h_dot_out, 1/local_eta), D, G_out, r.mDirection.z, h_dot_out, h_dot_in, 1/local_eta);
		r.mEta = local_eta;
	}
	return r;
}