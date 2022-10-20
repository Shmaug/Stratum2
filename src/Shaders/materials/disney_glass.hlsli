Real disneyglass_reflect_pdf(const Real F, const Real D, const Real G_in, const Real cos_theta_in) {
	return (F * D * G_in) / (4 * abs(cos_theta_in));
}
Real disneyglass_refract_pdf(const Real F, const Real D, const Real G_in, const Real cos_theta_in, const Real h_dot_in, const Real h_dot_out, const Real eta) {
	const Real sqrt_denom = h_dot_in + eta * h_dot_out;
	const Real dh_dout = eta * eta * h_dot_out / (sqrt_denom * sqrt_denom);
	return (1 - F) * D * G_in * abs(dh_dout * h_dot_in / cos_theta_in);
}

Spectrum disneyglass_eval_reflect(const Spectrum base_color, const Real F, const Real D, const Real G, const Real cos_theta_in) {
	return base_color * (F * D * G) / (4 * abs(cos_theta_in));
}
Spectrum disneyglass_eval_refract(const Spectrum base_color, const Real F, const Real D, const Real G, const Real cos_theta_in, const Real h_dot_in, const Real h_dot_out, const Real local_eta, const bool adjoint) {
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
	const Real sqrt_denom = h_dot_in + local_eta * h_dot_out;
	const Real eta_factor = adjoint ? (1 / (local_eta * local_eta)) : 1;
	return sqrt(base_color) * (eta_factor * (1 - F) * D * G * abs(h_dot_out * h_dot_in)) / (abs(cos_theta_in) * sqrt_denom * sqrt_denom);
}

void disneyglass_eval(const DisneyMaterialData bsdf, out MaterialEvalRecord r, const Vector3 dir_in, const Vector3 dir_out, const bool adjoint) {
	const Real local_eta = dir_in.z < 0 ? 1/bsdf.eta() : bsdf.eta();
	const bool reflect = dir_in.z * dir_out.z > 0;

    Vector3 h = normalize(reflect ? (dir_in + dir_out) : (dir_in + dir_out * local_eta));
	if (h.z * dir_in.z < 0) h = -h;

	const Real aspect = sqrt(1 - 0.9 * bsdf.anisotropic());
	const float2 alpha = max(0.0001, float2(bsdf.alpha() / aspect, bsdf.alpha() * aspect));

	const Real h_dot_in = dot(h, dir_in);
	const Real h_dot_out = dot(h, dir_out);
	const Real F = fresnel_dielectric(h_dot_in, local_eta);
	const Real D = Dm(alpha.x, alpha.y, h);
	const Real G_in  = G1(alpha.x, alpha.y, dir_in);
	const Real G_out = G1(alpha.x, alpha.y, dir_out);
	if (reflect) {
		r.f = disneyglass_eval_reflect(bsdf.base_color(), F, D, G_in * G_out, dir_in.z);
		r.pdf_fwd = disneyglass_reflect_pdf(F, D, G_in, dir_in.z);
		r.pdf_rev = disneyglass_reflect_pdf(fresnel_dielectric(h_dot_out, local_eta), D, G_out, dir_out.z);
	} else {
		r.f = disneyglass_eval_refract(bsdf.base_color(), F, D, G_in * G_out, dir_in.z, h_dot_in, h_dot_out, local_eta, adjoint);
		r.pdf_fwd = disneyglass_refract_pdf(F, D, G_in, dir_in.z, h_dot_in, h_dot_out, local_eta);
		r.pdf_rev = disneyglass_refract_pdf(fresnel_dielectric(h_dot_out, 1/local_eta), D, G_out, dir_out.z, h_dot_out, h_dot_in, 1/local_eta);
	}
}

Spectrum disneyglass_sample(const DisneyMaterialData bsdf, out MaterialSampleRecord r, const Vector3 rnd, const Vector3 dir_in, inout Spectrum beta, const bool adjoint) {
	const Real local_eta = dir_in.z < 0 ? 1/bsdf.eta() : bsdf.eta();
	const Real aspect = sqrt(1 - 0.9 * bsdf.anisotropic());
	const float2 alpha = max(0.0001, float2(bsdf.alpha() / aspect, bsdf.alpha() * aspect));
	r.roughness = bsdf.roughness();

	const Vector3 h = sample_visible_normals(dir_in, alpha.x, alpha.y, rnd.xy);

	const Real h_dot_in = dot(h, dir_in);
	const Real F = fresnel_dielectric(h_dot_in, local_eta);
	const Real D = Dm(alpha.x, alpha.y, h);
	const Real G_in = G1(alpha.x, alpha.y, dir_in);
	const Real h_dot_out_sq = 1 - (1 - h_dot_in * h_dot_in) / (local_eta * local_eta);
	if (h_dot_out_sq <= 0 || rnd.z <= F) {
		// Reflection
		r.dir_out = reflect(-dir_in, h);
		const Real G_out = G1(alpha.x, alpha.y, r.dir_out);
		const Real h_dot_out = dot(h, r.dir_out);
		r.pdf_fwd = disneyglass_reflect_pdf(F, D, G_in, dir_in.z);
		r.pdf_rev = disneyglass_reflect_pdf(fresnel_dielectric(h_dot_out, local_eta), D, G_out, r.dir_out.z);
		r.eta = 0;

		const Spectrum f = disneyglass_eval_reflect(bsdf.base_color(), F, D, G_in * G_out, dir_in.z);
		beta *= f / r.pdf_fwd;
		return f;
	} else {
		// Refraction
		r.dir_out = refract(-dir_in, h, 1/local_eta);
		const Real G_out = G1(alpha.x, alpha.y, r.dir_out);
		const Real h_dot_out = dot(h, r.dir_out);
		r.pdf_fwd = disneyglass_refract_pdf(F, D, G_in, dir_in.z, h_dot_in, h_dot_out, local_eta);
		r.pdf_rev = disneyglass_refract_pdf(fresnel_dielectric(h_dot_out, 1/local_eta), D, G_out, r.dir_out.z, h_dot_out, h_dot_in, 1/local_eta);
		r.eta = local_eta;

		const Spectrum f = disneyglass_eval_refract(bsdf.base_color(), F, D, G_in * G_out, dir_in.z, h_dot_in, h_dot_out, local_eta, adjoint);
		beta *= f / r.pdf_fwd;
		return f;
	}
}