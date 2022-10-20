Real disneymetal_eval_pdf(const Real D, const Real G_in, const Real cos_theta_in) {
	return D * G_in / (4 * abs(cos_theta_in));
}

Spectrum disneymetal_eval(const Spectrum base_color, const Real D, const Real G, const Vector3 dir_in, const Real h_dot_out) {
	return base_color * schlick_fresnel3(base_color, abs(h_dot_out)) * D * G / (4 * abs(dir_in.z));
}

void disneymetal_eval(const DisneyMaterialData bsdf, out MaterialEvalRecord r, const Vector3 dir_in, const Vector3 dir_out, const bool adjoint) {
	if (dir_in.z * dir_out.z < 0) {
		r.f = 0;
		r.pdf_fwd = r.pdf_rev = 0;
		return; // No light through the surface
	}
	const Real aspect = sqrt(1 - 0.9 * bsdf.anisotropic());
	const float2 alpha = max(0.0001, float2(bsdf.alpha() / aspect, bsdf.alpha() * aspect));

	const Vector3 h = normalize(dir_in + dir_out);
	const Real D = Dm(alpha.x, alpha.y, h);
	const Real G_in = G1(alpha.x, alpha.y, dir_in);
	const Real G_out = G1(alpha.x, alpha.y, dir_out);
	r.f = disneymetal_eval(bsdf.base_color(), D, G_in * G_out, dir_in, dot(h, dir_out));
	r.pdf_fwd = disneymetal_eval_pdf(D, G_in, dir_in.z);
	r.pdf_rev = disneymetal_eval_pdf(D, G_out, dir_out.z);
}

Spectrum disneymetal_sample(const DisneyMaterialData bsdf, out MaterialSampleRecord r, const Vector3 rnd, const Vector3 dir_in, inout Spectrum beta, const bool adjoint) {
	const Real aspect = sqrt(1 - 0.9 * bsdf.anisotropic());
	const float2 alpha = max(0.0001, float2(bsdf.alpha() / aspect, bsdf.alpha() * aspect));

	const Vector3 h = sample_visible_normals(dir_in, alpha.x, alpha.y, rnd.xy);
	r.dir_out = reflect(-dir_in, h);
	const Real D = Dm(alpha.x, alpha.y, h);
	const Real G_in = G1(alpha.x, alpha.y, dir_in);
	const Real G_out = G1(alpha.x, alpha.y, r.dir_out);
	r.pdf_fwd = disneymetal_eval_pdf(D, G_in, dir_in.z);
	r.pdf_rev = disneymetal_eval_pdf(D, G_out, r.dir_out.z);
	r.eta = 0;
	r.roughness = bsdf.roughness();
	const Spectrum f = disneymetal_eval(bsdf.base_color(), D, G_in * G_out, dir_in, dot(h, r.dir_out));
	beta *= f / r.pdf_fwd;
	return f;
}