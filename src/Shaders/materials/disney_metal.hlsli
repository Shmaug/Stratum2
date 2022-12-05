float disneymetal_eval_pdf(const float D, const float G_in, const float cos_theta_in) {
	return D * G_in / (4 * abs(cos_theta_in));
}

float3 disneymetal_eval(const float3 baseColor, const float D, const float G, const float3 dirIn, const float h_dot_out) {
	return baseColor * schlick_fresnel3(baseColor, abs(h_dot_out)) * D * G / (4 * abs(dirIn.z));
}

MaterialEvalRecord disneymetal_eval(const DisneyMaterialData bsdf, const float3 dirIn, const float3 dirOut, const bool adjoint) {
	MaterialEvalRecord r;
	if (dirIn.z * dirOut.z < 0) {
		r.mReflectance = 0;
		r.mFwdPdfW = r.mRevPdfW = 0;
		return r; // No light through the surface
	}
	const float aspect = sqrt(1 - 0.9 * bsdf.anisotropic());
	const float2 alpha = max(0.0001, float2(bsdf.alpha() / aspect, bsdf.alpha() * aspect));

	const float3 h = normalize(dirIn + dirOut);
	const float D = Dm(alpha.x, alpha.y, h);
	const float G_in = G1(alpha.x, alpha.y, dirIn);
	const float G_out = G1(alpha.x, alpha.y, dirOut);
	r.mReflectance = disneymetal_eval(bsdf.baseColor(), D, G_in * G_out, dirIn, dot(h, dirOut));
	r.mFwdPdfW = disneymetal_eval_pdf(D, G_in, dirIn.z);
    r.mRevPdfW = disneymetal_eval_pdf(D, G_out, dirOut.z);
    return r;
}

MaterialSampleRecord disneymetal_sample(const DisneyMaterialData bsdf, const float3 rnd, const float3 dirIn, const bool adjoint) {
	const float aspect = sqrt(1 - 0.9 * bsdf.anisotropic());
	const float2 alpha = max(0.0001, float2(bsdf.alpha() / aspect, bsdf.alpha() * aspect));

	const float3 h = sample_visible_normals(dirIn, alpha.x, alpha.y, rnd.xy);

	MaterialSampleRecord r;
	r.mDirection = reflect(-dirIn, h);
	const float D = Dm(alpha.x, alpha.y, h);
	const float G_in = G1(alpha.x, alpha.y, dirIn);
	const float G_out = G1(alpha.x, alpha.y, r.mDirection);
	r.mFwdPdfW = disneymetal_eval_pdf(D, G_in, dirIn.z);
	r.mRevPdfW = disneymetal_eval_pdf(D, G_out, r.mDirection.z);
	r.mEta = 0;
	r.mRoughness = bsdf.roughness();
	return r;
}