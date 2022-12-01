float disneyclearcoat_eval_pdf(const float D, const float3 h, const float hdotwo) {
    return D * abs(h.z) / (4*abs(hdotwo));
}

float3 disneyclearcoat_eval(const float D, const float3 dirIn, const float3 dirOut, const float3 h, const float hdotwo) {
    const float eta = 1.5;
    const float Fc = schlick_fresnel1(R0(eta), hdotwo);
    return Fc * D * Gc(dirIn) * Gc(dirOut) / (4 * abs(dirIn.z));
}

MaterialEvalRecord disneyclearcoat_eval(const DisneyMaterialData bsdf, const float3 dirIn, const float3 dirOut, const bool adjoint) {
	MaterialEvalRecord r;
	if (dirIn.z * dirOut.z < 0) {
		r.mReflectance = 0;
		r.mFwdPdfW = r.mRevPdfW = 0;
		return r; // No light through the surface
	}
    const float3 h = normalize(dirIn + dirOut);
    const float hdotwo = abs(dot(h, dirOut));

    const float alpha = (1 - bsdf.clearcoatGloss())*0.1 + bsdf.clearcoatGloss()*0.001;
	const float D = Dc(alpha, h.z);

	r.mReflectance = disneyclearcoat_eval(D, dirIn, dirOut, h, hdotwo);
	r.mFwdPdfW = disneyclearcoat_eval_pdf(D, h, hdotwo);
	r.mRevPdfW = disneyclearcoat_eval_pdf(D, h, dot(h, dirIn));
	return r;
}
MaterialSampleRecord disneyclearcoat_sample(const DisneyMaterialData bsdf, const float3 rnd, const float3 dirIn, const bool adjoint) {
    const float alpha = (1 - bsdf.clearcoatGloss())*0.1 + bsdf.clearcoatGloss()*0.001;

    const float alpha2 = alpha*alpha;
    const float cos_phi = sqrt((1 - pow(alpha2, 1 - rnd.x)) / (1 - alpha2));
    const float sin_phi = sqrt(1 - max(cos_phi*cos_phi, float(0)));
    const float theta = 2*M_PI * rnd.y;
    float3 h = float3(sin_phi*cos(theta), sin_phi*sin(theta), cos_phi);
	if (dirIn.z < 0) h = -h;

	const float D = Dc(alpha, h.z);

	MaterialSampleRecord r;
	r.mDirection = reflect(-dirIn, h);
	const float hdotwo = dot(h, r.mDirection);
	r.mFwdPdfW = disneyclearcoat_eval_pdf(D, h, hdotwo);
	r.mRevPdfW = disneyclearcoat_eval_pdf(D, h, dot(h, dirIn));
	r.mEta = 0;
	r.mRoughness = alpha;
	return r;
}