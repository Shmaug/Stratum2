Real disneyclearcoat_eval_pdf(const Real D, const Vector3 h, const Real hdotwo) {
    return D * abs(h.z) / (4*abs(hdotwo));
}

Spectrum disneyclearcoat_eval(const Real D, const Vector3 dir_in, const Vector3 dir_out, const Vector3 h, const Real hdotwo) {
    const Real eta = 1.5;
    const Real Fc = schlick_fresnel1(R0(eta), hdotwo);
    return Fc * D * Gc(dir_in) * Gc(dir_out) / (4 * abs(dir_in.z));
}

void disneyclearcoat_eval(const DisneyMaterialData bsdf, out MaterialEvalRecord r, const Vector3 dir_in, const Vector3 dir_out, const bool adjoint) {
	if (dir_in.z * dir_out.z < 0) {
		r.f = 0;
		r.pdf_fwd = r.pdf_rev = 0;
		return; // No light through the surface
	}
    const Vector3 h = normalize(dir_in + dir_out);
    const Real hdotwo = abs(dot(h, dir_out));

    const Real alpha = (1 - bsdf.clearcoat_gloss())*0.1 + bsdf.clearcoat_gloss()*0.001;
	const Real D = Dc(alpha, h.z);

	r.f = disneyclearcoat_eval(D, dir_in, dir_out, h, hdotwo);
	r.pdf_fwd = disneyclearcoat_eval_pdf(D, h, hdotwo);
	r.pdf_rev = disneyclearcoat_eval_pdf(D, h, dot(h, dir_in));
}
Spectrum disneyclearcoat_sample(const DisneyMaterialData bsdf, out MaterialSampleRecord r, const Vector3 rnd, const Vector3 dir_in, inout Spectrum beta, const bool adjoint) {
    const Real alpha = (1 - bsdf.clearcoat_gloss())*0.1 + bsdf.clearcoat_gloss()*0.001;

    const Real alpha2 = alpha*alpha;
    const Real cos_phi = sqrt((1 - pow(alpha2, 1 - rnd.x)) / (1 - alpha2));
    const Real sin_phi = sqrt(1 - max(cos_phi*cos_phi, Real(0)));
    const Real theta = 2*M_PI * rnd.y;
    Vector3 h = Vector3(sin_phi*cos(theta), sin_phi*sin(theta), cos_phi);
	if (dir_in.z < 0) h = -h;

	const Real D = Dc(alpha, h.z);

	r.dir_out = reflect(-dir_in, h);
	const Real hdotwo = dot(h, r.dir_out);
	r.pdf_fwd = disneyclearcoat_eval_pdf(D, h, hdotwo);
	r.pdf_rev = disneyclearcoat_eval_pdf(D, h, dot(h, dir_in));
	r.eta = 0;
	r.roughness = alpha;
	const Spectrum f = disneyclearcoat_eval(D, dir_in, r.dir_out, h, hdotwo);
	beta *= f / r.pdf_fwd;
	return f;
}