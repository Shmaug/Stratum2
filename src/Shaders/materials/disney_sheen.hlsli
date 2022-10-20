Spectrum disneysheen_eval(const DisneyMaterialData bsdf, const Vector3 dir_in, const Vector3 dir_out, const Vector3 h) {
	const Spectrum Ctint = luminance(bsdf.base_color()) > 1e-4 ? bsdf.base_color()/luminance(bsdf.base_color()) : 1;
	const Spectrum Csheen = (1 - bsdf.sheen_tint()) + bsdf.sheen_tint()*Ctint;
	return Csheen * pow(1 - abs(dot(h, dir_out)), 5) * abs(dir_out.z);
}

void disneysheen_eval(const DisneyMaterialData bsdf, out MaterialEvalRecord r, const Vector3 dir_in, const Vector3 dir_out, const bool adjoint) {
	if (dir_in.z * dir_out.z < 0) {
		r.f = 0;
		r.pdf_fwd = r.pdf_rev = 0;
		return; // No light through the surface
	}
	r.f = disneysheen_eval(bsdf, dir_in, dir_out, normalize(dir_in + dir_out));
	r.pdf_fwd = cosine_hemisphere_pdfW(abs(dir_out.z));
	r.pdf_rev = cosine_hemisphere_pdfW(abs(dir_in.z));
}

Spectrum disneysheen_sample(const DisneyMaterialData bsdf, out MaterialSampleRecord r, const Vector3 rnd, const Vector3 dir_in, inout Spectrum beta, const bool adjoint) {
	r.dir_out = sample_cos_hemisphere(rnd.x, rnd.y);
	if (dir_in.z < 0) r.dir_out = -r.dir_out;
	r.pdf_fwd = cosine_hemisphere_pdfW(abs(r.dir_out.z));
	r.pdf_rev = cosine_hemisphere_pdfW(abs(dir_in.z));
	r.eta = 0;
	r.roughness = 1;
	const Spectrum f = disneysheen_eval(bsdf, dir_in, r.dir_out, normalize(dir_in + r.dir_out));
	beta *= f / r.pdf_fwd;
	return f;
}