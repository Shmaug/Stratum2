float3 disneysheen_eval(const DisneyMaterialData bsdf, const float3 dirIn, const float3 dirOut, const float3 h) {
	const float3 Ctint = luminance(bsdf.baseColor()) > 1e-4 ? bsdf.baseColor()/luminance(bsdf.baseColor()) : 1;
	const float3 Csheen = (1 - bsdf.sheen_tint()) + bsdf.sheen_tint()*Ctint;
	return Csheen * pow(1 - abs(dot(h, dirOut)), 5) * abs(dirOut.z);
}

MaterialEvalRecord disneysheen_eval(const DisneyMaterialData bsdf, const float3 dirIn, const float3 dirOut, const bool adjoint) {
	MaterialEvalRecord r;
	if (dirIn.z * dirOut.z < 0) {
		r.mReflectance = 0;
		r.mFwdPdfW = r.mRevPdfW = 0;
		return r; // No light through the surface
	}
	r.mReflectance = disneysheen_eval(bsdf, dirIn, dirOut, normalize(dirIn + dirOut));
	r.mFwdPdfW = cosine_hemisphere_pdfW(abs(dirOut.z));
	r.mRevPdfW = cosine_hemisphere_pdfW(abs(dirIn.z));
	return r;
}

MaterialSampleRecord disneysheen_sample(const DisneyMaterialData bsdf, const float3 rnd, const float3 dirIn, const bool adjoint) {
	MaterialSampleRecord r;
	r.mDirection = sample_cos_hemisphere(rnd.x, rnd.y);
	if (dirIn.z < 0) r.mDirection = -r.dirOut;
	r.mFwdPdfW = cosine_hemisphere_pdfW(abs(r.mDirection.z));
	r.mRevPdfW = cosine_hemisphere_pdfW(abs(dirIn.z));
	r.mEta = 0;
	r.mRoughness = 1;
	return r;
}