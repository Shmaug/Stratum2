float3 disneydiffuse_eval(const DisneyMaterialData bsdf, const float3 dirIn, const float3 dirOut) {
	const float hdotwo = abs(dot(normalize(dirIn + dirOut), dirOut));

	const float FSS90 = bsdf.roughness() * hdotwo*hdotwo;
	const float FD90 = 0.5 + 2*FSS90;
	const float ndotwi5 = pow(1 - abs(dirIn.z), 5);
	const float ndotwo5 = pow(1 - abs(dirOut.z), 5);
	const float FDwi = 1 + (FD90 - 1) * ndotwi5;
	const float FDwo = 1 + (FD90 - 1) * ndotwo5;
	const float3 f_base_diffuse = (bsdf.baseColor() / M_PI) * FDwi * FDwo;

	const float FSSwi = 1 + (FSS90 - 1) * ndotwi5;
	const float FSSwo = 1 + (FSS90 - 1) * ndotwo5;
	const float3 f_subsurface = (1.25 * bsdf.baseColor() / M_PI) * (FSSwi * FSSwo * (1 / (abs(dirIn.z) + abs(dirOut.z)) - 0.5) + 0.5);

	return lerp(f_base_diffuse, f_subsurface, bsdf.subsurface()) * abs(dirOut.z);
}

MaterialEvalRecord disneydiffuse_eval(const DisneyMaterialData bsdf, const float3 dirIn, const float3 dirOut, const bool adjoint) {
	MaterialEvalRecord r;
	if (dirIn.z * dirOut.z < 0) {
		r.mReflectance = 0;
		r.mFwdPdfW = r.mRevPdfW = 0;
		return r; // No light through the surface
	}
	r.mReflectance = disneydiffuse_eval(bsdf, dirIn, dirOut);
	r.mFwdPdfW = cosine_hemisphere_pdfW(abs(dirOut.z));
	r.mRevPdfW = cosine_hemisphere_pdfW(abs(dirIn.z));
	return r;
}
MaterialSampleRecord disneydiffuse_sample(const DisneyMaterialData bsdf, const float3 rnd, const float3 dirIn, const bool adjoint) {
	MaterialSampleRecord r;
	r.mDirection = sample_cos_hemisphere(rnd.x, rnd.y);
	if (dirIn.z < 0) r.mDirection = -r.mDirection;
	r.mFwdPdfW = cosine_hemisphere_pdfW(abs(r.mDirection.z));
	r.mRevPdfW = cosine_hemisphere_pdfW(abs(dirIn.z));
	r.mEta = 0;
	r.mRoughness = 1;
	return r;
}