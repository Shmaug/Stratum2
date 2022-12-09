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


struct DisneyDiffuseMaterial : BSDF {
    DisneyMaterialData bsdf;

    [mutating]
    void load(const uint address, const float2 uv, const float uvScreenSize, inout uint packedShadingNormal, inout uint packedTangent, const bool flipBitangent) {
        for (int i = 0; i < DisneyMaterialData::gDataCount; i++)
            bsdf.data[i] = ImageValue4(gScene.mMaterialData, address + i * ImageValue4::PackedSize).eval(uv, uvScreenSize);

        // normal map
        if (CHECK_FEATURE(NormalMaps)) {
            const uint2 p = gScene.mMaterialData.Load<uint2>(address + (ImageValue4::PackedSize * DisneyMaterialData::gDataCount) + 4);
            ImageValue3 bump_img = { 1, p.x };
            if (bump_img.hasImage() && asfloat(p.y) > 0) {
                float3 bump = bump_img.eval(uv, uvScreenSize) * 2 - 1;
                bump = normalize(float3(bump.xy * asfloat(p.y), bump.z > 0 ? bump.z : 1));

                float3 n = unpackNormal(packedShadingNormal);
                float3 t = unpackNormal(packedTangent);

                n = normalize(t * bump.x + cross(n, t) * (flipBitangent ? -1 : 1) * bump.y + n * bump.z);
                t = normalize(t - n * dot(n, t));

                packedShadingNormal = packNormal(n);
                packedTangent = packNormal(t);
            }
        }
    }

    [mutating]
    void load(inout ShadingData sd) {
        load(sd.materialAddress, sd.mTexcoord, sd.mTexcoordScreenSize, sd.mPackedShadingNormal, sd.mPackedTangent, sd.isBitangentFlipped);
    }

    __init(uint address, const float2 uv, const float uvScreenSize, inout uint packedShadingNormal, inout uint packedTangent, const bool flipBitangent) {
        load(address, uv, uvScreenSize, packedShadingNormal, packedTangent, flipBitangent);
    }
    __init(inout ShadingData sd) {
        load(sd);
    }

    float3 emission() { return bsdf.baseColor() * bsdf.emission(); }
    float3 albedo() { return bsdf.baseColor(); }
    bool canEvaluate() { return bsdf.emission() <= 0 && any(bsdf.baseColor() > 0); }

    bool isSingular() { return false; }

    MaterialEvalRecord evaluate<let Adjoint : bool>(const float3 dirIn, const float3 dirOut) {
        MaterialEvalRecord r;
        if (dirIn.z * dirOut.z < 0) {
            r.mReflectance = 0;
            r.mFwdPdfW = r.mRevPdfW = 0;
            return r; // No light through the surface
        }
        r.mReflectance = disneydiffuse_eval(bsdf, dirIn, dirOut);
        r.mFwdPdfW = cosHemispherePdfW(abs(dirOut.z));
        r.mRevPdfW = cosHemispherePdfW(abs(dirIn.z));
        return r;
    }
    MaterialSampleRecord sample<let Adjoint : bool>(const float3 rnd, const float3 dirIn) {
        MaterialSampleRecord r;
        r.mDirection = sampleCosHemisphere(rnd.x, rnd.y);
        if (dirIn.z < 0) r.mDirection = -r.mDirection;
        r.mFwdPdfW = cosHemispherePdfW(abs(r.mDirection.z));
        r.mRevPdfW = cosHemispherePdfW(abs(dirIn.z));
        r.mEta = 0;
        r.mRoughness = 1;
        return r;
    }
};