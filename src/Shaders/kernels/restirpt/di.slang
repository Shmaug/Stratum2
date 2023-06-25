#include "pt.slang"

struct DirectIlluminationSample {
    Spectrum mRadiance;
    float mPdfW;
    float mBsdfPdfW;
    float G;
};

extension PathTracer {
    DirectIlluminationSample DirectIllumination(const PathVertex vertex, const Interaction bsdf, inout RandomSampler rng) {
        DirectIlluminationSample o;
        o.mRadiance = 0;
        o.mPdfW = 0;
        o.G = 0;
        o.mBsdfPdfW = 0;

		const IlluminationSampleRecord r = mScene.SampleIllumination(rng.nextFloat(), vertex.mShadingData.mPosition);
		if (all(r.mRadiance <= 0) || r.mPdf <= 0)
			return o;

		if (r.isFinite)
            o.G = r.mCosLight / pow2(r.mDistanceToLight);
        else
			o.G = 1;

		const bool isMedium = gHasMedia && vertex.mShadingData.isMedium();
		const float3 localDirOut = isMedium ? r.mDirectionToLight : vertex.mShadingData.toLocal(r.mDirectionToLight);

        o.mRadiance = r.mRadiance;

        const ReflectanceEvalRecord f = bsdf.evaluateReflectance<false>(vertex.mLocalDirIn, localDirOut);
        o.mRadiance *= f.mReflectance;
        if (!isMedium)
            o.mRadiance *= vertex.mShadingData.shadingNormalCorrection<false>(vertex.mLocalDirIn, localDirOut);

        o.mPdfW = safe_divide(r.mPdf, o.G);
        o.mBsdfPdfW = f.mFwdPdfW;

		// evaluate transmittance along shadow ray

		float transDirPdf, transNeePdf;
        mScene.traceVisibilityRay(
            makeRay(rayOffset(vertex.mShadingData, r.mDirectionToLight), r.mDirectionToLight, 0, AdjustIntersectionDistance(r.mDistanceToLight)),
            rng, vertex.mCurrentMedium, o.mRadiance, transDirPdf, transNeePdf);

        o.mPdfW *= transNeePdf;
        o.mBsdfPdfW *= transDirPdf;

        if (all(o.mRadiance <= 0) || transNeePdf <= 0) {
            o.mRadiance = 0;
            o.mPdfW = 0;
            o.mBsdfPdfW = 0;
            o.G = 0;
        }

        return o;
    }
}