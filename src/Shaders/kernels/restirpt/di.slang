#include "pt.slang"

float DirectMis(const float a, const float b) {
    const float a2 = a * a;
    return safe_divide(a2, a2 + b * b);
}

struct DirectIlluminationSample {
    float3 mRadiance;
    float mPdfA;
    float G;
    float mMisWeight;
};

extension PathTracer {
    DirectIlluminationSample DirectIllumination(const PathVertex vertex, const Interaction bsdf, inout RandomSampler rng) {
        DirectIlluminationSample o;
        o.mRadiance = 0;
        o.mPdfA = 0;
        o.G = 0;
        o.mMisWeight = 0;

		const IlluminationSampleRecord r = mScene.SampleIllumination(rng.nextFloat(), vertex.mShadingData.mPosition);
		if (all(r.mRadiance <= 0) || r.mPdf <= 0)
			return o;

		if (r.isFinite)
            o.G = r.mCosLight / pow2(r.mDistanceToLight);
        else
			o.G = 1;

		const bool isMedium = gHasMedia && vertex.mShadingData.isMedium();
		const float3 localDirOut = isMedium ? r.mDirectionToLight : vertex.mShadingData.toLocal(r.mDirectionToLight);

		ReflectanceEvalRecord f = bsdf.evaluateReflectance<false>(vertex.mLocalDirIn, localDirOut);
		if (!isMedium)
			f.mReflectance *= vertex.mShadingData.shadingNormalCorrection<false>(vertex.mLocalDirIn, localDirOut);

		o.mRadiance = r.mRadiance * f.mReflectance;

		o.mPdfA = r.mPdf;
		o.mMisWeight = DirectMis(safe_divide(r.mPdf, o.G), f.mFwdPdfW);

		// evaluate transmittance along shadow ray

		const float3 rayOrigin = isMedium ?
				vertex.mShadingData.mPosition :
				rayOffset(vertex.mShadingData.mPosition, vertex.mShadingData.getGeometryNormal(), r.mDirectionToLight);

		float transDirPdf, transNeePdf;
        mScene.traceVisibilityRay(
            makeRay(rayOrigin, r.mDirectionToLight, 0, AdjustIntersectionDistance(r.mDistanceToLight)),
            rng, vertex.mCurrentMedium, o.mRadiance, transDirPdf, transNeePdf);

		o.mPdfA *= transNeePdf;

        if (all(o.mRadiance <= 0) || transNeePdf <= 0) {
            o.G = 0;
            o.mPdfA = 0;
        }

        return o;
    }
}

/* -------- ReSTIR -------- */

struct ReservoirSampleDataDI {
    float4 mSample;
    float mSourcePdf;
    float3 mCachedContribution; // f * Le
    RayDesc mVisibilityRay;
    float mCachedWtoA;
    float mCachedBsdfPdfW;

    property float mTargetPdf { get { return luminance(mCachedContribution) * mCachedWtoA; } }
    property float mRISWeight { get { return mTargetPdf / mSourcePdf; } }

    __init(const SceneParameters scene, const PathVertex vertex, const Interaction bsdf, const float4 x) {
        mSample = x;
        mCachedContribution = 0;
        mCachedWtoA = 0;
        mCachedBsdfPdfW = 0;
        const IlluminationSampleRecord r = scene.SampleIllumination(mSample, vertex.mShadingData.mPosition);
        mSourcePdf = r.mPdf;
        if (luminance(r.mRadiance) > 0 && r.mPdf > 0) {
            const bool isMedium = gHasMedia && vertex.mShadingData.isMedium();

            const float3 localDirOut = isMedium ? r.mDirectionToLight : vertex.mShadingData.toLocal(r.mDirectionToLight);

            ReflectanceEvalRecord f = bsdf.evaluateReflectance<false>(vertex.mLocalDirIn, localDirOut);
            if (!isMedium)
                f.mReflectance *= vertex.mShadingData.shadingNormalCorrection<false>(vertex.mLocalDirIn, localDirOut);

            mCachedBsdfPdfW = f.mFwdPdfW;

            mCachedContribution = f.mReflectance * r.mRadiance;
            mCachedWtoA = r.isFinite ? r.mCosLight / pow2(r.mDistanceToLight) : 1;
            const float3 rayOrigin = isMedium ?
					vertex.mShadingData.mPosition :
					rayOffset(vertex.mShadingData.mPosition, vertex.mShadingData.getGeometryNormal(), r.mDirectionToLight);
            mVisibilityRay = makeRay(rayOrigin, r.mDirectionToLight, 0, AdjustIntersectionDistance(r.mDistanceToLight));
        }
    }
};
struct PackedReservoirDI {
    float3 mLocalPosition;
    uint mInstancePrimitiveIndex;
    float4 mSample;
    float mCachedTargetPdf;
    float W;
    float M;
    uint mPackedLocalDirIn;

    property float3 mLocalDirIn {
        get { return unpackNormal(mPackedLocalDirIn); }
        set { mPackedLocalDirIn = packNormal(newValue); }
    }

    uint getInstanceIndex()  { return BF_GET(mInstancePrimitiveIndex, 0, 16); }
    uint getPrimitiveIndex() { return BF_GET(mInstancePrimitiveIndex, 16, 16); }
    ShadingData getShadingData(const SceneParameters scene) {
        const uint instanceIndex = getInstanceIndex();
        ShadingData sd = scene.makeShadingData(scene.mInstances[instanceIndex], scene.mInstanceTransforms[instanceIndex], mLocalPosition, getPrimitiveIndex());
        sd.mTexcoordScreenSize = 0;
        return sd;
    }

    void Store(const RenderParams fb, const uint2 idx) {
        float4 data;
        data.xyz = mLocalPosition;
        data.w = asfloat(mInstancePrimitiveIndex);
        fb.mReservoirDataDI[0][idx] = data;
        fb.mReservoirDataDI[1][idx] = mSample;
        fb.mReservoirDataDI[2][idx] = float4(mCachedTargetPdf, W, M, asfloat(mPackedLocalDirIn));
    }

    [mutating]
    void LoadPrev(const RenderParams fb, const uint2 idx) {
        const float4 data = fb.mPrevReservoirDataDI[0][idx];
        mLocalPosition = data.xyz;
        mInstancePrimitiveIndex = asuint(data.w);
        mSample = fb.mPrevReservoirDataDI[1][idx];
        const float4 data2 = fb.mPrevReservoirDataDI[2][idx];
        mCachedTargetPdf = data2.x;
        W = data2.y;
        M = data2.z;
        mPackedLocalDirIn = asuint(data2.w);
    }
};

extension PathTracer {
	void ReservoirReuseDI(const PathVertex vertex, const Interaction bsdf, const PackedReservoirDI q, inout ReservoirSampleDataDI currentSample, inout float integrationWeight, inout float M, inout RandomSampler rng) {
		// q's sample in the current domain
		ReservoirSampleDataDI newSample = ReservoirSampleDataDI(mScene, vertex, bsdf, q.mSample);
		#ifdef gReSTIR_DI_Reuse_Visibility
		if (newSample.mTargetPdf > 0) {
			float transDirPdf, transNeePdf;
			mScene.traceVisibilityRay(newSample.mVisibilityRay, rng, vertex.mCurrentMedium, newSample.mCachedContribution, transDirPdf, transNeePdf);
		}
		#endif

		// target pdf in q's domain of currentSample
		float phat_q = 0;
		{
			const PathVertex v_q = PathVertex(mScene, q.getShadingData(mScene), q.mInstancePrimitiveIndex, vertex.mCurrentMedium, q.mPackedLocalDirIn);
			ReservoirSampleDataDI currentSample_q;
			if (gHasMedia && v_q.mShadingData.isMedium())
				currentSample_q = ReservoirSampleDataDI(mScene, v_q, Medium(mScene, v_q.mShadingData.getMaterialAddress()), currentSample.mSample);
			else
				currentSample_q = ReservoirSampleDataDI(mScene, v_q, mScene.LoadMaterial(v_q.mShadingData), currentSample.mSample);

			if (currentSample_q.mTargetPdf > 0) {
				#ifdef gReSTIR_DI_Reuse_Visibility
				// note: this ray should really be traced through the previous frame's acceleration structure
				float transDirPdf, transNeePdf;
				mScene.traceVisibilityRay(currentSample_q.mVisibilityRay, rng, vertex.mCurrentMedium, currentSample_q.mCachedContribution, transDirPdf, transNeePdf);
				#endif
				phat_q = currentSample_q.mTargetPdf;
			}
		}

		// Resampling MIS weights

		#ifdef gPairwiseMis
		const float R = M;
		M += q.M;
		// defensive pairwise mis
		const float m0 = (1 +      safe_divide(currentSample.mTargetPdf, R * currentSample.mTargetPdf + (M - R) * phat_q)) / M;
		const float m1 = (M - R) * safe_divide(q.mCachedTargetPdf, R * newSample.mTargetPdf + (M - R) * q.mCachedTargetPdf) / M;
		#elif defined(gTalbotMis)
		// generalized talbot mis
		const float m0 = safe_divide(M*currentSample.mTargetPdf, M*currentSample.mTargetPdf + q.M * phat_q);
		const float m1 = safe_divide(q.M * q.mCachedTargetPdf  , q.M * q.mCachedTargetPdf   + M * newSample.mTargetPdf);
		M += q.M;
		#else
		// naive mis
		const float m0 = safe_divide(M, M + (phat_q > 0 ? q.M : 0));
		const float m1 = safe_divide(q.M, q.M + (newSample.mTargetPdf > 0 ? M : 0));
		M += q.M;
		#endif

		// RIS weights

		const float w0 = m0 * currentSample.mTargetPdf * integrationWeight;
		const float w1 = m1 * newSample.mTargetPdf * q.W;

		// create new reservoir

		const float totalWeight = w0 + w1;
		if (rng.nextFloat().x * totalWeight <= w1) {
			currentSample = newSample;
		}

		if (currentSample.mTargetPdf > 0)
			integrationWeight = totalWeight / currentSample.mTargetPdf;
		else
			integrationWeight = 0;
	}

    DirectIlluminationSample DirectIlluminationReSTIR(const PathVertex vertex, const Interaction bsdf, inout RandomSampler rng) {
		ReservoirSampleDataDI sample;
		float wsum = 0;

		// initial ris

		for (uint i = 0; i < gPushConstants.mDICandidateSamples; i++) {
			const ReservoirSampleDataDI xi = ReservoirSampleDataDI(mScene, vertex, bsdf, rng.nextFloat());
			if (xi.mTargetPdf <= 0)
				continue;

			wsum += xi.mRISWeight;
			if (rng.nextFloat().x <= xi.mRISWeight / wsum)
				sample = xi;
		}

		float integrationWeight = 0;

		#if !defined(gReSTIR_DI_Reuse) || defined(gReSTIR_DI_Reuse_Visibility)
		if (sample.mTargetPdf > 0) {
			float transDirPdf, transNeePdf;
			mScene.traceVisibilityRay(sample.mVisibilityRay, rng, vertex.mCurrentMedium, sample.mCachedContribution, transDirPdf, transNeePdf);
		}
		#endif

		if (sample.mTargetPdf > 0)
			integrationWeight = wsum / (gPushConstants.mDICandidateSamples * sample.mTargetPdf);

		// reuse

		#ifdef gReSTIR_DI_Reuse

		float M = gPushConstants.mDICandidateSamples;

		if (gPushConstants.mReservoirHistoryValid != 0) {
			const float2 discRng = rng.nextFloat().xy;
			const int2 qi = int2(mFramebuffer.mPrevUVs[sPixelIndex] * gPushConstants.mOutputExtent + sampleConcentricDisc(discRng.x, discRng.y) * gPushConstants.mDIReuseRadius);
			if (all(qi >= 0) && all(qi < gPushConstants.mOutputExtent)) {
				PackedReservoirDI q;
				q.LoadPrev(mFramebuffer, qi);
				if (q.M > 0) {
					ReservoirReuseDI(vertex, bsdf, q, sample, integrationWeight, M, rng);
				}
			}
		}

		// store for reuse
		{
			PackedReservoirDI r;
			r.mLocalPosition = mScene.mInstanceInverseTransforms[vertex.mInstanceIndex].transformPoint(vertex.mShadingData.mPosition);
			r.mInstancePrimitiveIndex = vertex.mInstancePrimitiveIndex;
			r.mSample = sample.mSample;
			r.mCachedTargetPdf = sample.mTargetPdf;
			r.W = integrationWeight;
			r.M = min(M, gPushConstants.mDIMaxM * gPushConstants.mDICandidateSamples);
			r.mPackedLocalDirIn = vertex.mPackedLocalDirIn;
			r.Store(mFramebuffer, sPixelIndex);
		}

		#ifndef gReSTIR_DI_Reuse_Visibility
		if (sample.mTargetPdf > 0) {
			float transDirPdf, transNeePdf;
			mScene.traceVisibilityRay(sample.mVisibilityRay, rng, vertex.mCurrentMedium, sample.mCachedContribution, transDirPdf, transNeePdf);
		}
		#endif

		#endif

        DirectIlluminationSample o;
        o.mRadiance = sample.mCachedContribution;
        o.mPdfA = safe_divide(1, integrationWeight);
        o.G = sample.mCachedWtoA;
        o.mMisWeight = DirectMis(safe_divide(sample.mSourcePdf, sample.mCachedWtoA), sample.mCachedBsdfPdfW);
        return o;
	}
}
