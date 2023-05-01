#include "di.slang"

// Use the Vulkan Configurator to redirect DebugPrintf to stdout (requires --debugMessenger)
#ifdef gDebugPixel
#define DebugPrintf(pixel, ...) if ((pixel.y<<16 | (pixel.x&0xFFFF)) == gPushConstants.mPackedDebugPixelIndex) { printf(__VA_ARGS__); }
#else
#define DebugPrintf(pixel, ...)
#endif

struct VertexSampleRecord {
    float3 mLe;
	float mNeePdfW;
    float3 mNeeLe;
	float mNeeMisWeight;

    float3 mDirection;
    float mDirPdfW;
    float3 mBrdf;
    bool isSpecular;
    bool isSpecularLobe;
};

extension PathTracer {
	// samples a direction and samples direct illumination (nee)
	VertexSampleRecord ProcessVertex(const PathVertex vertex, const Interaction bsdf, inout RandomSampler rng, const bool sampleDirection) {
		VertexSampleRecord r;
		r.mLe = 0;
		r.mNeePdfW = 0;
		r.mNeeLe = 0;
		r.mNeeMisWeight = 0;
		r.mDirPdfW = 0;
		r.mBrdf = 0;
		r.isSpecular = bsdf.isSingular();
		r.isSpecularLobe = false;

		const bool isMedium = gHasMedia && vertex.mShadingData.isMedium();
		const float3 localDirIn = vertex.mLocalDirIn;

		if (isMedium || localDirIn.z > 0) {
			r.mLe = bsdf.emission();
		}

		if (sampleDirection && bsdf.canEvaluate()) {
			const DirectionSampleRecord s = bsdf.sampleDirection<false>(rng.nextFloat().xyz, localDirIn);
			r.isSpecularLobe = s.isSingular();
			r.mBrdf = s.mReflectance;
			r.mDirPdfW = s.mFwdPdfW;
			if (!isMedium) {
				r.mBrdf *= vertex.mShadingData.shadingNormalCorrection<false>(localDirIn, s.mDirection);
				r.mDirection = vertex.mShadingData.toWorld(s.mDirection);
			} else
				r.mDirection = s.mDirection;

			#ifdef gNee
			if (!r.isSpecular) {
				const DirectIlluminationSample s = DirectIllumination(vertex, bsdf, rng);
				r.mNeeLe = s.mRadiance;
				r.mNeePdfW = safe_divide(s.mPdfA, s.G);
				r.mNeeMisWeight = s.mMisWeight;
			}
			#endif
		}

		return r;
	}
	VertexSampleRecord ProcessVertex(const PathVertex vertex, inout RandomSampler rng, const bool sampleNext) {
		if (vertex.mShadingData.isEnvironment()) {
			VertexSampleRecord r;
			float tmp;
			r.mLe = mScene.EvaluateEnvironment(-vertex.mLocalDirIn, tmp);
			r.mNeeLe = 0;
			r.mNeePdfW = 0;
			r.mBrdf = 0;
			r.mDirPdfW = 0;
			r.isSpecular = false;
			r.isSpecularLobe = false;
			return r;
		}
		if (gHasMedia && vertex.mShadingData.isMedium())
			return ProcessVertex(vertex, Medium(mScene, vertex.mShadingData.getMaterialAddress()), rng, sampleNext);
		else
			return ProcessVertex(vertex, mScene.LoadMaterial(vertex.mShadingData), rng, sampleNext);
	}
	PathVertex TraceNext(const PathVertex vertex, inout RandomSampler rng, const float3 direction, inout float3 throughput, out float transDirPdf, out float directPdfA, out float WtoA) {
		const float3 origin = (gHasMedia && vertex.mShadingData.isMedium()) ?
			vertex.mShadingData.mPosition :
			rayOffset(vertex.mShadingData.mPosition, vertex.mShadingData.getGeometryNormal(), direction);

		uint medium = vertex.mCurrentMedium;

		IntersectionResult isect;
		float transNeePdf;
		const bool hit = mScene.traceScatteringRay(makeRay(origin, direction), rng, medium, throughput, transDirPdf, transNeePdf, isect);
		isect.mShadingData.mTexcoordScreenSize = 0; // disable mipmaps for now

		PathVertex v = PathVertex(mScene, isect.mShadingData, isect.mInstancePrimitiveIndex, medium, -direction);

		if (hit) {
			directPdfA = isect.LightSamplePdfA();
			WtoA = abs((gHasMedia && v.mShadingData.isMedium()) ? 1 : v.mLocalDirIn.z) / pow2(isect.mDistance);
		} else {
			mScene.EvaluateEnvironment(direction, directPdfA);
			if (gLightCount > 0)
				directPdfA *= gEnvironmentSampleProbability;
			WtoA = 1;
		}

		return v;
	}

	float3 EstimateRadiance(PathVertex vertex, inout RandomSampler rng, const uint2 pixelIndex) {
		float3 radiance = 0;
		float3 throughput = 1;
		float pathPdfW = 1;
		uint diffuseBounces = 0;
		float misWeight = 1;

        for (uint depth = 1; depth <= gPushConstants.mMaxDepth; depth++) {
            VertexSampleRecord s;
            if (depth == 1) {
				// get albedo from material
				if (gHasMedia && vertex.mShadingData.isMedium()) {
					const Medium m = Medium(mScene, vertex.mShadingData.getMaterialAddress());
					mFramebuffer.mAlbedo[pixelIndex] = float4(m.albedo(), 1);
					s = ProcessVertex(vertex, m, rng, depth < gPushConstants.mMaxDepth);
				} else {
					const PackedMaterialData m = mScene.LoadMaterial(vertex.mShadingData);
					mFramebuffer.mAlbedo[pixelIndex] = float4(m.albedo(), 1);
					s = ProcessVertex(vertex, m, rng, depth < gPushConstants.mMaxDepth);
				}
        	} else {
                s = ProcessVertex(vertex, rng, depth < gPushConstants.mMaxDepth);
            }


			// add emission from vertex and nee sample
            radiance += throughput * s.mLe * misWeight / pathPdfW;
			if (!s.isSpecular)
				diffuseBounces++;
            if (diffuseBounces > gPushConstants.mMaxDiffuseBounces)
                break;
			radiance += throughput * s.mNeeLe * safe_divide(s.mNeeMisWeight, pathPdfW * s.mNeePdfW);

			if (s.mDirPdfW <= 0 || all(s.mBrdf <= 0))
				break;

			throughput *= s.mBrdf;
			pathPdfW *= s.mDirPdfW;

			float transDirPdf, directPdfA, WtoA;
			vertex = TraceNext(vertex, rng, s.mDirection, throughput, transDirPdf, directPdfA, WtoA);
			if (transDirPdf <= 0)
				break;
			pathPdfW *= transDirPdf;

			#ifdef gNee
			misWeight = s.isSpecular ? 1 : DirectMis(s.mDirPdfW, directPdfA / WtoA);
			#endif
		}

		return radiance;
	}
}

/* -------- ReSTIR PT -------- */

struct PackedVertex {
    float3 mLocalPosition;
    uint mInstancePrimitiveIndex;

    uint getInstanceIndex()  { return BF_GET(mInstancePrimitiveIndex, 0, 16); }
    uint getPrimitiveIndex() { return BF_GET(mInstancePrimitiveIndex, 16, 16); }
    ShadingData getShadingData(const SceneParameters scene) {
        const uint instanceIndex = getInstanceIndex();
        ShadingData sd = scene.makeShadingData(scene.mInstances[instanceIndex], scene.mInstanceTransforms[instanceIndex], mLocalPosition, getPrimitiveIndex());
        sd.mTexcoordScreenSize = 0;
        return sd;
    }
};
struct ReconnectionVertex {
    PackedVertex mVertex;
    float3 mRadiance; // estimated radiance arriving at mVertex
    uint mPackedDirOut;

    property float3 mDirOut {
        get { return unpackNormal(mPackedDirOut); }
        set { mPackedDirOut = packNormal(newValue); }
	}
};

struct GISample {
    ReconnectionVertex mReconnectionVertex;
    uint4 mRng;
    float mReplayPdfW;
    uint mPathLength;
};


struct ReservoirSampleDataGI {
    GISample mSample;
    float3 mCachedContribution;
    float W;

    float3 mCachedDirectRadiance, mCachedAlbedo;

    property float mCachedTargetPdf { get { return luminance(mCachedContribution); } }

	// randomly replaces mCachedContribution and mCachedTargetPdf using resampling
    [mutating]
    void UpdatePath(const float3 pathContribution, const float pathPdfW, const uint pathLength, const float rnd) {
        if (pathPdfW <= 0 || all(pathContribution <= 0))
			return;

		const float phat = luminance(pathContribution);
		const float wi = phat / pathPdfW;

		const float wsum = wi + W * mCachedTargetPdf;

		if (rnd*wsum <= wi) {
            mCachedContribution = pathContribution;
            mSample.mReplayPdfW = pathPdfW;
            mSample.mPathLength = pathLength;
		}

		W = safe_divide(wsum, mCachedTargetPdf);
	}

	// traces a path from vertex using the random seed x
    [mutating]
    void make<let bIsCanonical : bool>(const PathTracer pt, PathVertex vertex, const uint4 x, const uint2 pixelIndex) {
        mSample.mRng = x;
        mSample.mReplayPdfW = 1;
        mSample.mReconnectionVertex.mRadiance = 0;
        mSample.mPathLength = 0;
        mCachedContribution = 0;
        W = 0;

        RandomSampler rng;
        rng.mState = x;

        float3 throughput = 1;
		uint diffuseBounces = 0;
        float misWeight = 1;

        for (uint depth = 1; depth <= gPushConstants.mMaxDepth; depth++) {
            VertexSampleRecord s;

            if (depth == 1) {
				if (gHasMedia && vertex.mShadingData.isMedium()) {
					const Medium m = Medium(pt.mScene, vertex.mShadingData.getMaterialAddress());
					if (bIsCanonical) mCachedAlbedo = m.albedo();
                    s = pt.ProcessVertex(vertex, m, rng, depth < gPushConstants.mMaxDepth);
				} else {
					const PackedMaterialData m = pt.mScene.LoadMaterial(vertex.mShadingData);
					if (bIsCanonical) mCachedAlbedo = m.albedo();
                    s = pt.ProcessVertex(vertex, m, rng, depth < gPushConstants.mMaxDepth);
				}
                if (bIsCanonical) mCachedDirectRadiance = s.mLe;
            } else {
				s = pt.ProcessVertex(vertex, rng, depth < gPushConstants.mMaxDepth);
				// Add direct light
				UpdatePath(throughput * s.mLe * misWeight, mSample.mReplayPdfW, depth, rng.nextFloat().x);
			}

			// Add NEE contribution
			#ifdef gNee
			UpdatePath(throughput * s.mNeeLe * s.mNeeMisWeight, mSample.mReplayPdfW * s.mNeePdfW, depth+1, rng.nextFloat().x);
			#endif

			if (!s.isSpecular)
				diffuseBounces++;
            if (s.mDirPdfW <= 0 || all(s.mBrdf <= 0) || diffuseBounces > gPushConstants.mMaxDiffuseBounces)
                return;

            throughput *= s.mBrdf;
            mSample.mReplayPdfW *= s.mDirPdfW;

			float transDirPdf, directPdfA, WtoA;
			vertex = pt.TraceNext(vertex, rng, s.mDirection, throughput, transDirPdf, directPdfA, WtoA);
            if (transDirPdf <= 0)
                return;
			throughput /= transDirPdf;

			// Compute MIS weight for direct light at the next vertex
			#ifdef gNee
            misWeight = s.isSpecular ? 1 : DirectMis(s.mDirPdfW, directPdfA / WtoA);
			#endif
		}
	}

    // construct canonical sample at vertex
    __init(const PathTracer pt, const PathVertex vertex, const uint4 x, const uint2 pixelIndex) {
        make<true>(pt, vertex, x, pixelIndex);
    }

    // shift sample to vertex's domain
    __init(const PathTracer pt, const PathVertex vertex, const GISample sample, out float shiftJacobian) {
        make<false>(pt, vertex, sample.mRng, -1);
        if (sample.mPathLength != 0 && mSample.mPathLength != 0) {
            if (sample.mPathLength == mSample.mPathLength)
                shiftJacobian = safe_divide(sample.mReplayPdfW, mSample.mReplayPdfW);
            else
				shiftJacobian = 0;
        } else
			shiftJacobian = 1;
    }
};

struct PackedReservoirGI {
    PackedVertex mVertex;
    float mCachedTargetPdf;
    float W;
    float M;
    uint mPackedLocalDirIn;
    GISample mSample;

    property float3 mLocalDirIn {
        get { return unpackNormal(mPackedLocalDirIn); }
        set { mPackedLocalDirIn = packNormal(newValue); }
    }

    void Store(const RenderParams fb, const uint2 idx) {
        fb.mReservoirDataGI[0][idx] = reinterpret<float4>(mVertex);
        fb.mReservoirDataGI[1][idx] = float4(mCachedTargetPdf, W, M, asfloat(mPackedLocalDirIn));
        fb.mReservoirDataGI[2][idx] = reinterpret<float4>(mSample.mRng);
        fb.mReservoirDataGI[3][idx] = float4(mSample.mReplayPdfW, asfloat(mSample.mPathLength), 0, 0);
        fb.mReservoirDataGI[4][idx] = reinterpret<float4>(mSample.mReconnectionVertex.mVertex);
        fb.mReservoirDataGI[5][idx] = float4(mSample.mReconnectionVertex.mRadiance, asfloat(mSample.mReconnectionVertex.mPackedDirOut));
    }

    [mutating]
    void LoadPrev(const RenderParams fb, const uint2 idx) {
        mVertex = reinterpret<PackedVertex>(fb.mPrevReservoirDataGI[0][idx]);
        const float4 data = fb.mPrevReservoirDataGI[1][idx];
        mCachedTargetPdf = data.x;
        W = data.y;
        M = data.z;
        mPackedLocalDirIn = asuint(data.w);

        mSample.mRng = reinterpret<uint4>(fb.mPrevReservoirDataGI[2][idx]);

        const float4 data1 = fb.mPrevReservoirDataGI[3][idx];
        mSample.mReplayPdfW = data1.x;
        mSample.mPathLength = asuint(data1.y);

		mSample.mReconnectionVertex.mVertex = reinterpret<PackedVertex>(fb.mPrevReservoirDataGI[4][idx]);
		const float4 data2 = fb.mPrevReservoirDataGI[5][idx];
		mSample.mReconnectionVertex.mRadiance = data2.rgb;
		mSample.mReconnectionVertex.mPackedDirOut = asuint(data2.w);
    }
};

extension PathTracer {
	void ReservoirReuseGI(const PathVertex vertex, const PackedReservoirGI q, inout ReservoirSampleDataGI currentSample, inout float M, inout RandomSampler rng) {
		// q's sample in the current domain
		float jac_QtoCur;
		const ReservoirSampleDataGI newSample = ReservoirSampleDataGI(this, vertex, q.mSample, jac_QtoCur);

		// target pdf in q's domain of currentSample
		float phat_from_q = 0;
		{
			const PathVertex v_q = PathVertex(mScene, q.mVertex.getShadingData(mScene), q.mVertex.mInstancePrimitiveIndex, vertex.mCurrentMedium, q.mPackedLocalDirIn);
			float jac_curToQ;
			const ReservoirSampleDataGI currentSample_q = ReservoirSampleDataGI(this, v_q, currentSample.mSample, jac_curToQ);
			phat_from_q = currentSample_q.mCachedTargetPdf * jac_curToQ;
			DebugPrintf(rng.mState.xy, "%f %f", jac_QtoCur, jac_curToQ);
		}

		// Resampling MIS weights

		#ifdef gPairwiseMis
		const float R = M;
		M += q.M;
		// defensive pairwise mis
		const float m0 = (1 +      safe_divide(currentSample.mCachedTargetPdf, R * currentSample.mCachedTargetPdf + (M - R) * phat_from_q)) / M;
		const float m1 = (M - R) * safe_divide(q.mCachedTargetPdf, R * newSample.mCachedTargetPdf*jac_QtoCur + (M - R) * q.mCachedTargetPdf) / M;
		#elif defined(gTalbotMis)
		// generalized talbot mis
		const float m0 = safe_divide(M*currentSample.mCachedTargetPdf, M*currentSample.mCachedTargetPdf + q.M * phat_from_q);
		const float m1 = safe_divide(q.M * q.mCachedTargetPdf  , q.M * q.mCachedTargetPdf   + M * newSample.mCachedTargetPdf*jac_QtoCur);
		M += q.M;
		#else
		// naive mis
		const float m0 = safe_divide(M  , M   + (phat_from_q > 0 ? q.M : 0));
		const float m1 = safe_divide(q.M, q.M + (newSample.mCachedTargetPdf*jac_QtoCur > 0 ? M : 0));
		M += q.M;
		#endif

		// RIS weights

		const float w0 = m0 * currentSample.mCachedTargetPdf * currentSample.W;
		const float w1 = m1 *     newSample.mCachedTargetPdf * safe_divide(q.W, jac_QtoCur);

		// create new reservoir

		const float totalWeight = w0 + w1;
		if (rng.nextFloat().x * totalWeight <= w1) {
			currentSample = newSample;
		}

		currentSample.W = safe_divide(totalWeight, currentSample.mCachedTargetPdf);
	}

	float3 EstimateRadianceReSTIR(const PathVertex vertex, inout RandomSampler rng, const uint2 pixelIndex) {
		ReservoirSampleDataGI sample;
		float wsum = 0;

		// initial ris

        float3 directRadiance = 0;

		for (uint i = 0; i < gPushConstants.mGICandidateSamples; i++) {
            const ReservoirSampleDataGI xi = ReservoirSampleDataGI(this, vertex, rng.mState, pixelIndex);
            if (i == 0) {
                mFramebuffer.mAlbedo[pixelIndex] = float4(xi.mCachedAlbedo, 1);
                directRadiance = xi.mCachedDirectRadiance;
            }
			rng.skipNext(gPushConstants.mMaxDepth*(2 + gPushConstants.mMaxNullCollisions));
			if (xi.mCachedTargetPdf > 0) {
				const float wi = xi.mCachedTargetPdf * xi.W;
				wsum += wi;
				if (rng.nextFloat().x*wsum <= wi)
					sample = xi;
			}
		}


		if (sample.mCachedTargetPdf > 0)
			sample.W = wsum / (gPushConstants.mGICandidateSamples * sample.mCachedTargetPdf);
		else
			sample.W = 0;

		// reuse

		#ifdef gReSTIR_GI_Reuse

		float M = 1;

        if (gPushConstants.mReservoirHistoryValid != 0) {
            for (uint i = 0; i < gPushConstants.mGIReuseSamples; i++) {
				const float2 discRng = rng.nextFloat().xy;
				const int2 qi = int2(mFramebuffer.mPrevUVs[pixelIndex] * gPushConstants.mOutputExtent + sampleConcentricDisc(discRng.x, discRng.y) * gPushConstants.mGIReuseRadius);
				if (all(qi >= 0) && all(qi < gPushConstants.mOutputExtent)) {
					PackedReservoirGI q;
					q.LoadPrev(mFramebuffer, qi);
					if (q.M > 0)
						ReservoirReuseGI(vertex, q, sample, M, rng);
				}
            }
		}

		// store for reuse
		{
			PackedReservoirGI r;
			r.mSample = sample.mSample;
			r.mVertex.mLocalPosition = mScene.mInstanceInverseTransforms[vertex.mInstanceIndex].transformPoint(vertex.mShadingData.mPosition);
			r.mVertex.mInstancePrimitiveIndex = vertex.mInstancePrimitiveIndex;
			r.mCachedTargetPdf = sample.mCachedTargetPdf;
			r.W = sample.W;
			r.M = min(M, gPushConstants.mGIMaxM);
			r.mPackedLocalDirIn = vertex.mPackedLocalDirIn;
			r.Store(mFramebuffer, pixelIndex);
		}

		#endif

		return directRadiance + sample.mCachedContribution * sample.W;
	}
}