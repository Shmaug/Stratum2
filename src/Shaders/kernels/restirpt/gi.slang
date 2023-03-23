#include "di.slang"

struct VertexSampleRecord {
    float3 mLe;
    float3 mNeeLe;
    float3 mBrdf;
    float3 mDirection;
    float mDirPdfW;
    bool isSpecular;
    bool isSpecularLobe;
};

VertexSampleRecord ProcessVertex(const PathVertex vertex, const Interaction bsdf, inout RandomSampler rng, const bool sampleDirection, const bool sampleNee) {
    VertexSampleRecord r;
    r.mLe = 0;
    r.mNeeLe = 0;
	r.mBrdf = 0;
	r.mDirPdfW = 0;

    const float3 localDirIn = vertex.mLocalDirIn;

    if (vertex.mShadingData.isEnvironment()) {
		float tmp;
		r.mLe = gScene.EvaluateEnvironment(-localDirIn, tmp);
        return r;
	}

    if ((gHasMedia && vertex.mShadingData.isMedium()) || localDirIn.z > 0) {
        r.mLe = bsdf.emission();
    }

    r.isSpecular = bsdf.isSingular();
    r.isSpecularLobe = false;

	if (sampleDirection) {
        const DirectionSampleRecord s = bsdf.sampleDirection<false>(rng.nextFloat().xyz, localDirIn);
        r.isSpecularLobe = s.isSingular();
		r.mBrdf = s.mReflectance * vertex.mShadingData.shadingNormalCorrection<false>(localDirIn, s.mDirection);
		r.mDirPdfW = s.mFwdPdfW;
		r.mDirection = (gHasMedia && vertex.mShadingData.isMedium()) ? s.mDirection : vertex.mShadingData.toWorld(s.mDirection);
	}

	if (!r.isSpecular && sampleNee) {
        float tmp;
        r.mNeeLe = DirectIlluminationSimple(vertex, bsdf, rng, tmp, sampleDirection);
    }

    return r;
}
VertexSampleRecord ProcessVertex(const PathVertex vertex, inout RandomSampler rng, const bool sampleNext, const bool sampleNee) {
    if (gHasMedia && vertex.mShadingData.isMedium())
        return ProcessVertex(vertex, Medium(vertex.mShadingData.getMaterialAddress()), rng, sampleNext, sampleNee);
	else
        return ProcessVertex(vertex, gScene.LoadMaterial(vertex.mShadingData), rng, sampleNext, sampleNee);
}
PathVertex TraceNext(const PathVertex vertex, inout RandomSampler rng, const float3 direction, inout float3 throughput, out float transDirPdf, out float directPdfA, out float WtoA) {
	const float3 origin = (gHasMedia && vertex.mShadingData.isMedium()) ?
		vertex.mShadingData.mPosition :
		rayOffset(vertex.mShadingData.mPosition, vertex.mShadingData.getGeometryNormal(), direction);

	uint medium = vertex.mCurrentMedium;

	IntersectionResult isect;
    float transNeePdf;
    const bool hit = gScene.traceScatteringRay(makeRay(origin, direction), rng, medium, throughput, transDirPdf, transNeePdf, isect);

    PathVertex v = PathVertex(isect.mShadingData, isect.mInstancePrimitiveIndex, medium, -direction); ;
    v.mShadingData.mTexcoordScreenSize = 0; // disable mipmaps for now

    if (hit) {
        directPdfA = isect.LightSamplePdfA();
        WtoA = abs((gHasMedia && v.mShadingData.isMedium()) ? 1 : v.mLocalDirIn.z) / pow2(isect.mDistance);
    } else {
        gScene.EvaluateEnvironment(direction, directPdfA);
        if (gLightCount > 0)
            directPdfA *= gEnvironmentSampleProbability;
        WtoA = 1;
    }

	return v;
}


struct PackedVertex {
    float3 mLocalPosition;
    uint mInstancePrimitiveIndex;

    uint getInstanceIndex()  { return BF_GET(mInstancePrimitiveIndex, 0, 16); }
    uint getPrimitiveIndex() { return BF_GET(mInstancePrimitiveIndex, 16, 16); }
    ShadingData getShadingData() {
        const uint instanceIndex = getInstanceIndex();
        ShadingData sd = gScene.makeShadingData(gScene.mInstances[instanceIndex], gScene.mInstanceTransforms[instanceIndex], mLocalPosition, getPrimitiveIndex());
        sd.mTexcoordScreenSize = 0;
        return sd;
    }
};
struct ReconnectionVertex {
    PackedVertex mVertex;
    float3 mRadiance;
    uint mPackedDirOut;

    property float3 mDirOut {
        get { return unpackNormal(mPackedDirOut); }
        set { mPackedDirOut = packNormal(newValue); }
	}
};

struct GISample {
    uint4 mRng;
    float mReplayPdfA;
    uint mPathLength;
};

struct ReservoirSampleDataGI {
    GISample mSample;
    float3 mCachedContribution;
    float mCachedTargetPdf;
    float W;

	[mutating]
    void make(PathVertex vertex, const uint4 x) {
        mSample.mRng = x;
        mSample.mReplayPdfA = 0;
        mSample.mPathLength = 0;
        mCachedContribution = 0;
        mCachedTargetPdf = 0;
        W = 0;

        float pdfA = 1;

        RandomSampler rng;
        rng.mState = x;

        float3 throughput = 1;
        for (uint pathLength = 1; pathLength <= gPushConstants.mMaxDepth; pathLength++) {
            const VertexSampleRecord s = ProcessVertex(vertex, rng, pathLength < gPushConstants.mMaxDepth, false);

            if (pathLength > 2 && any(s.mLe > 0)) {
                mCachedContribution = throughput * s.mLe;
                mCachedTargetPdf = luminance(mCachedContribution);
                W = 1 / pdfA;
                mSample.mReplayPdfA = pdfA;
                mSample.mPathLength = pathLength;
                break;
            }

            if (s.mDirPdfW <= 0 || all(s.mBrdf <= 0)) break;
            throughput *= s.mBrdf;
            pdfA *= s.mDirPdfW;

            float transDirPdf, directPdfA, WtoA;
            vertex = TraceNext(vertex, rng, s.mDirection, throughput, transDirPdf, directPdfA, WtoA);
            if (transDirPdf <= 0 || all(throughput <= 0)) break;
            pdfA *= transDirPdf;
            pdfA *= WtoA;
            throughput *= WtoA;
        }
	}

    // construct canonical sample at vertex
    __init(const PathVertex vertex, const uint4 x) {
        make(vertex, x);
    }

    // shift sample to vertex's domain
    __init(const PathVertex vertex, const GISample sample, out float shiftJacobian) {
        make(vertex, sample.mRng);
        if (mSample.mReplayPdfA > 0 && sample.mPathLength == mSample.mPathLength)
            shiftJacobian = sample.mReplayPdfA / mSample.mReplayPdfA;
        else
            shiftJacobian = 0;
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

    void Store(const uint2 idx) {
        gRenderParams.mReservoirDataGI[0][idx] = reinterpret<float4>(mVertex);
        gRenderParams.mReservoirDataGI[1][idx] = float4(mCachedTargetPdf, W, M, asfloat(mPackedLocalDirIn));
        gRenderParams.mReservoirDataGI[2][idx] = reinterpret<float4>(mSample.mRng);
        gRenderParams.mReservoirDataGI[3][idx] = float4(mSample.mReplayPdfA, asfloat(mSample.mPathLength), 0, 0);
    }

    [mutating]
    void LoadPrev(const uint2 idx) {
        mVertex = reinterpret<PackedVertex>(gRenderParams.mPrevReservoirDataGI[0][idx]);
        const float4 data = gRenderParams.mPrevReservoirDataGI[1][idx];
        mCachedTargetPdf = data.x;
        W = data.y;
        M = data.z;
        mPackedLocalDirIn = asuint(data.w);

        mSample.mRng = reinterpret<uint4>(gRenderParams.mPrevReservoirDataGI[2][idx]);

        const float4 data1 = gRenderParams.mPrevReservoirDataGI[3][idx];
        mSample.mReplayPdfA = data1.x;
        mSample.mPathLength = asuint(data1.y);
    }
};

void ReservoirReuseGI(const PathVertex vertex, const PackedReservoirGI q, inout ReservoirSampleDataGI currentSample, inout float M, inout RandomSampler rng) {
	// q's sample in the current domain
    float jac_QtoCur;
    const ReservoirSampleDataGI newSample = ReservoirSampleDataGI(vertex, q.mSample, jac_QtoCur);

    // target pdf in q's domain of currentSample
    float phat_from_q = 0;
    {
        const PathVertex v_q = PathVertex(q.mVertex.getShadingData(), q.mVertex.mInstancePrimitiveIndex, vertex.mCurrentMedium, q.mPackedLocalDirIn);
        float jac_curToQ;
        const ReservoirSampleDataGI currentSample_q = ReservoirSampleDataGI(v_q, currentSample.mSample, jac_curToQ);
        phat_from_q = currentSample_q.mCachedTargetPdf;
        // jacobian is 0 if the current path is 0, so dont apply it in that case
        if (currentSample.mSample.mReplayPdfA > 0)
            phat_from_q *= jac_curToQ;
        else
            jac_QtoCur = 1;
    }


    // Resampling MIS weights

    // naive mis
    // const float m0 = M / (M + (phat_from_q > 0 ? q.M : 0));
    // const float m1 = q.M / (q.M + (newSample.mTargetPdf > 0 ? M : 0));

    // generalized talbot mis
    const float m0 = safe_divide(currentSample.mCachedTargetPdf*M  , currentSample.mCachedTargetPdf*M   + phat_from_q*q.M);
    const float m1 = safe_divide(            q.mCachedTargetPdf*q.M,             q.mCachedTargetPdf*q.M + newSample.mCachedTargetPdf * jac_QtoCur*q.M);

    const float R = M;
    M += q.M;

    // defensive pairwise mis
    //const float m0 = (1 +      safe_divide(currentSample.mCachedTargetPdf, R * currentSample.mCachedTargetPdf + (M - R) * phat_from_q      )) / M;
    //const float m1 = (M - R) * safe_divide(            q.mCachedTargetPdf, R *     newSample.mCachedTargetPdf + (M - R) * q.mCachedTargetPdf) / M;

    // RIS weights

    const float w0 = m0 * currentSample.mCachedTargetPdf * currentSample.W;
    const float w1 = m1 * newSample.mCachedTargetPdf * q.W * jac_QtoCur;

    // create new reservoir

    const float totalWeight = w0 + w1;
    if (rng.nextFloat().x * totalWeight <= w1) {
        currentSample = newSample;
    }

    if (currentSample.mCachedTargetPdf > 0)
        currentSample.W = totalWeight / currentSample.mCachedTargetPdf;
    else
        currentSample.W = 0;
}

float3 GlobalIlluminationReSTIR(const PathVertex vertex, inout RandomSampler rng, const uint2 pixelIndex) {
    ReservoirSampleDataGI sample;

   	float wsum = 0;

    // initial ris

    for (uint i = 0; i < gPushConstants.mGICandidateSamples; i++) {
        const ReservoirSampleDataGI xi = ReservoirSampleDataGI(vertex, rng.mState);
        rng.skipNext(100);
        if (xi.mCachedTargetPdf > 0) {
            const float wi = xi.mCachedTargetPdf * xi.W;
            wsum += wi;
            if (rng.nextFloat().x <= wi / wsum)
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
        for (uint i = 0; i < 8; i++) {
			const float2 discRng = rng.nextFloat().xy;
			const int2 qi = int2(gRenderParams.mPrevUVs[pixelIndex] * gPushConstants.mOutputExtent + sampleConcentricDisc(discRng.x, discRng.y) * gPushConstants.mGIReuseRadius);
			if (all(qi >= 0) && all(qi < gPushConstants.mOutputExtent)) {
				PackedReservoirGI q;
                q.LoadPrev(qi);
                if (q.M <= 0 || q.mSample.mPathLength != sample.mSample.mPathLength)
                    continue;
				ReservoirReuseGI(vertex, q, sample, M, rng);
                break;
			}
        }
    }

    // store for reuse
    {
        PackedReservoirGI r;
        r.mSample = sample.mSample;
        r.mVertex.mLocalPosition = gScene.mInstanceInverseTransforms[vertex.mInstanceIndex].transformPoint(vertex.mShadingData.mPosition);
        r.mVertex.mInstancePrimitiveIndex = vertex.mInstancePrimitiveIndex;
        r.mCachedTargetPdf = sample.mCachedTargetPdf;
        r.W = sample.W;
        r.M = min(M, gPushConstants.mGIMaxM);
        r.mPackedLocalDirIn = vertex.mPackedLocalDirIn;
        r.Store(pixelIndex);
    }

	#endif

	return sample.mCachedContribution * sample.W;
}

float3 GlobalIlluminationSimple(PathVertex vertex, inout RandomSampler rng, const uint2 pixelIndex) {
    float3 radiance = 0;
    float3 throughput = 1;
    float misWeight = 0; // emission weight at first vertex is 0, since it's handled by the caller
    for (uint depth = 1; depth <= gPushConstants.mMaxDepth; depth++) {
        const VertexSampleRecord s = ProcessVertex(vertex, rng, depth < gPushConstants.mMaxDepth, depth < gPushConstants.mMaxDepth && depth > 1);

        // note: s.mNeeLe already has mis weight applied
		radiance += throughput * (s.mNeeLe + s.mLe * misWeight);

        if (depth == gPushConstants.mMaxDepth || s.mDirPdfW <= 0 || all(s.mBrdf <= 0))
            break;

        throughput *= s.mBrdf / s.mDirPdfW;

        float transDirPdf, directPdfA, WtoA;
        vertex = TraceNext(vertex, rng, s.mDirection, throughput, transDirPdf, directPdfA, WtoA);
        if (transDirPdf <= 0)
            break;
        throughput /= transDirPdf;

        if (s.isSpecular)
            misWeight = 1;
        else if (depth == 1)
            misWeight = 0; // diffuse direct light handled by DirectIllumination
        else
            misWeight = DirectMis(s.mDirPdfW * WtoA, directPdfA);
    }
    return radiance;
}

#ifdef gReSTIR_GI
#define GlobalIllumination GlobalIlluminationReSTIR
#else
#define GlobalIllumination GlobalIlluminationSimple
#endif