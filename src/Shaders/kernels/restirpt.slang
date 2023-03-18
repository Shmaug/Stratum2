struct PushConstants {
	float4 mSceneSphere;
    uint2 mOutputExtent;
	uint mViewCount;
    uint mMaxDepth;
	uint mRandomSeed;
    uint mMaxNullCollisions;
    uint mLightCount;
    uint mEnvironmentMaterialAddress;
    float mEnvironmentSampleProbability;
    float mMaxM;
    uint mCandidateSamples;
    uint mReservoirHistoryValid;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> gPushConstants;


#define gMaxNullCollisions            gPushConstants.mMaxNullCollisions
#define gLightCount                   gPushConstants.mLightCount
#define gEnvironmentMaterialAddress   gPushConstants.mEnvironmentMaterialAddress
#define gEnvironmentSampleProbability gPushConstants.mEnvironmentSampleProbability
#define gSceneSphere                  gPushConstants.mSceneSphere

#ifdef gLambertian
#include "materials/lambertian.hlsli"
#else
#include "materials/disney.hlsli"
#endif

#include "common/intersection.hlsli"
#include "common/lights.hlsli"

struct RenderParams {
    StructuredBuffer<ViewData> mViews;
    StructuredBuffer<TransformData> mViewTransforms;
    StructuredBuffer<TransformData> mViewInverseTransforms;
    StructuredBuffer<TransformData> mPrevViewInverseTransforms;
    StructuredBuffer<uint> mViewMediumIndices;
    RWTexture2D<float4> mOutput;
    RWTexture2D<float4> mAlbedo;
	RWTexture2D<float2> mPrevUVs;
    RWTexture2D<uint2> mVisibility;
	RWTexture2D<float4> mDepth;

    RWStructuredBuffer<PathState> mPathStates;

    RWTexture2D<float4> mReservoirData[3];
    RWTexture2D<float4> mPrevReservoirData[3];

    void StoreAuxillaryData(const ShadingData shadingData, const float3 albedo, const uint2 index, const uint instancePrimitiveIndex) {
        float2 prevPixelCoord = index + 0.5;

		const float depth = length(mViewInverseTransforms[0].transformPoint(shadingData.mPosition));

        float prevDepth = depth;
        if (BF_GET(instancePrimitiveIndex, 0, 16) != INVALID_INSTANCE) {
            const float3 prevCameraPos = mPrevViewInverseTransforms[0].transformPoint(gScene.mInstanceMotionTransforms[BF_GET(instancePrimitiveIndex, 0, 16)].transformPoint(shadingData.mPosition));
			prevDepth = length(prevCameraPos);
			mViews[0].toRaster(prevCameraPos, prevPixelCoord);
		}

        mPrevUVs[index] = prevPixelCoord / gPushConstants.mOutputExtent;
        mAlbedo[index] = float4(albedo, 1);

		VisibilityData v;
		v.mInstancePrimitiveIndex = instancePrimitiveIndex;
        v.mPackedNormal = shadingData.mPackedShadingNormal;
        mVisibility[index] = reinterpret<uint2>(v);

		DepthData d;
		d.mDepth = depth;
		d.mPrevDepth = prevDepth;
        d.mDepthDerivative = 1;
        mDepth[index] = reinterpret<float4>(d);
    }
};
ParameterBlock<RenderParams> gRenderParams;


float AdjustIntersectionDistance(const float d) {
    return max(d * 0.99, d - 1e-2);
}

interface Interaction {
    bool canEvaluate();
    bool isSingular();

    float3 emission();
    float emissionPdf();
    float3 albedo();

    float continuationProb();

    ReflectanceEvalRecord evaluateReflectance<let Adjoint : bool>(const float3 dirIn, const float3 dirOut);
    DirectionSampleRecord sampleDirection<let Adjoint : bool>(const float3 rnd, const float3 dirIn);

    float3 evaluateReflectanceFast<let Adjoint : bool>(const float3 localDirIn, const float3 localDirOut);
};
extension PackedMaterialData : Interaction {};
extension Medium : Interaction {};


struct PathVertex {
    ShadingData mShadingData;
    uint mInstancePrimitiveIndex;
    uint mCurrentMedium;
    uint mPackedLocalDirIn;
    property float3 mLocalDirIn {
        get { return unpackNormal(mPackedLocalDirIn); }
        set { mPackedLocalDirIn = packNormal(newValue); }
    };

    property uint mInstanceIndex {
        get { return BF_GET(mInstancePrimitiveIndex, 0, 16); }
        set { BF_SET(mInstancePrimitiveIndex, newValue, 0, 16); }
    };
    property uint mPrimitiveIndex {
        get { return BF_GET(mInstancePrimitiveIndex, 16, 16); }
        set { BF_SET(mInstancePrimitiveIndex, newValue, 16, 16); }
    };

    __init(const ShadingData shadingData, const uint instancePrimitiveIndex, const uint currentMedium, const uint packedLocalDirIn) {
        mShadingData = shadingData;
		mInstancePrimitiveIndex = instancePrimitiveIndex;
        mCurrentMedium = currentMedium;
		mPackedLocalDirIn = packedLocalDirIn;
        if (!(gHasMedia && mShadingData.isMedium())) {
            gScene.ApplyNormalMap(mShadingData);
        }
    }
    __init(const ShadingData shadingData, const uint instancePrimitiveIndex, const uint currentMedium, const float3 dirIn) {
        mShadingData = shadingData;
		mInstancePrimitiveIndex = instancePrimitiveIndex;
        mCurrentMedium = currentMedium;
        if (gHasMedia && mShadingData.isMedium()) {
            mLocalDirIn = dirIn;
        } else {
            gScene.ApplyNormalMap(mShadingData);
            mLocalDirIn = mShadingData.toLocal(dirIn);
        }
	}
};

// ----------------------------------------------------------------------------------
// ------------------------------ Direct illumination  ------------------------------
// ----------------------------------------------------------------------------------

struct ReservoirSampleDataDI {
    float4 mSample;
    float mSourcePdf;
    float mCachedWtoA;
    float3 mCachedContribution; // f * Le
    RayDesc mVisibilityRay;

    property float mTargetPdf { get { return luminance(mCachedContribution) * mCachedWtoA; } }
    property float mRISWeight { get { return mTargetPdf / mSourcePdf; } }

    __init(const PathVertex vertex, const Interaction bsdf, const float4 x) {
        mSample = x;
        mCachedContribution = 0;
        mCachedWtoA = 0;
        const IlluminationSampleRecord r = gScene.SampleIllumination(mSample, vertex.mShadingData.mPosition);
        mSourcePdf = r.mPdf;
        if (luminance(r.mRadiance) > 0 && r.mPdf > 0) {
            const bool isMedium = gHasMedia && vertex.mShadingData.isMedium();

            const float3 localDirOut = isMedium ? r.mDirectionToLight : vertex.mShadingData.toLocal(r.mDirectionToLight);

            ReflectanceEvalRecord f = bsdf.evaluateReflectance<false>(vertex.mLocalDirIn, localDirOut);
            if (!isMedium)
                f.mReflectance *= vertex.mShadingData.shadingNormalCorrection<false>(vertex.mLocalDirIn, localDirOut);
            const float contProb = bsdf.continuationProb();
            f.mFwdPdfW *= contProb;
            f.mRevPdfW *= contProb;

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
    ShadingData getShadingData() {
        const uint instanceIndex = getInstanceIndex();
        ShadingData sd = gScene.makeShadingData(gScene.mInstances[instanceIndex], gScene.mInstanceTransforms[instanceIndex], mLocalPosition, getPrimitiveIndex());
        sd.mTexcoordScreenSize = 0;
        return sd;
    }

    void Store(const uint2 idx) {
        float4 data;
        data.xyz = mLocalPosition;
        data.w = asfloat(mInstancePrimitiveIndex);
        gRenderParams.mReservoirData[0][idx] = data;
        gRenderParams.mReservoirData[1][idx] = mSample;
        gRenderParams.mReservoirData[2][idx] = float4(mCachedTargetPdf, W, M, asfloat(mPackedLocalDirIn));
    }

    [mutating]
    void LoadPrev(const uint2 idx) {
        const float4 data = gRenderParams.mPrevReservoirData[0][idx];
        mLocalPosition = data.xyz;
        mInstancePrimitiveIndex = asuint(data.w);
        mSample = gRenderParams.mPrevReservoirData[1][idx];
        const float4 data2 = gRenderParams.mPrevReservoirData[2][idx];
        mCachedTargetPdf = data2.x;
        W = data2.y;
        M = data2.z;
        mPackedLocalDirIn = asuint(data2.w);
    }
};

void ReservoirReuseDI(const PathVertex vertex, const Interaction bsdf, const PackedReservoirDI q, inout ReservoirSampleDataDI currentSample, inout float integrationWeight, inout float M, inout RandomSampler rng) {
    // q's sample in the current domain
    ReservoirSampleDataDI newSample = ReservoirSampleDataDI(vertex, bsdf, q.mSample);
	#ifdef gReSTIR_DI_Reuse_Visibility
    if (newSample.mTargetPdf > 0) {
        float transDirPdf, transNeePdf;
        gScene.traceVisibilityRay(newSample.mVisibilityRay, rng, vertex.mCurrentMedium, newSample.mCachedContribution, transDirPdf, transNeePdf);
    }
	#endif

    // target pdf in q's domain of currentSample
    float phat_q = 0;
    {
        const PathVertex v_q = PathVertex(q.getShadingData(), q.mInstancePrimitiveIndex, vertex.mCurrentMedium, q.mPackedLocalDirIn);
        ReservoirSampleDataDI currentSample_q;
        if (gHasMedia && v_q.mShadingData.isMedium())
            currentSample_q = ReservoirSampleDataDI(v_q, Medium(v_q.mShadingData.getMaterialAddress()), currentSample.mSample);
		else
        	currentSample_q = ReservoirSampleDataDI(v_q, gScene.LoadMaterial(v_q.mShadingData), currentSample.mSample);

        if (currentSample_q.mTargetPdf > 0) {
			#ifdef gReSTIR_DI_Reuse_Visibility
            // note: this ray should really be traced through the previous frame's acceleration structure
            float transDirPdf, transNeePdf;
            gScene.traceVisibilityRay(currentSample_q.mVisibilityRay, rng, vertex.mCurrentMedium, currentSample_q.mCachedContribution, transDirPdf, transNeePdf);
			#endif
            phat_q = currentSample_q.mTargetPdf;
        }
    }

    // Resampling MIS weights

    const float m0 = M / (M + (phat_q > 0 ? q.M : 0));
    const float m1 = q.M / (q.M + (newSample.mTargetPdf > 0 ? M : 0));
    // const float m0 = mis(M*currentSample.mTargetPdf, q.M * phat_q);
    // const float m1 = mis(q.M * q.mCachedTargetPdf  , M * newSample.mTargetPdf);

    // RIS weights

    const float w0 = m0 * currentSample.mTargetPdf * integrationWeight;
    const float w1 = m1 * newSample.mTargetPdf * q.W;

    M += q.M;

    // create new reservoir from current and new reservoirs

    const float totalWeight = w0 + w1;
    if (rng.nextFloat().x * totalWeight <= w1) {
        currentSample = newSample;
    }

    if (currentSample.mTargetPdf > 0)
        integrationWeight = totalWeight / currentSample.mTargetPdf;
    else
        integrationWeight = 0;
}

float3 DirectIlluminationReSTIR(const PathVertex vertex, const Interaction bsdf, inout RandomSampler rng, const uint2 pixelIndex) {
    ReservoirSampleDataDI sample;
    float wsum = 0;

    // initial ris

    for (uint i = 0; i < gPushConstants.mCandidateSamples; i++) {
        const ReservoirSampleDataDI xi = ReservoirSampleDataDI(vertex, bsdf, rng.nextFloat());
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
        gScene.traceVisibilityRay(sample.mVisibilityRay, rng, vertex.mCurrentMedium, sample.mCachedContribution, transDirPdf, transNeePdf);
    }
	#endif

	if (sample.mTargetPdf > 0)
		integrationWeight = wsum / (gPushConstants.mCandidateSamples * sample.mTargetPdf);

    // reuse

	#ifdef gReSTIR_DI_Reuse

    float M = gPushConstants.mCandidateSamples;

    if (gPushConstants.mReservoirHistoryValid != 0) {
        {
            const float2 discRng = rng.nextFloat().xy;
            const int2 qi = int2(gRenderParams.mPrevUVs[pixelIndex] * gPushConstants.mOutputExtent + sampleConcentricDisc(discRng.x, discRng.y) * 64);
            if (all(qi >= 0) && all(qi < gPushConstants.mOutputExtent)) {
                PackedReservoirDI q;
                q.LoadPrev(qi);
                if (q.M > 0) {
                    ReservoirReuseDI(vertex, bsdf, q, sample, integrationWeight, M, rng);
                }
            }
        }
    }

    // store for reuse
    {
        PackedReservoirDI r;
        r.mLocalPosition = gScene.mInstanceInverseTransforms[vertex.mInstanceIndex].transformPoint(vertex.mShadingData.mPosition);
        r.mInstancePrimitiveIndex = vertex.mInstancePrimitiveIndex;
        r.mSample = sample.mSample;
        r.mCachedTargetPdf = sample.mTargetPdf;
        r.W = integrationWeight;
        r.M = min(M, gPushConstants.mMaxM * gPushConstants.mCandidateSamples);
        r.mPackedLocalDirIn = vertex.mPackedLocalDirIn;
        r.Store(pixelIndex);
    }

	#ifndef gReSTIR_DI_Reuse_Visibility
    if (sample.mTargetPdf > 0) {
        float transDirPdf, transNeePdf;
        gScene.traceVisibilityRay(sample.mVisibilityRay, rng, vertex.mCurrentMedium, sample.mCachedContribution, transDirPdf, transNeePdf);
    }
	#endif

	#endif

	return sample.mCachedContribution * integrationWeight * sample.mCachedWtoA;
}

float3 DirectIlluminationSimple(const PathVertex vertex, const Interaction bsdf, inout RandomSampler rng, const uint2 pixelIndex) {
    const IlluminationSampleRecord r = gScene.SampleIllumination(rng.nextFloat(), vertex.mShadingData.mPosition);
    if (all(r.mRadiance <= 0) || r.mPdf <= 0)
        return 0;

    float WtoA = 1;
    if (r.isFinite)
        WtoA = r.mCosLight / pow2(r.mDistanceToLight);

    const bool isMedium = gHasMedia && vertex.mShadingData.isMedium();
    const float3 localDirOut = isMedium ? r.mDirectionToLight : vertex.mShadingData.toLocal(r.mDirectionToLight);

    ReflectanceEvalRecord f = bsdf.evaluateReflectance<false>(vertex.mLocalDirIn, localDirOut);
    if (!isMedium)
        f.mReflectance *= vertex.mShadingData.shadingNormalCorrection<false>(vertex.mLocalDirIn, localDirOut);
    const float contProb = bsdf.continuationProb();
    f.mFwdPdfW *= contProb;
    f.mRevPdfW *= contProb;

    float3 contrib = (r.mRadiance * f.mReflectance) * (WtoA / r.mPdf);

    // evaluate transmittance along shadow ray

    const float3 rayOrigin = isMedium ?
			vertex.mShadingData.mPosition :
			rayOffset(vertex.mShadingData.mPosition, vertex.mShadingData.getGeometryNormal(), r.mDirectionToLight);

    float transDirPdf, transNeePdf;
    gScene.traceVisibilityRay(
        makeRay(rayOrigin, r.mDirectionToLight, 0, AdjustIntersectionDistance(r.mDistanceToLight)),
        rng, vertex.mCurrentMedium, contrib, transDirPdf, transNeePdf);

    if (all(contrib <= 0) || transNeePdf <= 0)
        return 0;

    return contrib / transNeePdf;
}

#ifdef gReSTIR_DI
#define DirectIllumination DirectIlluminationReSTIR
#else
#define DirectIllumination DirectIlluminationSimple
#endif

// ----------------------------------------------------------------------------------
// --------------------------------- Path tracing  ----------------------------------
// ----------------------------------------------------------------------------------

struct PathState {
    float3 mRayOrigin;
    uint mPackedCurrentMedium;
    float3 mThroughput;
    uint mPackedRayDirection;
    RandomSampler mRng;
    float3 mRadiance;

    property uint mCurrentMedium {
        get { return BF_GET(mPackedCurrentMedium, 0, 16); }
        set { BF_SET(mPackedCurrentMedium, newValue, 0, 16); }
    }
    property uint mPathLength {
        get { return BF_GET(mPackedCurrentMedium, 16, 16); }
        set { BF_SET(mPackedCurrentMedium, newValue, 16, 16); }
    }

    property float3 mRayDirection {
        get { return unpackNormal(mPackedRayDirection); }
        set { mPackedRayDirection = packNormal(newValue); }
    }

	__init(const uint2 index) {
		mPathLength = 0;
		mThroughput = 1;
		mRadiance = 0;
        mRng = RandomSampler(gPushConstants.mRandomSeed, index);

        gRenderParams.mOutput[index] = float4(0, 0, 0, 1);

		#ifdef gPixelJitter
		const float2 offset = mRng.nextFloat().xy;
		#else
		const float2 offset = 0.5;
		#endif

		const uint viewIndex = getViewIndex(index, gPushConstants.mOutputExtent, gPushConstants.mViewCount);

		float2 uv;
		const float3 localDir = gRenderParams.mViews[viewIndex].toWorld(index + offset, /*out*/ uv);
		const TransformData t = gRenderParams.mViewTransforms[viewIndex];

		mRayOrigin = float3(t.m[0][3], t.m[1][3], t.m[2][3]);
		mCurrentMedium = gRenderParams.mViewMediumIndices[viewIndex];
		mRayDirection = normalize(t.transformVector(localDir));
	}

    [mutating]
    PathVertex TraceRay() {
		IntersectionResult isect;
		float transDirPdf, transNeePdf;
		const bool hit = gScene.traceScatteringRay(makeRay(mRayOrigin, mRayDirection), mRng, mCurrentMedium, mThroughput, transDirPdf, transNeePdf, isect);
		mThroughput /= transDirPdf;
		mPathLength++;

		if (!hit) {
			float dirPdfW;
			const float3 emission = gScene.EvaluateEnvironment(mRayDirection, dirPdfW);
            if (any(emission > 0)) {
                mRadiance += mThroughput * emission;
			}
			mThroughput = 0;
		}

        isect.mShadingData.mTexcoordScreenSize = 0; // disable mipmaps for now
        return PathVertex(isect.mShadingData, isect.mInstancePrimitiveIndex, mCurrentMedium, -mRayDirection);
    }

    [mutating]
    bool SampleDirection(const PathVertex vertex, const Interaction bsdf) {
        // only account for light paths with >2 segments
        // paths with <=2 segments are sampled explicitly
        if (vertex.mLocalDirIn.z > 0 && mPathLength > 2)
            mRadiance += mThroughput * bsdf.emission();

        if (!bsdf.canEvaluate() || mPathLength >= gPushConstants.mMaxDepth)
            return false;

        const float continueProb = min(1, bsdf.continuationProb());
        if (mRng.nextFloat().x > continueProb)
            return false;

        DirectionSampleRecord s = bsdf.sampleDirection<false>(mRng.nextFloat().xyz, vertex.mLocalDirIn);
        s.mFwdPdfW *= continueProb;
        s.mRevPdfW *= continueProb;

        mThroughput *= s.mReflectance / s.mFwdPdfW;

        if (gHasMedia && vertex.mShadingData.isMedium()) {
            mRayDirection = s.mDirection;
            mRayOrigin = vertex.mShadingData.mPosition;
        } else {
            mThroughput *= vertex.mShadingData.shadingNormalCorrection<false>(vertex.mLocalDirIn, s.mDirection);
            const float3 worldDir = vertex.mShadingData.toWorld(s.mDirection);
            mRayDirection = worldDir;
            mRayOrigin = rayOffset(vertex.mShadingData.mPosition, vertex.mShadingData.getGeometryNormal(), worldDir);
        }

        return true;
	}

    [mutating]
    void ProcessPrimaryVertex(const PathVertex vertex, const Interaction bsdf, const uint2 pixelIndex) {
		gRenderParams.StoreAuxillaryData(vertex.mShadingData, bsdf.albedo(), pixelIndex, vertex.mInstancePrimitiveIndex);

        if (!(gHasMedia && vertex.mShadingData.isSurface()) && vertex.mLocalDirIn.z > 0)
            gRenderParams.mOutput[pixelIndex].rgb += mThroughput * bsdf.emission();

        if (!bsdf.canEvaluate()) {
            mThroughput = 0;
            return;
        }
        if (!bsdf.isSingular() && gPushConstants.mMaxDepth > 1)
            gRenderParams.mOutput[pixelIndex].rgb += mThroughput * DirectIllumination(vertex, bsdf, mRng, pixelIndex);

		#ifndef gNoIndirect
        if (!SampleDirection(vertex, bsdf))
			mThroughput = 0;
		#endif
	}
};


[shader("compute")]
[numthreads(8, 4, 1)]
void Render(uint3 index: SV_DispatchThreadID) {
	if (any(index.xy >= gPushConstants.mOutputExtent))
		return;

    PathState pathState = PathState(index.xy);

	// trace primary ray and sample direct illumination
    {
        const PathVertex vertex = pathState.TraceRay();

        if (vertex.mShadingData.isEnvironment()) {
            gRenderParams.StoreAuxillaryData(vertex.mShadingData, 1, index.xy, vertex.mInstancePrimitiveIndex);
            return;
        }

        if (gHasMedia && vertex.mShadingData.isMedium())
            pathState.ProcessPrimaryVertex(vertex, Medium(vertex.mShadingData.getMaterialAddress()), index.xy);
        else
            pathState.ProcessPrimaryVertex(vertex, gScene.LoadMaterial(vertex.mShadingData), index.xy);
	}

	#ifndef gNoIndirect
    {
		const uint4 indirectLightSample = pathState.mRng.mState;

		// indirect light
		while (any(pathState.mThroughput > 0) && pathState.mPathLength + 1 <= gPushConstants.mMaxDepth) {
			const PathVertex vertex = pathState.TraceRay();
			if (all(pathState.mThroughput <= 0))
				break;
			if (gHasMedia && vertex.mShadingData.isMedium()) {
				if (!pathState.SampleDirection(vertex, Medium(vertex.mShadingData.getMaterialAddress())))
					break;
			} else {
				if (!pathState.SampleDirection(vertex, gScene.LoadMaterial(vertex.mShadingData)))
					break;
			}
		}

		if (any(pathState.mRadiance > 0))
			gRenderParams.mOutput[index.xy].rgb += pathState.mRadiance;
    }
	#endif
}