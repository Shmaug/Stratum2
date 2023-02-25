#ifndef gSampleDirectIllumination
#define gSampleDirectIllumination false
#endif
#ifndef gDebugPaths
#define gDebugPaths false
#endif
#ifndef gMultiDispatch
#define gMultiDispatch false
#endif
#ifndef gDeferShadowRays
#define gDeferShadowRays false
#endif
#ifndef gSortShadowRays
#define gSortShadowRays false
#endif

struct PushConstants {
	uint mRandomSeed;
    uint mRRDepth;
    uint mMaxDepth;
    uint mMaxNullCollisions;
    uint2 mOutputExtent;
    uint mViewCount;
    uint mLightCount;
    uint mEnvironmentMaterialAddress;
    float mEnvironmentSampleProbability;
    float4 mSceneSphere;
    uint mDebugPathLengths;
    uint mRenderIteration;
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
#include "common/hashgrid.hlsli"


struct PathState {
    float3 mRayOrigin;
    uint mCurrentMedium;
    float3 mThroughput;
    uint mPackedRayDirection;
    RandomSampler mRng;
    uint mPathLength; // number of segments
    uint mPixelIndex;
    uint pad;
    float mDirPdfW;

    float3 getRayDirection() {
		return unpackNormal(mPackedRayDirection);
	}

    uint2 getPixelCoord() {
        return uint2(mPixelIndex % gPushConstants.mOutputExtent.x, mPixelIndex / gPushConstants.mOutputExtent.x);
	}
};

struct ShadowRay {
    float3 mContribution;
    float pad;
    float3 mRayOrigin;
    uint mPathLength;
    float3 mRayDirection;
    float mDistance;
    float mNeePdfW;
    float mDirPdfW;
    uint mPixelIndex;
    uint mLightVertices;

    uint2 getPixelCoord() {
        return uint2(mPixelIndex % gPushConstants.mOutputExtent.x, mPixelIndex / gPushConstants.mOutputExtent.x);
    }
};
#define gQuantizationFactor 16384

struct RenderParams {
    StructuredBuffer<ViewData> mViews;
    StructuredBuffer<TransformData> mViewTransforms;
    StructuredBuffer<TransformData> mPrevViewInverseTransforms;
    StructuredBuffer<uint> mViewMediumIndices;
    RWTexture2D<float4> mOutput;
    RWTexture2D<float4> mAlbedo;
	RWTexture2D<float2> mPrevUVs;
    RWTexture2D<uint2> mVisibility;
	RWTexture2D<float4> mDepth;

    RWStructuredBuffer<PathState> mPathStates;

    HashGrid<ShadowRay> mShadowRays;
    RWStructuredBuffer<uint4> mOutputAtomic;

	void StoreAuxillaryData(const uint2 index, const IntersectionResult isect, const float3 albedo) {
		float2 prevPixelCoord = index + 0.5;

		float prevDepth = isect.mDistance;
		if (isect.mInstanceIndex != INVALID_INSTANCE) {
			const float3 prevCameraPos = mPrevViewInverseTransforms[0].transformPoint(gScene.mInstanceMotionTransforms[isect.mInstanceIndex].transformPoint(isect.mShadingData.mPosition));
			prevDepth = length(prevCameraPos);
			mViews[0].toRaster(prevCameraPos, prevPixelCoord);
		}

		mPrevUVs[index] = prevPixelCoord / gPushConstants.mOutputExtent;
		mAlbedo[index] = float4(albedo, 1);

		VisibilityData v;
		v.mInstancePrimitiveIndex = isect.mInstancePrimitiveIndex;
		v.mPackedNormal = isect.mShadingData.mPackedShadingNormal;
		mVisibility[index] = reinterpret<uint2>(v);

		DepthData d;
		d.mDepth = isect.mDistance;
		d.mPrevDepth = prevDepth;
		d.mDepthDerivative = 1;
        mDepth[index] = reinterpret<float4>(d);
    }

    void AddRadiance(const uint2 index, const float3 radiance, const float weight, const uint viewVertices, const uint lightVertices) {
        if (gDebugPaths) {
            if (BF_GET(gPushConstants.mDebugPathLengths, 0, 16) == viewVertices + lightVertices && BF_GET(gPushConstants.mDebugPathLengths, 16, 16) == lightVertices)
                mOutput[index].rgb += radiance;
            return;
        }
        if (any(radiance > 0) && weight > 0)
            mOutput[index].rgb += radiance * weight;
    }

    void AddRadianceInterlocked(const uint2 index, const float3 radiance) {
        const uint idx = index.y * gPushConstants.mOutputExtent.x + index.x;
        const int3 ri = int3(max(0,radiance) * gQuantizationFactor);
        uint3 p;
        InterlockedAdd(mOutputAtomic[idx][0], ri[0], p[0]);
        InterlockedAdd(mOutputAtomic[idx][1], ri[1], p[1]);
        InterlockedAdd(mOutputAtomic[idx][2], ri[2], p[2]);
	}

    void AddRadianceInterlocked(const uint2 index, const float3 radiance, const float weight, const uint viewVertices, const uint lightVertices) {
        if (gDebugPaths) {
            if (BF_GET(gPushConstants.mDebugPathLengths, 0, 16) == viewVertices + lightVertices && BF_GET(gPushConstants.mDebugPathLengths, 16, 16) == lightVertices)
                AddRadianceInterlocked(index, radiance);
            return;
        }
        if (any(radiance > 0) && weight > 0)
            AddRadianceInterlocked(index, radiance * weight);
    }
};
ParameterBlock<RenderParams> gRenderParams;



interface Interaction {
    bool canEvaluate();
    bool isSingular();

    float3 emission();
    float emissionPdf();
    float3 albedo();

    float continuationProb();

    ReflectanceEvalRecord evaluateReflectance<let Adjoint : bool>(const float3 dirIn, const float3 dirOut);
    DirectionSampleRecord sampleDirection<let Adjoint : bool>(const float3 rnd, const float3 dirIn);
};
extension PackedMaterialData : Interaction {};
extension Medium             : Interaction {};

float mis(const float a, const float b) {
    const float a2 = a * a;
    return a2 / (a2 + b * b);
}

float PathMisWeight(const uint viewPathLength, const uint lightVertices, const float dirSamplePdfW, const float lightSamplePdfW) {
    if (viewPathLength == 1 && lightVertices == 0)
        return 1;

    if (!gSampleDirectIllumination) {
        return lightVertices == 0 ? 1 : 0;
    }

    if (lightVertices == 0)
        return mis(dirSamplePdfW, lightSamplePdfW); // camera path
    else if (lightVertices == 1)
        return mis(lightSamplePdfW, dirSamplePdfW); // nee
    else
        return 0; // bdpt connections
}


static PathState sPathState;

void TraceConnectionRay(const float3 rayOrigin, const float3 rayDirection, const float rayDistance, const float neePdfW, const float dirPdfW, float3 contrib, const uint lightVertices) {
    if (gDeferShadowRays) {
        ShadowRay ray;
        ray.mContribution = sPathState.mThroughput * contrib;
        ray.mRayOrigin = rayOrigin;
        ray.mPathLength = sPathState.mPathLength;
        ray.mRayDirection = rayDirection;
        ray.mDistance = rayDistance;
        ray.mNeePdfW = neePdfW;
        ray.mDirPdfW = dirPdfW;
        ray.mPixelIndex = sPathState.mPixelIndex;
        ray.mLightVertices = 1;
        if (gSortShadowRays)
            gRenderParams.mShadowRays.Append(ray.mRayOrigin, ray);
        else {
            uint appendIndex;
            InterlockedAdd(gRenderParams.mShadowRays.mCounters[gRenderParams.mShadowRays.mConstants.mCellCount], 1, appendIndex);
            gRenderParams.mShadowRays.mData[appendIndex] = ray;
        }
    } else {
        float transDirPdf, transNeePdf;
        gScene.traceVisibilityRay(
            makeRay(rayOrigin, rayDirection, 0, rayDistance),
            sPathState.mRng, sPathState.mCurrentMedium, contrib, transDirPdf, transNeePdf);

        if (all(contrib <= 0) || transNeePdf <= 0)
            return;

        contrib /= transNeePdf;

        gRenderParams.AddRadiance(sPathState.getPixelCoord(), sPathState.mThroughput * contrib, PathMisWeight(sPathState.mPathLength, lightVertices, dirPdfW * transDirPdf, neePdfW * transNeePdf), sPathState.mPathLength + 1 + (lightVertices-1), lightVertices);
    }
}

void ConnectToLight(const Interaction bsdf, const ShadingData shadingData) {
    const IlluminationSampleRecord r = gScene.SampleIllumination(shadingData.mPosition, sPathState.mRng.nextFloat());
    if (all(r.mRadiance <= 0) || r.mPdf <= 0)
        return;

    const bool isMedium = gHasMedia && shadingData.isMedium();

    float AtoW = 1;
    if (r.isFinite)
        AtoW = pow2(r.mDistanceToLight) / r.mCosLight;
	float3 contrib = r.mRadiance / (r.mPdf*AtoW);

    float3 localDirIn, localDirOut;
    if (isMedium) {
        localDirIn  = -sPathState.getRayDirection();
        localDirOut = r.mDirectionToLight;
    } else {
        localDirIn  = shadingData.toLocal(-sPathState.getRayDirection());
        localDirOut = shadingData.toLocal(r.mDirectionToLight);
    }

    const ReflectanceEvalRecord f = bsdf.evaluateReflectance<false>(localDirIn, localDirOut);
    contrib *= f.mReflectance;
    if (!isMedium)
        contrib *= shadingData.shadingNormalCorrection<false>(localDirIn, localDirOut);

	// evaluate transmittance along shadow ray

    const float3 rayOrigin = isMedium ?
		shadingData.mPosition :
		rayOffset(shadingData.mPosition, shadingData.getGeometryNormal(), r.mDirectionToLight);

    TraceConnectionRay(rayOrigin, r.mDirectionToLight, r.mDistanceToLight * 0.99, r.mPdf * AtoW, f.mFwdPdfW, contrib, 1);
}

bool ProcessInteraction(const Interaction bsdf, const IntersectionResult isect, const float transDirPdf, const float transNeePdf) {
    const bool isMedium = gHasMedia && isect.mShadingData.isMedium();

    if (sPathState.mPathLength == 1)
        gRenderParams.StoreAuxillaryData(sPathState.getPixelCoord(), isect, bsdf.albedo());

    // surface lights are one-sided
    if (isMedium || dot(sPathState.getRayDirection(), isect.mShadingData.getGeometryNormal()) < 0) {
        gRenderParams.AddRadiance(
            sPathState.getPixelCoord(),
			sPathState.mThroughput * bsdf.emission(),
			PathMisWeight(sPathState.mPathLength, 0, transDirPdf*sPathState.mDirPdfW, transNeePdf*gScene.LightSamplePdfW(sPathState.getRayDirection(), isect)),
			sPathState.mPathLength + 1,
			0);
    }

    if (sPathState.mPathLength >= gPushConstants.mMaxDepth)
        return false;

    if (!bsdf.canEvaluate())
        return false;

    if (gSampleDirectIllumination)
		ConnectToLight(bsdf, isect.mShadingData);

    if (sPathState.mPathLength >= gPushConstants.mRRDepth) {
        const float continueProb = bsdf.continuationProb();
        if (continueProb < 1) {
            if (sPathState.mRng.nextFloat().x >= continueProb)
                return false;
            sPathState.mThroughput /= continueProb;
        }
    }

    const float3 localDirIn = isMedium ? -sPathState.getRayDirection() : isect.mShadingData.toLocal(-sPathState.getRayDirection());

    const DirectionSampleRecord s = bsdf.sampleDirection<false>(sPathState.mRng.nextFloat().xyz, localDirIn);
    sPathState.mThroughput *= s.mReflectance / s.mFwdPdfW;
    sPathState.mDirPdfW = s.mFwdPdfW;

    if (isMedium) {
        sPathState.mPackedRayDirection = packNormal(s.mDirection);
        sPathState.mRayOrigin = isect.mShadingData.mPosition;
    } else {
		sPathState.mThroughput  *= isect.mShadingData.shadingNormalCorrection<false>(localDirIn, s.mDirection);
		sPathState.mPackedRayDirection = packNormal(isect.mShadingData.toWorld(s.mDirection));
        sPathState.mRayOrigin = rayOffset(isect.mShadingData.mPosition, isect.mShadingData.getGeometryNormal(), sPathState.getRayDirection());
	}

    return !any(isnan(sPathState.mThroughput)) && any(sPathState.mThroughput > 0);
}

bool NextVertex() {
	IntersectionResult isect;
	float dirPdf, neePdf;
	const bool hit = gScene.traceScatteringRay(makeRay(sPathState.mRayOrigin, sPathState.getRayDirection()), sPathState.mRng, sPathState.mCurrentMedium, sPathState.mThroughput, dirPdf, neePdf, isect);
	sPathState.mThroughput /= dirPdf;
	sPathState.mPathLength++;
	isect.mShadingData.mTexcoordScreenSize = 0; // disable mipmaps for now

	if (!hit) {
        if (sPathState.mPathLength == 1)
            gRenderParams.StoreAuxillaryData(sPathState.getPixelCoord(), isect, 1);
		float envPdfW;
		const float3 emission = gScene.EvaluateEnvironment(sPathState.getRayDirection(), envPdfW);
        gRenderParams.AddRadiance(
            sPathState.getPixelCoord(),
			sPathState.mThroughput * emission,
			PathMisWeight(sPathState.mPathLength, 0, sPathState.mDirPdfW, envPdfW),
			sPathState.mPathLength+1,
			0);
		return false;
	}

	if (gHasMedia && isect.mShadingData.isMedium()) {
		return ProcessInteraction(Medium(isect.mShadingData.getMaterialAddress()), isect, dirPdf, neePdf);
	} else {
		gScene.ApplyNormalMap(isect.mShadingData);
		return ProcessInteraction(gScene.LoadMaterial(isect.mShadingData), isect, dirPdf, neePdf);
	}
}


void InitializeCameraPath(const uint2 index) {
    const uint viewIndex = getViewIndex(index, gPushConstants.mOutputExtent, gPushConstants.mViewCount);

	float2 uv;
	const float3 localDir = gRenderParams.mViews[viewIndex].toWorld(index + 0.5, /*out*/ uv);
	const TransformData t = gRenderParams.mViewTransforms[viewIndex];

    sPathState.mRayOrigin = float3(t.m[0][3], t.m[1][3], t.m[2][3]);
    sPathState.mCurrentMedium = gRenderParams.mViewMediumIndices[viewIndex];
    sPathState.mThroughput = 1;
    sPathState.mPackedRayDirection = packNormal(normalize(t.transformVector(localDir)));
    sPathState.mRng = RandomSampler(gPushConstants.mRandomSeed, index);
    sPathState.mPathLength = 0;
    sPathState.mPixelIndex = index.y * gPushConstants.mOutputExtent.x + index.x;
    sPathState.mDirPdfW = 0;

    gRenderParams.mOutput[index] = float4(0, 0, 0, 1);
}

void LoadPathState(const uint threadIndex) {
    sPathState = gRenderParams.mPathStates[threadIndex];
}

void StorePathState(const uint threadIndex) {
	gRenderParams.mPathStates[threadIndex] = sPathState;
}

[shader("compute")]
[numthreads(8, 4, 1)]
void Render(uint3 index: SV_DispatchThreadID) {
    if (any(index.xy >= gPushConstants.mOutputExtent)) return;

    InitializeCameraPath(index.xy);

    while (sPathState.mPathLength + 1 <= gPushConstants.mMaxDepth) {
        if (!NextVertex()) {
            sPathState.mThroughput = 0;
            break;
        }
        if (gMultiDispatch) {
            StorePathState(index.y * gPushConstants.mOutputExtent.x + index.x);
            break;
        }
	}
}

[shader("compute")]
[numthreads(8, 4, 1)]
void RenderIteration(uint3 index: SV_DispatchThreadID) {
    if (any(index.xy >= gPushConstants.mOutputExtent)) return;

    LoadPathState(index.y * gPushConstants.mOutputExtent.x + index.x);

    if (any(sPathState.mThroughput > 0) && sPathState.mPathLength + 1 <= gPushConstants.mMaxDepth) {
        if (!NextVertex())
            sPathState.mThroughput = 0;
        StorePathState(index.y * gPushConstants.mOutputExtent.x + index.x);
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void ProcessShadowRays(uint3 index: SV_DispatchThreadID) {
    const uint idx = index.y * gPushConstants.mOutputExtent.x + index.x;
    if (idx >= gRenderParams.mShadowRays.GetCurrentSize()) return;

    ShadowRay ray = gRenderParams.mShadowRays.mData[idx];

	float transDirPdf, transNeePdf;
    gScene.traceVisibilityRay(
        makeRay(ray.mRayOrigin, ray.mRayDirection, 0, ray.mDistance),
        sPathState.mRng, sPathState.mCurrentMedium, ray.mContribution, transDirPdf, transNeePdf);

    if (all(ray.mContribution <= 0) || transNeePdf <= 0)
        return;

    ray.mContribution /= transNeePdf;

    gRenderParams.AddRadianceInterlocked(ray.getPixelCoord(), ray.mContribution, PathMisWeight(ray.mPathLength, ray.mLightVertices, ray.mDirPdfW * transDirPdf, ray.mNeePdfW * transNeePdf), ray.mPathLength + 1 + (ray.mLightVertices-1), ray.mLightVertices);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void ProcessAtomicOutput(uint3 index: SV_DispatchThreadID) {
    if (any(index.xy >= gPushConstants.mOutputExtent)) return;
    uint4 c = gRenderParams.mOutputAtomic[index.y * gPushConstants.mOutputExtent.x + index.x];
    gRenderParams.mOutput[index.xy].rgb += c.rgb/float(gQuantizationFactor);
}