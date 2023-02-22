#ifndef gSampleDirectIllumination
#define gSampleDirectIllumination false
#endif
#ifndef gDebugPaths
#define gDebugPaths false
#endif
#ifndef gMultiDispatch
#define gMultiDispatch false
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
};
[[vk::push_constant]] ConstantBuffer<PushConstants> gPushConstants;

#define gMaxNullCollisions gPushConstants.mMaxNullCollisions
#define gLightCount gPushConstants.mLightCount
#define gEnvironmentMaterialAddress gPushConstants.mEnvironmentMaterialAddress
#define gEnvironmentSampleProbability gPushConstants.mEnvironmentSampleProbability
#define gSceneSphere gPushConstants.mSceneSphere

#include "materials/lambertian.hlsli"
#include "common/intersection.hlsli"
#include "common/lights.hlsli"


struct PathState {
    float3 mRayOrigin;
    uint mCurrentMedium;
    float3 mThroughput;
    uint mPackedRayDirection;
    RandomSampler mRng;
    uint mPathLength; // number of segments
    uint mPathIndex;
    uint pad;
    float mDirPdfW;

    float3 getRayDirection() {
		return unpackNormal(mPackedRayDirection);
	}

    uint2 getOutputIndex() {
        return uint2(mPathIndex % gPushConstants.mOutputExtent.x, mPathIndex / gPushConstants.mOutputExtent.x);
	}
};

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

float PathWeight(const uint viewPathLength, const uint lightVertices, const float dirSamplePdfW, const float lightSamplePdfW) {
    if (viewPathLength == 1 && lightVertices == 0)
        return 1;

    if (!gSampleDirectIllumination) {
        return lightVertices == 0 ? 1 : 0;
    }

    if (lightVertices == 0)
        return mis(dirSamplePdfW, lightSamplePdfW);
    else
        return mis(lightSamplePdfW, dirSamplePdfW);
}

void Connect(inout PathState state, const Interaction bsdf, const ShadingData shadingData) {
    const IlluminationSampleRecord r = gScene.SampleIllumination(shadingData.mPosition, state.mRng.nextFloat());
    if (all(r.mRadiance <= 0) || r.mPdf <= 0)
        return;

    const bool isMedium = gHasMedia && shadingData.isMedium();

    float AtoW = 1;
    if (r.isFinite)
        AtoW = pow2(r.mDistanceToLight) / r.mCosLight;
	float3 contrib = r.mRadiance / (r.mPdf*AtoW);

    float3 localDirIn, localDirOut;
    if (isMedium) {
        localDirIn  = -state.getRayDirection();
        localDirOut = r.mDirectionToLight;
    } else {
        localDirIn  = shadingData.toLocal(-state.getRayDirection());
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

    float dirPdf, neePdf;
    gScene.traceVisibilityRay(
        makeRay(rayOrigin, r.mDirectionToLight, 0, r.mDistanceToLight * 0.99),
        state.mRng, state.mCurrentMedium, contrib, dirPdf, neePdf);

	if (all(contrib <= 0) || neePdf <= 0)
        return;

    contrib /= neePdf;

    gRenderParams.AddRadiance(state.getOutputIndex(), state.mThroughput * contrib, PathWeight(state.mPathLength, 1, f.mFwdPdfW, r.mPdf * AtoW), state.mPathLength+1, 1);
}

bool ProcessInteraction(inout PathState state, const Interaction bsdf, const IntersectionResult isect) {
    const bool isMedium = gHasMedia && isect.mShadingData.isMedium();

    if (state.mPathLength == 1)
        gRenderParams.StoreAuxillaryData(state.getOutputIndex(), isect, bsdf.albedo());

    // surface lights are one-sided
    if (isMedium || dot(state.getRayDirection(), isect.mShadingData.getGeometryNormal()) < 0) {
        gRenderParams.AddRadiance(
            state.getOutputIndex(),
			state.mThroughput * bsdf.emission(),
			PathWeight(state.mPathLength, 0, state.mDirPdfW, gScene.LightSamplePdfW(state.getRayDirection(), isect)),
			state.mPathLength + 1,
			0);
    }

    if (state.mPathLength >= gPushConstants.mMaxDepth)
        return false;

    if (!bsdf.canEvaluate())
        return false;

    if (gSampleDirectIllumination)
		Connect(state, bsdf, isect.mShadingData);

    if (state.mPathLength >= gPushConstants.mRRDepth) {
        const float continueProb = bsdf.continuationProb();
        if (continueProb < 1) {
            if (state.mRng.nextFloat().x >= continueProb)
                return false;
            state.mThroughput /= continueProb;
        }
    }

    const float3 localDirIn = isMedium ? -state.getRayDirection() : isect.mShadingData.toLocal(-state.getRayDirection());

    const DirectionSampleRecord s = bsdf.sampleDirection<false>(state.mRng.nextFloat().xyz, localDirIn);

    state.mThroughput *= s.mReflectance / s.mFwdPdfW;
    state.mDirPdfW = s.mFwdPdfW;

    if (isMedium) {
        state.mPackedRayDirection = packNormal(s.mDirection);
        state.mRayOrigin = isect.mShadingData.mPosition;
    } else {
		state.mThroughput  *= isect.mShadingData.shadingNormalCorrection<false>(localDirIn, s.mDirection);
		state.mPackedRayDirection = packNormal(isect.mShadingData.toWorld(s.mDirection));
        state.mRayOrigin = rayOffset(isect.mShadingData.mPosition, isect.mShadingData.getGeometryNormal(), state.getRayDirection());
	}

    return !any(isnan(state.mThroughput)) && any(state.mThroughput > 0);
}

bool NextVertex(inout PathState state) {
	IntersectionResult isect;
	float dirPdf, neePdf;
	const bool hit = gScene.traceScatteringRay(makeRay(state.mRayOrigin, state.getRayDirection()), state.mRng, state.mCurrentMedium, state.mThroughput, dirPdf, neePdf, isect);
	state.mThroughput /= dirPdf;
	state.mPathLength++;
	isect.mShadingData.mTexcoordScreenSize = 0; // disable mipmaps for now

	if (!hit) {
        if (state.mPathLength == 1)
            gRenderParams.StoreAuxillaryData(state.getOutputIndex(), isect, 1);
		float envPdfW;
		const float3 emission = gScene.EvaluateEnvironment(state.getRayDirection(), envPdfW);
		gRenderParams.AddRadiance(
			state.getOutputIndex(),
			state.mThroughput * emission,
			PathWeight(state.mPathLength, 0, state.mDirPdfW, envPdfW),
			state.mPathLength+1,
			0);
		return false;
	}

	if (gHasMedia && isect.mShadingData.isMedium()) {
		return ProcessInteraction(state, Medium(isect.mShadingData.getMaterialAddress()), isect);
	} else {
		gScene.ApplyNormalMap(isect.mShadingData);
		return ProcessInteraction(state, gScene.LoadMaterial(isect.mShadingData), isect);
	}
}

PathState InitializeCameraPath(const uint2 index) {
    const uint viewIndex = getViewIndex(index, gPushConstants.mOutputExtent, gPushConstants.mViewCount);

    gRenderParams.mOutput[index] = float4(0, 0, 0, 1);

	float2 uv;
	const float3 localDir = gRenderParams.mViews[viewIndex].toWorld(index + 0.5, /*out*/ uv);
	const TransformData t = gRenderParams.mViewTransforms[viewIndex];

    PathState state;
    state.mRayOrigin = float3(t.m[0][3], t.m[1][3], t.m[2][3]);
    state.mPackedRayDirection = packNormal(normalize(t.transformVector(localDir)));
    state.mCurrentMedium = gRenderParams.mViewMediumIndices[viewIndex];
	state.mPathLength = 0;
    state.mRng = RandomSampler(gPushConstants.mRandomSeed, index);
    state.mThroughput = 1;
    state.mPathIndex = index.y * gPushConstants.mOutputExtent.x + index.x;
    return state;
}


[shader("compute")]
[numthreads(8, 8, 1)]
void Render(uint3 index: SV_DispatchThreadID, uint group_index: SV_GroupIndex) {
    if (any(index.xy >= gPushConstants.mOutputExtent)) return;

    PathState state = InitializeCameraPath(index.xy);

    while (state.mPathLength + 1 <= gPushConstants.mMaxDepth) {
        if (!NextVertex(state)) {
            state.mThroughput = 0;
            break;
        }
		if (gMultiDispatch)
			break;
	}
	if (gMultiDispatch)
		gRenderParams.mPathStates[index.y * gPushConstants.mOutputExtent.x + index.x] = state;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void RenderIteration(uint3 index: SV_DispatchThreadID, uint group_index: SV_GroupIndex) {
    if (any(index.xy >= gPushConstants.mOutputExtent)) return;

    PathState state = gRenderParams.mPathStates[index.y * gPushConstants.mOutputExtent.x + index.x];

    if (state.mPathLength + 1 <= gPushConstants.mMaxDepth) {
        if (!NextVertex(state))
            state.mThroughput = 0;
    }

    gRenderParams.mPathStates[index.y * gPushConstants.mOutputExtent.x + index.x] = state;
}