
struct PushConstants {
	uint mRandomSeed;
    uint mMinDepth;
    uint mMaxDepth;
    uint mMaxNullCollisions;
    uint2 mOutputExtent;
    uint mViewCount;
    uint mEnvironmentMaterialAddress;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> gPushConstants;

#define gMaxNullCollisions gPushConstants.mMaxNullCollisions

#include "materials/lambertian.hlsli"
#include "common/intersection.hlsli"


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
};
ParameterBlock<RenderParams> gRenderParams;

void StoreAuxillaryData(const uint2 index, const IntersectionResult isect, const float3 albedo) {
    float2 prevPixelCoord = index + 0.5;

    float prevDepth = isect.mDistance;
    if (isect.mInstanceIndex != INVALID_INSTANCE) {
        const float3 prevCameraPos = gRenderParams.mPrevViewInverseTransforms[0].transformPoint(gScene.mInstanceMotionTransforms[isect.mInstanceIndex].transformPoint(isect.mShadingData.mPosition));
        prevDepth = length(prevCameraPos);
        gRenderParams.mViews[0].toRaster(prevCameraPos, prevPixelCoord);
    }

    gRenderParams.mPrevUVs[index] = prevPixelCoord / gPushConstants.mOutputExtent;
    gRenderParams.mAlbedo[index] = float4(albedo, 1);

    VisibilityData v;
    v.mInstancePrimitiveIndex = isect.mInstancePrimitiveIndex;
    v.mPackedNormal = isect.mShadingData.mPackedShadingNormal;
    gRenderParams.mVisibility[index] = reinterpret<uint2>(v);

    DepthData d;
    d.mDepth = isect.mDistance;
    d.mPrevDepth = prevDepth;
    d.mDepthDerivative = 1;
    gRenderParams.mDepth[index] = reinterpret<float4>(d);
}

struct PathState {
    float3 mRayOrigin;
    uint mCurrentMedium;
    float3 mRayDirection;
    uint mPathLength;
    RandomSampler mRng;
    float3 mThroughput = 1;
};

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
extension Medium : Interaction {};

void Connect(const uint2 index, inout PathState state, const Interaction bsdf, const ShadingData shadingData) {
	// TODO: NEE, LVC
}

bool Bounce(const uint2 index, inout PathState state, const Interaction bsdf, const IntersectionResult isect) {
    if (state.mPathLength == 1)
        StoreAuxillaryData(index, isect, bsdf.albedo());

    // surface lights are one-sided
    if (!gHasMedia || !isect.mShadingData.isSurface() || dot(state.mRayDirection, isect.mShadingData.getGeometryNormal()) < 0)
        gRenderParams.mOutput[index].rgb += state.mThroughput * bsdf.emission();

    if (!bsdf.canEvaluate())
        return false;

    Connect(index, state, bsdf, isect.mShadingData);

    if (state.mPathLength >= gPushConstants.mMinDepth) {
        const float continueProb = bsdf.continuationProb();
        if (continueProb < 1) {
            if (state.mRng.nextFloat() >= continueProb)
                return false;
            state.mThroughput /= continueProb;
        }
    }

    const float3 localDirIn = (!gHasMedia || isect.mShadingData.isSurface()) ? isect.mShadingData.toLocal(-state.mRayDirection) : -state.mRayDirection;

    const DirectionSampleRecord s = bsdf.sampleDirection<false>(state.mRng.nextFloat3(), localDirIn);

    state.mThroughput *= s.mReflectance;
    if (!s.isSingular())
        state.mThroughput /= s.mFwdPdfW;

    if (!gHasMedia || isect.mShadingData.isSurface()) {
		state.mThroughput  *= isect.mShadingData.shadingNormalCorrection<false>(localDirIn, s.mDirection);
		state.mRayDirection = isect.mShadingData.toWorld(s.mDirection);
        state.mRayOrigin = rayOffset(isect.mShadingData.mPosition, isect.mShadingData.getGeometryNormal(), state.mRayDirection);
    } else {
        state.mRayDirection = s.mDirection;
        state.mRayOrigin = isect.mShadingData.mPosition;
	}

    return !any(isnan(state.mThroughput)) && any(state.mThroughput > 0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void render(uint3 index: SV_DispatchThreadID, uint group_index: SV_GroupIndex) {
    if (any(index.xy >= gPushConstants.mOutputExtent)) return;
    const uint viewIndex = getViewIndex(index.xy, gPushConstants.mOutputExtent, gPushConstants.mViewCount);

	float2 uv;
	const float3 localDir = gRenderParams.mViews[viewIndex].toWorld(index.xy + 0.5, /*out*/ uv);
	const TransformData t = gRenderParams.mViewTransforms[viewIndex];

    PathState state;

    state.mRayOrigin = float3(t.m[0][3], t.m[1][3], t.m[2][3]);
    state.mRayDirection = normalize(t.transformVector(localDir));

    state.mCurrentMedium = gRenderParams.mViewMediumIndices[viewIndex];
	state.mPathLength = 0;

    state.mRng = RandomSampler(gPushConstants.mRandomSeed, index.xy);
    state.mThroughput = 1;

    gRenderParams.mOutput[index.xy] = float4(0, 0, 0, 1);

    while (state.mPathLength < gPushConstants.mMaxDepth) {
        IntersectionResult isect;
        float dirPdf, neePdf;
        const bool hit = gScene.traceScatteringRay(makeRay(state.mRayOrigin, state.mRayDirection), state.mRng, state.mCurrentMedium, state.mThroughput, dirPdf, neePdf, isect);
        isect.mShadingData.mTexcoordScreenSize = 0; // disable mipmaps for now
        state.mPathLength++;

        state.mThroughput /= dirPdf;

        if (!hit) {
            if (state.mPathLength == 1)
                StoreAuxillaryData(index.xy, isect, 1);
            float environmentPdfW;
            gRenderParams.mOutput[index.xy].rgb += state.mThroughput * gScene.EvaluateEnvironment(gPushConstants.mEnvironmentMaterialAddress, state.mRayDirection, environmentPdfW);
            break;
        }

        if (!gHasMedia || isect.mShadingData.isSurface()) {
            PackedMaterialData bsdf = gScene.LoadMaterial(isect.mShadingData);
            gScene.ApplyNormalMap(isect.mShadingData);
            if (!Bounce(index.xy, state, bsdf, isect))
				break;
        } else {
            Medium bsdf = Medium(isect.mShadingData.getMaterialAddress());
            if (!Bounce(index.xy, state, bsdf, isect))
				break;
        }
    }
}