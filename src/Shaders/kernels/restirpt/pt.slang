#pragma once

struct PushConstants {
	float4 mSceneSphere;
    uint2 mOutputExtent;
	uint mViewCount;
    uint mMaxDepth;
	uint mMaxDiffuseBounces;
	uint mRandomSeed;
    uint mMaxNullCollisions;
    uint mLightCount;
    uint mEnvironmentMaterialAddress;
    float mEnvironmentSampleProbability;
    uint mDICandidateSamples;
    uint mGICandidateSamples;
    float mDIMaxM;
    float mGIMaxM;
    float mDIReuseRadius;
    float mGIReuseRadius;
    uint mReservoirHistoryValid;
    uint mPackedDebugPixelIndex;
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

    RWTexture2D<float4> mReservoirDataDI[3];
    RWTexture2D<float4> mPrevReservoirDataDI[3];

    RWTexture2D<float4> mReservoirDataGI[4];
    RWTexture2D<float4> mPrevReservoirDataGI[4];

    void StoreAuxillaryData(const SceneParameters scene, const ShadingData shadingData, const uint2 index, const uint instancePrimitiveIndex) {
        float2 prevPixelCoord = index + 0.5;

		const float depth = length(mViewInverseTransforms[0].transformPoint(shadingData.mPosition));

        float prevDepth = depth;
        if (BF_GET(instancePrimitiveIndex, 0, 16) != INVALID_INSTANCE) {
            const float3 prevCameraPos = mPrevViewInverseTransforms[0].transformPoint(scene.mInstanceMotionTransforms[BF_GET(instancePrimitiveIndex, 0, 16)].transformPoint(shadingData.mPosition));
			prevDepth = length(prevCameraPos);
			mViews[0].toRaster(prevCameraPos, prevPixelCoord);
		}

        mPrevUVs[index] = prevPixelCoord / gPushConstants.mOutputExtent;

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

    __init(const SceneParameters scene, const ShadingData shadingData, const uint instancePrimitiveIndex, const uint currentMedium, const uint packedLocalDirIn) {
        mShadingData = shadingData;
		mInstancePrimitiveIndex = instancePrimitiveIndex;
        mCurrentMedium = currentMedium;
		mPackedLocalDirIn = packedLocalDirIn;
        if (mShadingData.isSurface())
            scene.ApplyNormalMap(mShadingData);
    }
    __init(const SceneParameters scene, const ShadingData shadingData, const uint instancePrimitiveIndex, const uint currentMedium, const float3 dirIn) {
        mShadingData = shadingData;
		mInstancePrimitiveIndex = instancePrimitiveIndex;
        mCurrentMedium = currentMedium;
		if (mShadingData.isSurface()) {
            scene.ApplyNormalMap(mShadingData);
            mLocalDirIn = mShadingData.toLocal(dirIn);
        } else
            mLocalDirIn = dirIn;
	}
};

struct PathTracer {
    SceneParameters mScene;
    RenderParams mFramebuffer;

	PathVertex SampleVisibility(const uint2 index, inout RandomSampler rng, out float3 throughput) {
		const uint viewIndex = getViewIndex(index, gPushConstants.mOutputExtent, gPushConstants.mViewCount);

		float2 uv;
		const float3 localDir = mFramebuffer.mViews[viewIndex].toWorld(index + 0.5, /*out*/ uv);
		const TransformData t = mFramebuffer.mViewTransforms[viewIndex];
		const float3 direction = normalize(t.transformVector(localDir));

		uint medium = mFramebuffer.mViewMediumIndices[viewIndex];

		IntersectionResult isect;
		float transDirPdf, transNeePdf;
        throughput = 1;
        const bool hit = mScene.traceScatteringRay(makeRay(float3(t.m[0][3], t.m[1][3], t.m[2][3]), direction), rng, medium, throughput, transDirPdf, transNeePdf, isect);
		if (transDirPdf > 0)
			throughput /= transDirPdf;

		isect.mShadingData.mTexcoordScreenSize = 0; // disable mipmaps for now
		return PathVertex(mScene, isect.mShadingData, isect.mInstancePrimitiveIndex, medium, -direction);
	}
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

    float3 evaluateReflectanceFast<let Adjoint : bool>(const float3 localDirIn, const float3 localDirOut);
};
extension PackedMaterialData : Interaction {};
extension Medium : Interaction {};