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
    uint mGICandidateSamples;
    float mGIMaxM;
    float mGIReuseRadius;
    uint mGIReuseSamples;
    uint mReservoirHistoryValid;
    uint mDebugPixelIndex;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> gPushConstants;

#ifdef gDebugPixel
#define IsDebugPixel(pixel) (((pixel).y * gPushConstants.mOutputExtent.x + (pixel).x) == gPushConstants.mDebugPixelIndex)
#define DebugPrintf(pixel, ...) if (IsDebugPixel(pixel)) { printf(__VA_ARGS__); }
#else
#define IsDebugPixel(pixel) (false)
#define DebugPrintf(pixel, ...)
#endif

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

	static uint2 sPixelIndex;

	// samples a camera ray
	PathVertex SampleVisibility(const uint2 index, inout RandomSampler rng, out Spectrum throughput) {
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
		const PathVertex vertex = PathVertex(mScene, isect.mShadingData, isect.mInstancePrimitiveIndex, medium, -direction);

		// store aux data (depth/normals/motion)
        {
			float2 prevPixelCoord = index + 0.5;

			const float depth = length(mFramebuffer.mViewInverseTransforms[0].transformPoint(vertex.mShadingData.mPosition));

			float prevDepth = depth;
            if (vertex.mInstanceIndex != INVALID_INSTANCE) {
                const float3 prevCameraPos = tmul(mFramebuffer.mPrevViewInverseTransforms[0], mScene.mInstanceMotionTransforms[vertex.mInstanceIndex]).transformPoint(vertex.mShadingData.mPosition);
				prevDepth = length(prevCameraPos);
				mFramebuffer.mViews[0].toRaster(prevCameraPos, prevPixelCoord);
			}

			mFramebuffer.mPrevUVs[index] = prevPixelCoord / gPushConstants.mOutputExtent;

			VisibilityData v;
			v.mInstancePrimitiveIndex = vertex.mInstancePrimitiveIndex;
			v.mPackedNormal = vertex.mShadingData.mPackedShadingNormal;
			mFramebuffer.mVisibility[index] = reinterpret<uint2>(v);

			DepthData d;
			d.mDepth = depth;
			d.mPrevDepth = prevDepth;
			d.mDepthDerivative = 1;
			mFramebuffer.mDepth[index] = reinterpret<float4>(d);
		}

		// store albedo
        {
            if (vertex.mShadingData.isEnvironment())
                mFramebuffer.mAlbedo[sPixelIndex] = 0;
			else if (vertex.mShadingData.isMedium())
				mFramebuffer.mAlbedo[sPixelIndex] = float4(Medium(mScene, vertex.mShadingData.getMaterialAddress()).albedo(), 1);
			else
				mFramebuffer.mAlbedo[sPixelIndex] = float4(mScene.LoadMaterial(vertex.mShadingData).albedo(), 1);
        }

        return vertex;
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
    DirectionSampleRecord sampleDirection    <let Adjoint : bool>(const float3 rnd  , const float3 dirIn);

    float3 evaluateReflectanceFast<let Adjoint : bool>(const float3 localDirIn, const float3 localDirOut);
};
extension PackedMaterialData : Interaction {};
extension Medium : Interaction {};

// path tracing functions in path.slang