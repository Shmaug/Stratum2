#ifndef gMaxNullCollisions
#define gMaxNullCollisions 128
#endif

#ifndef gDebugMode
#define gDebugMode TinyPTDebugMode::eNone
#endif
#ifndef gFeatureFlags
#define gFeatureFlags 0
#endif
#define CHECK_FEATURE(feature) (bool)(gFeatureFlags & BIT((uint)TinyPTFeatureFlagBits::e ## feature ))

#include "compat/scene.h"
ParameterBlock<SceneParameters> gScene;

#include "compat/tinypt.h"
[[vk::push_constant]] ConstantBuffer<TinyPTPushConstants> gPushConstants;


#include "common/rng.hlsli"
#include "common/intersection.hlsli"
#include "compat/environment.h"
#include "materials/disney.hlsli"
#define Material DisneyMaterial

struct RenderParams {
	StructuredBuffer<ViewData> mViews;
	StructuredBuffer<TransformData> mViewTransforms;
    StructuredBuffer<uint> mViewMediumInstances;

    StructuredBuffer<ViewData> mPrevViews;
    StructuredBuffer<TransformData> mPrevInverseViewTransforms;

	RWTexture2D<float4> mOutput;
	RWTexture2D<float4> mAlbedo;
	RWTexture2D<float2> mPrevUVs;
	RWTexture2D<float2> mVisibility;
	RWTexture2D<float4> mDepth;
};

ParameterBlock<RenderParams> gRenderParams;


float3 randomColor(const uint idx) {
    uint r = pcg(idx);
    uint g = pcg(r);
    uint b = pcg(g);
    return unorm32ToFloat(uint3(r, g, b));
}

float mis(const float a) {
    return a * a;
}
float mis(const float a, const float b) {
    const float a2 = mis(a);
    return a2 / (a2 + mis(b));
}


struct EmissionVertex {
	float3 mPosition;
	uint mPackedGeometryNormal;
	float3 mEmission;
	float mPdfA;

	property float3 geometryNormal { get { return unpackNormal(mPackedGeometryNormal); } }
};
extension SceneParameters {
	EmissionVertex sampleEmission(inout RandomSampler rng) {
		EmissionVertex v = { 0, 0, 0, 0 };
		// TODO
		return v;
	}
}

struct PathVertex<let Adjoint : bool> {
	ShadingData mShadingData;
	uint mMediumInstance;

	property float3 position { get { return mShadingData.mPosition; } }
	property float3 geometryNormal { get { return mShadingData.geometryNormal; } }
	property float3 shadingNormal { get { return mShadingData.shadingNormal; } }

	float3 toLocal(const float3 v) { return mShadingData.toLocal(v); }
	float3 toWorld(const float3 v) { return mShadingData.toWorld(v); }

    float3 evaluateBsdf(BSDF bsdf, const float3 localDirIn, const float3 localDirOut, out float fwdPdfW, out float revPdfW) {
        const MaterialEvalRecord r = bsdf.evaluate<Adjoint>(localDirIn, localDirOut);
        fwdPdfW = r.mFwdPdfW;
        revPdfW = r.mRevPdfW;
		return r.mReflectance;
	}

    float3 sampleDirection(BSDF bsdf, const float3 localDirIn, const float3 rnd, out float fwdDirPdfW, out float revDirPdfW) {
        const MaterialSampleRecord r = bsdf.sample<Adjoint>(rnd, localDirIn);
        fwdDirPdfW = r.mFwdPdfW;
        revDirPdfW = r.mRevPdfW;
		return r.mDirection;
	}
};

struct PathIntegrator<let Adjoint : bool> {
	RandomSampler mRNG;

	uint2 mOutputIndex;

	uint mBounces;
	uint mDiffuseBounces;

	float3 mBeta;

    PathVertex<Adjoint> mVertex;

	float3 mDirection;
	float mDirFwdPdfW;
	float mTransmitDistance, mTransmitDirPdf, mTransmitNeePdf;
	float mShapePdfA;

	__init(const uint2 index, const PathVertex<Adjoint> v, const float3 direction) {
		mRNG = RandomSampler(gPushConstants.mRandomSeed, index);

		mOutputIndex = index;

		mBounces = 0;
		mDiffuseBounces = 0;
		mBeta = 1;

        mVertex = v;
        mDirection = direction;
        mDirFwdPdfW = 0;

    	gRenderParams.mOutput[mOutputIndex] = float4(0, 0, 0, 1);
	}

	[mutating]
	bool russianRoullette() {
		const float rr_prob = 2*luminance(mBeta);
		if (rr_prob < 1) {
			if (mRNG.nextFloat() >= rr_prob)
				return true;
			else
				mBeta /= rr_prob;
		}
		return false;
    }

	[mutating]
	float3 sampleDirectLight(BSDF bsdf, const float3 localDirIn) {
		const EmissionVertex lightVertex = gScene.sampleEmission(mRNG);

		float3 toLight = lightVertex.mPosition - mVertex.position;
		const float dist = length(toLight);
		toLight /= dist;

		const float cosLight = -dot(lightVertex.geometryNormal, toLight);
		if (cosLight < 0) return 0;

		const float3 localDirOut = mVertex.toLocal(toLight);

        float fwdPdfW, revPdfW;
        const float3 Le = lightVertex.mEmission * cosLight * mVertex.evaluateBsdf(bsdf, localDirIn, localDirOut, /*out*/ fwdPdfW, /*out*/ revPdfW);

		// trace shadow ray
		RayDesc ray;
		ray.Origin = mVertex.mShadingData.isSurface ? rayOffset(mVertex.position, mVertex.geometryNormal, toLight) : mVertex.position;
		ray.TMin = 0;
		ray.Direction = toLight;
		ray.TMax = dist*0.999;
		float3 beta;
		float dirPdf, neePdf;
		gScene.traceVisibilityRay(mRNG, ray, mVertex.mMediumInstance, /*out*/ beta, /*out*/ dirPdf, /*out*/ neePdf);
		if (any(beta > 1))
			return beta * Le * mis(neePdf*lightVertex.mPdfA, dirPdf*pdfWtoA(fwdPdfW, cosLight/(dist*dist))) / lightVertex.mPdfA;
		return 0;
	}

	// samples direct lighting, samples mDirection and mBeta
    [mutating]
    bool integrateVertex<let StoreAlbedo : bool>(BSDF bsdf) {
        if (StoreAlbedo)
            gRenderParams.mAlbedo[mOutputIndex] = float4(bsdf.albedo(), 1);

		// evaluate BSDF emission
		float3 Le = bsdf.emission();
		if (any(Le > 0)) {
			const float cosTheta = -dot(mDirection, mVertex.geometryNormal);
			if (cosTheta > 0) {
                // MIS with direct lighting
                if (CHECK_FEATURE(NEE) && mDirFwdPdfW > 0)
					Le *= mis(mTransmitDirPdf*pdfWtoA(mDirFwdPdfW, cosTheta/pow2(mTransmitDistance)), mTransmitNeePdf*mShapePdfA);

				gRenderParams.mOutput[mOutputIndex].rgb += mBeta * Le;
			}
		}

		if (mBounces >= gPushConstants.mMaxBounces || mDiffuseBounces > gPushConstants.mMaxDiffuseBounces || !bsdf.canEvaluate()) {
			mBeta = 0;
			return false;
		}

		if (mBounces >= gPushConstants.mMinBounces) {
			if (russianRoullette()) {
				mBeta = 0;
				return false;
			}
		}

		const float3 localDirIn = mVertex.toLocal(-mDirection);

		mBounces++;
		if (!bsdf.isSingular()) {
            mDiffuseBounces++;
            if (!Adjoint && CHECK_FEATURE(NEE))
				gRenderParams.mOutput[mOutputIndex].rgb += mBeta * sampleDirectLight(bsdf, localDirIn);
		}

		// sample BSDF
        float revPdfW;
        const float3 localDirOut = mVertex.sampleDirection(bsdf, localDirIn, float3(mRNG.nextFloat(), mRNG.nextFloat(), mRNG.nextFloat()), /*out*/ mDirFwdPdfW, /*out*/ revPdfW);

		// eval BSDF
		float fwdPdfW;
		mBeta *= mVertex.evaluateBsdf(bsdf, localDirIn, localDirOut, /*out*/ fwdPdfW, /*out*/ revPdfW) / mDirFwdPdfW;

		mDirection = mVertex.toWorld(localDirOut);
		return true;
	}

	// Called by step() below. traces a ray in mDirection to find the next vertex. returns true if the next vertex is valid
	[mutating]
	bool step<let StoreAlbedo : bool>() {
		RayDesc ray;
		ray.Origin = rayOffset(mVertex.position, mVertex.geometryNormal, mDirection);
        ray.TMin = 0;
        ray.Direction = mDirection;
        ray.TMax = POS_INFINITY;

		IntersectionResult isect;
		float3 transmitBeta;
        const bool hit = gScene.traceScatteringRay(mRNG, ray, mVertex.mMediumInstance, transmitBeta, mTransmitDirPdf, mTransmitNeePdf, isect, mVertex.mShadingData);
        mVertex.mShadingData.mTexcoordScreenSize = 0;
		mTransmitDistance = isect.mDistance;
		mShapePdfA = isect.mShapePdfA;

		mBeta *= transmitBeta;

		if (gDebugMode == TinyPTDebugMode::eDepth)             { gRenderParams.mOutput[mOutputIndex] = float4(mTransmitDistance.xxx / 10, 1); return false; }
		if (gDebugMode == TinyPTDebugMode::eGeometryNormal)    { gRenderParams.mOutput[mOutputIndex] = float4(mVertex.mShadingData.geometryNormal * .5 + .5, 1); return false; }
		if (gDebugMode == TinyPTDebugMode::eShadingNormal)     { gRenderParams.mOutput[mOutputIndex] = float4(mVertex.mShadingData.shadingNormal * .5 + .5, 1); return false; }
		if (gDebugMode == TinyPTDebugMode::eTextureCoordinate) { gRenderParams.mOutput[mOutputIndex] = float4(mVertex.mShadingData.mTexcoord, 0, 1); return false; }
		if (gDebugMode == TinyPTDebugMode::eMaterialAddress)   { gRenderParams.mOutput[mOutputIndex] = float4(randomColor(mVertex.mShadingData.materialAddress), 1); return false; }

		if (StoreAlbedo) {
			VisibilityData v;
			v.mInstancePrimitiveIndex = isect.mInstancePrimitiveIndex;
			v.mPackedNormal = mVertex.mShadingData.mPackedShadingNormal;
			gRenderParams.mVisibility[mOutputIndex] = reinterpret<float2>(v);

            if (hit) {
                uint2 extent;
                gRenderParams.mOutput.GetDimensions(extent.x, extent.y);
                const uint viewIndex = getViewIndex(mOutputIndex, extent, gPushConstants.mViewCount);

				const float3 prevWorldPos = gScene.mInstanceMotionTransforms[isect.instanceIndex].transformPoint(mVertex.position);
				const float3 prevViewPos = gRenderParams.mPrevInverseViewTransforms[viewIndex].transformPoint(prevWorldPos);

				gRenderParams.mPrevViews[viewIndex].toRaster(prevViewPos, /*out*/ gRenderParams.mPrevUVs[mOutputIndex]);


				DepthData d;
				d.mDepth = mTransmitDistance;
				d.mPrevDepth = length(prevViewPos);

                const ViewData view = gRenderParams.mViews[viewIndex];
                const TransformData t = gRenderParams.mViewTransforms[viewIndex];
                const float3 ng = mVertex.geometryNormal;
                float2 uv;
                d.mDepthDerivative.x = rayPlane(ray.Origin - mVertex.position, normalize(t.transformVector(view.toWorld(mOutputIndex + 0.5 + uint2(1, 0), /*out*/ uv))), ng) - mTransmitDistance;
                d.mDepthDerivative.y = rayPlane(ray.Origin - mVertex.position, normalize(t.transformVector(view.toWorld(mOutputIndex + 0.5 + uint2(0, 1), /*out*/ uv))), ng) - mTransmitDistance;

				gRenderParams.mDepth[mOutputIndex] = reinterpret<float4>(d);
            }
		}

        if (hit) {
			if (mVertex.mShadingData.isSurface)
				return integrateVertex<StoreAlbedo>(Material(mVertex.mShadingData));
			else if (mVertex.mShadingData.isMedium)
				return integrateVertex<StoreAlbedo>(Medium(mVertex.mShadingData.materialAddress));
			else
				return false;
		}

		// hit background

		if (gPushConstants.mEnvironmentMaterialAddress != -1) {
			const Environment environment = Environment(gPushConstants.mEnvironmentMaterialAddress);
			float3 Le = environment.evaluate(mDirection);

			// MIS with direct lighting
			if (CHECK_FEATURE(NEE) && mDirFwdPdfW > 0) {
				const float emissionPdfW = environment.evaluatePdfW(mDirection); // TODO: light pick pdf
				Le *= mis(mTransmitDirPdf*mDirFwdPdfW, mTransmitNeePdf*emissionPdfW);
			}

			gRenderParams.mOutput[mOutputIndex].rgb += mBeta * Le;
			if (StoreAlbedo)
				gRenderParams.mAlbedo[mOutputIndex] = float4(1, 1, 1, 1);
		}

		mBeta = 0;
		return false;
	}
};

PathVertex<false> generateEyeRay(const uint viewIndex, const uint2 index, out float3 direction) {
	const ViewData view = gRenderParams.mViews[viewIndex];
	float2 uv;
	const float3 localDir = view.toWorld(index + 0.5, /*out*/ uv);
	const TransformData t = gRenderParams.mViewTransforms[viewIndex];
	direction = normalize(t.transformVector(localDir));

	PathVertex<false> v;
	v.mShadingData.mPosition = float3(t.m[0][3], t.m[1][3], t.m[2][3]);
	v.mShadingData.mFlagsMaterialAddress = 0;
	v.mShadingData.mPackedGeometryNormal = v.mShadingData.mPackedShadingNormal = packNormal(t.transformVector(float3(0,0,1)));
	v.mShadingData.mShapeArea = view.mProjection.mSensorArea;
	v.mMediumInstance = gRenderParams.mViewMediumInstances[viewIndex];
	return v;
}

/*
uint mapPixelCoord(const uint2 pixel_coord, const uint2 groupId, const uint groupThreadIndex) {
	const uint dispatch_w = (gOutputExtent.x + GROUPSIZE_X - 1) / GROUPSIZE_X;
	const uint group_index = groupId.y*dispatch_w + groupId.x;
	return group_index*GROUPSIZE_X*GROUPSIZE_Y + groupThreadIndex;
}
*/

[shader("compute")]
[numthreads(8, 8, 1)]
void sampleViewPaths(uint3 index: SV_DispatchThreadID, uint group_index: SV_GroupIndex) {
    uint2 extent;
    gRenderParams.mOutput.GetDimensions(extent.x, extent.y);
    if (any(index.xy >= extent)) return;

    float3 direction;
    PathVertex<false> v = generateEyeRay(getViewIndex(index.xy, extent, gPushConstants.mViewCount), index.xy, /*out*/ direction);

    PathIntegrator<false> path = PathIntegrator<false>(index.xy, v, direction);

    if (!path.step<true>()) return;
    for (uint i = 0; i < 8; i++)
		if (!path.step<false>())
			return;
}