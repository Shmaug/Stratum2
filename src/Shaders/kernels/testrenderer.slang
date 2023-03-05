#ifndef gSampleDirectIllumination
#define gSampleDirectIllumination false
#endif
#ifndef gUseVC
#define gUseVC false
#endif
#ifndef gMultiDispatch
#define gMultiDispatch false
#endif
#ifndef gTraceFromLight
#define gTraceFromLight false
#endif
#ifndef gDeferShadowRays
#define gDeferShadowRays false
#endif
#ifndef gDebugPaths
#define gDebugPaths false
#endif
#ifndef gDebugPathWeights
#define gDebugPathWeights false
#endif

struct PushConstants {
    uint2 mOutputExtent;
    uint mLightSubpathCount;
    uint mMaxDepth;

	uint mRandomSeed;
    uint mMaxNullCollisions;
	uint mViewCount;
    uint mVolumeInstanceCount;

    uint mLightCount;
    uint mEnvironmentMaterialAddress;
    float mEnvironmentSampleProbability;
    uint mDebugPathLengths;

	float4 mSceneSphere;
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


struct PackedLightVertex {
    float3 mLocalPosition;
    uint mInstancePrimitiveIndex;
    float3 mThroughput;
    uint mPackedLocalDirIn;
    uint mPathLength;
    float dEL;
    float mFwdPdfA;
    float mRevWtoA;

    uint getInstanceIndex()  { return BF_GET(mInstancePrimitiveIndex, 0, 16); }
    uint getPrimitiveIndex() { return BF_GET(mInstancePrimitiveIndex, 16, 16); }
    float3 getLocalDirIn() { return unpackNormal(mPackedLocalDirIn); }
    ShadingData getShadingData() {
        const uint instanceIndex = getInstanceIndex();
        ShadingData sd = gScene.makeShadingData(gScene.mInstances[instanceIndex], gScene.mInstanceTransforms[instanceIndex], mLocalPosition, getPrimitiveIndex());
        sd.mTexcoordScreenSize = 0;
        return sd;
	}
};

struct PathState {
    float3 mRayOrigin;
    uint mPackedCurrentMedium;
    float3 mThroughput;
	// dE or dL
    float dEL;
    RandomSampler mRng;
    float2 mPackedRayDirection;
	// fwd pdf of sampling the ray direction at last bounce
    float mFwdPdfW;
    uint mPacked;

	// |normal * direction| at last bounce
    property float mCosOut {
        get { return f16tof32(BF_GET(mPacked, 0, 16)); }
        set { BF_SET(mPacked, f32tof16(newValue), 0, 16); }
    }

    // specifies if a light path came from a finite emitter (e.g. not the environment)
    property bool isFiniteLight {
        get { return bool(BF_GET(mPacked, 31, 1)); }
        set { BF_SET(mPacked, (newValue ? 1 : 0), 31, 1); }
    }
	// specifies if a view path is specular
    property bool isSpecular {
        get { return bool(BF_GET(mPacked, 31, 1)); }
        set { BF_SET(mPacked, (newValue ? 1 : 0), 31, 1); }
    }

    property uint mCurrentMedium {
		get { return BF_GET(mPackedCurrentMedium, 0, 16); }
		set { BF_SET(mPackedCurrentMedium, newValue, 0, 16); }
	}
    property uint mPathLength {
		get { return BF_GET(mPackedCurrentMedium, 16, 16); }
		set { BF_SET(mPackedCurrentMedium, newValue, 16, 16); }
	}

    float3 getRayDirection() {
		return unpackNormal2(mPackedRayDirection);
	}
};

struct ShadowRay {
    float3 mContribution;
    float mWeight;
    float3 mRayOrigin;
    float mDistance;
    float3 mRayDirection;
    uint mPixelIndex;
    uint mViewVertices;
    uint mLightVertices;
    uint pad0;
    uint pad1;

    uint2 getPixelCoord() {
        return uint2(mPixelIndex % gPushConstants.mOutputExtent.x, mPixelIndex / gPushConstants.mOutputExtent.x);
    }
};

#define gQuantizationFactor 16384


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
    RWStructuredBuffer<PackedLightVertex> mLightVertices;
    RWStructuredBuffer<ShadowRay> mShadowRays;
    RWStructuredBuffer<uint> mCounters; // 0 -> light vertices, 1 -> shadow rays

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
		if (weight <= 0 || all(radiance <= 0))
			return;

        if (gDebugPaths) {
            if (BF_GET(gPushConstants.mDebugPathLengths, 0, 16) == viewVertices + lightVertices && BF_GET(gPushConstants.mDebugPathLengths, 16, 16) == lightVertices) {
				if (gDebugPathWeights)
					mOutput[index].rgb += radiance * weight;
				else
                	mOutput[index].rgb += radiance;
			}
            return;
        }

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
		if (weight <= 0 || all(radiance <= 0))
			return;

        if (gDebugPaths) {
            if (BF_GET(gPushConstants.mDebugPathLengths, 0, 16) == viewVertices + lightVertices && BF_GET(gPushConstants.mDebugPathLengths, 16, 16) == lightVertices) {
				if (gDebugPathWeights)
					AddRadianceInterlocked(index, radiance * weight);
				else
                	AddRadianceInterlocked(index, radiance);
			}
            return;
        }

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
extension Medium : Interaction {};


#define Nkk 1
//#define Nkk (gPushConstants.mOutputExtent.x * gPushConstants.mOutputExtent.y)
#define N0k 1
//#define N0k gPushConstants.mLightSubpathCount

float mis(const float a) {
    return a * a;
}
float mis(const float a, const float b) {
    const float a2 = mis(a);
    return a2 / (a2 + mis(b));
}


static PathState sPathState;
static uint2 sPixelIndex;
static float sIntersectionDistance;


void TraceConnectionRay(const float3 rayOrigin, const float3 rayDirection, const float rayDistance, float3 contrib, const float weight, const uint viewVertices, const uint lightVertices) {
    if (gDeferShadowRays) {
        ShadowRay ray;
        ray.mContribution = contrib;
        ray.mWeight = weight;
        ray.mRayOrigin = rayOrigin;
        ray.mViewVertices = viewVertices;
        ray.mRayDirection = rayDirection;
        ray.mDistance = rayDistance;
        ray.mPixelIndex = sPixelIndex.y * gPushConstants.mOutputExtent.x + sPixelIndex.x;
        ray.mLightVertices = lightVertices;
		uint appendIndex;
		InterlockedAdd(gRenderParams.mCounters[1], 1, appendIndex);
		gRenderParams.mShadowRays[appendIndex] = ray;
    } else {
        float transDirPdf, transNeePdf;
        gScene.traceVisibilityRay(
            makeRay(rayOrigin, rayDirection, 0, rayDistance),
            sPathState.mRng, sPathState.mCurrentMedium, contrib, transDirPdf, transNeePdf);

        if (all(contrib <= 0) || transNeePdf <= 0)
            return;

        contrib /= transNeePdf;

        gRenderParams.AddRadiance(sPixelIndex, contrib, weight, viewVertices, lightVertices);
    }
}

float AdjustIntersectionDistance(const float d) {
    return max(d * 0.99, d - 1e-2);
}


void StoreLightVertex(const IntersectionResult isect, const float3 localDirIn) {
    PackedLightVertex v;
	v.mLocalPosition = gScene.mInstanceInverseTransforms[isect.mInstanceIndex].transformPoint(isect.mShadingData.mPosition);
	v.mInstancePrimitiveIndex = isect.mInstancePrimitiveIndex;
	v.mThroughput = sPathState.mThroughput;
	v.mPackedLocalDirIn = packNormal(localDirIn);
    v.mPathLength = sPathState.mPathLength;
    v.dEL = sPathState.dEL;
    v.mFwdPdfA = pdfWtoA(sPathState.mFwdPdfW, abs((gHasMedia && isect.mShadingData.isMedium()) ? 1 : localDirIn.z), sIntersectionDistance);
    v.mRevWtoA = sPathState.mCosOut / pow2(sIntersectionDistance);
    uint idx;
    InterlockedAdd(gRenderParams.mCounters[0], 1, idx);
    gRenderParams.mLightVertices[idx] = v;
}

struct LightSubpathSampleRecord {
    float3 mThroughput;
    float mFwdPdfW;
    float3 mReflectance;
    float mCosLight;
    float3 mDirectionToLight;
    float mDistanceToLight;
    uint mSubpathLength;
    float dL;
};
LightSubpathSampleRecord SampleLightSubpath(const float3 referencePosition, const uint rnd) {
    if (gRenderParams.mCounters[0] == 0)
        return { 0 };

    const PackedLightVertex v = gRenderParams.mLightVertices[rnd % gRenderParams.mCounters[0]];

	if (sPathState.mPathLength + 1 + v.mPathLength > gPushConstants.mMaxDepth)
		return { 0 };

    ShadingData shadingData = v.getShadingData();

    LightSubpathSampleRecord r;
	r.mThroughput = v.mThroughput;
    r.mDirectionToLight = shadingData.mPosition - referencePosition;
    r.mDistanceToLight = length(r.mDirectionToLight);
    r.mDirectionToLight /= r.mDistanceToLight;
	r.mSubpathLength = v.mPathLength;

    // the LVC paper connects to lightVertexCount/lightPathCount light vertices
    // we only connect to one light vertex, therefore we weight its contribution by lightVertexCount/lightPathCount
    r.mThroughput *= gRenderParams.mCounters[0] / (float)gPushConstants.mLightSubpathCount;

	ReflectanceEvalRecord f;
	float contProb;

    if (gHasMedia && shadingData.isMedium()) {
		r.mCosLight = 1;
        const Medium lightBsdf = Medium(shadingData.getMaterialAddress());
		f = lightBsdf.evaluateReflectance<true>(v.getLocalDirIn(), -r.mDirectionToLight);
		contProb = lightBsdf.continuationProb();
	} else {
		gScene.ApplyNormalMap(shadingData);
		const float3 localDirOut = shadingData.toLocal(-r.mDirectionToLight);
		r.mCosLight = abs(localDirOut.z);
		const PackedMaterialData lightBsdf = gScene.LoadMaterial(shadingData);
		f = lightBsdf.evaluateReflectance<true>(v.getLocalDirIn(), localDirOut);
		f.mReflectance *= shadingData.shadingNormalCorrection<true>(v.getLocalDirIn(), localDirOut);
		contProb = lightBsdf.continuationProb();
	}

	r.mReflectance = f.mReflectance;
	r.mFwdPdfW = f.mFwdPdfW * contProb;
    const float revPdfW = f.mRevPdfW * contProb;

    r.dL = (1 + mis(revPdfW * v.mRevWtoA) * v.dEL) / mis(v.mFwdPdfA);

    return r;
}
void ConnectToLightSubpath(const Interaction bsdf, const ShadingData shadingData, const float3 localDirIn) {
    const LightSubpathSampleRecord r = SampleLightSubpath(shadingData.mPosition, sPathState.mRng.next().x);
    if (all(r.mThroughput <= 0) || all(r.mReflectance <= 0))
        return;

    const bool isMedium = gHasMedia && shadingData.isMedium();
    const float3 localDirOut = isMedium ? r.mDirectionToLight : shadingData.toLocal(r.mDirectionToLight);

    ReflectanceEvalRecord f = bsdf.evaluateReflectance<false>(localDirIn, localDirOut);
    if (!isMedium)
        f.mReflectance *= shadingData.shadingNormalCorrection<false>(localDirIn, localDirOut);
    const float contProb = bsdf.continuationProb();
    f.mFwdPdfW *= contProb;
    f.mRevPdfW *= contProb;

    const float invDist2 = 1 / pow2(r.mDistanceToLight);
    const float3 contrib = (sPathState.mThroughput * r.mThroughput) * (f.mReflectance * r.mReflectance) * invDist2;

	// Convert pdfs to area pdf
	const float cameraBsdfDirPdfA = f.mFwdPdfW * r.mCosLight * invDist2;
	const float lightBsdfDirPdfA  = r.mFwdPdfW * abs(isMedium ? 1 : localDirOut.z) * invDist2;

    const float dE = (1 + mis(pdfWtoA(f.mRevPdfW, sPathState.mCosOut, sIntersectionDistance)) * sPathState.dEL) / mis(pdfWtoA(sPathState.mFwdPdfW, isMedium ? 1 : abs(localDirIn.z), sIntersectionDistance));
    const float misWeight = 1 / (dE * lightBsdfDirPdfA + 1 + r.dL * cameraBsdfDirPdfA);

    const float3 rayOrigin = isMedium ?
		shadingData.mPosition :
		rayOffset(shadingData.mPosition, shadingData.getGeometryNormal(), r.mDirectionToLight);

    TraceConnectionRay(rayOrigin, r.mDirectionToLight, AdjustIntersectionDistance(r.mDistanceToLight), contrib, misWeight, sPathState.mPathLength+1, r.mSubpathLength+1);
}


void ConnectToCamera(const Interaction bsdf, const ShadingData shadingData, const float3 localDirIn) {
	float2 pixelCoord;
    const float3 cameraPos = gRenderParams.mViewInverseTransforms[0].transformPoint(shadingData.mPosition);
    if (!gRenderParams.mViews[0].toRaster(cameraPos, pixelCoord))
        return;

    const float cosAtCamera = abs(normalize(cameraPos).z);

	float3 toCamera = gRenderParams.mViewTransforms[0].transformPoint(0) - shadingData.mPosition;
    const float dist = length(toCamera);
    toCamera /= dist;

    const bool isMedium = gHasMedia && shadingData.isMedium();
    const float3 localDirOut = isMedium ? toCamera : shadingData.toLocal(toCamera);
	const float cosOut = isMedium ? 1 : abs(localDirOut.z);

    ReflectanceEvalRecord f = bsdf.evaluateReflectance<true>(localDirIn, localDirOut);
    if (!isMedium)
        f.mReflectance *= shadingData.shadingNormalCorrection<true>(localDirIn, localDirOut);
    const float contProb = bsdf.continuationProb();
    f.mFwdPdfW *= contProb;
    f.mRevPdfW *= contProb;

    f.mFwdPdfW = 0; // cant hit lens with bsdf sampling...

    // Compute pdf conversion factor from image plane area to surface area
    const float imagePointToCameraDist = gRenderParams.mViews[0].imagePlaneDist() / cosAtCamera;
    const float imageToSolidAngleFactor = pow2(imagePointToCameraDist) / cosAtCamera;
    const float imageToSurfaceFactor = imageToSolidAngleFactor * cosOut / pow2(dist);

    // We put the virtual image plane at such a distance from the camera origin
    // that the pixel area is one and thus the image plane sampling pdf is 1.
    // The area pdf of aHitpoint as sampled from the camera is then equal to
    // the conversion factor from image plane area density to surface area density
    const float cameraPdfA = imageToSurfaceFactor;

    const float surfaceToImageFactor = 1.f / imageToSurfaceFactor;

    // We divide the contribution by surfaceToImageFactor to convert the (already
    // divided) pdf from surface area to image plane area, w.r.t. which the
    // pixel integral is actually defined. We also divide by the number of samples
    // this technique makes, which is equal to the number of light sub-paths
    float3 contrib = (sPathState.mThroughput * f.mReflectance) / (gPushConstants.mLightSubpathCount * surfaceToImageFactor * cosOut);

    // see 'Recursive MIS Computation for Streaming BDPT on the GPU' by Dietger van Antwerpen
    // eq 16, where s=1
    const float dL_1 = (1 + mis(pdfWtoA(f.mRevPdfW, sPathState.mCosOut, sIntersectionDistance)) * sPathState.dEL) / mis(pdfWtoA(sPathState.mFwdPdfW, isMedium ? 1 : abs(localDirIn.z), sIntersectionDistance));

    const float misWeight = gUseVC ? 1 / (N0k + dL_1 * cameraPdfA) : 1;

    const float3 rayOrigin = isMedium ? shadingData.mPosition : rayOffset(shadingData.mPosition, shadingData.getGeometryNormal(), toCamera);

    if (gDeferShadowRays) {
        ShadowRay ray;
        ray.mContribution = contrib;
        ray.mWeight = misWeight;
        ray.mRayOrigin = rayOrigin;
        ray.mViewVertices = 1;
        ray.mRayDirection = toCamera;
        ray.mDistance = dist;
        ray.mPixelIndex = int(pixelCoord.y) * gPushConstants.mOutputExtent.x + int(pixelCoord.x);
        ray.mLightVertices = (int)sPathState.mPathLength + 1;
		uint appendIndex;
		InterlockedAdd(gRenderParams.mCounters[1], 1, appendIndex);
		gRenderParams.mShadowRays[appendIndex] = ray;
    } else {
        float transDirPdf, transNeePdf;
        gScene.traceVisibilityRay(
            makeRay(rayOrigin, toCamera, 0, dist),
            sPathState.mRng, sPathState.mCurrentMedium, contrib, transDirPdf, transNeePdf);

        if (all(contrib <= 0) || transNeePdf <= 0)
            return;

        contrib /= transNeePdf;

        gRenderParams.AddRadianceInterlocked(uint2(pixelCoord), contrib, misWeight, 1, (int)sPathState.mPathLength + 1);
    }
}


void ConnectToLight(const Interaction bsdf, const ShadingData shadingData, const float3 localDirIn) {
    const IlluminationSampleRecord r = gScene.SampleIllumination(sPathState.mRng.nextFloat(), shadingData.mPosition);
    if (all(r.mRadiance <= 0) || r.mPdf <= 0)
        return;

    const bool isMedium = gHasMedia && shadingData.isMedium();

    float AtoW = 1;
    if (r.isFinite)
        AtoW = pow2(r.mDistanceToLight) / r.mCosLight;
    const float directPdfW = r.mPdf * AtoW;

    const float3 localDirOut = isMedium ? r.mDirectionToLight : shadingData.toLocal(r.mDirectionToLight);

    ReflectanceEvalRecord f = bsdf.evaluateReflectance<false>(localDirIn, localDirOut);
    if (!isMedium)
        f.mReflectance *= shadingData.shadingNormalCorrection<false>(localDirIn, localDirOut);
	const float contProb = bsdf.continuationProb();
	f.mFwdPdfW *= contProb;
	f.mRevPdfW *= contProb;

    const float3 contrib = (sPathState.mThroughput * r.mRadiance * f.mReflectance) / directPdfW;

	// compute mis weight

    float misWeight;
    if (gUseVC) {
        const float emissionPdfA = pdfWtoA(cosHemispherePdfW(r.mCosLight), abs(localDirOut.z), r.mDistanceToLight);
        const float dE = (1 + mis(pdfWtoA(f.mRevPdfW, sPathState.mCosOut, sIntersectionDistance)) * sPathState.dEL) / mis(pdfWtoA(sPathState.mFwdPdfW, isMedium ? 1 : abs(localDirIn.z), sIntersectionDistance));
        const float dL = 1 / mis(r.mPdf);
        misWeight = 1 / (dE * mis(emissionPdfA) + 1 + dL * mis(f.mFwdPdfW / AtoW));
    } else
        misWeight = mis(r.mPdf, f.mFwdPdfW / AtoW);

	// evaluate transmittance along shadow ray

    const float3 rayOrigin = isMedium ?
		shadingData.mPosition :
		rayOffset(shadingData.mPosition, shadingData.getGeometryNormal(), r.mDirectionToLight);

    TraceConnectionRay(rayOrigin, r.mDirectionToLight, AdjustIntersectionDistance(r.mDistanceToLight), contrib, misWeight, sPathState.mPathLength+1, 1);
}


void AddEmission(const float3 emission, const float directPdfA, const float emissionDirPdfA, const float WtoA) {
	float w = 1;
	if (sPathState.mPathLength > 1 && !sPathState.isSpecular) {
        if (gUseVC) {
            // see 'Recursive MIS Computation for Streaming BDPT on the GPU' by Dietger van Antwerpen
            // eq 16, where s=k
            // sPathState.dEL contains dE_{k-1} which should be computed before calling this function
            const float dE_k = (1 + mis(emissionDirPdfA) * sPathState.dEL) / mis(sPathState.mFwdPdfW * WtoA);
            w = 1 / (Nkk + dE_k * mis(directPdfA));
        } else if (gSampleDirectIllumination && sPathState.mFwdPdfW > 0) {
            w = mis(sPathState.mFwdPdfW*WtoA, directPdfA);
		}
	}

    gRenderParams.AddRadiance(
        sPixelIndex,
		emission,
		w,
		sPathState.mPathLength + 1,
		0);
}


bool ProcessInteraction(const Interaction bsdf, const IntersectionResult isect, const float transDirPdf, const float transNeePdf) {
    const bool isMedium = gHasMedia && isect.mShadingData.isMedium();
    const float3 localDirIn = isMedium ? -sPathState.getRayDirection() : isect.mShadingData.toLocal(-sPathState.getRayDirection());
    const float cosIn = isMedium ? 1 : abs(localDirIn.z);

    if (!gTraceFromLight) {
        if (sPathState.mPathLength == 1) {
            gRenderParams.StoreAuxillaryData(sPixelIndex, isect, bsdf.albedo());
            // dE_1 = N0k/cameraPdfA
            sPathState.dEL = N0k / mis(sPathState.mFwdPdfW * cosIn / pow2(sIntersectionDistance));
		}

		// surface lights are one-sided
        if (!isMedium && localDirIn.z > 0) {
            const float3 emission = sPathState.mThroughput * bsdf.emission();
            if (any(emission > 0)) {
                AddEmission(emission,
					isect.LightSamplePdfA(),
					pdfWtoA(cosHemispherePdfW(localDirIn.z), sPathState.mCosOut, sIntersectionDistance),
					cosIn / pow2(sIntersectionDistance));
			}
		}
    }

    if (sPathState.mPathLength >= gPushConstants.mMaxDepth)
        return false;

    if (!bsdf.canEvaluate())
        return false;

    if (!bsdf.isSingular()) {
        if (gTraceFromLight) {
            if (gUseVC && sPathState.mPathLength + 2 <= gPushConstants.mMaxDepth)
            	StoreLightVertex(isect, localDirIn);
			ConnectToCamera(bsdf, isect.mShadingData, localDirIn);
    	} else {
			if (gSampleDirectIllumination || gUseVC)
				ConnectToLight(bsdf, isect.mShadingData, localDirIn);
			if (gUseVC)
				ConnectToLightSubpath(bsdf, isect.mShadingData, localDirIn);
		}
    }

	const float continueProb = min(1, bsdf.continuationProb());
	if (sPathState.mRng.nextFloat().x > continueProb)
		return false;

    DirectionSampleRecord s = bsdf.sampleDirection<gTraceFromLight>(sPathState.mRng.nextFloat().xyz, localDirIn);
    s.mFwdPdfW *= continueProb;
    s.mRevPdfW *= continueProb;

    sPathState.mThroughput *= s.mReflectance / s.mFwdPdfW;

    // compute dE or dL for current vertex
    sPathState.dEL = (1 + mis(pdfWtoA(s.mRevPdfW, sPathState.mCosOut, sIntersectionDistance)) * sPathState.dEL) / mis(pdfWtoA(sPathState.mFwdPdfW, cosIn, sIntersectionDistance));

    if (s.isSingular()) {
        sPathState.mFwdPdfW = 0;
    } else {
        if (!gTraceFromLight)
            sPathState.isSpecular = false;
        sPathState.mFwdPdfW = s.mFwdPdfW;
	}

    if (isMedium) {
		sPathState.mCosOut = 1;
		sPathState.mPackedRayDirection = packNormal2(s.mDirection);
        sPathState.mRayOrigin = isect.mShadingData.mPosition;
    } else {
		sPathState.mCosOut = abs(s.mDirection.z);
        sPathState.mThroughput *= isect.mShadingData.shadingNormalCorrection<gTraceFromLight>(localDirIn, s.mDirection);
        const float3 worldDir = isect.mShadingData.toWorld(s.mDirection);
		sPathState.mPackedRayDirection = packNormal2(worldDir);
        sPathState.mRayOrigin = rayOffset(isect.mShadingData.mPosition, isect.mShadingData.getGeometryNormal(), worldDir);
	}

    return !any(isnan(sPathState.mThroughput)) && any(sPathState.mThroughput > 0);
}

bool NextVertex() {
	IntersectionResult isect;
	float transDirPdf, transNeePdf;
	const bool hit = gScene.traceScatteringRay(makeRay(sPathState.mRayOrigin, sPathState.getRayDirection()), sPathState.mRng, sPathState.mCurrentMedium, sPathState.mThroughput, transDirPdf, transNeePdf, isect);
	sPathState.mThroughput /= transDirPdf;
	sPathState.mPathLength++;
    sIntersectionDistance = isect.mDistance;
    isect.mShadingData.mTexcoordScreenSize = 0; // disable mipmaps for now

    if (!hit) {
        if (!gTraceFromLight) {
            if (sPathState.mPathLength == 1)
                gRenderParams.StoreAuxillaryData(sPixelIndex, isect, 1);

			float dirPdfW;
			const float3 emission = sPathState.mThroughput * gScene.EvaluateEnvironment(sPathState.getRayDirection(), dirPdfW);

            if (any(emission > 0)) {
                AddEmission(emission,
					concentricDiscPdfA() / pow2(gPushConstants.mSceneSphere.w),
					dirPdfW,
					1);
			}
        }
		return false;
	}

	if (gHasMedia && isect.mShadingData.isMedium()) {
		return ProcessInteraction(Medium(isect.mShadingData.getMaterialAddress()), isect, transDirPdf, transNeePdf);
	} else {
		gScene.ApplyNormalMap(isect.mShadingData);
		return ProcessInteraction(gScene.LoadMaterial(isect.mShadingData), isect, transDirPdf, transNeePdf);
	}
}


void InitializeCameraPath(const uint2 index) {
    const uint viewIndex = getViewIndex(index, gPushConstants.mOutputExtent, gPushConstants.mViewCount);
    sPixelIndex = index.xy;

	float2 uv;
	const float3 localDir = gRenderParams.mViews[viewIndex].toWorld(index + 0.5, /*out*/ uv);
	const TransformData t = gRenderParams.mViewTransforms[viewIndex];

    sPathState.mRayOrigin = float3(t.m[0][3], t.m[1][3], t.m[2][3]);
    sPathState.mCurrentMedium = gRenderParams.mViewMediumIndices[viewIndex];
    sPathState.mPathLength = 0;
    sPathState.mThroughput = 1;
    sPathState.mRng = RandomSampler(gPushConstants.mRandomSeed, index);
    sPathState.mPackedRayDirection = packNormal2(normalize(t.transformVector(localDir)));

    // Compute pdf conversion factor from area on image plane to solid angle on ray
    const float cosAtCamera = abs(localDir.z);
    const float imagePointToCameraDist = gRenderParams.mViews[viewIndex].imagePlaneDist() / cosAtCamera;
    const float imageToSolidAngleFactor = pow2(imagePointToCameraDist) / cosAtCamera;

    // We put the virtual image plane at such a distance from the camera origin
    // that the pixel area is one and thus the image plane sampling pdf is 1.
    // The solid angle ray pdf is then equal to the conversion factor from
    // image plane area density to ray solid angle density
    const float cameraPdfW = imageToSolidAngleFactor;

    sPathState.mFwdPdfW = cameraPdfW;
    sPathState.mCosOut = cosAtCamera;

    sPathState.isSpecular = true;
    sPathState.dEL = 1; // gets assigned in ProcessInteraction

    gRenderParams.mOutput[index] = float4(0, 0, 0, 1);
}
void InitializeLightPath(const uint2 index) {
    sPathState.mRng = RandomSampler(gPushConstants.mRandomSeed, index, 0xFFFF);
    const IlluminationSampleRecord r = gScene.SampleIllumination(sPathState.mRng.nextFloat());
    sPathState.isFiniteLight = r.isFinite;

	sPathState.mThroughput = r.mRadiance / r.mPdf;
	sPathState.dEL = Nkk / mis(r.mPdf);

    const float2 dirRng = sPathState.mRng.nextFloat().xy;
    if (sPathState.isFiniteLight) {
		const float3 dir = sampleCosHemisphere(dirRng.x, dirRng.y);
		const float3x3 onb = makeOrthonormal(r.getNormal());
        sPathState.mRayOrigin = rayOffset(r.mPosition, r.getNormal());
		sPathState.mPackedRayDirection = packNormal2(normalize(onb[0] * dir.x + onb[1] * dir.y + onb[2] * dir.z));
        sPathState.mFwdPdfW = cosHemispherePdfW(dir.z);
        sPathState.mCosOut = dir.z;
        sPathState.mThroughput *= sPathState.mCosOut / sPathState.mFwdPdfW;
    } else {
        // environment light
        const float2 xy = sampleConcentricDisc(dirRng.x, dirRng.y);
        const float3x3 onb = makeOrthonormal(-r.mDirectionToLight);
        sPathState.mRayOrigin = gPushConstants.mSceneSphere.xyz + gPushConstants.mSceneSphere.w * (onb[0] * xy.x + onb[1] * xy.y + r.mDirectionToLight);
        sPathState.mPackedRayDirection = packNormal2(-r.mDirectionToLight);
        sPathState.mFwdPdfW = r.mPdf;
	    sPathState.mCosOut = 1;
	}

    sPathState.mCurrentMedium = (gHasMedia && sPathState.isFiniteLight) ? gScene.GetMediumIndex(r.mPosition, gPushConstants.mVolumeInstanceCount) : INVALID_INSTANCE;
    sPathState.mPathLength = 0;
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
    if ((gTraceFromLight && index.y * gPushConstants.mOutputExtent.x + index.x >= gPushConstants.mLightSubpathCount) ||
        (!gTraceFromLight && any(index.xy >= gPushConstants.mOutputExtent)))
        return;

    if (gTraceFromLight)
		InitializeLightPath(index.xy);
    else
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
    if ((gTraceFromLight && index.y * gPushConstants.mOutputExtent.x + index.x > gPushConstants.mLightSubpathCount) ||
        (!gTraceFromLight && any(index.xy >= gPushConstants.mOutputExtent)))
		return;

    LoadPathState(index.y * gPushConstants.mOutputExtent.x + index.x);

    if (any(sPathState.mThroughput > 0) && sPathState.mPathLength + 1 <= gPushConstants.mMaxDepth) {
        if (!NextVertex())
            sPathState.mThroughput = 0;
        StorePathState(index.y * gPushConstants.mOutputExtent.x + index.x);
    }
}


[shader("compute")]
[numthreads(8, 4, 1)]
void ProcessShadowRays(uint3 index: SV_DispatchThreadID) {
    const uint idx = index.y * gPushConstants.mOutputExtent.x + index.x;
    if (idx >= gRenderParams.mCounters[1]) return;

    ShadowRay ray = gRenderParams.mShadowRays[idx];

	float transDirPdf, transNeePdf;
    gScene.traceVisibilityRay(
        makeRay(ray.mRayOrigin, ray.mRayDirection, 0, ray.mDistance),
        sPathState.mRng, sPathState.mCurrentMedium, ray.mContribution, transDirPdf, transNeePdf);

    if (all(ray.mContribution <= 0) || transNeePdf <= 0)
        return;

    ray.mContribution /= transNeePdf;

    gRenderParams.AddRadianceInterlocked(ray.getPixelCoord(), ray.mContribution, ray.mWeight, ray.mViewVertices, ray.mLightVertices);
}


#ifndef gClearImage
#define gClearImage false
#endif

[shader("compute")]
[numthreads(8, 8, 1)]
void ProcessAtomicOutput(uint3 index: SV_DispatchThreadID) {
    if (any(index.xy >= gPushConstants.mOutputExtent)) return;
    uint4 c = gRenderParams.mOutputAtomic[index.y * gPushConstants.mOutputExtent.x + index.x];
    if (gClearImage) {
        gRenderParams.mOutput[index.xy] = float4(c.rgb / float(gQuantizationFactor), 1);
    } else {
        gRenderParams.mOutput[index.xy].rgb += c.rgb / float(gQuantizationFactor);
    }
}