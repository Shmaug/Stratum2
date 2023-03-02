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
#ifndef gSortShadowRays
#define gSortShadowRays false
#endif
#ifndef gDebugPaths
#define gDebugPaths false
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
    float dVC;
    float dVCM;
    float pad0;
    float pad1;

    uint getInstanceIndex()  { return BF_GET(mInstancePrimitiveIndex, 0, 16); }
    uint getPrimitiveIndex() { return BF_GET(mInstancePrimitiveIndex, 16, 16); }
    float3 getLocalDirIn() { return unpackNormal(mPackedLocalDirIn); }
    ShadingData getShadingData() {
        return gScene.makeShadingData(gScene.mInstances[getInstanceIndex()], gScene.mInstanceTransforms[getInstanceIndex()], mLocalPosition, getPrimitiveIndex());
	}
};

struct PathState {
    float3 mRayOrigin;
    uint mPackedCurrentMedium;
    float3 mThroughput;
    uint mPackedRayDirection;
    RandomSampler mRng;
    uint2 mPackedMis;
    uint mPixelIndex;
    uint pad;

    property float mDirPdfW {
		get { return f16tof32(BF_GET(mPackedMis[0], 0, 16)); }
		set { BF_SET(mPackedMis[0], f32tof16(newValue), 0, 16); }
	}
    property float dVCM {
		get { return f16tof32(BF_GET(mPackedMis[0], 16, 16)); }
		set { BF_SET(mPackedMis[0], f32tof16(newValue), 16, 16); }
	}
    property float dVC {
		get { return f16tof32(BF_GET(mPackedMis[1], 0, 16)); }
        set { BF_SET(mPackedMis[1], f32tof16(newValue), 0, 16); }
    }
    property uint isFinite {
        get { return BF_GET(mPackedMis[1], 1, 31); }
        set { BF_SET(mPackedMis[1], newValue, 1, 31); }
    }
    property uint isSpecular {
        get { return BF_GET(mPackedMis[1], 1, 31); }
        set { BF_SET(mPackedMis[1], newValue, 1, 31); }
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
		return unpackNormal(mPackedRayDirection);
	}

    uint2 getPixelCoord() {
        return uint2(mPixelIndex % gPushConstants.mOutputExtent.x, mPixelIndex / gPushConstants.mOutputExtent.x);
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
    RWStructuredBuffer<uint> mLightVertexCounter;

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

struct LightSubpathSampleRecord {
    float3 mRadiance;
    float mDistanceToLight;
    float3 mDirectionToLight;
    float mCosLight;
    float dVCM;
    float dVC;
    float mFwdPdfW;
    float mRevPdfW;
    bool isMedium;
};
LightSubpathSampleRecord SampleLightSubpath(const float3 referencePosition, const uint rnd) {
    const PackedLightVertex v = gRenderParams.mLightVertices[rnd%gRenderParams.mLightVertexCounter[0]];

    ShadingData shadingData = v.getShadingData();

    LightSubpathSampleRecord r;
    r.mDirectionToLight = shadingData.mPosition - referencePosition;
    r.mDistanceToLight = length(r.mDirectionToLight);
    r.mDirectionToLight /= r.mDistanceToLight;
    r.dVCM = v.dVCM;
    r.dVC = v.dVC;
    r.isMedium = gHasMedia && shadingData.isMedium();

    r.mRadiance = v.mThroughput;

    // the LVC paper connects to lightVertexCount/lightPathCount light vertices
    // we only connect to one light vertex, therefore we weight its contribution by lightVertexCount/lightPathCount
    r.mRadiance *= gRenderParams.mLightVertexCounter[0] / (float)gPushConstants.mLightSubpathCount;

	ReflectanceEvalRecord f;
	float contProb;

    if (r.isMedium) {
		r.mCosLight = 1;
        Medium lightBsdf = Medium(shadingData.getMaterialAddress());
		f = lightBsdf.evaluateReflectance<true>(v.getLocalDirIn(), r.mDirectionToLight);
		contProb = lightBsdf.continuationProb();
	} else {
		gScene.ApplyNormalMap(shadingData);
		const float3 localDirOut = shadingData.toLocal(r.mDirectionToLight);
		r.mCosLight = localDirOut.z;
		PackedMaterialData lightBsdf = gScene.LoadMaterial(shadingData);
		f = lightBsdf.evaluateReflectance<true>(v.getLocalDirIn(), localDirOut);
		f.mReflectance *= shadingData.shadingNormalCorrection<true>(v.getLocalDirIn(), localDirOut);
		contProb = lightBsdf.continuationProb();
	}

	r.mRadiance *= f.mReflectance;
	r.mFwdPdfW = f.mFwdPdfW * contProb;
	r.mRevPdfW = f.mRevPdfW * contProb;

    return r;
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
};
extension PackedMaterialData : Interaction {};
extension Medium : Interaction {};

float mis(const float a) {
    return a * a;
}
float mis(const float a, const float b) {
    const float a2 = mis(a);
    return a2 / (a2 + mis(b));
}

static PathState sPathState;

void TraceConnectionRay(const float3 rayOrigin, const float3 rayDirection, const float rayDistance, float3 contrib, const float weight, const uint lightVertices) {
    if (gDeferShadowRays) {
        ShadowRay ray;
        ray.mContribution = sPathState.mThroughput * contrib;
        ray.mWeight = weight;
        ray.mRayOrigin = rayOrigin;
        ray.mViewVertices = (int)sPathState.mPathLength + 1 + (int(lightVertices) - 1);
        ray.mRayDirection = rayDirection;
        ray.mDistance = rayDistance;
        ray.mPixelIndex = sPathState.mPixelIndex;
        ray.mLightVertices = lightVertices;
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

        gRenderParams.AddRadiance(sPathState.getPixelCoord(), sPathState.mThroughput * contrib, weight, sPathState.mPathLength + 1 + (lightVertices-1), lightVertices);
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

    float3 contrib = r.mRadiance / directPdfW;

    const float3 localDirOut = isMedium ? r.mDirectionToLight : shadingData.toLocal(r.mDirectionToLight);

    ReflectanceEvalRecord f = bsdf.evaluateReflectance<false>(localDirIn, localDirOut);
	const float contProb = bsdf.continuationProb();
	f.mFwdPdfW *= contProb;
	f.mRevPdfW *= contProb;

    contrib *= f.mReflectance;
    if (!isMedium)
        contrib *= shadingData.shadingNormalCorrection<false>(localDirIn, localDirOut);


	// compute mis weight

    float misWeight;
    if (gUseVC) {
		const float emissionPdfW = r.mPdf * cosHemispherePdfW(r.mCosLight);

		// Partial light sub-path MIS weight [tech. rep. (44)].
		// Note that wLight is a ratio of area pdfs. But since both are on the
		// light source, their distance^2 and cosine terms cancel out.
		// Therefore we can write wLight as a ratio of solid angle pdfs,
		// both expressed w.r.t. the same shading point.
		const float wLight = mis(f.mFwdPdfW / directPdfW);

		// Partial eye sub-path MIS weight [tech. rep. (45)].
		//
		// In front of the sum in the parenthesis we have Mis(ratio), where
		//    ratio = emissionPdfA / directPdfA,
		// with emissionPdfA being the product of the pdfs for choosing the
		// point on the light source and sampling the outgoing direction.
		// What we are given by the light source instead are emissionPdfW
		// and directPdfW. Converting to area pdfs and plugging into ratio:
		//    emissionPdfA = emissionPdfW * cosToLight / dist^2
		//    directPdfA   = directPdfW * cosAtLight / dist^2
		//    ratio = (emissionPdfW * cosToLight / dist^2) / (directPdfW * cosAtLight / dist^2)
		//    ratio = (emissionPdfW * cosToLight) / (directPdfW * cosAtLight)
		//
		// Also note that both emissionPdfW and directPdfW should be
		// multiplied by lightPickProb, so it cancels out.
		const float wCamera = mis(emissionPdfW * abs(isMedium ? 1 : localDirOut.z) / (directPdfW * r.mCosLight)) * (sPathState.dVCM + sPathState.dVC * mis(f.mRevPdfW));

		// Full path MIS weight [tech. rep. (37)]
		misWeight = 1.f / (wLight + 1.f + wCamera);
    } else
        misWeight = mis(directPdfW, f.mFwdPdfW);

	// evaluate transmittance along shadow ray

    const float3 rayOrigin = isMedium ?
		shadingData.mPosition :
		rayOffset(shadingData.mPosition, shadingData.getGeometryNormal(), r.mDirectionToLight);

    TraceConnectionRay(rayOrigin, r.mDirectionToLight, r.mDistanceToLight * 0.99, contrib, misWeight, 1);
}

void ConnectToLightSubpath(const Interaction bsdf, const ShadingData shadingData, const float3 localDirIn) {
    const LightSubpathSampleRecord r = SampleLightSubpath(shadingData.mPosition, sPathState.mRng.next().x);
    if (all(r.mRadiance <= 0))
        return;

    const bool isMedium = gHasMedia && shadingData.isMedium();

    const float AtoW = pow2(r.mDistanceToLight) / (r.isMedium ? 1 : r.mCosLight);
    float3 contrib = r.mRadiance;

    const float3 localDirOut = isMedium ? r.mDirectionToLight : shadingData.toLocal(r.mDirectionToLight);

    ReflectanceEvalRecord f = bsdf.evaluateReflectance<false>(localDirIn, localDirOut);
    const float contProb = bsdf.continuationProb();
    f.mFwdPdfW *= contProb;
    f.mRevPdfW *= contProb;

    contrib *= f.mReflectance;
    if (!isMedium)
        contrib *= shadingData.shadingNormalCorrection<false>(localDirIn, localDirOut);

	// Convert pdfs to area pdf
	const float cameraBsdfDirPdfA = f.mFwdPdfW * pow2(r.mDistanceToLight) / abs(r.isMedium ? 1 : r.mCosLight);
	const float lightBsdfDirPdfA  = r.mFwdPdfW * pow2(r.mDistanceToLight) / abs(isMedium   ? 1 : localDirOut.z);

	// Partial light sub-path MIS weight [tech. rep. (40)]
	const float wLight = mis(cameraBsdfDirPdfA) * (r.dVCM + r.dVC * mis(r.mRevPdfW));

	// Partial eye sub-path MIS weight [tech. rep. (41)]
	const float wCamera = mis(lightBsdfDirPdfA) * (sPathState.dVCM + sPathState.dVC * mis(f.mRevPdfW));

	// Full path MIS weight [tech. rep. (37)]
	const float misWeight = 1.f / (wLight + 1.f + wCamera);

    const float3 rayOrigin = isMedium ?
		shadingData.mPosition :
		rayOffset(shadingData.mPosition, shadingData.getGeometryNormal(), r.mDirectionToLight);

    TraceConnectionRay(rayOrigin, r.mDirectionToLight, r.mDistanceToLight * 0.99, contrib, misWeight, 1);
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
    if (!isMedium) {
        f.mReflectance *= shadingData.shadingNormalCorrection<true>(localDirIn, localDirOut);
	}

    // Compute pdf conversion factor from image plane area to surface area
    const float imagePointToCameraDist = gRenderParams.mViews[0].imagePlaneDist() / cosAtCamera;
    const float imageToSolidAngleFactor = pow2(imagePointToCameraDist) / cosAtCamera;
    const float imageToSurfaceFactor = imageToSolidAngleFactor * cosOut / pow2(dist);

    // We put the virtual image plane at such a distance from the camera origin
    // that the pixel area is one and thus the image plane sampling pdf is 1.
    // The area pdf of aHitpoint as sampled from the camera is then equal to
    // the conversion factor from image plane area density to surface area density
    const float cameraPdfA = imageToSurfaceFactor;

    // Partial light sub-path weight [tech. rep. (46)]. Note the division by
    // mLightPathCount, which is the number of samples this technique uses.
    // This division also appears a few lines below in the framebuffer accumulation.
    const float wLight = mis(cameraPdfA / gPushConstants.mLightSubpathCount) * (sPathState.dVCM + sPathState.dVC * mis(f.mRevPdfW));

    // Partial eye sub-path weight is 0 [tech. rep. (47)]

    // Full path MIS weight [tech. rep. (37)]. No MIS for traditional light tracing.
    const float misWeight = (!gUseVC) ? 1.f : (1.f / (wLight + 1.f));

    const float surfaceToImageFactor = 1.f / imageToSurfaceFactor;

    // We divide the contribution by surfaceToImageFactor to convert the (already
    // divided) pdf from surface area to image plane area, w.r.t. which the
    // pixel integral is actually defined. We also divide by the number of samples
    // this technique makes, which is equal to the number of light sub-paths
    float3 contrib = sPathState.mThroughput * f.mReflectance / (gPushConstants.mLightSubpathCount * surfaceToImageFactor * cosOut);

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
            makeRay(rayOrigin, toCamera, 0, dist),
            sPathState.mRng, sPathState.mCurrentMedium, contrib, transDirPdf, transNeePdf);

        if (all(contrib <= 0) || transNeePdf <= 0)
            return;

        contrib /= transNeePdf;

        gRenderParams.AddRadianceInterlocked(uint2(pixelCoord), contrib, misWeight, 1, (int)sPathState.mPathLength + 1);
    }
}

void StoreLightVertex(const IntersectionResult isect, const float3 localDirIn) {
    PackedLightVertex v;
	v.mLocalPosition = gScene.mInstanceInverseTransforms[isect.mInstanceIndex].transformPoint(isect.mShadingData.mPosition);
	v.mInstancePrimitiveIndex = isect.mInstancePrimitiveIndex;
	v.mThroughput = sPathState.mThroughput;
	v.mPackedLocalDirIn = packNormal(localDirIn);
	v.dVC = sPathState.dVC;
    v.dVCM = sPathState.dVCM;
    uint idx;
    InterlockedAdd(gRenderParams.mLightVertexCounter[0], 1, idx);
    gRenderParams.mLightVertices[idx] = v;
}

bool ProcessInteraction(const Interaction bsdf, const IntersectionResult isect, const float transDirPdf, const float transNeePdf) {
    const bool isMedium = gHasMedia && isect.mShadingData.isMedium();

    if (!gTraceFromLight && sPathState.mPathLength == 1)
        gRenderParams.StoreAuxillaryData(sPathState.getPixelCoord(), isect, bsdf.albedo());

    const float3 localDirIn = isMedium ? -sPathState.getRayDirection() : isect.mShadingData.toLocal(-sPathState.getRayDirection());

    if (!gTraceFromLight || (sPathState.mPathLength > 1 || sPathState.isFinite == 1))
		sPathState.dVCM *= mis(pow2(isect.mDistance));
    if (!isMedium) {
		sPathState.dVCM /= mis(abs(localDirIn.z));
		sPathState.dVC  /= mis(abs(localDirIn.z));
    }

    if (!gTraceFromLight) {
		// surface lights are one-sided
        if (isMedium || dot(sPathState.getRayDirection(), isect.mShadingData.getGeometryNormal()) < 0) {
            float w = 1;
            if (sPathState.mPathLength > 1) {
				const float pdfA = transNeePdf * isect.LightSamplePdfA();
                if (gUseVC) {
					const float emissionPdfW = pdfA * cosHemispherePdfW(localDirIn.z);

					// Partial eye sub-path MIS weight [tech. rep. (43)].
					// If the last hit was specular, then dVCM == 0.
					const float wCamera = mis(pdfA) * sPathState.dVCM + mis(emissionPdfW) * sPathState.dVC;
                    w = 1 / (1 + wCamera);
                } else if (gSampleDirectIllumination) {
                    const float AtoW = pow2(isect.mDistance) / (isMedium ? 1 : abs(localDirIn.z));
                    w = mis(transDirPdf * sPathState.mDirPdfW, transNeePdf * pdfA * AtoW);
                }
            }

			gRenderParams.AddRadiance(
				sPathState.getPixelCoord(),
				sPathState.mThroughput * bsdf.emission(),
				w,
				sPathState.mPathLength + 1,
				0);
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
    sPathState.mDirPdfW = s.mFwdPdfW;

	const float cosOut = isMedium ? 1 : abs(s.mDirection.z);
    if (s.isSingular()) {
        // Specular scattering case [tech. rep. (53)-(55)] (partially, as noted above)
        sPathState.dVCM = 0.f;
        sPathState.dVC *= mis(cosOut);
    } else {
        // Implements [tech. rep. (34)-(36)] (partially, as noted above)
        sPathState.dVC = mis(cosOut / s.mFwdPdfW) * (sPathState.dVC * mis(s.mRevPdfW) + sPathState.dVCM);
        sPathState.dVCM = mis(1.f / s.mFwdPdfW);
        sPathState.isSpecular = 0;
	}

    if (isMedium) {
        sPathState.mPackedRayDirection = packNormal(s.mDirection);
        sPathState.mRayOrigin = isect.mShadingData.mPosition;
    } else {
		sPathState.mThroughput *= isect.mShadingData.shadingNormalCorrection<gTraceFromLight>(localDirIn, s.mDirection);
		sPathState.mPackedRayDirection = packNormal(isect.mShadingData.toWorld(s.mDirection));
        sPathState.mRayOrigin = rayOffset(isect.mShadingData.mPosition, isect.mShadingData.getGeometryNormal(), sPathState.getRayDirection());
	}

    return !any(isnan(sPathState.mThroughput)) && any(sPathState.mThroughput > 0);
}

bool NextVertex() {
	IntersectionResult isect;
	float transDirPdf, transNeePdf;
	const bool hit = gScene.traceScatteringRay(makeRay(sPathState.mRayOrigin, sPathState.getRayDirection()), sPathState.mRng, sPathState.mCurrentMedium, sPathState.mThroughput, transDirPdf, transNeePdf, isect);
	sPathState.mThroughput /= transDirPdf;
	sPathState.mPathLength++;
	isect.mShadingData.mTexcoordScreenSize = 0; // disable mipmaps for now

    if (!hit) {
        if (!gTraceFromLight) {
			if (sPathState.mPathLength == 1)
				gRenderParams.StoreAuxillaryData(sPathState.getPixelCoord(), isect, 1);

			float envPdfW;
			const float3 emission = gScene.EvaluateEnvironment(sPathState.getRayDirection(), envPdfW);

			if (any(emission > 0)) {
				float w = 1;
				if (sPathState.mPathLength > 1) {
					if (gUseVC) {
						const float pdfA = transNeePdf / (M_PI*pow2(gPushConstants.mSceneSphere.w));
						const float emissionPdfW = pdfA * envPdfW;

						// Partial eye sub-path MIS weight [tech. rep. (43)].
						// If the last hit was specular, then dVCM == 0.
						const float wCamera = mis(pdfA) * sPathState.dVCM + mis(emissionPdfW) * sPathState.dVC;
						w = 1 / (1 + wCamera);
					} else if (gSampleDirectIllumination)
						w = mis(sPathState.mDirPdfW*transDirPdf, envPdfW*transNeePdf);
				}

				gRenderParams.AddRadiance(
					sPathState.getPixelCoord(),
					sPathState.mThroughput * emission,
					w,
					sPathState.mPathLength + 1,
					0);
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

    // Compute pdf conversion factor from area on image plane to solid angle on ray
    const float cosAtCamera = localDir.z;
    const float imagePointToCameraDist = gRenderParams.mViews[viewIndex].imagePlaneDist() / cosAtCamera;
    const float imageToSolidAngleFactor = pow2(imagePointToCameraDist) / cosAtCamera;

    // We put the virtual image plane at such a distance from the camera origin
    // that the pixel area is one and thus the image plane sampling pdf is 1.
    // The solid angle ray pdf is then equal to the conversion factor from
    // image plane area density to ray solid angle density
    const float cameraPdfW = imageToSolidAngleFactor;

    // Eye sub-path MIS quantities. Implements [tech. rep. (31)-(33)] partially.
    // The evaluation is completed after tracing the camera ray in the eye sub-path loop.
    sPathState.dVCM = mis(gPushConstants.mLightSubpathCount / cameraPdfW);
    sPathState.dVC = 0;

    gRenderParams.mOutput[index] = float4(0, 0, 0, 1);
}
void InitializeLightPath(const uint2 index) {
    sPathState.mRng = RandomSampler(gPushConstants.mRandomSeed, index);
    const IlluminationSampleRecord r = gScene.SampleIllumination(sPathState.mRng.nextFloat());

	// TODO: environment light

    const float2 dirRng = sPathState.mRng.nextFloat().xy;
    const float3 dir = sampleCosHemisphere(dirRng.x, dirRng.y);

    sPathState.mRayOrigin = rayOffset(r.mPosition, r.getNormal());
    sPathState.mCurrentMedium = gHasMedia ? gScene.GetMediumIndex(r.mPosition, gPushConstants.mVolumeInstanceCount) : INVALID_INSTANCE;
    sPathState.mPackedRayDirection = packNormal(mul(dir, makeOrthonormal(r.getNormal())));
    sPathState.mPathLength = 0;
    sPathState.mPixelIndex = index.y * gPushConstants.mOutputExtent.x + index.x;
    sPathState.mDirPdfW = cosHemispherePdfW(dir.z);

    const float emissionPdfW = r.mPdf * sPathState.mDirPdfW;
    sPathState.mThroughput = r.mRadiance * dir.z / emissionPdfW;

    // Light sub-path MIS quantities. Implements [tech. rep. (31)-(33)] partially.
    // The evaluation is completed after tracing the emission ray in the light sub-path loop.
    // Delta lights are handled as well [tech. rep. (48)-(50)].

	sPathState.dVCM = mis(r.mPdf / emissionPdfW);

	if (!r.isSingular) {
		const float usedCosLight = r.isFinite ? dir.z : 1.f;
		sPathState.dVC = mis(usedCosLight / emissionPdfW);
	} else {
		sPathState.dVC = 0.f;
	}
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