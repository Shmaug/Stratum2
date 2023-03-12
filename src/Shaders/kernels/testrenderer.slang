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

    float mMaxM;
    uint mCandidateSamples;
    uint mPrevHashGridValid;
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
    float dVC;
    float dVCM;
    float pad;

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
	uint mInstancePrimitiveIndex;
    RandomSampler mRng;
    float2 mPackedRayDirection;
    uint2 mPacked;


    property float dVCM {
        get { return f16tof32(BF_GET(mPacked[0], 0, 16)); }
        set { BF_SET(mPacked[0], f32tof16(newValue), 0, 16); }
    }
    property float dVC {
        get { return f16tof32(BF_GET(mPacked[0], 16, 16)); }
        set { BF_SET(mPacked[0], f32tof16(newValue), 16, 16); }
    }

    // fwd pdf of sampling the ray direction at last bounce
    property float mDirPdfW {
        get { return f16tof32(BF_GET(mPacked[1], 0, 16)); }
        set { BF_SET(mPacked[1], f32tof16(newValue), 0, 16); }
    }

    // specifies if a light path came from a finite emitter (e.g. not the environment)
    property bool isFiniteLight {
        get { return bool(BF_GET(mPacked[1], 31, 1)); }
        set { BF_SET(mPacked[1], (newValue ? 1 : 0), 31, 1); }
    }
	// specifies if a view path is specular
    property bool isSpecular {
        get { return bool(BF_GET(mPacked[1], 31, 1)); }
        set { BF_SET(mPacked[1], (newValue ? 1 : 0), 31, 1); }
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
    uint getInstanceIndex()  { return BF_GET(mInstancePrimitiveIndex, 0, 16); }
    uint getPrimitiveIndex() { return BF_GET(mInstancePrimitiveIndex, 16, 16); }
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

struct HashGridData {
    float3 mLocalPosition;
    uint mInstancePrimitiveIndex;

    float4 mDI_Sample;

    float mDI_CachedTargetPdf;
    float mDI_W;
    float mDI_M;
    uint mPackedLocalDirIn;

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

#define gQuantizationFactor 16384
#define gMisVmWeightFactor 0

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

    HashGrid<HashGridData> mHashGrid;
    HashGrid<HashGridData> mPrevHashGrid;

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

        #ifdef gDebugPaths
            if (BF_GET(gPushConstants.mDebugPathLengths, 0, 16) == viewVertices + lightVertices && BF_GET(gPushConstants.mDebugPathLengths, 16, 16) == lightVertices) {
				#ifdef gDebugPathWeights
					mOutput[index].rgb += weight;
				#else
                	mOutput[index].rgb += radiance;
				#endif
			}
		#else
			mOutput[index].rgb += radiance * weight;
        #endif
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

        #ifdef gDebugPaths
            if (BF_GET(gPushConstants.mDebugPathLengths, 0, 16) == viewVertices + lightVertices && BF_GET(gPushConstants.mDebugPathLengths, 16, 16) == lightVertices) {
				#ifdef gDebugPathWeights
					AddRadianceInterlocked(index, weight);
				#else
                	AddRadianceInterlocked(index, radiance);
				#endif
			}
		#else
			AddRadianceInterlocked(index, radiance * weight);
        #endif
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

    float3 evaluateReflectanceFast<let Adjoint : bool>(const float3 localDirIn, const float3 localDirOut);
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
static uint2 sPixelIndex;


void TraceConnectionRay(const float3 rayOrigin, const float3 rayDirection, const float rayDistance, float3 contrib, const float weight, const uint viewVertices, const uint lightVertices) {
    #ifdef gDeferShadowRays
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
    #else
        float transDirPdf, transNeePdf;
        gScene.traceVisibilityRay(
            makeRay(rayOrigin, rayDirection, 0, rayDistance),
            sPathState.mRng, sPathState.mCurrentMedium, contrib, transDirPdf, transNeePdf);

        if (all(contrib <= 0) || transNeePdf <= 0)
            return;

        contrib /= transNeePdf;

        gRenderParams.AddRadiance(sPixelIndex, contrib, weight, viewVertices, lightVertices);
    #endif
}

float AdjustIntersectionDistance(const float d) {
    return max(d * 0.99, d - 1e-2);
}


void StoreLightVertex(const ShadingData shadingData, const float3 localDirIn) {
    PackedLightVertex v;
	v.mLocalPosition = gScene.mInstanceInverseTransforms[sPathState.getInstanceIndex()].transformPoint(shadingData.mPosition);
	v.mInstancePrimitiveIndex = sPathState.mInstancePrimitiveIndex;
	v.mThroughput = sPathState.mThroughput;
	v.mPackedLocalDirIn = packNormal(localDirIn);
    v.mPathLength = sPathState.mPathLength;
    v.dVC  = sPathState.dVC;
    v.dVCM = sPathState.dVCM;
    uint idx;
    InterlockedAdd(gRenderParams.mCounters[0], 1, idx);
    gRenderParams.mLightVertices[idx] = v;
}

struct LightVertexSampleRecord {
    float3 mThroughput;
    float mFwdPdfW;
    float mRevPdfW;
    float3 mReflectance;
    float mCosLight;
    float3 mDirectionToLight;
    float mDistanceToLight;
    uint mSubpathLength;
    float dVC;
    float dVCM;
};
LightVertexSampleRecord SampleLightVertex(const float3 referencePosition, const uint rnd) {
    if (gRenderParams.mCounters[0] == 0)
        return { 0 };

    const PackedLightVertex v = gRenderParams.mLightVertices[rnd % gRenderParams.mCounters[0]];

	if (sPathState.mPathLength + 1 + v.mPathLength > gPushConstants.mMaxDepth)
		return { 0 };

    ShadingData shadingData = v.getShadingData();

    LightVertexSampleRecord r;
	r.mThroughput = v.mThroughput;
    r.mDirectionToLight = shadingData.mPosition - referencePosition;
    r.mDistanceToLight = length(r.mDirectionToLight);
    r.mDirectionToLight /= r.mDistanceToLight;
	r.mSubpathLength = v.mPathLength;

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
    r.mRevPdfW = f.mRevPdfW * contProb;

    r.dVC = v.dVC;
    r.dVCM = v.dVCM;

    return r;
}
void ConnectToLightSubpath(const Interaction bsdf, const ShadingData shadingData, const float3 localDirIn) {
    const LightVertexSampleRecord r = SampleLightVertex(shadingData.mPosition, sPathState.mRng.next().x);
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

    // the LVC paper connects to lightVertexCount/lightPathCount light vertices
    // we only connect to one light vertex, therefore we weight its contribution by lightVertexCount/lightPathCount
    const float w = (gRenderParams.mCounters[0] / (float)gPushConstants.mLightSubpathCount);

    const float invDist2 = 1 / pow2(r.mDistanceToLight);
    const float3 contrib = (sPathState.mThroughput * r.mThroughput) * ((f.mReflectance * r.mReflectance) * invDist2 * w);

	// Convert pdfs to area pdf
	const float cameraBsdfDirPdfA = f.mFwdPdfW * r.mCosLight * invDist2;
	const float lightBsdfDirPdfA  = r.mFwdPdfW * abs(isMedium ? 1 : localDirOut.z) * invDist2;

    // Partial light sub-path MIS weight [tech. rep. (40)]
    const float wLight = mis(cameraBsdfDirPdfA) * (gMisVmWeightFactor + r.dVCM + r.dVC * mis(r.mRevPdfW));

    // Partial eye sub-path MIS weight [tech. rep. (41)]
    const float wCamera = mis(lightBsdfDirPdfA) * (gMisVmWeightFactor + sPathState.dVCM + sPathState.dVC * mis(f.mRevPdfW));

	const float misWeight = 1 / (wLight + 1.f + wCamera);

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

	#ifdef gUseVC
    // Partial light sub-path weight [tech. rep. (46)]. Note the division by
    // mLightPathCount, which is the number of samples this technique uses.
    // This division also appears a few lines below in the framebuffer accumulation.
    const float wLight = mis(cameraPdfA / gPushConstants.mLightSubpathCount) * (
		gMisVmWeightFactor + sPathState.dVCM + sPathState.dVC * mis(f.mRevPdfW));

	// Partial eye sub-path weight is 0 [tech. rep. (47)]

	// Full path MIS weight [tech. rep. (37)]. No MIS for traditional light tracing.
	const float misWeight = 1.f / (wLight + 1.f);
	#else
    const float misWeight = 1;
	#endif

    const float3 rayOrigin = isMedium ? shadingData.mPosition : rayOffset(shadingData.mPosition, shadingData.getGeometryNormal(), toCamera);

    #ifdef gDeferShadowRays
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
    #else
        float transDirPdf, transNeePdf;
        gScene.traceVisibilityRay(
            makeRay(rayOrigin, toCamera, 0, dist),
            sPathState.mRng, sPathState.mCurrentMedium, contrib, transDirPdf, transNeePdf);

        if (all(contrib <= 0) || transNeePdf <= 0)
            return;

        contrib /= transNeePdf;

        gRenderParams.AddRadianceInterlocked(uint2(pixelCoord), contrib, misWeight, 1, (int)sPathState.mPathLength + 1);
    #endif
}


float ComputeMIS_DI(const IlluminationSampleRecord r, const ReflectanceEvalRecord f, const float directPdfW, const float cosOut, const float AtoW) {
    #ifdef gUseVC
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
        const float wCamera = mis(emissionPdfW * cosOut / (directPdfW * r.mCosLight)) * (
			gMisVmWeightFactor + sPathState.dVCM + sPathState.dVC * mis(f.mRevPdfW));

        return 1 / (wLight + 1 + wCamera);
	#elif defined(gSampleDirectIlluminationOnly)
		return 1;
	#else
		// mis with BSDF sampling
        return mis(r.mPdf, f.mFwdPdfW / AtoW);
	#endif
}

struct ReservoirSampleData {
    float4 mSample;
	float mSourcePdf;
    float mCachedMisWeight;
    float mCachedWtoA;
    float3 mCachedContribution; // f * Le
    RayDesc mVisibilityRay;

    property float mTargetPdf { get { return luminance(mCachedContribution) * mCachedWtoA; } }
    property float mRISWeight { get { return mTargetPdf / mSourcePdf; } }

    __init(const Interaction bsdf, const ShadingData shadingData, const float3 localDirIn, const float4 x) {
       mSample = x;
       mCachedContribution = 0;
       mCachedWtoA = 0;
       const IlluminationSampleRecord r = gScene.SampleIllumination(mSample, shadingData.mPosition);
	   mSourcePdf = r.mPdf;
       if (luminance(r.mRadiance) > 0 && r.mPdf > 0) {
			const bool isMedium = gHasMedia && shadingData.isMedium();

			const float3 localDirOut = isMedium ? r.mDirectionToLight : shadingData.toLocal(r.mDirectionToLight);

			ReflectanceEvalRecord f = bsdf.evaluateReflectance<false>(localDirIn, localDirOut);
			if (!isMedium)
						f.mReflectance *= shadingData.shadingNormalCorrection<false>(localDirIn, localDirOut);
			const float contProb = bsdf.continuationProb();
			f.mFwdPdfW *= contProb;
			f.mRevPdfW *= contProb;

			mCachedContribution = f.mReflectance * r.mRadiance;
			mCachedWtoA = r.isFinite ? r.mCosLight / pow2(r.mDistanceToLight) : 1;

			const float AtoW = r.isFinite ? pow2(r.mDistanceToLight) / r.mCosLight : 1;
			mCachedMisWeight = ComputeMIS_DI(r, f, mSourcePdf * AtoW, isMedium ? 1 : abs(localDirOut.z), AtoW);

			const float3 rayOrigin = isMedium ?
			shadingData.mPosition :
			rayOffset(shadingData.mPosition, shadingData.getGeometryNormal(), r.mDirectionToLight);
			mVisibilityRay = makeRay(rayOrigin, r.mDirectionToLight, 0, AdjustIntersectionDistance(r.mDistanceToLight));
       }
    }
};

void ReservoirReuseDI(const Interaction bsdf, const ShadingData shadingData, const float3 localDirIn, const HashGridData q, inout ReservoirSampleData currentSample, inout float integrationWeight) {
	if (q.mDI_M <= 0)
		return;

	// target pdfs

    // currentSample in q's domain
    ReservoirSampleData currentSample_q;
    {
		ShadingData sd_i = q.getShadingData();
		if (gHasMedia && sd_i.isMedium()) {
			currentSample_q = ReservoirSampleData(Medium(shadingData.getMaterialAddress()), sd_i, q.getLocalDirIn(), currentSample.mSample);
		} else {
			gScene.ApplyNormalMap(sd_i);
			currentSample_q = ReservoirSampleData(gScene.LoadMaterial(sd_i), sd_i, q.getLocalDirIn(), currentSample.mSample);
		}
    }

	// resampling MIS weights

    const float M = gPushConstants.mCandidateSamples;
    const float m0 = mis(currentSample.mTargetPdf * M, currentSample_q.mTargetPdf * q.mDI_M);

    // q's sample in the current domain
    const ReservoirSampleData newSample = ReservoirSampleData(bsdf, shadingData, localDirIn, q.mDI_Sample);
    const float m1 = mis(q.mDI_CachedTargetPdf * q.mDI_M, newSample.mTargetPdf * M);

	// RIS weights

	const float shiftJacobian = 1;

    const float w0 = m0 * currentSample.mTargetPdf * integrationWeight;
    const float w1 = m1 * newSample.mTargetPdf * q.mDI_W * shiftJacobian;

	// create new reservoir from current and new reservoirs

	const float totalWeight = w0 + w1;
    if (sPathState.mRng.nextFloat().x * totalWeight <= w1) {
        currentSample = newSample;
	}

    integrationWeight = totalWeight / currentSample.mTargetPdf;
}

ReservoirSampleData SampleIlluminationReSTIR(const Interaction bsdf, const ShadingData shadingData, const float3 localDirIn, out float integrationWeight) {
	ReservoirSampleData sample;
	float wsum = 0;

	// initial ris

    for (uint i = 0; i < gPushConstants.mCandidateSamples; i++) {
        const ReservoirSampleData xi = ReservoirSampleData(bsdf, shadingData, localDirIn, sPathState.mRng.nextFloat());
        if (xi.mTargetPdf <= 0)
            continue;

        wsum += xi.mRISWeight;
        if (sPathState.mRng.nextFloat().x <= xi.mRISWeight / wsum)
            sample = xi;
    }

    integrationWeight = wsum / (gPushConstants.mCandidateSamples * sample.mTargetPdf);

	// reuse

	#ifdef gReSTIR_DI_Reuse

	// TODO: trace visibility ray, zero integrationWeight if occluded

    float Mq = 0;
    if (gPushConstants.mPrevHashGridValid) {
		const float3 queryPos = shadingData.mPosition;
        const float cellSize = gRenderParams.mPrevHashGrid.GetCellSize(queryPos);
        const uint cellIndex = gRenderParams.mPrevHashGrid.FindCellIndex<false>(queryPos, cellSize);
        if (cellIndex != -1) {
			const uint2 range = gRenderParams.mPrevHashGrid.GetCellDataRange(cellIndex);
            if (range.y > range.x) {
                const HashGridData q = gRenderParams.mPrevHashGrid.mData[range.x + sPathState.mRng.next().x % (range.y - range.x)];
                Mq = q.mDI_M;
                ReservoirReuseDI(bsdf, shadingData, localDirIn, q, sample, integrationWeight);
            }
        }
	}

	// store for reuse
    if (integrationWeight > 0) {
        HashGridData d;
        d.mLocalPosition = gScene.mInstanceInverseTransforms[sPathState.getInstanceIndex()].transformPoint(shadingData.mPosition);
        d.mInstancePrimitiveIndex = sPathState.mInstancePrimitiveIndex;
        d.mDI_Sample = sample.mSample;
        d.mDI_CachedTargetPdf = sample.mTargetPdf;
        d.mDI_W = integrationWeight;
        d.mDI_M = min(gPushConstants.mMaxM * gPushConstants.mCandidateSamples, gPushConstants.mCandidateSamples + Mq);
        gRenderParams.mHashGrid.Append(shadingData.mPosition, d, gRenderParams.mHashGrid.GetCellSize(shadingData.mPosition));
	}
	#endif

    return sample;
}

#ifdef gReSTIR_DI

void ConnectToLight(const Interaction bsdf, const ShadingData shadingData, const float3 localDirIn) {
    float integrationWeight;
    const ReservoirSampleData r = SampleIlluminationReSTIR(bsdf, shadingData, localDirIn, integrationWeight);

    if (integrationWeight <= 0)
		return;

	// convert 1/pdfA to 1/pdfW
	integrationWeight *= r.mCachedWtoA;

    const float3 contrib = (sPathState.mThroughput * r.mCachedContribution) * integrationWeight;

    TraceConnectionRay(r.mVisibilityRay.Origin, r.mVisibilityRay.Direction, r.mVisibilityRay.TMax, contrib, r.mCachedMisWeight, sPathState.mPathLength + 1, 1);
}

#else

void ConnectToLight(const Interaction bsdf, const ShadingData shadingData, const float3 localDirIn) {
	const IlluminationSampleRecord r = gScene.SampleIllumination(sPathState.mRng.nextFloat(), shadingData.mPosition);
	if (all(r.mRadiance <= 0) || r.mPdf <= 0)
		return;

	float AtoW = 1;
	if (r.isFinite)
		AtoW = pow2(r.mDistanceToLight) / r.mCosLight;
    const float directPdfW = r.mPdf * AtoW;

    const bool isMedium = gHasMedia && shadingData.isMedium();
    const float3 localDirOut = isMedium ? r.mDirectionToLight : shadingData.toLocal(r.mDirectionToLight);

    ReflectanceEvalRecord f = bsdf.evaluateReflectance<false>(localDirIn, localDirOut);
    if (!isMedium)
        f.mReflectance *= shadingData.shadingNormalCorrection<false>(localDirIn, localDirOut);
	const float contProb = bsdf.continuationProb();
	f.mFwdPdfW *= contProb;
	f.mRevPdfW *= contProb;

    const float3 contrib = (sPathState.mThroughput * r.mRadiance * f.mReflectance) / directPdfW;

    // compute mis weight
    const float misWeight = ComputeMIS_DI(r, f, directPdfW, abs(isMedium ? 1 : localDirOut.z), AtoW);

	// evaluate transmittance along shadow ray

    const float3 rayOrigin = isMedium ?
		shadingData.mPosition :
		rayOffset(shadingData.mPosition, shadingData.getGeometryNormal(), r.mDirectionToLight);

    TraceConnectionRay(rayOrigin, r.mDirectionToLight, AdjustIntersectionDistance(r.mDistanceToLight), contrib, misWeight, sPathState.mPathLength + 1, 1);
}

#endif


void AddEmission(const float3 emission, const float directPdfA, const float emissionPdfW, const float WtoA) {
	float w = 1;
	if (sPathState.mPathLength > 1 && !sPathState.isSpecular) {
        #ifdef gUseVC
			// Partial eye sub-path MIS weight [tech. rep. (43)].
			// If the last hit was specular, then dVCM == 0.
			const float wCamera = mis(directPdfA) * sPathState.dVCM + mis(emissionPdfW) * sPathState.dVC;
            w = 1 / (1 + wCamera);
		#elif defined(gSampleDirectIllumination)
			if (sPathState.mDirPdfW > 0)
				w = mis(sPathState.mDirPdfW*WtoA, directPdfA);
		#endif
	}

    gRenderParams.AddRadiance(
        sPixelIndex,
		emission,
		w,
		sPathState.mPathLength + 1,
		0);
}


bool ProcessInteraction<let bFromLight : bool>(const Interaction bsdf, const IntersectionResult isect) {
    const bool isMedium = gHasMedia && isect.mShadingData.isMedium();
    const float3 localDirIn = isMedium ? -sPathState.getRayDirection() : isect.mShadingData.toLocal(-sPathState.getRayDirection());
    const float cosIn = isMedium ? 1 : abs(localDirIn.z);

    if (bFromLight) {
		#ifdef gUseVC
		// Update the MIS quantities before storing them at the vertex.
		// These updates follow the initialization in GenerateLightSample() or
		// SampleScattering(), and together implement equations [tech. rep. (31)-(33)]
		// or [tech. rep. (34)-(36)], respectively.

		// Infinite lights use MIS handled via solid angle integration,
		// so do not divide by the distance for such lights [tech. rep. Section 5.1]
		if (sPathState.mPathLength > 1 || sPathState.isFiniteLight == 1)
			sPathState.dVCM *= mis(pow2(isect.mDistance));
		sPathState.dVCM /= mis(cosIn);
        sPathState.dVC /= mis(cosIn);
		#endif
	} else {
		#ifdef gUseVC
		// Update the MIS quantities, following the initialization in
		// GenerateLightSample() or SampleScattering(). Implement equations
		// [tech. rep. (31)-(33)] or [tech. rep. (34)-(36)], respectively.
		sPathState.dVCM *= mis(pow2(isect.mDistance));
		sPathState.dVCM /= mis(cosIn);
		sPathState.dVC  /= mis(cosIn);
		#endif

		#if !defined(gMultiDispatch) || (defined(gMultiDispatch) && defined(gMultiDispatchFirst))
		if (sPathState.mPathLength == 1)
            gRenderParams.StoreAuxillaryData(sPixelIndex, isect, bsdf.albedo());
		#endif

		// surface lights are one-sided
        if (isMedium || localDirIn.z > 0) {
            const float3 emission = sPathState.mThroughput * bsdf.emission();
            if (any(emission > 0)) {
                const float pdfA = isect.LightSamplePdfA();
                AddEmission(emission,
					pdfA,
					pdfA * cosHemispherePdfW(localDirIn.z),
					cosIn / pow2(isect.mDistance));
			}
        }
    }

    if (sPathState.mPathLength >= gPushConstants.mMaxDepth)
        return false;

    if (!bsdf.canEvaluate())
        return false;

    if (!bsdf.isSingular()) {
		if (bFromLight) {
            #ifdef gUseVC
				if (sPathState.mPathLength + 2 <= gPushConstants.mMaxDepth)
					StoreLightVertex(isect, localDirIn);
			#endif
                ConnectToCamera(bsdf, isect.mShadingData, localDirIn);
        } else {
			#if defined(gSampleDirectIllumination) || defined(gUseVC)
				ConnectToLight(bsdf, isect.mShadingData, localDirIn);
			#endif
			#ifdef gUseVC
				ConnectToLightSubpath(bsdf, isect.mShadingData, localDirIn);
			#endif
        }
    }

	#ifdef gSampleDirectIlluminationOnly
    return false;
	#endif

	const float continueProb = min(1, bsdf.continuationProb());
	if (sPathState.mRng.nextFloat().x > continueProb)
		return false;

    DirectionSampleRecord s = bsdf.sampleDirection<bFromLight>(sPathState.mRng.nextFloat().xyz, localDirIn);
    s.mFwdPdfW *= continueProb;
    s.mRevPdfW *= continueProb;

    sPathState.mThroughput *= s.mReflectance / s.mFwdPdfW;

    const float cosOut = isMedium ? 1 : abs(s.mDirection.z);

    // Sub-path MIS quantities for the next vertex. Only partial - the
    // evaluation is completed when the actual hit point is known,
    // i.e. after tracing the ray, in the sub-path loop.
    if (s.isSingular()) {
        sPathState.mDirPdfW = 0;
		#ifdef gUseVC
			// Specular scattering case [tech. rep. (53)-(55)] (partially, as noted above)
			sPathState.dVCM = 0.f;
			sPathState.dVC *= mis(cosOut);
		#endif
    } else {
		sPathState.mDirPdfW = s.mFwdPdfW;
		#ifdef gUseVC
			#ifndef gTraceFromLight
				sPathState.isSpecular = false;
			#endif
			// Implements [tech. rep. (34)-(36)] (partially, as noted above)
			sPathState.dVC = mis(cosOut / s.mFwdPdfW) * (sPathState.dVC * mis(s.mRevPdfW) + sPathState.dVCM + gMisVmWeightFactor);
			sPathState.dVCM = mis(1.f / s.mFwdPdfW);
		#endif
	}

    if (isMedium) {
		sPathState.mPackedRayDirection = packNormal2(s.mDirection);
        sPathState.mRayOrigin = isect.mShadingData.mPosition;
    } else {
        sPathState.mThroughput *= isect.mShadingData.shadingNormalCorrection<bFromLight>(localDirIn, s.mDirection);
        const float3 worldDir = isect.mShadingData.toWorld(s.mDirection);
		sPathState.mPackedRayDirection = packNormal2(worldDir);
        sPathState.mRayOrigin = rayOffset(isect.mShadingData.mPosition, isect.mShadingData.getGeometryNormal(), worldDir);
	}

    return !any(isnan(sPathState.mThroughput)) && any(sPathState.mThroughput > 0);
}

bool NextVertex<let bFromLight : bool>() {
	IntersectionResult isect;
	float transDirPdf, transNeePdf;
	const bool hit = gScene.traceScatteringRay(makeRay(sPathState.mRayOrigin, sPathState.getRayDirection()), sPathState.mRng, sPathState.mCurrentMedium, sPathState.mThroughput, transDirPdf, transNeePdf, isect);
	sPathState.mThroughput /= transDirPdf;
    sPathState.mPathLength++;
    sPathState.mInstancePrimitiveIndex = isect.mInstancePrimitiveIndex;
    isect.mShadingData.mTexcoordScreenSize = 0; // disable mipmaps for now

    if (!hit) {
        if (!bFromLight) {
			#if !defined(gMultiDispatch) || (defined(gMultiDispatch) && defined(gMultiDispatchFirst))
            if (sPathState.mPathLength == 1)
                gRenderParams.StoreAuxillaryData(sPixelIndex, isect, 1);
			#endif

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
		return ProcessInteraction<bFromLight>(Medium(isect.mShadingData.getMaterialAddress()), isect);
	} else {
		gScene.ApplyNormalMap(isect.mShadingData);
		return ProcessInteraction<bFromLight>(gScene.LoadMaterial(isect.mShadingData), isect);
	}
}


void InitializeCameraPath(const uint2 index) {
    const uint viewIndex = getViewIndex(index, gPushConstants.mOutputExtent, gPushConstants.mViewCount);

    sPathState.mRng = RandomSampler(gPushConstants.mRandomSeed, index);

	#ifdef gPixelJitter
    const float2 offset = sPathState.mRng.nextFloat().xy;
	#else
    const float2 offset = 0.5;
	#endif

    float2 uv;
    const float3 localDir = gRenderParams.mViews[viewIndex].toWorld(index + offset, /*out*/ uv);
	const TransformData t = gRenderParams.mViewTransforms[viewIndex];

    sPathState.mRayOrigin = float3(t.m[0][3], t.m[1][3], t.m[2][3]);
    sPathState.mCurrentMedium = gRenderParams.mViewMediumIndices[viewIndex];
    sPathState.mPathLength = 0;
    sPathState.mThroughput = 1;
    sPathState.mPackedRayDirection = packNormal2(normalize(t.transformVector(localDir)));

	#ifdef gUseVC
		// Compute pdf conversion factor from area on image plane to solid angle on ray
		const float cosAtCamera = abs(localDir.z);
		const float imagePointToCameraDist = gRenderParams.mViews[viewIndex].imagePlaneDist() / cosAtCamera;
		const float imageToSolidAngleFactor = pow2(imagePointToCameraDist) / cosAtCamera;

		// We put the virtual image plane at such a distance from the camera origin
		// that the pixel area is one and thus the image plane sampling pdf is 1.
		// The solid angle ray pdf is then equal to the conversion factor from
		// image plane area density to ray solid angle density
		const float cameraPdfW = imageToSolidAngleFactor;

		sPathState.mDirPdfW = cameraPdfW;

		sPathState.isSpecular = true;

		// Eye sub-path MIS quantities. Implements [tech. rep. (31)-(33)] partially.
		// The evaluation is completed after tracing the camera ray in the eye sub-path loop.
		sPathState.dVCM = mis(gPushConstants.mLightSubpathCount / cameraPdfW);
		sPathState.dVC = 0;
	#endif

    gRenderParams.mOutput[index] = float4(0, 0, 0, 1);
}
void InitializeLightPath(const uint2 index) {
    sPathState.mRng = RandomSampler(gPushConstants.mRandomSeed, index, 0xFFFF);
    const IlluminationSampleRecord r = gScene.SampleIllumination(sPathState.mRng.nextFloat());
    sPathState.isFiniteLight = r.isFinite;

    sPathState.mThroughput = r.mRadiance / r.mPdf;

    float directPdfA, emissionPdfW, cosLight;

    const float2 dirRng = sPathState.mRng.nextFloat().xy;
    if (sPathState.isFiniteLight) {
		// area light
		const float3 dir = sampleCosHemisphere(dirRng.x, dirRng.y);
		const float3x3 onb = makeOrthonormal(r.getNormal());
        sPathState.mRayOrigin = rayOffset(r.mPosition, r.getNormal());
		sPathState.mPackedRayDirection = packNormal2(normalize(onb[0] * dir.x + onb[1] * dir.y + onb[2] * dir.z));
        sPathState.mDirPdfW = cosHemispherePdfW(dir.z);
        sPathState.mThroughput *= dir.z / sPathState.mDirPdfW;
        directPdfA = r.mPdf;
        emissionPdfW = r.mPdf * sPathState.mDirPdfW;
        cosLight = dir.z;
    } else {
        // environment light
        const float2 xy = sampleConcentricDisc(dirRng.x, dirRng.y);
        const float3x3 onb = makeOrthonormal(-r.mDirectionToLight);
        sPathState.mRayOrigin = gPushConstants.mSceneSphere.xyz + gPushConstants.mSceneSphere.w * (onb[0] * xy.x + onb[1] * xy.y + r.mDirectionToLight);
        sPathState.mPackedRayDirection = packNormal2(-r.mDirectionToLight);
        sPathState.mDirPdfW = r.mPdf;
        // TODO: directPdfA and emissionPdfW for environment lights
        directPdfA = 1;
        emissionPdfW = 1;
        cosLight = 1;
	}

	#ifdef gUseVC
    sPathState.dVCM = mis(directPdfA / emissionPdfW);
    sPathState.dVC  = r.isSingular ? 0 : mis(cosLight / emissionPdfW);
	#endif

    sPathState.mCurrentMedium = (gHasMedia && sPathState.isFiniteLight) ? gScene.GetMediumIndex(r.mPosition, gPushConstants.mVolumeInstanceCount) : INVALID_INSTANCE;
    sPathState.mPathLength = 0;
}


void LoadPathState(const uint threadIndex) {
    sPathState = gRenderParams.mPathStates[threadIndex];
}
void StorePathState(const uint threadIndex) {
	gRenderParams.mPathStates[threadIndex] = sPathState;
}

#ifdef gTraceFromLight
#define bFromLight true
#else
#define bFromLight false
#endif

[shader("compute")]
[numthreads(8, 4, 1)]
void Render(uint3 index: SV_DispatchThreadID) {
    sPixelIndex = index.xy;

	#ifdef gTraceFromLight

		if (index.y * gPushConstants.mOutputExtent.x + index.x >= gPushConstants.mLightSubpathCount)
			return;
		InitializeLightPath(index.xy);

	#else

		if (any(index.xy >= gPushConstants.mOutputExtent))
			return;
		InitializeCameraPath(index.xy);

	#endif


	#ifdef gMultiDispatch

		if (!NextVertex<bFromLight>())
			sPathState.mThroughput = 0;
		StorePathState(index.y * gPushConstants.mOutputExtent.x + index.x);

	#else

        while (sPathState.mPathLength + 1 <= gPushConstants.mMaxDepth) {
			if (!NextVertex<bFromLight>()) {
				sPathState.mThroughput = 0;
				break;
			}
		}

	#endif
}

[shader("compute")]
[numthreads(8, 4, 1)]
void RenderIteration(uint3 index: SV_DispatchThreadID) {
    sPixelIndex = index.xy;

	#ifdef gTraceFromLight

    if (index.y * gPushConstants.mOutputExtent.x + index.x >= gPushConstants.mLightSubpathCount)
		return;

	#else

    if (any(index.xy >= gPushConstants.mOutputExtent))
        return;

	#endif


    LoadPathState(index.y * gPushConstants.mOutputExtent.x + index.x);

    if (any(sPathState.mThroughput > 0) && sPathState.mPathLength + 1 <= gPushConstants.mMaxDepth) {
        if (!NextVertex<bFromLight>())
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