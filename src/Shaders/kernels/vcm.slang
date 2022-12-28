#ifndef gHasMedia
#define gHasMedia (false)
#endif

#ifndef gAlphaTest
#define gAlphaTest (false)
#endif

#ifndef gNormalMaps
#define gNormalMaps (false)
#endif

#ifndef gMaxNullCollisions
#define gMaxNullCollisions (1024)
#endif

#ifndef gPerformanceCounters
#define gPerformanceCounters (false)
#endif

#ifndef gLightTraceQuantization
#define gLightTraceQuantization (16384)
#endif

// Do only unidirectional path tracing with next event estimation
#ifndef gPathTraceOnly
#define gPathTraceOnly (false)
#endif
// Do only light tracing
#ifndef gLightTraceOnly
#define gLightTraceOnly (false)
#endif
// Vertex merging (of some form) is used
#ifndef gUseVM
#define gUseVM (false)
#endif
// Vertex connection (BPT) is used
#ifndef gUseVC
#define gUseVC (false)
#endif
// Do PPM, same terminates camera after first merge
#ifndef gPpm
#define gPpm (false)
#endif

#include "common/path.hlsli"


struct AbstractCamera {
    uint mViewIndex;

    property ViewData view { get { return gRenderParams.mViews[mViewIndex]; } }
    property TransformData transform { get { return gRenderParams.mViewTransforms[mViewIndex]; } }
    property TransformData inverseTransform { get { return gRenderParams.mViewInverseTransforms[mViewIndex]; } }
    property TransformData prevInverseTransform { get { return gRenderParams.mPrevInverseViewTransforms[mViewIndex]; } }

}

struct LightRadianceRecord {
	float3 mRadiance;
    float mDirectPdfA;
	float mEmissionPdfW;
};
LightRadianceRecord GetBackgroundRadiance(const float3 aDirection) {
	if (!gHasEnvironment) return { 0 };

    LightRadianceRecord r;
    EnvironmentImage().evaluate(aDirection, cartesianToSphericalUv(aDirection), r.mRadiance, r.mDirectPdfA);
    if (gPushConstants.mLightCount > 0)
        r.mDirectPdfA *= gPushConstants.mEnvironmentSampleProbability;
    r.mEmissionPdfW = r.mDirectPdfA * concentricDiscPdfA() / pow2(gPushConstants.mSceneSphere.w);
	return r;
}
LightRadianceRecord GetSurfaceRadiance(const float3 aDirection, const IntersectionResult aIsect, const BSDF aBsdf) {
    if (gPushConstants.mLightCount == 0) return { 0 };

    const float cosTheta = -dot(aDirection, aIsect.mShadingData.geometryNormal);
    if (cosTheta < 0) return { 0 };

	LightRadianceRecord r;
    r.mRadiance = aBsdf.emission();
    r.mDirectPdfA = aBsdf.emissionPdf() * aIsect.mPrimitivePickPdf / (aIsect.mShadingData.mShapeArea * gPushConstants.mLightCount);
    if (gHasEnvironment)
        r.mDirectPdfA *= 1 - gPushConstants.mEnvironmentSampleProbability;
    r.mEmissionPdfW = r.mDirectPdfA * cosHemispherePdfW(cosTheta);
	return r;
}

float3 OffsetRayOrigin(const VcmVertex vertex, const float3 direction) {
	return rayOffset(vertex.mShadingData.mPosition, vertex.mShadingData.geometryNormal, direction);
}

void AddColor(const uint2 index, const float3 color) {
	gRenderParams.mOutput[index] += float4(color, 0);
}

void InterlockedAddColor(const int2 ipos, const uint2 extent, const float3 color) {
    if (all(color <= 0) || any(ipos < 0) || any(ipos >= extent))
        return;

    const uint3 icolor = uint3(color * gLightTraceQuantization);

    const uint address = 16 * (ipos.y * extent.x + ipos.x);
    uint overflowed = 0;
    uint3 prev;
    gRenderParams.mLightImage.InterlockedAdd(address + 0, icolor[0], prev[0]);
    gRenderParams.mLightImage.InterlockedAdd(address + 4, icolor[1], prev[1]);
    gRenderParams.mLightImage.InterlockedAdd(address + 8, icolor[2], prev[2]);
    for (uint i = 0; i < 3; i++) {
        if (icolor[i] >= 0xFFFFFFFF - prev[i])
            overflowed |= BIT(i);
    }
    gRenderParams.mLightImage.InterlockedOr(address + 12, overflowed);
}

typedef VcmVertex CameraVertex;
typedef VcmVertex LightVertex;

// The sole point of this structure is to make carrying around the ray baggage easier.
struct SubPathState {
    VcmVertex mVertex;
    float3 mDirection; // Where to go next
    float mFwdBsdfPdfW;
	bool mIsFiniteLight; // Just generate by finite light
	bool mSpecularPath; // All scattering events so far were specular
};

// Mis power, we use balance heuristic
float Mis(float aPdf) {
    return aPdf;
}

// ported from https://github.com/SmallVCM/SmallVCM/blob/master/src/vertexcm.hxx

struct VertexCM {
	static const AbstractCamera gCamera = { 0 };

    uint2 mIndex;
    uint2 mOutputExtent;
    RandomSampler mRng;

    uint mScreenPixelCount;
	uint mLightSubPathCount;

    float mMisVmWeightFactor;
	float mMisVcWeightFactor;
    float mVmNormalization;

    property uint pathIndex { get { return mIndex.y * mOutputExtent.x + mIndex.x; } };

    __init(const uint2 index, const uint2 extent, const int aIteration, const int aRngOffset) {
        mIndex = index;
        mOutputExtent = extent;
        mRng = RandomSampler(gPushConstants.mRandomSeed, index, aRngOffset);
        const uint pathCount = extent.x * extent.y;
        mScreenPixelCount  = pathCount;
        mLightSubPathCount = pathCount;

        // Setup our radius, 1st iteration has aIteration == 0, thus offset
        float radius = gPushConstants.mRadiusFactor * gPushConstants.mSceneSphere.w;
        radius /= pow(float(aIteration + 1), 0.5f * (1 - gPushConstants.mRadiusAlpha));
        // Purely for numeric stability
        radius = max(radius, 1e-7f);
        const float radiusSqr = pow2(radius);

        // Factor used to normalise vertex merging contribution.
        // We divide the summed up energy by disk radius and number of light paths
        mVmNormalization = 1.f / (radiusSqr * M_PI * mLightSubPathCount);

        // MIS weight constant [tech. rep. (20)], with n_VC = 1 and n_VM = mLightPathCount
        const float etaVCM = (M_PI * radiusSqr) * mLightSubPathCount;
        mMisVmWeightFactor = gUseVM ? Mis(etaVCM) : 0.f;
        mMisVcWeightFactor = gUseVC ? Mis(1.f / etaVCM) : 0.f;
    }

	void StoreLightVertex(const VcmVertex vertex) {
		const uint index = gRenderParams.mLightPathLengths[pathIndex];
		gRenderParams.mLightPathLengths[pathIndex] = index + 1;
		gRenderParams.mLightVertices[index * mLightSubPathCount + pathIndex] = vertex;
	}
	VcmVertex LoadLightVertex(const uint vertexIndex) {
		return gRenderParams.mLightVertices[vertexIndex * mLightSubPathCount + pathIndex];
	}

    void StoreAuxillaryData(const IntersectionResult isect, const float3 albedo) {
        float2 pixelCoord = mIndex + 0.5;
		if (isect.instanceIndex != INVALID_INSTANCE)
        	gCamera.view.toRaster(gCamera.prevInverseTransform.transformPoint(isect.mShadingData.mPosition), pixelCoord);

        gRenderParams.mPrevUVs[mIndex] = pixelCoord / mOutputExtent;

        VisibilityData v;
        v.mInstancePrimitiveIndex = isect.mInstancePrimitiveIndex;
        v.mPackedNormal = isect.mShadingData.mPackedShadingNormal;
        gRenderParams.mVisibility[mIndex] = reinterpret<uint2>(v);

        DepthData d;
        d.mDepth = isect.mDistance;
        d.mPrevDepth = isect.mDistance;
        d.mDepthDerivative = 1;
		gRenderParams.mDepth[mIndex] = reinterpret<float4>(d);
	}


    [mutating]
    void GenerateLightPaths() {
        if (gPathTraceOnly) return;

		SubPathState lightState;
		GenerateLightSample(lightState);

        gRenderParams.mLightPathLengths[pathIndex] = 0;

		//////////////////////////////////////////////////////////////////////////
		// Trace light path
		for (;; ++lightState.mVertex.mPathLength) {
            // Offset ray origin instead of setting tmin due to numeric
            // issues in ray-sphere intersection. The isect.dist has to be
            // extended by this EPS_RAY after hit point is determined
            const RayDesc ray = makeRay(OffsetRayOrigin(lightState.mVertex, lightState.mDirection), lightState.mDirection);

			IntersectionResult isect;
			if (!gScene.traceRay(ray, true, isect))
                break;

            lightState.mVertex.mShadingData = isect.mShadingData;
            lightState.mVertex.mShadingData.mTexcoordScreenSize = 0;
            lightState.mVertex.mLocalDirectionIn = packNormal(lightState.mVertex.mShadingData.toLocal(-lightState.mDirection));

			const Material bsdf = Material(lightState.mVertex.mShadingData);
			if (!bsdf.canEvaluate())
                break;

            // Update the MIS quantities before storing them at the vertex.
			// These updates follow the initialization in GenerateLightSample() or
			// SampleScattering(), and together implement equations [tech. rep. (31)-(33)]
			// or [tech. rep. (34)-(36)], respectively.
            {
                const float cosTheta = -dot(lightState.mVertex.mShadingData.geometryNormal, lightState.mDirection);

				// Infinite lights use MIS handled via solid angle integration,
				// so do not divide by the distance for such lights [tech. rep. Section 5.1]
				if (lightState.mVertex.mPathLength > 1 || lightState.mIsFiniteLight == 1)
					lightState.mVertex.dVCM *= Mis(pow2(isect.mDistance));
				lightState.mVertex.dVCM /= Mis(abs(cosTheta));
				lightState.mVertex.dVC /= Mis(abs(cosTheta));
				lightState.mVertex.dVM /= Mis(abs(cosTheta));
			}

            if (!bsdf.isSingular()) {
				// Store vertex, unless BSDF is purely specular, which prevents
				// vertex connections and merging
				if (gUseVC || gUseVM) {
					StoreLightVertex(lightState.mVertex);
				}

				// Connect to camera, unless BSDF is purely specular
				if (gUseVC || gLightTraceOnly) {
					if (lightState.mVertex.mPathLength + 1 >= gPushConstants.mMinPathLength)
						ConnectToCamera(lightState, bsdf);
				}
            }

			// Terminate if the path would become too long after scattering
			if (lightState.mVertex.mPathLength + 2 > gPushConstants.mMaxPathLength)
				break;

			// Continue random walk
			if (!SampleScattering<true>(bsdf, lightState))
				break;
		}
	}

	[mutating]
	void GenerateCameraPaths() {
		// initialize with light trace sample
        uint4 lightTraceSample = gRenderParams.mLightImage.Load4(int(pathIndex*16));
		// detect overflow from atomic adds
        for (uint i = 0; i < 3; i++)
            if ((lightTraceSample.w & BIT(i)) > 0)
                lightTraceSample[i] = 0xFFFFFFFF;
        gRenderParams.mOutput[mIndex] = float4(lightTraceSample.rgb / float(gLightTraceQuantization), 1);

        // Unless rendering with traditional light tracing
		if (gLightTraceOnly) return;

		SubPathState cameraState;
        const float2 screenSample = GenerateCameraSample(cameraState);

		//////////////////////////////////////////////////////////////////////
		// Trace camera path
		for (;; ++cameraState.mVertex.mPathLength) {
            // Offset ray origin instead of setting tmin due to numeric
            // issues in ray-sphere intersection. The isect.dist has to be
            // extended by this EPS_RAY after hit point is determined
            const RayDesc ray = makeRay(OffsetRayOrigin(cameraState.mVertex, cameraState.mDirection), cameraState.mDirection);

            IntersectionResult isect;
            const bool hit = gScene.traceRay(ray, true, isect);

			if (cameraState.mVertex.mPathLength == 1)
                StoreAuxillaryData(isect, 1);

            if (!hit) {
                if (cameraState.mVertex.mPathLength == 1)
	                gRenderParams.mAlbedo[mIndex] = 0;

				// Get radiance from environment
				const LightRadianceRecord background = GetBackgroundRadiance(cameraState.mDirection);
				if (any(background.mRadiance > 0)) {
					if (cameraState.mVertex.mPathLength >= gPushConstants.mMinPathLength) {
						AddColor(mIndex, cameraState.mVertex.mThroughput * GetLightRadiance(background, cameraState, 1));
					}
				}

				break;
            }

            cameraState.mVertex.mShadingData = isect.mShadingData;
            cameraState.mVertex.mShadingData.mTexcoordScreenSize = 0;
            cameraState.mVertex.mLocalDirectionIn = packNormal(cameraState.mVertex.mShadingData.toLocal(-cameraState.mDirection));

            float fwdG;

            // Update the MIS quantities, following the initialization in
            // GenerateLightSample() or SampleScattering(). Implement equations
            // [tech. rep. (31)-(33)] or [tech. rep. (34)-(36)], respectively.
            {
                const float cosTheta = -dot(cameraState.mVertex.mShadingData.geometryNormal, cameraState.mDirection);
				cameraState.mVertex.dVCM *= Mis(pow2(isect.mDistance));
				cameraState.mVertex.dVCM /= Mis(abs(cosTheta));
				cameraState.mVertex.dVC  /= Mis(abs(cosTheta));
                cameraState.mVertex.dVM  /= Mis(abs(cosTheta));

                fwdG = cosTheta / pow2(isect.mDistance);
			}

            const Material bsdf = Material(cameraState.mVertex.mShadingData);

            if (cameraState.mVertex.mPathLength == 1)
                gRenderParams.mAlbedo[mIndex] = float4(bsdf.albedo(), 1);

			// Light source has been hit; terminate afterwards, since
			// our light sources do not have reflective properties
            const LightRadianceRecord light = GetSurfaceRadiance(cameraState.mDirection, isect, bsdf);
            if (any(light.mRadiance > 0)) {
				if (cameraState.mVertex.mPathLength >= gPushConstants.mMinPathLength) {
					AddColor(mIndex, cameraState.mVertex.mThroughput * GetLightRadiance(light, cameraState, fwdG));
				}
				break;
			}

			if (!bsdf.canEvaluate())
				break;

			// Terminate if eye sub-path is too long for connections or merging
			if (cameraState.mVertex.mPathLength >= gPushConstants.mMaxPathLength)
				break;

            if (!bsdf.isSingular()) {
                ////////////////////////////////////////////////////////////////
                // Vertex connection: Connect to a light source
                if (gUseVC || gPathTraceOnly) {
					if (cameraState.mVertex.mPathLength + 1 >= gPushConstants.mMinPathLength) {
						AddColor(mIndex, cameraState.mVertex.mThroughput * DirectIllumination(cameraState, bsdf));
					}
				}

                ////////////////////////////////////////////////////////////////
                // Vertex connection: Connect to light vertices
                if (gUseVC && !gPathTraceOnly) {
					// For VC, each light sub-path is assigned to a particular eye
					// sub-path, as in traditional BPT. It is also possible to
					// connect to vertices from any light path, but MIS should
					// be revisited.
					const uint count = gRenderParams.mLightPathLengths[pathIndex];

					for (int i = 0; i < count; i++) {
                        LightVertex lightVertex = LoadLightVertex(i);
                        const uint fullPathLength = lightVertex.mPathLength + 1 + cameraState.mVertex.mPathLength;

						if (fullPathLength < gPushConstants.mMinPathLength)
							continue;

						// Light vertices are stored in increasing path length
						// order; once we go above the max path length, we can
						// skip the rest
						if (fullPathLength > gPushConstants.mMaxPathLength)
							break;

						AddColor(mIndex, cameraState.mVertex.mThroughput * lightVertex.mThroughput * ConnectVertices(lightVertex, bsdf, cameraState));
					}
				}

                ////////////////////////////////////////////////////////////////
                // Vertex merging: Merge with light vertices
                if (gUseVM && !gPathTraceOnly) {
					//RangeQuery query(*this, hitPoint, bsdf, cameraState);
					//mHashGrid.Process(mLightVertices, query);
					//AddColor(cameraState.mVertex.mThroughput * mVmNormalization * query.GetContrib());

					// PPM merges only at the first non-specular surface from camera
					if (gPpm) break;
				}
            }

			if (!SampleScattering<false>(bsdf, cameraState))
				break;
		}
    }


    //////////////////////////////////////////////////////////////////////////
    // Camera tracing methods
    //////////////////////////////////////////////////////////////////////////

    // Generates new camera sample given a pixel index
    [mutating]
    float2 GenerateCameraSample(out SubPathState oCameraState) {
        // Generate ray
        float2 sample;
        const float3 rayDirection = gCamera.view.toWorld(float2(mIndex) + 0.5, sample);

        // Compute pdf conversion factor from area on image plane to solid angle on ray
        const float cosAtCamera = abs(rayDirection.z);
        const float imagePointToCameraDist = gCamera.view.imagePlaneDist() / cosAtCamera;
        const float imageToSolidAngleFactor = pow2(imagePointToCameraDist) / cosAtCamera;

        // We put the virtual image plane at such a distance from the camera origin
        // that the pixel area is one and thus the image plane sampling pdf is 1.
        // The solid angle ray pdf is then equal to the conversion factor from
        // image plane area density to ray solid angle density
        const float cameraPdfW = imageToSolidAngleFactor;

        const TransformData cameraTransform = gCamera.transform;

        oCameraState.mDirection = normalize(cameraTransform.transformVector(rayDirection));

        oCameraState.mVertex.mShadingData.mPosition = cameraTransform.transformPoint(0);
        oCameraState.mVertex.mShadingData.mPackedGeometryNormal = oCameraState.mVertex.mShadingData.mPackedShadingNormal = packNormal(normalize(cameraTransform.transformVector(float3(0,0,sign(rayDirection.z)))));
        oCameraState.mVertex.mThroughput = 1;
        oCameraState.mVertex.mPathLength = 1;
        oCameraState.mSpecularPath = true;

        // Eye sub-path MIS quantities. Implements [tech. rep. (31)-(33)] partially.
        // The evaluation is completed after tracing the camera ray in the eye sub-path loop.
        oCameraState.mVertex.dVCM = Mis(mLightSubPathCount / cameraPdfW);
        oCameraState.mVertex.dVC = 0;
        oCameraState.mVertex.dVM = 0;

        return sample;
    }

    // Returns the radiance of a light source when hit by a random ray,
    // multiplied by MIS weight. Can be used for both Background and Area lights.
    //
    // For Background lights:
    //    Has to be called BEFORE updating the MIS quantities.
    //    Value of aHitpoint is irrelevant (passing float3(0))
    //
    // For Area lights:
    //    Has to be called AFTER updating the MIS quantities.
    float3 GetLightRadiance(const LightRadianceRecord aLight, const SubPathState aCameraState, const float aFwdG) {
        if (all(aLight.mRadiance <= 0))
            return 0;

        // If we see light source directly from camera, no weighting is required
        if (aCameraState.mVertex.mPathLength == 1)
            return aLight.mRadiance;


        // When using only vertex merging, we want purely specular paths
        // to give radiance (cannot get it otherwise). Rest is handled
        // by merging and we should return 0.
        if (gUseVM && !gUseVC)
            return aCameraState.mSpecularPath ? aLight.mRadiance : 0;


        // Partial eye sub-path MIS weight [tech. rep. (43)].
        // If the last hit was specular, then dVCM == 0.
        const float wCamera = Mis(aLight.mDirectPdfA) * aCameraState.mVertex.dVCM + Mis(aLight.mEmissionPdfW) * aCameraState.mVertex.dVC;

        // Partial light sub-path weight is 0 [tech. rep. (42)].

        float misWeight;
        if (gPathTraceOnly) {
			// Basic NEE MIS weight
            misWeight = 1 / (1 + Mis(pdfAtoW(aLight.mDirectPdfA, aFwdG)) / Mis(aCameraState.mFwdBsdfPdfW));
        } else {
            // Full path MIS weight [tech. rep. (37)].
            misWeight = 1.f / (1.f + wCamera);
        }

        return misWeight * aLight.mRadiance;
    }

    // Connects camera vertex to randomly chosen light point.
    // Returns emitted radiance multiplied by path MIS weight.
    // Has to be called AFTER updating the MIS quantities.
    [mutating]
    float3 DirectIllumination(const SubPathState aCameraState, const BSDF aBsdf) {
        const IlluminationSampleRecord emissionSample = gScene.sampleIllumination(aCameraState.mVertex.mShadingData.mPosition, mRng.nextFloat4());

        // If radiance == 0, other values are undefined, so have to early exit
        if (all(emissionSample.mRadiance <= 0))
            return 0;

        const float3 localDirIn   = unpackNormal(aCameraState.mVertex.mLocalDirectionIn);
        const float3 localToLight = aCameraState.mVertex.mShadingData.toLocal(emissionSample.mDirectionToLight);
		MaterialEvalRecord evaluateRecord = aBsdf.evaluate<false>(localDirIn, localToLight);
        evaluateRecord.mReflectance *= aCameraState.mVertex.mShadingData.shadingNormalCorrection<false>(localDirIn, localToLight);

        if (all(evaluateRecord.mReflectance <= 0))
            return 0;

        const float continuationProbability = aBsdf.continuationProb();

        // If the light is delta light, we can never hit it
        // by BSDF sampling, so the probability of this path is 0
        const float bsdfDirPdfW = evaluateRecord.mFwdPdfW * (emissionSample.isSingular ? 0.f : continuationProbability);

        const float bsdfRevPdfW = evaluateRecord.mRevPdfW * continuationProbability;

        // Partial light sub-path MIS weight [tech. rep. (44)].
        // Note that wLight is a ratio of area pdfs. But since both are on the
        // light source, their distance^2 and cosine terms cancel out.
        // Therefore we can write wLight as a ratio of solid angle pdfs,
        // both expressed w.r.t. the same shading point.
        const float wLight = Mis(bsdfDirPdfW / emissionSample.mDirectPdfW);

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
        const float wCamera = Mis(emissionSample.mEmissionPdfW * abs(localToLight.z) / (emissionSample.mDirectPdfW * emissionSample.mCosLight)) * (
			mMisVmWeightFactor + aCameraState.mVertex.dVCM + aCameraState.mVertex.dVC * Mis(bsdfRevPdfW));

        float misWeight;

        if (gPathTraceOnly) {
            // Basic NEE MIS weight
            misWeight = 1 / (1 + Mis(bsdfDirPdfW) / Mis(emissionSample.mDirectPdfW));
        } else {
        	// Full path MIS weight [tech. rep. (37)]
            misWeight = 1.f / (wLight + 1.f + wCamera);
		}

        const float3 contrib = (misWeight / emissionSample.mDirectPdfW) * (emissionSample.mRadiance * evaluateRecord.mReflectance);

        if (all(contrib <= 0))
            return 0;

        IntersectionResult isect;
        if (gScene.traceRay(makeRay(OffsetRayOrigin(aCameraState.mVertex, emissionSample.mDirectionToLight), emissionSample.mDirectionToLight, 0, emissionSample.mDistance*0.99), false, isect))
            return 0;

        return contrib;
    }

    // Connects an eye and a light vertex. Result multiplied by MIS weight, but
    // not multiplied by vertex throughputs. Has to be called AFTER updating MIS
    // constants. 'direction' is FROM eye TO light vertex.
    float3 ConnectVertices(inout LightVertex aLightVertex, const BSDF aCameraBsdf, const SubPathState aCameraState) {
        // Get the connection
        float3 direction = aLightVertex.mShadingData.mPosition - aCameraState.mVertex.mShadingData.mPosition;
        const float dist2 = dot(direction, direction);
        float distance = sqrt(dist2);
        direction /= distance;

        // Evaluate BSDF at camera vertex
        const float3 cameraLocalDirIn = unpackNormal(aCameraState.mVertex.mLocalDirectionIn);
        const float3 cameraLocalDirOut = aCameraState.mVertex.mShadingData.toLocal(direction);
        const float cosCamera = cameraLocalDirOut.z;
        const MaterialEvalRecord cameraBsdfEval = aCameraBsdf.evaluate<false>(cameraLocalDirIn, cameraLocalDirOut);
        const float3 cameraBsdfFactor = cameraBsdfEval.mReflectance * aCameraState.mVertex.mShadingData.shadingNormalCorrection<false>(cameraLocalDirIn, cameraLocalDirOut);
        float cameraBsdfDirPdfW = cameraBsdfEval.mFwdPdfW;
        float cameraBsdfRevPdfW = cameraBsdfEval.mRevPdfW;

        if (all(cameraBsdfFactor <= 0))
            return 0;

        // Camera continuation probability (for Russian roulette)
        const float cameraCont = aCameraBsdf.continuationProb();
        cameraBsdfDirPdfW *= cameraCont;
        cameraBsdfRevPdfW *= cameraCont;

        const BSDF lightBsdf = Material(aLightVertex.mShadingData);

        // Evaluate BSDF at light vertex
        const float3 lightLocalDirIn = unpackNormal(aLightVertex.mLocalDirectionIn);
        const float3 lightLocalDirOut = aLightVertex.mShadingData.toLocal(-direction);
        const float cosLight = lightLocalDirOut.z;
        const MaterialEvalRecord lightBsdfEval = lightBsdf.evaluate<true>(lightLocalDirIn, lightLocalDirOut);
        const float3 lightBsdfFactor = lightBsdfEval.mReflectance * aLightVertex.mShadingData.shadingNormalCorrection<true>(lightLocalDirIn, lightLocalDirOut);
        float lightBsdfDirPdfW = lightBsdfEval.mFwdPdfW;
        float lightBsdfRevPdfW = lightBsdfEval.mRevPdfW;

        if (all(lightBsdfFactor <= 0) || cosLight < 0)
            return 0;

        // Light continuation probability (for Russian roulette)
        const float lightCont = lightBsdf.continuationProb();
        lightBsdfDirPdfW *= lightCont;
        lightBsdfRevPdfW *= lightCont;

        // Compute geometry term. cosine terms are handled in bsdf evaluations
        const float geometryTerm = 1 / dist2;

        // Convert pdfs to area pdf
        const float cameraBsdfDirPdfA = pdfWtoA(cameraBsdfDirPdfW, abs(cosLight)/pow2(distance));
        const float lightBsdfDirPdfA  = pdfWtoA(lightBsdfDirPdfW, abs(cosCamera)/pow2(distance));

        // Partial light sub-path MIS weight [tech. rep. (40)]
        const float wLight = Mis(cameraBsdfDirPdfA) * (mMisVmWeightFactor + aLightVertex.dVCM + aLightVertex.dVC * Mis(lightBsdfRevPdfW));

        // Partial eye sub-path MIS weight [tech. rep. (41)]
        const float wCamera = Mis(lightBsdfDirPdfA) * (mMisVmWeightFactor + aCameraState.mVertex.dVCM + aCameraState.mVertex.dVC * Mis(cameraBsdfRevPdfW));

        // Full path MIS weight [tech. rep. (37)]
        const float misWeight = 1.f / (wLight + 1.f + wCamera);

        const float3 contrib = (misWeight * geometryTerm) * cameraBsdfFactor * lightBsdfFactor;

        if (all(contrib <= 0))
            return 0;

        IntersectionResult isect;
        if (gScene.traceRay(makeRay(OffsetRayOrigin(aCameraState.mVertex, direction), direction, 0, distance*0.99), false, isect))
            return 0;

        return contrib;
    }


    //////////////////////////////////////////////////////////////////////////
    // Light tracing methods
    //////////////////////////////////////////////////////////////////////////

    // Samples light emission
    [mutating]
    void GenerateLightSample(out SubPathState oLightState) {
        const EmissionSampleRecord emissionSample = gScene.sampleEmission(mRng.nextFloat4(), mRng.nextFloat2());

        oLightState.mVertex.mShadingData.mPosition = emissionSample.mPosition;
        oLightState.mVertex.mShadingData.mPackedGeometryNormal = oLightState.mVertex.mShadingData.mPackedShadingNormal = emissionSample.mPackedNormal;
        oLightState.mDirection = emissionSample.mDirection;

        oLightState.mVertex.mThroughput = emissionSample.mRadiance / emissionSample.mEmissionPdfW;
        oLightState.mVertex.mPathLength = 1;
        oLightState.mIsFiniteLight = emissionSample.isFinite;

        // Light sub-path MIS quantities. Implements [tech. rep. (31)-(33)] partially.
        // The evaluation is completed after tracing the emission ray in the light sub-path loop.
        // Delta lights are handled as well [tech. rep. (48)-(50)].
        {
            oLightState.mVertex.dVCM = Mis(emissionSample.mDirectPdfA / emissionSample.mEmissionPdfW);

            if (!emissionSample.isSingular) {
                const float usedCosLight = emissionSample.isFinite ? emissionSample.mCosLight : 1.f;
                oLightState.mVertex.dVC = Mis(usedCosLight / emissionSample.mEmissionPdfW);
            } else {
                oLightState.mVertex.dVC = 0.f;
            }

            oLightState.mVertex.dVM = oLightState.mVertex.dVC * mMisVcWeightFactor;
        }
    }

    // Computes contribution of light sample to camera by splatting is onto the
    // framebuffer. Multiplies by throughput (obviously, as nothing is returned).
    [mutating]
    void ConnectToCamera(const SubPathState aLightState, const BSDF aBsdf) {
        const TransformData cameraTransform = gCamera.transform;
        const float3 cameraPosition = cameraTransform.transformPoint(0);
        const float3 cameraForward  = cameraTransform.transformVector(float3(0, 0, 1));

        // Check it projects to the screen (and where)
        float2 imagePos;
        if (!gCamera.view.toRaster(gCamera.inverseTransform.transformPoint(aLightState.mVertex.mShadingData.mPosition), imagePos))
            return;

        // Compute distance and normalize direction to camera
        float3 directionToCamera = cameraPosition - aLightState.mVertex.mShadingData.mPosition;
        const float distance = length(directionToCamera);
        directionToCamera /= distance;

        // Get the BSDF
        const float3 localDirIn    = unpackNormal(aLightState.mVertex.mLocalDirectionIn);
        const float3 localToCamera = aLightState.mVertex.mShadingData.toLocal(directionToCamera);
        MaterialEvalRecord evaluateRecord = aBsdf.evaluate<true>(localDirIn, localToCamera);
        evaluateRecord.mReflectance *= aLightState.mVertex.mShadingData.shadingNormalCorrection<true>(localDirIn, localToCamera);

        if (all(evaluateRecord.mReflectance <= 0))
            return;

        const float bsdfRevPdfW = evaluateRecord.mRevPdfW * aBsdf.continuationProb();

        // Compute pdf conversion factor from image plane area to surface area
        const float cosAtCamera = abs(dot(cameraForward, directionToCamera));
        const float imagePointToCameraDist = gCamera.view.imagePlaneDist() / cosAtCamera;
        const float imageToSolidAngleFactor = pow2(imagePointToCameraDist) / cosAtCamera;
        const float imageToSurfaceFactor = imageToSolidAngleFactor * abs(localToCamera.z) / pow2(distance);

        // We put the virtual image plane at such a distance from the camera origin
        // that the pixel area is one and thus the image plane sampling pdf is 1.
        // The area pdf of aHitpoint as sampled from the camera is then equal to
        // the conversion factor from image plane area density to surface area density
        const float cameraPdfA = imageToSurfaceFactor;

        // Partial light sub-path weight [tech. rep. (46)]. Note the division by
        // mLightPathCount, which is the number of samples this technique uses.
        // This division also appears a few lines below in the framebuffer accumulation.
        const float wLight = Mis(cameraPdfA / mLightSubPathCount) * (mMisVmWeightFactor + aLightState.mVertex.dVCM + aLightState.mVertex.dVC * Mis(bsdfRevPdfW));

        // Partial eye sub-path weight is 0 [tech. rep. (47)]

        // Full path MIS weight [tech. rep. (37)]. No MIS for traditional light tracing.
        const float misWeight = gLightTraceOnly ? 1.f : (1.f / (wLight + 1.f));

        const float surfaceToImageFactor = 1.f / imageToSurfaceFactor;

		// divide by cosTheta to eliminate cosTheta in evaluateRecord.mReflectance
        const float3 bsdfFactor = evaluateRecord.mReflectance / abs(localToCamera.z);

        // We divide the contribution by surfaceToImageFactor to convert the (already
        // divided) pdf from surface area to image plane area, w.r.t. which the
        // pixel integral is actually defined. We also divide by the number of samples
        // this technique makes, which is equal to the number of light sub-paths
        const float3 contrib = misWeight * aLightState.mVertex.mThroughput * bsdfFactor / (mLightSubPathCount * surfaceToImageFactor);

        if (all(contrib <= 0))
			return;

        IntersectionResult isect;
        if (gScene.traceRay(makeRay(OffsetRayOrigin(aLightState.mVertex, directionToCamera), directionToCamera, 0, distance), false, isect))
			return;

		InterlockedAddColor(int2(imagePos), mOutputExtent, contrib);
	}

    // Samples a scattering direction camera/light sample according to BSDF.
    // Returns false for termination
    [mutating]
    bool SampleScattering<let Adjoint : bool>(const BSDF aBsdf, inout SubPathState aoState) {
		// Russian roulette
        const float contProb = aBsdf.continuationProb();
        if (mRng.nextFloat() > contProb)
            return false;

		const float3 localDirIn = unpackNormal(aoState.mVertex.mLocalDirectionIn);
        const MaterialSampleRecord sampleRecord = aBsdf.sample<Adjoint>(mRng.nextFloat3(), localDirIn);
        MaterialEvalRecord evaluateRecord = aBsdf.evaluate<Adjoint>(localDirIn, sampleRecord.mDirection);
        evaluateRecord.mReflectance *= aoState.mVertex.mShadingData.shadingNormalCorrection<Adjoint>(localDirIn, sampleRecord.mDirection);

        if (all(evaluateRecord.mReflectance <= 0))
            return false;

        const float cosThetaOut = sampleRecord.mDirection.z;
        const float bsdfDirPdfW = sampleRecord.mFwdPdfW * contProb;
        const float bsdfRevPdfW = sampleRecord.mRevPdfW * contProb;

        aoState.mDirection = aoState.mVertex.mShadingData.toWorld(sampleRecord.mDirection);

        // Sub-path MIS quantities for the next vertex. Only partial - the
        // evaluation is completed when the actual hit point is known,
        // i.e. after tracing the ray, in the sub-path loop.

        if (sampleRecord.isSingular()) {
            // Specular scattering case [tech. rep. (53)-(55)] (partially, as noted above)
            aoState.mVertex.dVCM = 0.f;
            // aoState.mVertex.dVC *= Mis(cosThetaOut / bsdfDirPdfW) * Mis(bsdfRevPdfW);
            // aoState.mVertex.dVM *= Mis(cosThetaOut / bsdfDirPdfW) * Mis(bsdfRevPdfW);
            //assert(bsdfDirPdfW == bsdfRevPdfW);
            aoState.mVertex.dVC *= Mis(cosThetaOut);
            aoState.mVertex.dVM *= Mis(cosThetaOut);

            // aoState.mSpecularPath &= 1;
            aoState.mFwdBsdfPdfW = POS_INFINITY;
        } else {
            // Implements [tech. rep. (34)-(36)] (partially, as noted above)
            aoState.mVertex.dVC = Mis(cosThetaOut / bsdfDirPdfW) * (aoState.mVertex.dVC * Mis(bsdfRevPdfW) + aoState.mVertex.dVCM + mMisVmWeightFactor);
            aoState.mVertex.dVM = Mis(cosThetaOut / bsdfDirPdfW) * (aoState.mVertex.dVM * Mis(bsdfRevPdfW) + aoState.mVertex.dVCM * mMisVcWeightFactor + 1.f);
            aoState.mVertex.dVCM = Mis(1.f / bsdfDirPdfW);

            aoState.mSpecularPath = false;
            aoState.mFwdBsdfPdfW = bsdfDirPdfW;
        }

        aoState.mVertex.mThroughput *= evaluateRecord.mReflectance / bsdfDirPdfW;

        return true;
    }
};

[shader("compute")]
[numthreads(8, 8, 1)]
void GenerateLightPaths(uint3 index: SV_DispatchThreadID) {
    uint2 extent;
    gRenderParams.mOutput.GetDimensions(extent.x, extent.y);
    if (any(index.xy >= extent)) return;

    VertexCM pt = VertexCM(index.xy, extent, 0, 0);

    pt.GenerateLightPaths();
}

[shader("compute")]
[numthreads(8, 8, 1)]
void GenerateCameraPaths(uint3 index: SV_DispatchThreadID) {
    uint2 extent;
    gRenderParams.mOutput.GetDimensions(extent.x, extent.y);
    if (any(index.xy >= extent)) return;

    VertexCM pt = VertexCM(index.xy, extent, 0, 16384);

    pt.GenerateCameraPaths();
}