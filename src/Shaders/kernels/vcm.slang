// Based on https://github.com/SmallVCM/SmallVCM/blob/master/src/vertexcm.hxx

#include "common/path.hlsli"

static const AbstractCamera gCamera = { 0 };

// For visibility tests: after the ray origin is offsetted with rayOffset(), TMax should be decreased
// so the ray doesn't intersect the surface/light (resulting in an occlusion)
float AdjustDistance(const float dist) {
	return dist * 0.99;
}
uint GetLightVertexIndex(const uint lightPathIndex, const uint vertexIndex) {
	// store light paths individually, so specific path vertices can be found
	// i.e. { path0.vertex0, ..., pathN.vertex0, path0.vertex1, ..., pathN.vertex1, ... }
	return vertexIndex * gPushConstants.mLightSubPathCount + lightPathIndex;
}


typedef VcmVertex CameraVertex;
typedef VcmVertex LightVertex;

// The sole point of this structure is to make carrying around the ray baggage easier.
// 128 bytes
struct SubPathState {
	VcmVertex mVertex;
	float3 mDirection; // Where to go next
    float mFwdBsdfPdfW; // for NEE MIS when doing regular path tracing. Other techniques use MIS quantities in mVertex.
    uint mInstancePrimitiveIndex;
	uint mFlags;
    //uint pad;
    //uint4 pad1;

	// Just generate by finite light
	property bool mIsFiniteLight {
		get { return BF_GET(mFlags, 0, 1); }
		set { BF_SET(mFlags, uint(newValue), 0, 1); }
	}
	// All scattering events so far were specular
	property bool mSpecularPath {
		get { return BF_GET(mFlags, 1, 1); }
		set { BF_SET(mFlags, uint(newValue), 1, 1); }
    }

    property uint mInstanceIndex {
        get { return BF_GET(mInstancePrimitiveIndex, 0, 16); }
        set { BF_SET(mInstancePrimitiveIndex, uint(newValue), 0, 16); }
    }
    property uint mPrimitiveIndex {
        get { return BF_GET(mInstancePrimitiveIndex, 16, 16); }
        set { BF_SET(mInstancePrimitiveIndex, uint(newValue), 16, 16); }
    }
};

struct VertexCM {
	RandomSampler mRng;
	uint2 mIndex;

    property uint mPathIndex { get { return mIndex.y * gPushConstants.mOutputExtent.x + mIndex.x; } };

    __init(const uint2 aThreadIndex, const int aRngOffset = 0) {
        mIndex = aThreadIndex;
        mRng = RandomSampler(gPushConstants.mRandomSeed, aThreadIndex, aRngOffset);
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
        const MaterialSampleRecord sampleRecord   = aBsdf.sample<Adjoint>(mRng.nextFloat3(), localDirIn);
        MaterialEvalRecord         evaluateRecord = aBsdf.evaluate<Adjoint>(localDirIn, sampleRecord.mDirection);
        evaluateRecord.mReflectance *= aoState.mVertex.mShadingData.shadingNormalCorrection<Adjoint>(localDirIn, sampleRecord.mDirection);

        if (all(evaluateRecord.mReflectance <= 0))
            return false;

        const float cosThetaOut = aoState.mVertex.mShadingData.isSurface() ? abs(sampleRecord.mDirection.z) : 1;
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
            aoState.mVertex.dVC = Mis(cosThetaOut / bsdfDirPdfW) * (aoState.mVertex.dVC * Mis(bsdfRevPdfW) + aoState.mVertex.dVCM + gRenderParams.mVcmConstants.mMisVmWeightFactor);
            aoState.mVertex.dVM = Mis(cosThetaOut / bsdfDirPdfW) * (aoState.mVertex.dVM * Mis(bsdfRevPdfW) + aoState.mVertex.dVCM * gRenderParams.mVcmConstants.mMisVcWeightFactor + 1.f);
            aoState.mVertex.dVCM = Mis(1.f / bsdfDirPdfW);

            aoState.mSpecularPath = false;
            aoState.mFwdBsdfPdfW = bsdfDirPdfW;
        }

        aoState.mVertex.mThroughput *= evaluateRecord.mReflectance / bsdfDirPdfW;
        aoState.mVertex.mPathPdfA *= bsdfDirPdfW;

        return true;
    }
}

// Camera tracing methods
extension VertexCM {
    // Generates new camera sample given a pixel index
    [mutating]
    float2 GenerateCameraSample(out SubPathState oCameraState) {
        // Generate ray
        float2 sample;
        const float3 rayDirection = gCamera.getView().toWorld(float2(mIndex) + 0.5, sample);

        // Compute pdf conversion factor from area on image plane to solid angle on ray
        const float cosAtCamera = abs(rayDirection.z);
        const float imagePointToCameraDist = gCamera.getView().imagePlaneDist() / cosAtCamera;
        const float imageToSolidAngleFactor = pow2(imagePointToCameraDist) / cosAtCamera;

        // We put the virtual image plane at such a distance from the camera origin
        // that the pixel area is one and thus the image plane sampling pdf is 1.
        // The solid angle ray pdf is then equal to the conversion factor from
        // image plane area density to ray solid angle density
        const float cameraPdfW = imageToSolidAngleFactor;

        const TransformData cameraTransform = gCamera.getTransform();

        oCameraState.mDirection = normalize(cameraTransform.transformVector(rayDirection));

        oCameraState.mVertex.mShadingData.mPosition = cameraTransform.transformPoint(0);
        oCameraState.mVertex.mShadingData.mPackedGeometryNormal = oCameraState.mVertex.mShadingData.mPackedShadingNormal = packNormal(normalize(cameraTransform.transformVector(float3(0,0,sign(rayDirection.z)))));
        oCameraState.mVertex.mThroughput = 1;
        oCameraState.mVertex.mPathLength = 1;
        oCameraState.mVertex.mPathPdfA = 1;
        oCameraState.mSpecularPath = true;

        // Eye sub-path MIS quantities. Implements [tech. rep. (31)-(33)] partially.
        // The evaluation is completed after tracing the camera ray in the eye sub-path loop.
        oCameraState.mVertex.dVCM = Mis(gPushConstants.mLightSubPathCount / cameraPdfW);
        oCameraState.mVertex.dVC = 0;
        oCameraState.mVertex.dVM = 0;

        return sample;
    }

    void StoreAuxillaryData(const IntersectionResult isect, const float3 albedo) {
        float2 prevPixelCoord = mIndex + 0.5;

        float prevDepth = isect.mDistance;
        if (isect.mInstanceIndex != INVALID_INSTANCE) {
            const float3 prevCameraPos = gCamera.getPrevInverseTransform().transformPoint(gScene.mInstanceMotionTransforms[isect.mInstanceIndex].transformPoint(isect.mShadingData.mPosition));
            prevDepth = length(prevCameraPos);
            gCamera.getView().toRaster(prevCameraPos, prevPixelCoord);
        }

        gRenderParams.mPrevUVs[mIndex] = prevPixelCoord / gPushConstants.mOutputExtent;

        VisibilityData v;
        v.mInstancePrimitiveIndex = isect.mInstancePrimitiveIndex;
        v.mPackedNormal           = isect.mShadingData.mPackedShadingNormal;
        gRenderParams.mVisibility[mIndex] = reinterpret<uint2>(v);

        DepthData d;
        d.mDepth     = isect.mDistance;
        d.mPrevDepth = prevDepth;
        d.mDepthDerivative = 1;
        gRenderParams.mDepth[mIndex] = reinterpret<float4>(d);
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
    static float3 GetLightRadiance(const LightRadianceRecord aLight, const SubPathState aCameraState, const float aFwdG) {
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
            misWeight = 1 / (1 + wCamera);
        }

        if (gDebugPaths && !gDebugPathWeights)
            misWeight = 1;

        return misWeight * aLight.mRadiance;
    }


    ////////////////////////////////
	// Direct illumination methods
	////////////////////////////////

    struct ReservoirStateDI {
        IlluminationSampleRecord mCachedSample;
        float4 mSampleRnd;
        float mCachedTargetPdf;
        float mTotalWeight;
        float M;

        property float mIntegrationWeight { get { return mCachedTargetPdf > 0 ? mTotalWeight / mCachedTargetPdf : 0; } }

        __init() {
			mCachedTargetPdf = 0;
            mTotalWeight = 0;
            M = 0;
		}

        static float GetTargetPdf<let bVisibility : bool>(const ShadingData aRefPos, const float4 aSample, out IlluminationSampleRecord oSampleRecord) {
            oSampleRecord = gScene.sampleIllumination(aRefPos.mPosition, aSample);
            const float cosTheta = dot(aRefPos.getShadingNormal(), oSampleRecord.mDirectionToLight);
            const float targetPdf = luminance(oSampleRecord.mRadiance) * abs(cosTheta) * oSampleRecord.getG();

            if (bVisibility && targetPdf > 0) {
                IntersectionResult isect;
                if (gScene.traceRay(makeRay(rayOffset(aRefPos.mPosition, aRefPos.getGeometryNormal(), oSampleRecord.mDirectionToLight), oSampleRecord.mDirectionToLight, 0, AdjustDistance(oSampleRecord.mDistance)), false, isect)) {
                    return 0;
                }
            }

            return targetPdf;
        }

		// assumes M has already been set to the correct sample count
        [mutating]
        bool ProcessSample(const SubPathState aCameraState, const BSDF aCameraBsdf, const float4 aSampleRnd, const float aRnd) {
            IlluminationSampleRecord sample;
            const float targetPdf = GetTargetPdf<false>(aCameraState.mVertex.mShadingData, aSampleRnd, sample);
            if (targetPdf <= 0)
				return false;

            const float wi = targetPdf / (sample.getDirectPdfA() * M);
            mTotalWeight += wi;
            if (aRnd * mTotalWeight <= wi) {
                mSampleRnd = aSampleRnd;
                mCachedTargetPdf = targetPdf;
                mCachedSample = sample;
                return true;
			}
			return false;
		}

        // turns this reservoir into a new one containing just the current and reused samples
        [mutating]
        bool Reuse(const SubPathState aCameraState, const BSDF aCameraBsdf, const DirectIlluminationReservoir aReservoir, const float aRnd) {
            if (aReservoir.M <= 0)
                return false;

            const float shiftJacobian = 1;

			// target pdfs

            IlluminationSampleRecord tmp, sample;
            const ShadingData sd_i = gScene.makeShadingData(gScene.mInstances[aReservoir.mInstanceIndex], gScene.mInstanceTransforms[aReservoir.mInstanceIndex], aReservoir.mLocalPosition, aReservoir.mPrimitiveIndex);
            // target pdf in aReservoir's domain of the current sample
			const float phat_i_old = GetTargetPdf<true>(sd_i, mSampleRnd, tmp);
            // target pdf in the current domain of aReservoir's sample
            const float phat_new   = GetTargetPdf<true>(aCameraState.mVertex.mShadingData, aReservoir.mRnd, sample);

			// resampling MIS weights

            const float m0 = safe_divide(mCachedTargetPdf * M, mCachedTargetPdf * M + phat_i_old * aReservoir.M);
            const float m1 = safe_divide(aReservoir.mCachedTargetPdf * aReservoir.M, aReservoir.mCachedTargetPdf * aReservoir.M + phat_new * M);

			M += aReservoir.M;

            // RIS weights

            float w0 = m0 * mCachedTargetPdf * mIntegrationWeight;
            float w1 = m1 * phat_new * aReservoir.mIntegrationWeight * shiftJacobian;

            // create new reservoir from current and new reservoirs

			mTotalWeight = w0 + w1;
            if (aRnd * mTotalWeight <= w1) {
                mSampleRnd = aReservoir.mRnd;
                mCachedTargetPdf = phat_new;
                mCachedSample = sample;
                return true;
            }
            return false;
        }
    };

    [mutating]
    IlluminationSampleRecord SampleDirectIlluminationReservoir(const SubPathState aCameraState, const BSDF aCameraBsdf) {
        ReservoirStateDI reservoir = ReservoirStateDI();

        // canonical samples
        reservoir.M = gPushConstants.mDIReservoirSampleCount;
        for (uint i = 0; i < gPushConstants.mDIReservoirSampleCount; i++)
            reservoir.ProcessSample(aCameraState, aCameraBsdf, mRng.nextFloat4(), mRng.nextFloat());

        // visibility ray for canonical samples
        if (reservoir.mCachedTargetPdf > 0) {
            IntersectionResult isect;
            if (gScene.traceRay(makeRay(aCameraState.mVertex.OffsetRay(reservoir.mCachedSample.mDirectionToLight), reservoir.mCachedSample.mDirectionToLight, 0, AdjustDistance(reservoir.mCachedSample.mDistance)), false, isect))
                reservoir.mCachedTargetPdf = 0;
        }

        if (bool(gDIReservoirFlags & (uint)VcmReservoirFlags::eReuse)) {
            if (gPushConstants.reservoirHistoryValid()) {
				const float2 jitter = mRng.nextFloat2() * 2 - 1;
                const float3x3 frame = makeOrthonormal(aCameraState.mVertex.mShadingData.getGeometryNormal());
                const float3 queryPos = aCameraState.mVertex.mShadingData.mPosition + gPushConstants.mHashGridJitterRadius * (frame[0] * jitter.x + frame[1] * jitter.y);
                const float cellSize = gRenderParams.mPrevDirectIlluminationReservoirs.GetCellSize(queryPos);
				const uint cellIndex = gRenderParams.mPrevDirectIlluminationReservoirs.FindCellIndex(queryPos, cellSize);
                if (cellIndex != -1) {
					const uint2 range = gRenderParams.mPrevDirectIlluminationReservoirs.GetCellDataRange(cellIndex);
                    if (range.y > range.x)
                        reservoir.Reuse(aCameraState, aCameraBsdf, gRenderParams.mPrevDirectIlluminationReservoirs.mData[range.x + mRng.next() % (range.y - range.x)], mRng.nextFloat());
                }
			}

            if (reservoir.mIntegrationWeight > 0) {
                DirectIlluminationReservoir storedReservoir = {};
                storedReservoir.mRnd = reservoir.mSampleRnd;
                storedReservoir.mLocalPosition = gScene.mInstanceInverseTransforms[aCameraState.mInstanceIndex].transformPoint(aCameraState.mVertex.mShadingData.mPosition);
                storedReservoir.mInstancePrimitiveIndex = aCameraState.mInstancePrimitiveIndex;
                storedReservoir.M = min(reservoir.M, gPushConstants.mDIReservoirMaxM * gPushConstants.mDIReservoirSampleCount);
                storedReservoir.mIntegrationWeight = reservoir.mIntegrationWeight;
                storedReservoir.mCachedTargetPdf = reservoir.mCachedTargetPdf;
                gRenderParams.mDirectIlluminationReservoirs.Append(
					aCameraState.mVertex.mShadingData.mPosition,
					aCameraState.mVertex.mShadingData.getShadingNormal(),
					luminance(aCameraState.mVertex.mThroughput),
					storedReservoir);
            }
        }

        reservoir.mCachedSample.mIntegrationWeight = reservoir.mIntegrationWeight * reservoir.mCachedSample.getG();
        return reservoir.mCachedSample;
	}

    // Connects camera vertex to randomly chosen light point.
    // Returns emitted radiance multiplied by path MIS weight.
    // Has to be called AFTER updating the MIS quantities.
    [mutating]
    float3 DirectIllumination(const SubPathState aCameraState, const BSDF aBsdf) {
        IlluminationSampleRecord emissionSample;
        if (bool(gDIReservoirFlags & (uint)VcmReservoirFlags::eRIS)) {
            emissionSample = SampleDirectIlluminationReservoir(aCameraState, aBsdf);
            if (emissionSample.mIntegrationWeight <= 0)
                return 0;
        } else
            emissionSample = gScene.sampleIllumination(aCameraState.mVertex.mShadingData.mPosition, mRng.nextFloat4());

        // If radiance == 0, other values are undefined, so have to early exit
        if (all(emissionSample.mRadiance <= 0) || emissionSample.mIntegrationWeight <= 0)
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
			gRenderParams.mVcmConstants.mMisVmWeightFactor + aCameraState.mVertex.dVCM + aCameraState.mVertex.dVC * Mis(bsdfRevPdfW));

        float misWeight;

        if (gPathTraceOnly) {
            // Basic NEE MIS weight
            misWeight = 1 / (1 + Mis(bsdfDirPdfW) / Mis(emissionSample.mDirectPdfW));
        } else {
        	// Full path MIS weight [tech. rep. (37)]
            misWeight = 1 / (wLight + 1 + wCamera);
		}

        if (gDebugPaths && !gDebugPathWeights)
            misWeight = 1;

        const float3 contrib = (misWeight * emissionSample.mIntegrationWeight) * (emissionSample.mRadiance * evaluateRecord.mReflectance);

        if (all(contrib <= 0))
            return 0;

		// for reservoir samples, visibility is handled by the reservoir's integration weight
        if (!bool(gDIReservoirFlags & (uint)VcmReservoirFlags::eRIS)) {
            IntersectionResult isect;
            if (gScene.traceRay(makeRay(aCameraState.mVertex.OffsetRay(emissionSample.mDirectionToLight), emissionSample.mDirectionToLight, 0, AdjustDistance(emissionSample.mDistance)), false, isect))
                return 0;
        }

        return contrib;
    }


    ////////////////////////////////
    // VC methods
    ////////////////////////////////

    // Connects an eye and a light vertex. Result multiplied by MIS weight, but
    // not multiplied by vertex throughputs. Has to be called AFTER updating MIS constants.
    static float3 ConnectVertices(const CameraVertex aCameraVertex, const BSDF aCameraBsdf, inout LightVertex aLightVertex, out RayDesc oVisibilityRay) {
        // Get the connection
        float3 direction = aLightVertex.mShadingData.mPosition - aCameraVertex.mShadingData.mPosition;
        const float dist2 = dot(direction, direction);
        float distance = sqrt(dist2);
        direction /= distance;

        // Evaluate BSDF at camera vertex
        const float3 cameraLocalDirIn = unpackNormal(aCameraVertex.mLocalDirectionIn);
        const float3 cameraLocalDirOut = aCameraVertex.mShadingData.toLocal(direction);
        const float cosCamera = aCameraVertex.mShadingData.isSurface() ? cameraLocalDirOut.z : 1;
        const MaterialEvalRecord cameraBsdfEval = aCameraBsdf.evaluate<false>(cameraLocalDirIn, cameraLocalDirOut);
        const float3 cameraBsdfFactor = cameraBsdfEval.mReflectance * aCameraVertex.mShadingData.shadingNormalCorrection<false>(cameraLocalDirIn, cameraLocalDirOut);
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
        const float cosLight = aLightVertex.mShadingData.isSurface() ? lightLocalDirOut.z : 1;
        const MaterialEvalRecord lightBsdfEval = lightBsdf.evaluate<true>(lightLocalDirIn, lightLocalDirOut);
        const float3 lightBsdfFactor = lightBsdfEval.mReflectance * aLightVertex.mShadingData.shadingNormalCorrection<true>(lightLocalDirIn, lightLocalDirOut);
        float lightBsdfDirPdfW = lightBsdfEval.mFwdPdfW;
        float lightBsdfRevPdfW = lightBsdfEval.mRevPdfW;

        if (all(lightBsdfFactor <= 0))
            return 0;

        // Light continuation probability (for Russian roulette)
        const float lightCont = lightBsdf.continuationProb();
        lightBsdfDirPdfW *= lightCont;
        lightBsdfRevPdfW *= lightCont;

        // Compute geometry term (cosine terms are handled in bsdf evaluations)
        const float geometryTerm = 1 / dist2;

        // Convert pdfs to area pdf
        const float cameraBsdfDirPdfA = pdfWtoA(cameraBsdfDirPdfW, abs(cosLight)/pow2(distance));
        const float lightBsdfDirPdfA  = pdfWtoA(lightBsdfDirPdfW, abs(cosCamera)/pow2(distance));

        // Partial light sub-path MIS weight [tech. rep. (40)]
        const float wLight = Mis(cameraBsdfDirPdfA) * (gRenderParams.mVcmConstants.mMisVmWeightFactor + aLightVertex.dVCM + aLightVertex.dVC * Mis(lightBsdfRevPdfW));

        // Partial eye sub-path MIS weight [tech. rep. (41)]
        const float wCamera = Mis(lightBsdfDirPdfA) * (gRenderParams.mVcmConstants.mMisVmWeightFactor + aCameraVertex.dVCM + aCameraVertex.dVC * Mis(cameraBsdfRevPdfW));

        // Full path MIS weight [tech. rep. (37)]
        float misWeight = (gDebugPaths && !gDebugPathWeights) ?
			1.f :
			1.f / (wLight + 1.f + wCamera);

        const float3 contrib = (misWeight * geometryTerm) * cameraBsdfFactor * lightBsdfFactor;

        if (all(contrib <= 0))
            return 0;

        oVisibilityRay = makeRay(aCameraVertex.OffsetRay(direction), direction, 0, AdjustDistance(distance));

        return contrib;
    }
    static float3 ConnectVertices(const SubPathState aCameraState, const BSDF aCameraBsdf, inout LightVertex aLightVertex) {
        RayDesc ray;
        const float3 contrib = ConnectVertices(aCameraState.mVertex, aCameraBsdf, aLightVertex, ray);

		if (all(contrib <= 0)) return 0;

		IntersectionResult isect;
		if (gScene.traceRay(ray, false, isect))
			return 0;

        return contrib;
	}

    struct ReservoirStateLVC {
        PackedVcmVertex mSample;
        RayDesc mCachedVisibilityRay;
        float3 mCachedContribution;
        float mCachedTargetPdf;
        float mTotalWeight;
        float M;

        property float mIntegrationWeight { get { return mCachedTargetPdf > 0 ? mTotalWeight / mCachedTargetPdf : 0; } }

        __init() {
            mCachedTargetPdf = 0;
            mTotalWeight = 0;
            M = 0;
        }

        static float GetTargetPdf(const VcmVertex aCameraVertex, const BSDF aCameraBsdf, const PackedVcmVertex aSample, out float3 oContrib, out RayDesc oRay) {
            const uint fullPathLength = aSample.mPathLength + 1 + aCameraVertex.mPathLength;
            if (fullPathLength < gPushConstants.mMinPathLength || fullPathLength > gPushConstants.mMaxPathLength)
                return 0;

            LightVertex lv = VcmVertex(aSample);
            oContrib = (aSample.mThroughput * aSample.mPathPdfA) * ConnectVertices(aCameraVertex, aCameraBsdf, lv, oRay);

            return luminance(oContrib);
        }
        static float GetTargetPdf(const VcmVertex aCameraVertex, const BSDF aCameraBsdf, const PackedVcmVertex aSample, out float3 oContrib) {
            RayDesc ray;
            LightVertex lv = VcmVertex(aSample);
            const float targetPdf = GetTargetPdf(aCameraVertex, aCameraBsdf, aSample, oContrib, ray);

            if (targetPdf > 0) {
                IntersectionResult isect;
                if (gScene.traceRay(ray, false, isect)) {
                    oContrib = 0;
                    return 0;
                }
			}

            return targetPdf;
		}

        // assumes M has already been set to the correct sample count
        [mutating]
        bool ProcessSample(const SubPathState aCameraState, const BSDF aCameraBsdf, const PackedVcmVertex aSample, const float aRnd) {
            RayDesc ray;
            float3 contrib;
            const float targetPdf = GetTargetPdf(aCameraState.mVertex, aCameraBsdf, aSample, contrib, ray);
            if (targetPdf <= 0)
                return false;

            const float wi = targetPdf / (aSample.mPathPdfA * M);
            mTotalWeight += wi;
            if (aRnd * mTotalWeight <= wi) {
                mSample = aSample;
                mCachedVisibilityRay = ray;
                mCachedContribution = contrib;
                mCachedTargetPdf = targetPdf;
                return true;
            }
            return false;
        }

        // turns this reservoir into a new one containing just the current and reused samples
        [mutating]
        bool Reuse(const SubPathState aCameraState, const BSDF aCameraBsdf, const LVCReservoir aReservoir, const float aRnd) {
            if (aReservoir.M <= 0)
                return false;

            const float shiftJacobian = 1;

            // target pdfs

			// target pdf in the current domain of aReservoir's sample
            float3 contrib;
            const float phat_new = GetTargetPdf(aCameraState.mVertex, aCameraBsdf, aReservoir.mLightVertex, contrib);

			// target pdf in aReservoir's domain of the current sample
            float phat_i_old;
            {
                CameraVertex v = VcmVertex(aReservoir.mCameraVertex);
                Material bsdf = Material(v.mShadingData);
                float3 tmp;
                phat_i_old = GetTargetPdf(v, bsdf, mSample, tmp);
            }

            // resampling MIS weights

            const float m0 = safe_divide(mCachedTargetPdf * M, mCachedTargetPdf * M + phat_i_old * aReservoir.M);
            const float m1 = safe_divide(aReservoir.mCachedTargetPdf * aReservoir.M, aReservoir.mCachedTargetPdf * aReservoir.M + phat_new * M);

            M += aReservoir.M;

            // RIS weights

            float w0 = m0 * mCachedTargetPdf * mIntegrationWeight;
            float w1 = m1 * phat_new * aReservoir.mIntegrationWeight * shiftJacobian;

            // create new reservoir from current and new reservoirs

            mTotalWeight = w0 + w1;
            if (aRnd * mTotalWeight <= w1) {
                mSample = aReservoir.mLightVertex;
                mCachedContribution = contrib;
                mCachedTargetPdf = phat_new;
                return true;
            }
            return false;
        }
    }

    [mutating]
    uint SampleLightCells(const SubPathState aCameraState, out float W) {
        uint2 selected = -1;
        float cachedTargetPdf = 0;
        float totalWeight = 0;

        const uint cellCount = gRenderParams.mLightHashGrid.mStats[1];
        const float sourcePdf = 1;// / float(cellCount);

        for (uint i = 0; i < gPushConstants.mLVCHashGridSampleCount; i++) {
            const uint   candidateCellIndex     = gRenderParams.mLightHashGrid.mActiveCells[mRng.next() % cellCount];
            const uint2  candidateCellDataRange = gRenderParams.mLightHashGrid.GetCellDataRange(candidateCellIndex);
            const float3 candidateCellPosition  = gRenderParams.mLightHashGrid.mCellCenters[candidateCellIndex].xyz;
            const float4 candidateCellNormal    = gRenderParams.mLightHashGrid.mCellNormals[candidateCellIndex] / float(gRenderParams.mLightHashGrid.gNormalQuantization);

            float3 toCell = candidateCellPosition - aCameraState.mVertex.mShadingData.mPosition;
            const float dist2 = dot(toCell, toCell);
            toCell /= sqrt(dist2);

            // note: the larger n is, the more concentrated the normals in the cell are
            const float n = length(candidateCellNormal.xyz);

            const float cosCamera = abs(dot(aCameraState.mVertex.mShadingData.getShadingNormal(), toCell));
            const float cosLight  = (n <= 0.5) ? 1 : abs(dot(candidateCellNormal.xyz / n, toCell));

            if (cosLight * cosCamera <= 0)
                continue;

            const float cellSize = gRenderParams.mLightHashGrid.GetCellSize(candidateCellPosition);
            const float targetPdf = (candidateCellDataRange.y - candidateCellDataRange.x) * candidateCellNormal.w / max(pow2(cellSize * .5), dist2) * cosCamera * cosLight;

            if (targetPdf <= 0)
                continue;

            const float wi = targetPdf / sourcePdf;

            totalWeight += wi;
            if (mRng.nextFloat() * totalWeight <= wi) {
                cachedTargetPdf = targetPdf;
                selected = candidateCellDataRange;
			}
        }

        if (selected.x == -1) {
            W = 1;
            return mRng.next() % gRenderParams.mLightPathLengths[0];
        }

		// return a random vertex from the selected cell

        W = totalWeight / (gPushConstants.mLVCHashGridSampleCount * cachedTargetPdf);
        return selected.x + (selected.y - selected.x) % mRng.next();
	}

    [mutating]
    void ConnectToLVCReservoir(const SubPathState aCameraState, const BSDF aCameraBsdf) {
        ReservoirStateLVC reservoir = ReservoirStateLVC();

        // canonical samples
        reservoir.M = gPushConstants.mLVCReservoirSampleCount;
        for (uint i = 0; i < gPushConstants.mLVCReservoirSampleCount; i++) {
            reservoir.ProcessSample(aCameraState, aCameraBsdf, gRenderParams.mLightVertices[mRng.next() % gRenderParams.mLightPathLengths[0]], mRng.nextFloat());
		}

        // visibility ray for canonical samples
        if (reservoir.mCachedTargetPdf > 0) {
            IntersectionResult isect;
            if (gScene.traceRay(reservoir.mCachedVisibilityRay, true, isect))
                reservoir.mCachedTargetPdf = 0;
        }

        if (bool(gLVCReservoirFlags & (uint)VcmReservoirFlags::eReuse)) {
            if (gPushConstants.reservoirHistoryValid()) {
                const float2 jitter = mRng.nextFloat2() * 2 - 1;
                const float3x3 frame = makeOrthonormal(aCameraState.mVertex.mShadingData.getGeometryNormal());
                const float3 queryPos = aCameraState.mVertex.mShadingData.mPosition + gPushConstants.mHashGridJitterRadius * (frame[0] * jitter.x + frame[1] * jitter.y);
                const float cellSize = gRenderParams.mPrevLVCReservoirs.GetCellSize(queryPos);
                const uint cellIndex = gRenderParams.mPrevLVCReservoirs.FindCellIndex(queryPos, cellSize);
                if (cellIndex != -1) {
                    const uint2 range = gRenderParams.mPrevLVCReservoirs.GetCellDataRange(cellIndex);
                    if (range.y > range.x)
                        reservoir.Reuse(aCameraState, aCameraBsdf, gRenderParams.mPrevLVCReservoirs.mData[range.x + mRng.next() % (range.y - range.x)], mRng.nextFloat());
                }
            }

            if (reservoir.mIntegrationWeight > 0) {
                LVCReservoir storedReservoir = {};
                storedReservoir.mLightVertex = reservoir.mSample;
                storedReservoir.mCameraVertex = PackedVcmVertex(aCameraState.mVertex, aCameraState.mInstancePrimitiveIndex);
                storedReservoir.M = min(reservoir.M, gPushConstants.mLVCReservoirMaxM * gPushConstants.mLVCReservoirSampleCount);
                storedReservoir.mIntegrationWeight = reservoir.mIntegrationWeight;
                storedReservoir.mCachedTargetPdf = reservoir.mCachedTargetPdf;
                gRenderParams.mLVCReservoirs.Append(
                    aCameraState.mVertex.mShadingData.mPosition,
                    aCameraState.mVertex.mShadingData.getShadingNormal(),
                    luminance(aCameraState.mVertex.mThroughput),
					storedReservoir);
            }
        }

        if (reservoir.mIntegrationWeight <= 0)
            return;

        const float3 contrib = aCameraState.mVertex.mThroughput * reservoir.mCachedContribution * reservoir.mIntegrationWeight;

        // the LVC paper connects to lightVertexCount/lightPathCount light vertices
        // we only connect to one light vertex, therefore we weight its contribution by lightVertexCount/lightPathCount
        const float w = gRenderParams.mLightPathLengths[0] / (float)gPushConstants.mLightSubPathCount;

        AddColor(mIndex, contrib * w, aCameraState.mVertex.mPathLength + 1, reservoir.mSample.mPathLength + 1);
    }

    // Connect to a random light vertex. Called by LVC-BPT instead of ConnectToLightPath().
    [mutating]
    void ConnectToLVC(const SubPathState aCameraState, const BSDF aCameraBsdf) {
        if (gRenderParams.mLightPathLengths[0] == 0)
            return; // no light vertices...

        uint idx;
		float W;

        if (gLVCHashGridSampling) {
            idx = SampleLightCells(aCameraState, W);
        } else {
            idx = mRng.next() % gRenderParams.mLightPathLengths[0];
            W = 1;
        }

        LightVertex lightVertex = VcmVertex(gRenderParams.mLightVertices[idx]);
        const uint fullPathLength = lightVertex.mPathLength + 1 + aCameraState.mVertex.mPathLength;
        if (fullPathLength < gPushConstants.mMinPathLength || fullPathLength > gPushConstants.mMaxPathLength)
            return;

        const float3 contrib = aCameraState.mVertex.mThroughput * lightVertex.mThroughput * ConnectVertices(aCameraState, aCameraBsdf, lightVertex);

        // the LVC paper connects to lightVertexCount/lightPathCount light vertices
        // we only connect to one light vertex, therefore we weight its contribution by lightVertexCount/lightPathCount
        const float w = gRenderParams.mLightPathLengths[0] / (float)gPushConstants.mLightSubPathCount;

        AddColor(mIndex, contrib * w * W, aCameraState.mVertex.mPathLength + 1, lightVertex.mPathLength + 1);
    }

    // Connect to a light sub-path. Has to be called AFTER updating MIS constants.
    [mutating]
    void ConnectToLightPath(const SubPathState aCameraState, const BSDF aCameraBsdf) {
        // For VC, each light sub-path is assigned to a particular eye
        // sub-path, as in traditional BPT. It is also possible to
        // connect to vertices from any light path, but MIS should
        // be revisited.

        const uint pathIndex = mPathIndex % gPushConstants.mLightSubPathCount;
        const uint count = gRenderParams.mLightPathLengths[pathIndex];

        for (int i = 0; i < count; i++) {
            LightVertex lightVertex = VcmVertex(gRenderParams.mLightVertices[GetLightVertexIndex(pathIndex, i)]);
            const uint fullPathLength = lightVertex.mPathLength + 1 + aCameraState.mVertex.mPathLength;
            if (fullPathLength < gPushConstants.mMinPathLength || all(lightVertex.mThroughput <= 0))
                continue;

            // Light vertices are stored in increasing path length
            // order; once we go above the max path length, we can
            // skip the rest
            if (fullPathLength > gPushConstants.mMaxPathLength)
                break;

            AddColor(mIndex, aCameraState.mVertex.mThroughput * lightVertex.mThroughput * ConnectVertices(aCameraState, aCameraBsdf, lightVertex), aCameraState.mVertex.mPathLength + 1, lightVertex.mPathLength + 1);
        }
    }


    ////////////////////////////////
    // VM methods
    ////////////////////////////////

	// Range query methods used for PPM, BPT, and VCM. When HashGrid finds a vertex
	// within range -- ProcessVertex() is called and vertex
	// merging is performed. BSDF of the camera vertex is used.
    [mutating]
	void MergeVertices(const SubPathState aCameraState, const BSDF aCameraBsdf) {
		const float3 queryPos = aCameraState.mVertex.mShadingData.mPosition;

        const int3 offsetSign = sign(frac(queryPos / gRenderParams.mLightHashGrid.GetCellSize(queryPos)) - 0.5);

        const float mergeRadiusSqr = pow2(gRenderParams.mVcmConstants.mMergeRadius);

		for (int j = 0; j < 8; j++) {
			const int3 offset = int3((j%2), (j%4)/2, j/4) * offsetSign;

			const uint cellIndex = gRenderParams.mLightHashGrid.FindCellIndex(queryPos, gRenderParams.mLightHashGrid.GetCellSize(queryPos), offset);
			if (cellIndex == -1)
				continue;

            uint2 range = gRenderParams.mLightHashGrid.GetCellDataRange(cellIndex);
            float w = 1;
            if (range.y - range.x > 128) {
                w = float(range.y - range.x) / 128;
                range.y = range.x + 128;
			}
			for (; range.x < range.y; range.x++) {
				LightVertex particle = VcmVertex(gRenderParams.mLightVertices[gRenderParams.mLightHashGrid.mData[range.x]]);

				const float3 toParticle = particle.mShadingData.mPosition - queryPos;
				const float distSqr     = dot(toParticle, toParticle);

				if (distSqr <= mergeRadiusSqr)
					ProcessVertex(aCameraState, aCameraBsdf, particle, w);
			}
		}
	}
	void ProcessVertex(const SubPathState aCameraState, const BSDF aCameraBsdf, inout LightVertex aLightVertex, const float aWeight = 1) {
		// Reject if full path length below/above min/max path length
        if ((aLightVertex.mPathLength + aCameraState.mVertex.mPathLength > gPushConstants.mMaxPathLength) ||
            (aLightVertex.mPathLength + aCameraState.mVertex.mPathLength < gPushConstants.mMinPathLength))
            return;

        // Retrieve light incoming direction
        const float3 localDirIn = unpackNormal(aCameraState.mVertex.mLocalDirectionIn);
		const float3 localDirOut = aCameraState.mVertex.mShadingData.toLocal(aLightVertex.mShadingData.toWorld(unpackNormal(aLightVertex.mLocalDirectionIn)));

        const MaterialEvalRecord eval = aCameraBsdf.evaluate<false>(localDirIn, localDirOut);
        const float3 cameraBsdfFactor = eval.mReflectance * aCameraState.mVertex.mShadingData.shadingNormalCorrection<false>(localDirIn, localDirOut);
		float cameraBsdfDirPdfW = eval.mFwdPdfW;
        float cameraBsdfRevPdfW = eval.mRevPdfW;

		if (all(cameraBsdfFactor <= 0))
			return;

		cameraBsdfDirPdfW *= aCameraBsdf.continuationProb();

		const Material lightBsdf = Material(aLightVertex.mShadingData);

		// Even though this is pdf from camera BSDF, the continuation probability
		// must come from light BSDF, because that would govern it if light path
		// actually continued
		cameraBsdfRevPdfW *= lightBsdf.continuationProb();

        // Partial light sub-path MIS weight [tech. rep. (38)]
        const float wLight = aLightVertex.dVCM * gRenderParams.mVcmConstants.mMisVcWeightFactor + aLightVertex.dVM * Mis(cameraBsdfDirPdfW);

        // Partial eye sub-path MIS weight [tech. rep. (39)]
        const float wCamera = aCameraState.mVertex.dVCM * gRenderParams.mVcmConstants.mMisVcWeightFactor + aCameraState.mVertex.dVM * Mis(cameraBsdfRevPdfW);

		// Full path MIS weight [tech. rep. (37)]. No MIS for PPM
		const float misWeight = gPpm ? 1.f : 1.f / (wLight + 1.f + wCamera);

		const float3 contrib = misWeight * cameraBsdfFactor * aLightVertex.mThroughput;
		AddColor(mIndex, aCameraState.mVertex.mThroughput * gRenderParams.mVcmConstants.mVmNormalization * contrib, aCameraState.mVertex.mPathLength + 1, aLightVertex.mPathLength);
	}


    ////////////////////////////////
    // Top-level methods
    ////////////////////////////////

	[mutating]
	void GenerateCameraPaths() {
		if (gPathTraceOnly) {
			gRenderParams.mOutput[mIndex] = float4(0, 0, 0, 1);
		} else {
			// initialize with light trace sample
			uint4 lightTraceSample = gRenderParams.mLightImage.Load4(int(mPathIndex*16));
			// detect overflow from atomic adds
			for (uint i = 0; i < 3; i++)
				if ((lightTraceSample.w & BIT(i)) > 0)
					lightTraceSample[i] = 0xFFFFFFFF;
			gRenderParams.mOutput[mIndex] = float4(lightTraceSample.rgb / float(gPushConstants.mLightImageQuantization), 1);

			// Unless rendering with traditional light tracing
			if (gLightTraceOnly) return;
		}

		SubPathState cameraState;
		const float2 screenSample = GenerateCameraSample(cameraState);

		//////////////////////////////////////////////////////////////////////
		// Trace camera path
		for (;; ++cameraState.mVertex.mPathLength) {
            // Offset ray origin instead of setting tmin due to numeric
            // issues in ray-sphere intersection. The isect.dist has to be
            // extended by this EPS_RAY after hit point is determined
            const RayDesc ray = makeRay(cameraState.mVertex.OffsetRay(cameraState.mDirection), cameraState.mDirection);

			IntersectionResult isect;
			const bool hit = gScene.traceRay(ray, true, isect);
            cameraState.mInstancePrimitiveIndex = isect.mInstancePrimitiveIndex;

			if (cameraState.mVertex.mPathLength == 1)
				StoreAuxillaryData(isect, 1);

			if (!hit) {
				if (cameraState.mVertex.mPathLength == 1)
					gRenderParams.mAlbedo[mIndex] = 0;

				// Get radiance from environment
				const LightRadianceRecord background = GetBackgroundRadiance(cameraState.mDirection);
				if (any(background.mRadiance > 0)) {
					if (cameraState.mVertex.mPathLength >= gPushConstants.mMinPathLength) {
						AddColor(mIndex, cameraState.mVertex.mThroughput * GetLightRadiance(background, cameraState, 1), cameraState.mVertex.mPathLength + 1, 0);
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
                const float cosTheta = cameraState.mVertex.mShadingData.isSurface() ? abs(dot(cameraState.mVertex.mShadingData.getShadingNormal(), cameraState.mDirection)) : 1;
				cameraState.mVertex.dVCM *= Mis(pow2(isect.mDistance));
				cameraState.mVertex.dVCM /= Mis(cosTheta);
				cameraState.mVertex.dVC  /= Mis(cosTheta);
				cameraState.mVertex.dVM  /= Mis(cosTheta);

                fwdG = cosTheta / pow2(isect.mDistance);
                cameraState.mVertex.mPathPdfA *= fwdG;
			}

			const Material bsdf = Material(cameraState.mVertex.mShadingData);

			if (cameraState.mVertex.mPathLength == 1)
				gRenderParams.mAlbedo[mIndex] = float4(bsdf.albedo(), 1);

			// Light source has been hit; terminate afterwards, since
			// our light sources do not have reflective properties
			const LightRadianceRecord light = GetSurfaceRadiance(cameraState.mDirection, isect, bsdf);
			if (any(light.mRadiance > 0)) {
				if (cameraState.mVertex.mPathLength >= gPushConstants.mMinPathLength) {
					AddColor(mIndex, cameraState.mVertex.mThroughput * GetLightRadiance(light, cameraState, fwdG), cameraState.mVertex.mPathLength + 1, 0);
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
						AddColor(mIndex, cameraState.mVertex.mThroughput * DirectIllumination(cameraState, bsdf), cameraState.mVertex.mPathLength + 1, 1);
					}
				}

				if (!gPathTraceOnly) {
					////////////////////////////////////////////////////////////////
					// Vertex connection: Connect to light vertices
                    if (gUseVC) {
                        if (gUseLVC) {
                            if (bool(gLVCReservoirFlags & (uint)VcmReservoirFlags::eRIS))
                                ConnectToLVCReservoir(cameraState, bsdf);
							else
                            	ConnectToLVC(cameraState, bsdf);
                        } else
							ConnectToLightPath(cameraState, bsdf);
					}

					////////////////////////////////////////////////////////////////
					// Vertex merging: Merge with light vertices
					if (gUseVM) {
						MergeVertices(cameraState, bsdf);

						// PPM merges only at the first non-specular surface from camera
						if (gPpm) break;
					}
				}
			}

			if (!SampleScattering<false>(bsdf, cameraState))
				break;
		}
	}
}

// Light tracing methods
extension VertexCM {
    // Samples light emission
    [mutating]
    void GenerateLightSample(out SubPathState oLightState) {
        const EmissionSampleRecord emissionSample = gScene.sampleEmission(mRng.nextFloat4(), mRng.nextFloat2());

        oLightState.mVertex.mShadingData.mPosition = emissionSample.mPosition;
        oLightState.mVertex.mShadingData.mPackedGeometryNormal = oLightState.mVertex.mShadingData.mPackedShadingNormal = emissionSample.mPackedNormal;
        oLightState.mDirection = emissionSample.mDirection;

        oLightState.mVertex.mThroughput = emissionSample.mRadiance / emissionSample.mEmissionPdfW;
        oLightState.mVertex.mPathLength = 1;
        oLightState.mVertex.mPathPdfA = emissionSample.mEmissionPdfW;
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

            oLightState.mVertex.dVM = oLightState.mVertex.dVC * gRenderParams.mVcmConstants.mMisVcWeightFactor;
        }
    }

    // Computes contribution of light sample to camera by splatting is onto the
    // framebuffer. Multiplies by throughput (obviously, as nothing is returned).
    [mutating]
    static void ConnectToCamera(const SubPathState aLightState, const BSDF aBsdf) {
        const TransformData cameraTransform = gCamera.getTransform();
        const float3 cameraPosition = cameraTransform.transformPoint(0);
        const float3 cameraForward  = cameraTransform.transformVector(float3(0, 0, 1));

        // Check it projects to the screen (and where)
        float2 imagePos;
        if (!gCamera.getView().toRaster(gCamera.getInverseTransform().transformPoint(aLightState.mVertex.mShadingData.mPosition), imagePos))
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
        const float imagePointToCameraDist = gCamera.getView().imagePlaneDist() / cosAtCamera;
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
        const float wLight = Mis(cameraPdfA / gPushConstants.mLightSubPathCount) * (gRenderParams.mVcmConstants.mMisVmWeightFactor + aLightState.mVertex.dVCM + aLightState.mVertex.dVC * Mis(bsdfRevPdfW));

        // Partial eye sub-path weight is 0 [tech. rep. (47)]

        // Full path MIS weight [tech. rep. (37)]. No MIS for traditional light tracing.
        const float misWeight = (gLightTraceOnly || (gDebugPaths && !gDebugPathWeights)) ?
			1.f :
			(1.f / (wLight + 1.f));

        const float surfaceToImageFactor = 1.f / imageToSurfaceFactor;

		// divide by cosTheta to eliminate cosTheta in evaluateRecord.mReflectance
        const float3 bsdfFactor = evaluateRecord.mReflectance / abs(localToCamera.z);

        // We divide the contribution by surfaceToImageFactor to convert the (already
        // divided) pdf from surface area to image plane area, w.r.t. which the
        // pixel integral is actually defined. We also divide by the number of samples
        // this technique makes, which is equal to the number of light sub-paths
        const float3 contrib = misWeight * aLightState.mVertex.mThroughput * bsdfFactor / (gPushConstants.mLightSubPathCount * surfaceToImageFactor);

        if (all(contrib <= 0))
			return;

        IntersectionResult isect;
        if (gScene.traceRay(makeRay(aLightState.mVertex.OffsetRay(directionToCamera), directionToCamera, 0, distance), false, isect))
			return;

        InterlockedAddColor(int2(imagePos), gPushConstants.mOutputExtent, contrib, 1, aLightState.mVertex.mPathLength + 1);
	}

    // For regular VC: uses GetLightVertexIndex to find index in mLightPathLengths
    // For LVC VC: atomically appends light vertices to mLightPathLengths
    void StoreLightVertex(const SubPathState aLightState) {
        uint idx;
        if (gUseLVC) {
            if (all(aLightState.mVertex.mThroughput <= 0))
            	return;
			// append to mLightVertices
            InterlockedAdd(gRenderParams.mLightPathLengths[0], 1, idx);
        } else {
			// append to path's vertex list
			const uint storedVertices = gRenderParams.mLightPathLengths[mPathIndex];
			gRenderParams.mLightPathLengths[mPathIndex] = storedVertices + 1;
            idx = GetLightVertexIndex(mPathIndex, storedVertices);
        }

        gRenderParams.mLightVertices[idx] = PackedVcmVertex(aLightState.mVertex, aLightState.mInstancePrimitiveIndex);
        if (gUseVM || gLVCHashGridSampling) {
            if (any(aLightState.mVertex.mThroughput > 0))
                gRenderParams.mLightHashGrid.Append(
                    aLightState.mVertex.mShadingData.mPosition,
                    aLightState.mVertex.mShadingData.getShadingNormal(),
                    luminance(aLightState.mVertex.mThroughput),
					idx);
        }
	}

	// Top-level methods

	[mutating]
	void GenerateLightPaths() {
		if (gPathTraceOnly) return;

		SubPathState lightState;
		GenerateLightSample(lightState);

		if (gUseVC && !gUseLVC)
			gRenderParams.mLightPathLengths[mPathIndex] = 0;

		//////////////////////////////////////////////////////////////////////////
		// Trace light path
		for (;; ++lightState.mVertex.mPathLength) {
            // Offset ray origin instead of setting tmin due to numeric
            // issues in ray-sphere intersection. The isect.dist has to be
            // extended by this EPS_RAY after hit point is determined
            const RayDesc ray = makeRay(lightState.mVertex.OffsetRay(lightState.mDirection), lightState.mDirection);

			IntersectionResult isect;
            if (!gScene.traceRay(ray, true, isect))
                break;
            lightState.mInstancePrimitiveIndex = isect.mInstancePrimitiveIndex;

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
                const float cosTheta = lightState.mVertex.mShadingData.isSurface() ? abs(dot(lightState.mVertex.mShadingData.getShadingNormal(), lightState.mDirection)) : 1;

				// Infinite lights use MIS handled via solid angle integration,
				// so do not divide by the distance for such lights [tech. rep. Section 5.1]
				if (lightState.mVertex.mPathLength > 1 || lightState.mIsFiniteLight == 1)
					lightState.mVertex.dVCM *= Mis(pow2(isect.mDistance));
				lightState.mVertex.dVCM /= Mis(cosTheta);
				lightState.mVertex.dVC /= Mis(cosTheta);
                lightState.mVertex.dVM /= Mis(cosTheta);
                lightState.mVertex.mPathPdfA *= cosTheta / pow2(isect.mDistance);
			}

			if (!bsdf.isSingular()) {
				// Store vertex, unless BSDF is purely specular, which prevents
				// vertex connections and merging
				if (gUseVC || gUseVM) {
					StoreLightVertex(lightState);
				}

				// Connect to camera, unless BSDF is purely specular
                if (gUseVC || gLightTraceOnly) {
                    if (lightState.mVertex.mPathLength + 1 >= gPushConstants.mMinPathLength && lightState.mVertex.mPathLength + 1 <= gPushConstants.mMaxPathLength)
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
};

// entry points

[shader("compute")]
[numthreads(8, 8, 1)]
void GenerateLightPaths(uint3 index: SV_DispatchThreadID) {
    if (index.y * gPushConstants.mOutputExtent.x + index.x >= gPushConstants.mLightSubPathCount) return;

    VertexCM vcm = VertexCM(index.xy, 16384);
    vcm.GenerateLightPaths();
}

[shader("compute")]
[numthreads(8, 8, 1)]
void GenerateCameraPaths(uint3 index: SV_DispatchThreadID) {
    if (any(index.xy >= gPushConstants.mOutputExtent)) return;

    VertexCM vcm = VertexCM(index.xy);
    vcm.GenerateCameraPaths();
}

#ifndef gHashGrid
#define gHashGrid gRenderParams.mLightHashGrid
#endif

[shader("compute")]
[numthreads(64, 1, 1)]
void ComputeHashGridIndices(uint3 index: SV_DispatchThreadID) {
	gHashGrid._ComputeIndices(index.y * gPushConstants.mOutputExtent.x + index.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void SwizzleHashGrid(uint3 index: SV_DispatchThreadID) {
	gHashGrid._Swizzle(index.y * gPushConstants.mOutputExtent.x + index.x);
}