#pragma once

#include "pt.slang"

#ifndef gGIMis
#define gGIMis ReuseMISType::eSimple
#endif

struct ShiftData {
    uint4 mRngSeed;
    ReconnectionVertex mReconnectionVertex;
    uint4 mPackedData;

    // number of path segments before the reconnection vertex
    property uint mPrefixLength {
        get { return BF_GET(mPackedData[0],           0, 8); }
        set {        BF_SET(mPackedData[0], newValue, 0, 8); }
    };
    // number of diffuse bounces after the reconnection vertex (not including the reconnection vertex)
    property uint mSuffixDiffuseBounces {
        get { return BF_GET(mPackedData[0],           8, 8); }
        set {        BF_SET(mPackedData[0], newValue, 8, 8); }
    };
    // number of path segments in the whole path
    property uint mPathLength {
        get { return BF_GET(mPackedData[0],           8, 16); }
        set {        BF_SET(mPackedData[0], newValue, 8, 16); }
    };
    // path pdf before the reconnection vertex
    property float mPrefixPdfW {
        get { return asfloat(mPackedData[1]); }
        set { mPackedData[1] = asuint(newValue); }
    };
    // base path reconnection distance. 0 indicates no reconnection vertex
    property float mReconnectionDist {
        get { return asfloat(mPackedData[2]); }
        set { mPackedData[2] = asuint(newValue); }
    };
    // base path reconnection cos term
    property float mReconnectionCos {
        get { return asfloat(mPackedData[3]); }
        set { mPackedData[3] = asuint(newValue); }
    };

    bool HasReconnectionVertex() {
        return mReconnectionDist > 0;
	}

    __init(const RandomSampler rng) {
        mRngSeed = rng.mState;
        mReconnectionVertex.mPackedData = 0;
        mPackedData = 0;
	}
};

struct Path {
    Spectrum mContribution;
	float mPdfW;
	uint mLength;
    uint mDiffuseBounces;
    bool mTerminated;

    ShiftData mShiftData; // output shift data
    float mRcJacobian; // reconnection jacobian, computed in Reconnect. -1 until a reconnection attempt is made.
    float mPrevDist;
    bool mIsPrevVertexDiffuse;

    __init(const RandomSampler rng) {
		mContribution = 1;
        mPdfW = 1;
        mLength = 1;
        mDiffuseBounces = 0;
        mTerminated = false;

        mShiftData = ShiftData(rng);
        mRcJacobian = -1;
        mPrevDist = 0;
        mIsPrevVertexDiffuse = false;
	}


    // Called when a reconnectable vertex is found
	// must be called BEFORE mContribution and mPdfW are modified with directionSample's reflectance and pdf
	[mutating]
    void SetReconnectionVertex(const SceneParameters scene, const PathVertex vertex, const DirectionSampleRecord directionSample) {
		mShiftData.mReconnectionVertex.mVertex.mLocalPosition = scene.mInstanceInverseTransforms[vertex.mInstanceIndex].transformPoint(vertex.mShadingData.mPosition);
        mShiftData.mReconnectionVertex.mVertex.mInstancePrimitiveIndex = vertex.mInstancePrimitiveIndex;
        mShiftData.mReconnectionVertex.mLocalDirOut = directionSample.mDirection;
        mShiftData.mReconnectionVertex.mRadiance = mContribution * directionSample.mReflectance; // temporarily stored here, finalized in Terminate()
        mShiftData.mPrefixLength = mLength;
        mShiftData.mSuffixDiffuseBounces = mDiffuseBounces; // temporarily stored here, finalized in Terminate()
        mShiftData.mPrefixPdfW = mPdfW;
        mShiftData.mReconnectionDist = mPrevDist;
        mShiftData.mReconnectionCos = abs(vertex.mLocalDirIn.z);
	}

	[mutating]
    void Terminate(const float3 emission) {
        mContribution *= emission;
        // finalize the reconnection vertex
        if (mShiftData.HasReconnectionVertex()) {
            // divide out the path contribution before/at the reconnection vertex
            mShiftData.mReconnectionVertex.mRadiance.r = safe_divide(mContribution.r, mShiftData.mReconnectionVertex.mRadiance.r);
            mShiftData.mReconnectionVertex.mRadiance.g = safe_divide(mContribution.g, mShiftData.mReconnectionVertex.mRadiance.g);
            mShiftData.mReconnectionVertex.mRadiance.b = safe_divide(mContribution.b, mShiftData.mReconnectionVertex.mRadiance.b);
            mShiftData.mSuffixDiffuseBounces = mDiffuseBounces - mShiftData.mSuffixDiffuseBounces;
        } else
            mShiftData.mPrefixPdfW = mPdfW;
		mShiftData.mPathLength = mLength;
        mTerminated = true;
	}
};

extension PathTracer {
    void Reconnect(inout Path p, const PathVertex vertex, const Interaction bsdf, inout RandomSampler rng, const ShiftData basePath) {
        if (p.mLength + 1 < basePath.mPrefixLength) {
			// need more bounces before we can reconnect. don't terminate yet.
			return;
		}

        if (p.mLength + 1 > basePath.mPrefixLength || p.mDiffuseBounces + basePath.mSuffixDiffuseBounces > gPushConstants.mMaxDiffuseBounces) {
			// path too long - can't reconnect
            p.Terminate(0);
            p.mRcJacobian = 0;
            return;
        }

        DebugPrintf(sPixelIndex, "\tAttempting to connect vertex %u to base path vertex %u/%u\n", p.mLength + 1, basePath.mPrefixLength + 1, basePath.mPathLength + 1);

        // terminate path into rcv vertex
		p.mLength = basePath.mPathLength;
		p.mDiffuseBounces += basePath.mSuffixDiffuseBounces;
        p.mTerminated = true;

        const ShadingData sd_rcv = basePath.mReconnectionVertex.mVertex.getShadingData(mScene);
        Vector3 toRcv = sd_rcv.mPosition - vertex.mShadingData.mPosition;
        const float dist2 = dot(toRcv, toRcv);
        const float dist = sqrt(dist2);
        toRcv /= dist;

        DebugPrintf(sPixelIndex, "\tConnecting to %0.3f %0.3f %0.3f\n", sd_rcv.mPosition.x, sd_rcv.mPosition.y, sd_rcv.mPosition.z);

        // evaluate bsdf at vertex
        {
			const Vector3 localDirOut = vertex.mShadingData.isSurface() ? vertex.mShadingData.toLocal(toRcv) : toRcv;
			p.mContribution *= bsdf.evaluateReflectance<false>(vertex.mLocalDirIn, localDirOut).mReflectance;
			if (vertex.mShadingData.isSurface())
				p.mContribution *= vertex.mShadingData.shadingNormalCorrection<false>(vertex.mLocalDirIn, localDirOut);
        }

		// evaluate bsdf or emission at rcv
		float cos_rcv = 1;
		if (gHasMedia && sd_rcv.isMedium()) {
			const Medium bsdf_rcv = Medium(mScene, sd_rcv.getMaterialAddress());
			const float3 le = bsdf_rcv.emission();
            if (any(le > 0)) {
                p.mContribution *= le;
            } else {
				p.mContribution *= bsdf_rcv.evaluateReflectance<false>(-toRcv, basePath.mReconnectionVertex.mLocalDirOut).mReflectance;
                p.mContribution *= basePath.mReconnectionVertex.mRadiance;
			}
		} else {
            const PackedMaterialData bsdf_rcv = mScene.LoadMaterial(sd_rcv);
            const Vector3 localDirIn_rcv = sd_rcv.toLocal(-toRcv);
			cos_rcv = abs(localDirIn_rcv.z);
			const Spectrum le = bsdf_rcv.emission();
            if (any(le > 0)) {
                p.mContribution *= le;
			} else {
                p.mContribution *= bsdf_rcv.evaluateReflectance<false>(localDirIn_rcv, basePath.mReconnectionVertex.mLocalDirOut).mReflectance;
                p.mContribution *= vertex.mShadingData.shadingNormalCorrection<false>(localDirIn_rcv, basePath.mReconnectionVertex.mLocalDirOut);
                p.mContribution *= basePath.mReconnectionVertex.mRadiance;
			}
        }

        p.mRcJacobian = safe_divide(cos_rcv, basePath.mReconnectionCos) * safe_divide(pow2(basePath.mReconnectionDist), dist2);

        DebugPrintf(sPixelIndex, "\tReconnection jacobian: %f\n", p.mRcJacobian);

		// trace visibility ray
        {
            float transDirPdf, transNeePdf;
            mScene.traceVisibilityRay(makeRay(rayOffset(vertex.mShadingData, toRcv), toRcv, 0, AdjustIntersectionDistance(dist)), rng, vertex.mCurrentMedium, p.mContribution, transDirPdf, transNeePdf);
            p.mContribution *= safe_divide(1, transNeePdf);
            if (transNeePdf <= 0) {
                p.mRcJacobian = 0;
                DebugPrintf(sPixelIndex, "\tBlocked!\n");
            }
        }
    }

    // Evaluates emission and samples a direction. Returns true when
    // a light path is found (when 'vertex' is emissive, or when vertex is connected to rcv)
    Vector3 IntegrateVertex(inout Path p, const PathVertex vertex, const Interaction bsdf, inout RandomSampler rng, const Optional<ShiftData> basePath = none) {
        const bool isCanonical = !basePath.hasValue;

        // store albedo at the first vertex
        if (isCanonical && p.mLength == 1)
			mFramebuffer.mAlbedo[sPixelIndex] = float4(bsdf.albedo(), 1);

        const bool isMedium = gHasMedia && vertex.mShadingData.isMedium();
        const bool isDiffuse = !bsdf.isSingular();

        // evaluate emission
        if (isMedium || vertex.mLocalDirIn.z > 0) {
            const Spectrum le = bsdf.emission();
            if (any(le > 0)) {
				/*
				#ifdef gUseReconnection
                // store reconnection vertex if a reconnection vertex hasn't been found yet
                if (isCanonical && p.mIsPrevVertexDiffuse && !p.ReconnectionVertexFound())
                    p.SetReconnectionVertex(mScene, vertex, 0);
				#endif
				*/
                p.Terminate(le);
                return 0;
			}
		}

        // terminate if too long, or bsdf is invalid
        if (p.mLength + 1 > gPushConstants.mMaxDepth || (isDiffuse && p.mDiffuseBounces + 1 > gPushConstants.mMaxDiffuseBounces) || !bsdf.canEvaluate()) {
            p.Terminate(0);
            return 0;
        }

        // sample direction
        DirectionSampleRecord s = bsdf.sampleDirection<false>(rng.nextFloat().xyz, vertex.mLocalDirIn);
        if (!isMedium)
            s.mReflectance *= vertex.mShadingData.shadingNormalCorrection<false>(vertex.mLocalDirIn, s.mDirection);

        if (all(s.mReflectance <= 0) || s.mFwdPdfW <= 0) {
            p.Terminate(0);
            return 0;
        }

        if (isDiffuse)
            p.mDiffuseBounces++;

		#ifdef gUseReconnection
        if (isCanonical && isDiffuse) {
            DebugPrintf(sPixelIndex, "\tDiffuse vertex %u at %0.3f %0.3f %0.3f\n", p.mLength+1, vertex.mShadingData.mPosition.x, vertex.mShadingData.mPosition.y, vertex.mShadingData.mPosition.z);
        }

        // attempt to connect to basePath
        if (!isCanonical && isDiffuse && basePath.value.HasReconnectionVertex()) {
            Reconnect(p, vertex, bsdf, rng, basePath.value);
			if (p.mTerminated)
            	return 0;
		}

		// store last reconnectable vertex
        if (isCanonical && p.mIsPrevVertexDiffuse && isDiffuse) {
            p.SetReconnectionVertex(mScene, vertex, s);
            DebugPrintf(sPixelIndex, "\tStoring rcv at %u\n", p.mLength+1);
        }
		#endif

		// modify throughput and pdf, return sampled direction

		p.mContribution *= s.mReflectance;
		p.mPdfW         *= s.mFwdPdfW;
        p.mIsPrevVertexDiffuse = isDiffuse;

        return isMedium ? s.mDirection : vertex.mShadingData.toWorld(s.mDirection);
    }

	PathVertex TraceNext(PathVertex vertex, inout RandomSampler rng, const float3 direction, inout float3 throughput, inout float pdf, out float dist) {
		uint medium = vertex.mCurrentMedium;

		IntersectionResult isect;
        float transDirPdf, transNeePdf;
        mScene.traceScatteringRay(makeRay(rayOffset(vertex.mShadingData, direction), direction), rng, medium, throughput, transDirPdf, transNeePdf, isect);
		isect.mShadingData.mTexcoordScreenSize = 0; // disable mipmaps for now

        pdf *= transDirPdf;

        vertex = PathVertex(mScene, isect.mShadingData, isect.mInstancePrimitiveIndex, medium, -direction);

        if (vertex.mShadingData.isEnvironment())
            dist = 1;
        else
            dist = isect.mDistance;

		return vertex;
	}

    // Generates a path with unidirectional path tracing. Attempts to reconnect to basePath is specified.
    Path GeneratePath(PathVertex vertex, inout RandomSampler rng, const Optional<ShiftData> basePath = none) {
        DebugPrintf(sPixelIndex, "Generating path\n\tat %0.3f %0.3f %0.3f\n\twith seed %u %u %u %u\n",
			vertex.mShadingData.mPosition.x, vertex.mShadingData.mPosition.y, vertex.mShadingData.mPosition.z,
			rng.mState.x, rng.mState.y, rng.mState.z, rng.mState.w );

        Path p = Path(rng);

        while (!p.mTerminated) {
			// Integrate current vertex
            Vector3 dir;
            if (gHasMedia && vertex.mShadingData.isMedium()) {
                const Medium m = Medium(mScene, vertex.mShadingData.getMaterialAddress());
                dir = IntegrateVertex(p, vertex, m, rng, basePath);
            } else if (vertex.mShadingData.isSurface()) {
                const PackedMaterialData m = mScene.LoadMaterial(vertex.mShadingData);
                dir = IntegrateVertex(p, vertex, m, rng, basePath);
            } else /*if (vertex.mShadingData.isEnvironment())*/ {
                float envPdfW;
                p.Terminate(mScene.EvaluateEnvironment(-vertex.mLocalDirIn, envPdfW));
            }

            // Trace ray to find next vertex
            if (!p.mTerminated) {
				vertex = TraceNext(vertex, rng, dir, p.mContribution, p.mPdfW, p.mPrevDist);
                p.mLength++;
            }
		}

		#ifdef gCountRays
        if (!basePath.hasValue) {
			InterlockedAdd(mFramebuffer.mDebugCounters[4], 1);
			if (p.mShiftData.HasReconnectionVertex())
                InterlockedAdd(mFramebuffer.mDebugCounters[3], 1);
        }
		#endif

        if (basePath.hasValue)
            p.mShiftData = basePath.value;

		return p;
    }
    Path GenerateShiftedPath(const PathVertex vertex, const ShiftData basePath, out float jacobian, out bool valid) {
		jacobian = 0;
		valid = false;

        RandomSampler rng = { basePath.mRngSeed };
        const Path p = GeneratePath(vertex, rng, basePath);

        if (p.mLength == basePath.mPathLength) {
            valid = true;
            jacobian = safe_divide(basePath.mPrefixPdfW, p.mPdfW);
			if (p.mRcJacobian != -1)
				jacobian *= p.mRcJacobian;
        }

		#ifdef gCountRays
        InterlockedAdd(mFramebuffer.mDebugCounters[0], 1);
        if (valid) {
            InterlockedAdd(mFramebuffer.mDebugCounters[1], 1);
			if (p.mRcJacobian > 0)
            	InterlockedAdd(mFramebuffer.mDebugCounters[2], 1);
        }
		#endif

        return p;
	}

	// Standard path tracing estimator
	Spectrum EstimateRadiance(const PathVertex vertex, inout RandomSampler rng) {
		const Path p = GeneratePath(vertex, rng);
		return p.mContribution * safe_divide(1, p.mPdfW);
	}


	struct PathReservoirData {
		PackedVertex mBaseVertex;
		ShiftData mShiftData;
        uint mPackedLocalDirIn;
		float W;
        float M;

		PathVertex GetBaseVertex(const SceneParameters scene) {
			return PathVertex(scene, mBaseVertex.getShadingData(scene), mBaseVertex.mInstancePrimitiveIndex, -1, mPackedLocalDirIn);
		}
    };
    PathReservoirData LoadReservoir(const uint2 pixelIndex) {
        PathReservoirData r;
        r.mBaseVertex = reinterpret<PackedVertex>(mFramebuffer.mPrevReservoirDataGI[0][pixelIndex]);
        r.mShiftData.mRngSeed = reinterpret<uint4>(mFramebuffer.mPrevReservoirDataGI[1][pixelIndex]);

        r.mShiftData.mPackedData = reinterpret<uint4>(mFramebuffer.mPrevReservoirDataGI[2][pixelIndex]);

        const float4 data3 = mFramebuffer.mPrevReservoirDataGI[3][pixelIndex];
        r.W = data3.x;
        r.M = data3.y;
        r.mPackedLocalDirIn = asuint(data3.z);

		#ifdef gUseReconnection
        r.mShiftData.mReconnectionVertex = LoadPrevReconnectionVertex(mFramebuffer, pixelIndex);
		#endif

		return r;
	}
    void StoreReservoir(const Path p, const float W, const float M, const PathVertex vertex) {
        PackedVertex baseVertex;
		baseVertex.mLocalPosition = mScene.mInstanceInverseTransforms[vertex.mInstanceIndex].transformPoint(vertex.mShadingData.mPosition);
        baseVertex.mInstancePrimitiveIndex = vertex.mInstancePrimitiveIndex;

        mFramebuffer.mReservoirDataGI[0][sPixelIndex] = reinterpret<float4>(baseVertex);
        mFramebuffer.mReservoirDataGI[1][sPixelIndex] = reinterpret<float4>(p.mShiftData.mRngSeed);
        mFramebuffer.mReservoirDataGI[2][sPixelIndex] = reinterpret<float4>(p.mShiftData.mPackedData);
        mFramebuffer.mReservoirDataGI[3][sPixelIndex] = float4(W, M, asfloat(vertex.mPackedLocalDirIn), 0);
        #ifdef gUseReconnection
		StoreReconnectionVertex(mFramebuffer, sPixelIndex, p.mShiftData.mReconnectionVertex);
		#endif
    }


	// shift estimator, simply averages shifted paths from neighbors
    float3 EstimateRadianceShift(const PathVertex vertex, inout RandomSampler rng) {
        DebugPrintf(sPixelIndex, "Estimating radiance...\n");
        // sample canonical path
        Path p = GeneratePath(vertex, rng);
        float W = safe_divide(1, p.mPdfW);

        DebugPrintf(sPixelIndex, "Base path     : %f / %f (prefix length: %u/%u)\n", p.mContribution.r, p.mShiftData.mPrefixPdfW, p.mShiftData.mPrefixLength, p.mLength);
        if (IsDebugPixel(sPixelIndex) && any(p.mContribution > 0)) {
            printf("\tPrefix length: %u. Has rcv: %u\n", p.mShiftData.mPrefixLength, (uint)p.mShiftData.HasReconnectionVertex());
            // Test the identity shift
            float jacobian;
            bool valid;
            const Path pi = GenerateShiftedPath(vertex, p.mShiftData, jacobian, valid);
            if (!valid)
                printf("Shift failed (%u != %u)\n", p.mLength, pi.mLength);
            else {
				printf("Offset path   : %f / %f\n", pi.mContribution.r, pi.mPdfW);
				printf("Shift jacobian: %f %f\n", jacobian, pi.mRcJacobian);
            }
        }

        #ifndef gReSTIR_GI_Reuse
        return p.mContribution * W;
        #endif

		Spectrum f = p.mContribution * W;

        // store the sampled path
        StoreReservoir(p, W, 1, vertex);

        if (gPushConstants.mReservoirHistoryValid != 0 && p.mLength > 1) {
            const float2 prevPixel = mFramebuffer.mPrevUVs[sPixelIndex] * gPushConstants.mOutputExtent;

            float norm = 1;

			for (uint i = 0; i < gPushConstants.mGIReuseSamples; i++) {
                const float2 discRng = rng.nextFloat().xy;
                const int2 candidatePixel = int2(prevPixel + sampleConcentricDisc(discRng.x, discRng.y) * gPushConstants.mGIReuseRadius);
                if (any(candidatePixel < 0) || any(candidatePixel >= gPushConstants.mOutputExtent))
                    continue;

                const PathReservoirData data = LoadReservoir(candidatePixel);

                if (data.M <= 0 || data.mShiftData.mPathLength == 1) // invalid sample
                    continue;

                // shift candidatePixel's path to vertex
                float jacobian_from_i;
                bool valid;
                const Path xi = GenerateShiftedPath(vertex, data.mShiftData, jacobian_from_i, valid);
                if (!valid)
                    continue;

                // add radiance from shifted path
                f += xi.mContribution * data.W * jacobian_from_i;

                norm += 1;
            }

            // average canonical and shifted samples
            f *= safe_divide(1,norm);
		}

		return f;
	}

	// ReSTIR estimator, resamples shifted paths from neighbors
	float3 EstimateRadianceReSTIR(const PathVertex vertex, inout RandomSampler rng) {
		// sample canonical paths
        Path p = GeneratePath(vertex, rng);
        float W = safe_divide(1, p.mPdfW);
        if (gPushConstants.mGICandidateSamples > 1) {
            float wsum = luminance(p.mContribution) * W;
            for (uint i = 1; i < gPushConstants.mGICandidateSamples; i++) {
                const Path pi = GeneratePath(vertex, rng);
                const float wi = safe_divide(luminance(pi.mContribution), pi.mPdfW);
                wsum += wi;
                if (wsum * rng.nextFloat().x <= wi)
                    p = pi;
            }
            W = safe_divide(wsum, luminance(p.mContribution) * gPushConstants.mGICandidateSamples);
        }

        float M = max(1, gPushConstants.mGICandidateSamples);

		#ifndef gReSTIR_GI_Reuse
		if (false)
		#endif
        {
			if (gPushConstants.mReservoirHistoryValid != 0 && p.mLength > 1) {
				const float2 prevPixel = mFramebuffer.mPrevUVs[sPixelIndex] * gPushConstants.mOutputExtent;
				const RandomSampler spatialRng = rng;
				rng.skipNext(gPushConstants.mGIReuseSamples);

				float wsum = M * luminance(p.mContribution) * W;

				uint sampleMask = 0;
				uint selectedIndex = -1;
				float newM = M;

				for (uint i = 0; i < gPushConstants.mGIReuseSamples; i++) {
					float2 discRng;
					{
						RandomSampler tmp_rng = spatialRng;
						tmp_rng.skipNext(i);
						discRng = tmp_rng.nextFloat().xy;
					}
					const int2 candidatePixel = int2(prevPixel + sampleConcentricDisc(discRng.x, discRng.y) * gPushConstants.mGIReuseRadius);
					if (any(candidatePixel < 0) || any(candidatePixel >= gPushConstants.mOutputExtent))
						continue;

					const PathReservoirData data = LoadReservoir(candidatePixel);

					if (data.M <= 0)
						continue;

					// shift candidatePixel's path to vertex
					float jacobian_from_i;
					bool valid;
					const Path xi = GenerateShiftedPath(vertex, data.mShiftData, jacobian_from_i, valid);
					if (!valid)
						continue;

					SET_BIT(sampleMask, i);

					// RIS weight
					const float wi = data.M * luminance(xi.mContribution) * data.W * jacobian_from_i;

					wsum += wi;
					if (rng.nextFloat().x * wsum <= wi) {
						p = xi;
						selectedIndex = i;
					}

					newM += data.M;
				}

				W = safe_divide(wsum, luminance(p.mContribution));

				// compute resampling MIS weight
				float Z = M;
				M = newM;

				for (uint i = 0; i < gPushConstants.mGIReuseSamples; i++) {
					if ((sampleMask & BIT(i)) == 0) continue;

					float2 discRng;
					{
						RandomSampler tmp_rng = spatialRng;
						tmp_rng.skipNext(i);
						discRng = tmp_rng.nextFloat().xy;
					}
					const int2 candidatePixel = int2(prevPixel + sampleConcentricDisc(discRng.x, discRng.y) * gPushConstants.mGIReuseRadius);

					const PathReservoirData data = LoadReservoir(candidatePixel);

					if (selectedIndex == i) {
						Z += data.M;
						continue;
					}

					// shift p to candidatePixel's vertex
                    float jacobian_to_i;
                    bool valid;
					const Path p_at_i = GenerateShiftedPath(data.GetBaseVertex(mScene), p.mShiftData, jacobian_to_i, valid);
					if (valid)
						Z += data.M;
				}

				W = safe_divide(W, Z);
			}

			if (any(p.mContribution > 0))
				StoreReservoir(p, W, min(M, gPushConstants.mGICandidateSamples * gPushConstants.mGIMaxM), vertex);
        }

		return p.mContribution * W;
	}
}