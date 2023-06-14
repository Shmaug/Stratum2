#pragma once

#include "pt.slang"

#ifndef gGIMis
#define gGIMis ReuseMISType::eSimple
#endif

#define gReconnectionDistance 0.2
#define gReconnectionCos 0.01

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
    float mRcJacobian;    // reconnection jacobian, computed in Reconnect. -1 until a reconnection attempt is made.
    float mPrevPdfW;
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
        mPrevPdfW = 0;
        mPrevDist = 0;
        mIsPrevVertexDiffuse = false;
	}


    // Called when a reconnectable vertex is found
	// must be called BEFORE mContribution and mPdfW are modified with directionSample's reflectance and pdf
	[mutating]
    void SetReconnectionVertex(const SceneParameters scene, const PathVertex vertex, const Optional<DirectionSampleRecord> directionSample = none) {
		mShiftData.mReconnectionVertex.mVertex.mLocalPosition = scene.mInstanceInverseTransforms[vertex.mInstanceIndex].transformPoint(vertex.mShadingData.mPosition);
        mShiftData.mReconnectionVertex.mVertex.mInstancePrimitiveIndex = vertex.mInstancePrimitiveIndex;
        mShiftData.mReconnectionVertex.mLocalDirOut = directionSample.hasValue ? directionSample.value.mDirection : 0;
        mShiftData.mReconnectionVertex.mRadiance = mContribution * (directionSample.hasValue ? directionSample.value.mReflectance : 1); // temporarily stored here, finalized in Terminate()
        mShiftData.mPrefixLength = mLength;
        mShiftData.mSuffixDiffuseBounces = mDiffuseBounces; // temporarily stored here, finalized in Terminate()
        mShiftData.mPrefixPdfW = mPrevPdfW;
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

T madfrac<T>(T a, T b) { return a * b - floor(a * b); }

float3 fib_to_dir(float i, float n) {
    float phi_i = i * 1.6180339887498948482045868343656;
    float phi = 2.0 * M_PI * (phi_i - floor(phi_i));
    float cosTheta = 1.0 - 2.0 * (i + 0.5) / n;
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    return float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

float RISTargetFunction(const Path p) { return luminance(p.mContribution); }

extension PathTracer {
    void Reconnect(inout Path p, const PathVertex vertex, const Interaction bsdf, inout RandomSampler rng, const ShiftData basePath) {
        if (p.mLength + 1 < basePath.mPrefixLength) {
			// need more bounces before we can reconnect. don't terminate yet.
			return;
		}

        DebugPrintf(sPixelIndex, "\tAttempting to connect vertex %u to base path vertex %u/%u\n", p.mLength + 1, basePath.mPrefixLength + 1, basePath.mPathLength + 1);

        if (p.mLength + 1 > basePath.mPrefixLength || p.mDiffuseBounces + basePath.mSuffixDiffuseBounces > gPushConstants.mMaxDiffuseBounces) {
			// path too long - can't reconnect
            p.Terminate(0);
            p.mRcJacobian = 0;
            DebugPrintf(sPixelIndex, "\t\tReconnection failed: path too long\n");
            return;
        }

        // terminate path into rcv vertex
		p.mLength = basePath.mPathLength;
		p.mDiffuseBounces += basePath.mSuffixDiffuseBounces;
        p.mTerminated = true;

        const ShadingData sd_rcv = basePath.mReconnectionVertex.mVertex.getShadingData(mScene);
        Vector3 toRcv = sd_rcv.mPosition - vertex.mShadingData.mPosition;
        const float dist2 = dot(toRcv, toRcv);
        const float dist = sqrt(dist2);
        toRcv /= dist;

        DebugPrintf(sPixelIndex, "\t\tConnecting to %0.3f %0.3f %0.3f from %0.3f %0.3f %0.3f\n",
				sd_rcv.mPosition.x, sd_rcv.mPosition.y, sd_rcv.mPosition.z,
				vertex.mShadingData.mPosition.x, vertex.mShadingData.mPosition.y, vertex.mShadingData.mPosition.z);

        if (dist < gReconnectionDistance) {
            p.Terminate(0);
			p.mRcJacobian = 0;
            DebugPrintf(sPixelIndex, "\t\tReconnection failed: dist < gReconnectionDistance\n");
            return;
        }

        // evaluate bsdf at vertex
        {
            const Vector3 localDirOut = vertex.mShadingData.isSurface() ? vertex.mShadingData.toLocal(toRcv) : toRcv;
            const ReflectanceEvalRecord s = bsdf.evaluateReflectance<false>(vertex.mLocalDirIn, localDirOut);
            p.mContribution *= s.mReflectance;
            if (vertex.mShadingData.isSurface())
                p.mContribution *= vertex.mShadingData.shadingNormalCorrection<false>(vertex.mLocalDirIn, localDirOut);
            if (s.mFwdPdfW <= 0) {
                p.Terminate(0);
                p.mRcJacobian = 0;
                DebugPrintf(sPixelIndex, "\t\tReconnection failed: BSDF sampling not possible at rcv-1\n");
                return;
			}
        }

		// evaluate bsdf or emission at rcv
		float cos_rcv = 1;
		if (gHasMedia && sd_rcv.isMedium()) {
			const Medium bsdf_rcv = Medium(mScene, sd_rcv.getMaterialAddress());
			const float3 le = bsdf_rcv.emission();
            if (any(le > 0)) {
                p.mContribution *= le;
            } else {
                const ReflectanceEvalRecord s = bsdf_rcv.evaluateReflectance<false>(-toRcv, basePath.mReconnectionVertex.mLocalDirOut);
                p.mContribution *= s.mReflectance;
                p.mContribution *= basePath.mReconnectionVertex.mRadiance;
                if (s.mFwdPdfW <= 0) {
                    p.Terminate(0);
                    p.mRcJacobian = 0;
                    DebugPrintf(sPixelIndex, "\t\tReconnection failed: BSDF sampling not possible at rcv\n");
                    return;
                }
			}
		} else { // surface
            const PackedMaterialData bsdf_rcv = mScene.LoadMaterial(sd_rcv);
            const Vector3 localDirIn_rcv = sd_rcv.toLocal(-toRcv);
			cos_rcv = abs(localDirIn_rcv.z);
			const Spectrum le = bsdf_rcv.emission();
            if (any(le > 0)) {
                p.mContribution *= le;
            } else {
                const ReflectanceEvalRecord s = bsdf_rcv.evaluateReflectance<false>(localDirIn_rcv, basePath.mReconnectionVertex.mLocalDirOut);
                p.mContribution *= s.mReflectance * vertex.mShadingData.shadingNormalCorrection<false>(localDirIn_rcv, basePath.mReconnectionVertex.mLocalDirOut);
                p.mContribution *= basePath.mReconnectionVertex.mRadiance;
                if (s.mFwdPdfW <= 0) {
                    p.Terminate(0);
                    p.mRcJacobian = 0;
                    DebugPrintf(sPixelIndex, "\t\tReconnection failed: BSDF sampling not possible at rcv\n");
                    return;
				}
			}
        }

        if (cos_rcv < gReconnectionCos) {
            p.Terminate(0);
            p.mRcJacobian = 0;
            DebugPrintf(sPixelIndex, "\t\tReconnection failed: cos_rcv < gReconnectionCos\n");
            return;
		}

        p.mRcJacobian = safe_divide(cos_rcv, basePath.mReconnectionCos) * pow2(safe_divide(basePath.mReconnectionDist, dist));

		// trace visibility ray
        {
            float transDirPdf, transNeePdf;
            mScene.traceVisibilityRay(makeRay(rayOffset(vertex.mShadingData, toRcv), toRcv, 0, AdjustIntersectionDistance(dist)), rng, vertex.mCurrentMedium, p.mContribution, transDirPdf, transNeePdf);
            p.mContribution *= safe_divide(1, transNeePdf);
            if (transNeePdf <= 0) {
                p.Terminate(0);
                p.mRcJacobian = 0;
                DebugPrintf(sPixelIndex, "\t\tReconnection failed: Occluded\n");
                return;
            }
        }

        DebugPrintf(sPixelIndex, "\tReconnection jacobian: %f\n", p.mRcJacobian);
        DebugPrintf(sPixelIndex, "\tPath reconnected. Contribution: %f\n", luminance(p.mContribution));
    }

    // Evaluates emission, performs reconnection, samples the BSDF, and returns the sampled direction
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
				#ifdef gUseReconnection
                // store reconnection vertex if a reconnection vertex hasn't been found yet
                if (isCanonical && p.mIsPrevVertexDiffuse && !p.mShiftData.HasReconnectionVertex() && p.mPrevDist >= gReconnectionDistance && abs(vertex.mLocalDirIn.z) >= gReconnectionCos) {
                    p.SetReconnectionVertex(mScene, vertex);
                }
				#endif
                p.Terminate(le);
        		DebugPrintf(sPixelIndex, "\tPath terminated. Contribution: %f | PDF: %f\n", luminance(p.mContribution), p.mPdfW);
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
        // attempt to connect to basePath
        if (!isCanonical && isDiffuse && basePath.value.HasReconnectionVertex()) {
            Reconnect(p, vertex, bsdf, rng, basePath.value);
			if (p.mTerminated)
            	return 0;
		}

        // store first reconnectable vertex
        if (isCanonical && p.mIsPrevVertexDiffuse && isDiffuse && !p.mShiftData.HasReconnectionVertex() && p.mPrevDist >= gReconnectionDistance && abs(vertex.mLocalDirIn.z) >= gReconnectionCos) {
            p.SetReconnectionVertex(mScene, vertex, s);
            DebugPrintf(sPixelIndex, "\t\tStoring rcv at %u (prefix pdf: %f)\n", p.mLength+1, p.mShiftData.mPrefixPdfW);
        }
		#endif

		// modify throughput and pdf, return sampled direction

        p.mPrevPdfW = p.mPdfW;
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
        DebugPrintf(sPixelIndex, "Generating path with seed %u %u %u %u\n", rng.mState.x, rng.mState.y, rng.mState.z, rng.mState.w);
		DebugPrintf(sPixelIndex, "\tat %0.3f %0.3f %0.3f\n", vertex.mShadingData.mPosition.x, vertex.mShadingData.mPosition.y, vertex.mShadingData.mPosition.z);

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

        if (basePath.hasValue)
            p.mShiftData = basePath.value;

		#ifdef gCountRays
        if (!basePath.hasValue) {
			InterlockedAdd(mFramebuffer.mDebugCounters[4], 1);
			if (p.mShiftData.HasReconnectionVertex())
                InterlockedAdd(mFramebuffer.mDebugCounters[3], 1);
        }
		#endif

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
            if (p.mRcJacobian != -1) {
                valid = (p.mRcJacobian > 0);
                jacobian *= p.mRcJacobian;
            }
            DebugPrintf(sPixelIndex, "Shift jacobian: %f | Rc jacobian: %f\n", jacobian, p.mRcJacobian);
        } else {
            DebugPrintf(sPixelIndex, "Shift failed: shifted %u != base %u\n", p.mLength, basePath.mPathLength);
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

	// ReSTIR estimator, resamples shifted paths from neighbors
    float3 EstimateRadianceReSTIR(const PathVertex vertex, inout RandomSampler rng) {
        DebugPrintf(sPixelIndex, "\n\n\n");
        DebugPrintf(sPixelIndex, "Sampling canonical paths...\n");

        // sample canonical paths
        Path p = GeneratePath(vertex, rng);
        float W = safe_divide(1, p.mPdfW);
        float M = 1;
        if (gPushConstants.mGICandidateSamples > 1) {
            float wsum = RISTargetFunction(p) * W;
            for (uint i = 1; i < gPushConstants.mGICandidateSamples; i++) {
                const Path pi = GeneratePath(vertex, rng);
                const float wi = safe_divide(RISTargetFunction(pi), pi.mPdfW);
                wsum += wi;
                if (wsum * rng.nextFloat().x <= wi)
                    p = pi;
            }
            M = gPushConstants.mGICandidateSamples;
            W = safe_divide(wsum, RISTargetFunction(p) * M);
        }

		#ifndef gReSTIR_GI_Reuse
		return p.mContribution * W;
		#endif

        if (RISTargetFunction(p) > 0 && p.mLength > 1) {
            if (gPushConstants.mReservoirHistoryValid != 0) {
                DebugPrintf(sPixelIndex, "Performing spatial reuse\n");
                DebugPrintf(sPixelIndex, "Base path contribution: %f\n", luminance(p.mContribution));
                const float2 prevPixel = sPixelIndex + 0.5;//mFramebuffer.mPrevUVs[sPixelIndex] * gPushConstants.mOutputExtent;
				const RandomSampler spatialRng = rng;
				rng.skipNext(gPushConstants.mGIReuseSamples);

				float wsum = M * RISTargetFunction(p) * W;
				float Z = M;

				for (uint i = 0; i < gPushConstants.mGIReuseSamples; i++) {
					int2 offset;
					{
						RandomSampler tmp_rng = spatialRng;
                        tmp_rng.skipNext(i);
                        const float2 discRng = tmp_rng.nextFloat().xy;
                        offset = int2(sampleConcentricDisc(discRng.x, discRng.y) * gPushConstants.mGIReuseRadius + 0.5);
					}
					const int2 candidatePixel = int2(prevPixel) + offset;
                    if (any(candidatePixel < 0) || any(candidatePixel >= gPushConstants.mOutputExtent))
                        continue;

					const PathReservoirData data = LoadReservoir(candidatePixel);

					if (data.M <= 0)
						continue;

                    DebugPrintf(sPixelIndex, "\tReusing pixel at %d, %d...\n", offset.x, offset.y);

					// shift candidatePixel's path to vertex
					float jacobian;
					bool valid;
                    const Path xi = GenerateShiftedPath(vertex, data.mShiftData, jacobian, valid);
                    if (!valid) {
                        DebugPrintf(sPixelIndex, "\t\tShift failed.\n");
                        continue;
                    }

                    DebugPrintf(sPixelIndex, "\t\tShift succeeded. Contribution: %f | Jacobian: %f\n", luminance(xi.mContribution), jacobian);

                    M += data.M;

					// RIS weight
					const float wi = data.M * RISTargetFunction(xi) * data.W * jacobian;

					wsum += wi;
                    if (rng.nextFloat().x * wsum <= wi) {
                        p = xi;
                    }
                }

                DebugPrintf(sPixelIndex, "\t\tSelected path with contribution %f and seed %u %u %u %u\n", luminance(p.mContribution), p.mShiftData.mRngSeed.x, p.mShiftData.mRngSeed.y, p.mShiftData.mRngSeed.z, p.mShiftData.mRngSeed.w);
                DebugPrintf(sPixelIndex, "Computing Z\n");

                for (uint i = 0; i < gPushConstants.mGIReuseSamples; i++) {
                    int2 offset;
                    {
                        RandomSampler tmp_rng = spatialRng;
                        tmp_rng.skipNext(i);
                        const float2 discRng = tmp_rng.nextFloat().xy;
                        offset = int2(sampleConcentricDisc(discRng.x, discRng.y) * gPushConstants.mGIReuseRadius + 0.5);
                    }
                    const int2 candidatePixel = int2(prevPixel) + offset;
					if (any(candidatePixel < 0) || any(candidatePixel >= gPushConstants.mOutputExtent))
						continue;

					const PathReservoirData data = LoadReservoir(candidatePixel);

                    DebugPrintf(sPixelIndex, "\tShifting path to pixel at %d, %d\n", offset.x, offset.y);

					// shift p to candidatePixel's vertex
					float jacobian;
					bool valid;
                    const Path p_at_i = GenerateShiftedPath(data.GetBaseVertex(mScene), p.mShiftData, jacobian, valid);
                    if (valid && any(p_at_i.mContribution > 0)) {
                        Z += data.M;
                        DebugPrintf(sPixelIndex, "\t\tShift succeeded. Jacobian: %f\n", jacobian);
                    } else {
                        DebugPrintf(sPixelIndex, "\t\tShift failed\n");
                    }
                }

                W = safe_divide(wsum, RISTargetFunction(p) * Z);
            }
            StoreReservoir(p, W, min(M, gPushConstants.mGIMaxM * max(gPushConstants.mGICandidateSamples, 1)), vertex);
        }

		return p.mContribution * W;
	}
}