#pragma once

#include "di.slang"

#define gMinReconnectionDistance 0.0

struct PathSuffix {
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
        get { return BF_GET(mPackedData[0],           16, 8); }
        set {        BF_SET(mPackedData[0], newValue, 16, 8); }
    };
	// specifies whether the last vertex was sampled with next event estimation. Always false if a reconnection vertex exists.
    property bool mReplayNee {
        get { return bool(BF_GET(mPackedData[0],                 24, 1)); }
        set {        bool(BF_SET(mPackedData[0], uint(newValue), 24, 1)); }
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

    PathSuffix mSuffix; // output shift data
    float mRcJacobian;  // reconnection jacobian, computed in Reconnect(). The value -1 indicates no reconnection was attempted.
    float mPrevPdfW;
    float mPrevDist;
    bool mIsPrevVertexDiffuse;

    __init(const RandomSampler rng) {
		mContribution = 1;
        mPdfW = 1;
        mLength = 1;
        mDiffuseBounces = 0;
        mTerminated = false;

        mSuffix = PathSuffix(rng);
        mRcJacobian = -1;
        mPrevPdfW = 0;
        mPrevDist = 0;
        mIsPrevVertexDiffuse = false;
	}


    // Called when a reconnectable vertex is found
	// must be called BEFORE mContribution and mPdfW are modified with directionSample's reflectance and pdf
	// when directionSample is none, the reconnection vertex is an emissive vertex
	// Note: mReconnectionVertex.mRadiance temporarily stored here, then finalized in Terminate()
	[mutating]
    void SetReconnectionVertex(const SceneParameters scene, const PathVertex vertex, const Optional<DirectionSampleRecord> directionSample = none) {
		mSuffix.mReconnectionVertex.mVertex.mLocalPosition = scene.mInstanceInverseTransforms[vertex.mInstanceIndex].transformPoint(vertex.mShadingData.mPosition);
        mSuffix.mReconnectionVertex.mVertex.mInstancePrimitiveIndex = vertex.mInstancePrimitiveIndex;
        if (directionSample.hasValue) {
            mSuffix.mReconnectionVertex.mLocalDirOut = directionSample.value.mDirection;
            mSuffix.mReconnectionVertex.mRadiance = mContribution * directionSample.value.mReflectance; // temporarily stored here, finalized in Terminate()
        } else {
            mSuffix.mReconnectionVertex.mLocalDirOut = 0;
            mSuffix.mReconnectionVertex.mRadiance = mContribution; // temporarily stored here, finalized in Terminate()
		}
        mSuffix.mPrefixLength = mLength;
        mSuffix.mSuffixDiffuseBounces = mDiffuseBounces; // temporarily stored here, finalized in Terminate()
        mSuffix.mPrefixPdfW = mPrevPdfW;
        mSuffix.mReconnectionDist = mPrevDist;
        mSuffix.mReconnectionCos = abs(vertex.mLocalDirIn.z);
        mSuffix.mReplayNee = false;
	}

	[mutating]
    void Terminate(const float3 emission, const bool isNee = false) {
        mContribution *= emission;
        // finalize the reconnection vertex
        if (mSuffix.HasReconnectionVertex()) {
            // divide out the path contribution before/at the reconnection vertex
            mSuffix.mReconnectionVertex.mRadiance.r = safe_divide(mContribution.r, mSuffix.mReconnectionVertex.mRadiance.r);
            mSuffix.mReconnectionVertex.mRadiance.g = safe_divide(mContribution.g, mSuffix.mReconnectionVertex.mRadiance.g);
            mSuffix.mReconnectionVertex.mRadiance.b = safe_divide(mContribution.b, mSuffix.mReconnectionVertex.mRadiance.b);
            mSuffix.mSuffixDiffuseBounces = mDiffuseBounces - mSuffix.mSuffixDiffuseBounces;
        } else {
            mSuffix.mPrefixPdfW = mPdfW;
            mSuffix.mReplayNee = isNee;
        }
		mSuffix.mPathLength = mLength;
        mTerminated = true;
	}
};

// path generation + basic path tracing estimator

extension PathTracer {
    void Reconnect(inout Path p, const PathVertex vertex, const Interaction bsdf, inout RandomSampler rng, const PathSuffix basePath) {
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

        // terminate path into rcv vertex
		p.mLength = basePath.mPathLength;
		p.mDiffuseBounces += basePath.mSuffixDiffuseBounces;
        p.mTerminated = true;

        const ShadingData sd_rcv = basePath.mReconnectionVertex.mVertex.getShadingData(mScene);
        Vector3 toRcv = sd_rcv.mPosition - vertex.mShadingData.mPosition;
        const float dist2 = dot(toRcv, toRcv);
        const float dist = sqrt(dist2);
        toRcv /= dist;

        if (dist < gMinReconnectionDistance) {
            p.Terminate(0);
			p.mRcJacobian = 0;
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
                    return;
				}
			}
        }

        p.mRcJacobian = safe_divide(cos_rcv, basePath.mReconnectionCos) * pow2(safe_divide(basePath.mReconnectionDist, dist));

        p.mSuffix = basePath;
        p.mSuffix.mPrefixPdfW = p.mPdfW;
        p.mSuffix.mReconnectionCos = cos_rcv;
        p.mSuffix.mReconnectionDist = dist;

        // trace visibility ray
        {
            float transDirPdf, transNeePdf;
            mScene.traceVisibilityRay(makeRay(rayOffset(vertex.mShadingData, toRcv), toRcv, 0, AdjustIntersectionDistance(dist)), rng, vertex.mCurrentMedium, p.mContribution, transDirPdf, transNeePdf);
            p.mContribution *= safe_divide(1, transNeePdf);
            if (transNeePdf <= 0) {
                p.Terminate(0);
                p.mRcJacobian = 0;
                return;
            }
        }
    }

    // Evaluates emission, performs reconnection, samples the BSDF, and returns the sampled direction
    Vector3 IntegrateVertex(inout Path p, const PathVertex vertex, const Interaction bsdf, inout RandomSampler rng, const Optional<PathSuffix> basePath = none) {
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
                if (isCanonical && p.mIsPrevVertexDiffuse && !p.mSuffix.HasReconnectionVertex() && p.mPrevDist >= gMinReconnectionDistance) {
                    p.SetReconnectionVertex(mScene, vertex);
                }
				#endif
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
        const DirectionSampleRecord s = bsdf.sampleDirection<false>(rng.nextFloat().xyz, vertex.mLocalDirIn);
        if (all(p.mContribution <= 0) || s.mFwdPdfW <= 0) {
            p.Terminate(0);
            return 0;
        }

		// connection strategies
        if (isDiffuse) {
			p.mDiffuseBounces++;

			// next event estimation
			#ifdef gNee
            const DirectIlluminationSample s = DirectIllumination(vertex, bsdf, rng);
			#endif

			#ifdef gUseReconnection
			// attempt to connect to basePath
			if (!isCanonical && basePath.value.HasReconnectionVertex()) {
				Reconnect(p, vertex, bsdf, rng, basePath.value);
				if (p.mTerminated)
					return 0;
			}

			// store first reconnectable vertex
			if (isCanonical && p.mIsPrevVertexDiffuse && !p.mSuffix.HasReconnectionVertex() && p.mPrevDist >= gMinReconnectionDistance) {
				p.SetReconnectionVertex(mScene, vertex, s);
			}
			#endif
        }

        p.mPrevPdfW = p.mPdfW;
        p.mIsPrevVertexDiffuse = isDiffuse;

        // modify contribution and pdf
		p.mPdfW         *= s.mFwdPdfW;
        p.mContribution *= s.mReflectance;
        if (!isMedium)
            p.mContribution *= vertex.mShadingData.shadingNormalCorrection<false>(vertex.mLocalDirIn, s.mDirection);


		// return sampled direction
        return isMedium ? s.mDirection : vertex.mShadingData.toWorld(s.mDirection);
    }

	// Trace a ray to find the next vertex
	void TraceNext(inout Path p, inout PathVertex vertex, inout RandomSampler rng, const float3 direction) {
		uint medium = vertex.mCurrentMedium;

		IntersectionResult isect;
        float transDirPdf, transNeePdf;
        mScene.traceScatteringRay(makeRay(rayOffset(vertex.mShadingData, direction), direction), rng, medium, p.mContribution, transDirPdf, transNeePdf, isect);
		isect.mShadingData.mTexcoordScreenSize = 0; // disable mipmaps for now

        p.mPdfW *= transDirPdf;
        p.mLength++;

        vertex = PathVertex(mScene, isect.mShadingData, isect.mInstancePrimitiveIndex, medium, -direction);

        if (vertex.mShadingData.isEnvironment())
            p.mPrevDist = POS_INFINITY;
        else
            p.mPrevDist = isect.mDistance;
	}

    // Generates a path with unidirectional path tracing. Attempts to reconnect to basePath is specified.
    Path GeneratePath(PathVertex vertex, inout RandomSampler rng, const Optional<PathSuffix> basePath = none) {
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
				TraceNext(p, vertex, rng, dir);
            }
		}

		#ifdef gCountRays
        if (!basePath.hasValue) {
			InterlockedAdd(mFramebuffer.mDebugCounters[4], 1);
			if (p.mSuffix.HasReconnectionVertex())
                InterlockedAdd(mFramebuffer.mDebugCounters[3], 1);
        }
		#endif

		return p;
    }
    Path GenerateShiftedPath(const PathVertex vertex, const PathSuffix basePath, out float jacobian) {
		RandomSampler rng = { basePath.mRngSeed };
        const Path p = GeneratePath(vertex, rng, basePath);

		jacobian = 0;
        if (any(p.mContribution > 0) && p.mLength == basePath.mPathLength) {
            jacobian = safe_divide(basePath.mPrefixPdfW, p.mPdfW);
            if (p.mRcJacobian != -1)
                jacobian *= p.mRcJacobian;
        }

        if (isnan(jacobian) || isinf(jacobian))
            jacobian = 0;

		#ifdef gCountRays
        InterlockedAdd(mFramebuffer.mDebugCounters[0], 1);
        if (jacobian > 0) {
            InterlockedAdd(mFramebuffer.mDebugCounters[1], 1);
			if (p.mRcJacobian > 0)
            	InterlockedAdd(mFramebuffer.mDebugCounters[2], 1);
        }
		#endif

        return p;
    }

	// Basic path tracing estimator
	Spectrum EstimateRadiance(const PathVertex vertex, inout RandomSampler rng) {
		const Path p = GeneratePath(vertex, rng);
		return p.mContribution * safe_divide(1, p.mPdfW);
	}
}
