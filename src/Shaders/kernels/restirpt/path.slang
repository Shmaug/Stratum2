#pragma once

#include "common.slang"

#ifdef gReconnection
static bool gEnableReconnection = true;
#else
static bool gEnableReconnection = false;
#endif
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
    float W; // only for canonical paths
    PathSuffix mSuffix;

    __init() {
        mContribution = 0;
        W = 0;
        mSuffix = PathSuffix({0});
	}
};

Path MergePaths(const Path p0, const Path p1, inout RandomSampler rng) {
    if (p0.W <= 0) return p1;
    if (p1.W <= 0) return p0;

    const float w0 = luminance(p0.mContribution) * p0.W;
    const float w1 = luminance(p1.mContribution) * p1.W;
    const float wsum = w0 + w1;

    if (rng.nextFloat().x*wsum <= w0)
		return { p0.mContribution, safe_divide(wsum, luminance(p0.mContribution)), p0.mSuffix };
	else
    	return { p1.mContribution, safe_divide(wsum, luminance(p1.mContribution)), p1.mSuffix };
}

struct PathState {
    Spectrum mContribution;
	float mPdfW; // full path pdfW
	uint mLength;
    uint mDiffuseBounces;
    bool mTerminated;

    PathSuffix mSuffix; // output shift data
    float mRcJacobian;  // reconnection jacobian, computed in Reconnect(). The value -1 indicates no reconnection was attempted.
    float mPrevPdfW;
    float mPrevDist;
    bool mIsPrevVertexDiffuse;

    float mBsdfPdfW;
    float mLightSamplePdfA; // NEE pdf of the current vertex

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
        mBsdfPdfW = 0;
        mLightSamplePdfA = 0;
        mIsPrevVertexDiffuse = false;
	}


    // Called when a reconnectable vertex is found
	// must be called BEFORE mContribution and mPdfW are modified with directionSample's reflectance and pdf
	// when directionSample is none, the reconnection vertex is an emissive vertex
	// Note: mReconnectionVertex.mRadiance temporarily stored here, then finalized in Terminate
	[mutating]
    void SetReconnectionVertex(const SceneParameters scene, const PathVertex vertex, const Optional<DirectionSampleRecord> directionSample = none) {
        if (vertex.mShadingData.isEnvironment()) {
            mSuffix.mReconnectionVertex.mVertex.mLocalPosition = vertex.mShadingData.mPosition;
        } else {
            mSuffix.mReconnectionVertex.mVertex.mLocalPosition = scene.mInstanceInverseTransforms[vertex.mInstanceIndex].transformPoint(vertex.mShadingData.mPosition);
        }
        mSuffix.mReconnectionVertex.mVertex.mInstancePrimitiveIndex = vertex.mInstancePrimitiveIndex;

        if (directionSample.hasValue) {
            mSuffix.mReconnectionVertex.mLocalDirOut = directionSample.value.mDirection;
            mSuffix.mReconnectionVertex.mRadiance = mContribution * directionSample.value.mReflectance; // temporarily stored here, finalized in Terminate
        } else {
            mSuffix.mReconnectionVertex.mLocalDirOut = 0;
            mSuffix.mReconnectionVertex.mRadiance = mContribution; // temporarily stored here, finalized in Terminate
		}
        mSuffix.mPrefixLength = mLength;
        mSuffix.mSuffixDiffuseBounces = mDiffuseBounces; // temporarily stored here, finalized in Terminate
        mSuffix.mPrefixPdfW = mPrevPdfW;
        mSuffix.mReconnectionDist = mPrevDist;
        mSuffix.mReconnectionCos = vertex.mShadingData.isSurface() ? abs(vertex.mLocalDirIn.z) : 1;
	}

    [mutating]
    void Kill() {
        mContribution = 0;
        mSuffix.mReconnectionVertex.mRadiance = 0;
        mSuffix.mPrefixPdfW = 0;
        mTerminated = true;
	}

	// called when an emissive vertex is found
	[mutating]
    void Terminate(const float3 emission, inout Path result, inout RandomSampler rng) {
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
        }
		mSuffix.mPathLength = mLength;
        mTerminated = true;

        // randomly replace result using resampling
        result = MergePaths(result, { mContribution, safe_divide(1, mPdfW), mSuffix }, rng);
	}
};

// path generation

extension PathTracer {
    void Reconnect(inout PathState p, const PathVertex vertex, const Interaction bsdf, inout RandomSampler rng, const PathSuffix basePath, inout Path result) {
        if (p.mLength + 1 < basePath.mPrefixLength) {
			// need more bounces before we can reconnect. don't terminate yet.
			return;
		}

        if (p.mLength + 1 > basePath.mPrefixLength || p.mDiffuseBounces + basePath.mSuffixDiffuseBounces > gPushConstants.mMaxDiffuseBounces) {
			// path too long - can't reconnect
            p.Kill();
            return;
        }

        const ShadingData sd_rcv = basePath.mReconnectionVertex.mVertex.getShadingData(mScene);
        Vector3 toRcv;
        float dist;
        if (sd_rcv.isEnvironment()) {
            toRcv = basePath.mReconnectionVertex.mVertex.mLocalPosition;
            dist = POS_INFINITY;
        } else {
			toRcv = sd_rcv.mPosition - vertex.mShadingData.mPosition;
			dist = length(toRcv);
			toRcv /= dist;
        }

        if (dist < gMinReconnectionDistance) {
            p.Kill();
            return;
        }

        const float cos_rcv = sd_rcv.isSurface() ? sd_rcv.toLocal(-toRcv).z : 1;

        if (sd_rcv.isEnvironment())
            p.mRcJacobian = 1;
		else
            p.mRcJacobian = safe_divide(cos_rcv, basePath.mReconnectionCos) * pow2(safe_divide(basePath.mReconnectionDist, dist));

        // evaluate bsdf at vertex
        {
            const Vector3 localDirOut = vertex.mShadingData.isSurface() ? vertex.mShadingData.toLocal(toRcv) : toRcv;
            const ReflectanceEvalRecord s = bsdf.evaluateReflectance<false>(vertex.mLocalDirIn, localDirOut);
            p.mContribution *= s.mReflectance;
            if (vertex.mShadingData.isSurface())
                p.mContribution *= vertex.mShadingData.shadingNormalCorrection<false>(vertex.mLocalDirIn, localDirOut);
            if (s.mFwdPdfW <= 0) {
                p.Kill();
                return;
			}
        }

        // evaluate bsdf at rcv, if rcv is not the last vertex in the path
        if (p.mLength + 1 < basePath.mPathLength) {
			if (gHasMedia && sd_rcv.isMedium()) {
				const Medium bsdf_rcv = Medium(mScene, sd_rcv.getMaterialAddress());
				const ReflectanceEvalRecord s = bsdf_rcv.evaluateReflectance<false>(-toRcv, basePath.mReconnectionVertex.mLocalDirOut);
				p.mContribution *= s.mReflectance;
				if (s.mFwdPdfW <= 0) {
					p.Kill();
					return;
				}
			} else if (sd_rcv.isSurface()) {
				const PackedMaterialData bsdf_rcv = mScene.LoadMaterial(sd_rcv);
				const Vector3 localDirIn_rcv = sd_rcv.toLocal(-toRcv);
                const ReflectanceEvalRecord s = bsdf_rcv.evaluateReflectance<false>(localDirIn_rcv, basePath.mReconnectionVertex.mLocalDirOut);
                p.mContribution *= s.mReflectance;
                p.mContribution *= vertex.mShadingData.shadingNormalCorrection<false>(localDirIn_rcv, basePath.mReconnectionVertex.mLocalDirOut);
				if (s.mFwdPdfW <= 0 || all(p.mContribution <= 0)) {
					p.Kill();
					return;
				}
			}
        }
		p.mContribution *= basePath.mReconnectionVertex.mRadiance;

        // terminate path into rcv vertex
		p.mLength = basePath.mPathLength;
		p.mDiffuseBounces += basePath.mSuffixDiffuseBounces;
        p.mTerminated = true;

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
                p.Kill();
                return;
            }
        }

		// no resampling for shifted paths, just assign directly
		result = { p.mContribution, safe_divide(1, p.mPdfW), p.mSuffix };
    }

    void SampleNee(PathState p, const PathVertex vertex, const Interaction bsdf, inout RandomSampler rng, inout Path result) {
        const EmissionSampleRecord emission = mScene.SampleEmission(rng.nextFloat());

        Vector3 toLight;
        float dist;
        if (emission.mShadingData.isEnvironment()) {
            toLight = emission.mShadingData.mPosition;
            dist = POS_INFINITY;
        } else {
			toLight = emission.mShadingData.mPosition - vertex.mShadingData.mPosition;
			dist = length(toLight);
			toLight /= dist;
        }

        const float cos_rcv = emission.mShadingData.isSurface() ? dot(-toLight, emission.mShadingData.getShadingNormal()) : 1;
        if (cos_rcv <= 0)
            return;

		// store rcv if no rcv exists
        if (!p.mSuffix.HasReconnectionVertex() && dist >= gMinReconnectionDistance)
            p.SetReconnectionVertex(mScene, PathVertex(mScene, emission.mShadingData, emission.mInstancePrimitiveIndex, vertex.mCurrentMedium, -toLight));

        float lightPdfW = emission.mPdf;
		if (!emission.mShadingData.isEnvironment())
			lightPdfW *= pow2(dist);
		if (emission.mShadingData.isSurface())
			lightPdfW /= cos_rcv;
		p.mPdfW *= lightPdfW;

		// evaluate bsdf at vertex
		{
			const Vector3 localDirOut = vertex.mShadingData.isSurface() ? vertex.mShadingData.toLocal(toLight) : toLight;
			const ReflectanceEvalRecord s = bsdf.evaluateReflectance<false>(vertex.mLocalDirIn, localDirOut);
			p.mContribution *= s.mReflectance;
			if (vertex.mShadingData.isSurface())
                p.mContribution *= vertex.mShadingData.shadingNormalCorrection<false>(vertex.mLocalDirIn, localDirOut);
            if (all(p.mContribution <= 0))
                return;

            // NEE MIS
        	p.mContribution *= pow2(lightPdfW) / (pow2(lightPdfW) + pow2(s.mFwdPdfW));
		}

        // trace visibility ray
        {
            float transDirPdf, transNeePdf;
            mScene.traceVisibilityRay(
				makeRay(rayOffset(vertex.mShadingData, toLight), toLight, 0, AdjustIntersectionDistance(dist)),
				rng, vertex.mCurrentMedium, p.mContribution, transDirPdf, transNeePdf);
			if (transNeePdf <= 0) {
				p.Kill();
				return;
			}
			p.mContribution /= transNeePdf;
        }

        Spectrum radiance;
        if (emission.mShadingData.isSurface()) {
            radiance = mScene.LoadMaterial(emission.mShadingData).emission();
        } else if (emission.mShadingData.isMedium()) {
            radiance = Medium(mScene, emission.mShadingData.getMaterialAddress()).emission();
        } else /*if (emission.mShadingData.isEnvironment())*/ {
            float tmp;
            radiance = mScene.EvaluateEnvironment(toLight, tmp);
		}

        p.Terminate(radiance, result, rng);
	}

    // Evaluates emission, performs reconnection, samples the BSDF, and returns the sampled direction
    Vector3 IntegrateVertex(inout PathState p, const PathVertex vertex, const Interaction bsdf, inout RandomSampler rng, const Optional<PathSuffix> basePath, inout Path result) {
        const bool isCanonical = !basePath.hasValue;
		const bool isMedium = gHasMedia && vertex.mShadingData.isMedium();
        const bool isDiffuse = !bsdf.isSingular();

        // evaluate emission
        if (isMedium || vertex.mLocalDirIn.z > 0) {
            const Spectrum le = bsdf.emission();
            if (any(le > 0)) {
                // store reconnection vertex if a reconnection vertex hasn't been found yet
                if (gEnableReconnection && isCanonical && p.mIsPrevVertexDiffuse && !p.mSuffix.HasReconnectionVertex() && p.mPrevDist >= gMinReconnectionDistance)
                    p.SetReconnectionVertex(mScene, vertex);
				#ifdef gNee
                // NEE MIS
                if (p.mIsPrevVertexDiffuse)
                	p.mContribution *= pow2(p.mBsdfPdfW) / (pow2(p.mBsdfPdfW) + pow2(p.mLightSamplePdfA * pow2(p.mPrevDist) / vertex.mLocalDirIn.z));
				#ifdef gDebugNee
				if (p.mIsPrevVertexDiffuse)
                	p.Kill();
				else
				#endif
				#endif
				p.Terminate(le, result, rng);
                return 0;
			}
		}

        if (isDiffuse)
            p.mDiffuseBounces++;

        // terminate if too long, or bsdf is invalid
        if (p.mLength + 1 > gPushConstants.mMaxDepth || p.mDiffuseBounces > gPushConstants.mMaxDiffuseBounces || !bsdf.canEvaluate()) {
            p.Kill();
            return 0;
        }

		// attempt to connect to basePath
		if (!isCanonical && isDiffuse && basePath.value.HasReconnectionVertex()) {
			Reconnect(p, vertex, bsdf, rng, basePath.value, result);
			if (p.mTerminated)
				return 0;
		}

		// next event estimation
		#ifdef gNee
		if (isCanonical && isDiffuse)
            SampleNee(p, vertex, bsdf, rng, result);
		#endif

        // sample direction
        const DirectionSampleRecord s = bsdf.sampleDirection<false>(rng.nextFloat().xyz, vertex.mLocalDirIn);
        if (all(s.mReflectance <= 0) || s.mFwdPdfW <= 0) {
            p.Kill();
            return 0;
        }

        // store first reconnectable vertex
        if (gEnableReconnection && isCanonical && isDiffuse && p.mIsPrevVertexDiffuse && !p.mSuffix.HasReconnectionVertex() && p.mPrevDist >= gMinReconnectionDistance)
			p.SetReconnectionVertex(mScene, vertex, s);

        p.mIsPrevVertexDiffuse = isDiffuse;

        // modify contribution and pdf
        p.mPrevPdfW = p.mPdfW;
		p.mPdfW    *= s.mFwdPdfW;
        p.mBsdfPdfW = s.mFwdPdfW;
        p.mContribution *= s.mReflectance;
        if (!isMedium)
            p.mContribution *= vertex.mShadingData.shadingNormalCorrection<false>(vertex.mLocalDirIn, s.mDirection);

		// return sampled direction
        return isMedium ? s.mDirection : vertex.mShadingData.toWorld(s.mDirection);
    }

    // Generates a path with unidirectional path tracing. Attempts to reconnect to basePath if specified.
    Path GeneratePath(inout PathState p, inout PathVertex vertex, inout RandomSampler rng, const Optional<PathSuffix> basePath) {
        Path result = Path();

        while (!p.mTerminated) {
			// Integrate current vertex
			Vector3 direction;
			if (gHasMedia && vertex.mShadingData.isMedium()) {
				const Medium m = Medium(mScene, vertex.mShadingData.getMaterialAddress());
				direction = IntegrateVertex(p, vertex, m, rng, basePath, result);
			} else if (vertex.mShadingData.isSurface()) {
				const PackedMaterialData m = mScene.LoadMaterial(vertex.mShadingData);
				direction = IntegrateVertex(p, vertex, m, rng, basePath, result);
			} else /*if (vertex.mShadingData.isEnvironment())*/ {
				// store reconnection vertex if a reconnection vertex hasn't been found yet
                if (gEnableReconnection && !basePath.hasValue && p.mIsPrevVertexDiffuse && !p.mSuffix.HasReconnectionVertex())
                    p.SetReconnectionVertex(mScene, vertex);
				// NEE MIS
                p.mContribution *= pow2(p.mBsdfPdfW) / (pow2(p.mBsdfPdfW) + pow2(p.mLightSamplePdfA));
				float envPdfW;
				p.Terminate(mScene.EvaluateEnvironment(-vertex.mLocalDirIn, envPdfW), result, rng);
			}

			// Trace ray to find next vertex
			if (!p.mTerminated) {
				uint medium = vertex.mCurrentMedium;

				IntersectionResult isect;
				float transDirPdf, transNeePdf;
				mScene.traceScatteringRay(makeRay(rayOffset(vertex.mShadingData, direction), direction), rng, medium, p.mContribution, transDirPdf, transNeePdf, isect);
				isect.mShadingData.mTexcoordScreenSize = 0; // disable mipmaps for now

                p.mPdfW *= transDirPdf;
                p.mLightSamplePdfA = isect.LightSamplePdfA();
				p.mLength++;

                vertex = PathVertex(mScene, isect.mShadingData, isect.mInstancePrimitiveIndex, medium, -direction);

				if (vertex.mShadingData.isEnvironment())
					p.mPrevDist = POS_INFINITY;
				else
					p.mPrevDist = isect.mDistance;
            }
        }

        return result;
    }

    Path GeneratePath(PathVertex vertex, inout RandomSampler rng) {
        PathState p = PathState(rng);
        const Path result = GeneratePath(p, vertex, rng, none);

		#ifdef gCountRays
		InterlockedAdd(mFramebuffer.mDebugCounters[4], 1);
		if (p.mSuffix.HasReconnectionVertex())
			InterlockedAdd(mFramebuffer.mDebugCounters[3], 1);
		#endif

        return result;
    }
    Path GenerateShiftedPath(PathVertex vertex, const PathSuffix basePath, out float jacobian) {
		RandomSampler rng = { basePath.mRngSeed };
        PathState p = PathState(rng);
        const Path result = GeneratePath(p, vertex, rng, basePath);

		// compute jacobian
		jacobian = 0;
        if (any(p.mContribution > 0) && p.mLength == basePath.mPathLength) {
            jacobian = safe_divide(basePath.mPrefixPdfW, p.mSuffix.mPrefixPdfW);
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

        return result;
    }
}
