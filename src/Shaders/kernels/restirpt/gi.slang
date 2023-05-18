#pragma once

#include "pt.slang"

enum ShiftType {
    eRandomReplay = 1,
    eReconnection = 2,
    eHybrid = 3
};

#ifndef gShiftType
#define gShiftType ShiftType::eRandomReplay
#endif
#ifndef gGIMis
#define gGIMis ReuseMISType::eSimple
#endif

bool IsReconnectionShift(const ShiftType t) {
    return (uint(t) & uint(ShiftType::eReconnection)) == uint(ShiftType::eReconnection);
}

struct Path {
	RandomSampler mRngSeed;
	float3 mContribution;
	float mPdfW;
	uint mLength;
    uint mDiffuseBounces;
    ReconnectionVertex mReconnectionVertex;
    float3 mThroughputAtReconnectionVertex; // Used in GeneratePath to cancel the contribution before the reconnection vertex
    float mReconnectionG;
    bool mIsPrevVertexDiffuse;
    bool mTerminated;

    __init(const RandomSampler rng) {
		mRngSeed = rng;
		mContribution = 1;
        mPdfW = 1;
        mLength = 1;
        mReconnectionG = 1;
        mDiffuseBounces = 0;
        mReconnectionVertex = ReconnectionVertex();
        mThroughputAtReconnectionVertex = 0;
        mIsPrevVertexDiffuse = false;
        mTerminated = false;
	}

	bool ReconnectionVertexFound() {
		return any(mThroughputAtReconnectionVertex > 0);
	}

	// Called when a reconnectable vertex is first found, *after* mContribution is modified with s.mReflectance
	[mutating]
	void SetReconnectionVertex(const SceneParameters scene, const PathVertex vertex, const float3 dirOut) {
		mThroughputAtReconnectionVertex = mContribution / mPdfW;
		mReconnectionVertex.mVertex.mLocalPosition = scene.mInstanceInverseTransforms[vertex.mInstanceIndex].transformPoint(vertex.mShadingData.mPosition);
        mReconnectionVertex.mVertex.mInstancePrimitiveIndex = vertex.mInstancePrimitiveIndex;
        mReconnectionVertex.mReconnectionLength = mLength;
        mReconnectionVertex.mLocalDirOut = dirOut;
	}

	[mutating]
    void Terminate(const float3 emission) {
        mContribution *= emission;
        if (ReconnectionVertexFound()) {
			mReconnectionVertex.mRadiance = (mContribution / mPdfW) / mThroughputAtReconnectionVertex;
            mReconnectionVertex.mPathLength = mLength;
        }
        mTerminated = true;
	}
};

struct PathReuseData {
    RandomSampler mRngSeed;
    PackedVertex mVertex;
    uint4 mPacked; // { pathLength | diffuseBounces<<8 | M<<16, pdfPdfW, W, localDirIn  }
    Optional<ReconnectionVertex> mReconnectionVertex;

    property uint mPathLength {
        get { return BF_GET(mPacked[0],            0, 8); }
        set {        BF_SET(mPacked[0], newValue,  0, 8); }
    };
    property uint mDiffuseBounces {
        get { return BF_GET(mPacked[0],           8, 8); }
        set {        BF_SET(mPacked[0], newValue, 8, 8); }
    };
    property float M {
        get { return f16tof32(BF_GET(mPacked[0],     16, 16)); }
        set { BF_SET(mPacked[0], f32tof16(newValue), 16, 16); }
    };
    property float mPathPdfW {
        get { return asfloat(mPacked[1]); }
        set { mPacked[1] = asuint(newValue); }
    };
    property float W {
        get { return asfloat(mPacked[2]); }
        set { mPacked[2] = asuint(newValue); }
    };
    property uint mPackedLocalDirIn {
        get { return mPacked[3]; }
        set { mPacked[3] = newValue; }
    };
};

extension PathTracer {
    void Reconnect(inout Path p, const PathVertex vertex, const Interaction bsdf, inout RandomSampler rng, const ReconnectionVertex rcv) {
        // attempt to connect to reconnection vertex
        const ShadingData sd_rcv = rcv.mVertex.getShadingData(mScene);
        float3 dirOut = sd_rcv.mPosition - vertex.mShadingData.mPosition;
        const float dist2 = dot(dirOut, dirOut);
        const float dist = sqrt(dist2);
        dirOut /= dist;

		// evaluate bsdf at vertex
        const float3 localDirOut = vertex.mShadingData.isSurface() ? vertex.mShadingData.toLocal(dirOut) : dirOut;
        float3 f = bsdf.evaluateReflectance<false>(vertex.mLocalDirIn, localDirOut).mReflectance;
        if (vertex.mShadingData.isSurface())
            f *= vertex.mShadingData.shadingNormalCorrection<false>(vertex.mLocalDirIn, localDirOut);

		// evaluate bsdf/emission at rcv
		float cos_rcv = 1;
		if (gHasMedia && sd_rcv.isMedium()) {
			const Medium bsdf_rcv = Medium(mScene, sd_rcv.getMaterialAddress());
			const float3 le = bsdf_rcv.emission();
			if (any(le > 0)) {
				f *= le;
            } else {
                f *= rcv.mRadiance * bsdf_rcv.evaluateReflectance<false>(-dirOut, rcv.mLocalDirOut).mReflectance;
			}
		} else {
			const PackedMaterialData bsdf_rcv = mScene.LoadMaterial(sd_rcv);
			const float3 localDirIn_rcv = sd_rcv.toLocal(-dirOut);
			cos_rcv = abs(localDirIn_rcv.z);
			const float3 le = bsdf_rcv.emission();
			if (any(le > 0)) {
				f *= le;
			} else {
                f *= rcv.mRadiance * bsdf_rcv.evaluateReflectance<false>(localDirIn_rcv, rcv.mLocalDirOut).mReflectance;
                f *= vertex.mShadingData.shadingNormalCorrection<false>(localDirIn_rcv, rcv.mLocalDirOut);
			}
        }

		// trace visibility ray
        {
            float transDirPdf, transNeePdf;
            mScene.traceVisibilityRay(makeRay(rayOffset(vertex.mShadingData, dirOut), dirOut, 0, AdjustIntersectionDistance(dist)), rng, vertex.mCurrentMedium, f, transDirPdf, transNeePdf);
            f *= safe_divide(1, transNeePdf);
		}

		// terminate path

        const float G = cos_rcv / dist2;
        if (any(f > 0) && cos_rcv > 0) {
			// manually Terminate path
            p.mContribution *= f * G;
			p.mReconnectionG = G;
            p.mReconnectionVertex = rcv;
            p.mReconnectionVertex.mReconnectionLength = p.mLength + 1;
            p.mReconnectionVertex.mPathLength = p.mLength + 1 + (rcv.mPathLength - rcv.mReconnectionLength);
            p.mLength = p.mReconnectionVertex.mPathLength;
            p.mTerminated = true;
        }
	}

    // Evaluates emission and samples a direction. Returns true when
    // a light path is found (when 'vertex' is emissive, or when vertex is connected to rcv)
    float3 IntegrateVertex(inout Path p, const PathVertex vertex, const Interaction bsdf, inout RandomSampler rng, const Optional<ReconnectionVertex> rcv = none) {
		// store albedo at the first vertex
		if (!rcv.hasValue && p.mLength == 1)
			mFramebuffer.mAlbedo[sPixelIndex] = float4(bsdf.albedo(), 1);

        const bool isMedium = gHasMedia && vertex.mShadingData.isMedium();

        // evaluate emission
        if (isMedium || vertex.mLocalDirIn.z > 0) {
            const float3 le = bsdf.emission();
            if (any(le > 0)) {
                // store reconnection vertex if a reconnection vertex hasn't been found yet
                if (!rcv.hasValue && p.mIsPrevVertexDiffuse && !p.ReconnectionVertexFound())
                    p.SetReconnectionVertex(mScene, vertex, 0);
                p.Terminate(le);
                return 0;
			}
		}

        const bool isDiffuse = !bsdf.isSingular();
        if (isDiffuse)
            p.mDiffuseBounces++;

        // terminate if too long, or bsdf is invalid
        if (p.mLength >= gPushConstants.mMaxDepth || p.mDiffuseBounces > gPushConstants.mMaxDiffuseBounces || !bsdf.canEvaluate()) {
            p.Terminate(0);
            return 0;
        }

		// attempt to connect to rcv
        if (rcv.hasValue && isDiffuse) {
            Reconnect(p, vertex, bsdf, rng, rcv.value);
			if (p.mTerminated)
            	return 0;
		}

        // sample direction, modify throughput
        DirectionSampleRecord s = bsdf.sampleDirection<false>(rng.nextFloat().xyz, vertex.mLocalDirIn);
		p.mContribution *= s.mReflectance;
		p.mPdfW         *= s.mFwdPdfW;
        if (!isMedium)
            p.mContribution *= vertex.mShadingData.shadingNormalCorrection<false>(vertex.mLocalDirIn, s.mDirection);

        if (all(p.mContribution <= 0) || p.mPdfW <= 0) {
            p.Terminate(0);
            return 0;
        }

		// store first reconnectable vertex
        if (!rcv.hasValue && isDiffuse && p.mIsPrevVertexDiffuse && !p.ReconnectionVertexFound())
			p.SetReconnectionVertex(mScene, vertex, s.mDirection);
        p.mIsPrevVertexDiffuse = isDiffuse;

        return isMedium ? s.mDirection : vertex.mShadingData.toWorld(s.mDirection);
    }

	PathVertex TraceNext(const PathVertex vertex, inout RandomSampler rng, const float3 direction, inout float3 throughput, inout float pdf) {
		uint medium = vertex.mCurrentMedium;

		IntersectionResult isect;
        float transDirPdf, transNeePdf;
        mScene.traceScatteringRay(makeRay(rayOffset(vertex.mShadingData, direction), direction), rng, medium, throughput, transDirPdf, transNeePdf, isect);
		isect.mShadingData.mTexcoordScreenSize = 0; // disable mipmaps for now

        pdf *= transDirPdf;

		return PathVertex(mScene, isect.mShadingData, isect.mInstancePrimitiveIndex, medium, -direction);
	}

    // Generates a path with unidirectional path tracing. Attempts to reconnect to rcv ASAP, if specified.
    Path GeneratePath(PathVertex vertex, inout RandomSampler rng, const Optional<ReconnectionVertex> rcv = none) {
        Path p = Path(rng);

        while (!p.mTerminated) {
			// Integrate current vertex
            float3 dir;
            if (gHasMedia && vertex.mShadingData.isMedium()) {
                const Medium m = Medium(mScene, vertex.mShadingData.getMaterialAddress());
                dir = IntegrateVertex(p, vertex, m, rng, rcv);
            } else if (vertex.mShadingData.isSurface()) {
                const PackedMaterialData m = mScene.LoadMaterial(vertex.mShadingData);
                dir = IntegrateVertex(p, vertex, m, rng, rcv);
            } else /*if (vertex.mShadingData.isEnvironment())*/ {
                float envPdfW;
                p.Terminate(mScene.EvaluateEnvironment(-vertex.mLocalDirIn, envPdfW));
            }

            // Trace ray to find next vertex
            if (!p.mTerminated) {
				vertex = TraceNext(vertex, rng, dir, p.mContribution, p.mPdfW);
                p.mLength++;
            }
		}

		return p;
    }
    Path GenerateShiftedPath(const PathVertex vertex, RandomSampler rng, const Optional<ReconnectionVertex> rcv, const uint baseLength, const uint baseDiffuseBounces, const float basePdfW, out float jacobian) {
        const Path p = GeneratePath(vertex, rng, rcv);

		jacobian = safe_divide(basePdfW, p.mPdfW);

		#ifdef gCountRays
        InterlockedAdd(mFramebuffer.mDebugCounters[0], 1);
        if (jacobian > 0 && p.mLength == baseLength && p.mDiffuseBounces == baseDiffuseBounces)
            InterlockedAdd(mFramebuffer.mDebugCounters[1], 1);
		#endif

        return p;
	}

	// Standard path tracing estimator
	float3 EstimateRadiance(const PathVertex vertex, inout RandomSampler rng) {
		const Path p = GeneratePath(vertex, rng);
		return p.mContribution * safe_divide(1, p.mPdfW);
	}

    PathReuseData LoadForReuse(const uint2 pixelIndex) {
        PathReuseData r;
        r.mRngSeed = reinterpret<RandomSampler>(mFramebuffer.mPrevReservoirDataGI[0][pixelIndex]);
        r.mVertex =  reinterpret<PackedVertex> (mFramebuffer.mPrevReservoirDataGI[1][pixelIndex]);
        r.mPacked =  reinterpret<uint4>        (mFramebuffer.mPrevReservoirDataGI[2][pixelIndex]);
        if (IsReconnectionShift(gShiftType))
            r.mReconnectionVertex = LoadPrevReconnectionVertex(mFramebuffer, pixelIndex);
        else
            r.mReconnectionVertex = none;
        return r;
	}
    void StoreForReuse(const Path p, const float W, const float M, const PathVertex vertex) {
        PathReuseData data;
        data.mPathLength = p.mLength;
        data.mDiffuseBounces = p.mDiffuseBounces;
        data.M = M;
        data.mPathPdfW = p.mPdfW;
        data.W = W;
        data.mPackedLocalDirIn = vertex.mPackedLocalDirIn;
		data.mVertex.mLocalPosition = mScene.mInstanceInverseTransforms[vertex.mInstanceIndex].transformPoint(vertex.mShadingData.mPosition);
        data.mVertex.mInstancePrimitiveIndex = vertex.mInstancePrimitiveIndex;
        mFramebuffer.mReservoirDataGI[0][sPixelIndex] = reinterpret<float4>(p.mRngSeed);
        mFramebuffer.mReservoirDataGI[1][sPixelIndex] = reinterpret<float4>(data.mVertex);
        mFramebuffer.mReservoirDataGI[2][sPixelIndex] = reinterpret<float4>(data.mPacked);
        if (IsReconnectionShift(gShiftType))
            StoreReconnectionVertex(mFramebuffer, sPixelIndex, p.mReconnectionVertex);
    }
    Path ShiftPathToVertex(const PathReuseData p, const PathVertex dstVertex, out float jacobian) {
        return GenerateShiftedPath(dstVertex, p.mRngSeed, p.mReconnectionVertex, p.mPathLength, p.mDiffuseBounces, p.mPathPdfW, jacobian);
	}
    Path ShiftPathToPixel(const Path p, const PathReuseData dstVertex, out float jacobian) {
        const PathVertex vertex_i = PathVertex(mScene, dstVertex.mVertex.getShadingData(mScene), dstVertex.mVertex.mInstancePrimitiveIndex, -1, dstVertex.mPackedLocalDirIn);
        if (IsReconnectionShift(gShiftType))
            return GenerateShiftedPath(vertex_i, p.mRngSeed, p.mReconnectionVertex, p.mLength, p.mDiffuseBounces, p.mPdfW, jacobian);
        else
            return GenerateShiftedPath(vertex_i, p.mRngSeed, none, p.mLength, p.mDiffuseBounces, p.mPdfW, jacobian);
    }

	// shift estimator, simply averages shifted paths from neighbors
	float3 EstimateRadianceShift(const PathVertex vertex, inout RandomSampler rng) {
        // sample canonical path
        Path p = GeneratePath(vertex, rng);
        float W = safe_divide(1, p.mPdfW);

		float3 f = p.mContribution * W;

		#ifdef gReSTIR_GI_Reuse

        // store the sampled path
        StoreForReuse(p, W, 1, vertex);

        if (gPushConstants.mReservoirHistoryValid != 0 && p.mLength > 1) {
            const float2 prevPixel = mFramebuffer.mPrevUVs[sPixelIndex] * gPushConstants.mOutputExtent;

            float norm = 1;

            for (uint i = 0; i < gPushConstants.mGIReuseSamples; i++) {
                const float2 discRng = rng.nextFloat().xy;
                const int2 candidatePixel = int2(prevPixel + sampleConcentricDisc(discRng.x, discRng.y) * gPushConstants.mGIReuseRadius);
                if (any(candidatePixel < 0) || any(candidatePixel >= gPushConstants.mOutputExtent))
                    continue;

                const PathReuseData data = LoadForReuse(candidatePixel);

                if (data.M <= 0 || data.mPathLength == 1) // invalid sample
                    continue;

				// shift candidatePixel's path to vertex
                float jacobian_from_i;
                const Path xi = ShiftPathToVertex(data, vertex, jacobian_from_i);

                if (xi.mLength != p.mLength || xi.mDiffuseBounces != p.mDiffuseBounces)
                    continue;

                norm++;

                // add radiance from shifted path
                f += xi.mContribution * data.W * jacobian_from_i;
            }

            // average canonical and shifted samples
            f /= norm;
		}

		#endif

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

		#ifdef gReSTIR_GI_Reuse

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

                const PathReuseData data = LoadForReuse(candidatePixel);

                if (data.M <= 0)
                    continue;

				// shift candidatePixel's path to vertex
                float jacobian_from_i;
                const Path xi = ShiftPathToVertex(data, vertex, jacobian_from_i);

                if (xi.mLength != p.mLength || xi.mDiffuseBounces != p.mDiffuseBounces)
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

                const PathReuseData data = LoadForReuse(candidatePixel);

                if (selectedIndex == i) {
                    Z += data.M;
                    continue;
                }

                // shift p to candidatePixel's vertex
                float jacobian_to_i;
                const Path p_at_i = ShiftPathToPixel(p, data, jacobian_to_i);

                if (luminance(p_at_i.mContribution) * jacobian_to_i > 0)
                    Z += data.M;
            }

            W = safe_divide(W, Z);
		}

        if (any(p.mContribution > 0))
            StoreForReuse(p, W, min(M, gPushConstants.mGICandidateSamples * gPushConstants.mGIMaxM), vertex);

		#endif

		return p.mContribution * W;
	}
}