#pragma once

#include "path.slang"

struct PathReservoir {
    Path p;
    float W;
    float M;

    __init() {
        p.mContribution = 0;
        W = 0;
        M = 0;
    }
    __init(const Path p) {
        this.p = p;
        this.W = safe_divide(1, p.mPdfW);
        this.M = 1;
    }
    __init(const Path p, const float W, const float M) {
        this.p = p;
        this.W = W;
        this.M = M;
    }

    Spectrum GetEstimate() {
        return p.mContribution * W;
    }

    // Target PDF
    float GetSampleTargetWeight() {
        return luminance(p.mContribution);
    }

    // The weight for selecting this reservoir's sample during merging
    float GetResamplingWeight() {
        return GetSampleTargetWeight() * W;
    }

    // Prepares W for merging, instead of integration
    [mutating]
    void PrepareMerge(const float misWeight = 1) {
        W *= GetSampleTargetWeight() * misWeight;
    }

    // Finalizes merging, preparing W for integration
    [mutating]
    void FinalizeMerge() {
        W = safe_divide(W, GetSampleTargetWeight());
    }

    [mutating]
    bool Merge(inout RandomSampler rng, const PathReservoir r, const float misWeight = 1, const float jacobian = 1) {
        M += r.M;
        const float w = r.GetResamplingWeight() * jacobian * misWeight;
        W += w;
        if (rng.nextFloat().x * W <= w) {
            p = r.p;
            return true;
        }
        return false;
	}
};
struct PackedPathReservoir {
    PackedVertex mBaseVertex;
    PathSuffix mSuffix;
    float W;
    float M;
    float mCachedTargetWeight;
    uint mLocalDirIn;

    PathVertex GetBaseVertex(const SceneParameters scene, const RenderParams framebuffer) {
        const ShadingData sd = mBaseVertex.getShadingData(scene);
        const float3 localDirIn = sd.toLocal(normalize(framebuffer.mViewTransforms[0].transformPoint(0) - sd.mPosition));
        return PathVertex(scene, sd, mBaseVertex.mInstancePrimitiveIndex, -1, packNormal(localDirIn));
    }
};

// Spatial reuse pattern
struct SampleLocationGenerator {
    RandomSampler mSeed;
    int2 mCenter;

    __init(inout RandomSampler rng, const int2 center) {
        mSeed = rng;
        rng.skipNext(gPushConstants.mGIReuseSamples);
        mCenter = center;
	}

    __subscript(uint index) -> int2 {
        get {
			RandomSampler tmp_rng = mSeed;
			tmp_rng.skipNext(index);
            const float2 discRng = tmp_rng.nextFloat().xy;
            return mCenter + int2(sampleConcentricDisc(discRng.x, discRng.y) * gPushConstants.mGIReuseRadius);
        }
    }
};

extension PathTracer {
    Optional<PackedPathReservoir> LoadReservoir(const int2 pixelIndex) {
        if (any(pixelIndex < 0) || any(pixelIndex >= gPushConstants.mOutputExtent))
            return none;

        PackedPathReservoir r;

        const float4 data3 = mFramebuffer.mPrevPathReservoirData[3][pixelIndex];
        r.W = data3.x;
        r.M = data3.y;
        r.mCachedTargetWeight = data3.z;
        r.mLocalDirIn = asuint(data3.w);

        if (r.M <= 0) // null reservoir
            return none;

        r.mBaseVertex  = reinterpret<PackedVertex>(mFramebuffer.mPrevPathReservoirData[0][pixelIndex]);
        r.mSuffix.mRngSeed    = reinterpret<uint4>(mFramebuffer.mPrevPathReservoirData[1][pixelIndex]);
        r.mSuffix.mPackedData = reinterpret<uint4>(mFramebuffer.mPrevPathReservoirData[2][pixelIndex]);

		#ifdef gUseReconnection
        r.mSuffix.mReconnectionVertex = LoadPrevReconnectionVertex(mFramebuffer, pixelIndex);
		#endif

		return r;
    }
    void StoreReservoir(const PathReservoir r, const PathVertex vertex) {
        PackedVertex baseVertex;
		baseVertex.mLocalPosition = mScene.mInstanceInverseTransforms[vertex.mInstanceIndex].transformPoint(vertex.mShadingData.mPosition);
        baseVertex.mInstancePrimitiveIndex = vertex.mInstancePrimitiveIndex;

        mFramebuffer.mPathReservoirData[0][sPixelIndex] = reinterpret<float4>(baseVertex);
        mFramebuffer.mPathReservoirData[1][sPixelIndex] = reinterpret<float4>(r.p.mSuffix.mRngSeed);
        mFramebuffer.mPathReservoirData[2][sPixelIndex] = reinterpret<float4>(r.p.mSuffix.mPackedData);
        mFramebuffer.mPathReservoirData[3][sPixelIndex] = float4(
            r.GetSampleTargetWeight() > 0 ? r.W : 0,
            min(r.M, gPushConstants.mGIMaxM),
            0,
            0);

        #ifdef gUseReconnection
		StoreReconnectionVertex(mFramebuffer, sPixelIndex, r.p.mSuffix.mReconnectionVertex);
		#endif
    }

    // Store reservoir for reuse & merge with neighbors
    void ReservoirReuse(const PathVertex vertex, inout RandomSampler rng, inout PathReservoir r) {
		#ifndef gTemporalReuse
		// without temporal reuse, we store the reservoir before merging with previous reservoirs
        StoreReservoir(r, vertex);
		#endif

		if (gPushConstants.mReservoirHistoryValid != 0) {
			const float2 prevPixel = sPixelIndex + 0.5; //mFramebuffer.mPrevUVs[sPixelIndex] * gPushConstants.mOutputExtent;
			const SampleLocationGenerator sg = SampleLocationGenerator(rng, int2(prevPixel));

            // apply mis weight to current sample
            r.PrepareMerge(gPushConstants.mGICandidateSamples);

            int selected_i = -1;

			for (uint i = 0; i < gPushConstants.mGIReuseSamples; i++) {
                const Optional<PackedPathReservoir> candidate = LoadReservoir(sg[i]);
                if (!candidate.hasValue)
                    continue;

                if (candidate.value.W <= 0) {
                    r.M += candidate.value.M;
                    continue;
                }

                PathReservoir ri;
                ri.M = candidate.value.M;
                ri.W = candidate.value.W;
                float jacobian;
                ri.p = GenerateShiftedPath(vertex, candidate.value.mSuffix, jacobian);
                if (jacobian <= 0) {
					// shift failed
                    r.M += ri.M;
                    continue;
                }

                if (r.Merge(rng, ri, ri.M, jacobian)) {
                    selected_i = int(i);
                }
            }

			// finalize MIS weight
            if (r.W > 0) {
                float Z = gPushConstants.mGICandidateSamples;
                for (uint i = 0; i < gPushConstants.mGIReuseSamples; i++) {
                    const Optional<PackedPathReservoir> candidate = LoadReservoir(sg[i]);
                    if (!candidate.hasValue)
                        continue;

					if (i == selected_i) {
                        Z += candidate.value.M;
                        continue;
                    }

                    float jacobian;
                    const Path pi = GenerateShiftedPath(candidate.value.GetBaseVertex(mScene, mFramebuffer), r.p.mSuffix, jacobian);
                    if (any(pi.mContribution > 0) && jacobian > 0)
                        Z += candidate.value.M;
                }
                const float before = r.W;
                r.W = safe_divide(r.W, Z);
            }

            r.FinalizeMerge();
		}

		#ifdef gTemporalReuse
		// with temporal reuse, we store the reservior after merging with previous reservoirs
        StoreReservoir(r, vertex);
		#endif
	}

    // ReSTIR estimator
    Spectrum EstimateRadianceReSTIR(const PathVertex vertex, inout RandomSampler rng) {
        PathReservoir r = PathReservoir();

        // sample canonical paths
        {
            for (uint i = 0; i < gPushConstants.mGICandidateSamples; i++) {
                r.Merge(rng, PathReservoir(GeneratePath(vertex, rng)));
            }
            r.FinalizeMerge();
            // Equivalent to multiplying each sample's weight by 1/M
            if (r.M > 0)
                r.W /= r.M;
        }

		#ifdef gReSTIR_GI_Reuse
		// no resampling for paths with no bounces
        if (!(r.p.mLength == 1 && max3(r.p.mContribution) > 0 && r.W > 0))
            ReservoirReuse(vertex, rng, r);
        #endif

		return r.GetEstimate();
	}
}