#include "di.slang"

enum ShiftType {
	eRandomReplay,
	eReconnection,
	eHybrid
};

#ifndef gShiftType
#define gShiftType ShiftType::eRandomReplay
#endif

struct ReconnectionVertex {
    float3 mLocalPosition;
    uint mInstancePrimitiveIndex;
    uint4 mPackedData;

    property uint mInstanceIndex  {
		get { return BF_GET(mInstancePrimitiveIndex, 0, 16); }
		set { BF_SET(mInstancePrimitiveIndex, newValue, 0, 16); }
	}
    property uint mPrimitiveIndex {
		get { return BF_GET(mInstancePrimitiveIndex, 16, 16); }
		set { BF_SET(mInstancePrimitiveIndex, newValue, 16, 16); }
	}

	// estimated radiance leaving mVertex towards mDirOut
    property float3 mRadiance {
        get {
            return float3(
                f16tof32(BF_GET(mPackedData[0],  0, 16)),
                f16tof32(BF_GET(mPackedData[0], 16, 16)),
                f16tof32(BF_GET(mPackedData[1],  0, 16)));
        } set {
			BF_SET(mPackedData[0], f32tof16(newValue[0]),  0, 16);
			BF_SET(mPackedData[0], f32tof16(newValue[1]), 16, 16);
			BF_SET(mPackedData[1], f32tof16(newValue[2]),  0, 16);
		}
	};

	// number of path segments before the reconnection vertex
    property uint mReconnectionLength {
        get { return BF_GET(mPackedData[2], 0, 16); }
        set { BF_SET(mPackedData[2], newValue, 0, 16); }
    };

	// number of path segments in the whole path
    property uint mPathLength {
        get { return BF_GET(mPackedData[2], 16, 16); }
        set { BF_SET(mPackedData[2], newValue, 16, 16); }
    };

    property float3 mDirOut {
        get { return unpackNormal(mPackedData[3]); }
        set { mPackedData[3] = packNormal(newValue); }
	}

    ShadingData getShadingData(const SceneParameters scene) {
        ShadingData sd = scene.makeShadingData(scene.mInstances[mInstanceIndex], scene.mInstanceTransforms[mInstanceIndex], mLocalPosition, mPrimitiveIndex);
        sd.mTexcoordScreenSize = 0;
        return sd;
    }
};

struct Path {
	RandomSampler mRng;
	float3 mContribution;
	float mPdfW;
	uint mLength;
    ReconnectionVertex mReconnectionVertex;
    float3 mThroughputAtReconnectionVertex; // Used in GeneratePath to divide out the path contribution before the reconnection vertex

	bool ReconnectionVertexFound() {
		return any(mThroughputAtReconnectionVertex > 0);
	}

	// Called when a reconnectable vertex is first found, after mContribution is modified with s.mReflectance
	[mutating]
	void MakeReconnectionVertex(const SceneParameters scene, const PathVertex vertex, const DirectionSampleRecord s) {
		mThroughputAtReconnectionVertex = mContribution / mPdfW;
		mReconnectionVertex.mLocalPosition = scene.mInstanceInverseTransforms[vertex.mInstanceIndex].transformPoint(vertex.mShadingData.mPosition);
		mReconnectionVertex.mInstancePrimitiveIndex = vertex.mInstancePrimitiveIndex;
		mReconnectionVertex.mDirOut = s.mDirection;
		mReconnectionVertex.mReconnectionLength = mLength;
	}
	// Called after a full path is constructed
	[mutating]
	void FinalizeReconnectionVertex() {
		mReconnectionVertex.mRadiance = (mContribution/mPdfW) / mThroughputAtReconnectionVertex;
		mReconnectionVertex.mPathLength = mLength;
	}
};

extension PathTracer {
	// Evaluates emission and samples a direction. Returns true when
	// a light path is found (when the vertex is emissive)
	bool SampleNext(inout Path p, const PathVertex vertex, const Interaction bsdf, inout RandomSampler rng, out DirectionSampleRecord s) {
		// store albedo at the first vertex
		if (p.mLength == 1)
			mFramebuffer.mAlbedo[sPixelIndex] = float4(bsdf.albedo(), 1);

        const bool isMedium = gHasMedia && vertex.mShadingData.isMedium();

        // evaluate emission
        if (isMedium || vertex.mLocalDirIn.z > 0) {
			const float3 le = bsdf.emission();
			if (any(le > 0)) {
				p.mContribution *= le;
				// store radiance at reconnection vertex, if any
				if (p.ReconnectionVertexFound())
					p.FinalizeReconnectionVertex();

				return true;
			}
		}

		// sample direction
		if (p.mLength < gPushConstants.mMaxDepth && bsdf.canEvaluate()) {
            s = bsdf.sampleDirection<false>(rng.nextFloat().xyz, vertex.mLocalDirIn);
            if (!isMedium) {
				s.mReflectance *= vertex.mShadingData.shadingNormalCorrection<false>(vertex.mLocalDirIn, s.mDirection);
                s.mDirection = vertex.mShadingData.toWorld(s.mDirection);
            }
		}

		return false;
	}

	PathVertex TraceNext(const PathVertex vertex, inout RandomSampler rng, const float3 direction, inout float3 throughput) {
		const float3 origin = (gHasMedia && vertex.mShadingData.isMedium()) ?
			vertex.mShadingData.mPosition :
			rayOffset(vertex.mShadingData.mPosition, vertex.mShadingData.getGeometryNormal(), direction);

		uint medium = vertex.mCurrentMedium;

		IntersectionResult isect;
		float transDirPdf, transNeePdf;
		mScene.traceScatteringRay(makeRay(origin, direction), rng, medium, throughput, transDirPdf, transNeePdf, isect);
		isect.mShadingData.mTexcoordScreenSize = 0; // disable mipmaps for now

		return PathVertex(mScene, isect.mShadingData, isect.mInstancePrimitiveIndex, medium, -direction);
	}


	// Generates a path with unidirectional path tracing
	Path GeneratePath(PathVertex vertex, inout RandomSampler rng) {
		uint diffuseBounces = 0;
		bool wasLastDiffuse = false;

		Path p;
		p.mRng = rng;
		p.mContribution = 1;
		p.mPdfW = 1;
		p.mThroughputAtReconnectionVertex = 0;

        for (p.mLength = 1; p.mLength <= gPushConstants.mMaxDepth; p.mLength++) {
            DirectionSampleRecord s;
			bool isDiffuse;

			if (gHasMedia && vertex.mShadingData.isMedium()) {
                const Medium m = Medium(mScene, vertex.mShadingData.getMaterialAddress());
				if (SampleNext(p, vertex, m, rng, s))
					return p;
				isDiffuse = !m.isSingular();
			} else if (vertex.mShadingData.isSurface()) {
				const PackedMaterialData m = mScene.LoadMaterial(vertex.mShadingData);
				if (SampleNext(p, vertex, m, rng, s))
					return p;
                isDiffuse = !m.isSingular();
            } else {
                // ray missed scene
                float envPdfW;
                const float3 le = mScene.EvaluateEnvironment(-vertex.mLocalDirIn, envPdfW);
				p.mContribution *= le;
				// store radiance at reconnection vertex, if any
				if (p.ReconnectionVertexFound())
					p.FinalizeReconnectionVertex();
                return p;
            }

			if (p.mLength >= gPushConstants.mMaxDepth || s.mFwdPdfW <= 0 || all(s.mReflectance <= 0))
				break;

			p.mContribution *= s.mReflectance;
			p.mPdfW *= s.mFwdPdfW;

			if (isDiffuse) {
				// store first reconnectable vertex
				if (wasLastDiffuse && !p.ReconnectionVertexFound())
					p.MakeReconnectionVertex(mScene, vertex, s);

				diffuseBounces++;
				if (diffuseBounces > gPushConstants.mMaxDiffuseBounces)
					break;
			}
			wasLastDiffuse = isDiffuse;

			// trace ray
			vertex = TraceNext(vertex, rng, s.mDirection, p.mContribution);
		}

		p.mContribution = 0;
		p.mPdfW = 0;
		return p;
    }

    Path GenerateShiftedPath(PathVertex vertex, RandomSampler rng, const ReconnectionVertex rcv, const ShiftType shiftType, out float jacobian) {
		return {};
	}
    Path GenerateShiftedPath(const PathVertex vertex, const uint2 candidatePixel, const ShiftType shiftType, out float jacobian) {
		const RandomSampler candidateRng = reinterpret<RandomSampler>(mFramebuffer.mPrevReservoirDataGI[0][candidatePixel]);
		ReconnectionVertex candidateRcv;
		if (gShiftType != ShiftType::eRandomReplay) {
			// load reconnection vertex
			const float4 data0 = mFramebuffer.mPrevReservoirDataGI[1][candidatePixel];
			candidateRcv.mLocalPosition = data0.xyz;
			candidateRcv.mInstancePrimitiveIndex = asuint(data0.w);
			candidateRcv.mPackedData = asuint(mFramebuffer.mPrevReservoirDataGI[2][candidatePixel]);
        }
        return GenerateShiftedPath(vertex, candidateRng, candidateRcv, shiftType, jacobian);
	}

	float3 EstimateRadiance(const PathVertex vertex, inout RandomSampler rng) {
		const Path p = GeneratePath(vertex, rng);
		return p.mContribution * safe_divide(1, p.mPdfW);
	}

	float3 EstimateRadianceReSTIR(const PathVertex vertex, inout RandomSampler rng) {
		// (re)sample canonical path
		Path p = GeneratePath(vertex, rng);
		float wsum = luminance(p.mContribution) * safe_divide(1, p.mPdfW);
		for (uint i = 1; i < gPushConstants.mGICandidateSamples; i++) {
			const Path xi = GeneratePath(vertex, rng);
			const float wi = luminance(xi.mContribution) * safe_divide(1, xi.mPdfW);
			wsum += wi;
			if (rng.nextFloat().x * wsum < wi) {
				p = xi;
			}
		}

		float W = safe_divide(wsum, luminance(p.mContribution) * gPushConstants.mGICandidateSamples);

		// reuse
		#ifdef gReSTIR_GI_Reuse

		if (gPushConstants.mReservoirHistoryValid != 0) {

			wsum = m0 * W * luminance(p.mContribution);

			for (uint i = 0; i < gPushConstants.mGIReuseSamples; i++) {
				const float2 discRng = rng.nextFloat().xy;
				const int2 qi = int2(mFramebuffer.mPrevUVs[sPixelIndex] * gPushConstants.mOutputExtent + sampleConcentricDisc(discRng.x, discRng.y) * gPushConstants.mGIReuseRadius);
				if (all(qi >= 0) && all(qi < gPushConstants.mOutputExtent)) {
                    float jacobian;
                    const Path xi = GenerateShiftedPath(vertex, qi, gShiftType, jacobian);
                    const float wi = mi * luminance(xi.mContribution) * Wi * jacobian;
                    wsum += wi;
					if (rng.nextFloat().x * wsum < wi) {
						p = xi;
					}
				}
			}

			W = safe_divide(wsum, luminance(p.mContribution));
		}

		// store for reuse
		mFramebuffer.mReservoirDataGI[0][sPixelIndex] = reinterpret<float4>(p.mRng);
		if (gShiftType != ShiftType::eRandomReplay) {
			// store reconnection vertex
			mFramebuffer.mReservoirDataGI[1][sPixelIndex] = float4(p.mReconnectionVertex.mLocalPosition, asfloat(p.mReconnectionVertex.mInstancePrimitiveIndex));
			mFramebuffer.mReservoirDataGI[2][sPixelIndex] = asfloat(p.mReconnectionVertex.mPackedData);
		}

		#endif

		return p.mContribution * W;
	}
}