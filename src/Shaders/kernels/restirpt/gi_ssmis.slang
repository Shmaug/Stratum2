#include "di.slang"

struct PathState {
    float3 radiance;
    uint depth;
    float3 throughput;
    uint diffuseBounces;
    float cachedMisWeight;
};

extension PathTracer {
	// samples a direction and samples direct illumination (nee)
	DirectionSampleRecord ProcessVertex(const PathVertex vertex, const Interaction bsdf, inout RandomSampler rng, inout PathState state) {
		DirectionSampleRecord r;
		r.mReflectance = 0;

		const bool isMedium = gHasMedia && vertex.mShadingData.isMedium();
		const float3 localDirIn = vertex.mLocalDirIn;

		if (isMedium || localDirIn.z > 0) {
			state.radiance += state.cachedMisWeight * state.throughput * bsdf.emission();
		}

        if (!bsdf.canEvaluate())
			return r;

        state.depth++;
        if (!bsdf.isSingular()) {
            state.diffuseBounces++;
            if (state.diffuseBounces > gPushConstants.mMaxDiffuseBounces)
                return r;
		}
        if (state.depth >= gPushConstants.mMaxDepth)
			return r;

		float bsdfSamplePdf = bsdf.isSingular() ? 1 : 0.5;
		if (rng.nextFloat().x < bsdfSamplePdf) {
			state.throughput /= bsdfSamplePdf;
			r = bsdf.sampleDirection<false>(rng.nextFloat().xyz, localDirIn);
			if (!isMedium) {
				r.mReflectance *= vertex.mShadingData.shadingNormalCorrection<false>(localDirIn, r.mDirection);
				r.mDirection = vertex.mShadingData.toWorld(r.mDirection);
			}
		} else {
			state.throughput /= 1 - bsdfSamplePdf;

			const DirectIlluminationSample s = DirectIllumination(vertex, bsdf, rng);
			state.radiance += state.throughput * s.mRadiance * (s.mMisWeight * safe_divide(s.G, s.mPdfA));
		}

		return r;
    }
    DirectionSampleRecord ProcessVertex(const PathVertex vertex, inout RandomSampler rng, inout PathState state) {
        if (vertex.mShadingData.isEnvironment()) {
            DirectionSampleRecord r;
            r.mReflectance = 0;
            float tmp;
            state.radiance += state.cachedMisWeight * state.throughput * mScene.EvaluateEnvironment(-vertex.mLocalDirIn, tmp);
			return r;
		}
		if (gHasMedia && vertex.mShadingData.isMedium())
			return ProcessVertex(vertex, Medium(mScene, vertex.mShadingData.getMaterialAddress()), rng, state);
		else
			return ProcessVertex(vertex, mScene.LoadMaterial(vertex.mShadingData), rng, state);
	}
	PathVertex TraceNext(const PathVertex vertex, inout RandomSampler rng, const float3 direction, inout float3 throughput, out float transDirPdf, out float directPdfA, out float WtoA) {
		const float3 origin = (gHasMedia && vertex.mShadingData.isMedium()) ?
			vertex.mShadingData.mPosition :
			rayOffset(vertex.mShadingData.mPosition, vertex.mShadingData.getGeometryNormal(), direction);

		uint medium = vertex.mCurrentMedium;

		IntersectionResult isect;
		float transNeePdf;
		const bool hit = mScene.traceScatteringRay(makeRay(origin, direction), rng, medium, throughput, transDirPdf, transNeePdf, isect);
		isect.mShadingData.mTexcoordScreenSize = 0; // disable mipmaps for now

		PathVertex v = PathVertex(mScene, isect.mShadingData, isect.mInstancePrimitiveIndex, medium, -direction);

		if (hit) {
			directPdfA = isect.LightSamplePdfA();
			WtoA = abs((gHasMedia && v.mShadingData.isMedium()) ? 1 : v.mLocalDirIn.z) / pow2(isect.mDistance);
		} else {
			mScene.EvaluateEnvironment(direction, directPdfA);
			if (gLightCount > 0)
				directPdfA *= gEnvironmentSampleProbability;
			WtoA = 1;
		}

		return v;
	}

    float3 EstimateRadianceSSMIS(PathVertex vertex, inout RandomSampler rng, const uint2 pixelIndex) {
        PathState state;
		state.radiance = 0;
		state.throughput = 1;
        state.cachedMisWeight = 1;
        state.diffuseBounces = 0;
        for (state.depth = 1; state.depth <= gPushConstants.mMaxDepth;) {
            DirectionSampleRecord s;
            if (state.depth == 1) {
				// get albedo from material
				if (gHasMedia && vertex.mShadingData.isMedium()) {
					const Medium m = Medium(mScene, vertex.mShadingData.getMaterialAddress());
					mFramebuffer.mAlbedo[pixelIndex] = float4(m.albedo(), 1);
					s = ProcessVertex(vertex, m, rng, state);
				} else {
					const PackedMaterialData m = mScene.LoadMaterial(vertex.mShadingData);
					mFramebuffer.mAlbedo[pixelIndex] = float4(m.albedo(), 1);
					s = ProcessVertex(vertex, m, rng, state);
				}
        	} else {
                s = ProcessVertex(vertex, rng, state);
            }

			// trace ray towards s.mDirection

			if (all(s.mReflectance <= 0))
				break;

			state.throughput *= s.mReflectance / s.mFwdPdfW;

			float transDirPdf, directPdfA, WtoA;
			vertex = TraceNext(vertex, rng, s.mDirection, state.throughput, transDirPdf, directPdfA, WtoA);
			if (transDirPdf <= 0)
                break;
            state.throughput /= transDirPdf;

            // bsdf sampling mis weight
            state.cachedMisWeight = s.isSingular() ? 1 : DirectMis(s.mFwdPdfW, directPdfA / WtoA);
		}

		return state.radiance;
	}
}