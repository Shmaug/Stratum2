#include "di.slang"


struct VertexSampleRecord {
    float3 mLe;
	float mNeePdfW;
    float3 mNeeLe;
	float mNeeMisWeight;

    float3 mDirection;
    float mDirPdfW;
    float3 mBrdf;
    bool isSpecular;
    bool isSpecularLobe;
};

extension PathTracer {
	// samples a direction and samples direct illumination (nee)
	VertexSampleRecord ProcessVertex(const PathVertex vertex, const Interaction bsdf, inout RandomSampler rng, const bool sampleDirection) {
		VertexSampleRecord r;
		r.mLe = 0;
		r.mNeePdfW = 0;
		r.mNeeLe = 0;
		r.mNeeMisWeight = 0;
		r.mDirPdfW = 0;
		r.mBrdf = 0;
		r.isSpecular = bsdf.isSingular();
		r.isSpecularLobe = false;

		const bool isMedium = gHasMedia && vertex.mShadingData.isMedium();
		const float3 localDirIn = vertex.mLocalDirIn;

		if (isMedium || localDirIn.z > 0) {
			r.mLe = bsdf.emission();
		}

		if (sampleDirection && bsdf.canEvaluate()) {
			const DirectionSampleRecord s = bsdf.sampleDirection<false>(rng.nextFloat().xyz, localDirIn);
			r.isSpecularLobe = s.isSingular();
			r.mBrdf = s.mReflectance;
			r.mDirPdfW = s.mFwdPdfW;
			if (!isMedium) {
				r.mBrdf *= vertex.mShadingData.shadingNormalCorrection<false>(localDirIn, s.mDirection);
				r.mDirection = vertex.mShadingData.toWorld(s.mDirection);
			} else
				r.mDirection = s.mDirection;

			#ifdef gNee
			if (!r.isSpecular) {
				const DirectIlluminationSample s = DirectIllumination(vertex, bsdf, rng);
				r.mNeeLe = s.mRadiance;
				r.mNeePdfW = safe_divide(s.mPdfA, s.G);
				r.mNeeMisWeight = s.mMisWeight;
			}
			#endif
		}

		return r;
	}
	VertexSampleRecord ProcessVertex(const PathVertex vertex, inout RandomSampler rng, const bool sampleNext) {
		if (vertex.mShadingData.isEnvironment()) {
			VertexSampleRecord r;
			float tmp;
			r.mLe = mScene.EvaluateEnvironment(-vertex.mLocalDirIn, tmp);
			r.mNeeLe = 0;
			r.mNeePdfW = 0;
			r.mBrdf = 0;
			r.mDirPdfW = 0;
			r.isSpecular = false;
			r.isSpecularLobe = false;
			return r;
		}
		if (gHasMedia && vertex.mShadingData.isMedium())
			return ProcessVertex(vertex, Medium(mScene, vertex.mShadingData.getMaterialAddress()), rng, sampleNext);
		else
			return ProcessVertex(vertex, mScene.LoadMaterial(vertex.mShadingData), rng, sampleNext);
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

	float3 EstimateRadiancePathGuiding(PathVertex vertex, inout RandomSampler rng, const uint2 pixelIndex) {
		float3 radiance = 0;
		float3 throughput = 1;
		float pathPdfW = 1;
		uint diffuseBounces = 0;
		float misWeight = 1;

        for (uint depth = 2; depth <= gPushConstants.mMaxDepth; depth++) {
            VertexSampleRecord s;
            if (depth == 1) {
				// get albedo from material
				if (gHasMedia && vertex.mShadingData.isMedium()) {
					const Medium m = Medium(mScene, vertex.mShadingData.getMaterialAddress());
					mFramebuffer.mAlbedo[pixelIndex] = float4(m.albedo(), 1);
					s = ProcessVertex(vertex, m, rng, depth < gPushConstants.mMaxDepth);
				} else {
					const PackedMaterialData m = mScene.LoadMaterial(vertex.mShadingData);
					mFramebuffer.mAlbedo[pixelIndex] = float4(m.albedo(), 1);
					s = ProcessVertex(vertex, m, rng, depth < gPushConstants.mMaxDepth);
				}
        	} else {
                s = ProcessVertex(vertex, rng, depth < gPushConstants.mMaxDepth);
            }


			// add emission from vertex and nee sample
            radiance += throughput * s.mLe * misWeight / pathPdfW;
			if (!s.isSpecular)
				diffuseBounces++;
            if (diffuseBounces > gPushConstants.mMaxDiffuseBounces)
                break;
			radiance += throughput * s.mNeeLe * safe_divide(s.mNeeMisWeight, pathPdfW * s.mNeePdfW);

			if (s.mDirPdfW <= 0 || all(s.mBrdf <= 0))
				break;

			throughput *= s.mBrdf;
			pathPdfW *= s.mDirPdfW;

			float transDirPdf, directPdfA, WtoA;
			vertex = TraceNext(vertex, rng, s.mDirection, throughput, transDirPdf, directPdfA, WtoA);
			if (transDirPdf <= 0)
				break;
			pathPdfW *= transDirPdf;

			#ifdef gNee
			misWeight = s.isSpecular ? 1 : DirectMis(s.mDirPdfW, directPdfA / WtoA);
			#endif
		}

		return radiance;
	}
}