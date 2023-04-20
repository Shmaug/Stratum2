#ifdef gSingleSampleMIS
#include "gi_ssmis.slang"
#else
#include "gi.slang"
#endif

#ifdef gReSTIR_GI
#define EstimateRadianceFn EstimateRadianceReSTIR
#else
#define EstimateRadianceFn EstimateRadiance
#endif

#ifdef gReSTIR_DI
#define DirectIlluminationFn DirectIlluminationReSTIR
#else
#define DirectIlluminationFn DirectIllumination
#endif

extension PathTracer {
	void Render(const uint2 index) {
		if (any(index >= gPushConstants.mOutputExtent))
			return;

		RandomSampler rng = RandomSampler(gPushConstants.mRandomSeed, index);

		float3 throughput;
		const PathVertex primaryVertex = SampleVisibility(index, rng, throughput);

		mFramebuffer.StoreAuxillaryData(mScene, primaryVertex.mShadingData, index, primaryVertex.mInstancePrimitiveIndex);

		if (primaryVertex.mShadingData.isEnvironment()) {
			float tmp;
			mFramebuffer.mAlbedo[index] = float4(0, 0, 0, 1);
			mFramebuffer.mOutput[index] = float4(throughput * mScene.EvaluateEnvironment(-primaryVertex.mLocalDirIn, tmp), 1);
			return;
		}

		#ifdef gNoGI

        DirectIlluminationSample s;
		if (gHasMedia && primaryVertex.mShadingData.isMedium()) {
			const Medium m = Medium(mScene, primaryVertex.mShadingData.getMaterialAddress());
			mFramebuffer.mAlbedo[index] = float4(m.albedo(), 1);
			s = DirectIlluminationFn(primaryVertex, m, rng, index);
		} else {
			const PackedMaterialData m = mScene.LoadMaterial(primaryVertex.mShadingData);
			mFramebuffer.mAlbedo[index] = float4(m.albedo(), 1);
			s = DirectIlluminationFn(primaryVertex, m, rng, index);
		}
		mFramebuffer.mOutput[index] = float4(throughput * s.mRadiance * safe_divide(s.G, s.mPdfA), 1);

		#else // gNoGI

		#ifdef gSingleSampleMIS
		mFramebuffer.mOutput[index] = float4(throughput * EstimateRadianceSSMIS(primaryVertex, rng, index), 1);
		#else
        mFramebuffer.mOutput[index] = float4(throughput * EstimateRadianceFn(primaryVertex, rng, index), 1);
		#endif

		#endif
	}
};

ParameterBlock<PathTracer> gPathTracer;

[shader("compute")]
[numthreads(8, 4, 1)]
void Render(uint3 index: SV_DispatchThreadID) {
	gPathTracer.Render(index.xy);
}