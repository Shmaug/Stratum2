#include "gi.slang"

#ifdef gReSTIR_DI
#define DirectIlluminationFn DirectIlluminationReSTIR
#else
#define DirectIlluminationFn DirectIllumination
#endif

#ifdef gReSTIR_GI
#define EstimateRadianceFn EstimateRadianceReSTIR
#else
#define EstimateRadianceFn EstimateRadiance
#endif

extension PathTracer {
	void Render(const uint2 pixelIndex) {
		sPixelIndex = pixelIndex;
		if (any(sPixelIndex >= gPushConstants.mOutputExtent))
			return;

		RandomSampler rng = RandomSampler(gPushConstants.mRandomSeed, sPixelIndex);

		float3 throughput;
		const PathVertex primaryVertex = SampleVisibility(sPixelIndex, rng, throughput);

		if (primaryVertex.mShadingData.isEnvironment()) {
			float tmp;
			mFramebuffer.mAlbedo[sPixelIndex] = float4(0, 0, 0, 1);
			mFramebuffer.mOutput[sPixelIndex] = float4(throughput * mScene.EvaluateEnvironment(-primaryVertex.mLocalDirIn, tmp), 1);
			return;
		}

		#ifdef gNoGI

		// get albedo from first bounce

        DirectIlluminationSample s;
		if (gHasMedia && primaryVertex.mShadingData.isMedium()) {
			const Medium m = Medium(mScene, primaryVertex.mShadingData.getMaterialAddress());
			mFramebuffer.mAlbedo[sPixelIndex] = float4(m.albedo(), 1);
			s = DirectIlluminationFn(primaryVertex, m, rng);
		} else {
			const PackedMaterialData m = mScene.LoadMaterial(primaryVertex.mShadingData);
			mFramebuffer.mAlbedo[sPixelIndex] = float4(m.albedo(), 1);
			s = DirectIlluminationFn(primaryVertex, m, rng);
		}
		mFramebuffer.mOutput[sPixelIndex] = float4(throughput * s.mRadiance * safe_divide(s.G, s.mPdfA), 1);

		#else // gNoGI

        mFramebuffer.mOutput[sPixelIndex] = float4(throughput * EstimateRadianceFn(primaryVertex, rng), 1);

		#endif
	}
};

ParameterBlock<PathTracer> gPathTracer;

[shader("compute")]
[numthreads(8, 4, 1)]
void Render(uint3 index: SV_DispatchThreadID) {
	gPathTracer.Render(index.xy);
}