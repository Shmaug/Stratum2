#include "gi.slang"

extension PathTracer {
	void Render(const uint2 pixelIndex) {
		sPixelIndex = pixelIndex;
		if (any(sPixelIndex >= gPushConstants.mOutputExtent))
			return;

		RandomSampler rng = RandomSampler(gPushConstants.mRandomSeed, sPixelIndex);

		Spectrum throughput;
		const PathVertex primaryVertex = SampleVisibility(sPixelIndex, rng, throughput);

		if (primaryVertex.mShadingData.isEnvironment()) {
			float tmp;
			mFramebuffer.mAlbedo[sPixelIndex] = float4(0, 0, 0, 1);
			mFramebuffer.mOutput[sPixelIndex] = float4(throughput * mScene.EvaluateEnvironment(-primaryVertex.mLocalDirIn, tmp), 1);
			return;
		}

        mFramebuffer.mOutput[sPixelIndex] = float4(throughput * EstimateRadiance(primaryVertex, rng), 1);
	}
};

ParameterBlock<PathTracer> gPathTracer;

[shader("compute")]
[numthreads(8, 4, 1)]
void Render(uint3 index: SV_DispatchThreadID) {
	gPathTracer.Render(index.xy);
}