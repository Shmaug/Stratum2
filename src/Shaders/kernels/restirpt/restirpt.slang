#include "gi.slang"

extension PathTracer {
	void Render(const uint2 pixelIndex, const uint2 randomIndex) {
		sPixelIndex = pixelIndex;

        RandomSampler rng = RandomSampler(gPushConstants.mRandomSeed, randomIndex);

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

// map a thread index to a unique random pixel
uint2 GetOutputIndex(const uint2 threadIndex) {
	//#ifdef gCoherentRNG
    //return pcg4d(uint4(threadIndex, gPushConstants.mRandomSeed, 0)).xy % gPushConstants.mOutputExtent.xy;
	//#endif
	return threadIndex;
}
uint2 GetRandomSeed(const uint2 threadIndex) {
    #ifdef gCoherentRNG
	return WaveReadLaneFirst(threadIndex);
	#else
	return threadIndex;
	#endif
}

[shader("compute")]
[numthreads(8, 4, 1)]
void Render(uint3 threadIndex: SV_DispatchThreadID) {
    const uint2 pixelIndex = GetOutputIndex(threadIndex.xy);
    if (any(pixelIndex.xy >= gPushConstants.mOutputExtent))
		return;

    gPathTracer.Render(pixelIndex, GetRandomSeed(threadIndex.xy));
}