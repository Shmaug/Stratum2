#include "gi.slang"

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

[shader("compute")]
[numthreads(8, 4, 1)]
void Render(uint3 index: SV_DispatchThreadID) {
	if (any(index.xy >= gPushConstants.mOutputExtent))
        return;

    RandomSampler rng = RandomSampler(gPushConstants.mRandomSeed, index.xy);

    float3 throughput;
    const PathVertex primaryVertex = SampleVisibility(index.xy, rng, throughput);

	gRenderParams.StoreAuxillaryData(primaryVertex.mShadingData, index.xy, primaryVertex.mInstancePrimitiveIndex);

    if (primaryVertex.mShadingData.isEnvironment()) {
        float tmp;
    	gRenderParams.mAlbedo[index.xy] = float4(0, 0, 0, 1);
        gRenderParams.mOutput[index.xy] = float4(throughput * gScene.EvaluateEnvironment(-primaryVertex.mLocalDirIn, tmp), 1);
        return;
	}

	#ifdef gFullGI

    gRenderParams.mOutput[index.xy] = float4(EstimateRadianceFn(primaryVertex, rng, index.xy), 1);

	#else // gFullGI

    // load load material
    if (gHasMedia && primaryVertex.mShadingData.isMedium()) {
        const Medium m = Medium(primaryVertex.mShadingData.getMaterialAddress());
        gRenderParams.mAlbedo[index.xy] = float4(m.albedo(), 1);
        gRenderParams.mOutput[index.xy] = float4(DirectIlluminationFn(primaryVertex, m, rng, index.xy), 1);
    } else {
        const PackedMaterialData m = gScene.LoadMaterial(primaryVertex.mShadingData);
        gRenderParams.mAlbedo[index.xy] = float4(m.albedo(), 1);
        gRenderParams.mOutput[index.xy] = float4(DirectIlluminationFn(primaryVertex, m, rng, index.xy), 1);
	}

	#endif
}