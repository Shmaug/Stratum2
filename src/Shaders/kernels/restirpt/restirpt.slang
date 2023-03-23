#include "gi.slang"

float3 Radiance(const PathVertex vertex, const Interaction bsdf, inout RandomSampler rng, const uint2 index) {
	gRenderParams.mAlbedo[index] = float4(bsdf.albedo(), 1);

    float3 radiance = 0;

    if (vertex.mShadingData.isSurface() && vertex.mLocalDirIn.z > 0)
        radiance += bsdf.emission();

    if (!bsdf.canEvaluate() || gPushConstants.mMaxDepth < 2)
		return radiance;

	#ifdef gDirectLight
    if (!bsdf.isSingular())
        radiance = DirectIllumination(vertex, bsdf, rng, index);
	#endif

	#ifdef gIndirectLight
	radiance += GlobalIllumination(vertex, rng, index);
	#endif

    return radiance;
}

PathVertex SampleVisibility(const uint2 index, inout RandomSampler rng, out float3 throughput) {
    const uint viewIndex = getViewIndex(index, gPushConstants.mOutputExtent, gPushConstants.mViewCount);

    float2 uv;
    const float3 localDir = gRenderParams.mViews[viewIndex].toWorld(index + 0.5, /*out*/ uv);
    const TransformData t = gRenderParams.mViewTransforms[viewIndex];
    const float3 direction = normalize(t.transformVector(localDir));

    uint medium = gRenderParams.mViewMediumIndices[viewIndex];

    IntersectionResult isect;
    float transDirPdf, transNeePdf;
    throughput = 1;
    const bool hit = gScene.traceScatteringRay(makeRay(float3(t.m[0][3], t.m[1][3], t.m[2][3]), direction), rng, medium, throughput, transDirPdf, transNeePdf, isect);
    if (transDirPdf > 0)
        throughput /= transDirPdf;

    isect.mShadingData.mTexcoordScreenSize = 0; // disable mipmaps for now
    return PathVertex(isect.mShadingData, isect.mInstancePrimitiveIndex, medium, -direction);
}

[shader("compute")]
[numthreads(8, 4, 1)]
void Render(uint3 index: SV_DispatchThreadID) {
	if (any(index.xy >= gPushConstants.mOutputExtent))
        return;

    gRenderParams.mOutput[index.xy] = float4(0, 0, 0, 1);
    gRenderParams.mAlbedo[index.xy] = float4(0, 0, 0, 1);

    RandomSampler rng = RandomSampler(gPushConstants.mRandomSeed, index.xy);

    float3 throughput;
    const PathVertex primaryVertex = SampleVisibility(index.xy, rng, throughput);

	gRenderParams.StoreAuxillaryData(primaryVertex.mShadingData, index.xy, primaryVertex.mInstancePrimitiveIndex);

    if (primaryVertex.mShadingData.isEnvironment()) {
        float dirPdfW;
        const float3 emission = gScene.EvaluateEnvironment(-primaryVertex.mLocalDirIn, dirPdfW);
		gRenderParams.mOutput[index.xy].rgb += throughput * emission;
	} else if (gHasMedia && primaryVertex.mShadingData.isMedium())
        gRenderParams.mOutput[index.xy].rgb += throughput * Radiance(primaryVertex, Medium(primaryVertex.mShadingData.getMaterialAddress()), rng, index.xy);
    else
        gRenderParams.mOutput[index.xy].rgb += throughput * Radiance(primaryVertex, gScene.LoadMaterial(primaryVertex.mShadingData), rng, index.xy);
}