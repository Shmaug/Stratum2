#include "compat/scene.h"
#include "compat/image_value.h"
#include "compat/material.h"
#include "common/intersection.hlsli"

struct PushConstants {
	uint mViewCount;
    uint mRandomSeed;
    float mAODistance;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> gPushConstants;

struct RenderParams {
    StructuredBuffer<ViewData> mViews;
    StructuredBuffer<TransformData> mViewTransforms;
    StructuredBuffer<TransformData> mPrevViewInverseTransforms;
	RWTexture2D<float4> mOutput;
    RWTexture2D<float4> mAlbedo;
	RWTexture2D<float2> mPrevUVs;
	RWTexture2D<uint2> mVisibility;
	RWTexture2D<float4> mDepth;
};
ParameterBlock<RenderParams> gRenderParams;

static uint2 gOutputExtent;

void StoreAuxillaryData(const uint2 index, const IntersectionResult isect, const float3 albedo) {
    float2 prevPixelCoord = index + 0.5;

    float prevDepth = isect.mDistance;
    if (isect.mInstanceIndex != INVALID_INSTANCE) {
        const float3 prevCameraPos = gRenderParams.mPrevViewInverseTransforms[0].transformPoint(gScene.mInstanceMotionTransforms[isect.mInstanceIndex].transformPoint(isect.mShadingData.mPosition));
        prevDepth = length(prevCameraPos);
        gRenderParams.mViews[0].toRaster(prevCameraPos, prevPixelCoord);
    }

    gRenderParams.mPrevUVs[index] = prevPixelCoord / gOutputExtent;

    VisibilityData v;
    v.mInstancePrimitiveIndex = isect.mInstancePrimitiveIndex;
    v.mPackedNormal = isect.mShadingData.mPackedShadingNormal;
    gRenderParams.mVisibility[index] = reinterpret<uint2>(v);

    DepthData d;
    d.mDepth = isect.mDistance;
    d.mPrevDepth = prevDepth;
    d.mDepthDerivative = 1;
    gRenderParams.mDepth[index] = reinterpret<float4>(d);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void render(uint3 index: SV_DispatchThreadID, uint group_index: SV_GroupIndex) {
    gRenderParams.mOutput.GetDimensions(gOutputExtent.x, gOutputExtent.y);
    if (any(index.xy >= gOutputExtent)) return;
    const uint viewIndex = getViewIndex(index.xy, gOutputExtent, gPushConstants.mViewCount);


	float2 uv;
	const float3 localDir = gRenderParams.mViews[viewIndex].toWorld(index.xy + 0.5, /*out*/ uv);
	const TransformData t = gRenderParams.mViewTransforms[viewIndex];

    RayDesc ray;
    ray.Origin = float3(t.m[0][3], t.m[1][3], t.m[2][3]);
    ray.TMin = 0;
	ray.Direction = normalize(t.transformVector(localDir));
    ray.TMax = POS_INFINITY;

    float3 color = 0;

    IntersectionResult isect;
    const bool hit = gScene.traceRay(ray, true, isect);

    if (hit) color = isect.mShadingData.getShadingNormal() * .5 + .5;

	StoreAuxillaryData(index.xy, isect, hit ? color : 1);

    if (hit) {
		const bool front = dot(isect.mShadingData.getGeometryNormal(), ray.Direction) <= 0;

		RandomSampler rng = RandomSampler(gPushConstants.mRandomSeed, index.xy);
        ray.Direction = isect.mShadingData.toWorld(sampleCosHemisphere(rng.nextFloat(), rng.nextFloat()));
        if (!front) ray.Direction = -ray.Direction;

		ray.Origin = rayOffset(isect.mShadingData.mPosition, isect.mShadingData.getGeometryNormal(), ray.Direction);

		if (gScene.traceRay(ray, true, isect))
        	color *= gPushConstants.mAODistance*isect.mDistance / (1 + gPushConstants.mAODistance*isect.mDistance);
    }

	gRenderParams.mOutput[index.xy] = float4(color, 1);
}