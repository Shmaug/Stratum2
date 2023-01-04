#define gPerformanceCounters (false)

#include "compat/path_tracer.h"
struct PushConstants {
    uint mLightSubPathCount;
    float mLineRadius;
    uint mHashGridCellCount;
    uint mVmIteration;

    float4 mSceneSphere;
    float mHashGridRadiusFactor;
    float mHashGridRadiusAlpha;

    float VcmMergeRadius() {
        // Setup our radius, 1st iteration has aIteration == 0, thus offset
        float radius = mHashGridRadiusFactor * mSceneSphere[3];
        radius /= pow(float(mVmIteration + 1), 0.5f * (1 - mHashGridRadiusAlpha));
        // Purely for numeric stability
        return max(radius, 1e-7f);
    }
};
[[vk::push_constant]] ConstantBuffer<PushConstants> gPushConstants;

#include "common/hashgrid.hlsli"

struct Params {
	StructuredBuffer<ViewData> mViews;
	StructuredBuffer<TransformData> mViewTransforms;
	StructuredBuffer<TransformData> mViewInverseTransforms;

	RWStructuredBuffer<VcmVertex> mLightVertices;
	RWStructuredBuffer<uint> mLightPathLengths;
	HashGrid<uint> mLightHashGrid;
    Texture2D<float4> mDepth;
};

ParameterBlock<Params> gParams;

struct VSIn {
    float3 p : POSITION;
};
struct VSOut {
    float4 position : SV_Position;
    float3 color : COLOR0;
    float3 camPos : TEXCOORD0;
};


[shader("vertex")]
VSOut LightVertexVS(VSIn v, uint instanceID: SV_InstanceID, uint vertexID: SV_VertexID) {
	static const float2 gStripVertices[6] = {
		float2(0, 0),
		float2(1, 1),
		float2(1, 0),

		float2(0, 1),
		float2(1, 1),
		float2(0, 0) };

    VSOut o;

	const uint lightPathIndex = instanceID;
	const uint count = gParams.mLightPathLengths[lightPathIndex];

    if (count < 2) {
        o.position = POS_INFINITY;
        o.color = 0;
        return o;
    }

	const uint lightVertexIndex = min(count - 2, vertexID / 6);
	const VcmVertex lightVertex  = gParams.mLightVertices[lightVertexIndex * gPushConstants.mLightSubPathCount + lightPathIndex];
    const VcmVertex lightVertex1 = gParams.mLightVertices[(lightVertexIndex + 1) * gPushConstants.mLightSubPathCount + lightPathIndex];
    const float3 cameraPos  = gParams.mViewInverseTransforms[0].transformPoint(lightVertex.mShadingData.mPosition);
    const float3 cameraPos1 = gParams.mViewInverseTransforms[0].transformPoint(lightVertex1.mShadingData.mPosition);

    const float3 dir = cameraPos1 - cameraPos;
    float3 up = cross(dir, float3(0,0,1));
    up.z = 0;
    up = normalize(up);

    const float2 p = gStripVertices[vertexID % 6];

    const float misFactor = lerp(lightVertex.dVC, lightVertex1.dVC, p.x);
    const float r = gPushConstants.mLineRadius;// * exp(-abs(misFactor));

    o.camPos = lerp(cameraPos, cameraPos1, p.x) + r * up * (p.y * 2 - 1);
    o.position = gParams.mViews[0].mProjection.projectPoint(o.camPos);
    o.position.y = -o.position.y;
    o.color = lerp(lightVertex.mThroughput, lightVertex1.mThroughput, p.x);
    return o;
}

[shader("fragment")]
float4 LightVertexFS(VSOut i) : SV_Target {
    if (length(i.camPos) > reinterpret<DepthData>(gParams.mDepth[int2(i.position.xy)]).mDepth)
        discard;
    return float4(i.color/luminance(i.color), 1);
}