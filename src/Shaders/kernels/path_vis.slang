#include "compat/path_tracer.h"
struct PushConstants {
    uint mLightSubPathCount;
    uint mHashGridCellCount;
    uint mSegmentIndex;
    float mLineRadius;
    float mLineLength;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> gPushConstants;

#include "common/hashgrid.hlsli"
#include "common/scene.hlsli"

struct Params {
	StructuredBuffer<ViewData> mViews;
	StructuredBuffer<TransformData> mViewTransforms;
	StructuredBuffer<TransformData> mViewInverseTransforms;

	RWStructuredBuffer<VcmVertex> mLightVertices;
	RWStructuredBuffer<uint> mLightPathLengths;
    Texture2D<float4> mDepth;
};

ParameterBlock<Params> gParams;

struct VSOut {
    float4 position : SV_Position;
    float3 color : COLOR0;
    float3 camPos : TEXCOORD0;
};

[shader("vertex")]
VSOut LightVertexVS(float3 p: POSITION, uint instanceID: SV_InstanceID, uint vertexID: SV_VertexID) {
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
    const uint segmentIndex = vertexID / 6;

    if (segmentIndex >= count || (gPushConstants.mSegmentIndex != -1 && segmentIndex != gPushConstants.mSegmentIndex)) {
        o.position = float4(0,0,0,0);
        o.color = 0;
        o.camPos = 0;
        return o;
    }

    const float2 p = gStripVertices[vertexID % 6];

	const VcmVertex lightVertex  = gParams.mLightVertices[segmentIndex * gPushConstants.mLightSubPathCount + lightPathIndex];
    const float3 cameraPos  = gParams.mViewInverseTransforms[0].transformPoint(lightVertex.mShadingData.mPosition);

    o.color = lightVertex.mThroughput / luminance(lightVertex.mThroughput);

    const float3 dir = gParams.mViewInverseTransforms[0].transformVector(lightVertex.mShadingData.toWorld(unpackNormal(lightVertex.mLocalDirectionIn)));
    float3 up = cross(dir, float3(0,0,1));
    up.z = 0;

    const float d = length(cameraPos);

    o.camPos = cameraPos + normalize(dir) * gPushConstants.mLineLength * d * p.x + normalize(up) * gPushConstants.mLineRadius * d * (p.y * 2 - 1);

    o.position = gParams.mViews[0].mProjection.projectPoint(o.camPos);
    o.position.y = -o.position.y;
    return o;
}

[shader("fragment")]
float4 LightVertexFS(VSOut i) : SV_Target {
    if (length(i.camPos) > reinterpret<DepthData>(gParams.mDepth[int2(i.position.xy)]).mDepth)
        discard;
    return float4(i.color, 1);
}