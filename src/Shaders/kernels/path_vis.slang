#include "compat/scene.h"
#include "common/shading_data.hlsli"

struct PackedLightVertex {
    float3 mLocalPosition;
    uint mInstancePrimitiveIndex;
    float3 mThroughput;
    uint mPackedLocalDirIn;
    uint mPathLength;
    float dVC;
    float dVCM;
    float pad;

    uint getInstanceIndex() { return BF_GET(mInstancePrimitiveIndex, 0, 16); }
    uint getPrimitiveIndex() { return BF_GET(mInstancePrimitiveIndex, 16, 16); }
    float3 getLocalDirIn() { return unpackNormal(mPackedLocalDirIn); }
    ShadingData getShadingData() {
        const uint instanceIndex = getInstanceIndex();
        ShadingData sd = gScene.makeShadingData(gScene.mInstances[instanceIndex], gScene.mInstanceTransforms[instanceIndex], mLocalPosition, getPrimitiveIndex());
        sd.mTexcoordScreenSize = 0;
        return sd;
    }
};

struct Params {
	StructuredBuffer<ViewData> mViews;
    StructuredBuffer<TransformData> mViewInverseTransforms;

    RWStructuredBuffer<PackedLightVertex> mLightVertices;

    RWStructuredBuffer<uint> mCounters; // 0 -> light vertices, 1 -> shadow rays
    Texture2D<float4> mDepth;
};

ParameterBlock<Params> gParams;

struct PushConstants {
    uint mDepth;
    float mVertexPercent;
    float mLineRadius;
    float mLineLength;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> gPushConstants;

struct VSOut {
    float4 position : SV_Position;
    float3 color : COLOR0;
    float3 camPos : TEXCOORD0;
};

[shader("vertex")]
VSOut LightVertexVS(float3 p: POSITION, uint vertexID: SV_VertexID) {
	static const float2 gStripVertices[6] = {
		float2(0, 0),
		float2(1, 1),
		float2(1, 0),

		float2(0, 1),
		float2(1, 1),
		float2(0, 0)
	};

    VSOut o;

    const uint segmentIndex = vertexID / 6;

    if (segmentIndex >= uint(gPushConstants.mVertexPercent * gParams.mCounters[1])) {
        o.position = float4(0,0,0,0);
        o.color = 0;
        o.camPos = 0;
        return o;
    }

    const PackedLightVertex lightVertex = gParams.mLightVertices[segmentIndex];

    if (gPushConstants.mDepth != -1 && lightVertex.mPathLength != gPushConstants.mDepth) {
        o.position = float4(0, 0, 0, 0);
        o.color = 0;
        o.camPos = 0;
        return o;
	}

    const ShadingData sd = lightVertex.getShadingData();

    const float3 cameraPos = gParams.mViewInverseTransforms[0].transformPoint(sd.mPosition);

    o.color = lightVertex.mThroughput / luminance(lightVertex.mThroughput);

    const float3 dir = gParams.mViewInverseTransforms[0].transformVector(sd.toWorld(lightVertex.getLocalDirIn()));
    float3 up = cross(dir, float3(0,0,1));
    up.z = 0;
    const float d = 1;//length(cameraPos);///gParams.mViews[0].imagePlaneDist();
    const float2 p = gStripVertices[vertexID % 6];
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