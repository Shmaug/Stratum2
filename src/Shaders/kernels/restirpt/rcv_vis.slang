#include "rcv.slang"

ParameterBlock<SceneParameters> gScene;
ParameterBlock<RenderParams> gFramebuffer;

struct RasterPushConstants {
    uint2 mOutputExtent;
    uint mPixelIndex;
    uint mDepth;
    float mLineRadius;
    float mLineLength;
};
[[vk::push_constant]] ConstantBuffer<RasterPushConstants> gRasterPushConstants;

struct VSOut {
    float4 position : SV_Position;
    float3 color    : COLOR0;
    float3 camPos   : TEXCOORD0;
};

[shader("vertex")]
VSOut ShowReconnectionVertexVS(float3 p: POSITION, uint vertexID: SV_VertexID) {
    static const float2 gStripVertices[6] = {
        float2(0, 0),
        float2(0.25, 1),
        float2(1, 0),

        float2(-0.25, 1),
        float2(0.25, 1),
        float2(0, 0)
	};

    VSOut o;

    const uint pixelIndex1d = vertexID / 6;

    if (pixelIndex1d >= gRasterPushConstants.mOutputExtent.x * gRasterPushConstants.mOutputExtent.y || (gRasterPushConstants.mPixelIndex != -1 && gRasterPushConstants.mPixelIndex != pixelIndex1d)) {
        o.position = float4(0,0,0,0);
        o.color = 0;
        o.camPos = 0;
        return o;
    }

    const ReconnectionVertex rcv = LoadPrevReconnectionVertex(gFramebuffer, uint2(pixelIndex1d%gRasterPushConstants.mOutputExtent.x, pixelIndex1d/gRasterPushConstants.mOutputExtent.x));

    if ((gRasterPushConstants.mDepth != -1 && rcv.mReconnectionLength != gRasterPushConstants.mDepth) || all(rcv.mRadiance <= 0)) {
        o.position = float4(0, 0, 0, 0);
        o.color = 0;
        o.camPos = 0;
        return o;
	}

    const ShadingData sd = rcv.getShadingData(gScene);

    const float3 pos = gFramebuffer.mViewInverseTransforms[0].transformPoint(sd.mPosition);
    const float2 p = gStripVertices[vertexID % 6];

    o.color = rcv.mRadiance / luminance(rcv.mRadiance);

    const float3 worldDir = rcv.mPathLength == rcv.mReconnectionLength ? sd.getShadingNormal() : sd.toWorld(rcv.mLocalDirOut);
    const float3 dir = gFramebuffer.mViewInverseTransforms[0].transformVector(worldDir);
    float3 up = cross(dir, float3(0,0,1));
    up.z = 0;
    const float d = gRasterPushConstants.mPixelIndex == -1 ? 1 : length(pos);
    o.camPos = pos + normalize(dir) * gRasterPushConstants.mLineLength * d * p.y + normalize(up) * gRasterPushConstants.mLineRadius * d * p.x;

	o.position = gFramebuffer.mViews[0].mProjection.projectPoint(o.camPos);
    o.position.y = -o.position.y;
    return o;
}

[shader("fragment")]
float4 ShowReconnectionVertexFS(VSOut i) : SV_Target {
    if (length(i.camPos) > reinterpret<DepthData>(gFramebuffer.mDepth[int2(i.position.xy)]).mDepth)
        discard;
    return float4(i.color, 1);
}