#include "common/shading_data.hlsli"
#include "common/material.hlsli"
#include "common/noneuclid.hlsli"

struct Params {
	StructuredBuffer<ViewData> mViews;
	StructuredBuffer<TransformData> mViewTransforms;
	StructuredBuffer<TransformData> mViewInverseTransforms;
};
ParameterBlock<Params> gParams;


struct PushConstants {
    uint mViewIndex;
    uint mInstanceIndex;
    uint mMaterialAddress;
    uint pad;
    float mLorentzSign;
    float mScale;
    uint pad1;
    uint pad2;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> gPushConstants;

struct VSOut {
    float4 position : SV_Position;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
    nointerpolation uint primId : TEXCOORD1;
};

[shader("vertex")]
VSOut vsmain(uint vertexID: SV_VertexID) {
    const MeshInstanceData instance = reinterpret<MeshInstanceData>(gScene.mInstances[gPushConstants.mInstanceIndex]);
    const MeshVertexInfo vertexInfo = gScene.mMeshVertexInfo[instance.vertexInfoIndex()];
    const uint index = gScene.LoadTriangleIndicesUniform(vertexInfo, vertexID / 3)[vertexID % 3];
    const float3 vertex = LoadVertexAttribute<float3>(gScene.mVertexBuffers[vertexInfo.positionBuffer()], vertexInfo.positionOffset(), vertexInfo.positionStride(), index);
    const float3 normal = LoadVertexAttribute<float3>(gScene.mVertexBuffers[vertexInfo.normalBuffer()]  , vertexInfo.normalOffset()  , vertexInfo.normalStride()  , index);
    const float2 uv     = LoadVertexAttribute<float2>(gScene.mVertexBuffers[vertexInfo.texcoordBuffer()], vertexInfo.texcoordOffset(), vertexInfo.texcoordStride(), index);

    const float3 worldPos = gScene.mInstanceTransforms[gPushConstants.mInstanceIndex].transformPoint(vertex);

    const float4 viewPos = mul(
        ViewMat(gParams.mViewTransforms[gPushConstants.mViewIndex].transformPoint(0), gParams.mViewInverseTransforms[gPushConstants.mViewIndex], gPushConstants.mLorentzSign),
        euclidianToCurved(worldPos * gPushConstants.mScale, gPushConstants.mLorentzSign));

    ProjectionData proj = gParams.mViews[gPushConstants.mViewIndex].mProjection;
    proj.mFarPlane  = sin(proj.mFarPlane*gPushConstants.mScale);
    proj.mNearPlane = sin(proj.mNearPlane*gPushConstants.mScale);

    VSOut o;
    o.position = proj.projectPoint(viewPos);
    o.position.y = -o.position.y;
    o.normal = normal;
    o.uv = uv;
    o.primId = vertexID / 3;
    return o;
}

void ApplyAlphaMask(const float2 uv) {
	#ifdef gUseAlphaMask
    uint alphaMask;
    float alphaCutoff;
    gScene.getMaterialAlphaMask(gPushConstants.mMaterialAddress, alphaMask, alphaCutoff);
    if (alphaMask < gImageCount) {
        if (gScene.mImage1s[alphaMask].Sample(gScene.mStaticSampler, uv) < alphaCutoff)
            discard;
    }
	#endif
}

[shader("fragment")]
void fsmain(VSOut i, out float4 outputColor: SV_Target0, out uint2 visibility: SV_Target1) {
    ApplyAlphaMask(i.uv);

    const PackedMaterialData material = gScene.LoadMaterialUniform(gPushConstants.mMaterialAddress, i.uv);

    outputColor = float4(material.getBaseColor() + material.getEmission(), 1);

    VisibilityData vis;
    vis.mInstancePrimitiveIndex = (gPushConstants.mInstanceIndex & 0xFFFF) | (i.primId << 16);
    vis.mPackedNormal = packNormal(i.normal);
    visibility = reinterpret<uint2>(vis);
}