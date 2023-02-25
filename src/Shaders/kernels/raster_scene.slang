#include "common/shading_data.hlsli"
#include "common/material.hlsli"

struct PushConstants {
    uint mViewIndex;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> gPushConstants;

struct Params {
	StructuredBuffer<ViewData> mViews;
	StructuredBuffer<TransformData> mViewTransforms;
	StructuredBuffer<TransformData> mViewInverseTransforms;
};
ParameterBlock<Params> gParams;

struct VSOut {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float3 color : COLOR0;
    nointerpolation uint imageIndex : TEXCOORD1;
};

[shader("vertex")]
VSOut vsmain(uint instanceID: SV_InstanceID, uint vertexID: SV_VertexID) {
    const MeshInstanceData instance = reinterpret<MeshInstanceData>(gScene.mInstances[instanceID]);

    const MeshVertexInfo vertexInfo = gScene.mMeshVertexInfo[instance.vertexInfoIndex()];
    const uint index = gScene.LoadTriangleIndicesUniform(vertexInfo, vertexID / 3)[vertexID % 3];
    const float3 vertex = LoadVertexAttribute<float3>(gScene.mVertexBuffers[vertexInfo.positionBuffer()], vertexInfo.positionOffset(), vertexInfo.positionStride(), index);
    const float2 uv     = LoadVertexAttribute<float2>(gScene.mVertexBuffers[vertexInfo.texcoordBuffer()], vertexInfo.texcoordOffset(), vertexInfo.texcoordStride(), index);

    const TransformData objectToCamera = tmul(gParams.mViewInverseTransforms[gPushConstants.mViewIndex], gScene.mInstanceTransforms[instanceID]);

    VSOut o;
    o.position = gParams.mViews[gPushConstants.mViewIndex].mProjection.projectPoint(objectToCamera.transformPoint(vertex));
    o.position.y = -o.position.y;
    o.uv = uv;

    uint4 imageIndices;
    const PackedMaterialData material = gScene.LoadMaterial(instance.getMaterialAddress(), imageIndices);

    o.color = material.getBaseColor();
    o.imageIndex = imageIndices[0];
    return o;
}

[shader("fragment")]
float4 fsmain(VSOut i) : SV_Target {
    float3 c = i.color;
	if (i.imageIndex < gImageCount)
	    c *= gScene.mImages[i.imageIndex].Sample(gScene.mStaticSampler, i.uv).rgb;
    return float4(c, 1);
}