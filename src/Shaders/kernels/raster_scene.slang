#include "common/shading_data.hlsli"
#include "common/material.hlsli"

struct PushConstants {
    uint mViewIndex;
    uint mInstanceIndex;
    uint mMaterialAddress;
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

    const TransformData objectToCamera = tmul(gParams.mViewInverseTransforms[gPushConstants.mViewIndex], gScene.mInstanceTransforms[gPushConstants.mInstanceIndex]);

    VSOut o;
    o.position = gParams.mViews[gPushConstants.mViewIndex].mProjection.projectPoint(objectToCamera.transformPoint(vertex));
    o.position.y = -o.position.y;
    o.normal = normal;
    o.uv = uv;
    o.primId = vertexID / 3;
    return o;
}

[shader("fragment")]
void fsmain(VSOut i, out float4 outputColor: SV_Target0, out uint2 visibility: SV_Target1) {
    const PackedMaterialData material = gScene.LoadMaterialUniform(gPushConstants.mMaterialAddress, i.uv);

	#ifdef gUseAlphaMask
	uint alphaMask;
	float alphaCutoff;
	gScene.getMaterialAlphaMask(gPushConstants.mMaterialAddress, alphaMask, alphaCutoff);
	if (alphaMask < gImageCount) {
		if (gScene.mImage1s[alphaMask].SampleLevel(gScene.mStaticSampler, i.uv, 0) < alphaCutoff)
			discard;
	}
	#endif

    outputColor = float4(material.getBaseColor() + material.getEmission(), 1);

    VisibilityData vis;
    vis.mInstancePrimitiveIndex = (gPushConstants.mInstanceIndex & 0xFFFF) | (i.primId << 16);
    vis.mPackedNormal = packNormal(i.normal);
    visibility = reinterpret<uint2>(vis);
}