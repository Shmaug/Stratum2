#include "common/material.hlsli"
#include "common/shading_data.hlsli"

typedef float3 EuclidianCoordinate;
typedef float4 CurvedCoordinate;

float dotProduct(CurvedCoordinate u, CurvedCoordinate v, const float lorentzSign) {
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2] + lorentzSign * u[3] * v[3];
}

CurvedCoordinate direction(CurvedCoordinate to, CurvedCoordinate from, const float lorentzSign) {
    if (lorentzSign > 0) {
        float cosd = dotProduct(from, to, lorentzSign);
        float sind = sqrt(1 - cosd * cosd);
        return (to - from * cosd) / sind;
    }
    if (lorentzSign < 0) {
        float coshd = -dotProduct(from, to, lorentzSign);
        float sinhd = sqrt(coshd * coshd - 1);
        return (to - from * coshd) / sinhd;
    }
    return normalize(to - from);
}

CurvedCoordinate euclidianToCurved(CurvedCoordinate eucPoint, const float lorentzSign) {
    const EuclidianCoordinate P = eucPoint.xyz;
    const float distance = length(P);
    if (distance < 1e-5f) return eucPoint;
    if (lorentzSign > 0) return float4(P / distance * sin(distance) , cos(distance));
    if (lorentzSign < 0) return float4(P / distance * sinh(distance), cosh(distance));
    return eucPoint;
}
CurvedCoordinate euclidianToCurved(EuclidianCoordinate eucPoint, const float lorentzSign) {
    return euclidianToCurved(CurvedCoordinate(eucPoint[0], eucPoint[1], eucPoint[2], 1), lorentzSign);
}

float4x4 TranslateMatrix(CurvedCoordinate to, const float lorentzSign) {
    if (lorentzSign != 0) {
        const float denom = 1 / (1 + to[3]);
        return transpose(float4x4(1 - lorentzSign * to[0] * to[0] * denom,    -lorentzSign * to[0] * to[1] * denom,    -lorentzSign * to[0] * to[2] * denom, -lorentzSign * to[0],
                                     -lorentzSign * to[1] * to[0] * denom, 1 - lorentzSign * to[1] * to[1] * denom,    -lorentzSign * to[1] * to[2] * denom, -lorentzSign * to[1],
                                     -lorentzSign * to[2] * to[0] * denom,    -lorentzSign * to[2] * to[1] * denom, 1 - lorentzSign * to[2] * to[2] * denom, -lorentzSign * to[2],
                                     to[0], to[1], to[2], to[3]));
    }
    return transpose(float4x4(1, 0, 0, 0,
                              0, 1, 0, 0,
                              0, 0, 1, 0,
                              to[0], to[1], to[2], 1));
}

float4x4 ViewMat(const EuclidianCoordinate cameraPosition, const TransformData worldToCamera, const float lorentzSign) {
    const CurvedCoordinate ic = CurvedCoordinate(worldToCamera.m[0].xyz, 0);
    const CurvedCoordinate jc = CurvedCoordinate(worldToCamera.m[1].xyz, 0);
    const CurvedCoordinate kc = CurvedCoordinate(worldToCamera.m[2].xyz, 0);

    const CurvedCoordinate geomEye = euclidianToCurved(cameraPosition, lorentzSign);

    const float4x4 eyeTranslate = TranslateMatrix(geomEye, lorentzSign);
    const CurvedCoordinate icp = mul(eyeTranslate, ic);
    const CurvedCoordinate jcp = mul(eyeTranslate, jc);
    const CurvedCoordinate kcp = mul(eyeTranslate, kc);

    if (abs(lorentzSign) < 1e-4) {
        return transpose(float4x4(
            icp[0], jcp[0], kcp[0], 0,
            icp[1], jcp[1], kcp[1], 0,
            icp[2], jcp[2], kcp[2], 0,
            -dotProduct(icp, geomEye, lorentzSign), -dotProduct(jcp, geomEye, lorentzSign), -dotProduct(kcp, geomEye, lorentzSign), 1));
    }
    return transpose(float4x4(
        icp[0], jcp[0], kcp[0], lorentzSign * geomEye[0],
        icp[1], jcp[1], kcp[1], lorentzSign * geomEye[1],
        icp[2], jcp[2], kcp[2], lorentzSign * geomEye[2],
        lorentzSign * icp[3], lorentzSign * jcp[3], lorentzSign * kcp[3], geomEye[3]));
}

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
    uint mRenderFlipped;
    float mLorentzSign;
    float mScale;
    uint pad1;
    uint pad2;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> gPushConstants;

struct VSOut {
    float4 position : SV_Position;
    float3 normal   : NORMAL;
    float3 worldPos : TEXCOORD0;
    float2 uv       : TEXCOORD1;
    nointerpolation uint primId : TEXCOORD2;
};

[shader("vertex")]
VSOut vsmain(uint vertexID: SV_VertexID) {
    const MeshInstanceData instance = reinterpret<MeshInstanceData>(gScene.mInstances[gPushConstants.mInstanceIndex]);
    const MeshVertexInfo vertexInfo = gScene.mMeshVertexInfo[instance.vertexInfoIndex()];
	const uint3 triangle = gScene.LoadTriangleIndicesUniform(vertexInfo, vertexID / 3);
    float3 v[3];
    float3 n[3];
    float2 uv[3];
	LoadTriangleAttribute<float3>(gScene.mVertexBuffers[vertexInfo.positionBuffer()], vertexInfo.positionOffset(), vertexInfo.positionStride(), triangle,  v[0],  v[1],  v[2]);
	LoadTriangleAttribute<float3>(gScene.mVertexBuffers[vertexInfo.normalBuffer()]  , vertexInfo.normalOffset()  , vertexInfo.normalStride()  , triangle,  n[0],  n[1],  n[2]);
	LoadTriangleAttribute<float2>(gScene.mVertexBuffers[vertexInfo.texcoordBuffer()], vertexInfo.texcoordOffset(), vertexInfo.texcoordStride(), triangle, uv[0], uv[1], uv[2]);

    VSOut o;
    o.normal =  n[vertexID % 3];
    o.uv     = uv[vertexID % 3];
    o.primId = vertexID / 3;

    o.worldPos = gScene.mInstanceTransforms[gPushConstants.mInstanceIndex].transformPoint(v[vertexID % 3]);
    float4 geomPos = euclidianToCurved(o.worldPos * gPushConstants.mScale, gPushConstants.mLorentzSign);

    if (gPushConstants.mRenderFlipped == 1) {
        geomPos  = -geomPos;
        o.normal = -o.normal;
    }

    const float4 viewPos = mul(
        ViewMat(gParams.mViewTransforms[gPushConstants.mViewIndex].transformPoint(0), gParams.mViewInverseTransforms[gPushConstants.mViewIndex], gPushConstants.mLorentzSign),
        geomPos);

    ProjectionData proj = gParams.mViews[gPushConstants.mViewIndex].mProjection;
    proj.mFarPlane  = sin(proj.mFarPlane *gPushConstants.mScale);
    proj.mNearPlane = sin(proj.mNearPlane*gPushConstants.mScale);
    o.position = proj.projectPoint(viewPos);
    o.position.y = -o.position.y;
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