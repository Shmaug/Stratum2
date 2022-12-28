#pragma once

#include "compat/scene.h"
#include "compat/path_tracer.h"

struct RenderParams {
    StructuredBuffer<ViewData> mViews;
    StructuredBuffer<TransformData> mViewTransforms;
    StructuredBuffer<TransformData> mViewInverseTransforms;
	StructuredBuffer<uint> mViewMediumInstances;

	StructuredBuffer<ViewData> mPrevViews;
	StructuredBuffer<TransformData> mPrevInverseViewTransforms;

    RWStructuredBuffer<VcmVertex> mLightVertices;
    RWStructuredBuffer<uint> mLightPathLengths;

    RWByteAddressBuffer mLightImage;
    RWTexture2D<float4> mOutput;
	RWTexture2D<float4> mAlbedo;
	RWTexture2D<float2> mPrevUVs;
	RWTexture2D<uint2> mVisibility;
    RWTexture2D<float4> mDepth;
};

// shader parameters

ParameterBlock<SceneParameters> gScene;
ParameterBlock<RenderParams> gRenderParams;
[[vk::push_constant]] ConstantBuffer<PathTracerPushConstants> gPushConstants;


// includes which reference shader parameters

#define gHasEnvironment (gPushConstants.mEnvironmentMaterialAddress != -1)

#include "intersection.hlsli"
#include "materials/environment.hlsli"
#include "materials/disney.hlsli"
typedef DisneyMaterial Material;

extension ShadingData {
    float shadingNormalCorrection<let Adjoint : bool>(const float3 localDirIn, const float3 localDirOut) {
        if (isMedium)
            return 1;

        const float3 localGeometryNormal = toLocal(geometryNormal);
        const float ngdotin = dot(localGeometryNormal, localDirIn);
        const float ngdotout = dot(localGeometryNormal, localDirOut);

        // light leak fix
        if (sign(ngdotout) != sign(localDirOut.z) || sign(ngdotin) != sign(localDirIn.z))
            return 0;

        float G = 1;

        if (Adjoint) {
            const float num = ngdotout * localDirIn.z;
            const float denom = localDirOut.z * ngdotin;
            if (abs(denom) > 1e-5)
                G *= abs(num / denom);
        }

        return G;
    }
}

struct IlluminationSampleRecord {
    float3 mRadiance;
    float mDistance;
    float3 mDirectionToLight;
    float mDirectPdfW;
    float mEmissionPdfW;
    float mCosLight;

    bool isSingular;
    bool isFinite;
};
struct EmissionSampleRecord {
    float3 mRadiance;
    float mDirectPdfA;
    float3 mPosition;
    float mEmissionPdfW;
    float3 mDirection;
    float mCosLight;
    uint mPackedNormal;

    bool isSingular;
    bool isFinite;
};

extension SceneParameters {
    IlluminationSampleRecord sampleIllumination(const float3 referencePosition, const float4 rnd) {
        IlluminationSampleRecord r;

        if (gHasEnvironment) {
            if (gPushConstants.mLightCount == 0 || rnd.w < gPushConstants.mEnvironmentSampleProbability) {
                float2 uv;
                r.mRadiance = EnvironmentImage().sample(rnd.xy, /*out*/ r.mDirectionToLight, /*out*/ uv, /*out*/ r.mDirectPdfW);
                if (gPushConstants.mLightCount > 0)
                    r.mDirectPdfW *= gPushConstants.mEnvironmentSampleProbability;
				r.mDistance = POS_INFINITY;
                r.mEmissionPdfW = r.mDirectPdfW * concentricDiscPdfA() / pow2(gPushConstants.mSceneSphere.w);
                r.mCosLight = 1;
                r.isSingular = false;
                r.isFinite = false;
                return r;
            }
        }

        if (gPushConstants.mLightCount == 0)
            return { 0 };

        const uint lightInstanceIndex = mLightInstanceMap[uint(rnd.z * gPushConstants.mLightCount) % gPushConstants.mLightCount];
        const InstanceData instance = mInstances[lightInstanceIndex];
        const TransformData transform = mInstanceTransforms[lightInstanceIndex];

        float pdfA = 1 / (float)gPushConstants.mLightCount;
        if (gHasEnvironment)
            pdfA *= 1 - gPushConstants.mEnvironmentSampleProbability;

        ShadingData shadingData;
        if (instance.type() == InstanceType::eMesh) {
            const MeshInstanceData mesh = reinterpret<MeshInstanceData>(instance);
            const uint primitiveIndex = uint(rnd.w * mesh.primitiveCount()) % mesh.primitiveCount();
            shadingData = makeTriangleShadingData(mesh, transform, primitiveIndex, sampleUniformTriangle(rnd.x, rnd.y));
            pdfA /= mesh.primitiveCount();
        } else if (instance.type() == InstanceType::eSphere) {
            const SphereInstanceData sphere = reinterpret<SphereInstanceData>(instance);
            shadingData = makeSphereShadingData(sphere, transform, sphere.radius() * sampleUniformSphereCartesian(rnd.x, rnd.y));
        } else
            return { 0 };


        r.mDirectionToLight = shadingData.mPosition - referencePosition;
        r.mDistance = length(r.mDirectionToLight);
        r.mDirectionToLight /= r.mDistance;

        r.mCosLight = -dot(shadingData.geometryNormal, r.mDirectionToLight);
        if (r.mCosLight < 0)
            return { 0 };

        const Material m = Material(shadingData);
        pdfA *= m.emissionPdf() / shadingData.mShapeArea;

        r.mRadiance = m.emission();
        r.mDirectPdfW = pdfAtoW(pdfA, r.mCosLight / pow2(r.mDistance));
        r.mEmissionPdfW = pdfA * cosHemispherePdfW(r.mCosLight);
        r.isSingular = false;
        r.isFinite = true;
        return r;
    }
    EmissionSampleRecord sampleEmission(const float4 posRnd, const float2 dirRnd) {
        EmissionSampleRecord r;

        if (gHasEnvironment) {
            if (gPushConstants.mLightCount == 0 || posRnd.w < gPushConstants.mEnvironmentSampleProbability) {
                float2 uv;
                r.mRadiance = EnvironmentImage().sample(dirRnd, /*out*/ r.mDirection, /*out*/ uv, /*out*/ r.mDirectPdfA);
                if (gPushConstants.mLightCount > 0)
                    r.mDirectPdfA *= gPushConstants.mEnvironmentSampleProbability;
                const float3x3 frame = makeOrthonormal(r.mDirection);
                r.mPosition = gPushConstants.mSceneSphere.xyz + gPushConstants.mSceneSphere.w * (frame[0] * posRnd.x + frame[1] * posRnd.y - r.mDirection);
                r.mEmissionPdfW = r.mDirectPdfA * concentricDiscPdfA() / pow2(gPushConstants.mSceneSphere.w);
                r.mPackedNormal = packNormal(r.mDirection);
				r.mCosLight = 1;
                r.isSingular = false;
                r.isFinite = false;
                return r;
            }
        }

        if (gPushConstants.mLightCount == 0)
            return { 0 };


        const uint lightInstanceIndex = mLightInstanceMap[uint(posRnd.z * gPushConstants.mLightCount) % gPushConstants.mLightCount];
        const InstanceData instance = mInstances[lightInstanceIndex];
        const TransformData transform = mInstanceTransforms[lightInstanceIndex];

        float pdfA = 1 / (float)gPushConstants.mLightCount;
        if (gHasEnvironment)
            pdfA *= 1 - gPushConstants.mEnvironmentSampleProbability;

        ShadingData shadingData;
        if (instance.type() == InstanceType::eMesh) {
			const MeshInstanceData mesh = reinterpret<MeshInstanceData>(instance);
            const uint primitiveIndex = uint(posRnd.w * mesh.primitiveCount()) % mesh.primitiveCount();
            shadingData = makeTriangleShadingData(mesh, transform, primitiveIndex, sampleUniformTriangle(posRnd.x, posRnd.y));
            pdfA /= mesh.primitiveCount();
        } else if (instance.type() == InstanceType::eSphere) {
            const SphereInstanceData sphere = reinterpret<SphereInstanceData>(instance);
            shadingData = makeSphereShadingData(sphere, transform, sphere.radius() * sampleUniformSphereCartesian(posRnd.x, posRnd.y));
        } else
            return { 0 };

        const Material m = Material(shadingData);
		pdfA *= m.emissionPdf() / shadingData.mShapeArea;

        const float3 localDirOut = sampleCosHemisphere(dirRnd.x, dirRnd.y);
        r.mCosLight = localDirOut.z;

        r.mRadiance = m.emission() * r.mCosLight;
        r.mDirectPdfA = pdfA;
        r.mPosition = shadingData.mPosition;
        r.mEmissionPdfW = pdfA * cosHemispherePdfW(r.mCosLight);
        r.mDirection = shadingData.toWorld(localDirOut);
        r.mPackedNormal = shadingData.mPackedGeometryNormal;
        r.isSingular = false;
        r.isFinite = true;
        return r;
    }
}