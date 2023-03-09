#pragma once

#ifndef gDIReservoirFlags
#define gDIReservoirFlags 0
#endif
#ifndef gLVCReservoirFlags
#define gLVCReservoirFlags 0
#endif
#ifndef gPpm
#define gPpm false
#endif
#ifndef gUseVC
#define gUseVC false
#endif
#ifndef gUseVM
#define gUseVM false
#endif
#ifndef gUseLVC
#define gUseLVC false
#endif
#ifndef gPathTraceOnly
#define gPathTraceOnly false
#endif
#ifndef gLightTraceOnly
#define gLightTraceOnly false
#endif
#ifndef gDebugPaths
#define gDebugPaths false
#endif
#ifndef gDebugPathWeights
#define gDebugPathWeights false
#endif



#include "compat/vcm.h"
#include "compat/scene.h"

#ifdef gLambertian
#include "materials/lambertian.hlsli"
#else
#include "materials/disney.hlsli"
#endif
#include "intersection.hlsli"

// shader parameters

[[vk::push_constant]] ConstantBuffer<VcmPushConstants> gPushConstants;

#include "hashgrid.hlsli"

struct RenderParams {
    StructuredBuffer<ViewData> mViews;
    StructuredBuffer<TransformData> mViewTransforms;
    StructuredBuffer<TransformData> mViewInverseTransforms;
	StructuredBuffer<uint> mViewMediumInstances;

	StructuredBuffer<TransformData> mPrevInverseViewTransforms;

    RWByteAddressBuffer mLightImage;

    RWTexture2D<float4> mOutput;
	RWTexture2D<float4> mAlbedo;
	RWTexture2D<float2> mPrevUVs;
	RWTexture2D<uint2> mVisibility;
    RWTexture2D<float4> mDepth;

    RWStructuredBuffer<PackedVcmVertex> mLightVertices;
    RWStructuredBuffer<uint> mLightPathLengths;

    ConstantBuffer<VcmConstants> mVcmConstants;
    HashGrid<uint> mLightHashGrid;

    HashGrid<DirectIlluminationReservoir> mDirectIlluminationReservoirs;
    HashGrid<DirectIlluminationReservoir> mPrevDirectIlluminationReservoirs;

    HashGrid<LVCReservoir> mLVCReservoirs;
    HashGrid<LVCReservoir> mPrevLVCReservoirs;
};

ParameterBlock<RenderParams> gRenderParams;

#define gHasEnvironment (gPushConstants.mEnvironmentMaterialAddress != -1)

// includes which reference shader parameters



// useful struct properties

extension PackedVcmVertex {
    property uint mInstanceIndex {
		get { return BF_GET(mInstancePrimitiveIndex, 0, 16); }
		set { BF_SET(mInstancePrimitiveIndex, newValue, 0, 16); }
	}
    property uint mPrimitiveIndex {
		get { return BF_GET(mInstancePrimitiveIndex, 16, 16); }
		set { BF_SET(mInstancePrimitiveIndex, newValue, 16, 16); }
    }

    property uint mPathLength {
        get { return BF_GET(mPackedData, 0, 16); }
        set { BF_SET(mPackedData, newValue, 0, 16); }
    }
    property float mPathPdfA {
        get { return f16tof32(BF_GET(mPackedData, 16, 16)); }
        set { BF_SET(mPackedData, f32tof16(newValue), 16, 16); }
    }

    __init(const VcmVertex v, const uint instancePrimitiveIndex) {
        mLocalPosition = gScene.mInstanceInverseTransforms[BF_GET(instancePrimitiveIndex, 0, 16)].transformPoint(v.mShadingData.mPosition);
        mInstancePrimitiveIndex = instancePrimitiveIndex;
        mThroughput = v.mThroughput;
        mPackedData = v.mPackedData;
        dVCM = v.dVCM;
        dVC = v.dVC;
        dVM = v.dVM;
        mLocalDirectionIn = v.mLocalDirectionIn;
    }
}

extension VcmVertex {
    property uint mPathLength {
        get { return BF_GET(mPackedData, 0, 16); }
        set { BF_SET(mPackedData, newValue, 0, 16); }
    }
    property float mPathPdfA {
        get { return f16tof32(BF_GET(mPackedData, 16, 16)); }
        set { BF_SET(mPackedData, f32tof16(newValue), 16, 16); }
    }

    __init() {
        mThroughput = 0;
    }
    __init(PackedVcmVertex v) {
        mShadingData = gScene.makeShadingData(gScene.mInstances[v.mInstanceIndex], gScene.mInstanceTransforms[v.mInstanceIndex], v.mLocalPosition, v.mPrimitiveIndex);
        mThroughput = v.mThroughput;
        mPackedData = v.mPackedData;
        dVCM = v.dVCM;
        dVC = v.dVC;
        dVM = v.dVM;
        mLocalDirectionIn = v.mLocalDirectionIn;
    }

    float3 OffsetRay(const float3 direction) {
        return rayOffset(mShadingData.mPosition, mShadingData.getGeometryNormal(), direction);
    }
}

extension DirectIlluminationReservoir {
    property uint mInstanceIndex {
        get { return BF_GET(mInstancePrimitiveIndex, 0, 16); }
        set { BF_SET(mInstancePrimitiveIndex, newValue, 0, 16); }
    }
    property uint mPrimitiveIndex {
        get { return BF_GET(mInstancePrimitiveIndex, 16, 16); }
        set { BF_SET(mInstancePrimitiveIndex, newValue, 16, 16); }
    }
}

// light sampling

struct IlluminationSampleRecord {
    float3 mRadiance;
    float mIntegrationWeight;
    float3 mDirectionToLight;
    float mDistance;
    float mDirectPdfW;
    float mEmissionPdfW;
    float mCosLight;
    uint mFlags;

    float3 mPosition;
    uint mPackedNormal;

    property float3 mNormal {
        get { return unpackNormal(mPackedNormal); }
        set { mPackedNormal = packNormal(newValue); }
	}

    float getG() {
        return mCosLight / pow2(mDistance);
    }
    float getDirectPdfA() {
        return pdfWtoA(mDirectPdfW, mCosLight, mDistance);
	}

    property bool isSingular {
        get { return BF_GET(mFlags, 0, 1); }
        set { BF_SET(mFlags, (uint)newValue, 0, 1); }
	}
    property bool isFinite {
        get { return BF_GET(mFlags, 1, 1); }
        set { BF_SET(mFlags, (uint)newValue, 1, 1); }
    }

    __init() {
		mRadiance = 0;
		mFlags = 0;
	}
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
	// uniformly samples a light instance and primitive index, then uniformly samples the primitive's area
	// note: referencePosition is not used during sampling
    IlluminationSampleRecord sampleIllumination(const float3 referencePosition, const float4 rnd) {
        IlluminationSampleRecord r;

        if (gPushConstants.mLightCount == 0)
            return { 0 };

        const uint lightInstanceIndex = mLightInstanceMap[uint(rnd.z * gPushConstants.mLightCount) % gPushConstants.mLightCount];
        const InstanceData instance = mInstances[lightInstanceIndex];
        const TransformData transform = mInstanceTransforms[lightInstanceIndex];

        float pdfA = 1 / (float)gPushConstants.mLightCount;
        if (gHasEnvironment)
            pdfA *= 1 - gPushConstants.mEnvironmentSampleProbability;

        ShadingData shadingData;
        if (instance.getType() == InstanceType::eMesh) {
			// triangle
            const MeshInstanceData mesh = reinterpret<MeshInstanceData>(instance);
            const uint primitiveIndex = uint(rnd.w * mesh.primitiveCount()) % mesh.primitiveCount();
            shadingData = makeTriangleShadingData(mesh, transform, primitiveIndex, sampleUniformTriangle(rnd.x, rnd.y));
            pdfA /= mesh.primitiveCount();
        } else if (instance.getType() == InstanceType::eSphere) {
			// sphere
            const SphereInstanceData sphere = reinterpret<SphereInstanceData>(instance);
            shadingData = makeSphereShadingData(sphere, transform, sphere.radius() * sampleUniformSphere(rnd.x, rnd.y));
        } else
            return { 0 }; // shouldn't happen as only environment, mesh and sphere lights are supported

        r.mDirectionToLight = shadingData.mPosition - referencePosition;
        r.mDistance = length(r.mDirectionToLight);
        r.mDirectionToLight /= r.mDistance;

        r.mCosLight = -dot(shadingData.getGeometryNormal(), r.mDirectionToLight);

        const PackedMaterialData m = gScene.LoadMaterial(shadingData);
        pdfA *= m.emissionPdf() / shadingData.mShapeArea;

        r.mRadiance = r.mCosLight <= 0 ? 0 : m.emission();
        r.mDirectPdfW = pdfAtoW(pdfA, r.mCosLight, r.mDistance);
        r.mEmissionPdfW = pdfA * cosHemispherePdfW(r.mCosLight);
        r.mIntegrationWeight = 1 / r.mDirectPdfW;
        r.mPosition = shadingData.mPosition;
        r.mPackedNormal = shadingData.mPackedGeometryNormal;
        r.isSingular = false;
        r.isFinite = true;
        return r;
    }
    EmissionSampleRecord sampleEmission(const float4 posRnd, const float2 dirRnd) {
        EmissionSampleRecord r;

        if (gPushConstants.mLightCount == 0)
            return { 0 };


        const uint lightInstanceIndex = mLightInstanceMap[uint(posRnd.z * gPushConstants.mLightCount) % gPushConstants.mLightCount];
        const InstanceData instance = mInstances[lightInstanceIndex];
        const TransformData transform = mInstanceTransforms[lightInstanceIndex];

        float pdfA = 1 / (float)gPushConstants.mLightCount;
        if (gHasEnvironment)
            pdfA *= 1 - gPushConstants.mEnvironmentSampleProbability;

        ShadingData shadingData;
        if (instance.getType() == InstanceType::eMesh) {
			const MeshInstanceData mesh = reinterpret<MeshInstanceData>(instance);
            const uint primitiveIndex = uint(posRnd.w * mesh.primitiveCount()) % mesh.primitiveCount();
            shadingData = makeTriangleShadingData(mesh, transform, primitiveIndex, sampleUniformTriangle(posRnd.x, posRnd.y));
            pdfA /= mesh.primitiveCount();
        } else if (instance.getType() == InstanceType::eSphere) {
            const SphereInstanceData sphere = reinterpret<SphereInstanceData>(instance);
            shadingData = makeSphereShadingData(sphere, transform, sphere.radius() * sampleUniformSphere(posRnd.x, posRnd.y));
        } else
            return { 0 };

        const PackedMaterialData m = gScene.LoadMaterial(shadingData);
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
struct LightRadianceRecord {
	float3 mRadiance;
    float mDirectPdfA;
	float mEmissionPdfW;
};
LightRadianceRecord GetBackgroundRadiance(const float3 aDirection) {
	if (!gHasEnvironment) return { 0 };

    LightRadianceRecord r;
    r.mRadiance = 0;
    r.mDirectPdfA = 0;
    if (gPushConstants.mLightCount > 0)
        r.mDirectPdfA *= gPushConstants.mEnvironmentSampleProbability;
    r.mEmissionPdfW = r.mDirectPdfA * concentricDiscPdfA() / pow2(gRenderParams.mVcmConstants.mSceneSphere.w);
	return r;
}
LightRadianceRecord GetSurfaceRadiance(const float3 aDirection, const IntersectionResult aIsect, const PackedMaterialData aBsdf) {
    if (gPushConstants.mLightCount == 0) return { 0 };

    const float cosTheta = -dot(aDirection, aIsect.mShadingData.getGeometryNormal());
    if (cosTheta < 0) return { 0 };

	LightRadianceRecord r;
    r.mRadiance = aBsdf.emission();
    r.mDirectPdfA = aBsdf.emissionPdf() * aIsect.mPrimitivePickPdf / (aIsect.mShadingData.mShapeArea * gPushConstants.mLightCount);
    if (gHasEnvironment)
        r.mDirectPdfA *= 1 - gPushConstants.mEnvironmentSampleProbability;
    r.mEmissionPdfW = r.mDirectPdfA * cosHemispherePdfW(cosTheta);
	return r;
}

// path tracing functions

void AddColor(const uint2 index, const float3 color, const uint cameraVertices, const uint lightVertices) {
    if (gDebugPaths && !(cameraVertices == gPushConstants.debugCameraPathLength() && lightVertices == gPushConstants.debugLightPathLength()))
        return;

	gRenderParams.mOutput[index] += float4(color, 0);
}

void InterlockedAddColor(const int2 ipos, const uint2 extent, const float3 color, const uint cameraVertices, const uint lightVertices) {
    if (gDebugPaths && !(cameraVertices == gPushConstants.debugCameraPathLength() && lightVertices == gPushConstants.debugLightPathLength()))
		return;

    if (all(color <= 0) || any(ipos < 0) || any(ipos >= extent))
        return;

    const uint3 icolor = uint3(color * gPushConstants.mLightImageQuantization);

    const uint address = 16 * (ipos.y * extent.x + ipos.x);
    uint overflowed = 0;
    uint3 prev;
    gRenderParams.mLightImage.InterlockedAdd(address + 0, icolor[0], prev[0]);
    gRenderParams.mLightImage.InterlockedAdd(address + 4, icolor[1], prev[1]);
    gRenderParams.mLightImage.InterlockedAdd(address + 8, icolor[2], prev[2]);
    for (uint i = 0; i < 3; i++) {
        if (icolor[i] >= 0xFFFFFFFF - prev[i])
            overflowed |= BIT(i);
    }
    gRenderParams.mLightImage.InterlockedOr(address + 12, overflowed);
}

struct AbstractCamera {
    uint mViewIndex;

    ViewData      getView()                 { return gRenderParams.mViews[mViewIndex]; }
    TransformData getTransform()            { return gRenderParams.mViewTransforms[mViewIndex]; }
    TransformData getInverseTransform()     { return gRenderParams.mViewInverseTransforms[mViewIndex]; }
    TransformData getPrevInverseTransform() { return gRenderParams.mPrevInverseViewTransforms[mViewIndex]; }
};
