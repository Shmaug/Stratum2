#pragma once

#include "common/shading_data.hlsli"

// this is a separate file so that it can be included by the path tracer and the rcv visualizer (rcv_vis.slang)

struct RenderParams {
    StructuredBuffer<ViewData> mViews;
    StructuredBuffer<TransformData> mViewTransforms;
    StructuredBuffer<TransformData> mViewInverseTransforms;
    StructuredBuffer<TransformData> mPrevViewInverseTransforms;
    StructuredBuffer<uint> mViewMediumIndices;
    RWTexture2D<float4> mOutput;
    RWTexture2D<float4> mAlbedo;
    RWTexture2D<float2> mPrevUVs;
    RWTexture2D<uint2> mVisibility;
    RWTexture2D<float4> mDepth;

    RWStructuredBuffer<uint> mDebugCounters;

    RWTexture2D<float4> mPathReservoirData[6];
    RWTexture2D<float4> mPrevPathReservoirData[6];
};

typedef float3 Vector3;
typedef float3 Spectrum;

struct PackedVertex {
	float3 mLocalPosition;
	uint mInstancePrimitiveIndex;

    property uint mInstanceIndex  {
		get { return BF_GET(mInstancePrimitiveIndex, 0, 16); }
		set { BF_SET(mInstancePrimitiveIndex, newValue, 0, 16); }
	}
    property uint mPrimitiveIndex {
		get { return BF_GET(mInstancePrimitiveIndex, 16, 16); }
		set { BF_SET(mInstancePrimitiveIndex, newValue, 16, 16); }
	}

    ShadingData getShadingData(const SceneParameters scene) {
        ShadingData sd;
        if (mInstanceIndex == INVALID_INSTANCE) {
            sd.mPosition = mLocalPosition;
            sd.mShapeArea = -1;
		} else
        	sd = scene.makeShadingData(scene.mInstances[mInstanceIndex], scene.mInstanceTransforms[mInstanceIndex], mLocalPosition, mPrimitiveIndex);
        sd.mTexcoordScreenSize = 0;
        return sd;
    }
};
struct ReconnectionVertex {
    PackedVertex mVertex;
    uint4 mPackedData; // { radiance, localDirOut }

	// radiance arriving at mVertex from mLocalDirOut
    property Spectrum mRadiance {
        get {
            return asfloat(mPackedData.xyz);
        }
        set {
            mPackedData.xyz = asuint(newValue);
		}
	};

    property Vector3 mLocalDirOut {
        get { return unpackNormal(mPackedData[3]); }
        set { mPackedData[3] = packNormal(newValue); }
	}
};
ReconnectionVertex LoadPrevReconnectionVertex(const RenderParams framebuffer, const uint2 pixelIndex) {
    ReconnectionVertex r;
	r.mVertex = reinterpret<PackedVertex>(framebuffer.mPrevPathReservoirData[4][pixelIndex]);
	r.mPackedData = reinterpret<uint4>(framebuffer.mPrevPathReservoirData[5][pixelIndex]);
	return r;
}
void StoreReconnectionVertex(const RenderParams framebuffer, const uint2 pixelIndex, const ReconnectionVertex rcv) {
	framebuffer.mPathReservoirData[4][pixelIndex] = reinterpret<float4>(rcv.mVertex);
	framebuffer.mPathReservoirData[5][pixelIndex] = reinterpret<float4>(rcv.mPackedData);
}
