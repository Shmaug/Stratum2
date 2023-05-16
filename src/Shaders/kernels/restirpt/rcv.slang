#pragma once

#include "common/shading_data.hlsli"

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

    RWTexture2D<float4> mReservoirDataDI[3];
    RWTexture2D<float4> mPrevReservoirDataDI[3];

    RWTexture2D<float4> mReservoirDataGI[6];
    RWTexture2D<float4> mPrevReservoirDataGI[6];
};

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
        ShadingData sd = scene.makeShadingData(scene.mInstances[mInstanceIndex], scene.mInstanceTransforms[mInstanceIndex], mLocalPosition, mPrimitiveIndex);
        sd.mTexcoordScreenSize = 0;
        return sd;
    }
};
struct ReconnectionVertex {
    PackedVertex mVertex;
    uint4 mPackedData; // { radiance, radianceScale, rcLength | pathLength<<16, localDirOut }


	// estimated radiance leaving mVertex towards mDirOut
    property float3 mRadiance {
        get {
            return D3DX_R10G10B10A2_UNORM_to_FLOAT4(mPackedData[0]).xyz * asfloat(mPackedData[1]);
        }
		set {
            const float scale = max3(newValue);
            mPackedData[0] = D3DX_FLOAT4_to_R10G10B10A2_UNORM(float4(newValue / scale, 0));
            mPackedData[1] = asuint(scale);
		}
	};

	// number of path segments before the reconnection vertex
    property uint mReconnectionLength {
        get { return BF_GET(mPackedData[2]          , 0, 16); }
        set {        BF_SET(mPackedData[2], newValue, 0, 16); }
    };

	// number of path segments in the whole path
    property uint mPathLength {
        get { return BF_GET(mPackedData[2]          , 16, 16); }
        set {        BF_SET(mPackedData[2], newValue, 16, 16); }
    };

    property float3 mLocalDirOut {
        get { return unpackNormal(mPackedData[3]); }
        set { mPackedData[3] = packNormal(newValue); }
	}

    __init() {
        mRadiance = 0;
        mReconnectionLength = 0;
        mPathLength = 0;
	}
};
ReconnectionVertex LoadPrevReconnectionVertex(const RenderParams framebuffer, const uint2 pixelIndex) {
    ReconnectionVertex r;
	r.mVertex = reinterpret<PackedVertex>(framebuffer.mPrevReservoirDataGI[3][pixelIndex]);
	r.mPackedData = reinterpret<uint4>(framebuffer.mPrevReservoirDataGI[4][pixelIndex]);
	return r;
}
void StoreReconnectionVertex(const RenderParams framebuffer, const uint2 pixelIndex, const ReconnectionVertex rcv) {
	framebuffer.mReservoirDataGI[3][pixelIndex] = reinterpret<float4>(rcv.mVertex);
	framebuffer.mReservoirDataGI[4][pixelIndex] = reinterpret<float4>(rcv.mPackedData);
}
