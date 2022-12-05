#include "compat/scene.h"

float sampleImage1(const uint imageIndex, const float2 uv, const float uvScreenSize) {
	float lod = 0;
    if (uvScreenSize > 0) {
        float w, h;
        gScene.mImage1s[imageIndex].GetDimensions(w, h);
		lod = log2(max(uvScreenSize * max(w, h), 1e-6f));
    }
    return gScene.mImage1s[NonUniformResourceIndex(imageIndex)].SampleLevel(gScene.mStaticSampler, uv, lod);
}
float4 sampleImage4(const uint imageIndex, const float2 uv, const float uvScreenSize) {
    float lod = 0;
    if (uvScreenSize > 0) {
        float w, h;
        gScene.mImages[imageIndex].GetDimensions(w, h);
		lod = log2(max(uvScreenSize * max(w, h), 1e-6f));
    }
    return gScene.mImages[NonUniformResourceIndex(imageIndex)].SampleLevel(gScene.mStaticSampler, uv, lod);
}

struct ImageValue1 {
	static const uint PackedSize = 8;

	float mValue;
	uint mImage;

	__init(ByteAddressBuffer buf, const uint address) {
		mValue = buf.Load<float>(address);
		mImage = buf.Load<uint>(address + 4);
	}

	bool hasImage() { return mImage < gImageCount; }
	float eval(const float2 uv, const float uvScreenSize) {
		if (!hasImage()) return mValue;
		return mValue * sampleImage1(mImage, uv, uvScreenSize);
	}
};

struct ImageValue2 {
	static const uint PackedSize = 12;

	float2 mValue;
	uint mImage;

	__init(ByteAddressBuffer buf, const uint address) {
		mValue = buf.Load<float2>(address);
		mImage = buf.Load<uint>(address + 8);
	}

	bool hasImage() { return mImage < gImageCount; }
	float2 eval(const float2 uv, const float uvScreenSize) {
		if (!hasImage()) return mValue;
		return mValue * sampleImage4(mImage, uv, uvScreenSize).rg;
	}
};

struct ImageValue3 {
	static const uint PackedSize = 16;

	float3 mValue;
	uint mImage;

	__init(ByteAddressBuffer buf, const uint address) {
		mValue = buf.Load<float3>(address);
		mImage = buf.Load<uint>(address + 12);
	}

	bool hasImage() { return mImage < gImageCount; }
	float3 eval(const float2 uv, const float uvScreenSize) {
		if (!hasImage()) return mValue;
		return mValue * sampleImage4(mImage, uv, uvScreenSize).rgb;
	}
};

struct ImageValue4 {
	static const uint PackedSize = 20;

	float4 mValue;
	uint mImage;

	__init(ByteAddressBuffer buf, const uint address) {
		mValue = buf.Load<float4>(address);
		mImage = buf.Load<uint>(address + 16);
	}

	bool hasImage() { return mImage < gImageCount; }
	float4 eval(const float2 uv, const float uvScreenSize) {
		if (!hasImage()) return mValue;
		return mValue * sampleImage4(mImage, uv, uvScreenSize);
	}
};