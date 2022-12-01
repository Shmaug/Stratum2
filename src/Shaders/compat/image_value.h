#ifndef IMAGE_VALUE_H
#define IMAGE_VALUE_H

#include "common.h"

#ifdef __cplusplus
#include <Core/Image.hpp>
#endif

STM_NAMESPACE_BEGIN

#ifdef __cplusplus
inline uint4 channelMappingSwizzle(vk::ComponentMapping m) {
	uint4 c;
	c[0] = (m.r < vk::ComponentSwizzle::eR) ? 0 : ((uint32_t)m.r - (uint32_t)vk::ComponentSwizzle::eR);
	c[1] = (m.g < vk::ComponentSwizzle::eR) ? 1 : ((uint32_t)m.g - (uint32_t)vk::ComponentSwizzle::eR);
	c[2] = (m.b < vk::ComponentSwizzle::eR) ? 2 : ((uint32_t)m.b - (uint32_t)vk::ComponentSwizzle::eR);
	c[3] = (m.a < vk::ComponentSwizzle::eR) ? 3 : ((uint32_t)m.a - (uint32_t)vk::ComponentSwizzle::eR);
	return c;
}

template<unsigned int N>
struct ImageValue {
	VectorType<float,N> mValue;
	Image::View mImage;

	inline void store(MaterialResources& resources) const {
		resources.mMaterialData.AppendN(mValue);
		resources.mMaterialData.Append(resources.getIndex(mImage));
	}

	// Implemented in Scene.cpp
	void drawGui(const string& label);
};

using ImageValue1 = ImageValue<1>;
using ImageValue2 = ImageValue<2>;
using ImageValue3 = ImageValue<3>;
using ImageValue4 = ImageValue<4>;
#endif

#ifdef __HLSL__

struct ImageValue1 {
	float mValue;
	uint mImage;

	__init(inout uint address) {
		mValue = gScene.mMaterialData.Load<float>(address); address += 4;
		mImage = gScene.mMaterialData.Load<uint>(address); address += 4;
	}

	bool hasImage() { return mImage < gImageCount; }
	Texture2D<float> image() { return gScene.mImage1s[NonUniformResourceIndex(mImage)]; }
	float eval(const float2 uv, const float uvScreenSize) {
		if (!hasImage()) return mValue;
		return mValue * sampleImage(image(), uv, uvScreenSize);
	}
};

struct ImageValue2 {
	float2 mValue;
	uint mImage;

	__init(inout uint address) {
		mValue = gScene.mMaterialData.Load<float2>(address); address += 8;
		mImage = gScene.mMaterialData.Load<uint>(address); address += 4;
	}

	bool hasImage() { return mImage < gImageCount; }
	Texture2D<float4> image() { return gScene.mImages[NonUniformResourceIndex(mImage)]; }
	float2 eval(const float2 uv, const float uvScreenSize) {
		if (!hasImage()) return mValue;
		return mValue * sampleImage(image(), uv, uvScreenSize).rg;
	}
};

struct ImageValue3 {
	float3 mValue;
	uint mImage;

	__init(inout uint address) {
		mValue = gScene.mMaterialData.Load<float3>(address); address += 12;
		mImage = gScene.mMaterialData.Load<uint>(address); address += 4;
	}

	bool hasImage() { return mImage < gImageCount; }
	Texture2D<float4> image() { return gScene.mImages[NonUniformResourceIndex(mImage)]; }
	float3 eval(const float2 uv, const float uvScreenSize) {
		if (!hasImage()) return mValue;
		return mValue * sampleImage(image(), uv, uvScreenSize).rgb;
	}
};

struct ImageValue4 {
	float4 mValue;
	uint mImage;

	__init(inout uint address) {
		mValue = gScene.mMaterialData.Load<float4>(address); address += 16;
		mImage = gScene.mMaterialData.Load<uint>(address); address += 4;
	}

	bool hasImage() { return mImage < gImageCount; }
	Texture2D<float4> image() { return gScene.mImages[NonUniformResourceIndex(mImage)]; }
	float4 eval(const float2 uv, const float uvScreenSize) {
		if (!hasImage()) return mValue;
		return mValue * sampleImage(image(), uv, uvScreenSize);
	}
};
#endif

STM_NAMESPACE_END

#endif