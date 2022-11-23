#ifndef IMAGE_VALUE_H
#define IMAGE_VALUE_H

#include "common.h"

#ifdef __cplusplus
#include <Core/Image.hpp>
namespace tinyvkpt {

inline uint4 channelMappingSwizzle(vk::ComponentMapping m) {
	uint4 c;
	c[0] = (m.r < vk::ComponentSwizzle::eR) ? 0 : ((uint32_t)m.r - (uint32_t)vk::ComponentSwizzle::eR);
	c[1] = (m.g < vk::ComponentSwizzle::eR) ? 1 : ((uint32_t)m.g - (uint32_t)vk::ComponentSwizzle::eR);
	c[2] = (m.b < vk::ComponentSwizzle::eR) ? 2 : ((uint32_t)m.b - (uint32_t)vk::ComponentSwizzle::eR);
	c[3] = (m.a < vk::ComponentSwizzle::eR) ? 3 : ((uint32_t)m.a - (uint32_t)vk::ComponentSwizzle::eR);
	return c;
}

#endif // __cplusplus

struct ImageValue1 {
	float value;
#ifdef __HLSL__
	uint image_index_channel;
	bool has_image() { return BF_GET(image_index_channel,0,30) < gImageCount; }
	uint channel() { return BF_GET(image_index_channel,30,2); }
	Texture2D<float4> image() { return gSceneParams.gImages[NonUniformResourceIndex(BF_GET(image_index_channel,0,30))]; }
	float eval(const float2 uv, const float uv_screen_size) {
		if (value <= 0) return 0;
		if (!has_image()) return value;
		return value * sample_image(image(), uv, uv_screen_size)[channel()];
	}
	SLANG_MUTATING
	void load(inout uint address) {
		value               = gSceneParams.gMaterialData.Load<float>((int)address); address += 4;
		image_index_channel = gSceneParams.gMaterialData.Load<uint>((int)address); address += 4;
	}
#endif
#ifdef __cplusplus
	Image::View image;
	inline void store(MaterialResources& resources) const {
		resources.mMaterialData.Appendf(value);
		const uint image_index = resources.getIndex(image);
		const uint32_t channel = channelMappingSwizzle(image.componentMapping())[0];
		uint image_index_and_channel = 0;
		BF_SET(image_index_and_channel, image_index, 0, 30);
		BF_SET(image_index_and_channel, channel, 30, 2);
		resources.mMaterialData.Append(image_index_and_channel);
	}
#endif
};

struct ImageValue2 {
	float2 value;
#ifdef __HLSL__
	uint image_index;
	bool has_image() { return image_index < gImageCount; }
	Texture2D<float4> image() { return gSceneParams.gImages[NonUniformResourceIndex(image_index)]; }
	SLANG_MUTATING
	void load(inout uint address) {
		value       = gSceneParams.gMaterialData.Load<float2>(address); address += 8;
		image_index = gSceneParams.gMaterialData.Load<uint>(address); address += 4;
	}
	float2 eval(const float2 uv, const float uv_screen_size) {
		if (!has_image()) return value;
		if (!any(value > 0)) return 0;
		return value * sample_image(image(), uv, uv_screen_size).rg;
	}
#endif
#ifdef __cplusplus
	Image::View image;
	inline void store(MaterialResources& resources) const {
		resources.mMaterialData.AppendN(value);
		resources.mMaterialData.Append(resources.getIndex(image));
	}
#endif
};

struct ImageValue3 {
	float3 value;
#ifdef __HLSL__
	uint image_index;
	bool has_image() { return image_index < gImageCount; }
	Texture2D<float4> image() { return gSceneParams.gImages[NonUniformResourceIndex(image_index)]; }
	SLANG_MUTATING
	void load(inout uint address) {
		value       = gSceneParams.gMaterialData.Load<float3>(address); address += 12;
		image_index = gSceneParams.gMaterialData.Load<uint>(address); address += 4;
	}
	float3 eval(const float2 uv, const float uv_screen_size) {
		if (!has_image()) return value;
		if (!any(value > 0)) return 0;
		return value * sample_image(image(), uv, uv_screen_size).rgb;
	}
#endif
#ifdef __cplusplus
	Image::View image;
	inline void store(MaterialResources& resources) const {
		resources.mMaterialData.AppendN(value);
		resources.mMaterialData.Append(resources.getIndex(image));
	}
#endif
};

struct ImageValue4 {
	float4 value;
#ifdef __HLSL__
	uint image_index;
	bool has_image() { return image_index < gImageCount; }
	Texture2D<float4> image() { return gSceneParams.gImages[NonUniformResourceIndex(image_index)]; }
	SLANG_MUTATING
	void load(inout uint address) {
		value       = gSceneParams.gMaterialData.Load<float4>(address); address += 16;
		image_index = gSceneParams.gMaterialData.Load<uint>(address); address += 4;
	}
	float4 eval(const float2 uv, const float uv_screen_size) {
		if (!has_image()) return value;
		if (!any(value > 0)) return 0;
		return value * sample_image(image(), uv, uv_screen_size);
	}
#endif
#ifdef __cplusplus
	Image::View image;
	inline void store(MaterialResources& resources) const {
		resources.mMaterialData.AppendN(value);
		resources.mMaterialData.Append(resources.getIndex(image));
	}
#endif
};

#ifdef __cplusplus

// Defined in Scene.cpp

void imageValueField(const char* label, ImageValue1& v);
void imageValueField(const char* label, ImageValue2& v);
void imageValueField(const char* label, ImageValue3& v);
void imageValueField(const char* label, ImageValue4& v);

} // namespace tinyvkpt

#endif // __cplusplus

#endif