#pragma once

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

template<int N>
struct ImageValue {
	VectorType<float,N> mValue;
	Image::View mImage;

	inline void store(MaterialResources& resources) const {
		resources.mMaterialData.AppendN(mValue);
		resources.mMaterialData.Append(resources.getIndex(mImage));
	}

	// Implemented in Scene.cpp
	bool drawGui(const string& label);
};

using ImageValue1 = ImageValue<1>;
using ImageValue2 = ImageValue<2>;
using ImageValue3 = ImageValue<3>;
using ImageValue4 = ImageValue<4>;
#endif

#ifdef __SLANG_COMPILER__
#include "common/image_value.hlsli"
#endif

STM_NAMESPACE_END