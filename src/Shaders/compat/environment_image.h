#pragma once

#include "scene.h"
#include "image_value.h"

STM_NAMESPACE_BEGIN

struct EnvironmentImage {
	ImageValue3 mEmission;

#ifdef __cplusplus
	inline void store(MaterialResources& resources) const {
		mEmission.store(resources);
	}
	inline void drawGui() {
		mEmission.drawGui("Emission");
	}
#endif
};

STM_NAMESPACE_END