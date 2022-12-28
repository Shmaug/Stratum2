#pragma once

#include "common.h"

STM_NAMESPACE_BEGIN

struct VcmPushConstants {
	float mRadiusAlpha;       // Radius reduction rate parameter
	float mBaseRadius;        // Initial merging radius
	float mMisVmWeightFactor; // Weight of vertex merging (used in VC)
	float mMisVcWeightFactor; // Weight of vertex connection (used in VM)
	float mScreenPixelCount;  // Number of pixels
	float mLightSubPathCount; // Number of light sub-paths
    float mVmNormalization;   // 1 / (Pi * radius^2 * light_path_count)
};

STM_NAMESPACE_END