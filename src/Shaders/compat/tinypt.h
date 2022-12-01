#ifndef TINYPT_H
#define TINYPT_H

#include "common.h"

STM_NAMESPACE_BEGIN

struct TinyPTPushConstants {
	uint mRandomSeed;
	uint mViewCount;

	uint mEnvironmentMaterialAddress;
	uint mLightCount;

	uint mMinBounces;
	uint mMaxBounces;
	uint mMaxDiffuseBounces;
};

STM_NAMESPACE_END

#endif