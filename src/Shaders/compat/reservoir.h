#ifndef RESERVOIR_H
#define RESERVOIR_H

#include "hlslcompat.h"

STM_NAMESPACE_BEGIN

struct Reservoir {
	float total_weight;
	uint M;

	SLANG_MUTATING
	inline void init() {
		total_weight = 0;
		M = 0;
	}

	SLANG_MUTATING
	inline bool update(const float rnd, const float w) {
		M++;
		total_weight += w;
		return rnd*total_weight <= w;
	}
};

STM_NAMESPACE_END

#endif