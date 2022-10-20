#ifndef RESERVOIR_H
#define RESERVOIR_H

#ifdef __cplusplus
#pragma pack(push)
#pragma pack(1)
namespace tinyvkpt {
#endif

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

#ifdef __cplusplus
} // namespace tinyvkpt
#pragma pack(pop)
#endif

#endif