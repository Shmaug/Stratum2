#ifndef RNG_H
#define RNG_H

// xxhash (https://github.com/Cyan4973/xxHash)
//   From https://www.shadertoy.com/view/Xt3cDn
uint xxhash32(const uint p) {
	const uint PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
	const uint PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
	uint h32 = p + PRIME32_5;
	h32 = PRIME32_4*((h32 << 17) | (h32 >> (32 - 17)));
    h32 = PRIME32_2*(h32^(h32 >> 15));
    h32 = PRIME32_3*(h32^(h32 >> 13));
    return h32^(h32 >> 16);
}

// https://www.pcg-random.org/
uint pcg(uint v) {
	uint state = v * 747796405u + 2891336453u;
	uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}
uint4 pcg4d(uint4 v) {
	v = v * 1664525u + 1013904223u;
	v.x += v.y * v.w;
	v.y += v.z * v.x;
	v.z += v.x * v.y;
	v.w += v.y * v.z;
	v = v ^ (v >> 16u);
	v.x += v.y * v.w;
	v.y += v.z * v.x;
	v.z += v.x * v.y;
	v.w += v.y * v.z;
	return v;
}
uint pcg_next_uint(inout uint4 v) {
	v.w++;
	return pcg4d(v).x;
}
float pcg_next_float(inout uint4 v) {
    return asfloat(0x3f800000 | (pcg_next_uint(v) >> 9)) - 1;
}

typedef uint4 rng_state_t;
rng_state_t rng_init(const uint2 pixel, const uint offset = 0) { return uint4(pixel, gRandomSeed, offset); }
void  rng_skip_next (inout rng_state_t state) { state.w++; }
uint  rng_next_uint (inout rng_state_t state) { return pcg_next_uint(state); }
float rng_next_float(inout rng_state_t state) { return pcg_next_float(state); }

#endif