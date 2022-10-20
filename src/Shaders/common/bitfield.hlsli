#include "D3DX_DXGIFormatConvert.inl"

inline float2 pack_normal_octahedron2(const float3 v) {
	// Project the sphere onto the octahedron, and then onto the xy plane
	const float2 p = v.xy * (1 / (abs(v.x) + abs(v.y) + abs(v.z)));
	// Reflect the folds of the lower hemisphere over the diagonals
	return (v.z <= 0) ? ((1 - abs(p.yx)) * lerp(-1, 1, float2(p >= 0))) : p;
}
inline float3 unpack_normal_octahedron2(const float2 p) {
	float3 v = float3(p, 1 - dot(1, abs(p)));
	if (v.z < 0) v.xy = (1 - abs(v.yx)) * lerp(-1, 1, float2(v.xy >= 0));
	return normalize(v);
}

inline uint pack_normal_octahedron(const float3 v) {
	return D3DX_FLOAT2_to_R16G16_SNORM(pack_normal_octahedron2(v));
}
inline float3 unpack_normal_octahedron(const uint packed) {
	return unpack_normal_octahedron2(D3DX_R16G16_SNORM_to_FLOAT2(packed));
}
