#ifndef SHADING_DATA_H
#define SHADING_DATA_H

#include "scene.h"

#define SHADING_FLAG_FRONT_FACE BIT(0)
#define SHADING_FLAG_FLIP_BITANGENT BIT(1)

#ifdef __cplusplus
namespace tinyvkpt {
#endif

// 48 bytes
struct ShadingData {
	float3 position;
	uint flags;
	uint packed_geometry_normal;
	uint packed_shading_normal;
	uint packed_tangent;
	float shape_area;
	float2 uv;
	float uv_screen_size;
	float mean_curvature;
#ifdef __HLSL__
	inline float3 geometry_normal() { return unpack_normal_octahedron(packed_geometry_normal); }
	inline float3 shading_normal() { return unpack_normal_octahedron(packed_shading_normal); }
	inline float3 tangent() { return unpack_normal_octahedron(packed_tangent); }
	inline bool flip_bitangent() { return (bool)(flags & SHADING_FLAG_FLIP_BITANGENT); }

	inline float3 to_world(const float3 v) {
		const float3 n = shading_normal();
		const float3 t = tangent();
		return v.x*t + v.y*cross(n, t)*(flip_bitangent() ? -1 : 1) + v.z*n;
	}
	inline float3 to_local(const float3 v) {
		const float3 n = shading_normal();
		const float3 t = tangent();
		return float3(dot(v, t), dot(v, cross(n, t)*(flip_bitangent() ? -1 : 1)), dot(v, n));
	}
#endif
};

#ifdef __HLSL__
#include "../common/shading_data.hlsli"
#endif

#ifdef __cplusplus
} // namespace tinyvkpt
#endif

#endif
