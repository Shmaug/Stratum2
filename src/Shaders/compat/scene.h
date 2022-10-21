#ifndef SCENE_H
#define SCENE_H

#include "common.h"
#include "bitfield.h"
#include "transform.h"

#ifdef __cplusplus
#pragma pack(push)
#pragma pack(1)
namespace tinyvkpt {
#endif

#define INSTANCE_TYPE_TRIANGLES 0
#define INSTANCE_TYPE_SPHERE 1
#define INSTANCE_TYPE_VOLUME 2

#define BVH_FLAG_NONE 0
#define BVH_FLAG_TRIANGLES BIT(0)
#define BVH_FLAG_SPHERES BIT(1)
#define BVH_FLAG_VOLUME BIT(2)
#define BVH_FLAG_EMITTER BIT(3)

#define INVALID_INSTANCE 0xFFFF
#define INVALID_PRIMITIVE 0xFFFF

#define gImageCount 4096
#define gVolumeCount 8

struct InstanceData {
	uint4 packed;

	inline uint type() CONST_CPP { return BF_GET(packed[0], 0, 4); }
	inline uint material_address() CONST_CPP { return BF_GET(packed[0], 4, 28); }
	inline uint light_index() CONST_CPP { return BF_GET(packed[1], 0, 12); }

	// mesh
	inline uint prim_count() CONST_CPP { return BF_GET(packed[1], 12, 16); }
	inline uint index_stride() CONST_CPP { return BF_GET(packed[1], 28, 4); }
	inline uint first_vertex() CONST_CPP { return packed[2]; }
	inline uint indices_byte_offset() CONST_CPP { return packed[3]; }

	// sphere
	inline float radius() CONST_CPP { return asfloat(packed[2]); }

	// volume
	inline uint volume_index() CONST_CPP { return packed[2]; }
};

inline TransformData make_instance_motion_transform(const TransformData instance_inv_transform, const TransformData prevObjectToWorld) { return tmul(prevObjectToWorld, instance_inv_transform); }
inline InstanceData make_instance_triangles(const uint materialAddress, const uint primCount, const uint firstVertex, const uint indexByteOffset, const uint indexStride) {
	InstanceData r;
	r.packed = 0;
	BF_SET(r.packed[0], INSTANCE_TYPE_TRIANGLES, 0, 4);
	BF_SET(r.packed[0], materialAddress, 4, 28);
	BF_SET(r.packed[1], -1, 0, 12);
	BF_SET(r.packed[1], primCount, 12, 16);
	BF_SET(r.packed[1], indexStride, 28, 4);
	r.packed[2] = firstVertex;
	r.packed[3] = indexByteOffset;
	return r;
}
inline InstanceData make_instance_sphere(const uint materialAddress, const float radius) {
	InstanceData r;
	r.packed = 0;
	BF_SET(r.packed[0], INSTANCE_TYPE_SPHERE, 0, 4);
	BF_SET(r.packed[0], materialAddress, 4, 28);
	BF_SET(r.packed[1], -1, 0, 12);
	r.packed[2] = asuint(radius);
	return r;
}
inline InstanceData make_instance_volume(const uint materialAddress, const uint volume_index) {
	InstanceData r;
	r.packed = 0;
	BF_SET(r.packed[0], INSTANCE_TYPE_VOLUME, 0, 4);
	BF_SET(r.packed[0], materialAddress, 4, 28);
	BF_SET(r.packed[1], -1, 0, 12);
	r.packed[2] = volume_index;
	return r;
}

struct ViewData {
	ProjectionData projection;
	int2 image_min;
	int2 image_max;
	inline int2 extent() CONST_CPP { return image_max - image_min; }
#ifdef __cplusplus
	inline bool test_inside(const int2 p) const { return (p >= image_min).all() && (p < image_max).all(); }
#endif
#ifdef __HLSL__
	inline bool test_inside(const int2 p) { return all(p >= image_min) && all(p < image_max); }
	inline float image_plane_dist() { return abs(image_max.y - image_min.y) / (2 * tan(projection.vertical_fov/2)); }
	inline float sensor_pdfW(const float cos_theta) {
		//return 1 / (cos_theta / pow2(image_plane_dist() / cos_theta));
		return pow2(image_plane_dist()) / pow3(cos_theta);
	}
	inline float3 to_world(const float2 pixel_coord, out float2 uv) {
		uv = (pixel_coord - image_min)/extent();
		float2 clip_pos = 2*uv - 1;
		clip_pos.y = -clip_pos.y;
		return normalize(projection.back_project(clip_pos));
	}
	inline bool to_raster(const float3 pos, out float2 uv) {
		float4 screen_pos = projection.project_point(pos);
		screen_pos.y = -screen_pos.y;
		screen_pos.xyz /= screen_pos.w;
		if (any(abs(screen_pos.xyz) >= 1) || screen_pos.z <= 0) return false;
		uv = image_min + extent() * (screen_pos.xy*.5 + .5);
		return true;
	}
#endif
};

struct VisibilityInfo {
	uint instance_primitive_index;
	uint packed_normal;

	inline uint instance_index()  { return BF_GET(instance_primitive_index, 0, 16); }
	inline uint primitive_index() { return BF_GET(instance_primitive_index, 16, 16); }
#ifdef __HLSL__
	inline float3 normal()   { return unpack_normal_octahedron(packed_normal); }
#endif
};
struct DepthInfo {
	float z;
	float prev_z;
	float2 dz_dxy;
};

struct PackedVertexData {
	float3 position;
	float u;
	float3 normal;
	float v;
	inline float2 uv() { return float2(u, v); }
#ifdef __cplusplus
	inline PackedVertexData(const float3& p, const float3& n, const float2& uv) {
#else
	__init(const float3 p, const float3 n, const float2 uv) {
#endif
		position = p;
		u = uv[0];
		normal = n;
		v= uv[1];
	}
};

#ifdef __HLSL__
#include "../common/scene.hlsli"
#endif // __HLSL__

#ifdef __cplusplus
} // namespace tinyvkpt
#pragma pack(pop)
#endif

#endif