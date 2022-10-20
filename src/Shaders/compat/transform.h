#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "quatf.h"

#ifdef __cplusplus
namespace tinyvkpt {
#endif

struct TransformData {
	float3x4 m;

	inline float3 transform_vector(float3 v) CONST_CPP {
#ifdef __cplusplus
		return m.block<3, 3>(0, 0).matrix() * v.matrix();
#else
		return mul(m, float4(v,0));
#endif
	}

	inline float3 transform_point(float3 v) CONST_CPP {
#ifdef __cplusplus
		return (m.matrix() * v.matrix().homogeneous()).col(3).head<3>();
#else
		return mul(m, float4(v, 1));
#endif
	}

	inline TransformData inverse() CONST_CPP {
		TransformData r;
#ifdef __cplusplus
		Eigen::Matrix4f m4x4;
		m4x4.topRows<3>() = m;
		m4x4.bottomRows<1>() = float4(0,0,0,1);
		r.m = m4x4.inverse().topRows<3>();
#else
		float a00 = m[0][0], a01 = m[0][1], a02 = m[0][2];
		float a10 = m[1][0], a11 = m[1][1], a12 = m[1][2];
		float a20 = m[2][0], a21 = m[2][1], a22 = m[2][2];
		float b01 = a22 * a11 - a12 * a21;
		float b11 = -a22 * a10 + a12 * a20;
		float b21 = a21 * a10 - a11 * a20;
		float det = 1 / (a00 * b01 + a01 * b11 + a02 * b21);
		r.m[0] = float4(b01 * det, (-a22 * a01 + a02 * a21) * det, (a12 * a01 - a02 * a11) * det, -m[0][3]);
		r.m[1] = float4(b11 * det, (a22 * a00 - a02 * a20) * det, (-a12 * a00 + a02 * a10) * det, -m[1][3]);
		r.m[2] = float4(b21 * det, (-a21 * a00 + a01 * a20) * det, (a11 * a00 - a01 * a10) * det, -m[2][3]);
#endif
		return r;
	}
};

inline TransformData make_transform(const float3 t, const quatf r, const float3 s) {
	TransformData result;
#ifdef __cplusplus
	result.m = Eigen::Affine3f(Eigen::Translation3f(t)).matrix().topRows<3>();
	result.m.block<3, 3>(0, 0) = (Eigen::Quaternionf(r.w, r.xyz[0], r.xyz[1], r.xyz[2]) * Eigen::Scaling(s.matrix())).matrix();
#else
	// https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
	float sqw = r.w * r.w;
	float sqx = r.xyz[0] * r.xyz[0];
	float sqy = r.xyz[1] * r.xyz[1];
	float sqz = r.xyz[2] * r.xyz[2];

	// invs (inverse square length) is only required if quaternion is not already normalised
	float invs = 1 / (sqx + sqy + sqz + sqw);
	result.m[0][0] = (sqx - sqy - sqz + sqw) * invs; // since sqw + sqx + sqy + sqz =1/invs*invs
	result.m[1][1] = (-sqx + sqy - sqz + sqw) * invs;
	result.m[2][2] = (-sqx - sqy + sqz + sqw) * invs;

	float tmp1 = r.xyz[0] * r.xyz[2];
	float tmp2 = r.xyz[1] * r.w;
	result.m[2][0] = 2 * (tmp1 + tmp2) * invs;
	result.m[0][2] = 2 * (tmp1 - tmp2) * invs;

	tmp1 = r.xyz[0] * r.xyz[1];
	tmp2 = r.xyz[2] * r.w;
	result.m[1][0] = 2 * (tmp1 - tmp2) * invs;
	result.m[0][1] = 2 * (tmp1 + tmp2) * invs;

	tmp1 = r.xyz[2] * r.xyz[1];
	tmp2 = r.xyz[0] * r.w;
	result.m[1][2] = 2 * (tmp1 + tmp2) * invs;
	result.m[2][1] = 2 * (tmp1 - tmp2) * invs;

	result.m[0][3] = t[0];
	result.m[1][3] = t[1];
	result.m[2][3] = t[2];
#endif
	return result;
}

inline TransformData tmul(const TransformData lhs, const TransformData rhs) {
	TransformData r;
#ifdef __cplusplus
	Eigen::Matrix4f m4x4;
	m4x4.topRows<3>() = rhs.m;
	m4x4.bottomRows<1>() = float4(0,0,0,1);
	r.m = (lhs.m.matrix() * m4x4).topRows<3>();
#else
	float4x4 rhs4x4;
	rhs4x4[0] = rhs.m[0];
	rhs4x4[1] = rhs.m[1];
	rhs4x4[2] = rhs.m[2];
	rhs4x4[3] = float4(0,0,0,1);
	r.m = (float3x4)mul(lhs.m, rhs4x4);
#endif
	return r;
}

inline float3x4 to_float3x4(const TransformData t) { return t.m; }
inline TransformData from_float3x4(const float3x4 m) { TransformData t = { m }; return t; }

struct ProjectionData {
	float2 scale;
	float2 offset;
	float near_plane;
	float far_plane;
	float sensor_area;
	float vertical_fov;

	inline bool orthographic() CONST_CPP { return vertical_fov < 0; }

	// uses reversed z (1 at near plane -> 0 at far plane)
	inline float4 project_point(const float3 v) CONST_CPP {
		float4 r;
		if (orthographic()) {
			r[0] = v[0] * scale[0] + offset[0];
			r[1] = v[1] * scale[1] + offset[1];
			r[2] = (v[2] - far_plane) / (near_plane - far_plane);
			r[3] = 1;
		} else { // perspective
			// infinite far plane
			r[0] = v[0] * scale[0] + v[2] * offset[0];
			r[1] = v[1] * scale[1] + v[2] * offset[1];
			r[2] = abs(near_plane);
			r[3] = v[2] * sign(near_plane);
		}
		return r;
	}
	inline float3 back_project(const float2 v) CONST_CPP {
		float3 r;
		if (orthographic()) {
			r[0] = (v[0] - offset[0]) / scale[0];
			r[1] = (v[1] - offset[1]) / scale[1];
		} else { // perspective
			r[0] = near_plane * (v[0] * sign(near_plane) - offset[0]) / scale[0];
			r[1] = near_plane * (v[1] * sign(near_plane) - offset[1]) / scale[1];
		}
		r[2] = near_plane;
		return r;
	}
};

inline ProjectionData make_orthographic(const float2 size, const float2 offset, const float znear, const float zfar) {
	ProjectionData r;
	r.scale = 2 / size;
	r.offset = offset;
	r.near_plane = znear;
	r.far_plane = zfar;
	r.vertical_fov = -1;
	return r;
}
inline ProjectionData make_perspective(const float fovy, const float aspect, const float2 offset, const float znear) {
	ProjectionData r;
	r.scale[1] = 1 / tan(fovy / 2);
	r.scale[0] = aspect * r.scale[1];
	r.offset = offset;
	r.near_plane = znear;
	r.far_plane = 0;
	r.vertical_fov = fovy;
	return r;
}

#ifdef __cplusplus
} // namespace tinyvkpt
#endif

#endif
