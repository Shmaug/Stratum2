#ifndef QUATF_H
#define QUATF_H

#include "common.h"

#ifdef __cplusplus
namespace tinyvkpt {
#endif

struct quatf {
	float3 xyz;
	float w;
};
inline quatf make_quatf(const float x, const float y, const float z, const float w) {
	quatf q;
	q.xyz[0] = x;
	q.xyz[1] = y;
	q.xyz[2] = z;
	q.w = w;
	return q;
}
inline quatf make_quatf(const float3 xyz, const float w) {
	quatf q;
	q.xyz = xyz;
	q.w = w;
	return q;
}
inline quatf quatf_identity() { return make_quatf(0,0,0,1); }

inline quatf angle_axis(const float angle, const float3 axis) {
	return make_quatf(axis*sin(angle/2), cos(angle/2));
}

inline quatf qnormalize(const quatf q) {
	const float nrm = 1 / sqrt(q.xyz[0]*q.xyz[0] + q.xyz[1]*q.xyz[1] + q.xyz[2]*q.xyz[2] + q.w*q.w);
	return make_quatf(q.xyz*nrm, q.w*nrm);
}
inline quatf inverse(const quatf q) { return qnormalize( make_quatf(-q.xyz, q.w) ); }
inline quatf qmul(const quatf q1, const quatf q2) {
	return make_quatf(
		q1.w*q2.xyz + q2.w*q1.xyz + cross(q1.xyz,q2.xyz),
		q1.w*q2.w - (q1.xyz[0]*q2.xyz[0] + q1.xyz[1]*q2.xyz[1] + q1.xyz[2]*q2.xyz[2]) );
}
inline float3 rotate_vector(const quatf q, const float3 v) {
  return qmul(q, qmul(make_quatf(v,0), inverse(q))).xyz;
}
inline quatf slerp(const quatf a, quatf b, const float t) {
	float cosHalfAngle = a.w * b.w + a.xyz[0]*b.xyz[0] + a.xyz[1]*b.xyz[1] + a.xyz[2]*b.xyz[2];

	if (cosHalfAngle >= 1 || cosHalfAngle <= -1)
		return a;
	else if (cosHalfAngle < 0) {
		b.xyz = -b.xyz;
		b.w = -b.w;
		cosHalfAngle = -cosHalfAngle;
	}

	float blendA, blendB;
	if (cosHalfAngle < 0.999) {
		// do proper slerp for big angles
		const float halfAngle = acos(cosHalfAngle);
		const float oneOverSinHalfAngle = 1/sin(halfAngle);
		blendA = sin(halfAngle * (1-t)) * oneOverSinHalfAngle;
		blendB = sin(halfAngle * t) * oneOverSinHalfAngle;
	} else {
		// do lerp if angle is really small.
		blendA = 1-t;
		blendB = t;
	}
 	return qnormalize(make_quatf(blendA*a.xyz + blendB*b.xyz, blendA*a.w + blendB*b.w));
}

// a, b, c expected to be orthonormal
inline quatf make_quatf(const float3 a, const float3 b, const float3 c) {
	quatf q;
	float T = a[0] + b[1] + c[2];
	if (T > 0) {
		const float s = sqrt(T + 1) * 2;
		q.xyz[0] = (b[2] - c[1]) / s;
		q.xyz[1] = (c[0] - a[2]) / s;
		q.xyz[2] = (a[1] - b[0]) / s;
		q.w = 0.25f * s;
	} else if ( a[0] > b[1] && a[0] > c[2]) {
		const float s = sqrt(1 + a[0] - b[1] - c[2]) * 2;
		q.xyz[0] = 0.25f * s;
		q.xyz[1] = (a[1] + b[0]) / s;
		q.xyz[2] = (c[0] + a[2]) / s;
		q.w = (b[2] - c[1]) / s;
	} else if (b[1] > c[2]) {
		const float s = sqrt(1 + b[1] - a[0] - c[2]) * 2;
		q.xyz[0] = (a[1] + b[0]) / s;
		q.xyz[1] = 0.25f * s;
		q.xyz[2] = (b[2] + c[1]) / s;
		q.w = (c[0] - a[2]) / s;
	} else {
		const float s = sqrt(1 + c[2] - a[0] - b[1]) * 2;
		q.xyz[0] = (c[0] + a[2]) / s;
		q.xyz[1] = (b[2] + c[1]) / s;
		q.xyz[2] = 0.25f * s;
		q.w = (a[1] - b[0]) / s;
	}
	return qnormalize(q);
}

#ifdef __cplusplus
} // namespace tinyvkpt
#endif

#endif