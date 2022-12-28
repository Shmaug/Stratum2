#pragma once

#include "common.h"

STM_NAMESPACE_BEGIN

struct quatf {
	float3 xyz;
	float w;

	inline static quatf identity() { return quatf(0,0,0,1); }

	SLANG_CTOR(quatf) (const float _x, const float _y, const float _z, const float _w) {
		xyz[0] = _x;
		xyz[1] = _y;
		xyz[2] = _z;
		w = _w;
	}
	SLANG_CTOR(quatf) (const float3 _xyz, const float _w) {
		xyz = _xyz;
		w = _w;
	}

	// a, b, c expected to be orthonormal
	SLANG_CTOR(quatf) (const float3 a, const float3 b, const float3 c) {
		float T = a[0] + b[1] + c[2];
		if (T > 0) {
			const float s = sqrt(T + 1) * 2;
			xyz[0] = (b[2] - c[1]) / s;
			xyz[1] = (c[0] - a[2]) / s;
			xyz[2] = (a[1] - b[0]) / s;
			w = 0.25f * s;
		} else if ( a[0] > b[1] && a[0] > c[2]) {
			const float s = sqrt(1 + a[0] - b[1] - c[2]) * 2;
			xyz[0] = 0.25f * s;
			xyz[1] = (a[1] + b[0]) / s;
			xyz[2] = (c[0] + a[2]) / s;
			w = (b[2] - c[1]) / s;
		} else if (b[1] > c[2]) {
			const float s = sqrt(1 + b[1] - a[0] - c[2]) * 2;
			xyz[0] = (a[1] + b[0]) / s;
			xyz[1] = 0.25f * s;
			xyz[2] = (b[2] + c[1]) / s;
			w = (c[0] - a[2]) / s;
		} else {
			const float s = sqrt(1 + c[2] - a[0] - b[1]) * 2;
			xyz[0] = (c[0] + a[2]) / s;
			xyz[1] = (b[2] + c[1]) / s;
			xyz[2] = 0.25f * s;
			w = (a[1] - b[0]) / s;
		}

		const float nrm = 1 / sqrt(dot(xyz, xyz) + w*w);
		xyz *= nrm;
		w *= nrm;
	}

	inline static quatf angleAxis(const float angle, const float3 axis) {
		return quatf(axis*sin(angle/2), cos(angle/2));
	}
};


inline quatf normalize(const quatf q) {
	const float nrm = 1 / sqrt(dot(q.xyz, q.xyz) + q.w*q.w);
	return quatf(q.xyz*nrm, q.w*nrm);
}
inline quatf inverse(const quatf q) { return normalize( quatf(-q.xyz, q.w) ); }
inline quatf qmul(const quatf q1, const quatf q2) {
	return quatf(
		q1.w*q2.xyz + q2.w*q1.xyz + cross(q1.xyz,q2.xyz),
		q1.w*q2.w - (q1.xyz[0]*q2.xyz[0] + q1.xyz[1]*q2.xyz[1] + q1.xyz[2]*q2.xyz[2]) );
}
inline float3 rotateVector(const quatf q, const float3 v) {
return qmul(q, qmul(quatf(v,0), inverse(q))).xyz;
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
	return normalize(quatf(blendA*a.xyz + blendB*b.xyz, blendA*a.w + blendB*b.w));
}

STM_NAMESPACE_END