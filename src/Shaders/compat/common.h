#pragma once

#include "hlslcompat.h"

#ifdef __cplusplus
#include <Core/math.hpp>
#endif

STM_NAMESPACE_BEGIN

inline float max3(const float3 x) { return max(max(x[0], x[1]), x[2]); }
inline float max4(const float4 x) { return max(max(x[0], x[1]), max(x[2], x[3])); }
inline float min3(const float3 x) { return min(min(x[0], x[1]), x[2]); }
inline float min4(const float4 x) { return min(min(x[0], x[1]), min(x[2], x[3])); }

#define DECLARE_INTEGER_POW_FNS(T) \
	inline T pow2(const T x) { return x*x; } \
	inline T pow3(const T x) { return pow2(x)*x; } \
	inline T pow4(const T x) { return pow2(x)*pow2(x); } \
	inline T pow5(const T x) { return pow4(x)*x; }
DECLARE_INTEGER_POW_FNS(float)

#ifdef __cplusplus
#undef DECLARE_INTEGER_POW_FNS
#define DECLARE_INTEGER_POW_FNS(T) \
	inline T pow2(const T x) { return x.cwiseProduct(x); } \
	inline T pow3(const T x) { return pow2(x).cwiseProduct(x); } \
	inline T pow4(const T x) { return pow2(x).cwiseProduct(pow2(x)); } \
	inline T pow5(const T x) { return pow4(x).cwiseProduct(x); }
#endif
DECLARE_INTEGER_POW_FNS(float2)
DECLARE_INTEGER_POW_FNS(float3)
DECLARE_INTEGER_POW_FNS(float4)
#undef DECLARE_INTEGER_POW_FNS

inline float safe_divide(const float numerator, const float denominator, const float value = 0) { return denominator == 0 ? value : numerator / denominator; }

inline float average(const float2 x) { return (x[0] + x[1])/2; }
inline float average(const float3 x) { return (x[0] + x[1] + x[2])/3; }
inline float average(const float4 x) { return (x[0] + x[1] + x[2] + x[3])/4; }

inline float  radians(const float  x) { return x * M_PI/180; }
inline float2 radians(const float2 x) { return x * M_PI/180; }
inline float3 radians(const float3 x) { return x * M_PI/180; }
inline float4 radians(const float4 x) { return x * M_PI/180; }
inline float  degrees(const float  x) { return x * 180/M_PI; }
inline float2 degrees(const float2 x) { return x * 180/M_PI; }
inline float3 degrees(const float3 x) { return x * 180/M_PI; }
inline float4 degrees(const float4 x) { return x * 180/M_PI; }

inline float luminance(const float3 color) { return dot(color, float3(0.2126, 0.7152, 0.0722)); }

inline float3 hueToRgb(const float hue) {
	const float x = 6*hue;
	return saturate(float3(abs(x-3) - 1, 2 - abs(x-2), 2 - abs(x-4)));
}
inline float3 hsvToRgb(const float3 hsv) {
	const float3 rgb = hueToRgb(hsv[0]) - float3(1,1,1);
	return (rgb * hsv[1] + float3(1,1,1)) * hsv[2];
}
inline float3 rgbToHcv(const float3 rgb) {
	// Based on work by Sam Hocevar and Emil Persson
	const float4 P = (rgb[1] < rgb[2]) ? float4(rgb[2], rgb[1], -1, 2.f/3.f) : float4(rgb[1], rgb[2], 0, -1.f/3.f);
	const float4 Q = (rgb[0] < P[0]) ? float4(P[0], P[1], P[3], rgb[0]) : float4(rgb[0], P[1], P[2], P[0]);
	const float C = Q[0] - min(Q[3], Q[1]);
	const float H = abs((Q[3] - Q[1]) / (6*C + 1e-6f) + Q[2]);
	return float3(H, C, Q[0]);
}
inline float3 rgbToHsv(const float3 rgb) {
	const float3 hcv = rgbToHcv(rgb);
	return float3(hcv[0], hcv[1] / (hcv[2] + 1e-6f), hcv[2]);
}

inline float3 xyzToRgb(const float3 xyz) {
	return float3(
			3.240479f * xyz[0] - 1.537150 * xyz[1] - 0.498535 * xyz[2],
		   -0.969256f * xyz[0] + 1.875991 * xyz[1] + 0.041556 * xyz[2],
			0.055648f * xyz[0] - 0.204043 * xyz[1] + 1.057311 * xyz[2]);
}

inline float3 srgbToRgb(const float3 srgb) {
	// https://en.wikipedia.org/wiki/SRGB#From_sRGB_to_CIE_XYZ
	float3 rgb;
	for (int i = 0; i < 3; i++)
		rgb[i] = srgb[i] <= 0.04045 ? srgb[i] / 12.92 : pow((srgb[i] + 0.055) / 1.055, 2.4);
	return rgb;
}
inline float3 rgbToSrgb(const float3 rgb) {
	// https://en.wikipedia.org/wiki/SRGB#From_CIE_XYZ_to_sRGB
	float3 srgb;
	for (int i = 0; i < 3; i++)
		srgb[i] = rgb[i] <= 0.0031308 ? rgb[i] * 12.92 : pow(rgb[i] * 1.055, 1/2.4) - 0.055;
	return srgb;
}

inline float3 viridisQuintic(const float x) {
	// from https://www.shadertoy.com/view/XtGGzG
	float4 x1 = float4(1, x, x*x, x*x*x); // 1 x x2 x3
	float2 x2 = float2(x1[1], x1[2]) * x1[3]; // x4 x5
	return float3(
		dot(x1, float4( 0.280268003, -0.143510503,   2.225793877, -14.815088879)) + dot(x2, float2( 25.212752309, -11.772589584)),
		dot(x1, float4(-0.002117546,  1.617109353,  -1.909305070,   2.701152864)) + dot(x2, float2(-1.685288385 ,   0.178738871)),
		dot(x1, float4( 0.300805501,  2.614650302, -12.019139090,  28.933559110)) + dot(x2, float2(-33.491294770,  13.762053843)));
}


inline float rayPlane(const float3 origin, const float3 dir, const float3 normal) {
	const float denom = dot(normal, dir);
	if (abs(denom) > 0)
		return -dot(origin, normal) / denom;
	else
		return POS_INFINITY;
}
inline float2 rayAabb(const float3 origin, const float3 inv_dir, const float3 mn, const float3 mx) {
#ifdef __cplusplus
	const float3 t0 = (mn - origin).cwiseProduct(inv_dir);
	const float3 t1 = (mx - origin).cwiseProduct(inv_dir);
#else
	const float3 t0 = (mn - origin) * inv_dir;
	const float3 t1 = (mx - origin) * inv_dir;
#endif
	return float2(max3(min(t0, t1)), min3(max(t0, t1)));
}
inline float2 raySphere(const float3 origin, const float3 dir, const float3 p, const float r) {
	const float3 f = origin - p;
	const float a = dot(dir, dir);
	const float b = dot(f, dir);
	const float3 l = a*f - dir*b;
	float det = pow2(a*r) - dot(l,l);
	if (det < 0) return float2(0,0);
	const float inv_a = 1/a;
	det = sqrt(det * inv_a) * inv_a;
	return -float2(1,1)*b*inv_a + float2(-det, det);
}


inline float stableAtan2(const float y, const float x) {
	return x == 0.0 ? (y == 0 ? 0 : (y < 0 ? -M_PI/2 : M_PI/2)) : atan2(y, x);
}

inline float2 cartesianToSphericalUv(const float3 v) {
	const float theta = stableAtan2(v[2], v[0]);
	return float2(theta*M_1_PI*.5 + .5, acos(clamp(v[1], -1.f, 1.f))*M_1_PI);
}
inline float3 sphericalUvToCartesian(float2 uv) {
	uv[0] = uv[0]*2 - 1;
	uv *= M_PI;
	const float sinPhi = sin(uv[1]);
	return float3(sinPhi*cos(uv[0]), cos(uv[1]), sinPhi*sin(uv[0]));
}

inline float pdfWtoA(const float pdfW, const float cosTheta, const float dist) {
	return pdfW * cosTheta / pow2(dist);
}
inline float pdfAtoW(const float pdfA, const float cosTheta, const float dist) {
	return pdfA * pow2(dist) / cosTheta;
}

#ifdef __SLANG_COMPILER__
#include "common/common.hlsli"
#endif

STM_NAMESPACE_END