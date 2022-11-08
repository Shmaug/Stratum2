#ifndef SHADER_COMMON_H
#define SHADER_COMMON_H

#include "hlslcompat.h"

#define POS_INFINITY asfloat(0x7F800000)
#define NEG_INFINITY asfloat(0xFF800000)

#ifndef M_PI
#define M_PI (3.1415926535897932)
#endif
#ifndef M_1_PI
#define M_1_PI (1/M_PI)
#endif

#ifdef __cplusplus
namespace tinyvkpt {
#endif

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

inline float  radians(const float  x) { return x * M_PI/180; }
inline float2 radians(const float2 x) { return x * M_PI/180; }
inline float3 radians(const float3 x) { return x * M_PI/180; }
inline float4 radians(const float4 x) { return x * M_PI/180; }
inline float  degrees(const float  x) { return x * 180/M_PI; }
inline float2 degrees(const float2 x) { return x * 180/M_PI; }
inline float3 degrees(const float3 x) { return x * 180/M_PI; }
inline float4 degrees(const float4 x) { return x * 180/M_PI; }

inline float len_sqr(const float3 v) { return dot(v,v); }

inline float luminance(const float3 color) { return dot(color, float3(0.2126, 0.7152, 0.0722)); }

inline float3 hue_to_rgb(const float hue) {
	float x = 6*hue;
	return saturate(float3(abs(x-3) - 1, 2 - abs(x-2), 2 - abs(x-4)));
}
inline float3 hsv_to_rgb(const float3 hsv) {
	float3 rgb = hue_to_rgb(hsv[0]) - float3(1,1,1);
	return (rgb * hsv[1] + float3(1,1,1)) * hsv[2];
}
inline float3 rgb_to_hcv(const float3 rgb) {
	// Based on work by Sam Hocevar and Emil Persson
	float4 P = (rgb[1] < rgb[2]) ? float4(rgb[2], rgb[1], -1, 2.f/3.f) : float4(rgb[1], rgb[2], 0, -1.f/3.f);
	float4 Q = (rgb[0] < P[0]) ? float4(P[0], P[1], P[3], rgb[0]) : float4(rgb[0], P[1], P[2], P[0]);
	float C = Q[0] - min(Q[3], Q[1]);
	float H = abs((Q[3] - Q[1]) / (6*C + 1e-6f) + Q[2]);
	return float3(H, C, Q[0]);
}
inline float3 rgb_to_hsv(const float3 rgb) {
	float3 hcv = rgb_to_hcv(rgb);
	return float3(hcv[0], hcv[1] / (hcv[2] + 1e-6f), hcv[2]);
}

inline float3 xyz_to_rgb(const float3 xyz) {
	return float3(
			3.240479f * xyz[0] - 1.537150 * xyz[1] - 0.498535 * xyz[2],
		 -0.969256f * xyz[0] + 1.875991 * xyz[1] + 0.041556 * xyz[2],
			0.055648f * xyz[0] - 0.204043 * xyz[1] + 1.057311 * xyz[2]);
}

inline float3 srgb_to_rgb(const float3 srgb) {
	// https://en.wikipedia.org/wiki/SRGB#From_sRGB_to_CIE_XYZ
	float3 rgb;
	for (int i = 0; i < 3; i++)
		rgb[i] = srgb[i] <= 0.04045 ? srgb[i] / 12.92 : pow((srgb[i] + 0.055) / 1.055, 2.4);
	return rgb;
}
inline float3 rgb_to_srgb(const float3 rgb) {
	// https://en.wikipedia.org/wiki/SRGB#From_CIE_XYZ_to_sRGB
	float3 srgb;
	for (int i = 0; i < 3; i++)
		srgb[i] = rgb[i] <= 0.0031308 ? rgb[i] * 12.92 : pow(rgb[i] * 1.055, 1/2.4) - 0.055;
	return srgb;
}

inline float3 viridis_quintic(const float x) {
	// from https://www.shadertoy.com/view/XtGGzG
	float4 x1 = float4(1, x, x*x, x*x*x); // 1 x x2 x3
	float2 x2 = float2(x1[1], x1[2]) * x1[3]; // x4 x5
	return float3(
		dot(x1, float4( 0.280268003, -0.143510503,   2.225793877, -14.815088879)) + dot(x2, float2( 25.212752309, -11.772589584)),
		dot(x1, float4(-0.002117546,  1.617109353,  -1.909305070,   2.701152864)) + dot(x2, float2(-1.685288385 ,   0.178738871)),
		dot(x1, float4( 0.300805501,  2.614650302, -12.019139090,  28.933559110)) + dot(x2, float2(-33.491294770,  13.762053843)));
}

inline void make_orthonormal(const float3 N, __hlsl_out(float3) T, __hlsl_out(float3) B) {
	if (N[0] != N[1] || N[0] != N[2])
		T = float3(N[2] - N[1], N[0] - N[2], N[1] - N[0]);  //(1,1,1)x N
	else
		T = float3(N[2] - N[1], N[0] + N[2], -N[1] - N[0]);  //(-1,1,1)x N
	T = normalize(T);
	B = cross(N, T);
}

inline float stable_atan2(const float y, const float x) {
	return x == 0.0 ? (y == 0 ? 0 : (y < 0 ? -M_PI/2 : M_PI/2)) : atan2(y, x);
}

inline float2 cartesian_to_spherical_uv(const float3 v) {
	const float theta = stable_atan2(v[2], v[0]);
	return float2(theta*M_1_PI*.5 + .5, acos(clamp(v[1], -1.f, 1.f))*M_1_PI);
}
inline float3 spherical_uv_to_cartesian(float2 uv) {
	uv[0] = uv[0]*2 - 1;
	uv *= M_PI;
	const float sinPhi = sin(uv[1]);
	return float3(sinPhi*cos(uv[0]), cos(uv[1]), sinPhi*sin(uv[0]));
}

// use spherical_uv_to_cartesian to get XYZ
inline float2 sample_uniform_sphere(const float u1, const float u2) {
	return float2(2 * M_PI * u2, acos(2*u1 - 1));
}
inline float uniform_sphere_pdfW() { return 1/(4*M_PI); }
inline float3 sample_cos_hemisphere(const float u1, const float u2) {
	const float phi = (2*M_PI) * u2;
	float2 xy = sqrt(u1) * float2(cos(phi), sin(phi));
	return float3(xy[0], xy[1], sqrt(max(0.f, 1 - dot(xy, xy))));
}
inline float cosine_hemisphere_pdfW(const float cos_theta) {
	return max(cos_theta, 0.f) / M_PI;
}

inline float2 ray_sphere(const float3 origin, const float3 dir, const float3 p, const float r) {
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
inline float2 ray_aabb(const float3 origin, const float3 inv_dir, const float3 mn, const float3 mx) {
	#ifdef __cplusplus
	const float3 t0 = (mn - origin).cwiseProduct(inv_dir);
	const float3 t1 = (mx - origin).cwiseProduct(inv_dir);
	#else
	const float3 t0 = (mn - origin) * inv_dir;
	const float3 t1 = (mx - origin) * inv_dir;
	#endif
	return float2(max3(min(t0, t1)), min3(max(t0, t1)));
}
inline float ray_plane(const float3 origin, const float3 dir, const float3 normal) {
	const float denom = dot(normal, dir);
	if (abs(denom) > 0)
		return -dot(origin, normal) / denom;
	else
		return POS_INFINITY;
}

inline float average(const float2 x) { return (x[0] + x[1])/2; }
inline float average(const float3 x) { return (x[0] + x[1] + x[2])/3; }
inline float average(const float4 x) { return (x[0] + x[1] + x[2] + x[3])/4; }

// returns pdfW * G, where G = cos_theta / pow2(dist);
inline float pdfWtoA(const float pdfW, const float G) {
	return pdfW * G;
}
// returns pdfW / G, where G = cos_theta / pow2(dist);
inline float pdfAtoW(const float pdfA, const float G) {
	return pdfA / G;
}


// To support spectral data, we need to convert spectral measurements (how much energy at each wavelength) to
// RGB. To do this, we first convert the spectral data to CIE XYZ, by
// integrating over the XYZ response curve. Here we use an analytical response
// curve proposed by Wyman et al.: https://jcgt.org/published/0002/02/01/
inline float xFit_1931(float wavelength) {
	const float t1 = (wavelength - 442.0f) * ((wavelength < 442.0f) ? 0.0624f : 0.0374f);
	const float t2 = (wavelength - 599.8f) * ((wavelength < 599.8f) ? 0.0264f : 0.0323f);
	const float t3 = (wavelength - 501.1f) * ((wavelength < 501.1f) ? 0.0490f : 0.0382f);
	return 0.362f * exp(-0.5f * t1 * t1) + 1.056f * exp(-0.5f * t2 * t2) - 0.065f * exp(-0.5f * t3 * t3);
}
inline float yFit_1931(float wavelength) {
	const float t1 = (wavelength - 568.8) * ((wavelength < 568.8) ? 0.0213 : 0.0247f);
	const float t2 = (wavelength - 530.9) * ((wavelength < 530.9) ? 0.0613 : 0.0322f);
	return 0.821f * exp(-0.5f * t1 * t1) + 0.286f * exp(-0.5f * t2 * t2);
}
inline float zFit_1931(float wavelength) {
	const float t1 = (wavelength - 437.0) * ((wavelength < 437.0) ? 0.0845 : 0.0278f);
	const float t2 = (wavelength - 459.0) * ((wavelength < 459.0) ? 0.0385 : 0.0725f);
	return 1.217f * exp(-0.5f * t1 * t1) +0.681f * exp(-0.5f * t2 * t2);
}
inline float3 XYZintegral_coeff(const float wavelength) {
	return float3(xFit_1931(wavelength), yFit_1931(wavelength), zFit_1931(wavelength));
}

#ifdef __cplusplus
inline float3 integrate_XYZ(const std::vector<std::pair<float, float>>& data) {
	static const float CIE_Y_integral = 106.856895f;
	static const float wavelength_beg = 400;
	static const float wavelength_end = 700;
	if (data.size() == 0) {
		return float3::Zero();
	}
	float3 ret = float3::Zero();
	int data_pos = 0;
	// integrate from wavelength 400 nm to 700 nm, increment by 1nm at a time
	// linearly interpolate from the data
	for (float wavelength = wavelength_beg; wavelength <= wavelength_end; wavelength += float(1)) {
		// assume the spectrum data is sorted by wavelength
		// move data_pos such that wavelength is between two data or at one end
		while(data_pos < (int)data.size() - 1 && !((data[data_pos].first <= wavelength && data[data_pos + 1].first > wavelength) || data[0].first > wavelength)) {
			data_pos += 1;
		}
		float measurement = 0;
		if (data_pos < (int)data.size() - 1 && data[0].first <= wavelength) {
			const float curr_data = data[data_pos].second;
			const float next_data = data[std::min(data_pos + 1, (int)data.size() - 1)].second;
			const float curr_wave = data[data_pos].first;
			const float next_wave = data[std::min(data_pos + 1, (int)data.size() - 1)].first;
			// linearly interpolate
			measurement = curr_data * (next_wave - wavelength) / (next_wave - curr_wave) +
										next_data * (wavelength - curr_wave) / (next_wave - curr_wave);
		} else {
			// assign the endpoint
			measurement = data[data_pos].second;
		}
		const float3 coeff = XYZintegral_coeff(wavelength);
		ret += coeff * measurement;
	}
	const float wavelength_span = wavelength_end - wavelength_beg;
	ret *= (wavelength_span / (CIE_Y_integral * (wavelength_end - wavelength_beg)));
	return ret;
}

} // namespace tinyvkpt
#endif

#endif