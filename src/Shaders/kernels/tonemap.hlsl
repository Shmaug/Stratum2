#if 0
#pragma compile slangc -profile sm_6_6 -lang slang -entry main
#pragma compile slangc -profile sm_6_6 -lang slang -entry reduce_max
#endif

#include "../compat/common.h"
#include "../compat/tonemap.h"

#ifndef gMode
#define gMode 0
#endif
#ifndef gModulateAlbedo
#define gModulateAlbedo true
#endif
#ifndef gGammaCorrection
#define gGammaCorrection true
#endif

Texture2D<float4> gInput;
RWTexture2D<float4> gOutput;
Texture2D<float4> gAlbedo;
RWByteAddressBuffer gMax;
RWByteAddressBuffer gPrevMax;

struct PushConstants {
	float gExposure;
	float gExposureAlpha;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> gPushConstants;

float3 tonemap_reinhard(const float3 c) {
	//return c / (1 + c);
	const float l = luminance(c);
	const float3 tc = c / (1 + c);
	return lerp(c/(1 + l), tc, tc);
}
float3 tonemap_reinhard_extended(const float3 c, const float3 max_c) {
	return c / (1 + c) * (1 + c / pow2(lerp(max_c, 1, float3(max_c == 0))));
}

float3 tonemap_reinhard_luminance(const float3 c) {
	const float l = luminance(c);
	const float l1 = l / (1 + l);
	return c * (l1 / l);
}
float3 tonemap_reinhard_luminance_extended(const float3 c, const float max_l) {
	const float l = luminance(c);
	const float l1 = (l / (1 + l)) * (1 + l / pow2(max_l == 0 ? 1 : max_l));
	return c * (l1 / l);
}

float tonemap_uncharted2_partial1(const float x) {
  	static const float A = 0.15;
  	static const float B = 0.50;
  	static const float C = 0.10;
  	static const float D = 0.20;
  	static const float E = 0.02;
  	static const float F = 0.30;
	return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}
float3 tonemap_uncharted2_partial(const float3 x) {
  	static const float A = 0.15;
  	static const float B = 0.50;
  	static const float C = 0.10;
  	static const float D = 0.20;
  	static const float E = 0.02;
  	static const float F = 0.30;
	return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}
float3 tonemap_uncharted2(const float3 c, const float max_l) {
	return tonemap_uncharted2_partial(c) / tonemap_uncharted2_partial1(max_l == 0 ? 1 : max_l);
}

float3 tonemap_filmic(float3 c) {
	c = max(0, c - 0.004f);
	return (c * (6.2f * c + 0.5f)) / (c * (6.2f * c + 1.7f) + 0.06f);
}

float3 rtt_and_odt_fit(float3 v) {
    const float3 a = v * (v + 0.0245786f) - 0.000090537f;
    const float3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
    return a / b;
}
float3 aces_fitted(float3 v) {
	// https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
	static const float3x3 aces_input_matrix = {
		{ 0.59719, 0.35458, 0.04823 },
		{ 0.07600, 0.90834, 0.01566 },
		{ 0.02840, 0.13383, 0.83777 }
	};
	// ODT_SAT => XYZ => D60_2_D65 => sRGB
	static const float3x3 aces_output_matrix = {
		{  1.60475, -0.53108, -0.07367 },
		{ -0.10208,  1.10813, -0.00605 },
		{ -0.00327, -0.07276,  1.07602 }
	};
    v = mul(aces_input_matrix, v);
    v = rtt_and_odt_fit(v);
    return saturate(mul(aces_output_matrix, v));
}

float3 aces_approx(float3 v) {
    static const float a = 2.51f;
    static const float b = 0.03f;
    static const float c = 2.43f;
    static const float d = 0.59f;
    static const float e = 0.14f;
    v *= 0.6f;
    return saturate((v*(a*v+b))/(v*(c*v+d)+e));
}

#define gMaxQuantization 16384

SLANG_SHADER("compute")
[numthreads(8, 8, 1)]
void reduce_max(uint3 index : SV_DispatchThreadID, uint group_index : SV_GroupIndex) {
	uint2 resolution;
	gInput.GetDimensions(resolution.x, resolution.y);
    if (any(index.xy >= resolution)) return;

	float4 v = float4(gInput[index.xy].rgb, 0);
	if (gModulateAlbedo) v.rgb *= gAlbedo[index.xy].rgb;
	v.w = luminance(v.rgb);

	/*
	static const int r = 2;
	float2 moments = float2(v.w, pow2(v.w));
	float4 avg = v;
	float4 mn = v;
	for (int x = -r; x <= r; x++)
		for (int y = -r; y <= r; y++) {
			if (x == 0 && y == 0) continue;
			const int2 p = int2(index.xy) + int2(x,y);
    		if (any(p < 0) || any(p >= resolution)) continue;

			float4 vp = float4(gInput[p].rgb, 0);
			if (gModulateAlbedo) vp.rgb *= gAlbedo[p].rgb;
			vp.w = luminance(vp.rgb);

			avg += vp;
			moments += float2(vp.w, pow2(vp.w));
			mn = min(vp, mn);
		}
	avg     /= pow2(2*r+1);
	moments /= pow2(2*r+1);

	v = avg;
	*/

	if (any(v != v) || v.w <= 0) return;

	const uint4 vi = (uint4)clamp(v*gMaxQuantization, 0, float(0xFFFFFFFF));
	uint4 prev;
	gMax.InterlockedMax(0 , vi.x);
	gMax.InterlockedMax(4 , vi.y);
	gMax.InterlockedMax(8 , vi.z);
	gMax.InterlockedMax(12, vi.w);
	//if (vi.x > 0xFFFFFFFF - prev.x) gMax.InterlockedOr(40, (1 << 0));
	//if (vi.y > 0xFFFFFFFF - prev.y) gMax.InterlockedOr(40, (1 << 1));
	//if (vi.z > 0xFFFFFFFF - prev.z) gMax.InterlockedOr(40, (1 << 2));
	//if (vi.w > 0xFFFFFFFF - prev.w) gMax.InterlockedOr(40, (1 << 3));
}

SLANG_SHADER("compute")
[numthreads(8,8,1)]
void main(uint3 index : SV_DispatchThreadID) {
	uint2 resolution;
	gOutput.GetDimensions(resolution.x, resolution.y);
	if (any(index.xy >= resolution)) return;

	float4 cur_max = gMax.Load<uint4>(0)/(float)gMaxQuantization;

	//const uint overflow = gMax.Load(40);
	//if (overflow & (1 << 0)) cur_max.x = 0xFFFFFFFF/(float)gMaxQuantization;
	//if (overflow & (1 << 1)) cur_max.y = 0xFFFFFFFF/(float)gMaxQuantization;
	//if (overflow & (1 << 2)) cur_max.z = 0xFFFFFFFF/(float)gMaxQuantization;
	//if (overflow & (1 << 3)) cur_max.w = 0xFFFFFFFF/(float)gMaxQuantization;

	float2 cur_moments = float2(cur_max.w, pow2(cur_max.w));
	if (gPushConstants.gExposureAlpha > 0 && gPushConstants.gExposureAlpha < 1) {
		const float2 prev_moments = gPrevMax.Load<float2>(32);
		if (all(prev_moments == prev_moments) && prev_moments.x > 0)
			cur_moments = lerp(prev_moments, cur_moments, sqrt(gPushConstants.gExposureAlpha));

		const float4 prev_max = gPrevMax.Load<float4>(16);
		if (all(prev_max == prev_max) && prev_max.w > 0)
			cur_max = lerp(prev_max, cur_max, gPushConstants.gExposureAlpha);
	}
	if (all(index == 0)) {
		gMax.Store<float4>(16, cur_max);
		gMax.Store<float2>(32, cur_moments);
	}
	//cur_max += abs(sqrt(cur_moments.y) - cur_moments.x);

	float3 radiance = gInput[index.xy].rgb;
	const float3 albedo = gAlbedo[index.xy].rgb;
	if (gModulateAlbedo) radiance *= (1e-2 + albedo);

	radiance *= pow(2, gPushConstants.gExposure);
	switch (gMode) {
	case (uint)TonemapMode::eReinhard:
		radiance = tonemap_reinhard(radiance);
		break;
	case (uint)TonemapMode::eReinhardExtended:
		radiance = tonemap_reinhard_extended(radiance, cur_max.rgb);
		break;
	case (uint)TonemapMode::eReinhardLuminance:
		radiance = tonemap_reinhard_luminance(radiance);
		break;
	case (uint)TonemapMode::eReinhardLuminanceExtended:
		radiance = tonemap_reinhard_luminance_extended(radiance, cur_max.w);
		break;
	case (uint)TonemapMode::eUncharted2:
		radiance = tonemap_uncharted2(radiance, cur_max.w);
		break;
	case (uint)TonemapMode::eFilmic:
		radiance = tonemap_filmic(radiance);
		break;
	case (uint)TonemapMode::eACES:
		radiance = aces_fitted(radiance);
		break;
	case (uint)TonemapMode::eACESApprox:
		radiance = aces_approx(radiance);
		break;
	case (uint)TonemapMode::eViridisR:
		radiance = viridis_quintic(saturate(luminance(radiance)));
		break;
	case (uint)TonemapMode::eViridisLengthRGB: {
		float m = cur_max.w;
		if (m == 0) m = 1;
		radiance = viridis_quintic(saturate(luminance(radiance) / m));
		break;
	}
	}
	if (gGammaCorrection) radiance = rgb_to_srgb(radiance);

	gOutput[index.xy] = float4(radiance, 1);
}