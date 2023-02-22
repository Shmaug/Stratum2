#pragma once

#include "D3DX_DXGIFormatConvert.inl"

float2 packNormal2(const float3 v) {
	// Project the sphere onto the octahedron, and then onto the xy plane
	const float2 p = v.xy * (1 / (abs(v.x) + abs(v.y) + abs(v.z)));
	// Reflect the folds of the lower hemisphere over the diagonals
	return (v.z <= 0) ? ((1 - abs(p.yx)) * lerp(-1, 1, float2(p >= 0))) : p;
}
float3 unpackNormal2(const float2 p) {
	float3 v = float3(p, 1 - dot(1, abs(p)));
	if (v.z < 0) v.xy = (1 - abs(v.yx)) * lerp(-1, 1, float2(v.xy >= 0));
	return normalize(v);
}

uint packNormal(const float3 v) {
	return D3DX_FLOAT2_to_R16G16_SNORM(packNormal2(v));
}
float3 unpackNormal(const uint packed) {
	return unpackNormal2(D3DX_R16G16_SNORM_to_FLOAT2(packed));
}

float3x3 makeOrthonormal(const float3 N) {
    float3x3 r;
	if (N[0] != N[1] || N[0] != N[2])
		r[0] = float3(N[2] - N[1], N[0] - N[2], N[1] - N[0]);  // (1,1,1) x N
	else
		r[0] = float3(N[2] - N[1], N[0] + N[2], -N[1] - N[0]);  // (-1,1,1) x N
	r[0] = normalize(r[0]);
	r[1] = cross(N, r[0]);
    r[2] = N;
    return r;
}

float2 sampleUniformTriangle(const float u1, const float u2) {
    const float a = sqrt(u1);
    return float2(1 - a, a * u2);
}
float3 sampleUniformSphere(const float u1, const float u2) {
	const float z = u2 * 2 - 1;
    const float r = sqrt(max(1 - z * z, 0));
    const float phi = 2 * M_PI * u1;
	return float3(r * cos(phi), r * sin(phi), z);
}

float2 sampleConcentricDisc(const float u1, const float u2) {
	// from pbrtv3, sampling.cpp line 113

    // Map uniform random numbers to $[-1,1]^2$
    const float2 uOffset = 2 * float2(u1,u2) - 1;

    // Handle degeneracy at the origin
    if (uOffset.x == 0 && uOffset.y == 0) return 0;

    // Apply concentric mapping to point
    float theta, r;
    if (abs(uOffset.x) > abs(uOffset.y)) {
        r = uOffset.x;
        theta = M_PI/4 * (uOffset.y / uOffset.x);
    } else {
        r = uOffset.y;
        theta = M_PI/2 - M_PI/4 * (uOffset.x / uOffset.y);
    }
    return r * float2(cos(theta), sin(theta));
}
float concentricDiscPdfA() {
    return 1.0 / M_PI;
}

float3 sampleCosHemisphere(const float u1, const float u2) {
    const float2 xy = sampleConcentricDisc(u1, u2);
	return float3(xy, sqrt(max(0, 1 - dot(xy,xy))));
}
float cosHemispherePdfW(const float cosTheta) {
	return max(cosTheta, 0.f) / M_PI;
}