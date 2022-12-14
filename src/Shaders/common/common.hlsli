#include "D3DX_DXGIFormatConvert.inl"

inline float2 packNormal2(const float3 v) {
	// Project the sphere onto the octahedron, and then onto the xy plane
	const float2 p = v.xy * (1 / (abs(v.x) + abs(v.y) + abs(v.z)));
	// Reflect the folds of the lower hemisphere over the diagonals
	return (v.z <= 0) ? ((1 - abs(p.yx)) * lerp(-1, 1, float2(p >= 0))) : p;
}
inline float3 unpackNormal2(const float2 p) {
	float3 v = float3(p, 1 - dot(1, abs(p)));
	if (v.z < 0) v.xy = (1 - abs(v.yx)) * lerp(-1, 1, float2(v.xy >= 0));
	return normalize(v);
}

inline uint packNormal(const float3 v) {
	return D3DX_FLOAT2_to_R16G16_SNORM(packNormal2(v));
}
inline float3 unpackNormal(const uint packed) {
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

inline float2 sampleUniformSphere(const float u1, const float u2) {
	return float2(2 * M_PI * u2, acos(2*u1 - 1));
}
inline float3 sampleCosHemisphere(const float u1, const float u2) {
	const float phi = (2*M_PI) * u2;
	float2 xy = sqrt(u1) * float2(cos(phi), sin(phi));
	return float3(xy[0], xy[1], sqrt(max(0.f, 1 - dot(xy, xy))));
}
inline float cosHemispherePdfW(const float cosTheta) {
	return max(cosTheta, 0.f) / M_PI;
}

float2 sampleTexel(Texture2D<float4> image, float2 rnd, out float pdf, const uint maxIterations = 10) {
 	uint2 imageExtent;
	uint levelCount;
	image.GetDimensions(0, imageExtent.x, imageExtent.y, levelCount);

	pdf = 1;
	int2 coord = 0;
	uint2 lastExtent = 1;
 	for (uint i = 1; i < min(maxIterations+1, levelCount-1); i++) {
		const uint level = levelCount-1 - i;
		uint tmp;
		uint2 extent;
		image.GetDimensions(level, extent.x, extent.y, tmp);
		coord *= int2(extent/lastExtent);

		float4 p = 0;
		if (extent.x > 1) {
			const float inv_h = 1/(float)extent.y;
			float sy = sin(M_PI * (coord.y + 0.5f)*inv_h);
			p[0] = luminance(image.Load(int3(coord + int2(0,0), (int)level)).rgb) * sy;
			p[1] = luminance(image.Load(int3(coord + int2(1,0), (int)level)).rgb) * sy;
			if (extent.y > 1) {
				sy = sin(M_PI * (coord.y + 1 + 0.5f)*inv_h);
				p[2] = luminance(image.Load(int3(coord + int2(0,1), (int)level)).rgb) * sy;
				p[3] = luminance(image.Load(int3(coord + int2(1,1), (int)level)).rgb) * sy;
			}
		}
		const float sum = dot(p, 1);
		if (sum < 1e-6) continue;
		p /= sum;

		for (int j = 0; j < 4; j++) {
			if (j == 3 || rnd.x < p[j]) {
				coord += int2(j%2, j/2);
				pdf *= p[j];
				rnd.x /= p[j];
				break;
			}
			rnd.x -= p[j];
		}

		lastExtent = extent;
	}

	pdf *= lastExtent.x*lastExtent.y;

	return (float2(coord) + rnd) / float2(lastExtent);
}
float sampleTexelPdf(Texture2D<float4> image, const float2 uv, const uint maxIterations = 10) {
 	uint2 imageExtent;
	uint levelCount;
	image.GetDimensions(0, imageExtent.x, imageExtent.y, levelCount);

	float pdf = 1;
	uint2 lastExtent = 1;
 	for (uint i = 1; i < min(maxIterations+1, levelCount-1); i++) {
		const uint level = levelCount-1 - i;
		uint tmp;
		uint2 size;
		image.GetDimensions(level, size.x, size.y, tmp);

		const int2 coord = int2(float2(size)*uv/2)*2;

		float4 p = 0;
		if (size.x > 1) {
			const float inv_h = 1/(float)size.y;
			float sy = sin(M_PI * (coord.y + 0.5f)*inv_h);
			p[0] = luminance(image.Load(int3(coord + int2(0,0), (int)level)).rgb) * sy;
			p[1] = luminance(image.Load(int3(coord + int2(1,0), (int)level)).rgb) * sy;
			if (size.y > 1) {
				sy = sin(M_PI * (coord.y + 1 + 0.5f)*inv_h);
				p[2] = luminance(image.Load(int3(coord + int2(0,1), (int)level)).rgb) * sy;
				p[3] = luminance(image.Load(int3(coord + int2(1,1), (int)level)).rgb) * sy;
			}
		}
		const float sum = dot(p, 1);
		if (sum < 1e-6) continue;
		p /= sum;

		const uint2 o = min(uint2(uv*size) - coord, 1);
		pdf *= p[o.y*2 + o.x];

		lastExtent = size;
	}

	pdf *= lastExtent.x*lastExtent.y;

	return pdf;
}
