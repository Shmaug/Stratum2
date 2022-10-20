#ifndef BDPT_H
#define BDPT_H

#include "scene.h"
#include "reservoir.h"

#ifdef __cplusplus
#pragma pack(push)
#pragma pack(1)
namespace tinyvkpt {
#endif

enum BDPTFlagBits {
	ePerformanceCounters,
	eRemapThreads,
	eCoherentRR,
	eCoherentSampling,
	eFlipTriangleUVs,
	eFlipNormalMaps,
	eAlphaTest,
	eNormalMaps,
	eShadingNormalShadowFix,
	eRayCones,
	eNoViewPaths,
	eNEE,
	eNEEReservoirs,
	eNEEReservoirReuse,
	eMIS,
	eSampleLightPower,
	eUniformSphereSampling,
	ePresampleLights,
	eDeferShadowRays,
	eConnectToViews,
	eConnectToLightPaths,
	eLVC,
	eLVCReservoirs,
	eLVCReservoirReuse,
	eReferenceBDPT,
	eHashGridJitter,
	eSampleEnvironmentMapDirectly,
	eBDPTFlagCount,
};

#define BDPT_CHECK_FLAG(mask, flag) (bool)(mask & BIT((uint)flag))
#define BDPT_SET_FLAG(mask, flag) mask |= BIT((uint)flag)
#define BDPT_UNSET_FLAG(mask, flag) mask &= ~BIT((uint)flag)

#define BDPT_FLAG_HAS_ENVIRONMENT 			BIT(0)
#define BDPT_FLAG_HAS_EMISSIVES 			BIT(1)
#define BDPT_FLAG_HAS_MEDIA 				BIT(2)
#define BDPT_FLAG_TRACE_LIGHT				BIT(3)

struct BDPTPushConstants {
	uint2 gOutputExtent;
	uint gViewCount;
	uint gLightCount;

	uint gLightDistributionPDF;
	uint gLightDistributionCDF;
	uint gEnvironmentMaterialAddress;
	float gEnvironmentSampleProbability;

	uint gRandomSeed;
	uint gMinPathVertices;
	uint gMaxPathVertices;
	uint gMaxDiffuseVertices;

	uint gMaxNullCollisions;
	uint gLightPresampleTileSize;
	uint gLightPresampleTileCount;
	uint gLightPathCount;

	uint gReservoirM;
	uint gReservoirMaxM;
	uint gReservoirSpatialM;
	uint gHashGridBucketCount;

	float gHashGridMinBucketRadius;
	float gHashGridBucketPixelRadius;
	uint gDebugViewPathLength;
	uint gDebugLightPathLength;

	#ifdef __cplusplus
	inline BDPTPushConstants() {
		gMinPathVertices = 4;
		gMaxPathVertices = 8;
		gMaxDiffuseVertices = 2;
		gMaxNullCollisions = 64;
		gEnvironmentSampleProbability = 0.5f;
		gLightPresampleTileSize = 1024;
		gLightPresampleTileCount = 128;
		gLightPathCount = 64;
		gReservoirM = 16;
		gReservoirMaxM = 64;
		gReservoirSpatialM = 4;
		gHashGridBucketCount = 200000;
		gHashGridMinBucketRadius = 0.1f;
		gHashGridBucketPixelRadius = 6;
	}
	#endif
};

struct ShadowRayData {
	float3 contribution;
	uint rng_offset;
	float3 ray_origin;
	uint medium;
	float3 ray_direction;
	float ray_distance;
};

struct PresampledLightPoint {
	float3 position;
	uint packed_geometry_normal;
	float3 Le;
	float pdfA; // negative for environment map samples
#ifdef __HLSL__
	inline float3 geometry_normal() { return unpack_normal_octahedron(packed_geometry_normal); }
#endif
};

#define PATH_VERTEX_FLAG_FLIP_BITANGENT	BIT(0)
#define PATH_VERTEX_FLAG_IS_BACKGROUND	BIT(1)
#define PATH_VERTEX_FLAG_IS_MEDIUM 		BIT(2)
#define PATH_VERTEX_FLAG_IS_PREV_DELTA  BIT(3)

// 64 bytes
struct PathVertex {
	float3 position;
	uint packed_geometry_normal;
	uint2 packed_beta;
	float2 uv;
	uint material_address;
	uint packed_local_dir_in;
	uint packed_shading_normal;
	uint packed_tangent;
	float prev_dVC; // dL at previous light vertex (dL_{s+2})
	float G_rev; // prev_cos_out/dist2_prev
	float prev_pdfA_fwd; // P(s+1 <- s+2)
	float path_pdf;
#ifdef __HLSL__
	inline float3 geometry_normal() { return unpack_normal_octahedron(packed_geometry_normal); }
	SLANG_MUTATING
	inline void pack_beta(const float3 beta, const uint subpath_length, const uint diffuse_vertices, const uint flags) {
		BF_SET(packed_beta[0], f32tof16(beta[0])  , 0 , 16);
		BF_SET(packed_beta[0], f32tof16(beta[1])  , 16, 16);
		BF_SET(packed_beta[1], f32tof16(beta[2])  , 0 , 16);
		BF_SET(packed_beta[1], subpath_length     , 16, 7);
		BF_SET(packed_beta[1], diffuse_vertices   , 23, 5);
		BF_SET(packed_beta[1], flags              , 28, 4);
	}
	inline float3 beta() CONST_CPP {
		return float3(f16tof32(packed_beta[0]), f16tof32(packed_beta[0] >> 16), f16tof32(packed_beta[1]));
	}
	inline uint subpath_length()   { return BF_GET(packed_beta[1], 16, 7); }
	inline uint diffuse_vertices() { return BF_GET(packed_beta[1], 23, 5); }
	inline bool is_background()    { return (bool)BF_GET(packed_beta[1], 28, 4) & PATH_VERTEX_FLAG_IS_BACKGROUND; }
	inline bool is_medium()        { return (bool)BF_GET(packed_beta[1], 28, 4) & PATH_VERTEX_FLAG_IS_MEDIUM; }
	inline bool is_prev_delta()    { return (bool)BF_GET(packed_beta[1], 28, 4) & PATH_VERTEX_FLAG_IS_PREV_DELTA; }
	inline bool flip_bitangent()   { return (bool)BF_GET(packed_beta[1], 28, 4) & PATH_VERTEX_FLAG_FLIP_BITANGENT; }

	inline float3 local_dir_in()   { return unpack_normal_octahedron(packed_local_dir_in); }
	inline float3 shading_normal() { return unpack_normal_octahedron(packed_shading_normal); }
	inline float3 tangent()        { return unpack_normal_octahedron(packed_tangent); }
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

struct ReservoirData {
	Reservoir r;
	float W;
	uint pad;
};

enum class BDPTDebugMode {
	eNone,
	eAlbedo,
	eSpecular,
	eEmission,
	eShadingNormal,
	eGeometryNormal,
	eDirOut,
	eApproxBSDF,
	ePrevUV,
	eEnvironmentSampleTest,
	eEnvironmentSamplePDF,
	eReservoirWeight,
	ePathLengthContribution,
	eLightTraceContribution,
	eViewTraceContribution,
	eDebugModeCount
};

#ifdef __cplusplus
}
#pragma pack(pop)

namespace std {
inline string to_string(const tinyvkpt::BDPTFlagBits& m) {
	switch (m) {
		default: return "Unknown";
		case tinyvkpt::BDPTFlagBits::ePerformanceCounters: return "Performance counters";
		case tinyvkpt::BDPTFlagBits::eRemapThreads: return "Remap threads";
		case tinyvkpt::BDPTFlagBits::eCoherentRR: return "Coherent RR";
		case tinyvkpt::BDPTFlagBits::eCoherentSampling: return "Coherent sampling";
		case tinyvkpt::BDPTFlagBits::eFlipTriangleUVs: return "Flip triangle UVs";
		case tinyvkpt::BDPTFlagBits::eFlipNormalMaps: return "Flip normal maps";
		case tinyvkpt::BDPTFlagBits::eAlphaTest: return "Alpha test";
		case tinyvkpt::BDPTFlagBits::eNormalMaps: return "Normal maps";
		case tinyvkpt::BDPTFlagBits::eShadingNormalShadowFix: return "Shading normal shadow fix";
		case tinyvkpt::BDPTFlagBits::eRayCones: return "Ray cones";
		case tinyvkpt::BDPTFlagBits::eNoViewPaths: return "No view paths";
		case tinyvkpt::BDPTFlagBits::eNEE: return "NEE";
		case tinyvkpt::BDPTFlagBits::eNEEReservoirs: return "NEE reservoirs";
		case tinyvkpt::BDPTFlagBits::eNEEReservoirReuse: return "NEE reservoir reuse";
		case tinyvkpt::BDPTFlagBits::eMIS: return "MIS";
		case tinyvkpt::BDPTFlagBits::eSampleLightPower: return "Sample light power";
		case tinyvkpt::BDPTFlagBits::eUniformSphereSampling: return "Uniform sphere sampling";
		case tinyvkpt::BDPTFlagBits::ePresampleLights: return "Presample lights";
		case tinyvkpt::BDPTFlagBits::eDeferShadowRays: return "Defer shadow rays";
		case tinyvkpt::BDPTFlagBits::eConnectToViews: return "Connect to views";
		case tinyvkpt::BDPTFlagBits::eConnectToLightPaths: return "Connect to light paths";
		case tinyvkpt::BDPTFlagBits::eLVC: return "Light vertex cache";
		case tinyvkpt::BDPTFlagBits::eLVCReservoirs: return "LVC reservoirs";
		case tinyvkpt::BDPTFlagBits::eLVCReservoirReuse: return "LVC reservoir reuse";
		case tinyvkpt::BDPTFlagBits::eReferenceBDPT: return "Reference BDPT";
		case tinyvkpt::BDPTFlagBits::eHashGridJitter: return "Jitter hash grid lookups ";
		case tinyvkpt::BDPTFlagBits::eSampleEnvironmentMapDirectly: return "Sample environment map directly";
	}
}
inline string to_string(const tinyvkpt::BDPTDebugMode& m) {
	switch (m) {
		default: return "Unknown";
		case tinyvkpt::BDPTDebugMode::eNone: return "None";
		case tinyvkpt::BDPTDebugMode::eAlbedo: return "Albedo";
		case tinyvkpt::BDPTDebugMode::eSpecular: return "Specular";
		case tinyvkpt::BDPTDebugMode::eEmission: return "Emission";
		case tinyvkpt::BDPTDebugMode::eShadingNormal: return "Shading normal";
		case tinyvkpt::BDPTDebugMode::eGeometryNormal: return "Geometry normal";
		case tinyvkpt::BDPTDebugMode::eDirOut: return "Bounce direction";
		case tinyvkpt::BDPTDebugMode::eApproxBSDF: return "Approximate BSDF";
		case tinyvkpt::BDPTDebugMode::ePrevUV: return "Prev UV";
		case tinyvkpt::BDPTDebugMode::eEnvironmentSampleTest: return "Environment map sampling test";
		case tinyvkpt::BDPTDebugMode::eEnvironmentSamplePDF: return "Environment map sampling PDF";
		case tinyvkpt::BDPTDebugMode::eReservoirWeight: return "Reservoir weight";
		case tinyvkpt::BDPTDebugMode::ePathLengthContribution: return "Path contribution (per length)";
		case tinyvkpt::BDPTDebugMode::eLightTraceContribution: return "Light trace contribution";
		case tinyvkpt::BDPTDebugMode::eViewTraceContribution: return "View trace contribution";
	}
};
}
#endif

#endif