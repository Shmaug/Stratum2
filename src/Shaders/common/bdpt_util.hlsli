#ifndef BDPT_UTIL_H
#define BDPT_UTIL_H

#ifndef gSceneFlags
#define gSceneFlags 0
#endif
#ifndef gSpecializationFlags
#define gSpecializationFlags 0
#endif
#ifndef gDebugMode
#define gDebugMode 0
#endif
#ifndef gLightTraceQuantization
#define gLightTraceQuantization 16384
#endif
#ifndef gCoherentRNG
#define gCoherentRNG false
#endif
//#define FORCE_LAMBERTIAN

#define gHasEnvironment                (bool)(gSceneFlags & BDPT_FLAG_HAS_ENVIRONMENT)
#define gHasEmissives                  (bool)(gSceneFlags & BDPT_FLAG_HAS_EMISSIVES)
#define gHasMedia                      (bool)(gSceneFlags & BDPT_FLAG_HAS_MEDIA)
#define gTraceLight                    (bool)(gSceneFlags & BDPT_FLAG_TRACE_LIGHT)

#define gUsePerformanceCounters        BDPT_CHECK_FLAG(gSpecializationFlags, BDPTFlagBits::ePerformanceCounters)
#define gRemapThreadIndex              BDPT_CHECK_FLAG(gSpecializationFlags, BDPTFlagBits::eRemapThreads)
#define gCoherentRR                    BDPT_CHECK_FLAG(gSpecializationFlags, BDPTFlagBits::eCoherentRR)
#define gCoherentSampling              BDPT_CHECK_FLAG(gSpecializationFlags, BDPTFlagBits::eCoherentSampling)
#define gFlipTriangleUVs               BDPT_CHECK_FLAG(gSpecializationFlags, BDPTFlagBits::eFlipTriangleUVs)
#define gFlipNormalMaps	               BDPT_CHECK_FLAG(gSpecializationFlags, BDPTFlagBits::eFlipNormalMaps)
#define gAlphaTest                     BDPT_CHECK_FLAG(gSpecializationFlags, BDPTFlagBits::eAlphaTest)
#define gUseNormalMaps                 BDPT_CHECK_FLAG(gSpecializationFlags, BDPTFlagBits::eNormalMaps)
#define gShadingNormalFix              BDPT_CHECK_FLAG(gSpecializationFlags, BDPTFlagBits::eShadingNormalShadowFix)
#define gUseRayCones                   BDPT_CHECK_FLAG(gSpecializationFlags, BDPTFlagBits::eRayCones)
#define gNoViewPaths                   BDPT_CHECK_FLAG(gSpecializationFlags, BDPTFlagBits::eNoViewPaths)
#define gSampleLightPower              BDPT_CHECK_FLAG(gSpecializationFlags, BDPTFlagBits::eSampleLightPower)
#define gUniformSphereSampling         BDPT_CHECK_FLAG(gSpecializationFlags, BDPTFlagBits::eUniformSphereSampling)
#define gUseMIS                        BDPT_CHECK_FLAG(gSpecializationFlags, BDPTFlagBits::eMIS)
#define gConnectToLights               BDPT_CHECK_FLAG(gSpecializationFlags, BDPTFlagBits::eNEE)
#define gUseNEEReservoirs              BDPT_CHECK_FLAG(gSpecializationFlags, BDPTFlagBits::eNEEReservoirs)
#define gUseNEEReservoirReuse          BDPT_CHECK_FLAG(gSpecializationFlags, BDPTFlagBits::eNEEReservoirReuse)
#define gPresampleLights               BDPT_CHECK_FLAG(gSpecializationFlags, BDPTFlagBits::ePresampleLights)
#define gDeferShadowRays               BDPT_CHECK_FLAG(gSpecializationFlags, BDPTFlagBits::eDeferShadowRays)
#define gConnectToViews                BDPT_CHECK_FLAG(gSpecializationFlags, BDPTFlagBits::eConnectToViews)
#define gConnectToLightPaths           BDPT_CHECK_FLAG(gSpecializationFlags, BDPTFlagBits::eConnectToLightPaths)
#define gReferenceBDPT                 BDPT_CHECK_FLAG(gSpecializationFlags, BDPTFlagBits::eReferenceBDPT)
#define gLightVertexCache              BDPT_CHECK_FLAG(gSpecializationFlags, BDPTFlagBits::eLVC)
#define gUseLVCReservoirs              BDPT_CHECK_FLAG(gSpecializationFlags, BDPTFlagBits::eLVCReservoirs)
#define gUseLVCReservoirReuse          BDPT_CHECK_FLAG(gSpecializationFlags, BDPTFlagBits::eLVCReservoirReuse)
#define gHashGridJitter                BDPT_CHECK_FLAG(gSpecializationFlags, BDPTFlagBits::eHashGridJitter)
#define gSampleEnvironmentMap          BDPT_CHECK_FLAG(gSpecializationFlags, BDPTFlagBits::eSampleEnvironmentMapDirectly)

#define gOutputExtent                  gPushConstants.gOutputExtent
#define gViewCount                     gPushConstants.gViewCount
#define gLightCount                    gPushConstants.gLightCount
#define gLightDistributionPDF          gPushConstants.gLightDistributionPDF
#define gLightDistributionCDF          gPushConstants.gLightDistributionCDF
#define gEnvironmentMaterialAddress    gPushConstants.gEnvironmentMaterialAddress
#define gEnvironmentSampleProbability  gPushConstants.gEnvironmentSampleProbability
#define gRandomSeed                    gPushConstants.gRandomSeed
#define gMinPathVertices               gPushConstants.gMinPathVertices
#define gMaxPathVertices               gPushConstants.gMaxPathVertices
#define gMaxDiffuseVertices            gPushConstants.gMaxDiffuseVertices
#define gMaxNullCollisions             gPushConstants.gMaxNullCollisions
#define gLightPresampleTileSize        gPushConstants.gLightPresampleTileSize
#define gLightPresampleTileCount       gPushConstants.gLightPresampleTileCount
#define gReservoirM                    gPushConstants.gReservoirM
#define gReservoirMaxM                 gPushConstants.gReservoirMaxM
#define gReservoirSpatialM             gPushConstants.gReservoirSpatialM
#define gLightPathCount                gPushConstants.gLightPathCount
#define gHashGridBucketCount           gPushConstants.gHashGridBucketCount
#define gHashGridMinBucketRadius       gPushConstants.gHashGridMinBucketRadius
#define gHashGridBucketPixelRadius     gPushConstants.gHashGridBucketPixelRadius


uint map_pixel_coord(const uint2 pixel_coord, const uint2 group_id, const uint group_thread_index) {
	if (gRemapThreadIndex) {
		const uint dispatch_w = (gOutputExtent.x + GROUPSIZE_X - 1) / GROUPSIZE_X;
		const uint group_index = group_id.y*dispatch_w + group_id.x;
		return group_index*GROUPSIZE_X*GROUPSIZE_Y + group_thread_index;
	} else
		return pixel_coord.y*gOutputExtent.x + pixel_coord.x;
}

float2 sample_texel(Texture2D<float4> img, float2 rnd, out float pdf, const uint max_iterations = 10) {
	static const uint2 offsets[4] = {
		uint2(0,0),
		uint2(1,0),
		uint2(0,1),
		uint2(1,1),
	};

 	uint2 full_size;
	uint level_count;
	img.GetDimensions(0, full_size.x, full_size.y, level_count);

	pdf = 1;
	uint2 coord = 0;
	uint2 last_size = 1;
 	for (uint i = 1; i < min(max_iterations+1, level_count-1); i++) {
		const uint level = level_count-1 - i;
		uint tmp;
		uint2 size;
		img.GetDimensions(level, size.x, size.y, tmp);
		coord *= size/last_size;

		const float inv_h = 1/(float)size.y;

		uint j;
		float4 p = 0;
		if (size.x > 1)
			for (j = 0; j < 2; j++)
				p[j] = luminance(img.Load((int3)uint3(coord + offsets[j], level)).rgb) * sin(M_PI * (coord.y + offsets[j].y + 0.5f)*inv_h);
		if (size.y > 1)
			for (j = 2; j < 4; j++)
				p[j] = luminance(img.Load((int3)uint3(coord + offsets[j], level)).rgb) * sin(M_PI * (coord.y + offsets[j].y + 0.5f)*inv_h);
		const float sum = dot(p, 1);
		if (sum < 1e-6) continue;
		p /= sum;

		for (j = 0; j < 4; j++) {
			if (rnd.x < p[j]) {
				coord += offsets[j];
				pdf *= p[j];
				rnd.x /= p[j];
				break;
			}
			rnd.x -= p[j];
		}
		last_size = size;
	}

	pdf *= last_size.x*last_size.y;

	return (float2(coord) + rnd) / float2(last_size);
}
float sample_texel_pdf(Texture2D<float4> img, const float2 uv, const uint max_iterations = 10) {
	static const uint2 offsets[4] = {
		uint2(0,0),
		uint2(1,0),
		uint2(0,1),
		uint2(1,1),
	};

 	uint2 full_size;
	uint level_count;
	img.GetDimensions(0, full_size.x, full_size.y, level_count);

	float pdf = 1;
	uint2 last_size = 1;
 	for (uint i = 1; i < min(max_iterations+1, level_count-1); i++) {
		const uint level = level_count-1 - i;
		uint tmp;
		uint2 size;
		img.GetDimensions(level, size.x, size.y, tmp);

		const uint2 coord = floor(size*uv/2)*2;

		const float inv_h = 1/(float)size.y;

		uint j;
		float4 p = 0;
		if (size.x > 1)
			for (j = 0; j < 2; j++)
				p[j] = luminance(img.Load(uint3(coord + offsets[j], level)).rgb) * sin(M_PI * (coord.y + offsets[j].y + 0.5f)*inv_h);
		if (size.y > 1)
			for (j = 2; j < 4; j++)
				p[j] = luminance(img.Load(uint3(coord + offsets[j], level)).rgb) * sin(M_PI * (coord.y + offsets[j].y + 0.5f)*inv_h);
		const float sum = dot(p, 1);
		if (sum < 1e-6) continue;
		p /= sum;

		const uint2 o = saturate(uint2(uv*size) - coord);
		pdf *= p[o.y*2 + o.x];

		last_size = size;
	}
	pdf *= last_size.x*last_size.y;
	return pdf;
}


#endif