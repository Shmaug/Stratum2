#ifndef MEDIUM_H
#define MEDIUM_H

#ifndef gMaxNullCollisions
#define gMaxNullCollisions 2048
#endif

#include "common/rng.hlsli"
#include "compat/scene.h"

#define PNANOVDB_HLSL
#include "../../../extern/nanovdb/PNanoVDB.h"

struct Medium {
	float3 mDensityScale;
	uint mPackedAlbedoScale;
	uint mDensityVolumeIndex;
    uint mAlbedoVolumeIndex;
    float mAnisotropy;

    __init(const uint address) {
        const uint4 data0 = gScene.mMaterialData.Load<uint4>((int)address);
        mDensityScale = asfloat(data0.xyz);
        mPackedAlbedoScale = data0.w;
        const uint3 data1 = gScene.mMaterialData.Load<uint3>((int)address + 16);
        mDensityVolumeIndex = data1[0];
        mAlbedoVolumeIndex = data1[1];
        mAnisotropy = asfloat(data1[2]);
    }

    float3 emission() { return 0; }
    float emissionPdf() { return 0; }
    float3 albedo() { return D3DX_R8G8B8A8_UNORM_to_FLOAT4(mPackedAlbedoScale).rgb; }
    bool canEvaluate() { return any(mDensityScale > 0) && mPackedAlbedoScale > 0; }
    float3 density() { return mDensityScale; }
    bool isSingular() { return abs(mAnisotropy) > 0.999; }
    float continuationProb() { return saturate(luminance(albedo() * density())); }

	// henyey greenstein phasefunction sampling

    ReflectanceEvalRecord evaluateReflectance<let Adjoint : bool>(const float3 dirIn, const float3 dirOut) {
        const float v = 1 / (4 * M_PI) * (1 - mAnisotropy * mAnisotropy) / pow(1 + mAnisotropy * mAnisotropy + 2 * mAnisotropy * dot(dirIn, dirOut), 1.5);
        ReflectanceEvalRecord r;
		r.mReflectance = v;
		r.mFwdPdfW = v;
		r.mRevPdfW = v;
		return r;
	}
    DirectionSampleRecord sampleDirection<let Adjoint : bool>(const float3 rnd, const float3 dirIn) {
        const float anisotropy2 = mAnisotropy * mAnisotropy;
		DirectionSampleRecord r;
        if (abs(mAnisotropy) < 1e-4) {
            r.mDirection = sampleUniformSphere(rnd.x, rnd.y);
		} else {
			const float tmp = (anisotropy2 - 1) / (2 * rnd.x * mAnisotropy - (mAnisotropy + 1));
			const float cos_elevation = (tmp * tmp - (1 + anisotropy2)) / (2 * mAnisotropy);
			const float sin_elevation = sqrt(max(1 - cos_elevation * cos_elevation, 0));
            const float azimuth = 2 * M_PI * rnd.y;
            const float3x3 frame = makeOrthonormal(dirIn);
			r.mDirection =
				frame[0] * sin_elevation * cos(azimuth) +
				frame[1] * sin_elevation * sin(azimuth) +
				dirIn    * cos_elevation;
		}
		const float v = 1/(4*M_PI) * (1 - anisotropy2) / pow(1 + anisotropy2 + 2 * mAnisotropy * dot(dirIn, r.mDirection), 1.5);
        r.mReflectance = v;
		r.mFwdPdfW = v;
        r.mRevPdfW = v;
		r.mEta = -1;
		r.mRoughness = 1 - abs(mAnisotropy);
		return r;
	}

	// density/transmittance sampling

    StructuredBuffer<uint> getDensityVolume() { return gScene.mVolumes[NonUniformResourceIndex(mDensityVolumeIndex)]; }
	StructuredBuffer<uint> getAlbedoVolume()  { return gScene.mVolumes[NonUniformResourceIndex(mAlbedoVolumeIndex)]; }

    float3 readDensity(inout pnanovdb_readaccessor_t density_accessor, const float3 pos_index) {
        if (mDensityVolumeIndex == -1)
            return 1;
		else
        	return pnanovdb_read_float(getDensityVolume(), pnanovdb_readaccessor_get_value_address(PNANOVDB_GRID_TYPE_FLOAT, getDensityVolume(), density_accessor, (int3)floor(pos_index)));
    }
    float3 readAlbedo(inout pnanovdb_readaccessor_t albedo_accessor, const float3 pos_index) {
		if (mAlbedoVolumeIndex == -1)
			return 1;
        else
            return pnanovdb_read_float(getAlbedoVolume(), pnanovdb_readaccessor_get_value_address(PNANOVDB_GRID_TYPE_FLOAT, getAlbedoVolume(), albedo_accessor, (int3)floor(pos_index)));
	}

	// returns hit position inside medium. multiplies beta by transmittance*sigma_s
    float3 deltaTrack<let Scattering : bool>(inout RandomSampler rng, float3 origin, float3 direction, float tmax, inout float3 beta, out float3 dirPdf, out float3 neePdf, out bool scattered) {
		scattered = false;
		dirPdf = 1;
		neePdf = 1;

		const uint channel = rng.next().x % 3;

        if (mDensityVolumeIndex == -1) {
			// homogeneous volume

            const float3 sigma_t = density();
            if (sigma_t[channel] < 1e-6) return 0;
            const float t = -log(1 - rng.nextFloat().x) / sigma_t[channel];
            if (t < tmax && Scattering) {
                scattered = true;
                beta   *= exp(-sigma_t * t) * sigma_t * albedo();
                dirPdf *= exp(-sigma_t[channel] * t) * sigma_t[channel];
                return origin + direction*t;
            }
			beta   *= exp(-sigma_t * tmax);
			dirPdf *= exp(-sigma_t[channel] * tmax);
			return 0;
		}

		pnanovdb_grid_handle_t gridHandle = { 0 };

		pnanovdb_readaccessor_t density_accessor, albedo_accessor;
		pnanovdb_readaccessor_init(density_accessor, pnanovdb_tree_get_root(getDensityVolume(), pnanovdb_grid_get_tree(getDensityVolume(), gridHandle)));
        if (mAlbedoVolumeIndex != -1)
            pnanovdb_readaccessor_init(albedo_accessor, pnanovdb_tree_get_root(getAlbedoVolume(), pnanovdb_grid_get_tree(getAlbedoVolume(), gridHandle)));


        float3 majorant = density() * pnanovdb_read_float(getDensityVolume(), pnanovdb_root_get_max_address(PNANOVDB_GRID_TYPE_FLOAT, getDensityVolume(), density_accessor.root));
        if (majorant[channel] < 1e-6) return 0;

		origin    = pnanovdb_grid_world_to_indexf    (getDensityVolume(), gridHandle, origin);
		direction = pnanovdb_grid_world_to_index_dirf(getDensityVolume(), gridHandle, direction);

        const float invMajorantChannel = 1 / majorant[channel];
        const float invMajorantMax = 1 / max3(majorant);

        for (uint iteration = 0; iteration < gMaxNullCollisions && any(beta > 0); iteration++) {
            const float t = -log(1 - rng.nextFloat().x) * invMajorantChannel;

			if (t >= tmax) {
				// transmitted without scattering
				const float3 tr = exp(-majorant * tmax);
				beta *= tr;
				neePdf *= tr;
				dirPdf *= tr;
				break;
			}

			origin += direction*t;
			tmax -= t;

            const float3 sigma_t = density() * readDensity(density_accessor, origin);
            const float3 sigma_s = albedo()  * readAlbedo(albedo_accessor, origin);

            const float3 tr = exp(-majorant * t) * invMajorantMax;

            if (Scattering && rng.nextFloat().x < sigma_t[channel] * invMajorantChannel) {
				// real particle
                beta   *= tr * sigma_t * sigma_s; // note: multiplication by localSigmaS is really part of BSDF computation
                dirPdf *= tr * sigma_t;
                scattered = true;
				return pnanovdb_grid_index_to_worldf(getDensityVolume(), gridHandle, origin);
			} else {
				// fake particle
				beta   *= tr * (majorant - sigma_t);
				dirPdf *= tr * (majorant - sigma_t);
                neePdf *= tr * majorant;
			}
		}

        return 0;
	}
};

#endif