#ifndef MEDIUM_H
#define MEDIUM_H

#include "common/rng.hlsli"
#include "bsdf.hlsli"

#include "compat/scene.h"

#define PNANOVDB_HLSL
#include "nanovdb/PNanoVDB.h"

struct Medium : BSDF {
	float3 mDensityScale;
	float mAnisotropy;
	float3 mAlbedoScale;
	float mAttenuationUnit;
	uint mDensityVolumeIndex;
    uint mAlbedoVolumeIndex;

	__init(uint address) {
		mDensityScale		= gScene.mMaterialData.Load<float3>((int)address); address += 12;
		mAnisotropy       	= gScene.mMaterialData.Load<float>((int)address); address += 4;
		mAlbedoScale        = gScene.mMaterialData.Load<float3>((int)address); address += 12;
		mAttenuationUnit 	= gScene.mMaterialData.Load<float>((int)address); address += 4;
		mDensityVolumeIndex = gScene.mMaterialData.Load((int)address); address += 4;
		mAlbedoVolumeIndex  = gScene.mMaterialData.Load((int)address); address += 4;
	}

    float3 emission() { return 0; }
    float emissionPdf() { return 0; }
	float3 albedo() { return mAlbedoScale; }
    bool canEvaluate() { return any(mDensityScale > 0); }
    bool isSingular() { return abs(mAnisotropy) > 0.999; }
    float continuationProb() { return saturate(luminance(mAlbedoScale * mDensityScale)); }

	MaterialEvalRecord evaluate<let Adjoint : bool>(const float3 dirIn, const float3 dirOut) {
		const float v = 1/(4*M_PI) * (1 - mAnisotropy * mAnisotropy) / pow(1 + mAnisotropy * mAnisotropy + 2 * mAnisotropy * dot(dirIn, dirOut), 1.5);
		MaterialEvalRecord r;
		r.mReflectance = v;
		r.mFwdPdfW = v;
		r.mRevPdfW = v;
		return r;
	}
	MaterialSampleRecord sample<let Adjoint : bool>(const float3 rnd, const float3 dirIn) {
		MaterialSampleRecord r;
		if (abs(mAnisotropy) < 1e-3) {
			const float z = 1 - 2 * rnd.x;
			const float phi = 2 * M_PI * rnd.y;
			r.mDirection = float3(sqrt(max(0, 1 - z * z)) * float2(cos(phi), sin(phi)), z);
		} else {
			const float tmp = (mAnisotropy * mAnisotropy - 1) / (2 * rnd.x * mAnisotropy - (mAnisotropy + 1));
			const float cos_elevation = (tmp * tmp - (1 + mAnisotropy * mAnisotropy)) / (2 * mAnisotropy);
			const float sin_elevation = sqrt(max(1 - cos_elevation * cos_elevation, 0));
            const float azimuth = 2 * M_PI * rnd.y;
            const float3x3 frame = makeOrthonormal(dirIn);
			r.mDirection = frame[0] * sin_elevation * cos(azimuth) + frame[1] * sin_elevation * sin(azimuth) + dirIn * cos_elevation;
		}
		const float v = 1/(4*M_PI) * (1 - mAnisotropy * mAnisotropy) / pow(1 + mAnisotropy * mAnisotropy + 2 * mAnisotropy * dot(dirIn, r.mDirection), 1.5);
		r.mFwdPdfW = v;
		r.mRevPdfW = v;
		r.mEta = -1;
		r.mRoughness = 1 - abs(mAnisotropy);
		return r;
	}

	float3 readDensity(inout pnanovdb_readaccessor_t density_accessor, pnanovdb_address_t address) {
		return pnanovdb_read_float(gScene.mVolumes[mDensityVolumeIndex], address);
	}
	float3 readDensity(inout pnanovdb_readaccessor_t density_accessor, const float3 pos_index) {
		return readDensity(density_accessor, pnanovdb_readaccessor_get_value_address(PNANOVDB_GRID_TYPE_FLOAT, gScene.mVolumes[mDensityVolumeIndex], density_accessor, (int3)floor(pos_index)));
	}
	float3 readAlbedo(inout pnanovdb_readaccessor_t density_accessor, pnanovdb_address_t address) {
		return pnanovdb_read_float(gScene.mVolumes[mAlbedoVolumeIndex], address);
	}
	float3 readAlbedo(inout pnanovdb_readaccessor_t albedo_accessor, const float3 pos_index) {
		if (mAlbedoVolumeIndex == -1)
			return 1;
		else
			return readAlbedo(albedo_accessor, pnanovdb_readaccessor_get_value_address(PNANOVDB_GRID_TYPE_FLOAT, gScene.mVolumes[mAlbedoVolumeIndex], albedo_accessor, (int3)floor(pos_index)));
	}

    StructuredBuffer<uint> getDensityVolume() { return gScene.mVolumes[mDensityVolumeIndex]; }
	StructuredBuffer<uint> getAlbedoVolume() { return gScene.mVolumes[mAlbedoVolumeIndex]; }

	// returns hit position inside medium. multiplies beta by transmittance*sigma_s
    float3 deltaTrack<let Scattering : bool>(inout RandomSampler rng, float3 origin, float3 direction, float tmax, inout float3 beta, out float3 dirPdf, out float3 neePdf, out bool scattered) {
		pnanovdb_grid_handle_t gridHandle = { 0 };
		scattered = false;

		pnanovdb_readaccessor_t density_accessor, albedo_accessor;
		pnanovdb_readaccessor_init(density_accessor, pnanovdb_tree_get_root(getDensityVolume(), pnanovdb_grid_get_tree(getDensityVolume(), gridHandle)));
        if (mAlbedoVolumeIndex != -1)
            pnanovdb_readaccessor_init(albedo_accessor, pnanovdb_tree_get_root(getAlbedoVolume(), pnanovdb_grid_get_tree(getAlbedoVolume(), gridHandle)));

		dirPdf = 1;
		neePdf = 1;

		const float3 majorant = mDensityScale * readDensity(density_accessor, pnanovdb_root_get_max_address(PNANOVDB_GRID_TYPE_FLOAT, getDensityVolume(), density_accessor.root));
        const uint channel = rng.next() % 3;
        if (majorant[channel] < 1e-6) return 0;

		origin    = pnanovdb_grid_world_to_indexf    (getDensityVolume(), gridHandle, origin);
		direction = pnanovdb_grid_world_to_index_dirf(getDensityVolume(), gridHandle, direction);

		for (uint iteration = 0; iteration < gMaxNullCollisions && any(beta > 0); iteration++) {
            const float t = mAttenuationUnit * -log(1 - rng.nextFloat()) / majorant[channel];

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

            const float3 localSigmaS = mAlbedoScale * readAlbedo(albedo_accessor, origin);
            const float3 localSigmaA = 1 - localSigmaS;
			const float3 localDensity = mDensityScale * readDensity(density_accessor, origin);
			const float3 localSigmaT = localDensity * (localSigmaS + localSigmaA);

			const float3 tr = exp(-majorant * t) / max3(majorant);
			const float3 realProb = localSigmaT / majorant;

            if (Scattering && rng.nextFloat() < realProb[channel]) {
				// real particle
                beta *= tr * localSigmaS; // note: multiplication by localSigmaS is really part of BSDF computation
                dirPdf *= tr * majorant * realProb;
                scattered = true;
				return pnanovdb_grid_index_to_worldf(getDensityVolume(), gridHandle, origin);
			} else {
				// fake particle
				beta *= tr * (majorant - localSigmaT);
				dirPdf *= tr * majorant * (1 - realProb);
                neePdf *= tr * majorant;
			}
		}

        return 0;
	}
};

#endif