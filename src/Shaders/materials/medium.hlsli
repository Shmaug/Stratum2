#ifndef MEDIUM_H
#define MEDIUM_H

#include "bsdf.hlsli"

#include "compat/scene.h"

#ifdef __HLSL__
#define PNANOVDB_HLSL
#include "nanovdb/PNanoVDB.h"
#endif

struct Medium : BSDF {
	float3 mDensityScale;
	float mAnisotropy;
	float3 mAlbedoScale;
	float mAttenuationUnit;
	uint mDensityVolumeIndex;
	uint mAlbedoVolumeIndex;

	__init(uint address) {
		mDensityScale		= gScene.mMaterialData.Load<float3>(address); address += 12;
		mAnisotropy       	= gScene.mMaterialData.Load<float>(address); address += 4;
		mAlbedoScale        = gScene.mMaterialData.Load<float3>(address); address += 12;
		mAttenuationUnit 	= gScene.mMaterialData.Load<float>(address); address += 4;
		mDensityVolumeIndex = gScene.mMaterialData.Load(address); address += 4;
		mAlbedoVolumeIndex  = gScene.mMaterialData.Load(address); address += 4;
	}

	float3 emission() { return 0; }
	float3 albedo() { return mAlbedoScale; }
	bool canEvaluate() { return any(mDensityScale > 0); }
	bool isSingular() { return abs(mAnisotropy) > 0.999; }

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
			r.mDirection = mul(makeOrthonormal(dirIn), float3(sin_elevation * cos(azimuth), + sin_elevation * sin(azimuth), cos_elevation));
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

	// returns hit position inside medium. multiplies beta by transmittance*sigma_s
	float3 deltaTrack(inout RandomSampler rng, float3 origin, float3 direction, float tmax, inout float3 beta, out float3 dirPdf, out float3 neePdf, const bool scatter = true) {
		pnanovdb_readaccessor_t density_accessor, albedo_accessor;
		pnanovdb_grid_handle_t grid_handle = { 0 };
		pnanovdb_readaccessor_init(density_accessor, pnanovdb_tree_get_root(gScene.mVolumes[mDensityVolumeIndex], pnanovdb_grid_get_tree(gScene.mVolumes[mDensityVolumeIndex], grid_handle)));
		if (mAlbedoVolumeIndex != -1)
			pnanovdb_readaccessor_init(albedo_accessor, pnanovdb_tree_get_root(gScene.mVolumes[mAlbedoVolumeIndex], pnanovdb_grid_get_tree(gScene.mVolumes[mAlbedoVolumeIndex], grid_handle)));

		dirPdf = 1;
		neePdf = 1;

		const float3 majorant = mDensityScale * readDensity(density_accessor, pnanovdb_root_get_max_address(PNANOVDB_GRID_TYPE_FLOAT, gScene.mVolumes[mDensityVolumeIndex], density_accessor.root));
		const uint channel = rng.next() % 3;
		if (majorant[channel] < 1e-6) return POS_INFINITY;

		origin    = pnanovdb_grid_world_to_indexf    (gScene.mVolumes[mDensityVolumeIndex], grid_handle, origin);
		direction = pnanovdb_grid_world_to_index_dirf(gScene.mVolumes[mDensityVolumeIndex], grid_handle, direction);

		for (uint iteration = 0; iteration < gMaxNullCollisions && any(beta > 0); iteration++) {
			const float2 rnd = float2(rng.nextFloat(), rng.nextFloat());
			const float t = mAttenuationUnit * -log(1 - rnd.x) / majorant[channel];
			if (t < tmax) {
				origin += direction*t;
				tmax -= t;

				const float3 local_density = mDensityScale * readDensity(density_accessor, origin);
				const float3 local_albedo  = mAlbedoScale * readAlbedo(albedo_accessor, origin);

				const float3 local_sigma_s = local_density * local_albedo;
				const float3 local_sigma_a = local_density * (1 - local_albedo);
				const float3 local_sigma_t = local_sigma_s + local_sigma_a;

				const float3 real_prob = local_sigma_t / majorant;
				const float3 tr = exp(-majorant * t) / max3(majorant);

				if (scatter && rnd.y < real_prob[channel]) {
					// real particle
					beta *= tr * local_sigma_s;
					dirPdf *= tr * majorant * real_prob;
					return pnanovdb_grid_index_to_worldf(gScene.mVolumes[mDensityVolumeIndex], grid_handle, origin);
				} else {
					// fake particle
					beta *= tr * (majorant - local_sigma_t);
					dirPdf *= tr * majorant * (1 - real_prob);
					neePdf *= tr * majorant;
					return POS_INFINITY;
				}
			} else {
				// transmitted without scattering
				const float3 tr = exp(-majorant * tmax);
				beta *= tr;
				neePdf *= tr;
				dirPdf *= tr;
				break;
			}
		}

		return POS_INFINITY;
	}
};

#endif