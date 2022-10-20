#include "../compat/scene.h"
#include "bsdf.hlsli"

#ifdef __HLSL__
#define PNANOVDB_HLSL
#include "nanovdb/PNanoVDB.h"
#endif

struct Medium : BSDF {
	Spectrum density_scale;
	float anisotropy;
	Spectrum albedo_scale;
	float attenuation_unit;
	uint density_volume_index;
	uint albedo_volume_index;

	SLANG_MUTATING
	void load(SceneParameters scene, uint address) {
		density_scale		 = scene.gMaterialData.Load<float3>(address); address += 12;
		anisotropy       	 = scene.gMaterialData.Load<float>(address); address += 4;
		albedo_scale         = scene.gMaterialData.Load<float3>(address); address += 12;
		attenuation_unit 	 = scene.gMaterialData.Load<float>(address); address += 4;
		density_volume_index = scene.gMaterialData.Load(address); address += 4;
		albedo_volume_index  = scene.gMaterialData.Load(address); address += 4;
	}

	Spectrum Le() { return 0; }
	Spectrum albedo() { return albedo_scale; }
	bool can_eval() { return any(density_scale > 0); }
	bool is_specular() { return abs(anisotropy) > 0.999; }

	Spectrum eval_approx(const Vector3 dir_in, const Vector3 dir_out, const bool adjoint = false) {
		return 1/(4*M_PI) * (1 - anisotropy * anisotropy) / pow(1 + anisotropy * anisotropy + 2 * anisotropy * dot(dir_in, dir_out), 1.5);
	}
	void eval(out MaterialEvalRecord r, const Vector3 dir_in, const Vector3 dir_out, const bool adjoint = false) {
		const Real v = 1/(4*M_PI) * (1 - anisotropy * anisotropy) / pow(1 + anisotropy * anisotropy + 2 * anisotropy * dot(dir_in, dir_out), 1.5);
		r.f = v;
		r.pdf_fwd = v;
		r.pdf_rev = v;
	}
	Spectrum sample(out MaterialSampleRecord r, const Vector3 rnd, const Vector3 dir_in, inout Spectrum beta, const bool adjoint = false) {
		if (abs(anisotropy) < 1e-3) {
			const Real z = 1 - 2 * rnd.x;
			const Real phi = 2 * M_PI * rnd.y;
			r.dir_out = Vector3(sqrt(max(0, 1 - z * z)) * float2(cos(phi), sin(phi)), z);
		} else {
			const Real tmp = (anisotropy * anisotropy - 1) / (2 * rnd.x * anisotropy - (anisotropy + 1));
			const Real cos_elevation = (tmp * tmp - (1 + anisotropy * anisotropy)) / (2 * anisotropy);
			const Real sin_elevation = sqrt(max(1 - cos_elevation * cos_elevation, 0));
			const Real azimuth = 2 * M_PI * rnd.y;
			float3 t, b;
			make_orthonormal(dir_in, t, b);
			r.dir_out = sin_elevation * cos(azimuth) * t + sin_elevation * sin(azimuth) * b + cos_elevation * dir_in;
		}
		const Real v = 1/(4*M_PI) * (1 - anisotropy * anisotropy) / pow(1 + anisotropy * anisotropy + 2 * anisotropy * dot(dir_in, r.dir_out), 1.5);
		r.pdf_fwd = v;
		r.pdf_rev = v;
		r.eta = -1;
		r.roughness = 1 - abs(anisotropy);
		return v;
	}

	Spectrum read_density(inout pnanovdb_readaccessor_t density_accessor, pnanovdb_address_t address) {
		return pnanovdb_read_float(gSceneParams.gVolumes[density_volume_index], address);
	}
	Spectrum read_density(inout pnanovdb_readaccessor_t density_accessor, const Vector3 pos_index) {
		return read_density(density_accessor, pnanovdb_readaccessor_get_value_address(PNANOVDB_GRID_TYPE_FLOAT, gSceneParams.gVolumes[density_volume_index], density_accessor, (int3)floor(pos_index)));
	}
	Spectrum read_albedo(inout pnanovdb_readaccessor_t density_accessor, pnanovdb_address_t address) {
		return pnanovdb_read_float(gSceneParams.gVolumes[albedo_volume_index], address);
	}
	Spectrum read_albedo(inout pnanovdb_readaccessor_t albedo_accessor, const Vector3 pos_index) {
		if (albedo_volume_index == -1)
			return 1;
		else
			return read_albedo(albedo_accessor, pnanovdb_readaccessor_get_value_address(PNANOVDB_GRID_TYPE_FLOAT, gSceneParams.gVolumes[albedo_volume_index], albedo_accessor, (int3)floor(pos_index)));
	}

	// returns hit position inside medium. multiplies beta by transmittance*sigma_s
	Vector3 delta_track(inout rng_state_t rng_state, Vector3 origin, Vector3 direction, float t_max, inout Spectrum beta, inout Spectrum dir_pdf, inout Spectrum nee_pdf, const bool can_scatter = true) {
		pnanovdb_readaccessor_t density_accessor, albedo_accessor;
		pnanovdb_grid_handle_t grid_handle = { 0 };
		pnanovdb_readaccessor_init(density_accessor, pnanovdb_tree_get_root(gSceneParams.gVolumes[density_volume_index], pnanovdb_grid_get_tree(gSceneParams.gVolumes[density_volume_index], grid_handle)));
		if (albedo_volume_index != -1)
			pnanovdb_readaccessor_init(albedo_accessor, pnanovdb_tree_get_root(gSceneParams.gVolumes[albedo_volume_index], pnanovdb_grid_get_tree(gSceneParams.gVolumes[albedo_volume_index], grid_handle)));

		const Spectrum majorant = density_scale * read_density(density_accessor, pnanovdb_root_get_max_address(PNANOVDB_GRID_TYPE_FLOAT, gSceneParams.gVolumes[density_volume_index], density_accessor.root));
		const uint channel = rng_next_uint(rng_state)%3;
		if (majorant[channel] < 1e-6) return POS_INFINITY;

		origin    = pnanovdb_grid_world_to_indexf    (gSceneParams.gVolumes[density_volume_index], grid_handle, origin);
		direction = pnanovdb_grid_world_to_index_dirf(gSceneParams.gVolumes[density_volume_index], grid_handle, direction);

		for (uint iteration = 0; iteration < gMaxNullCollisions && any(beta > 0); iteration++) {
			const float2 rnd = float2(rng_next_float(rng_state), rng_next_float(rng_state));
			const float t = attenuation_unit * -log(1 - rnd.x) / majorant[channel];
			if (t < t_max) {
				origin += direction*t;
				t_max -= t;

				const Spectrum local_density = density_scale * read_density(density_accessor, origin);
				const Spectrum local_albedo  = albedo_scale * read_albedo(albedo_accessor, origin);

				const Spectrum local_sigma_s = local_density * local_albedo;
				const Spectrum local_sigma_a = local_density * (1 - local_albedo);
				const Spectrum local_sigma_t = local_sigma_s + local_sigma_a;

				const Spectrum real_prob = local_sigma_t / majorant;
				const Spectrum tr = exp(-majorant * t) / max3(majorant);

				if (can_scatter && rnd.y < real_prob[channel]) {
					// real particle
					beta *= tr * local_sigma_s;
					dir_pdf *= tr * majorant * real_prob;
					return pnanovdb_grid_index_to_worldf(gSceneParams.gVolumes[density_volume_index], grid_handle, origin);
				} else {
					// fake particle
					beta *= tr * (majorant - local_sigma_t);
					dir_pdf *= tr * majorant * (1 - real_prob);
					nee_pdf *= tr * majorant;
					return POS_INFINITY;
				}
			} else {
				// transmitted without scattering
				const Spectrum tr = exp(-majorant * t_max);
				beta *= tr;
				nee_pdf *= tr;
				dir_pdf *= tr;
				break;
			}
		}

		return POS_INFINITY;
	}
};