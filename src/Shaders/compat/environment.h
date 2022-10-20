#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include "scene.h"
#include "image_value.h"

#ifdef __cplusplus
namespace tinyvkpt {
#endif

struct Environment {
	ImageValue3 emission;

#ifdef __cplusplus
	inline void store(ByteAppendBuffer& bytes, MaterialResources& resources) const {
		emission.store(bytes, resources);
	}
	inline void inspectorGui() {
		image_value_field("Emission", emission);
	}
#endif
#ifdef __HLSL__
	SLANG_MUTATING
	inline void load(uint address) {
		emission.load(address);
	}

	inline float3 eval(const float3 dir_out) {
		if (!emission.has_image()) return emission.value;
		uint w, h;
		emission.image().GetDimensions(w, h);
		const float2 uv = cartesian_to_spherical_uv(dir_out);
		return sample_image(emission.image(), uv, 0).rgb * emission.value;
	}

	inline float3 sample(const float2 rnd, out float3 dir_out, out float pdf) {
		if (!emission.has_image()) {
			const float2 uv = sample_uniform_sphere(rnd.x, rnd.y);
			dir_out = spherical_uv_to_cartesian(uv);
			pdf = uniform_sphere_pdfW();
			return emission.value;
		} else {
			uint w, h;
			emission.image().GetDimensions(w, h);
			const float2 uv = sample_texel(emission.image(), rnd, pdf);
			dir_out = spherical_uv_to_cartesian(uv);
			pdf /= (2 * M_PI * M_PI * sqrt(1 - dir_out.y*dir_out.y));
			return emission.value*emission.image().SampleLevel(gSceneParams.gStaticSampler, uv, 0).rgb;
		}
	}

	inline float eval_pdf(const float3 dir_out) {
		if (!emission.has_image())
			return uniform_sphere_pdfW();
		else {
			const float2 uv = cartesian_to_spherical_uv(dir_out);
			const float pdf = sample_texel_pdf(emission.image(), uv);
			return pdf / (2 * M_PI * M_PI * sqrt(1 - dir_out.y*dir_out.y));
		}
	}
#endif
};

#ifdef __cplusplus

inline Environment loadEnvironment(CommandBuffer& commandBuffer, const filesystem::path& filename) {
	// TODO
	//ImageData image = loadImageData(commandBuffer.mDevice, filename, false);
	shared_ptr<Image> image; //= make_shared<Image>(commandBuffer, filename.stem().string(), filename);
	Environment e;
	e.emission = ImageValue3(float3::Ones(), Image::View(image, vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, image->levels(), 0, image->layers())));
	return e;
}

} // namespace vkpt
#endif

#endif