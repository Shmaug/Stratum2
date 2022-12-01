#pragma once

#include <Core/Image.hpp>

#include <imgui/imgui.h>
#include <imgui/imgui_impl_vulkan.h>

namespace stm2 {

class Gui {
public:
	inline static void scalarField(const char* label, const vk::Format format, void* data) {
		switch (format) {
		default:
			cerr << "Unsupported scalar format " << to_string(format) << endl;
			break;
		case vk::Format::eR8Uint:
			ImGui::InputScalarN(label, ImGuiDataType_U8, data, 1);
			break;
		case vk::Format::eR8G8Uint:
			ImGui::InputScalarN(label, ImGuiDataType_U8, data, 2);
			break;
		case vk::Format::eR8G8B8Uint:
			ImGui::InputScalarN(label, ImGuiDataType_U8, data, 3);
			break;
		case vk::Format::eR8G8B8A8Uint:
			ImGui::InputScalarN(label, ImGuiDataType_U8, data, 4);
			break;

		case vk::Format::eR8Sint:
			ImGui::InputScalarN(label, ImGuiDataType_S8, data, 1);
			break;
		case vk::Format::eR8G8Sint:
			ImGui::InputScalarN(label, ImGuiDataType_S8, data, 2);
			break;
		case vk::Format::eR8G8B8Sint:
			ImGui::InputScalarN(label, ImGuiDataType_S8, data, 3);
			break;
		case vk::Format::eR8G8B8A8Sint:
			ImGui::InputScalarN(label, ImGuiDataType_S8, data, 4);
			break;

		case vk::Format::eR16Uint:
			ImGui::InputScalarN(label, ImGuiDataType_U16, data, 1);
			break;
		case vk::Format::eR16G16Uint:
			ImGui::InputScalarN(label, ImGuiDataType_U16, data, 2);
			break;
		case vk::Format::eR16G16B16Uint:
			ImGui::InputScalarN(label, ImGuiDataType_U16, data, 3);
			break;
		case vk::Format::eR16G16B16A16Uint:
			ImGui::InputScalarN(label, ImGuiDataType_U16, data, 4);
			break;

		case vk::Format::eR16Sint:
			ImGui::InputScalarN(label, ImGuiDataType_S16, data, 1);
			break;
		case vk::Format::eR16G16Sint:
			ImGui::InputScalarN(label, ImGuiDataType_S16, data, 2);
			break;
		case vk::Format::eR16G16B16Sint:
			ImGui::InputScalarN(label, ImGuiDataType_S16, data, 3);
			break;
		case vk::Format::eR16G16B16A16Sint:
			ImGui::InputScalarN(label, ImGuiDataType_S16, data, 4);
			break;

		case vk::Format::eR32Uint:
			ImGui::InputScalarN(label, ImGuiDataType_U32, data, 1);
			break;
		case vk::Format::eR32G32Uint:
			ImGui::InputScalarN(label, ImGuiDataType_U32, data, 2);
			break;
		case vk::Format::eR32G32B32Uint:
			ImGui::InputScalarN(label, ImGuiDataType_U32, data, 3);
			break;
		case vk::Format::eR32G32B32A32Uint:
			ImGui::InputScalarN(label, ImGuiDataType_U32, data, 4);
			break;

		case vk::Format::eR32Sint:
			ImGui::InputScalarN(label, ImGuiDataType_S32, data, 1);
			break;
		case vk::Format::eR32G32Sint:
			ImGui::InputScalarN(label, ImGuiDataType_S32, data, 2);
			break;
		case vk::Format::eR32G32B32Sint:
			ImGui::InputScalarN(label, ImGuiDataType_S32, data, 3);
			break;
		case vk::Format::eR32G32B32A32Sint:
			ImGui::InputScalarN(label, ImGuiDataType_S32, data, 4);
			break;

		case vk::Format::eR32Sfloat:
			ImGui::InputScalarN(label, ImGuiDataType_Float, data, 1);
			break;
		case vk::Format::eR32G32Sfloat:
			ImGui::InputScalarN(label, ImGuiDataType_Float, data, 2);
			break;
		case vk::Format::eR32G32B32Sfloat:
			ImGui::InputScalarN(label, ImGuiDataType_Float, data, 3);
			break;
		case vk::Format::eR32G32B32A32Sfloat:
			ImGui::InputScalarN(label, ImGuiDataType_Float, data, 4);
			break;

		case vk::Format::eR64Sfloat:
			ImGui::InputScalarN(label, ImGuiDataType_Double, data, 1);
			break;
		case vk::Format::eR64G64Sfloat:
			ImGui::InputScalarN(label, ImGuiDataType_Double, data, 2);
			break;
		case vk::Format::eR64G64B64Sfloat:
			ImGui::InputScalarN(label, ImGuiDataType_Double, data, 3);
			break;
		case vk::Format::eR64G64B64A64Sfloat:
			ImGui::InputScalarN(label, ImGuiDataType_Double, data, 4);
			break;
		}
	}

	template<typename T, typename Ty>
	inline static bool enumDropdown(const char* label, Ty& selected, const uint32_t count) {
		bool ret = false;
		if (ImGui::BeginCombo(label, to_string((T)selected).c_str())) {
			for (uint32_t i = 0; i < count; i++)
				if (ImGui::Selectable(to_string((T)i).c_str(), (uint32_t)selected == i)) {
					selected = (Ty)i;
					ret = true;
				}
			ImGui::EndCombo();
		}
		return ret;
	}


	static unordered_map<Image::View, pair<vk::raii::DescriptorSet, vk::raii::Sampler>> gTextureIDs;
	static unordered_set<Image::View> gFrameTextures;
	static ImTextureID getTextureID(const Image::View& image) {
		auto it = gTextureIDs.find(image);
		if (it == gTextureIDs.end()) {
			vk::raii::Sampler sampler(*image.image()->mDevice, vk::SamplerCreateInfo({}, vk::Filter::eNearest, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear));
			vk::raii::DescriptorSet descriptorSet(
				*image.image()->mDevice,
				ImGui_ImplVulkan_AddTexture(*sampler, *image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
				*image.image()->mDevice.descriptorPool());

			it = gTextureIDs.emplace(image, pair{ move(descriptorSet), move(sampler) }).first;
		}
		gFrameTextures.emplace(image);
		return (VkDescriptorSet)*it->second.first;

	}

	Gui(Swapchain& swapchain, vk::raii::Queue queue, const uint32_t queueFamily, const vk::ImageLayout dstLayout, const bool clear);
	~Gui();

	Gui() = default;
	Gui(Gui&&) = default;
	Gui& operator=(Gui&&) = default;

	Gui(const Gui&) = delete;
	Gui& operator=(const Gui&) = delete;

	void newFrame();

	// converts renderTarget to ColorAttachmentOptimal before rendering
	void render(CommandBuffer& commandBuffer, const Image::View& renderTarget, const vk::ClearValue& clearValue = {});

private:
	vk::raii::RenderPass mRenderPass;
	uint32_t mQueueFamily;
	vk::ImageLayout mDstLayout;
	unordered_map<vk::Image, vk::raii::Framebuffer> mFramebuffers;
};

}