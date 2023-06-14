#pragma once

#include <Core/Image.hpp>
#include <imgui/imgui.h>

namespace stm2 {

class Gui {
private:
	static shared_ptr<vk::raii::DescriptorPool> gImGuiDescriptorPool;

public:
	template<typename T>
	inline static ImGuiDataType getImGuiDataType() {
		if constexpr(is_floating_point_v<T>)
			return sizeof(T) == sizeof(float) ? ImGuiDataType_Float : ImGuiDataType_Double;
		else if constexpr (sizeof(T) == sizeof(uint64_t))
			return is_signed_v<T> ? ImGuiDataType_S64 : ImGuiDataType_U64;
		else if constexpr (sizeof(T) == sizeof(uint32_t))
			return is_signed_v<T> ? ImGuiDataType_S32 : ImGuiDataType_U32;
		else if constexpr (sizeof(T) == sizeof(uint16_t))
			return is_signed_v<T> ? ImGuiDataType_S16 : ImGuiDataType_U16;
		else
			return ImGuiDataType_COUNT;
	}

	template<typename T>
	inline static bool scalarField(const string& label, T* ptr, const T& mn = 0, const T& mx = 0, const float dragSpeed = 1) {
		if (dragSpeed == 0 && mn != mx) {
			ImGui::SetNextItemWidth(75);
			return ImGui::SliderScalar(label.c_str(), getImGuiDataType<T>(), ptr, &mn, &mx);
		} else {
			ImGui::SetNextItemWidth(50);
			return ImGui::DragScalar(label.c_str(), getImGuiDataType<T>(), ptr, dragSpeed, &mn, &mx);
		}
	}

	inline static bool scalarField(const char* label, const vk::Format format, void* data) {
		static unordered_map<vk::Format, pair<ImGuiDataType, int>> sFormatMap = {
			{ vk::Format::eR8Uint,             { ImGuiDataType_U8, 1 } },
			{ vk::Format::eR8G8Uint,           { ImGuiDataType_U8, 2 } },
			{ vk::Format::eR8G8B8Uint,         { ImGuiDataType_U8, 3 } },
			{ vk::Format::eR8G8B8A8Uint,       { ImGuiDataType_U8, 4 } },

			{ vk::Format::eR8Sint,             { ImGuiDataType_S8, 1 } },
			{ vk::Format::eR8G8Sint,           { ImGuiDataType_S8, 2 } },
			{ vk::Format::eR8G8B8Sint,         { ImGuiDataType_S8, 3 } },
			{ vk::Format::eR8G8B8A8Sint,       { ImGuiDataType_S8, 4 } },

			{ vk::Format::eR16Uint,            { ImGuiDataType_U16, 1 } },
			{ vk::Format::eR16G16Uint,         { ImGuiDataType_U16, 2 } },
			{ vk::Format::eR16G16B16Uint,      { ImGuiDataType_U16, 3 } },
			{ vk::Format::eR16G16B16A16Uint,   { ImGuiDataType_U16, 4 } },

			{ vk::Format::eR16Sint,            { ImGuiDataType_S16, 1 } },
			{ vk::Format::eR16G16Sint,         { ImGuiDataType_S16, 2 } },
			{ vk::Format::eR16G16B16Sint,      { ImGuiDataType_S16, 3 } },
			{ vk::Format::eR16G16B16A16Sint,   { ImGuiDataType_S16, 4 } },

			{ vk::Format::eR32Uint,            { ImGuiDataType_U32, 1 } },
			{ vk::Format::eR32G32Uint,         { ImGuiDataType_U32, 2 } },
			{ vk::Format::eR32G32B32Uint,      { ImGuiDataType_U32, 3 } },
			{ vk::Format::eR32G32B32A32Uint,   { ImGuiDataType_U32, 4 } },

			{ vk::Format::eR32Sint,            { ImGuiDataType_S32, 1 } },
			{ vk::Format::eR32G32Sint,         { ImGuiDataType_S32, 2 } },
			{ vk::Format::eR32G32B32Sint,      { ImGuiDataType_S32, 3 } },
			{ vk::Format::eR32G32B32A32Sint,   { ImGuiDataType_S32, 4 } },

			{ vk::Format::eR32Sfloat,          { ImGuiDataType_Float, 1 } },
			{ vk::Format::eR32G32Sfloat,       { ImGuiDataType_Float, 2 } },
			{ vk::Format::eR32G32B32Sfloat,    { ImGuiDataType_Float, 3 } },
			{ vk::Format::eR32G32B32A32Sfloat, { ImGuiDataType_Float, 4 } },

			{ vk::Format::eR64Sfloat,          { ImGuiDataType_Double, 1 } },
			{ vk::Format::eR64G64Sfloat,       { ImGuiDataType_Double, 2 } },
			{ vk::Format::eR64G64B64Sfloat,    { ImGuiDataType_Double, 3 } },
			{ vk::Format::eR64G64B64A64Sfloat, { ImGuiDataType_Double, 4 } }
		};
		const auto&[dataType, components] = sFormatMap.at(format);
		ImGui::SetNextItemWidth(50);
		return ImGui::InputScalarN(label, dataType, data, components);

	}

	template<typename EnumType, typename T>
	inline static bool enumDropdown(const char* label, T& selected, const uint32_t count) {
		bool ret = false;
		const string previewstr = to_string((EnumType)selected);
		if (ImGui::BeginCombo(label, previewstr.c_str())) {
			for (uint32_t i = 0; i < count; i++) {
				const string stri = to_string((EnumType)i);
				if (ImGui::Selectable(stri.c_str(), (uint32_t)selected == i)) {
					selected = (T)i;
					ret = true;
				}
			}
			ImGui::EndCombo();
		}
		return ret;
	}

	static void progressSpinner(const char* label, const float radius = 15, const float thickness = 6, const bool center = true);

	static ImFont* gHeaderFont;
	static unordered_map<Image::View, pair<vk::raii::DescriptorSet, vk::raii::Sampler>> gTextureIDs;
	static unordered_set<Image::View> gFrameTextures;
	static ImTextureID getTextureID(const Image::View& image);

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