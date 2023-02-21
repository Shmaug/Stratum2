#include "DeviceResourcePool.hpp"
#include <imgui/imgui.h>

namespace stm2 {

void DeviceResourcePool::drawGui() {
	ImGui::PushID(this);

	for (const auto&[name, sets] : mDescriptorSets) {
		if (ImGui::CollapsingHeader(name.c_str())) {
			for (const auto& descriptorSet : sets) {
				ImGui::Text("%llu descriptors (%u frames ago)",
					descriptorSet->descriptors().size(),
					descriptorSet->mDevice.frameIndex() - descriptorSet->lastFrameUsed());
			}
		}
	}

	for (const auto&[name, images] : mImages) {
		if (ImGui::CollapsingHeader(name.c_str())) {
			for (const auto& image : images) {
				ImGui::Text("%ux%ux%u (%u frames ago)",
					image.extent().width, image.extent().height, image.extent().depth,
					image.image()->mDevice.frameIndex() - image.image()->lastFrameUsed());
				ImGui::Indent();
				ImGui::Text("%s", to_string(image.image()->format()).c_str());
				ImGui::Unindent();
			}
		}
	}

	for (const auto&[name, buffers] : mBuffers) {
		if (ImGui::CollapsingHeader(name.c_str())) {
			for (const auto& buffer : buffers) {
				const auto[size, sizeUnit] = formatBytes(buffer.sizeBytes());
				ImGui::Text("%llu %s %s (%u frames ago)",
					size, sizeUnit,
					(buffer.buffer()->memoryUsage() & vk::MemoryPropertyFlagBits::eDeviceLocal) ? "Device" : "Host",
					buffer.buffer()->mDevice.frameIndex() - buffer.buffer()->lastFrameUsed());
			}
		}
	}

	ImGui::PopID();
}

}