#pragma once

#include <unordered_map>

#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_hash.hpp>

#include <Utils/common.hpp>

namespace tinyvkpt {

class Gui {
public:
	Gui(Swapchain& swapchain, vk::raii::Queue queue, const uint32_t queueFamily);
	~Gui();
	void newFrame();
	void render(CommandBuffer& commandBuffer, Image& backBuffer);
private:
	vk::raii::RenderPass mRenderPass;
	unordered_map<vk::Image, vk::raii::Framebuffer> mFramebuffers;
};

}