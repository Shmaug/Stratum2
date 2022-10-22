#pragma once

#include <Core/Image.hpp>

namespace tinyvkpt {

class Gui {
public:
	Gui(Swapchain& swapchain, vk::raii::Queue queue, const uint32_t queueFamily, const vk::ImageLayout dstLayout, const bool clear);
	~Gui();

	Gui() = default;
	Gui(Gui&&) = default;
	Gui& operator=(Gui&&) = default;

	Gui(const Gui&) = delete;
	Gui& operator=(const Gui&) = delete;

	void newFrame();
	void render(CommandBuffer& commandBuffer, const Image::View& backBuffer, const vk::ClearValue& clearValue);
private:
	vk::raii::RenderPass mRenderPass;
	uint32_t mQueueFamily;
	vk::ImageLayout mDstLayout;
	unordered_map<vk::Image, vk::raii::Framebuffer> mFramebuffers;
};

}