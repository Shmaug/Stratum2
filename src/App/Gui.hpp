#pragma once

#include <Core/Image.hpp>

namespace tinyvkpt {

class Gui {
public:
	Gui(Swapchain& swapchain, vk::raii::Queue queue, const uint32_t queueFamily, const vk::ImageLayout srcLayout, const vk::ImageLayout dstLayout, const bool clear);
	~Gui();
	void newFrame();
	void render(CommandBuffer& commandBuffer, const Image::View& backBuffer, const vk::ClearValue& clearValue);
private:
	vk::raii::RenderPass mRenderPass;
	uint32_t mQueueFamily;
	vk::ImageLayout mDstLayout;
	unordered_map<vk::Image, vk::raii::Framebuffer> mFramebuffers;
};

}