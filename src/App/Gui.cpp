#include "Gui.hpp"
#include <Core/Instance.hpp>
#include <Core/Window.hpp>
#include <Core/Swapchain.hpp>
#include <Core/CommandBuffer.hpp>

#include <imgui/imgui_impl_vulkan.h>
#include <imgui/imgui_impl_glfw.h>

namespace tinyvkpt {

Gui::Gui(Swapchain& swapchain, vk::raii::Queue queue, const uint32_t queueFamily, const vk::ImageLayout dstLayout, const bool clear)
	: mRenderPass(nullptr), mQueueFamily(queueFamily), mDstLayout(dstLayout) {

	// create renderpass
	vk::AttachmentReference attachmentReference(0, vk::ImageLayout::eColorAttachmentOptimal);
	vk::SubpassDescription subpass({}, vk::PipelineBindPoint::eGraphics, {}, attachmentReference, {}, {}, {});
	vk::AttachmentDescription attachment({},
		swapchain.format().format,
		vk::SampleCountFlagBits::e1,
		clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
		vk::AttachmentStoreOp::eStore,
		vk::AttachmentLoadOp::eDontCare,
		vk::AttachmentStoreOp::eDontCare,
		vk::ImageLayout::eColorAttachmentOptimal,
		dstLayout );
	mRenderPass = vk::raii::RenderPass(*swapchain.mDevice, vk::RenderPassCreateInfo({}, attachment, subpass, {}));

	ImGui::CreateContext();

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();

	// Setup Platform/Renderer backends
	ImGui_ImplGlfw_InitForVulkan(swapchain.mWindow.window(), true);

	ImGui_ImplVulkan_InitInfo init_info = {};
	init_info.Instance = **swapchain.mDevice.mInstance;
	init_info.PhysicalDevice = *swapchain.mDevice.physical();
	init_info.Device = **swapchain.mDevice;
	init_info.QueueFamily = mQueueFamily;
	init_info.Queue = *queue;
	init_info.PipelineCache  = *swapchain.mDevice.pipelineCache();
	init_info.DescriptorPool = *swapchain.mDevice.descriptorPool();
	init_info.Subpass = 0;
	init_info.MinImageCount = swapchain.backBufferCount();
	init_info.ImageCount    = swapchain.backBufferCount();
	init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
	init_info.Allocator = nullptr;
	//init_info.CheckVkResultFn = check_vk_result;
	ImGui_ImplVulkan_Init(&init_info, *mRenderPass);

	// Upload Fonts

	shared_ptr<CommandBuffer> commandBufferPtr = swapchain.mDevice.getCommandBuffer(0);
	CommandBuffer& commandBuffer = *commandBufferPtr;
	commandBuffer->begin(vk::CommandBufferBeginInfo());

	ImGui_ImplVulkan_CreateFontsTexture(**commandBuffer);

	commandBuffer->end();
	swapchain.mDevice.submit(vk::raii::Queue(*swapchain.mDevice, 0, 0), commandBufferPtr);
	if (swapchain.mDevice->waitForFences(**commandBuffer.fence(), true, ~0ull) != vk::Result::eSuccess)
		throw runtime_error("Error: waitForFences failed");

	ImGui_ImplVulkan_DestroyFontUploadObjects();
}
Gui::~Gui() {
	if (*mRenderPass) {
		ImGui_ImplVulkan_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
	}
}

void Gui::newFrame() {
	ImGui_ImplVulkan_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
}

void Gui::render(CommandBuffer& commandBuffer, const Image::View& backBuffer, const vk::ClearValue& clearValue) {
	ImGui::Render();
	ImDrawData* drawData = ImGui::GetDrawData();
	if (drawData->DisplaySize.x <= 0.0f || drawData->DisplaySize.y <= 0.0f) return;

	const vk::Extent2D extent((uint32_t)drawData->DisplaySize.x, (uint32_t)drawData->DisplaySize.y);

	// create framebuffer

	auto it = mFramebuffers.find(**backBuffer.image());
	if (it == mFramebuffers.end()) {
		vk::raii::Framebuffer fb(*commandBuffer.mDevice, vk::FramebufferCreateInfo({}, *mRenderPass, *backBuffer, extent.width, extent.height, 1));
		it = mFramebuffers.emplace(**backBuffer.image(), move(fb)).first;
	}

	backBuffer.barrier(commandBuffer, vk::ImageLayout::eColorAttachmentOptimal, vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::AccessFlagBits::eColorAttachmentWrite);

	// render gui

	commandBuffer->beginRenderPass(
		vk::RenderPassBeginInfo(*mRenderPass, *it->second, vk::Rect2D({0,0}, extent), clearValue),
		vk::SubpassContents::eInline);

	// Record dear imgui primitives into command buffer
	ImGui_ImplVulkan_RenderDrawData(drawData, **commandBuffer);

	// Submit command buffer
	commandBuffer->endRenderPass();
	backBuffer.updateState(mDstLayout, vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::AccessFlagBits::eColorAttachmentRead);
}

}