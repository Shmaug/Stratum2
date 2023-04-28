#include "Gui.hpp"
#include <Core/Instance.hpp>
#include <Core/Window.hpp>
#include <Core/Swapchain.hpp>
#include <Core/CommandBuffer.hpp>

#include <imgui/imgui_impl_vulkan.h>
#include <imgui/imgui_impl_glfw.h>
#include <ImGuizmo.h>

namespace stm2 {

unordered_map<Image::View, pair<vk::raii::DescriptorSet, vk::raii::Sampler>> Gui::gTextureIDs;
unordered_set<Image::View> Gui::gFrameTextures;
shared_ptr<vk::raii::DescriptorPool> Gui::gImGuiDescriptorPool;


ImTextureID Gui::getTextureID(const Image::View& image) {
	auto it = gTextureIDs.find(image);
	if (it == gTextureIDs.end()) {
		vk::raii::Sampler sampler(*image.image()->mDevice, vk::SamplerCreateInfo({}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear));
		vk::raii::DescriptorSet descriptorSet(
			*image.image()->mDevice,
			ImGui_ImplVulkan_AddTexture(*sampler, *image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
			**gImGuiDescriptorPool);

		it = gTextureIDs.emplace(image, pair{ move(descriptorSet), move(sampler) }).first;
	}
	gFrameTextures.emplace(image);
	return (VkDescriptorSet)*it->second.first;
}

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
	swapchain.mDevice.setDebugName(*mRenderPass, "Gui::mRenderPass");


	ImGui::CreateContext();

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();
	ImGui::GetStyle().WindowRounding = 4.0f;
	ImGui::GetStyle().GrabRounding = 4.0f;
	ImGui::GetStyle().IndentSpacing *= 0.75f;

	ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
	ImGui::GetIO().ConfigWindowsMoveFromTitleBarOnly = true;

	gImGuiDescriptorPool = swapchain.mDevice.getDescriptorPool();

	ImGui_ImplGlfw_InitForVulkan(swapchain.mWindow.window(), true);

	ImGui_ImplVulkan_InitInfo init_info = {};
	init_info.Instance = **swapchain.mDevice.mInstance;
	init_info.PhysicalDevice = *swapchain.mDevice.physical();
	init_info.Device = **swapchain.mDevice;
	init_info.QueueFamily = mQueueFamily;
	init_info.Queue = *queue;
	init_info.PipelineCache  = *swapchain.mDevice.pipelineCache();
	init_info.DescriptorPool = **gImGuiDescriptorPool;
	init_info.Subpass = 0;
	init_info.MinImageCount = max(swapchain.minImageCount(), 2u);
	init_info.ImageCount    = max(swapchain.imageCount(), 2u);
	init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
	init_info.Allocator = nullptr;
	//init_info.CheckVkResultFn = check_vk_result;
	ImGui_ImplVulkan_Init(&init_info, *mRenderPass);

	float scale = 1;
	if (auto arg = swapchain.mDevice.mInstance.findArgument("guiScale"); arg) {
		scale = stof(*arg);
		ImGui::GetStyle().ScaleAllSizes(scale);
		ImGui::GetStyle().IndentSpacing /= scale;
	}

	// Upload Fonts

	for (auto fontstr : swapchain.mDevice.mInstance.findArguments("font")) {
		const size_t delim = fontstr.find(',');
		if (delim == string::npos)
			ImGui::GetIO().Fonts->AddFontFromFileTTF(fontstr.c_str(), scale*16.f);
		else {
			string font = fontstr.substr(0, delim);
			ImGui::GetIO().Fonts->AddFontFromFileTTF(font.c_str(), scale*(float)atof(fontstr.c_str() + delim + 1));
		}
	}

	shared_ptr<CommandBuffer> commandBufferPtr = make_shared<CommandBuffer>(swapchain.mDevice, "ImGui CreateFontsTexture", swapchain.mDevice.findQueueFamily());
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
	gFrameTextures.clear();
	gTextureIDs.clear();
	ImGui_ImplVulkan_Shutdown();
	gImGuiDescriptorPool.reset();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

void Gui::newFrame() {
	ImGui_ImplVulkan_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
	ImGuizmo::BeginFrame();
}

void Gui::render(CommandBuffer& commandBuffer, const Image::View& renderTarget, const vk::ClearValue& clearValue) {
	ImGui::Render();
	ImDrawData* drawData = ImGui::GetDrawData();
	if (drawData->DisplaySize.x <= 0.0f || drawData->DisplaySize.y <= 0.0f) return;

	const vk::Extent2D extent((uint32_t)drawData->DisplaySize.x, (uint32_t)drawData->DisplaySize.y);

	// create framebuffer

	auto it = mFramebuffers.find(**renderTarget.image());
	if (it == mFramebuffers.end()) {
		vk::raii::Framebuffer fb(*commandBuffer.mDevice, vk::FramebufferCreateInfo({}, *mRenderPass, *renderTarget, extent.width, extent.height, 1));
		it = mFramebuffers.emplace(**renderTarget.image(), move(fb)).first;
	}

	renderTarget.barrier(commandBuffer, vk::ImageLayout::eColorAttachmentOptimal, vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::AccessFlagBits::eColorAttachmentWrite);

	for (const Image::View& v : gFrameTextures)
		v.barrier(commandBuffer, vk::ImageLayout::eShaderReadOnlyOptimal, vk::PipelineStageFlagBits::eFragmentShader, vk::AccessFlagBits::eShaderRead);
	gFrameTextures.clear();

	// render gui

	commandBuffer->beginRenderPass(
		vk::RenderPassBeginInfo(*mRenderPass, *it->second, vk::Rect2D({0,0}, extent), clearValue),
		vk::SubpassContents::eInline);

	// Record dear imgui primitives into command buffer
	ImGui_ImplVulkan_RenderDrawData(drawData, **commandBuffer);

	// Submit command buffer
	commandBuffer->endRenderPass();
	renderTarget.updateState(mDstLayout, vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::AccessFlagBits::eColorAttachmentRead);
}

}