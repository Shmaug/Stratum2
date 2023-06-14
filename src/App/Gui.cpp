#include "Gui.hpp"
#include <Core/Instance.hpp>
#include <Core/Window.hpp>
#include <Core/Swapchain.hpp>
#include <Core/CommandBuffer.hpp>

#include <imgui/imgui_impl_vulkan.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_internal.h>
#include <ImGuizmo.h>

namespace stm2 {

unordered_map<Image::View, pair<vk::raii::DescriptorSet, vk::raii::Sampler>> Gui::gTextureIDs;
unordered_set<Image::View> Gui::gFrameTextures;
shared_ptr<vk::raii::DescriptorPool> Gui::gImGuiDescriptorPool;
ImFont* Gui::gHeaderFont;


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

void Gui::progressSpinner(const char* label, const float radius, const float thickness, bool center) {
	ImGuiWindow* window = ImGui::GetCurrentWindow();
	ImDrawList* drawList = window->DrawList;
	const ImGuiStyle& style = ImGui::GetStyle();

	ImVec2 pos = window->DC.CursorPos;
	if (center)
    	pos.x += (ImGui::GetContentRegionAvail().x - 2*radius) * .5f;

	const ImRect bb(pos, ImVec2(pos.x + radius*2, pos.y + (radius + style.FramePadding.y)*2));
	ImGui::ItemSize(bb, style.FramePadding.y);
	if (!ImGui::ItemAdd(bb, window->GetID(label)))
		return;

	const float t = ImGui::GetCurrentContext()->Time;

	const int num_segments = drawList->_CalcCircleAutoSegmentCount(radius);

	const int start = abs(sin(t * 1.8f))*(num_segments-5);
	const float a_min = float(M_PI*2) * ((float)start) / (float)num_segments;
	const float a_max = float(M_PI*2) * ((float)num_segments-3) / (float)num_segments;

	const ImVec2 c = ImVec2(pos.x + radius, pos.y + radius + style.FramePadding.y);

	drawList->PathClear();

	for (int i = 0; i < num_segments; i++) {
		const float a = a_min + ((float)i / (float)num_segments) * (a_max - a_min);
		drawList->PathLineTo(ImVec2(
			c.x + cos(a + t*8) * radius,
			c.y + sin(a + t*8) * radius));
	}

	drawList->PathStroke(ImGui::GetColorU32(ImGuiCol_Text), 0, thickness);
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
	ImGui::GetStyle().GrabRounding   = 4.0f;
	ImGui::GetStyle().IndentSpacing *= 0.75f;

	{
		auto& colors = ImGui::GetStyle().Colors;
		colors[ImGuiCol_WindowBg] = ImVec4{ 0.1f, 0.1f, 0.1f, 0.9f };
		colors[ImGuiCol_DockingEmptyBg] = colors[ImGuiCol_WindowBg];

		colors[ImGuiCol_Header] = colors[ImGuiCol_WindowBg];
		colors[ImGuiCol_HeaderActive]  = ImVec4{ 0.15f, 0.15f, 0.15f, 1.0f };
		colors[ImGuiCol_HeaderHovered] = ImVec4{ 0.20f, 0.20f, 0.20f, 1.0f };

		colors[ImGuiCol_TitleBg]          = ImVec4{ 0.15f, 0.15f, 0.15f, 1.0f };
		colors[ImGuiCol_TitleBgActive]    = ImVec4{ 0.2f, 0.2f, 0.2f, 1.0f };
		colors[ImGuiCol_TitleBgCollapsed] = colors[ImGuiCol_TitleBg];

		colors[ImGuiCol_Tab]                = colors[ImGuiCol_TitleBgActive];
		colors[ImGuiCol_TabHovered]         = ImVec4{ 0.45f, 0.45f, 0.45f, 1.0f };
		colors[ImGuiCol_TabActive]          = ImVec4{ 0.35f, 0.35f, 0.35f, 1.0f };
		colors[ImGuiCol_TabUnfocused]       = colors[ImGuiCol_TitleBg];
		colors[ImGuiCol_TabUnfocusedActive] = colors[ImGuiCol_TitleBg];

		colors[ImGuiCol_FrameBg]            = ImVec4{ 0.15f, 0.15f, 0.15f, 1.0f };
		colors[ImGuiCol_FrameBgHovered]     = ImVec4{ 0.19f, 0.19f, 0.19f, 1.0f };
		colors[ImGuiCol_FrameBgActive]      = ImVec4{ 0.18f, 0.18f, 0.18f, 1.0f };

		colors[ImGuiCol_Button]             = ImVec4{ 0.2f, 0.2f, 0.2f, 1.0f };
		colors[ImGuiCol_ButtonHovered]      = ImVec4{ 0.25f, 0.25f, 0.25f, 1.0f };
		colors[ImGuiCol_ButtonActive]       = ImVec4{ 0.175f, 0.175f, 0.175f, 1.0f };
		colors[ImGuiCol_CheckMark]          = ImVec4{ 0.75f, 0.75f, 0.75f, 1.0f };
		colors[ImGuiCol_SliderGrab]         = ImVec4{ 0.75f, 0.75f, 0.75f, 1.0f };
		colors[ImGuiCol_SliderGrabActive]   = ImVec4{ 0.8f, 0.8f, 0.8f, 1.0f };

		colors[ImGuiCol_ResizeGrip]        = colors[ImGuiCol_ButtonActive];
		colors[ImGuiCol_ResizeGripActive]  = colors[ImGuiCol_ButtonActive];
		colors[ImGuiCol_ResizeGripHovered] = colors[ImGuiCol_ButtonActive];

		colors[ImGuiCol_DragDropTarget]    = colors[ImGuiCol_ButtonActive];
	}

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
	if (auto arg = swapchain.mDevice.mInstance.findArgument("guiScale")) {
		scale = stof(*arg);
		ImGui::GetStyle().ScaleAllSizes(scale);
		ImGui::GetStyle().IndentSpacing /= scale;
	}

	// Upload Fonts

	if (auto arg = swapchain.mDevice.mInstance.findArgument("font")) {
		ImGui::GetIO().Fonts->AddFontFromFileTTF(arg->c_str(), scale*16.f);
		gHeaderFont = ImGui::GetIO().Fonts->AddFontFromFileTTF(arg->c_str(), scale*20.f);
	} else
		gHeaderFont = ImGui::GetFont();

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