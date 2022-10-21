#include <iostream>

#include <GLFW/glfw3.h>

#include "Core/Instance.hpp"
#include "Core/Window.hpp"
#include "Core/Swapchain.hpp"
#include "Core/CommandBuffer.hpp"
#include "Core/Profiler.hpp"
#include "Core/PipelineContext.hpp"

#include "App/Gui.hpp"

#include <imgui/imgui.h>

using namespace tinyvkpt;

void run(Instance& instance) {
	Window window(instance, "tinyvkpt", { 1600, 900 });
	auto[physicalDevice, presentQueueFamily] = window.findPhysicalDevice();
	Device device(instance, physicalDevice);
	vk::raii::Queue presentQueue(*device, presentQueueFamily, 0);
	Swapchain swapchain(device, "tinyvkpt", window);

	Gui gui(swapchain, presentQueue, presentQueueFamily, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR, false);

	//shared_ptr<ShaderSource> shaderSource = make_shared<ShaderSource>("kernels/test.hlsl", "main");
	//ComputePipelineContext pipelineContext(make_shared<ComputePipeline>("test", make_shared<Shader>(device, shaderSource), ComputePipeline::Metadata());

	while (window.isOpen()) {
		Profiler::beginFrame();

		device.checkCommandBuffers();

		if (swapchain.dirty())
			swapchain.create();

		gui.newFrame();

		if (swapchain.acquireImage()) {
			// imgui
			if (ImGui::Begin("Profiler"))
				Profiler::frameTimesGui();
			ImGui::End();
			if (Profiler::hasHistory()) {
				if (ImGui::Begin("Timeline"))
					Profiler::sampleTimelineGui();
				ImGui::End();
			}
			ImGui::ShowDemoWindow();


			// render
			shared_ptr<CommandBuffer> commandBufferPtr = device.getCommandBuffer(presentQueueFamily);
			CommandBuffer& commandBuffer = *commandBufferPtr;
			commandBuffer->begin(vk::CommandBufferBeginInfo());


			Image::Metadata imageInfo = {};
			imageInfo.mExtent = vk::Extent3D(1024,1024,1);
			imageInfo.mFormat = vk::Format::eR8G8B8A8Unorm;
			imageInfo.mQueueFamilies.emplace_back(presentQueueFamily);
			shared_ptr<Image> image = make_shared<Image>(device, "testImage", imageInfo);
			commandBuffer.trackResource(image);

			image->barrier(commandBuffer, vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1), vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite);
			commandBuffer->clearColorImage(**image, vk::ImageLayout::eTransferDstOptimal, vk::ClearColorValue(array<float,4>{1.f,1.f,0.f,1.f}), vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

			swapchain.backBuffer()->barrier(commandBuffer, vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1), vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite);
			image->barrier(commandBuffer, vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1), vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferRead);
			commandBuffer->blitImage(
				**image, vk::ImageLayout::eTransferSrcOptimal,
				**swapchain.backBuffer(), vk::ImageLayout::eTransferDstOptimal,
				vk::ImageBlit(
					vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1), { vk::Offset3D(0,0,0), vk::Offset3D(image->extent().width, image->extent().height, 1) },
					vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1), { vk::Offset3D(0,0,0), vk::Offset3D(swapchain.extent().width, swapchain.extent().height, 1) } ),
				vk::Filter::eLinear);

			gui.render(commandBuffer, Image::View(swapchain.backBuffer(), vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)), vk::ClearColorValue(array<float,4>{0.f,0.f,0.f,1.f}));


			commandBuffer->end();
			device.submit(presentQueue, commandBufferPtr);
			swapchain.present(presentQueue);
		}

		glfwPollEvents();
	}

	device->waitIdle();
	device.checkCommandBuffers();
}

int main(int argc, char** argv) {
	vector<string> args(argc);
	for (uint32_t i = 0; i < argc; i++)
		args[i] = argv[i];
	Instance instance(args);

	run(instance);

	glfwTerminate();

	return EXIT_SUCCESS;
}