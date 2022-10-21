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

struct App {
	Swapchain& mSwapchain;
	uint32_t mPresentQueueFamily;
	vk::raii::Queue mPresentQueue;
	Gui& mGui;

	int mProfilerHistoryCount = 3;

	ResourcePool<Image> mImages;

	void update() {
		if (ImGui::Begin("Profiler")) {
			Profiler::frameTimesGui();

			ImGui::SliderInt("Count", &mProfilerHistoryCount, 1, 256);
			ImGui::PushID("Show timeline");
			if (ImGui::Button(Profiler::hasHistory() ? "Hide timeline" : "Show timeline"))
				Profiler::resetHistory(Profiler::hasHistory() ? 0 : mProfilerHistoryCount);
			ImGui::PopID();
		}
		ImGui::End();

		if (Profiler::hasHistory()) {
			if (ImGui::Begin("Timeline"))
				Profiler::sampleTimelineGui();
			ImGui::End();
		}
		ImGui::ShowDemoWindow();
	}

	shared_ptr<vk::raii::Semaphore> render() {
		Device& device = mSwapchain.mDevice;

		Image::Metadata imageInfo = {};
		imageInfo.mExtent = vk::Extent3D(1024,1024,1);
		imageInfo.mFormat = vk::Format::eR8G8B8A8Unorm;
		shared_ptr<Image> image = mImages.get();
		if (!image) {
			image = make_shared<Image>(device, "testImage", imageInfo);
			mImages.emplace(image);
		}


		shared_ptr<CommandBuffer> commandBufferPtr = device.getCommandBuffer(mPresentQueueFamily);
		CommandBuffer& commandBuffer = *commandBufferPtr;
		commandBuffer->begin(vk::CommandBufferBeginInfo());


		commandBuffer.trackResource(image);
		image->barrier(commandBuffer, vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1), vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite);
		commandBuffer->clearColorImage(**image, vk::ImageLayout::eTransferDstOptimal, vk::ClearColorValue(array<float,4>{1.f,1.f,0.f,1.f}), vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

		mSwapchain.backBuffer()->barrier(commandBuffer, vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1), vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite);
		image->barrier(commandBuffer, vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1), vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferRead);
		commandBuffer->blitImage(
			**image, vk::ImageLayout::eTransferSrcOptimal,
			**mSwapchain.backBuffer(), vk::ImageLayout::eTransferDstOptimal,
			vk::ImageBlit(
				vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1), { vk::Offset3D(0,0,0), vk::Offset3D(image->extent().width, image->extent().height, 1) },
				vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1), { vk::Offset3D(0,0,0), vk::Offset3D(mSwapchain.backBuffer()->extent().width, mSwapchain.backBuffer()->extent().height, 1) } ),
			vk::Filter::eLinear);


		mGui.render(commandBuffer, Image::View(mSwapchain.backBuffer(), vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)), vk::ClearColorValue(array<float,4>{0.f,0.f,0.f,1.f}));

		commandBuffer->end();

		shared_ptr<vk::raii::Semaphore> semaphore = make_shared<vk::raii::Semaphore>(*device, vk::SemaphoreCreateInfo());
		commandBuffer.trackVulkanResource(semaphore);

		device.submit(mPresentQueue, commandBufferPtr, {}, semaphore);

		return semaphore;
	}

	void run() {
		shared_ptr<ShaderSource> shaderSource = make_shared<ShaderSource>("../../src/Shaders/kernels/test.hlsl", "main");
		//ComputePipelineContext pipelineContext(make_shared<ComputePipeline>("test", make_shared<Shader>(mDevice, shaderSource), ComputePipeline::Metadata());

		while (mSwapchain.mWindow.isOpen()) {
			Profiler::beginFrame();

			mSwapchain.mDevice.checkCommandBuffers();

			if (mSwapchain.dirty())
				mSwapchain.create();

			mGui.newFrame();

			if (mSwapchain.acquireImage()) {
				update();
				auto semaphore = render();
				mSwapchain.present(mPresentQueue, semaphore);
			}

			glfwPollEvents();
		}
	}
};

int main(int argc, char** argv) {
	vector<string> args(argc);
	for (uint32_t i = 0; i < argc; i++)
		args[i] = argv[i];
	{
		Instance instance(args);
		Window window(instance, "tinyvkpt", { 1600, 900 });
		auto[physicalDevice, presentQueueFamily] = window.findPhysicalDevice();
		Device device(instance, physicalDevice);
		Swapchain swapchain(device, "tinyvkpt", window);

		vk::raii::Queue presentQueue(*device, presentQueueFamily, 0);
		Gui gui(swapchain, presentQueue, presentQueueFamily, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR, false);

		App app(swapchain, presentQueueFamily, presentQueue, gui);
		app.run();

		device->waitIdle();
		device.checkCommandBuffers();
	}
	glfwTerminate();

	return EXIT_SUCCESS;
}