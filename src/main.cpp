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


// TODO: push constant reflection is wrong

struct App {
	Swapchain& mSwapchain;
	uint32_t mPresentQueueFamily;
	vk::raii::Queue mPresentQueue;
	Gui mGui;

	int mProfilerHistoryCount = 3;

	float3 mBias = float3::Zero();
	float mExposure = 0;

	ComputePipelineContext computePass1, computePass2;

	inline App(Swapchain& swapchain, const uint32_t presentQueueFamily) :
		mSwapchain(swapchain),
		mPresentQueueFamily(presentQueueFamily),
		mPresentQueue(*swapchain.mDevice, presentQueueFamily, 0),
		mGui(mSwapchain, mPresentQueue, presentQueueFamily, vk::ImageLayout::ePresentSrcKHR, false) {

		computePass1 = ComputePipelineContext(make_shared<ComputePipeline>("testPass1", make_shared<Shader>(swapchain.mDevice, ShaderSource("../../src/Shaders/kernels/test.hlsl", "pass1")), ComputePipeline::Metadata()));
		computePass2 = ComputePipelineContext(make_shared<ComputePipeline>("testPass2", make_shared<Shader>(swapchain.mDevice, ShaderSource("../../src/Shaders/kernels/test.hlsl", "pass2")), ComputePipeline::Metadata()));
	}

	void update() {
		ProfilerScope ps("App::update");
		mGui.newFrame();

		if (ImGui::Begin("vkpt")) {
			Profiler::frameTimesGui();

			ImGui::SliderInt("Count", &mProfilerHistoryCount, 1, 256);
			ImGui::PushID("Show timeline");
			if (ImGui::Button(Profiler::hasHistory() ? "Hide timeline" : "Show timeline"))
				Profiler::resetHistory(Profiler::hasHistory() ? 0 : mProfilerHistoryCount);
			ImGui::PopID();

			ImGui::DragFloat3("Bias", mBias.data());
			ImGui::DragFloat("Exposure", &mExposure);

			ImGui::Text("%u Back buffers", mSwapchain.backBufferCount());
			ImGui::Text("%u Pipeline resources", computePass1.resourceCount() + computePass2.resourceCount());
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
		ProfilerScope ps("App::render");
		Device& device = mSwapchain.mDevice;

		shared_ptr<CommandBuffer> commandBufferPtr = device.getCommandBuffer(mPresentQueueFamily);
		CommandBuffer& commandBuffer = *commandBufferPtr;
		commandBuffer->begin(vk::CommandBufferBeginInfo());

		Image::View renderTarget(mSwapchain.backBuffer(), vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

		auto descriptors = computePass1.getDescriptorSets(commandBuffer.mDevice, Descriptors{
			{ { "gParams.mInput" , 0 }, DescriptorValue{ ImageDescriptor{ renderTarget, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite, {} } } },
			{ { "gParams.mOutput", 0 }, DescriptorValue{ ImageDescriptor{ renderTarget, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite, {} } } }
		});

		computePass1(commandBuffer, computePass1.dispatchDimensions(renderTarget.extent()), descriptors);
		computePass2(commandBuffer, computePass2.dispatchDimensions(renderTarget.extent()), descriptors, {},
			PushConstants{
				{ "mBias"    , PushConstantValue(mBias) },
				{ "mExposure", PushConstantValue(mExposure) }
			} );

		mGui.render(commandBuffer, renderTarget, vk::ClearColorValue(array<float,4>{0.f,0.f,0.f,1.f}));

		commandBuffer->end();

		pair<shared_ptr<vk::raii::Semaphore>, vk::PipelineStageFlags> waitSemaphore { mSwapchain.imageAvailableSemaphore(), vk::PipelineStageFlagBits::eComputeShader };
		shared_ptr<vk::raii::Semaphore> signalSemaphore = make_shared<vk::raii::Semaphore>(*device, vk::SemaphoreCreateInfo());
		commandBuffer.trackVulkanResource(signalSemaphore);
		device.submit(mPresentQueue, commandBufferPtr, waitSemaphore, signalSemaphore);
		return signalSemaphore;
	}

	void run() {
		while (mSwapchain.mWindow.isOpen()) {
			Profiler::beginFrame();

			mSwapchain.mDevice.updateFrame();

			bool valid = true;
			if (mSwapchain.dirty()) {
				mSwapchain.mDevice->waitIdle();
				valid = mSwapchain.create();
			}

			if (valid && mSwapchain.acquireImage()) {
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
		Swapchain swapchain(device, "tinyvkpt/Swapchain", window, 2, vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eColorAttachment);
		App app(swapchain, presentQueueFamily);
		app.run();

		device->waitIdle();
		device.updateFrame();
	}
	glfwTerminate();

	return EXIT_SUCCESS;
}