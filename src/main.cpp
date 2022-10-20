#include <iostream>

#include <GLFW/glfw3.h>

#include "Core/Instance.hpp"
#include "Core/Window.hpp"
#include "Core/Swapchain.hpp"
#include "Core/CommandBuffer.hpp"
#include "Core/Profiler.hpp"

#include "App/Gui.hpp"

#include <imgui/imgui.h>

using namespace tinyvkpt;

void run(Instance& instance) {
	Window window(instance, "tinyvkpt", { 1600, 900 });
	auto[physicalDevice, presentQueueFamily] = window.findPhysicalDevice();
	Device device(instance, physicalDevice);
	Swapchain swapchain(device, "tinyvkpt", window);

	vk::raii::Queue presentQueue(*device, presentQueueFamily, 0);

	Gui gui(swapchain, presentQueue, presentQueueFamily);

	while (window.isOpen()) {
		Profiler::beginFrame();

		device.checkCommandBuffers();

		if (swapchain.dirty())
			swapchain.create();

		gui.newFrame();

		if (swapchain.acquireImage()) {
			if (ImGui::Begin("vkpt")) {
				Profiler::frameTimesGui();
			}
			ImGui::End();

			if (Profiler::hasHistory()) {
				if (ImGui::Begin("Timeline"))
					Profiler::sampleTimelineGui();
				ImGui::End();
			}

			ImGui::ShowDemoWindow();

			shared_ptr<CommandBuffer> commandBufferPtr = device.getCommandBuffer(presentQueueFamily);
			CommandBuffer& commandBuffer = *commandBufferPtr;
			commandBuffer->begin(vk::CommandBufferBeginInfo());

			gui.render(commandBuffer, *swapchain.backBuffer());

			commandBuffer->end();
			device.submit(presentQueue, commandBufferPtr);

			swapchain.present(presentQueue);
		}

		glfwPollEvents();
	}

	device->waitIdle();
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