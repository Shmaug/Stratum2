#include <Core/Instance.hpp>
#include <Core/Window.hpp>
#include <Core/Swapchain.hpp>
#include <Core/CommandBuffer.hpp>
#include <Core/Profiler.hpp>
#include <Core/Pipeline.hpp>

#include <App/Gui.hpp>
#include <App/Scene.hpp>

#include <GLFW/glfw3.h>

using namespace tinyvkpt;

struct App {
	unique_ptr<Instance> mInstance;
	unique_ptr<Window> mWindow;
	shared_ptr<Device> mDevice;
	uint32_t mPresentQueueFamily;
	vk::raii::Queue mPresentQueue;

	shared_ptr<Swapchain> mSwapchain;
	shared_ptr<Gui> mGui;

	NodePtr mRootNode;

	int mProfilerHistoryCount = 5;

	float3 mBias = float3::Zero();
	float mExposure = 0;

	inline App(const vector<string>& args) : mPresentQueue(nullptr) {
		mInstance     = make_unique<Instance>(args);
		mWindow       = make_unique<Window>(*mInstance, "tinyvkpt", vk::Extent2D{ 1600, 900 });

		auto[physicalDevice, presentQueueFamily] = mWindow->findPhysicalDevice();

		mDevice       = make_shared<Device>(*mInstance, physicalDevice);
		mPresentQueue = vk::raii::Queue(**mDevice, presentQueueFamily, 0);
		mSwapchain    = make_shared<Swapchain>(*mDevice, "tinyvkpt/Swapchain", *mWindow, 2, vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eColorAttachment);
		mGui          = make_shared<Gui>(*mSwapchain, mPresentQueue, presentQueueFamily, vk::ImageLayout::ePresentSrcKHR, true);

		mRootNode = Node::create("Root");
		mRootNode->addComponent(mDevice);
		mRootNode->addComponent(mSwapchain);
		mRootNode->addComponent(mGui);
	}
	inline ~App() {
		(*mDevice)->waitIdle();
		glfwTerminate();
	}

	inline void update() {
		ProfilerScope ps("App::update");
		mGui->newFrame();

		if (ImGui::Begin("vkpt")) {
			Profiler::frameTimesGui();

			ImGui::SliderInt("Count", &mProfilerHistoryCount, 1, 256);
			ImGui::PushID("Show timeline");
			if (ImGui::Button(Profiler::hasHistory() ? "Hide timeline" : "Show timeline"))
				Profiler::resetHistory(Profiler::hasHistory() ? 0 : mProfilerHistoryCount);
			ImGui::PopID();

			ImGui::DragFloat3("Bias", mBias.data());
			ImGui::DragFloat("Exposure", &mExposure);

			ImGui::Text("%u Back buffers", mSwapchain->backBufferCount());
		}
		ImGui::End();

		if (Profiler::hasHistory()) {
			if (ImGui::Begin("Timeline"))
				Profiler::sampleTimelineGui();
			ImGui::End();
		}
		ImGui::ShowDemoWindow();
	}

	// returns semaphore which signals when rendering completes
	inline shared_ptr<vk::raii::Semaphore> render() {
		ProfilerScope ps("App::render");
		Device& device = mSwapchain->mDevice;

		// build commandBuffer

		shared_ptr<CommandBuffer> commandBufferPtr = device.getCommandBuffer(mPresentQueueFamily);
		CommandBuffer& commandBuffer = *commandBufferPtr;
		commandBuffer->begin(vk::CommandBufferBeginInfo());

		Image::View renderTarget(mSwapchain->backBuffer(), vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
		mGui->render(commandBuffer, renderTarget, vk::ClearColorValue(array<float,4>{0.f,0.f,0.f,1.f}));

		commandBuffer->end();

		// submit commandBuffer

		pair<shared_ptr<vk::raii::Semaphore>, vk::PipelineStageFlags> waitSemaphore { mSwapchain->imageAvailableSemaphore(), vk::PipelineStageFlagBits::eComputeShader };
		shared_ptr<vk::raii::Semaphore> signalSemaphore = make_shared<vk::raii::Semaphore>(*device, vk::SemaphoreCreateInfo());
		commandBuffer.trackVulkanResource(signalSemaphore);
		device.submit(mPresentQueue, commandBufferPtr, waitSemaphore, signalSemaphore);
		return signalSemaphore;
	}

	inline void run() {
		// main loop
		while (mWindow->isOpen()) {
			Profiler::beginFrame();

			mDevice->updateFrame();

			bool swapchainValid = true;
			if (mSwapchain->dirty() || mWindow->extent() != mSwapchain->extent()) {
				(*mDevice)->waitIdle();
				swapchainValid = mSwapchain->create();
			}

			if (swapchainValid && mSwapchain->acquireImage()) {
				update();
				auto semaphore = render();
				mSwapchain->present(mPresentQueue, semaphore);
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
		App app(args);
		app.run();
	}

	return EXIT_SUCCESS;
}