#include <Core/Instance.hpp>
#include <Core/Window.hpp>
#include <Core/Swapchain.hpp>
#include <Core/CommandBuffer.hpp>
#include <Core/Profiler.hpp>
#include <Core/Pipeline.hpp>

#include <App/Gui.hpp>
#include <App/Scene.hpp>
#include <App/Inspector.hpp>
#include <App/BDPT.hpp>
#include <App/FlyCamera.hpp>

#include <GLFW/glfw3.h>

using namespace tinyvkpt;

struct App {
	shared_ptr<Node> mRootNode;

	shared_ptr<Instance> mInstance;
	shared_ptr<Window> mWindow;
	shared_ptr<Device> mDevice;
	uint32_t mPresentQueueFamily;
	vk::raii::Queue mPresentQueue;

	shared_ptr<Node> mSwapchainNode;
	shared_ptr<Swapchain> mSwapchain;
	shared_ptr<Gui> mGui;
	shared_ptr<Inspector> mInspector;
	shared_ptr<Scene> mScene;

	shared_ptr<BDPT> mRenderer;

	shared_ptr<FlyCamera> mFlyCamera;
	shared_ptr<Camera> mCamera;

	chrono::high_resolution_clock::time_point mLastUpdate;
	int mProfilerHistoryCount = 3;

	inline App(const vector<string>& args) : mPresentQueue(nullptr) {
		mRootNode = Node::create("Root");
		mInstance = mRootNode->makeComponent<Instance>(args);

		const shared_ptr<Node> deviceNode = mRootNode->addChild("Device");
		mWindow = deviceNode->makeComponent<Window>(*mInstance, "tinyvkpt", vk::Extent2D{ 1600, 900 });

		vk::raii::PhysicalDevice physicalDevice = nullptr;
		tie(physicalDevice, mPresentQueueFamily) = mWindow->findPhysicalDevice();

		mDevice       = deviceNode->makeComponent<Device>(*mInstance, physicalDevice);
		mPresentQueue = vk::raii::Queue(**mDevice, mPresentQueueFamily, 0);

		mSwapchainNode = deviceNode->addChild("Swapchain");
		mSwapchain     = mSwapchainNode->makeComponent<Swapchain>(*mDevice, "tinyvkpt/Swapchain", *mWindow, 2, vk::ImageUsageFlagBits::eTransferDst|vk::ImageUsageFlagBits::eColorAttachment);

		mGui = make_shared<Gui>(*mSwapchain, mPresentQueue, mPresentQueueFamily, vk::ImageLayout::ePresentSrcKHR, false);
		mInspector = mSwapchainNode->makeComponent<Inspector>(*mSwapchainNode);

		auto sceneNode = deviceNode->addChild("Scene");
		mScene = sceneNode->makeComponent<Scene>(*sceneNode);
		mRenderer = sceneNode->makeComponent<BDPT>(*sceneNode);

		auto cameraNode = sceneNode->addChild("Camera");
		cameraNode->makeComponent<TransformData>( float3::Zero(), quatf_identity(), float3::Ones() );
		mFlyCamera = cameraNode->makeComponent<FlyCamera>(*cameraNode);
		mCamera = cameraNode->makeComponent<Camera>();

		mLastUpdate = chrono::high_resolution_clock::now();
	}
	inline ~App() {
		(*mDevice)->waitIdle();
	}

	inline void drawGui() {
		ProfilerScope ps("App::drawGui");

		// profiler timings

		if (ImGui::Begin("vkpt")) {
			Profiler::frameTimesGui();

			ImGui::SliderInt("Count", &mProfilerHistoryCount, 1, 32);
			ImGui::PushID("Show timeline");
			if (ImGui::Button(Profiler::hasHistory() ? "Hide timeline" : "Show timeline"))
				Profiler::resetHistory(Profiler::hasHistory() ? 0 : mProfilerHistoryCount);
			ImGui::PopID();

			ImGui::Text("%u Back buffers", mSwapchain->imageCount());
		}
		ImGui::End();

		// frame timeline

		if (Profiler::hasHistory()) {
			if (ImGui::Begin("Timeline"))
				Profiler::sampleTimelineGui();
			ImGui::End();
		}
	}

	inline void update(CommandBuffer& commandBuffer) {
		auto now = chrono::high_resolution_clock::now();
		const float deltaTime = chrono::duration_cast<chrono::duration<float>>(now - mLastUpdate).count();
		mLastUpdate = now;

		mGui->newFrame();

		drawGui();
		mInspector->draw();

		mScene->update(commandBuffer, deltaTime);
		mRenderer->update(commandBuffer, deltaTime);
		mFlyCamera->update(deltaTime);
	}
	inline void render(CommandBuffer& commandBuffer, const Image::View& renderTarget) {
		// force camera projection aspect ratio to match renderTarget apsect ratio
		mCamera->mImageRect = vk::Rect2D{ { 0, 0 }, vk::Extent2D(renderTarget.extent().width, renderTarget.extent().height) };
		const float aspect = mCamera->mImageRect.extent.height / (float)mCamera->mImageRect.extent.width;
		if (abs(mCamera->mProjection.scale[0] / mCamera->mProjection.scale[1] - aspect) > 1e-5) {
			const float fovy = 2 * atan(1 / mCamera->mProjection.scale[1]);
			mCamera->mProjection = make_perspective(fovy, aspect, mCamera->mProjection.offset, mCamera->mProjection.near_plane);
		}

		mRenderer->render(commandBuffer, renderTarget);
		mGui->render(commandBuffer, renderTarget);
	}

	// returns semaphore which signals when commands/rendering completes
	inline void doFrame() {
		ProfilerScope ps("App::doFrame");

		shared_ptr<CommandBuffer> commandBufferPtr = mDevice->getCommandBuffer(mPresentQueueFamily);
		CommandBuffer& commandBuffer = *commandBufferPtr;
		commandBuffer->begin(vk::CommandBufferBeginInfo());

		update(commandBuffer);
		render(commandBuffer, Image::View(mSwapchain->image(), vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)));

		commandBuffer->end();

		pair<shared_ptr<vk::raii::Semaphore>, vk::PipelineStageFlags> waitSemaphore { mSwapchain->imageAvailableSemaphore(), vk::PipelineStageFlagBits::eComputeShader };
		shared_ptr<vk::raii::Semaphore> signalSemaphore = make_shared<vk::raii::Semaphore>(**mDevice, vk::SemaphoreCreateInfo());
		commandBuffer.trackVulkanResource(signalSemaphore);
		mDevice->submit(mPresentQueue, commandBufferPtr, waitSemaphore, signalSemaphore);

		// present

		mSwapchain->present(mPresentQueue, signalSemaphore);

		commandBuffer.trackVulkanResource(mSwapchain->imageAvailableSemaphore());
	}

	inline void run() {
		// main loop
		while (mWindow->isOpen()) {
			glfwPollEvents();

			Profiler::beginFrame();

			mDevice->updateFrame();

			// recreate swapchain if needed

			if (mSwapchain->isDirty()) {
				if (!mSwapchain->create())
					continue;

				// recreate swapchain-dependent resources
				(*mDevice)->waitIdle();
				mGui.reset();
				mGui = make_shared<Gui>(*mSwapchain, mPresentQueue, mPresentQueueFamily, vk::ImageLayout::ePresentSrcKHR, true);
			}

			// do frame

			if (mSwapchain->acquireImage())
				doFrame();
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