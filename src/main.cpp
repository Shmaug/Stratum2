#include <Core/Instance.hpp>
#include <Core/Window.hpp>
#include <Core/Swapchain.hpp>
#include <Core/CommandBuffer.hpp>
#include <Core/Profiler.hpp>
#include <Core/Pipeline.hpp>

#include <App/Gui.hpp>
#include <App/Scene.hpp>
#include <App/Inspector.hpp>
#include <App/FlyCamera.hpp>

#include <GLFW/glfw3.h>

#include <App/PathTracer.hpp>
#include <App/Denoiser.hpp>
#include <App/Tonemapper.hpp>
#define RendererType PathTracer

namespace stm2 {

struct App {
	shared_ptr<Node> mRootNode;

	shared_ptr<Instance> mInstance;
	shared_ptr<Window> mWindow;
	shared_ptr<Device> mDevice;
	uint32_t mPresentQueueFamily;
	vk::raii::Queue mPresentQueue;

	vector<shared_ptr<CommandBuffer>> mCommandBuffers;

	shared_ptr<Node> mSwapchainNode;
	shared_ptr<Swapchain> mSwapchain;
	shared_ptr<Gui> mGui;
	shared_ptr<Inspector> mInspector;
	shared_ptr<Scene> mScene;

	shared_ptr<FlyCamera> mFlyCamera;
	shared_ptr<Camera> mCamera;

	shared_ptr<RendererType> mRenderer;

	chrono::high_resolution_clock::time_point mLastUpdate;
	int mProfilerHistoryCount = 3;

	inline App(const vector<string>& args) : mPresentQueue(nullptr) {
		mRootNode = Node::create("Root");
		mInstance = mRootNode->makeComponent<Instance>(args);

		const shared_ptr<Node> deviceNode = mRootNode->addChild("Device");
		mWindow = deviceNode->makeComponent<Window>(*mInstance, "Stratum2", vk::Extent2D{ 1600, 900 });

		vk::raii::PhysicalDevice physicalDevice = nullptr;
		tie(physicalDevice, mPresentQueueFamily) = mWindow->findPhysicalDevice();

		mDevice       = deviceNode->makeComponent<Device>(*mInstance, physicalDevice);
		mPresentQueue = vk::raii::Queue(**mDevice, mPresentQueueFamily, 0);

		mSwapchainNode = deviceNode->addChild("Swapchain");
		mSwapchain     = mSwapchainNode->makeComponent<Swapchain>(*mDevice, "Swapchain", *mWindow, 2);

		mCommandBuffers.resize(mSwapchain->imageCount());
		for (auto& cb : mCommandBuffers) {
			cb = make_shared<CommandBuffer>(*mDevice, "CommandBuffer", mPresentQueueFamily);
			mDevice->incrementFrameIndex();
		}
		mDevice->updateLastFrameDone(0);

		mGui = make_shared<Gui>(*mSwapchain, mPresentQueue, mPresentQueueFamily, vk::ImageLayout::ePresentSrcKHR, false);
		mInspector = mSwapchainNode->makeComponent<Inspector>(*mSwapchainNode);

		auto sceneNode = deviceNode->addChild("Scene");
		mInspector->select(sceneNode);
		mScene = sceneNode->makeComponent<Scene>(*sceneNode);
		mRenderer = sceneNode->makeComponent<RendererType>(*sceneNode);
		sceneNode->makeComponent<Denoiser>(*sceneNode);
		sceneNode->makeComponent<Tonemapper>(*sceneNode);

		auto cameraNode = sceneNode->addChild("Camera");
		cameraNode->makeComponent<TransformData>( float3(0,1.5f,0), quatf::identity(), float3::Ones() );
		mFlyCamera = cameraNode->makeComponent<FlyCamera>(*cameraNode);
		mCamera = cameraNode->makeComponent<Camera>(ProjectionData::makePerspective(radians(70.f), mSwapchain->extent().width / mSwapchain->extent().height, float2::Zero(), .001f));

		mLastUpdate = chrono::high_resolution_clock::now();
	}
	inline ~App() {
		(*mDevice)->waitIdle();
	}

	inline void drawGui() {
		ProfilerScope ps("App::drawGui");

		// profiler timings

		if (ImGui::Begin("Profiler")) {
			Profiler::frameTimesGui();

			ImGui::SliderInt("Count", &mProfilerHistoryCount, 1, 32);
			ImGui::PushID("Show timeline");
			if (ImGui::Button(Profiler::hasHistory() ? "Hide timeline" : "Show timeline"))
				Profiler::resetHistory(Profiler::hasHistory() ? 0 : mProfilerHistoryCount);
			ImGui::PopID();
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
		mFlyCamera->update(deltaTime);
	}
	inline void render(CommandBuffer& commandBuffer, const Image::View& renderTarget) {
		// force camera projection aspect ratio to match renderTarget apsect ratio
		mCamera->mImageRect = vk::Rect2D{ { 0, 0 }, vk::Extent2D(renderTarget.extent().width, renderTarget.extent().height) };
		const float aspect = mCamera->mImageRect.extent.height / (float)mCamera->mImageRect.extent.width;
		if (abs(mCamera->mProjection.mScale[0] / mCamera->mProjection.mScale[1] - aspect) > 1e-5) {
			const float fovy = 2 * atan(1 / mCamera->mProjection.mScale[1]);
			mCamera->mProjection = ProjectionData::makePerspective(fovy, aspect, mCamera->mProjection.mOffset, mCamera->mProjection.mNearPlane);
		}

		mRenderer->render(commandBuffer, renderTarget);
		mGui->render(commandBuffer, renderTarget);
	}

	// returns semaphore which signals when commands/rendering completes
	inline void doFrame() {
		ProfilerScope ps("App::doFrame");

		mDevice->incrementFrameIndex();

		shared_ptr<CommandBuffer> commandBufferPtr = mCommandBuffers[mDevice->frameIndex() % mSwapchain->imageCount()];
		CommandBuffer& commandBuffer = *commandBufferPtr;

		if (commandBuffer.fence()) {
			if ((*mDevice)->waitForFences(**commandBuffer.fence(), true, ~0ull) != vk::Result::eSuccess)
				throw runtime_error("Error: waitForFences failed");
		}
		mDevice->updateLastFrameDone(mDevice->frameIndex() - mCommandBuffers.size());

		// Record commands

		commandBuffer.reset();
		commandBuffer->begin(vk::CommandBufferBeginInfo());

		update(commandBuffer);
		render(commandBuffer, Image::View(mSwapchain->image(), vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)));

		commandBuffer->end();

		// submit commands

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

			// recreate swapchain if needed

			if (mSwapchain->isDirty()) {
				(*mDevice)->waitIdle();
				if (!mSwapchain->create())
					continue;

				// recreate swapchain-dependent resources
				mCommandBuffers.resize(mSwapchain->imageCount());
				for (auto& cb : mCommandBuffers)
					cb = make_shared<CommandBuffer>(*mDevice, "CommandBuffer", mPresentQueueFamily);

				mGui.reset();
				mGui = make_shared<Gui>(*mSwapchain, mPresentQueue, mPresentQueueFamily, vk::ImageLayout::ePresentSrcKHR, false);
			}

			// do frame

			if (mSwapchain->acquireImage())
				doFrame();
		}
	}
};

}

int main(int argc, char** argv) {
	using namespace std;

	vector<string> args(argc);
	ranges::copy_n(argv, argc, args.begin());

	{
		stm2::App app(args);
		app.run();
	}

	return EXIT_SUCCESS;
}