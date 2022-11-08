#include "Swapchain.hpp"
#include "Window.hpp"
#include "Profiler.hpp"

#include <stdexcept>

namespace tinyvkpt {

Swapchain::Swapchain(Device& device, const string& name, Window& window, const uint32_t minImages, const vk::ImageUsageFlags backBufferUsage, const vk::SurfaceFormatKHR surfaceFormat, const vk::PresentModeKHR presentMode)
	: Device::Resource(device, name), mSwapchain(nullptr), mWindow(window), mMinImageCount(minImages), mUsage(backBufferUsage) {
	// select the format of the swapchain
	const auto formats = mDevice.physical().getSurfaceFormatsKHR(*mWindow.surface());
	mSurfaceFormat = formats.front();
	for (const vk::SurfaceFormatKHR& format : formats)
		if (format == surfaceFormat) {
			mSurfaceFormat = format;
			break;
		}

	mPresentMode = vk::PresentModeKHR::eFifo; // required to be supported
	for (const vk::PresentModeKHR& mode : mDevice.physical().getSurfacePresentModesKHR(*mWindow.surface()))
		if (mode == presentMode) {
			mPresentMode = mode;
			break;
		}

	create();
}

bool Swapchain::create() {
	ProfilerScope ps("Swapchain::create");

	// get the size of the swapchain
	const vk::SurfaceCapabilitiesKHR capabilities = mDevice.physical().getSurfaceCapabilitiesKHR(*mWindow.surface());
	mExtent = capabilities.currentExtent;
	if (mExtent.width == 0 || mExtent.height == 0 || mExtent.width > mDevice.limits().maxImageDimension2D || mExtent.height > mDevice.limits().maxImageDimension2D)
		return false;

 	vk::raii::SwapchainKHR oldSwapchain(move(mSwapchain));

	vk::SwapchainCreateInfoKHR info = {};
	info.surface = *mWindow.surface();
	info.oldSwapchain = *oldSwapchain;
	info.minImageCount = mMinImageCount;
	info.imageFormat = mSurfaceFormat.format;
	info.imageColorSpace = mSurfaceFormat.colorSpace;
	info.imageExtent = mExtent;
	info.imageArrayLayers = 1;
	info.imageUsage = mUsage;
	info.imageSharingMode = vk::SharingMode::eExclusive;
	info.preTransform = capabilities.currentTransform;
	info.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
	info.presentMode = mPresentMode;
	info.clipped = VK_FALSE;
	mSwapchain = vk::raii::SwapchainKHR(*mDevice, info);
	mDevice.setDebugName(*mSwapchain, resourceName());

	const vector<VkImage> images = mSwapchain.getImages();
	mImages.resize(images.size());
	mImageAvailableSemaphores.resize(images.size());
	for (uint32_t i = 0; i < mImages.size(); i++) {
		Image::Metadata m = {};
		m.mFormat = mSurfaceFormat.format;
		m.mExtent = vk::Extent3D(mExtent, 1);
		m.mUsage = info.imageUsage;
		m.mQueueFamilies = mWindow.queueFamilies(mDevice.physical());
		mImages[i] = make_shared<Image>(mDevice, "SwapchainImage " + to_string(i), images[i], m);
		mImageAvailableSemaphores[i] = make_shared<vk::raii::Semaphore>(*mDevice, vk::SemaphoreCreateInfo{});
	}

	mBackBufferIndex = 0;
	mImageAvailableSemaphoreIndex = 0;
	mDirty = false;
	return true;
}

bool Swapchain::acquireImage() {
	ProfilerScope ps("Window::acquire_image");

	const uint32_t semaphore_index = (mImageAvailableSemaphoreIndex + 1) % mImageAvailableSemaphores.size();

	vk::Result result;
	std::tie(result, mBackBufferIndex) = mSwapchain.acquireNextImage(mAcquireImageTimeout.count(), **mImageAvailableSemaphores[semaphore_index], {});

	if (result == vk::Result::eNotReady || result == vk::Result::eTimeout)
		return false;
	else if (result == vk::Result::eSuboptimalKHR || result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eErrorSurfaceLostKHR) {
		mDirty = true;
		return false;
	} else if (result != vk::Result::eSuccess)
		throw runtime_error("Failed to acquire next image");

	mImageAvailableSemaphoreIndex = semaphore_index;

	return true;
}

void Swapchain::present(const vk::raii::Queue queue, const vk::ArrayProxy<shared_ptr<vk::raii::Semaphore>>& waitSemaphores) {
	ProfilerScope ps("Window::present");

	vector<vk::Semaphore> semaphores(waitSemaphores.size());
	ranges::transform(waitSemaphores, semaphores.begin(), [](const auto s) { return **s; });

	const vk::Result result = queue.presentKHR(vk::PresentInfoKHR(semaphores, *mSwapchain, mBackBufferIndex));
	if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eErrorSurfaceLostKHR)
		mDirty = true;
	mPresentCount++;
}

}