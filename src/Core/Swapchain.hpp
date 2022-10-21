#pragma once

#include "Image.hpp"

namespace tinyvkpt {

class Swapchain : public Device::Resource {
public:
	Window& mWindow;

	Swapchain(Device& device, const string& name, Window& window,
		const uint32_t minImages = 2,
		const vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst,
		const vk::SurfaceFormatKHR preferredSurfaceFormat = vk::SurfaceFormatKHR(vk::Format::eR8G8B8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear),
		const vk::PresentModeKHR presentMode = vk::PresentModeKHR::eImmediate);

	DECLARE_DEREFERENCE_OPERATORS(vk::raii::SwapchainKHR, mSwapchain)

	inline vk::Extent2D extent() const { return mExtent; }
	inline vk::SurfaceFormatKHR format() const { return mSurfaceFormat; }
	inline vk::PresentModeKHR presentMode() const { return mPresentMode; }
	inline const shared_ptr<vk::raii::Semaphore>& imageAvailableSemaphore() const { return mImageAvailableSemaphores[mImageAvailableSemaphoreIndex]; }

	inline void acquireImageTimeout(const chrono::nanoseconds& v) { mAcquireImageTimeout = v; }
	inline const chrono::nanoseconds& acquireImageTimeout() const { return mAcquireImageTimeout; }

	inline const uint32_t& minImageCount() const { return mMinImageCount; }

	inline uint32_t backBufferCount() const { return (uint32_t)mImages.size(); }
	inline uint32_t backBufferIndex() const { return mBackBufferIndex; }
	inline const shared_ptr<Image>& backBuffer() const { return mImages[backBufferIndex()]; }
	inline const shared_ptr<Image>& backBuffer(uint32_t i) const { return mImages[i]; }
	inline bool dirty() const { return mDirty; }

	void create();

	bool acquireImage();
	// Waits on waitSemaphores
	void present(const vk::raii::Queue queue, const vk::ArrayProxy<shared_ptr<vk::raii::Semaphore>>& waitSemaphores = {});

	// Number of times present has been called
	inline size_t presentCount() const { return mPresentCount; }

private:
	vk::raii::SwapchainKHR mSwapchain;
	vk::Extent2D mExtent;
	vector<shared_ptr<Image>> mImages;
	vector<shared_ptr<vk::raii::Semaphore>> mImageAvailableSemaphores;
	uint32_t mMinImageCount;
	uint32_t mBackBufferIndex;
	uint32_t mImageAvailableSemaphoreIndex;
	vk::ImageUsageFlags mUsage;

	vk::SurfaceFormatKHR mSurfaceFormat;
	vk::PresentModeKHR mPresentMode;
	chrono::nanoseconds mAcquireImageTimeout = 10s;
	size_t mPresentCount = 0;
	bool mDirty = false;
};

}