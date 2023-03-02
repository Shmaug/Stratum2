#pragma once

#include "Device.hpp"

namespace stm2 {

class CommandBuffer : public Device::Resource {
public:
	CommandBuffer(Device& device, const string& name, const uint32_t queueFamily);

	DECLARE_DEREFERENCE_OPERATORS(vk::raii::CommandBuffer, mCommandBuffer)

	inline const shared_ptr<vk::raii::Fence>& fence() const { return mFence; }
	inline uint32_t queueFamily() const { return mQueueFamily; }
	inline size_t frameIndex() const { return mFrameIndex; }

	inline void trackResource(const shared_ptr<Device::Resource>& r) {
		r->markUsed();
		mResources.emplace(r);
	}

	template<typename T>
	inline void trackVulkanResource(const shared_ptr<T>& r) {
		class ResourceWrapper : public Device::Resource {
		public:
			shared_ptr<T> mResource;
			inline ResourceWrapper(Device& device, const string& name, shared_ptr<T>&& r)
				: Device::Resource(device, name), mResource(r) {}
			inline ResourceWrapper(Device& device, const string& name, const shared_ptr<T>& r)
				: Device::Resource(device, name), mResource(r) {}
			ResourceWrapper(ResourceWrapper&&) = default;
			ResourceWrapper(const ResourceWrapper&) = default;
		};

		trackResource(make_shared<ResourceWrapper>(mDevice, resourceName() + "/ResourceWrapper", r));
	}

	void reset();

private:
	friend class Device;
	vk::raii::CommandBuffer mCommandBuffer;
	shared_ptr<vk::raii::Fence> mFence;
	uint32_t mQueueFamily;
	unordered_set<shared_ptr<Device::Resource>> mResources;
	size_t mFrameIndex;
};

}
