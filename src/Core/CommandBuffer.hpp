#pragma once

#include "Device.hpp"

namespace tinyvkpt {

// removes commandbuffer from deviceresource's 'in flight' list upon destruction
struct CommandBufferResourceTracker {
private:
	friend class CommandBuffer;
	CommandBuffer* mCommandBuffer = nullptr;
	shared_ptr<Device::Resource> mResource;

	inline CommandBufferResourceTracker(CommandBuffer* cb, const shared_ptr<Device::Resource>& r) : mCommandBuffer(cb), mResource(r) {
		if (mCommandBuffer && mResource)
			mResource->mInFlight.emplace(mCommandBuffer);
	}

public:
	inline ~CommandBufferResourceTracker() {
		if (mCommandBuffer && mResource)
			mResource->mInFlight.erase(mCommandBuffer);
	}

	CommandBufferResourceTracker(CommandBufferResourceTracker&& r) {
		mCommandBuffer = r.mCommandBuffer;
		mResource = move(r.mResource);
		r.mCommandBuffer = nullptr;
	}
	CommandBufferResourceTracker& operator=(CommandBufferResourceTracker&& r) {
		mCommandBuffer = r.mCommandBuffer;
		mResource = move(r.mResource);
		r.mCommandBuffer = nullptr;
	}

	CommandBufferResourceTracker() = default;
	CommandBufferResourceTracker(const CommandBufferResourceTracker&) = delete;
	CommandBufferResourceTracker& operator=(const CommandBufferResourceTracker&) = delete;
};

class CommandBuffer : public Device::Resource {
public:
	CommandBuffer(Device& device, const string& name, const uint32_t queueFamily);

	DECLARE_DEREFERENCE_OPERATORS(vk::raii::CommandBuffer, mCommandBuffer)

	inline const shared_ptr<vk::raii::Fence>& fence() const { return mFence; }
	inline uint32_t queueFamily() const { return mQueueFamily; }

	inline void trackResource(const shared_ptr<Device::Resource>& r) {
		mResources.emplace(r.get(), CommandBufferResourceTracker(this, r));
	}

	template<typename T>
	inline void trackVulkanResource(const shared_ptr<T>& vr) {
		class ResourceWrapperType : public Device::Resource {
		public:
			shared_ptr<T> mResource;
			inline ResourceWrapperType(Device& device, const string& name, shared_ptr<T>&& r)
				: Device::Resource(device, name), mResource(r) {}
			inline ResourceWrapperType(Device& device, const string& name, const shared_ptr<T>& r)
				: Device::Resource(device, name), mResource(r) {}
			ResourceWrapperType(ResourceWrapperType&&) = default;
			ResourceWrapperType(const ResourceWrapperType&) = default;
			ResourceWrapperType& operator=(ResourceWrapperType&&) = default;
			ResourceWrapperType& operator=(const ResourceWrapperType&) = default;
		};
		mResources.emplace(vr.get(), CommandBufferResourceTracker(this, make_shared<ResourceWrapperType>(mDevice, resourceName() + "/ResourceWrapper", vr)));
	}

	void reset();

private:
	friend class Device;
	vk::raii::CommandBuffer mCommandBuffer;
	shared_ptr<vk::raii::Fence> mFence;
	uint32_t mQueueFamily;
	unordered_map<void*, CommandBufferResourceTracker> mResources;
};

}
