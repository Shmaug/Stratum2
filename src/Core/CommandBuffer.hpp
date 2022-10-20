#pragma once

#include "Device.hpp"

namespace tinyvkpt {

// removes commandbuffer from deviceresource's 'in flight' list upon destruction
class CommandBufferResourceTracker {
friend class CommandBuffer;
private:
	shared_ptr<Device::Resource> mResource;
	CommandBuffer* mCommandBuffer;

	inline CommandBufferResourceTracker(const shared_ptr<Device::Resource>& r, CommandBuffer* cb) : mResource(r), mCommandBuffer(cb) {
		mResource->mInFlight.emplace(mCommandBuffer);
	}

public:
	CommandBufferResourceTracker(const CommandBufferResourceTracker&) = delete;
	CommandBufferResourceTracker& operator=(const CommandBufferResourceTracker&& r) = delete;

	inline CommandBufferResourceTracker() : mResource(nullptr), mCommandBuffer(nullptr) {}
	inline CommandBufferResourceTracker(CommandBufferResourceTracker&& r) {
		mResource = move(r.mResource);
		mCommandBuffer = r.mCommandBuffer;
		r.mCommandBuffer = nullptr;
	}
	inline ~CommandBufferResourceTracker() {
		if (mResource && mCommandBuffer)
			mResource->mInFlight.erase(mCommandBuffer);
	}
	inline CommandBufferResourceTracker& operator=(CommandBufferResourceTracker&& r) {
		mResource = move(r.mResource);
		mCommandBuffer = r.mCommandBuffer;
		r.mCommandBuffer = nullptr;
		return *this;
	}

	inline shared_ptr<Device::Resource>& operator*() { return mResource; }
	inline shared_ptr<Device::Resource>* operator->() { return &mResource; }
	inline const shared_ptr<Device::Resource>& operator*() const { return mResource; }
	inline const shared_ptr<Device::Resource>* operator->() const { return &mResource; }

	inline bool operator!=(const CommandBufferResourceTracker& v) const { return mResource != v.mResource; }
	inline bool operator==(const CommandBufferResourceTracker& v) const { return mResource == v.mResource; }
};

}

namespace std {
template<>
struct hash<tinyvkpt::CommandBufferResourceTracker> {
	inline size_t operator()(const tinyvkpt::CommandBufferResourceTracker& t) const {
		return hash<shared_ptr<tinyvkpt::Device::Resource>>()(*t);
	}
};
}

namespace tinyvkpt {

class CommandBuffer : public Device::Resource {
public:
	CommandBuffer(Device& device, const string& name, const uint32_t queueFamily);

	inline vk::raii::CommandBuffer& operator*() { return mCommandBuffer; }
	inline vk::raii::CommandBuffer* operator->() { return &mCommandBuffer; }
	inline const vk::raii::CommandBuffer& operator*() const { return mCommandBuffer; }
	inline const vk::raii::CommandBuffer* operator->() const { return &mCommandBuffer; }

	inline const shared_ptr<vk::raii::Fence>& fence() const { return mFence; }
	inline uint32_t queueFamily() const { return mQueueFamily; }

	inline void trackResource(const shared_ptr<Device::Resource>& r) {
		CommandBufferResourceTracker t(r, this);
		mResources.emplace(move(t));
	}

private:
	friend class Device;
	vk::raii::CommandBuffer mCommandBuffer;
	shared_ptr<vk::raii::Fence> mFence;
	uint32_t mQueueFamily;
	unordered_set<CommandBufferResourceTracker> mResources;
};

}
