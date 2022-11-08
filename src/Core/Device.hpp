#pragma once

#include <unordered_set>

#include <vk_mem_alloc.h>

#include <Utils/fwd.hpp>
#include <Utils/utils.hpp>

namespace tinyvkpt {

inline uint32_t findQueueFamily(vk::raii::PhysicalDevice& physicalDevice, const vk::QueueFlags flags = vk::QueueFlagBits::eGraphics|vk::QueueFlagBits::eCompute|vk::QueueFlagBits::eTransfer) {
	const auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
	for (uint32_t i = 0; i < queueFamilyProperties.size(); i++) {
		if (queueFamilyProperties[i].queueFlags | flags)
			return i;
	}
	return -1;
}

class Device {
public:
	class Resource {
	public:
		Device& mDevice;
		inline Resource(Device& device, const string& name) : mDevice(device), mName(name) {}
		inline string resourceName() const { return mName; }
		inline bool inFlight() const { return mLastFrameUsed >  mDevice.lastFrameDone(); }
		inline void markUsed() { mLastFrameUsed = mDevice.frameIndex(); }
	private:
		const string mName;
		size_t mLastFrameUsed;
		friend class Device;
	};

	Instance& mInstance;

	Device(Instance& instance, vk::raii::PhysicalDevice physicalDevice);
	~Device();

	DECLARE_DEREFERENCE_OPERATORS(vk::raii::Device, mDevice)

	inline vk::raii::PhysicalDevice physical() const { return mPhysicalDevice; }
	inline vk::raii::PipelineCache& pipelineCache() { return mPipelineCache; }
	inline const vk::raii::PipelineCache& pipelineCache() const { return mPipelineCache; }
	inline VmaAllocator allocator() const { return mAllocator; }

	inline const vk::PhysicalDeviceLimits& limits() const { return mLimits; }
	inline const vk::PhysicalDeviceFeatures& features() const { return mFeatures; }
	inline const vk::PhysicalDeviceDescriptorIndexingFeatures&       descriptorIndexingFeatures() const    { return get<vk::PhysicalDeviceDescriptorIndexingFeatures>(mFeatureChain); }
	inline const vk::PhysicalDeviceBufferDeviceAddressFeatures&      bufferDeviceAddressFeatures() const   { return get<vk::PhysicalDeviceBufferDeviceAddressFeatures>(mFeatureChain); }
	inline const vk::PhysicalDeviceAccelerationStructureFeaturesKHR& accelerationStructureFeatures() const { return get<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>(mFeatureChain); }
	inline const vk::PhysicalDeviceRayTracingPipelineFeaturesKHR&    ray_tracingPipelineFeatures() const   { return get<vk::PhysicalDeviceRayTracingPipelineFeaturesKHR>(mFeatureChain); }
	inline const vk::PhysicalDeviceRayQueryFeaturesKHR&              rayQueryFeatures() const              { return get<vk::PhysicalDeviceRayQueryFeaturesKHR>(mFeatureChain); }

	template<typename T> requires(convertible_to<decltype(T::objectType), vk::ObjectType>)
	inline void setDebugName(const T& object, const string& name) {
		vk::DebugUtilsObjectNameInfoEXT info = {};
		info.objectHandle = *reinterpret_cast<const uint64_t*>(&object);
		info.objectType = T::objectType;
		info.pObjectName = name.c_str();
		mDevice.setDebugUtilsObjectNameEXT(info);
	}

	vk::raii::CommandPool& commandPool(const uint32_t queueFamily);
	vk::raii::DescriptorPool& descriptorPool();

	inline uint32_t frameIndex() const { return mFrameIndex; }
	inline uint32_t lastFrameDone() const { return mLastFrameDone; }

	inline uint32_t findQueueFamily(const vk::QueueFlags flags = vk::QueueFlagBits::eGraphics|vk::QueueFlagBits::eCompute|vk::QueueFlagBits::eTransfer) {
		return tinyvkpt::findQueueFamily(mPhysicalDevice, flags);
	}

	shared_ptr<CommandBuffer> getCommandBuffer(const uint32_t queueFamily);
	void submit(
		const vk::raii::Queue queue,
		const vk::ArrayProxy<shared_ptr<CommandBuffer>>& commandBuffers,
		const vk::ArrayProxy<pair<shared_ptr<vk::raii::Semaphore>, vk::PipelineStageFlags>>& waitSemaphores = {},
		const vk::ArrayProxy<shared_ptr<vk::raii::Semaphore>>& signalSemaphores = {});

	void updateFrame();

private:
	vk::raii::Device mDevice;
 	vk::raii::PhysicalDevice mPhysicalDevice;
	vk::raii::PipelineCache mPipelineCache;
	unordered_map<uint32_t, vk::raii::CommandPool> mCommandPools;
	vk::raii::DescriptorPool mDescriptorPool;

	unordered_map<uint32_t, vector<shared_ptr<CommandBuffer>>> mCommandBuffersInFlight;
	unordered_map<uint32_t, stack<shared_ptr<CommandBuffer>>> mCommandBufferPool;
	VmaAllocator mAllocator;

	size_t mFrameIndex;
	size_t mLastFrameDone;

	vk::PhysicalDeviceFeatures mFeatures;
	vk::StructureChain<
		vk::DeviceCreateInfo,
		vk::PhysicalDeviceDescriptorIndexingFeatures,
		vk::PhysicalDeviceBufferDeviceAddressFeatures,
		vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
		vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,
		vk::PhysicalDeviceRayQueryFeaturesKHR
	> mFeatureChain;
	vk::PhysicalDeviceLimits mLimits;
};

}