#pragma once

#include <unordered_set>

#include <vk_mem_alloc.h>

#include "fwd.hpp"
#include "utils.hpp"

namespace stm2 {

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
		inline size_t lastFrameUsed() const { return mLastFrameUsed; }
		inline bool inFlight() const { return mLastFrameUsed > mDevice.lastFrameDone(); }
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
	inline const vk::PhysicalDeviceVulkan13Features&                 vulkan13Features() const              { return get<vk::PhysicalDeviceVulkan13Features>(mFeatureChain); }
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

	const shared_ptr<vk::raii::DescriptorPool>& allocateDescriptorPool();
	const shared_ptr<vk::raii::DescriptorPool>& getDescriptorPool();

	inline uint32_t findQueueFamily(const vk::QueueFlags flags = vk::QueueFlagBits::eGraphics|vk::QueueFlagBits::eCompute|vk::QueueFlagBits::eTransfer) {
		return stm2::findQueueFamily(mPhysicalDevice, flags);
	}

	void submit(
		const vk::raii::Queue queue,
		const vk::ArrayProxy<const shared_ptr<CommandBuffer>>& commandBuffers,
		const vk::ArrayProxy<pair<shared_ptr<vk::raii::Semaphore>, vk::PipelineStageFlags>>& waitSemaphores = {},
		const vk::ArrayProxy<shared_ptr<vk::raii::Semaphore>>& signalSemaphores = {});

	inline size_t frameIndex() const { return mFrameIndex; }
	inline size_t lastFrameDone() const { return mLastFrameDone; }
	void incrementFrameIndex() { mFrameIndex++; }
	void updateLastFrameDone(const size_t v) { mLastFrameDone = v; }

	void drawGui();

private:
	vk::raii::Device mDevice;
 	vk::raii::PhysicalDevice mPhysicalDevice;
	vk::raii::PipelineCache mPipelineCache;
	unordered_map<uint32_t, vk::raii::CommandPool> mCommandPools;
	stack<shared_ptr<vk::raii::DescriptorPool>> mDescriptorPools;

	VmaAllocator mAllocator;

	size_t mFrameIndex;
	size_t mLastFrameDone;

	vk::PhysicalDeviceFeatures mFeatures;
	vk::StructureChain<
		vk::DeviceCreateInfo,
		vk::PhysicalDeviceVulkan13Features,
		vk::PhysicalDeviceDescriptorIndexingFeatures,
		vk::PhysicalDeviceBufferDeviceAddressFeatures,
		vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
		vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,
		vk::PhysicalDeviceRayQueryFeaturesKHR
	> mFeatureChain;
	vk::PhysicalDeviceLimits mLimits;
};

}