#pragma once

#include <mutex>
#include <unordered_set>
#include <stack>

#include <vk_mem_alloc.h>

#include <Utils/utils.hpp>
#include <vulkan/vulkan_raii.hpp>

namespace tinyvkpt {

class Device {
public:
	Instance& mInstance;

	class Resource {
	public:
		Device& mDevice;
		inline Resource(Device& device, const string& name) : mDevice(device), mName(name) {}
		inline string resourceName() const { return mName; }
		inline bool inFlight() const { return !mInFlight.empty(); }
	private:
		friend class CommandBufferResourceTracker;
		// maintained by CommandBuffers
		unordered_set<CommandBuffer*> mInFlight;
		const string mName;
	};

	template<typename T>
	class ResourceWrapper : Resource {
	public:
		T mObject;
		ResourceWrapper(ResourceWrapper&& v) = default;
		ResourceWrapper& operator=(ResourceWrapper&& v) = default;
		inline ResourceWrapper(Device& device, const string& name, T&& v) : mDevice(device), mName(name), mObject(move(v)) {}
		inline T& operator*() { return mObject; }
		inline T* operator->() { return &mObject; }
		inline const T& operator*() const { return mObject; }
		inline const T* operator->() const { return &mObject; }
	};

	Device(Instance& instance, vk::raii::PhysicalDevice physicalDevice);
	~Device();

	inline vk::raii::Device& operator*() { return mDevice; }
	inline vk::raii::Device* operator->() { return &mDevice; }
	inline const vk::raii::Device& operator*() const { return mDevice; }
	inline const vk::raii::Device* operator->() const { return &mDevice; }

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

	shared_ptr<CommandBuffer> getCommandBuffer(const uint32_t queueFamily);
	void submit(const vk::raii::Queue queue, const vk::ArrayProxy<shared_ptr<CommandBuffer>>& commandBuffers);

	void checkCommandBuffers();

private:
	vk::raii::Device mDevice;
 	vk::raii::PhysicalDevice mPhysicalDevice;
	vk::raii::PipelineCache mPipelineCache;
	unordered_map<uint32_t, vk::raii::CommandPool> mCommandPools;
	vk::raii::DescriptorPool mDescriptorPool;

	unordered_map<uint32_t, vector<shared_ptr<CommandBuffer>>> mCommandBuffersInFlight;
	unordered_map<uint32_t, stack<shared_ptr<CommandBuffer>>> mCommandBufferPool;
	VmaAllocator mAllocator;

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