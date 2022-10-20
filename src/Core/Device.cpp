#define VMA_IMPLEMENTATION
#include "Device.hpp"
#include "Instance.hpp"
#include "CommandBuffer.hpp"

#include <iostream>
#include <algorithm>

namespace tinyvkpt {

Device::Device(Instance& instance, vk::raii::PhysicalDevice physicalDevice) : mInstance(instance), mPhysicalDevice(physicalDevice), mDevice(nullptr), mPipelineCache(nullptr), mDescriptorPool(nullptr) {
	unordered_set<string> deviceExtensions;
	for (const string& s : mInstance.findArguments("deviceExtension"))
		deviceExtensions.emplace(s);
	deviceExtensions.emplace(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
	if (deviceExtensions.contains(VK_KHR_RAY_QUERY_EXTENSION_NAME)) {
		deviceExtensions.emplace(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
		deviceExtensions.emplace(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
		deviceExtensions.emplace(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
	}
	if (deviceExtensions.contains(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME))
		deviceExtensions.emplace(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);

	// Queue create infos

	vector<vk::QueueFamilyProperties> queueFamilyProperties = mPhysicalDevice.getQueueFamilyProperties();
	vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
	float queuePriority = 1.0f;
	for (uint32_t i = 0; i < queueFamilyProperties.size(); i++) {
		if (queueFamilyProperties[i].queueFlags & (vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eTransfer)) {
			queueCreateInfos.emplace_back(vk::DeviceQueueCreateInfo({}, i, 1, &queuePriority));
		}
	}

	// Logical device features

	mFeatures.fillModeNonSolid = true;
	mFeatures.sparseBinding = true;
	mFeatures.samplerAnisotropy = true;
	mFeatures.shaderImageGatherExtended = true;
	mFeatures.shaderStorageImageExtendedFormats = true;
	mFeatures.wideLines = true;
	mFeatures.largePoints = true;
	mFeatures.sampleRateShading = true;
	mFeatures.shaderFloat64 = true; // needed by slang?
	mFeatures.shaderUniformBufferArrayDynamicIndexing = true;
	mFeatures.shaderStorageBufferArrayDynamicIndexing = true;
	mFeatures.shaderSampledImageArrayDynamicIndexing = true;
	mFeatures.shaderStorageImageArrayDynamicIndexing = true;

	auto& difeatures = get<vk::PhysicalDeviceDescriptorIndexingFeatures>(mFeatureChain);
	difeatures.shaderUniformBufferArrayNonUniformIndexing = true;
	difeatures.shaderStorageBufferArrayNonUniformIndexing = true;
	difeatures.shaderSampledImageArrayNonUniformIndexing = true;
	difeatures.shaderStorageImageArrayNonUniformIndexing = true;
	difeatures.shaderUniformTexelBufferArrayNonUniformIndexing = true;
	difeatures.shaderStorageTexelBufferArrayNonUniformIndexing = true;
	difeatures.descriptorBindingPartiallyBound = true;
	get<vk::PhysicalDeviceBufferDeviceAddressFeatures>(mFeatureChain).bufferDeviceAddress = deviceExtensions.contains(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
	get<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>(mFeatureChain).accelerationStructure = deviceExtensions.contains(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
	auto& rtfeatures = get<vk::PhysicalDeviceRayTracingPipelineFeaturesKHR>(mFeatureChain);
	rtfeatures.rayTracingPipeline = deviceExtensions.contains(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
	rtfeatures.rayTraversalPrimitiveCulling = rtfeatures.rayTracingPipeline;
	get<vk::PhysicalDeviceRayQueryFeaturesKHR>(mFeatureChain).rayQuery = deviceExtensions.contains(VK_KHR_RAY_QUERY_EXTENSION_NAME);

	// Create logical device

	vector<const char*> deviceExts;
	for (const string& s : deviceExtensions)
		deviceExts.emplace_back(s.c_str());

	vector<const char*> validationLayers;
	for (const string& v : instance.validationLayers())
		validationLayers.push_back(v.c_str());

	const vk::PhysicalDeviceProperties properties = mPhysicalDevice.getProperties();
	mLimits = properties.limits;

	auto& createInfo = get<vk::DeviceCreateInfo>(mFeatureChain);
	createInfo.setQueueCreateInfos(queueCreateInfos);
	createInfo.setPEnabledLayerNames(validationLayers);
	createInfo.setPEnabledExtensionNames(deviceExts);
	createInfo.setPEnabledFeatures(&mFeatures);
	createInfo.setPNext(&difeatures);
	mDevice = mPhysicalDevice.createDevice(createInfo);
	setDebugName(*mDevice, "[" + to_string(properties.deviceID) + "]: " + properties.deviceName.data());

	// Load pipeline cache

	vk::PipelineCacheCreateInfo cacheInfo = {};
	string tmp;
	if (!mInstance.findArgument("noPipelineCache")) {
		try {
			ifstream cacheFile(filesystem::temp_directory_path() / "pcache", ios::binary | ios::ate);
			if (cacheFile.is_open()) {
				cacheInfo.initialDataSize = cacheFile.tellg();
				cacheInfo.pInitialData = new char[cacheInfo.initialDataSize];
				cacheFile.seekg(0, ios::beg);
				cacheFile.read((char*)cacheInfo.pInitialData, cacheInfo.initialDataSize);
				cout << "Read pipeline cache (" << fixed << showpoint << setprecision(2) << cacheInfo.initialDataSize/1024.f << "KiB)" << endl;
			}
		} catch (exception& e) {
			cerr << "Warning: Failed to read pipeline cache: " << e.what() << endl;
		}
	}
	mPipelineCache = vk::raii::PipelineCache(mDevice, cacheInfo);
	if (cacheInfo.pInitialData) delete[] cacheInfo.pInitialData;

	// Create VMA allocator

	VmaAllocatorCreateInfo allocatorInfo{};
	allocatorInfo.physicalDevice = *mPhysicalDevice;
	allocatorInfo.device = *mDevice;
	allocatorInfo.instance = **mInstance;
	allocatorInfo.vulkanApiVersion = mInstance.vulkanVersion();
	allocatorInfo.preferredLargeHeapBlockSize = 1024 * 1024;
	allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	vmaCreateAllocator(&allocatorInfo, &mAllocator);
}
Device::~Device() {
	vmaDestroyAllocator(mAllocator);
}

vk::raii::CommandPool& Device::commandPool(const uint32_t queueFamily) {
	if (auto it = mCommandPools.find(queueFamily); it != mCommandPools.end())
		return it->second;
	return mCommandPools.emplace(queueFamily, vk::raii::CommandPool(mDevice, vk::CommandPoolCreateInfo({}, queueFamily))).first->second;
}

vk::raii::DescriptorPool& Device::descriptorPool() {
	if (!*mDescriptorPool) {
		vector<vk::DescriptorPoolSize> poolSizes {
			vk::DescriptorPoolSize(vk::DescriptorType::eSampler,              min(4096u, mLimits.maxDescriptorSetSamplers)),
			vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, min(4096u, mLimits.maxDescriptorSetSampledImages)),
			vk::DescriptorPoolSize(vk::DescriptorType::eInputAttachment,      min(4096u, mLimits.maxDescriptorSetInputAttachments)),
			vk::DescriptorPoolSize(vk::DescriptorType::eSampledImage,         min(4096u, mLimits.maxDescriptorSetSampledImages)),
			vk::DescriptorPoolSize(vk::DescriptorType::eStorageImage,         min(4096u, mLimits.maxDescriptorSetStorageImages)),
			vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer,        min(4096u, mLimits.maxDescriptorSetUniformBuffers)),
			vk::DescriptorPoolSize(vk::DescriptorType::eUniformBufferDynamic, min(4096u, mLimits.maxDescriptorSetUniformBuffersDynamic)),
			vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer,        min(4096u, mLimits.maxDescriptorSetStorageBuffers)),
			vk::DescriptorPoolSize(vk::DescriptorType::eStorageBufferDynamic, min(4096u, mLimits.maxDescriptorSetStorageBuffersDynamic))
		};
		mDescriptorPool = vk::raii::DescriptorPool(mDevice, vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 4096, poolSizes));
	}
	return mDescriptorPool;
}

shared_ptr<CommandBuffer> Device::getCommandBuffer(const uint32_t queueFamily) {
	auto& pool = mCommandBufferPool[queueFamily];
	if (pool.empty())
		return make_shared<CommandBuffer>(*this, "CommandBuffer", queueFamily);
	else {
		const shared_ptr<CommandBuffer> cb = pool.top();
		pool.pop();
		(*cb)->reset();
		cb->mResources.clear();
		return cb;
	}
}

void Device::submit(const vk::raii::Queue queue, const vk::ArrayProxy<shared_ptr<CommandBuffer>>& commandBuffers) {
	// create or reuse fence
	shared_ptr<vk::raii::Fence> fence;
	for (auto cb : commandBuffers)
		if (cb->fence()) {
			fence = cb->fence();
			break;
		}
	if (!fence) fence = make_shared<vk::raii::Fence>(mDevice, vk::FenceCreateInfo());

	// assign fence, track in-flight commandbuffers
	vector<vk::CommandBuffer> vkbufs;
	for (auto cb : commandBuffers) {
		cb->mFence = fence;
		vkbufs.emplace_back(***cb);
		mCommandBuffersInFlight[cb->queueFamily()].emplace_back(cb);
	}

	vk::SubmitInfo submitInfo;
	submitInfo.setCommandBuffers(vkbufs);
	queue.submit(vk::ArrayProxy(submitInfo), **fence);
}

void Device::checkCommandBuffers() {
	for (auto&[queueFamily, pool] : mCommandBuffersInFlight) {
		for (auto it = pool.begin(); it != pool.end();) {
			if ((*it)->fence()->getStatus() == vk::Result::eSuccess) {
				mCommandBufferPool[queueFamily].push(*it);
				it = pool.erase(it);
			} else
				it++;
		}
	}
}

}