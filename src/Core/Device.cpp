#define VMA_IMPLEMENTATION
#include "Device.hpp"
#include "Instance.hpp"
#include "CommandBuffer.hpp"
#include "Profiler.hpp"

#include <imgui/imgui.h>
#include <algorithm>

namespace stm2 {

Device::Device(Instance& instance, vk::raii::PhysicalDevice physicalDevice) :
	mInstance(instance),
	mPhysicalDevice(physicalDevice),
	mDevice(nullptr),
	mPipelineCache(nullptr),
	mDescriptorPool(nullptr),
	mFrameIndex(0),
	mLastFrameDone(0) {
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
	mFeatures.samplerAnisotropy = true;
	mFeatures.shaderImageGatherExtended = true;
	mFeatures.shaderStorageImageExtendedFormats = true;
	mFeatures.wideLines = true;
	mFeatures.largePoints = true;
	mFeatures.sampleRateShading = true;
	mFeatures.shaderFloat64 = true; // needed by slang?
	mFeatures.shaderStorageBufferArrayDynamicIndexing = true;
	mFeatures.shaderSampledImageArrayDynamicIndexing = true;
	mFeatures.shaderStorageImageArrayDynamicIndexing = true;

	vk::PhysicalDeviceVulkan13Features& vk13features = get<vk::PhysicalDeviceVulkan13Features>(mFeatureChain);
	vk13features.dynamicRendering = true;

	vk::PhysicalDeviceDescriptorIndexingFeatures& difeatures = get<vk::PhysicalDeviceDescriptorIndexingFeatures>(mFeatureChain);
	difeatures.shaderStorageBufferArrayNonUniformIndexing = true;
	difeatures.shaderSampledImageArrayNonUniformIndexing = true;
	difeatures.shaderStorageImageArrayNonUniformIndexing = true;
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
	createInfo.setPNext(&vk13features);
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
	if (get<vk::PhysicalDeviceBufferDeviceAddressFeatures>(mFeatureChain).bufferDeviceAddress)
		allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	else
		allocatorInfo.flags = 0;
	vmaCreateAllocator(&allocatorInfo, &mAllocator);
}
Device::~Device() {
	vmaDestroyAllocator(mAllocator);
}

vk::raii::CommandPool& Device::commandPool(const uint32_t queueFamily) {
	if (auto it = mCommandPools.find(queueFamily); it != mCommandPools.end())
		return it->second;
	return mCommandPools.emplace(queueFamily, vk::raii::CommandPool(mDevice, vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamily))).first->second;
}

const shared_ptr<vk::raii::DescriptorPool>& Device::descriptorPool() {
	if (!mDescriptorPool) {
		vector<vk::DescriptorPoolSize> poolSizes {
			vk::DescriptorPoolSize(vk::DescriptorType::eSampler,              min(16384u, mLimits.maxDescriptorSetSamplers)),
			vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, min(16384u, mLimits.maxDescriptorSetSampledImages)),
			vk::DescriptorPoolSize(vk::DescriptorType::eInputAttachment,      min(16384u, mLimits.maxDescriptorSetInputAttachments)),
			vk::DescriptorPoolSize(vk::DescriptorType::eSampledImage,         min(16384u, mLimits.maxDescriptorSetSampledImages)),
			vk::DescriptorPoolSize(vk::DescriptorType::eStorageImage,         min(16384u, mLimits.maxDescriptorSetStorageImages)),
			vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer,        min(16384u, mLimits.maxDescriptorSetUniformBuffers)),
			vk::DescriptorPoolSize(vk::DescriptorType::eUniformBufferDynamic, min(16384u, mLimits.maxDescriptorSetUniformBuffersDynamic)),
			vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer,        min(16384u, mLimits.maxDescriptorSetStorageBuffers)),
			vk::DescriptorPoolSize(vk::DescriptorType::eStorageBufferDynamic, min(16384u, mLimits.maxDescriptorSetStorageBuffersDynamic))
		};
		mDescriptorPool = make_shared<vk::raii::DescriptorPool>(mDevice, vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 4096, poolSizes));
	}
	return mDescriptorPool;
}

void Device::submit(const vk::raii::Queue queue, const vk::ArrayProxy<const shared_ptr<CommandBuffer>>& commandBuffers, const vk::ArrayProxy<pair<shared_ptr<vk::raii::Semaphore>, vk::PipelineStageFlags>>& waitSemaphores, const vk::ArrayProxy<shared_ptr<vk::raii::Semaphore>>& signalSemaphores) {
	// create or reuse fence
	shared_ptr<vk::raii::Fence> fence;
	for (auto cb : commandBuffers)
		if (cb->fence()) {
			fence = cb->fence();
			mDevice.resetFences(**fence);
			break;
		}
	if (!fence)
		fence = make_shared<vk::raii::Fence>(mDevice, vk::FenceCreateInfo());

	// assign fence, get vkbufs
	vector<vk::CommandBuffer> vkbufs;
	for (const shared_ptr<CommandBuffer>& cb : commandBuffers) {
		cb->mFrameIndex = frameIndex();
		cb->mFence = fence;
		vkbufs.emplace_back(***cb);
	}

	vector<vk::Semaphore> vkWaitSemaphores(waitSemaphores.size());
	vector<vk::PipelineStageFlags> waitStages(waitSemaphores.size());
	vector<vk::Semaphore> vkSignalSemaphores(signalSemaphores.size());
	ranges::transform(waitSemaphores, vkWaitSemaphores.begin(), [](auto s){ return **s.first; } );
	ranges::transform(waitSemaphores, waitStages.begin(), &pair<shared_ptr<vk::raii::Semaphore>, vk::PipelineStageFlags>::second);
	ranges::transform(signalSemaphores, vkSignalSemaphores.begin(), [](auto s){ return **s; } );
	queue.submit(vk::SubmitInfo(vkWaitSemaphores, waitStages, vkbufs, vkSignalSemaphores), **fence);
}

void Device::drawGui() {
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS];
	vmaGetHeapBudgets(mAllocator, budgets);
	const vk::PhysicalDeviceMemoryProperties properties = mPhysicalDevice.getMemoryProperties();
	for (uint32_t heapIndex = 0; heapIndex < properties.memoryHeapCount; heapIndex++) {
		const auto[usage, usageUnit] = formatBytes(budgets[heapIndex].usage);
		const auto[budget, budgetUnit] = formatBytes(budgets[heapIndex].budget);
		const auto[allocationBytes, allocationBytesUnit] = formatBytes(budgets[heapIndex].statistics.allocationBytes);
		const auto[blockBytes, blockBytesUnit] = formatBytes(budgets[heapIndex].statistics.blockBytes);
		ImGui::Text("Heap %u %s", heapIndex, (properties.memoryHeaps[heapIndex].flags & vk::MemoryHeapFlagBits::eDeviceLocal) ? "(device local)" : "");
		ImGui::Text("%llu %s used, %llu %s budgeted", usage, usageUnit, budget, budgetUnit);
		ImGui::Indent();
		ImGui::Text("%u allocations (%llu %s)", budgets[heapIndex].statistics.allocationCount, allocationBytes, allocationBytesUnit);
		ImGui::Text("%u device memory blocks (%llu %s)", budgets[heapIndex].statistics.blockCount, blockBytes, blockBytesUnit);
		ImGui::Unindent();
	}
}

}