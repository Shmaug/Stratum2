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

	vector<uint8_t> cacheData;
	string tmp;
	vk::PipelineCacheCreateInfo cacheInfo = {};
	if (!mInstance.findArgument("noPipelineCache")) {
		try {
			cacheData = readFile<vector<uint8_t>>(filesystem::temp_directory_path() / "stm2_pcache");
			if (!cacheData.empty()) {
				cacheInfo.pInitialData = cacheData.data();
				cacheInfo.initialDataSize = cacheData.size();
				cout << "Read pipeline cache (" << fixed << showpoint << setprecision(2) << cacheData.size()/1024.f << "KiB)" << endl;
			}
		} catch (exception& e) {
			cerr << "Warning: Failed to read pipeline cache: " << e.what() << endl;
		}
	}
	mPipelineCache = vk::raii::PipelineCache(mDevice, cacheInfo);

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
	if (!mInstance.findArgument("noPipelineCache")) {
		try {
			const vector<uint8_t> cacheData = mPipelineCache.getData();
			if (!cacheData.empty())
				writeFile(filesystem::temp_directory_path() / "stm2_pcache", cacheData);
		} catch (exception& e) {
			cerr << "Warning: Failed to write pipeline cache: " << e.what() << endl;
		}
	}
	vmaDestroyAllocator(mAllocator);
}

vk::raii::CommandPool& Device::commandPool(const uint32_t queueFamily) {
	scoped_lock l(mCommandPoolMutex);
	auto& pools = mCommandPools[this_thread::get_id()];
	if (auto it = pools.find(queueFamily); it != pools.end())
		return it->second;
	return pools.emplace(queueFamily, vk::raii::CommandPool(mDevice, vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamily))).first->second;
}

const shared_ptr<vk::raii::DescriptorPool>& Device::allocateDescriptorPool() {
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
	mDescriptorPools.push(make_shared<vk::raii::DescriptorPool>(mDevice, vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 8192, poolSizes)));
	return mDescriptorPools.top();
}
const shared_ptr<vk::raii::DescriptorPool>& Device::getDescriptorPool() {
	if (mDescriptorPools.empty())
		return allocateDescriptorPool();
	return mDescriptorPools.top();
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
	if (ImGui::CollapsingHeader("Heap budgets")) {
		VmaBudget budgets[VK_MAX_MEMORY_HEAPS];
		vmaGetHeapBudgets(mAllocator, budgets);
		const vk::PhysicalDeviceMemoryProperties properties = mPhysicalDevice.getMemoryProperties();
		for (uint32_t heapIndex = 0; heapIndex < properties.memoryHeapCount; heapIndex++) {
			ImGui::Text("Heap %u %s", heapIndex, (properties.memoryHeaps[heapIndex].flags & vk::MemoryHeapFlagBits::eDeviceLocal) ? "(device local)" : "");
			ImGui::Indent();

			const auto[usage, usageUnit] = formatBytes(budgets[heapIndex].usage);
			const auto[budget, budgetUnit] = formatBytes(budgets[heapIndex].budget);
			ImGui::Text("%llu %s used, %llu %s budgeted", usage, usageUnit, budget, budgetUnit);

			const auto[allocationBytes, allocationBytesUnit] = formatBytes(budgets[heapIndex].statistics.allocationBytes);
			ImGui::Text("%u allocations\t(%llu %s)", budgets[heapIndex].statistics.allocationCount, allocationBytes, allocationBytesUnit);

			const auto[blockBytes, blockBytesUnit] = formatBytes(budgets[heapIndex].statistics.blockBytes);
			ImGui::Text("%u memory blocks\t(%llu %s)", budgets[heapIndex].statistics.blockCount, blockBytes, blockBytesUnit);

			ImGui::Unindent();
		}
	}
}

}