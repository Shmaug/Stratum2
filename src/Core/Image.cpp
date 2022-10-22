#include "Image.hpp"

#include "CommandBuffer.hpp"

namespace tinyvkpt {

Image::Image(Device& device, const string& name, const Metadata& metadata, const vk::MemoryPropertyFlags memoryFlags) : Device::Resource(device, name), mImage(nullptr), mMetadata(metadata) {
	VmaAllocationCreateInfo allocationCreateInfo;
	allocationCreateInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    allocationCreateInfo.requiredFlags = (VkMemoryPropertyFlags)memoryFlags;
    allocationCreateInfo.memoryTypeBits = 0;
    allocationCreateInfo.pool = VK_NULL_HANDLE;
    allocationCreateInfo.pUserData = VK_NULL_HANDLE;
    allocationCreateInfo.priority = 0;

	vk::ImageCreateInfo createInfo(
		{},
		type(),
		format(),
		extent(),
		levels(),
		layers(),
		samples(),
		tiling(),
		usage(),
		sharingMode(),
		queueFamilies(),
		vk::ImageLayout::eUndefined );

	vmaCreateImage(mDevice.allocator(), &(const VkImageCreateInfo&)createInfo, &allocationCreateInfo, &(VkImage&)mImage, &mAllocation, &mAllocationInfo);
	mOwnsImage = true;
	device.setDebugName(mImage, resourceName());
	mSubresourceStates = vector<vector<Image::SubresourceLayoutState>>(
		metadata.mLayers,
		vector<Image::SubresourceLayoutState>(
			metadata.mLevels,
			Image::SubresourceLayoutState{
				vk::ImageLayout::eUndefined,
				vk::PipelineStageFlagBits::eTopOfPipe,
				vk::AccessFlagBits::eNone,
				queueFamilies().empty() ? VK_QUEUE_FAMILY_IGNORED : queueFamilies().front() }));
}
Image::Image(Device& device, const string& name, const vk::Image image, const Metadata& metadata) : Device::Resource(device, name), mImage(image), mMetadata(metadata) {
	mOwnsImage = false;
	mAllocation = nullptr;
	if (mImage) device.setDebugName(mImage, resourceName());
	mSubresourceStates = vector<vector<Image::SubresourceLayoutState>>(
		metadata.mLayers,
		vector<Image::SubresourceLayoutState>(
			metadata.mLevels,
			Image::SubresourceLayoutState{
				vk::ImageLayout::eUndefined,
				vk::PipelineStageFlagBits::eTopOfPipe,
				vk::AccessFlagBits::eNone,
				queueFamilies().empty() ? VK_QUEUE_FAMILY_IGNORED : queueFamilies().front() }));
}
Image::~Image() {
	if (mOwnsImage && mImage && mAllocation)
		vmaDestroyImage(mDevice.allocator(), mImage, mAllocation);
}

const vk::ImageView Image::view(const vk::ImageSubresourceRange& subresource, const vk::ImageViewType viewType, const vk::ComponentMapping& componentMapping) {
	auto it = mViews.find(tie(subresource, viewType, componentMapping));
	if (it == mViews.end()) {
		vk::raii::ImageView v(*mDevice, vk::ImageViewCreateInfo({},
			mImage,
			viewType,
			format(),
			componentMapping,
			subresource));
		it = mViews.emplace(tie(subresource, viewType, componentMapping), move(v)).first;
	}
	return *it->second;
}

void Image::barrier(CommandBuffer& commandBuffer, const vk::ImageSubresourceRange& subresource, const Image::SubresourceLayoutState& newState) {
	const auto& [ newLayout, newStage, dstAccessMask, dstQueueFamilyIndex ] = newState;

	unordered_map<pair<vk::PipelineStageFlags, vk::PipelineStageFlags>, vector<vk::ImageMemoryBarrier>> barriers;

	for (uint32_t arrayLayer = subresource.baseArrayLayer; arrayLayer < subresource.layerCount; arrayLayer++) {
		for (uint32_t level = subresource.baseMipLevel; level < subresource.levelCount; level++) {
			auto& oldState = mSubresourceStates[arrayLayer][level];

			// try to combine barrier with one for previous mip level
			const auto& [ oldLayout, curStage, srcAccessMask, srcQueueFamilyIndex ] = oldState;
			if (oldState != newState) {
				vector<vk::ImageMemoryBarrier>& b = barriers[make_pair(curStage, newStage)];
				if (!b.empty()) {
					vk::ImageMemoryBarrier& prev = b.back();
					if (prev.oldLayout == oldLayout &&
						prev.srcAccessMask == srcAccessMask &&
						prev.srcQueueFamilyIndex == srcQueueFamilyIndex &&
						prev.subresourceRange.baseArrayLayer == arrayLayer &&
						prev.subresourceRange.baseMipLevel + prev.subresourceRange.levelCount == level) {

						prev.subresourceRange.levelCount++;
						oldState = newState;
						continue;
					}
				}

				b.emplace_back(vk::ImageMemoryBarrier(
					srcAccessMask, dstAccessMask,
					oldLayout, newLayout,
					dstQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED ? VK_QUEUE_FAMILY_IGNORED : srcQueueFamilyIndex, srcQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED ? VK_QUEUE_FAMILY_IGNORED : dstQueueFamilyIndex,
					mImage, vk::ImageSubresourceRange(subresource.aspectMask, level, 1, arrayLayer, 1) ));

				oldState = newState;
			}
		}
	}

	for (const auto&[stages, b] : barriers)
		commandBuffer->pipelineBarrier(stages.first, stages.second, vk::DependencyFlagBits::eByRegion, {}, {}, b);
}

void Image::updateState(const vk::ImageSubresourceRange& subresource, const Image::SubresourceLayoutState& newState) {
	for (uint32_t arrayLayer = subresource.baseArrayLayer; arrayLayer < subresource.layerCount; arrayLayer++) {
		for (uint32_t level = subresource.baseMipLevel; level < subresource.levelCount; level++) {
			mSubresourceStates[arrayLayer][level] = newState;
		}
	}
}

}