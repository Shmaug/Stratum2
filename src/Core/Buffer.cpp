#include "Buffer.hpp"
#include "Image.hpp"

namespace tinyvkpt {

Buffer::Buffer(Device& device, const string& name, const vk::BufferCreateInfo& createInfo, const vk::MemoryPropertyFlags memoryFlags, const bool randomHostAccess)
	: Device::Resource(device, name), mSize(createInfo.size), mMemoryFlags(memoryFlags) {
	VmaAllocationCreateInfo allocationCreateInfo;
	allocationCreateInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | (randomHostAccess ? VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT : VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
    allocationCreateInfo.usage = (memoryFlags & vk::MemoryPropertyFlagBits::eDeviceLocal) ? VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE : VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    allocationCreateInfo.requiredFlags = (VkMemoryPropertyFlags)memoryFlags;
    allocationCreateInfo.memoryTypeBits = 0;
    allocationCreateInfo.pool = VK_NULL_HANDLE;
    allocationCreateInfo.pUserData = VK_NULL_HANDLE;
    allocationCreateInfo.priority = 0;
	vk::Result result = (vk::Result)vmaCreateBuffer(mDevice.allocator(), &(const VkBufferCreateInfo&)createInfo, &allocationCreateInfo, &(VkBuffer&)mBuffer, &mAllocation, &mAllocationInfo);
	if (result != vk::Result::eSuccess)
		vk::throwResultException(result, "vmaCreateBuffer");
	device.setDebugName(mBuffer, resourceName());
}
Buffer::~Buffer() {
	if (mBuffer && mAllocation)
		vmaDestroyBuffer(mDevice.allocator(), mBuffer, mAllocation);
}

void Buffer::fill(CommandBuffer& commandBuffer, const uint32_t data, const vk::DeviceSize offset, const vk::DeviceSize size) const {
	commandBuffer->fillBuffer(mBuffer, offset, size, data);
}

void Buffer::barrier(CommandBuffer& commandBuffer,
	const vk::PipelineStageFlags srcStage, const vk::PipelineStageFlags dstStage,
	const vk::AccessFlags srcAccess, const vk::AccessFlags dstAccess,
	const uint32_t srcQueue, const uint32_t dstQueue, const vk::DeviceSize offset, const vk::DeviceSize size) const {
	commandBuffer->pipelineBarrier(
		srcStage, dstStage,
		vk::DependencyFlagBits::eByRegion,
		{},
		vk::BufferMemoryBarrier(
			srcAccess, dstAccess,
			srcQueue, dstQueue,
			mBuffer, offset, size),
		{});
}

void Buffer::barriers(CommandBuffer& commandBuffer, const vector<Buffer::View<byte>>& buffers,
	const vk::PipelineStageFlags srcStage, const vk::PipelineStageFlags dstStage,
	const vk::AccessFlags srcAccess, const vk::AccessFlags dstAccess,
	const uint32_t srcQueue, const uint32_t dstQueue) {

	for (const auto& b : buffers)
		commandBuffer.trackResource(b.buffer());

	vector<vk::BufferMemoryBarrier> bufferMemoryBarriers(buffers.size());
	ranges::transform(buffers, bufferMemoryBarriers.begin(), [=](const auto& v){ return vk::BufferMemoryBarrier(
			srcAccess, dstAccess,
			srcQueue, dstQueue,
			**v.buffer(), v.offset(), v.sizeBytes()); });

	commandBuffer->pipelineBarrier(
		srcStage, dstStage,
		vk::DependencyFlagBits::eByRegion,
		{},
		bufferMemoryBarriers,
		{});
}

void Buffer::copy(CommandBuffer& commandBuffer, const Buffer::View<byte>& src, const Buffer::View<byte>& dst) {
	if (src.sizeBytes() != dst.sizeBytes())
		throw runtime_error("src and dst size must match");

	commandBuffer.trackResource(src.buffer());
	commandBuffer.trackResource(dst.buffer());

	commandBuffer->copyBuffer(**src.buffer(), **dst.buffer(), vk::BufferCopy(src.offset(), dst.offset(), src.sizeBytes()));
}

void Buffer::copyToImage(CommandBuffer& commandBuffer, const shared_ptr<Image>& dst, const vk::DeviceSize offset) const {
	commandBuffer.trackResource(dst);
	dst->barrier(commandBuffer, vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1), vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite);
	commandBuffer->copyBufferToImage(mBuffer, **dst, vk::ImageLayout::eTransferDstOptimal,
		vk::BufferImageCopy(offset, 0, 0, vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor,0,0,1), {0,0,0}, dst->extent()));
}
void Buffer::copyToImage(CommandBuffer& commandBuffer, const shared_ptr<Image>& dst, const vk::ArrayProxy<vk::BufferImageCopy>& copies) const {
	commandBuffer.trackResource(dst);
	dst->barrier(commandBuffer, vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1), vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite);
	commandBuffer->copyBufferToImage(mBuffer, **dst, vk::ImageLayout::eTransferDstOptimal, copies);
}

}