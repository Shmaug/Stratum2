#include "Buffer.hpp"

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
	vmaCreateBuffer(mDevice.allocator(), &(const VkBufferCreateInfo&)createInfo, &allocationCreateInfo, &(VkBuffer&)mBuffer, &mAllocation, &mAllocationInfo);
	device.setDebugName(mBuffer, resourceName());
}
Buffer::~Buffer() {
	if (mBuffer && mAllocation)
		vmaDestroyBuffer(mDevice.allocator(), mBuffer, mAllocation);
}

}