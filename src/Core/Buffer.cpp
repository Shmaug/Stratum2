#include "Buffer.hpp"

namespace tinyvkpt {

Buffer::Buffer(Device& device, const string& name, const vk::DeviceSize size, const vk::BufferUsageFlags usage, const vk::MemoryPropertyFlags memoryFlags, const vk::SharingMode sharingMode)
	: Device::Resource(device, name), mSize(size), mUsage(usage), mMemoryFlags(memoryFlags), mSharingMode(sharingMode) {
	// TODO: implement vma
	device.setDebugName(mBuffer, resourceName());
}
Buffer::~Buffer() {
	if (mBuffer && mAllocation)
		vmaDestroyBuffer(mDevice.allocator(), mBuffer, mAllocation);
}

}