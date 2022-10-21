#include "Buffer.hpp"

namespace tinyvkpt {

Buffer::Buffer(Device& device, const string& name, const vk::DeviceSize size, const vk::BufferUsageFlags usage, const vk::MemoryPropertyFlags memoryFlags, const vk::SharingMode sharingMode)
	: Device::Resource(device, name), mBuffer(nullptr), mSize(size), mUsage(usage), mMemoryFlags(memoryFlags), mSharingMode(sharingMode) {
	mBuffer = vk::raii::Buffer(*device, vk::BufferCreateInfo( ));
	device.setDebugName(*mBuffer, resourceName());
	// TODO: implement vma
}

}