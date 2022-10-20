#include "Buffer.hpp"

namespace tinyvkpt {

Buffer::Buffer(Device& device, const string& name, const vk::DeviceSize size, const vk::BufferUsageFlags usage, const VmaMemoryUsage memoryUsage, const vk::SharingMode sharingMode)
	: Device::Resource(device, name), mBuffer(nullptr), mSize(size), mUsage(usage), mMemoryUsage(memoryUsage), mSharingMode(sharingMode) {
	mBuffer = vk::raii::Buffer(*device, vk::BufferCreateInfo( ));
	device.setDebugName(*mBuffer, resourceName());
}

}