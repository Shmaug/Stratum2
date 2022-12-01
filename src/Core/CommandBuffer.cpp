#include "CommandBuffer.hpp"
#include "Profiler.hpp"

#include <Utils/hlslmath.hpp>

namespace stm2 {

CommandBuffer::CommandBuffer(Device& device, const string& name, const uint32_t queueFamily) : Device::Resource(device, name), mCommandBuffer(nullptr), mQueueFamily(queueFamily) {
	vk::raii::CommandBuffers commandBuffers(*mDevice, vk::CommandBufferAllocateInfo(*mDevice.commandPool(queueFamily), vk::CommandBufferLevel::ePrimary, 1));
	mCommandBuffer = move(commandBuffers[0]);
	device.setDebugName(*mCommandBuffer, resourceName());
}

void CommandBuffer::reset() {
	mResources.clear();
	mCommandBuffer.reset();
}

}