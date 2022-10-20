#include "CommandBuffer.hpp"
#include "Profiler.hpp"

#include <Utils/math.hpp>

namespace tinyvkpt {

ProfilerScope::ProfilerScope(const string& label, CommandBuffer* cmd, const float4& color) : mCommandBuffer(cmd) {
	Profiler::beginSample(label, color);
	if (mCommandBuffer) {
		vk::DebugUtilsLabelEXT info = {};
		copy_n(color.data(), 4, info.color.data());
		info.pLabelName = label.c_str();
		(*mCommandBuffer)->beginDebugUtilsLabelEXT(info);
	}
}
ProfilerScope::~ProfilerScope() {
	if (mCommandBuffer)
		(*mCommandBuffer)->endDebugUtilsLabelEXT();
	Profiler::endSample();
}

CommandBuffer::CommandBuffer(Device& device, const string& name, const uint32_t queueFamily) : Device::Resource(device, name), mCommandBuffer(nullptr) {
	vk::raii::CommandBuffers commandBuffers(*mDevice, vk::CommandBufferAllocateInfo(*mDevice.commandPool(queueFamily), vk::CommandBufferLevel::ePrimary, 1));
	mCommandBuffer = move(commandBuffers[0]);
	device.setDebugName(*mCommandBuffer, resourceName());
}

}