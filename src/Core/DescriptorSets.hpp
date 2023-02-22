#pragma once

#include <variant>

#include "Buffer.hpp"
#include "Image.hpp"

namespace stm2 {

using BufferDescriptor = Buffer::View<byte>;
using ImageDescriptor = tuple<Image::View, vk::ImageLayout, vk::AccessFlags, shared_ptr<vk::raii::Sampler>>;
using DescriptorValue = variant<BufferDescriptor, ImageDescriptor, shared_ptr<vk::raii::AccelerationStructureKHR>>;
using Descriptors = unordered_map<pair<string, uint32_t>, DescriptorValue>;

class DescriptorSets : public Device::Resource {
public:
	Pipeline& mPipeline;

	DescriptorSets(Pipeline& pipeline, const string& name);
	DescriptorSets() = default;
	DescriptorSets(const DescriptorSets&) = default;
	DescriptorSets(DescriptorSets&&) = default;
	DescriptorSets& operator=(const DescriptorSets&) = default;
	DescriptorSets& operator=(DescriptorSets&&) = default;

	void write(const Descriptors& descriptors = {});

	void transitionImages(CommandBuffer& commandBuffer);
	void bind(CommandBuffer& commandBuffer, const unordered_map<string, uint32_t>& dynamicOffsets = {});

	inline const Descriptors& descriptors() const { return mDescriptors; }

private:
	shared_ptr<vk::raii::DescriptorPool> mDescriptorPool;
	vector<shared_ptr<vk::raii::DescriptorSet>> mDescriptorSets;
	vector<shared_ptr<vk::raii::DescriptorSetLayout>> mDescriptorSetLayouts;
	Descriptors mDescriptors;
};

}