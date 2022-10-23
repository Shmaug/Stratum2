#pragma once

#include <variant>

#include "Buffer.hpp"
#include "Image.hpp"
#include "Pipeline.hpp"
#include "ResourcePool.hpp"

namespace tinyvkpt {

class PushConstantValue {
public:
	template<typename T>
	inline PushConstantValue(const T& value) {
		mData.resize(sizeof(value));
		*reinterpret_cast<T*>(mData.data()) = value;
	}

	PushConstantValue() = default;
	PushConstantValue(PushConstantValue&&) = default;
	PushConstantValue(const PushConstantValue&) = default;
	PushConstantValue& operator=(PushConstantValue&&) = default;
	PushConstantValue& operator=(const PushConstantValue&) = default;

	inline const vector<byte>& data() const { return mData; }
private:
	vector<byte> mData;
};

using PushConstants = unordered_map<string, PushConstantValue>;


using BufferDescriptor = Buffer::View<byte>;
using ImageDescriptor = tuple<Image::View, vk::ImageLayout, vk::AccessFlags, shared_ptr<vk::raii::Sampler>>;
using DescriptorValue = variant<BufferDescriptor, ImageDescriptor>;
using Descriptors = unordered_map<pair<string, uint32_t>, DescriptorValue>;


class DescriptorSets : public Device::Resource {
public:
	shared_ptr<ComputePipeline> mPipeline;
	vector<shared_ptr<vk::raii::DescriptorSet>> mDescriptorSets;
	Descriptors mDescriptors;

	DescriptorSets(Device& device, const string& name, const shared_ptr<ComputePipeline>& pipeline, const Descriptors& descriptors = {});
	DescriptorSets() = default;
	DescriptorSets(const DescriptorSets&) = default;
	DescriptorSets(DescriptorSets&&) = default;
	DescriptorSets& operator=(const DescriptorSets&) = default;
	DescriptorSets& operator=(DescriptorSets&&) = default;

	void write(const Descriptors& descriptors = {});

	void bind(CommandBuffer& commandBuffer, const unordered_map<string, uint32_t>& dynamicOffsets = {});
};

// Handles DescriptorSet creation and re-use. operator() binds and dispatches the pipeline with the given arguments.
class ComputePipelineContext {
public:
	ComputePipelineContext() = default;
	ComputePipelineContext(const ComputePipelineContext&) = default;
	ComputePipelineContext(ComputePipelineContext&&) = default;
	ComputePipelineContext& operator=(const ComputePipelineContext&) = default;
	ComputePipelineContext& operator=(ComputePipelineContext&&) = default;

	inline ComputePipelineContext(const shared_ptr<ComputePipeline>& pipeline) : mPipeline(pipeline) {}

	inline const shared_ptr<ComputePipeline>& pipeline() const { return mPipeline; }
	inline vk::Extent3D dispatchDimensions(const vk::Extent3D& extent) {
		const vk::Extent3D& s = mPipeline->shader()->workgroupSize();
		return {
			(extent.width  + s.width - 1)  / s.width,
			(extent.height + s.height - 1) / s.height,
			(extent.depth  + s.depth - 1)  / s.depth };
	}
	inline size_t resourceCount() const { return mResources.size(); }

	// gets descriptorsets
	shared_ptr<DescriptorSets> getDescriptorSets(Device& device, const Descriptors& descriptors = {});

	void pushConstants(CommandBuffer& commandBuffer, const PushConstants& constants) const;

	// binds the descriptors and calls pushConstants
	void operator()(CommandBuffer& commandBuffer, const vk::Extent3D& dim, const shared_ptr<DescriptorSets>& descriptors, const unordered_map<string, uint32_t>& dynamicOffsets = {}, const PushConstants& constants = {});
	void operator()(CommandBuffer& commandBuffer, const vk::Extent3D& dim, const Descriptors& descriptors = {}, const unordered_map<string, uint32_t>& dynamicOffsets = {}, const PushConstants& constants = {});

private:
	shared_ptr<ComputePipeline> mPipeline;
	ResourcePool<DescriptorSets> mResources;
};

}