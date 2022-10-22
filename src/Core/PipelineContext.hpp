#pragma once

#include <variant>

#include "Buffer.hpp"
#include "Image.hpp"
#include "Pipeline.hpp"
#include "ResourcePool.hpp"

namespace tinyvkpt {

using BufferDescriptor = Buffer::View<byte>;
using ImageDescriptor = tuple<Image::View, vk::ImageLayout, vk::AccessFlagBits, shared_ptr<vk::raii::Sampler>>;
using DescriptorValue = variant<BufferDescriptor, ImageDescriptor>;
using Descriptors = unordered_map<pair<string, uint32_t>, DescriptorValue>;

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

	void bindDescriptors(CommandBuffer& commandBuffer, const Descriptors& descriptors = {}, const unordered_map<string, uint32_t>& dynamicOffsets = {});
	void pushConstants(CommandBuffer& commandBuffer, const PushConstants& constants) const;
	inline size_t resourceCount() const { return mResources.size(); }

	void operator()(CommandBuffer& commandBuffer, const vk::Extent3D& dim, const Descriptors& descriptors = {}, const unordered_map<string, uint32_t>& dynamicOffsets = {}, const PushConstants& constants = {});
	inline void operator()(CommandBuffer& commandBuffer, const vk::Extent3D& dim, const Descriptors& descriptors = {}, const PushConstants& constants = {}) {
		operator()(commandBuffer, dim, descriptors, {}, constants);
	}

private:
	class DispatchResources : public Device::Resource {
    public:
		vector<shared_ptr<vk::raii::DescriptorSet>> mDescriptorSets;
		Descriptors mDescriptors;

		DispatchResources(Device& device, const string& name) : Device::Resource(device, name) {}
		DispatchResources() = default;
		DispatchResources(const DispatchResources&) = default;
		DispatchResources(DispatchResources&&) = default;
		DispatchResources& operator=(const DispatchResources&) = default;
		DispatchResources& operator=(DispatchResources&&) = default;
	};

	shared_ptr<ComputePipeline> mPipeline;
	ResourcePool<DispatchResources> mResources;
};

}