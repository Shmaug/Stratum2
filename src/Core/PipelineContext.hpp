#pragma once

#include <variant>

#include "Buffer.hpp"
#include "Image.hpp"
#include "Pipeline.hpp"

namespace tinyvkpt {

using BufferDescriptor = Buffer::View<byte>;
using ImageDescriptor = tuple<Image::View, vk::ImageLayout, vk::AccessFlagBits, shared_ptr<vk::raii::Sampler>>;

using DescriptorValue = variant<BufferDescriptor, ImageDescriptor>;
using Descriptors = unordered_map<pair<string, uint32_t>, DescriptorValue>;

class PushConstantValue {
public:
	template<typename T> requires(is_trivially_copyable_v<T>)
	inline PushConstantValue(const T& value) {
		mData.resize(sizeof(value));
		ranges::uninitialized_copy();
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
	inline ComputePipelineContext(const shared_ptr<ComputePipeline>& pipeline) : mPipeline(pipeline) {}

	inline const shared_ptr<ComputePipeline>& pipeline() const { return mPipeline; }

	void bindDescriptors(CommandBuffer& commandBuffer, const Descriptors& descriptors = {}, const unordered_map<string, uint32_t>& dynamicOffsets = {});
	void pushConstants(CommandBuffer& commandBuffer, const PushConstants& constants) const;

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
	unordered_set<shared_ptr<DispatchResources>> mResources;
};

}