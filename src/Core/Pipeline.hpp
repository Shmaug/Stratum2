#pragma once

#include "Shader.hpp"
#include "Buffer.hpp"
#include "Image.hpp"
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

	inline void clear() { mData.clear(); }
	inline const vector<byte>& data() const { return mData; }

	template<typename T>
	inline T& get() {
		if (mData.empty())
			mData.resize(sizeof(T));
		return *reinterpret_cast<T*>(mData.data());
	}

	template<typename T>
	inline T& operator=(const T& value) {
		mData.resize(sizeof(value));
		return *reinterpret_cast<T*>(mData.data()) = value;
	}

private:
	vector<byte> mData;
};

using PushConstants = unordered_map<string, PushConstantValue>;

using BufferDescriptor = Buffer::View<byte>;
using ImageDescriptor = tuple<Image::View, vk::ImageLayout, vk::AccessFlags, shared_ptr<vk::raii::Sampler>>;
using DescriptorValue = variant<BufferDescriptor, ImageDescriptor, shared_ptr<vk::raii::AccelerationStructureKHR>>;
using Descriptors = unordered_map<pair<string, uint32_t>, DescriptorValue>;

class DescriptorSets : public Device::Resource {
public:
	ComputePipeline& mPipeline;

	DescriptorSets(ComputePipeline& pipeline, const string& name, const Descriptors& descriptors = {});
	DescriptorSets() = default;
	DescriptorSets(const DescriptorSets&) = default;
	DescriptorSets(DescriptorSets&&) = default;
	DescriptorSets& operator=(const DescriptorSets&) = default;
	DescriptorSets& operator=(DescriptorSets&&) = default;

	void write(const Descriptors& descriptors = {});

	void bind(CommandBuffer& commandBuffer, const unordered_map<string, uint32_t>& dynamicOffsets = {});

private:
	vector<shared_ptr<vk::raii::DescriptorSet>> mDescriptorSets;
	Descriptors mDescriptors;
};

class ComputePipeline : public Device::Resource {
public:
	struct Metadata {
		vk::PipelineShaderStageCreateFlags mStageLayoutFlags;
		vk::PipelineLayoutCreateFlags mLayoutFlags;
		vk::PipelineCreateFlags mFlags;
		vk::DescriptorSetLayoutCreateFlags mDescriptorSetLayoutFlags;
		unordered_map<string, vector<shared_ptr<vk::raii::Sampler>>> mImmutableSamplers;
		unordered_map<string, vk::DescriptorBindingFlags> mBindingFlags;
	};

	ComputePipeline(const string& name, const shared_ptr<Shader>& shader, const Metadata& metadata = {});

	DECLARE_DEREFERENCE_OPERATORS(vk::raii::Pipeline, mPipeline)

	inline const shared_ptr<Shader>& shader() const { return mShader; }
	inline const shared_ptr<vk::raii::PipelineLayout>& layout() const { return mLayout; }
	inline const vector<shared_ptr<vk::raii::DescriptorSetLayout>>& descriptorSetLayouts() const { return mDescriptorSetLayouts; }

	void pushConstants(CommandBuffer& commandBuffer, const PushConstants& constants) const;


	inline vk::Extent3D dispatchDimensions(const vk::Extent3D& extent) {
		const vk::Extent3D& s = shader()->workgroupSize();
		return {
			(extent.width  + s.width - 1)  / s.width,
			(extent.height + s.height - 1) / s.height,
			(extent.depth  + s.depth - 1)  / s.depth };
	}
	inline size_t resourceCount() const { return mResources.size(); }

	// lazily creates and caches DescriptorSets
	[[nodiscard]] shared_ptr<DescriptorSets> getDescriptorSets(const Descriptors& descriptors = {});

	void dispatch(CommandBuffer& commandBuffer, const vk::Extent3D& dim, const shared_ptr<DescriptorSets>& descriptors, const unordered_map<string, uint32_t>& dynamicOffsets = {}, const PushConstants& constants = {});
	void dispatch(CommandBuffer& commandBuffer, const vk::Extent3D& dim, const Descriptors& descriptors = {}, const unordered_map<string, uint32_t>& dynamicOffsets = {}, const PushConstants& constants = {});

	inline void dispatchTiled(CommandBuffer& commandBuffer, const vk::Extent3D& dim, const shared_ptr<DescriptorSets>& descriptors, const unordered_map<string, uint32_t>& dynamicOffsets = {}, const PushConstants& constants = {}) {
		dispatch(commandBuffer, dispatchDimensions(dim), descriptors, dynamicOffsets, constants);
	}
	inline void dispatchTiled(CommandBuffer& commandBuffer, const vk::Extent3D& dim, const Descriptors& descriptors = {}, const unordered_map<string, uint32_t>& dynamicOffsets = {}, const PushConstants& constants = {}) {
		dispatch(commandBuffer, dispatchDimensions(dim), descriptors, dynamicOffsets, constants);
	}

private:
	shared_ptr<Shader> mShader;
	vk::raii::Pipeline mPipeline;
	shared_ptr<vk::raii::PipelineLayout> mLayout;
	vector<shared_ptr<vk::raii::DescriptorSetLayout>> mDescriptorSetLayouts;
	Metadata mMetadata;

	ResourcePool<DescriptorSets> mResources;
};

class ComputePipelineCache {
public:
	ComputePipelineCache(const filesystem::path& sourceFile,
		const string& entryPoint = "main",
		const string& profile = "sm_6_7",
		const vector<string>& compileArgs = {},
		const ComputePipeline::Metadata& pipelineMetadata = {}) :
		mSourceFile(sourceFile),
		mEntryPoint(entryPoint),
		mProfile(profile),
		mCompileArgs(compileArgs),
		mPipelineMetadata(pipelineMetadata) {}
	ComputePipelineCache() = default;
	ComputePipelineCache(const ComputePipelineCache&) = default;
	ComputePipelineCache(ComputePipelineCache&&) = default;
	ComputePipelineCache& operator=(const ComputePipelineCache&) = default;
	ComputePipelineCache& operator=(ComputePipelineCache&&) = default;

	[[nodiscard]] shared_ptr<ComputePipeline> get(Device& device, const Defines& defines = {});

private:
	filesystem::path mSourceFile;
	string mEntryPoint;
	string mProfile;
	vector<string> mCompileArgs;
	ComputePipeline::Metadata mPipelineMetadata;
	unordered_map<size_t, shared_ptr<ComputePipeline>> mPipelines;
};

}