#pragma once

#include "Shader.hpp"
#include "DescriptorSets.hpp"

#include <future>
#include <shared_mutex>

namespace stm2 {

class PushConstantValue : public vector<byte> {
public:
	template<typename T>
	inline PushConstantValue(const T& value) {
		resize(sizeof(value));
		*reinterpret_cast<T*>(data()) = value;
	}

	PushConstantValue() = default;
	PushConstantValue(PushConstantValue&&) = default;
	PushConstantValue(const PushConstantValue&) = default;
	PushConstantValue& operator=(PushConstantValue&&) = default;
	PushConstantValue& operator=(const PushConstantValue&) = default;

	template<typename T>
	inline T& get() {
		if (empty())
			resize(sizeof(T));
		return *reinterpret_cast<T*>(data());
	}

	template<typename T>
	inline T& operator=(const T& value) {
		resize(sizeof(value));
		return *reinterpret_cast<T*>(data()) = value;
	}
};

using PushConstants = unordered_map<string, PushConstantValue>;

class Pipeline : public Device::Resource {
public:
	struct Metadata {
		vk::PipelineShaderStageCreateFlags mStageLayoutFlags;
		vk::PipelineLayoutCreateFlags mLayoutFlags;
		vk::PipelineCreateFlags mFlags;
		vk::DescriptorSetLayoutCreateFlags mDescriptorSetLayoutFlags;
		unordered_map<string, vector<shared_ptr<vk::raii::Sampler>>> mImmutableSamplers;
		unordered_map<string, vk::DescriptorBindingFlags> mBindingFlags;
	};

	using ShaderStageMap = unordered_map<vk::ShaderStageFlagBits, shared_ptr<Shader>>;

	Pipeline(Device& device, const string& name, const ShaderStageMap& shaders, const Metadata& metadata = {}, const vector<std::shared_ptr<vk::raii::DescriptorSetLayout>>& descriptorSetLayouts = {});

	DECLARE_DEREFERENCE_OPERATORS(vk::raii::Pipeline, mPipeline)

	inline const shared_ptr<vk::raii::PipelineLayout>& layout() const { return mLayout; }
	inline const vector<shared_ptr<vk::raii::DescriptorSetLayout>>& descriptorSetLayouts() const { return mDescriptorSetLayouts; }
	inline const Metadata& metadata() const { return mMetadata; }
	inline const unordered_map<string, Shader::DescriptorBinding>& descriptorMap() const { return mDescriptorMap; }
	inline shared_ptr<Shader> shaderStage(vk::ShaderStageFlagBits stage) const {
		if (auto it = mShaders.find(stage); it != mShaders.end())
			return it->second;
		return nullptr;
	}

	void pushConstants(CommandBuffer& commandBuffer, const PushConstants& constants) const;

	// creates + caches DescriptorSets
	[[nodiscard]] shared_ptr<DescriptorSets> getDescriptorSets(const Descriptors& descriptors = {});

protected:
	vk::raii::Pipeline mPipeline;
	shared_ptr<vk::raii::PipelineLayout> mLayout;
	vector<shared_ptr<vk::raii::DescriptorSetLayout>> mDescriptorSetLayouts;
	unordered_map<string, Shader::DescriptorBinding> mDescriptorMap;

	Metadata mMetadata;

	ShaderStageMap mShaders;

	list<shared_ptr<DescriptorSets>> mDescriptorSetCache;
	shared_mutex mDescriptorSetMutex;
};

class GraphicsPipeline : public Pipeline {
public:
	struct ColorBlendState {
		vk::PipelineColorBlendStateCreateFlags flags = {};
		bool mLogicOpEnable = false;
		vk::LogicOp mLogicOp = vk::LogicOp::eClear;
		vector<vk::PipelineColorBlendAttachmentState> mAttachments;
		array<float,4> mBlendConstants = { 1, 1, 1, 1 };
	};
	struct DynamicRenderingState {
		uint32_t mViewMask = 0;
		vector<vk::Format> mColorFormats;
		vk::Format mDepthFormat = vk::Format::eUndefined;
		vk::Format mStencilFormat = vk::Format::eUndefined;
	};

	struct GraphicsMetadata : public Pipeline::Metadata {
		optional<vk::PipelineVertexInputStateCreateInfo>   mVertexInputState;
		optional<vk::PipelineInputAssemblyStateCreateInfo> mInputAssemblyState;
		optional<vk::PipelineTessellationStateCreateInfo>  mTessellationState;
		optional<vk::PipelineRasterizationStateCreateInfo> mRasterizationState;
		optional<vk::PipelineMultisampleStateCreateInfo>   mMultisampleState;
		optional<vk::PipelineDepthStencilStateCreateInfo>  mDepthStencilState;
		vector<vk::Viewport> mViewports;
		vector<vk::Rect2D> mScissors;
		optional<ColorBlendState> mColorBlendState;
		vector<vk::DynamicState> mDynamicStates;
		optional<DynamicRenderingState> mDynamicRenderingState;
		vk::RenderPass mRenderPass;
		uint32_t mSubpassIndex;

	};

	GraphicsPipeline(const string& name, const ShaderStageMap& shaders, const GraphicsMetadata& metadata = {}, const vector<std::shared_ptr<vk::raii::DescriptorSetLayout>>& descriptorSetLayouts = {});
};

class ComputePipeline : public Pipeline {
public:
	ComputePipeline(const string& name, const shared_ptr<Shader>& shader, const Metadata& metadata = {}, const vector<std::shared_ptr<vk::raii::DescriptorSetLayout>>& descriptorSetLayouts = {});

	inline const shared_ptr<Shader>& shader() const { return mShaders.at(vk::ShaderStageFlagBits::eCompute); }

	inline vk::Extent3D calculateDispatchDim(const vk::Extent3D& extent) {
		const vk::Extent3D& s = shader()->workgroupSize();
		return {
			(extent.width  + s.width - 1)  / s.width,
			(extent.height + s.height - 1) / s.height,
			(extent.depth  + s.depth - 1)  / s.depth };
	}

	void dispatch(CommandBuffer& commandBuffer, const vk::Extent3D& dim, const shared_ptr<DescriptorSets>& descriptors, const unordered_map<string, uint32_t>& dynamicOffsets = {}, const PushConstants& constants = {});
	inline void dispatch(CommandBuffer& commandBuffer, const vk::Extent3D& dim, const Descriptors& descriptors = {}, const unordered_map<string, uint32_t>& dynamicOffsets = {}, const PushConstants& constants = {}) {
		dispatch(commandBuffer, dim, getDescriptorSets(descriptors), dynamicOffsets, constants);
	}
	inline void dispatchTiled(CommandBuffer& commandBuffer, const vk::Extent3D& dim, const shared_ptr<DescriptorSets>& descriptors, const unordered_map<string, uint32_t>& dynamicOffsets = {}, const PushConstants& constants = {}) {
		dispatch(commandBuffer, calculateDispatchDim(dim), descriptors, dynamicOffsets, constants);
	}
	inline void dispatchTiled(CommandBuffer& commandBuffer, const vk::Extent3D& dim, const Descriptors& descriptors = {}, const unordered_map<string, uint32_t>& dynamicOffsets = {}, const PushConstants& constants = {}) {
		dispatch(commandBuffer, calculateDispatchDim(dim), descriptors, dynamicOffsets, constants);
	}
};


// compiles + caches pipelines
class GraphicsPipelineCache {
public:
	struct ShaderSourceInfo {
		filesystem::path mSourceFile;
		string mEntryPoint;
		string mProfile;
	};

	GraphicsPipelineCache(const unordered_map<vk::ShaderStageFlagBits, ShaderSourceInfo>& sourceFiles,
		const vector<string>& compileArgs = {},
		const GraphicsPipeline::GraphicsMetadata& pipelineMetadata = {}) :
		mEntryPointProfiles(sourceFiles),
		mCompileArgs(compileArgs),
		mPipelineMetadata(pipelineMetadata),
		mMutex(make_shared<shared_mutex>()) {}
	GraphicsPipelineCache() = default;
	GraphicsPipelineCache(const GraphicsPipelineCache&) = default;
	GraphicsPipelineCache(GraphicsPipelineCache&&) = default;
	GraphicsPipelineCache& operator=(const GraphicsPipelineCache&) = default;
	GraphicsPipelineCache& operator=(GraphicsPipelineCache&&) = default;

	inline operator bool() const { return !mEntryPointProfiles.empty(); }

	inline const GraphicsPipeline::GraphicsMetadata& pipelineMetadata() const { return mPipelineMetadata; }

	[[nodiscard]] shared_ptr<GraphicsPipeline> get(Device& device, const Defines& defines = {}, const vector<shared_ptr<vk::raii::DescriptorSetLayout>>& descriptorSetLayouts = {}, optional<pair<vk::RenderPass, uint32_t>> renderPass = {});

	inline void clear() {
		mCachedShaders.clear();
		mCachedPipelines.clear();
	}

private:
	unordered_map<vk::ShaderStageFlagBits, ShaderSourceInfo> mEntryPointProfiles;
	vector<string> mCompileArgs;
	GraphicsPipeline::GraphicsMetadata mPipelineMetadata;

	unordered_map<size_t, Pipeline::ShaderStageMap> mCachedShaders;
	unordered_map<size_t, shared_ptr<GraphicsPipeline>> mCachedPipelines;
	shared_ptr<shared_mutex> mMutex;
};


// compiles + caches pipelines
class ComputePipelineCache {
public:
	ComputePipelineCache(const filesystem::path& sourceFile,
		const string& entryPoint = "main",
		const string& profile = "sm_6_6",
		const vector<string>& compileArgs = {},
		const Pipeline::Metadata& pipelineMetadata = {}) :
		mSourceFile(sourceFile),
		mEntryPoint(entryPoint),
		mProfile(profile),
		mCompileArgs(compileArgs),
		mPipelineMetadata(pipelineMetadata),
		mMutex(make_shared<shared_mutex>()) {}
	ComputePipelineCache() = default;
	ComputePipelineCache(const ComputePipelineCache&) = default;
	ComputePipelineCache(ComputePipelineCache&&) = default;
	ComputePipelineCache& operator=(const ComputePipelineCache&) = default;
	ComputePipelineCache& operator=(ComputePipelineCache&&) = default;

	inline operator bool() const { return !mSourceFile.empty(); }

	inline const Pipeline::Metadata& pipelineMetadata() const { return mPipelineMetadata; }

	[[nodiscard]] shared_ptr<ComputePipeline> get(Device& device, const Defines& defines = {}, const vector<shared_ptr<vk::raii::DescriptorSetLayout>>& descriptorSetLayouts = {});
	[[nodiscard]] shared_ptr<ComputePipeline> getAsync(Device& device, const Defines& defines = {}, const vector<shared_ptr<vk::raii::DescriptorSetLayout>>& descriptorSetLayouts = {});

	inline void clear() {
		mCachedPipelines.clear();
	}

private:
	filesystem::path mSourceFile;
	string mEntryPoint;
	string mProfile;
	vector<string> mCompileArgs;
	Pipeline::Metadata mPipelineMetadata;

	unordered_map<size_t, shared_ptr<ComputePipeline>> mCachedPipelines;
	unordered_map<size_t, future<shared_ptr<ComputePipeline>>> mCompileJobs;
	shared_ptr<shared_mutex> mMutex;
};

}