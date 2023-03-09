#include "Pipeline.hpp"
#include "CommandBuffer.hpp"

#include <map>

namespace stm2 {

DescriptorSets::DescriptorSets(Pipeline& pipeline, const string& name) :
	Device::Resource(pipeline.mDevice, name), mPipeline(pipeline), mDescriptorSetLayouts(mPipeline.descriptorSetLayouts()) {
	// get descriptor set layouts
	vector<vk::DescriptorSetLayout> layouts(mDescriptorSetLayouts.size());
	ranges::transform(mDescriptorSetLayouts, layouts.begin(), [](auto& l){ return **l; });

	// allocate descriptor sets
	vk::raii::DescriptorSets sets = nullptr;
	try{
		mDescriptorPool = mDevice.getDescriptorPool();
		sets = vk::raii::DescriptorSets(*mDevice, vk::DescriptorSetAllocateInfo(**mDescriptorPool, layouts));
	} catch(vk::OutOfPoolMemoryError e) {
		mDescriptorPool = mDevice.allocateDescriptorPool();
		sets = vk::raii::DescriptorSets(*mDevice, vk::DescriptorSetAllocateInfo(**mDescriptorPool, layouts));
	}

	mDescriptorSets.resize(sets.size());
	for (uint32_t i = 0; i < sets.size(); i++) {
		mDescriptorSets[i] = make_shared<vk::raii::DescriptorSet>(move(sets[i]));
		mDevice.setDebugName(**mDescriptorSets[i], resourceName() + "[" + to_string(i) + "]");
	}
}

void DescriptorSets::write(const Descriptors& descriptors) {
	union DescriptorInfo {
		vk::DescriptorBufferInfo buffer;
		vk::DescriptorImageInfo image;
		vk::WriteDescriptorSetAccelerationStructureKHR accelerationStructure;
	};

	vector<DescriptorInfo> descriptorInfos;
	vector<vk::WriteDescriptorSet> writes;
	descriptorInfos.reserve(descriptors.size());
	writes.reserve(descriptors.size());

	for (const auto& [id, descriptorValue] : descriptors) {
		const auto& [name, arrayIndex] = id;
		auto it = mPipeline.descriptorMap().find(name);
		if (it == mPipeline.descriptorMap().end()) {
			cerr << "Warning: Descriptor " << name << " does not exist in pipeline " << mPipeline.resourceName() << endl;
			continue;
		}
		const Shader::DescriptorBinding& binding = it->second;

		// skip if descriptor already written
		if (auto dit = mDescriptors.find(id); dit != mDescriptors.end() && dit->second == descriptorValue)
			continue;

		// write descriptor

		vk::WriteDescriptorSet& w = writes.emplace_back(vk::WriteDescriptorSet(**mDescriptorSets[binding.mSet], binding.mBinding, arrayIndex, 1, binding.mDescriptorType));
		DescriptorInfo& info = descriptorInfos.emplace_back(DescriptorInfo{});
		switch (descriptorValue.index()) {
		case 0: {
			const BufferDescriptor& v = get<BufferDescriptor>(descriptorValue);

			switch (binding.mDescriptorType) {
			case vk::DescriptorType::eUniformBuffer:
			case vk::DescriptorType::eStorageBuffer:
			case vk::DescriptorType::eInlineUniformBlock:
			case vk::DescriptorType::eUniformBufferDynamic:
			case vk::DescriptorType::eStorageBufferDynamic:
			case vk::DescriptorType::eUniformTexelBuffer:
			case vk::DescriptorType::eStorageTexelBuffer:
				if (!v) cerr << "Warning: Writing null buffer at " << id.first << " array index " << id.second << endl;
				break;

			default:
				cerr << "Warning: Invalid descriptor type " << to_string(binding.mDescriptorType) << " while writing " << id.first << " array index " << id.second << endl;
				break;
			}

			info.buffer = vk::DescriptorBufferInfo(**v.buffer(), v.offset(), v.sizeBytes());
			w.setBufferInfo(info.buffer);
			break;
		}
		case 1: {
			const auto& [image, layout, accessFlags, sampler] = get<ImageDescriptor>(descriptorValue);

			switch (binding.mDescriptorType) {
			case vk::DescriptorType::eSampler:
				if (!sampler) cerr << "Warning: Writing null sampler at " << id.first << " array index " << id.second << endl;
				break;
			case vk::DescriptorType::eCombinedImageSampler:
				if (!sampler) cerr << "Warning: Writing null sampler at " << id.first << " array index " << id.second << endl;
			case vk::DescriptorType::eSampledImage:
			case vk::DescriptorType::eStorageImage:
				if (!image) cerr << "Warning: Writing null image at " << id.first << " array index " << id.second << endl;
				break;
			default:
				cerr << "Warning: Invalid descriptor type " << to_string(binding.mDescriptorType) << " while writing " << id.first << " array index " << id.second << endl;
				break;
			}

			info.image = vk::DescriptorImageInfo(sampler ? **sampler : nullptr, *image, layout);
			w.setImageInfo(info.image);
			break;
		}
		case 2: {
			const auto& v = get<shared_ptr<vk::raii::AccelerationStructureKHR>>(descriptorValue);
			if (!v) cerr << "Warning: Writing null acceleration structure at " << id.first << " array index " << id.second << endl;
			if (binding.mDescriptorType != vk::DescriptorType::eAccelerationStructureKHR)
				cerr << "Warning: Invalid descriptor type " << to_string(binding.mDescriptorType) << " while writing " << id.first << " array index " << id.second << endl;
			info.accelerationStructure = vk::WriteDescriptorSetAccelerationStructureKHR(**v);
			w.descriptorCount = info.accelerationStructure.accelerationStructureCount;
			w.pNext = &info;
			break;
		}
		}
	}

	mDescriptors = descriptors;

	if (!writes.empty())
		mDevice->updateDescriptorSets(writes, {});
}

void DescriptorSets::transitionImages(CommandBuffer& commandBuffer) {
	const bool compute = mPipeline.shaderStage(vk::ShaderStageFlagBits::eCompute) != nullptr;
	// transition image descriptors to required layout
	for (const auto& [id, descriptorValue] : mDescriptors) {
		if (descriptorValue.index() == 1) {
			const auto& [image, layout, accessFlags, sampler] = get<ImageDescriptor>(descriptorValue);
			image.barrier(commandBuffer, layout, compute ? vk::PipelineStageFlagBits::eComputeShader : vk::PipelineStageFlagBits::eFragmentShader, accessFlags, commandBuffer.queueFamily());
		}
	}
}

void DescriptorSets::bind(CommandBuffer& commandBuffer, const unordered_map<string, uint32_t>& dynamicOffsets) {
	const bool compute = mPipeline.shaderStage(vk::ShaderStageFlagBits::eCompute) != nullptr;

	if (compute)
		transitionImages(commandBuffer);

	#ifdef _DEBUG
	/*
	for (const auto&[name, value] : mPipeline.descriptorMap()) {
		if (mDescriptors.find({name,0}) == mDescriptors.end() && mPipeline.metadata().mImmutableSamplers.find(name) == mPipeline.metadata().mImmutableSamplers.end()) {
			if (auto bf = mPipeline.metadata().mBindingFlags.find(name); bf != mPipeline.metadata().mBindingFlags.end())
				if (bf->second & vk::DescriptorBindingFlagBits::ePartiallyBound)
					continue;
			cerr << "Warning: Binding DescriptorSets with missing descriptor: " << name << endl;
		}
	}*/
	#endif

	// dynamic offsets
	vector<uint32_t> dynamicOffsetValues(dynamicOffsets.size());
	{
		vector<pair<uint32_t/*set+binding*/, uint32_t/*offset*/>> dynamicOffsetPairs;
		dynamicOffsetPairs.resize(dynamicOffsets.size());
		for (const auto& [name, offset] : dynamicOffsets) {
			auto it = mPipeline.descriptorMap().find(name);
			if (it == mPipeline.descriptorMap().end()) {
				const string msg = "Descriptor " + name + " does not exist in pipeline " + mPipeline.resourceName();
				cerr << "Error: " << msg << endl;
				throw runtime_error(msg);
			}
			dynamicOffsetPairs.emplace_back((uint32_t(it->second.mSet) << 16) | it->second.mBinding, offset);
		}

		ranges::sort(dynamicOffsetPairs);

		ranges::transform(dynamicOffsetPairs, dynamicOffsetValues.begin(), &pair<uint32_t, uint32_t>::second);
	}

	// bind descriptor sets
	vector<vk::DescriptorSet> descriptorSets(mDescriptorSets.size());
	ranges::transform(mDescriptorSets, descriptorSets.begin(), [](const auto& ds) { return **ds; });
	commandBuffer->bindDescriptorSets(compute ? vk::PipelineBindPoint::eCompute : vk::PipelineBindPoint::eGraphics, **mPipeline.layout(), 0, descriptorSets, dynamicOffsetValues);
}


Pipeline::Pipeline(Device& device, const string& name, const ShaderStageMap& shaders, const Metadata& metadata, const vector<std::shared_ptr<vk::raii::DescriptorSetLayout>>& descriptorSetLayouts) :
	Device::Resource(device, name), mShaders(shaders), mPipeline(nullptr), mMetadata(metadata), mDescriptorSetLayouts(descriptorSetLayouts) {

	// gather descriptorset bindings

	vector<map<uint32_t, tuple<vk::DescriptorSetLayoutBinding, optional<vk::DescriptorBindingFlags>, vector<vk::Sampler>>>> bindings;

	for (const auto&[stage, shader] : mShaders) {
		for (const auto&[id, binding] : shader->descriptorMap()) {
			mDescriptorMap[id] = binding;

			// compute total array size
			uint32_t descriptorCount = 1;
			for (const uint32_t v : binding.mArraySize)
				descriptorCount *= v;

			// get binding flags
			optional<vk::DescriptorBindingFlags> flags;
			if (auto b_it = mMetadata.mBindingFlags.find(id); b_it != mMetadata.mBindingFlags.end())
				flags = b_it->second;

			// get immutable samplers
			vector<vk::Sampler> samplers;
			if (auto s_it = mMetadata.mImmutableSamplers.find(id); s_it != mMetadata.mImmutableSamplers.end()) {
				samplers.resize(s_it->second.size());
				ranges::transform(s_it->second, samplers.begin(), [](const shared_ptr<vk::raii::Sampler>& s){ return **s; });
			}

			// increase set count if needed
			if (binding.mSet >= bindings.size())
				bindings.resize(binding.mSet + 1);

			// copy bindings

			auto& setBindings = bindings[binding.mSet];

			auto it = setBindings.find(binding.mBinding);
			if (it == setBindings.end())
				it = setBindings.emplace(binding.mBinding,
					tuple{
						vk::DescriptorSetLayoutBinding(
							binding.mBinding,
							binding.mDescriptorType,
							descriptorCount,
							shader->stage(), {}),
						flags,
						samplers }).first;
			else {
				auto&[setLayoutBinding, flags, samplers] = it->second;

				if (setLayoutBinding.descriptorType != binding.mDescriptorType)
					throw logic_error("Shader modules contain descriptors of different types at the same binding");
				if (setLayoutBinding.descriptorCount != descriptorCount)
					throw logic_error("Shader modules contain descriptors with different counts at the same binding");

				setLayoutBinding.stageFlags |= shader->stage();
			}
		}
	}

	// create DescriptorSetLayouts

	mDescriptorSetLayouts.resize(bindings.size());
	for (uint32_t i = 0; i < bindings.size(); i++) {
		if (mDescriptorSetLayouts[i]) continue;
		vector<vk::DescriptorSetLayoutBinding> layoutBindings;
		vector<vk::DescriptorBindingFlags> bindingFlags;
		bool hasFlags = false;
		for (const auto&[bindingIndex, binding_] : bindings[i]) {
			const auto&[binding, flag, samplers] = binding_;
			if (flag) hasFlags = true;
			bindingFlags.emplace_back(flag ? *flag : vk::DescriptorBindingFlags{});

			auto& b = layoutBindings.emplace_back(binding);
			if (samplers.size())
				b.setImmutableSamplers(samplers);
		}

		vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo(bindingFlags);
		mDescriptorSetLayouts[i] = make_shared<vk::raii::DescriptorSetLayout>(*mDevice, vk::DescriptorSetLayoutCreateInfo(mMetadata.mDescriptorSetLayoutFlags, layoutBindings, hasFlags ? &bindingFlagsInfo : nullptr));
		mDevice.setDebugName(**mDescriptorSetLayouts[i], resourceName() + "/DescriptorSetLayout[" + to_string(i) + "]");
	}


    // populate pushConstantRanges

	vector<vk::PushConstantRange> pushConstantRanges;

	// determine push constant range
	uint32_t rangeBegin = numeric_limits<uint32_t>::max();
	uint32_t rangeEnd = 0;
	vk::ShaderStageFlags stages = vk::ShaderStageFlags{0};
	for (const auto&[stage, shader] : mShaders) {
		if (shader->pushConstants().empty())
			continue;
		stages |= stage;
		for (const auto& [id, p] : shader->pushConstants()) {
			rangeBegin = std::min(rangeBegin, p.mOffset);
			rangeEnd   = std::max(rangeEnd  , p.mOffset + p.mTypeSize);
		}
	}
	if (stages != vk::ShaderStageFlags{0})
		pushConstantRanges.emplace_back(stages, rangeBegin, rangeEnd - rangeBegin);

	// create pipelinelayout from descriptors and pushconstants

	vector<vk::DescriptorSetLayout> vklayouts(mDescriptorSetLayouts.size());
	ranges::transform(mDescriptorSetLayouts, vklayouts.begin(), [](auto ds) { return **ds; });
	mLayout = make_shared<vk::raii::PipelineLayout>(*mDevice, vk::PipelineLayoutCreateInfo(mMetadata.mLayoutFlags, vklayouts, pushConstantRanges));
	mDevice.setDebugName(**mLayout, resourceName() + "/Layout");
}

void Pipeline::pushConstants(CommandBuffer& commandBuffer, const PushConstants& constants) const {
	for (const auto[name, pushConstant] : constants) {
		vk::ShaderStageFlags stages = vk::ShaderStageFlags{0};
		uint32_t offset = 0;

		for (const auto&[stage, shader] : mShaders) {
			if (name == "") {
				stages |= stage;
				offset = 0;
				continue;
			}

			auto it = shader->pushConstants().find(name);
			if (it == shader->pushConstants().end())
				continue;

			if (it->second.mTypeSize != pushConstant.size())
				cerr << "Warning: " << "Push constant " << name << " (" << pushConstant.size() << "B)"
					<< " does not match size declared by shader (" << it->second.mTypeSize << "B) for shader " + shader->resourceName() << endl;

			stages |= stage;
			offset = it->second.mOffset;
		}

		if (stages != vk::ShaderStageFlags{0})
			commandBuffer->pushConstants<byte>(**layout(), stages, offset, pushConstant);
		else
			cerr << "Warning: Push constant " + name + " does not exist in pipeline " + resourceName() << endl;
	}
}

shared_ptr<DescriptorSets> Pipeline::getDescriptorSets(const Descriptors& descriptors) {
	shared_ptr<DescriptorSets> r;
	for (auto ds : mDescriptorSetCache) {
		if (!ds->inFlight()) {
			r = ds;
			break;
		}
	}
	if (!r) {
		r = make_shared<DescriptorSets>(*this, resourceName() + "/Resources");
		mDescriptorSetCache.emplace_back(r);
	}
	r->write(descriptors);
	r->markUsed();
	return r;
}


GraphicsPipeline::GraphicsPipeline(const string& name, const ShaderStageMap& shaders, const GraphicsMetadata& metadata, const vector<std::shared_ptr<vk::raii::DescriptorSetLayout>>& descriptorSetLayouts)
	: Pipeline(shaders.begin()->second->mDevice, name, shaders, metadata, descriptorSetLayouts) {

	// Pipeline constructor creates mLayout, mDescriptorSetLayouts, and mDescriptorMap

	// create pipeline

	vector<vk::PipelineShaderStageCreateInfo> stages;
	for (const auto&[stage, shader] : shaders)
		stages.emplace_back(vk::PipelineShaderStageCreateInfo(mMetadata.mStageLayoutFlags, shader->stage(), ***shader, "main"));

	vk::PipelineColorBlendStateCreateInfo colorBlendState;
	if (metadata.mColorBlendState)
		colorBlendState = vk::PipelineColorBlendStateCreateInfo(
			{},
			metadata.mColorBlendState->mLogicOpEnable,
			metadata.mColorBlendState->mLogicOp,
			metadata.mColorBlendState->mAttachments,
			metadata.mColorBlendState->mBlendConstants);

	vk::PipelineDynamicStateCreateInfo dynamicState({}, metadata.mDynamicStates);

	vk::PipelineRenderingCreateInfo dynamicRenderingState;
	if (metadata.mDynamicRenderingState)
		dynamicRenderingState = vk::PipelineRenderingCreateInfo(
			metadata.mDynamicRenderingState->mViewMask,
			metadata.mDynamicRenderingState->mColorFormats,
			metadata.mDynamicRenderingState->mDepthFormat,
			metadata.mDynamicRenderingState->mStencilFormat);

	vk::PipelineViewportStateCreateInfo viewportState({}, metadata.mViewports, metadata.mScissors);

	mPipeline = vk::raii::Pipeline(*mDevice, mDevice.pipelineCache(), vk::GraphicsPipelineCreateInfo(
		mMetadata.mFlags,
		stages,
		metadata.mVertexInputState.has_value()   ? &metadata.mVertexInputState.value() : nullptr,
		metadata.mInputAssemblyState.has_value() ? &metadata.mInputAssemblyState.value() : nullptr,
		metadata.mTessellationState.has_value()  ? &metadata.mTessellationState.value() : nullptr,
		&viewportState,
		metadata.mRasterizationState.has_value() ? &metadata.mRasterizationState.value() : nullptr,
		metadata.mMultisampleState.has_value()   ? &metadata.mMultisampleState.value() : nullptr,
		metadata.mDepthStencilState.has_value()  ? &metadata.mDepthStencilState.value() : nullptr,
		metadata.mColorBlendState.has_value()    ? &colorBlendState : nullptr,
		&dynamicState,
		**mLayout,
		metadata.mRenderPass,
		metadata.mSubpassIndex, {}, {},
		metadata.mDynamicRenderingState.has_value() ? &dynamicRenderingState : nullptr));
	mDevice.setDebugName(*mPipeline, resourceName());
}

ComputePipeline::ComputePipeline(const string& name, const shared_ptr<Shader>& shader_, const Metadata& metadata, const vector<std::shared_ptr<vk::raii::DescriptorSetLayout>>& descriptorSetLayouts)
	: Pipeline(shader_->mDevice, name, { { shader_->stage(), shader_ } }, metadata, descriptorSetLayouts) {

	// Pipeline constructor creates mLayout, mDescriptorSetLayouts, and mDescriptorMap

	// create pipeline

	mPipeline = vk::raii::Pipeline(*mDevice, mDevice.pipelineCache(), vk::ComputePipelineCreateInfo(
		mMetadata.mFlags,
		vk::PipelineShaderStageCreateInfo(mMetadata.mStageLayoutFlags, vk::ShaderStageFlagBits::eCompute, ***shader_, "main"),
		**mLayout));
	mDevice.setDebugName(*mPipeline, resourceName());
}


void ComputePipeline::dispatch(CommandBuffer& commandBuffer, const vk::Extent3D& dim, const shared_ptr<DescriptorSets>& descriptors, const unordered_map<string, uint32_t>& dynamicOffsets, const PushConstants& constants) {
	commandBuffer.trackResource(descriptors);

	commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, *mPipeline);
	descriptors->bind(commandBuffer, dynamicOffsets);
	pushConstants(commandBuffer, constants);
	commandBuffer->dispatch(dim.width, dim.height, dim.depth);
}


shared_ptr<GraphicsPipeline> GraphicsPipelineCache::get(Device& device, const Defines& defines, const vector<shared_ptr<vk::raii::DescriptorSetLayout>>& descriptorSetLayouts, optional<pair<vk::RenderPass, uint32_t>> renderPass) {
	size_t key = 0;
	for (const auto& d : defines)
		key = hashArgs(key, d.first, d.second);
	const size_t defineKey = key;
	for (const auto& l : descriptorSetLayouts)
		key = hashArgs(key, l);
	if (renderPass.has_value())
		key = hashArgs(key, *renderPass);

	if (auto it = mCachedPipelines.find(key); it != mCachedPipelines.end())
		return it->second;

	Pipeline::ShaderStageMap stages;
	if (auto it = mCachedShaders.find(defineKey); it != mCachedShaders.end())
		stages = it->second;
	else {
		for (const auto&[stage, info] : mEntryPointProfiles)
			stages[stage] = make_shared<Shader>(device, info.mSourceFile, info.mEntryPoint, info.mProfile, mCompileArgs, defines);
		mCachedShaders.emplace(defineKey, stages);
	}

	GraphicsPipeline::GraphicsMetadata metadata = mPipelineMetadata;
	if (renderPass.has_value())
		tie(metadata.mRenderPass, metadata.mSubpassIndex) = renderPass.value();

	const shared_ptr<GraphicsPipeline> pipeline = make_shared<GraphicsPipeline>("Graphics pipeline", stages, mPipelineMetadata, descriptorSetLayouts);
	return mCachedPipelines.emplace(key, pipeline).first->second;
}

shared_ptr<ComputePipeline> ComputePipelineCache::get(Device& device, const Defines& defines, const vector<shared_ptr<vk::raii::DescriptorSetLayout>>& descriptorSetLayouts) {
	size_t key = 0;
	for (const auto& d : defines)
		key = hashArgs(key, d.first, d.second);
	for (const auto& l : descriptorSetLayouts)
		key = hashArgs(key, l);

	if (auto it = mCachedPipelines.find(key); it != mCachedPipelines.end())
		return it->second;

	const shared_ptr<Shader> shader = make_shared<Shader>(device, mSourceFile, mEntryPoint, mProfile, mCompileArgs, defines);
	const shared_ptr<ComputePipeline> pipeline = make_shared<ComputePipeline>(mSourceFile.stem().string() + "_" + mEntryPoint, shader, mPipelineMetadata, descriptorSetLayouts);
	return mCachedPipelines.emplace(key, pipeline).first->second;
}

shared_ptr<ComputePipeline> ComputePipelineCache::getAsync(Device& device, const Defines& defines, const vector<shared_ptr<vk::raii::DescriptorSetLayout>>& descriptorSetLayouts) {
	size_t key = 0;
	for (const auto& d : defines)
		key = hashArgs(key, d.first, d.second);
	for (const auto& l : descriptorSetLayouts)
		key = hashArgs(key, l);

	// check if the pipeline is already compiled
	if (auto it = mCachedPipelines.find(key); it != mCachedPipelines.end())
		return it->second;

	// check if the pipeline is currently compiling
	if (auto it = mCompileJobs.find(key); it != mCompileJobs.end()) {
		if (it->second.wait_for(10us) == future_status::ready) {
			// compile job completed
			const shared_ptr<ComputePipeline> pipeline = it->second.get();
			mCompileJobs.erase(it);
			mCachedPipelines.emplace(key, pipeline);
			return pipeline;
		}
		return nullptr; // currently compiling on another thread
	}

	// compile the pipeline asynchronously
	mCompileJobs.emplace(key, move(async(launch::async, [=, &device]() {
		const shared_ptr<Shader> shader = make_shared<Shader>(device, mSourceFile, mEntryPoint, mProfile, mCompileArgs, defines);
		return make_shared<ComputePipeline>(mSourceFile.stem().string() + "_" + mEntryPoint, shader, mPipelineMetadata, descriptorSetLayouts);
	})));

	return nullptr;
}

}