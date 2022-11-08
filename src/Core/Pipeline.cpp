#include "Pipeline.hpp"
#include "CommandBuffer.hpp"

#include <map>

namespace tinyvkpt {


DescriptorSets::DescriptorSets(ComputePipeline& pipeline, const string& name, const Descriptors& descriptors) :
	Device::Resource(pipeline.mDevice, name), mPipeline(pipeline) {
	// get descriptor set layouts
	vector<vk::DescriptorSetLayout> layouts(mPipeline.descriptorSetLayouts().size());
	ranges::transform(mPipeline.descriptorSetLayouts(), layouts.begin(), [](auto& l){ return **l; });

	// allocate descriptor sets
	vk::raii::DescriptorSets sets(*mPipeline.mDevice, vk::DescriptorSetAllocateInfo(*mPipeline.mDevice.descriptorPool(), layouts));

	mDescriptorSets.resize(sets.size());
	ranges::transform(sets, mDescriptorSets.begin(), [](vk::raii::DescriptorSet& ds) {
		return make_shared<vk::raii::DescriptorSet>(move(ds));
	});

	write(descriptors);
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
		auto it = mPipeline.shader()->descriptorMap().find(name);
		if (it == mPipeline.shader()->descriptorMap().end())
			cerr << "Warning: Descriptor " << name << " does not exist in shader " << mPipeline.shader()->resourceName() << endl;
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
			info.accelerationStructure.setAccelerationStructures(**v);
			w.descriptorCount = info.accelerationStructure.accelerationStructureCount;
			w.pNext = &info;
		}
		}
	}

	mDescriptors = descriptors;

	if (!writes.empty())
		mDevice->updateDescriptorSets(writes, {});
}

void DescriptorSets::bind(CommandBuffer& commandBuffer, const unordered_map<string, uint32_t>& dynamicOffsets) {
	// transition image descriptors to required layout
	for (const auto& [id, descriptorValue] : mDescriptors) {
		if (descriptorValue.index() == 1) {
			const auto& [image, layout, accessFlags, sampler] = get<ImageDescriptor>(descriptorValue);
			image.barrier(commandBuffer, layout, vk::PipelineStageFlagBits::eComputeShader, accessFlags, commandBuffer.queueFamily());
		}
	}

	// dynamic offsets
	vector<uint32_t> dynamicOffsetValues(dynamicOffsets.size());
	{
		vector<pair<uint32_t/*set+binding*/, uint32_t/*offset*/>> dynamicOffsetPairs;
		dynamicOffsetPairs.resize(dynamicOffsets.size());
		for (const auto& [name, offset] : dynamicOffsets) {
			auto it = mPipeline.shader()->descriptorMap().find(name);
			if (it == mPipeline.shader()->descriptorMap().end()) {
				const string msg = "Descriptor " + name + " does not exist in shader " + mPipeline.shader()->resourceName();
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
	commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, **mPipeline.layout(), 0, descriptorSets, dynamicOffsetValues);
}



ComputePipeline::ComputePipeline(const string& name, const shared_ptr<Shader>& shader, const Metadata& metadata)
	: Device::Resource(shader->mDevice, name), mShader(shader), mPipeline(nullptr), mMetadata(metadata) {

	// populate bindings

	vector<map<uint32_t, pair<vk::DescriptorSetLayoutBinding, optional<vk::DescriptorBindingFlags>>>> bindings;
	unordered_map<uint64_t, vector<vk::Sampler>> staticSamplers;

    for (const auto&[id, binding] : shader->descriptorMap()) {
		uint32_t descriptorCount = 1;
		for (const uint32_t v : binding.mArraySize)
			descriptorCount *= v;

		// increase set count if needed
		if (bindings.size() <= binding.mSet)
			bindings.resize(binding.mSet + 1);

		auto& setBindings = bindings[binding.mSet];

		optional<vk::DescriptorBindingFlags> flags;
		if (auto b_it = mMetadata.mBindingFlags.find(id); b_it != mMetadata.mBindingFlags.end())
			flags = b_it->second;

		auto it = setBindings.find(binding.mBinding);
		if (it == setBindings.end())
			it = setBindings.emplace(binding.mBinding, pair{ vk::DescriptorSetLayoutBinding(
				binding.mBinding,
				binding.mDescriptorType,
				descriptorCount,
				shader->stage(), {}),
				flags }).first;
		else {
			auto&[setLayoutBinding, flags] = it->second;
			if (setLayoutBinding.descriptorCount != descriptorCount)
				throw logic_error("Shader modules contain descriptors with different counts at the same binding");
			if (setLayoutBinding.descriptorType != binding.mDescriptorType)
				throw logic_error("Shader modules contain descriptors of different types at the same binding");

			setLayoutBinding.stageFlags |= shader->stage();
		}

		// immutable samplers

		if (auto samplers_it = mMetadata.mImmutableSamplers.find(id); samplers_it != mMetadata.mImmutableSamplers.end())
			for (auto s : samplers_it->second)
				staticSamplers[(uint64_t(binding.mSet) << 32) | binding.mBinding].emplace_back(**s);
    }

	for (auto[b, samplers] : staticSamplers)
		for (const auto& s : samplers)
			bindings[b >> 32].at(b&UINT_MAX).first.setImmutableSamplers(s);

	// create DescriptorSetLayouts
	mDescriptorSetLayouts.resize(bindings.size());
	for (uint32_t i = 0; i < bindings.size(); i++) {
		vector<vk::DescriptorSetLayoutBinding> layoutBindings(bindings[i].size());
		vector<vk::DescriptorBindingFlags> bindingFlags;
		for (const auto&[setIndex, binding] : bindings[i]) {
			const auto&[b, flag] = binding;
			layoutBindings[b.binding] = b;
			if (flag)
				bindingFlags.emplace_back(*flag);
		}
		vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo(bindingFlags);
		mDescriptorSetLayouts[i] = make_shared<vk::raii::DescriptorSetLayout>(*mDevice, vk::DescriptorSetLayoutCreateInfo(mMetadata.mDescriptorSetLayoutFlags, layoutBindings, &bindingFlagsInfo));
	}

    // populate pushConstantRanges

	vector<vk::PushConstantRange> pushConstantRanges;

    if (!shader->pushConstants().empty()) {
		// determine push constant range
		uint32_t rangeBegin = numeric_limits<uint32_t>::max();
		uint32_t rangeEnd = 0;
		for (const auto& [id, p] : shader->pushConstants()) {
			rangeBegin = std::min(rangeBegin, p.mOffset);
			rangeEnd   = std::max(rangeEnd  , p.mOffset + p.mTypeSize);
		}
		pushConstantRanges.emplace_back(shader->stage(), rangeBegin, rangeEnd - rangeBegin);
    }

	// create pipelinelayout from descriptors and pushconstants

	vector<vk::DescriptorSetLayout> vklayouts(mDescriptorSetLayouts.size());
	ranges::transform(mDescriptorSetLayouts, vklayouts.begin(), [](auto ds) { return **ds; });
	mLayout = make_shared<vk::raii::PipelineLayout>(*mDevice, vk::PipelineLayoutCreateInfo(mMetadata.mLayoutFlags, vklayouts, pushConstantRanges));

	// create pipeline

	mPipeline = vk::raii::Pipeline(*shader->mDevice, shader->mDevice.pipelineCache(), vk::ComputePipelineCreateInfo(
		mMetadata.mFlags,
		vk::PipelineShaderStageCreateInfo(mMetadata.mStageLayoutFlags, vk::ShaderStageFlagBits::eCompute, ***shader, "main"),
		**mLayout));
}

void ComputePipeline::pushConstants(CommandBuffer& commandBuffer, const PushConstants& constants) const {
	for (const auto[name, pushConstant] : constants) {
		auto it = shader()->pushConstants().find(name);
		if (it == shader()->pushConstants().end()) {
			if (name == "") {
				commandBuffer->pushConstants<byte>(**layout(), shader()->stage(), 0, pushConstant.data());
				continue;
			} else {
				const string msg = "Push constant " + name + " does not exist in shader " + shader()->resourceName();
				cerr << "Error: " << msg << endl;
				throw runtime_error(msg);
			}
		}

		if (it->second.mTypeSize != pushConstant.data().size())
			cerr << "Warning: " << "Push constant " << name << " (" << pushConstant.data().size() << "B)"
				 << " does not match size declared by shader (" << it->second.mTypeSize << "B) for shader " + shader()->resourceName() << endl;

		commandBuffer->pushConstants<byte>(**layout(), shader()->stage(), it->second.mOffset, pushConstant.data());
	}
}


shared_ptr<DescriptorSets> ComputePipeline::getDescriptorSets(const Descriptors& descriptors) {
	shared_ptr<DescriptorSets> r = mResources.get();
	if (r) {
		r->write(descriptors);
	} else {
		r = make_shared<DescriptorSets>(*this, resourceName() + "/Resources", descriptors);
		mResources.emplace(r);
	}
	return r;
}

void ComputePipeline::dispatch(CommandBuffer& commandBuffer, const vk::Extent3D& dim, const shared_ptr<DescriptorSets>& descriptors, const unordered_map<string, uint32_t>& dynamicOffsets, const PushConstants& constants) {
	commandBuffer.trackResource(descriptors);

	commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, *mPipeline);
	descriptors->bind(commandBuffer, dynamicOffsets);
	pushConstants(commandBuffer, constants);
	commandBuffer->dispatch(dim.width, dim.height, dim.depth);
}
void ComputePipeline::dispatch(CommandBuffer& commandBuffer, const vk::Extent3D& dim, const Descriptors& descriptors, const unordered_map<string, uint32_t>& dynamicOffsets, const PushConstants& constants) {
	dispatch(commandBuffer, dim, getDescriptorSets(descriptors), dynamicOffsets, constants);
}


shared_ptr<ComputePipeline> ComputePipelineCache::get(Device& device, const Defines& defines) {
	size_t key = 0;
	for (auto[d,v] : defines)
		key = hashArgs(key, d, v);

	if (auto it = mPipelines.find(key); it != mPipelines.end())
		return it->second;

	auto pipeline = make_shared<ComputePipeline>(
		mSourceFile.stem().string() + "_" + mEntryPoint,
		make_shared<Shader>(device, mSourceFile, mEntryPoint, mProfile, mCompileArgs, defines),
		mPipelineMetadata );
	return mPipelines.emplace(key, pipeline).first->second;
}


}