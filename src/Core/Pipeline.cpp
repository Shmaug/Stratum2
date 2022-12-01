#include "Pipeline.hpp"
#include "CommandBuffer.hpp"

#include <map>

namespace stm2 {


DescriptorSets::DescriptorSets(ComputePipeline& pipeline, const string& name) :
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
		if (it == mPipeline.shader()->descriptorMap().end()) {
			cerr << "Warning: Descriptor " << name << " does not exist in shader " << mPipeline.shader()->resourceName() << endl;
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

	// warn if unbound descriptors
	/*
	for (const auto&[id, binding] : mPipeline.shader()->descriptorMap()) {
		if (mPipeline.metadata().mImmutableSamplers.find(id) != mPipeline.metadata().mImmutableSamplers.end())
			continue;
		if (auto flag_it = mPipeline.metadata().mBindingFlags.find(id); flag_it != mPipeline.metadata().mBindingFlags.end())
			if (flag_it->second & vk::DescriptorBindingFlagBits::ePartiallyBound)
				continue;
		uint32_t total = 1;
		for (uint32_t s : binding.mArraySize)
			total *= s;
		for (uint32_t i = 0; i < total; i++)
			if (mDescriptors.find({ id, i }) == mDescriptors.end())
				cout << "Warning: Unbound descriptor " << id << " (" << binding.mSet << "." << binding.mBinding << "[" << i << "]) while binding DescriptorSets " << resourceName() << endl;
	}*/
}



ComputePipeline::ComputePipeline(const string& name, const shared_ptr<Shader>& shader, const Metadata& metadata)
	: Device::Resource(shader->mDevice, name), mShader(shader), mPipeline(nullptr), mMetadata(metadata) {

	// gather descriptorset bindings

	vector<map<uint32_t, tuple<vk::DescriptorSetLayoutBinding, optional<vk::DescriptorBindingFlags>, vector<vk::Sampler>>>> bindings;

    for (const auto&[id, binding] : shader->descriptorMap()) {
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

			if (setLayoutBinding.descriptorCount != descriptorCount)
				throw logic_error("Shader modules contain descriptors with different counts at the same binding");
			if (setLayoutBinding.descriptorType != binding.mDescriptorType)
				throw logic_error("Shader modules contain descriptors of different types at the same binding");

			setLayoutBinding.stageFlags |= shader->stage();
		}
    }

	// create DescriptorSetLayouts

	mDescriptorSetLayouts.resize(bindings.size());
	for (uint32_t i = 0; i < bindings.size(); i++) {
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
				commandBuffer->pushConstants<byte>(**layout(), shader()->stage(), 0, pushConstant);
				continue;
			} else {
				const string msg = "Push constant " + name + " does not exist in shader " + shader()->resourceName();
				cerr << "Error: " << msg << endl;
				throw runtime_error(msg);
			}
		}

		if (it->second.mTypeSize != pushConstant.size())
			cerr << "Warning: " << "Push constant " << name << " (" << pushConstant.size() << "B)"
				 << " does not match size declared by shader (" << it->second.mTypeSize << "B) for shader " + shader()->resourceName() << endl;

		commandBuffer->pushConstants<byte>(**layout(), shader()->stage(), it->second.mOffset, pushConstant);
	}
}


shared_ptr<DescriptorSets> ComputePipeline::getDescriptorSets(const Descriptors& descriptors) {
	shared_ptr<DescriptorSets> r = mResources.get();
	if (!r)
		r = mResources.emplace(make_shared<DescriptorSets>(*this, resourceName() + "/Resources"));
	r->write(descriptors);
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