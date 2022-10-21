#include "Pipeline.hpp"

namespace tinyvkpt {

ComputePipeline::ComputePipeline(const string& name, const shared_ptr<Shader>& shader, const Metadata& metadata)
	: Device::Resource(shader->mDevice, name), mShader(shader), mPipeline(nullptr), mMetadata(metadata) {

	// populate bindings

	vector<unordered_map<uint32_t, vk::DescriptorSetLayoutBinding>> bindings;
	unordered_map<uint64_t, vector<vk::Sampler>> staticSamplers;

    for (const auto&[id, binding] : shader->descriptorMap()) {
		uint32_t descriptorCount = 1;
		for (const uint32_t v : binding.mArraySize)
			descriptorCount *= v;

		// increase set count if needed
		if (bindings.size() <= binding.mSet)
			bindings.resize(binding.mSet + 1);

		auto& setBindings = bindings[binding.mSet];

		auto it = setBindings.find(binding.mBinding);
		if (it == setBindings.end())
			it = setBindings.emplace(binding.mBinding, vk::DescriptorSetLayoutBinding(
				binding.mBinding,
				binding.mDescriptorType,
				descriptorCount,
				shader->stage(),
				{})).first;
		else {
			if (it->second.descriptorCount != descriptorCount)
				throw logic_error("Shader modules contain descriptors with different counts at the same binding");
			if (it->second.descriptorType != binding.mDescriptorType)
				throw logic_error("Shader modules contain descriptors of different types at the same binding");

			it->second.stageFlags |= shader->stage();
		}

		// immutable samplers

		if (auto samplers_it = mMetadata.mImmutableSamplers.find(id); samplers_it != mMetadata.mImmutableSamplers.end())
			for (auto s : samplers_it->second)
				staticSamplers[(uint64_t(binding.mSet) << 32) | binding.mBinding].emplace_back(**s);
    }

	for (auto[b, samplers] : staticSamplers)
		for (const auto& s : samplers)
			bindings[b >> 32].at(b&0xFFFFFFFFu).setImmutableSamplers(s);

    // populate pushConstantRanges

	vector<vk::PushConstantRange> pushConstantRanges;

    if (!shader->pushConstants().empty()) {
		// determine push constant range
		uint32_t rangeBegin = numeric_limits<uint32_t>::max();
		uint32_t rangeEnd = 0;
		for (const auto& [id, p] : shader->pushConstants()) {
			uint32_t sz = p.mTypeSize;
			if (!p.mArraySize.empty()) {
				sz = p.mArrayStride;
				for (const uint32_t v : p.mArraySize)
				sz *= v;
			}
			rangeBegin = std::min(rangeBegin, p.mOffset);
			rangeEnd   = std::max(rangeEnd  , p.mOffset + sz);
		}
		pushConstantRanges.emplace_back(shader->stage(), rangeBegin, rangeEnd - rangeBegin);
    }

	// create DescriptorSetLayouts
	mDescriptorSetLayouts.resize(bindings.size());
	for (uint32_t i = 0; i < bindings.size(); i++) {
		vector<vk::DescriptorSetLayoutBinding> layoutBindings(bindings[i].size());
		for (const auto&[setIndex, binding] : bindings[i])
			layoutBindings[binding.binding] = binding;
		mDescriptorSetLayouts[i] = make_shared<vk::raii::DescriptorSetLayout>(*mDevice, vk::DescriptorSetLayoutCreateInfo(mMetadata.mDescriptorSetLayoutFlags, layoutBindings));
	}

	vector<vk::DescriptorSetLayout> vklayouts(mDescriptorSetLayouts.size());
	ranges::transform(mDescriptorSetLayouts, vklayouts.begin(), [](auto ds) { return **ds; });
	mLayout = make_shared<vk::raii::PipelineLayout>(*mDevice, vk::PipelineLayoutCreateInfo(mMetadata.mLayoutFlags, vklayouts, pushConstantRanges));

	mPipeline = vk::raii::Pipeline(*shader->mDevice, shader->mDevice.pipelineCache(), vk::ComputePipelineCreateInfo(
		mMetadata.mFlags,
		vk::PipelineShaderStageCreateInfo(mMetadata.mStageLayoutFlags, vk::ShaderStageFlagBits::eCompute, *shader->module(), shader->source()->entryPoint().c_str()),
		**mLayout));
}

}