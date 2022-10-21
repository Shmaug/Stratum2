#include "PipelineContext.hpp"
#include "CommandBuffer.hpp"

#include <iostream>

namespace tinyvkpt {

void ComputePipelineContext::bindDescriptors(CommandBuffer& commandBuffer, const Descriptors& descriptors, const unordered_map<string, uint32_t>& dynamicOffsets) {
	shared_ptr<DispatchResources> r = mResources.get();
	// create new resources if necessary
	if (!r) {
		r = mResources.emplace(make_shared<DispatchResources>(commandBuffer.mDevice, mPipeline->resourceName() + "/Resources"));

		vector<vk::DescriptorSetLayout> layouts(mPipeline->descriptorSetLayouts().size());
		ranges::transform(mPipeline->descriptorSetLayouts(), layouts.begin(), [](auto& l){ return **l; });

		vk::raii::DescriptorSets sets(*mPipeline->mDevice, vk::DescriptorSetAllocateInfo(*mPipeline->mDevice.descriptorPool(), layouts));

		r->mDescriptorSets.resize(sets.size());
		ranges::transform(sets, r->mDescriptorSets.begin(), [](vk::raii::DescriptorSet& ds) {
			return make_shared<vk::raii::DescriptorSet>(move(ds));
		});

		mResources.emplace(r);
	}

	r->mDescriptors = descriptors;

	// write descriptors

	union DescriptorInfo {
		vk::DescriptorBufferInfo buffer;
		vk::DescriptorImageInfo image;
	};

	vector<DescriptorInfo> descriptorInfos;
	vector<vk::WriteDescriptorSet> writes;
	descriptorInfos.reserve(descriptors.size());
	writes.reserve(descriptors.size());

	for (const auto& [id, descriptorValue] : descriptors) {
		const auto& [name, arrayIndex] = id;
		auto it = mPipeline->shader()->descriptorMap().find(name);
		if (it == mPipeline->shader()->descriptorMap().end()) {
			const string msg = "Descriptor " + name + " does not exist in shader " + mPipeline->shader()->resourceName();
			cerr << "Error: " << msg << endl;
			throw runtime_error(msg);
		}
		const Shader::DescriptorBinding& binding = it->second;

		// TODO: warn/error on invalid descriptor
		switch (binding.mDescriptorType) {
			case vk::DescriptorType::eSampler:
				break;
			case vk::DescriptorType::eCombinedImageSampler:
				break;
			case vk::DescriptorType::eSampledImage:
				break;
			case vk::DescriptorType::eStorageImage:
				break;
			case vk::DescriptorType::eUniformTexelBuffer:
				break;
			case vk::DescriptorType::eStorageTexelBuffer:
				break;
			case vk::DescriptorType::eUniformBuffer:
				break;
			case vk::DescriptorType::eStorageBuffer:
				break;
			case vk::DescriptorType::eUniformBufferDynamic:
				break;
			case vk::DescriptorType::eStorageBufferDynamic:
				break;
			case vk::DescriptorType::eInputAttachment:
				break;
			case vk::DescriptorType::eInlineUniformBlock:
				break;
		}

		vk::WriteDescriptorSet& w = writes.emplace_back(vk::WriteDescriptorSet(**r->mDescriptorSets[binding.mSet], binding.mBinding, arrayIndex, 1, binding.mDescriptorType));
		DescriptorInfo& info = descriptorInfos.emplace_back(DescriptorInfo{});
		switch (descriptorValue.index()) {
		case 0: {
			const BufferDescriptor& v = get<BufferDescriptor>(descriptorValue);
			info.buffer = vk::DescriptorBufferInfo(**v.buffer(), v.offset(), v.sizeBytes());
			w.setBufferInfo(info.buffer);
			break;
		}
		case 1: {
			const auto& [image, layout, accessFlags, sampler] = get<ImageDescriptor>(descriptorValue);
			info.image = vk::DescriptorImageInfo(**sampler, *image, layout);
			image.barrier(commandBuffer, layout, vk::PipelineStageFlagBits::eComputeShader, accessFlags, commandBuffer.queueFamily());
			w.setImageInfo(info.image);
			break;
		}
		}
	}

	if (!writes.empty())
		commandBuffer.mDevice->updateDescriptorSets(writes, {});

	// dynamic offsets

	vector<pair<uint32_t/*set+binding*/, uint32_t/*offset*/>> dynamicOffsetPairs;
	dynamicOffsetPairs.resize(dynamicOffsets.size());
	for (const auto& [name, offset] : dynamicOffsets) {
		auto it = mPipeline->shader()->descriptorMap().find(name);
		if (it == mPipeline->shader()->descriptorMap().end()) {
			const string msg = "Descriptor " + name + " does not exist in shader " + mPipeline->shader()->resourceName();
			cerr << "Error: " << msg << endl;
			throw runtime_error(msg);
		}
		dynamicOffsetPairs.emplace_back((uint32_t(it->second.mSet) << 16) | it->second.mBinding, offset);
	}
	ranges::sort(dynamicOffsetPairs);

	vector<uint32_t> dynamicOffsetValues(dynamicOffsetPairs.size());
	ranges::transform(dynamicOffsetPairs, dynamicOffsetValues.begin(), &pair<uint32_t, uint32_t>::second);

	// bind descriptorsets

	vector<vk::DescriptorSet> descriptorSets(r->mDescriptorSets.size());
	ranges::transform(r->mDescriptorSets, descriptorSets.begin(), [](const auto& ds) { return **ds; });
	commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, **mPipeline->layout(), 0, descriptorSets, dynamicOffsetValues);
	commandBuffer.trackResource(r);
}

void ComputePipelineContext::pushConstants(CommandBuffer& commandBuffer, const PushConstants& constants) const {
	for (const auto[name, pushConstant] : constants) {
		auto it = mPipeline->shader()->pushConstants().find(name);
		if (it == mPipeline->shader()->pushConstants().end()) {
			const string msg = "Push constant " + name + " does not exist in shader " + mPipeline->shader()->resourceName();
			cerr << "Error: " << msg << endl;
			throw runtime_error(msg);
		}

		uint32_t typeSize = it->second.mTypeSize;
		for (const uint32_t c : it->second.mArraySize)
			typeSize *= c;
		if (typeSize != constants.size())
			cerr << "Warning: " << "Push constant " << name << " with size " << constants.size() << "B"
				 << " does not match size " << typeSize << "B declared by shader " + mPipeline->shader()->resourceName() << endl;

		commandBuffer->pushConstants<byte>(**mPipeline->layout(), mPipeline->shader()->stage(), it->second.mOffset, pushConstant.data());
	}
}

void ComputePipelineContext::operator()(CommandBuffer& commandBuffer, const vk::Extent3D& dim, const Descriptors& descriptors, const unordered_map<string, uint32_t>& dynamicOffsets, const PushConstants& constants) {
	commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, ***mPipeline);
	commandBuffer.trackResource(mPipeline);

	bindDescriptors(commandBuffer, descriptors);
	pushConstants(commandBuffer, constants);
	commandBuffer->dispatch(dim.width, dim.height, dim.depth);
}


}