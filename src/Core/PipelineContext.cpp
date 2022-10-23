#include "PipelineContext.hpp"
#include "CommandBuffer.hpp"

namespace tinyvkpt {

DescriptorSets::DescriptorSets(Device& device, const string& name, const shared_ptr<ComputePipeline>& pipeline, const Descriptors& descriptors) :
	Device::Resource(device, name), mPipeline(pipeline) {
	// get descriptor set layouts
	vector<vk::DescriptorSetLayout> layouts(mPipeline->descriptorSetLayouts().size());
	ranges::transform(mPipeline->descriptorSetLayouts(), layouts.begin(), [](auto& l){ return **l; });

	// allocate descriptor sets
	vk::raii::DescriptorSets sets(*mPipeline->mDevice, vk::DescriptorSetAllocateInfo(*mPipeline->mDevice.descriptorPool(), layouts));

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
	};

	vector<DescriptorInfo> descriptorInfos;
	vector<vk::WriteDescriptorSet> writes;
	descriptorInfos.reserve(descriptors.size());
	writes.reserve(descriptors.size());

	for (const auto& [id, descriptorValue] : descriptors) {
		const auto& [name, arrayIndex] = id;
		auto it = mPipeline->shader()->descriptorMap().find(name);
		if (it == mPipeline->shader()->descriptorMap().end())
			cerr << "Warning: Descriptor " << name << " does not exist in shader " << mPipeline->shader()->resourceName() << endl;
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
			// TODO: write acceleration structure descriptor
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
			auto it = mPipeline->shader()->descriptorMap().find(name);
			if (it == mPipeline->shader()->descriptorMap().end()) {
				const string msg = "Descriptor " + name + " does not exist in shader " + mPipeline->shader()->resourceName();
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
	commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, **mPipeline->layout(), 0, descriptorSets, dynamicOffsetValues);
}


shared_ptr<DescriptorSets> ComputePipelineContext::getDescriptorSets(Device& device, const Descriptors& descriptors) {
	shared_ptr<DescriptorSets> r = mResources.get();
	if (r) {
		r->write(descriptors);
	} else {
		r = make_shared<DescriptorSets>(device, mPipeline->resourceName() + "/Resources", mPipeline, descriptors);
		mResources.emplace(r);
	}
	return r;
}

void ComputePipelineContext::pushConstants(CommandBuffer& commandBuffer, const PushConstants& constants) const {
	for (const auto[name, pushConstant] : constants) {
		auto it = mPipeline->shader()->pushConstants().find(name);
		if (it == mPipeline->shader()->pushConstants().end()) {
			const string msg = "Push constant " + name + " does not exist in shader " + mPipeline->shader()->resourceName();
			cerr << "Error: " << msg << endl;
			throw runtime_error(msg);
		}

		if (it->second.mTypeSize != pushConstant.data().size())
			cerr << "Warning: " << "Push constant " << name << " (" << pushConstant.data().size() << "B)"
				 << " does not match size declared by shader (" << it->second.mTypeSize << "B) for shader " + mPipeline->shader()->resourceName() << endl;

		commandBuffer->pushConstants<byte>(**mPipeline->layout(), mPipeline->shader()->stage(), it->second.mOffset, pushConstant.data());
	}
}

void ComputePipelineContext::operator()(CommandBuffer& commandBuffer, const vk::Extent3D& dim, const shared_ptr<DescriptorSets>& descriptors, const unordered_map<string, uint32_t>& dynamicOffsets, const PushConstants& constants) {
	commandBuffer.trackResource(mPipeline);
	commandBuffer.trackResource(descriptors);

	commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, ***mPipeline);
	descriptors->bind(commandBuffer, dynamicOffsets);
	pushConstants(commandBuffer, constants);
	commandBuffer->dispatch(dim.width, dim.height, dim.depth);
}
void ComputePipelineContext::operator()(CommandBuffer& commandBuffer, const vk::Extent3D& dim, const Descriptors& descriptors, const unordered_map<string, uint32_t>& dynamicOffsets, const PushConstants& constants) {
	operator()(commandBuffer, dim, getDescriptorSets(commandBuffer.mDevice, descriptors), dynamicOffsets, constants);
}


}