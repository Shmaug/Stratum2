#pragma once

#include "Shader.hpp"

namespace tinyvkpt {

class ComputePipeline : public Device::Resource {
public:
	struct Metadata {
		vk::PipelineShaderStageCreateFlags mStageLayoutFlags = {};
		vk::PipelineLayoutCreateFlags mLayoutFlags = {};
		vk::PipelineCreateFlags mFlags = {};
		vk::DescriptorSetLayoutCreateFlags mDescriptorSetLayoutFlags = {};
		unordered_map<string, vector<shared_ptr<vk::raii::Sampler>>> mImmutableSamplers;
		unordered_map<string, vk::DescriptorBindingFlags> mBindingFlags;
	};

	ComputePipeline(const string& name, const shared_ptr<Shader>& shader, const Metadata& metadata = {});

	inline vk::raii::Pipeline& operator*() { return mPipeline; }
	inline vk::raii::Pipeline* operator->() { return &mPipeline; }
	inline const vk::raii::Pipeline& operator*() const { return mPipeline; }
	inline const vk::raii::Pipeline* operator->() const { return &mPipeline; }

	inline const shared_ptr<Shader>& shader() const { return mShader; }
	inline const shared_ptr<vk::raii::PipelineLayout>& layout() const { return mLayout; }
	inline const vector<shared_ptr<vk::raii::DescriptorSetLayout>>& descriptorSetLayouts() const { return mDescriptorSetLayouts; }

private:
	shared_ptr<Shader> mShader;
	vk::raii::Pipeline mPipeline;
	shared_ptr<vk::raii::PipelineLayout> mLayout;
	vector<shared_ptr<vk::raii::DescriptorSetLayout>> mDescriptorSetLayouts;
	Metadata mMetadata;
};

}