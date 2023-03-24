#pragma once

#include <unordered_map>
#include <variant>

#include "Device.hpp"

struct GPUPrinting;

namespace stm2 {

using Defines = unordered_map<string, string>;

class Shader : public Device::Resource {
public:
	struct DescriptorBinding {
		uint32_t mSet;
		uint32_t mBinding;
		vk::DescriptorType mDescriptorType;
		vector<uint32_t> mArraySize;
		uint32_t mInputAttachmentIndex;
	};
	struct PushConstant {
		uint32_t mOffset;
		uint32_t mTypeSize;
	};
	struct Variable {
		uint32_t mLocation;
		vk::Format mFormat;
		string mSemantic;
		uint32_t mSemanticIndex;
	};

	Shader(Device& device, const filesystem::path& sourceFile, const string& entryPoint = "main", const string& profile = "sm_6_7", const vector<string>& compileArgs = {}, const Defines& defines = {});
	Shader(Shader&&) = default;
	Shader& operator=(Shader&&) = default;

	DECLARE_DEREFERENCE_OPERATORS(vk::raii::ShaderModule, mModule)

	inline const vk::ShaderStageFlagBits& stage() const { return mStage; }
	inline const unordered_map<string, DescriptorBinding>& descriptorMap() const { return mDescriptorMap; }
	inline const unordered_map<string, PushConstant>& pushConstants() const { return mPushConstants; }
	inline const unordered_map<string, Variable>& inputVariables() const { return mInputVariables; }
	inline const unordered_map<string, Variable>& outputVariables() const { return mOutputVariables; }
	inline const vk::Extent3D& workgroupSize() const { return mWorkgroupSize; }

	void processGPUPrintCommands(const void* data, const size_t dataSize) const;

private:
	vk::raii::ShaderModule mModule;

	vk::ShaderStageFlagBits mStage;
	unordered_map<string, DescriptorBinding> mDescriptorMap;
	unordered_map<string, PushConstant> mPushConstants;
	unordered_map<string, Variable> mInputVariables;
	unordered_map<string, Variable> mOutputVariables;
	vk::Extent3D mWorkgroupSize;
	shared_ptr<GPUPrinting> mGPUPrinting;
};

}
