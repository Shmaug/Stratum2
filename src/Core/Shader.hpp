#pragma once

#include <unordered_map>
#include <variant>

#include "Device.hpp"

namespace tinyvkpt {

class ShaderSource {
public:
	inline ShaderSource(const filesystem::path& sourceFile, const string& entryPoint = "main", const string& profile = "sm_6_6", const vector<string>& compileArgs = {})
		: mSourceFile(sourceFile), mEntryPoint(entryPoint), mShaderProfile(profile), mCompileArgs(compileArgs) {
		if (!filesystem::exists(mSourceFile))
			throw runtime_error(mSourceFile.string() + " does not exist");
	}

	inline const filesystem::path& sourceFile() const { return mSourceFile; }
	inline const string& entryPoint() const { return mEntryPoint; }
	inline const string& profile() const { return mShaderProfile; }
	inline const vector<string>& compileArgs() const { return mCompileArgs; }

private:
	filesystem::path mSourceFile;
	string mEntryPoint;
	string mShaderProfile;
	vector<string> mCompileArgs;
};

struct Shader : Device::Resource {
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
		uint32_t mArrayStride;
		vector<uint32_t> mArraySize; // uint32_t for literal, string for specialization constant
	};
	struct Variable {
		uint32_t mLocation;
		vk::Format mFormat;
		string mSemantic;
		uint32_t mSemanticIndex;
	};

	Shader(Device& device, const shared_ptr<ShaderSource>& source, const unordered_map<string, string>& defines = {});
	Shader(Shader&&) = default;
	Shader& operator=(Shader&&) = default;

	inline const shared_ptr<ShaderSource>& source() const { return mSource; }
	inline const vk::ShaderStageFlagBits& stage() const { return mStage; }
	inline const vk::raii::ShaderModule& module() const { return mModule; }
	inline const unordered_map<string, DescriptorBinding>& descriptorMap() const { return mDescriptorMap; }
	inline const unordered_map<string, PushConstant>& pushConstants() const { return mPushConstants; }
	inline const unordered_map<string, Variable>& inputVariables() const { return mInputVariables; }
	inline const unordered_map<string, Variable>& outputVariables() const { return mOutputVariables; }
	inline const vk::Extent3D& workgroupSize() const { return mWorkgroupSize; }

private:
	shared_ptr<ShaderSource> mSource;
	vk::ShaderStageFlagBits mStage;
	vk::raii::ShaderModule mModule;
	unordered_map<string, DescriptorBinding> mDescriptorMap;
	unordered_map<string, PushConstant> mPushConstants;
	unordered_map<string, Variable> mInputVariables;
	unordered_map<string, Variable> mOutputVariables;
	vk::Extent3D mWorkgroupSize;
};

}
