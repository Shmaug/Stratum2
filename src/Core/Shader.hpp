#pragma once

#include <unordered_map>
#include <variant>

#include "Device.hpp"

namespace tinyvkpt {

class Shader : public Device::Resource {
public:
	struct DefineValue {
		variant<uint32_t, string> mValue;
		const bool is_number() const { return mValue.index() == 0; }
		const uint32_t get_number() const { return std::get<uint32_t>(mValue); }
		const string get_string() const { return std::get<string>(mValue); }
	};

	struct DescriptorBinding {
		uint32_t mSet;
		uint32_t mBinding;
		vk::DescriptorType mDescriptorType;
		vector<DefineValue> mArraySize;
		uint32_t mInputAttachmentIndex;
	};
	struct PushConstant {
		uint32_t mOffset;
		uint32_t mTypeSize;
		uint32_t mArrayStride;
		vector<DefineValue> mArraySize; // uint32_t for literal, string for specialization constant
	};
	struct Variable {
		uint32_t mLocation;
		vk::Format mFormat;
		string mSemantic;
		uint32_t mSemanticIndex;
	};

	Shader(Device& device, const filesystem::path& filepath, const string& entrypoint, const vector<string>& compile_args = {}, const unordered_map<string, DefineValue>& defines = {}, const string& profile = "sm_6_6");

	// lazily compiles spirv for the given defines
	const vk::raii::ShaderModule& get(const unordered_map<string, DefineValue>& defines = {});

	inline const vk::ShaderStageFlagBits& stage() const { return mStage; }
	inline const string& entryPoint() const { return mEntryPoint; }
	inline const auto& specializationConstants() const { return mSpecializationConstants; }
	inline const auto& descriptors() const { return mDescriptorMap; }
	inline const auto& pushConstants() const { return mPushConstants; }
	inline const auto& inputVariables() const { return mInputVariables; }
	inline const auto& outputVariables() const { return mOutputVariables; }
	inline const vk::Extent3D& workgroupSize() const { return mWorkgroupSize; }

private:
	filesystem::path mShaderFile;
	unordered_map<string, vk::raii::ShaderModule> mShaderModules;
	vk::ShaderStageFlagBits mStage;
	string mEntryPoint;
	string mShaderProfile;
	vector<string> mCompileArgs;

	unordered_map<string, DescriptorBinding> mDescriptorMap;
	unordered_map<string, pair<uint32_t/*id*/,uint32_t/*default value*/>> mSpecializationConstants;
	unordered_map<string, PushConstant> mPushConstants;
	unordered_map<string, Variable> mInputVariables;
	unordered_map<string, Variable> mOutputVariables;
	vk::Extent3D mWorkgroupSize;

	static string getKey(const unordered_map<string, DefineValue>& defines);
	vector<uint32_t> compile_reflect(const unordered_map<string, DefineValue>& defines);
};

}
