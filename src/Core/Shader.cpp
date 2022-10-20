#include "Shader.hpp"
#include "Instance.hpp"

#include <iostream>

#include <slang/slang.h>

namespace tinyvkpt {

vector<uint32_t> Shader::compile_reflect(const unordered_map<string, DefineValue>& defines) {
	SlangSession* session = spCreateSession();
	SlangCompileRequest* request = spCreateCompileRequest(session);

	// process mCompileArgs

	vector<const char*> args;
	for (const string& arg : mCompileArgs) args.emplace_back(arg.c_str());
	if (SLANG_FAILED(spProcessCommandLineArguments(request, args.data(), args.size())))
		cout << "Failed to process compile arguments" << endl;

	// defines

	int targetIndex = spAddCodeGenTarget(request, SLANG_SPIRV);
	spAddPreprocessorDefine(request, "__SLANG__", "");
	spAddPreprocessorDefine(request, "__HLSL__", "");
	vector<string> uintDefs;
	for (const auto&[n,d] : defines)
		if (d.is_number())
			spAddPreprocessorDefine(request, n.c_str(), uintDefs.emplace_back(to_string(d.get_number())).c_str());
		else
			spAddPreprocessorDefine(request, n.c_str(), d.get_string().c_str());

	// include paths

	for (const string& inc : mDevice.mInstance.findArguments("shaderInclude"))
		spAddSearchPath(request, inc.c_str());
	int translationUnitIndex = spAddTranslationUnit(request, SLANG_SOURCE_LANGUAGE_SLANG, "");
	spAddTranslationUnitSourceFile(request, translationUnitIndex, mShaderFile.string().c_str());
	int entryPointIndex = spAddEntryPoint(request, translationUnitIndex, mEntryPoint.c_str(), SLANG_STAGE_NONE);
	spSetTargetProfile(request, targetIndex, spFindProfile(session, mShaderProfile.c_str()));
	spSetTargetFloatingPointMode(request, targetIndex, SLANG_FLOATING_POINT_MODE_FAST);

	// compile

	SlangResult r = spCompile(request);
	cout << spGetDiagnosticOutput(request);
	if (SLANG_FAILED(r)) throw runtime_error(spGetDiagnosticOutput(request));

	slang::ShaderReflection* shaderReflection = slang::ShaderReflection::get(request);

	static const unordered_map<SlangTypeKind, const char*> type_kind_name_map = {
		{ SLANG_TYPE_KIND_NONE, "None" },
		{ SLANG_TYPE_KIND_STRUCT, "Struct" },
		{ SLANG_TYPE_KIND_ARRAY, "Array" },
		{ SLANG_TYPE_KIND_MATRIX, "Matrix" },
		{ SLANG_TYPE_KIND_VECTOR, "Vector" },
		{ SLANG_TYPE_KIND_SCALAR, "Scalar" },
		{ SLANG_TYPE_KIND_CONSTANT_BUFFER, "ConstantBuffer" },
		{ SLANG_TYPE_KIND_RESOURCE, "Resource" },
		{ SLANG_TYPE_KIND_SAMPLER_STATE, "SamplerState" },
		{ SLANG_TYPE_KIND_TEXTURE_BUFFER, "TextureBuffer" },
		{ SLANG_TYPE_KIND_SHADER_STORAGE_BUFFER, "ShaderStorageBuffer" },
		{ SLANG_TYPE_KIND_PARAMETER_BLOCK, "ParameterBlock" },
		{ SLANG_TYPE_KIND_GENERIC_TYPE_PARAMETER, "GenericTypeParameter" },
		{ SLANG_TYPE_KIND_INTERFACE, "Interface" },
		{ SLANG_TYPE_KIND_OUTPUT_STREAM, "OutputStream" },
		{ SLANG_TYPE_KIND_SPECIALIZED, "Specialized" },
		{ SLANG_TYPE_KIND_FEEDBACK, "Feedback" }
	};
	static const unordered_map<SlangBindingType, const char*> binding_type_name_map = {
		{ SLANG_BINDING_TYPE_UNKNOWN, "Unknown" },
		{ SLANG_BINDING_TYPE_SAMPLER, "Sampler" },
		{ SLANG_BINDING_TYPE_TEXTURE, "Texture" },
		{ SLANG_BINDING_TYPE_CONSTANT_BUFFER, "ConstantBuffer" },
		{ SLANG_BINDING_TYPE_PARAMETER_BLOCK, "ParameterBlock" },
		{ SLANG_BINDING_TYPE_TYPED_BUFFER, "TypedBuffer" },
		{ SLANG_BINDING_TYPE_RAW_BUFFER, "RawBuffer" },
		{ SLANG_BINDING_TYPE_COMBINED_TEXTURE_SAMPLER, "CombinedTextureSampler" },
		{ SLANG_BINDING_TYPE_INPUT_RENDER_TARGET, "InputRenderTarget" },
		{ SLANG_BINDING_TYPE_INLINE_UNIFORM_DATA, "InlineUniformData" },
		{ SLANG_BINDING_TYPE_RAY_TRACTING_ACCELERATION_STRUCTURE, "RayTracingAccelerationStructure" },
		{ SLANG_BINDING_TYPE_VARYING_INPUT, "VaryingInput" },
		{ SLANG_BINDING_TYPE_VARYING_OUTPUT, "VaryingOutput" },
		{ SLANG_BINDING_TYPE_EXISTENTIAL_VALUE, "ExistentialValue" },
		{ SLANG_BINDING_TYPE_PUSH_CONSTANT, "PushConstant" },
		{ SLANG_BINDING_TYPE_MUTABLE_FLAG, "MutableFlag" },
		{ SLANG_BINDING_TYPE_MUTABLE_TETURE, "MutableTexture" },
		{ SLANG_BINDING_TYPE_MUTABLE_TYPED_BUFFER, "MutableTypedBuffer" },
		{ SLANG_BINDING_TYPE_MUTABLE_RAW_BUFFER, "MutableRawBuffer" },
		{ SLANG_BINDING_TYPE_BASE_MASK, "BaseMask" },
		{ SLANG_BINDING_TYPE_EXT_MASK, "ExtMask" },
	};
	static const unordered_map<SlangParameterCategory, const char*> category_name_map = {
		{ SLANG_PARAMETER_CATEGORY_NONE, "None" },
		{ SLANG_PARAMETER_CATEGORY_MIXED, "Mixed" },
		{ SLANG_PARAMETER_CATEGORY_CONSTANT_BUFFER, "ConstantBuffer" },
		{ SLANG_PARAMETER_CATEGORY_SHADER_RESOURCE, "ShaderResource" },
		{ SLANG_PARAMETER_CATEGORY_UNORDERED_ACCESS, "UnorderedAccess" },
		{ SLANG_PARAMETER_CATEGORY_VARYING_INPUT, "VaryingInput" },
		{ SLANG_PARAMETER_CATEGORY_VARYING_OUTPUT, "VaryingOutput" },
		{ SLANG_PARAMETER_CATEGORY_SAMPLER_STATE, "SamplerState" },
		{ SLANG_PARAMETER_CATEGORY_UNIFORM, "Uniform" },
		{ SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT, "DescriptorTableSlot" },
		{ SLANG_PARAMETER_CATEGORY_SPECIALIZATION_CONSTANT, "SpecializationConstant" },
		{ SLANG_PARAMETER_CATEGORY_PUSH_CONSTANT_BUFFER, "PushConstantBuffer" },
		{ SLANG_PARAMETER_CATEGORY_REGISTER_SPACE, "RegisterSpace" },
		{ SLANG_PARAMETER_CATEGORY_GENERIC, "GenericResource" },
		{ SLANG_PARAMETER_CATEGORY_RAY_PAYLOAD, "RayPayload" },
		{ SLANG_PARAMETER_CATEGORY_HIT_ATTRIBUTES, "HitAttributes" },
		{ SLANG_PARAMETER_CATEGORY_CALLABLE_PAYLOAD, "CallablePayload" },
		{ SLANG_PARAMETER_CATEGORY_SHADER_RECORD, "ShaderRecord" },
		{ SLANG_PARAMETER_CATEGORY_EXISTENTIAL_TYPE_PARAM, "ExistentialTypeParam" },
		{ SLANG_PARAMETER_CATEGORY_EXISTENTIAL_OBJECT_PARAM, "ExistentialObjectParam" },
	};
	static const unordered_map<SlangScalarType, size_t> scalar_size_map = {
		{ SLANG_SCALAR_TYPE_NONE, 0 },
		{ SLANG_SCALAR_TYPE_VOID, sizeof(uint32_t) },
		{ SLANG_SCALAR_TYPE_BOOL, sizeof(uint32_t) },
		{ SLANG_SCALAR_TYPE_INT32, sizeof(int32_t) },
		{ SLANG_SCALAR_TYPE_UINT32, sizeof(uint32_t) },
		{ SLANG_SCALAR_TYPE_INT64, sizeof(int64_t) },
		{ SLANG_SCALAR_TYPE_UINT64, sizeof(uint64_t) },
		{ SLANG_SCALAR_TYPE_FLOAT16, sizeof(uint16_t) },
		{ SLANG_SCALAR_TYPE_FLOAT32, sizeof(uint32_t) },
		{ SLANG_SCALAR_TYPE_FLOAT64, sizeof(uint64_t) },
		{ SLANG_SCALAR_TYPE_INT8, sizeof(int8_t) },
		{ SLANG_SCALAR_TYPE_UINT8, sizeof(uint8_t) },
		{ SLANG_SCALAR_TYPE_INT16, sizeof(int16_t) },
		{ SLANG_SCALAR_TYPE_UINT16, sizeof(uint16_t) }
	};
	static const unordered_map<SlangBindingType, vk::DescriptorType> descriptor_type_map = {
		{ SLANG_BINDING_TYPE_SAMPLER, vk::DescriptorType::eSampler },
		{ SLANG_BINDING_TYPE_TEXTURE, vk::DescriptorType::eSampledImage},
		{ SLANG_BINDING_TYPE_CONSTANT_BUFFER, vk::DescriptorType::eUniformBuffer },
		{ SLANG_BINDING_TYPE_TYPED_BUFFER, vk::DescriptorType::eUniformTexelBuffer },
		{ SLANG_BINDING_TYPE_RAW_BUFFER, vk::DescriptorType::eStorageBuffer },
		{ SLANG_BINDING_TYPE_COMBINED_TEXTURE_SAMPLER, vk::DescriptorType::eCombinedImageSampler },
		{ SLANG_BINDING_TYPE_INPUT_RENDER_TARGET, vk::DescriptorType::eInputAttachment },
		{ SLANG_BINDING_TYPE_INLINE_UNIFORM_DATA, vk::DescriptorType::eInlineUniformBlock },
		{ SLANG_BINDING_TYPE_RAY_TRACTING_ACCELERATION_STRUCTURE, vk::DescriptorType::eAccelerationStructureKHR },
		{ SLANG_BINDING_TYPE_MUTABLE_TETURE, vk::DescriptorType::eStorageImage },
		{ SLANG_BINDING_TYPE_MUTABLE_TYPED_BUFFER, vk::DescriptorType::eStorageTexelBuffer },
		{ SLANG_BINDING_TYPE_MUTABLE_RAW_BUFFER, vk::DescriptorType::eStorageBuffer },
	};
	static const unordered_map<SlangStage, vk::ShaderStageFlagBits> stage_map = {
		{ SLANG_STAGE_VERTEX, vk::ShaderStageFlagBits::eVertex },
		{ SLANG_STAGE_HULL, vk::ShaderStageFlagBits::eTessellationControl },
		{ SLANG_STAGE_DOMAIN, vk::ShaderStageFlagBits::eTessellationEvaluation },
		{ SLANG_STAGE_GEOMETRY, vk::ShaderStageFlagBits::eGeometry },
		{ SLANG_STAGE_FRAGMENT, vk::ShaderStageFlagBits::eFragment },
		{ SLANG_STAGE_COMPUTE, vk::ShaderStageFlagBits::eCompute },
		{ SLANG_STAGE_RAY_GENERATION, vk::ShaderStageFlagBits::eRaygenKHR },
		{ SLANG_STAGE_INTERSECTION, vk::ShaderStageFlagBits::eIntersectionKHR },
		{ SLANG_STAGE_ANY_HIT, vk::ShaderStageFlagBits::eAnyHitKHR },
		{ SLANG_STAGE_CLOSEST_HIT, vk::ShaderStageFlagBits::eClosestHitKHR },
		{ SLANG_STAGE_MISS, vk::ShaderStageFlagBits::eMissKHR },
		{ SLANG_STAGE_CALLABLE, vk::ShaderStageFlagBits::eCallableKHR },
		{ SLANG_STAGE_MESH, vk::ShaderStageFlagBits::eMeshNV },
	};

	mStage = stage_map.at(shaderReflection->getEntryPointByIndex(0)->getStage());
	if (mStage == vk::ShaderStageFlagBits::eCompute) {
		SlangUInt sz[3];
		shaderReflection->getEntryPointByIndex(0)->getComputeThreadGroupSize(3, &sz[0]);
		mWorkgroupSize.width  = (uint32_t)sz[0];
		mWorkgroupSize.height = (uint32_t)sz[1];
		mWorkgroupSize.depth  = (uint32_t)sz[2];
	}

	function<void(const uint32_t, const string, slang::VariableLayoutReflection*, const uint32_t)> reflectParameter;
	reflectParameter = [&](const uint32_t set_index, const string baseName, slang::VariableLayoutReflection* parameter, const uint32_t offset) {
		slang::TypeReflection* type = parameter->getType();
		slang::TypeLayoutReflection* typeLayout = parameter->getTypeLayout();

		vector<DefineValue> arraySize;
		if (typeLayout->getKind() == slang::TypeReflection::Kind::Array)
			arraySize.emplace_back((uint32_t)typeLayout->getTotalArrayElementCount());

		const string name = baseName + "." + parameter->getName();
		const uint32_t binding_index = offset + parameter->getBindingIndex();

		if (type->getFieldCount() == 0) {
			const vk::DescriptorType descriptorType = descriptor_type_map.at((SlangBindingType)typeLayout->getBindingRangeType(0));
			mDescriptorMap.emplace(name, DescriptorBinding(set_index, binding_index, descriptorType, arraySize, 0));
		} else {
			for (uint32_t i = 0; i < type->getFieldCount(); i++)
				reflectParameter(set_index, name, typeLayout->getFieldByIndex(i), binding_index);
		}
	};

	for (uint32_t parameter_index = 0; parameter_index < shaderReflection->getParameterCount(); parameter_index++) {
		slang::VariableLayoutReflection* parameter = shaderReflection->getParameterByIndex(parameter_index);
		slang::ParameterCategory category = parameter->getCategory();
		slang::TypeReflection* type = parameter->getType();
		slang::TypeLayoutReflection* typeLayout = parameter->getTypeLayout();

		if (category == slang::ParameterCategory::PushConstantBuffer) {
			// TODO: get offset from slang. currently assumes tightly packed scalar push constants
			size_t offset = 0;
			for (uint32_t i = 0; i < type->getElementType()->getFieldCount(); i++) {
				slang::VariableReflection* field = type->getElementType()->getFieldByIndex(i);
				slang::TypeLayoutReflection* fieldTypeLayout = typeLayout->getElementTypeLayout()->getFieldByIndex(i)->getTypeLayout();
				slang::TypeReflection* fieldType = field->getType();
				if (fieldType->getScalarType() == slang::TypeReflection::ScalarType::None) {
					cerr << "Warning: Non-scalar push constants unsupported" << endl;
					continue;
				}
				auto& dst = mPushConstants[field->getName()];
				dst.mOffset = (uint32_t)offset;
				dst.mArrayStride = fieldTypeLayout->getStride();
				if (fieldType->getTotalArrayElementCount())
					dst.mArraySize.emplace_back((uint32_t)fieldType->getTotalArrayElementCount());
				dst.mTypeSize = scalar_size_map.at((SlangScalarType)fieldType->getScalarType());
				offset += max<size_t>(fieldType->getTotalArrayElementCount(), 1) * fieldTypeLayout->getStride();
			}
		} else if (category == slang::ParameterCategory::DescriptorTableSlot) {
			const vk::DescriptorType descriptorType = descriptor_type_map.at((SlangBindingType)typeLayout->getBindingRangeType(0));
			vector<DefineValue> array_size;
			if (typeLayout->getKind() == slang::TypeReflection::Kind::Array)
				array_size.emplace_back((uint32_t)typeLayout->getTotalArrayElementCount());
			mDescriptorMap.emplace(parameter->getName(), DescriptorBinding(parameter->getBindingSpace(), parameter->getBindingIndex(), descriptorType, array_size, 0));
		} else if (category == slang::ParameterCategory::RegisterSpace) {
			for (uint32_t i = 0; i < type->getElementType()->getFieldCount(); i++)
				reflectParameter(parameter->getBindingIndex(), parameter->getName(), typeLayout->getElementTypeLayout()->getFieldByIndex(i), 0);
		} else
			cerr << "Warning: Unsupported resource category: " << category_name_map.at((SlangParameterCategory)category) << endl;
	}

	slang::IBlob* blob;
	r = spGetEntryPointCodeBlob(request, entryPointIndex, targetIndex, &blob);
	if (SLANG_FAILED(r)) throw runtime_error(to_string(r));

	vector<uint32_t> spirv(blob->getBufferSize()/sizeof(uint32_t));
	memcpy(spirv.data(), blob->getBufferPointer(), blob->getBufferSize());

	spDestroyCompileRequest(request);
	spDestroySession(session);

	return spirv;
}

Shader::Shader(Device& device, const filesystem::path& filename, const string& entrypoint, const vector<string>& args, const unordered_map<string, Shader::DefineValue>& defines, const string& profile) :
	Device::Resource(device, filename.stem().string() + "_" + entrypoint), mShaderFile(filename), mEntryPoint(entrypoint), mCompileArgs(args) {

	if (!filesystem::exists(filename)) throw runtime_error(filename.string() + " does not exist");

	// Compile source with slang
	vector<uint32_t> spv = compile_reflect(defines);
	cout << "Compiled " << mShaderFile << ": " << mEntryPoint << " (" << spv.size()*sizeof(uint32_t) << " bytes)" << endl;

	mShaderModules.emplace("", vk::raii::ShaderModule(*device, vk::ShaderModuleCreateInfo({}, spv)));
}

string Shader::getKey(const unordered_map<string, Shader::DefineValue>& defines) {
	string define_args = "";
	for (const auto&[n,d] : defines)
		if (d.is_number())
			define_args += " -D" + n + "=" + to_string(d.get_number());
		else
			define_args += " -D" + n + "=" + d.get_string();
	return define_args;
}

const vk::raii::ShaderModule& Shader::get(const unordered_map<string, Shader::DefineValue>& defines) {
	if (defines.empty())
		return mShaderModules.at("");

	const string key = getKey(defines);
	if (auto it = mShaderModules.find(key); it != mShaderModules.end())
		return it->second;

	const filesystem::path spvpath = (filesystem::temp_directory_path().string() / mShaderFile.stem()).string() + "_" + mEntryPoint + ".spv";

	string compile_cmd = "..\\..\\extern\\slang\\bin\\windows-x64\\release\\slangc.exe " + mShaderFile.string() + " -entry " + mEntryPoint;
	compile_cmd += " -profile " + mShaderProfile + " -lang slang -target spirv -o " + spvpath.string();
	for (const string& arg : mCompileArgs)
		compile_cmd += " " + arg;
	compile_cmd += " -D__HLSL__ -D__SLANG__";
	for (const auto&[n,d] : defines)
		if (d.is_number())
			compile_cmd += " -D" + n + "=" + to_string(d.get_number());
		else
			compile_cmd += " -D" + n + "=" + d.get_string();
	if (system(compile_cmd.c_str()) != EXIT_SUCCESS)
		throw runtime_error("Error: Shader compilation failed");
	cout << "Compiled " << mShaderFile << ": " << mEntryPoint << key << endl;

	const vector<uint32_t> spv = readFile<vector<uint32_t>>(spvpath);
	return mShaderModules.emplace(key, vk::raii::ShaderModule(*mDevice, vk::ShaderModuleCreateInfo({}, spv))).first->second;
}

}