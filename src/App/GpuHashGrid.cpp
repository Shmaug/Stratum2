#include "GpuHashGrid.hpp"

#include <Core/Instance.hpp>

namespace stm2 {

GpuHashGrid::GpuHashGrid(Device& device, const uint32_t elementSize, const uint32_t cellCount, const float cellSize) {
	mElementSize = elementSize;
	mCellCount = cellCount;
	mCellSize = cellSize;

	const filesystem::path shaderPath = *device.mInstance.findArgument("shaderKernelPath");
	mComputeIndicesPipeline = ComputePipelineCache(shaderPath / "hashgrid.slang", "ComputeIndices", "sm_6_6", { "-O3", "-matrix-layout-row-major", "-capability", "spirv_1_5" });
	mSwizzlePipeline        = ComputePipelineCache(shaderPath / "hashgrid.slang", "Swizzle"       , "sm_6_6", { "-O3", "-matrix-layout-row-major", "-capability", "spirv_1_5" });
}

GpuHashGrid::FrameData GpuHashGrid::init(CommandBuffer& commandBuffer, Descriptors& outputDescriptors, const string& name, const Metadata& md, const string& prevName) {
	struct HashGridConstants {
		float mCellPixelRadius;
		float mMinCellSize;
		uint32_t mCellCount;
		uint32_t pad;
		float3 mCameraPosition;
		float mDistanceScale;
	};
	HashGridConstants constants;
	constants.mCellCount = mCellCount;
	constants.mCellPixelRadius = mCellPixelRadius;
	constants.mMinCellSize = mCellSize;
	constants.mCameraPosition = md.mCameraPosition;
	constants.mDistanceScale = tan(constants.mCellPixelRadius * md.mVerticalFoV * max(1.0f / md.mImageExtent[1], md.mImageExtent[1] / (float)(md.mImageExtent[0]*md.mImageExtent[0])));

	if (!prevName.empty()) {
		Buffer::View<byte> emptyBuffer = mResourcePool.getBuffer<byte>(commandBuffer.mDevice, "Empty", 16, vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eUniformBuffer|vk::BufferUsageFlagBits::eTransferDst);
		for (const char* bufferName : {
			"mChecksums",
			"mCellCounters",
			"mOtherCounters",
			"mAppendDataIndices",
			"mAppendData",
			"mIndices",
			"mDataIndices",
			"mConstants" }) {
			const Buffer::View<byte>& prevBuf = mResourcePool.getLastBuffer<byte>(name + "." + bufferName);
			outputDescriptors[{"gRenderParams." + prevName + "." + bufferName, 0 }] = prevBuf ? prevBuf : emptyBuffer;
		}
	}

	const uint32_t bufCount = prevName.empty() ? 0 : 1;

	const unordered_map<string, Buffer::View<byte>> buffers {
		{ "mChecksums"        , mResourcePool.getBuffer<uint32_t>          (commandBuffer.mDevice, name + ".mChecksums",         mCellCount,         vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal, bufCount) },
		{ "mCellCounters"     , mResourcePool.getBuffer<uint32_t>          (commandBuffer.mDevice, name + ".mCellCounters",      mCellCount,         vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal, bufCount) },
		{ "mOtherCounters"    , mResourcePool.getBuffer<uint32_t>          (commandBuffer.mDevice, name + ".mOtherCounters",     4,                  vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal, bufCount) },
		{ "mAppendDataIndices", mResourcePool.getBuffer<uint2>             (commandBuffer.mDevice, name + ".mAppendDataIndices", mSize,              vk::BufferUsageFlagBits::eStorageBuffer,                                       vk::MemoryPropertyFlagBits::eDeviceLocal, bufCount) },
		{ "mAppendData"       , mResourcePool.getBuffer<byte>              (commandBuffer.mDevice, name + ".mAppendData",        mSize*mElementSize, vk::BufferUsageFlagBits::eStorageBuffer,                                       vk::MemoryPropertyFlagBits::eDeviceLocal, bufCount) },
		{ "mDataIndices"      , mResourcePool.getBuffer<uint32_t>          (commandBuffer.mDevice, name + ".mDataIndices",       mSize,              vk::BufferUsageFlagBits::eStorageBuffer,                                       vk::MemoryPropertyFlagBits::eDeviceLocal, bufCount) },
		{ "mIndices"          , mResourcePool.getBuffer<uint32_t>          (commandBuffer.mDevice, name + ".mIndices",           mCellCount,         vk::BufferUsageFlagBits::eStorageBuffer,                                       vk::MemoryPropertyFlagBits::eDeviceLocal, bufCount) },
		{ "mConstants"        , mResourcePool.uploadData<HashGridConstants>(commandBuffer        , name + ".mConstants",         constants,          vk::BufferUsageFlagBits::eUniformBuffer|vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal, bufCount) },
	};

	Descriptors hashGridDescriptors;
	for (const auto&[bufferName, buf] : buffers) {
		hashGridDescriptors[{ "gHashGrid." + bufferName, 0 }] = buf;
		outputDescriptors[{"gRenderParams." + name + "." + bufferName, 0 }] = buf;
	}
	return hashGridDescriptors;
};

void GpuHashGrid::clear(CommandBuffer& commandBuffer, const GpuHashGrid::FrameData& hashGridDescriptors) {
	get<BufferDescriptor>(hashGridDescriptors.at({ "gHashGrid.mChecksums"    , 0 })).cast<uint32_t>().fill(commandBuffer, 0);
	get<BufferDescriptor>(hashGridDescriptors.at({ "gHashGrid.mCellCounters" , 0 })).cast<uint32_t>().fill(commandBuffer, 0);
	get<BufferDescriptor>(hashGridDescriptors.at({ "gHashGrid.mOtherCounters", 0 })).cast<uint32_t>().fill(commandBuffer, 0);

	Buffer::barriers(commandBuffer, {
		get<BufferDescriptor>(hashGridDescriptors.at({"gHashGrid.mChecksums",0})),
		get<BufferDescriptor>(hashGridDescriptors.at({"gHashGrid.mCellCounters",0})),
		get<BufferDescriptor>(hashGridDescriptors.at({"gHashGrid.mOtherCounters",0})) },
		vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
		vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);
};

void GpuHashGrid::build(CommandBuffer& commandBuffer, const GpuHashGrid::FrameData& hashGridDescriptors) {
	Defines defs { { "N", to_string(mElementSize/sizeof(float)) } };
	const auto computeIndicesPipeline = mComputeIndicesPipeline.get(commandBuffer.mDevice, defs);
	const auto swizzlePipeline        = mSwizzlePipeline       .get(commandBuffer.mDevice, defs);

	const auto descriptorSets = computeIndicesPipeline->getDescriptorSets(hashGridDescriptors);

	get<BufferDescriptor>(hashGridDescriptors.at({"gHashGrid.mCellCounters",0})).barrier(
		commandBuffer,
		vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
		vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);

	computeIndicesPipeline->dispatchTiled(commandBuffer, vk::Extent3D{1024, (mCellCount + 1023)/1024, 1}, descriptorSets);

	Buffer::barriers(commandBuffer, {
		get<BufferDescriptor>(hashGridDescriptors.at({"gHashGrid.mOtherCounters",0})),
		get<BufferDescriptor>(hashGridDescriptors.at({"gHashGrid.mIndices",0})),
		get<BufferDescriptor>(hashGridDescriptors.at({"gHashGrid.mAppendDataIndices",0})) },
		vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
		vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);

	swizzlePipeline->dispatchTiled(commandBuffer, vk::Extent3D{1024, (mSize + 1023)/1024, 1}, descriptorSets);

	get<BufferDescriptor>(hashGridDescriptors.at({ "gHashGrid.mDataIndices", 0 })).barrier(
		commandBuffer,
		vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
		vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);
}

}