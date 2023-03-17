#pragma once

#include <Core/DeviceResourcePool.hpp>
#include <Core/math.hpp>

namespace stm2 {

struct GpuHashGrid {
	uint32_t mSize; // # of elements
	uint32_t mElementSize;
	uint32_t mCellCount;

	float mCellSize = 0.1f;
	float mCellPixelRadius = 0;

	using FrameData = Descriptors;

	ComputePipelineCache mComputeIndicesPipeline, mSwizzlePipeline;
	DeviceResourcePool mResourcePool;

	GpuHashGrid() = default;
	GpuHashGrid(const GpuHashGrid&) = default;
	GpuHashGrid(GpuHashGrid&&) = default;
	GpuHashGrid(Device& device, const uint32_t elementSize, const uint32_t cellCount, const float cellSize);

	GpuHashGrid& operator=(const GpuHashGrid&) = default;
	GpuHashGrid& operator=(GpuHashGrid&&) = default;

	struct Metadata {
		float3 mCameraPosition = float3::Zero();
		float mVerticalFoV = 0;
		uint2 mImageExtent = uint2::Zero();
	};
	FrameData init(CommandBuffer& commandBuffer, Descriptors& outputDescriptors, const string& name, const Metadata& md, const string& prevName = "");
	void clear(CommandBuffer& commandBuffer, const FrameData& hashGridDescriptors);
	void build(CommandBuffer& commandBuffer, const FrameData& hashGridDescriptors);
};

};