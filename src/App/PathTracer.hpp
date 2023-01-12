#pragma once

#include "Scene.hpp"

#include <Shaders/compat/scene.h>
#include <Shaders/compat/path_tracer.h>

namespace stm2 {

class PathTracer {
private:
	class FrameResources : public Device::Resource {
	public:
		inline FrameResources(Device& device) : Device::Resource(device, "PathTracer::FrameResources") {}

		chrono::high_resolution_clock::time_point mTime;

		vector<pair<ViewData,TransformData>> mViews;

		shared_ptr<Scene::FrameResources> mSceneData;
		Buffer::View<VisibilityData> mSelectionData;
		bool mSelectionDataValid;
		bool mSelectionShift;

		shared_ptr<DescriptorSets> mDescriptorSets;

		unordered_map<string, ImageDescriptor> mImages;
		unordered_map<string, Buffer::View<byte>> mBuffers;

		Image::View mRasterDepthBuffer;

		template<typename T>
		inline Buffer::View<T> getBuffer(const string& name, const vk::DeviceSize count, const vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer, const vk::MemoryPropertyFlags memoryProperties = vk::MemoryPropertyFlagBits::eDeviceLocal) {
			if (auto it = mBuffers.find(name); it != mBuffers.end())
				if (it->second.sizeBytes() >= sizeof(T)*count)
					return it->second.cast<T>();

			auto b = make_shared<Buffer>(mDevice, name, sizeof(T)*count, usage, memoryProperties);
			mBuffers[name] = b;
			return b;
		}

		inline Image::View getImage(const string& name, const vk::Extent3D& extent, const vk::Format format, const vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eStorage) {
			if (auto it = mImages.find(name); it != mImages.end()) {
				const auto&[img,layout,access,sampler] = it->second;
				if (img.extent().width >= extent.width && img.extent().height >= extent.height && img.extent().depth >= extent.depth && img.image()->format() == format && (img.image()->usage() & usage))
					return img;
			}

			Image::Metadata md = {};
			md.mExtent = extent;
			md.mFormat = format;
			md.mUsage = usage;
			Image::View img = make_shared<Image>(mDevice, name, md);
			mImages[name] = ImageDescriptor{img, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {}};
			return img;
		}
	};
public:
	Node& mNode;

	PathTracer(Node& node);

	void createPipelines(Device& device);

	void drawGui();
	void render(CommandBuffer& commandBuffer, const Image::View& renderTarget);
	void rasterLightPaths(CommandBuffer& commandBuffer, const Image::View& renderTarget, FrameResources& frame);

	inline Image::View resultImage() const { return mLastResultImage; }

private:
	void createRasterPipeline(Device& device, const vk::Extent2D& extent, const vk::Format format);

	enum RenderPipelineIndex {
		eGenerateLightPaths,
		eGenerateCameraPaths,
		eHashGridComputeIndices,
		eHashGridSwizzle,
		ePipelineCount
	};
	array<ComputePipelineCache, RenderPipelineIndex::ePipelineCount> mRenderPipelines;

	GraphicsPipelineCache mRasterLightPathPipeline;

	PathTracerPushConstants mPushConstants;

	bool mRandomPerFrame = true;
	bool mDenoise = true;
	bool mTonemap = true;

	bool mDebugPaths = false;
	bool mDebugPathWeights = false;

	float mVmRadiusAlpha  = 0.01f;
	float mVmRadiusFactor = 0.025f;

	VcmReservoirFlags mDIReservoirFlags = VcmReservoirFlags::eNone;
	VcmReservoirFlags mLVCReservoirFlags = VcmReservoirFlags::eNone;

	bool mVisualizeLightPaths = false;
	uint32_t mVisualizeLightPathCount = 128;
	float mVisualizeLightPathRadius = 0.00075f;
	float mVisualizeLightPathLength = 0.05;
	uint32_t mVisualizeSegmentIndex = -1;

	VcmAlgorithmType mAlgorithm = VcmAlgorithmType::kBpt;
	bool mUsePerformanceCounters = false;
	float mLightPathPercent = 1;

	uint2 mHashGridStats = uint2::Zero();
	Buffer::View<uint32_t> mPerformanceCounters;
	vector<uint32_t> mPrevPerformanceCounters;
	vector<float> mPerformanceCounterPerSecond;
	float mPerformanceCounterTimer;

	DeviceResourcePool<FrameResources> mFrameResourcePool;
	shared_ptr<FrameResources> mPrevFrame;
	Image::View mLastResultImage;
	chrono::high_resolution_clock::time_point mLastSceneUpdateTime;
};

}