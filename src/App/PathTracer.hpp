#pragma once

#include "Scene.hpp"

#include <Shaders/compat/scene.h>
#include <Shaders/compat/path_tracer.h>

namespace stm2 {

class PathTracer {
public:
	Node& mNode;

	PathTracer(Node& node);

	void createPipelines(Device& device);

	void drawGui();
	void render(CommandBuffer& commandBuffer, const Image::View& renderTarget);

	inline Image::View resultImage() const { return mLastResultImage; }

private:
	enum RenderPipelineIndex {
		eGenerateLightPaths,
		eGenerateCameraPaths,
		eHashGridComputeIndices,
		eHashGridSwizzle,
		ePipelineCount
	};
	array<ComputePipelineCache, RenderPipelineIndex::ePipelineCount> mRenderPipelines;

	PathTracerPushConstants mPushConstants;

	bool mHalfColorPrecision = false;
	bool mRandomPerFrame = true;
	bool mDenoise = true;
	bool mTonemap = true;
	bool mDebugPaths = false;

	VcmAlgorithmType mAlgorithm = VcmAlgorithmType::kBpt;
	bool mUsePerformanceCounters = false;
	uint32_t mLightTraceQuantization = 16384;

	Buffer::View<uint32_t> mPerformanceCounters;
	vector<uint32_t> mPrevPerformanceCounters;
	vector<float> mPerformanceCounterPerSecond;
	float mPerformanceCounterTimer;

	class FrameResources : public Device::Resource {
	public:
		inline FrameResources(Device& device) : Device::Resource(device, "PathTracer::FrameResources") {}

		chrono::high_resolution_clock::time_point mTime;

		vector<pair<ViewData,TransformData>> mViews;

		shared_ptr<Scene::FrameResources> mSceneData;
		Buffer::View<VisibilityData> mSelectionData;
		bool mSelectionDataValid;

		shared_ptr<DescriptorSets> mDescriptorSets;

		unordered_map<string, ImageDescriptor> mImages;
		unordered_map<string, Buffer::View<byte>> mBuffers;

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

	DeviceResourcePool<FrameResources> mFrameResourcePool;
	shared_ptr<FrameResources> mPrevFrame;
	Image::View mLastResultImage;
	chrono::high_resolution_clock::time_point mLastSceneUpdateTime;
};

}