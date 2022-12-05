#pragma once

#include "Scene.hpp"

#include <Shaders/compat/scene.h>
#include <Shaders/compat/tinypt.h>

namespace stm2 {

class PathTracer {
public:
	Node& mNode;

	PathTracer(Node& node);

	void createPipelines(Device& device);

	void drawGui();
	void render(CommandBuffer& commandBuffer, const Image::View& renderTarget);

private:
	enum RenderPipelineIndex {
		eTraceViewPaths,
		ePipelineCount
	};
	array<ComputePipelineCache, RenderPipelineIndex::ePipelineCount> mRenderPipelines;

	TinyPTPushConstants mPushConstants;

	bool mHalfColorPrecision = false;
	bool mRandomPerFrame = true;
	bool mDenoise = true;
	bool mTonemap = true;

	TinyPTDebugMode mDebugMode = TinyPTDebugMode::eNone;
	uint32_t mFeatureFlags = BIT((uint32_t)TinyPTFeatureFlagBits::ePerformanceCounters);

	Buffer::View<uint32_t> mRayCount;
	vector<uint32_t> mPrevRayCount;
	vector<float> mRaysPerSecond;
	float mRayCountTimer;

	class FrameResources : public Device::Resource {
	public:
		inline FrameResources(Device& device) : Device::Resource(device, "PathTracer::FrameResources") {}

		chrono::high_resolution_clock::time_point mTime;

		shared_ptr<Scene::FrameResources> mSceneData;
		Buffer::View<VisibilityData> mSelectionData;
		bool mSelectionDataValid;

		shared_ptr<DescriptorSets> mDescriptorSets;

		unordered_map<string, ImageDescriptor> mImages;
		unordered_map<string, Buffer::View<byte>> mBuffers;

		template<typename T>
		inline Buffer::View<T> getBuffer(const string& name, const vk::DeviceSize count, const vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorage, const vk::MemoryPropertyFlags memoryProperties = vk::MemoryPropertyFlagBits::eDeviceLocal) {
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
};

}