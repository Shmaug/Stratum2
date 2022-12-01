#pragma once

#include <Core/Pipeline.hpp>

#include "Node.hpp"

namespace stm2 {

class TestRenderer {
public:
	Node& mNode;

	TestRenderer(Node&);

	void createPipelines(Device& device);

	void drawGui();

	void render(CommandBuffer& commandBuffer, const Image::View& renderTarget);

private:
	ComputePipelineCache mPipeline;
	PushConstants mPushConstants;

	bool mRandomPerFrame;


	class FrameResources : public Device::Resource {
	public:
		inline FrameResources(Device& device) : Device::Resource(device, "TestRenderer::FrameResources") {}

		shared_ptr<DescriptorSets> mDescriptors;

		unordered_map<string, ImageDescriptor> mImages;
		unordered_map<string, Buffer::View<byte>> mBuffers;

		template<typename T>
		inline Buffer::View<T> getBuffer(const string& name, const vk::DeviceSize count, const vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorage, const vk::MemoryPropertyFlags memoryProperties = vk::MemoryPropertyFlags::eDeviceLocal) {
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

};