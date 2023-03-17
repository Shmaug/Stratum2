#pragma once

#include "Pipeline.hpp"

namespace stm2 {

class DeviceResourcePool {
private:
	unordered_map<string, list<shared_ptr<DescriptorSets>>> mDescriptorSets;
	unordered_map<string, list<Image::View>> mImages;
	unordered_map<string, list<Buffer::View<byte>>> mBuffers;

public:
	inline void clear() {
		mDescriptorSets.clear();
		mImages.clear();
		mBuffers.clear();
	}

	inline void clean(uint32_t maxAge = 16) {
		for (auto&[name, items] : mDescriptorSets) {
			for (auto it = items.begin(); it != items.end();) {
				const auto& resource = *it;
				if (resource->mDevice.frameIndex() - resource->lastFrameUsed() > maxAge)
					it = items.erase(it);
				else
					it++;
			}
		}
		for (auto&[name, items] : mImages) {
			for (auto it = items.begin(); it != items.end();) {
				const auto& resource = it->image();
				if (resource->mDevice.frameIndex() - resource->lastFrameUsed() > maxAge)
					it = items.erase(it);
				else
					it++;
			}
		}
		for (auto&[name, items] : mBuffers) {
			for (auto it = items.begin(); it != items.end();) {
				const auto& resource = it->buffer();
				if (resource->mDevice.frameIndex() - resource->lastFrameUsed() > maxAge)
					it = items.erase(it);
				else
					it++;
			}
		}
	}

	inline Image::View getLastImage(const string& name) const {
		auto images_it = mImages.find(name);
		if (images_it == mImages.end()) return {};
		const list<Image::View>& images = images_it->second;

		// find newest resource
		Image::View img;
		uint32_t frame = 0;
		for (const Image::View& i : images) {
			if (i.image()->lastFrameUsed() >= frame) {
				frame = i.image()->lastFrameUsed();
				img = i;
			}
		}
		return img;
	}
	template<typename T>
	inline Buffer::View<T> getLastBuffer(const string& name) const {
		auto buffers_it = mBuffers.find(name);
		if (buffers_it == mBuffers.end()) return {};
		const list<Buffer::View<byte>>& buffers = buffers_it->second;

		// find newest resource
		Buffer::View<T> buf;
		uint32_t frame = 0;
		for (const Buffer::View<byte>& b : buffers) {
			if (b.buffer()->lastFrameUsed() >= frame) {
				frame = b.buffer()->lastFrameUsed();
				buf = b.cast<T>();
			}
		}
		return buf;
	}

	template<typename T>
	inline Buffer::View<T> getBuffer(Device& device, const string& name, const vk::DeviceSize count, const vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer, const vk::MemoryPropertyFlags memoryProperties = vk::MemoryPropertyFlagBits::eDeviceLocal, const uint32_t bufferCount = 1) {
		auto& buffers = mBuffers[name];

		for (const auto& buf : buffers) {
			if (buf.sizeBytes() < sizeof(T)*count ||
				!(buf.buffer()->usage() & usage) ||
				!(buf.buffer()->memoryUsage() & memoryProperties))
				continue;
			if (buf.buffer()->lastFrameUsed() + bufferCount < device.frameIndex()) {
				buf.buffer()->markUsed();
				return buf.cast<T>();
			}
		}

		Buffer::View<T> b = make_shared<Buffer>(device, name, sizeof(T)*count, usage, memoryProperties);
		buffers.emplace_back(b);
		b.buffer()->markUsed();
		return b;
	}

	template<typename T>
	inline Buffer::View<T> uploadData(CommandBuffer& commandBuffer, const string& name, const vk::ArrayProxy<T>& data, const vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer, const vk::MemoryPropertyFlags memoryProperties = vk::MemoryPropertyFlagBits::eDeviceLocal, const uint32_t bufferCount = 1) {
		Buffer::View<T> src = getBuffer<T>(commandBuffer.mDevice, name+" (Staging)", data.size(), vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, commandBuffer.mDevice.frameIndex() - commandBuffer.mDevice.lastFrameDone() - 1);
		Buffer::View<T> dst = getBuffer<T>(commandBuffer.mDevice, name             , data.size(), vk::BufferUsageFlagBits::eTransferDst|usage, memoryProperties, bufferCount);
		ranges::uninitialized_copy(data, src);
		Buffer::copy(commandBuffer, src, dst);
		commandBuffer.trackResource(src.buffer());
		commandBuffer.trackResource(dst.buffer());
		return dst;
	}

	inline Image::View getImage(Device& device, const string& name, const Image::Metadata& metadata, const uint32_t bufferCount = 1) {
		auto& images = mImages[name];

		for (const auto& img : images) {
			if (img.extent().width  != metadata.mExtent.width ||
				img.extent().height != metadata.mExtent.height ||
				img.extent().depth  != metadata.mExtent.depth ||
				img.image()->format() != metadata.mFormat ||
				!(img.image()->usage() & metadata.mUsage))
				continue;
			if (img.image()->lastFrameUsed() + bufferCount < device.frameIndex()) {
				img.image()->markUsed();
				return img;
			}
		}

		Image::View img = make_shared<Image>(device, name, metadata);
		images.emplace_back(img);
		img.image()->markUsed();
		return img;
	}

	inline shared_ptr<DescriptorSets> getDescriptorSets(Pipeline& pipeline, const string& name, const Descriptors& descriptors = {}) {
		auto& descriptorSets = mDescriptorSets[name];

		for (const auto& sets : descriptorSets) {
			if (sets->inFlight() || sets->mPipeline.descriptorSetLayouts().size() != pipeline.descriptorSetLayouts().size())
				continue;
			bool same = true;
			for (uint32_t i = 0; i < sets->mPipeline.descriptorSetLayouts().size(); i++) {
				if (sets->mPipeline.descriptorSetLayouts()[i] == pipeline.descriptorSetLayouts()[i])
					continue;
				same = false;
				break;
			}
			if (same) {
				sets->markUsed();
				sets->write(descriptors);
				return sets;
			}
		}

		shared_ptr<DescriptorSets> sets = make_shared<DescriptorSets>(pipeline, name);
		descriptorSets.emplace_back(sets);
		sets->markUsed();
		sets->write(descriptors);
		return sets;
	}

	void drawGui();
};

}