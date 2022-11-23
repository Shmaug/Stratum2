#include "Image.hpp"
#include "Buffer.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>

#define TINYEXR_USE_MINIZ 1
#include <miniz.h>
#define TINYEXR_IMPLEMENTATION
#include <tiny_exr.h>

#define TINYDDSLOADER_IMPLEMENTATION
#include <tinyddsloader.h>

namespace tinyvkpt {


inline vk::Format dxgiToVulkan(tinyddsloader::DDSFile::DXGIFormat format, const bool alphaFlag) {
	switch (format) {
		case tinyddsloader::DDSFile::DXGIFormat::BC1_UNorm: {
			if (alphaFlag) return vk::Format::eBc1RgbaUnormBlock;
			else return vk::Format::eBc1RgbUnormBlock;
		}
		case tinyddsloader::DDSFile::DXGIFormat::BC1_UNorm_SRGB: {
			if (alphaFlag) return vk::Format::eBc1RgbaSrgbBlock;
			else return vk::Format::eBc1RgbSrgbBlock;
		}

		case tinyddsloader::DDSFile::DXGIFormat::BC2_UNorm:       return vk::Format::eBc2UnormBlock;
		case tinyddsloader::DDSFile::DXGIFormat::BC2_UNorm_SRGB:  return vk::Format::eBc2SrgbBlock;
		case tinyddsloader::DDSFile::DXGIFormat::BC3_UNorm:       return vk::Format::eBc3UnormBlock;
		case tinyddsloader::DDSFile::DXGIFormat::BC3_UNorm_SRGB:  return vk::Format::eBc3SrgbBlock;
		case tinyddsloader::DDSFile::DXGIFormat::BC4_UNorm:       return vk::Format::eBc4UnormBlock;
		case tinyddsloader::DDSFile::DXGIFormat::BC4_SNorm:       return vk::Format::eBc4SnormBlock;
		case tinyddsloader::DDSFile::DXGIFormat::BC5_UNorm:       return vk::Format::eBc5UnormBlock;
		case tinyddsloader::DDSFile::DXGIFormat::BC5_SNorm:       return vk::Format::eBc5SnormBlock;

		case tinyddsloader::DDSFile::DXGIFormat::R8G8B8A8_UNorm:      return vk::Format::eR8G8B8A8Unorm;
		case tinyddsloader::DDSFile::DXGIFormat::R8G8B8A8_UNorm_SRGB: return vk::Format::eR8G8B8A8Srgb;
		case tinyddsloader::DDSFile::DXGIFormat::R8G8B8A8_UInt:       return vk::Format::eR8G8B8A8Uint;
		case tinyddsloader::DDSFile::DXGIFormat::R8G8B8A8_SNorm:      return vk::Format::eR8G8B8A8Snorm;
		case tinyddsloader::DDSFile::DXGIFormat::R8G8B8A8_SInt:       return vk::Format::eR8G8B8A8Sint;
		case tinyddsloader::DDSFile::DXGIFormat::B8G8R8A8_UNorm:      return vk::Format::eB8G8R8A8Unorm;
		case tinyddsloader::DDSFile::DXGIFormat::B8G8R8A8_UNorm_SRGB: return vk::Format::eB8G8R8A8Srgb;

		case tinyddsloader::DDSFile::DXGIFormat::R16G16B16A16_Float:  return vk::Format::eR16G16B16A16Sfloat;
		case tinyddsloader::DDSFile::DXGIFormat::R16G16B16A16_SInt:   return vk::Format::eR16G16B16A16Sint;
		case tinyddsloader::DDSFile::DXGIFormat::R16G16B16A16_UInt:   return vk::Format::eR16G16B16A16Uint;
		case tinyddsloader::DDSFile::DXGIFormat::R16G16B16A16_UNorm:  return vk::Format::eR16G16B16A16Unorm;
		case tinyddsloader::DDSFile::DXGIFormat::R16G16B16A16_SNorm:  return vk::Format::eR16G16B16A16Snorm;

		default: return vk::Format::eUndefined;
	}
}

Image::PixelData Image::loadFile(Device& device, const filesystem::path& filename, const bool srgb, int desiredChannels) {
	if (!filesystem::exists(filename)) throw invalid_argument("File does not exist: " + filename.string());
	if (filename.extension() == ".exr") {
		float* data = nullptr;
		int width;
		int height;
		const char* err = nullptr;
		int ret = LoadEXR(&data, &width, &height, filename.string().c_str(), &err);
		if (ret != TINYEXR_SUCCESS) {
			std::cerr << "OpenEXR error: " << err << std::endl;
			FreeEXRErrorMessage(err);
			throw runtime_error(std::string("Failure when loading image: ") + filename.string());
		}
		auto buf = make_shared<Buffer>(device, filename.stem().string(), width*height*sizeof(float)*4, vk::BufferUsageFlagBits::eTransferSrc|vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		memcpy(buf->data(), data, buf->size());
		free(data);
		return Image::PixelData{buf, vk::Format::eR32G32B32A32Sfloat, vk::Extent3D(width,height,1)};
	} else if (filename.extension() == ".dds") {
		using namespace tinyddsloader;
		DDSFile dds;
    	auto ret = dds.Load(filename.string().c_str());
		if (tinyddsloader::Result::tinydds_Success != ret) throw runtime_error("Failed to load " + filename.string());
		dds.GetBitsPerPixel(dds.GetFormat());

		dds.Flip();

		const DDSFile::ImageData* img = dds.GetImageData(0, 0);

		auto buf = make_shared<Buffer>(device, filename.stem().string(), img->m_memSlicePitch, vk::BufferUsageFlagBits::eTransferSrc|vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		memcpy(buf->data(), img->m_mem, buf->size());
		return Image::PixelData{buf, dxgiToVulkan(dds.GetFormat(), false), vk::Extent3D(dds.GetWidth(), dds.GetHeight(), dds.GetDepth())};
	} else {
		int x,y,channels;
		stbi_info(filename.string().c_str(), &x, &y, &channels);

		if (channels == 3) desiredChannels = 4;

		byte* pixels = nullptr;
		vk::Format format = vk::Format::eUndefined;
		if (stbi_is_hdr(filename.string().c_str())) {
			pixels = (byte*)stbi_loadf(filename.string().c_str(), &x, &y, &channels, desiredChannels);
			switch(desiredChannels ? desiredChannels : channels) {
				case 1: format = vk::Format::eR32Sfloat; break;
				case 2: format = vk::Format::eR32G32Sfloat; break;
				case 3: format = vk::Format::eR32G32B32Sfloat; break;
				case 4: format = vk::Format::eR32G32B32A32Sfloat; break;
			}
		} else if (stbi_is_16_bit(filename.string().c_str())) {
			pixels = (byte*)stbi_load_16(filename.string().c_str(), &x, &y, &channels, desiredChannels);
			switch(desiredChannels ? desiredChannels : channels) {
				case 1: format = vk::Format::eR16Unorm; break;
				case 2: format = vk::Format::eR16G16Unorm; break;
				case 3: format = vk::Format::eR16G16B16Unorm; break;
				case 4: format = vk::Format::eR16G16B16A16Unorm; break;
			}
		} else {
			pixels = (byte*)stbi_load(filename.string().c_str(), &x, &y, &channels, desiredChannels);
			switch (desiredChannels ? desiredChannels : channels) {
				case 1: format = srgb ? vk::Format::eR8Srgb : vk::Format::eR8Unorm; break;
				case 2: format = srgb ? vk::Format::eR8G8Srgb : vk::Format::eR8G8Unorm; break;
				case 3: format = srgb ? vk::Format::eR8G8B8Srgb : vk::Format::eR8G8B8Unorm; break;
				case 4: format = srgb ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm; break;
			}
		}
		if (!pixels) throw invalid_argument("Could not load " + filename.string());
		cout << "Loaded " << filename << " (" << x << "x" << y << ")" << endl;
		if (desiredChannels) channels = desiredChannels;

		auto buf = make_shared<Buffer>(device, filename.stem().string() + "/Staging", x*y*texelSize(format), vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		memcpy(buf->data(), pixels, buf->size());
		stbi_image_free(pixels);
		return Image::PixelData{buf, format, vk::Extent3D(x,y,1)};
	}
}


Image::Image(Device& device, const string& name, const Metadata& metadata, const vk::MemoryPropertyFlags memoryFlags) : Device::Resource(device, name), mImage(nullptr), mMetadata(metadata) {
	VmaAllocationCreateInfo allocationCreateInfo;
	allocationCreateInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    allocationCreateInfo.usage = (memoryFlags & vk::MemoryPropertyFlagBits::eDeviceLocal) ? VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE : VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    allocationCreateInfo.requiredFlags = (VkMemoryPropertyFlags)memoryFlags;
    allocationCreateInfo.memoryTypeBits = 0;
    allocationCreateInfo.pool = VK_NULL_HANDLE;
    allocationCreateInfo.pUserData = VK_NULL_HANDLE;
    allocationCreateInfo.priority = 0;

	vk::ImageCreateInfo createInfo(
		mMetadata.mCreateFlags,
		type(),
		format(),
		extent(),
		levels(),
		layers(),
		samples(),
		tiling(),
		usage(),
		sharingMode(),
		queueFamilies(),
		vk::ImageLayout::eUndefined );

	vk::Result result = (vk::Result)vmaCreateImage(mDevice.allocator(), &(const VkImageCreateInfo&)createInfo, &allocationCreateInfo, &(VkImage&)mImage, &mAllocation, nullptr);
	if (result != vk::Result::eSuccess)
		vk::throwResultException(result, "vmaCreateImage");
	device.setDebugName(mImage, resourceName());
	mSubresourceStates = vector<vector<Image::SubresourceLayoutState>>(
		metadata.mLayers,
		vector<Image::SubresourceLayoutState>(
			metadata.mLevels,
			Image::SubresourceLayoutState{
				vk::ImageLayout::eUndefined,
				vk::PipelineStageFlagBits::eTopOfPipe,
				vk::AccessFlagBits::eNone,
				queueFamilies().empty() ? VK_QUEUE_FAMILY_IGNORED : queueFamilies().front() }));
}
Image::Image(Device& device, const string& name, const vk::Image image, const Metadata& metadata) : Device::Resource(device, name), mImage(image), mMetadata(metadata), mAllocation(nullptr) {
	mAllocation = nullptr;
	if (mImage) device.setDebugName(mImage, resourceName());
	mSubresourceStates = vector<vector<Image::SubresourceLayoutState>>(
		metadata.mLayers,
		vector<Image::SubresourceLayoutState>(
			metadata.mLevels,
			Image::SubresourceLayoutState{
				vk::ImageLayout::eUndefined,
				vk::PipelineStageFlagBits::eTopOfPipe,
				vk::AccessFlagBits::eNone,
				queueFamilies().empty() ? VK_QUEUE_FAMILY_IGNORED : queueFamilies().front() }));
}
Image::~Image() {
	if (mImage && mAllocation)
		vmaDestroyImage(mDevice.allocator(), mImage, mAllocation);
}

const vk::ImageView Image::view(const vk::ImageSubresourceRange& subresource, const vk::ImageViewType viewType, const vk::ComponentMapping& componentMapping) {
	auto it = mViews.find(tie(subresource, viewType, componentMapping));
	if (it == mViews.end()) {
		vk::raii::ImageView v(*mDevice, vk::ImageViewCreateInfo({},
			mImage,
			viewType,
			format(),
			componentMapping,
			subresource));
		it = mViews.emplace(tie(subresource, viewType, componentMapping), move(v)).first;
	}
	return *it->second;
}

void Image::upload(CommandBuffer& commandBuffer, const PixelData& pixels) {
	barrier(commandBuffer,
		vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1),
		vk::ImageLayout::eTransferDstOptimal,
		vk::PipelineStageFlagBits::eTransfer,
		vk::AccessFlagBits::eTransferWrite);

	const auto& [buf, format, extent] = pixels;

	commandBuffer.trackResource(buf);

	commandBuffer->copyBufferToImage(**buf, mImage, vk::ImageLayout::eTransferDstOptimal,
		vk::BufferImageCopy(0, 0, 0, vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1), vk::Offset3D{0,0,0}, extent ));
}

void Image::barrier(CommandBuffer& commandBuffer, const vk::ImageSubresourceRange& subresource, const Image::SubresourceLayoutState& newState) {
	const auto& [ newLayout, newStage, dstAccessMask, dstQueueFamilyIndex ] = newState;

	unordered_map<pair<vk::PipelineStageFlags, vk::PipelineStageFlags>, vector<vk::ImageMemoryBarrier>> barriers;

	const vk::AccessFlags writeAccess =
		vk::AccessFlagBits::eShaderWrite |
		vk::AccessFlagBits::eColorAttachmentWrite |
		vk::AccessFlagBits::eDepthStencilAttachmentWrite |
		vk::AccessFlagBits::eTransferWrite |
		vk::AccessFlagBits::eHostWrite |
		vk::AccessFlagBits::eMemoryWrite |
		vk::AccessFlagBits::eAccelerationStructureWriteKHR;

	const uint32_t maxLayer = min(mMetadata.mLayers, subresource.baseArrayLayer+subresource.layerCount);
	const uint32_t maxLevel = min(mMetadata.mLevels, subresource.baseMipLevel+subresource.levelCount);

	for (uint32_t arrayLayer = subresource.baseArrayLayer; arrayLayer < maxLayer; arrayLayer++) {
		for (uint32_t level = subresource.baseMipLevel; level < maxLevel; level++) {
			auto& oldState = mSubresourceStates[arrayLayer][level];

			// try to combine barrier with one for previous mip level
			const auto& [ oldLayout, curStage, srcAccessMask, srcQueueFamilyIndex ] = oldState;
			if (oldState != newState || (srcAccessMask & writeAccess)) {
				vector<vk::ImageMemoryBarrier>& b = barriers[make_pair(curStage, newStage)];
				if (!b.empty()) {
					vk::ImageMemoryBarrier& prev = b.back();
					if (prev.oldLayout == oldLayout &&
						prev.srcAccessMask == srcAccessMask &&
						prev.srcQueueFamilyIndex == srcQueueFamilyIndex &&
						prev.subresourceRange.baseArrayLayer == arrayLayer &&
						prev.subresourceRange.baseMipLevel + prev.subresourceRange.levelCount == level) {

						prev.subresourceRange.levelCount++;
						oldState = newState;
						continue;
					}
				}

				b.emplace_back(vk::ImageMemoryBarrier(
					srcAccessMask, dstAccessMask,
					oldLayout, newLayout,
					dstQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED ? VK_QUEUE_FAMILY_IGNORED : srcQueueFamilyIndex, srcQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED ? VK_QUEUE_FAMILY_IGNORED : dstQueueFamilyIndex,
					mImage, vk::ImageSubresourceRange(subresource.aspectMask, level, 1, arrayLayer, 1) ));

				oldState = newState;
			}
		}
	}

	for (const auto&[stages, b] : barriers)
		commandBuffer->pipelineBarrier(stages.first, stages.second, vk::DependencyFlagBits::eByRegion, {}, {}, b);
}

void Image::updateState(const vk::ImageSubresourceRange& subresource, const Image::SubresourceLayoutState& newState) {
	for (uint32_t arrayLayer = subresource.baseArrayLayer; arrayLayer < subresource.layerCount; arrayLayer++) {
		for (uint32_t level = subresource.baseMipLevel; level < subresource.levelCount; level++) {
			mSubresourceStates[arrayLayer][level] = newState;
		}
	}
}

void Image::generateMipMaps(CommandBuffer& commandBuffer, const vk::Filter filter, const vk::ImageAspectFlags aspect) {
	barrier(commandBuffer,
		vk::ImageSubresourceRange(aspect, 1, levels()-1, 0, layers()),
		vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite);
	vk::ImageBlit blit = {};
	blit.srcOffsets[0] = blit.dstOffsets[0] = vk::Offset3D(0, 0, 0);
	blit.srcOffsets[1] = vk::Offset3D((int32_t)extent().width, (int32_t)extent().height, (int32_t)extent().depth);
	blit.srcSubresource.aspectMask = aspect;
	blit.srcSubresource.baseArrayLayer = 0;
	blit.srcSubresource.layerCount = layers();
	blit.dstSubresource = blit.srcSubresource;
	for (uint32_t i = 1; i < levels(); i++) {
		barrier(commandBuffer,
			vk::ImageSubresourceRange(aspect, i-1, 1, 0, layers()),
			vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferRead );
		blit.srcSubresource.mipLevel = i - 1;
		blit.dstSubresource.mipLevel = i;
		blit.dstOffsets[1].x = max(1, blit.srcOffsets[1].x / 2);
		blit.dstOffsets[1].y = max(1, blit.srcOffsets[1].y / 2);
		blit.dstOffsets[1].z = max(1, blit.srcOffsets[1].z / 2);
		commandBuffer->blitImage(
			mImage, vk::ImageLayout::eTransferSrcOptimal,
			mImage, vk::ImageLayout::eTransferDstOptimal,
			blit, vk::Filter::eLinear);
		blit.srcOffsets[1] = blit.dstOffsets[1];
	}
}

void Image::clearColor(CommandBuffer& commandBuffer, const vk::ClearColorValue& clearValue, const vk::ArrayProxy<const vk::ImageSubresourceRange>& subresources) {
	for (const auto& subresource : subresources)
		barrier(commandBuffer, subresource, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite);
	commandBuffer->clearColorImage(mImage, vk::ImageLayout::eTransferDstOptimal, clearValue, subresources);
}

void Image::copy(CommandBuffer& commandBuffer, const shared_ptr<Image>& src, const shared_ptr<Image>& dst, const vk::ArrayProxy<const vk::ImageCopy>& regions) {
	commandBuffer.trackResource(src);
	commandBuffer.trackResource(dst);
	for (const vk::ImageCopy& region : regions) {
		const auto& s = region.srcSubresource;
		src->barrier(commandBuffer, vk::ImageSubresourceRange(region.srcSubresource.aspectMask, region.srcSubresource.mipLevel, 1, region.srcSubresource.baseArrayLayer, region.srcSubresource.layerCount), vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferRead);
		dst->barrier(commandBuffer, vk::ImageSubresourceRange(region.dstSubresource.aspectMask, region.dstSubresource.mipLevel, 1, region.dstSubresource.baseArrayLayer, region.dstSubresource.layerCount), vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite);
	}
	commandBuffer->copyImage(**src, vk::ImageLayout::eTransferSrcOptimal, **dst, vk::ImageLayout::eTransferDstOptimal, regions);
}
void Image::blit(CommandBuffer& commandBuffer, const shared_ptr<Image>& src, const shared_ptr<Image>& dst, const vk::ArrayProxy<const vk::ImageBlit>& regions, const vk::Filter filter) {
	commandBuffer.trackResource(src);
	commandBuffer.trackResource(dst);
	for (const vk::ImageBlit& region : regions) {
		src->barrier(commandBuffer, vk::ImageSubresourceRange(region.srcSubresource.aspectMask, region.srcSubresource.mipLevel, 1, region.srcSubresource.baseArrayLayer, region.srcSubresource.layerCount), vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferRead);
		dst->barrier(commandBuffer, vk::ImageSubresourceRange(region.dstSubresource.aspectMask, region.dstSubresource.mipLevel, 1, region.dstSubresource.baseArrayLayer, region.dstSubresource.layerCount), vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite);
	}
	commandBuffer->blitImage(**src, vk::ImageLayout::eTransferSrcOptimal, **dst, vk::ImageLayout::eTransferDstOptimal, regions, filter);
}

}