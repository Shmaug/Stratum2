#pragma once

#include <bit>

#include "CommandBuffer.hpp"

#include <Utils/hash.hpp>

namespace stm2 {

class Image : public Device::Resource {
public:
	using SubresourceLayoutState = tuple<vk::ImageLayout, vk::PipelineStageFlags, vk::AccessFlags, uint32_t /*queueFamily*/>;

	struct Metadata {
		vk::ImageCreateFlags mCreateFlags = {};
		vk::ImageType mType = vk::ImageType::e2D;
		vk::Format mFormat;
		vk::Extent3D mExtent;
		uint32_t mLevels = 1;
		uint32_t mLayers = 1;
		vk::SampleCountFlagBits mSamples = vk::SampleCountFlagBits::e1;
		vk::ImageUsageFlags mUsage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
		vk::ImageTiling mTiling = vk::ImageTiling::eOptimal;
		vk::SharingMode mSharingMode = vk::SharingMode::eExclusive;
		vector<uint32_t> mQueueFamilies;
	};

	inline static uint32_t maxMipLevels(const vk::Extent3D& extent) {
		return 32 - (uint32_t)countl_zero(max(max(extent.width, extent.height), extent.depth));
	}

	using PixelData = tuple<shared_ptr<Buffer>, vk::Format, vk::Extent3D>;
	static PixelData loadFile(Device& device, const filesystem::path& filename, const bool srgb = true, int desiredChannels = 0);

	Image(Device& device, const string& name, const Metadata& metadata, const vk::MemoryPropertyFlags memoryFlags = vk::MemoryPropertyFlagBits::eDeviceLocal);
	Image(Device& device, const string& name, const vk::Image image, const Metadata& metadata);
	~Image();

	DECLARE_DEREFERENCE_OPERATORS(vk::Image, mImage)

	inline operator bool() const { return mImage; }

	inline Metadata metadata() const { return mMetadata; }
	inline vk::ImageType type() const { return mMetadata.mType; }
	inline vk::Format format() const { return mMetadata.mFormat; }
	inline vk::Extent3D extent(const uint32_t level = 0) const {
		uint32_t s = 1 << level;
		const vk::Extent3D& e = mMetadata.mExtent;
		return vk::Extent3D(max(e.width / s, 1u), max(e.height / s, 1u), max(e.depth / s, 1u));
	}
	inline uint32_t levels() const { return mMetadata.mLevels; }
	inline uint32_t layers() const { return mMetadata.mLayers; }
	inline vk::SampleCountFlagBits samples() const { return mMetadata.mSamples; }
	inline vk::ImageUsageFlags usage() const { return mMetadata.mUsage; }
	inline vk::ImageTiling tiling() const { return mMetadata.mTiling; }
	inline vk::SharingMode sharingMode() const { return mMetadata.mSharingMode; }
	inline const vector<uint32_t>& queueFamilies() const { return mMetadata.mQueueFamilies; }

	const vk::ImageView view(const vk::ImageSubresourceRange& subresource, const vk::ImageViewType viewType = vk::ImageViewType::e2D, const vk::ComponentMapping& componentMapping = {});

	void generateMipMaps(CommandBuffer& commandBuffer, const vk::Filter filter = vk::Filter::eLinear, const vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor);

	void upload(CommandBuffer& commandBuffer, const PixelData& pixels);

	void barrier(CommandBuffer& commandBuffer, const vk::ImageSubresourceRange& subresource, const SubresourceLayoutState& newState);
	inline void barrier(CommandBuffer& commandBuffer, const vk::ImageSubresourceRange& subresource, const vk::ImageLayout layout, const vk::PipelineStageFlags stage, const vk::AccessFlags accessMask, uint32_t queueFamily = VK_QUEUE_FAMILY_IGNORED) {
		barrier(commandBuffer, subresource, { layout, stage, accessMask, queueFamily });
	}

	// use to update tracked state e.g. after a renderpass
	void updateState(const vk::ImageSubresourceRange& subresource, const SubresourceLayoutState& newState);
	inline void updateState(const vk::ImageSubresourceRange& subresource, const vk::ImageLayout layout, const vk::PipelineStageFlags stage, const vk::AccessFlags accessMask, const uint32_t queueFamily = VK_QUEUE_FAMILY_IGNORED) {
		updateState(subresource, { layout, stage, accessMask, queueFamily });
	}

	void clearColor(CommandBuffer& commandBuffer, const vk::ClearColorValue& clearValue, const vk::ArrayProxy<const vk::ImageSubresourceRange>& subresources);

	class View {
	public:
		View() = default;
		View(const View&) = default;
		View(View&&) = default;
		inline View(const shared_ptr<Image>& image, const vk::ImageSubresourceRange& subresource = { vk::ImageAspectFlagBits::eColor, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS }, vk::ImageViewType viewType = vk::ImageViewType::e2D, const vk::ComponentMapping& componentMapping = {})
			: mImage(image), mSubresource(subresource), mType(viewType), mComponentMapping(componentMapping) {
			if (image) {
				if (mSubresource.levelCount == VK_REMAINING_MIP_LEVELS) mSubresource.levelCount = image->levels();
				if (mSubresource.layerCount == VK_REMAINING_ARRAY_LAYERS) mSubresource.layerCount = image->layers();
				mView = image->view(subresource, viewType, componentMapping);
			}
		}
		View& operator=(const View&) = default;
		View& operator=(View&& v) = default;

		DECLARE_DEREFERENCE_OPERATORS(vk::ImageView, mView)

		inline bool operator==(const View& rhs) const { return mView == rhs.mView; }
		inline bool operator!=(const View& rhs) const { return mView != rhs.mView; }

		inline operator bool() const { return mView; }

		inline const shared_ptr<Image>& image() const { return mImage; }
		inline const vk::ImageSubresourceRange& subresourceRange() const { return mSubresource; }
		inline vk::ImageSubresourceLayers subresourceLayer(const uint32_t levelOffset = 0) const {
			return vk::ImageSubresourceLayers(mSubresource.aspectMask, mSubresource.baseMipLevel + levelOffset, mSubresource.baseArrayLayer, mSubresource.layerCount);
		}
		inline const vk::ImageViewType type() const { return mType; }
		inline const vk::ComponentMapping& componentMapping() const { return mComponentMapping; }
		inline vk::ImageSubresourceRange subresource() const { return mSubresource; }

		inline vk::Extent3D extent(const uint32_t levelOffset = 0) const {
			return mImage->extent(mSubresource.baseMipLevel + levelOffset);
		}

		inline void barrier(CommandBuffer& commandBuffer, const Image::SubresourceLayoutState& newState) const {
			mImage->barrier(commandBuffer, mSubresource, newState);
		}
		inline void barrier(CommandBuffer& commandBuffer, const vk::ImageLayout layout, const vk::PipelineStageFlags stage, const vk::AccessFlags accessMask, uint32_t queueFamily = VK_QUEUE_FAMILY_IGNORED) const {
			mImage->barrier(commandBuffer, mSubresource, { layout, stage, accessMask, queueFamily });
		}

		inline void updateState(const SubresourceLayoutState& newState) const {
			mImage->updateState(mSubresource, newState);
		}
		inline void updateState(const vk::ImageLayout layout, const vk::PipelineStageFlags stage, const vk::AccessFlags accessMask, const uint32_t queueFamily = VK_QUEUE_FAMILY_IGNORED) const {
			updateState({ layout, stage, accessMask, queueFamily });
		}

		inline void clearColor(CommandBuffer& commandBuffer, const vk::ClearColorValue& clearValue) const {
			mImage->clearColor(commandBuffer, clearValue, mSubresource);
			commandBuffer.trackResource(mImage);
		}

	private:
		friend class Image;
		shared_ptr<Image> mImage;
		vk::ImageView mView;
		vk::ImageSubresourceRange mSubresource;
		vk::ImageViewType mType;
		vk::ComponentMapping mComponentMapping;
	};

	static void copy(CommandBuffer& commandBuffer, const shared_ptr<Image>& src, const shared_ptr<Image>& dst, const vk::ArrayProxy<const vk::ImageCopy>& regions);
	static void blit(CommandBuffer& commandBuffer, const shared_ptr<Image>& src, const shared_ptr<Image>& dst, const vk::ArrayProxy<const vk::ImageBlit>& regions, const vk::Filter filter = vk::Filter::eLinear);

	inline static void copy(CommandBuffer& commandBuffer, const View& src, const View& dst) {
		vk::ImageCopy c(
				src.subresourceLayer(), vk::Offset3D(0,0,0),
				dst.subresourceLayer(), vk::Offset3D(0,0,0),
				dst.extent());
		copy(commandBuffer, src.image(), dst.image(), c);
	}
	inline static void blit(CommandBuffer& commandBuffer, const View& src, const View& dst, const vk::Filter filter = vk::Filter::eLinear) {
		vk::ImageBlit c(
				src.subresourceLayer(), { vk::Offset3D(0,0,0), vk::Offset3D(src.extent().width, src.extent().height, src.extent().depth) },
				dst.subresourceLayer(), { vk::Offset3D(0,0,0), vk::Offset3D(dst.extent().width, dst.extent().height, dst.extent().depth) });
		blit(commandBuffer, src.image(), dst.image(), c, filter);
	}

private:
	vk::Image mImage;
	VmaAllocation mAllocation;
	Metadata mMetadata;
	unordered_map<tuple<vk::ImageSubresourceRange, vk::ImageViewType, vk::ComponentMapping>, vk::raii::ImageView> mViews;
	vector<vector<SubresourceLayoutState>> mSubresourceStates; // mSubresourceStates[arrayLayer][level]
};

}

namespace std {

template<>
struct hash<stm2::Image::View> {
	inline size_t operator()(const stm2::Image::View& v) const {
		return hash<vk::ImageView>()(*v);
	}
};

}
