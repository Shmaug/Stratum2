#pragma once

#include "Device.hpp"

#include <Utils/hash.hpp>

namespace tinyvkpt {

class Image : public Device::Resource {
public:
	using SubresourceLayoutState = tuple<vk::ImageLayout, vk::PipelineStageFlags, vk::AccessFlags, uint32_t /*queueFamily*/>;

	struct Metadata {
		vk::ImageCreateFlags mCreateFlags;
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

	Image(Device& device, const string& name, const Metadata& metadata, const vk::MemoryPropertyFlags memoryFlags = vk::MemoryPropertyFlagBits::eDeviceLocal);
	Image(Device& device, const string& name, const vk::Image image, const Metadata& metadata);
	~Image();

	DECLARE_DEREFERENCE_OPERATORS(vk::Image, mImage)

	inline operator bool() const { return mImage; }

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

	void barrier(CommandBuffer& commandBuffer, const vk::ImageSubresourceRange& subresource, const SubresourceLayoutState& newState);
	inline void barrier(CommandBuffer& commandBuffer, const vk::ImageSubresourceRange& subresource, const vk::ImageLayout layout, const vk::PipelineStageFlags stage, const vk::AccessFlags accessMask, uint32_t queueFamily = VK_QUEUE_FAMILY_IGNORED) {
		barrier(commandBuffer, subresource, { layout, stage, accessMask, queueFamily });
	}

	// use to update tracked state e.g. after a renderpass
	void updateState(const vk::ImageSubresourceRange& subresource, const SubresourceLayoutState& newState);
	inline void updateState(const vk::ImageSubresourceRange& subresource, const vk::ImageLayout layout, const vk::PipelineStageFlags stage, const vk::AccessFlags accessMask, const uint32_t queueFamily = VK_QUEUE_FAMILY_IGNORED) {
		updateState(subresource, { layout, stage, accessMask, queueFamily });
	}

	class View {
	public:
		View() = default;
		View(const View&) = default;
		View(View&&) = default;
		inline View(const shared_ptr<Image>& image, const vk::ImageSubresourceRange& subresource, vk::ImageViewType viewType = vk::ImageViewType::e2D, const vk::ComponentMapping& componentMapping = {})
			: mImage(image), mSubresource(subresource), mType(viewType), mComponentMapping(componentMapping) {
			if (image)
				mView = image->view(subresource, viewType, componentMapping);
		}
		View& operator=(const View&) = default;
		View& operator=(View&& v) = default;

		DECLARE_DEREFERENCE_OPERATORS(vk::ImageView, mView)

		inline bool operator==(const View& rhs) const { return mView == rhs.mView; }
		inline bool operator!=(const View& rhs) const { return mView != rhs.mView; }

		inline operator bool() const { return mView; }

		inline const shared_ptr<Image>& image() const { return mImage; }
		inline const vk::ImageSubresourceRange& subresourceRange() const { return mSubresource; }
		inline const vk::ImageViewType type() const { return mType; }
		inline const vk::ComponentMapping& componentMapping() const { return mComponentMapping; }
		inline vk::ImageSubresourceRange subresource() const { return mSubresource; }

		inline vk::Extent3D extent(const uint32_t levelOffset = 0) const {
			return mImage->extent(mSubresource.baseMipLevel + levelOffset);
		}

		inline void barrier(CommandBuffer& commandBuffer, const Image::SubresourceLayoutState& newState) const {
			mImage->barrier(commandBuffer, mSubresource, newState);
		}
		void barrier(CommandBuffer& commandBuffer, const vk::ImageLayout layout, const vk::PipelineStageFlags stage, const vk::AccessFlags accessMask, uint32_t queueFamily = VK_QUEUE_FAMILY_IGNORED) const {
			mImage->barrier(commandBuffer, mSubresource, { layout, stage, accessMask, queueFamily });
		}

		inline void updateState(const SubresourceLayoutState& newState) const {
			mImage->updateState(mSubresource, newState);
		}
		inline void updateState(const vk::ImageLayout layout, const vk::PipelineStageFlags stage, const vk::AccessFlags accessMask, const uint32_t queueFamily = VK_QUEUE_FAMILY_IGNORED) const {
			updateState({ layout, stage, accessMask, queueFamily });
		}

	private:
		friend class Image;
		shared_ptr<Image> mImage;
		vk::ImageView mView;
		vk::ImageSubresourceRange mSubresource;
		vk::ImageViewType mType;
		vk::ComponentMapping mComponentMapping;
	};

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
struct hash<tinyvkpt::Image::View> {
	inline size_t operator()(const tinyvkpt::Image::View& v) const {
		return hash<vk::ImageView>()(*v);
	}
};

}
