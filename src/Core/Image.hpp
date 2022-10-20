#pragma once

#include "Device.hpp"

#include <Utils/hash.hpp>

namespace tinyvkpt {

class Image : public Device::Resource {
public:
	struct Metadata {
		vk::ImageType mType = vk::ImageType::e2D;
		vk::Format mFormat;
		vk::Extent3D mExtent;
		uint32_t mLevels = 1;
		uint32_t mLayers = 1;
		vk::SampleCountFlagBits mSamples = vk::SampleCountFlagBits::e1;
		vk::ImageUsageFlags mUsage;
		vk::ImageTiling mTiling = vk::ImageTiling::eOptimal;
		vk::SharingMode mSharingMode = vk::SharingMode::eExclusive;
		vector<uint32_t> mQueueFamilies;
	};

	Image(Device& device, const string& name, const Metadata& metadata);
	Image(Device& device, const string& name, vk::raii::Image&& image, const Metadata& metadata);
	Image(Device& device, const string& name, const vk::Image image, const Metadata& metadata);
	~Image();

	inline vk::raii::Image& operator*() { return mImage; }
	inline vk::raii::Image* operator->() { return &mImage; }
	inline const vk::raii::Image& operator*() const { return mImage; }
	inline const vk::raii::Image* operator->() const { return &mImage; }

	inline vk::ImageType type() const { return mMetadata.mType; }
	inline vk::Extent3D extent() const { return mMetadata.mExtent; }
	inline vk::Format format() const { return mMetadata.mFormat; }
	inline uint32_t levels() const { return mMetadata.mLevels; }
	inline uint32_t layers() const { return mMetadata.mLayers; }
	inline vk::SampleCountFlagBits samples() const { return mMetadata.mSamples; }
	inline vk::ImageUsageFlags usage() const { return mMetadata.mUsage; }
	inline vk::ImageTiling tiling() const { return mMetadata.mTiling; }
	inline vk::SharingMode sharingMode() const { return mMetadata.mSharingMode; }
	inline const vector<uint32_t>& queueFamilies() const { return mMetadata.mQueueFamilies; }

	const vk::ImageView view(const vk::ImageSubresourceRange& subresource, const vk::ImageViewType viewType = vk::ImageViewType::e2D, const vk::ComponentMapping& componentMapping = {});

	class View {
		friend class Image;
	private:
		shared_ptr<Image> mImage;
		vk::ImageView mView;
		vk::ImageSubresourceRange mSubresource;
		vk::ImageViewType mType;
		vk::ComponentMapping mComponentMapping;

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

		inline vk::ImageView& operator*() { return mView; }
		inline vk::ImageView* operator->() { return &mView; }
		inline const vk::ImageView& operator*() const { return mView; }
		inline const vk::ImageView* operator->() const { return &mView; }

		inline bool operator==(const View& rhs) const { return mView == rhs.mView; }
		inline bool operator!=(const View& rhs) const { return mView != rhs.mView; }

		inline operator bool() const { return mView; }

		inline const shared_ptr<Image>& image() const { return mImage; }
		inline const vk::ImageSubresourceRange& subresourceRange() const { return mSubresource; }
		inline const vk::ImageViewType type() const { return mType; }
		inline const vk::ComponentMapping& componentMapping() const { return mComponentMapping; }
		inline vk::ImageSubresourceLayers subresource(uint32_t level) const {
			return vk::ImageSubresourceLayers(mSubresource.aspectMask, mSubresource.baseMipLevel + level, mSubresource.baseArrayLayer, mSubresource.layerCount);
		}
		inline vk::Extent3D extent(uint32_t level = 0) const {
			uint32_t s = 1 << (mSubresource.baseMipLevel + level);
			const vk::Extent3D& e = mImage->extent();
			return vk::Extent3D(max(e.width / s, 1u), max(e.height / s, 1u), max(e.depth / s, 1u));
		}
	};

private:
	vk::raii::Image mImage;
	bool mOwned;
	Metadata mMetadata;
	unordered_map<tuple<vk::ImageSubresourceRange, vk::ImageViewType, vk::ComponentMapping>, vk::raii::ImageView> mViews;
	unordered_map<tuple<vk::ImageAspectFlagBits, uint32_t/*level*/, uint32_t/*layer*/>, tuple<vk::ImageLayout, vk::PipelineStageFlags, vk::AccessFlags>> mSubresourceStates;
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
