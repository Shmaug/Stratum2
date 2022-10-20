#include "Image.hpp"

namespace tinyvkpt {

Image::Image(Device& device, const string& name, const Metadata& metadata) : Device::Resource(device, name), mImage(nullptr), mMetadata(metadata) {
	mImage = vk::raii::Image(*device, vk::ImageCreateInfo(
		{},
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
		vk::ImageLayout::eUndefined ));
	mOwned = true;
	device.setDebugName(*mImage, resourceName());
}
Image::Image(Device& device, const string& name, vk::raii::Image&& image, const Metadata& metadata) : Device::Resource(device, name), mImage(move(image)), mMetadata(metadata) {
	mOwned = true;
	if (*mImage) device.setDebugName(*mImage, resourceName());
}
Image::Image(Device& device, const string& name, const vk::Image image, const Metadata& metadata) : Device::Resource(device, name), mImage(*device, image), mMetadata(metadata) {
	mOwned = false;
	if (*mImage) device.setDebugName(*mImage, resourceName());
}
Image::~Image() {
	if (!mOwned) memset(&mImage, 0, sizeof(vk::raii::Image));
}

const vk::ImageView Image::view(const vk::ImageSubresourceRange& subresource, const vk::ImageViewType viewType, const vk::ComponentMapping& componentMapping) {
	auto it = mViews.find(tie(subresource, viewType, componentMapping));
	if (it == mViews.end()) {
		vk::raii::ImageView v(*mDevice, vk::ImageViewCreateInfo({},
			*mImage,
			viewType,
			format(),
			componentMapping,
			subresource));
		it = mViews.emplace(tie(subresource, viewType, componentMapping), move(v)).first;
	}
	return *it->second;
}

}