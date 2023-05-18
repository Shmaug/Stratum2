#include "ImageComparer.hpp"
#include "Inspector.hpp"
#include "Renderer.hpp"
#include "Denoiser.hpp"

#include <Core/Instance.hpp>
#include <Core/Profiler.hpp>

namespace stm2 {

ImageComparer::ImageComparer(Node& node) : mNode(node) {
	if (shared_ptr<Inspector> inspector = mNode.root()->findDescendant<Inspector>())
		inspector->setInspectCallback<ImageComparer>();

	const filesystem::path shaderPath(*mNode.root()->findDescendant<Instance>()->findArgument("shaderKernelPath"));
	mPipeline = ComputePipelineCache(shaderPath / "image_compare.slang");
}

Buffer::View<uint2> ImageComparer::compare(CommandBuffer& commandBuffer, const Image::View& img0, const Image::View& img1) {
	ProfilerScope ps("Image compare", &commandBuffer);
	const Buffer::View<uint2> resultBuffer    = make_shared<Buffer>(commandBuffer.mDevice, "ImageComparer/ResultBufferGpu", sizeof(uint2), vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferSrc|vk::BufferUsageFlagBits::eTransferDst);
	const Buffer::View<uint2> resultBufferCpu = make_shared<Buffer>(commandBuffer.mDevice, "ImageComparer/ResultBuffer"   , sizeof(uint2), vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);

	resultBuffer.fill(commandBuffer, 0);
	resultBuffer.barrier(commandBuffer,
		vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
		vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);

	mPipeline.get(commandBuffer.mDevice, {
		{ "gMode", to_string((uint32_t)mMode) }
	})->dispatchTiled(commandBuffer, img0.extent(), Descriptors{
		{ {"gImage1",0}, ImageDescriptor(img0, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {}) },
		{ {"gImage2",0}, ImageDescriptor(img1, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {}) },
		{ {"gOutput",0}, resultBuffer },
	}, {}, PushConstants{
		{ "mExtent", PushConstantValue(uint2(img0.extent().width, img0.extent().height)) },
		{ "mQuantization", PushConstantValue(mQuantization) },
	});

	resultBuffer.barrier(commandBuffer,
		vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer,
		vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead);

	Buffer::copy(commandBuffer, resultBuffer, resultBufferCpu);

	return resultBufferCpu;
}

void ImageComparer::postRender(CommandBuffer& commandBuffer, const Image::View& renderTarget) {
	ProfilerScope ps("ImageComparer::update");

	string storeSuffix;
	if (mStoreFrame) {
		auto denoiser = mNode.root()->findDescendant<Denoiser>();
		if (denoiser) {
			if (denoiser->accumulatedFrames() == *mStoreFrame) {
				mStore = true;
				storeSuffix = "_" + to_string(denoiser->accumulatedFrames());
			}
		}
	}

	// store/copy the renderTarget
	if (mStore) {
		const string name = string(mStoreLabel.data()) + storeSuffix;
		Image::Metadata md = renderTarget.image()->metadata();
		md.mUsage = vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eTransferSrc|vk::ImageUsageFlagBits::eTransferDst;
		const Image::View image = make_shared<Image>(commandBuffer.mDevice, "ImageComparer/" + name, md);

		Image::copy(commandBuffer, renderTarget, image);
		image.image()->generateMipMaps(commandBuffer);

		mImages.emplace_back(name, image, Buffer::View<uint2>{});
		mStore = false;
	}

	// compute comparisons
	if (mReferenceImage && !mReferenceImage.image()->inFlight()) {
		for (auto&[label, img, resultBuffer] : mImages) {
			if (img != mReferenceImage && !resultBuffer && img && !img.image()->inFlight()) {
				resultBuffer = compare(commandBuffer, mReferenceImage, img);
			}
		}
	}

	// show comparison window
	if (mSelected < mImages.size()) {
		const Image::View currentImage = get<Image::View>(mImages[mSelected]);
		if (ImGui::Begin("Image compare")) {
			// image pan/zoom
			const uint32_t w = ImGui::GetWindowSize().x - 4;
			commandBuffer.trackResource(currentImage.image());
			const float aspect = (float)currentImage.extent().height / (float)currentImage.extent().width;
			ImVec2 uvMin(mOffset[0], mOffset[1]);
			ImVec2 uvMax(mOffset[0] + mZoom, mOffset[1] + mZoom);
			ImGui::Image(Gui::getTextureID(currentImage), ImVec2(w, w * aspect), uvMin, uvMax);
			if (ImGui::IsItemHovered()) {
				mZoom *= 1 - 0.05f*ImGui::GetIO().MouseWheel;
				mZoom      = clamp(mZoom, 0.f, 1.f);
				mOffset[0] = clamp(mOffset[0], 0.f, 1 - mZoom);
				mOffset[1] = clamp(mOffset[1], 0.f, 1 - mZoom);
				if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
					const ImGuiIO& io = ImGui::GetIO();
					mOffset[0] -= (uvMax.x - uvMin.x) * io.MouseDelta.x / w;
					mOffset[1] -= (uvMax.y - uvMin.y) * io.MouseDelta.y / (w*aspect);
					mOffset[0] = clamp(mOffset[0], 0.f, 1 - mZoom);
					mOffset[1] = clamp(mOffset[1], 0.f, 1 - mZoom);
				}
			}
		}
		ImGui::End();
	}
}

void ImageComparer::drawGui() {
	if (ImGui::Button("Clear")) {
		mImages.clear();
		mReferenceImage = {};
	}
	ImGui::SameLine();
	if (ImGui::Button("Reload shaders")) {
		mPipeline.clear();
	}

	// properties
	{
		bool changed = false;
		ImGui::SetNextItemWidth(80);
		if (ImGui::DragFloat("Quantization", &mQuantization))
			changed = true;
		ImGui::SetNextItemWidth(160);
		if (Gui::enumDropdown<ImageCompareMode>("Error mode", mMode, (uint32_t)ImageCompareMode::eCompareMetricCount))
			changed = true;

		if (changed)
			for (auto& [label,img,resultBuffer] : mImages)
				resultBuffer.reset();
	}

	// save button
	{
		mStoreLabel.resize(64);
		ImGui::InputText("##StoreLabel", mStoreLabel.data(), mStoreLabel.size());
		ImGui::SameLine();
		if (ImGui::Button("Save"))
			mStore = true;
	}

	bool v = mStoreFrame.has_value();
	ImGui::Checkbox("Store frame number", &v);
	if (v) {
		if (!mStoreFrame) mStoreFrame = 0;
		ImGui::SameLine();
		ImGui::SetNextItemWidth(30);
		ImGui::DragScalar("##StoreFrame", ImGuiDataType_U32, &mStoreFrame.value());
	} else {
		mStoreFrame = nullopt;
	}

	// list images and mse values
	for (auto it = mImages.begin(); it != mImages.end();) {
		auto&[label,img,resultBuffer] = *it;
		const uint32_t index = it - mImages.begin();

		ImGui::PushID(hashArgs(int64_t(this), index));
		if (ImGui::Selectable(label.c_str(), mSelected == index)) {
			mSelected = mSelected == index ? -1 : index;
		}
		bool d = false;
		if (ImGui::BeginPopupContextItem()) {
			if (mReferenceImage != img && ImGui::Selectable("Set as reference")) {
				mReferenceImage = img;
				for (auto& [label,img,resultBuffer] : mImages)
					resultBuffer.reset();
			}
			d = ImGui::Selectable("Delete");
			ImGui::EndPopup();
		}

		if (resultBuffer) {
			const bool overflowed = resultBuffer[0][1] > 0;
			float result = ((overflowed ? ~uint32_t(0) : resultBuffer[0][0]) / mQuantization) / (3 * img.extent().width * img.extent().height);
			if (mMode == ImageCompareMode::eMSE)
				result = sqrt(result);
			ImGui::SameLine();
			ImGui::Text("%s%f", overflowed ? ">" : "", result);
		}

		ImGui::PopID();

		if (d) {
			if (mReferenceImage == img)
				mReferenceImage = {};
			if (mSelected == index)
				mSelected = -1;
			else if (mSelected < mImages.size() && mSelected > index)
				mSelected--;
			it = mImages.erase(it);
		} else
			it++;
	}
}

}