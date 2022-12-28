#include "ImageComparer.hpp"
#include "Inspector.hpp"
#include "PathTracer.hpp"

#include <Core/Instance.hpp>
#include <Core/Profiler.hpp>

namespace stm2 {

ImageComparer::ImageComparer(Node& node) : mNode(node) {
	if (shared_ptr<Inspector> inspector = mNode.root()->findDescendant<Inspector>())
		inspector->setInspectCallback<ImageComparer>();

	const filesystem::path shaderPath(*mNode.root()->findDescendant<Instance>()->findArgument("shaderKernelPath"));
	mPipeline = ComputePipelineCache(shaderPath / "image_compare.slang");
}

void ImageComparer::update(CommandBuffer& commandBuffer) {
	ProfilerScope ps("ImageComparer::update");
	for (auto&[original, img] : mImages|views::values) {
		if (!img) {
			Image::Metadata md = original.image()->metadata();
			md.mUsage = vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eTransferSrc|vk::ImageUsageFlagBits::eTransferDst;
			img = make_shared<Image>(commandBuffer.mDevice, original.image()->resourceName() + "/ImageComparer Copy", md);
			Image::copy(commandBuffer, original, img);
			img.image()->generateMipMaps(commandBuffer);
		}
	}

	if (mComparing.empty()) return;

	if (ImGui::Begin("Compare result")) {
		if (mComparing.size() == 1) {
			mCurrent = *mComparing.begin();
		} else {
			bool s = false;
			for (auto c : mComparing) {
				if (s) ImGui::SameLine();
				s = true;
				if (ImGui::Button(c.c_str()))
					mCurrent = c;
			}

			if (mComparing.size() == 2) {
				auto& resultBuffer = mCompareResult[ *mComparing.begin() + "_" + *(++mComparing.begin())];

				bool update = !resultBuffer;
				ImGui::SameLine();
				ImGui::SetNextItemWidth(80);
				if (ImGui::DragScalar("Quantization", ImGuiDataType_U32, &mQuantization)) update = true;
				ImGui::SameLine();
				ImGui::SetNextItemWidth(160);
				if (Gui::enumDropdown<ImageCompareMode>("Error mode", mMode, (uint32_t)ImageCompareMode::eCompareMetricCount))
					update = true;

				if (update) {
					if (!resultBuffer)
						resultBuffer = make_shared<Buffer>(commandBuffer.mDevice, "ImageComparer/ResultBuffer", 2*sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
					resultBuffer[0] = 0;
					resultBuffer[1] = 0;

					const Defines defines {
						{"gMode", to_string((uint32_t)mMode)},
						{"gQuantization", to_string(mQuantization)}
					};

					{
						ProfilerScope ps("Image compare", &commandBuffer);
						mPipeline.get(commandBuffer.mDevice, defines)->dispatchTiled(commandBuffer, mImages.at(*mComparing.begin()).second.extent(), Descriptors{
							{ {"gImage1",0}, ImageDescriptor(mImages.at(*mComparing.begin()).second  , vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {}) },
							{ {"gImage2",0}, ImageDescriptor(mImages.at(*++mComparing.begin()).second, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {}) },
							{ {"gOutput",0}, resultBuffer },
						});
					}
				}

				if (resultBuffer) {
					ImGui::SameLine();
					if (resultBuffer[1])
						ImGui::Text("OVERFLOW");
					else
						ImGui::Text("%f", mMode == ImageCompareMode::eMSE ? sqrt(resultBuffer[0]/(float)mQuantization) : resultBuffer[0]/(float)mQuantization);
				}
			}
		}

		// image pan/zoom
		const uint32_t w = ImGui::GetWindowSize().x - 4;
		Image::View& img = mImages.at(mCurrent).second;
		commandBuffer.trackResource(img.image());
		const float aspect = (float)img.extent().height / (float)img.extent().width;
		ImVec2 uvMin(mOffset[0], mOffset[1]);
		ImVec2 uvMax(mOffset[0] + mZoom, mOffset[1] + mZoom);
		ImGui::Image(Gui::getTextureID(img), ImVec2(w, w * aspect), uvMin, uvMax);
		if (ImGui::IsItemHovered()) {
			mZoom *= 1 - 0.05f*ImGui::GetIO().MouseWheel;
			mZoom = clamp(mZoom, 0.f, 1.f);
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

void ImageComparer::drawGui() {
	if (ImGui::Button("Clear")) {
		mImages.clear();
		mComparing.clear();
		mCurrent.clear();
	}

	static char label[64];
	ImGui::InputText("##", label, sizeof(label));
	ImGui::SameLine();
	if (ImGui::Button("Save") && !string(label).empty()) {
		Image::View img;
		if (auto pt = mNode.root()->findDescendant<PathTracer>())
			img = pt->resultImage();
		if (img)
			mImages.emplace(label, pair<Image::View, Image::View>{ img, {} });
	}

	for (auto it = mImages.begin(); it != mImages.end(); ) {
		ImGui::PushID(&it->second);
		const bool d = ImGui::Button("x");
		bool c = mComparing.find(it->first) != mComparing.end();
		ImGui::PopID();
		if (d) {
			if (c) mComparing.erase(it->first);
			it = mImages.erase(it);
			continue;
		}
		ImGui::SameLine();
		if (ImGui::Checkbox(it->first.c_str(), &c)) {
			if (c)
				mComparing.emplace(it->first);
			else
				mComparing.erase(it->first);
		}
		it++;
	}
}

}