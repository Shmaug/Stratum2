#include "Denoiser.hpp"
#include "Gui.hpp"
#include "Inspector.hpp"

#include <Core/Instance.hpp>
#include <Core/Swapchain.hpp>
#include <Core/CommandBuffer.hpp>
#include <Core/Profiler.hpp>

#include <random>

#include <Shaders/compat/filter_type.h>

namespace stm2 {


Denoiser::Denoiser(Node& node) : mNode(node) {
	if (shared_ptr<Inspector> inspector = mNode.root()->findDescendant<Inspector>())
		inspector->setInspectCallback<Denoiser>();

	Device& device = *mNode.findAncestor<Device>();
	createPipelines(device);

	mReprojection = true;
	mCheckNormal = true;
	mCheckDepth = true;

	const auto instance = mNode.findAncestor<Instance>();
	if (instance->findArgument("noReprojection")) mReprojection = false;
	if (instance->findArgument("noNormalCheck")) mCheckNormal = false;
	if (instance->findArgument("noDepthCheck")) mCheckDepth = false;
}


void Denoiser::createPipelines(Device& device) {
	const filesystem::path shaderPath = filesystem::path(*device.mInstance.findArgument("shaderKernelPath")) / "svgf";
	mTemporalAccumulationPipeline = ComputePipelineCache(shaderPath / "temporal_accumulation.slang", "main"    , "cs_6_6");
	mEstimateVariancePipeline     = ComputePipelineCache(shaderPath / "estimate_variance.slang"    , "main"    , "cs_6_6");
	mAtrousPipeline               = ComputePipelineCache(shaderPath / "atrous.slang"               , "main"    , "cs_6_6");
	mCopyRGBPipeline              = ComputePipelineCache(shaderPath / "atrous.slang"               , "copy_rgb", "cs_6_6");
}

void Denoiser::drawGui() {
	ImGui::PushID(this);
	if (ImGui::Button("Clear resources")) {
		Device& device = *mNode.findAncestor<Device>();
		device->waitIdle();
		mResourcePool.clear();
		mPrevVisibility = {};
		mPrevDepth = {};
	}
	if (ImGui::Button("Reload shaders")) {
		Device& device = *mNode.findAncestor<Device>();
		device->waitIdle();
		createPipelines(device);
	}

	ImGui::Text("%u frames accumulated", mAccumulatedFrames);
	ImGui::SameLine();
	if (ImGui::Button("Reset"))
		resetAccumulation();

	if (ImGui::CollapsingHeader("Configuration")) {
		ImGui::SetNextItemWidth(200);
		Gui::enumDropdown<DenoiserDebugMode>("Debug Mode", mDebugMode, (uint32_t)DenoiserDebugMode::eDebugModeCount);

		if (ImGui::Checkbox("Demodulate albedo", &mDemodulateAlbedo))
			resetAccumulation();
		ImGui::Checkbox("Reprojection", &mReprojection);
		if (mReprojection) {
			ImGui::Checkbox("Check normal", &mCheckNormal);
			ImGui::Checkbox("Check depth", &mCheckDepth);
		}

		ImGui::PushItemWidth(40);
		ImGui::DragScalar("Target sample count", ImGuiDataType_U32, &mHistoryLimit);
		ImGui::DragScalar("Filter iterations", ImGuiDataType_U32, &mAtrousIterations, 0.1f);

		if (mAtrousIterations > 0) {
			ImGui::Indent();
			ImGui::DragScalar("Variance boost frames", ImGuiDataType_U32, &mVarianceBoostLength);
			ImGui::DragFloat("Sigma luminance boost", &mSigmaLuminanceBoost, .1f, 0, 0, "%.2f");
			ImGui::PopItemWidth();
			Gui::enumDropdown<FilterKernelType>("Filter", mFilterType, (uint32_t)FilterKernelType::eFilterKernelTypeCount);
			ImGui::PushItemWidth(40);
			ImGui::DragScalar("History tap iteration", ImGuiDataType_U32, &mHistoryTap, 0.1f);
			ImGui::Unindent();
		}
		ImGui::PopItemWidth();
	}

	if (ImGui::CollapsingHeader("Resources")) {
		ImGui::Indent();
		mResourcePool.drawGui();
		ImGui::Unindent();
	}
	ImGui::PopID();
}

Image::View Denoiser::denoise(
	CommandBuffer& commandBuffer,
	const Image::View& radiance,
	const Image::View& albedo,
	const Image::View& prevUVs,
	const Image::View& visibility,
	const Image::View& depth,
	const Buffer::View<ViewData>& views) {
	ProfilerScope ps("Denoiser::denoise", &commandBuffer);

	mResourcePool.clean();

	if (ImGui::IsKeyPressed(ImGuiKey_F5))
		mResetAccumulation = true;

	// Initialize resources

	const vk::Extent3D extent = radiance.extent();

	Image::View accumColor, accumMoments;
	array<Image::View,2> temp;
	auto prevAccumColor   = mResourcePool.getLastImage("mAccumColor");
	auto prevAccumMoments = mResourcePool.getLastImage("mAccumMoments");

	{
		ProfilerScope ps("Create images");

		accumColor = mResourcePool.getImage(commandBuffer.mDevice, "mAccumColor", Image::Metadata{
			.mFormat = radiance.image()->format(),
			.mExtent = extent,
			.mUsage = vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eTransferSrc|vk::ImageUsageFlagBits::eTransferDst,
		});

		accumMoments = mResourcePool.getImage(commandBuffer.mDevice, "mAccumMoments", Image::Metadata{
			.mFormat = vk::Format::eR32G32Sfloat,
			.mExtent = extent,
			.mUsage = vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eTransferSrc|vk::ImageUsageFlagBits::eTransferDst,
		});

		for (uint32_t i = 0; i < temp.size(); i++)
			temp[i] = mResourcePool.getImage(commandBuffer.mDevice, "mTemp" + to_string(i), Image::Metadata{
				.mFormat = radiance.image()->format(),
				.mExtent = extent,
				.mUsage = vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eTransferSrc|vk::ImageUsageFlagBits::eTransferDst,
			}, 0);

		accumColor  .clearColor(commandBuffer, vk::ClearColorValue{ array<float,4>{ 0.f, 0.f, 0.f, 0.f } });
		accumMoments.clearColor(commandBuffer, vk::ClearColorValue{ array<float,4>{ 0.f, 0.f, 0.f, 0.f } });
	}

	Image::View output = radiance;

	Defines defines {
		{"gReprojection"    , mReprojection     ? "true" : "false" },
		{"gDemodulateAlbedo", mDemodulateAlbedo ? "true" : "false" },
		{"gCheckNormal"     , mCheckNormal      ? "true" : "false" },
		{"gCheckDepth"      , mCheckDepth       ? "true" : "false" },
		{"gFilterKernelType", to_string((uint32_t)mFilterType) },
		{"gDebugMode", "(DenoiserDebugMode)" + to_string((uint32_t)mDebugMode) },
	};

	auto instanceIndexMap = mNode.findAncestor<Scene>()->frameData().mResourcePool.getLastBuffer<uint32_t>("mInstanceIndexMap");
	if (!mResetAccumulation && instanceIndexMap && prevAccumColor && prevAccumMoments && mPrevVisibility && mPrevVisibility.extent() == visibility.extent()) {
		Descriptors descriptors;
		descriptors[{"gParams.mViews",0}]            = views;
		descriptors[{"gParams.mInstanceIndexMap",0}] = instanceIndexMap;
		descriptors[{"gParams.mVisibility",0}]       = ImageDescriptor{ visibility       , vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{"gParams.mPrevVisibility",0}]   = ImageDescriptor{ mPrevVisibility  , vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{"gParams.mDepth",0}]            = ImageDescriptor{ depth            , vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{"gParams.mPrevDepth",0}]        = ImageDescriptor{ mPrevDepth       , vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{"gParams.mPrevUVs",0}]          = ImageDescriptor{ prevUVs          , vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{"gParams.mInput",0}]            = ImageDescriptor{ radiance         , vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{"gParams.mAlbedo",0}]           = ImageDescriptor{ albedo           , vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{"gParams.mAccumColor",0}]       = ImageDescriptor{ accumColor       , vk::ImageLayout::eGeneral              , vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite, {} };
		descriptors[{"gParams.mAccumMoments",0}]     = ImageDescriptor{ accumMoments     , vk::ImageLayout::eGeneral              , vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite, {} };
		descriptors[{"gParams.mFilterImages", 0}]    = ImageDescriptor{ temp[0]          , vk::ImageLayout::eGeneral              , vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite, {} };
		descriptors[{"gParams.mFilterImages", 1}]    = ImageDescriptor{ temp[1]          , vk::ImageLayout::eGeneral              , vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite, {} };
		descriptors[{"gParams.mPrevAccumColor",0}]   = ImageDescriptor{ prevAccumColor   , vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{"gParams.mPrevAccumMoments",0}] = ImageDescriptor{ prevAccumMoments , vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };

		output = accumColor;

		{ // temporal accumulation
			ProfilerScope ps("Temporal accumulation", &commandBuffer);
			auto accumulationPipeline = mTemporalAccumulationPipeline.get(commandBuffer.mDevice, defines);
			accumulationPipeline->dispatchTiled(commandBuffer, extent, mResourcePool.getDescriptorSets(*accumulationPipeline, "AccumulationDescriptors", descriptors), {}, {
				{ "mViewCount", PushConstantValue((uint32_t)views.size()) },
				{ "mHistoryLimit", PushConstantValue(mHistoryLimit) },
			});
		}

		if (mAtrousIterations > 0) {

			{ // estimate variance
				temp[0].barrier(commandBuffer, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderWrite);
				accumColor.barrier(commandBuffer, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);
				accumMoments.barrier(commandBuffer, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);

				ProfilerScope ps("Estimate variance", &commandBuffer);
				auto estimateVariancePipeline = mEstimateVariancePipeline.get(commandBuffer.mDevice, defines);
				estimateVariancePipeline->dispatchTiled(commandBuffer, extent, mResourcePool.getDescriptorSets(*estimateVariancePipeline, "EstimateVarianceDescriptors", descriptors), {}, {
					{ "mViewCount", PushConstantValue((uint32_t)views.size()) },
					{ "mHistoryLimit", PushConstantValue(mHistoryLimit) },
					{ "mVarianceBoostLength", PushConstantValue(mVarianceBoostLength) },
				});
			}

			ProfilerScope ps("Filter image", &commandBuffer);
			auto atrousPipeline = mAtrousPipeline.get(commandBuffer.mDevice, defines);
			auto atrousDescriptorSets = mResourcePool.getDescriptorSets(*atrousPipeline, "AtrousDescriptors", descriptors);

			for (uint32_t i = 0; i < mAtrousIterations; i++) {
				temp[0].barrier(commandBuffer, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);
				temp[1].barrier(commandBuffer, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);

				atrousPipeline->dispatchTiled(commandBuffer, extent, atrousDescriptorSets, {}, {
					{ "mViewCount", PushConstantValue((uint32_t)views.size()) },
					{ "mSigmaLuminanceBoost", PushConstantValue(mSigmaLuminanceBoost) },
					{ "mIteration", PushConstantValue(i) },
					{ "mStepSize", PushConstantValue(1 << i) },
				});

				if (i+1 == mHistoryTap) {
					// copy rgb (not alpha channel) to AccumColor
					temp[0].barrier(commandBuffer, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);
					temp[1].barrier(commandBuffer, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);
					auto copyRgb = mCopyRGBPipeline.get(commandBuffer.mDevice, defines);
					copyRgb->dispatchTiled(commandBuffer, extent, mResourcePool.getDescriptorSets(*copyRgb, "CopyRGBDescriptors", descriptors));
				}
			}
			output = temp[mAtrousIterations%2];
		}
		mAccumulatedFrames++;
	} else {
		accumColor.clearColor(commandBuffer, vk::ClearColorValue{ array<float,4>{ 0.f, 0.f, 0.f, 0.f } });
		mResetAccumulation = false;
		mAccumulatedFrames = 0;
	}

	mPrevVisibility = visibility;
	mPrevDepth = depth;

	return output;
}

}