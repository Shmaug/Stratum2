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
		inspector->setTypeCallback<Denoiser>();

	Device& device = *mNode.findAncestor<Device>();
	createPipelines(device);
}


void Denoiser::createPipelines(Device& device) {
	auto samplerClamp = make_shared<vk::raii::Sampler>(*device, vk::SamplerCreateInfo({},
		vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear,
		vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge,
		0, true, 8, false, vk::CompareOp::eAlways, 0, VK_LOD_CLAMP_NONE));

	ComputePipeline::Metadata md;
	md.mImmutableSamplers["mStaticSampler"] = { samplerClamp };

	const filesystem::path shaderPath = filesystem::path(*device.mInstance.findArgument("shaderKernelPath")) / "svgf";
	mTemporalAccumulationPipeline = ComputePipelineCache(shaderPath / "temporal_accumulation.slang", "main"    , "cs_6_6", {}, md);
	mEstimateVariancePipeline     = ComputePipelineCache(shaderPath / "estimate_variance.slang"    , "main"    , "cs_6_6", {}, md);
	mAtrousPipeline               = ComputePipelineCache(shaderPath / "atrous.slang"               , "main"    , "cs_6_6", {}, md);
	mCopyRGBPipeline              = ComputePipelineCache(shaderPath / "atrous.slang"               , "copy_rgb", "cs_6_6", {}, md);
}

void Denoiser::drawGui() {
	ImGui::PushID(this);
	if (ImGui::Button("Reload shaders")) {
		Device& device = *mNode.findAncestor<Device>();
		device->waitIdle();
		createPipelines(device);
	}

	ImGui::SetNextItemWidth(200);
	Gui::enumDropdown<DenoiserDebugMode>("Debug Mode", mDebugMode, (uint32_t)DenoiserDebugMode::eDebugModeCount);
	ImGui::PopID();

	ImGui::Text("%u frames accumulated", mAccumulatedFrames);
	ImGui::SameLine();
	if (ImGui::Button("Reset accumulation"))
		resetAccumulation();

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

Image::View Denoiser::denoise(
	CommandBuffer& commandBuffer,
	const Image::View& radiance,
	const Image::View& albedo,
	const Image::View& prevUVs,
	const Image::View& visibility,
	const Image::View& depth,
	const Buffer::View<ViewData>& views) {
	ProfilerScope ps("Denoiser::denoise", &commandBuffer);

	if (ImGui::IsKeyPressed(ImGuiKey_F5))
		mResetAccumulation = true;

	// Initialize resources

	{
		ProfilerScope ps("Allocate Frame Resources", &commandBuffer);
		if (mNode.root()->findDescendant<Swapchain>()->imageCount() <= 1) {
			if (!mPrevFrame) mPrevFrame = make_shared<FrameResources>(commandBuffer.mDevice);
			if (!mCurFrame)  mCurFrame  = make_shared<FrameResources>(commandBuffer.mDevice);
			swap(mCurFrame, mPrevFrame);
		} else {
			if (mCurFrame)
				mPrevFrame = mCurFrame;

			mCurFrame = mFrameResources.get();
			if (!mCurFrame)
				mCurFrame = mFrameResources.emplace(make_shared<FrameResources>(commandBuffer.mDevice));
		}
	}

	commandBuffer.trackResource(mCurFrame);

	mCurFrame->mViews = views;
	mCurFrame->mRadiance = radiance;
	mCurFrame->mAlbedo = albedo;
	mCurFrame->mVisibility = visibility;
	mCurFrame->mDepth = depth;

	const vk::Extent3D extent = radiance.extent();
	if (!mCurFrame->mAccumColor || mCurFrame->mAccumColor.extent() != extent) {
		ProfilerScope ps("Create images");

		Image::Metadata md = radiance.image()->metadata();
		md.mFormat = vk::Format::eR32G32Sfloat;

		mCurFrame->mAccumColor   = make_shared<Image>(commandBuffer.mDevice, "mAccumColor", radiance.image()->metadata());
		mCurFrame->mAccumMoments = make_shared<Image>(commandBuffer.mDevice, "mAccumMoments", md);
		for (uint32_t i = 0; i < mCurFrame->mTemp.size(); i++)
			mCurFrame->mTemp[i]  = make_shared<Image>(commandBuffer.mDevice, "mTemp" + to_string(i), radiance.image()->metadata());

		mCurFrame->mAccumColor.clearColor(commandBuffer  , vk::ClearColorValue{ array<float,4>{ 0.f, 0.f, 0.f, 0.f } });
		mCurFrame->mAccumMoments.clearColor(commandBuffer, vk::ClearColorValue{ array<float,4>{ 0.f, 0.f, 0.f, 0.f } });
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

	if (!mResetAccumulation && mPrevFrame && mPrevFrame->mRadiance && mPrevFrame->mRadiance.extent() == mCurFrame->mRadiance.extent()) {
		Descriptors descriptors;
		descriptors[{"gParams.mViews",0}]            = mCurFrame->mViews;
		descriptors[{"gParams.mInstanceIndexMap",0}] = mNode.findAncestor<Scene>()->resources()->mInstanceIndexMap;
		descriptors[{"gParams.mVisibility",0}]       = ImageDescriptor{ mCurFrame->mVisibility   , vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{"gParams.mPrevVisibility",0}]   = ImageDescriptor{ mPrevFrame->mVisibility  , vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{"gParams.mDepth",0}]            = ImageDescriptor{ mCurFrame->mDepth        , vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{"gParams.mPrevDepth",0}]        = ImageDescriptor{ mPrevFrame->mDepth       , vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{"gParams.mPrevUVs",0}]          = ImageDescriptor{ prevUVs                  , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite, {} };
		descriptors[{"gParams.mInput",0}]            = ImageDescriptor{ mCurFrame->mRadiance     , vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{"gParams.mAlbedo",0}]           = ImageDescriptor{ mCurFrame->mAlbedo       , vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{"gParams.mAccumColor",0}]       = ImageDescriptor{ mCurFrame->mAccumColor   , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite, {} };
		descriptors[{"gParams.mAccumMoments",0}]     = ImageDescriptor{ mCurFrame->mAccumMoments , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite, {} };
		descriptors[{"gParams.mFilterImages", 0}]    = ImageDescriptor{ mCurFrame->mTemp[0]      , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite, {} };
		descriptors[{"gParams.mFilterImages", 1}]    = ImageDescriptor{ mCurFrame->mTemp[1]      , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite, {} };
		descriptors[{"gParams.mPrevAccumColor",0}]   = ImageDescriptor{ mPrevFrame->mAccumColor  , vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{"gParams.mPrevAccumMoments",0}] = ImageDescriptor{ mPrevFrame->mAccumMoments, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };


		//mCurFrame->mDescriptors = temporalAccumulation->getDescriptorSets(descriptors);

		output = mCurFrame->mAccumColor;

		{ // temporal accumulation
			ProfilerScope ps("Temporal accumulation", &commandBuffer);
			auto temporalAccumulation = mTemporalAccumulationPipeline.get(commandBuffer.mDevice, defines);
			temporalAccumulation->dispatchTiled(commandBuffer, extent, descriptors, {}, {
				{ "mViewCount", PushConstantValue((uint32_t)views.size()) },
				{ "mHistoryLimit", PushConstantValue(mHistoryLimit) },
			});
		}

		if (mAtrousIterations > 0) {
			{ // estimate variance
				mCurFrame->mTemp[0].barrier(commandBuffer, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderWrite);
				mCurFrame->mAccumColor.barrier(commandBuffer, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);
				mCurFrame->mAccumMoments.barrier(commandBuffer, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);

				ProfilerScope ps("Estimate variance", &commandBuffer);
				auto estimateVariance = mEstimateVariancePipeline.get(commandBuffer.mDevice, defines);
				estimateVariance->dispatchTiled(commandBuffer, extent, descriptors, {}, {
					{ "mViewCount", PushConstantValue((uint32_t)views.size()) },
					{ "mHistoryLimit", PushConstantValue(mHistoryLimit) },
					{ "mVarianceBoostLength", PushConstantValue(mVarianceBoostLength) },
				});
			}

			ProfilerScope ps("Filter image", &commandBuffer);
			auto atrousPipeline = mAtrousPipeline.get(commandBuffer.mDevice, defines);
			auto atrousDescriptors = atrousPipeline->getDescriptorSets(descriptors);

			for (uint32_t i = 0; i < mAtrousIterations; i++) {
				mCurFrame->mTemp[0].barrier(commandBuffer, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);
				mCurFrame->mTemp[1].barrier(commandBuffer, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);

				atrousPipeline->dispatchTiled(commandBuffer, extent, atrousDescriptors, {}, {
					{ "mViewCount", PushConstantValue((uint32_t)views.size()) },
					{ "mSigmaLuminanceBoost", PushConstantValue(mSigmaLuminanceBoost) },
					{ "mIteration", PushConstantValue(i) },
					{ "mStepSize", PushConstantValue(1 << i) },
				});

				if (i+1 == mHistoryTap) {
					// copy rgb (keep w) to AccumColor
					mCurFrame->mTemp[0].barrier(commandBuffer, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);
					mCurFrame->mTemp[1].barrier(commandBuffer, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);
					auto copyRgb = mCopyRGBPipeline.get(commandBuffer.mDevice, defines);
					copyRgb->dispatchTiled(commandBuffer, extent, descriptors, {}, {});
				}
			}
			output = mCurFrame->mTemp[mAtrousIterations%2];
		}
		mAccumulatedFrames++;
	} else {
		mCurFrame->mAccumColor.clearColor(commandBuffer, vk::ClearColorValue{ array<float,4>{ 0.f, 0.f, 0.f, 0.f } });
		mResetAccumulation = false;
		mAccumulatedFrames = 0;
	}

	return output;
}

}