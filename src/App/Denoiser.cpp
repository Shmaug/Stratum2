#include "Denoiser.hpp"
#include "Gui.hpp"
#include "Inspector.hpp"

#include <Core/Instance.hpp>
#include <Core/CommandBuffer.hpp>
#include <Core/Profiler.hpp>

#include <random>

#include <Shaders/compat/filter_type.h>

namespace stm2 {


Denoiser::Denoiser(Node& node) : mNode(node) {
	if (shared_ptr<Inspector> inspector = mNode.root()->findDescendant<Inspector>())
		inspector->setTypeCallback<Denoiser>();
}


void Denoiser::createPipelines(Device& device) {
	auto samplerClamp = make_shared<vk::raii::Sampler>(*device, vk::SamplerCreateInfo({},
		vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear,
		vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge,
		0, true, 8, false, vk::CompareOp::eAlways, 0, VK_LOD_CLAMP_NONE));

	ComputePipeline::Metadata md;
	md.mImmutableSamplers["mStaticSampler"] = { samplerClamp };

	const filesystem::path shaderPath = filesystem::path(*device.mInstance.findArgument("shaderKernelPath")) / "svgf";
	mTemporalAccumulationPipeline = ComputePipelineCache(shaderPath / "temporal_accumulation.hlsl", "main", "cs_6_7", {}, md);
	mEstimateVariancePipeline     = ComputePipelineCache(shaderPath / "estimate_variance.hlsl", "main", "cs_6_7", {}, md);
	mAtrousPipeline               = ComputePipelineCache(shaderPath / "atrous.hlsl", "main", "cs_6_7", {}, md);
	mCopyRGBPipeline              = ComputePipelineCache(shaderPath / "atrous.hlsl", "copy_rgb", "cs_6_7", {}, md);

	mPushConstants["gHistoryLimit"] = 0.f;
	mPushConstants["gSigmaLuminanceBoost"] = 3.f;
}

void Denoiser::drawGui() {
	if (ImGui::Button("Reload Denoiser Shaders")) {
		Device& device = *mNode.findAncestor<Device>();
		device->waitIdle();
		createPipelines(device);
	}

	ImGui::SetNextItemWidth(200);
	Gui::enumDropdown<DenoiserDebugMode>("Denoiser Debug Mode", mDebugMode, (uint32_t)DenoiserDebugMode::eDebugModeCount);

	ImGui::Text("%u frames accumulated", mAccumulatedFrames);
	ImGui::SameLine();
	if (ImGui::Button("Reset Accumulation"))
		resetAccumulation();

	ImGui::Checkbox("Reprojection", &mReprojection);
	if (ImGui::Checkbox("Demodulate Albedo", &mDemodulateAlbedo))
		resetAccumulation();

	ImGui::PushItemWidth(40);
	ImGui::DragFloat("Target Sample Count", &mPushConstants["gHistoryLimit"].get<float>());
	ImGui::DragScalar("Filter Iterations", ImGuiDataType_U32, &mAtrousIterations, 0.1f);

	if (mAtrousIterations > 0) {
		ImGui::Indent();
		ImGui::DragFloat("Variance Boost Frames", &mPushConstants["gVarianceBoostLength"].get<float>());
		ImGui::DragFloat("Sigma Luminance Boost", &mPushConstants["gSigmaLuminanceBoost"].get<float>(), .1f, 0, 0, "%.2f");
		ImGui::PopItemWidth();
		Gui::enumDropdown<FilterKernelType>("Filter", mFilterType, (uint32_t)FilterKernelType::eFilterKernelTypeCount);
		ImGui::PushItemWidth(40);
		ImGui::DragScalar("History Tap Iteration", ImGuiDataType_U32, &mHistoryTap, 0.1f);
		ImGui::Unindent();
	}
	ImGui::PopItemWidth();
}

void Denoiser::denoise(
	CommandBuffer& commandBuffer,
	const Image::View& radiance,
	const Image::View& albedo,
	const Image::View& prevUVs,
	const Buffer::View<ViewData>& views,
	const Buffer::View<VisibilityData>& visibility,
	const Buffer::View<DepthData>& depth) {
	ProfilerScope ps("Denoiser::denoise", &commandBuffer);

	if (ImGui::IsKeyPressed(ImGuiKey_F5))
		mResetAccumulation = true;

	// Initialize resources

	{
		ProfilerScope ps("Allocate Frame Resources", &commandBuffer);
		if (mCurFrame)
			mPrevFrame = mCurFrame;

		mCurFrame = mFrameResources.get();
		if (!mCurFrame)
			mCurFrame = mFrameResources.emplace(make_shared<FrameResources>(commandBuffer.mDevice));
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

		mCurFrame->mAccumColor   = make_shared<Image>(commandBuffer.mDevice, "gAccumColor", radiance.image()->metadata());
		mCurFrame->mAccumMoments = make_shared<Image>(commandBuffer.mDevice, "gAccumMoments", md);
		for (uint32_t i = 0; i < mCurFrame->mTemp.size(); i++)
			mCurFrame->mTemp[i]  = make_shared<Image>(commandBuffer.mDevice, "pingpong" + to_string(i), radiance.image()->metadata());
		mCurFrame->mDebugImage   = make_shared<Image>(commandBuffer.mDevice, "gDebugImage", radiance.image()->metadata());

		mCurFrame->mAccumColor.clearColor(commandBuffer  , vk::ClearColorValue{ array<float,4>{ 0.f, 0.f, 0.f, 0.f } });
		mCurFrame->mAccumMoments.clearColor(commandBuffer, vk::ClearColorValue{ array<float,4>{ 0.f, 0.f, 0.f, 0.f } });
	}

	Image::View output = radiance;

	PushConstants pushConstants = mPushConstants;
	pushConstants["gViewCount"] = (uint32_t)views.size();

	Defines defines;
	// TODO: defines

	if (!mResetAccumulation && mPrevFrame && mPrevFrame->mRadiance && mPrevFrame->mRadiance.extent() == mCurFrame->mRadiance.extent()) {
		Descriptors descriptors;
		descriptors[{"gViews",0}]            = mCurFrame->mViews;
		descriptors[{"gInstanceIndexMap",0}] = mNode.findAncestor<Scene>()->resources()->mInstanceIndexMap;
		descriptors[{"gVisibility",0}]       = mCurFrame->mVisibility;
		descriptors[{"gPrevVisibility",0}]   = mPrevFrame->mVisibility;
		descriptors[{"gDepth",0}]            = mCurFrame->mDepth;
		descriptors[{"gPrevDepth",0}]        = mPrevFrame->mDepth;
		descriptors[{"gPrevUVs",0}]          = ImageDescriptor{ prevUVs                  , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite, {} };
		descriptors[{"gRadiance",0}]         = ImageDescriptor{ mCurFrame->mRadiance     , vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{"gAlbedo",0}]           = ImageDescriptor{ mCurFrame->mAlbedo       , vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{"gAccumColor",0}]       = ImageDescriptor{ mCurFrame->mAccumColor   , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite, {} };
		descriptors[{"gAccumMoments",0}]     = ImageDescriptor{ mCurFrame->mAccumMoments , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite, {} };
		descriptors[{"gFilterImages", 0}]    = ImageDescriptor{ mCurFrame->mTemp[0]      , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite, {} };
		descriptors[{"gFilterImages", 1}]    = ImageDescriptor{ mCurFrame->mTemp[1]      , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite, {} };
		descriptors[{"gPrevAccumColor",0}]   = ImageDescriptor{ mPrevFrame->mAccumColor  , vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{"gPrevAccumMoments",0}] = ImageDescriptor{ mPrevFrame->mAccumMoments, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{"gDebugImage",0}]       = ImageDescriptor{ mCurFrame->mDebugImage   , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };

		auto temporalAccumulation = mTemporalAccumulationPipeline.get(commandBuffer.mDevice, defines);
		mCurFrame->mDescriptors = temporalAccumulation->getDescriptorSets(descriptors);
		output = mCurFrame->mAccumColor;

		mCurFrame->mVisibility.barrier(commandBuffer, vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);

		{ // temporal accumulation
			ProfilerScope ps("Temporal accumulation", &commandBuffer);
			temporalAccumulation->dispatchTiled(commandBuffer, extent, mCurFrame->mDescriptors, {}, pushConstants);
		}

		if (mAtrousIterations > 0) {
			{ // estimate variance
				mCurFrame->mTemp[0].barrier(commandBuffer, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderWrite);
				mCurFrame->mAccumColor.barrier(commandBuffer, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);
				mCurFrame->mAccumMoments.barrier(commandBuffer, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);

				ProfilerScope ps("Estimate variance", &commandBuffer);
				mEstimateVariancePipeline.get(commandBuffer.mDevice, defines)->dispatchTiled(commandBuffer, extent, mCurFrame->mDescriptors, {}, pushConstants);
			}

			ProfilerScope ps("Filter image", &commandBuffer);
			auto atrous = mAtrousPipeline.get(commandBuffer.mDevice, defines);

			for (uint32_t i = 0; i < mAtrousIterations; i++) {
				mCurFrame->mTemp[0].barrier(commandBuffer, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);
				mCurFrame->mTemp[1].barrier(commandBuffer, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);

				pushConstants["gIteration"] = i;
				pushConstants["gStepSize"] = 1 << i;
				atrous->dispatchTiled(commandBuffer, extent, mCurFrame->mDescriptors, {}, pushConstants);

				if (i+1 == mHistoryTap) {
					// copy rgb (keep w) to AccumColor
					mCurFrame->mTemp[0].barrier(commandBuffer, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);
					mCurFrame->mTemp[1].barrier(commandBuffer, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);
					mCopyRGBPipeline.get(commandBuffer.mDevice, defines)->dispatchTiled(commandBuffer, extent, mCurFrame->mDescriptors, {}, {});
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
}

}