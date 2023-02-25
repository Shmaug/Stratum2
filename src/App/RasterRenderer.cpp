#include "RasterRenderer.hpp"
#include "Tonemapper.hpp"
#include "Gui.hpp"
#include "Inspector.hpp"

#include <stb_image_write.h>
#include <ImGuizmo.h>
#include <random>

#include <Core/Instance.hpp>
#include <Core/Swapchain.hpp>
#include <Core/CommandBuffer.hpp>
#include <Core/Profiler.hpp>

namespace stm2 {

RasterRenderer::RasterRenderer(Node& node) : mNode(node) {
	if (shared_ptr<Inspector> inspector = mNode.root()->findDescendant<Inspector>())
		inspector->setInspectCallback<RasterRenderer>();
}

void RasterRenderer::createPipelines(Device& device, const vk::Extent3D renderExtent, const vk::Format renderFormat) {
	const auto samplerRepeat = make_shared<vk::raii::Sampler>(*device, vk::SamplerCreateInfo({},
		vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear,
		vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat,
		0, true, 8, false, vk::CompareOp::eAlways, 0, VK_LOD_CLAMP_NONE));

	const vector<string>& args = {
		"-matrix-layout-row-major",
		"-O3",
		"-Wno-30081",
		"-capability", "spirv_1_5",
	};
	const filesystem::path shaderPath = *device.mInstance.findArgument("shaderKernelPath");
	const filesystem::path kernelPath = shaderPath / "vcm.slang";

	GraphicsPipeline::GraphicsMetadata gmd;
	gmd.mColorBlendState = GraphicsPipeline::ColorBlendState();
	gmd.mColorBlendState->mAttachments = { vk::PipelineColorBlendAttachmentState(
		false,
		vk::BlendFactor::eZero,
		vk::BlendFactor::eZero,
		vk::BlendOp::eAdd,
		vk::BlendFactor::eZero,
		vk::BlendFactor::eZero,
		vk::BlendOp::eAdd,
		vk::ColorComponentFlags{vk::FlagTraits<vk::ColorComponentFlagBits>::allFlags}) };

	gmd.mDynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };

	gmd.mDynamicRenderingState = GraphicsPipeline::DynamicRenderingState();
	gmd.mDynamicRenderingState->mColorFormats = { format };
	gmd.mDynamicRenderingState->mDepthFormat = vk::Format::eD32Sfloat;

	gmd.mViewports = { vk::Viewport(0, 0, extent.width, extent.height, 0, 1) };
	gmd.mScissors = { vk::Rect2D({0,0}, extent) };

	gmd.mVertexInputState   = vk::PipelineVertexInputStateCreateInfo();
	gmd.mInputAssemblyState = vk::PipelineInputAssemblyStateCreateInfo({}, vk::PrimitiveTopology::eTriangleList);
	gmd.mRasterizationState = vk::PipelineRasterizationStateCreateInfo(
		{},    // flags
		false, // depthClampEnable_
		false, // rasterizerDiscardEnable_
		vk::PolygonMode::eFill,
		vk::CullModeFlagBits::eNone,
		vk::FrontFace::eCounterClockwise,
		false, // depthBiasEnable_
		{}, // depthBiasConstantFactor_
		{}, // depthBiasClamp_
		{}, // depthBiasSlopeFactor_
		{}  // lineWidth_
	);
	gmd.mMultisampleState   = vk::PipelineMultisampleStateCreateInfo();
	gmd.mDepthStencilState  = vk::PipelineDepthStencilStateCreateInfo(
		{},    // flags_
		true,  // depthTestEnable_
		true,  // depthWriteEnable_
		vk::CompareOp::eGreater, // depthCompareOp_
		false, // depthBoundsTestEnable_
		false, // stencilTestEnable_
		{},    // front_
		{},    // back_
		{},    // minDepthBounds_
		{}     // maxDepthBounds_
	);

	const filesystem::path shaderPath = *device.mInstance.findArgument("shaderKernelPath");
	const filesystem::path rasterShaderPath = shaderPath / "vcm_vis.slang";
	const vector<string>& rasterArgs = {
		"-matrix-layout-row-major",
		"-O3",
		"-Wno-30081",
		"-capability", "spirv_1_5",
	};
	mRasterLightPathPipeline = GraphicsPipelineCache({
		{ vk::ShaderStageFlagBits::eVertex  , GraphicsPipelineCache::ShaderSourceInfo(rasterShaderPath, "LightVertexVS", "sm_6_6") },
		{ vk::ShaderStageFlagBits::eFragment, GraphicsPipelineCache::ShaderSourceInfo(rasterShaderPath, "LightVertexFS", "sm_6_6") }
	}, rasterArgs, gmd);
}

void RasterRenderer::drawGui() {
	ImGui::PushID(this);
	if (ImGui::Button("Clear resources")) {
		Device& device = *mNode.findAncestor<Device>();
		device->waitIdle();
		mResourcePool.clear();
	}
	if (ImGui::Button("Reload shaders")) {
		Device& device = *mNode.findAncestor<Device>();
		device->waitIdle();
		createPipelines(device);
	}
	ImGui::PopID();

	if (auto tonemapper = mNode.getComponent<Tonemapper>(); tonemapper)
		ImGui::Checkbox("Enable tonemapper", &mTonemap);
}

void RasterRenderer::render(CommandBuffer& commandBuffer, const Image::View& renderTarget) {
	if (renderTarget.image()->format() != mRasterLightPathPipeline.pipelineMetadata().mDynamicRenderingState->mColorFormats[0])
		createPipelines(commandBuffer.mDevice, vk::Extent2D(renderTarget.extent().width, renderTarget.extent().height), renderTarget.image()->format());

	Descriptors descriptors;

	for (auto d : {
		"mViews",
		"mViewTransforms",
		"mViewInverseTransforms",
		"mLightVertices",
		"mLightPathLengths" }) {
		descriptors[{string("gParams.")+d,0}] = frame.mBuffers.at(d);
	}
	descriptors[{"gParams.mDepth",0}] = ImageDescriptor{get<Image::View>(frame.mImages.at("mDepth")), vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {}};

	const vk::Extent3D extent = renderTarget.extent();

	if (!frame.mRasterDepthBuffer || extent.width > frame.mRasterDepthBuffer.extent().width || extent.height > frame.mRasterDepthBuffer.extent().height) {
		Image::Metadata md = {};
		md.mExtent = extent;
		md.mFormat = vk::Format::eD32Sfloat;
		md.mUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment|vk::ImageUsageFlagBits::eTransferDst;
		frame.mRasterDepthBuffer = make_shared<Image>(commandBuffer.mDevice, "gRasterDepthBuffer", md);
	}

	auto lightPathPipeline = mRasterLightPathPipeline.get(commandBuffer.mDevice);
	auto descriptorSets = lightPathPipeline->getDescriptorSets(descriptors);

	// render light paths
	{
		renderTarget.barrier(commandBuffer, vk::ImageLayout::eColorAttachmentOptimal, vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::AccessFlagBits::eColorAttachmentWrite);
		frame.mRasterDepthBuffer.barrier(commandBuffer, vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::PipelineStageFlagBits::eEarlyFragmentTests, vk::AccessFlagBits::eDepthStencilAttachmentRead|vk::AccessFlagBits::eDepthStencilAttachmentWrite);

		descriptorSets->transitionImages(commandBuffer);

		vk::RenderingAttachmentInfo colorAttachment(
			*renderTarget, vk::ImageLayout::eColorAttachmentOptimal,
			vk::ResolveModeFlagBits::eNone,	{}, vk::ImageLayout::eUndefined,
			vk::AttachmentLoadOp::eLoad,
			vk::AttachmentStoreOp::eStore,
			vk::ClearValue{});
		vk::RenderingAttachmentInfo depthAttachment(
			*frame.mRasterDepthBuffer, vk::ImageLayout::eDepthStencilAttachmentOptimal,
			vk::ResolveModeFlagBits::eNone,	{}, vk::ImageLayout::eUndefined,
			vk::AttachmentLoadOp::eClear,
			vk::AttachmentStoreOp::eDontCare,
			vk::ClearValue{vk::ClearDepthStencilValue{0,0}});
		commandBuffer->beginRendering(vk::RenderingInfo(
			vk::RenderingFlags{},
			vk::Rect2D(vk::Offset2D(0,0), vk::Extent2D(extent.width, extent.height)),
			1, 0, colorAttachment, &depthAttachment, nullptr));

		commandBuffer->setViewport(0, vk::Viewport(0, 0, extent.width, extent.height, 0, 1));
		commandBuffer->setScissor(0, vk::Rect2D(vk::Offset2D(0,0), vk::Extent2D(extent.width, extent.height)));

		commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, ***lightPathPipeline);
		descriptorSets->bind(commandBuffer);
		lightPathPipeline->pushConstants(commandBuffer, {
			{ "mLightSubPathCount", mPushConstants.mLightSubPathCount },
			{ "mHashGridCellCount", mPushConstants.mHashGridCellCount },
			{ "mSegmentIndex", mVisualizeSegmentIndex },
			{ "mLineRadius", mVisualizeLightPathRadius },
			{ "mLineLength", mVisualizeLightPathLength },
			{ "mMergeRadius", frame.mBuffers.at("mVcmConstants").cast<VcmConstants>()[0].mMergeRadius },
		});
		commandBuffer->draw((mPushConstants.mMaxPathLength+1)*6, min(mVisualizeLightPathCount, mPushConstants.mLightSubPathCount), 0, 0);

		commandBuffer.trackResource(lightPathPipeline);
		commandBuffer.trackResource(descriptorSets);

		commandBuffer->endRendering();
	}
}

}