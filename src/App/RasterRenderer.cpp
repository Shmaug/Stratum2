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

void RasterRenderer::createPipelines(Device& device, const vk::Format renderFormat) {
	vk::PipelineColorBlendAttachmentState blendState(
		false,
		vk::BlendFactor::eZero,
		vk::BlendFactor::eZero,
		vk::BlendOp::eAdd,
		vk::BlendFactor::eZero,
		vk::BlendFactor::eZero,
		vk::BlendOp::eAdd,
		vk::ColorComponentFlags{vk::FlagTraits<vk::ColorComponentFlagBits>::allFlags});

	GraphicsPipeline::GraphicsMetadata gmd;
	gmd.mColorBlendState = GraphicsPipeline::ColorBlendState();
	gmd.mColorBlendState->mAttachments = { blendState, blendState };

	gmd.mDynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };

	gmd.mDynamicRenderingState = GraphicsPipeline::DynamicRenderingState();
	gmd.mDynamicRenderingState->mColorFormats = { renderFormat, vk::Format::eR32G32Uint };
	gmd.mDynamicRenderingState->mDepthFormat = vk::Format::eD32Sfloat;

	gmd.mViewports = { vk::Viewport(0, 0, 0, 0, 0, 1) };
	gmd.mScissors  = { vk::Rect2D({ 0, 0 }, { 0, 0 }) };

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

	gmd.mImmutableSamplers["gScene.mStaticSampler"] = { make_shared<vk::raii::Sampler>(*device, vk::SamplerCreateInfo({},
		vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear,
		vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat,
		0, true, 8, false, vk::CompareOp::eAlways, 0, VK_LOD_CLAMP_NONE)) };
	gmd.mBindingFlags["gScene.mVertexBuffers"] = vk::DescriptorBindingFlagBits::ePartiallyBound;
	gmd.mBindingFlags["gScene.mImages"]  = vk::DescriptorBindingFlagBits::ePartiallyBound;
	gmd.mBindingFlags["gScene.mImage2s"] = vk::DescriptorBindingFlagBits::ePartiallyBound;
	gmd.mBindingFlags["gScene.mImage1s"] = vk::DescriptorBindingFlagBits::ePartiallyBound;
	gmd.mBindingFlags["gScene.mVolumes"] = vk::DescriptorBindingFlagBits::ePartiallyBound;

	const filesystem::path shaderPath = *device.mInstance.findArgument("shaderKernelPath");
	const filesystem::path rasterShaderPath = shaderPath / "raster_scene.slang";
	const vector<string>& rasterArgs = {
		"-matrix-layout-row-major",
		"-O3",
		"-Wno-30081",
		"-capability", "spirv_1_5",
	};
	mRasterPipeline = GraphicsPipelineCache({
		{ vk::ShaderStageFlagBits::eVertex  , GraphicsPipelineCache::ShaderSourceInfo(rasterShaderPath, "vsmain", "sm_6_6") },
		{ vk::ShaderStageFlagBits::eFragment, GraphicsPipelineCache::ShaderSourceInfo(rasterShaderPath, "fsmain", "sm_6_6") }
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
		mRasterPipeline.clear();
	}
	ImGui::PopID();

	ImGui::Checkbox("Alpha masks", &mAlphaMasks);

	if (auto tonemapper = mNode.getComponent<Tonemapper>(); tonemapper)
		ImGui::Checkbox("Enable tonemapper", &mTonemap);
}

void RasterRenderer::render(CommandBuffer& commandBuffer, const Image::View& renderTarget) {
	ProfilerScope ps("RasterRenderer::render", &commandBuffer);

	if (!mRasterPipeline.pipelineMetadata().mDynamicRenderingState || renderTarget.image()->format() != mRasterPipeline.pipelineMetadata().mDynamicRenderingState->mColorFormats[0])
		createPipelines(commandBuffer.mDevice, renderTarget.image()->format());

	const shared_ptr<Scene> scene = mNode.findAncestor<Scene>();


	// scene object picker
	for (auto it = mSelectionData.begin(); it != mSelectionData.end();) {
		if (it->first.buffer()->inFlight())
			break;

		const uint32_t selectedInstance = it->first.cast<VisibilityData>()[0].instanceIndex();
		if (shared_ptr<Inspector> inspector = mNode.root()->findDescendant<Inspector>()) {
			if (selectedInstance == INVALID_INSTANCE || selectedInstance >= scene->frameData().mInstanceNodes.size())
				inspector->select(nullptr);
			else {
				if (shared_ptr<Node> selected = scene->frameData().mInstanceNodes[selectedInstance].lock()) {
					if (it->second) {
						shared_ptr<Node> tn;
						if (selected->findAncestor<TransformData>(&tn))
							inspector->select(tn);
						else
							inspector->select(selected);
					} else
						inspector->select(selected);
				}
			}
		}

		it = mSelectionData.erase(it);
	}


	vector<ViewData> views;
	vector<TransformData> viewTransforms;
	vector<TransformData> viewInverseTransforms;
	scene->mNode.forEachDescendant<Camera>([&](Node& node, const shared_ptr<Camera>& camera) {
		views.emplace_back(camera->view());
		viewInverseTransforms.emplace_back( viewTransforms.emplace_back(nodeToWorld(node)).inverse() );
	});

	Descriptors descriptors;
	descriptors[{"gParams.mViews",0}]                 = mResourcePool.uploadData<ViewData>     (commandBuffer, "mViews"                , views);
	descriptors[{"gParams.mViewTransforms",0}]        = mResourcePool.uploadData<TransformData>(commandBuffer, "mViewTransforms"       , viewTransforms);
	descriptors[{"gParams.mViewInverseTransforms",0}] = mResourcePool.uploadData<TransformData>(commandBuffer, "mViewInverseTransforms", viewInverseTransforms);
	for (const auto&[id,descriptor] : scene->frameData().mDescriptors)
		descriptors[{ "gScene." + id.first, id.second }] = descriptor;


	const vk::Extent3D extent = renderTarget.extent();

	auto colorBuffer = (renderTarget.image()->usage() & (vk::ImageUsageFlagBits::eColorAttachment|vk::ImageUsageFlagBits::eTransferSrc)) ?
		renderTarget : mResourcePool.getImage(commandBuffer.mDevice, "ColorBuffer", Image::Metadata{
		.mFormat = vk::Format::eR8G8B8A8Unorm,
		.mExtent = extent,
		.mUsage = vk::ImageUsageFlagBits::eColorAttachment|vk::ImageUsageFlagBits::eTransferDst|vk::ImageUsageFlagBits::eTransferSrc });
	auto visibilityBuffer = mResourcePool.getImage(commandBuffer.mDevice, "VisibilityBuffer", Image::Metadata{
		.mFormat = vk::Format::eR32G32Uint,
		.mExtent = extent,
		.mUsage = vk::ImageUsageFlagBits::eColorAttachment|vk::ImageUsageFlagBits::eTransferDst|vk::ImageUsageFlagBits::eTransferSrc });
	auto depthBuffer = mResourcePool.getImage(commandBuffer.mDevice, "DepthBuffer", Image::Metadata{
		.mFormat = vk::Format::eD32Sfloat,
		.mExtent = extent,
		.mUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment });

	auto pipeline = mRasterPipeline.get(commandBuffer.mDevice);
	auto descriptorSets = pipeline->getDescriptorSets(descriptors);

	// render

	colorBuffer.barrier(commandBuffer, vk::ImageLayout::eColorAttachmentOptimal, vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::AccessFlagBits::eColorAttachmentWrite);
	visibilityBuffer.barrier(commandBuffer, vk::ImageLayout::eColorAttachmentOptimal, vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::AccessFlagBits::eColorAttachmentWrite);
	depthBuffer.barrier(commandBuffer, vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::PipelineStageFlagBits::eEarlyFragmentTests, vk::AccessFlagBits::eDepthStencilAttachmentRead|vk::AccessFlagBits::eDepthStencilAttachmentWrite);\
	descriptorSets->transitionImages(commandBuffer);

	vector<vk::RenderingAttachmentInfo> colorAttachments{
		{
			*colorBuffer, vk::ImageLayout::eColorAttachmentOptimal,
			vk::ResolveModeFlagBits::eNone,	{}, vk::ImageLayout::eUndefined,
			vk::AttachmentLoadOp::eClear,
			vk::AttachmentStoreOp::eStore,
			vk::ClearValue{vk::ClearColorValue{array<uint32_t,4>{~0u,0u,0u,0u}}} },
		{
			*visibilityBuffer, vk::ImageLayout::eColorAttachmentOptimal,
			vk::ResolveModeFlagBits::eNone,	{}, vk::ImageLayout::eUndefined,
			vk::AttachmentLoadOp::eClear,
			vk::AttachmentStoreOp::eStore,
			vk::ClearValue{vk::ClearColorValue{array<float,4>{0,0,0,0}}} } };
	vk::RenderingAttachmentInfo depthAttachment(
		*depthBuffer, vk::ImageLayout::eDepthStencilAttachmentOptimal,
		vk::ResolveModeFlagBits::eNone,	{}, vk::ImageLayout::eUndefined,
		vk::AttachmentLoadOp::eClear,
		vk::AttachmentStoreOp::eDontCare,
		vk::ClearValue{vk::ClearDepthStencilValue{0,0}});
	commandBuffer->beginRendering(vk::RenderingInfo(
		vk::RenderingFlags{},
		vk::Rect2D(vk::Offset2D(0,0), vk::Extent2D(extent.width, extent.height)),
		1, 0, colorAttachments, &depthAttachment, nullptr));

	commandBuffer->setViewport(0, vk::Viewport(0, 0, extent.width, extent.height, 0, 1));
	commandBuffer->setScissor(0, vk::Rect2D(vk::Offset2D(0,0), vk::Extent2D(extent.width, extent.height)));

	commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, ***pipeline);
	descriptorSets->bind(commandBuffer);
	commandBuffer.trackResource(descriptorSets);

	const auto& instances = scene->frameData().mInstances;

	vector<uint3> alphaMasked;
	alphaMasked.reserve(instances.size());

	struct PushConstants {
		uint mViewIndex;
		uint mInstanceIndex;
		uint mMaterialAddress;
	};
	PushConstants pushConstants;

	pushConstants.mViewIndex = 0;

	for (uint32_t instanceIndex = 0; instanceIndex < instances.size(); instanceIndex++) {
		const auto&[instanceData, material, transform] = instances[instanceIndex];
		const MeshInstanceData* instance = reinterpret_cast<const MeshInstanceData*>(&instanceData);

		if (mAlphaMasks && material->alphaTest()) {
			alphaMasked.emplace_back(instance->getMaterialAddress(), instance->primitiveCount()*3, instanceIndex);
			continue;
		}
		pushConstants.mInstanceIndex = instanceIndex;
		pushConstants.mMaterialAddress = instance->getMaterialAddress();
		pipeline->pushConstants(commandBuffer, { { "", pushConstants } });
		commandBuffer->draw(instance->primitiveCount()*3, 1, 0, instanceIndex);
	}

	if (mAlphaMasks && !alphaMasked.empty()) {
		auto alphaPipeline = mRasterPipeline.get(commandBuffer.mDevice, {{ "gUseAlphaMask", "1" }}, pipeline->descriptorSetLayouts());

		commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, ***alphaPipeline);
		descriptorSets->bind(commandBuffer);
		commandBuffer.trackResource(descriptorSets);
		for (const auto& data : alphaMasked) {
			pushConstants.mInstanceIndex = data[2];
			pushConstants.mMaterialAddress = data[0];
			alphaPipeline->pushConstants(commandBuffer, { { "", pushConstants } });
			commandBuffer->draw(data[1], 1, 0, data[2]);
		}
	}


	commandBuffer->endRendering();

	// copy VisibilityData for selected pixel for scene object picking
	if (!ImGui::GetIO().WantCaptureMouse && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGuizmo::IsUsing()) {
		const ImVec2 c = ImGui::GetIO().MousePos;
		for (const ViewData& view : views)
			if (view.isInside(int2(c.x, c.y))) {
				Buffer::View<VisibilityData> selectionBuffer = make_shared<Buffer>(commandBuffer.mDevice, "SelectionData", sizeof(uint2), vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
				visibilityBuffer.barrier(commandBuffer, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferRead);
				selectionBuffer.copyFromImage(commandBuffer, visibilityBuffer.image(), visibilityBuffer.subresourceLayer(), vk::Offset3D{int(c.x), int(c.y), 0}, vk::Extent3D{1,1,1});
				mSelectionData.push_back(make_pair(selectionBuffer, ImGui::GetIO().KeyShift));
				break;
			}
	}

	if (colorBuffer != renderTarget)
		Image::copy(commandBuffer, colorBuffer, renderTarget);
}

}