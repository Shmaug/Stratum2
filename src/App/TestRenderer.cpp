#include "TestRenderer.hpp"
#include "Inspector.hpp"
#include "Scene.hpp"
#include "Denoiser.hpp"
#include "Tonemapper.hpp"

#include <Core/Instance.hpp>
#include <Core/Profiler.hpp>

#include <imgui/imgui.h>
#include <ImGuizmo.h>

namespace stm2 {

TestRenderer::TestRenderer(Node& node) : mNode(node) {
	if (shared_ptr<Inspector> inspector = mNode.root()->findDescendant<Inspector>())
		inspector->setInspectCallback<TestRenderer>();

	mDefines = {
		{ "gAlphaTest", true },
		{ "gNormalMaps", true },
		{ "gShadingNormals", true },
		{ "gPixelJitter", false },
		{ "gLambertian", false },
		{ "gDebugFastBRDF", false },
		{ "gDebugPaths", false },
		{ "gDebugPathWeights", false },
		{ "gMultiDispatch", true },
		{ "gDeferShadowRays", true },
		{ "gSampleDirectIllumination", true },
		{ "gSampleDirectIlluminationOnly", false },
		{ "gReSTIR_DI", false },
		{ "gReSTIR_DI_Reuse", false },
		{ "gReSTIR_DI_Reuse_Visibility", false },
		{ "gUseVC", false },
	};

	mPushConstants["mMaxDepth"] = 5u;
	mPushConstants["mMaxNullCollisions"] = 1000;
	mPushConstants["mDebugPathLengths"] = 3 | (1<<16);
	mPushConstants["mEnvironmentSampleProbability"] = 0.9f;
	mPushConstants["mCandidateSamples"] = 32;
	mPushConstants["mMaxM"] = 3.f;

    mRasterPushConstants["mDepth"] = -1;
    mRasterPushConstants["mLineRadius"] = .0001f;
    mRasterPushConstants["mLineLength"] = .001f;
    mRasterPushConstants["mVertexPercent"] = .1f;

	Device& device = *mNode.findAncestor<Device>();

	mHashGrid = GpuHashGrid(device, sizeof(float4)*8, 100000, 0.5f);

	mStaticSampler = make_shared<vk::raii::Sampler>(*device, vk::SamplerCreateInfo({},
		vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear,
		vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat,
		0, true, 8, false, vk::CompareOp::eAlways, 0, VK_LOD_CLAMP_NONE));
	device.setDebugName(**mStaticSampler, "TestRenderer/Sampler");

	createPipelines(device);
}

void TestRenderer::createPipelines(Device& device) {
	ComputePipeline::Metadata md;
	md.mImmutableSamplers["gScene.mStaticSampler"]  = { mStaticSampler };
	md.mImmutableSamplers["gScene.mStaticSampler1"] = { mStaticSampler };
	md.mBindingFlags["gScene.mVertexBuffers"] = vk::DescriptorBindingFlagBits::ePartiallyBound;
	md.mBindingFlags["gScene.mImages"]  = vk::DescriptorBindingFlagBits::ePartiallyBound;
	md.mBindingFlags["gScene.mImage2s"] = vk::DescriptorBindingFlagBits::ePartiallyBound;
	md.mBindingFlags["gScene.mImage1s"] = vk::DescriptorBindingFlagBits::ePartiallyBound;
	md.mBindingFlags["gScene.mVolumes"] = vk::DescriptorBindingFlagBits::ePartiallyBound;

	const vector<string>& args = {
		"-matrix-layout-row-major",
		"-O3",
		"-Wno-30081",
		"-capability", "spirv_1_5",
		"-capability", "GL_EXT_ray_tracing"
	};

	const filesystem::path shaderPath = *device.mInstance.findArgument("shaderKernelPath");
	mPipelines.clear();
	mPipelines.emplace("Render",                 ComputePipelineCache(shaderPath / "testrenderer.slang", "Render"             , "sm_6_6", args, md));
	mPipelines.emplace("RenderIteration",        ComputePipelineCache(shaderPath / "testrenderer.slang", "RenderIteration"    , "sm_6_6", args, md));
	mPipelines.emplace("ProcessShadowRays",      ComputePipelineCache(shaderPath / "testrenderer.slang", "ProcessShadowRays"  , "sm_6_6", args, md));
	mPipelines.emplace("ProcessAtomicOutput",    ComputePipelineCache(shaderPath / "testrenderer.slang", "ProcessAtomicOutput", "sm_6_6", args, md));
}

void TestRenderer::drawGui() {
	bool changed = false;

	ImGui::PushID(this);
	if (ImGui::Button("Clear resources")) {
		Device& device = *mNode.findAncestor<Device>();
		device->waitIdle();
		mResourcePool.clear();
		mPrevViewTransforms.clear();
	}
	if (ImGui::Button("Reload shaders")) {
		Device& device = *mNode.findAncestor<Device>();
		device->waitIdle();
		createPipelines(device);
		changed = true;
	}

	if (ImGui::Checkbox("Light tracing", &mLightTrace)) changed = true;

	if (ImGui::CollapsingHeader("Defines")) {
		for (auto&[define, enabled] : mDefines) {
			if (ImGui::Checkbox(define.c_str(), &enabled)) changed = true;
		}

		if (changed) {
			// make sure defines are consistent

			if (mDefines.at("gDebugPathWeights"))
				mDefines.at("gDebugPaths") = true;

			if (mLightTrace) {
				mDefines.at("gUseVC") = false;
				mDefines.at("gSampleDirectIllumination") = false;
				mDefines.at("gSampleDirectIlluminationOnly") = false;
				mDefines.at("gReSTIR_DI") = false;
			}

			if (mDefines.at("gUseVC")) {
				mDefines.at("gSampleDirectIllumination") = false;
				mDefines.at("gSampleDirectIlluminationOnly") = false;
			}

			if (!mDefines.at("gReSTIR_DI"))
				mDefines.at("gReSTIR_DI_Reuse") = false;
			if (!mDefines.at("gReSTIR_DI_Reuse"))
				mDefines.at("gReSTIR_DI_Reuse_Visibility") = false;
		}
	}

	if (ImGui::CollapsingHeader("Configuration")) {
		if (mDefines.at("gDebugPaths")) {
			ImGui::SetNextItemWidth(40);
			if (ImGui::DragScalarN("Length, light vertices", ImGuiDataType_U16, &mPushConstants["mDebugPathLengths"].get<uint32_t>(), 2, .2f)) changed = true;
		}
		if (ImGui::Checkbox("Random frame seed", &mRandomPerFrame)) changed = true;
		ImGui::PushItemWidth(40);
		uint32_t one = 1;
		if (ImGui::DragScalar("Max depth", ImGuiDataType_U32, &mPushConstants["mMaxDepth"].get<uint32_t>(), .2f, &one)) changed = true;
		if (ImGui::DragScalar("Max null collisions", ImGuiDataType_U32, &mPushConstants["mMaxNullCollisions"].get<uint32_t>())) changed = true;

		if (!mLightTrace) {
			if (ImGui::SliderFloat("Environment sample p", &mPushConstants["mEnvironmentSampleProbability"].get<float>(), 0, 1)) changed = true;
		}

		if (mDefines.at("gUseVC") || mLightTrace) {
			if (ImGui::SliderFloat("Light subpath count", &mLightSubpathCount, 0, 2)) changed = true;
		}

		if (mDefines.at("gReSTIR_DI")) {
			if (ImGui::DragScalar("Candidate samples", ImGuiDataType_U32, &mPushConstants["mCandidateSamples"].get<uint32_t>())) changed = true;
		}

		if (mDefines.at("gReSTIR_DI_Reuse")) {
			if (ImGui::DragFloat("Max M", &mPushConstants["mMaxM"].get<float>(), .01f)) changed = true;
			if (ImGui::CollapsingHeader("Hash grid")) {
				ImGui::Indent();
				if (ImGui::DragScalar("Cell count", ImGuiDataType_U32, &mHashGrid.mCellCount)) changed = true;
				if (ImGui::DragFloat("Min cell size", &mHashGrid.mCellSize, .01f)) changed = true;
				if (ImGui::DragFloat("Cell pixel radius", &mHashGrid.mCellPixelRadius, .5f, 0, 1000)) changed = true;
				ImGui::Unindent();
			}
		}
		ImGui::PopItemWidth();

		if (mDefines.at("gUseVC")) {
			ImGui::Checkbox("Visualize light paths", &mVisualizeLightPaths);
			if (mVisualizeLightPaths){
				ImGui::PushItemWidth(40);
				ImGui::Indent();
				ImGui::DragScalar("Depth", ImGuiDataType_S32, &mRasterPushConstants["mDepth"].get<int32_t>());
				ImGui::SliderFloat("% vertices", &mRasterPushConstants["mVertexPercent"].get<float>(), 0, 1);
				ImGui::DragFloat("Line radius", &mRasterPushConstants["mLineRadius"].get<float>(), .01f);
				ImGui::DragFloat("Line length", &mRasterPushConstants["mLineLength"].get<float>(), .5f, 0, .5);
				ImGui::Unindent();
				ImGui::PopItemWidth();
			}
		}

		if (ImGui::Checkbox("Denoise ", &mDenoise)) changed = true;
		if (ImGui::Checkbox("Tonemap", &mTonemap)) changed = true;
	}

	if (changed && mDenoise) {
		const shared_ptr<Denoiser> denoiser = mNode.findDescendant<Denoiser>();
		if (denoiser)
			denoiser->resetAccumulation();
	}

	if (ImGui::CollapsingHeader("Resources")) {
		ImGui::Indent();
		mResourcePool.drawGui();
		ImGui::Unindent();
	}
	ImGui::PopID();
}

GraphicsPipelineCache createRasterPipeline(Device& device, const vk::Extent2D& extent, const vk::Format format) {
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
	const filesystem::path rasterShaderPath = shaderPath / "path_vis.slang";
	const vector<string>& rasterArgs = {
		"-matrix-layout-row-major",
		"-O3",
		"-Wno-30081",
		"-capability", "spirv_1_5",
	};

	auto staticSampler = make_shared<vk::raii::Sampler>(*device, vk::SamplerCreateInfo({},
		vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear,
		vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat,
		0, true, 8, false, vk::CompareOp::eAlways, 0, VK_LOD_CLAMP_NONE));
	device.setDebugName(**staticSampler, "TestRenderer/Sampler (Raster)");
	gmd.mImmutableSamplers["gScene.mStaticSampler"]  = { staticSampler };
	gmd.mImmutableSamplers["gScene.mStaticSampler1"] = { staticSampler };
	gmd.mBindingFlags["gScene.mVertexBuffers"] = vk::DescriptorBindingFlagBits::ePartiallyBound;
	gmd.mBindingFlags["gScene.mImages"]  = vk::DescriptorBindingFlagBits::ePartiallyBound;
	gmd.mBindingFlags["gScene.mImage2s"] = vk::DescriptorBindingFlagBits::ePartiallyBound;
	gmd.mBindingFlags["gScene.mImage1s"] = vk::DescriptorBindingFlagBits::ePartiallyBound;
	gmd.mBindingFlags["gScene.mVolumes"] = vk::DescriptorBindingFlagBits::ePartiallyBound;

	return GraphicsPipelineCache({
		{ vk::ShaderStageFlagBits::eVertex  , GraphicsPipelineCache::ShaderSourceInfo(rasterShaderPath, "LightVertexVS", "sm_6_6") },
		{ vk::ShaderStageFlagBits::eFragment, GraphicsPipelineCache::ShaderSourceInfo(rasterShaderPath, "LightVertexFS", "sm_6_6") }
	}, rasterArgs, gmd);
}

void TestRenderer::render(CommandBuffer& commandBuffer, const Image::View& renderTarget) {
	ProfilerScope ps("TestRenderer::render", &commandBuffer);

	mResourcePool.clean();

	const vk::Extent3D extent = renderTarget.extent();

	const shared_ptr<Scene>      scene      = mNode.findAncestor<Scene>();
	const shared_ptr<Denoiser>   denoiser   = mNode.findDescendant<Denoiser>();
	const shared_ptr<Tonemapper> tonemapper = mNode.findDescendant<Tonemapper>();

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

	// allocate images

	const Image::View outputImage = mResourcePool.getImage(commandBuffer.mDevice, "mOutput", Image::Metadata{
		.mFormat = vk::Format::eR32G32B32A32Sfloat,
		.mExtent = extent,
		.mUsage = vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eTransferSrc,
	}, 0);
	const Image::View albedoImage = mResourcePool.getImage(commandBuffer.mDevice, "mAlbedo", Image::Metadata{
		.mFormat = vk::Format::eR32G32B32A32Sfloat,
		.mExtent = extent,
		.mUsage = vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eTransferSrc,
	}, 0);
	const Image::View prevUVsImage = mResourcePool.getImage(commandBuffer.mDevice, "mPrevUVs", Image::Metadata{
		.mFormat = vk::Format::eR32G32Sfloat,
		.mExtent = extent,
		.mUsage = vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eTransferSrc,
	}, 0);
	const Image::View visibilityImage = mResourcePool.getImage(commandBuffer.mDevice, "mVisibility", Image::Metadata{
		.mFormat = vk::Format::eR32G32Uint,
		.mExtent = extent,
		.mUsage = vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eTransferSrc,
	});
	const Image::View depthImage = mResourcePool.getImage(commandBuffer.mDevice, "mDepth", Image::Metadata{
		.mFormat = vk::Format::eR32G32B32A32Sfloat,
		.mExtent = extent,
		.mUsage = vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eTransferSrc,
	});

	// assign descriptors

	Descriptors descriptors;

	auto pathStates   = mResourcePool.getBuffer<float4x4>(commandBuffer.mDevice, "mPathStates", mDefines.at("gMultiDispatch") ? extent.width*extent.height : 1, vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, 0);
	auto atomicOutput = mResourcePool.getBuffer<uint4>(commandBuffer.mDevice, "mOutputAtomic", (mDefines.at("gDeferShadowRays")||mDefines.at("gUseVC")||mLightTrace) ? extent.width*extent.height : 1, vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal, 0);

	descriptors[{ "gRenderParams.mOutput", 0 }]     = ImageDescriptor{ outputImage    , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };
	descriptors[{ "gRenderParams.mAlbedo", 0 }]     = ImageDescriptor{ albedoImage    , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };
	descriptors[{ "gRenderParams.mPrevUVs", 0 }]    = ImageDescriptor{ prevUVsImage   , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };
	descriptors[{ "gRenderParams.mVisibility", 0 }] = ImageDescriptor{ visibilityImage, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };
	descriptors[{ "gRenderParams.mDepth", 0 }]      = ImageDescriptor{ depthImage     , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };
	descriptors[{ "gRenderParams.mPathStates", 0 }] = pathStates;
	descriptors[{ "gRenderParams.mOutputAtomic", 0 }] = atomicOutput;

	bool changed = false;
	bool hasHeterogeneousMedia = false;

	vector<ViewData> viewsBufferData;
	vector<TransformData> viewTransformsBufferData;
	Buffer::View<ViewData> viewsBuffer;

	// find views, assign scene descriptors
	{
		const Scene::FrameData& sceneData = scene->frameData();
		if (sceneData.mDescriptors.empty()) {
			renderTarget.clearColor(commandBuffer, vk::ClearColorValue(array<uint32_t, 4>{ 0, 0, 0, 0 }));
			return;
		}


		if (scene->lastUpdate() > mLastSceneVersion) {
			changed = true;
			mLastSceneVersion = scene->lastUpdate();
		}

		for (auto& [name, d] : sceneData.mDescriptors)
			descriptors[{ "gScene." + name.first, name.second }] = d;
		descriptors[{ "gScene.mRayCount", 0u }] = make_shared<Buffer>(commandBuffer.mDevice, "mRayCount", 2*sizeof(uint32_t), vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

		// track resources which are not held by the descriptorset
		commandBuffer.trackResource(sceneData.mAccelerationStructureBuffer.buffer());

		// find views

		vector<pair<ViewData, TransformData>> views;
		mNode.root()->forEachDescendant<Camera>([&](Node& node, const shared_ptr<Camera>& camera) {
			views.emplace_back(pair{ camera->view(), nodeToWorld(node) });
		});

		if (views.empty()) {
			renderTarget.clearColor(commandBuffer, vk::ClearColorValue(array<uint32_t, 4>{ 0, 0, 0, 0 }));
			return;
		}

		viewsBufferData.resize(views.size());
		viewTransformsBufferData.resize(views.size());
		vector<TransformData> inverseViewTransformsData(views.size());
		vector<TransformData> prevInverseViewTransformsData(views.size());
		for (uint32_t i = 0; i < views.size(); i++) {
			const auto&[view,viewTransform] = views[i];
			viewsBufferData[i] = view;
			viewTransformsBufferData[i] = viewTransform;
			inverseViewTransformsData[i] = viewTransform.inverse();
			if (mPrevViewTransforms.size() == views.size()) {
				prevInverseViewTransformsData[i] = mPrevViewTransforms[i].inverse();
				if ((mPrevViewTransforms[i].m != viewTransform.m).any())
					changed = true;
			} else
				prevInverseViewTransformsData[i] = viewTransform.inverse();
		}

		mPrevViewTransforms = viewTransformsBufferData;

		viewsBuffer = mResourcePool.uploadData<ViewData>(commandBuffer, "mViews", viewsBufferData);

		descriptors[{ "gRenderParams.mViews", 0 }]                     = viewsBuffer;
		descriptors[{ "gRenderParams.mViewTransforms", 0 }]            = mResourcePool.uploadData<TransformData>(commandBuffer, "mViewTransforms", viewTransformsBufferData);
		descriptors[{ "gRenderParams.mViewInverseTransforms", 0 }]     = mResourcePool.uploadData<TransformData>(commandBuffer, "mViewInverseTransforms", inverseViewTransformsData);
		descriptors[{ "gRenderParams.mPrevViewInverseTransforms", 0 }] = mResourcePool.uploadData<TransformData>(commandBuffer, "mPrevViewInverseTransforms", prevInverseViewTransformsData);


		// find if views are inside a volume
		vector<uint32_t> viewMediumIndices(views.size());
		ranges::fill(viewMediumIndices, INVALID_INSTANCE);
		for (const auto& info : sceneData.mInstanceVolumeInfo) {
			for (uint32_t i = 0; i < views.size(); i++) {
				const auto&[instance, material, transform] = sceneData.mInstances[info.mInstanceIndex];
				if (reinterpret_cast<const VolumeInstanceData*>(&instance)->volumeIndex() != -1)
					hasHeterogeneousMedia = true;
				const float3 localViewPos = transform.inverse().transformPoint( viewTransformsBufferData[i].transformPoint(float3::Zero()) );
				if ((localViewPos >= info.mMin).all() && (localViewPos <= info.mMax).all()) {
					viewMediumIndices[i] = info.mInstanceIndex;
				}
			}
		}

		descriptors[{ "gRenderParams.mViewMediumIndices", 0 }] = mResourcePool.uploadData<uint>(commandBuffer, "mViewMediumIndices", viewMediumIndices);
	}

	{
		const Scene::FrameData& sceneData = scene->frameData();

		mPushConstants["mOutputExtent"] = uint2(renderTarget.extent().width, renderTarget.extent().height);
		mPushConstants["mLightSubpathCount"] = max(1u, uint32_t(renderTarget.extent().width * renderTarget.extent().height * mLightSubpathCount));
		mPushConstants["mViewCount"] = (uint32_t)viewsBufferData.size();
		mPushConstants["mEnvironmentMaterialAddress"] = sceneData.mEnvironmentMaterialAddress;
		mPushConstants["mLightCount"] = sceneData.mLightCount;
		mPushConstants["mVolumeInstanceCount"] = (uint32_t)sceneData.mInstanceVolumeInfo.size();
		float4 sphere;
		sphere.head<3>() = (sceneData.mAabbMax + sceneData.mAabbMin) / 2;
		sphere[3] = length<float,3>(sceneData.mAabbMax - sphere.head<3>());
		mPushConstants["mSceneSphere"] = sphere;

		if (changed && mRandomPerFrame && mDenoise && denoiser && !denoiser->reprojection()) {
			denoiser->resetAccumulation();
		}

		if (mRandomPerFrame)
			mPushConstants["mRandomSeed"] = (mDenoise && denoiser) ? denoiser->accumulatedFrames() : (uint32_t)rand();
		else
			mPushConstants["mRandomSeed"] = 0u;

		if (!mHashGrid.mResourcePool.getLastBuffer<byte>("mHashGrid.mChecksums"))
			mPushConstants["mPrevHashGridValid"] = 0u;
		else
			mPushConstants["mPrevHashGridValid"] = 1u;
	}

	const uint32_t maxShadowRays = mDefines.at("gDeferShadowRays") ? (extent.width*extent.height*2 + (mLightTrace || mDefines.at("gUseVC") ? mPushConstants["mLightSubpathCount"].get<uint32_t>() : 0))*(mPushConstants["mMaxDepth"].get<uint32_t>()-1) : 0;

	auto lightVertexBuffer = mResourcePool.getBuffer<array<float4,3>>(commandBuffer.mDevice, "mLightVertices", mDefines.at("gUseVC") ? max(1u, mPushConstants["mLightSubpathCount"].get<uint32_t>()*(mPushConstants["mMaxDepth"].get<uint32_t>()-1)) : 1, vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, 0);
	auto counterBuffer = mResourcePool.getBuffer<uint32_t>(commandBuffer.mDevice, "mCounters", 2, vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal, 0);
	descriptors[{ "gRenderParams.mLightVertices", 0 }] = lightVertexBuffer;
	descriptors[{ "gRenderParams.mCounters", 0 }] = counterBuffer;
	descriptors[{ "gRenderParams.mShadowRays", 0 }] = mResourcePool.getBuffer<array<float4,4>>(commandBuffer.mDevice, "mShadowRays", max(1u, maxShadowRays), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, 0);

	mHashGrid.mSize = mDefines.at("gReSTIR_DI_Reuse") ? max(1u, extent.width*extent.height*(mPushConstants["mMaxDepth"].get<uint32_t>()-1)) : 1;
	const auto hashGrid = mHashGrid.init(commandBuffer, descriptors, "mHashGrid", GpuHashGrid::Metadata{
			.mCameraPosition = viewTransformsBufferData[0].transformPoint(float3::Zero()),
			.mVerticalFoV = viewsBufferData[0].mProjection.mVerticalFoV,
			.mImageExtent = { extent.width, extent.height },
		}, "mPrevHashGrid");

	if (!mDefines.at("gReSTIR_DI_Reuse")) {
		mHashGrid.mResourcePool.clear();
		mPrevHashGridEvent.reset();
	}

	// setup shader defines

	Defines defines;
	for (const auto&[define,enabled] : mDefines)
		if (enabled)
			defines[define] = to_string(enabled);

	if (mPushConstants["mVolumeInstanceCount"].get<uint32_t>() > 0)
		defines.emplace("gHasMedia", "true");
	if (hasHeterogeneousMedia)
		defines.emplace("gHasHeterogeneousMedia", "true");


	// create pipelines

	bool loading = false;

	auto loadPipeline = [&](const string& name, const Defines& defs) {
		const auto& p = mPipelines.at(name).getAsync(commandBuffer.mDevice, defs);
		if (!p)
			loading = true;
		return p;
	};

	shared_ptr<ComputePipeline> renderPipeline, renderIterationPipeline, renderLightPipeline, renderLightIterationPipeline, processShadowRaysPipeline, processAtomicOutputPipeline;
	{
		{
			Defines tmp = defines;
			if (mDefines.at("gMultiDispatch")) {
				renderIterationPipeline = loadPipeline("RenderIteration", tmp);
				tmp["gMultiDispatchFirst"] = "true";
			}
			renderPipeline = loadPipeline("Render", tmp);
		}
		if (mDefines.at("gUseVC") || mLightTrace) {
			Defines tmp = defines;
			tmp["gTraceFromLight"] = "true";
			if (mDefines.at("gMultiDispatch")) {
				renderLightIterationPipeline = loadPipeline("RenderIteration", tmp);
				tmp["gMultiDispatchFirst"] = "true";
			}
			renderLightPipeline = loadPipeline("Render", tmp);
		}
		if (mDefines.at("gDeferShadowRays"))
			processShadowRaysPipeline = loadPipeline("ProcessShadowRays", defines);
		if (mDefines.at("gDeferShadowRays") || mDefines.at("gUseVC") || mLightTrace)
			processAtomicOutputPipeline = loadPipeline("ProcessAtomicOutput", Defines{ { "gClearImage", to_string(mLightTrace) }});
	}
	if (loading) {
		// compiling shaders...
		renderTarget.clearColor(commandBuffer, vk::ClearColorValue(array<float, 4>{ 0.5f, 0.5f, 0.5f, 0 }));
		return;
	}

	// create descriptor sets
	const shared_ptr<DescriptorSets> descriptorSets = mResourcePool.getDescriptorSets(*renderPipeline, "DescriptorSets", descriptors);

	// render
	{
		if (mDefines.at("gDeferShadowRays") || mDefines.at("gUseVC")) {
			counterBuffer.fill(commandBuffer, 0);
			counterBuffer.barrier(commandBuffer,
				vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
				vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);
		}

		// light paths
		if (mDefines.at("gUseVC") || mLightTrace) {
			ProfilerScope ps("Light paths", &commandBuffer);

			atomicOutput.fill(commandBuffer, 0);
			atomicOutput.barrier(commandBuffer,
				vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
				vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);

			const vk::Extent3D lightExtent = { extent.width, (mPushConstants["mLightSubpathCount"].get<uint32_t>() + extent.width-1)/extent.width, 1 };
			renderLightPipeline->dispatchTiled(commandBuffer, lightExtent, descriptorSets, {}, mPushConstants);
			if (mDefines.at("gMultiDispatch")) {
				for (uint32_t i = 1; i < mPushConstants["mMaxDepth"].get<uint32_t>(); i++) {
					pathStates.barrier(commandBuffer,
						vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
						vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);
					renderLightIterationPipeline->dispatchTiled(commandBuffer, lightExtent, descriptorSets, {}, mPushConstants);
				}
			}

			lightVertexBuffer.barrier(commandBuffer,
				vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
				vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);
		}

		// view paths
		if (!mLightTrace) {
			ProfilerScope ps("View paths", &commandBuffer);

			if (mDefines.at("gReSTIR_DI_Reuse")) {
				if (mPrevHashGridEvent) {
					vector<vk::BufferMemoryBarrier> barriers;
					for (const auto&[id, buf] : hashGrid)
						barriers.emplace_back(vk::BufferMemoryBarrier{
							vk::AccessFlagBits::eShaderWrite,
							vk::AccessFlagBits::eShaderRead,
							VK_QUEUE_FAMILY_IGNORED,
							VK_QUEUE_FAMILY_IGNORED,
							**get<BufferDescriptor>(buf).buffer(),
							get<BufferDescriptor>(buf).offset(),
							get<BufferDescriptor>(buf).size() });
					commandBuffer->waitEvents(**mPrevHashGridEvent, vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, barriers, {});
					commandBuffer.trackVulkanResource(mPrevHashGridEvent);
				}
				mHashGrid.clear(commandBuffer, hashGrid);
			}

			renderPipeline->dispatchTiled(commandBuffer, extent, descriptorSets, {}, mPushConstants);
			if (mDefines.at("gMultiDispatch")) {
				for (uint32_t i = 1; i < mPushConstants["mMaxDepth"].get<uint32_t>(); i++) {
					pathStates.barrier(commandBuffer,
						vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
						vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);
					renderIterationPipeline->dispatchTiled(commandBuffer, extent, descriptorSets, {}, mPushConstants);
				}
			}

			if (mDefines.at("gReSTIR_DI_Reuse")) {
				mHashGrid.build(commandBuffer, hashGrid);

				mPrevHashGridEvent = make_shared<vk::raii::Event>(*commandBuffer.mDevice, vk::EventCreateInfo{ vk::EventCreateFlagBits::eDeviceOnly });
				commandBuffer->setEvent(**mPrevHashGridEvent, vk::PipelineStageFlagBits::eComputeShader);
				commandBuffer.trackVulkanResource(mPrevHashGridEvent);
			}
		}

		if (mDefines.at("gDeferShadowRays")) {
			ProfilerScope ps("Shadow rays", &commandBuffer);
			if (!mDefines.at("gUseVC") && !mLightTrace) {
				atomicOutput.fill(commandBuffer, 0);
				Buffer::barriers(commandBuffer, { atomicOutput },
					vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
					vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);
			}
			processShadowRaysPipeline->dispatchTiled(commandBuffer, vk::Extent3D{extent.width, (maxShadowRays + extent.width-1) / extent.width, 1}, descriptorSets, {}, mPushConstants);
		}

		if (mDefines.at("gDeferShadowRays") || mDefines.at("gUseVC") || mLightTrace) {
			Buffer::barriers(commandBuffer, { atomicOutput },
				vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
				vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);
			processAtomicOutputPipeline->dispatchTiled(commandBuffer, extent, descriptorSets, {}, mPushConstants);
		}
	}


	// post processing

	Image::View processedOutput = outputImage;

	// run denoiser
	if (mDenoise && denoiser && mRandomPerFrame) {
		processedOutput = denoiser->denoise(
			commandBuffer,
			outputImage,
			albedoImage,
			prevUVsImage,
			visibilityImage,
			depthImage,
			viewsBuffer );
	}

	// run tonemapper
	if (mTonemap && tonemapper) {
		tonemapper->render(commandBuffer, processedOutput, outputImage, (mDenoise && denoiser && denoiser->demodulateAlbedo()) ? albedoImage : Image::View{});
		// copy outputImage to renderTarget
		if (outputImage.image()->format() == renderTarget.image()->format())
			Image::copy(commandBuffer, outputImage, renderTarget);
		else
			Image::blit(commandBuffer, outputImage, renderTarget);
	} else {
		// copy processedOutput to renderTarget
		if (processedOutput.image()->format() == renderTarget.image()->format())
			Image::copy(commandBuffer, processedOutput, renderTarget);
		else
			Image::blit(commandBuffer, processedOutput, renderTarget);
	}


	// copy VisibilityData for selected pixel for scene object picking
	if (!ImGui::GetIO().WantCaptureMouse && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGuizmo::IsUsing()) {
		const ImVec2 c = ImGui::GetIO().MousePos;
		for (const ViewData& view : viewsBufferData)
			if (view.isInside(int2(c.x, c.y))) {
				Buffer::View<VisibilityData> selectionBuffer = make_shared<Buffer>(commandBuffer.mDevice, "SelectionData", sizeof(uint2), vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
				visibilityImage.barrier(commandBuffer, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferRead);
				selectionBuffer.copyFromImage(commandBuffer, visibilityImage.image(), visibilityImage.subresourceLayer(), vk::Offset3D{int(c.x), int(c.y), 0}, vk::Extent3D{1,1,1});
				mSelectionData.push_back(make_pair(selectionBuffer, ImGui::GetIO().KeyShift));
				break;
			}
	}


	if (mDefines.at("gUseVC") && mVisualizeLightPaths) {
		if (!mRasterLightPathPipeline || renderTarget.image()->format() != mRasterLightPathPipeline.pipelineMetadata().mDynamicRenderingState->mColorFormats[0])
			mRasterLightPathPipeline = createRasterPipeline(commandBuffer.mDevice, vk::Extent2D(renderTarget.extent().width, renderTarget.extent().height), renderTarget.image()->format());

		Descriptors rasterDescriptors;
		for (auto& [name, d] : scene->frameData().mDescriptors)
			rasterDescriptors[{ "gScene." + name.first, name.second }] = d;
		rasterDescriptors.erase({"gScene.mAccelerationStructure", 0});

		for (auto d : {
			"mViews",
			"mViewInverseTransforms",
			"mLightVertices",
			"mCounters" }) {
			rasterDescriptors[{string("gParams.")+d,0}] = descriptors.at({string("gRenderParams.")+d, 0});
		}
		rasterDescriptors[{"gParams.mDepth",0}] = ImageDescriptor{get<Image::View>(get<ImageDescriptor>(descriptors.at({"gRenderParams.mDepth",0}))), vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {}};

		const Image::View rasterDepthBuffer = mResourcePool.getImage(commandBuffer.mDevice, "mRasterDepthBuffer", Image::Metadata{
				.mFormat = vk::Format::eD32Sfloat,
				.mExtent = extent,
				.mUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment|vk::ImageUsageFlagBits::eTransferDst });
		auto lightPathPipeline = mRasterLightPathPipeline.get(commandBuffer.mDevice, { { "NO_SCENE_ACCELERATION_STRUCTURE", "1" } });
		auto descriptorSets = lightPathPipeline->getDescriptorSets(rasterDescriptors);

		// render light paths
		{
			renderTarget.barrier(commandBuffer, vk::ImageLayout::eColorAttachmentOptimal, vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::AccessFlagBits::eColorAttachmentWrite);
			rasterDepthBuffer.barrier(commandBuffer, vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::PipelineStageFlagBits::eEarlyFragmentTests, vk::AccessFlagBits::eDepthStencilAttachmentRead|vk::AccessFlagBits::eDepthStencilAttachmentWrite);

			descriptorSets->transitionImages(commandBuffer);

			vk::RenderingAttachmentInfo colorAttachment(
				*renderTarget, vk::ImageLayout::eColorAttachmentOptimal,
				vk::ResolveModeFlagBits::eNone,	{}, vk::ImageLayout::eUndefined,
				vk::AttachmentLoadOp::eLoad,
				vk::AttachmentStoreOp::eStore,
				vk::ClearValue{});
			vk::RenderingAttachmentInfo depthAttachment(
				*rasterDepthBuffer, vk::ImageLayout::eDepthStencilAttachmentOptimal,
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
			lightPathPipeline->pushConstants(commandBuffer, mRasterPushConstants);
			const uint32_t vertexCount = mPushConstants["mLightSubpathCount"].get<uint32_t>()*mPushConstants["mMaxDepth"].get<uint32_t>();
			commandBuffer->draw(uint32_t(mRasterPushConstants["mVertexPercent"].get<float>() * vertexCount)*6, 1, 0, 0);

			commandBuffer.trackResource(lightPathPipeline);
			commandBuffer.trackResource(descriptorSets);

			commandBuffer->endRendering();
		}
	}
}

}