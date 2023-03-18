#include "VCM.hpp"
#include "Denoiser.hpp"
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

VCM::VCM(Node& node) : mNode(node) {
	if (shared_ptr<Inspector> inspector = mNode.root()->findDescendant<Inspector>())
		inspector->setInspectCallback<VCM>();

	Device& device = *mNode.findAncestor<Device>();
	createPipelines(device);

	mLightHashGrid = GpuHashGrid(device, sizeof(uint32_t), 100000, 0.1f);
	mDIHashGrid  = GpuHashGrid(device, sizeof(DirectIlluminationReservoir), 100000, 0.1f);
	mLVCHashGrid = GpuHashGrid(device, sizeof(LVCReservoir), 100000, 0.1f);

	mPushConstants.mMinPathLength = 0;
	mPushConstants.mMaxPathLength = 6;
	mPushConstants.mEnvironmentSampleProbability = 0.5f;
	mPushConstants.mLightImageQuantization = 16384;
	mPushConstants.mDIReservoirSampleCount = 32;
	mPushConstants.mDIReservoirMaxM = 3;
	mPushConstants.mLVCReservoirSampleCount = 8;
	mPushConstants.mLVCReservoirMaxM = 3;

	// initialize constants
	if (auto arg = device.mInstance.findArgument("minPathLength"); arg) mPushConstants.mMinPathLength = atoi(arg->c_str());
	if (auto arg = device.mInstance.findArgument("maxPathLength"); arg) mPushConstants.mMaxPathLength = atoi(arg->c_str());
}

void VCM::createPipelines(Device& device) {
	const auto samplerRepeat = make_shared<vk::raii::Sampler>(*device, vk::SamplerCreateInfo({},
		vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear,
		vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat,
		0, true, 8, false, vk::CompareOp::eAlways, 0, VK_LOD_CLAMP_NONE));

	ComputePipeline::Metadata md;
	md.mImmutableSamplers["gScene.mStaticSampler"]  = { samplerRepeat };
	md.mImmutableSamplers["gScene.mStaticSampler1"] = { samplerRepeat };
	md.mBindingFlags["gScene.mVertexBuffers"] = vk::DescriptorBindingFlagBits::ePartiallyBound;
	md.mBindingFlags["gScene.mImages"]  = vk::DescriptorBindingFlagBits::ePartiallyBound;
	md.mBindingFlags["gScene.mImage1s"] = vk::DescriptorBindingFlagBits::ePartiallyBound;
	md.mBindingFlags["gScene.mImage2s"] = vk::DescriptorBindingFlagBits::ePartiallyBound;
	md.mBindingFlags["gScene.mVolumes"] = vk::DescriptorBindingFlagBits::ePartiallyBound;

	const filesystem::path shaderPath = *device.mInstance.findArgument("shaderKernelPath");
	const vector<string>& args = {
		"-matrix-layout-row-major",
		"-O3",
		"-Wno-30081",
		"-capability", "spirv_1_5",
		"-capability", "GL_EXT_ray_tracing"
	};
	const filesystem::path kernelPath = shaderPath / "vcm.slang";
	mRenderPipelines[RenderPipelineIndex::eGenerateLightPaths]     = ComputePipelineCache(kernelPath, "GenerateLightPaths"    , "sm_6_6", args, md);
	mRenderPipelines[RenderPipelineIndex::eGenerateCameraPaths]    = ComputePipelineCache(kernelPath, "GenerateCameraPaths"   , "sm_6_6", args, md);
	mRenderPipelines[RenderPipelineIndex::eHashGridComputeIndices] = ComputePipelineCache(shaderPath / "hashgrid.slang", "ComputeIndices", "sm_6_6", { "-O3", "-matrix-layout-row-major", "-capability", "spirv_1_5" });
	mRenderPipelines[RenderPipelineIndex::eHashGridSwizzle]        = ComputePipelineCache(shaderPath / "hashgrid.slang", "Swizzle"       , "sm_6_6", { "-O3", "-matrix-layout-row-major", "-capability", "spirv_1_5" });

	auto swapchain = mNode.root()->findDescendant<Swapchain>();
	createRasterPipeline(device, swapchain->extent(), swapchain->format().format);
}

void VCM::createRasterPipeline(Device& device, const vk::Extent2D& extent, const vk::Format format) {
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

void VCM::drawGui() {
	bool changed = false;

	ImGui::PushID(this);
	if (ImGui::Button("Clear resources")) {
		Device& device = *mNode.findAncestor<Device>();
		device->waitIdle();
		mResourcePool.clear();
	}
	if (ImGui::Button("Reload shaders")) {
		changed = true;
		Device& device = *mNode.findAncestor<Device>();
		device->waitIdle();
		createPipelines(device);
	}
	ImGui::PopID();

	ImGui::Checkbox("Pause rendering", &mPauseRendering);

	{
		if (Gui::enumDropdown<VcmAlgorithmType>("Algorithm", mAlgorithm, VcmAlgorithmType::kNumVcmAlgorithmType)) changed = true;
		if (ImGui::Checkbox("Shading normals", &mUseShadingNormals)) changed = true;
		if (mUseShadingNormals) {
			if (ImGui::Checkbox("Normal maps", &mUseNormalMaps)) changed = true;
		}
		if (ImGui::Checkbox("Alpha testing", &mUseAlphaTesting)) changed = true;
		if (ImGui::Checkbox("Random frame seed", &mRandomPerFrame)) changed = true;
		ImGui::PushItemWidth(60);

		uint32_t one = 1;
		if (ImGui::DragScalar("Max path length", ImGuiDataType_U32, &mPushConstants.mMaxPathLength, 1, &one)) changed = true;
		if (ImGui::DragScalar("Min path length", ImGuiDataType_U32, &mPushConstants.mMinPathLength, 1, &one)) changed = true;

		if (mPushConstants.mEnvironmentMaterialAddress != -1 && mPushConstants.mLightCount > 0)
			if (ImGui::SliderFloat("Environment sample probability", &mPushConstants.mEnvironmentSampleProbability, 0, 1)) changed = true;

		if (mAlgorithm == VcmAlgorithmType::kBpt)
			if (ImGui::Checkbox("Light vertex cache", &mUseLightVertexCache)) changed = true;

		// reservoirs
		if (mAlgorithm != VcmAlgorithmType::kLightTrace) {
			if (ImGui::CheckboxFlags("DI reservoir resampling", reinterpret_cast<uint32_t*>(&mDIReservoirFlags), (uint32_t)VcmReservoirFlags::eRIS)) changed = true;
			if (mDIReservoirFlags & VcmReservoirFlags::eRIS) {
				ImGui::Indent();
				ImGui::PushID(&mDIReservoirFlags);
				if (ImGui::DragScalar("RIS samples", ImGuiDataType_U32, &mPushConstants.mDIReservoirSampleCount, 1, &one)) changed = true;
				if (ImGui::CheckboxFlags("Reuse", reinterpret_cast<uint32_t*>(&mDIReservoirFlags), (uint32_t)VcmReservoirFlags::eReuse)) changed = true;
				if (mDIReservoirFlags & VcmReservoirFlags::eReuse) {
					if (ImGui::DragFloat("Max M", &mPushConstants.mDIReservoirMaxM)) changed = true;
				}
				ImGui::PopID();
				ImGui::Unindent();
			}

			if (mAlgorithm == VcmAlgorithmType::kBpt && mUseLightVertexCache) {
				if (ImGui::CheckboxFlags("LVC reservoir resampling", reinterpret_cast<uint32_t*>(&mLVCReservoirFlags), (uint32_t)VcmReservoirFlags::eRIS)) changed = true;
				if (mLVCReservoirFlags & VcmReservoirFlags::eRIS) {
					ImGui::Indent();
					ImGui::PushID(&mLVCReservoirFlags);
					if (ImGui::DragScalar("RIS samples", ImGuiDataType_U32, &mPushConstants.mLVCReservoirSampleCount, 1, &one)) changed = true;
					if (ImGui::CheckboxFlags("Reuse", reinterpret_cast<uint32_t*>(&mLVCReservoirFlags), (uint32_t)VcmReservoirFlags::eReuse)) changed = true;
					if (mLVCReservoirFlags & VcmReservoirFlags::eReuse) {
						if (ImGui::DragFloat("Max M", &mPushConstants.mLVCReservoirMaxM)) changed = true;
					}
					ImGui::PopID();
					ImGui::Unindent();
				}
			}
		}

		// light subpaths
		if (mAlgorithm != VcmAlgorithmType::kPathTrace) {
			if (ImGui::DragScalar("Light image quantization", ImGuiDataType_U32, &mPushConstants.mLightImageQuantization)) changed = true;
			if (ImGui::SliderFloat("Relative light path count", &mLightPathPercent, 0, 2)) changed = true;

			ImGui::Checkbox("Visualize light paths", &mVisualizeLightPaths);
			if (mVisualizeLightPaths) {
				ImGui::DragScalar("Num paths visualized", ImGuiDataType_U32, &mVisualizeLightPathCount);
				ImGui::DragScalar("Segment index", ImGuiDataType_U32, &mVisualizeSegmentIndex);
				ImGui::DragFloat("Line width", &mVisualizeLightPathRadius, .0005f);
				ImGui::DragFloat("Line length", &mVisualizeLightPathLength, .0005f);
			}
		}

		// hash grid
		if (mAlgorithm == VcmAlgorithmType::kPpm || mAlgorithm == VcmAlgorithmType::kBpm || mAlgorithm == VcmAlgorithmType::kVcm) {
			if (ImGui::DragScalar("Photon hash grid cells", ImGuiDataType_U32, &mLightHashGrid.mCellCount)) changed = true;
			if (ImGui::DragFloat("Photon hash grid min cell size", &mLightHashGrid.mCellSize, .01f, 1e-4f, 1000)) changed = true;
			if (ImGui::DragFloat("Photon hash grid cell pixel radius", &mLightHashGrid.mCellPixelRadius, .5f, 0, 1000)) changed = true;
		}
		if (mAlgorithm != VcmAlgorithmType::kLightTrace && (mDIReservoirFlags & VcmReservoirFlags::eReuse))  {
			if (ImGui::DragScalar("DI hash grid cells", ImGuiDataType_U32, &mDIHashGrid.mCellCount)) changed = true;
			if (ImGui::DragFloat("DI hash grid min cell size", &mDIHashGrid.mCellSize, .01f, 1e-4f, 1000)) changed = true;
			if (ImGui::DragFloat("DI hash grid cell pixel radius", &mDIHashGrid.mCellPixelRadius, .5f, 0, 1000)) changed = true;
		}
		if (mAlgorithm == VcmAlgorithmType::kBpt && mUseLightVertexCache && (mLVCReservoirFlags & VcmReservoirFlags::eReuse)) {
			if (ImGui::DragScalar("LVC hash grid cells", ImGuiDataType_U32, &mLVCHashGrid.mCellCount)) changed = true;
			if (ImGui::DragFloat("LVC hash grid min cell size", &mLVCHashGrid.mCellSize, .01f, 1e-4f, 1000)) changed = true;
			if (ImGui::DragFloat("LVC hash grid cell pixel radius", &mLVCHashGrid.mCellPixelRadius, .5f, 0, 1000)) changed = true;
		}

		if (mAlgorithm == VcmAlgorithmType::kPpm || mAlgorithm == VcmAlgorithmType::kBpm || mAlgorithm == VcmAlgorithmType::kVcm) {
			if (ImGui::SliderFloat("Merge radius alpha", &mVmRadiusAlpha, -1, 1)) changed = true;
			if (ImGui::SliderFloat("Merge radius factor", &mVmRadiusFactor, 0, 1)) changed = true;
		}

		ImGui::PopItemWidth();

		if (ImGui::Checkbox("Debug paths", &mDebugPaths)) changed = true;
		if (mDebugPaths) {
			if (ImGui::Checkbox("Weights", &mDebugPathWeights)) changed = true;
			ImGui::PushItemWidth(60);
			uint32_t cameraLength = mPushConstants.debugCameraPathLength();
			uint32_t lightLength = mPushConstants.debugLightPathLength();
			if (ImGui::DragScalar("Camera vertices", ImGuiDataType_U32, &cameraLength)) changed = true;
			if (ImGui::DragScalar("Light vertices", ImGuiDataType_U32, &lightLength)) changed = true;
			mPushConstants.debugCameraPathLength(cameraLength);
			mPushConstants.debugLightPathLength(lightLength);
			ImGui::PopItemWidth();
		}

		if (auto denoiser = mNode.getComponent<Denoiser>(); denoiser) {
			if (ImGui::Checkbox("Enable denoiser", &mDenoise))
				changed = true;

			if (changed && !ImGui::GetIO().KeyAlt)
				denoiser->resetAccumulation();
		}

		if (auto tonemapper = mNode.getComponent<Tonemapper>(); tonemapper)
			ImGui::Checkbox("Enable tonemapper", &mTonemap);
	}

	if (changed && mDenoise) {
		if (auto denoiser = mNode.getComponent<Denoiser>(); denoiser)
			denoiser->resetAccumulation();
	}
}

void VCM::render(CommandBuffer& commandBuffer, const Image::View& renderTarget) {
	if (mPauseRendering) return;

	ProfilerScope ps("VCM::render", &commandBuffer);

	const shared_ptr<Scene> scene = mNode.findAncestor<Scene>();
	const Scene::FrameData& sceneData = scene->frameData();
	if (sceneData.mDescriptors.empty()) {
		renderTarget.clearColor(commandBuffer, vk::ClearColorValue(array<uint32_t, 4>{ 0, 0, 0, 0 }));
		return;
	}

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

	bool changed = false;
	if (scene->lastUpdate() > mLastSceneVersion) {
		changed = true;
		mLastSceneVersion = scene->lastUpdate();
	}

	bool has_volumes = false;

	Descriptors descriptors;

	// copy camera/view data to GPU, compute gViewMediumInstances
	vector<ViewData>      views;
	vector<TransformData> viewTransforms;
	{
		ProfilerScope ps("Upload views", &commandBuffer);

		vector<TransformData> viewInverseTransforms;

		mNode.root()->forEachDescendant<Camera>([&](Node& node, const shared_ptr<Camera>& camera) {
			views.emplace_back(camera->view());
			TransformData t = nodeToWorld(node);
			viewTransforms.emplace_back(t);
			viewInverseTransforms.emplace_back(t.inverse());
		});
		mPushConstants.mViewCount = (uint32_t)views.size();

		if (views.empty()) {
			renderTarget.clearColor(commandBuffer, vk::ClearColorValue(array<uint32_t, 4>{ 0, 0, 0, 0 }));
			return;
		}

		descriptors[{"gRenderParams.mViews",0}]                 = mResourcePool.uploadData<ViewData>     (commandBuffer, "mViews"                , views);
		descriptors[{"gRenderParams.mViewTransforms",0}]        = mResourcePool.uploadData<TransformData>(commandBuffer, "mViewTransforms"       , viewTransforms);
		descriptors[{"gRenderParams.mViewInverseTransforms",0}] = mResourcePool.uploadData<TransformData>(commandBuffer, "mViewInverseTransforms", viewInverseTransforms);

		vector<TransformData> prevInverseViewTransforms(views.size());
		for (uint32_t i = 0; i < mPrevViewTransforms.size(); i++) {
			prevInverseViewTransforms[i] = mPrevViewTransforms[i].inverse();
		}
		descriptors[{"gRenderParams.mPrevInverseViewTransforms",0}] = mResourcePool.uploadData<TransformData>(commandBuffer, "mPrevInverseViewTransforms", prevInverseViewTransforms);

		if (!mPrevViewTransforms.empty() && (mPrevViewTransforms[0].m != viewTransforms[0].m).any())
			changed = true;
		mPrevViewTransforms = viewTransforms;

		// find if views are inside a volume
		vector<uint32_t> viewMediumIndices(views.size());
		ranges::fill(viewMediumIndices, INVALID_INSTANCE);
		mNode.forEachDescendant<Medium>([&](Node& node, const shared_ptr<Medium>& vol) {
			has_volumes = true;
			for (uint32_t i = 0; i < views.size(); i++) {
				const float3 localViewPos = nodeToWorld(node).inverse().transformPoint( viewTransforms[i].transformPoint(float3::Zero()) );
				if (vol->mDensityGrid->grid<float>()->worldBBox().isInside(nanovdb::Vec3R(localViewPos[0], localViewPos[1], localViewPos[2])))
					viewMediumIndices[i] = sceneData.mInstanceTransformMap.at(vol.get()).second;
			}
		});
		descriptors[{"gRenderParams.mViewMediumInstances",0}] = mResourcePool.uploadData<uint32_t>(commandBuffer, "mViewMediumInstances", viewMediumIndices);
	}

	const vk::Extent3D extent = renderTarget.extent();
	mPushConstants.mOutputExtent = uint2(extent.width, extent.height);
	mPushConstants.mScreenPixelCount = extent.width * extent.height;
	mPushConstants.mLightSubPathCount = max<uint32_t>(1, mPushConstants.mScreenPixelCount * mLightPathPercent);
	const uint32_t maxLightVertices  = mPushConstants.mLightSubPathCount*mPushConstants.mMaxPathLength;
	const uint32_t maxCameraVertices = mPushConstants.mLightSubPathCount*mPushConstants.mMaxPathLength;

	// specify defines
	bool useVC = false;
	bool useVM = false;
	Defines defines {
		{ "gHasMedia",            to_string(has_volumes) },
		{ "gDebugPaths",          to_string(mDebugPaths) },
		{ "gDebugPathWeights",    to_string(mDebugPathWeights) },
		{ "gDIReservoirFlags",    to_string((uint32_t)mDIReservoirFlags) },
		{ "gLVCReservoirFlags",   to_string((uint32_t)mLVCReservoirFlags) },
		{ "gNormalMaps",          to_string(mUseNormalMaps) },
		{ "gShadingNormals",      to_string(mUseShadingNormals) },
		{ "gAlphaTest" ,          to_string(mUseAlphaTesting) },
	};
	switch (mAlgorithm) {
	case VcmAlgorithmType::kPathTrace:
		defines["gPathTraceOnly"] = "true";
		break;
	case VcmAlgorithmType::kLightTrace:
		defines["gLightTraceOnly"] = "true";
		break;
	case VcmAlgorithmType::kPpm:
		defines["gPpm"]   = "true";
		defines["gUseVM"] = "true";
		useVM = true;
		break;
	case VcmAlgorithmType::kBpm:
		defines["gUseVM"] = "true";
		useVM = true;
		break;
	case VcmAlgorithmType::kBpt:
		defines["gUseVC"] = "true";
		useVC = true;
		if (mUseLightVertexCache) {
			defines["gUseLVC"] = "true";
		}
		break;
	case VcmAlgorithmType::kVcm:
		defines["gUseVC"] = "true";
		defines["gUseVM"] = "true";
		useVC = true;
		useVM = true;
		break;
	default:
		printf("Unknown algorithm type\n");
		break;
	}


	GpuHashGrid::FrameData lightHashGrid, diReservoirHashGrid, lvcReservoirHashGrid;

	// update values, allocate data
	{
		mPushConstants.reservoirHistoryValid(!mPrevDescriptors.empty() && !ImGui::IsKeyPressed(ImGuiKey_F5));
		mPushConstants.mEnvironmentMaterialAddress = sceneData.mEnvironmentMaterialAddress;
		mPushConstants.mLightCount = sceneData.mLightCount;
		if (mRandomPerFrame) mPushConstants.mRandomSeed = (mDenoise && denoiser) ? denoiser->accumulatedFrames() : rand();

		// update vcm constants
		{
			VcmConstants constants;

			const float3 center = (sceneData.mAabbMin + sceneData.mAabbMax) / 2;
			constants.mSceneSphere = float4(center[0], center[1], center[2], length<float,3>(sceneData.mAabbMax - center));

			const uint32_t vmIteration = (mDenoise && denoiser) ? denoiser->accumulatedFrames() : 0;

			// Setup our radius, 1st iteration has aIteration == 0, thus offset
			float radius = mVmRadiusFactor * constants.mSceneSphere[3];
			radius /= pow(float(vmIteration + 1), 0.5f * (1 - mVmRadiusAlpha));
			// Purely for numeric stability
			constants.mMergeRadius = max(radius, 1e-7f);

			const float mergeRadiusSqr = pow2(constants.mMergeRadius);

			// Factor used to normalise vertex merging contribution.
			// We divide the summed up energy by disk radius and number of light paths
			constants.mVmNormalization = 1.f / (mergeRadiusSqr * M_PI * mPushConstants.mLightSubPathCount);

			// MIS weight constant [tech. rep. (20)], with n_VC = 1 and n_VM = mLightPathCount
			const float etaVCM = (M_PI * mergeRadiusSqr) * mPushConstants.mLightSubPathCount;
			constants.mMisVmWeightFactor = useVM ? Mis(etaVCM) : 0.f;
			constants.mMisVcWeightFactor = useVC ? Mis(1.f / etaVCM) : 0.f;

			descriptors[{"gRenderParams.mVcmConstants",0}] = mResourcePool.uploadData<VcmConstants>(commandBuffer, "gRenderParams.mVcmConstants", constants, vk::BufferUsageFlagBits::eUniformBuffer);
		}

		// allocate data

		ProfilerScope ps("Allocate data");

		auto usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;

		descriptors[{"gRenderParams.mOutput",0}]     = ImageDescriptor{mResourcePool.getImage(commandBuffer.mDevice, "mOutput",     Image::Metadata{ .mFormat = vk::Format::eR32G32B32A32Sfloat, .mExtent = extent, .mUsage = usage }), vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {}};
		descriptors[{"gRenderParams.mAlbedo",0}]     = ImageDescriptor{mResourcePool.getImage(commandBuffer.mDevice, "mAlbedo",     Image::Metadata{ .mFormat = vk::Format::eR16G16B16A16Sfloat, .mExtent = extent, .mUsage = usage }), vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {}};
		descriptors[{"gRenderParams.mPrevUVs",0}]    = ImageDescriptor{mResourcePool.getImage(commandBuffer.mDevice, "mPrevUVs",    Image::Metadata{ .mFormat = vk::Format::eR32G32Sfloat,       .mExtent = extent, .mUsage = usage }), vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {}};
		descriptors[{"gRenderParams.mVisibility",0}] = ImageDescriptor{mResourcePool.getImage(commandBuffer.mDevice, "mVisibility", Image::Metadata{ .mFormat = vk::Format::eR32G32Uint,         .mExtent = extent, .mUsage = usage }), vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {}};
		descriptors[{"gRenderParams.mDepth",0}]      = ImageDescriptor{mResourcePool.getImage(commandBuffer.mDevice, "mDepth" ,     Image::Metadata{ .mFormat = vk::Format::eR32G32B32A32Sfloat, .mExtent = extent, .mUsage = usage }), vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {}};

		descriptors[{"gRenderParams.mLightImage",0}]       = mResourcePool.getBuffer<uint4>          (commandBuffer.mDevice, "mLightImage", mPushConstants.mScreenPixelCount*sizeof(uint4), vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eStorageBuffer);
		descriptors[{"gRenderParams.mLightVertices",0}]    = mResourcePool.getBuffer<PackedVcmVertex>(commandBuffer.mDevice, "mLightVertices", maxLightVertices);
		descriptors[{"gRenderParams.mLightPathLengths",0}] = mResourcePool.getBuffer<uint32_t>       (commandBuffer.mDevice, "mLightPathLengths", mPushConstants.mLightSubPathCount, vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eStorageBuffer);

		mLightHashGrid.mSize = maxLightVertices;
		mDIHashGrid.mSize    = maxCameraVertices;
		mLVCHashGrid.mSize   = maxCameraVertices;

		GpuHashGrid::Metadata md{
			.mCameraPosition = viewTransforms[0].transformPoint(float3::Zero()),
			.mVerticalFoV = views[0].mProjection.mVerticalFoV,
			.mImageExtent = mPushConstants.mOutputExtent,
		};
		lightHashGrid        = mLightHashGrid.init(commandBuffer, descriptors, "mLightHashGrid", md);
		diReservoirHashGrid  = mDIHashGrid.init(commandBuffer, descriptors, "mDirectIlluminationReservoirs", md, "mPrevDirectIlluminationReservoirs");
		lvcReservoirHashGrid = mLVCHashGrid.init(commandBuffer, descriptors, "mLVCReservoirs", md, "mPrevLVCReservoirs");
	}

	mPrevDescriptors = descriptors;

	// create mDescriptorSets
	shared_ptr<DescriptorSets> descriptorSets;
	{
		ProfilerScope ps("Assign descriptors", &commandBuffer);

		for (auto& [name, d] : sceneData.mDescriptors)
			descriptors[{ "gScene." + name.first, name.second }] = d;
		descriptors[{ "gScene.mRayCount", 0u }] = make_shared<Buffer>(commandBuffer.mDevice, "mRayCount", 2*sizeof(uint32_t), vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

		// track resources which are not held by the descriptorset
		commandBuffer.trackResource(sceneData.mAccelerationStructureBuffer.buffer());

		descriptorSets = mRenderPipelines[eGenerateLightPaths].get(commandBuffer.mDevice, defines)->getDescriptorSets(descriptors);
	}


	bool loading = false;

	auto loadPipeline = [&](const RenderPipelineIndex idx, const Defines& defs) {
		const auto& p = mRenderPipelines[idx].getAsync(commandBuffer.mDevice, defs);
		if (!p)
			loading = true;
		return p;
	};

	shared_ptr<ComputePipeline> generateLightPathsPipeline, generateCameraPathsPipeline;

	if (mAlgorithm != VcmAlgorithmType::kPathTrace)
		generateLightPathsPipeline = loadPipeline(eGenerateLightPaths, defines);
	generateCameraPathsPipeline = loadPipeline(eGenerateCameraPaths, defines);

	if (loading) {
		// compiling shaders...
		renderTarget.clearColor(commandBuffer, vk::ClearColorValue(array<float, 4>{ 0.5f, 0.5f, 0.5f, 0 }));
		return;
	}

	// rendering
	{
		// clearing things
		{
			if (useVM)
				mLightHashGrid.clear(commandBuffer, lightHashGrid);
			if ((mDIReservoirFlags & VcmReservoirFlags::eReuse) && mAlgorithm != VcmAlgorithmType::kPpm && mAlgorithm != VcmAlgorithmType::kLightTrace)
				mDIHashGrid.clear(commandBuffer, diReservoirHashGrid);
			if ((mLVCReservoirFlags & VcmReservoirFlags::eReuse) && mAlgorithm == VcmAlgorithmType::kBpt && mUseLightVertexCache)
				mLVCHashGrid.clear(commandBuffer, lvcReservoirHashGrid);

			// clear light image
			if (mAlgorithm != VcmAlgorithmType::kPathTrace) {
				get<BufferDescriptor>(descriptors.at({"gRenderParams.mLightImage",0})).fill(commandBuffer, 0);
				get<BufferDescriptor>(descriptors.at({"gRenderParams.mLightImage",0})).barrier(commandBuffer,
					vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
					vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
			}

			// clear light vertex counter
			if (mAlgorithm == VcmAlgorithmType::kBpt && mUseLightVertexCache) {
				const Buffer::View<byte>& lightPathLengths = get<BufferDescriptor>(descriptors.at({"gRenderParams.mLightPathLengths",0}));
				Buffer::View<uint32_t> lightPathLengths0(lightPathLengths.buffer(), lightPathLengths.offset(), 2);

				lightPathLengths0.fill(commandBuffer, 0);
				lightPathLengths0.barrier(commandBuffer,
					vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
					vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
			}
		}

		// generate light paths
		if (mAlgorithm != VcmAlgorithmType::kPathTrace) {
			ProfilerScope ps("Generate light paths", &commandBuffer);

			const vk::Extent3D lightExtent(extent.width, (mPushConstants.mLightSubPathCount + extent.width-1) / extent.width, 1);
			generateLightPathsPipeline->dispatchTiled(commandBuffer, lightExtent, descriptorSets, {}, { { "", PushConstantValue(mPushConstants) } });

			commandBuffer->pipelineBarrier(
				vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
				vk::DependencyFlagBits::eByRegion,
				vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead), {}, {});

			// Build hash grid over light vertices
			if (useVM)
				mLightHashGrid.build(commandBuffer, lightHashGrid);
		}

		// generate camera paths
		{
			ProfilerScope ps("Generate camera paths", &commandBuffer);
			generateCameraPathsPipeline->dispatchTiled(commandBuffer, extent, descriptorSets, {}, { { "", PushConstantValue(mPushConstants) } });
		}

		// build hashgrids
		if ((mDIReservoirFlags & VcmReservoirFlags::eReuse) && mAlgorithm != VcmAlgorithmType::kPpm && mAlgorithm != VcmAlgorithmType::kLightTrace)
			mDIHashGrid.build(commandBuffer, diReservoirHashGrid);
		if ((mLVCReservoirFlags & VcmReservoirFlags::eReuse) && mAlgorithm == VcmAlgorithmType::kBpt && mUseLightVertexCache)
			mLVCHashGrid.build(commandBuffer, lvcReservoirHashGrid);
	}

	const Image::View& outputImage = get<Image::View>(get<ImageDescriptor>(descriptors.at({"gRenderParams.mOutput",0})));
	const Image::View& albedoImage = get<Image::View>(get<ImageDescriptor>(descriptors.at({"gRenderParams.mAlbedo",0})));
	Image::View processedOutput = outputImage;


	// post processing
	{
		// run denoiser
		if (mDenoise && denoiser && mRandomPerFrame) {
			if (changed && !denoiser->reprojection())
				denoiser->resetAccumulation();

			processedOutput = denoiser->denoise(
				commandBuffer,
				outputImage,
				albedoImage,
				get<Image::View>(get<ImageDescriptor>(descriptors.at({"gRenderParams.mPrevUVs",0}))),
				get<Image::View>(get<ImageDescriptor>(descriptors.at({"gRenderParams.mVisibility",0}))),
				get<Image::View>(get<ImageDescriptor>(descriptors.at({"gRenderParams.mDepth",0}))),
				get<BufferDescriptor>(descriptors.at({"gRenderParams.mViews",0})).cast<ViewData>() );
		}

		// run tonemapper
		if (mTonemap && tonemapper) {
			tonemapper->render(commandBuffer, processedOutput, outputImage, (mDenoise && denoiser && denoiser->demodulateAlbedo()) ? albedoImage : Image::View{});
			// copy outputImage to renderTarget
			if (outputImage.image()->format() == renderTarget.image()->format())
				Image::copy(commandBuffer, outputImage, renderTarget);
			else
				Image::blit(commandBuffer, outputImage, renderTarget);
			mLastResultImage = outputImage;
		} else {
			// copy processedOutput to renderTarget
			if (processedOutput.image()->format() == renderTarget.image()->format())
				Image::copy(commandBuffer, processedOutput, renderTarget);
			else
				Image::blit(commandBuffer, processedOutput, renderTarget);
			mLastResultImage = processedOutput;
		}
	}

	// copy VisibilityData for selected pixel for scene object picking
	if (!ImGui::GetIO().WantCaptureMouse && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGuizmo::IsUsing()) {
		const ImVec2 c = ImGui::GetIO().MousePos;
		for (const ViewData& view : views)
			if (view.isInside(int2(c.x, c.y))) {
				const Image::View& visibilityImage = get<Image::View>(get<ImageDescriptor>(descriptors.at({"gRenderParams.mVisibility", 0})));
				Buffer::View<VisibilityData> selectionBuffer = make_shared<Buffer>(commandBuffer.mDevice, "SelectionData", sizeof(uint2), vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
				visibilityImage.barrier(commandBuffer, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferRead);
				selectionBuffer.copyFromImage(commandBuffer, visibilityImage.image(), visibilityImage.subresourceLayer(), vk::Offset3D{int(c.x), int(c.y), 0}, vk::Extent3D{1,1,1});
				mSelectionData.push_back(make_pair(selectionBuffer, ImGui::GetIO().KeyShift));
				break;
			}
	}

	// visualize paths
	if (mVisualizeLightPaths && mAlgorithm != VcmAlgorithmType::kPathTrace)
		rasterLightPaths(commandBuffer, renderTarget);

}

void VCM::rasterLightPaths(CommandBuffer& commandBuffer, const Image::View& renderTarget) {
#if 0
	if (renderTarget.image()->format() != mRasterLightPathPipeline.pipelineMetadata().mDynamicRenderingState->mColorFormats[0])
		createRasterPipeline(commandBuffer.mDevice, vk::Extent2D(renderTarget.extent().width, renderTarget.extent().height), renderTarget.image()->format());

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
#endif
}

}