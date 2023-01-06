#include "PathTracer.hpp"
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

PathTracer::PathTracer(Node& node) : mNode(node) {
	if (shared_ptr<Inspector> inspector = mNode.root()->findDescendant<Inspector>())
		inspector->setInspectCallback<PathTracer>();

	Device& device = *mNode.findAncestor<Device>();
	createPipelines(device);

	mPushConstants.mMinPathLength = 0;
	mPushConstants.mMaxPathLength = 6;
	mPushConstants.mEnvironmentSampleProbability = 0.5f;
	mPushConstants.mHashGridCellCount = 131072;
	mPushConstants.mLightImageQuantization = 16384;

	// initialize constants
	if (auto arg = device.mInstance.findArgument("minPathLength"); arg) mPushConstants.mMinPathLength = atoi(arg->c_str());
	if (auto arg = device.mInstance.findArgument("maxPathLength"); arg) mPushConstants.mMaxPathLength = atoi(arg->c_str());
}

void PathTracer::createPipelines(Device& device) {
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
	mRenderPipelines[RenderPipelineIndex::eHashGridComputeIndices] = ComputePipelineCache(kernelPath, "ComputeHashGridIndices", "sm_6_6", args, md);
	mRenderPipelines[RenderPipelineIndex::eHashGridSwizzle]        = ComputePipelineCache(kernelPath, "SwizzleHashGrid"       , "sm_6_6", args, md);

	auto swapchain = mNode.root()->findDescendant<Swapchain>();
	createRasterPipeline(device, swapchain->extent(), swapchain->format().format);

	mPerformanceCounters = make_shared<Buffer>(device, "mPerformanceCounters", 4*sizeof(uint32_t), vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	mPrevPerformanceCounters.resize(mPerformanceCounters.size());
	mPerformanceCounterPerSecond.resize(mPerformanceCounters.size());
	ranges::fill(mPerformanceCounters, 0);
	ranges::fill(mPrevPerformanceCounters, 0);
	mPerformanceCounterTimer = 0;
}

void PathTracer::createRasterPipeline(Device& device, const vk::Extent2D& extent, const vk::Format format) {
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

void PathTracer::drawGui() {
	bool changed = false;

	ImGui::PushID(this);
	if (ImGui::Button("Clear resources")) {
		mFrameResourcePool.clear();
		mPrevFrame.reset();
	}

	if (ImGui::Button("Reload shaders")) {
		changed = true;
		Device& device = *mNode.findAncestor<Device>();
		device->waitIdle();
		createPipelines(device);
	}

	ImGui::PopID();

	if (changed && mDenoise) {
		if (auto denoiser = mNode.getComponent<Denoiser>(); denoiser)
			denoiser->resetAccumulation();
	}

	{
		bool changed = false;
		if (Gui::enumDropdown<VcmAlgorithmType>("Algorithm", mAlgorithm, VcmAlgorithmType::kNumVcmAlgorithmType)) changed = true;
		if (ImGui::Checkbox("Random frame seed", &mRandomPerFrame)) changed = true;
		if (ImGui::Checkbox("Performance counters", &mUsePerformanceCounters)) changed = true;

		ImGui::PushItemWidth(60);


		if (ImGui::DragScalar("Max path length", ImGuiDataType_U32, &mPushConstants.mMaxPathLength)) changed = true;
		if (ImGui::DragScalar("Min path length", ImGuiDataType_U32, &mPushConstants.mMinPathLength)) changed = true;

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
		if (mAlgorithm == VcmAlgorithmType::kPpm || mAlgorithm == VcmAlgorithmType::kBpm || mAlgorithm == VcmAlgorithmType::kVcm) {
			if (ImGui::DragScalar("Hash grid size", ImGuiDataType_U32, &mPushConstants.mHashGridCellCount)) changed = true;
			if (ImGui::SliderFloat("Merge radius alpha", &mVmRadiusAlpha, -1, 1)) changed = true;
			if (ImGui::SliderFloat("Merge radius factor", &mVmRadiusFactor, 0, 1)) changed = true;
			ImGui::Text("Current merge radius: %f", mPushConstants.mMergeRadius);
		}

		if (mPushConstants.mEnvironmentMaterialAddress != -1 && mPushConstants.mLightCount > 0)
			if (ImGui::SliderFloat("Environment sample probability", &mPushConstants.mEnvironmentSampleProbability, 0, 1)) changed = true;

		ImGui::PopItemWidth();

		if (ImGui::Checkbox("Debug paths", &mDebugPaths)) changed = true;
		if (mDebugPaths) {
			if (ImGui::Checkbox("Weights", &mDebugPathWeights)) changed = true;
			ImGui::PushItemWidth(60);
			if (ImGui::DragScalar("Camera vertices", ImGuiDataType_U32, &mPushConstants.mDebugCameraPathLength)) changed = true;
			if (ImGui::DragScalar("Light vertices", ImGuiDataType_U32, &mPushConstants.mDebugLightPathLength)) changed = true;
			ImGui::PopItemWidth();
		}

		if (auto denoiser = mNode.getComponent<Denoiser>(); denoiser) {
			if (ImGui::Checkbox("Enable denoiser", &mDenoise))
				changed = true;

			if (changed)
				denoiser->resetAccumulation();
		}

		if (auto tonemapper = mNode.getComponent<Tonemapper>(); tonemapper)
			ImGui::Checkbox("Enable tonemapper", &mTonemap);
	}

	if (mUsePerformanceCounters) {
		const auto [rps, ext] = formatNumber(mPerformanceCounterPerSecond[0]);
		ImGui::Text("%.2f%s Rays/second (%u%% visibility)", rps, ext, mPerformanceCounterPerSecond[0] == 0 ? 0 : (uint32_t)(100*mPerformanceCounterPerSecond[1] / mPerformanceCounterPerSecond[0]));
		ImGui::Text("%u failed inserts, %u Buckets used", mHashGridStats[0], mHashGridStats[1]);
	}
}

void PathTracer::render(CommandBuffer& commandBuffer, const Image::View& renderTarget) {
	ProfilerScope ps("PathTracer::render", &commandBuffer);

	const shared_ptr<Scene> scene = mNode.findAncestor<Scene>();
	const shared_ptr<Scene::FrameResources> sceneResources = scene->resources();
	if (!sceneResources) {
		renderTarget.clearColor(commandBuffer, vk::ClearColorValue(array<uint32_t, 4>{ 0, 0, 0, 0 }));
		return;
	}

	const shared_ptr<Denoiser>   denoiser   = mNode.findDescendant<Denoiser>();
	const shared_ptr<Tonemapper> tonemapper = mNode.findDescendant<Tonemapper>();

	// reuse old frame resources
	auto frame = mFrameResourcePool.get();
	if (!frame)
		frame = mFrameResourcePool.emplace(make_shared<FrameResources>(commandBuffer.mDevice));
	commandBuffer.trackResource(frame);

	frame->mSceneData = sceneResources;

	// count rays per second
	{
		frame->mTime = chrono::high_resolution_clock::now();
		if (mPrevFrame) {
			const float deltaTime = chrono::duration_cast<chrono::duration<float>>(frame->mTime - mPrevFrame->mTime).count();
			mPerformanceCounterTimer += deltaTime;
			if (mPerformanceCounterTimer >= 1) {
				for (uint32_t i = 0; i < mPerformanceCounterPerSecond.size(); i++)
					mPerformanceCounterPerSecond[i] = (mPerformanceCounters[i] - mPrevPerformanceCounters[i]) / mPerformanceCounterTimer;
				ranges::copy(mPerformanceCounters, mPrevPerformanceCounters.begin());
				mPerformanceCounterTimer -= 1;
			}
		}
	}

	// scene object picking using frame's mSelectionData
	{
		ProfilerScope ps("Object picking");
		if (frame && frame->mSelectionData && frame->mSelectionDataValid) {
			const uint32_t selectedInstance = frame->mSelectionData.data()->instanceIndex();
			if (shared_ptr<Inspector> inspector = mNode.root()->findDescendant<Inspector>()) {
				if (selectedInstance == INVALID_INSTANCE || selectedInstance >= frame->mSceneData->mInstanceNodes.size())
					inspector->select(nullptr);
				else {
					if (shared_ptr<Node> selected = frame->mSceneData->mInstanceNodes[selectedInstance].lock()) {
						if (frame->mSelectionShift) {
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
		}
	}

	if (auto it = frame->mBuffers.find("mLightHashGrid.mStats"); it != frame->mBuffers.end())
		mHashGridStats = it->second.cast<uint2>()[0];

	bool has_volumes = false;

	// copy camera/view data to GPU, compute gViewMediumInstances
	{
		ProfilerScope ps("Upload views", &commandBuffer);

		frame->mViews.clear();
		mNode.root()->forEachDescendant<Camera>([&](Node& node, const shared_ptr<Camera>& camera) {
			frame->mViews.emplace_back(pair{ camera->view(), nodeToWorld(node) });
		});
		mPushConstants.mViewCount = (uint32_t)frame->mViews.size();

		if (frame->mViews.empty()) {
			renderTarget.clearColor(commandBuffer, vk::ClearColorValue(array<uint32_t, 4>{ 0, 0, 0, 0 }));
			cout << "Warning: No views" << endl;
			return;
		}

		// upload viewdata
		auto viewsBuffer                 = frame->getBuffer<ViewData>     ("mViews"                , frame->mViews.size(), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		auto viewTransformsBuffer        = frame->getBuffer<TransformData>("mViewTransforms"       , frame->mViews.size(), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		auto viewInverseTransformsBuffer = frame->getBuffer<TransformData>("mViewInverseTransforms", frame->mViews.size(), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		for (uint32_t i = 0; i < frame->mViews.size(); i++) {
			viewsBuffer[i] = frame->mViews[i].first;
			viewTransformsBuffer[i] = frame->mViews[i].second;
			viewInverseTransformsBuffer[i] = frame->mViews[i].second.inverse();
		}

		auto prevViews                 = frame->getBuffer<ViewData>     ("mPrevViews"                , frame->mViews.size(), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		auto prevInverseViewTransforms = frame->getBuffer<TransformData>("mPrevInverseViewTransforms", frame->mViews.size(), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		if (mPrevFrame) {
			for (uint32_t i = 0; i < mPrevFrame->mViews.size(); i++) {
				prevViews[i] = mPrevFrame->mViews[i].first;
				prevInverseViewTransforms[i] = mPrevFrame->mViews[i].second.inverse();
			}
		}


		// find if views are inside a volume
		auto viewMediumIndicesBuffer = frame->getBuffer<uint32_t>("mViewMediumInstances", frame->mViews.size(), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		ranges::fill(viewMediumIndicesBuffer, INVALID_INSTANCE);

		mNode.forEachDescendant<Medium>([&](Node& node, const shared_ptr<Medium>& vol) {
			has_volumes = true;
			for (uint32_t i = 0; i < frame->mViews.size(); i++) {
				const float3 localViewPos = nodeToWorld(node).inverse().transformPoint( frame->mViews[i].second.transformPoint(float3::Zero()) );
				if (vol->mDensityGrid->grid<float>()->worldBBox().isInside(nanovdb::Vec3R(localViewPos[0], localViewPos[1], localViewPos[2])))
					viewMediumIndicesBuffer[i] = frame->mSceneData->mInstanceTransformMap.at(vol.get()).second;
			}
		});
	}


	auto makeHashGridBuffers = [&](const string& name, const uint32_t cellCount, const uint32_t maxElements) {
		frame->getBuffer<uint> (name + ".mChecksums"    , cellCount, vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eStorageBuffer);
		frame->getBuffer<uint> (name + ".mCounters"     , cellCount, vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eStorageBuffer);
		frame->getBuffer<uint2>(name + ".mAppendIndices", 1+maxElements, vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eStorageBuffer);
		frame->getBuffer<uint> (name + ".mAppendData"   , maxElements);
		frame->getBuffer<uint> (name + ".mIndices"      , cellCount);
		frame->getBuffer<uint> (name + ".mData"         , maxElements);
		frame->getBuffer<uint> (name + ".mStats", 2, vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
	};

	auto clearHashGrid = [&](const string& name) {
		const Buffer::View<byte>& appendIndices = frame->mBuffers.at(name + ".mAppendIndices");
		Buffer::View<byte> appendIndices0(appendIndices.buffer(), appendIndices.offset(), sizeof(uint2));

		frame->mBuffers.at(name + ".mChecksums").fill(commandBuffer, 0);
		frame->mBuffers.at(name + ".mCounters").fill(commandBuffer, 0);
		appendIndices0.fill(commandBuffer, 0);

		Buffer::barriers(commandBuffer,
			{
				frame->mBuffers.at(name + ".mChecksums"),
				frame->mBuffers.at(name + ".mCounters"),
				appendIndices0
			},
			vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
			vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);

		if (mUsePerformanceCounters) {
			frame->mBuffers.at(name + ".mStats").cast<uint2>()[0] = uint2::Zero();
		}
	};


	const vk::Extent3D extent = renderTarget.extent();
	mPushConstants.mOutputExtent = uint2(extent.width, extent.height);
	mPushConstants.mScreenPixelCount = extent.width * extent.height;
	mPushConstants.mLightSubPathCount = max<uint32_t>(1, mPushConstants.mScreenPixelCount * mLightPathPercent);
	const uint32_t maxLightVertices = mPushConstants.mLightSubPathCount*mPushConstants.mMaxPathLength;

	// allocate data if needed
	{
		ProfilerScope ps("Allocate data", &commandBuffer);

		auto usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;

		frame->getImage("mOutput",     extent, vk::Format::eR32G32B32A32Sfloat, usage);
		frame->getImage("mAlbedo",     extent, vk::Format::eR16G16B16A16Sfloat, usage);
		frame->getImage("mPrevUVs",    extent, vk::Format::eR32G32Sfloat, usage);
		frame->getImage("mVisibility", extent, vk::Format::eR32G32Uint, usage);
		frame->getImage("mDepth" ,     extent, vk::Format::eR32G32B32A32Sfloat, usage);

		frame->getBuffer<uint4>("mLightImage", mPushConstants.mScreenPixelCount*sizeof(uint4), vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eStorageBuffer);
		frame->getBuffer<VcmVertex>("mLightVertices", maxLightVertices);
		frame->getBuffer<uint32_t>("mLightPathLengths", mPushConstants.mLightSubPathCount + 1, vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eStorageBuffer);

		makeHashGridBuffers("mLightHashGrid", mPushConstants.mHashGridCellCount, maxLightVertices);

		if (!frame->mSelectionData)
			frame->mSelectionData = make_shared<Buffer>(commandBuffer.mDevice, "mSelectionData", sizeof(VisibilityData), vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
	}

	// specify defines
	Defines defines {
		{ "gHasMedia", to_string(has_volumes) },
		{ "gPerformanceCounters", to_string(mUsePerformanceCounters) },
		{ "gDebugPaths", to_string(mDebugPaths) },
		{ "gDebugPathWeights", to_string(mDebugPathWeights) },
	};
	switch (mAlgorithm) {
	case VcmAlgorithmType::kPathTrace:
		defines["gPathTraceOnly"] = "true";
		break;
	case VcmAlgorithmType::kLightTrace:
		defines["gLightTraceOnly"] = "true";
		break;
	case VcmAlgorithmType::kPpm:
		defines["gPpm"] = "true";
		defines["gUseVM"] = "true";
		break;
	case VcmAlgorithmType::kBpm:
		defines["gUseVM"] = "true";
		break;
	case VcmAlgorithmType::kBpt:
		defines["gUseVC"] = "true";
		break;
	case VcmAlgorithmType::kBptLvc:
		defines["gUseVC"] = "true";
		defines["gUseLVC"] = "true";
		break;
	case VcmAlgorithmType::kVcm:
		defines["gUseVC"] = "true";
		defines["gUseVM"] = "true";
		break;
	default:
		printf("Unknown algorithm type\n");
		break;
	}

	// update push constants
	{
		mPushConstants.mEnvironmentMaterialAddress = frame->mSceneData->mEnvironmentMaterialAddress;
		mPushConstants.mLightCount = frame->mSceneData->mLightCount;

		if (mRandomPerFrame) mPushConstants.mRandomSeed = rand();

		const float3 center = (frame->mSceneData->mAabbMin + frame->mSceneData->mAabbMax) / 2;
		mPushConstants.mSceneSphere = float4(center[0], center[1], center[2], length<float,3>(frame->mSceneData->mAabbMax - center));

		const uint32_t vmIteration = (mDenoise && denoiser) ? denoiser->accumulatedFrames() : 0;

        // Setup our radius, 1st iteration has aIteration == 0, thus offset
        float radius = mVmRadiusFactor * mPushConstants.mSceneSphere[3];
        radius /= pow(float(vmIteration + 1), 0.5f * (1 - mVmRadiusAlpha));
        // Purely for numeric stability
        mPushConstants.mMergeRadius = max(radius, 1e-7f);

        const float mergeRadiusSqr = pow2(mPushConstants.mMergeRadius);

        // Factor used to normalise vertex merging contribution.
        // We divide the summed up energy by disk radius and number of light paths
        mPushConstants.mVmNormalization = 1.f / (mergeRadiusSqr * M_PI * mPushConstants.mLightSubPathCount);

        // MIS weight constant [tech. rep. (20)], with n_VC = 1 and n_VM = mLightPathCount
        const float etaVCM = (M_PI * mergeRadiusSqr) * mPushConstants.mLightSubPathCount;
        mPushConstants.mMisVmWeightFactor = defines.find("gUseVM") != defines.end() ? Mis(etaVCM) : 0.f;
        mPushConstants.mMisVcWeightFactor = defines.find("gUseVC") != defines.end() ? Mis(1.f / etaVCM) : 0.f;
	}

	// create mDescriptorSets
	{
		ProfilerScope ps("Assign descriptors", &commandBuffer);

		Descriptors descriptors;

		for (auto& [name, d] : frame->mSceneData->getDescriptors())
			descriptors[{ "gScene." + name.first, name.second }] = d;
		descriptors[{ "gScene.mPerformanceCounters", 0u }] = mPerformanceCounters;

		for (const auto&[name, buffer] : frame->mBuffers)
			descriptors[{ "gRenderParams." + name, 0 }] = buffer;
		for (const auto&[name, image] : frame->mImages)
			descriptors[{ "gRenderParams." + name, 0 }] = image;

		frame->mDescriptorSets = mRenderPipelines[eGenerateLightPaths].get(commandBuffer.mDevice, defines)->getDescriptorSets(descriptors);
	}

	// clear light image
	if (mAlgorithm != VcmAlgorithmType::kPathTrace) {
		frame->mBuffers.at("mLightImage").fill(commandBuffer, 0);
		frame->mBuffers.at("mLightImage").barrier(commandBuffer,
			vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
			vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);
	}

	clearHashGrid("mLightHashGrid");

	if (mAlgorithm == VcmAlgorithmType::kBptLvc) {
		const Buffer::View<byte>& lightPathLengths = frame->mBuffers.at("mLightPathLengths");
		Buffer::View<uint32_t> lightPathLengths0(lightPathLengths.buffer(), lightPathLengths.offset(), 2);

		lightPathLengths0.fill(commandBuffer, 0);
		lightPathLengths0.barrier(commandBuffer,
			vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
			vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);
	}


	///////////////////////////////////////////////////////////
	// Rendering


	// generate light paths
	if (mAlgorithm != VcmAlgorithmType::kPathTrace) {
		ProfilerScope ps("Generate light paths", &commandBuffer);

		const vk::Extent3D lightExtent(extent.width, (mPushConstants.mLightSubPathCount + extent.width-1) / extent.width, 1);
		mRenderPipelines[eGenerateLightPaths].get(commandBuffer.mDevice, defines)->dispatchTiled(commandBuffer, lightExtent, frame->mDescriptorSets, {}, { { "", PushConstantValue(mPushConstants) } });

		commandBuffer->pipelineBarrier(
			vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
			vk::DependencyFlagBits::eByRegion,
			vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead), {}, {});

		if (defines.find("gUseVM") != defines.end()) {
			// Build hash grid over light vertices
			{
				ProfilerScope ps("Compute HashGrid indices", &commandBuffer);
				const vk::Extent3D indicesExtent(extent.width, (mPushConstants.mHashGridCellCount + extent.width-1) / extent.width, 1);
				mRenderPipelines[eHashGridComputeIndices].get(commandBuffer.mDevice, defines)->dispatchTiled(commandBuffer, indicesExtent, frame->mDescriptorSets, {}, { { "", PushConstantValue(mPushConstants) } });

				frame->mBuffers.at("mLightHashGrid.mIndices").barrier(commandBuffer,
					vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
					vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);
			}
			{
				ProfilerScope ps("Swizzle HashGrid", &commandBuffer);
				const vk::Extent3D swizzleExtent(extent.width, (maxLightVertices + extent.width-1) / extent.width, 1);
				mRenderPipelines[eHashGridSwizzle].get(commandBuffer.mDevice, defines)->dispatchTiled(commandBuffer, swizzleExtent, frame->mDescriptorSets, {}, { { "", PushConstantValue(mPushConstants) } });

				frame->mBuffers.at("mLightHashGrid.mData").barrier(commandBuffer,
					vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
					vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);
			}
		}
	}

	// generate camera paths
	{
		ProfilerScope ps("Generate camera paths", &commandBuffer);
		mRenderPipelines[eGenerateCameraPaths].get(commandBuffer.mDevice, defines)->dispatchTiled(commandBuffer, extent, frame->mDescriptorSets, {}, { { "", PushConstantValue(mPushConstants) } });
	}

	if (mUsePerformanceCounters) {
		frame->mBuffers.at("mLightHashGrid.mStats").barrier(commandBuffer,
			vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eHost,
			vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eHostRead);
	}


	///////////////////////////////////////////////////////////
	// Post processing


	// run denoiser

	Image::View renderResult = get<Image::View>(frame->mImages.at("mOutput"));

	if (mDenoise && denoiser && mRandomPerFrame) {
		bool changed = mPrevFrame && (frame->mBuffers.at("mViewTransforms").cast<TransformData>()[0].m != mPrevFrame->mBuffers.at("mViewTransforms").cast<TransformData>()[0].m).any();

		if (scene->lastUpdate() > mLastSceneUpdateTime) {
			changed = true;
			mLastSceneUpdateTime = scene->lastUpdate();
		}

		if (changed && !denoiser->reprojection())
			denoiser->resetAccumulation();

		renderResult = denoiser->denoise(
			commandBuffer,
			renderResult,
			get<Image::View>(frame->mImages.at("mAlbedo")),
			get<Image::View>(frame->mImages.at("mPrevUVs")),
			get<Image::View>(frame->mImages.at("mVisibility")),
			get<Image::View>(frame->mImages.at("mDepth")),
			frame->mBuffers.at("mViews").cast<ViewData>() );
	}

	// run tonemapper

	if (mTonemap && tonemapper)
		tonemapper->render(commandBuffer, renderResult, renderTarget, (mDenoise && denoiser && denoiser->demodulateAlbedo()) ? get<Image::View>(frame->mImages.at("mAlbedo")) : Image::View{});
	else {
		// just copy renderResult to renderTarget
		if (renderResult.image()->format() == renderTarget.image()->format())
			Image::copy(commandBuffer, renderResult, renderTarget);
		else
			Image::blit(commandBuffer, renderResult, renderTarget);
	}


	// copy VisibilityData for selected pixel
	frame->mSelectionDataValid = false;
	if (!ImGui::GetIO().WantCaptureMouse && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGuizmo::IsUsing()) {
		const Image::View v = get<Image::View>(frame->mImages.at("mVisibility"));
		v.barrier(commandBuffer, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferRead);

		const ImVec2 c = ImGui::GetIO().MousePos;
		for (const auto&[view, transform] : frame->mViews)
			if (view.isInside(int2(c.x, c.y))) {
				frame->mSelectionData.copyFromImage(commandBuffer, v.image(), v.subresourceLayer(), vk::Offset3D{int(c.x), int(c.y), 0}, vk::Extent3D{1,1,1});
				frame->mSelectionDataValid = true;
				frame->mSelectionShift = ImGui::GetIO().KeyShift;
			}
	}

	// visualize paths
	if (mVisualizeLightPaths && mAlgorithm != VcmAlgorithmType::kPathTrace)
		rasterLightPaths(commandBuffer, renderTarget, *frame);

	mLastResultImage = renderResult;
	mPrevFrame = frame;
}

void PathTracer::rasterLightPaths(CommandBuffer& commandBuffer, const Image::View& renderTarget, FrameResources& frame) {
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
			{ "mMergeRadius", mPushConstants.mMergeRadius },
		});
		commandBuffer->draw((mPushConstants.mMaxPathLength+1)*6, min(mVisualizeLightPathCount, mPushConstants.mLightSubPathCount), 0, 0);

		commandBuffer.trackResource(lightPathPipeline);
		commandBuffer.trackResource(descriptorSets);

		commandBuffer->endRendering();
	}
}

}