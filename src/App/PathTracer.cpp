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
	mPushConstants.mHashGridCellCount = 65536;
	mPushConstants.mHashGridCellPixelRadius = 0;
	mPushConstants.mHashGridMinCellSize = .1f;
	mPushConstants.mHashGridJitterRadius = 0.05f;
	mPushConstants.mLightImageQuantization = 16384;
	mPushConstants.mDIReservoirSampleCount = 32;
	mPushConstants.mDIReservoirMaxM = 3;
	mPushConstants.mLVCReservoirSampleCount = 8;
	mPushConstants.mLVCReservoirMaxM = 3;
	mPushConstants.mLVCHashGridSampleCount = 8;

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
		if (ImGui::Checkbox("Performance counters", &mUsePerformanceCounters)) changed = true;
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
				if (ImGui::Checkbox("Cell resampling", &mLVCHashGridSampling)) changed = true;
				if (mLVCHashGridSampling) {
					if (ImGui::DragScalar("Cell RIS samples", ImGuiDataType_U32, &mPushConstants.mLVCHashGridSampleCount, 1, &one)) changed = true;
				}

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
		if (mAlgorithm == VcmAlgorithmType::kPpm || mAlgorithm == VcmAlgorithmType::kBpm || mAlgorithm == VcmAlgorithmType::kVcm ||
			(mAlgorithm != VcmAlgorithmType::kLightTrace && (mDIReservoirFlags & VcmReservoirFlags::eReuse)) ||
			(mAlgorithm == VcmAlgorithmType::kBpt && mUseLightVertexCache && ((mLVCReservoirFlags & VcmReservoirFlags::eReuse) || mLVCHashGridSampling))) {
			if (ImGui::DragScalar("Hash grid cells", ImGuiDataType_U32, &mPushConstants.mHashGridCellCount)) changed = true;
			if (ImGui::DragFloat("Hash grid min cell size", &mPushConstants.mHashGridMinCellSize, .01f, 1e-4f, 1000)) changed = true;
			if (ImGui::DragFloat("Hash grid cell pixel radius", &mPushConstants.mHashGridCellPixelRadius, .5f, 0, 1000)) changed = true;
		}
		if ((mAlgorithm != VcmAlgorithmType::kLightTrace && (mDIReservoirFlags & VcmReservoirFlags::eReuse)) ||
			(mAlgorithm == VcmAlgorithmType::kBpt && mUseLightVertexCache && (mLVCReservoirFlags & VcmReservoirFlags::eReuse))) {
			if (ImGui::DragFloat("Hash grid jitter radius", &mPushConstants.mHashGridJitterRadius, .005f)) changed = true;
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

	if (mUsePerformanceCounters) {
		const auto [rps, ext] = formatNumber(mPerformanceCounterPerSecond[0]);
		ImGui::Text("%.2f%s Rays/second (%u%% visibility)", rps, ext, mPerformanceCounterPerSecond[0] == 0 ? 0 : (uint32_t)(100*mPerformanceCounterPerSecond[1] / mPerformanceCounterPerSecond[0]));
		ImGui::Text("%u failed inserts, %u buckets used", mHashGridStats[0], mHashGridStats[1]);
	}
}

void PathTracer::render(CommandBuffer& commandBuffer, const Image::View& renderTarget) {
#if 0
	if (mPauseRendering) return;

	ProfilerScope ps("PathTracer::render", &commandBuffer);

	const shared_ptr<Scene> scene = mNode.findAncestor<Scene>();
	const Scene::FrameData& sceneData = scene->frameData();
	if (sceneData.mLightCount == 0 && sceneData.mEnvironmentMaterialAddress == -1) {
		renderTarget.clearColor(commandBuffer, vk::ClearColorValue(array<uint32_t, 4>{ 0, 0, 0, 0 }));
		return;
	}

	const shared_ptr<Denoiser>   denoiser   = mNode.findDescendant<Denoiser>();
	const shared_ptr<Tonemapper> tonemapper = mNode.findDescendant<Tonemapper>();

	// read gpu->cpu data from frame
	{
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
			if (frame->mSelectionData && frame->mSelectionDataValid) {
				const uint32_t selectedInstance = frame->mSelectionData.data()->instanceIndex();
				if (shared_ptr<Inspector> inspector = mNode.root()->findDescendant<Inspector>()) {
					if (selectedInstance == INVALID_INSTANCE || selectedInstance >= sceneData.mInstanceNodes.size())
						inspector->select(nullptr);
					else {
						if (shared_ptr<Node> selected = sceneData.mInstanceNodes[selectedInstance].lock()) {
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
	}

	bool has_volumes = false;

	Descriptors descriptors;

	// copy camera/view data to GPU, compute gViewMediumInstances
	{
		ProfilerScope ps("Upload views", &commandBuffer);

		vector<ViewData>      views;
		vector<TransformData> viewTransforms;
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

		auto prevViews                 = frame->getBuffer<ViewData>     ("mPrevViews"                , views.size(), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		auto prevInverseViewTransforms = frame->getBuffer<TransformData>("mPrevInverseViewTransforms", views.size(), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		if (mPrevFrame) {
			for (uint32_t i = 0; i < mPrevFrame->mViews.size(); i++) {
				prevViews[i] = mPrevFrame->mViews[i].first;
				prevInverseViewTransforms[i] = mPrevFrame->mViews[i].second.inverse();
			}
		}


		// find if views are inside a volume
		auto viewMediumIndicesBuffer = mResourcePool.getBuffer<uint32_t>("mViewMediumInstances", views.size(), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		ranges::fill(viewMediumIndicesBuffer, INVALID_INSTANCE);
		descriptors[{"gRenderParams.mViewMediumInstances",0}] = viewMediumIndicesBuffer;

		mNode.forEachDescendant<Medium>([&](Node& node, const shared_ptr<Medium>& vol) {
			has_volumes = true;
			for (uint32_t i = 0; i < views.size(); i++) {
				const float3 localViewPos = nodeToWorld(node).inverse().transformPoint( viewTransforms[i].transformPoint(float3::Zero()) );
				if (vol->mDensityGrid->grid<float>()->worldBBox().isInside(nanovdb::Vec3R(localViewPos[0], localViewPos[1], localViewPos[2])))
					viewMediumIndicesBuffer[i] = sceneData.mInstanceTransformMap.at(vol.get()).second;
			}
		});
	}

	shared_ptr<DescriptorSets> descriptorSets;

	auto makeHashGridBuffers = [&]<typename T>(const string& name, const uint32_t cellCount, const uint32_t maxElements) {
		descriptors[{"gRenderParams." + name + ".mChecksums",0}]     = mResourcePool.getBuffer<uint>  (commandBuffer.mDevice, name + ".mChecksums"    , cellCount    , vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eStorageBuffer);
		descriptors[{"gRenderParams." + name + ".mCounters",0}]      = mResourcePool.getBuffer<uint>  (commandBuffer.mDevice, name + ".mCounters"     , cellCount    , vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eStorageBuffer);
		descriptors[{"gRenderParams." + name + ".mAppendIndices",0}] = mResourcePool.getBuffer<uint2> (commandBuffer.mDevice, name + ".mAppendIndices", 1+maxElements, vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eStorageBuffer);
		descriptors[{"gRenderParams." + name + ".mAppendData",0}]    = mResourcePool.getBuffer<T>     (commandBuffer.mDevice, name + ".mAppendData"   , maxElements);
		descriptors[{"gRenderParams." + name + ".mIndices",0}]       = mResourcePool.getBuffer<uint>  (commandBuffer.mDevice, name + ".mIndices"      , cellCount);
		descriptors[{"gRenderParams." + name + ".mData",0}]          = mResourcePool.getBuffer<T>     (commandBuffer.mDevice, name + ".mData"         , maxElements);
		descriptors[{"gRenderParams." + name + ".mActiveCells",0}]   = mResourcePool.getBuffer<uint>  (commandBuffer.mDevice, name + ".mActiveCells"  , cellCount);
		descriptors[{"gRenderParams." + name + ".mCellCenters",0}]   = mResourcePool.getBuffer<float4>(commandBuffer.mDevice, name + ".mCellCenters"  , cellCount);
		descriptors[{"gRenderParams." + name + ".mCellNormals",0}]   = mResourcePool.getBuffer<int4>  (commandBuffer.mDevice, name + ".mCellNormals"  , cellCount, vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eStorageBuffer);
		descriptors[{"gRenderParams." + name + ".mStats",0}]         = mResourcePool.getBuffer<uint>  (commandBuffer.mDevice, name + ".mStats", 2, vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
	};
	auto clearHashGrid = [&](const string& name) {
		const Buffer::View<byte>& appendIndices = frame->mBuffers.at(name + ".mAppendIndices");
		Buffer::View<byte> appendIndices0(appendIndices.buffer(), appendIndices.offset(), sizeof(uint2));

		frame->mBuffers.at(name + ".mChecksums").fill(commandBuffer, 0);
		frame->mBuffers.at(name + ".mCounters").fill(commandBuffer, 0);
		frame->mBuffers.at(name + ".mCellNormals").fill(commandBuffer, 0);
		appendIndices0.fill(commandBuffer, 0);

		Buffer::barriers(commandBuffer,
			{
				frame->mBuffers.at(name + ".mChecksums"),
				frame->mBuffers.at(name + ".mCounters"),
				appendIndices0
			},
			vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
			vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);

		frame->mBuffers.at(name + ".mStats").cast<uint2>()[0] = uint2::Zero();
	};
	auto buildHashGrid = [&](const string& name, const uint32_t size) {
		const uint32_t w = renderTarget.extent().width;
		const Defines defines { { "gHashGrid", "gRenderParams." + name } };
		{
			ProfilerScope ps("Compute HashGrid indices (" + name + ")", &commandBuffer);
			const vk::Extent3D indicesExtent(w, (mPushConstants.mHashGridCellCount + w-1) / w, 1);
			mRenderPipelines[eHashGridComputeIndices].get(commandBuffer.mDevice, defines)->dispatchTiled(commandBuffer, indicesExtent, descriptorSets);

			frame->mBuffers.at(name+".mIndices").barrier(commandBuffer,
				vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
				vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);
		}
		{
			ProfilerScope ps("Swizzle HashGrid (" + name + ")", &commandBuffer);
			const vk::Extent3D swizzleExtent(w, (size + w-1) / w, 1);
			mRenderPipelines[eHashGridSwizzle].get(commandBuffer.mDevice, defines)->dispatchTiled(commandBuffer, swizzleExtent, descriptorSets);

			frame->mBuffers.at(name+".mData").barrier(commandBuffer,
				vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
				vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);
		}
	};


	// specify defines
	bool useVC = false;
	bool useVM = false;
	Defines defines {
		{ "gHasMedia",            to_string(has_volumes) },
		{ "gPerformanceCounters", to_string(mUsePerformanceCounters) },
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
			if (mLVCHashGridSampling)
				defines["gLVCHashGridSampling"] = "true";
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


	const vk::Extent3D extent = renderTarget.extent();
	mPushConstants.mOutputExtent = uint2(extent.width, extent.height);
	mPushConstants.mScreenPixelCount = extent.width * extent.height;
	mPushConstants.mLightSubPathCount = max<uint32_t>(1, mPushConstants.mScreenPixelCount * mLightPathPercent);
	const uint32_t maxLightVertices  = mPushConstants.mLightSubPathCount*mPushConstants.mMaxPathLength;
	const uint32_t maxCameraVertices = mPushConstants.mLightSubPathCount*mPushConstants.mMaxPathLength;

	// update values, allocate data
	{
		mPushConstants.reservoirHistoryValid((bool)mPrevFrame && !ImGui::IsKeyPressed(ImGuiKey_F5));
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

			descriptors["gRenderParams.mVcmConstants"] = mResourcePool.uploadData<VcmConstants>(commandBuffer, "gRenderParams.mVcmConstants", &constants, vk::BufferUsageFlagBits::eUniformBuffer);
		}

		// allocate data
		{
			ProfilerScope ps("Allocate data");

			auto usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;

			frame->getImage("mOutput",     extent, vk::Format::eR32G32B32A32Sfloat, usage);
			frame->getImage("mAlbedo",     extent, vk::Format::eR16G16B16A16Sfloat, usage);
			frame->getImage("mPrevUVs",    extent, vk::Format::eR32G32Sfloat, usage);
			frame->getImage("mVisibility", extent, vk::Format::eR32G32Uint, usage);
			frame->getImage("mDepth" ,     extent, vk::Format::eR32G32B32A32Sfloat, usage);

			frame->getBuffer<uint4>("mLightImage", mPushConstants.mScreenPixelCount*sizeof(uint4), vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eStorageBuffer);
			frame->getBuffer<PackedVcmVertex>("mLightVertices", maxLightVertices);
			frame->getBuffer<uint32_t>("mLightPathLengths", mPushConstants.mLightSubPathCount, vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eStorageBuffer);

			makeHashGridBuffers.operator()<uint>("mLightHashGrid", mPushConstants.mHashGridCellCount, maxLightVertices);
			makeHashGridBuffers.operator()<DirectIlluminationReservoir>("mDirectIlluminationReservoirs", mPushConstants.mHashGridCellCount, maxCameraVertices);
			makeHashGridBuffers.operator()<LVCReservoir>("mLVCReservoirs", mPushConstants.mHashGridCellCount, maxCameraVertices);

			if (!frame->mSelectionData)
				frame->mSelectionData = make_shared<Buffer>(commandBuffer.mDevice, "mSelectionData", sizeof(VisibilityData), vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
		}

		for (const string d : {
			"mChecksums",
			"mCounters",
			"mAppendIndices",
			"mAppendData",
			"mIndices",
			"mData",
			"mActiveCells",
			"mStats" }) {
			frame->mBuffers["mPrevDirectIlluminationReservoirs."+d] = (mPrevFrame ? mPrevFrame : frame)->mBuffers.at("mDirectIlluminationReservoirs."+d);
			frame->mBuffers["mPrevLVCReservoirs."+d]                = (mPrevFrame ? mPrevFrame : frame)->mBuffers.at("mLVCReservoirs."+d);
		}
	}

	// create mDescriptorSets
	{
		ProfilerScope ps("Assign descriptors", &commandBuffer);

		for (auto& [name, d] : sceneData.mDescriptors)
			descriptors[{ "gScene." + name.first, name.second }] = d;
		descriptors[{ "gScene.mPerformanceCounters", 0u }] = mPerformanceCounters;

		descriptorSets = mRenderPipelines[eGenerateLightPaths].get(commandBuffer.mDevice, defines)->getDescriptorSets(descriptors);
	}

	// rendering
	{
		// clearing things
		{
			if (useVM || (mLVCHashGridSampling && mAlgorithm == VcmAlgorithmType::kBpt && mUseLightVertexCache))
				clearHashGrid("mLightHashGrid");

			if ((mDIReservoirFlags & VcmReservoirFlags::eReuse) && mAlgorithm != VcmAlgorithmType::kPpm && mAlgorithm != VcmAlgorithmType::kLightTrace)
				clearHashGrid("mDirectIlluminationReservoirs");
			if ((mLVCReservoirFlags & VcmReservoirFlags::eReuse) && mAlgorithm == VcmAlgorithmType::kBpt && mUseLightVertexCache)
				clearHashGrid("mLVCReservoirs");

			// clear light image
			if (mAlgorithm != VcmAlgorithmType::kPathTrace) {
				frame->mBuffers.at("mLightImage").fill(commandBuffer, 0);
				frame->mBuffers.at("mLightImage").barrier(commandBuffer,
					vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
					vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
			}

			// clear light vertex counter
			if (mAlgorithm == VcmAlgorithmType::kBpt && mUseLightVertexCache) {
				const Buffer::View<byte>& lightPathLengths = frame->mBuffers.at("mLightPathLengths");
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
			mRenderPipelines[eGenerateLightPaths].get(commandBuffer.mDevice, defines)->dispatchTiled(commandBuffer, lightExtent, descriptorSets, {}, { { "", PushConstantValue(mPushConstants) } });

			commandBuffer->pipelineBarrier(
				vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
				vk::DependencyFlagBits::eByRegion,
				vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead), {}, {});

			// Build hash grid over light vertices
			if (useVM || (mLVCHashGridSampling && mAlgorithm == VcmAlgorithmType::kBpt && mUseLightVertexCache))
				buildHashGrid("mLightHashGrid", maxLightVertices);
		}

		// generate camera paths
		{
			ProfilerScope ps("Generate camera paths", &commandBuffer);
			mRenderPipelines[eGenerateCameraPaths].get(commandBuffer.mDevice, defines)->dispatchTiled(commandBuffer, extent, descriptorSets, {}, { { "", PushConstantValue(mPushConstants) } });
		}

		// build hashgrids
		if ((mDIReservoirFlags & VcmReservoirFlags::eReuse) && mAlgorithm != VcmAlgorithmType::kPpm && mAlgorithm != VcmAlgorithmType::kLightTrace)
			buildHashGrid("mDirectIlluminationReservoirs", maxLightVertices);
		if ((mLVCReservoirFlags & VcmReservoirFlags::eReuse) && mAlgorithm == VcmAlgorithmType::kBpt && mUseLightVertexCache)
			buildHashGrid("mLVCReservoirs", maxLightVertices);
	}

	Image::View renderResult = get<Image::View>(frame->mImages.at("mOutput"));

	// post processing
	{
		// run denoiser
		if (mDenoise && denoiser && mRandomPerFrame) {
			bool changed = mPrevFrame && (frame->mBuffers.at("mViewTransforms").cast<TransformData>()[0].m != mPrevFrame->mBuffers.at("mViewTransforms").cast<TransformData>()[0].m).any();

			if (scene->lastUpdate() > mLastSceneVersion) {
				changed = true;
				mLastSceneVersion = scene->lastUpdate();
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
	}


	// copy VisibilityData for selected pixel for scene object picking
	frame->mSelectionDataValid = false;
	if (!ImGui::GetIO().WantCaptureMouse && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGuizmo::IsUsing()) {
		const Image::View v = get<Image::View>(frame->mImages.at("mVisibility"));
		v.barrier(commandBuffer, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferRead);

		const ImVec2 c = ImGui::GetIO().MousePos;
		for (const auto&[view, transform] : views)
			if (view.isInside(int2(c.x, c.y))) {
				frame->mSelectionData.copyFromImage(commandBuffer, v.image(), v.subresourceLayer(), vk::Offset3D{int(c.x), int(c.y), 0}, vk::Extent3D{1,1,1});
				frame->mSelectionDataValid = true;
				frame->mSelectionShift = ImGui::GetIO().KeyShift;
			}
	}


	// visualize paths
	if (mVisualizeLightPaths && mAlgorithm != VcmAlgorithmType::kPathTrace)
		rasterLightPaths(commandBuffer, renderTarget);

	mLastResultImage = renderResult;
#endif
}

void PathTracer::rasterLightPaths(CommandBuffer& commandBuffer, const Image::View& renderTarget) {
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