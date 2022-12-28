#include "PathTracer.hpp"
#include "Denoiser.hpp"
#include "Tonemapper.hpp"
#include "Gui.hpp"
#include "Inspector.hpp"

#include <stb_image_write.h>
#include <random>

#include <Core/Instance.hpp>
#include <Core/CommandBuffer.hpp>
#include <Core/Profiler.hpp>

namespace stm2 {

PathTracer::PathTracer(Node& node) : mNode(node) {
	if (shared_ptr<Inspector> inspector = mNode.root()->findDescendant<Inspector>())
		inspector->setInspectCallback<PathTracer>();

	Device& device = *mNode.findAncestor<Device>();
	createPipelines(device);

	mPushConstants.mMinPathLength = 0;
	mPushConstants.mMaxPathLength = 5;
	mPushConstants.mRadiusAlpha = 0.5f;
	mPushConstants.mRadiusFactor = 0.5f;
	mPushConstants.mEnvironmentSampleProbability = 0.5f;

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
	mRenderPipelines[RenderPipelineIndex::eGenerateLightPaths]  = ComputePipelineCache(kernelPath, "GenerateLightPaths" , "sm_6_6", args, md);
	mRenderPipelines[RenderPipelineIndex::eGenerateCameraPaths] = ComputePipelineCache(kernelPath, "GenerateCameraPaths", "sm_6_6", args, md);

	mPerformanceCounters = make_shared<Buffer>(device, "mPerformanceCounters", 4*sizeof(uint32_t), vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	mPrevPerformanceCounters.resize(mPerformanceCounters.size());
	mPerformanceCounterPerSecond.resize(mPerformanceCounters.size());
	ranges::fill(mPerformanceCounters, 0);
	ranges::fill(mPrevPerformanceCounters, 0);
	mPerformanceCounterTimer = 0;
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

	if (ImGui::CollapsingHeader("Configuration")) {
		bool changed = false;
		if (Gui::enumDropdown<VcmAlgorithmType>("Algorithm", mAlgorithm, VcmAlgorithmType::kNumVcmAlgorithmType)) changed = true;
		if (ImGui::Checkbox("Random frame seed", &mRandomPerFrame)) changed = true;
		if (ImGui::Checkbox("Half precision", &mHalfColorPrecision)) changed = true;
		if (ImGui::Checkbox("Performance counters", &mUsePerformanceCounters)) changed = true;
		if (ImGui::DragScalar("LightTraceQuantization", ImGuiDataType_U32, &mLightTraceQuantization)) changed = true;

		ImGui::PushItemWidth(60);
		if (ImGui::DragScalar("Max path length", ImGuiDataType_U32, &mPushConstants.mMaxPathLength)) changed = true;
		if (ImGui::DragScalar("Min path length", ImGuiDataType_U32, &mPushConstants.mMinPathLength)) changed = true;
		if (mPushConstants.mEnvironmentMaterialAddress != -1 && mPushConstants.mLightCount > 0)
			if (ImGui::SliderFloat("Environment sample probability", &mPushConstants.mEnvironmentSampleProbability, 0, 1)) changed = true;
		if (ImGui::SliderFloat("Radius alpha", &mPushConstants.mRadiusAlpha, 0, 1)) changed = true;
		if (ImGui::SliderFloat("Radius factor", &mPushConstants.mRadiusFactor, 0, 1)) changed = true;
		ImGui::PopItemWidth();

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
	}

	/*
	if (mPrevFrame && ImGui::CollapsingHeader("Export")) {
		ImGui::Indent();
		static char path[256]{ 'i', 'm', 'a', 'g', 'e', '.', 'h', 'd', 'r', '\0' };
		ImGui::InputText("##", path, sizeof(path));
		if (ImGui::Button("Save")) {
			Device& d = mPrevFrame->mDevice;
			auto queueFamily = d.findQueueFamily(vk::QueueFlagBits::eTransfer);
			auto cb = d.getCommandBuffer(queueFamily);
			(*cb)->begin({});

			Image::View src = (mDenoise && mNode.findDescendant<Denoiser>()) ? mPrevFrame->mDenoiseResult : mPrevFrame->mRadiance;
			if (src.image()->format() != vk::Format::eR32G32B32A32Sfloat) {
				Image::Metadata md = {};
				md.mExtent = src.extent();
				md.mFormat = vk::Format::eR32G32B32A32Sfloat;
				md.mUsage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
				Image::View tmp = make_shared<Image>(cb->mDevice, "gRadiance", md);
				Image::blit(*cb, src, tmp);
				src = tmp;
			}

			src.barrier(*cb, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferRead);

			Buffer::View<float> pixels = make_shared<Buffer>(d, "image copy tmp", src.extent().width * src.extent().height * sizeof(float) * 4, vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
			(*cb)->copyImageToBuffer(
				**src.image(), vk::ImageLayout::eTransferSrcOptimal,
				**pixels.buffer(), vk::BufferImageCopy(pixels.offset(), 0, 0, src.subresourceLayer(), vk::Offset3D{0,0,0}, src.extent()));

			(*cb)->end();

			d.submit(d->getQueue(queueFamily, 0),cb);
			if (d->waitForFences(**cb->fence(), true, ~0llu) != vk::Result::eSuccess)
				cerr << "Failed to wait for fence" << endl;

			stbi_write_hdr(path, src.extent().width, src.extent().height, 4, pixels.data());
			ImGui::Unindent();
		}
	}
	*/
}

void PathTracer::render(CommandBuffer& commandBuffer, const Image::View& renderTarget) {
	auto scene = mNode.findAncestor<Scene>();
	auto sceneResources = scene->resources();
	if (!sceneResources) {
		renderTarget.clearColor(commandBuffer, vk::ClearColorValue(array<uint32_t, 4>{ 0, 0, 0, 0 }));
		return;
	}

	ProfilerScope ps("PathTracer::render", &commandBuffer);

	// reuse old frame resources
	auto frame = mFrameResourcePool.get();
	if (!frame)
		frame = mFrameResourcePool.emplace(make_shared<FrameResources>(commandBuffer.mDevice));
	commandBuffer.trackResource(frame);

	// scene object picking using frame's mSelectionData
	{
		ProfilerScope ps("Object picking");
		if (frame && frame->mSelectionData && frame->mSelectionDataValid) {
			const uint32_t selectedInstance = frame->mSelectionData.data()->instanceIndex();
			if (shared_ptr<Inspector> inspector = mNode.findAncestor<Inspector>()) {
				if (selectedInstance == INVALID_INSTANCE || selectedInstance >= frame->mSceneData->mInstanceNodes.size())
					inspector->select(nullptr);
				else {
					if (shared_ptr<Node> selected = frame->mSceneData->mInstanceNodes[selectedInstance].lock())
						inspector->select(selected);
				}
			}
		}
	}

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

	frame->mSceneData = sceneResources;


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

	if (mRandomPerFrame) mPushConstants.mRandomSeed = rand();
	mPushConstants.mEnvironmentMaterialAddress = frame->mSceneData->mEnvironmentMaterialAddress;
	mPushConstants.mLightCount = frame->mSceneData->mLightCount;

	mPushConstants.mSceneSphere = float4(0,0,0,100);

	const vk::Extent3D extent = renderTarget.extent();

	// allocate data if needed
	{
		ProfilerScope ps("Allocate data", &commandBuffer);

		const uint32_t pixelCount = extent.width * extent.height;

		auto format = mHalfColorPrecision ? vk::Format::eR16G16B16A16Sfloat : vk::Format::eR32G32B32A32Sfloat;
		auto usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;

		frame->getImage("mOutput",     extent, format, usage);
		frame->getImage("mAlbedo",     extent, format, usage);
		frame->getImage("mPrevUVs",    extent, vk::Format::eR32G32Sfloat, usage);
		frame->getImage("mVisibility", extent, vk::Format::eR32G32Uint, usage);
		frame->getImage("mDepth" ,     extent, vk::Format::eR32G32B32A32Sfloat, usage);

		frame->getBuffer<uint4>("mLightImage", pixelCount*sizeof(uint4), vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eStorageBuffer);
		frame->getBuffer<VcmVertex>("mLightVertices", pixelCount*mPushConstants.mMaxPathLength);
		frame->getBuffer<uint32_t>("mLightPathLengths", pixelCount);

		if (!frame->mSelectionData)
			frame->mSelectionData = make_shared<Buffer>(commandBuffer.mDevice, "mSelectionData", sizeof(VisibilityData), vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
	}


	Defines defines {
		{ "gHasMedia", to_string(has_volumes) },
		{ "gLightTraceQuantization", to_string(mLightTraceQuantization) },
		{ "gPerformanceCounters", to_string(mUsePerformanceCounters) },
	};

	{
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
		case VcmAlgorithmType::kVcm:
			defines["gUseVC"] = "true";
			defines["gUseVM"] = "true";
			break;
		default:
			printf("Unknown algorithm type\n");
			break;
		}
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

	frame->mBuffers.at("mLightImage").fill(commandBuffer, 0);
	frame->mBuffers.at("mLightImage").barrier(commandBuffer,
		vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
		vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);

	// generate light paths
	if (mAlgorithm != VcmAlgorithmType::kPathTrace) {
		ProfilerScope ps("Generate light paths", &commandBuffer);
		mRenderPipelines[eGenerateLightPaths].get(commandBuffer.mDevice, defines)->dispatchTiled(commandBuffer, extent, frame->mDescriptorSets, {}, { { "", PushConstantValue(mPushConstants) } });
		frame->mBuffers.at("mLightImage").barrier(commandBuffer,
			vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
			vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);
	}


	// generate camera paths
	{
		ProfilerScope ps("Generate camera paths", &commandBuffer);
		mRenderPipelines[eGenerateCameraPaths].get(commandBuffer.mDevice, defines)->dispatchTiled(commandBuffer, extent, frame->mDescriptorSets, {}, { { "", PushConstantValue(mPushConstants) } });
	}


	Image::View renderResult = get<Image::View>(frame->mImages.at("mOutput"));

	shared_ptr<Denoiser> denoiser = mNode.findDescendant<Denoiser>();
	if (mDenoise && denoiser) {
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

	shared_ptr<Tonemapper> tonemapper = mNode.findDescendant<Tonemapper>();
	if (mTonemap && tonemapper)
		tonemapper->render(commandBuffer, renderResult, renderTarget, (mDenoise && denoiser && denoiser->demodulateAlbedo()) ? get<Image::View>(frame->mImages.at("mAlbedo")) : Image::View{});
	else {
		// copy renderResult to rendertarget directly
		if (renderResult.image()->format() == renderTarget.image()->format())
			Image::copy(commandBuffer, renderResult, renderTarget);
		else
			Image::blit(commandBuffer, renderResult, renderTarget);
	}

	mLastResultImage = renderResult;

	// copy VisibilityData for selected pixel
	frame->mSelectionDataValid = false;
	if (!ImGui::GetIO().WantCaptureMouse && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
		const Image::View v = get<Image::View>(frame->mImages.at("mVisibility"));
		v.barrier(commandBuffer, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferRead);

		const ImVec2 c = ImGui::GetIO().MousePos;
		for (const auto&[view, transform] : frame->mViews)
			if (view.isInside(int2(c.x, c.y))) {
				frame->mSelectionData.copyFromImage(commandBuffer, v.image(), v.subresourceLayer(), vk::Offset3D{int(c.x), int(c.y), 0}, vk::Extent3D{1,1,1});
				frame->mSelectionDataValid = true;
			}
	}

	mPrevFrame = frame;
}

}