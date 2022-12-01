#include "TinyPT.hpp"
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

TinyPT::TinyPT(Node& node) : mNode(node) {
	if (shared_ptr<Inspector> inspector = mNode.root()->findDescendant<Inspector>())
		inspector->setTypeCallback<TinyPT>();

	Device& device = *mNode.findAncestor<Device>();
	createPipelines(device);

	// initialize constants
	if (auto arg = device.mInstance.findArgument("minBounces"); arg)        mPushConstants.mMinBounces        = atoi(arg->c_str());
	if (auto arg = device.mInstance.findArgument("maxBounces"); arg)        mPushConstants.mMaxBounces        = atoi(arg->c_str());
	if (auto arg = device.mInstance.findArgument("maxDiffuseBounces"); arg) mPushConstants.mMaxDiffuseBounces = atoi(arg->c_str());
}

void TinyPT::createPipelines(Device& device) {
	const auto samplerRepeat = make_shared<vk::raii::Sampler>(*device, vk::SamplerCreateInfo({},
		vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear,
		vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat,
		0, true, 8, false, vk::CompareOp::eAlways, 0, VK_LOD_CLAMP_NONE));

	ComputePipeline::Metadata md;
	md.mImmutableSamplers["gScene.mStaticSampler"]  = { samplerRepeat };
	md.mImmutableSamplers["gScene.mStaticSampler1"] = { samplerRepeat };
	md.mBindingFlags["gScene.gVolumes"] = vk::DescriptorBindingFlagBits::ePartiallyBound;
	md.mBindingFlags["gScene.gImages"]  = vk::DescriptorBindingFlagBits::ePartiallyBound;
	md.mBindingFlags["gScene.gImage1s"] = vk::DescriptorBindingFlagBits::ePartiallyBound;

	const filesystem::path shaderPath = *device.mInstance.findArgument("shaderKernelPath");
	const vector<string>& args = { "-matrix-layout-row-major", "-capability", "spirv_1_5", "-Wno-30081", "-capability", "GL_EXT_ray_tracing" };
	const filesystem::path kernelPath = shaderPath / "tinypt.slang";
	mRenderPipelines[RenderPipelineIndex::eTraceViewPaths] = ComputePipelineCache(kernelPath, "sampleViewPaths", "sm_6_6", args, md);

	mRayCount = make_shared<Buffer>(device, "gCounters", 4*sizeof(uint32_t), vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	mPrevRayCount.resize(mRayCount.size());
	mRaysPerSecond.resize(mRayCount.size());
	memset(mRayCount.data(), 0, mRayCount.sizeBytes());
	memset(mPrevRayCount.data(), 0, mPrevRayCount.size()*sizeof(uint32_t));
	mRayCountTimer = 0;

	mPushConstants.mMinBounces = 1;
	mPushConstants.mMaxBounces = 3;
	mPushConstants.mMaxDiffuseBounces = 2;
}

void TinyPT::drawGui() {
	ImGui::PushID(this);
	if (ImGui::Button("Reload shaders")) {
		Device& device = *mNode.findAncestor<Device>();
		device->waitIdle();
		createPipelines(device);
	}

	if (ImGui::Button("Clear resources")) {
		mFrameResourcePool.clear();
		mPrevFrame.reset();
	}
	ImGui::PopID();

	ImGui::Checkbox("Performance counters", &mPerformanceCounters);
	if (mPerformanceCounters) {
		const auto [rps, ext] = formatNumber(mRaysPerSecond[0]);
		ImGui::Text("%.2f%s Rays/second (%u%% shadow rays)", rps, ext, (uint32_t)(100 - (100*(uint64_t)mRaysPerSecond[1]) / mRaysPerSecond[0]));
	}

	if (ImGui::CollapsingHeader("Configuration")) {
		ImGui::Checkbox("Random frame seed", &mRandomPerFrame);
		ImGui::Checkbox("Half precision", &mHalfColorPrecision);
		ImGui::Checkbox("Show albedo", &mShowAlbedo);

		if (auto denoiser = mNode.getComponent<Denoiser>(); denoiser)
			if (ImGui::Checkbox("Enable denoiser", &mDenoise))
				denoiser->resetAccumulation();

		if (auto tonemapper = mNode.getComponent<Tonemapper>(); tonemapper)
			ImGui::Checkbox("Enable tonemapper", &mTonemap);
	}

	if (ImGui::CollapsingHeader("Path tracing")) {
		ImGui::PushItemWidth(60);
		ImGui::DragScalar("Max bounces", ImGuiDataType_U32, &mPushConstants.mMaxBounces);
		ImGui::DragScalar("Min bounces", ImGuiDataType_U32, &mPushConstants.mMinBounces);
		ImGui::DragScalar("Max diffuse bounces", ImGuiDataType_U32, &mPushConstants.mMaxDiffuseBounces);
		ImGui::PopItemWidth();
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

void TinyPT::render(CommandBuffer& commandBuffer, const Image::View& renderTarget) {
	auto sceneResources = mNode.getComponent<Scene>()->resources();
	if (!sceneResources) {
		renderTarget.clearColor(commandBuffer, vk::ClearColorValue(array<uint32_t, 4>{ 0, 0, 0, 0 }));
		return;
	}

	ProfilerScope ps("TinyPT::render", &commandBuffer);

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
			if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
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
	}

	// count rays per second
	{
		frame->mTime = chrono::high_resolution_clock::now();
		if (mPrevFrame) {
			const float deltaTime = chrono::duration_cast<chrono::duration<float>>(frame->mTime - mPrevFrame->mTime).count();
			mRayCountTimer += deltaTime;
			if (mRayCountTimer > 1) {
				for (uint32_t i = 0; i < mRaysPerSecond.size(); i++)
					mRaysPerSecond[i] = (mRayCount[i] - mPrevRayCount[i]) / mRayCountTimer;
				ranges::copy(mRayCount, mPrevRayCount.begin());
				mRayCountTimer = 0;
			}
		}
	}

	frame->mSceneData = sceneResources;


	bool has_volumes = false;

	// copy camera/view data to GPU, compute gViewMediumInstances
	vector<pair<ViewData,TransformData>> views;
	{
		ProfilerScope ps("Upload views", &commandBuffer);

		mNode.root()->forEachDescendant<Camera>([&](Node& node, const shared_ptr<Camera>& camera) {
			views.emplace_back(pair{ camera->view(), nodeToWorld(node) });
		});
		mPushConstants.mViewCount = (uint32_t)views.size();

		if (views.empty()) {
			renderTarget.clearColor(commandBuffer, vk::ClearColorValue(array<uint32_t, 4>{ 0, 0, 0, 0 }));
			cout << "Warning: No views" << endl;
			return;
		}

		// upload viewdata
		auto viewsBuffer                 = frame->getBuffer<ViewData>     ("mViews"                , views.size(), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		auto viewTransformsBuffer        = frame->getBuffer<TransformData>("mViewTransforms"       , views.size(), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		auto viewInverseTransformsBuffer = frame->getBuffer<TransformData>("mViewInverseTransforms", views.size(), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		for (uint32_t i = 0; i < views.size(); i++) {
			viewsBuffer[i] = views[i].first;
			viewTransformsBuffer[i] = views[i].second;
			viewInverseTransformsBuffer[i] = views[i].second.inverse();
		}

		// find if views are inside a volume
		auto viewMediumIndicesBuffer = frame->getBuffer<uint32_t>("mViewMediumInstances", views.size(), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		ranges::fill(viewMediumIndicesBuffer, INVALID_INSTANCE);
		mNode.forEachDescendant<Medium>([&](Node& node, const shared_ptr<Medium>& vol) {
			has_volumes = true;
			for (uint32_t i = 0; i < views.size(); i++) {
				const float3 localViewPos = nodeToWorld(node).inverse().transformPoint( views[i].second.transformPoint(float3::Zero()) );
				if (vol->mDensityGrid->grid<float>()->worldBBox().isInside(nanovdb::Vec3R(localViewPos[0], localViewPos[1], localViewPos[2])))
					viewMediumIndicesBuffer[i] = frame->mSceneData->mInstanceTransformMap.at(vol.get()).second;
			}
		});
	}

	if (mRandomPerFrame) mPushConstants.mRandomSeed = rand();
	mPushConstants.mEnvironmentMaterialAddress = frame->mSceneData->mEnvironmentMaterialAddress;
	mPushConstants.mLightCount = (uint32_t)frame->mSceneData->mLightInstanceMap.size();

	Defines defines;

	const vk::Extent3D extent = renderTarget.extent();

	// allocate data if needed
	{
		ProfilerScope ps("Allocate data", &commandBuffer);

		const uint32_t pixelCount = extent.width * extent.height;

		auto format = mHalfColorPrecision ? vk::Format::eR16G16B16A16Sfloat : vk::Format::eR32G32B32A32Sfloat;
		auto usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;

		frame->getImage("mRadiance", extent, format, usage);
		frame->getImage("mAlbedo",   extent, format, usage);
		frame->getImage("mPrevUVs",  extent, vk::Format::eR32G32Sfloat, usage);

		frame->getBuffer<VisibilityData>("mVisibility",        pixelCount, vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferSrc);
		frame->getBuffer<DepthData>     ("mDepth",             pixelCount, vk::BufferUsageFlagBits::eStorageBuffer);
		frame->getBuffer<float4>        ("mLightTraceSamples", pixelCount, vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst);

		if (!frame->mSelectionData)
			frame->mSelectionData = make_shared<Buffer>(commandBuffer.mDevice, "mSelectionData", sizeof(VisibilityData), vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
	}

	// create descriptorsets
	{
		ProfilerScope ps("Assign descriptors", &commandBuffer);

		Descriptors descriptors;

		for (auto& [name, d] : frame->mSceneData->getDescriptors())
			descriptors[{ "gScene." + name.first, name.second }] = d;
		descriptors[{ "gScene.mPerformanceCounters", 0u }] = mRayCount;

		for (const auto&[name, buffer] : frame->mBuffers)
			descriptors[{ "gRenderParams." + name, 0 }] = buffer;
		for (const auto&[name, image] : frame->mImages)
			descriptors[{ "gRenderParams." + name, 0 }] = image;

		frame->mDescriptorSets = mRenderPipelines[eTraceViewPaths].get(commandBuffer.mDevice, defines)->getDescriptorSets(descriptors);
	}

	auto zeroBuffers = [&](const vector<Buffer::View<byte>>& buffers, const vk::PipelineStageFlags dstStage = vk::PipelineStageFlagBits::eComputeShader, const vk::AccessFlags dstAccess = vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite) {
		for (auto& b : buffers)
			b.fill(commandBuffer, 0);
		Buffer::barriers(commandBuffer, buffers, vk::PipelineStageFlagBits::eTransfer, dstStage, vk::AccessFlagBits::eTransferWrite, dstAccess);
	};

	const PushConstants pushConstants = { { "", PushConstantValue(mPushConstants) } };

	// trace view paths
	{
		ProfilerScope ps("Sample visibility", &commandBuffer);
		mRenderPipelines[eTraceViewPaths].get(commandBuffer.mDevice, defines)->dispatchTiled(commandBuffer, extent, frame->mDescriptorSets, {}, pushConstants);
	}


	Image::View renderResult = get<Image::View>(frame->mImages.at(mShowAlbedo ? "mAlbedo" : "mOutput"));

	shared_ptr<Denoiser> denoiser = mNode.findDescendant<Denoiser>();
	if (mDenoise && denoiser) {
		const bool changed = mPrevFrame && (frame->mBuffers.at("mViewTransforms").cast<TransformData>()[0].m != mPrevFrame->mBuffers.at("mViewTransforms").cast<TransformData>()[0].m).any();
		if (changed && !denoiser->reprojection())
			denoiser->resetAccumulation();

		denoiser->denoise(
			commandBuffer,
			renderResult,
			get<Image::View>(frame->mImages.at("mAlbedo")),
			get<Image::View>(frame->mImages.at("mPrevUVs")),
			frame->mBuffers.at("mViews").cast<ViewData>(),
			frame->mBuffers.at("mVisibility").cast<VisibilityData>(),
			frame->mBuffers.at("mDepth").cast<DepthData>() );
	}

	shared_ptr<Tonemapper> tonemapper = mNode.findDescendant<Tonemapper>();
	if (mTonemap && tonemapper)
		tonemapper->render(commandBuffer, renderResult, (mDenoise && denoiser && denoiser->demodulateAlbedo()) ? get<Image::View>(frame->mImages.at("mAlbedo")) : Image::View{});

	// copy renderResult to rendertarget directly
	if (renderResult.image()->format() == renderTarget.image()->format())
		Image::copy(commandBuffer, renderResult, renderTarget);
	else
		Image::blit(commandBuffer, renderResult, renderTarget);

	// copy VisibilityData for selected pixel
	if (!ImGui::GetIO().WantCaptureMouse && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
		const Buffer::View<VisibilityData> v = frame->mBuffers.at("mVisibility").cast<VisibilityData>();
		v.barrier(commandBuffer, vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead);

		const ImVec2 c = ImGui::GetCursorPos();
		const int2 cp = int2((int)c.x, (int)c.y);
		frame->mSelectionDataValid = false;
		for (const auto&[view, transform] : views)
			if (view.isInside(cp)) {
				Buffer::copy(commandBuffer, Buffer::View<VisibilityData>(v, cp.y() * view.extent().x() + cp.x(), 1), frame->mSelectionData);
				frame->mSelectionDataValid = true;
			}
	}

	mPrevFrame = frame;
}

}