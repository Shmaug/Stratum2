#include "ReSTIRPT.hpp"
#include "Inspector.hpp"
#include "Scene.hpp"
#include "Denoiser.hpp"
#include "Tonemapper.hpp"

#include <Core/Instance.hpp>
#include <Core/Profiler.hpp>

#include "Gui.hpp"
#include <ImGuizmo.h>

namespace stm2 {

ReSTIRPT::ReSTIRPT(Node& node) : mNode(node) {
	if (shared_ptr<Inspector> inspector = mNode.root()->findDescendant<Inspector>())
		inspector->setInspectCallback<ReSTIRPT>();

	{
		mDefines["gCountRays"]      = false;
		mDefines["gAlphaTest"]      = false;
		mDefines["gShadingNormals"] = true;
		mDefines["gNormalMaps"]     = true;
		mDefines["gNee"]            = false;
		mDefines["gLambertian"]     = false;
		mDefines["gDebugFastBRDF"]  = false;
		mDefines["gDebugPixel"]     = false;
		mDefines["gDebugNee"]       = false;

		mPushConstants["mMaxDepth"] = 8u;
		mPushConstants["mMaxDiffuseBounces"] = 3u;
		mPushConstants["mMaxNullCollisions"] = 1000u;
		mPushConstants["mEnvironmentSampleProbability"] = 0.f;

		mDefines["gReSTIR_GI"] = false;
		mPushConstants["mGICandidateSamples"] = 4u;
		mDefines["gReSTIR_GI_Reuse"] = false;
		mPushConstants["mGIMaxM"]         = 4.f;
		mPushConstants["mGIReuseRadius"]  = 16.f;
		mPushConstants["mGIReuseSamples"] = 3u;
		mDefines["gReconnection"] = false;
		mDefines["gTemporalReuse"] = false;

		mRasterPushConstants["mLineRadius"] = .0025f;
		mRasterPushConstants["mLineLength"] = .02f;
	}

	Device& device = *mNode.findAncestor<Device>();

	mStaticSampler = make_shared<vk::raii::Sampler>(*device, vk::SamplerCreateInfo({},
		vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear,
		vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat,
		0, true, 8, false, vk::CompareOp::eAlways, 0, VK_LOD_CLAMP_NONE));
	device.setDebugName(**mStaticSampler, "TestRenderer/Sampler");

	createPipelines(device);
}

void ReSTIRPT::createPipelines(Device& device) {
	ComputePipeline::Metadata md;
	md.mImmutableSamplers["gPathTracer.mScene.mStaticSampler"]  = { mStaticSampler };
	md.mImmutableSamplers["gPathTracer.mScene.mStaticSampler1"] = { mStaticSampler };
	md.mBindingFlags["gPathTracer.mScene.mVertexBuffers"] = vk::DescriptorBindingFlagBits::ePartiallyBound;
	md.mBindingFlags["gPathTracer.mScene.mImages"]  = vk::DescriptorBindingFlagBits::ePartiallyBound;
	md.mBindingFlags["gPathTracer.mScene.mImage2s"] = vk::DescriptorBindingFlagBits::ePartiallyBound;
	md.mBindingFlags["gPathTracer.mScene.mImage1s"] = vk::DescriptorBindingFlagBits::ePartiallyBound;
	md.mBindingFlags["gPathTracer.mScene.mVolumes"] = vk::DescriptorBindingFlagBits::ePartiallyBound;

	const vector<string>& args = {
		"-matrix-layout-row-major",
		"-O3",
		"-Wno-30081",
		"-capability", "spirv_1_5",
		"-capability", "GL_EXT_ray_tracing"
	};

	const filesystem::path shaderPath = *device.mInstance.findArgument("shaderKernelPath");
	mPipelines.clear();
	mPipelines.emplace("Render", ComputePipelineCache(shaderPath / "restirpt" / "restirpt.slang", "Render", "sm_6_6", args, md));
}

void ReSTIRPT::drawGui() {
	bool changed = false;

	ImGui::PushID(this);

	if (mDefines.at("gCountRays") && mRayCount.mBuffer) {
		const auto[rps,rpsUnit] = formatNumber(mRayCount.mCurrentValue[0]/mCounterDt);
		const auto[shift,shiftUnit] = formatNumber(mDebugCounters.mCurrentValue[0]/mCounterDt);

		ImGui::Text("%.1f%s rays/s (%.1f%% occlusion)",
			rps, rpsUnit,
			100*(mRayCount.mCurrentValue[1]/(float)mRayCount.mCurrentValue[0]));
		ImGui::Text("%.1f%s shifts/s (%.1f%% success, %.1f%% reconnections, %.1f%% stored)",
			shift, shiftUnit,
			100*(mDebugCounters.mCurrentValue[1]/(float)mDebugCounters.mCurrentValue[0]),
			100*(mDebugCounters.mCurrentValue[2]/(float)mDebugCounters.mCurrentValue[0]),
			100*(mDebugCounters.mCurrentValue[3]/(float)mDebugCounters.mCurrentValue[4]));
	}

	if (ImGui::Button("Clear resources")) {
		Device& device = *mNode.findAncestor<Device>();
		device->waitIdle();
		mResourcePool.clear();
		mPrevViewTransforms.clear();
		changed = true;
	}
	ImGui::SameLine();
	if (ImGui::Button("Reload shaders")) {
		Device& device = *mNode.findAncestor<Device>();
		device->waitIdle();
		createPipelines(device);
		mRasterPipeline.clear();
		changed = true;
	}
	ImGui::SliderFloat("Render scale", &mRenderScale, 0.01f, 2.f);

	ImGui::Checkbox("Pause", &mPauseRender);
	if (mPauseRender) {
		ImGui::SameLine();
		if (ImGui::Button("Render once")) {
			mRenderOnce = true;
		}
	}

	const shared_ptr<Denoiser> denoiser = mDenoise ? mNode.findDescendant<Denoiser>() : nullptr;
	if (denoiser) {
		ImGui::Text("%u frames accumulated", denoiser->accumulatedFrames());
		ImGui::SameLine();
		if (ImGui::Button("Reset"))
		denoiser->resetAccumulation();
	}

	ImGui::Separator();

	{
		auto defineCheckbox = [&](const char* label, const char* name) {
			if (ImGui::Checkbox(label, &mDefines.at(name)))
				changed = true;
		};
		auto pushConstantField = [&]<typename T>(const char* label, const char* name, const T mn = 0, const T mx = 0, const float spd = 1) {
			if (Gui::scalarField<T>(label, &mPushConstants[name].get<T>(), mn, mx, spd))
				changed = true;
		};

		defineCheckbox("Alpha testing",      "gAlphaTest");
		defineCheckbox("Shading normals",    "gShadingNormals");
		defineCheckbox("Normal maps",        "gNormalMaps");
		defineCheckbox("Force lambertian",   "gLambertian");
		defineCheckbox("NEE",                "gNee");
		defineCheckbox("Debug NEE",          "gDebugNee");
		defineCheckbox("Debug fast BRDF",    "gDebugFastBRDF");
		defineCheckbox("Debug pixel (ctrl)", "gDebugPixel");
		defineCheckbox("Count rays",         "gCountRays");
		ImGui::Separator();

		ImGui::Checkbox("Fix random seed", &mFixSeed);
		pushConstantField.operator()<uint32_t>("Max depth",                      "mMaxDepth", 1, 0, 0.1f);
		pushConstantField.operator()<uint32_t>("Max diffuse bounces",            "mMaxDiffuseBounces", 1, 0, 0.1f);
		pushConstantField.operator()<uint32_t>("Max null collisions",            "mMaxNullCollisions");
		pushConstantField.operator()<float>   ("Environment sample probability", "mEnvironmentSampleProbability", 0, 1, 0);
		ImGui::Separator();

		{
			defineCheckbox("ReSTIR GI", "gReSTIR_GI");
			if (mDefines.at("gReSTIR_GI")) {
				ImGui::Indent();
				pushConstantField.operator()<uint32_t>("GI candidate samples", "mGICandidateSamples", 0, 128, 0.25f);
				defineCheckbox("ReSTIR GI reuse", "gReSTIR_GI_Reuse");
				if (mDefines.at("gReSTIR_GI_Reuse")) {
					defineCheckbox("Enable reconnection", "gReconnection");
					defineCheckbox("Temporal reuse", "gTemporalReuse");
					if (mDefines.at("gTemporalReuse"))
						pushConstantField.operator()<float>   ("GI max M", "mGIMaxM", 0, 10, .1f);
					pushConstantField.operator()<float>   ("GI reuse radius", "mGIReuseRadius", 0, 1000);
					pushConstantField.operator()<uint32_t>("GI reuse samples", "mGIReuseSamples", 0, 32);
				}
				ImGui::Unindent();
			}
		}
		ImGui::Separator();

		ImGui::Checkbox("Denoise", &mDenoise);
		ImGui::Checkbox("Tonemap", &mTonemap);
		ImGui::Separator();

		ImGui::Checkbox("Visualize reconnection vertices", &mShowRcVertices);
		if (mShowRcVertices && mDefines.at("gReSTIR_GI")) {
			ImGui::Indent();
			Gui::scalarField<float>("Line radius", &mRasterPushConstants.at("mLineRadius").get<float>(), 0, 1, .01f);
			Gui::scalarField<float>("Line length", &mRasterPushConstants.at("mLineLength").get<float>(), 0, 1, .01f);
			ImGui::Unindent();
		}
	}

	if (changed && denoiser)
		denoiser->resetAccumulation();

	if (ImGui::CollapsingHeader("Resources")) {
		ImGui::Indent();
		mResourcePool.drawGui();
		ImGui::Unindent();
	}

	ImGui::PopID();
}

void ReSTIRPT::render(CommandBuffer& commandBuffer, const Image::View& renderTarget) {
	if (mPauseRender) {
		if (mRenderOnce)
			mRenderOnce = false;
		else
			return;
	}

	ProfilerScope ps("TestRenderer::render", &commandBuffer);

	mResourcePool.clean();

	if (!mRayCount.mBuffer) {
		mRayCount = {
			make_shared<Buffer>(commandBuffer.mDevice, "mRayCount", 2*sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent),
			vector<uint32_t>(2),
			vector<uint32_t>(2) };
		ranges::fill(mRayCount.mBuffer, 0);
		ranges::fill(mRayCount.mLastValue, 0);
	}
	if (!mDebugCounters.mBuffer) {
		mDebugCounters = {
			make_shared<Buffer>(commandBuffer.mDevice, "mDebugCounters", 8*sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent),
			vector<uint32_t>(8),
			vector<uint32_t>(8) };
		ranges::fill(mDebugCounters.mBuffer, 0);
		ranges::fill(mDebugCounters.mLastValue, 0);
	}
	if (mDefines.at("gCountRays")) {
		const auto t1 = chrono::high_resolution_clock::now();
		const auto dt = t1 - mCounterTimer;
		if (dt > 1s) {
			mCounterTimer = t1;
			mCounterDt = chrono::duration_cast<chrono::duration<float>>(dt).count();

			auto updateCounter = [](auto& c) {
				for (uint32_t i = 0; i < c.mBuffer.size(); i++) {
					if (c.mBuffer[i] >= c.mLastValue[i])
						c.mCurrentValue[i] = c.mBuffer[i] - c.mLastValue[i];
					c.mLastValue[i] = c.mBuffer[i];
				}
			};
			updateCounter(mRayCount);
			updateCounter(mDebugCounters);
		}
	}

	vk::Extent3D extent = renderTarget.extent();
	extent.width  = max<uint32_t>(1, extent.width * mRenderScale);
	extent.height = max<uint32_t>(1, extent.height * mRenderScale);

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

	descriptors[{ "gPathTracer.mFramebuffer.mOutput", 0 }]     = ImageDescriptor{ outputImage    , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };
	descriptors[{ "gPathTracer.mFramebuffer.mAlbedo", 0 }]     = ImageDescriptor{ albedoImage    , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };
	descriptors[{ "gPathTracer.mFramebuffer.mPrevUVs", 0 }]    = ImageDescriptor{ prevUVsImage   , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };
	descriptors[{ "gPathTracer.mFramebuffer.mVisibility", 0 }] = ImageDescriptor{ visibilityImage, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };
	descriptors[{ "gPathTracer.mFramebuffer.mDepth", 0 }]      = ImageDescriptor{ depthImage     , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };
	descriptors[{ "gPathTracer.mFramebuffer.mDebugCounters", 0u }] = mDebugCounters.mBuffer;
	descriptors[{ "gPathTracer.mScene.mRayCount", 0u }]            = mRayCount.mBuffer;

	mPushConstants["mReservoirHistoryValid"] = ImGui::IsKeyDown(ImGuiKey_F5) || (mDenoise && denoiser && denoiser->accumulatedFrames() == 0) ? 0u : 1u;
	for (uint32_t i = 0; i < mPrevPathReservoirData.size(); i++) {
		const string id = "mReservoirDataGI["+to_string(i)+"]";
		const Image::View& prev = mPrevPathReservoirData[i];
		const Image::View reservoirData = mResourcePool.getImage(commandBuffer.mDevice, id, Image::Metadata{
			.mFormat = vk::Format::eR32G32B32A32Sfloat,
			.mExtent = extent,
			.mUsage = vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eTransferSrc|vk::ImageUsageFlagBits::eTransferDst,
		});
		descriptors[{ "gPathTracer.mFramebuffer.mPathReservoirData", i }] = ImageDescriptor{ reservoirData, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };
		descriptors[{ "gPathTracer.mFramebuffer.mPrevPathReservoirData", i }] = ImageDescriptor{ prev ? prev : reservoirData, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead, {} };
		if (!prev)
			mPushConstants["mReservoirHistoryValid"] = 0u;

		if (mDefines.at("gReSTIR_GI"))
			reservoirData.clearColor(commandBuffer, vk::ClearColorValue{array<float,4>{0,0,0,0}});

		mPrevPathReservoirData[i] = reservoirData;
	}

	bool changed = false;
	bool hasMedia = false;
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
			descriptors[{ "gPathTracer.mScene." + name.first, name.second }] = d;

		// track resources which are not held by the descriptorset
		commandBuffer.trackResource(sceneData.mAccelerationStructureBuffer.buffer());

		// find views

		vector<pair<ViewData, TransformData>> views;
		mNode.root()->forEachDescendant<Camera>([&](Node& node, const shared_ptr<Camera>& camera) {
			ViewData v = camera->view();
			v.mImageMin = (v.mImageMin.cast<float>() * mRenderScale).cast<int32_t>();
			v.mImageMax = (v.mImageMax.cast<float>() * mRenderScale).cast<int32_t>();
			views.emplace_back(pair{ v, nodeToWorld(node) });
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

		descriptors[{ "gPathTracer.mFramebuffer.mViews", 0 }]                     = viewsBuffer;
		descriptors[{ "gPathTracer.mFramebuffer.mViewTransforms", 0 }]            = mResourcePool.uploadData<TransformData>(commandBuffer, "mViewTransforms", viewTransformsBufferData);
		descriptors[{ "gPathTracer.mFramebuffer.mViewInverseTransforms", 0 }]     = mResourcePool.uploadData<TransformData>(commandBuffer, "mViewInverseTransforms", inverseViewTransformsData);
		descriptors[{ "gPathTracer.mFramebuffer.mPrevViewInverseTransforms", 0 }] = mResourcePool.uploadData<TransformData>(commandBuffer, "mPrevViewInverseTransforms", prevInverseViewTransformsData);


		// find if views are inside a volume
		vector<uint32_t> viewMediumIndices(views.size());
		ranges::fill(viewMediumIndices, INVALID_INSTANCE);
		for (const auto& info : sceneData.mInstanceVolumeInfo) {
			hasMedia = true;
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

		descriptors[{ "gPathTracer.mFramebuffer.mViewMediumIndices", 0 }] = mResourcePool.uploadData<uint>(commandBuffer, "mViewMediumIndices", viewMediumIndices);
	}

	// push constants
	{
		const Scene::FrameData& sceneData = scene->frameData();

		mPushConstants["mOutputExtent"] = uint2(extent.width, extent.height);
		mPushConstants["mViewCount"] = (uint32_t)viewsBufferData.size();
		mPushConstants["mEnvironmentMaterialAddress"] = sceneData.mEnvironmentMaterialAddress;
		mPushConstants["mLightCount"] = sceneData.mLightCount;
		float4 sphere;
		sphere.head<3>() = (sceneData.mAabbMax + sceneData.mAabbMin) / 2;
		sphere[3] = length<float,3>(sceneData.mAabbMax - sphere.head<3>());
		mPushConstants["mSceneSphere"] = sphere;

		if (changed && !mFixSeed && mDenoise && denoiser && !denoiser->reprojection()) {
			denoiser->resetAccumulation();
			mPushConstants["mReservoirHistoryValid"] = 0u;
		}

		if (mFixSeed)
			mPushConstants["mRandomSeed"] = 0u;
		else
			mPushConstants["mRandomSeed"] = (mDenoise && denoiser) ? denoiser->accumulatedFrames() : (uint32_t)rand();

		if (mDefines.at("gDebugPixel")) {
			const float2 c = float2(ImGui::GetIO().MousePos.x, ImGui::GetIO().MousePos.y) * mRenderScale;
			mPushConstants["mDebugPixelIndex"] = ImGui::GetIO().KeyCtrl ? int(c[1]) * extent.width + int(c[0]) : -1;
		}
	}

	// setup shader defines

	Defines defines;
	for (const auto&[define,enabled] : mDefines)
		if (enabled)
			defines[define] = to_string(enabled);

	if (hasMedia)
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

	shared_ptr<ComputePipeline> renderPipeline = loadPipeline("Render", defines);
	if (loading) {
		// compiling shaders...
		renderTarget.clearColor(commandBuffer, vk::ClearColorValue(array<float, 4>{ 0.5f, 0.5f, 0.5f, 0 }));

		const ImVec2 size = ImGui::GetMainViewport()->WorkSize;
		ImGui::SetNextWindowPos(ImVec2(size.x/2, size.y/2));
		if (ImGui::Begin("Compiling shaders", nullptr, ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoNav|ImGuiWindowFlags_NoDecoration|ImGuiWindowFlags_NoInputs)) {
			ImGui::Text("Compiling shaders...");
			Gui::progressSpinner("Compiling shaders");
		}
		ImGui::End();
		return;
	}

	// create descriptor sets
	const shared_ptr<DescriptorSets> descriptorSets = mResourcePool.getDescriptorSets(*renderPipeline, "DescriptorSets", descriptors);

	// render
	{
		ProfilerScope ps("Trace paths", &commandBuffer);
		renderPipeline->dispatchTiled(commandBuffer, extent, descriptorSets, {}, mPushConstants);
	}

	// post processing
	{
		Image::View processedOutput = outputImage;

		// run denoiser
		if (mDenoise && denoiser && !mFixSeed) {
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
			if (outputImage.image()->format() == renderTarget.image()->format() && outputImage.extent() == renderTarget.extent())
				Image::copy(commandBuffer, outputImage, renderTarget);
			else
				Image::blit(commandBuffer, outputImage, renderTarget, vk::Filter::eNearest);
		} else {
			// copy processedOutput to renderTarget
			if (processedOutput.image()->format() == renderTarget.image()->format() && processedOutput.extent() == renderTarget.extent())
				Image::copy(commandBuffer, processedOutput, renderTarget);
			else
				Image::blit(commandBuffer, processedOutput, renderTarget, vk::Filter::eNearest);
		}
	}

	// scene object picking
	if (!ImGui::GetIO().WantCaptureMouse && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGuizmo::IsUsing()) {
		// copy VisibilityData for selected pixel ()
		const int2 c = (float2(ImGui::GetIO().MousePos.x, ImGui::GetIO().MousePos.y) * mRenderScale).cast<int32_t>();
		for (const ViewData& view : viewsBufferData)
			if (view.isInside(c)) {
				Buffer::View<VisibilityData> selectionBuffer = make_shared<Buffer>(commandBuffer.mDevice, "SelectionData", sizeof(uint2), vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
				visibilityImage.barrier(commandBuffer, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferRead);
				selectionBuffer.copyFromImage(commandBuffer, visibilityImage.image(), visibilityImage.subresourceLayer(), vk::Offset3D{c[0], c[1], 0}, vk::Extent3D{1,1,1});
				mSelectionData.push_back(make_pair(selectionBuffer, ImGui::GetIO().KeyShift));
				break;
			}
	}

	// rasterize reconnection vertices
	if (mShowRcVertices && mDefines.at("gReSTIR_GI")) {
		if (!mRasterPipeline || renderTarget.image()->format() != mRasterPipeline.pipelineMetadata().mDynamicRenderingState->mColorFormats[0]) {
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
			gmd.mDynamicRenderingState->mColorFormats = { renderTarget.image()->format() };
			gmd.mDynamicRenderingState->mDepthFormat = vk::Format::eD32Sfloat;

			gmd.mViewports = { vk::Viewport(0, 0, extent.width, extent.height, 0, 1) };
			gmd.mScissors  = { vk::Rect2D({0,0}, { renderTarget.extent().width, renderTarget.extent().height }) };

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

			const filesystem::path shaderPath = *commandBuffer.mDevice.mInstance.findArgument("shaderKernelPath");
			const filesystem::path rasterShaderPath = shaderPath / "restirpt" / "rcv_vis.slang";
			const vector<string>& rasterArgs = {
				"-matrix-layout-row-major",
				"-O3",
				"-Wno-30081",
				"-capability", "spirv_1_5",
			};

			auto staticSampler = make_shared<vk::raii::Sampler>(*commandBuffer.mDevice, vk::SamplerCreateInfo({},
				vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear,
				vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat,
				0, true, 8, false, vk::CompareOp::eAlways, 0, VK_LOD_CLAMP_NONE));
			commandBuffer.mDevice.setDebugName(**staticSampler, "TestRenderer/Sampler (Raster)");
			gmd.mImmutableSamplers["gScene.mStaticSampler"]  = { staticSampler };
			gmd.mImmutableSamplers["gScene.mStaticSampler1"] = { staticSampler };
			gmd.mBindingFlags["gScene.mVertexBuffers"] = vk::DescriptorBindingFlagBits::ePartiallyBound;
			gmd.mBindingFlags["gScene.mImages"]  = vk::DescriptorBindingFlagBits::ePartiallyBound;
			gmd.mBindingFlags["gScene.mImage2s"] = vk::DescriptorBindingFlagBits::ePartiallyBound;
			gmd.mBindingFlags["gScene.mImage1s"] = vk::DescriptorBindingFlagBits::ePartiallyBound;
			gmd.mBindingFlags["gScene.mVolumes"] = vk::DescriptorBindingFlagBits::ePartiallyBound;

			mRasterPipeline = GraphicsPipelineCache({
				{ vk::ShaderStageFlagBits::eVertex  , GraphicsPipelineCache::ShaderSourceInfo(rasterShaderPath, "ShowReconnectionVertexVS", "sm_6_6") },
				{ vk::ShaderStageFlagBits::eFragment, GraphicsPipelineCache::ShaderSourceInfo(rasterShaderPath, "ShowReconnectionVertexFS", "sm_6_6") }
			}, rasterArgs, gmd);
		}

		Descriptors rasterDescriptors;
		for (auto& [name, d] : scene->frameData().mDescriptors)
			rasterDescriptors[{ "gScene." + name.first, name.second }] = d;
		rasterDescriptors.erase({"gScene.mAccelerationStructure", 0});

		for (auto [name, d] : descriptors) {
			if (name.first.starts_with("gPathTracer.mFramebuffer."))
				rasterDescriptors[{string("gFramebuffer.") + name.first.substr(25), name.second}] = d;
		}

		const Image::View rasterDepthBuffer = mResourcePool.getImage(commandBuffer.mDevice, "mRasterDepthBuffer", Image::Metadata{
				.mFormat = vk::Format::eD32Sfloat,
				.mExtent = extent,
				.mUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment|vk::ImageUsageFlagBits::eTransferDst });
		auto rasterPipeline = mRasterPipeline.get(commandBuffer.mDevice);
		auto descriptorSets = rasterPipeline->getDescriptorSets(rasterDescriptors);

		renderTarget.barrier     (commandBuffer, vk::ImageLayout::eColorAttachmentOptimal       , vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::AccessFlagBits::eColorAttachmentWrite);
		rasterDepthBuffer.barrier(commandBuffer, vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::PipelineStageFlagBits::eEarlyFragmentTests   , vk::AccessFlagBits::eDepthStencilAttachmentRead|vk::AccessFlagBits::eDepthStencilAttachmentWrite);

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

		commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, ***rasterPipeline);
		descriptorSets->bind(commandBuffer);
		rasterPipeline->pushConstants(commandBuffer, mRasterPushConstants);
		mRasterPushConstants["mOutputExtent"] = uint2(renderTarget.extent().width, renderTarget.extent().height);
		mRasterPushConstants["mPixelIndex"] = mDefines.at("gDebugPixel") ? mPushConstants["mDebugPixelIndex"] : -1u;
		commandBuffer->draw((renderTarget.extent().width * renderTarget.extent().height)*6, 1, 0, 0);

		commandBuffer.trackResource(rasterPipeline);
		commandBuffer.trackResource(descriptorSets);

		commandBuffer->endRendering();
	}
}

}