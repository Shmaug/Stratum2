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
		mDefines["gAlphaTest"]      = false;
		mDefines["gShadingNormals"] = true;
		mDefines["gNormalMaps"]     = true;
		mDefines["gLambertian"]     = false;
		mDefines["gNoGI"]           = false;
		mDefines["gDebugFastBRDF"]  = false;
		mDefines["gDebugPixel"]     = false;

		mFixSeed = false;
		mPushConstants["mMaxDepth"] = 8u;
		mPushConstants["mMaxDiffuseBounces"] = 3u;
		mPushConstants["mMaxNullCollisions"] = 1000u;
		mPushConstants["mEnvironmentSampleProbability"] = 0.f;

		mDefines["gReSTIR_DI"] = false;
		mPushConstants["mDICandidateSamples"] = 32u;
		mDefines["gReSTIR_DI_Reuse"] = false;
		mDefines["gReSTIR_DI_Reuse_Visibility"] = false;
		mPushConstants["mDIMaxM"] = 3.f;
		mPushConstants["mDIReuseRadius"] = 32.f;

		mDefines["gReSTIR_GI"] = false;
		mPushConstants["mGICandidateSamples"] = 4u;
		mDefines["gReSTIR_GI_Reuse"] = false;
		mPushConstants["mGIMaxM"]         = 4.f;
		mPushConstants["mGIReuseRadius"]  = 16.f;
		mPushConstants["mGIReuseSamples"] = 3u;
		mDefines["gReSTIR_GI_Shift_Test"] = false;

		mDenoise = true;
		mTonemap = true;

		mShowRcVertices = false;
		mRasterPushConstants["mDepth"] = -1;
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

	{
		auto defineCheckbox = [&](const char* label, const char* name) {
			if (ImGui::Checkbox(label, &mDefines.at(name)))
				changed = true;
		};
		auto pushConstantField = [&]<typename T>(const char* label, const char* name, const T mn = 0, const T mx = 0, const float spd = 1) {
			if (Gui::scalarField<T>(label, &mPushConstants[name].get<T>(), mn, mx, spd))
				changed = true;
		};

		defineCheckbox("Alpha testing",    "gAlphaTest");
		defineCheckbox("Shading normals",  "gShadingNormals");
		defineCheckbox("Normal maps",      "gNormalMaps");
		defineCheckbox("Force lambertian", "gLambertian");
		defineCheckbox("Disable GI",       "gNoGI");
		defineCheckbox("Debug fast BRDF",  "gDebugFastBRDF");
		defineCheckbox("DebugPixel",       "gDebugPixel");
		ImGui::Separator();

		ImGui::Checkbox("Fix random seed", &mFixSeed);
		pushConstantField.operator()<uint32_t>("Max depth",                      "mMaxDepth", 1, 0, 0.1f);
		pushConstantField.operator()<uint32_t>("Max diffuse bounces",            "mMaxDiffuseBounces", 1, 0, 0.1f);
		pushConstantField.operator()<uint32_t>("Max null collisions",            "mMaxNullCollisions");
		pushConstantField.operator()<float>   ("Environment sample probability", "mEnvironmentSampleProbability", 0, 1, 0);
		ImGui::Separator();

		if (mDefines.at("gNoGI")) {
			defineCheckbox("ReSTIR DI", "gReSTIR_DI");
			if (mDefines.at("gReSTIR_DI")) {
				ImGui::Indent();
				pushConstantField.operator()<uint32_t>("DI candidate samples", "mDICandidateSamples", 0, 128, 0.25f);
				defineCheckbox("ReSTIR DI reuse", "gReSTIR_DI_Reuse");
				defineCheckbox("ReSTIR DI reuse visibility", "gReSTIR_DI_Reuse_Visibility");
				pushConstantField.operator()<float>("DI max M", "mDIMaxM", 0, 10, .1f);
				pushConstantField.operator()<float>("DI reuse radius", "mDIReuseRadius", 0, 1000);
				ImGui::Unindent();
			}
		} else {
			defineCheckbox("ReSTIR GI", "gReSTIR_GI");
			if (mDefines.at("gReSTIR_GI")) {
				ImGui::Indent();
				pushConstantField.operator()<uint32_t>("GI candidate samples", "mGICandidateSamples", 0, 128, 0.25f);
				defineCheckbox("ReSTIR GI reuse", "gReSTIR_GI_Reuse");
				pushConstantField.operator()<float>   ("GI max M", "mGIMaxM", 0, 10, .1f);
				pushConstantField.operator()<float>   ("GI reuse radius", "mGIReuseRadius", 0, 1000);
				pushConstantField.operator()<uint32_t>("GI reuse samples", "mGIReuseSamples", 0, 32);
				if (mDefines.at("gReSTIR_GI_Reuse"))
					defineCheckbox("Debug shift map", "gReSTIR_GI_Shift_Test");
				ImGui::Unindent();
			}
		}
		ImGui::Separator();

		ImGui::Checkbox("Denoise", &mDenoise);
		ImGui::Checkbox("Tonemap", &mTonemap);
		ImGui::Separator();

		ImGui::Checkbox("Visualize reconnection vertices", &mShowRcVertices);
		if (mShowRcVertices && !mDefines.at("gNoGI") && mDefines.at("gReSTIR_GI")) {
			ImGui::Indent();
			Gui::scalarField("Depth", &mRasterPushConstants.at("mDepth").get<int32_t>());
			Gui::scalarField<float>("Line radius", &mRasterPushConstants.at("mLineRadius").get<float>(), 0, 1, .01f);
			Gui::scalarField<float>("Line length", &mRasterPushConstants.at("mLineLength").get<float>(), 0, 1, .01f);
			ImGui::Unindent();
		}
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

void ReSTIRPT::render(CommandBuffer& commandBuffer, const Image::View& renderTarget) {
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

	descriptors[{ "gPathTracer.mFramebuffer.mOutput", 0 }]     = ImageDescriptor{ outputImage    , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };
	descriptors[{ "gPathTracer.mFramebuffer.mAlbedo", 0 }]     = ImageDescriptor{ albedoImage    , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };
	descriptors[{ "gPathTracer.mFramebuffer.mPrevUVs", 0 }]    = ImageDescriptor{ prevUVsImage   , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };
	descriptors[{ "gPathTracer.mFramebuffer.mVisibility", 0 }] = ImageDescriptor{ visibilityImage, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };
	descriptors[{ "gPathTracer.mFramebuffer.mDepth", 0 }]      = ImageDescriptor{ depthImage     , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };

	mPushConstants["mReservoirHistoryValid"] = 1u;
	for (uint32_t i = 0; i < 3; i++) {
		const string id = "mReservoirDataDI["+to_string(i)+"]";
		const Image::View prev = mResourcePool.getLastImage(id);
		const Image::View reservoirData = mResourcePool.getImage(commandBuffer.mDevice, id, Image::Metadata{
			.mFormat = vk::Format::eR32G32B32A32Sfloat,
			.mExtent = extent,
			.mUsage = vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eTransferSrc|vk::ImageUsageFlagBits::eTransferDst,
		});
		descriptors[{ "gPathTracer.mFramebuffer.mReservoirDataDI", i }] = ImageDescriptor{ reservoirData, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };
		descriptors[{ "gPathTracer.mFramebuffer.mPrevReservoirDataDI", i }] = ImageDescriptor{ prev ? prev : reservoirData, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead, {} };
		if (!prev)
			mPushConstants["mReservoirHistoryValid"] = 0u;

		if (mDefines.at("gReSTIR_DI_Reuse"))
			reservoirData.clearColor(commandBuffer, vk::ClearColorValue{array<float,4>{0,0,0,0}});
	}
	for (uint32_t i = 0; i < 6; i++) {
		const string id = "mReservoirDataGI["+to_string(i)+"]";
		const Image::View prev = mResourcePool.getLastImage(id);
		const Image::View reservoirData = mResourcePool.getImage(commandBuffer.mDevice, id, Image::Metadata{
			.mFormat = vk::Format::eR32G32B32A32Sfloat,
			.mExtent = extent,
			.mUsage = vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eTransferSrc|vk::ImageUsageFlagBits::eTransferDst,
		});
		descriptors[{ "gPathTracer.mFramebuffer.mReservoirDataGI", i }] = ImageDescriptor{ reservoirData, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };
		descriptors[{ "gPathTracer.mFramebuffer.mPrevReservoirDataGI", i }] = ImageDescriptor{ prev ? prev : reservoirData, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead, {} };
		if (!prev)
			mPushConstants["mReservoirHistoryValid"] = 0u;

		if (mDefines.at("gReSTIR_GI"))
			reservoirData.clearColor(commandBuffer, vk::ClearColorValue{array<float,4>{0,0,0,0}});
	}
	if (ImGui::IsKeyDown(ImGuiKey_F5))
		mPushConstants["mReservoirHistoryValid"] = 0u;

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
		descriptors[{ "gPathTracer.mScene.mRayCount", 0u }] = make_shared<Buffer>(commandBuffer.mDevice, "mRayCount", 2*sizeof(uint32_t), vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

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

		mPushConstants["mOutputExtent"] = uint2(renderTarget.extent().width, renderTarget.extent().height);
		mPushConstants["mViewCount"] = (uint32_t)viewsBufferData.size();
		mPushConstants["mEnvironmentMaterialAddress"] = sceneData.mEnvironmentMaterialAddress;
		mPushConstants["mLightCount"] = sceneData.mLightCount;
		float4 sphere;
		sphere.head<3>() = (sceneData.mAabbMax + sceneData.mAabbMin) / 2;
		sphere[3] = length<float,3>(sceneData.mAabbMax - sphere.head<3>());
		mPushConstants["mSceneSphere"] = sphere;

		if (changed && !mFixSeed && mDenoise && denoiser && !denoiser->reprojection())
			denoiser->resetAccumulation();

		if (mFixSeed)
			mPushConstants["mRandomSeed"] = 0u;
		else
			mPushConstants["mRandomSeed"] = (mDenoise && denoiser) ? denoiser->accumulatedFrames() : (uint32_t)rand();

		if (mDefines.at("gDebugPixel")) {
			const ImVec2 c = ImGui::GetIO().MousePos;
			mPushConstants["mDebugPixelIndex"] = (int)(c.y * extent.width) + (int)c.x;
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

	shared_ptr<ComputePipeline> renderPipeline;
	{
		renderPipeline = loadPipeline("Render", defines);
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
	}

	// scene object picking
	if (!ImGui::GetIO().WantCaptureMouse && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGuizmo::IsUsing()) {
		// copy VisibilityData for selected pixel ()
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

	// rasterize reconnection vertices
	if (mShowRcVertices && !mDefines.at("gNoGI") && mDefines.at("gReSTIR_GI")) {
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