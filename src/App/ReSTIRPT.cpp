#include "ReSTIRPT.hpp"
#include "Inspector.hpp"
#include "Scene.hpp"
#include "Denoiser.hpp"
#include "Tonemapper.hpp"

#include <Core/Instance.hpp>
#include <Core/Profiler.hpp>

#include <imgui/imgui.h>
#include <ImGuizmo.h>

namespace stm2 {

ReSTIRPT::ReSTIRPT(Node& node) : mNode(node) {
	if (shared_ptr<Inspector> inspector = mNode.root()->findDescendant<Inspector>())
		inspector->setInspectCallback<ReSTIRPT>();

	mProperties = {
		initialize_property<bool>    ("Alpha testing",                  true, [this](){ return &mDefines["gAlphaTest"]; }),
		initialize_property<bool>    ("Shading normals",                true, [this](){ return &mDefines["gShadingNormals"]; }),
		initialize_property<bool>    ("Normal maps",                    true, [this](){ return &mDefines["gNormalMaps"]; }),
		initialize_property<bool>    ("Lambertian",                    false, [this](){ return &mDefines["gLambertian"]; }),
		initialize_property<bool>    ("No GI",                         false, [this](){ return &mDefines["gNoGI"]; }),
		initialize_property<bool>    ("Nee",                           false, [this](){ return &mDefines["gNee"]; }),
		initialize_property<bool>    ("Single sample MIS",             false, [this](){ return &mDefines["gSingleSampleMIS"]; }),
		initialize_property<bool>    ("Debug fast BRDF",               false, [this](){ return &mDefines["gDebugFastBRDF"]; }),
		initialize_property<bool>    ("DebugPixel",                    false, [this](){ return &mDefines["gDebugPixel"]; }),
		initialize_property<bool>    ("Random per frame",              false, [this](){ return &mRandomPerFrame; }),

		initialize_property<uint32_t>("Max depth",                         8, [this](){ return &mPushConstants["mMaxDepth"].get<uint32_t>(); }, 1, 0, 0.1f),
		initialize_property<uint32_t>("Max diffuse bounces",               3, [this](){ return &mPushConstants["mMaxDiffuseBounces"].get<uint32_t>(); }, 1, 0, 0.1f),
		initialize_property<uint32_t>("Max null collisions",            1000, [this](){ return &mPushConstants["mMaxNullCollisions"].get<uint32_t>(); }),
		initialize_property<float>   ("Environment sample probability", 0.9f, [this](){ return &mPushConstants["mEnvironmentSampleProbability"].get<float>(); }, 0, 1, 0),

		initialize_property<bool>    ("ReSTIR DI",                     false, [this](){ return &mDefines["gReSTIR_DI"]; }),
		initialize_property<uint32_t>("DI candidate samples",             32, [this](){ return &mPushConstants["mDICandidateSamples"].get<uint32_t>(); }, 0, 128, 0.25f),
		initialize_property<bool>    ("ReSTIR DI reuse",               false, [this](){ return &mDefines["gReSTIR_DI_Reuse"]; }),
		initialize_property<bool>    ("ReSTIR DI reuse visibility",    false, [this](){ return &mDefines["gReSTIR_DI_Reuse_Visibility"]; }),
		initialize_property<float>   ("DI max M",                          3, [this](){ return &mPushConstants["mDIMaxM"].get<float>(); }, 0, 10, .1f),
		initialize_property<float>   ("DI reuse radius",                  32, [this](){ return &mPushConstants["mDIReuseRadius"].get<float>(); }, 0, 1000),

		initialize_property<bool>    ("ReSTIR GI",                     false, [this](){ return &mDefines["gReSTIR_GI"]; }),
		initialize_property<uint32_t>("GI candidate samples",              4, [this](){ return &mPushConstants["mGICandidateSamples"].get<uint32_t>(); }, 0, 128, 0.25f),
		initialize_property<bool>    ("ReSTIR GI reuse",               false, [this](){ return &mDefines["gReSTIR_GI_Reuse"]; }),
		initialize_property<float>   ("GI max M",                          4, [this](){ return &mPushConstants["mGIMaxM"].get<float>(); }, 0, 10, .1f),
		initialize_property<float>   ("GI reuse radius",                  16, [this](){ return &mPushConstants["mGIReuseRadius"].get<float>(); }, 0, 1000),
		initialize_property<uint32_t>("GI reuse samples",                  3, [this](){ return &mPushConstants["mGIReuseSamples"].get<uint32_t>(); }, 0, 100),

		initialize_property<bool>    ("PairwiseMis",                   false, [this](){ return &mDefines["gPairwiseMis"]; }),
		initialize_property<bool>    ("TalbotMis",                     false, [this](){ return &mDefines["gTalbotMis"]; }),

		initialize_property<bool>    ("Denoise",                       false, [this](){ return &mDenoise; }),
		initialize_property<bool>    ("Tonemap",                       false, [this](){ return &mTonemap; }),
	};

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
		changed = true;
	}

	for (const auto& p : mProperties)
		if (p->drawGui())
			changed = true;

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

		if (changed && mRandomPerFrame && mDenoise && denoiser && !denoiser->reprojection()) {
			denoiser->resetAccumulation();
		}

		if (mRandomPerFrame)
			mPushConstants["mRandomSeed"] = (mDenoise && denoiser) ? denoiser->accumulatedFrames() : (uint32_t)rand();
		else
			mPushConstants["mRandomSeed"] = 0u;
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

	if (mDefines.at("gDebugPixel")) {
		const ImVec2 c = ImGui::GetIO().MousePos;
		mPushConstants["mPackedDebugPixelIndex"] = ((int)c.x & 0xFFFF) | ((int)c.y << 16);
	}

	// create descriptor sets
	const shared_ptr<DescriptorSets> descriptorSets = mResourcePool.getDescriptorSets(*renderPipeline, "DescriptorSets", descriptors);

	// render
	{
		ProfilerScope ps("Trace paths", &commandBuffer);
		renderPipeline->dispatchTiled(commandBuffer, extent, descriptorSets, {}, mPushConstants);
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


	// copy VisibilityData for selected pixel (scene object picking)
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
}

}