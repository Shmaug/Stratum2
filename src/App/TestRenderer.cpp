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

	Device& device = *mNode.findAncestor<Device>();
	createPipelines(device);

	mPushConstants["mRRDepth"] = 2u;
	mPushConstants["mMaxDepth"] = 5u;
	mPushConstants["mMaxNullCollisions"] = 1000;
	mPushConstants["mDebugPathLengths"] = 3 | (1<<16);
	mPushConstants["mEnvironmentSampleProbability"] = 0.9f;
}

void TestRenderer::createPipelines(Device& device) {
	const auto samplerRepeat = make_shared<vk::raii::Sampler>(*device, vk::SamplerCreateInfo({},
		vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear,
		vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat,
		0, true, 8, false, vk::CompareOp::eAlways, 0, VK_LOD_CLAMP_NONE));
	device.setDebugName(**samplerRepeat, "TestRenderer/Sampler");

	ComputePipeline::Metadata md;
	md.mImmutableSamplers["gScene.mStaticSampler"]  = { samplerRepeat };
	md.mImmutableSamplers["gScene.mStaticSampler1"] = { samplerRepeat };
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
	mPipeline = ComputePipelineCache(shaderPath / "testrenderer.slang", "render", "sm_6_6", args, md);
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

	if (ImGui::CollapsingHeader("Configuration")) {
		if (ImGui::Checkbox("Alpha testing", &mAlphaTest)) changed = true;
		if (ImGui::Checkbox("Normal maps", &mNormalMaps)) changed = true;
		if (ImGui::Checkbox("Shading normals", &mShadingNormals)) changed = true;
		if (ImGui::Checkbox("Sample direct illumination", &mSampleDirectIllumination)) changed = true;

		if (ImGui::Checkbox("Performance counters", &mPerformanceCounters)) changed = true;
		if (ImGui::Checkbox("Debug paths", &mDebugPaths)) changed = true;
		if (mDebugPaths) {
			if (ImGui::DragScalarN("Length, light vertices", ImGuiDataType_U16, &mPushConstants["mDebugPathLengths"].get<uint32_t>(), 2, .2f)) changed = true;
		}

		if (ImGui::Checkbox("Random frame seed", &mRandomPerFrame)) changed = true;
		ImGui::PushItemWidth(40);
		if (ImGui::DragScalar("RR depth",  ImGuiDataType_U32, &mPushConstants["mRRDepth"].get<uint32_t>(), .2f)) changed = true;
		if (ImGui::DragScalar("Max depth", ImGuiDataType_U32, &mPushConstants["mMaxDepth"].get<uint32_t>(), .2f)) changed = true;
		if (ImGui::DragScalar("Max null collisions", ImGuiDataType_U32, &mPushConstants["mMaxNullCollisions"].get<uint32_t>())) changed = true;
		if (ImGui::SliderFloat("Environment sample p", &mPushConstants["mEnvironmentSampleProbability"].get<float>(), 0, 1)) changed = true;
		ImGui::PopItemWidth();
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

void TestRenderer::render(CommandBuffer& commandBuffer, const Image::View& renderTarget) {
	ProfilerScope ps("PathTracer::render", &commandBuffer);

	mResourcePool.clean();

	const shared_ptr<Scene>      scene      = mNode.findAncestor<Scene>();
	const shared_ptr<Denoiser>   denoiser   = mNode.findDescendant<Denoiser>();
	const shared_ptr<Tonemapper> tonemapper = mNode.findDescendant<Tonemapper>();

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

	const vk::Extent3D extent = renderTarget.extent();

	const Image::View outputImage = mResourcePool.getImage(commandBuffer.mDevice, "mOutput", Image::Metadata{
		.mFormat = vk::Format::eR32G32B32A32Sfloat,
		.mExtent = extent,
		.mUsage = vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eTransferSrc,
	});
	const Image::View albedoImage = mResourcePool.getImage(commandBuffer.mDevice, "mAlbedo", Image::Metadata{
		.mFormat = vk::Format::eR32G32B32A32Sfloat,
		.mExtent = extent,
		.mUsage = vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eTransferSrc,
	});
	const Image::View prevUVsImage = mResourcePool.getImage(commandBuffer.mDevice, "mPrevUVs", Image::Metadata{
		.mFormat = vk::Format::eR32G32Sfloat,
		.mExtent = extent,
		.mUsage = vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eTransferSrc,
	});
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
	descriptors[{ "gRenderParams.mOutput", 0 }]     = ImageDescriptor{ outputImage    , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };
	descriptors[{ "gRenderParams.mAlbedo", 0 }]     = ImageDescriptor{ albedoImage    , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };
	descriptors[{ "gRenderParams.mPrevUVs", 0 }]    = ImageDescriptor{ prevUVsImage   , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };
	descriptors[{ "gRenderParams.mVisibility", 0 }] = ImageDescriptor{ visibilityImage, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };
	descriptors[{ "gRenderParams.mDepth", 0 }]      = ImageDescriptor{ depthImage     , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };

	bool changed = false;
	bool hasMedia = false;

	vector<ViewData> viewsBufferData;
	Buffer::View<ViewData> viewsBuffer;

	// scene descriptors + views
	{
		const Scene::FrameData& sceneData = scene->frameData();
		if (sceneData.mDescriptors.empty())
			return;

		if (scene->lastUpdate() > mLastSceneVersion) {
			changed = true;
			mLastSceneVersion = scene->lastUpdate();
		}

		for (auto& [name, d] : sceneData.mDescriptors)
			descriptors[{ "gScene." + name.first, name.second }] = d;
		descriptors[{ "gScene.mPerformanceCounters", 0u }] = make_shared<Buffer>(commandBuffer.mDevice, "mPerformanceCounters", 4*sizeof(uint32_t), vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

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
		vector<TransformData> viewTransformsBufferData(views.size());
		vector<TransformData> prevInverseViewTransformsData(views.size());
		for (uint32_t i = 0; i < views.size(); i++) {
			const auto&[view,viewTransform] = views[i];
			viewsBufferData[i] = view;
			viewTransformsBufferData[i] = viewTransform;
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
		descriptors[{ "gRenderParams.mPrevViewInverseTransforms", 0 }] = mResourcePool.uploadData<TransformData>(commandBuffer, "mPrevViewInverseTransforms", prevInverseViewTransformsData);


		// find if views are inside a volume
		vector<uint32_t> viewMediumIndices(views.size());
		ranges::fill(viewMediumIndices, INVALID_INSTANCE);

		scene->mNode.forEachDescendant<Medium>([&](Node& node, const shared_ptr<Medium>& vol) {
			hasMedia = true;
			for (uint32_t i = 0; i < views.size(); i++) {
				const float3 localViewPos = nodeToWorld(node).inverse().transformPoint( viewTransformsBufferData[i].transformPoint(float3::Zero()) );
				if (vol->mDensityGrid->grid<float>()->worldBBox().isInside(nanovdb::Vec3R(localViewPos[0], localViewPos[1], localViewPos[2])))
					viewMediumIndices[i] = sceneData.mInstanceTransformMap.at(vol.get()).second;
			}
		});

		descriptors[{ "gRenderParams.mViewMediumIndices", 0 }] = mResourcePool.uploadData<uint>(commandBuffer, "mViewMediumIndices", viewMediumIndices);

		mPushConstants["mOutputExtent"] = uint2(renderTarget.extent().width, renderTarget.extent().height);
		mPushConstants["mViewCount"] = (uint32_t)views.size();
		mPushConstants["mEnvironmentMaterialAddress"] = sceneData.mEnvironmentMaterialAddress;
		mPushConstants["mLightCount"] = sceneData.mLightCount;
		float4 sphere;
		sphere.head<3>() = (sceneData.mAabbMax + sceneData.mAabbMin) / 2;
		sphere[3] = length<float,3>(sceneData.mAabbMax - sphere.head<3>());
		mPushConstants["mSceneSphere"] = sphere;
	}


	Defines defines;
	if (hasMedia)                  defines["gHasMedia"]                 = "true";
	if (mAlphaTest)                defines["gAlphaTest"]                = "true";
	if (mNormalMaps)               defines["gNormalMaps"]               = "true";
	if (mShadingNormals)           defines["gShadingNormals"]           = "true";
	if (mPerformanceCounters)      defines["gPerformanceCounters"]      = "true";
	if (mSampleDirectIllumination) defines["gSampleDirectIllumination"] = "true";
	if (mDebugPaths)               defines["gDebugPaths"]               = "true";

	auto pipeline = mPipeline.get(commandBuffer.mDevice, defines);

	// create descriptor sets
	const shared_ptr<DescriptorSets> descriptorSets = mResourcePool.getDescriptorSets(*pipeline, "DescriptorSets", descriptors);

	if (mRandomPerFrame)
		mPushConstants["mRandomSeed"] = (uint32_t)rand();

	// render
	{
		ProfilerScope ps("Sample visibility", &commandBuffer);
		pipeline->dispatchTiled(commandBuffer, extent, descriptorSets, {}, mPushConstants);
	}

	// post processing

	Image::View processedOutput = outputImage;

	// run denoiser
	if (mDenoise && denoiser && mRandomPerFrame) {
		if (changed && !denoiser->reprojection())
			denoiser->resetAccumulation();

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
	if (mTonemap && tonemapper)
		tonemapper->render(commandBuffer, processedOutput, renderTarget, (mDenoise && denoiser && denoiser->demodulateAlbedo() && processedOutput != outputImage) ? albedoImage : Image::View{});
	else {
		// just copy processedOutput to renderTarget
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
}

}