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
		{ "gSampleDirectIllumination", true },
		{ "gUseVC", true },
		{ "gMultiDispatch", false },
		{ "gDeferShadowRays", false },
		{ "gDebugPaths", false },
		{ "gDebugPathWeights", false },
		{ "gLambertian", false },
	};

	mPushConstants["mMaxDepth"] = 5u;
	mPushConstants["mMaxNullCollisions"] = 1000;
	mPushConstants["mDebugPathLengths"] = 3 | (1<<16);
	mPushConstants["mEnvironmentSampleProbability"] = 0.9f;

	Device& device = *mNode.findAncestor<Device>();

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
	mPipelines.emplace("HashGridComputeIndices", ComputePipelineCache(shaderPath / "hashgrid.slang", "ComputeIndices", "sm_6_6", { "-O3", "-matrix-layout-row-major", "-capability", "spirv_1_5" }));
	mPipelines.emplace("HashGridSwizzle",        ComputePipelineCache(shaderPath / "hashgrid.slang", "Swizzle"       , "sm_6_6", { "-O3", "-matrix-layout-row-major", "-capability", "spirv_1_5" }));
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
	}

	if (ImGui::CollapsingHeader("Configuration")) {
		if (mDefines.at("gDebugPaths")) {
			if (ImGui::DragScalarN("Length, light vertices", ImGuiDataType_U16, &mPushConstants["mDebugPathLengths"].get<uint32_t>(), 2, .2f)) changed = true;
		}
		if (ImGui::Checkbox("Random frame seed", &mRandomPerFrame)) changed = true;
		ImGui::PushItemWidth(40);
		uint32_t one = 1;
		if (ImGui::DragScalar("Max depth", ImGuiDataType_U32, &mPushConstants["mMaxDepth"].get<uint32_t>(), .2f, &one)) changed = true;
		if (ImGui::DragScalar("Max null collisions", ImGuiDataType_U32, &mPushConstants["mMaxNullCollisions"].get<uint32_t>())) changed = true;
		if (ImGui::SliderFloat("Environment sample p", &mPushConstants["mEnvironmentSampleProbability"].get<float>(), 0, 1)) changed = true;

		if (mDefines.at("gUseVC") || mLightTrace) {
			if (ImGui::SliderFloat("Light subpath count", &mLightSubpathCount, 0, 2)) changed = true;
		}

		if (ImGui::CollapsingHeader("Hash grid")) {
			ImGui::Indent();
			if (ImGui::DragScalar("Cell count", ImGuiDataType_U32, &mHashGridCellCount)) changed = true;
			if (ImGui::DragFloat("Min cell size", &mHashGridCellSize, .01f)) changed = true;
			if (ImGui::DragFloat("Cell pixel radius", &mHashGridCellPixelRadius, .5f, 0, 1000)) changed = true;
			ImGui::Unindent();
		}

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

	auto pathStates   = mResourcePool.getBuffer<float4x4>(commandBuffer.mDevice, "mPathStates", mDefines.at("gMultiDispatch") ? extent.width*extent.height : 1);
	auto atomicOutput = mResourcePool.getBuffer<uint4>(commandBuffer.mDevice, "mOutputAtomic", (mDefines.at("gDeferShadowRays")||mDefines.at("gUseVC")||mLightTrace) ? extent.width*extent.height : 1, vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst);

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
		if (sceneData.mDescriptors.empty())
			return;

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

		mPushConstants["mOutputExtent"] = uint2(renderTarget.extent().width, renderTarget.extent().height);
		mPushConstants["mLightSubpathCount"] = max(1u, uint32_t(renderTarget.extent().width * renderTarget.extent().height * mLightSubpathCount));
		mPushConstants["mViewCount"] = (uint32_t)views.size();
		mPushConstants["mEnvironmentMaterialAddress"] = sceneData.mEnvironmentMaterialAddress;
		mPushConstants["mLightCount"] = sceneData.mLightCount;
		mPushConstants["mVolumeInstanceCount"] = (uint32_t)sceneData.mInstanceVolumeInfo.size();
		float4 sphere;
		sphere.head<3>() = (sceneData.mAabbMax + sceneData.mAabbMin) / 2;
		sphere[3] = length<float,3>(sceneData.mAabbMax - sphere.head<3>());
		mPushConstants["mSceneSphere"] = sphere;
		if (mRandomPerFrame)
			mPushConstants["mRandomSeed"] = (uint32_t)rand();
		else
			mPushConstants["mRandomSeed"] = 0u;
	}

	auto makeHashGrid = [&]<typename T>(const string& name, const uint32_t size, const uint cellCount) {
		struct HashGridConstants {
			uint mCellCount;
			uint mCellPixelRadius;
			uint mMinCellSize;
			uint pad;
			float3 mCameraPosition;
			float mDistanceScale;
		};
		HashGridConstants constants;
		constants.mCellCount = cellCount;
		constants.mCellPixelRadius = mHashGridCellPixelRadius;
		constants.mMinCellSize = mHashGridCellSize;
		constants.mCameraPosition = viewTransformsBufferData[0].transformPoint(float3::Zero());
		constants.mDistanceScale = tan(constants.mCellPixelRadius * viewsBufferData[0].mProjection.mVerticalFoV * max(1.0f / extent.height, extent.height / (float)pow2(extent.width)));

		const unordered_map<string, Buffer::View<byte>> buffers {
			{ "mChecksums"        , mResourcePool.getBuffer<uint32_t>          (commandBuffer.mDevice, name + ".mChecksums", cellCount, vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst) },
			{ "mCounters"         , mResourcePool.getBuffer<uint32_t>          (commandBuffer.mDevice, name + ".mCounters" , cellCount+3, vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst) },
			{ "mAppendDataIndices", mResourcePool.getBuffer<uint2>             (commandBuffer.mDevice, name + ".mAppendDataIndices", size) },
			{ "mAppendData"       , mResourcePool.getBuffer<T>                 (commandBuffer.mDevice, name + ".mAppendData", size) },
			{ "mIndices"          , mResourcePool.getBuffer<uint32_t>          (commandBuffer.mDevice, name + ".mIndices", cellCount) },
			{ "mActiveCellIndices", mResourcePool.getBuffer<uint32_t>          (commandBuffer.mDevice, name + ".mActiveCellIndices", cellCount) },
			{ "mData"             , mResourcePool.getBuffer<T>                 (commandBuffer.mDevice, name + ".mData", size) },
			{ "mConstants"        , mResourcePool.uploadData<HashGridConstants>(commandBuffer        , name + ".mConstants", constants, vk::BufferUsageFlagBits::eUniformBuffer|vk::BufferUsageFlagBits::eTransferDst) },
		};

		Descriptors hashGridDescriptors;
		for (const auto&[bufferName, buf] : buffers) {
			descriptors[{"gRenderParams." + name + "." + bufferName, 0 }] = buf;
			hashGridDescriptors[{ "gHashGrid." + bufferName, 0 }] = buf;
		}
		return hashGridDescriptors;
	};
	auto clearHashGrid = [&](const Descriptors& hashGridDescriptors) {
		get<BufferDescriptor>(hashGridDescriptors.at({ "gHashGrid.mChecksums", 0 })).fill(commandBuffer, 0);
		get<BufferDescriptor>(hashGridDescriptors.at({ "gHashGrid.mCounters", 0 })).fill(commandBuffer, 0);

		Buffer::barriers(commandBuffer, {
			get<BufferDescriptor>(hashGridDescriptors.at({"gHashGrid.mChecksums",0})),
			get<BufferDescriptor>(hashGridDescriptors.at({"gHashGrid.mCounters",0})) },
			vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
			vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);
	};
	auto buildHashGrid = [&](const Descriptors& hashGridDescriptors, const string& dataTypeStr, const uint32_t size, const uint cellCount) {
		auto computeIndicesPipeline = mPipelines.at("HashGridComputeIndices").get(commandBuffer.mDevice, { { "DataType", dataTypeStr } });
		const auto hashGridDescriptorSets = computeIndicesPipeline->getDescriptorSets(hashGridDescriptors);

		{
			get<BufferDescriptor>(hashGridDescriptors.at({"gHashGrid.mCounters",0})).barrier(
				commandBuffer,
				vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
				vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);

			computeIndicesPipeline->dispatchTiled(commandBuffer, vk::Extent3D{cellCount%1024, cellCount/1024, 1}, hashGridDescriptorSets);
		}

		{
			Buffer::barriers(commandBuffer, {
				get<BufferDescriptor>(hashGridDescriptors.at({"gHashGrid.mIndices",0})),
				get<BufferDescriptor>(hashGridDescriptors.at({"gHashGrid.mAppendData",0})),
				get<BufferDescriptor>(hashGridDescriptors.at({"gHashGrid.mAppendDataIndices",0})) },
				vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
				vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);

			auto swizzlePipeline = mPipelines.at("HashGridSwizzle").get(commandBuffer.mDevice, { { "DataType", dataTypeStr } }, computeIndicesPipeline->descriptorSetLayouts());
			swizzlePipeline->dispatchTiled(commandBuffer, vk::Extent3D{size%1024, size/1024, 1}, hashGridDescriptorSets);

			get<BufferDescriptor>(hashGridDescriptors.at({"gHashGrid.mData",0})).barrier(
				commandBuffer,
				vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
				vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);
		}
	};


	const uint32_t maxShadowRays = mDefines.at("gDeferShadowRays") ? max(1u, (2*extent.width*extent.height + mPushConstants["mLightSubpathCount"].get<uint32_t>())*(mPushConstants["mMaxDepth"].get<uint32_t>()-1)) : 1;

	auto lightVertexBuffer = mResourcePool.getBuffer<array<float4,3>>(commandBuffer.mDevice, "mLightVertices", mDefines.at("gUseVC") ? max(1u, mPushConstants["mLightSubpathCount"].get<uint32_t>()*(mPushConstants["mMaxDepth"].get<uint32_t>()-1)) : 1);
	auto counterBuffer = mResourcePool.getBuffer<uint32_t>(commandBuffer.mDevice, "mCounters", 2, vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst);
	descriptors[{ "gRenderParams.mLightVertices", 0 }] = lightVertexBuffer;
	descriptors[{ "gRenderParams.mCounters", 0 }] = counterBuffer;
	descriptors[{ "gRenderParams.mShadowRays", 0 }] = mResourcePool.getBuffer<array<float4,4>>(commandBuffer.mDevice, "mShadowRays", max(1u, maxShadowRays));


	Defines defines;
	for (const auto&[define,enabled] : mDefines)
		if (enabled)
			defines[define] = to_string(enabled);
	if (mPushConstants["mVolumeInstanceCount"].get<uint32_t>() > 0)
		defines.emplace("gHasMedia", "true");
	if (hasHeterogeneousMedia)
		defines.emplace("gHasHeterogeneousMedia", "true");
	if (mDefines.at("gDebugPathWeights") && !mDefines.at("gDebugPaths"))
		defines.emplace("gDebugPaths", "true");

	if (mLightTrace) {
		defines.erase("gUseVC");
		defines.erase("gSampleDirectIllumination");
	}
	if (mDefines.at("gUseVC"))
		defines.erase("gSampleDirectIllumination");

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
		renderPipeline = loadPipeline("Render", defines);
		if (mDefines.at("gMultiDispatch"))
			renderIterationPipeline = loadPipeline("RenderIteration", defines);
		if (mDefines.at("gUseVC") || mLightTrace) {
			Defines tmp = defines;
			tmp["gTraceFromLight"] = "true";
			renderLightPipeline = loadPipeline("Render", tmp);
			if (mDefines.at("gMultiDispatch"))
				renderLightIterationPipeline = loadPipeline("RenderIteration", tmp);
		}
		if (mDefines.at("gDeferShadowRays"))
			processShadowRaysPipeline = loadPipeline("ProcessShadowRays", defines);
		if (mDefines.at("gDeferShadowRays") || mDefines.at("gUseVC") || mLightTrace)
			processAtomicOutputPipeline = loadPipeline("ProcessAtomicOutput", Defines{ { "gClearImage", to_string(mLightTrace) }});
	}
	if (loading) {
		// compiling shaders...
		renderTarget.clearColor(commandBuffer, vk::ClearColorValue(array<uint32_t, 4>{ 0, 0, 0, 0 }));
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

			renderPipeline->dispatchTiled(commandBuffer, extent, descriptorSets, {}, mPushConstants);
			if (mDefines.at("gMultiDispatch")) {
				for (uint32_t i = 1; i < mPushConstants["mMaxDepth"].get<uint32_t>(); i++) {
					pathStates.barrier(commandBuffer,
						vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
						vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);
					renderIterationPipeline->dispatchTiled(commandBuffer, extent, descriptorSets, {}, mPushConstants);
				}
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
	if (mTonemap && tonemapper) {
		tonemapper->render(commandBuffer, processedOutput, outputImage, (mDenoise && denoiser && denoiser->demodulateAlbedo() && processedOutput != outputImage) ? albedoImage : Image::View{});
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
}

}