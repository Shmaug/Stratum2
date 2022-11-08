#include "BDPT.hpp"
#include "Denoiser.hpp"
#include "Gui.hpp"

#include <stb_image_write.h>
#include <random>

#include <Core/Instance.hpp>
#include <Core/CommandBuffer.hpp>
#include <Core/Profiler.hpp>

#include <Shaders/compat/tonemap.h>

namespace tinyvkpt {

/*
struct HashGridData {
	unordered_map<string, Buffer::View<byte>> mBuffers;
	HashGridData(const string& name, const uint32_t bucket_count, const uint32_t size) {
		mBuffers["g"+name+"HashGrid.mChecksums"]     = make_shared<Buffer>(commandBuffer.mDevice, "g"+name+"HashGrid.mChecksums", bucket_count * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal);
		mBuffers["g"+name+"HashGrid.mCounters"]      = make_shared<Buffer>(commandBuffer.mDevice, "g"+name+"HashGrid.mCounters" , bucket_count * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal);
		mBuffers["g"+name+"HashGrid.mIndices"]       = make_shared<Buffer>(commandBuffer.mDevice, "g"+name+"HashGrid.mIndices"  , bucket_count * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
		mBuffers["g"+name+"HashGrid.mStats"]         = make_shared<Buffer>(commandBuffer.mDevice, "g"+name+"HashGrid.mStats"    , 4 * sizeof(uint32_t)           , vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		mBuffers["g"+name+"HashGrid.mData"]          = make_shared<Buffer>(commandBuffer.mDevice, "g"+name+"HashGrid.mData"         , size * sizeof(NEEReservoir), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
		mBuffers["g"+name+"HashGrid.mAppendData"]    = make_shared<Buffer>(commandBuffer.mDevice, "g"+name+"HashGrid.mAppendData"   , size * sizeof(NEEReservoir), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
		mBuffers["g"+name+"HashGrid.mAppendIndices"] = make_shared<Buffer>(commandBuffer.mDevice, "g"+name+"HashGrid.mAppendIndices", size * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal);
	}

	void assign_descriptors(const shared_ptr<DescriptorSet>& descriptorSet, const unordered_map<string,uint32_t>& descriptorMap) {
		for (const auto&[name, buf] : mBuffers)
			descriptorSet->insert_or_assign(descriptorMap.at("gFrameParams." + name), buf);
	}
};
*/

void BDPT::createPipelines(Device& device) {
	// initialize constants
	if (mTonemapPushConstants.empty()) {
		mTonemapPushConstants["gExposure"] = 0.f;
		mTonemapPushConstants["gExposureAlpha"] = 0.1f;

		mSamplingFlags = 0;
		BDPT_SET_FLAG(mSamplingFlags, BDPTFlagBits::eRemapThreads);
		BDPT_SET_FLAG(mSamplingFlags, BDPTFlagBits::eRayCones);
		BDPT_SET_FLAG(mSamplingFlags, BDPTFlagBits::eCoherentRR);
		BDPT_SET_FLAG(mSamplingFlags, BDPTFlagBits::eNormalMaps);
		BDPT_SET_FLAG(mSamplingFlags, BDPTFlagBits::eNEE);
		BDPT_SET_FLAG(mSamplingFlags, BDPTFlagBits::eMIS);
		BDPT_SET_FLAG(mSamplingFlags, BDPTFlagBits::eDeferShadowRays);
		mPushConstants = BDPTPushConstants();

		if (auto arg = device.mInstance.findArgument("minPathVertices"); arg)              mPushConstants.gMinPathVertices = atoi(arg->c_str());
		if (auto arg = device.mInstance.findArgument("maxPathVertices"); arg)              mPushConstants.gMaxPathVertices = atoi(arg->c_str());
		if (auto arg = device.mInstance.findArgument("maxDiffuseVertices"); arg)           mPushConstants.gMaxDiffuseVertices = atoi(arg->c_str());
		if (auto arg = device.mInstance.findArgument("maxNullCollisions"); arg)            mPushConstants.gMaxNullCollisions = atoi(arg->c_str());
		if (auto arg = device.mInstance.findArgument("environmentSampleProbability"); arg) mPushConstants.gEnvironmentSampleProbability = atof(arg->c_str());
		if (auto arg = device.mInstance.findArgument("lightPresampleTileSize"); arg)       mPushConstants.gLightPresampleTileSize = atoi(arg->c_str());
		if (auto arg = device.mInstance.findArgument("lightPresampleTileCount"); arg)      mPushConstants.gLightPresampleTileCount = atoi(arg->c_str());
		if (auto arg = device.mInstance.findArgument("lightPathCount"); arg)               mPushConstants.gLightPathCount = atoi(arg->c_str());
		if (auto arg = device.mInstance.findArgument("reservoirM"); arg)                   mPushConstants.gReservoirM = atoi(arg->c_str());
		if (auto arg = device.mInstance.findArgument("reservoirMaxM"); arg)                mPushConstants.gReservoirMaxM = atoi(arg->c_str());
		if (auto arg = device.mInstance.findArgument("reservoirSpatialM"); arg)            mPushConstants.gReservoirSpatialM = atoi(arg->c_str());
		if (auto arg = device.mInstance.findArgument("hashGridBucketCount"); arg)          mPushConstants.gHashGridBucketCount = atoi(arg->c_str());
		if (auto arg = device.mInstance.findArgument("hashGridMinBucketRadius"); arg)      mPushConstants.gHashGridMinBucketRadius = atof(arg->c_str());
		if (auto arg = device.mInstance.findArgument("hashGridBucketPixelRadius"); arg)    mPushConstants.gHashGridBucketPixelRadius = atoi(arg->c_str());
		if (auto arg = device.mInstance.findArgument("minPathVertices"); arg)              mPushConstants.gMinPathVertices = atoi(arg->c_str());
		if (auto arg = device.mInstance.findArgument("maxPathVertices"); arg)              mPushConstants.gMaxPathVertices = atoi(arg->c_str());
		if (auto arg = device.mInstance.findArgument("exposure"); arg) mTonemapPushConstants["gExposure"] = atof(arg->c_str());
		if (auto arg = device.mInstance.findArgument("exposureAlpha"); arg) mTonemapPushConstants["gExposureAlpha"] = atof(arg->c_str());

		for (string arg : device.mInstance.findArguments("bdptFlag")) {
			if (arg.empty()) continue;

			bool set = true;
			if (arg[0] == '~' || arg[0] == '!') {
				set = false;
				arg = arg.substr(1);
			}

			for (char& c : arg) c = tolower(c);

			BDPTFlagBits flag;
			for (uint32_t i = 0; i < BDPTFlagBits::eBDPTFlagCount; i++) {
				const string flag_str = to_string((BDPTFlagBits)i);

				vector<char> str;
				str.reserve(flag_str.size()+1);
				memset(str.data(), 0, flag_str.size()+1);
				for (char c : flag_str)
					if (c != ' ')
						str.emplace_back(tolower(c));

				if (strcmp(arg.data(), str.data()) == 0) {
					if (set)
						BDPT_SET_FLAG(mSamplingFlags, i);
					else
						BDPT_UNSET_FLAG(mSamplingFlags, i);
					break;
				}
			}
		}
	}

	// create pipelines

	const auto samplerRepeat = make_shared<vk::raii::Sampler>(*device, vk::SamplerCreateInfo({},
		vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear,
		vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat,
		0, true, 8, false, vk::CompareOp::eAlways, 0, VK_LOD_CLAMP_NONE));

	ComputePipeline::Metadata md;
	md.mImmutableSamplers["gStaticSampler"] = { samplerRepeat };
	md.mImmutableSamplers["gStaticSampler1"] = { samplerRepeat };
	md.mBindingFlags["gVolumes"] = vk::DescriptorBindingFlagBits::ePartiallyBound;
	md.mBindingFlags["gImages"] = vk::DescriptorBindingFlagBits::ePartiallyBound;
	md.mBindingFlags["gImage1s"] = vk::DescriptorBindingFlagBits::ePartiallyBound;

	const filesystem::path src_path = *device.mInstance.findArgument("shaderPath") + "/kernels/renderers/bdpt.hlsl";
	const vector<string>& args_rt = { "-matrix-layout-row-major", "-capability", "spirv_1_5" };
	const vector<string>& args = { "-matrix-layout-row-major", "-capability", "spirv_1_5", "-capability", "GL_EXT_ray_tracing" };
	mRenderPipelines[RenderPipelineIndex::eSamplePhotons]          = ComputePipelineCache(src_path, "sample_photons"          , "sm_6_7", args_rt, md);
	mRenderPipelines[RenderPipelineIndex::eSampleVisibility]       = ComputePipelineCache(src_path, "sample_visibility"       , "sm_6_7", args_rt, md);
	mRenderPipelines[RenderPipelineIndex::eSubpathConnect]         = ComputePipelineCache(src_path, "connect"                 , "sm_6_7", args, md);
	mRenderPipelines[RenderPipelineIndex::eTraceShadows]           = ComputePipelineCache(src_path, "trace_shadows"           , "sm_6_7", args_rt, md);
	mRenderPipelines[RenderPipelineIndex::ePresampleLights]        = ComputePipelineCache(src_path, "presample_lights"        , "sm_6_7", args, md);
	mRenderPipelines[RenderPipelineIndex::eHashGridComputeIndices] = ComputePipelineCache(src_path, "hashgrid_compute_indices", "sm_6_7", args, md);
	mRenderPipelines[RenderPipelineIndex::eHashGridSwizzle]        = ComputePipelineCache(src_path, "hashgrid_swizzle"        , "sm_6_7", args, md);
	mRenderPipelines[RenderPipelineIndex::eAddLightTrace]          = ComputePipelineCache(src_path, "add_light_trace"         , "sm_6_7", args, md);

	mTonemapPipeline = ComputePipelineCache("Shaders/kernels/tonemap.hlsl");
	mTonemapMaxReducePipeline = ComputePipelineCache("Shaders/kernels/tonemap.hlsl", "reduce_max");

	mRayCount = make_shared<Buffer>(device, "gCounters", 4*sizeof(uint32_t), vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
	mPrevRayCount.resize(mRayCount.size());
	mRaysPerSecond.resize(mRayCount.size());
	memset(mRayCount.data(), 0, mRayCount.sizeBytes());
	memset(mPrevRayCount.data(), 0, mPrevRayCount.size()*sizeof(uint32_t));
	mRayCountTimer = 0;
}

void BDPT::drawGui() {
	if (ImGui::Button("Reload BDPT Shaders")) {
		Device& device = *mNode.findAncestor<Device>();
		device->waitIdle();
		createPipelines(device);
	}
	if (ImGui::Button("Clear resources")) {
		mFrameResourcePool.clear();
		mCurFrame.reset();
		mPrevFrame.reset();
	}

	ImGui::SetNextItemWidth(200);
	Gui::enumDropdown<BDPTDebugMode>("BDPT debug mode", mDebugMode, (uint32_t)BDPTDebugMode::eDebugModeCount);
	if (mDebugMode == BDPTDebugMode::ePathLengthContribution) {
		ImGui::Indent();
		ImGui::PushItemWidth(40);
		ImGui::DragScalar("View path length", ImGuiDataType_U32, &mPushConstants.gDebugViewPathLength, 1);
		ImGui::DragScalar("Light path length", ImGuiDataType_U32, &mPushConstants.gDebugLightPathLength, 1);
		ImGui::PopItemWidth();
		ImGui::Unindent();
	}

	if (BDPT_CHECK_FLAG(mSamplingFlags, BDPTFlagBits::ePerformanceCounters)) {
		const auto [rps, ext] = formatNumber(mRaysPerSecond[0]);
		ImGui::Text("%.2f%s Rays/second (%u%% shadow rays)", rps, ext, (uint32_t)(100 - (100*(uint64_t)mRaysPerSecond[1]) / mRaysPerSecond[0]));
	}

	ImGui::Checkbox("Pause rendering", &mPauseRendering);

	if (ImGui::CollapsingHeader("Configuration")) {
		ImGui::Checkbox("Random frame seed", &mRandomPerFrame);
		ImGui::Checkbox("Half precision", &mHalfColorPrecision);
		ImGui::Checkbox("Force lambertian", &mForceLambertian);
		for (uint i = 0; i < BDPTFlagBits::eBDPTFlagCount; i++)
			ImGui::CheckboxFlags(to_string((BDPTFlagBits)i).c_str(), &mSamplingFlags, BIT(i));
	}

	if (ImGui::CollapsingHeader("Path tracing")) {
		ImGui::PushItemWidth(60);
		ImGui::DragScalar("Max Path vertices", ImGuiDataType_U32, &mPushConstants.gMaxPathVertices);
		ImGui::DragScalar("Min path vertices", ImGuiDataType_U32, &mPushConstants.gMinPathVertices);
		ImGui::DragScalar("Max diffuse vertices", ImGuiDataType_U32, &mPushConstants.gMaxDiffuseVertices);
		ImGui::DragScalar("Max null collisions", ImGuiDataType_U32, &mPushConstants.gMaxNullCollisions);

		if (BDPT_CHECK_FLAG(mSamplingFlags, BDPTFlagBits::eConnectToViews))
			ImGui::InputScalar("Light trace quantization", ImGuiDataType_U32, &mLightTraceQuantization);

		if (BDPT_CHECK_FLAG(mSamplingFlags, BDPTFlagBits::eLVC)) {
			uint32_t mn = 1;
			ImGui::DragScalar("Light path count", ImGuiDataType_U32, &mPushConstants.gLightPathCount, 1, &mn);
		}

		if (BDPT_CHECK_FLAG(mSamplingFlags, BDPTFlagBits::eNEE) && mPushConstants.gEnvironmentMaterialAddress != -1 && mPushConstants.gLightCount > 0)
			ImGui::DragFloat("Environment sample probability", &mPushConstants.gEnvironmentSampleProbability, .1f, 0, 1);

		if (BDPT_CHECK_FLAG(mSamplingFlags, BDPTFlagBits::ePresampleLights)) {
			ImGui::Indent();
			ImGui::DragScalar("Presample tile size", ImGuiDataType_U32, &mPushConstants.gLightPresampleTileSize);
			ImGui::DragScalar("Presample tile count", ImGuiDataType_U32, &mPushConstants.gLightPresampleTileCount);
			mPushConstants.gLightPresampleTileSize  = max(mPushConstants.gLightPresampleTileSize, 1u);
			mPushConstants.gLightPresampleTileCount = max(mPushConstants.gLightPresampleTileCount, 1u);
			const auto [n, ext] = formatNumber(mPushConstants.gLightPresampleTileSize*mPushConstants.gLightPresampleTileCount*sizeof(PresampledLightPoint));
			ImGui::Text("Presampled light buffer is %.2f%s bytes", n, ext);
			ImGui::Unindent();
		}

		if (BDPT_CHECK_FLAG(mSamplingFlags, BDPTFlagBits::eNEEReservoirs) || BDPT_CHECK_FLAG(mSamplingFlags, BDPTFlagBits::eLVCReservoirs)) {
			ImGui::Indent();
			ImGui::DragScalar("Reservoir candidate samples", ImGuiDataType_U32, &mPushConstants.gReservoirM);
			if (mPushConstants.gReservoirM == 0) mPushConstants.gReservoirM = 1;
			if (BDPT_CHECK_FLAG(mSamplingFlags, BDPTFlagBits::eNEEReservoirReuse) || BDPT_CHECK_FLAG(mSamplingFlags, BDPTFlagBits::eLVCReservoirReuse)) {
				ImGui::DragScalar("Spatial candidates", ImGuiDataType_U32, &mPushConstants.gReservoirSpatialM);
				ImGui::DragScalar("Max M", ImGuiDataType_U32, &mPushConstants.gReservoirMaxM);
				ImGui::DragScalar("Bucket count", ImGuiDataType_U32, &mPushConstants.gHashGridBucketCount);
				ImGui::DragFloat("Bucket pixel radius", &mPushConstants.gHashGridBucketPixelRadius);
				ImGui::DragFloat("Min bucket radius", &mPushConstants.gHashGridMinBucketRadius);
				mPushConstants.gHashGridBucketCount = max(mPushConstants.gHashGridBucketCount, 1u);
				if (BDPT_CHECK_FLAG(mSamplingFlags, BDPTFlagBits::ePerformanceCounters)) {
					if (mCurFrame && mCurFrame->mPathData.contains("gNEEHashGrid.mStats")) {
						if (BDPT_CHECK_FLAG(mSamplingFlags, BDPTFlagBits::eNEEReservoirReuse)) {
							Buffer::View<uint32_t> data = mCurFrame->mPathData.at("gNEEHashGrid.mStats").cast<uint32_t>();
							ImGui::Text("NEE: %u failed inserts, %u%% buckets used", data[0], (100*data[1])/mPushConstants.gHashGridBucketCount);
						}
						if (BDPT_CHECK_FLAG(mSamplingFlags, BDPTFlagBits::eLVCReservoirReuse)) {
							Buffer::View<uint32_t> data = mCurFrame->mPathData.at("gLVCHashGrid.mStats").cast<uint32_t>();
							ImGui::Text("LVC: %u failed inserts, %u%% buckets used", data[0], (100*data[1])/mPushConstants.gHashGridBucketCount);
						}
					}
				}
			}
			ImGui::Unindent();
		}

		ImGui::PopItemWidth();
	}

	if (ImGui::CollapsingHeader("Post processing")) {
		ImGui::Indent();
		if (auto denoiser = mNode.getComponent<Denoiser>(); denoiser)
			if (ImGui::Checkbox("Enable denoiser", &mDenoise))
				denoiser->resetAccumulation();
		Gui::enumDropdown<TonemapMode>("Tonemap", mTonemapMode, (uint32_t)TonemapMode::eTonemapModeCount);
		ImGui::PushItemWidth(40);
		ImGui::DragFloat("Exposure", &mTonemapPushConstants["gExposure"].get<float>(), .1f, -10, 10);
		ImGui::DragFloat("Exposure alpha", &mTonemapPushConstants["gExposureAlpha"].get<float>(), .1f, 0, 1);
		ImGui::PopItemWidth();
		ImGui::Checkbox("Gamma correct", &mGammaCorrect);
		ImGui::Unindent();
	}

	if (mPrevFrame && ImGui::CollapsingHeader("Export")) {
		ImGui::Indent();
		static char path[256]{ 'i', 'm', 'a', 'g', 'e', '.', 'h', 'd', 'r', '\0' };
		ImGui::InputText("##", path, sizeof(path));
		if (ImGui::Button("Save")) {
			Device& d = mPrevFrame->mRadiance.image()->mDevice;
			auto cb = d.getCommandBuffer(d.findQueueFamily(vk::QueueFlagBits::eTransfer));

			Image::View src = mDenoise ? mPrevFrame->mDenoiseResult : mPrevFrame->mRadiance;
			if (src.image()->format() != vk::Format::eR32G32B32A32Sfloat) {
				Image::Metadata md = {};
				md.mExtent = src.extent();
				md.mFormat = vk::Format::eR32G32B32A32Sfloat;
				md.mUsage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eStorage;
				Image::View tmp = make_shared<Image>(cb->mDevice, "gRadiance", md);
				Image::blit(*cb, src, tmp);
				src = tmp;
			}

			src.barrier(*cb, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferRead);

			Buffer::View<float> pixels = make_shared<Buffer>(d, "image copy tmp", src.extent().width * src.extent().height * sizeof(float) * 4, vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
			(*cb)->copyImageToBuffer(
				**src.image(), vk::ImageLayout::eTransferSrcOptimal,
				**pixels.buffer(), vk::BufferImageCopy(pixels.offset(), 0, 0, src.subresourceLayer(), vk::Offset3D{0,0,0}, src.extent()));

			d->waitIdle(); // wait for mPrevFrame->mRadiance
			d.submit(0,cb);
			if (d->waitForFences(**cb->fence(), true, ~0llu) != vk::Result::eSuccess)
				cerr << "Failed to wait for fence" << endl;

			stbi_write_hdr(path, src.extent().width, src.extent().height, 4, pixels.data());
			ImGui::Unindent();
		}
	}
}

void BDPT::update(CommandBuffer& commandBuffer, const float deltaTime) {
	ProfilerScope ps("BDPT::update", &commandBuffer);

	// reuse old frame resources
	{
		ProfilerScope ps("Allocate Frame Resources", &commandBuffer);
		mPrevFrame = mCurFrame;
		mCurFrame = mFrameResourcePool.get();
		if (!mCurFrame) {
			mCurFrame = make_shared<FrameResources>(commandBuffer.mDevice);
			mFrameResourcePool.emplace(mCurFrame);
		}

		/*
		// scene object picker
		if (!ImGui::GetIO().WantCaptureMouse && mCurFrame && mCurFrame->mSelectionData && mCurFrame->mSelectionDataValid) {
			const uint32_t selectedInstance = mCurFrame->mSelectionData.data()->instance_index();
			if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
				shared_ptr<Inspector> inspector = mNode.findAncestor<Inspector>();
				if (selectedInstance == INVALID_INSTANCE || selectedInstance >= mCurFrame->mSceneData->mInstanceNodes.size())
					inspector->select(nullptr);
				else {
					Node* selected = mCurFrame->mSceneData->mInstanceNodes[selectedInstance];
					inspector->select(mNode.node_graph().contains(selected) ? selected : nullptr);
				}
			}
		}
		*/
		if (mPrevFrame)
			mCurFrame->mFrameNumber = mPrevFrame->mFrameNumber + 1;
		else
			mCurFrame->mFrameNumber = 0;
	}

	mCurFrame->mSceneData = mNode.findAncestor<Scene>()->resources();
	if (!mCurFrame->mSceneData) return;

	mRayCountTimer += deltaTime;
	if (mRayCountTimer > 1) {
		for (uint32_t i = 0; i < mRaysPerSecond.size(); i++)
			mRaysPerSecond[i] = (mRayCount[i] - mPrevRayCount[i]) / mRayCountTimer;
		ranges::copy(mRayCount, mPrevRayCount.begin());
		mRayCountTimer = 0;
	}

	mPushConstants.gEnvironmentMaterialAddress = mCurFrame->mSceneData->mEnvironmentMaterialAddress;
	mPushConstants.gLightCount = (uint32_t)mCurFrame->mSceneData->mLightInstanceMap.size();
}

void BDPT::render(CommandBuffer& commandBuffer, const Image::View& renderTarget, const vector<pair<ViewData,TransformData>>& views) {
	if (!mCurFrame || !mCurFrame->mSceneData) {
		renderTarget.clearColor(commandBuffer, vk::ClearColorValue(array<uint32_t, 4>{ 0, 0, 0, 0 }));
		return;
	}

	commandBuffer.trackResource(mCurFrame);

	if (mPauseRendering) {
		if (mCurFrame->mTonemapResult.image()->format() == renderTarget.image()->format())
			Image::copy(commandBuffer, mCurFrame->mTonemapResult, renderTarget);
		else
			Image::blit(commandBuffer, mCurFrame->mTonemapResult, renderTarget);
		return;
	}

	ProfilerScope ps("BDPT::render", &commandBuffer);

	const vk::Extent3D extent = renderTarget.extent();

	bool has_volumes = false;

	// copy camera/view data to GPU, compute gViewMediumInstances
	{
		ProfilerScope ps("Upload views", &commandBuffer);
		// upload viewdata
		mCurFrame->mViews = make_shared<Buffer>(commandBuffer.mDevice, "gViews", views.size() * sizeof(ViewData), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		mCurFrame->mViewTransforms = make_shared<Buffer>(commandBuffer.mDevice, "gViewTransforms", views.size() * sizeof(TransformData), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		mCurFrame->mViewInverseTransforms = make_shared<Buffer>(commandBuffer.mDevice, "gViewInverseTransforms", views.size() * sizeof(TransformData), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		for (uint32_t i = 0; i < views.size(); i++) {
			mCurFrame->mViews[i] = views[i].first;
			mCurFrame->mViewTransforms[i] = views[i].second;
			mCurFrame->mViewInverseTransforms[i] = views[i].second.inverse();
		}

		// find if views are inside a volume
		mCurFrame->mViewMediumIndices = make_shared<Buffer>(commandBuffer.mDevice, "gViewMediumInstances", views.size() * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		ranges::fill(mCurFrame->mViewMediumIndices, INVALID_INSTANCE);
		mNode.forEachDescendant<Medium>([&](const NodePtr& node, const shared_ptr<Medium>& vol) {
			has_volumes = true;
			for (uint32_t i = 0; i < views.size(); i++) {
				const float3 localViewPos = nodeToWorld(*node).inverse().transform_point( views[i].second.transform_point(float3::Zero()) );
				if (vol->mDensityGrid->grid<float>()->worldBBox().isInside(nanovdb::Vec3R(localViewPos[0], localViewPos[1], localViewPos[2])))
					mCurFrame->mViewMediumIndices[i] = mCurFrame->mSceneData->mInstanceTransformMap.at(vol.get()).second;
			}
		});
	}

	if (!BDPT_CHECK_FLAG(mSamplingFlags, BDPTFlagBits::eLVC))
		mPushConstants.gLightPathCount = extent.width * extent.height;

	shared_ptr<Denoiser> denoiser = mNode.findDescendant<Denoiser>();
	const bool reprojection = denoiser ? denoiser->reprojection() : false;
	const bool changed = mPrevFrame && mPrevFrame->mViewTransforms && (mCurFrame->mViewTransforms[0].m != mPrevFrame->mViewTransforms[0].m).any();

	// per-frame push constants
	BDPTPushConstants push_constants = mPushConstants;
	push_constants.gOutputExtent = uint2(extent.width, extent.height);
	push_constants.gViewCount = (uint32_t)views.size();
	if (mRandomPerFrame) push_constants.gRandomSeed = mCurFrame->mFrameNumber;

	if ((changed && !reprojection) || !mPrevFrame || (!mPrevFrame->mPathData["gNEEHashGrid.mChecksums"] && !mPrevFrame->mPathData["gLVCHashGrid.mChecksums"]))
		push_constants.gReservoirSpatialM = 0;

	// determine sampling flags
	uint32_t sceneFlags = 0;
	uint32_t samplingFlags = mSamplingFlags;
	{
		if (push_constants.gEnvironmentMaterialAddress != -1)
			sceneFlags |= BDPT_FLAG_HAS_ENVIRONMENT;
		else
			push_constants.gEnvironmentSampleProbability = 0;

		if (push_constants.gLightCount)
			sceneFlags |= BDPT_FLAG_HAS_EMISSIVES;
		else
			push_constants.gEnvironmentSampleProbability = 1;

		if (has_volumes)
			sceneFlags |= BDPT_FLAG_HAS_MEDIA;
		else
			push_constants.gMaxNullCollisions = 0;

		if (push_constants.gLightCount == 0 && push_constants.gEnvironmentMaterialAddress == -1) {
			// no lights...
			BDPT_UNSET_FLAG(samplingFlags, BDPTFlagBits::eNEE);
			BDPT_UNSET_FLAG(samplingFlags, BDPTFlagBits::eConnectToViews);
			BDPT_UNSET_FLAG(samplingFlags, BDPTFlagBits::eConnectToLightPaths);
		}

		if (BDPT_CHECK_FLAG(samplingFlags, BDPTFlagBits::eReferenceBDPT)) {
			BDPT_UNSET_FLAG(samplingFlags, BDPTFlagBits::eNEE);
			BDPT_UNSET_FLAG(samplingFlags, BDPTFlagBits::eConnectToViews);
			BDPT_UNSET_FLAG(samplingFlags, BDPTFlagBits::eConnectToLightPaths);
		}

		if (!BDPT_CHECK_FLAG(samplingFlags, BDPTFlagBits::eNEE)) {
			BDPT_UNSET_FLAG(samplingFlags, BDPTFlagBits::ePresampleLights);
			BDPT_UNSET_FLAG(samplingFlags, BDPTFlagBits::eNEEReservoirs);
			BDPT_UNSET_FLAG(samplingFlags, BDPTFlagBits::eNEEReservoirReuse);
		}

		if (!BDPT_CHECK_FLAG(samplingFlags, BDPTFlagBits::eLVC)) {
			BDPT_UNSET_FLAG(samplingFlags, BDPTFlagBits::eLVCReservoirs);
			BDPT_UNSET_FLAG(samplingFlags, BDPTFlagBits::eLVCReservoirReuse);
		}

		if (!BDPT_CHECK_FLAG(samplingFlags, BDPTFlagBits::eLVC) && !BDPT_CHECK_FLAG(samplingFlags, BDPTFlagBits::eNEE))
			BDPT_UNSET_FLAG(samplingFlags, BDPTFlagBits::eDeferShadowRays);
	}

	const bool reservoirReuse = BDPT_CHECK_FLAG(samplingFlags, BDPTFlagBits::eNEEReservoirReuse) || BDPT_CHECK_FLAG(samplingFlags, BDPTFlagBits::eLVCReservoirReuse);

	Defines defines;
	Defines presampleDefines;
	Defines samplePhotonsDefines;
	{
		defines["gSceneFlags"] = to_string(sceneFlags);
		defines["gSpecializationFlags"] = to_string(samplingFlags);
		defines["gDebugMode"] = to_string((uint32_t)mDebugMode);
		defines["gLightTraceQuantization"] = to_string(mLightTraceQuantization);
		if (mForceLambertian)
			defines["FORCE_LAMBERTIAN"] = "1";

		presampleDefines = defines;
		samplePhotonsDefines = defines;

		uint32_t tmp = samplingFlags;
		BDPT_SET_FLAG(tmp, BDPTFlagBits::eUniformSphereSampling);
		BDPT_UNSET_FLAG(tmp, BDPTFlagBits::eRayCones);
		presampleDefines["gSpecializationFlags"] = tmp;
		samplePhotonsDefines["gSpecializationFlags"] = tmp;
		samplePhotonsDefines["gSceneFlags"] = sceneFlags | BDPT_FLAG_TRACE_LIGHT;
	}

	// allocate data on GPU
	{
		ProfilerScope ps("Allocate data", &commandBuffer);

		const uint32_t pixelCount = extent.width * extent.height;
		if (!mCurFrame->mRadiance || mCurFrame->mRadiance.extent() != extent) {
			ProfilerScope ps("Allocate data");

			Image::Metadata md = {};
			md.mExtent = extent;
			md.mFormat = mHalfColorPrecision ? vk::Format::eR16G16B16A16Sfloat : vk::Format::eR32G32B32A32Sfloat;
			md.mUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;

			mCurFrame->mRadiance 	  = make_shared<Image>(commandBuffer.mDevice, "gRadiance",   md);
			mCurFrame->mAlbedo 		  = make_shared<Image>(commandBuffer.mDevice, "gAlbedo",     md);
			mCurFrame->mDebugImage 	  = make_shared<Image>(commandBuffer.mDevice, "gDebugImage", md);
			mCurFrame->mTonemapResult = make_shared<Image>(commandBuffer.mDevice, "gOutput",     md);
			md.mFormat = vk::Format::eR32G32Sfloat;
			mCurFrame->mPrevUVs 	  = make_shared<Image>(commandBuffer.mDevice, "gPrevUVs",    md);

			mCurFrame->mPathData["gVisibility"]        = make_shared<Buffer>(commandBuffer.mDevice, "gVisibility",        pixelCount * sizeof(VisibilityInfo),  vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eDeviceLocal);
			mCurFrame->mPathData["gDepth"]             = make_shared<Buffer>(commandBuffer.mDevice, "gDepth",             pixelCount * sizeof(DepthInfo),       vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
			mCurFrame->mPathData["gLightTraceSamples"] = make_shared<Buffer>(commandBuffer.mDevice, "gLightTraceSamples", pixelCount * sizeof(float4), 		    vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal);
			mCurFrame->mFrameNumber = 0;

			mCurFrame->mTonemapMax    = make_shared<Buffer>(commandBuffer.mDevice, "gMax", sizeof(uint4)*3, vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal);
			mCurFrame->mSelectionData = make_shared<Buffer>(commandBuffer.mDevice, "gSelectionData", sizeof(VisibilityInfo), vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		}

		const uint32_t lightVertexCount = (BDPT_CHECK_FLAG(samplingFlags, BDPTFlagBits::eConnectToLightPaths) || BDPT_CHECK_FLAG(samplingFlags, BDPTFlagBits::eReferenceBDPT)) ?
			push_constants.gLightPathCount * (push_constants.gMaxDiffuseVertices+1) : 1;
		if (!mCurFrame->mPathData.contains("gLightPathVertices") || mCurFrame->mPathData.at("gLightPathVertices").sizeBytes() < lightVertexCount * sizeof(PathVertex)) {
			mCurFrame->mPathData["gLightPathVertices"]    = make_shared<Buffer>(commandBuffer.mDevice, "gLightPathVertices", lightVertexCount * sizeof(PathVertex), vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal, 32);
			mCurFrame->mPathData["gLightPathVertexCount"] = make_shared<Buffer>(commandBuffer.mDevice, "gLightPathVertexCount", sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal, 32);
		}
		const uint32_t viewVertexCount = BDPT_CHECK_FLAG(samplingFlags, BDPTFlagBits::eReferenceBDPT) ? pixelCount * push_constants.gMaxDiffuseVertices : 1;
		if (!mCurFrame->mPathData.contains("gViewPathVertices") || mCurFrame->mPathData.at("gViewPathVertices").sizeBytes() < viewVertexCount * sizeof(PathVertex)) {
			mCurFrame->mPathData["gViewPathVertices"] = make_shared<Buffer>(commandBuffer.mDevice, "gViewPathVertices", viewVertexCount * sizeof(PathVertex), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, 32);
			mCurFrame->mPathData["gViewPathLengths"]  = make_shared<Buffer>(commandBuffer.mDevice, "gViewPathLengths", pixelCount * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, 32);
			mCurFrame->mPathData["gLightPathLengths"] = make_shared<Buffer>(commandBuffer.mDevice, "gLightPathLengths", pixelCount * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, 32);
		}

		const uint32_t shadowRayCount = BDPT_CHECK_FLAG(samplingFlags, BDPTFlagBits::eDeferShadowRays) ? pixelCount * max(push_constants.gMaxDiffuseVertices,1u) : 1;
		if (!mCurFrame->mPathData.contains("gShadowRays") || mCurFrame->mPathData.at("gShadowRays").sizeBytes() < shadowRayCount * sizeof(ShadowRayData))
			mCurFrame->mPathData["gShadowRays"] = make_shared<Buffer>(commandBuffer.mDevice, "gShadowRays", shadowRayCount * sizeof(ShadowRayData), vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal, 32);

		const uint32_t presampledLightCount = BDPT_CHECK_FLAG(samplingFlags, BDPTFlagBits::ePresampleLights) ? max(1u,push_constants.gLightPresampleTileCount*push_constants.gLightPresampleTileSize) : 1;
		if (!mCurFrame->mPathData.contains("gPresampledLights") || mCurFrame->mPathData.at("gPresampledLights").sizeBytes() < presampledLightCount * sizeof(PresampledLightPoint))
			mCurFrame->mPathData["gPresampledLights"] = make_shared<Buffer>(commandBuffer.mDevice, "gPresampledLights", presampledLightCount * sizeof(PresampledLightPoint), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);

		const uint32_t neeHashgridBucketCount = BDPT_CHECK_FLAG(samplingFlags, BDPTFlagBits::eNEEReservoirReuse) ? max(1u,push_constants.gHashGridBucketCount) : 1;
		if (!mCurFrame->mPathData["gNEEHashGrid.mChecksums"] || mCurFrame->mPathData.at("gNEEHashGrid.mChecksums").sizeBytes() < neeHashgridBucketCount * sizeof(uint32_t)) {
			mCurFrame->mPathData["gNEEHashGrid.mChecksums"] = make_shared<Buffer>(commandBuffer.mDevice, "gNEEHashGrid.mChecksums", neeHashgridBucketCount * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal);
			mCurFrame->mPathData["gNEEHashGrid.mCounters"]  = make_shared<Buffer>(commandBuffer.mDevice, "gNEEHashGrid.mCounters" , neeHashgridBucketCount * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal);
			mCurFrame->mPathData["gNEEHashGrid.mIndices"]   = make_shared<Buffer>(commandBuffer.mDevice, "gNEEHashGrid.mIndices"  , neeHashgridBucketCount * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
			mCurFrame->mPathData["gNEEHashGrid.mStats"]     = make_shared<Buffer>(commandBuffer.mDevice, "gNEEHashGrid.mStats"    , 4 * sizeof(uint32_t)                    , vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		}
		const uint32_t neeHashgridReservoirCount  = BDPT_CHECK_FLAG(samplingFlags, BDPTFlagBits::eNEEReservoirReuse) ? pixelCount * max(1u,push_constants.gMaxDiffuseVertices) : 1;
		if (!mCurFrame->mPathData["gNEEHashGrid.mReservoirs"] || mCurFrame->mPathData.at("gNEEHashGrid.mReservoirs").sizeBytes() < neeHashgridReservoirCount * sizeof(ReservoirData)) {
			mCurFrame->mPathData["gNEEHashGrid.mReservoirs"]             = make_shared<Buffer>(commandBuffer.mDevice, "gNEEHashGrid.mReservoirs"            , neeHashgridReservoirCount * sizeof(ReservoirData), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
			mCurFrame->mPathData["gNEEHashGrid.mSampleSources"]          = make_shared<Buffer>(commandBuffer.mDevice, "gNEEHashGrid.mSampleSources"         , neeHashgridReservoirCount * sizeof(PathVertex), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
			mCurFrame->mPathData["gNEEHashGrid.mReservoirSamples"]       = make_shared<Buffer>(commandBuffer.mDevice, "gNEEHashGrid.mReservoirSamples"      , neeHashgridReservoirCount * sizeof(PresampledLightPoint), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
			mCurFrame->mPathData["gNEEHashGrid.mAppendIndices"]          = make_shared<Buffer>(commandBuffer.mDevice, "gNEEHashGrid.mAppendIndices"         , neeHashgridReservoirCount * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal);
			mCurFrame->mPathData["gNEEHashGrid.mAppendReservoirs"]       = make_shared<Buffer>(commandBuffer.mDevice, "gNEEHashGrid.mAppendReservoirs"      , neeHashgridReservoirCount * sizeof(ReservoirData), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
			mCurFrame->mPathData["gNEEHashGrid.mAppendSampleSources"]    = make_shared<Buffer>(commandBuffer.mDevice, "gNEEHashGrid.mAppendSampleSources"   , neeHashgridReservoirCount * sizeof(PathVertex), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
			mCurFrame->mPathData["gNEEHashGrid.mAppendReservoirSamples"] = make_shared<Buffer>(commandBuffer.mDevice, "gNEEHashGrid.mAppendReservoirSamples", neeHashgridReservoirCount * sizeof(PresampledLightPoint), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
		}

		const uint32_t lvcHashgridBucketCount = BDPT_CHECK_FLAG(samplingFlags, BDPTFlagBits::eLVCReservoirReuse) ? max(1u,push_constants.gHashGridBucketCount) : 1;
		if (!mCurFrame->mPathData["gLVCHashGrid.mChecksums"] || mCurFrame->mPathData.at("gLVCHashGrid.mChecksums").sizeBytes() < lvcHashgridBucketCount * sizeof(uint32_t)) {
			mCurFrame->mPathData["gLVCHashGrid.mChecksums"] = make_shared<Buffer>(commandBuffer.mDevice, "gLVCHashGrid.mChecksums", lvcHashgridBucketCount * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal);
			mCurFrame->mPathData["gLVCHashGrid.mCounters"]  = make_shared<Buffer>(commandBuffer.mDevice, "gLVCHashGrid.mCounters" , lvcHashgridBucketCount * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal);
			mCurFrame->mPathData["gLVCHashGrid.mIndices"]   = make_shared<Buffer>(commandBuffer.mDevice, "gLVCHashGrid.mIndices"  , lvcHashgridBucketCount * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
			mCurFrame->mPathData["gLVCHashGrid.mStats"]     = make_shared<Buffer>(commandBuffer.mDevice, "gLVCHashGrid.mStats"    , 4 * sizeof(uint32_t)                        , vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		}
		const uint32_t lvcHashgridReservoirCount  = BDPT_CHECK_FLAG(samplingFlags, BDPTFlagBits::eLVCReservoirReuse) ? pixelCount * max(1u,push_constants.gMaxDiffuseVertices) : 1;
		if (!mCurFrame->mPathData["gLVCHashGrid.mReservoirs"] || mCurFrame->mPathData.at("gLVCHashGrid.mReservoirs").sizeBytes() < lvcHashgridReservoirCount * sizeof(ReservoirData)) {
			mCurFrame->mPathData["gLVCHashGrid.mReservoirs"]             = make_shared<Buffer>(commandBuffer.mDevice, "gLVCHashGrid.mReservoirs"            , lvcHashgridReservoirCount * sizeof(ReservoirData), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
			mCurFrame->mPathData["gLVCHashGrid.mSampleSources"]          = make_shared<Buffer>(commandBuffer.mDevice, "gLVCHashGrid.mSampleSources"         , lvcHashgridReservoirCount * sizeof(PathVertex), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
			mCurFrame->mPathData["gLVCHashGrid.mReservoirSamples"]       = make_shared<Buffer>(commandBuffer.mDevice, "gLVCHashGrid.mReservoirSamples"      , lvcHashgridReservoirCount * sizeof(PathVertex), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
			mCurFrame->mPathData["gLVCHashGrid.mAppendIndices"]          = make_shared<Buffer>(commandBuffer.mDevice, "gLVCHashGrid.mAppendIndices"         , lvcHashgridReservoirCount * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal);
			mCurFrame->mPathData["gLVCHashGrid.mAppendReservoirs"]       = make_shared<Buffer>(commandBuffer.mDevice, "gLVCHashGrid.mAppendReservoirs"      , lvcHashgridReservoirCount * sizeof(PathVertex), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
			mCurFrame->mPathData["gLVCHashGrid.mAppendSampleSources"]    = make_shared<Buffer>(commandBuffer.mDevice, "gLVCHashGrid.mAppendSampleSources"   , lvcHashgridReservoirCount * sizeof(PathVertex), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
			mCurFrame->mPathData["gLVCHashGrid.mAppendReservoirSamples"] = make_shared<Buffer>(commandBuffer.mDevice, "gLVCHashGrid.mAppendReservoirSamples", lvcHashgridReservoirCount * sizeof(PathVertex), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
		}
	}

	// create descriptorsets
	{
		ProfilerScope ps("Assign descriptors", &commandBuffer);
		Descriptors descriptors;

		descriptors[{ "gSceneParams.gAccelerationStructure", 0u }] = mCurFrame->mSceneData->mAccelerationStructure;
		descriptors[{ "gSceneParams.gVertices", 0u }] = mCurFrame->mSceneData->mVertices;
		descriptors[{ "gSceneParams.gIndices", 0u }] = mCurFrame->mSceneData->mIndices;
		descriptors[{ "gSceneParams.gInstances", 0u }] = mCurFrame->mSceneData->mInstances;
		descriptors[{ "gSceneParams.gInstanceTransforms", 0u }] = mCurFrame->mSceneData->mInstanceTransforms;
		descriptors[{ "gSceneParams.gInstanceInverseTransforms", 0u }] = mCurFrame->mSceneData->mInstanceInverseTransforms;
		descriptors[{ "gSceneParams.gInstanceMotionTransforms", 0u }] = mCurFrame->mSceneData->mInstanceMotionTransforms;
		descriptors[{ "gSceneParams.gMaterialData", 0u }] = mCurFrame->mSceneData->mMaterialData;
		descriptors[{ "gSceneParams.gLightInstances", 0u }] = mCurFrame->mSceneData->mLightInstanceMap;
		descriptors[{ "gSceneParams.gPerformanceCounters", 0u }] = mRayCount;
		for (const auto& [vol, index] : mCurFrame->mSceneData->mMaterialResources.mVolumeDataMap)
			descriptors[{"gSceneParams.gVolumes", index}] = vol;
		for (const auto& [image, index] : mCurFrame->mSceneData->mMaterialResources.mImage4s)
			descriptors[{"gSceneParams.gImages", index}] = ImageDescriptor{ image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		for (const auto& [image, index] : mCurFrame->mSceneData->mMaterialResources.mImage1s)
			descriptors[{"gSceneParams.gImage1s", index}] = ImageDescriptor{ image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };

		descriptors[{"gFrameParams.gViews",0}] = mCurFrame->mViews;
		descriptors[{"gFrameParams.gPrevViews",0}] = (mPrevFrame && mPrevFrame->mViews ? mPrevFrame : mCurFrame)->mViews;
		descriptors[{"gFrameParams.gViewTransforms",0}] = mCurFrame->mViewTransforms;
		descriptors[{"gFrameParams.gInverseViewTransforms",0}] = mCurFrame->mViewInverseTransforms;
		descriptors[{"gFrameParams.gPrevInverseViewTransforms",0}] = (mPrevFrame && mPrevFrame->mViews ? mPrevFrame : mCurFrame)->mViewInverseTransforms;
		descriptors[{"gFrameParams.gViewMediumInstances",0}] = mCurFrame->mViewMediumIndices;
		descriptors[{"gFrameParams.gRadiance",0}]   = ImageDescriptor{ mCurFrame->mRadiance  , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };
		descriptors[{"gFrameParams.gAlbedo",0}]     = ImageDescriptor{ mCurFrame->mAlbedo    , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };
		descriptors[{"gFrameParams.gDebugImage",0}] = ImageDescriptor{ mCurFrame->mDebugImage, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };
		descriptors[{"gFrameParams.gPrevUVs",0}]    = ImageDescriptor{ mCurFrame->mPrevUVs   , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };
		descriptors[{"gFrameParams.gPrevNEEHashGrid.mChecksums", 0}]        = (push_constants.gReservoirSpatialM > 0 ? mPrevFrame : mCurFrame)->mPathData.at("gNEEHashGrid.mChecksums");
		descriptors[{"gFrameParams.gPrevNEEHashGrid.mCounters", 0}]         = (push_constants.gReservoirSpatialM > 0 ? mPrevFrame : mCurFrame)->mPathData.at("gNEEHashGrid.mCounters");
		descriptors[{"gFrameParams.gPrevNEEHashGrid.mIndices", 0}]          = (push_constants.gReservoirSpatialM > 0 ? mPrevFrame : mCurFrame)->mPathData.at("gNEEHashGrid.mIndices");
		descriptors[{"gFrameParams.gPrevNEEHashGrid.mReservoirs", 0}]       = (push_constants.gReservoirSpatialM > 0 ? mPrevFrame : mCurFrame)->mPathData.at("gNEEHashGrid.mReservoirs");
		descriptors[{"gFrameParams.gPrevNEEHashGrid.mSampleSources", 0}]    = (push_constants.gReservoirSpatialM > 0 ? mPrevFrame : mCurFrame)->mPathData.at("gNEEHashGrid.mSampleSources");
		descriptors[{"gFrameParams.gPrevNEEHashGrid.mReservoirSamples", 0}] = (push_constants.gReservoirSpatialM > 0 ? mPrevFrame : mCurFrame)->mPathData.at("gNEEHashGrid.mReservoirSamples");
		descriptors[{"gFrameParams.gPrevNEEHashGrid.mStats", 0}]            = (push_constants.gReservoirSpatialM > 0 ? mPrevFrame : mCurFrame)->mPathData.at("gNEEHashGrid.mStats");
		descriptors[{"gFrameParams.gPrevLVCHashGrid.mChecksums", 0}]        = (push_constants.gReservoirSpatialM > 0 ? mPrevFrame : mCurFrame)->mPathData.at("gLVCHashGrid.mChecksums");
		descriptors[{"gFrameParams.gPrevLVCHashGrid.mCounters", 0}]         = (push_constants.gReservoirSpatialM > 0 ? mPrevFrame : mCurFrame)->mPathData.at("gLVCHashGrid.mCounters");
		descriptors[{"gFrameParams.gPrevLVCHashGrid.mIndices", 0}]          = (push_constants.gReservoirSpatialM > 0 ? mPrevFrame : mCurFrame)->mPathData.at("gLVCHashGrid.mIndices");
		descriptors[{"gFrameParams.gPrevLVCHashGrid.mReservoirs", 0}]       = (push_constants.gReservoirSpatialM > 0 ? mPrevFrame : mCurFrame)->mPathData.at("gLVCHashGrid.mReservoirs");
		descriptors[{"gFrameParams.gPrevLVCHashGrid.mSampleSources", 0}]    = (push_constants.gReservoirSpatialM > 0 ? mPrevFrame : mCurFrame)->mPathData.at("gLVCHashGrid.mSampleSources");
		descriptors[{"gFrameParams.gPrevLVCHashGrid.mReservoirSamples", 0}] = (push_constants.gReservoirSpatialM > 0 ? mPrevFrame : mCurFrame)->mPathData.at("gLVCHashGrid.mReservoirSamples");
		descriptors[{"gFrameParams.gPrevLVCHashGrid.mStats", 0}]            = (push_constants.gReservoirSpatialM > 0 ? mPrevFrame : mCurFrame)->mPathData.at("gLVCHashGrid.mStats");
		for (const auto&[name, buf] : mCurFrame->mPathData)
			descriptors[{"gFrameParams." + name, 0}] = buf;

		mCurFrame->mDescriptors = mRenderPipelines[eSampleVisibility].get(commandBuffer.mDevice, defines)->getDescriptorSets(descriptors);
	}

	auto zeroBuffers = [&](const vector<Buffer::View<byte>>& buffers, const vk::PipelineStageFlags dstStage = vk::PipelineStageFlagBits::eComputeShader, const vk::AccessFlags dstAccess = vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite) {
		for (auto& b : buffers)
			b.fill(commandBuffer, 0);
		Buffer::barriers(commandBuffer, buffers, vk::PipelineStageFlagBits::eTransfer, dstStage, vk::AccessFlagBits::eTransferWrite, dstAccess);
	};

	mCurFrame->mRadiance.barrier(commandBuffer, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderWrite);
	mCurFrame->mAlbedo.barrier(commandBuffer, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderWrite);
	mCurFrame->mPrevUVs.barrier(commandBuffer, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderWrite);
	mCurFrame->mDebugImage.barrier(commandBuffer, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderWrite);

	PushConstants pushConstants = { { "", PushConstantValue(mPushConstants) } };

	// presample lights
	if (push_constants.gMaxPathVertices > 2 && BDPT_CHECK_FLAG(samplingFlags, BDPTFlagBits::ePresampleLights)) {
		ProfilerScope ps("Presample lights", &commandBuffer);
		mRenderPipelines[ePresampleLights].get(commandBuffer.mDevice, presampleDefines)->dispatchTiled(commandBuffer,
			vk::Extent3D(push_constants.gLightPresampleTileSize * push_constants.gLightPresampleTileCount, 1, 1),
			mCurFrame->mDescriptors, {},
			pushConstants);
	}

	// trace light paths
	if ((BDPT_CHECK_FLAG(samplingFlags, BDPTFlagBits::eReferenceBDPT) || BDPT_CHECK_FLAG(samplingFlags, BDPTFlagBits::eConnectToViews) || BDPT_CHECK_FLAG(samplingFlags, BDPTFlagBits::eConnectToLightPaths)) && push_constants.gMaxPathVertices > 2) {
		zeroBuffers({
			mCurFrame->mPathData.at("gLightTraceSamples"),
			mCurFrame->mPathData.at("gLightPathVertices"),
			mCurFrame->mPathData.at("gLightPathVertexCount") });
		ProfilerScope ps("Sample photons", &commandBuffer);
		mRenderPipelines[eSamplePhotons].get(commandBuffer.mDevice, samplePhotonsDefines)->dispatchTiled(commandBuffer,
			vk::Extent3D((push_constants.gLightPathCount + extent.width-1) / extent.width, 1, 1),
			mCurFrame->mDescriptors, {},
			pushConstants);
	}

	// clear gShadowRays
	if (BDPT_CHECK_FLAG(samplingFlags, BDPTFlagBits::eDeferShadowRays))
		zeroBuffers({mCurFrame->mPathData.at("gShadowRays")}, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead);

	// clear nee hash grid
	if (BDPT_CHECK_FLAG(samplingFlags, BDPTFlagBits::eNEEReservoirReuse)) {
		zeroBuffers({
			mCurFrame->mPathData.at("gNEEHashGrid.mCounters"),
			mCurFrame->mPathData.at("gNEEHashGrid.mChecksums"),
			mCurFrame->mPathData.at("gNEEHashGrid.mAppendIndices") });
		memset(mCurFrame->mPathData.at("gNEEHashGrid.mStats").data(), 0, mCurFrame->mPathData.at("gNEEHashGrid.mStats").sizeBytes());
	}

	// clear lvc hash grid
	if (BDPT_CHECK_FLAG(samplingFlags, BDPTFlagBits::eLVCReservoirReuse)) {
		zeroBuffers({
			mCurFrame->mPathData.at("gLVCHashGrid.mCounters"),
			mCurFrame->mPathData.at("gLVCHashGrid.mChecksums"),
			mCurFrame->mPathData.at("gLVCHashGrid.mAppendIndices") });
		memset(mCurFrame->mPathData.at("gLVCHashGrid.mStats").data(), 0, mCurFrame->mPathData.at("gLVCHashGrid.mStats").sizeBytes());
	}

	// trace view paths
	{
		if (BDPT_CHECK_FLAG(samplingFlags, BDPTFlagBits::eConnectToLightPaths))
			mCurFrame->mPathData.at("gLightPathVertices").barrier(commandBuffer, vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);
			mCurFrame->mPathData.at("gLightPathVertexCount").barrier(commandBuffer, vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);
		if (BDPT_CHECK_FLAG(samplingFlags, BDPTFlagBits::ePresampleLights))
			mCurFrame->mPathData.at("gPresampledLights").barrier(commandBuffer, vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);

		// trace visibility
		{

			ProfilerScope ps("Sample visibility", &commandBuffer);
			mRenderPipelines[eSampleVisibility].get(commandBuffer.mDevice, defines)->dispatchTiled(commandBuffer, extent, mCurFrame->mDescriptors, {}, pushConstants);
		}

		mCurFrame->mPrevUVs.barrier(commandBuffer, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead);

		// trace shadow rays
		if (BDPT_CHECK_FLAG(samplingFlags, BDPTFlagBits::eDeferShadowRays)) {
			mCurFrame->mRadiance.barrier(commandBuffer, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
			mCurFrame->mPathData.at("gShadowRays").barrier(commandBuffer, vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);
			ProfilerScope ps("Trace shadow rays", &commandBuffer);
			mRenderPipelines[eTraceShadows].get(commandBuffer.mDevice, defines)->dispatchTiled(commandBuffer, extent, mCurFrame->mDescriptors, {}, pushConstants);
		}

		// store hashgrid
		if (reservoirReuse) {
			{
				Buffer::barriers(commandBuffer, {
					mCurFrame->mPathData.at("gNEEHashGrid.mCounters"),
					mCurFrame->mPathData.at("gLVCHashGrid.mCounters"),
				}, vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);
				ProfilerScope ps("Compute hash grid indices", &commandBuffer);
				mRenderPipelines[eHashGridComputeIndices].get(commandBuffer.mDevice, defines)->dispatchTiled(commandBuffer,
					vk::Extent3D(extent.width, (push_constants.gHashGridBucketCount + extent.width-1) / extent.width, 1),
					mCurFrame->mDescriptors, {}, pushConstants);
			}
			{
				Buffer::barriers(commandBuffer, {
					mCurFrame->mPathData.at("gNEEHashGrid.mIndices"),
					mCurFrame->mPathData.at("gNEEHashGrid.mAppendReservoirs"),
					mCurFrame->mPathData.at("gNEEHashGrid.mAppendSampleSources"),
					mCurFrame->mPathData.at("gNEEHashGrid.mAppendReservoirSamples"),
					mCurFrame->mPathData.at("gNEEHashGrid.mAppendIndices"),
					mCurFrame->mPathData.at("gLVCHashGrid.mIndices"),
					mCurFrame->mPathData.at("gLVCHashGrid.mAppendReservoirs"),
					mCurFrame->mPathData.at("gLVCHashGrid.mAppendSampleSources"),
					mCurFrame->mPathData.at("gLVCHashGrid.mAppendReservoirSamples"),
					mCurFrame->mPathData.at("gLVCHashGrid.mAppendIndices")
				}, vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);
				ProfilerScope ps("Swizzle hash grid", &commandBuffer);
				mRenderPipelines[eHashGridSwizzle].get(commandBuffer.mDevice, defines)->dispatchTiled(commandBuffer, vk::Extent3D(extent.width, extent.height * push_constants.gMaxDiffuseVertices, 1),
					mCurFrame->mDescriptors, {}, pushConstants);
			}
		}
	}

	// connect subpaths
	if (BDPT_CHECK_FLAG(samplingFlags, BDPTFlagBits::eReferenceBDPT)) {
		Buffer::barriers(commandBuffer, {
			mCurFrame->mPathData.at("gDepth"),
			mCurFrame->mPathData.at("gLightPathVertices"),
			mCurFrame->mPathData.at("gViewPathVertices"),
			mCurFrame->mPathData.at("gViewPathLengths"),
			mCurFrame->mPathData.at("gLightPathLengths")
		}, vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);
		mCurFrame->mRadiance.barrier(commandBuffer, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
		ProfilerScope ps("Subpath connections", &commandBuffer);
		mRenderPipelines[eSubpathConnect].get(commandBuffer.mDevice, defines)->dispatchTiled(commandBuffer, extent, mCurFrame->mDescriptors, {}, pushConstants);
	}

	// add light trace
	else if (BDPT_CHECK_FLAG(samplingFlags, BDPTFlagBits::eConnectToViews)) {
		mCurFrame->mRadiance.barrier(commandBuffer, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
		mCurFrame->mPathData.at("gLightTraceSamples").barrier(commandBuffer, vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead);
		ProfilerScope ps("Add light trace", &commandBuffer);
		mRenderPipelines[eAddLightTrace].get(commandBuffer.mDevice, defines)->dispatchTiled(commandBuffer, extent, mCurFrame->mDescriptors, {}, pushConstants);
	}

	Image::View result = (mDebugMode == BDPTDebugMode::eNone) ? mCurFrame->mRadiance : mCurFrame->mDebugImage;

	Defines tonemapDefines;
	tonemapDefines.emplace("gMode", to_string((uint32_t)mTonemapMode));
	if (mGammaCorrect)
		tonemapDefines.emplace("gGammaCorrection", "1");

	// accumulate/denoise
	if (mDenoise && denoiser) {
		if (changed && !reprojection) denoiser->resetAccumulation();
		mCurFrame->mDenoiseResult = denoiser->denoise(
			commandBuffer,
			result,
			mCurFrame->mAlbedo,
			mCurFrame->mViews,
			mCurFrame->mPathData.at("gVisibility").cast<VisibilityInfo>(),
			mCurFrame->mPathData.at("gDepth").cast<DepthInfo>(),
			mCurFrame->mPrevUVs);
		result = mCurFrame->mDenoiseResult;

		if (denoiser->demodulateAlbedo())
			tonemapDefines.emplace("gModulateAlbedo", "1");
	}

	// tone map
	{
		result.barrier(commandBuffer, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead);

		{
			ProfilerScope ps("Tonemap Reduce", &commandBuffer);
			commandBuffer->fillBuffer(**mCurFrame->mTonemapMax.buffer(), mCurFrame->mTonemapMax.offset(), mCurFrame->mTonemapMax.sizeBytes(), 0);
			mCurFrame->mTonemapMax.barrier(commandBuffer, vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);

			mTonemapMaxReducePipeline.get(commandBuffer.mDevice, tonemapDefines)->dispatchTiled(commandBuffer, extent, {
					{ {"gInput", 0}, ImageDescriptor{ result, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead, {}} },
					{ {"gAlbedo", 0}, ImageDescriptor{ mCurFrame->mAlbedo, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead, {}} },
					{ {"gMax", 0}, mCurFrame->mTonemapMax }
				}, {}, {});
		}

		mCurFrame->mAlbedo.barrier(commandBuffer, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead);
		mCurFrame->mTonemapResult.barrier(commandBuffer, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderWrite);
		mCurFrame->mTonemapMax.barrier(commandBuffer, vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite,  vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);

		{
			ProfilerScope ps("Tonemap", &commandBuffer);
			mTonemapPipeline.get(commandBuffer.mDevice, tonemapDefines)->dispatchTiled(commandBuffer, extent, {
				{ { "gInput", 0 }, ImageDescriptor{ result, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead, {} } },
				{ { "gOutput", 0 }, ImageDescriptor{ mCurFrame->mTonemapResult, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} } },
				{ { "gAlbedo", 0 }, ImageDescriptor{ mCurFrame->mAlbedo, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead, {} } },
				{ { "gMax", 0 }, mCurFrame->mTonemapMax },
				{ { "gPrevMax", 0 }, mPrevFrame && mPrevFrame->mTonemapMax ? mPrevFrame->mTonemapMax : mCurFrame->mTonemapMax }
				}, {}, mTonemapPushConstants);
		}
	}

	if (mCurFrame->mTonemapResult.image()->format() == renderTarget.image()->format())
		Image::copy(commandBuffer, mCurFrame->mTonemapResult, renderTarget);
	else
		Image::blit(commandBuffer, mCurFrame->mTonemapResult, renderTarget);

	// copy selection data
	{
		Buffer::View<VisibilityInfo> v = mCurFrame->mPathData.at("gVisibility").cast<VisibilityInfo>();
		v.barrier(commandBuffer, vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead);
		const ImVec2 c = ImGui::GetCursorPos();
		const int2 cp = int2((int)c.x, (int)c.y);
		mCurFrame->mSelectionDataValid = false;
		for (const auto&[view, transform] : views)
			if (view.test_inside(cp)) {
				Buffer::copy(commandBuffer, Buffer::View<VisibilityInfo>(v, cp.y() * view.extent().x() + cp.x(), 1), mCurFrame->mSelectionData);
				mCurFrame->mSelectionDataValid = true;
			}
	}
}

}