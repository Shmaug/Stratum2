#include "TestRenderer.hpp"
#include "Inspector.hpp"
#include "Scene.hpp"

#include <Core/Instance.hpp>
#include <Core/Profiler.hpp>

#include <imgui/imgui.h>

namespace stm2 {

TestRenderer::TestRenderer(Node& node) : mNode(node) {
	if (shared_ptr<Inspector> inspector = mNode.root()->findDescendant<Inspector>())
		inspector->setTypeCallback<TestRenderer>();

	Device& device = *mNode.findAncestor<Device>();
	createPipelines(device);
}

void TestRenderer::createPipelines(Device& device) {
	const vector<string>& args = { "-matrix-layout-row-major", "-capability", "spirv_1_5", "-Wno-30081" };
	const filesystem::path shaderPath = *device.mInstance.findArgument("shaderKernelPath");
	mPipeline = ComputePipelineCache(shaderPath / "testrenderer.slang", "render", "sm_6_6", args);
}

void TestRenderer::drawGui() {
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

	if (ImGui::CollapsingHeader("Configuration")) {
		ImGui::Checkbox("Random frame seed", &mRandomPerFrame);
	}

	if (ImGui::CollapsingHeader("Path tracing")) {
		ImGui::PushItemWidth(60);
		ImGui::DragScalar("Max bounces", ImGuiDataType_U32, &mPushConstants["mMaxBounces"].get<uint32_t>());
		ImGui::DragScalar("Min bounces", ImGuiDataType_U32, &mPushConstants["mMinBounces"].get<uint32_t>());
		ImGui::DragScalar("Max diffuse bounces", ImGuiDataType_U32, &mPushConstants["mMaxDiffuseBounces"].get<uint32_t>());
		ImGui::PopItemWidth();
	}
}

void TestRenderer::render(CommandBuffer& commandBuffer, const Image::View& renderTarget) {
	ProfilerScope ps("PathTracer::render", &commandBuffer);

	auto frame = mFrameResourcePool.get();
	if (!frame)
		frame = mFrameResourcePool.emplace(make_shared<FrameResources>(commandBuffer.mDevice));
	commandBuffer.trackResource(frame);

	const vk::Extent3D extent = renderTarget.extent();

	frame->getImage("mOutput", extent, vk::Format::eR32G32B32A32Sfloat, vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eTransferSrc);

	// upload views
	{
		ProfilerScope ps("Upload views", &commandBuffer);

		vector<pair<ViewData, TransformData>> views;
		mNode.root()->forEachDescendant<Camera>([&](Node& node, const shared_ptr<Camera>& camera) {
			views.emplace_back(pair{ camera->view(), nodeToWorld(node) });
		});
		mPushConstants["mViewCount"] = (uint32_t)views.size();

		if (views.empty()) {
			renderTarget.clearColor(commandBuffer, vk::ClearColorValue(array<uint32_t, 4>{ 0, 0, 0, 0 }));
			cout << "Warning: No views" << endl;
			return;
		}

		// upload viewdata
		auto viewsBuffer          = frame->getBuffer<ViewData>     ("mViews"         , views.size(), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		auto viewTransformsBuffer = frame->getBuffer<TransformData>("mViewTransforms", views.size(), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		for (uint32_t i = 0; i < views.size(); i++) {
			viewsBuffer[i] = views[i].first;
			viewTransformsBuffer[i] = views[i].second;
		}
	}

	mPushConstants["mRandomSeed"] = (uint32_t)rand();

	Defines defines;

	auto pipeline = mPipeline.get(commandBuffer.mDevice, defines);

	// create descriptorsets
	{
		ProfilerScope ps("Assign descriptors", &commandBuffer);

		Descriptors descriptors;
		for (const auto&[name, buffer] : frame->mBuffers)
			descriptors[{ "gRenderParams." + name, 0 }] = buffer;
		for (const auto&[name, image] : frame->mImages)
			descriptors[{ "gRenderParams." + name, 0 }] = image;

		frame->mDescriptors = pipeline->getDescriptorSets(descriptors);
	}

	// render
	{
		ProfilerScope ps("Sample visibility", &commandBuffer);
		pipeline->dispatchTiled(commandBuffer, extent, frame->mDescriptors, {}, mPushConstants);
	}

	const Image::View& result = get<Image::View>(frame->mImages["mOutput"]);
	// copy render result to rendertarget
	if (result.image()->format() == renderTarget.image()->format())
		Image::copy(commandBuffer, result, renderTarget);
	else
		Image::blit(commandBuffer, result, renderTarget);

	mPrevFrame = frame;
}

}