#include "Tonemapper.hpp"
#include "Inspector.hpp"
#include "Gui.hpp"

#include <Core/Instance.hpp>
#include <Core/CommandBuffer.hpp>
#include <Core/Profiler.hpp>

#include <Shaders/compat/tonemap.h>

#include <imgui/imgui.h>

namespace stm2 {

Tonemapper::Tonemapper(Node& node) : mNode(node) {
	if (shared_ptr<Inspector> inspector = mNode.root()->findDescendant<Inspector>())
		inspector->setInspectCallback<Tonemapper>();
	createPipelines(*mNode.findAncestor<Device>());
}

void Tonemapper::createPipelines(Device& device) {
	if (mPushConstants.empty()) {
		mPushConstants["mExposure"] = 0.f;

		if (auto arg = device.mInstance.findArgument("exposure"); arg) mPushConstants["mExposure"] = atof(arg->c_str());
	}

	const filesystem::path shaderPath = *device.mInstance.findArgument("shaderKernelPath");
	mPipeline          = ComputePipelineCache(shaderPath / "tonemap.slang");
	mMaxReducePipeline = ComputePipelineCache(shaderPath / "tonemap.slang", "maxReduce");
}

void Tonemapper::drawGui() {
	ImGui::PushID(this);
	if (ImGui::Button("Reload shaders")) {
		Device& device = *mNode.findAncestor<Device>();
		device->waitIdle();
		createPipelines(device);
	}
	ImGui::PopID();

	Gui::enumDropdown<TonemapMode>("Mode", mMode, (uint32_t)TonemapMode::eTonemapModeCount);
	ImGui::PushItemWidth(40);
	ImGui::DragFloat("Exposure", &mPushConstants["mExposure"].get<float>(), .1f, -10, 10);
	ImGui::PopItemWidth();
	ImGui::Checkbox("Gamma correct", &mGammaCorrect);
}

void Tonemapper::render(CommandBuffer& commandBuffer, const Image::View& input, const Image::View& output, const Image::View& albedo) {
	ProfilerScope ps("Tonemapper::render", &commandBuffer);

	Buffer::View maxBuf = make_shared<Buffer>(commandBuffer.mDevice, "Tonemap max", sizeof(uint4), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst);

	Defines defines;
	defines.emplace("gMode", to_string((uint32_t)mMode));
	if (albedo)        defines.emplace("gModulateAlbedo", "true");
	if (mGammaCorrect) defines.emplace("gGammaCorrection", "true");

	const vk::Extent3D extent = input.extent();

	// get maximum value in image
	{
		ProfilerScope ps("Tonemap reduce", &commandBuffer);

		commandBuffer->fillBuffer(**maxBuf.buffer(), maxBuf.offset(), maxBuf.sizeBytes(), 0);
		maxBuf.barrier(commandBuffer, vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);

		mMaxReducePipeline.get(commandBuffer.mDevice, defines)->dispatchTiled(commandBuffer, extent, {
				{ {"gInput", 0} , ImageDescriptor{ input, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {}} },
				{ {"gAlbedo", 0}, ImageDescriptor{ albedo.image()?albedo:input, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {}} },
				{ {"gMax", 0}, maxBuf }
			}, {}, {});
	}

	maxBuf.barrier(commandBuffer, vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite,  vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);

	// tonemap
	{
		ProfilerScope ps("Tonemap", &commandBuffer);
		mPipeline.get(commandBuffer.mDevice, defines)->dispatchTiled(commandBuffer, extent, {
			{ { "gInput", 0 }, ImageDescriptor{ input, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} } },
			{ { "gOutput", 0 }, ImageDescriptor{ output, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} } },
			{ { "gAlbedo", 0 }, ImageDescriptor{ albedo.image()?albedo:input, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} } },
			{ { "gMax", 0 }, maxBuf },
			}, {}, mPushConstants);
	}
}

}