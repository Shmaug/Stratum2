#pragma once

#include "Scene.hpp"

#include <Shaders/compat/scene.h>

namespace stm2 {

class RasterRenderer {
public:
	Node& mNode;

	RasterRenderer(Node& node);

	void createPipelines(Device& device, const vk::Format renderFormat);

	void drawGui();
	void render(CommandBuffer& commandBuffer, const Image::View& renderTarget);

private:
	GraphicsPipelineCache mRasterPipeline;

	bool mAlphaMasks = true;
	bool mTonemap = true;

	DeviceResourcePool mResourcePool;
	list<pair<Buffer::View<byte>, bool>> mSelectionData;
};

}