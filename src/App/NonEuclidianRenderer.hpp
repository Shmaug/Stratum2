#pragma once

#include "Scene.hpp"

#include <Shaders/compat/scene.h>

namespace stm2 {

class NonEuclidianRenderer {
public:
	Node& mNode;

	NonEuclidianRenderer(Node& node);

	void createPipelines(Device& device, const vk::Format renderFormat);

	void drawGui();
	void render(CommandBuffer& commandBuffer, const Image::View& renderTarget);

private:
	GraphicsPipelineCache mRasterPipeline;

	float mLorentzSign = 1;
	float mScaleFactor = 1;

	bool mAlphaMasks = true;
	bool mTonemap = true;

	DeviceResourcePool mResourcePool;
	list<pair<Buffer::View<byte>, bool>> mSelectionData;
};

}