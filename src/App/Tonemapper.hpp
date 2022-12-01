#pragma once

#include "Node.hpp"
#include <Core/Pipeline.hpp>
#include <Shaders/compat/tonemap.h>

namespace stm2 {

class Tonemapper {
public:
	Node& mNode;

	Tonemapper(Node& node);

	void createPipelines(Device& device);

	void drawGui();
	void render(CommandBuffer& commandBuffer, const Image::View& image, const Image::View& albedo);

private:
	ComputePipelineCache mPipeline;
	ComputePipelineCache mMaxReducePipeline;

	PushConstants mPushConstants;

	bool mGammaCorrect = true;
	TonemapMode mMode = TonemapMode::eRaw;
};

}