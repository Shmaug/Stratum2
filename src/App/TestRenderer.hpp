#pragma once

#include <Core/Pipeline.hpp>
#include <Core/DeviceResourcePool.hpp>

#include "Node.hpp"

namespace stm2 {

class TestRenderer {
public:
	Node& mNode;

	TestRenderer(Node&);

	void createPipelines(Device& device);

	void drawGui();

	void render(CommandBuffer& commandBuffer, const Image::View& renderTarget);

private:
	ComputePipelineCache mPipeline;
	PushConstants mPushConstants;

	bool mRandomPerFrame = true;
	bool mDenoise = true;
	bool mTonemap = true;

	chrono::high_resolution_clock::time_point mLastSceneVersion;

	DeviceResourcePool mResourcePool;
};

};