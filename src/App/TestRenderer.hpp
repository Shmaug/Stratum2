#pragma once

#include <Core/DeviceResourcePool.hpp>

#include "Node.hpp"

#include <Shaders/compat/transform.h>

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

	bool mPerformanceCounters = false;
	bool mAlphaTest = true;
	bool mNormalMaps = true;
	bool mShadingNormals = true;
	bool mSampleDirectIllumination = true;
	bool mDebugPaths = false;

	bool mRandomPerFrame = true;

	bool mDenoise = true;
	bool mTonemap = true;

	chrono::high_resolution_clock::time_point mLastSceneVersion;

	DeviceResourcePool mResourcePool;
	list<pair<Buffer::View<byte>, bool>> mSelectionData;
	vector<TransformData> mPrevViewTransforms;
};

};