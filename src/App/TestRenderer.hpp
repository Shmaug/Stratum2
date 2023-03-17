#pragma once

#include "GpuHashGrid.hpp"
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
	shared_ptr<vk::raii::Sampler> mStaticSampler;
	GraphicsPipelineCache mRasterLightPathPipeline;
	unordered_map<string, ComputePipelineCache> mPipelines;
	PushConstants mPushConstants;
	PushConstants mRasterPushConstants;

	float mLightSubpathCount = 1;
	bool mLightTrace = false;

	unordered_map<string, bool> mDefines;

	shared_ptr<vk::raii::Event> mPrevHashGridEvent;

	GpuHashGrid mHashGrid;


	bool mRandomPerFrame = true;

	bool mDenoise = true;
	bool mTonemap = true;

	bool mVisualizeLightPaths = false;

	chrono::high_resolution_clock::time_point mLastSceneVersion;

	DeviceResourcePool mResourcePool;
	list<pair<Buffer::View<byte>, bool>> mSelectionData;
	vector<TransformData> mPrevViewTransforms;
};

};