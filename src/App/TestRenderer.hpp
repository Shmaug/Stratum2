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
	shared_ptr<vk::raii::Sampler> mStaticSampler;
	unordered_map<string, ComputePipelineCache> mPipelines;
	PushConstants mPushConstants;

	uint32_t mHashGridCellCount = 100000;
	float mHashGridCellSize = 0.1f;
	float mHashGridCellPixelRadius = 0;
	float mLightSubpathCount = 1;
	bool mLightTrace = false;

	unordered_map<string, bool> mDefines;


	bool mRandomPerFrame = true;

	bool mDenoise = true;
	bool mTonemap = true;

	chrono::high_resolution_clock::time_point mLastSceneVersion;

	DeviceResourcePool mResourcePool;
	list<pair<Buffer::View<byte>, bool>> mSelectionData;
	vector<TransformData> mPrevViewTransforms;
};

};