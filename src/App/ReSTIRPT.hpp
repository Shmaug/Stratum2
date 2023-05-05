#pragma once

#include "GpuHashGrid.hpp"
#include "Node.hpp"

#include <Shaders/compat/transform.h>

namespace stm2 {

class ReSTIRPT {
public:
	Node& mNode;

	ReSTIRPT(Node&);

	void createPipelines(Device& device);

	void drawGui();

	void render(CommandBuffer& commandBuffer, const Image::View& renderTarget);

private:
	shared_ptr<vk::raii::Sampler> mStaticSampler;
	unordered_map<string, ComputePipelineCache> mPipelines;
	PushConstants mPushConstants;
	PushConstants mRasterPushConstants;
	unordered_map<string, bool> mDefines;

	bool mFixSeed;
	bool mDenoise;
	bool mTonemap;
	bool mShowRcVertices;
	GraphicsPipelineCache mRasterPipeline;

	chrono::high_resolution_clock::time_point mLastSceneVersion;

	DeviceResourcePool mResourcePool;
	list<pair<Buffer::View<byte>, bool>> mSelectionData;
	list<pair<Buffer::View<byte>, shared_ptr<Shader>>> mGpuPrintData;
	vector<TransformData> mPrevViewTransforms;
};

};