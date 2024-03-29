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
	uint32_t mMisType = 0;

	float mRenderScale = 1.f;

	bool mPauseRender = false;
	bool mRenderOnce = false;

	bool mFixSeed = false;
	bool mDenoise = true;
	bool mTonemap = true;
	bool mShowRcVertices = false;
	GraphicsPipelineCache mRasterPipeline;

	array<Image::View, 6> mPrevPathReservoirData;

	chrono::high_resolution_clock::time_point mLastSceneVersion;

	DeviceResourcePool mResourcePool;
	list<pair<Buffer::View<byte>, bool>> mSelectionData;
	vector<TransformData> mPrevViewTransforms;

	struct CounterData {
		Buffer::View<uint32_t> mBuffer;
		vector<uint32_t> mLastValue;
		vector<uint32_t> mCurrentValue;
	};
	CounterData mRayCount, mDebugCounters;
	chrono::high_resolution_clock::time_point mCounterTimer;
	float mCounterDt;
};

};