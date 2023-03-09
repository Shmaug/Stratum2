#pragma once

#include "Scene.hpp"

#include <Shaders/compat/scene.h>
#include <Shaders/compat/vcm.h>

namespace stm2 {

class VCM {
public:
	Node& mNode;

	VCM(Node& node);

	void createPipelines(Device& device);

	void drawGui();
	void render(CommandBuffer& commandBuffer, const Image::View& renderTarget);
	void rasterLightPaths(CommandBuffer& commandBuffer, const Image::View& renderTarget);

	inline Image::View resultImage() const { return mLastResultImage; }

private:
	void createRasterPipeline(Device& device, const vk::Extent2D& extent, const vk::Format format);

	enum RenderPipelineIndex {
		eGenerateLightPaths,
		eGenerateCameraPaths,
		eHashGridComputeIndices,
		eHashGridSwizzle,
		ePipelineCount
	};
	array<ComputePipelineCache, RenderPipelineIndex::ePipelineCount> mRenderPipelines;

	GraphicsPipelineCache mRasterLightPathPipeline;

	VcmPushConstants mPushConstants;

	bool mPauseRendering = false;

	bool mRandomPerFrame = true;
	bool mDenoise = true;
	bool mTonemap = true;

	bool mDebugPaths = false;
	bool mDebugPathWeights = false;

	bool mUseShadingNormals = true;
	bool mUseNormalMaps = true;
	bool mUseAlphaTesting = false;
	bool mUseLightVertexCache = true;
	bool mLVCHashGridSampling = false;

	float mVmRadiusAlpha  = 0.01f;
	float mVmRadiusFactor = 0.025f;

	VcmReservoirFlags mDIReservoirFlags = VcmReservoirFlags::eNone;
	VcmReservoirFlags mLVCReservoirFlags = VcmReservoirFlags::eNone;


	VcmAlgorithmType mAlgorithm = VcmAlgorithmType::kBpt;
	float mLightPathPercent = 1;

	DeviceResourcePool mResourcePool;
	Image::View mLastResultImage;
	chrono::high_resolution_clock::time_point mLastSceneVersion;

	list<pair<Buffer::View<byte>, bool>> mSelectionData;
	Descriptors mPrevDescriptors;
	vector<TransformData> mPrevViewTransforms;


	bool mVisualizeLightPaths = false;
	uint32_t mVisualizeLightPathCount = 128;
	float mVisualizeLightPathRadius = 0.00075f;
	float mVisualizeLightPathLength = 0.05;
	uint32_t mVisualizeSegmentIndex = -1;
};

}