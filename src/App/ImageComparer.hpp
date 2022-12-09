#pragma once

#include <Core/Pipeline.hpp>

#include <Utils/hlslmath.hpp>
#include <Shaders/compat/image_compare.h>

#include "Node.hpp"
#include "Gui.hpp"

namespace stm2 {

class ImageComparer {
public:
	Node& mNode;

	ImageComparer(Node& node);

	void drawGui();
	void update(CommandBuffer& commandBuffer);

private:
	ComputePipelineCache mPipeline;
	unordered_map<string, Buffer::View<uint32_t>> mCompareResult;
	ImageCompareMode mMode = ImageCompareMode::eSMAPE;
	uint32_t mQuantization = 1024;

	// stores original image object, and a copy
	unordered_map<string, pair<Image::View, Image::View>> mImages;
	unordered_set<string> mComparing;
	string mCurrent;
	float2 mOffset = float2::Zero();
	float mZoom = 1;
};

}