#pragma once

#include <Core/Pipeline.hpp>

#include <Core/math.hpp>
#include <Shaders/compat/image_compare.h>

#include "Node.hpp"
#include "Gui.hpp"

namespace stm2 {

class ImageComparer {
public:
	Node& mNode;

	ImageComparer(Node& node);

	void drawGui();
	void postRender(CommandBuffer& commandBuffer, const Image::View& renderTarget);

private:
	Buffer::View<uint2> compare(CommandBuffer& commandBuffer, const Image::View& img0, const Image::View& img1);

	ComputePipelineCache mPipeline;
	ImageCompareMode mMode = ImageCompareMode::eMSE;
	float mQuantization = 16384;

	Image::View mReferenceImage;
	vector<tuple<string, Image::View, Buffer::View<uint2>>> mImages;
	uint32_t mSelected = -1;

	vector<char> mStoreLabel;
	bool mStore = false;
	optional<uint32_t> mStoreFrame;

	float2 mOffset = float2::Zero();
	float mZoom = 1;
};

}