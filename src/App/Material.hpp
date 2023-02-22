#pragma once

#include "MaterialResources.hpp"

#include <Shaders/compat/scene.h>
#include <Shaders/compat/material_data.h>

namespace stm2 {

struct Material {
	PackedMaterialData mMaterialData;
    array<Image::View,4> mImages;

	Image::View mAlphaMask;
	float mAlphaCutoff = 0.5f;
	Buffer::View<uint32_t> mMinAlpha;
	Image::View mBumpImage;
	float mBumpStrength = 1;

	bool alphaTest() { return (!mMinAlpha || mMinAlpha.buffer()->inFlight()) ? false : mMinAlpha[0] < 255; }

    inline void store(MaterialResources& resources) const {
		resources.mMaterialData.AppendN(mMaterialData.mPackedData);

		for (const Image::View& img : mImages)
			resources.mMaterialData.Append(resources.getIndex(img));

		resources.mMaterialData.Appendf(mMaterialData.mEmissionScale);
		resources.mMaterialData.Append(resources.getIndex(mBumpImage));
		resources.mMaterialData.Appendf(mBumpStrength);
		resources.mMaterialData.Append(mBumpImage ? channelCount(mBumpImage.image()->format()) : 0);

		resources.mMaterialData.Append(resources.getIndex(mAlphaMask));
		resources.mMaterialData.Appendf(mAlphaCutoff);
		// pad to 64 bytes
		resources.mMaterialData.Append(0);
		resources.mMaterialData.Append(0);
    }

    void drawGui(Node& node);
};

struct Medium {
	float3 mDensityScale;
	float3 mAlbedoScale;
	float mAnisotropy;
	shared_ptr<nanovdb::GridHandle<nanovdb::HostBuffer>> mDensityGrid, mAlbedoGrid;
	Buffer::View<byte> mDensityBuffer, mAlbedoBuffer;

	inline void store(MaterialResources& resources) const {
		auto D3DX_FLOAT4_to_R8G8B8A8_UNORM = [](const float3 unpackedInput) -> uint32_t {
			auto D3DX_FLOAT_to_UINT = [](float _V, float _Scale) {
				return (uint32_t)floor(_V * _Scale + 0.5f);
			};
			uint32_t packedOutput;
			packedOutput = ((D3DX_FLOAT_to_UINT(saturate(unpackedInput[0]), 255))     |
							(D3DX_FLOAT_to_UINT(saturate(unpackedInput[1]), 255)<< 8) |
							(D3DX_FLOAT_to_UINT(saturate(unpackedInput[2]), 255)<<16) |
							(D3DX_FLOAT_to_UINT(saturate(             1.f), 255)<<24) );
			return packedOutput;
		};

		resources.mMaterialData.AppendN(mDensityScale);
		resources.mMaterialData.Append(D3DX_FLOAT4_to_R8G8B8A8_UNORM(mAlbedoScale));
		resources.mMaterialData.Append(resources.getIndex(mDensityBuffer));
		resources.mMaterialData.Append(resources.getIndex(mAlbedoBuffer));
		resources.mMaterialData.Appendf(mAnisotropy);
	}

	void drawGui(Node& node);
};

}