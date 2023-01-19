#pragma once

#include "MaterialResources.hpp"

#include <Shaders/compat/scene.h>
#include <Shaders/compat/environment_image.h>
#include <Shaders/compat/material.h>

namespace stm2 {

struct Material {
    ImageValue<4> mValues[MaterialData::gDataCount];
	Image::View mAlphaMask;
	Buffer::View<uint32_t> mMinAlpha;
	Image::View mBumpImage;
	float mBumpStrength;

	bool alphaTest() { return (!mMinAlpha || mMinAlpha.buffer()->inFlight()) ? false : mMinAlpha[0] < 0xFFFFFFFF/2; }

	auto baseColor()         { return mValues[0].mValue.head<3>(); }
	float& emission()        { return mValues[0].mValue[3]; }
	float& metallic()        { return mValues[1].mValue[0]; }
	float& roughness()       { return mValues[1].mValue[1]; }
	float& anisotropic()     { return mValues[1].mValue[2]; }
	float& subsurface()      { return mValues[1].mValue[3]; }
	float& clearcoat()       { return mValues[2].mValue[0]; }
	float& clearcoatGloss()  { return mValues[2].mValue[1]; }
	float& transmission()    { return mValues[2].mValue[2]; }
	float& eta()             { return mValues[2].mValue[3]; }

    inline void store(MaterialResources& resources) const {
		for (uint32_t i = 0; i < MaterialData::gDataCount; i++)
        	mValues[i].store(resources);
		resources.mMaterialData.Append(resources.getIndex(mAlphaMask));
		resources.mMaterialData.Append(resources.getIndex(mBumpImage));
		resources.mMaterialData.Appendf(mBumpStrength);
    }

    void drawGui(Node& node);
};

struct Medium {
	float3 mDensityScale;
	float mAnisotropy;
	float3 mAlbedoScale;
	float mAttenuationUnit;
	shared_ptr<nanovdb::GridHandle<nanovdb::HostBuffer>> mDensityGrid, mAlbedoGrid;
	Buffer::View<byte> mDensityBuffer, mAlbedoBuffer;

	inline void store(MaterialResources& resources) const {
		resources.mMaterialData.AppendN(mDensityScale);
		resources.mMaterialData.Appendf(mAnisotropy);
		resources.mMaterialData.AppendN(mAlbedoScale);
		resources.mMaterialData.Appendf(mAttenuationUnit);
		resources.mMaterialData.Append(resources.getIndex(mDensityBuffer));
		resources.mMaterialData.Append(resources.getIndex(mAlbedoBuffer));
	}

	void drawGui(Node& node);
};

}