#pragma once

#include "MaterialResources.hpp"

#include <Shaders/compat/scene.h>
#include <Shaders/compat/environment.h>
#include <Shaders/compat/disney_data.h>

namespace tinyvkpt {

struct Material {
    ImageValue4 mValues[DISNEY_DATA_N];
	Image::View mAlphaMask;
	Buffer::View<uint32_t> mMinAlpha;
	Image::View mBumpImage;
	float mBumpStrength;

	bool alphaTest() { return (!mMinAlpha || mMinAlpha.buffer()->inFlight()) ? false : mMinAlpha[0] < 0xFFFFFFFF/2; }

	auto baseColor()         { return mValues[0].value.head<3>(); }
	float& emission()        { return mValues[0].value[3]; }
	float& metallic()        { return mValues[1].value[0]; }
	float& roughness()       { return mValues[1].value[1]; }
	float& anisotropic()     { return mValues[1].value[2]; }
	float& subsurface()      { return mValues[1].value[3]; }
	float& clearcoat()       { return mValues[2].value[0]; }
	float& clearcoatGloss()  { return mValues[2].value[1]; }
	float& transmission()    { return mValues[2].value[2]; }
	float& eta()             { return mValues[2].value[3]; }

    inline void store(MaterialResources& resources) const {
		for (int i = 0; i < DISNEY_DATA_N; i++)
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