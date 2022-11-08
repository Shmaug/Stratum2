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

    inline void store(ByteAppendBuffer& bytes, MaterialResources& resources) const {
		for (int i = 0; i < DISNEY_DATA_N; i++)
        	mValues[i].store(bytes, resources);
		bytes.Append(resources.getIndex(mAlphaMask));
		bytes.Append(resources.getIndex(mBumpImage));
		bytes.Appendf(mBumpStrength);
    }

    inline void drawGui() {
		ImGui::ColorEdit3("Base Color"        , baseColor().data());
		ImGui::PushItemWidth(80);
		ImGui::DragFloat("Emission"           , &emission());
		ImGui::DragFloat("Metallic"           , &metallic(), 0.1, 0, 1);
		ImGui::DragFloat("Roughness"          , &roughness(), 0.1, 0, 1);
		ImGui::DragFloat("Anisotropic"        , &anisotropic(), 0.1, 0, 1);
		ImGui::DragFloat("Subsurface"         , &subsurface(), 0.1, 0, 1);
		ImGui::DragFloat("Clearcoat"          , &clearcoat(), 0.1, 0, 1);
		ImGui::DragFloat("Clearcoat Gloss"    , &clearcoatGloss(), 0.1, 0, 1);
		ImGui::DragFloat("Transmission"       , &transmission(), 0.1, 0, 1);
		ImGui::DragFloat("Index of Refraction", &eta(), 0.1, 0, 2);
		if (mBumpImage) ImGui::DragFloat("Bump Strength", &mBumpStrength, 0.1, 0, 10);
		ImGui::PopItemWidth();

		const float w = ImGui::CalcItemWidth() - 4;
		for (uint i = 0; i < DISNEY_DATA_N; i++)
			if (mValues[i].image) {
				ImGui::Text(mValues[i].image.image()->resourceName().c_str());
				ImGui::Image(&mValues[i].image, ImVec2(w, w * mValues[i].image.extent().height / (float)mValues[i].image.extent().width));
			}
		if (mAlphaMask) {
			ImGui::Text(mAlphaMask.image()->resourceName().c_str());
			ImGui::Image(&mAlphaMask, ImVec2(w, w * mAlphaMask.extent().height / (float)mAlphaMask.extent().width));
		}
		if (mBumpImage) {
			ImGui::Text(mBumpImage.image()->resourceName().c_str());
			ImGui::Image(&mBumpImage, ImVec2(w, w * mBumpImage.extent().height / (float)mBumpImage.extent().width));
		}
    }
};

struct Medium {
	float3 mDensityScale;
	float mAnisotropy;
	float3 mAlbedoScale;
	float mAttenuationUnit;
	shared_ptr<nanovdb::GridHandle<nanovdb::HostBuffer>> mDensityGrid, mAlbedoGrid;
	Buffer::View<byte> mDensityBuffer, mAlbedoBuffer;

	inline void store(ByteAppendBuffer& bytes, MaterialResources& resources) const {
		bytes.AppendN(mDensityScale);
		bytes.Appendf(mAnisotropy);
		bytes.AppendN(mAlbedoScale);
		bytes.Appendf(mAttenuationUnit);
		bytes.Append(resources.getIndex(mDensityBuffer));
		bytes.Append(resources.getIndex(mAlbedoBuffer));
	}
	inline void drawGui() {
		ImGui::ColorEdit3("Density", mDensityScale.data(), ImGuiColorEditFlags_HDR|ImGuiColorEditFlags_Float);
		ImGui::ColorEdit3("Albedo", mAlbedoScale.data(), ImGuiColorEditFlags_Float);
		ImGui::SliderFloat("Anisotropy", &mAnisotropy, -.999f, .999f);
		ImGui::SliderFloat("Attenuation Unit", &mAttenuationUnit, 0, 1);
	}
};

}