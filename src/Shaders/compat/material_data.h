#pragma once

#ifdef __cplusplus
#include <Core/math.hpp>
#endif

#include "bitfield.h"
#include "common.h"

STM_NAMESPACE_BEGIN

struct PackedMaterialData {
    // The data is packed as a series of 8 bit quantities:
    // { baseColor.r, baseColor.g   , baseColor.b , padding    }
    // { emission.r , emission.g    , emission.b  , padding    }
	// { metallic   , roughness     , anisotropic , subsurface }
	// { clearcoat  , clearcoatGloss, transmission, eta        }
	// Note: image indices, alpha mask, and normal map data are
	// stored sequentially after this struct in the material data buffer (see src/App/Material.hpp)
    uint4 mPackedData;
    float mEmissionScale;

	float3 getBaseColor() {
        return float3(
			BF_GET(mPackedData[0],  0, 8) / float(255),
			BF_GET(mPackedData[0],  8, 8) / float(255),
			BF_GET(mPackedData[0], 16, 8) / float(255) );
    }
    float3 getEmission() {
        return float3(
            BF_GET(mPackedData[1],  0, 8) / float(255),
            BF_GET(mPackedData[1],  8, 8) / float(255),
            BF_GET(mPackedData[1], 16, 8) / float(255)) * mEmissionScale;
    }
	float getMetallic()       { return BF_GET(mPackedData[2],  0, 8) / float(255); }
	float getRoughness()      { return BF_GET(mPackedData[2],  8, 8) / float(255); }
	float getAnisotropic()    { return BF_GET(mPackedData[2], 16, 8) / float(255); }
	float getSubsurface()     { return BF_GET(mPackedData[2], 24, 8) / float(255); }
	float getClearcoat()      { return BF_GET(mPackedData[3],  0, 8) / float(255); }
	float getClearcoatGloss() { return BF_GET(mPackedData[3],  8, 8) / float(255); }
	float getTransmission()   { return BF_GET(mPackedData[3], 16, 8) / float(255); }
	float getEta()            { return BF_GET(mPackedData[3], 24, 8) / float(255); }


    SLANG_MUTATING
    void setBaseColor(const float3 newValue) {
		BF_SET(mPackedData[0], (uint)floor(saturate(newValue[0])*255 + 0.5f),  0, 8);
		BF_SET(mPackedData[0], (uint)floor(saturate(newValue[1])*255 + 0.5f),  8, 8);
		BF_SET(mPackedData[0], (uint)floor(saturate(newValue[2])*255 + 0.5f), 16, 8);
    }
    SLANG_MUTATING
    void setEmission(float3 newValue) {
        mEmissionScale = max3(newValue);
        newValue /= mEmissionScale;
		BF_SET(mPackedData[1], (uint)floor(saturate(newValue[0])*255 + 0.5f),  0, 8);
		BF_SET(mPackedData[1], (uint)floor(saturate(newValue[1])*255 + 0.5f),  8, 8);
        BF_SET(mPackedData[1], (uint)floor(saturate(newValue[2])*255 + 0.5f), 16, 8);
    }
	SLANG_MUTATING void setMetallic      (const float newValue) { BF_SET(mPackedData[2], (uint)floor(saturate(newValue)*255 + 0.5f),  0, 8); }
	SLANG_MUTATING void setRoughness     (const float newValue) { BF_SET(mPackedData[2], (uint)floor(saturate(newValue)*255 + 0.5f),  8, 8); }
	SLANG_MUTATING void setAnisotropic   (const float newValue) { BF_SET(mPackedData[2], (uint)floor(saturate(newValue)*255 + 0.5f), 16, 8); }
	SLANG_MUTATING void setSubsurface    (const float newValue) { BF_SET(mPackedData[2], (uint)floor(saturate(newValue)*255 + 0.5f), 24, 8); }
	SLANG_MUTATING void setClearcoat     (const float newValue) { BF_SET(mPackedData[3], (uint)floor(saturate(newValue)*255 + 0.5f),  0, 8); }
	SLANG_MUTATING void setClearcoatGloss(const float newValue) { BF_SET(mPackedData[3], (uint)floor(saturate(newValue)*255 + 0.5f),  8, 8); }
	SLANG_MUTATING void setTransmission  (const float newValue) { BF_SET(mPackedData[3], (uint)floor(saturate(newValue)*255 + 0.5f), 16, 8); }
	SLANG_MUTATING void setEta           (const float newValue) { BF_SET(mPackedData[3], (uint)floor(saturate(newValue)*255 + 0.5f), 24, 8); }
};

STM_NAMESPACE_END