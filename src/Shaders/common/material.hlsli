#pragma once

#ifndef MATERIAL_STORE_FN

#ifndef gNormalMaps
#define gNormalMaps false
#endif

#include "shading_data.hlsli"
#include "D3DX_DXGIFormatConvert.inl"

extension SceneParameters {
    // see Material::store() in Material.hpp
    PackedMaterialData LoadMaterial(const ShadingData shadingData) {
		PackedMaterialData m;
		m.mPackedData = mMaterialData.Load<uint4>(int(shadingData.getMaterialAddress()));
		const uint4 indices = mMaterialData.Load<uint4>(int(shadingData.getMaterialAddress() + 16));

		m.mEmissionScale = mMaterialData.Load<float>(int(shadingData.getMaterialAddress() + 32));

		if (indices.x < gImageCount) {
			m.setBaseColor(m.getBaseColor() * SampleImage4(indices.x, shadingData.mTexcoord, shadingData.mTexcoordScreenSize).rgb);
		}
		if (indices.y < gImageCount) {
			m.setEmission(m.getEmission() * SampleImage4(indices.y, shadingData.mTexcoord, shadingData.mTexcoordScreenSize).rgb);
		}
		if (indices.z < gImageCount) {
			m.mPackedData[2] = D3DX_FLOAT4_to_R8G8B8A8_UNORM(D3DX_R8G8B8A8_UNORM_to_FLOAT4(m.mPackedData[2]) * SampleImage4(indices.z, shadingData.mTexcoord, shadingData.mTexcoordScreenSize));
		}
		if (indices.w < gImageCount) {
			m.mPackedData[3] = D3DX_FLOAT4_to_R8G8B8A8_UNORM(D3DX_R8G8B8A8_UNORM_to_FLOAT4(m.mPackedData[3]) * SampleImage4(indices.w, shadingData.mTexcoord, shadingData.mTexcoordScreenSize));
		}
		return m;
    }

    void getMaterialAlphaMask(const uint materialAddress, out uint alphaMask, out float alphaCutoff) {
        const uint2 data = mMaterialData.Load<uint2>(int(materialAddress) + 48);
        alphaMask = data.x;
        alphaCutoff = asfloat(data.y);
    }

    void ApplyNormalMap(inout ShadingData aoShadingData) {
        if (!gNormalMaps) return;

        const uint3 p = gScene.mMaterialData.Load<uint3>(int(aoShadingData.getMaterialAddress() + 36));
        if (p.x >= gImageCount) return;
        const float scale = asfloat(p.y);
        if (scale <= 0) return;

        float3 bump;
        if (p.z == 2) {
            bump.xy = SampleImage2(p.x, aoShadingData.mTexcoord, aoShadingData.mTexcoordScreenSize);
            bump.z = 1;
        } else {
            bump = SampleImage4(p.x, aoShadingData.mTexcoord, aoShadingData.mTexcoordScreenSize).rgb;
        }
        bump.xy = bump.xy * 2 - 1;
		bump.y = -bump.y;
        bump.xy *= scale;

        float3 n = aoShadingData.getShadingNormal();
        float3 t = aoShadingData.getTangent();

        n = normalize(t * bump.x + cross(n, t) * (aoShadingData.isBitangentFlipped() ? -1 : 1) * bump.y + n * bump.z);
        t = normalize(t - n * dot(n, t));

        aoShadingData.mPackedShadingNormal = packNormal(n);
        aoShadingData.mPackedTangent = packNormal(t);
    }
}

#else

#include "compat/material_data.h"

extension PackedMaterialData {
    void Store(RWTexture2D<float4> images[4], const uint2 index) {
        images[0][index] = D3DX_R8G8B8A8_UNORM_to_FLOAT4(mPackedData[0]);
        images[1][index] = D3DX_R8G8B8A8_UNORM_to_FLOAT4(mPackedData[1]);
        images[2][index] = D3DX_R8G8B8A8_UNORM_to_FLOAT4(mPackedData[2]);
        images[3][index] = D3DX_R8G8B8A8_UNORM_to_FLOAT4(mPackedData[3]);
	}
}

#endif