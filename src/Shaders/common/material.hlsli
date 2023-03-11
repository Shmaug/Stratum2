#pragma once

// used by material conversion kernels
#ifndef MATERIAL_STORE_FN

#ifndef gNormalMaps
#define gNormalMaps false
#endif

#include "shading_data.hlsli"
#include "D3DX_DXGIFormatConvert.inl"

extension SceneParameters {
    // see Material::store() in Material.hpp
    PackedMaterialData LoadMaterial(const uint address, out uint4 imageIndices) {
        PackedMaterialData m;
        m.mPackedData    = mMaterialData.Load<uint4>(int(address));
        imageIndices     = mMaterialData.Load<uint4>(int(address + 16));
        m.mEmissionScale = mMaterialData.Load<float>(int(address + 32));
        return m;
    }
    PackedMaterialData LoadMaterial(const uint address, const float2 uv, const float uvScreenSize) {
        uint4 imageIndices;
        PackedMaterialData m = LoadMaterial(address, imageIndices);

        if (imageIndices.x < gImageCount) {
            m.setBaseColor(m.getBaseColor() * SampleImage4(imageIndices.x, uv, uvScreenSize).rgb);
        }
        if (imageIndices.y < gImageCount) {
            m.setEmission(m.getEmission()   * SampleImage4(imageIndices.y, uv, uvScreenSize).rgb);
        }
        if (imageIndices.z < gImageCount) {
            float4 v = D3DX_R8G8B8A8_UNORM_to_FLOAT4(m.mPackedData[2]);
            const float4 vi = SampleImage4(imageIndices.z, uv, uvScreenSize);
            v.xzw *= vi.xzw;
			// convert "roughness" to "smoothness" before modulation, so that materials can be made rougher
            v.y = 1 - (1 - v.y) * (1 - vi.y);
            m.mPackedData[2] = D3DX_FLOAT4_to_R8G8B8A8_UNORM(v);
        }
        if (imageIndices.w < gImageCount) {
            m.mPackedData[3] = D3DX_FLOAT4_to_R8G8B8A8_UNORM(D3DX_R8G8B8A8_UNORM_to_FLOAT4(m.mPackedData[3]) * SampleImage4(imageIndices.w, uv, uvScreenSize));
        }
        return m;
    }
    PackedMaterialData LoadMaterialUniform(const uint address, const float2 uv) {
        uint4 imageIndices;
        PackedMaterialData m = LoadMaterial(address, imageIndices);

        if (imageIndices.x < gImageCount) {
            m.setBaseColor(m.getBaseColor() * mImages[imageIndices.x].Sample(mStaticSampler, uv).rgb);
        }
        if (imageIndices.y < gImageCount) {
            m.setEmission(m.getEmission()   * mImages[imageIndices.y].Sample(mStaticSampler, uv).rgb);
        }
        if (imageIndices.z < gImageCount) {
            m.mPackedData[2] = D3DX_FLOAT4_to_R8G8B8A8_UNORM(D3DX_R8G8B8A8_UNORM_to_FLOAT4(m.mPackedData[2]) * mImages[imageIndices.z].Sample(mStaticSampler, uv));
        }
        if (imageIndices.w < gImageCount) {
            m.mPackedData[3] = D3DX_FLOAT4_to_R8G8B8A8_UNORM(D3DX_R8G8B8A8_UNORM_to_FLOAT4(m.mPackedData[3]) * mImages[imageIndices.w].Sample(mStaticSampler, uv));
        }
        return m;
    }
    PackedMaterialData LoadMaterial(const ShadingData shadingData) {
        return LoadMaterial(shadingData.getMaterialAddress(), shadingData.mTexcoord, shadingData.mTexcoordScreenSize);
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