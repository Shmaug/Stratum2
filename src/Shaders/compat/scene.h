#ifndef SCENE_H
#define SCENE_H

#include "bitfield.h"
#include "transform.h"

STM_NAMESPACE_BEGIN

#define BVH_FLAG_NONE 0
#define BVH_FLAG_TRIANGLES BIT(0)
#define BVH_FLAG_SPHERES BIT(1)
#define BVH_FLAG_VOLUME BIT(2)
#define BVH_FLAG_EMITTER BIT(3)

#define INSTANCE_TYPE_MESH 0
#define INSTANCE_TYPE_SPHERE 1
#define INSTANCE_TYPE_VOLUME 2

#define INVALID_INSTANCE 0xFFFF
#define INVALID_PRIMITIVE 0xFFFF

#define gVertexBufferCount 1024
#define gImageCount 1024
#define gVolumeCount 8

inline static TransformData makeMotionTransform(const TransformData worldToObject, const TransformData prevObjectToWorld) {
	return tmul(prevObjectToWorld, worldToObject);
}

struct InstanceData {
	uint mTypeMaterialAddress;
	uint mData;

	inline uint type() CONST_CPP            { return BF_GET(mTypeMaterialAddress, 0, 4); }
	inline uint materialAddress() CONST_CPP { return BF_GET(mTypeMaterialAddress, 4, 28); }

#ifdef __cplusplus
	inline InstanceData(const uint type, const uint materialAddress) {
		mTypeMaterialAddress = 0;
		BF_SET(mTypeMaterialAddress, type, 0, 4);
		BF_SET(mTypeMaterialAddress, materialAddress, 4, 28);
	}
#endif
};

struct MeshInstanceData : InstanceData {
	inline uint vertexInfoIndex() CONST_CPP { return BF_GET(mData,  0, 16); }
	inline uint primitiveCount() CONST_CPP  { return BF_GET(mData, 16, 16); }

#ifdef __cplusplus
	inline MeshInstanceData(const uint materialAddress, const uint vertexInfoIndex, const uint primitiveCount)
		: InstanceData(INSTANCE_TYPE_MESH, materialAddress) {
		mData = 0;
		BF_SET(mData, vertexInfoIndex,  0, 16);
		BF_SET(mData, primitiveCount , 16, 16);
	}
#endif
};

struct SphereInstanceData : InstanceData {
	inline float radius() CONST_CPP { return asfloat(mData); }

#ifdef __cplusplus
	inline SphereInstanceData(const uint materialAddress, const float radius)
		: InstanceData(INSTANCE_TYPE_SPHERE, materialAddress) {
		mData = asuint(radius);
	}
#endif
};

struct VolumeInstanceData : InstanceData {
	inline uint volumeIndex() CONST_CPP { return mData; }

#ifdef __cplusplus
	inline VolumeInstanceData(const uint materialAddress, const uint volumeIndex)
		: InstanceData(INSTANCE_TYPE_VOLUME, materialAddress) {
		mData = volumeIndex;
	}
#endif
};

struct MeshVertexInfo {
	uint2 mPackedBufferIndices;
	uint mPackedStrides;
	uint pad;
	uint4 mPackedOffsets;

	inline uint indexBuffer()    CONST_CPP { return BF_GET(mPackedBufferIndices[0],  0, 16); }
	inline uint positionBuffer() CONST_CPP { return BF_GET(mPackedBufferIndices[0], 16, 16); }
	inline uint normalBuffer()   CONST_CPP { return BF_GET(mPackedBufferIndices[1],  0, 16); }
	inline uint texcoordBuffer() CONST_CPP { return BF_GET(mPackedBufferIndices[1], 16, 16); }

	inline uint indexOffset()    CONST_CPP { return mPackedOffsets[0]; };
	inline uint positionOffset() CONST_CPP { return mPackedOffsets[1]; };
	inline uint normalOffset()   CONST_CPP { return mPackedOffsets[2]; };
	inline uint texcoordOffset() CONST_CPP { return mPackedOffsets[3]; };

	inline uint indexStride()    CONST_CPP { return BF_GET(mPackedStrides,  0, 8); }
	inline uint positionStride() CONST_CPP { return BF_GET(mPackedStrides,  8, 8); }
	inline uint normalStride()   CONST_CPP { return BF_GET(mPackedStrides, 16, 8); }
	inline uint texcoordStride() CONST_CPP { return BF_GET(mPackedStrides, 24, 8); }

#ifdef __cplusplus
	inline MeshVertexInfo(
		const uint _indexBuffer   , const uint _indexOffset   , const uint _indexStride,
		const uint _positionBuffer, const uint _positionOffset, const uint _positionStride,
		const uint _normalBuffer  , const uint _normalOffset  , const uint _normalStride,
		const uint _texcoordBuffer, const uint _texcoordOffset, const uint _texcoordStride) {
		BF_SET(mPackedBufferIndices[0], _indexBuffer   ,  0, 16);
		BF_SET(mPackedBufferIndices[0], _positionBuffer, 16, 16);
		BF_SET(mPackedBufferIndices[1], _normalBuffer  ,  0, 16);
		BF_SET(mPackedBufferIndices[1], _texcoordBuffer, 16, 16);

		BF_SET(mPackedStrides, _indexStride   ,  0, 8);
		BF_SET(mPackedStrides, _positionStride,  8, 8);
		BF_SET(mPackedStrides, _normalStride  , 16, 8);
		BF_SET(mPackedStrides, _texcoordStride, 24, 8);

		mPackedOffsets[0] = _indexOffset;
		mPackedOffsets[1] = _positionOffset;
		mPackedOffsets[2] = _normalOffset;
		mPackedOffsets[3] = _texcoordOffset;
	}
#endif
};

struct ViewData {
	ProjectionData mProjection;
	int2 mImageMin;
	int2 mImageMax;

	inline int2 extent() CONST_CPP { return mImageMax - mImageMin; }

#ifdef __cplusplus
	inline bool isInside(const int2 p) const { return (p >= mImageMin).all() && (p < mImageMax).all(); }
#endif

#ifdef __HLSL__
	inline bool isInside(const int2 p) { return all(p >= mImageMin) && all(p < mImageMax); }
	inline float imagePlaneDist() { return abs(mImageMax.y - mImageMin.y) / (2 * tan(mProjection.mVerticalFoV/2)); }
	inline float sensorPdfW(const float cosTheta) {
		//return 1 / (cos_theta / pow2(image_plane_dist() / cos_theta));
		return pow2(imagePlaneDist()) / pow3(cosTheta);
	}
	inline float3 toWorld(const float2 pixelCoord, out float2 uv) {
		uv = (pixelCoord - mImageMin)/extent();
		float2 clip_pos = 2*uv - 1;
		clip_pos.y = -clip_pos.y;
		return normalize(mProjection.backProject(clip_pos));
	}
	inline bool toRaster(const float3 pos, out float2 uv) {
		float4 screen_pos = mProjection.projectPoint(pos);
		screen_pos.y = -screen_pos.y;
		screen_pos.xyz /= screen_pos.w;
		if (any(abs(screen_pos.xyz) >= 1) || screen_pos.z <= 0) return false;
		uv = mImageMin + extent() * (screen_pos.xy*.5 + .5);
		return true;
	}
#endif
};

struct VisibilityData {
	uint mInstancePrimitiveIndex;
	uint mPackedNormal;

	inline uint instanceIndex()  { return BF_GET(mInstancePrimitiveIndex, 0, 16); }
	inline uint primitiveIndex() { return BF_GET(mInstancePrimitiveIndex, 16, 16); }
#ifdef __HLSL__
	inline float3 normal()   { return unpackNormal(mPackedNormal); }
#endif
};
struct DepthData {
	float mDepth;
	float mPrevDepth;
	float2 mDepthDerivative;
};

STM_NAMESPACE_END

#ifdef __HLSL__
#include "../common/scene.hlsli"
#endif // __HLSL__

#endif