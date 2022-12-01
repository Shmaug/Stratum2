#ifndef SCENE_H
#define SCENE_H

#include "bitfield.h"
#include "transform.h"

STM_NAMESPACE_BEGIN

#define INSTANCE_TYPE_TRIANGLES 0
#define INSTANCE_TYPE_SPHERE 1
#define INSTANCE_TYPE_VOLUME 2

#define BVH_FLAG_NONE 0
#define BVH_FLAG_TRIANGLES BIT(0)
#define BVH_FLAG_SPHERES BIT(1)
#define BVH_FLAG_VOLUME BIT(2)
#define BVH_FLAG_EMITTER BIT(3)

#define INVALID_INSTANCE 0xFFFF
#define INVALID_PRIMITIVE 0xFFFF

#define gImageCount 1024
#define gVolumeCount 8

struct InstanceData {
	uint4 packed;

	inline uint type() CONST_CPP              { return BF_GET(packed[0], 0, 4); }
	inline uint materialAddress() CONST_CPP   { return BF_GET(packed[0], 4, 28); }
	inline uint lightIndex() CONST_CPP        { return BF_GET(packed[1], 0, 12); }

	inline static TransformData makeMotionTransform(const TransformData worldToObject, const TransformData prevObjectToWorld) {
		return tmul(prevObjectToWorld, worldToObject);
	}

#ifdef __cplusplus
	inline InstanceData(const uint type, const uint materialAddress) {
		packed = 0;
		BF_SET(packed[0], type, 0, 4);
		BF_SET(packed[0], materialAddress, 4, 28);
		BF_SET(packed[1], -1, 0, 12);
	}
#endif
};

struct TrianglesInstanceData : InstanceData {
	inline uint primitiveCount() CONST_CPP    { return BF_GET(packed[1], 12, 16); }
	inline uint indexStride() CONST_CPP       { return BF_GET(packed[1], 28, 4); }
	inline uint firstVertex() CONST_CPP       { return packed[2]; }
	inline uint indicesByteOffset() CONST_CPP { return packed[3]; }

#ifdef __cplusplus
	inline TrianglesInstanceData(const uint materialAddress, const uint primCount, const uint firstVertex, const uint indexByteOffset, const uint indexStride)
		: InstanceData(INSTANCE_TYPE_TRIANGLES, materialAddress) {
		BF_SET(packed[1], primCount, 12, 16);
		BF_SET(packed[1], indexStride, 28, 4);
		packed[2] = firstVertex;
		packed[3] = indexByteOffset;
	}
#endif
};

struct SphereInstanceData : InstanceData {
	inline float radius() CONST_CPP { return asfloat(packed[2]); }

#ifdef __cplusplus
	inline SphereInstanceData(const uint materialAddress, const float radius)
		: InstanceData(INSTANCE_TYPE_SPHERE, materialAddress) {
		packed[2] = asuint(radius);
	}
#endif
};

struct VolumeInstanceData : InstanceData {
	inline uint volumeIndex() CONST_CPP { return packed[2]; }

#ifdef __cplusplus
	inline VolumeInstanceData(const uint materialAddress, const uint volumeIndex)
		: InstanceData(INSTANCE_TYPE_VOLUME, materialAddress) {
		packed[2] = volumeIndex;
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

struct PackedVertexData {
	float3 mPosition;
	float mTexcoord0;
	float3 mNormal;
	float mTexcoord1;

	inline float2 uv() { return float2(mTexcoord0, mTexcoord1); }

	SLANG_CTOR(PackedVertexData) (const float3 p, const float3 n, const float2 uv) {
		mPosition = p;
		mTexcoord0 = uv[0];
		mNormal = n;
		mTexcoord1= uv[1];
	}
};


#define SHADING_FLAG_FLIP_BITANGENT BIT(0)

// 48 bytes
struct ShadingData {
	float3 mPosition;
	uint mFlagsMaterialAddress;

	uint mPackedGeometryNormal;
	uint mPackedShadingNormal;
	uint mPackedTangent;
	float mShapeArea;

	float2 mTexcoord;
	float mTexcoordScreenSize;
	float mMeanCurvature;

#ifdef __HLSL__
	property bool isSurface    { get { return mShapeArea > 0; } }
	property bool isBackground { get { return mShapeArea < 0; } }
	property bool isMedium     { get { return mShapeArea == 0; } }
	property uint materialAddress { get { return BF_GET(mFlagsMaterialAddress, 4, 28); } }

	property bool isBitangentFlipped { get { return (bool)(mFlagsMaterialAddress & SHADING_FLAG_FLIP_BITANGENT); } }
	property int bitangentDirection  { get { return isBitangentFlipped ? -1 : 1; } }

	property float3 geometryNormal   { get { return unpackNormal(mPackedGeometryNormal); } }
	property float3 shadingNormal    { get { return unpackNormal(mPackedShadingNormal); } }
	property float3 tangent          { get { return unpackNormal(mPackedTangent); } }

	float3 toWorld(const float3 v) {
		const float3 n = shadingNormal;
		const float3 t = tangent;
		return v.x*t + v.y*cross(n, t)*bitangentDirection + v.z*n;
	}
	float3 toLocal(const float3 v) {
		const float3 n = shadingNormal;
		const float3 t = tangent;
		return float3(dot(v, t), dot(v, cross(n, t)*bitangentDirection), dot(v, n));
	}
#endif
};

STM_NAMESPACE_END

#ifdef __HLSL__
#include "../common/scene.hlsli"
#endif // __HLSL__

#endif