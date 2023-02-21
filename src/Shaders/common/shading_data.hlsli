#pragma once

#ifndef gShadingNormals
#define gShadingNormals false
#endif
#ifndef gAlphaTest
#define gAlphaTest false
#endif
#ifndef gHasMedia
#define gHasMedia false
#endif
#ifndef gMaxNullCollisions
#define gMaxNullCollisions 1000
#endif
#ifndef gPerformanceCounters
#define gPerformanceCounters false
#endif

#include "scene.hlsli"

#define SHADING_FLAG_FLIP_BITANGENT BIT(0)

extension ShadingData {
    bool isSurface() { return mShapeArea > 0; }
	bool isMedium() { return mShapeArea == 0; }
	bool isEnvironment() { return mShapeArea < 0; }
	uint getMaterialAddress() { return BF_GET(mFlagsMaterialAddress, 4, 28); }

	bool isBitangentFlipped() { return (bool)(mFlagsMaterialAddress & SHADING_FLAG_FLIP_BITANGENT); }
	int getBitangentDirection() { return isBitangentFlipped() ? -1 : 1; }

	float3 getGeometryNormal() { return unpackNormal(mPackedGeometryNormal); }
	float3 getShadingNormal() { return unpackNormal(mPackedShadingNormal); }
	float3 getTangent() { return unpackNormal(mPackedTangent); }

	float3 toWorld(const float3 v) {
		const float3 n = getShadingNormal();
		const float3 t = getTangent();
		return v.x * t + v.y * cross(n, t) * getBitangentDirection() + v.z * n;
	}
	float3 toLocal(const float3 v) {
		const float3 n = getShadingNormal();
		const float3 t = getTangent();
		return float3(dot(v, t), dot(v, cross(n, t) * getBitangentDirection()), dot(v, n));
    }

    float shadingNormalCorrection<let Adjoint : bool>(const float3 localDirIn, const float3 localDirOut) {
        if (isMedium())
            return 1;

        const float3 localGeometryNormal = toLocal(getGeometryNormal());
        const float ngdotin = dot(localGeometryNormal, localDirIn);
        const float ngdotout = dot(localGeometryNormal, localDirOut);

        // light leak fix
        if (sign(ngdotout) != sign(localDirOut.z) || sign(ngdotin) != sign(localDirIn.z))
            return 0;

        float G = 1;

        if (Adjoint) {
            const float num = ngdotout * localDirIn.z;
            const float denom = localDirOut.z * ngdotin;
            if (abs(denom) > 1e-5)
                G *= abs(num / denom);
        }

        return G;
    }
};

uint3 loadTriangleIndices(ByteAddressBuffer indices, const uint offset, const uint indexStride, const uint primitiveIndex) {
	const int offsetBytes = (int)(offset + primitiveIndex*3*indexStride);
	uint3 tri;
	if (indexStride == 2) {
		// https://github.com/microsoft/DirectX-Graphics-Samples/blob/master/Samples/Desktop/D3D12Raytracing/src/D3D12RaytracingSimpleLighting/Raytracing.hlsl
		const int dwordAlignedOffset = offsetBytes & ~3;
		const uint2 four16BitIndices = indices.Load2(dwordAlignedOffset);
		if (dwordAlignedOffset == offsetBytes) {
				tri.x = four16BitIndices.x & 0xffff;
				tri.y = (four16BitIndices.x >> 16) & 0xffff;
				tri.z = four16BitIndices.y & 0xffff;
		} else {
				tri.x = (four16BitIndices.x >> 16) & 0xffff;
				tri.y = four16BitIndices.y & 0xffff;
				tri.z = (four16BitIndices.y >> 16) & 0xffff;
		}
	} else
		tri = indices.Load3(offsetBytes);
	return tri;
}
void loadTriangleAttribute<T>(const ByteAddressBuffer vertexBuffer, const uint offset, const uint stride, const uint3 tri, out T v0, out T v1, out T v2) {
	v0 = vertexBuffer.Load<T>(int(offset + stride*tri[0]));
	v1 = vertexBuffer.Load<T>(int(offset + stride*tri[1]));
	v2 = vertexBuffer.Load<T>(int(offset + stride*tri[2]));
}

extension SceneParameters {
	// doesn't assign mPosition
	ShadingData makeTriangleShadingDataWithoutPosition(const uint materialAddress, const TransformData transform, const MeshVertexInfo vertexInfo, const uint3 tri, const float2 bary, const float3 v0, const float3 v1, const float3 v2) {
		ShadingData r;
		r.mFlagsMaterialAddress = 0;
		BF_SET(r.mFlagsMaterialAddress, materialAddress, 4, 28);

		const float3 dPds = transform.transformVector(v0 - v2);
		const float3 dPdt = transform.transformVector(v1 - v2);
		float3 geometryNormal = cross(dPds, dPdt);
		const float area2 = length(geometryNormal);
		geometryNormal /= area2;
		r.mPackedGeometryNormal = packNormal(geometryNormal);
		r.mShapeArea = area2/2;

		float2 t0,t1,t2;
		if (vertexInfo.texcoordBuffer() < gVertexBufferCount)
			loadTriangleAttribute(mVertexBuffers[NonUniformResourceIndex(vertexInfo.texcoordBuffer())], vertexInfo.texcoordOffset(), vertexInfo.texcoordStride(), tri, t0, t1, t2);
        else
            t0 = t1 = t2 = 0;

		const float2 duvds = t2 - t0;
		const float2 duvdt = t2 - t1;

		r.mTexcoord = t0 + (t1 - t0)*bary.x + duvds*bary.y;

		// [du/ds, du/dt]
		// [dv/ds, dv/dt]
		// The inverse of this matrix is
		// (1/det) [ dv/dt, -du/dt]
		//         [-dv/ds,  du/ds]
		// where det = duds * dvdt - dudt * dvds
		const float det = duvds[0] * duvdt[1] - duvdt[0] * duvds[1];
		const float invDet = 1/det;
		const float dsdu =  duvdt[1] * invDet;
		const float dtdu = -duvds[1] * invDet;
		const float dsdv =  duvdt[0] * invDet;
		const float dtdv = -duvds[0] * invDet;
		float3 dPdu,dPdv;
		if (det != 0) {
			// Now we just need to do the matrix multiplication
			dPdu = -(dPds * dsdu + dPdt * dtdu);
			dPdv = -(dPds * dsdv + dPdt * dtdv);
			r.mTexcoordScreenSize = 1 / max(length(dPdu), length(dPdv));
		} else {
			const float3x3 onb = makeOrthonormal(geometryNormal);
			dPdu = onb[0];
			dPdv = onb[1];
			r.mTexcoordScreenSize = 1;
		}

        bool shadingNormalValid = false;
        float3 shadingNormal;
        float3 n0, n1, n2;
        if (gShadingNormals && vertexInfo.normalBuffer() < gVertexBufferCount) {
			loadTriangleAttribute(mVertexBuffers[NonUniformResourceIndex(vertexInfo.normalBuffer())], vertexInfo.normalOffset(), vertexInfo.normalStride(), tri, n0, n1, n2);

			shadingNormal = n0 + (n1 - n0)*bary.x + (n2 - n0)*bary.y;
			shadingNormalValid = !(all(shadingNormal.xyz == 0) || any(isnan(shadingNormal)));
        }

        if (!shadingNormalValid) {
			r.mPackedShadingNormal = r.mPackedGeometryNormal;
			r.mPackedTangent = packNormal(normalize(dPdu));
			r.mMeanCurvature = 0;
		} else {
			shadingNormal = normalize(transform.transformVector(shadingNormal));
			const float3 tangent = normalize(dPdu - shadingNormal*dot(shadingNormal, dPdu));
			r.mPackedShadingNormal = packNormal(shadingNormal);
			r.mPackedTangent = packNormal(tangent);

			// force geometry normal to agree with shading normal
			if (dot(shadingNormal, geometryNormal) < 0)
				r.mPackedGeometryNormal = packNormal(-geometryNormal);

			const float3 dNds = n2 - n0;
			const float3 dNdt = n2 - n1;
			const float3 dNdu = dNds * dsdu + dNdt * dtdu;
			const float3 dNdv = dNds * dsdv + dNdt * dtdv;
			const float3 bitangent = normalize(cross(shadingNormal, tangent));
			r.mMeanCurvature = (dot(dNdu, tangent) + dot(dNdv, bitangent)) / 2;
		}
		return r;
	}

	ShadingData makeTriangleShadingData(const MeshInstanceData instance, const TransformData transform, const uint primitiveIndex, const float2 bary) {
		const MeshVertexInfo vertexInfo = mMeshVertexInfo[instance.vertexInfoIndex()];
		const uint3 tri = loadTriangleIndices(mVertexBuffers[NonUniformResourceIndex(vertexInfo.indexBuffer())], vertexInfo.indexOffset(), vertexInfo.indexStride(), primitiveIndex);

		float3 v0,v1,v2;
		loadTriangleAttribute(mVertexBuffers[NonUniformResourceIndex(vertexInfo.positionBuffer())], vertexInfo.positionOffset(), vertexInfo.positionStride(), tri, v0, v1, v2);

		ShadingData r = makeTriangleShadingDataWithoutPosition(instance.getMaterialAddress(), transform, vertexInfo, tri, bary, v0, v1, v2);
		r.mPosition = transform.transformPoint(v0 + (v1 - v0)*bary.x + (v2 - v0)*bary.y);
		return r;
	}
	ShadingData makeTriangleShadingData(const MeshInstanceData instance, const TransformData transform, const uint primitiveIndex, const float3 localPosition) {
		const MeshVertexInfo vertexInfo = mMeshVertexInfo[instance.vertexInfoIndex()];
		const uint3 tri = loadTriangleIndices(mVertexBuffers[NonUniformResourceIndex(vertexInfo.indexBuffer())], vertexInfo.indexOffset(), vertexInfo.indexStride(), primitiveIndex);

		float3 v0,v1,v2;
		loadTriangleAttribute(mVertexBuffers[NonUniformResourceIndex(vertexInfo.positionBuffer())], vertexInfo.positionOffset(), vertexInfo.positionStride(), tri, v0, v1, v2);

		const float3 v1v0 = v1 - v0;
		const float3 v2v0 = v2 - v0;
		const float3 p_v0 = localPosition - v0;
		const float d00 = dot(v1v0, v1v0);
		const float d01 = dot(v1v0, v2v0);
		const float d11 = dot(v2v0, v2v0);
		const float d20 = dot(p_v0, v1v0);
		const float d21 = dot(p_v0, v2v0);
		const float2 bary = float2(d11 * d20 - d01 * d21, d00 * d21 - d01 * d20) / (d00 * d11 - d01 * d01);

		ShadingData r = makeTriangleShadingDataWithoutPosition(instance.getMaterialAddress(), transform, vertexInfo, tri, bary, v0, v1, v2);
		r.mPosition = transform.transformPoint(localPosition);
		return r;
	}
	ShadingData makeSphereShadingData  (const SphereInstanceData instance, const TransformData transform, const float3 localPosition) {
		ShadingData r;
		const float3 normal = normalize(transform.transformVector(localPosition));
		r.mPosition = transform.transformPoint(localPosition);
		r.mFlagsMaterialAddress = 0;
		BF_SET(r.mFlagsMaterialAddress, instance.getMaterialAddress(), 4, 28);
		r.mPackedGeometryNormal = r.mPackedShadingNormal = packNormal(normal);
		const float radius = instance.radius();
		r.mShapeArea = 4*M_PI*radius*radius;
		r.mMeanCurvature = 1/radius;
		r.mTexcoord = cartesianToSphericalUv(normalize(localPosition));
		const float3 dpdu = transform.transformVector(float3(-sin(r.mTexcoord[0]) * sin(r.mTexcoord[1]), 0                   , cos(r.mTexcoord[0]) * sin(r.mTexcoord[1])));
		const float3 dpdv = transform.transformVector(float3( cos(r.mTexcoord[0]) * cos(r.mTexcoord[1]), -sin(r.mTexcoord[1]), sin(r.mTexcoord[0]) * cos(r.mTexcoord[1])));
		r.mPackedTangent = packNormal(normalize(dpdu - normal*dot(normal, dpdu)));
		r.mTexcoordScreenSize = 1/max(length(dpdu), length(dpdv));
		return r;
	}
	ShadingData makeVolumeShadingData  (const VolumeInstanceData instance, const float3 position) {
		ShadingData r;
		r.mPosition = position;
		r.mFlagsMaterialAddress = 0;
		BF_SET(r.mFlagsMaterialAddress, instance.getMaterialAddress(), 4, 28);
		r.mShapeArea = 0;
		r.mTexcoordScreenSize = 0;
		return r;
    }

    ShadingData makeShadingData(const InstanceData instance, const TransformData transform, const float3 localPosition, const uint primitiveIndex = -1) {
        switch (instance.getType()) {
        case InstanceType::eMesh:
            return makeTriangleShadingData(reinterpret<MeshInstanceData>(instance), transform, primitiveIndex, localPosition);
        case InstanceType::eSphere:
            return makeSphereShadingData(reinterpret<SphereInstanceData>(instance), transform, localPosition);
		default:
		case InstanceType::eVolume:
			return makeVolumeShadingData(reinterpret<VolumeInstanceData>(instance), transform.transformPoint(localPosition));
		}
	}
}