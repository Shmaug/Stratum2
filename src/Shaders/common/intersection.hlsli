#pragma once

#include "compat/scene.h"
#include "materials/medium.hlsli"
#include "rng.hlsli"

#define SHADING_FLAG_FLIP_BITANGENT BIT(0)

extension ShadingData {
	property bool isSurface       { get { return mShapeArea > 0; } }
	property bool isMedium        { get { return mShapeArea == 0; } }
	property bool isEnvironment   { get { return mShapeArea < 0; } }
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
};

struct IntersectionResult {
	uint mInstancePrimitiveIndex;
	float mPrimitivePickPdf;
	float mDistance;
	ShadingData mShadingData;

	property uint instanceIndex {
        get { return BF_GET(mInstancePrimitiveIndex, 0, 16); }
        set { BF_SET(mInstancePrimitiveIndex, newValue, 0, 16); }
	}
	property uint primitiveIndex {
		get { return BF_GET(mInstancePrimitiveIndex, 16, 16); }
		set { BF_SET(mInstancePrimitiveIndex, newValue, 16, 16); }
	}
	property InstanceData  instance  { get { return gScene.mInstances[instanceIndex]; } }
	property TransformData transform { get { return gScene.mInstanceTransforms[instanceIndex]; } }
	property uint lightIndex         { get { return gScene.mInstanceLightMap[instanceIndex]; } }
};

float3 rayOffset(const float3 P, const float3 Ng) {
	// This function should be used to compute a modified ray start position for
	// rays leaving from a surface. This is from "A Fast and Robust Method for Avoiding
	// Self-Intersection" see https://research.nvidia.com/publication/2019-03_A-Fast-and
  float int_scale = 256.0;
  int3 of_i = int3(int_scale * Ng);

  float origin = 1.0 / 32.0;
  float float_scale = 1.0 / 65536.0;
  return float3(abs(P.x) < origin ? P.x + float_scale * Ng.x : asfloat(asint(P.x) + ((P.x < 0.0) ? -of_i.x : of_i.x)),
                abs(P.y) < origin ? P.y + float_scale * Ng.y : asfloat(asint(P.y) + ((P.y < 0.0) ? -of_i.y : of_i.y)),
                abs(P.z) < origin ? P.z + float_scale * Ng.z : asfloat(asint(P.z) + ((P.z < 0.0) ? -of_i.z : of_i.z)));
}
float3 rayOffset(const float3 P, const float3 Ng, float3 dir) {
	return rayOffset(P, dot(Ng, dir) < 0 ? -Ng : Ng);
}

RayDesc makeRay(const float3 origin, const float3 direction, const float tmin = 0, const float tmax = POS_INFINITY) {
	RayDesc ray;
	ray.Origin = origin;
	ray.TMin = tmin;
	ray.Direction = direction;
	ray.TMax = tmax;
	return ray;
}

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
	ShadingData makeTriangleShadingData_(const uint materialAddress, const TransformData transform, const MeshVertexInfo vertexInfo, const uint3 tri, const float2 bary, const float3 v0, const float3 v1, const float3 v2) {
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
		loadTriangleAttribute(mVertexBuffers[NonUniformResourceIndex(vertexInfo.texcoordBuffer())], vertexInfo.texcoordOffset(), vertexInfo.texcoordStride(), tri, t0, t1, t2);

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

		float3 n0,n1,n2;
		loadTriangleAttribute(mVertexBuffers[NonUniformResourceIndex(vertexInfo.normalBuffer())], vertexInfo.normalOffset(), vertexInfo.normalStride(), tri, n0, n1, n2);

		float3 shadingNormal = n0 + (n1 - n0)*bary.x + (n2 - n0)*bary.y;
		if (all(shadingNormal.xyz == 0) || any(isnan(shadingNormal))) {
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

		ShadingData r = makeTriangleShadingData_(instance.materialAddress(), transform, vertexInfo, tri, bary, v0, v1, v2);
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

		ShadingData r = makeTriangleShadingData_(instance.materialAddress(), transform, vertexInfo, tri, bary, v0, v1, v2);
		r.mPosition = transform.transformPoint(localPosition);
		return r;
	}
	ShadingData makeSphereShadingData  (const SphereInstanceData instance, const TransformData transform, const float3 localPosition) {
		ShadingData r;
		const float3 normal = normalize(transform.transformVector(localPosition));
		r.mPosition = transform.transformPoint(localPosition);
		r.mFlagsMaterialAddress = 0;
		BF_SET(r.mFlagsMaterialAddress, instance.materialAddress(), 4, 28);
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
		BF_SET(r.mFlagsMaterialAddress, instance.materialAddress(), 4, 28);
		r.mShapeArea = 0;
		r.mTexcoordScreenSize = 0;
		return r;
	}

    bool traceRay(const RayDesc ray, const bool closest, out IntersectionResult isect) {
        if (gPerformanceCounters)
			InterlockedAdd(mPerformanceCounters[0], 1);

		// trace ray

		RayQuery<RAY_FLAG_NONE> rayQuery;
		rayQuery.TraceRayInline(mAccelerationStructure, closest ? RAY_FLAG_NONE : RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, ~0, ray);
		while (rayQuery.Proceed()) {
			const uint instanceIndex = rayQuery.CandidateInstanceID();
			switch (rayQuery.CandidateType()) {
				case CANDIDATE_PROCEDURAL_PRIMITIVE: {
					const InstanceData instance = mInstances[instanceIndex];
					switch (instance.type()) {
						case InstanceType::eSphere: {
							const float2 st = raySphere(rayQuery.CandidateObjectRayOrigin(), rayQuery.CandidateObjectRayDirection(), 0, reinterpret<SphereInstanceData>(instance).radius());
							if (st.x < st.y) {
								const float t = st.x > rayQuery.RayTMin() ? st.x : st.y;
								if (t < rayQuery.CommittedRayT() && t > rayQuery.RayTMin())
									rayQuery.CommitProceduralPrimitiveHit(t);
							}
							break;
						}

						case InstanceType::eVolume: {
							pnanovdb_buf_t volumeBuffer = mVolumes[NonUniformResourceIndex(reinterpret<VolumeInstanceData>(instance).volumeIndex())];
							pnanovdb_grid_handle_t gridHandle = {0};
							pnanovdb_root_handle_t root = pnanovdb_tree_get_root(volumeBuffer, pnanovdb_grid_get_tree(volumeBuffer, gridHandle));
							const float3 origin    = pnanovdb_grid_world_to_indexf(volumeBuffer, gridHandle, rayQuery.CandidateObjectRayOrigin());
							const float3 direction = pnanovdb_grid_world_to_index_dirf(volumeBuffer, gridHandle, rayQuery.CandidateObjectRayDirection());
							const pnanovdb_coord_t bbox_min = pnanovdb_root_get_bbox_min(volumeBuffer, root);
							const pnanovdb_coord_t bbox_max = pnanovdb_root_get_bbox_max(volumeBuffer, root) + 1;
							const float3 t0 = (bbox_min - origin) / direction;
							const float3 t1 = (bbox_max - origin) / direction;
							const float2 st = float2(max3(min(t0, t1)), min3(max(t0, t1)));
							const float t = st.x > rayQuery.RayTMin() ? st.x : st.y;
							if (t < rayQuery.CommittedRayT() && t > rayQuery.RayTMin()) {
								const float3 normal = normalize(mInstanceTransforms[instanceIndex].transformVector(pnanovdb_grid_index_to_world_dirf(volumeBuffer, gridHandle, float3(t1 == t) - float3(t0 == t))));
								isect.mShadingData.mPackedGeometryNormal = isect.mShadingData.mPackedShadingNormal = packNormal(normal);
								rayQuery.CommitProceduralPrimitiveHit(t);
							}
							break;
						}
					}
				}
				case CANDIDATE_NON_OPAQUE_TRIANGLE: {
					if (!gAlphaTest) {
						rayQuery.CommitNonOpaqueTriangleHit();
						break;
					}
                    const MeshInstanceData instance = reinterpret<MeshInstanceData>(mInstances[instanceIndex]);
                    const uint alphaImage = mMaterialData.Load<uint>(instance.materialAddress() + ImageValue4::PackedSize * DisneyMaterialData::gDataCount);
					if (alphaImage >= gImageCount)
						break;
                    const MeshVertexInfo vertexInfo = mMeshVertexInfo[instance.vertexInfoIndex()];
                    const uint3 tri = loadTriangleIndices(mVertexBuffers[NonUniformResourceIndex(vertexInfo.indexBuffer())], vertexInfo.indexStride(), vertexInfo.indexStride(), rayQuery.CandidatePrimitiveIndex());
					float2 v0,v1,v2;
					loadTriangleAttribute(mVertexBuffers[NonUniformResourceIndex(vertexInfo.texcoordBuffer())], vertexInfo.texcoordOffset(), vertexInfo.texcoordStride(), tri, v0, v1, v2);
					const float2 barycentrics = rayQuery.CandidateTriangleBarycentrics();
					const float2 uv = v0 + (v1 - v0)*barycentrics.x + (v2 - v0)*barycentrics.y;
					if (mImage1s[NonUniformResourceIndex(alphaImage)].SampleLevel(mStaticSampler, uv, 0) >= 0.25)
						rayQuery.CommitNonOpaqueTriangleHit();
					break;
				}
			}
        }

		// create IntersectionResult

		switch (rayQuery.CommittedStatus()) {
			case COMMITTED_NOTHING: {
				// ray missed scene
				isect.mDistance = ray.TMax;
                isect.instanceIndex = INVALID_INSTANCE;
                isect.mShadingData.mPosition = ray.Direction;
                isect.mShadingData.mPackedGeometryNormal = isect.mShadingData.mPackedShadingNormal = packNormal(ray.Direction);
				isect.mShadingData.mShapeArea = -1;
				isect.mShadingData.mTexcoord = cartesianToSphericalUv(ray.Direction);
				return false;
			}
			case COMMITTED_TRIANGLE_HIT: {
				isect.mDistance = rayQuery.CommittedRayT();
				isect.instanceIndex = rayQuery.CommittedInstanceID();
				isect.primitiveIndex = rayQuery.CommittedPrimitiveIndex();

                MeshInstanceData meshInstance = reinterpret<MeshInstanceData>(isect.instance);
                isect.mPrimitivePickPdf = 1.0/meshInstance.primitiveCount();
				isect.mShadingData = makeTriangleShadingData(meshInstance, isect.transform, rayQuery.CommittedPrimitiveIndex(), rayQuery.CommittedTriangleBarycentrics());
				break;
			}
			case COMMITTED_PROCEDURAL_PRIMITIVE_HIT: {
				isect.mDistance = rayQuery.CommittedRayT();
				isect.instanceIndex = rayQuery.CommittedInstanceID();
                isect.primitiveIndex = INVALID_PRIMITIVE;
                isect.mPrimitivePickPdf = 1;
				switch (isect.instance.type()) {
					case InstanceType::eSphere:
						isect.mShadingData = makeSphereShadingData(reinterpret<SphereInstanceData>(isect.instance), isect.transform, rayQuery.CommittedObjectRayOrigin() + rayQuery.CommittedObjectRayDirection() * rayQuery.CommittedRayT());
						break;
					case InstanceType::eVolume: {
						const uint n = isect.mShadingData.mPackedGeometryNormal; // assigned in the rayQuery loop above
						isect.mShadingData = makeVolumeShadingData(reinterpret<VolumeInstanceData>(isect.instance), ray.Origin + ray.Direction * rayQuery.CommittedRayT());
						isect.mShadingData.mPackedGeometryNormal = n;
						break;
					}
				}
				break;
			}
		}
		return true;
	}

	bool traceScatteringRay(RayDesc ray, inout RandomSampler rng, inout uint curMediumInstance, inout float3 beta, out float dirPdf, out float neePdf, out IntersectionResult isect) {
		dirPdf = 1;
		neePdf = 1;

        if (!gHasMedia)
            return traceRay(ray, true, /*out*/ isect);

		// load medium
		Medium medium;
		if (curMediumInstance != INVALID_INSTANCE)
			medium = Medium(mInstances[curMediumInstance].materialAddress());

        const float3 origin_ = ray.Origin;

        ray.Origin += ray.Direction * ray.TMin;
		ray.TMin = 0;

        while (ray.TMax > 1e-6) {
            const bool hit = traceRay(ray, true, /*out*/ isect);

			// delta track through current medium
			if (curMediumInstance != INVALID_INSTANCE) {
				const TransformData invTransform = mInstanceInverseTransforms[curMediumInstance];

                float3 _dirPdf, _neePdf;
                bool scattered;
				const float3 p = medium.deltaTrack<true>(
					rng,
					invTransform.transformPoint(ray.Origin),
					invTransform.transformVector(ray.Direction),
					isect.mDistance,
					/*inout*/ beta,
					/*out*/ _dirPdf,
					/*out*/ _neePdf,
					/*out*/ scattered);

				dirPdf *= average(_dirPdf);
				neePdf *= average(_neePdf);

				if (scattered) {
					// medium scattering event
                    isect.mShadingData = makeVolumeShadingData(reinterpret<VolumeInstanceData>(mInstances[curMediumInstance]), mInstanceTransforms[curMediumInstance].transformPoint(p));
                    isect.mDistance = length(isect.mShadingData.mPosition - origin_);
					isect.instanceIndex = curMediumInstance;
                    isect.primitiveIndex = INVALID_PRIMITIVE;
                    isect.mPrimitivePickPdf = 0;
					return true;
				}
			}

			if (!hit) return false; // missed scene

            if (isect.instance.type() != InstanceType::eVolume)
				return true;

			const float3 ng = isect.mShadingData.geometryNormal;
			if (dot(ng, ray.Direction) < 0) {
				// entering medium
				curMediumInstance = isect.instanceIndex;
				medium = Medium(isect.mShadingData.materialAddress);
				ray.Origin = rayOffset(isect.mShadingData.mPosition, -ng);
			} else {
				// leaving medium
				curMediumInstance = INVALID_INSTANCE;
				ray.Origin = rayOffset(isect.mShadingData.mPosition, ng);
            }
            ray.TMax -= isect.mDistance;
		}

        return false;
	}

    void traceVisibilityRay(RayDesc ray, inout RandomSampler rng, uint curMediumInstance, out float3 beta, out float dirPdf, out float neePdf) {
        if (gPerformanceCounters)
            InterlockedAdd(mPerformanceCounters[1], 1);

		dirPdf = 1;
		neePdf = 1;
		beta = 1;

		if (!gHasMedia) {
            IntersectionResult isect;
			if (traceRay(ray, false, /*out*/ isect)) {
				beta = 0;
				dirPdf = 0;
				neePdf = 0;
			}
			return;
		}

		// load medium
		Medium medium;
		if (curMediumInstance != INVALID_INSTANCE)
            medium = Medium(mInstances[curMediumInstance].materialAddress());

        ray.Origin += ray.Direction * ray.TMin;
        ray.TMin = 0;

		while (ray.TMax > 1e-6f) {
            IntersectionResult isect;
            const bool hit = traceRay(ray, true, /*out*/ isect);

            if (hit && isect.instance.type() != InstanceType::eVolume) {
				// hit a surface
				beta = 0;
				dirPdf = 0;
				neePdf = 0;
				break;
			}

			// delta track through current medium
			if (curMediumInstance != INVALID_INSTANCE) {
				const TransformData invTransform = mInstanceInverseTransforms[curMediumInstance];

                float3 _dirPdf, _neePdf;
                bool scattered;
				medium.deltaTrack<false>(
					rng,
					invTransform.transformPoint(ray.Origin),
					invTransform.transformVector(ray.Direction),
					isect.mDistance,
					/*inout*/ beta,
					/*out*/ _dirPdf,
					/*out*/ _neePdf,
					/*out*/ scattered);

				dirPdf *= average(_dirPdf);
				neePdf *= average(_neePdf);
                if (all(beta) <= 0)
					return;
			}

			if (!hit) return; // successful transmission

			// hit medium aabb
			const float3 ng = isect.mShadingData.geometryNormal;
			if (dot(ng, ray.Direction) < 0) {
				// entering medium
				curMediumInstance = isect.instanceIndex;
				medium = Medium(isect.mShadingData.materialAddress);
				ray.Origin = rayOffset(isect.mShadingData.mPosition, -ng);
			} else {
				// leaving medium
				curMediumInstance = INVALID_INSTANCE;
				ray.Origin = rayOffset(isect.mShadingData.mPosition, ng);
			}
			ray.TMax -= isect.mDistance;
		}
	}
};