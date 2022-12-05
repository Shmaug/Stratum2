#include "compat/scene.h"
#include "materials/medium.hlsli"

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
};

struct IntersectionResult {
	float mDistance;
	uint mInstancePrimitiveIndex;
	property uint instanceIndex {
        get { return BF_GET(mInstancePrimitiveIndex, 0, 16); }
        set { BF_SET(mInstancePrimitiveIndex, newValue, 0, 16); }
	}
	property uint primitiveIndex {
		get { return BF_GET(mInstancePrimitiveIndex, 16, 16); }
		set { BF_SET(mInstancePrimitiveIndex, newValue, 16, 16); }
	}
	property InstanceData instance { get { return gScene.mInstances[instanceIndex]; } }
	property TransformData transform { get { return gScene.mInstanceTransforms[instanceIndex]; } }
	float mShapePdfA;
};

float3 rayOffset(const float3 P, const float3 Ng) {
	// This function should be used to compute a modified ray start position for
	// rays leaving from a surface. This is from "A Fast and Robust Method for Avoiding
	// Self-Intersection" see https://research.nvidia.com/publication/2019-03_A-Fast-and
  float int_scale = 256.0;
  int3 of_i = int_scale * Ng;

  float origin = 1.0 / 32.0;
  float float_scale = 1.0 / 65536.0;
  return float3(abs(P.x) < origin ? P.x + float_scale * Ng.x : asfloat(asint(P.x) + ((P.x < 0.0) ? -of_i.x : of_i.x)),
                abs(P.y) < origin ? P.y + float_scale * Ng.y : asfloat(asint(P.y) + ((P.y < 0.0) ? -of_i.y : of_i.y)),
                abs(P.z) < origin ? P.z + float_scale * Ng.z : asfloat(asint(P.z) + ((P.z < 0.0) ? -of_i.z : of_i.z)));
}
float3 rayOffset(const float3 P, const float3 Ng, float3 dir) {
	return rayOffset(P, dot(Ng, dir) < 0 ? -Ng : Ng);
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
	v0 = vertexBuffer.Load<T>(offset + stride*tri[0]);
	v1 = vertexBuffer.Load<T>(offset + stride*tri[1]);
	v2 = vertexBuffer.Load<T>(offset + stride*tri[2]);
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
	ShadingData makeVolumeShadingData  (const VolumeInstanceData instance, const TransformData transform, const float3 localPosition) {
		ShadingData r;
		r.mPosition = transform.transformPoint(localPosition);
		r.mFlagsMaterialAddress = 0;
		BF_SET(r.mFlagsMaterialAddress, instance.materialAddress(), 4, 28);
		r.mShapeArea = 0;
		r.mTexcoordScreenSize = 0;
		return r;
	}

    bool traceRay(const RayDesc ray, const bool closest, out IntersectionResult isect, out ShadingData shadingData) {
        if (CHECK_FEATURE(PerformanceCounters))
			InterlockedAdd(mPerformanceCounters[0], 1);

		RayQuery<RAY_FLAG_NONE> rayQuery;
		rayQuery.TraceRayInline(mAccelerationStructure, closest ? RAY_FLAG_NONE : RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, ~0, ray);
		while (rayQuery.Proceed()) {
			const uint instanceIndex = rayQuery.CandidateInstanceID();
			switch (rayQuery.CandidateType()) {
				case CANDIDATE_PROCEDURAL_PRIMITIVE: {
					const InstanceData instance = mInstances[instanceIndex];
					switch (instance.type()) {
						case INSTANCE_TYPE_SPHERE: {
							const float2 st = raySphere(rayQuery.CandidateObjectRayOrigin(), rayQuery.CandidateObjectRayDirection(), 0, reinterpret<SphereInstanceData>(instance).radius());
							if (st.x < st.y) {
								const float t = st.x > rayQuery.RayTMin() ? st.x : st.y;
								if (t < rayQuery.CommittedRayT() && t > rayQuery.RayTMin())
									rayQuery.CommitProceduralPrimitiveHit(t);
							}
							break;
						}

						case INSTANCE_TYPE_VOLUME: {
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
								shadingData.mPackedGeometryNormal = shadingData.mPackedShadingNormal = packNormal(normal);
								rayQuery.CommitProceduralPrimitiveHit(t);
							}
							break;
						}
					}
				}
				case CANDIDATE_NON_OPAQUE_TRIANGLE: {
					if (!CHECK_FEATURE(AlphaTest)) {
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

		if (rayQuery.CommittedStatus() == COMMITTED_NOTHING)
			return false;

		isect.mDistance = rayQuery.CommittedRayT();
		isect.instanceIndex = rayQuery.CommittedInstanceID();

		if (rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
			MeshInstanceData meshInstance = reinterpret<MeshInstanceData>(isect.instance);
			isect.primitiveIndex = rayQuery.CommittedPrimitiveIndex();
			shadingData = makeTriangleShadingData(meshInstance, isect.transform, rayQuery.CommittedPrimitiveIndex(), rayQuery.CommittedTriangleBarycentrics());
			isect.mShapePdfA = 1 / (shadingData.mShapeArea * meshInstance.primitiveCount());
		} else if (rayQuery.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT) {
			isect.primitiveIndex = INVALID_PRIMITIVE;
			const float3 localPosition = rayQuery.CommittedObjectRayOrigin() + rayQuery.CommittedObjectRayDirection()*rayQuery.CommittedRayT();
			switch (isect.instance.type()) {
				case INSTANCE_TYPE_SPHERE:
					shadingData = makeSphereShadingData(reinterpret<SphereInstanceData>(isect.instance), isect.transform, localPosition);
					isect.mShapePdfA = 1/shadingData.mShapeArea;
					break;
				case INSTANCE_TYPE_VOLUME:
					// shadingData.mPackedGeometryNormal set in the rayQuery loop above
					const uint n = shadingData.mPackedGeometryNormal;
					shadingData = makeVolumeShadingData(reinterpret<VolumeInstanceData>(isect.instance), isect.transform, localPosition);
					isect.mShapePdfA = 1;
					shadingData.mPackedGeometryNormal = n;
					break;
			}
		}

		return true;
	}

	bool traceScatteringRay(inout RandomSampler rng, RayDesc ray, inout uint curMediumInstance, out float3 beta, out float dirPdf, out float neePdf, out IntersectionResult isect, out ShadingData shadingData) {
		dirPdf = 1;
		neePdf = 1;
		beta = 1;

        if (!CHECK_FEATURE(Media))
            return traceRay(ray, true, /*out*/ isect, /*out*/ shadingData);

		// load medium
		Medium medium;
		if (curMediumInstance != INVALID_INSTANCE)
			medium = Medium(mInstances[curMediumInstance].materialAddress());

        while (ray.TMax > 1e-6) {
            const bool hit = traceRay(ray, true, /*out*/ isect, /*out*/ shadingData);

			// delta track through current medium
			if (curMediumInstance != INVALID_INSTANCE) {
				const TransformData invTransform = mInstanceInverseTransforms[curMediumInstance];

				float3 _dirPdf, _neePdf;
				const float3 p = medium.deltaTrack(
					rng,
					invTransform.transformPoint(ray.Origin + ray.Direction*ray.TMin),
					invTransform.transformVector(ray.Direction),
					isect.mDistance - ray.TMin,
					/*inout*/ beta,
					/*out*/ _dirPdf,
					/*out*/ _neePdf,
					true);

				dirPdf *= average(_dirPdf);
				neePdf *= average(_neePdf);

				if (all(isfinite(p))) {
					// medium scattering event
					isect.instanceIndex = curMediumInstance;
					isect.primitiveIndex = INVALID_PRIMITIVE;
					shadingData.mPosition = p;
					shadingData.mShapeArea = 0;
					isect.mDistance = length(p - ray.Origin);
					return true;
				}
			}

			if (!hit) return false; // missed scene

			if (isect.instance.type() != INSTANCE_TYPE_VOLUME)
				return true;

			const float3 ng = shadingData.geometryNormal;
			if (dot(ng, ray.Direction) < 0) {
				// entering volume
				curMediumInstance = isect.instanceIndex;
				medium = Medium(isect.instance.materialAddress());
				ray.Origin = rayOffset(shadingData.mPosition, -ng);
			} else {
				// leaving volume
				curMediumInstance = INVALID_INSTANCE;
				ray.Origin = rayOffset(shadingData.mPosition, ng);
			}
			// TODO: ray.Origin shouldn't be modified, instead offset ray.TMin somehow
			ray.TMin += isect.mDistance;
		}
	}

	void traceVisibilityRay(inout RandomSampler rng, RayDesc ray, uint curMediumInstance, out float3 beta, out float dirPdf, out float neePdf) {
		dirPdf = 1;
		neePdf = 1;
		beta = 1;

		if (!CHECK_FEATURE(Media)) {
            IntersectionResult isect;
            ShadingData shadingData;
			if (traceRay(ray, false, /*out*/ isect, /*out*/ shadingData)) {
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

		while (ray.TMax > 1e-6f) {
            IntersectionResult isect;
            ShadingData shadingData;
            const bool hit = traceRay(ray, true, /*out*/ isect, /*out*/ shadingData);

			if (hit && isect.instance.type() != INSTANCE_TYPE_VOLUME) {
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
				medium.deltaTrack(
					rng,
					invTransform.transformPoint(ray.Origin + ray.Direction*ray.TMin),
					invTransform.transformVector(ray.Direction),
					isect.mDistance - ray.TMin,
					/*inout*/ beta,
					/*out*/ _dirPdf,
					/*out*/ _neePdf,
					false);

				dirPdf *= average(_dirPdf);
				neePdf *= average(_neePdf);
			}

			if (!hit) return; // successful transmission

			// hit medium aabb
			const float3 ng = shadingData.geometryNormal;
			if (dot(ng, ray.Direction) < 0) {
				// entering medium
				curMediumInstance = isect.instanceIndex;
				medium = Medium(shadingData.materialAddress);
				ray.Origin = rayOffset(shadingData.mPosition, -ng);
			} else {
				// leaving medium
				curMediumInstance = INVALID_INSTANCE;
				ray.Origin = rayOffset(shadingData.mPosition, ng);
			}
			ray.TMin = 0;
			ray.TMax -= isect.mDistance;
		}
	}
};