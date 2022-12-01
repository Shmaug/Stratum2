#include "materials/medium.hlsli"

extension SceneParameters {
	uint3 loadTriangle(const uint indexByteOffset, const uint indexStride, const uint primitiveIndex) {
		const int offsetBytes = (int)(indexByteOffset + primitiveIndex*3*indexStride);
		uint3 tri;
		if (indexStride == 2) {
			// https://github.com/microsoft/DirectX-Graphics-Samples/blob/master/Samples/Desktop/D3D12Raytracing/src/D3D12RaytracingSimpleLighting/Raytracing.hlsl
			const int dwordAlignedOffset = offsetBytes & ~3;
			const uint2 four16BitIndices = mIndices.Load2(dwordAlignedOffset);
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
			tri = mIndices.Load3(offsetBytes);
		return tri;
	}
	uint3 loadTriangle(const TrianglesInstanceData instance, uint primitiveIndex) {
		return instance.firstVertex() + loadTriangle(instance.indicesByteOffset(), instance.indexStride(), primitiveIndex);
	}


	ShadingData makeTriangleShadingData(const uint materialAddress, const TransformData transform, const float2 bary, const PackedVertexData v0, const PackedVertexData v1, const PackedVertexData v2) {
		ShadingData r;
		r.mPosition = transform.transformPoint(v0.mPosition + (v1.mPosition - v0.mPosition)*bary.x + (v2.mPosition - v0.mPosition)*bary.y);
		r.mFlagsMaterialAddress = 0;
		BF_SET(r.mFlagsMaterialAddress, materialAddress, 4, 28);
		r.mTexcoord = v0.uv() + (v1.uv() - v0.uv())*bary.x + (v2.uv() - v0.uv())*bary.y;

		const float3 dPds = transform.transformVector(v0.mPosition - v2.mPosition);
		const float3 dPdt = transform.transformVector(v1.mPosition - v2.mPosition);
		float3 geometryNormal = cross(dPds, dPdt);
		const float area2 = length(geometryNormal);
		geometryNormal /= area2;
		r.mPackedGeometryNormal = packNormal(geometryNormal);
		r.mShapeArea = area2/2;

		// [du/ds, du/dt]
		// [dv/ds, dv/dt]
		const float2 duvds = v2.uv() - v0.uv();
		const float2 duvdt = v2.uv() - v1.uv();
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

		float3 shadingNormal = v0.mNormal + (v1.mNormal - v0.mNormal)*bary.x + (v2.mNormal - v0.mNormal)*bary.y;
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

			const float3 dNds = v2.mNormal - v0.mNormal;
			const float3 dNdt = v2.mNormal - v1.mNormal;
			const float3 dNdu = dNds * dsdu + dNdt * dtdu;
			const float3 dNdv = dNds * dsdv + dNdt * dtdv;
			const float3 bitangent = normalize(cross(shadingNormal, tangent));
			r.mMeanCurvature = (dot(dNdu, tangent) + dot(dNdv, bitangent)) / 2;
		}
		return r;
	}
	ShadingData makeTriangleShadingData(const TrianglesInstanceData instance, const TransformData transform, const uint primitiveIndex, const float2 bary) {
		const uint3 tri = loadTriangle(instance, primitiveIndex);
		return makeTriangleShadingData(instance.materialAddress(), transform, bary, mVertices[tri.x], mVertices[tri.y], mVertices[tri.z]);
	}
	ShadingData makeTriangleShadingData(const TrianglesInstanceData instance, const TransformData transform, const uint primitiveIndex, const float3 localPosition) {
		const uint3 tri = loadTriangle(instance, primitiveIndex);
		const PackedVertexData v0 = mVertices[tri.x];
		const PackedVertexData v1 = mVertices[tri.y];
		const PackedVertexData v2 = mVertices[tri.z];
		const float3 v1v0 = v1.mPosition - v0.mPosition;
		const float3 v2v0 = v2.mPosition - v0.mPosition;
		const float3 p_v0 = localPosition - v0.mPosition;
		const float d00 = dot(v1v0, v1v0);
		const float d01 = dot(v1v0, v2v0);
		const float d11 = dot(v2v0, v2v0);
		const float d20 = dot(p_v0, v1v0);
		const float d21 = dot(p_v0, v2v0);
		const float2 bary = float2(d11 * d20 - d01 * d21, d00 * d21 - d01 * d20) / (d00 * d11 - d01 * d01);
		ShadingData r = makeTriangleShadingData(instance.materialAddress(), transform, bary, v0, v1, v2);
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

	ShadingData makeBackgroundShadingData(const float3 direction) {
		ShadingData r;
		r.mPosition = direction;
		r.mShapeArea = -1;
		r.mFlagsMaterialAddress = 0;
		BF_SET(r.mFlagsMaterialAddress, gPushConstants.mEnvironmentMaterialAddress, 4, 28);
		r.mTexcoordScreenSize = 0;
		return r;
	}

	ShadingData makeShadingData(const uint instanceIndex, const uint primitiveIndex, const float3 localPosition) {
		if (instanceIndex == INVALID_INSTANCE)
			return makeBackgroundShadingData(localPosition);

		const InstanceData instance = mInstances[instanceIndex];
		const TransformData transform = mInstanceTransforms[instanceIndex];
		switch (instance.type()) {
		case INSTANCE_TYPE_TRIANGLES:
			return makeTriangleShadingData(reinterpret<TrianglesInstanceData>(instance), transform, primitiveIndex, localPosition.xy);
		default:
		case INSTANCE_TYPE_SPHERE:
			return makeSphereShadingData(reinterpret<SphereInstanceData>(instance), transform, localPosition);
		case INSTANCE_TYPE_VOLUME:
			return makeVolumeShadingData(reinterpret<VolumeInstanceData>(instance), transform, localPosition);
		}
	}
};


struct IntersectionResult {
	float t;
	uint instancePrimitiveIndex;
	property uint instanceIndex {
		get { return BF_GET(instancePrimitiveIndex, 0, 16); }
		set { BF_SET(instancePrimitiveIndex, newValue, 0, 16); }
	}
	property uint primitiveIndex {
		get { return BF_GET(instancePrimitiveIndex, 16, 16); }
		set { BF_SET(instancePrimitiveIndex, newValue, 16, 16); }
	}
	property InstanceData instance { get { return gScene.mInstances[instanceIndex]; } }
	property TransformData transform { get { return gScene.mInstanceTransforms[instanceIndex]; } }
	ShadingData shadingData;
	float shapePdfA;
};

bool traceRay(const SceneParameters scene, const RayDesc ray, const bool closest, out IntersectionResult isect) {
	//if (gUsePerformanceCounters)
	//	InterlockedAdd(mPerformanceCounters[0], 1);

	RayQuery<RAY_FLAG_NONE> rayQuery;
	rayQuery.TraceRayInline(scene.mAccelerationStructure, closest ? RAY_FLAG_NONE : RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, ~0, ray);
	while (rayQuery.Proceed()) {
		const uint instanceIndex = rayQuery.CandidateInstanceID();
		switch (rayQuery.CandidateType()) {
			case CANDIDATE_PROCEDURAL_PRIMITIVE: {
				const InstanceData instance = scene.mInstances[instanceIndex];
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
						pnanovdb_buf_t volumeBuffer = scene.mVolumes[NonUniformResourceIndex(reinterpret<VolumeInstanceData>(instance).volumeIndex())];
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
							const float3 normal = normalize(scene.mInstanceTransforms[instanceIndex].transformVector(pnanovdb_grid_index_to_world_dirf(volumeBuffer, gridHandle, float3(t1 == t) - float3(t0 == t))));
							isect.shadingData.mPackedGeometryNormal = isect.shadingData.mPackedShadingNormal = packNormal(normal);
							rayQuery.CommitProceduralPrimitiveHit(t);
						}
						break;
					}
				}
			}
			case CANDIDATE_NON_OPAQUE_TRIANGLE: {
				if (gEnableAlphaTest) {
					const TrianglesInstanceData instance = reinterpret<TrianglesInstanceData>(scene.mInstances[instanceIndex]);
					const uint alphaImage = scene.mMaterialData.Load<uint>(instance.materialAddress() + DisneyMaterialData::gAlphaMaterialOffset);
					if (alphaImage < gImageCount) {
						const uint3 tri = scene.loadTriangle(instance, rayQuery.CandidatePrimitiveIndex());
						const PackedVertexData v0 = scene.mVertices[tri.x];
						const PackedVertexData v1 = scene.mVertices[tri.y];
						const PackedVertexData v2 = scene.mVertices[tri.z];
						const float2 barycentrics = rayQuery.CandidateTriangleBarycentrics();
						const float2 uv = v0.uv() + (v1.uv() - v0.uv())*barycentrics.x + (v2.uv() - v0.uv())*barycentrics.y;
						if (scene.mImage1s[NonUniformResourceIndex(alphaImage)].SampleLevel(scene.mStaticSampler, uv, 0) >= 0.25)
							rayQuery.CommitNonOpaqueTriangleHit();
					}
				} else
					rayQuery.CommitNonOpaqueTriangleHit();
				break;
			}
		}
	}

	if (rayQuery.CommittedStatus() == COMMITTED_NOTHING)
		return false;

	isect.t = rayQuery.CommittedRayT();
	isect.instanceIndex = rayQuery.CommittedInstanceID();
	//isect.instance  = scene.mInstances[isect.instanceIndex];
	//isect.transform = scene.mInstanceTransforms[isect.instanceIndex];

	if (rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
		TrianglesInstanceData tri = reinterpret<TrianglesInstanceData>(isect.instance);
		isect.primitiveIndex = rayQuery.CommittedPrimitiveIndex();
		isect.shadingData = scene.makeTriangleShadingData(tri, isect.transform, rayQuery.CommittedPrimitiveIndex(), rayQuery.CommittedTriangleBarycentrics());
		isect.shapePdfA = 1 / (isect.shadingData.mShapeArea * tri.primitiveCount());
	} else if (rayQuery.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT) {
		isect.primitiveIndex = INVALID_PRIMITIVE;
		const float3 localPosition = rayQuery.CommittedObjectRayOrigin() + rayQuery.CommittedObjectRayDirection()*rayQuery.CommittedRayT();
		switch (isect.instance.type()) {
			case INSTANCE_TYPE_SPHERE:
				isect.shadingData = scene.makeSphereShadingData(reinterpret<SphereInstanceData>(isect.instance), isect.transform, localPosition);
				isect.shapePdfA = 1/isect.shadingData.mShapeArea;
				break;
			case INSTANCE_TYPE_VOLUME:
				// shadingData.mPackedGeometryNormal set in the rayQuery loop above
				const uint n = isect.shadingData.mPackedGeometryNormal;
				isect.shadingData = scene.makeVolumeShadingData(reinterpret<VolumeInstanceData>(isect.instance), isect.transform, localPosition);
				isect.shapePdfA = 1;
				isect.shadingData.mPackedGeometryNormal = n;
				break;
		}
	}

	return true;
}

bool traceRay(inout RandomSampler rng, RayDesc ray, inout uint curMediumInstance, out float3 beta, out float dirPdf, out float neePdf, out IntersectionResult isect) {
	dirPdf = 1;
	neePdf = 1;
	beta = 1;

	if (!gEnableMedia)
		return traceRay(gScene, ray, true, /*out*/ isect);

	// load medium
	Medium medium;
	if (curMediumInstance != INVALID_INSTANCE)
		medium = Medium(gScene, gScene.mInstances[curMediumInstance].materialAddress());

	while (ray.TMax > 1e-6) {
		const bool hit = traceRay(gScene, ray, true, isect);

		// delta track through current medium
		if (curMediumInstance != INVALID_INSTANCE) {
			const TransformData invTransform = gScene.mInstanceInverseTransforms[curMediumInstance];

			float3 _dirPdf, _neePdf;
			const float3 p = medium.deltaTrack(
				rng,
				invTransform.transformPoint(ray.Origin + ray.Direction*ray.TMin),
				invTransform.transformVector(ray.Direction),
				isect.t - ray.TMin,
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
				isect.shadingData.mPosition = p;
				isect.shadingData.mShapeArea = 0;
				isect.t = length(p - ray.Origin);
				return true;
			}
		}

		if (!hit) return false; // missed scene

		if (isect.instance.type() != INSTANCE_TYPE_VOLUME)
			return true;

		const float3 ng = isect.shadingData.geometryNormal;
		if (dot(ng, ray.Direction) < 0) {
			// entering volume
			curMediumInstance = isect.instanceIndex;
			medium = Medium(gScene, isect.instance.materialAddress());
			ray.Origin = rayOffset(isect.shadingData.mPosition, -ng);
		} else {
			// leaving volume
			curMediumInstance = INVALID_INSTANCE;
			ray.Origin = rayOffset(isect.shadingData.mPosition, ng);
		}
		// TODO: ray.Origin shouldn't be modified, instead offset ray.TMin somehow
		ray.TMin += isect.t;
	}
}

void traceVisibilityRay(inout RandomSampler rng, RayDesc ray, uint curMediumInstance, out float3 beta, out float dirPdf, out float neePdf) {
	dirPdf = 1;
	neePdf = 1;
	beta = 1;

	if (!gEnableMedia) {
		IntersectionResult isect;
		if (traceRay(gScene, ray, false, /*out*/ isect)) {
			beta = 0;
			dirPdf = 0;
			neePdf = 0;
		}
		return;
	}

	// load medium
	Medium medium;
	if (curMediumInstance != INVALID_INSTANCE)
		medium = Medium(gScene, gScene.mInstances[curMediumInstance].materialAddress());

	while (ray.TMax > 1e-6f) {
		IntersectionResult isect;
		const bool hit = traceRay(gScene, ray, true, /*out*/ isect);

		if (hit && isect.instance.type() != INSTANCE_TYPE_VOLUME) {
			// hit a surface
			beta = 0;
			dirPdf = 0;
			neePdf = 0;
			break;
		}

		// delta track through current medium
		if (curMediumInstance != INVALID_INSTANCE) {
			const TransformData invTransform = gScene.mInstanceInverseTransforms[curMediumInstance];

			float3 _dirPdf, _neePdf;
			medium.deltaTrack(
				rng,
				invTransform.transformPoint(ray.Origin + ray.Direction*ray.TMin),
				invTransform.transformVector(ray.Direction),
				isect.t - ray.TMin,
				/*inout*/ beta,
				/*out*/ _dirPdf,
				/*out*/ _neePdf,
				false);

			dirPdf *= average(_dirPdf);
			neePdf *= average(_neePdf);
		}

		if (!hit) return; // successful transmission

		// hit medium aabb
		const float3 ng = isect.shadingData.geometryNormal;
		if (dot(ng, ray.Direction) < 0) {
			// entering medium
			curMediumInstance = isect.instanceIndex;
			medium = Medium(gScene, isect.instance.materialAddress());
			ray.Origin = rayOffset(isect.shadingData.mPosition, -ng);
		} else {
			// leaving medium
			curMediumInstance = INVALID_INSTANCE;
			ray.Origin = rayOffset(isect.shadingData.mPosition, ng);
		}
		ray.TMin = 0;
		ray.TMax -= isect.t;
	}
}