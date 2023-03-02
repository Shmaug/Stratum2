#pragma once

#ifndef gHasMedia
#define gHasMedia false
#endif
#ifndef gAlphaTest
#define gAlphaTest false
#endif
#ifndef gCountRays
#define gCountRays false
#endif

#include "material.hlsli"
#include "materials/medium.hlsli"
#include "rng.hlsli"

struct IntersectionResult {
	uint mInstancePrimitiveIndex;
	float mPrimitivePickPdf;
	float mDistance;
	ShadingData mShadingData;

	property uint mInstanceIndex {
        get { return BF_GET(mInstancePrimitiveIndex, 0, 16); }
        set { BF_SET(mInstancePrimitiveIndex, newValue, 0, 16); }
	}
	property uint mPrimitiveIndex {
		get { return BF_GET(mInstancePrimitiveIndex, 16, 16); }
		set { BF_SET(mInstancePrimitiveIndex, newValue, 16, 16); }
	}
	InstanceData  getInstance()   { return gScene.mInstances[mInstanceIndex]; }
	TransformData getTransform()  { return gScene.mInstanceTransforms[mInstanceIndex]; }
	uint          getLightIndex() { return gScene.mInstanceLightMap[mInstanceIndex]; }
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

extension SceneParameters {
    bool traceRay(const RayDesc ray, const bool closest, out IntersectionResult isect) {
        if (gCountRays)
            InterlockedAdd(mRayCount[0], 1);

		// trace ray

		RayQuery<RAY_FLAG_NONE> rayQuery;
		rayQuery.TraceRayInline(mAccelerationStructure, closest ? RAY_FLAG_NONE : RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, ~0, ray);
		while (rayQuery.Proceed()) {
			const uint mInstanceIndex = rayQuery.CandidateInstanceID();
			switch (rayQuery.CandidateType()) {
				case CANDIDATE_PROCEDURAL_PRIMITIVE: {
					const InstanceData instance = mInstances[mInstanceIndex];
					switch (instance.getType()) {
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
							float3 origin,direction, bbox_min, bbox_max;
							const uint volumeIndex = reinterpret<VolumeInstanceData>(instance).volumeIndex();
							if (volumeIndex == -1) {
								origin = rayQuery.CandidateObjectRayOrigin();
								direction = rayQuery.CandidateObjectRayDirection();
								bbox_min = -1;
								bbox_max = 1;
							} else {
								pnanovdb_buf_t volumeBuffer = mVolumes[NonUniformResourceIndex(volumeIndex)];
								pnanovdb_grid_handle_t gridHandle = {0};
								pnanovdb_root_handle_t root = pnanovdb_tree_get_root(volumeBuffer, pnanovdb_grid_get_tree(volumeBuffer, gridHandle));
								origin    = pnanovdb_grid_world_to_indexf    (volumeBuffer, gridHandle, rayQuery.CandidateObjectRayOrigin());
								direction = pnanovdb_grid_world_to_index_dirf(volumeBuffer, gridHandle, rayQuery.CandidateObjectRayDirection());
								bbox_min = pnanovdb_root_get_bbox_min(volumeBuffer, root);
								bbox_max = pnanovdb_root_get_bbox_max(volumeBuffer, root) + 1;
							}
							const float3 t0 = (bbox_min - origin) / direction;
							const float3 t1 = (bbox_max - origin) / direction;
							const float2 st = float2(max3(min(t0, t1)), min3(max(t0, t1)));
							const float t = st.x > rayQuery.RayTMin() ? st.x : st.y;
							if (t < rayQuery.CommittedRayT() && t > rayQuery.RayTMin()) {
								float3 localNormal = float3(t1 == t) - float3(t0 == t);
								if (volumeIndex != -1)
									localNormal = pnanovdb_grid_index_to_world_dirf(mVolumes[NonUniformResourceIndex(volumeIndex)], {0}, localNormal);
								const float3 normal = normalize(mInstanceTransforms[mInstanceIndex].transformVector(localNormal));
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

                    const MeshInstanceData instance = reinterpret<MeshInstanceData>(mInstances[mInstanceIndex]);

                    uint alphaMask;
                    float alphaCutoff;
					getMaterialAlphaMask(instance.getMaterialAddress(), alphaMask, alphaCutoff);
					if (alphaMask >= gImageCount)
                        break;

					const MeshVertexInfo vertexInfo = mMeshVertexInfo[instance.vertexInfoIndex()];
                    if (vertexInfo.texcoordBuffer() >= gVertexBufferCount)
                        break;

                    const uint3 tri = LoadTriangleIndices(mVertexBuffers[NonUniformResourceIndex(vertexInfo.indexBuffer())], vertexInfo.indexStride(), vertexInfo.indexStride(), rayQuery.CandidatePrimitiveIndex());
					float2 v0,v1,v2;
					LoadTriangleAttribute(mVertexBuffers[NonUniformResourceIndex(vertexInfo.texcoordBuffer())], vertexInfo.texcoordOffset(), vertexInfo.texcoordStride(), tri, v0, v1, v2);

					const float2 barycentrics = rayQuery.CandidateTriangleBarycentrics();
                    const float2 uv = v0 + (v1 - v0) * barycentrics.x + (v2 - v0) * barycentrics.y;
                    if (mImage1s[NonUniformResourceIndex(alphaMask)].SampleLevel(mStaticSampler, uv, 0) >= alphaCutoff)
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
                isect.mInstanceIndex = INVALID_INSTANCE;
                isect.mShadingData.mPosition = ray.Direction;
                isect.mShadingData.mPackedGeometryNormal = isect.mShadingData.mPackedShadingNormal = packNormal(ray.Direction);
				isect.mShadingData.mShapeArea = -1;
				return false;
			}
			case COMMITTED_TRIANGLE_HIT: {
				isect.mDistance = rayQuery.CommittedRayT();
				isect.mInstanceIndex = rayQuery.CommittedInstanceID();
				isect.mPrimitiveIndex = rayQuery.CommittedPrimitiveIndex();

                MeshInstanceData meshInstance = reinterpret<MeshInstanceData>(isect.getInstance());
                isect.mPrimitivePickPdf = 1.0 / float(meshInstance.primitiveCount());
				isect.mShadingData = makeTriangleShadingData(meshInstance, isect.getTransform(), rayQuery.CommittedPrimitiveIndex(), rayQuery.CommittedTriangleBarycentrics());
				break;
			}
			case COMMITTED_PROCEDURAL_PRIMITIVE_HIT: {
				isect.mDistance = rayQuery.CommittedRayT();
				isect.mInstanceIndex = rayQuery.CommittedInstanceID();
                isect.mPrimitiveIndex = INVALID_PRIMITIVE;
                isect.mPrimitivePickPdf = 1;
				switch (isect.getInstance().getType()) {
					case InstanceType::eSphere:
						isect.mShadingData = makeSphereShadingData(reinterpret<SphereInstanceData>(isect.getInstance()), isect.getTransform(), rayQuery.CommittedObjectRayOrigin() + rayQuery.CommittedObjectRayDirection() * rayQuery.CommittedRayT());
						break;
					case InstanceType::eVolume: {
						const uint n = isect.mShadingData.mPackedGeometryNormal; // assigned in the rayQuery loop above
						isect.mShadingData = makeVolumeShadingData(reinterpret<VolumeInstanceData>(isect.getInstance()), ray.Origin + ray.Direction * rayQuery.CommittedRayT());
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

        const float3 origin = ray.Origin;

        ray.Origin += ray.Direction * ray.TMin;
		ray.TMin = 0;

        while (ray.TMax > 1e-6) {
            const bool hit = traceRay(ray, true, /*out*/ isect);

			// delta track through current medium
			if (curMediumInstance != INVALID_INSTANCE) {
				const TransformData invTransform = mInstanceInverseTransforms[curMediumInstance];

                float3 dirPdf3, neePdf3;
                bool scattered;
                const float3 p = Medium(mInstances[curMediumInstance].getMaterialAddress()).deltaTrack<true>(
					rng,
					invTransform.transformPoint(ray.Origin),
					invTransform.transformVector(ray.Direction),
					isect.mDistance,
					/*inout*/ beta,
					/*out*/ dirPdf3,
					/*out*/ neePdf3,
					/*out*/ scattered);

				dirPdf *= average(dirPdf3);
				neePdf *= average(neePdf3);

				if (scattered) {
					// medium scattering event
                    isect.mShadingData = makeVolumeShadingData(reinterpret<VolumeInstanceData>(mInstances[curMediumInstance]), mInstanceTransforms[curMediumInstance].transformPoint(p));
                    isect.mDistance = length(isect.mShadingData.mPosition - origin);
					isect.mInstanceIndex = curMediumInstance;
                    isect.mPrimitiveIndex = INVALID_PRIMITIVE;
                    isect.mPrimitivePickPdf = 0;
					return true;
				}
			}

			if (!hit) return false; // missed scene

            if (isect.mShadingData.isSurface())
				return true; // hit surface

            if (dot(isect.mShadingData.getGeometryNormal(), ray.Direction) > 0) {
				// leaving medium
				curMediumInstance = INVALID_INSTANCE;
			} else {
				// entering medium
				curMediumInstance = isect.mInstanceIndex;
            }
			ray.Origin = rayOffset(isect.mShadingData.mPosition, isect.mShadingData.getGeometryNormal(), ray.Direction);
            ray.TMax -= isect.mDistance;
		}

        return false;
	}

    void traceVisibilityRay(RayDesc ray, inout RandomSampler rng, uint curMediumInstance, inout float3 beta, out float dirPdf, out float neePdf) {
        if (gCountRays)
            InterlockedAdd(mRayCount[1], 1);

		dirPdf = 1;
		neePdf = 1;

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
            medium = Medium(mInstances[curMediumInstance].getMaterialAddress());

        ray.Origin += ray.Direction * ray.TMin;
        ray.TMin = 0;

		while (ray.TMax > 1e-6f) {
            IntersectionResult isect;
            const bool hit = traceRay(ray, true, /*out*/ isect);

            if (hit && isect.getInstance().getType() != InstanceType::eVolume) {
				// hit a surface
				beta = 0;
				dirPdf = 0;
				neePdf = 0;
				break;
			}

			// delta track through current medium
			if (curMediumInstance != INVALID_INSTANCE) {
				const TransformData invTransform = mInstanceInverseTransforms[curMediumInstance];

                float3 dirPdf3, neePdf3;
                bool scattered;
				medium.deltaTrack<false>(
					rng,
					invTransform.transformPoint(ray.Origin),
					invTransform.transformVector(ray.Direction),
					isect.mDistance,
					/*inout*/ beta,
					/*out*/ dirPdf3,
					/*out*/ neePdf3,
					/*out*/ scattered);

				dirPdf *= average(dirPdf3);
				neePdf *= average(neePdf3);
                if (all(beta) <= 0)
					return;
			}

			if (!hit) return; // successful transmission

			// hit medium aabb
			const float3 ng = isect.mShadingData.getGeometryNormal();
			if (dot(ng, ray.Direction) < 0) {
				// entering medium
				curMediumInstance = isect.mInstanceIndex;
				medium = Medium(isect.mShadingData.getMaterialAddress());
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