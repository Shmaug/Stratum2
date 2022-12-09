#ifndef gMaxNullCollisions
#define gMaxNullCollisions 128
#endif

#ifndef gDebugMode
#define gDebugMode PathTracerDebugMode::eNone
#endif
#ifndef gFeatureFlags
#define gFeatureFlags 0
#endif
#define CHECK_FEATURE(feature) (bool)(gFeatureFlags & BIT((uint)PathTracerFeatureFlagBits::e ## feature ))

#include "compat/scene.h"
ParameterBlock<SceneParameters> gScene;

#include "compat/path_tracer.h"
[[vk::push_constant]] ConstantBuffer<PathTracerPushConstants> gPushConstants;

#include "common/intersection.hlsli"
#include "common/rng.hlsli"
#include "compat/environment.h"
#include "materials/disney.hlsli"

#define Material DisneyMaterial

struct RenderParams {
	StructuredBuffer<ViewData> mViews;
	StructuredBuffer<TransformData> mViewTransforms;
	StructuredBuffer<uint> mViewMediumInstances;

	StructuredBuffer<ViewData> mPrevViews;
	StructuredBuffer<TransformData> mPrevInverseViewTransforms;

	RWTexture2D<float4> mOutput;
	RWTexture2D<float4> mAlbedo;
	RWTexture2D<float2> mPrevUVs;
	RWTexture2D<float2> mVisibility;
	RWTexture2D<float4> mDepth;
};

ParameterBlock<RenderParams> gRenderParams;


float3 randomColor(const uint idx) {
	uint r = pcg(idx);
	uint g = pcg(r);
	uint b = pcg(g);
	return unorm32ToFloat(uint3(r, g, b));
}

float mis(const float a) {
	return a * a;
}
float mis(const float a, const float b) {
	const float a2 = mis(a);
	return a2 / (a2 + mis(b));
}
float neeMis(const float a, const float b) {
	if (CHECK_FEATURE(NeeMis))
		return mis(a, b);
	else
		return 0.5;
}


struct EmissionVertex {
	float3 mPosition;
	uint mPackedGeometryNormal;
	float3 mEmission;
	float mPdfA;
	bool mIsEnvironment;

	property float3 geometryNormal { get { return unpackNormal(mPackedGeometryNormal); } }
};

extension SceneParameters {
	EmissionVertex sampleEmission(const float4 rnd) {
		if (gPushConstants.mEnvironmentMaterialAddress != -1) {
			if (gPushConstants.mLightCount == 0 || rnd.w < gPushConstants.mEnvironmentSampleProbability) {
				EmissionVertex v;
				v.mEmission = Environment(gPushConstants.mEnvironmentMaterialAddress).sample(rnd.xy, v.mPosition, v.mPdfA);
				if (gPushConstants.mLightCount > 0)
					v.mPdfA *= gPushConstants.mEnvironmentSampleProbability;
				v.mIsEnvironment = true;
				return v;
			}
		}

		if (gPushConstants.mLightCount == 0)
			return { 0, 0, 0, 0 };

        float pickPdf = 1 / (float)gPushConstants.mLightCount;

		if (gPushConstants.mEnvironmentMaterialAddress != -1)
			pickPdf *= 1 - gPushConstants.mEnvironmentSampleProbability;

        const uint lightInstanceIndex = mLightInstanceMap[uint(rnd.z * gPushConstants.mLightCount) % gPushConstants.mLightCount];
		const InstanceData instance = mInstances[lightInstanceIndex];
		const TransformData transform = mInstanceTransforms[lightInstanceIndex];

		ShadingData shadingData;

		if (instance.type() == InstanceType::eMesh) {
			const MeshInstanceData mesh = reinterpret<MeshInstanceData>(instance);
			shadingData = makeTriangleShadingData(mesh, transform, uint(rnd.w * mesh.primitiveCount()) % mesh.primitiveCount(), sampleUniformTriangle(rnd.x, rnd.y));
			pickPdf /= mesh.primitiveCount();
		} else if (instance.type() == InstanceType::eSphere) {
			const SphereInstanceData sphere = reinterpret<SphereInstanceData>(instance);
			shadingData = makeSphereShadingData(sphere, transform, sphere.radius() * sampleUniformSphereCartesian(rnd.x, rnd.y));
		} else
			return { 0, 0, 0, 0 };

		EmissionVertex v;
		v.mPosition = shadingData.mPosition;
		v.mPackedGeometryNormal = shadingData.mPackedGeometryNormal;
		v.mEmission = Material(shadingData).emission();
		v.mPdfA = pickPdf / shadingData.mShapeArea;
		v.mIsEnvironment = false;
		return v;
	}
}

struct PathVertex {
	ShadingData mShadingData;
	uint mMediumInstance;

	property float3 position       { get { return mShadingData.mPosition; } }
	property float3 geometryNormal { get { return mShadingData.geometryNormal; } }
	property float3 shadingNormal  { get { return mShadingData.shadingNormal; } }

	float3 toLocal(const float3 v) { return mShadingData.toLocal(v); }
	float3 toWorld(const float3 v) { return mShadingData.toWorld(v); }
};

struct PathIntegrator<let Adjoint : bool> {
	RandomSampler mRNG;

	uint2 mOutputIndex;

	uint mBounces;
	uint mDiffuseBounces;


	PathVertex mVertex;

	float3 mBeta;
	float3 mDirection;
	float mDirFwdPdfW;
	float mTransmitDistance, mTransmitDirPdf, mTransmitNeePdf;
	float mSurfaceEmissionPdfA; // probability of sampling the current point when sampling emission

	__init(const uint2 index, const PathVertex v, const float3 direction) {
		mRNG = RandomSampler(gPushConstants.mRandomSeed, index);

		mOutputIndex = index;

		mBounces = 0;
		mDiffuseBounces = 0;
		mBeta = 1;

		mVertex = v;
		mDirection = direction;
		mDirFwdPdfW = 0;

		gRenderParams.mOutput[mOutputIndex] = float4(0, 0, 0, 1);
	}

	void addDebugContribution(const PathTracerDebugMode debugMode, const float3 contrib) {
		if (gDebugMode == debugMode)
			gRenderParams.mOutput[mOutputIndex].rgb += contrib;
	}
	void addContribution(const float3 contrib, const float w, const uint viewVertices, const uint lightVertices) {
		if (gDebugMode != PathTracerDebugMode::ePathTypeContribution) {
			gRenderParams.mOutput[mOutputIndex].rgb += contrib * w;
			return;
		}
		if (viewVertices == gPushConstants.mDebugViewVertices && lightVertices == gPushConstants.mDebugLightVertices)
			gRenderParams.mOutput[mOutputIndex].rgb += contrib;
	}


	[mutating]
	void evalDirectLight(BSDF bsdf, const float3 localDirIn, const float3 L, const float G, const float3 toLight, const float dist, const float pdfA) {
		const MaterialEvalRecord r = bsdf.evaluate<Adjoint>(localDirIn, mVertex.toLocal(toLight));
		if (r.mFwdPdfW <= 0 || all(r.mReflectance <= 0))
			return;

		// trace shadow ray
		RayDesc ray;
		ray.Origin = mVertex.mShadingData.isSurface ? rayOffset(mVertex.position, mVertex.geometryNormal, toLight) : mVertex.position;
		ray.TMin = 0;
		ray.Direction = toLight;
		ray.TMax = dist*0.99;

		float3 transmitBeta;
		float transmitDirPdf, transmitNeePdf;
		gScene.traceVisibilityRay(mRNG, ray, mVertex.mMediumInstance, /*out*/ transmitBeta, /*out*/ transmitDirPdf, /*out*/ transmitNeePdf);
		if (all(transmitBeta <= 0))
			return;

		const float w = neeMis(pdfA * transmitNeePdf, pdfWtoA(r.mFwdPdfW, G) * transmitDirPdf);

		addContribution(mBeta * transmitBeta * G * r.mReflectance * L / pdfA, w, mBounces + 1, 1);
	}
	[mutating]
    void sampleDirectLight(BSDF bsdf, const float3 localDirIn) {
        const EmissionVertex lightVertex = gScene.sampleEmission(float4(mRNG.nextFloat(), mRNG.nextFloat(), mRNG.nextFloat(), mRNG.nextFloat()));
		if (lightVertex.mPdfA <= 0 || all(lightVertex.mEmission <= 0))
			return;

		if (lightVertex.mIsEnvironment) {
			evalDirectLight(bsdf, localDirIn, lightVertex.mEmission, 1, lightVertex.mPosition, POS_INFINITY, lightVertex.mPdfA);
			return;
		}

		float3 toLight = lightVertex.mPosition - mVertex.position;
		const float dist = length(toLight);
		toLight /= dist;

		const float cosLight = -dot(lightVertex.geometryNormal, toLight);
		if (cosLight < 0)
			return;

		const float G = cosLight / (dist * dist);
		evalDirectLight(bsdf, localDirIn, lightVertex.mEmission, G, toLight, dist, lightVertex.mPdfA);
	}


	[mutating]
	void addEmission(const float3 L, const float dirPdf, const float lightPdf) {
		float w;
		if (mDirFwdPdfW > 0 && CHECK_FEATURE(Nee))
			// MIS with NEE
			w = neeMis(dirPdf * mTransmitDirPdf, lightPdf * mTransmitNeePdf);
		else
			w = 1;

		addContribution(mBeta * L, w, mBounces + 2, 0);
	}

	[mutating]
	bool sampleDirection(BSDF bsdf, const float3 localDirIn) {
		// sample BSDF
		const MaterialSampleRecord r = bsdf.sample<Adjoint>(float3(mRNG.nextFloat(), mRNG.nextFloat(), mRNG.nextFloat()), localDirIn);
		if (r.mFwdPdfW <= 0)
			return false;

		// eval BSDF
		mBeta *= bsdf.evaluate<Adjoint>(localDirIn, r.mDirection).mReflectance / r.mFwdPdfW;

		mDirection = mVertex.toWorld(r.mDirection);
		mDirFwdPdfW = r.mFwdPdfW;
		return true;
	}


	[mutating]
	bool russianRoullette() {
		const float rr_prob = 2*luminance(mBeta);
        if (rr_prob < 1) {
            if (mRNG.nextFloat() >= rr_prob) {
                mBeta = 0;
                return true;
            } else
				mBeta /= rr_prob;
		}
		return false;
	}

	// samples direct lighting, samples mDirection and mBeta
	[mutating]
	bool integrateVertex<let StoreAlbedo : bool>(BSDF bsdf) {
		if (StoreAlbedo)
			gRenderParams.mAlbedo[mOutputIndex] = float4(bsdf.albedo(), 1);

		// evaluate BSDF emission
		const float3 L = bsdf.emission();
		if (any(L > 0)) {
			const float cosTheta = mVertex.mShadingData.isSurface ? -dot(mDirection, mVertex.geometryNormal) : 1;
			if (cosTheta > 0) {
				const float dirPdfA = pdfWtoA(mDirFwdPdfW, cosTheta / pow2(mTransmitDistance));
				addEmission(L, dirPdfA, mSurfaceEmissionPdfA);
			}
		}

        if (!bsdf.canEvaluate() || mBounces >= gPushConstants.mMaxBounces) {
			mBeta = 0;
			return false;
		}

		if (mBounces >= gPushConstants.mMinBounces) {
			if (russianRoullette())
				return false;
		}

		const float3 localDirIn = mVertex.toLocal(-mDirection);

		mBounces++;

		if (!bsdf.isSingular()) {
			if (mDiffuseBounces >= gPushConstants.mMaxDiffuseBounces) {
				mBeta = 0;
				return false;
			}

			mDiffuseBounces++;

			if (!Adjoint && CHECK_FEATURE(Nee))
				sampleDirectLight(bsdf, localDirIn);
		}

		return sampleDirection(bsdf, localDirIn);
	}


	// traces a ray in mDirection to find the next vertex, then samples the next direction
	// returns true if the next vertex is valid
	[mutating]
	bool step<let StoreAuxillaryData : bool>() {
		RayDesc ray;
		ray.Origin = rayOffset(mVertex.position, mVertex.geometryNormal, mDirection);
		ray.TMin = 0;
		ray.Direction = mDirection;
		ray.TMax = POS_INFINITY;

		IntersectionResult isect;
		float3 transmitBeta;
		const bool hit = gScene.traceScatteringRay(mRNG, ray, mVertex.mMediumInstance, transmitBeta, mTransmitDirPdf, mTransmitNeePdf, isect, mVertex.mShadingData);

		mBeta *= transmitBeta;
		mTransmitDistance = isect.mDistance;

		if (gPushConstants.mLightCount > 0) {
			mSurfaceEmissionPdfA = isect.mLightPickPdf / (mVertex.mShadingData.mShapeArea * gPushConstants.mLightCount);
			if (gPushConstants.mEnvironmentMaterialAddress != -1)
				mSurfaceEmissionPdfA *= 1 - gPushConstants.mEnvironmentSampleProbability;
		} else
			mSurfaceEmissionPdfA = 0;


		// store meta information
		if (!Adjoint && StoreAuxillaryData) {
			addDebugContribution(PathTracerDebugMode::eDepth,             mTransmitDistance.xxx / 10);
			addDebugContribution(PathTracerDebugMode::eGeometryNormal,    mVertex.mShadingData.geometryNormal * .5 + .5);
			addDebugContribution(PathTracerDebugMode::eShadingNormal,     mVertex.mShadingData.shadingNormal * .5 + .5);
			addDebugContribution(PathTracerDebugMode::eTextureCoordinate, float3(mVertex.mShadingData.mTexcoord, 0));
			addDebugContribution(PathTracerDebugMode::eMaterialAddress,   randomColor(mVertex.mShadingData.materialAddress));

			VisibilityData v;
			v.mInstancePrimitiveIndex = isect.mInstancePrimitiveIndex;
			v.mPackedNormal = mVertex.mShadingData.mPackedShadingNormal;
			gRenderParams.mVisibility[mOutputIndex] = reinterpret<float2>(v);

			if (hit) {
				uint2 extent;
				gRenderParams.mOutput.GetDimensions(extent.x, extent.y);
				const uint viewIndex = getViewIndex(mOutputIndex, extent, gPushConstants.mViewCount);

				const float3 prevWorldPos = gScene.mInstanceMotionTransforms[isect.instanceIndex].transformPoint(mVertex.position);
				const float3 prevViewPos  = gRenderParams.mPrevInverseViewTransforms[viewIndex].transformPoint(prevWorldPos);

				gRenderParams.mPrevViews[viewIndex].toRaster(prevViewPos, /*out*/ gRenderParams.mPrevUVs[mOutputIndex]);
				addDebugContribution(PathTracerDebugMode::ePrevUV, float3(gRenderParams.mPrevUVs[mOutputIndex], 0));

				DepthData d;
				d.mDepth = mTransmitDistance;
				d.mPrevDepth = length(prevViewPos);

				const ViewData view = gRenderParams.mViews[viewIndex];
				const TransformData t = gRenderParams.mViewTransforms[viewIndex];
				const float3 ng = mVertex.geometryNormal;
				float2 uv;
				d.mDepthDerivative.x = rayPlane(ray.Origin - mVertex.position, normalize(t.transformVector(view.toWorld(mOutputIndex + 0.5 + uint2(1, 0), /*out*/ uv))), ng) - mTransmitDistance;
				d.mDepthDerivative.y = rayPlane(ray.Origin - mVertex.position, normalize(t.transformVector(view.toWorld(mOutputIndex + 0.5 + uint2(0, 1), /*out*/ uv))), ng) - mTransmitDistance;

				gRenderParams.mDepth[mOutputIndex] = reinterpret<float4>(d);
			}
		}

		mVertex.mShadingData.mTexcoordScreenSize = 0;

		if (hit) {
			if (mVertex.mShadingData.isSurface)
				return integrateVertex<StoreAuxillaryData>(Material(mVertex.mShadingData));
			else if (mVertex.mShadingData.isMedium)
				return integrateVertex<StoreAuxillaryData>(Medium(mVertex.mShadingData.materialAddress));
			else
				return false;
		}

		// hit background
		if (!Adjoint) {
			if (StoreAuxillaryData)
				gRenderParams.mAlbedo[mOutputIndex] = float4(1, 1, 1, 1);

			if (gPushConstants.mEnvironmentMaterialAddress != -1) {
				const Environment environment = Environment(gPushConstants.mEnvironmentMaterialAddress);
				const float3 L = environment.evaluate(mDirection);
				const float pickPdf = gPushConstants.mLightCount > 0 ? gPushConstants.mEnvironmentSampleProbability : 1;
				addEmission(L, mDirFwdPdfW, environment.evaluatePdfW(mDirection) * pickPdf);
			}
		}

		mBeta = 0;
		return false;
	}
};

PathVertex generateEyeRay(const uint viewIndex, const uint2 index, out float3 direction) {
	const ViewData view = gRenderParams.mViews[viewIndex];
	float2 uv;
	const float3 localDir = view.toWorld(index + 0.5, /*out*/ uv);
	const TransformData t = gRenderParams.mViewTransforms[viewIndex];
	direction = normalize(t.transformVector(localDir));

	PathVertex v;
	v.mShadingData.mPosition = float3(t.m[0][3], t.m[1][3], t.m[2][3]);
	v.mShadingData.mFlagsMaterialAddress = 0;
	v.mShadingData.mPackedGeometryNormal = v.mShadingData.mPackedShadingNormal = packNormal(t.transformVector(float3(0,0,1)));
	v.mShadingData.mShapeArea = view.mProjection.mSensorArea;
	v.mMediumInstance = gRenderParams.mViewMediumInstances[viewIndex];
	return v;
}

/*
uint mapPixelCoord(const uint2 pixel_coord, const uint2 groupId, const uint groupThreadIndex) {
	const uint dispatch_w = (gOutputExtent.x + GROUPSIZE_X - 1) / GROUPSIZE_X;
	const uint group_index = groupId.y*dispatch_w + groupId.x;
	return group_index*GROUPSIZE_X*GROUPSIZE_Y + groupThreadIndex;
}
*/

[shader("compute")]
[numthreads(8, 8, 1)]
void sampleViewPaths(uint3 index: SV_DispatchThreadID, uint group_index: SV_GroupIndex) {
	uint2 extent;
	gRenderParams.mOutput.GetDimensions(extent.x, extent.y);
	if (any(index.xy >= extent)) return;

	float3 direction;
	PathVertex v = generateEyeRay(getViewIndex(index.xy, extent, gPushConstants.mViewCount), index.xy, /*out*/ direction);

	PathIntegrator<false> path = PathIntegrator<false>(index.xy, v, direction);

	if (!path.step<true>()) return;
	for (uint i = 0; i < 8; i++)
		if (!path.step<false>())
			return;
}