#ifndef gEnableMedia
#define gEnableMedia false
#endif
#ifndef gMaxNullCollisions
#define gMaxNullCollisions 128
#endif

#ifndef gEnableAlphaTest
#define gEnableAlphaTest false
#endif

#ifndef gUseNormalMaps
#define gUseNormalMaps false
#endif

#ifndef gEnableNEE
#define gEnableNEE false
#endif

#include "compat/tinypt.h"
[[vk::push_constant]] ConstantBuffer<TinyPTPushConstants> gPushConstants;

#include "compat/scene.h"
#include "compat/environment.h"

float sampleImage(Texture2D<float> image, const float2 uv, const float uvScreenSize) {
	float w, h;
	image.GetDimensions(w, h);
	float lod = 0;
	if (uvScreenSize > 0)
		lod = log2(max(uvScreenSize * max(w, h), 1e-6f));
	return image.SampleLevel(gScene.mStaticSampler, uv, lod);
}
float4 sampleImage(Texture2D<float4> image, const float2 uv, const float uvScreenSize) {
	float w, h;
	image.GetDimensions(w, h);
	float lod = 0;
	if (uvScreenSize > 0)
		lod = log2(max(uvScreenSize * max(w, h), 1e-6f));
	return image.SampleLevel(gScene.mStaticSampler, uv, lod);
}

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

#include "materials/lambertian.hlsli"
#define Material LambertianMaterial

#include "common/intersection.hlsli"
#include "common/rng.hlsli"

struct RenderParams {
	StructuredBuffer<ViewData> mViews;
	StructuredBuffer<TransformData> mViewTransforms;

	RWTexture2D<float4> mOutput;
	RWTexture2D<float4> mAlbedo;
	RWTexture2D<float2> mPrevUVs;

	inline uint getViewIndex(const uint2 index) {
		for (uint i = 0; i < gPushConstants.mViewCount; i++)
			if (all(index >= mViews[i].mImageMin) && all(index < mViews[i].mImageMax))
				return i;
		return -1;
	}
};

ParameterBlock<SceneParameters> gScene;
ParameterBlock<RenderParams> gRenderParams;


float mis(const float a, const float b) {
    float a2 = a*a;
    return a2 / (a2 + b*b);
}

class EmissionVertex {
	float3 mPosition;
	uint mPackedGeometryNormal;
	float3 mEmission;
	float mPdfA;

	property float3 geometryNormal { get { return unpackNormal(mPackedGeometryNormal); } }
};
extension SceneParameters {
	EmissionVertex sampleEmission(inout RandomSampler rng) {
		EmissionVertex v;
		v.mEmission = 0;
		v.mPdfA = 0;
		return v;
	}
}

struct PathVertex {
	ShadingData mShadingData;
	uint mMediumInstance;

	property float3 position { get { return mShadingData.mPosition; } }
	property float3 geometryNormal { get { return mShadingData.geometryNormal; } }
	property float3 shadingNormal { get { return mShadingData.shadingNormal; } }

	float3 toLocal(const float3 v) { return mShadingData.toLocal(v); }
	float3 toWorld(const float3 v) { return mShadingData.toWorld(v); }

	float3 evaluateBsdf(BSDF bsdf, const float3 localDirIn, const float3 localDirOut, out float pdfFwd, out float pdfRev) {
		return 0;
	}

	float3 sampleDirection(BSDF bsdf, const float3 localDirIn, out float dirPdfWFwd, out float dirPdfWRev) {
		return 0;
	}

	float3 sampleDirectLight(BSDF bsdf, inout RandomSampler rng, const float3 localDirIn) {
		const EmissionVertex lightVertex = gScene.sampleEmission(rng);

		float3 toLight = lightVertex.mPosition - position;
		const float dist = length(toLight);
		toLight /= dist;

		const float cosLight = -dot(lightVertex.geometryNormal, toLight);
		if (cosLight < 0) return 0;

		const float3 localDirOut = toLocal(toLight);

		float fwdPdfW, revPdfW;
		const float3 Le = lightVertex.mEmission * evaluateBsdf(bsdf, localDirIn, localDirOut, /*out*/ fwdPdfW, /*out*/ revPdfW) * cosLight;

		// cast shadow ray
		RayDesc ray;
		ray.Origin = mShadingData.isSurface ? rayOffset(position, geometryNormal, toLight) : position;
		ray.TMin = 0;
		ray.Direction = toLight;
		ray.TMax = dist;
		float3 beta;
		float dirPdf, neePdf;
		traceVisibilityRay(rng, ray, mMediumInstance, /*out*/ beta, /*out*/ dirPdf, /*out*/ neePdf);
		if (any(beta > 1))
			return beta * Le * mis(neePdf*lightVertex.mPdfA, dirPdf*pdfWtoA(fwdPdfW, cosLight/(dist*dist))) / lightVertex.mPdfA;
		return 0;
	}
};

class PathIntegrator {
	RandomSampler mRNG;

	uint2 mOutputIndex;

	uint mBounces;
	uint mDiffuseBounces;

	PathVertex mVertex;

	float3 mBeta;

	float mDirFwdPdfW;
	float mTransmitDistance, mTransmitDirPdf, mTransmitNeePdf;
	float mShapePdfA;

	__init(const uint2 index, const PathVertex v) {
		mRNG = new RandomSampler(gPushConstants.mRandomSeed, index);

		mBounces = 0;
		mDiffuseBounces = 0;

		mVertex = v;

		mOutputIndex = index;
		mBeta = 1;

    	gRenderParams.mOutput[mOutputIndex] = float4(0, 0, 0, 1);
	}

	[mutating]
	bool russianRoullette() {
		const float rr_prob = 2*luminance(mBeta);
		if (rr_prob < 1) {
			if (mRNG.nextFloat() >= rr_prob)
				return true;
			else
				mBeta /= rr_prob;
		}
		return false;
	}

	// traces a ray in rayDirection to find the next vertex. returns true if the next vertex is valid
	[mutating]
	bool traceRay(const float3 rayDirection) {
		IntersectionResult isect;
		float3 transmitBeta;
		RayDesc ray;
		ray.Origin = rayOffset(mVertex.position, mVertex.geometryNormal, rayDirection);
		ray.TMin = 0;
		ray.TMax = POS_INFINITY;
		const bool hit = traceRay(mRNG, ray, mVertex.mMediumInstance, transmitBeta, mTransmitDirPdf, mTransmitNeePdf, isect);
		mBeta *= transmitBeta;
		if (!hit) {
			if (gPushConstants.mEnvironmentMaterialAddress != -1) {
				const Environment environment = Environment(gPushConstants.mEnvironmentMaterialAddress);
				float3 Le = environment.evaluate(rayDirection);

				// MIS with direct lighting
				if (gEnableNEE && mDirFwdPdfW > 0) {
					const float emissionPdfW = environment.evaluatePdfW(rayDirection); // TODO: light pick pdf
					Le *= mis(mTransmitDirPdf*mDirFwdPdfW, mTransmitNeePdf*emissionPdfW);
				}

				gRenderParams.mOutput[mOutputIndex].rgb += mBeta * Le;
			}

			mBeta = 0;
			return false;
		}
		mVertex.mShadingData = isect.shadingData;
		mTransmitDistance = isect.t;
		mShapePdfA = isect.shapePdfA;
		return true;
	}

	// samples direct lighting, samples a direction, and modifies mBeta
	[mutating]
	bool integrateVertex(BSDF bsdf, inout float3 dirIn, const bool storeAlbedo) {
		if (storeAlbedo)
			gRenderParams.mAlbedo[mOutputIndex] = float4(bsdf.albedo(), 1);

		// evaluate BSDF emission
		float3 Le = bsdf.emission();
		if (any(Le > 0)) {
			const float cosTheta = -dot(dirIn, mVertex.geometryNormal);
			if (cosTheta > 0) {
				// MIS with direct lighting
				if (gEnableNEE && mDirFwdPdfW > 0)
					Le *= mis(mTransmitDirPdf*pdfWtoA(mDirFwdPdfW, cosTheta/pow2(mTransmitDistance)), mTransmitNeePdf*mShapePdfA);

				gRenderParams.mOutput[mOutputIndex].rgb += mBeta * Le;
			}
		}

		if (mBounces >= gPushConstants.mMaxBounces || mDiffuseBounces > gPushConstants.mMaxDiffuseBounces || !bsdf.canEvaluate()) {
			mBeta = 0;
			return false;
		}

		if (mBounces >= gPushConstants.mMinBounces) {
			if (russianRoullette()) {
				mBeta = 0;
				return false;
			}
		}

		const float3 localDirIn = mVertex.toLocal(-dirIn);

		mBounces++;
		if (!bsdf.isSingular()) {
			mDiffuseBounces++;
			if (gEnableNEE)
				gRenderParams.mOutput[mOutputIndex].rgb += mBeta * mVertex.sampleDirectLight(bsdf, mRNG, localDirIn);
		}

		// sample BSDF
		float revPdfW;
		const float3 localDirOut = mVertex.sampleDirection(bsdf, localDirIn, /*out*/ mDirFwdPdfW, /*out*/ revPdfW);

		// eval BSDF
		float fwdPdfW;
		mBeta *= mVertex.evaluateBsdf(bsdf, localDirIn, localDirOut, /*out*/ fwdPdfW, /*out*/ revPdfW) / mDirFwdPdfW;

		dirIn = mVertex.toWorld(localDirOut);
		return true;
	}

	[mutating]
	bool step(inout float3 direction, const bool storeAlbedo) {
		if (!traceRay(direction)) {
			if (storeAlbedo)
				gRenderParams.mAlbedo[mOutputIndex] = float4(1, 1, 1, 1);
			return false;
		}

		if (mVertex.mShadingData.isSurface)
			return integrateVertex(Material(gScene, mVertex.mShadingData), /*inout*/ direction, storeAlbedo);
		else
			return integrateVertex(Medium(gScene, mVertex.mShadingData.materialAddress), /*inout*/ direction, storeAlbedo);
	}
};

PathVertex generateEyeRay(const uint viewIndex, const uint2 index, out float3 direction) {
	const ViewData view = gRenderParams.mViews[viewIndex];
	float2 uv;
	const float3 localDir = view.toWorld(index + 0.5, uv);
	const TransformData t = gRenderParams.mViewTransforms[viewIndex];
	direction = normalize(t.transformVector(localDir));

	PathVertex v;
	v.mShadingData.mPosition = float3(t.m[0][3], t.m[1][3], t.m[2][3] );
	v.mShadingData.mFlagsMaterialAddress = 0;
	v.mShadingData.mPackedGeometryNormal = v.mShadingData.mPackedShadingNormal = packNormal(t.transformVector(float3(0,0,1)));
	v.mShadingData.mShapeArea = view.mProjection.mSensorArea;
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
void sampleViewPaths(uint3 index : SV_DispatchThreadID, uint group_index : SV_GroupIndex) {
	const uint viewIndex = gRenderParams.getViewIndex(index.xy);
	if (viewIndex != -1) return;

    float3 direction;
    PathIntegrator path = new PathIntegrator(index.xy, generateEyeRay(viewIndex, index.xy, direction));

	//if (path.step(/*inout*/ direction, true)) {
	//	while (any(path.mBeta > 0))
	//		if (!path.step(/*inout*/ direction, false))
	//			break;
	//}
}