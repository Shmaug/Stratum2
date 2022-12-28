#include "common/path.hlsli"

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

float3 randomColor(const uint idx) {
	uint r = pcg(idx);
	uint g = pcg(r);
	uint b = pcg(g);
	return unorm32ToFloat(uint3(r, g, b));
}


struct PathIntegrator<let Adjoint : bool> {
	uint2 mOutputIndex;
	uint mOutputIndex1d;
	uint mOutputStride;

	uint mBounces;
	uint mDiffuseBounces;

	RandomSampler mRNG;

	PathVertex mVertex;

	float3 mDirection; // assigned in sampleDirection()
	float mDirPdfW; // assigned in sampleDirection()
	float mRevG; // assigned in step()
	float mTransmitNeePdf; // assigned in step()
	float mTransmitDistance; // assigned in step()
	float mIntersectionLightPickPdf; // assigned in step()

	[mutating]
	void initializeViewVertex(const uint viewIndex) {
		// initialize view vertex
		const ViewData view = gRenderParams.mViews[viewIndex];
		const TransformData t = gRenderParams.mViewTransforms[viewIndex];

		mVertex.mShadingData.mPosition = float3(t.m[0][3], t.m[1][3], t.m[2][3]);
		mVertex.mShadingData.mFlagsMaterialAddress = 0;
		mVertex.mShadingData.mPackedGeometryNormal = mVertex.mShadingData.mPackedShadingNormal = packNormal(t.transformVector(float3(0, 0, sign(view.mProjection.mNearPlane))));
		mVertex.mShadingData.mShapeArea = view.mProjection.mSensorArea;
		mVertex.mMediumInstance = gRenderParams.mViewMediumInstances[viewIndex];
		mVertex.mPathPdfFwdA = 1;
		mVertex.mPathPdfRevA = 1;
		mVertex.mBeta = 1;

		// generate eye ray
		float2 uv;
		mDirection = normalize(t.transformVector(view.toWorld(mOutputIndex + 0.5, /*out*/ uv)));
		mDirPdfW = 1;

		gRenderParams.mOutput[mOutputIndex] = float4(0, 0, 0, 1);
	}

	[mutating]
	void initializeLightVertex() {
		// initialize emissive vertex
		mVertex = gScene.sampleEmission(float4(mRNG.nextFloat(), mRNG.nextFloat(), mRNG.nextFloat(), mRNG.nextFloat()));

		if (CHECK_FEATURE(ReferenceBDPT))
			gRenderParams.mLightPathVertices[mOutputIndex1d] = mVertex;

		mDirection = sampleCosHemisphere(mRNG.nextFloat(), mRNG.nextFloat());
		mDirPdfW = cosHemispherePdfW(mDirection.z);
		mVertex.mPathPdfFwdA *= mDirPdfW;
		mDirection = mVertex.mShadingData.toWorld(mDirection);
	}

	__init(const uint2 index, const uint2 extent) {
		mOutputIndex = index;
		mOutputIndex1d = mOutputIndex.y * extent.x + mOutputIndex.x;
		mOutputStride = extent.x + extent.y;

		mBounces = 0;
		mDiffuseBounces = 0;

		mRNG = RandomSampler(gPushConstants.mRandomSeed, mOutputIndex);

		if (Adjoint)
			initializeLightVertex();
		else
			initializeViewVertex(getViewIndex(mOutputIndex, extent, gPushConstants.mViewCount));
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
		const float3 localToLight = mVertex.mShadingData.toLocal(toLight);

		MaterialEvalRecord r = bsdf.evaluate<Adjoint>(localDirIn, localToLight);
		r.mReflectance *= mVertex.mShadingData.correctShadingNormal<Adjoint>(localDirIn, localToLight);
		if (r.mFwdPdfW <= 0 || all(r.mReflectance <= 0))
			return;

		// trace shadow ray
		RayDesc ray;
		ray.Origin = mVertex.mShadingData.isSurface ? rayOffset(mVertex.mShadingData.mPosition, mVertex.mShadingData.geometryNormal, toLight) : mVertex.mShadingData.mPosition;
		ray.TMin = 0;
		ray.Direction = toLight;
		ray.TMax = dist*0.99;

		float3 transmitBeta;
		float transmitDirPdf, transmitNeePdf;
		gScene.traceVisibilityRay(ray, mRNG, mVertex.mMediumInstance, /*out*/ transmitBeta, /*out*/ transmitDirPdf, /*out*/ transmitNeePdf);
		if (all(transmitBeta <= 0))
			return;

		const float w = neeMis(pdfA * transmitNeePdf, pdfWtoA(r.mFwdPdfW, G) * transmitDirPdf);

		addContribution(mVertex.mBeta * transmitBeta * G * r.mReflectance * L / pdfA, w, mBounces + 2, 1);
	}
	[mutating]
	void sampleDirectLight(BSDF bsdf, const float3 localDirIn) {
		const PathVertex lightVertex = gScene.sampleEmission(float4(mRNG.nextFloat(), mRNG.nextFloat(), mRNG.nextFloat(), mRNG.nextFloat()));
		if (lightVertex.mPathPdfFwdA <= 0 || all(lightVertex.mBeta <= 0))
			return;

		// environment emission
		if (lightVertex.mShadingData.isEnvironment) {
			evalDirectLight(bsdf, localDirIn, lightVertex.mBeta, 1, lightVertex.mShadingData.mPosition, POS_INFINITY, lightVertex.mPathPdfFwdA);
			return;
		}

		float3 toLight = lightVertex.mShadingData.mPosition - mVertex.mShadingData.mPosition;
		const float dist = length(toLight);
		toLight /= dist;

		const float cosLight = -dot(lightVertex.mShadingData.geometryNormal, toLight);
		if (cosLight < 0)
			return;

		const float G = cosLight / (dist * dist);
		evalDirectLight(bsdf, localDirIn, lightVertex.mBeta, G, toLight, dist, lightVertex.mPathPdfFwdA);
	}

	[mutating]
	bool sampleDirection(BSDF bsdf, const float3 localDirIn) {
		// sample BSDF
		const MaterialSampleRecord r = bsdf.sample<Adjoint>(float3(mRNG.nextFloat(), mRNG.nextFloat(), mRNG.nextFloat()), localDirIn);
		if (r.mFwdPdfW <= 0) {
			mVertex.mBeta = 0;
			return false;
		}

		// eval BSDF
		mVertex.mBeta *= bsdf.evaluate<Adjoint>(localDirIn, r.mDirection).mReflectance / r.mFwdPdfW;
		mVertex.mBeta *= mVertex.mShadingData.correctShadingNormal<Adjoint>(localDirIn, r.mDirection);
		mVertex.mPathPdfFwdA *= r.mFwdPdfW;
		mVertex.mPathPdfRevA *= pdfWtoA(r.mRevPdfW, mRevG);

		mDirection = mVertex.mShadingData.toWorld(r.mDirection);
		mDirPdfW = r.mFwdPdfW;
		return true;
	}


	void storeAuxillaryData(const RayDesc ray, const bool hit, const IntersectionResult isect) {
		addDebugContribution(PathTracerDebugMode::eDepth,             mTransmitDistance.xxx / 10);
		addDebugContribution(PathTracerDebugMode::eGeometryNormal,    mVertex.mShadingData.geometryNormal * .5 + .5);
		addDebugContribution(PathTracerDebugMode::eShadingNormal,     mVertex.mShadingData.shadingNormal * .5 + .5);
		addDebugContribution(PathTracerDebugMode::eTextureCoordinate, float3(mVertex.mShadingData.mTexcoord, 0));
		addDebugContribution(PathTracerDebugMode::eMaterialAddress, randomColor(mVertex.mShadingData.materialAddress));

		VisibilityData v;
		v.mInstancePrimitiveIndex = isect.mInstancePrimitiveIndex;
		v.mPackedNormal = mVertex.mShadingData.mPackedShadingNormal;
		gRenderParams.mVisibility[mOutputIndex] = reinterpret<float2>(v);

		DepthData d;
		d.mDepth = mTransmitDistance;

		if (hit) {
			uint2 extent;
			gRenderParams.mOutput.GetDimensions(extent.x, extent.y);
			const uint viewIndex = getViewIndex(mOutputIndex, extent, gPushConstants.mViewCount);

			const float3 prevWorldPos = gScene.mInstanceMotionTransforms[isect.instanceIndex].transformPoint(mVertex.mShadingData.mPosition);
			const float3 prevViewPos  = gRenderParams.mPrevInverseViewTransforms[viewIndex].transformPoint(prevWorldPos);
			d.mPrevDepth = length(prevViewPos);

			const ViewData view = gRenderParams.mViews[viewIndex];
			const TransformData t = gRenderParams.mViewTransforms[viewIndex];
			const float3 ng = mVertex.mShadingData.geometryNormal;
			float2 uv;
			d.mDepthDerivative.x = rayPlane(ray.Origin - mVertex.mShadingData.mPosition, normalize(t.transformVector(view.toWorld(mOutputIndex + 0.5 + uint2(1, 0), /*out*/ uv))), ng) - mTransmitDistance;
			d.mDepthDerivative.y = rayPlane(ray.Origin - mVertex.mShadingData.mPosition, normalize(t.transformVector(view.toWorld(mOutputIndex + 0.5 + uint2(0, 1), /*out*/ uv))), ng) - mTransmitDistance;

			gRenderParams.mPrevViews[viewIndex].toRaster(prevViewPos, /*out*/ gRenderParams.mPrevUVs[mOutputIndex]);
			addDebugContribution(PathTracerDebugMode::ePrevUV, float3(gRenderParams.mPrevUVs[mOutputIndex], 0));
		} else {
			d.mDepthDerivative = 0;
			d.mPrevDepth = POS_INFINITY;
		}

		gRenderParams.mDepth[mOutputIndex] = reinterpret<float4>(d);
	}


	// stores auxillary data, stores path vertex, samples direct lighting and next direction, does russian roullette
	[mutating]
	void integrateVertex<let StoreAlbedo : bool>(BSDF bsdf) {
		if (StoreAlbedo)
			gRenderParams.mAlbedo[mOutputIndex] = float4(bsdf.albedo(), 1);

		// evaluate BSDF emission
		if (!Adjoint && !CHECK_FEATURE(ReferenceBDPT)) {
			const float3 L = bsdf.emission();
			if (any(L > 0)) {
				const float cosTheta = mVertex.mShadingData.isSurface ? -dot(mDirection, mVertex.mShadingData.geometryNormal) :  1;
				if (cosTheta > 0) {
					float w = 1;

					if (mDirPdfW > 0 && mBounces > 0 && CHECK_FEATURE(Nee)) {
						// MIS with NEE

						// compute probability of sampling this point with NEE
						float lightPdfA;
						if (mVertex.mShadingData.isEnvironment) {
							if (gPushConstants.mLightCount == 0) {
								lightPdfA = 1;
							} else {
								lightPdfA = gPushConstants.mEnvironmentSampleProbability;
							}
						} else {
							if (gPushConstants.mLightCount == 0) {
								lightPdfA = 0;
							} else {
								lightPdfA = mIntersectionLightPickPdf / (mVertex.mShadingData.mShapeArea * gPushConstants.mLightCount);
								if (gHasEnvironment)
									lightPdfA *= 1 - gPushConstants.mEnvironmentSampleProbability;
							}
						}

						const float G = mVertex.mShadingData.isEnvironment ? 1 : (cosTheta / pow2(mTransmitDistance));
						w = neeMis(pdfWtoA(mDirPdfW, G), lightPdfA * bsdf.emissionPdf() * mTransmitNeePdf);
					}

					addContribution(mVertex.mBeta * L, w, mBounces + 2, 0);
				}
			}
		}

		if (!bsdf.canEvaluate() || mBounces >= gPushConstants.mMaxBounces || mDiffuseBounces >= gPushConstants.mMaxDiffuseBounces) {
			mVertex.mBeta = 0;
			return;
		}

		if (mBounces >= gPushConstants.mMinBounces)
			if (mVertex.russianRoullette(mRNG.nextFloat()))
				return;

		const bool singular = bsdf.isSingular();

		if (!singular && CHECK_FEATURE(ReferenceBDPT)) {
			if (Adjoint)
				gRenderParams.mLightPathVertices[mOutputIndex1d + (mDiffuseBounces+1) * mOutputStride] = mVertex;
			else
				gRenderParams.mViewPathVertices[mOutputIndex1d + mDiffuseBounces * mOutputStride] = mVertex;
		}

		const float3 localDirIn = mVertex.mShadingData.toLocal(-mDirection);

		if (!singular && !Adjoint && CHECK_FEATURE(Nee) && !CHECK_FEATURE(ReferenceBDPT))
			sampleDirectLight(bsdf, localDirIn);

		if (sampleDirection(bsdf, localDirIn)) {
			mBounces++;
			if (!singular)
				mDiffuseBounces++;
		}
	}

	// traces a ray in mDirection to find the next vertex, then samples the next direction
	// returns true if the next vertex is valid
	[mutating]
	void step<let StoreAuxillaryData : bool>() {
		if (mVertex.mShadingData.isSurface)
			mRevG = dot(mVertex.mShadingData.geometryNormal, mDirection);
		else
			mRevG = 1;

		RayDesc ray;
		ray.Origin = rayOffset(mVertex.mShadingData.mPosition, mVertex.mShadingData.geometryNormal, mDirection);
		ray.TMin = 0;
		ray.Direction = mDirection;
		ray.TMax = POS_INFINITY;

		IntersectionResult isect;
		float transmitDirPdf;
		const bool hit = gScene.traceScatteringRay(
			ray,
			mRNG,
			mVertex.mMediumInstance,
			/*inout*/ mVertex.mBeta,
			/*out*/ transmitDirPdf,
			/*out*/ mTransmitNeePdf,
			/*out*/ isect,
			/*out*/ mVertex.mShadingData);
		mVertex.mInstancePrimitiveIndex = isect.mInstancePrimitiveIndex;
		mVertex.directionIn = -mDirection;
		mTransmitDistance = isect.mDistance;
		mIntersectionLightPickPdf = isect.mLightPickPdf;

		mDirPdfW *= transmitDirPdf;
		mVertex.mPathPdfFwdA *= transmitDirPdf;
		// convert bsdf sample pdf to area measure
		mVertex.mPathPdfFwdA *= abs(dot(mVertex.mShadingData.geometryNormal, mDirection)) / pow2(mTransmitDistance);
		mRevG /= pow2(mTransmitDistance);

		// TODO: transmit/scattering reverse pdf?

		mVertex.mShadingData.mTexcoordScreenSize = 0;

		if (StoreAuxillaryData && !Adjoint)
			storeAuxillaryData(ray, hit, isect);

		if (mVertex.mShadingData.isSurface)
			integrateVertex<StoreAuxillaryData>(Material(mVertex.mShadingData));
		else if (mVertex.mShadingData.isMedium)
			integrateVertex<StoreAuxillaryData>(Medium(mVertex.mShadingData.materialAddress));
		else if (!hit)
			integrateVertex<StoreAuxillaryData>(EnvironmentBSDF(mVertex.mShadingData));
		else
			mVertex.mBeta = 0;
	}

	[mutating]
	void integrate() {
		step<!Adjoint>();
		while (any(mVertex.mBeta > 0) && !any(isnan(mVertex.mBeta)))
			step<false>();
	}
};


void connectPaths(const uint2 index, const uint2 extent, const uint viewVertexCount, const uint lightVertexCount) {
	const uint index1d = index.y * extent.x + index.x;
	const uint stride = extent.x * extent.y;

	gRenderParams.mOutput[index] = float4(0, 0, 0, 1);

	for (uint curVertexCount = 1; curVertexCount <= viewVertexCount + lightVertexCount; curVertexCount++) {
		float4 contribution = 0;

		// consider all techniques which result in paths with curVertexCount vertices
		for (uint t = 0; t < min(lightVertexCount, curVertexCount); t++) {
			const uint s = curVertexCount - t;
			if (s == 0 || s >= viewVertexCount)
				continue;

			PathVertex viewVertex  = gRenderParams.mViewPathVertices [index1d + stride * (s-1)];
			PathVertex lightVertex = gRenderParams.mLightPathVertices[index1d + stride * t];

			float3 toLight = lightVertex.mShadingData.mPosition - viewVertex.mShadingData.mPosition;
			const float dist = length(toLight);
			toLight /= dist;

			const float3 c =
				viewVertex.mBeta * viewVertex.evaluateBsdf<false>(viewVertex.directionIn, toLight) *
				(1 / pow2(dist)) *
				lightVertex.mBeta * lightVertex.evaluateBsdf<true>(lightVertex.directionIn, -toLight);

			const float w = mis(viewVertex.mPathPdfFwdA * lightVertex.mPathPdfFwdA);

			if (all(c <= 0) || any(isnan(c)) || w <= 0 || isnan(w))
				continue;

			contribution += float4(c * w, w);
		}

		gRenderParams.mOutput[index].rgb += contribution.rgb / contribution.w;
	}
}


[shader("compute")]
[numthreads(8, 8, 1)]
void integratePaths(uint3 index: SV_DispatchThreadID) {
	uint2 extent;
	gRenderParams.mOutput.GetDimensions(extent.x, extent.y);
	if (any(index.xy >= extent)) return;

	uint viewVertexCount, lightVertexCount;

	{
		PathIntegrator<false> path = PathIntegrator<false>(index.xy, extent);
		path.integrate();
		viewVertexCount = path.mDiffuseBounces + 1;
	}

	if (CHECK_FEATURE(ReferenceBDPT)) {
		{
			PathIntegrator<true> path = PathIntegrator<true>(index.xy, extent);
			path.integrate();
			lightVertexCount = path.mDiffuseBounces + 2;
		}

		connectPaths(index.xy, extent, viewVertexCount, lightVertexCount);
	}
}

#if 0
[shader("compute")]
[numthreads(8, 8, 1)]
void bdpt(uint3 index: SV_DispatchThreadID) {
	uint2 extent;
	gRenderParams.mOutput.GetDimensions(extent.x, extent.y);
	if (any(index.xy >= extent)) return;

	// initialize view vertex
	const ViewData view = gRenderParams.mViews[viewIndex];
	const TransformData t = gRenderParams.mViewTransforms[viewIndex];

	const float3 position = float3(t.m[0][3], t.m[1][3], t.m[2][3]);
	const float3 sensorNormal = packNormal(t.transformVector(float3(0, 0, sign(view.mProjection.mNearPlane))));
	uint curMedium = gRenderParams.mViewMediumInstances[viewIndex];

	// generate eye ray
	float2 uv;
	const float3 direction = normalize(t.transformVector(view.toWorld(mOutputIndex + 0.5, /*out*/ uv)));

	float3 beta = 1;
	float3 radiance = 0;

	for (uint vertexCount = 1; vertexCount < gPushConstants.mMaxBounces + 1; vertexCount++) {

	}
	gRenderParams.mOutput[index.xy] = float4(radiance, 1);
}
#endif