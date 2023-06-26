#include "intersection.hlsli"

#ifndef gEnvironmentMaterialAddress
#define gEnvironmentMaterialAddress -1
#endif
#ifndef gEnvironmentSampleProbability
#define gEnvironmentSampleProbability 0.5
#endif
#ifndef gLightCount
#define gLightCount 0
#endif
#ifndef gSceneSphere
#define gSceneSphere float4(0)
#endif

struct EmissionSampleRecord {
    ShadingData mShadingData;
    uint mInstancePrimitiveIndex;
    float mPdf;
    bool isSingular; // unused (only area lights are implemented)

    property uint mInstanceIndex {
        get { return BF_GET(mInstancePrimitiveIndex, 0, 16); }
        set { BF_SET(mInstancePrimitiveIndex, newValue, 0, 16); }
    }
    property uint mPrimitiveIndex {
        get { return BF_GET(mInstancePrimitiveIndex, 16, 16); }
        set { BF_SET(mInstancePrimitiveIndex, newValue, 16, 16); }
    }
};

struct IlluminationSampleRecord {
    float3 mRadiance;
    float mPdf; // area-measure except for environment lights
    float3 mDirectionToLight;
    float mDistanceToLight;
    float3 mPosition;
    uint mPackedNormal;

    float mCosLight;
    bool isFinite;
    bool isSingular; // unused (only area lights are implemented)

    float3 getNormal() { return unpackNormal(mPackedNormal); }
};


extension SceneParameters {
	float2 SampleTexel(Texture2D<float4> image, float2 rnd, out float pdf, const uint maxIterations = 10) {
		uint2 imageExtent;
		uint levelCount;
		image.GetDimensions(0, imageExtent.x, imageExtent.y, levelCount);

		pdf = 1;
		int2 coord = 0;
		uint2 lastExtent = 1;
		for (uint i = 1; i < min(maxIterations + 1, levelCount - 1); i++) {
			const uint level = levelCount - 1 - i;
			uint tmp;
			uint2 extent;
			image.GetDimensions(level, extent.x, extent.y, tmp);
			const float inv_h = 1 / (float)extent.y;

			coord *= int2(extent / lastExtent);

			float4 p = 0;
			if (extent.x - coord.x > 1) {
				const float sy = sin(M_PI * (coord.y + 0.5f) * inv_h);
				p[0] = luminance(image.Load(int3(coord + int2(0, 0), (int)level)).rgb) * sy;
				p[1] = luminance(image.Load(int3(coord + int2(1, 0), (int)level)).rgb) * sy;
			}
			if (extent.y - coord.y > 1) {
				const float sy = sin(M_PI * (coord.y + 1.5f) * inv_h);
				p[2] = luminance(image.Load(int3(coord + int2(0, 1), (int)level)).rgb) * sy;
				p[3] = luminance(image.Load(int3(coord + int2(1, 1), (int)level)).rgb) * sy;
			}
			const float sum = dot(p, 1);
			if (sum < 1e-6) continue;
			p /= sum;

			for (int j = 0; j < 4; j++) {
				if (j == 3 || rnd.x < p[j]) {
					coord += int2(j % 2, j / 2);
					pdf *= p[j];
					rnd.x /= p[j];
					break;
				}
				rnd.x -= p[j];
			}

			lastExtent = extent;
		}

		pdf *= lastExtent.x * lastExtent.y;

		return (float2(coord) + rnd) / float2(lastExtent);
	}
	float SampleTexelPdf(Texture2D<float4> image, const float2 uv, const uint maxIterations = 10) {
		uint2 imageExtent;
		uint levelCount;
		image.GetDimensions(0, imageExtent.x, imageExtent.y, levelCount);

		float pdf = 1;
		uint2 lastExtent = 1;
		for (uint i = 1; i < min(maxIterations + 1, levelCount - 1); i++) {
			const uint level = levelCount - 1 - i;
			uint tmp;
			uint2 size;
            image.GetDimensions(level, size.x, size.y, tmp);
			const float inv_h = 1 / (float)size.y;

            const int2 uvi = int2(float2(size) * uv);
            const int2 coord = (uvi / 2) * 2;
            const uint2 o = min(uvi - coord, 1);

			float4 p = 0;
			if (size.x - coord.x > 1) {
				const float sy = sin(M_PI * (coord.y + 0.5f) * inv_h);
				p[0] = luminance(image.Load(int3(coord + int2(0, 0), (int)level)).rgb) * sy;
				p[1] = luminance(image.Load(int3(coord + int2(1, 0), (int)level)).rgb) * sy;
			}
			if (size.y - coord.y > 1) {
				const float sy = sin(M_PI * (coord.y + 1.5f) * inv_h);
				p[2] = luminance(image.Load(int3(coord + int2(0, 1), (int)level)).rgb) * sy;
				p[3] = luminance(image.Load(int3(coord + int2(1, 1), (int)level)).rgb) * sy;
			}
			const float sum = dot(p, 1);
			if (sum < 1e-6) continue;
			p /= sum;

			pdf *= p[o.y * 2 + o.x];

			lastExtent = size;
		}

		pdf *= lastExtent.x * lastExtent.y;

		return pdf;
	}


    // returns emission
    float3 EvaluateEnvironment(const float3 direction, out float pdfW) {
        if (gEnvironmentMaterialAddress == -1)
            return 0;

        const uint4 packedData = mMaterialData.Load<uint4>((int)gEnvironmentMaterialAddress);
        float3 emission = asfloat(packedData.rgb);
        const uint environmentImage = packedData.w;

        if (environmentImage < gImageCount) {
            const float2 uv = cartesianToSphericalUv(direction);
            emission *= mImages[environmentImage].SampleLevel(mStaticSampler, uv, 0).rgb;
            pdfW = SampleTexelPdf(mImages[environmentImage], uv) / (2 * M_PI * M_PI * sqrt(1 - direction.y * direction.y));
        } else {
            pdfW = 1 / (4 * M_PI);
        }

        if (gLightCount > 0)
        	pdfW *= gEnvironmentSampleProbability;

        return emission;
    }

    // uniformly samples a light instance and primitive index, then uniformly samples the primitive's area
    EmissionSampleRecord SampleEmission(const float4 rnd) {
        EmissionSampleRecord r;
        r.isSingular = false;
        if (gEnvironmentMaterialAddress != -1 && (gLightCount == 0 || rnd.w < gEnvironmentSampleProbability)) {
            // sample environment light
            r.mShadingData.mShapeArea = -1;
            r.mInstanceIndex = INVALID_INSTANCE;

			const uint4 packedData = mMaterialData.Load<uint4>((int)gEnvironmentMaterialAddress);
			const uint environmentImage = packedData.w;
			if (environmentImage < gImageCount) {
                r.mShadingData.mPosition = sphericalUvToCartesian(SampleTexel(mImages[environmentImage], rnd.xy, r.mPdf));
                // jacobian from sphericalUvToCartesian
                r.mPdf /= (2 * M_PI * M_PI * sqrt(1 - pow2(r.mShadingData.mPosition.y)));
            } else {
                r.mShadingData.mPosition = sampleUniformSphere(rnd.x, rnd.y);
				r.mPdf = 1 / (4 * M_PI);
			}

			if (gLightCount > 0)
				r.mPdf *= gEnvironmentSampleProbability;
			return r;
        }

        if (gLightCount == 0)
            return { 0, 0 };

        r.mInstanceIndex = mLightInstanceMap[uint(rnd.z * gLightCount) % gLightCount];
        r.mPdf = 1 / (float)gLightCount;

        if (gEnvironmentMaterialAddress != -1)
            r.mPdf *= 1 - gEnvironmentSampleProbability;

        const InstanceData instance = mInstances[r.mInstanceIndex];
        const TransformData transform = mInstanceTransforms[r.mInstanceIndex];

		if (instance.getType() == InstanceType::eMesh) {
            // triangle
            const MeshInstanceData mesh = reinterpret<MeshInstanceData>(instance);
            r.mPdf /= (float)mesh.primitiveCount();
            r.mPrimitiveIndex = uint(rnd.w * mesh.primitiveCount()) % mesh.primitiveCount();
            r.mShadingData = makeTriangleShadingData(mesh, transform, r.mPrimitiveIndex, sampleUniformTriangle(rnd.x, rnd.y));
        } else if (instance.getType() == InstanceType::eSphere) {
            // sphere
            const SphereInstanceData sphere = reinterpret<SphereInstanceData>(instance);
            r.mShadingData = makeSphereShadingData(sphere, transform, sphere.radius() * sampleUniformSphere(rnd.x, rnd.y));
        } else
            return { 0 }; // volume lights are unsupported

        r.mPdf /= r.mShadingData.mShapeArea;
        return r;
    }
    // uniformly samples a light instance and primitive index, then uniformly samples the primitive's area
    IlluminationSampleRecord SampleIllumination(const float4 rnd, const float3 referencePosition) {
        const EmissionSampleRecord emissionVertex = SampleEmission(rnd);

        IlluminationSampleRecord r;
        r.mPdf = emissionVertex.mPdf;
        r.isSingular = false;
        if (emissionVertex.mShadingData.isEnvironment()) {
			// sample environment light
			r.mDistanceToLight = POS_INFINITY;
			r.mPosition = POS_INFINITY;
			r.mCosLight = 1;
			r.isFinite = false;

            float tmp;
            r.mRadiance = EvaluateEnvironment(emissionVertex.mShadingData.mPosition, tmp);
            return r;
		}

		if (gLightCount == 0)
			return { 0, 0 };

        r.isFinite = true;

        r.mRadiance = LoadMaterial(emissionVertex.mShadingData).getEmission();
        r.mPosition = emissionVertex.mShadingData.mPosition;
        r.mPackedNormal = emissionVertex.mShadingData.mPackedShadingNormal;
		r.mDirectionToLight = r.mPosition - referencePosition;
		r.mDistanceToLight  = length(r.mDirectionToLight);
		r.mDirectionToLight /= r.mDistanceToLight;
        r.mCosLight = -dot(r.getNormal(), r.mDirectionToLight);
        if (r.mCosLight <= 0) {
			// lights are one-sided
            return { 0, 0 };
		}
        return r;
    }
}

extension IntersectionResult {
    float LightSamplePdfA() {
        if (gLightCount == 0)
            return 0;
		float pdfA = mPrimitivePickPdf / (mShadingData.mShapeArea * gLightCount);
        if (gEnvironmentMaterialAddress != -1)
            pdfA *= 1 - gEnvironmentSampleProbability;
		return pdfA;
	}
};
