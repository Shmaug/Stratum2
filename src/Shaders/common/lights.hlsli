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

struct IlluminationSampleRecord {
    float3 mRadiance;
    float mPdf; // area-measure except for environment lights
    float3 mDirectionToLight;
    float mDistanceToLight;
    float3 mPosition;
    uint mPackedNormal;

    float mCosLight;
    bool isFinite;
    bool isSingular;

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
    IlluminationSampleRecord SampleIllumination(const float4 rnd, const Optional<float3> referencePosition = none) {
        IlluminationSampleRecord r;
        r.isSingular = false;
        if (gEnvironmentMaterialAddress != -1) {
            if (gLightCount == 0 || rnd.w < gEnvironmentSampleProbability) {
				// sample environment light
                r.mDistanceToLight = POS_INFINITY;
                r.mPosition = POS_INFINITY;
                r.mCosLight = 1;
                r.isFinite = false;

                const uint4 packedData = mMaterialData.Load<uint4>((int)gEnvironmentMaterialAddress);
                r.mRadiance = asfloat(packedData.rgb);
                const uint environmentImage = packedData.w;

                if (environmentImage < gImageCount) {
                    const float2 uv = SampleTexel(mImages[environmentImage], rnd.xy, r.mPdf);
                    r.mRadiance *= mImages[environmentImage].SampleLevel(mStaticSampler, uv, 0).rgb;
                    r.mDirectionToLight = sphericalUvToCartesian(uv);
                    r.mPdf /= (2 * M_PI * M_PI * sqrt(1 - r.mDirectionToLight.y * r.mDirectionToLight.y));
                } else {
                    r.mDirectionToLight = sampleUniformSphere(rnd.x, rnd.y);
                    r.mPdf = 1 / (4 * M_PI);
                }
                if (gLightCount > 0)
                    r.mPdf *= gEnvironmentSampleProbability;
				return r;
			}
		}

		if (gLightCount == 0)
			return { 0, 0 };

		const uint lightInstanceIndex = mLightInstanceMap[uint(rnd.z * gLightCount) % gLightCount];
		const InstanceData instance = mInstances[lightInstanceIndex];
		const TransformData transform = mInstanceTransforms[lightInstanceIndex];

        r.isFinite = true;

        r.mPdf = 1 / (float)gLightCount;
        if (gEnvironmentMaterialAddress != -1)
			r.mPdf *= 1 - gEnvironmentSampleProbability;

		ShadingData shadingData;
		if (instance.getType() == InstanceType::eMesh) {
			// triangle
			const MeshInstanceData mesh = reinterpret<MeshInstanceData>(instance);
			const uint primitiveIndex = uint(rnd.w * mesh.primitiveCount()) % mesh.primitiveCount();
			shadingData = makeTriangleShadingData(mesh, transform, primitiveIndex, sampleUniformTriangle(rnd.x, rnd.y));
			r.mPdf /= mesh.primitiveCount();
		} else if (instance.getType() == InstanceType::eSphere) {
			// sphere
			const SphereInstanceData sphere = reinterpret<SphereInstanceData>(instance);
			shadingData = makeSphereShadingData(sphere, transform, sphere.radius() * sampleUniformSphere(rnd.x, rnd.y));
		} else
			return { 0 }; // volume lights are unsupported

		r.mPdf /= shadingData.mShapeArea;
		r.mPosition = shadingData.mPosition;
		r.mPackedNormal = shadingData.mPackedShadingNormal;

        if (referencePosition.hasValue) {
			r.mDirectionToLight = r.mPosition - referencePosition.value;
			r.mDistanceToLight  = length(r.mDirectionToLight);
			r.mDirectionToLight /= r.mDistanceToLight;
			r.mCosLight = -dot(r.getNormal(), r.mDirectionToLight);
        }
        r.mRadiance = (referencePosition.hasValue && r.mCosLight <= 0) ? 0 : LoadMaterial(shadingData).getEmission();

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
