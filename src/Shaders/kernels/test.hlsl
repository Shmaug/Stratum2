struct Params {
	RWTexture2D<float4> mOutput;
};
ParameterBlock<Params> gParams;

struct PushConstants {
	float3 mBias;
	float mExposure;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> gPushConstants;

[shader("compute")]
[numthreads(8,8,1)]
void main(uint3 index : SV_DispatchThreadId) {
	uint2 dim = 1;
	gParams.mOutput.GetDimensions(dim.x, dim.y);
	if (any(index.xy > dim)) return;
	gParams.mOutput[index.xy] = float4(float3(pow(2,gPushConstants.mExposure) * (index.xy + 0.5) / dim, 0) + gPushConstants.mBias, 1);
}