struct Params {
	Texture2D<float4> mInput;
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
void pass1(uint3 index : SV_DispatchThreadId) {
	uint2 dim = 1;
	gParams.mOutput.GetDimensions(dim.x, dim.y);
	if (any(index.xy > dim)) return;
	gParams.mOutput[index.xy] = float4((index.xy + 0.5) / dim, 0, 1);
}

[shader("compute")]
[numthreads(8,8,1)]
void pass2(uint3 index : SV_DispatchThreadId) {
	uint2 dim = 1;
	gParams.mOutput.GetDimensions(dim.x, dim.y);
	if (any(index.xy > dim)) return;
	gParams.mOutput[index.xy] = float4(gParams.mInput[index.xy].rgb * pow(2,gPushConstants.mExposure) + gPushConstants.mBias, 1);
}