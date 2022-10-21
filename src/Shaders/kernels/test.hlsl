struct Params {
	RWTexture2D<float4> mInput;
	RWTexture2D<float4> mOutput;
};

ParameterBlock<Params> gParams;

struct PushConstants {
	float4 mBias;
	float4 mExposure;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> gPushConstants;

SLANG_SHADER("compute")
[numthreads(8,8,1)]
void main(uint3 index : SV_DispatchThreadId) {
	gOutput[index.xy] = gInput[index.xy] * gPushConstants.mBias + gPushConstants.mExposure;
}