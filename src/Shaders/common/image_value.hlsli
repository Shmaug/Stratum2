float sample_image(Texture2D<float> img, const float2 uv, const float uv_screen_size) {
	float w, h;
	img.GetDimensions(w, h);
	float lod = 0;
	if (gUseRayCones && uv_screen_size > 0)
		lod = log2(max(uv_screen_size * max(w, h), 1e-6f));
	return img.SampleLevel(gSceneParams.gStaticSampler, uv, lod);
}
float4 sample_image(Texture2D<float4> img, const float2 uv, const float uv_screen_size) {
	float w, h;
	img.GetDimensions(w, h);
	float lod = 0;
	if (gUseRayCones && uv_screen_size > 0)
		lod = log2(max(uv_screen_size * max(w, h), 1e-6f));
	return img.SampleLevel(gSceneParams.gStaticSampler, uv, lod);
}

float eval_image_value1(inout uint address, const float2 uv, const float uv_screen_size) {
	ImageValue1 img;
	img.load(address);
	return img.eval(uv, uv_screen_size);
}
float2 eval_image_value2(inout uint address, const float2 uv, const float uv_screen_size) {
	ImageValue2 img;
	img.load(address);
	return img.eval(uv, uv_screen_size);
}
float3 eval_image_value3(inout uint address, const float2 uv, const float uv_screen_size) {
	ImageValue3 img;
	img.load(address);
	return img.eval(uv, uv_screen_size);
}
float4 eval_image_value4(inout uint address, const float2 uv, const float uv_screen_size) {
	ImageValue4 img;
	img.load(address);
	return img.eval(uv, uv_screen_size);
}