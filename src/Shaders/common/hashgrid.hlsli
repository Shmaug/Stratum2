#pragma once

float hashgrid_cell_size(const float3 pos) {
	if (gHashGridBucketPixelRadius < 0)
		return gHashGridMinBucketRadius;
	const uint view_index = 0;
	const TransformData t = gFrameParams.gViewTransforms[view_index];
	const float dist = length(pos - float3(t.m[0][3], t.m[1][3], t.m[2][3]));
	const float2 extent = gFrameParams.gViews[view_index].mImageMax - gFrameParams.gViews[view_index].mImageMin;
	const float step = dist * tan(gHashGridBucketPixelRadius * gFrameParams.gViews[view_index].mProjection.mVerticalFoV * max(1/extent.y, extent.y/pow2(extent.x)));
	return gHashGridMinBucketRadius * (1 << uint(log2(step / gHashGridMinBucketRadius)));
}
uint hashgrid_bucket_index(const float3 pos, out uint checksum, const float cell_size) {
	// compute index in hash grid
	const int3 p = floor(pos/cell_size) + 0.5;
	checksum = max(1, xxhash32(cell_size + xxhash32(p.z + xxhash32(p.y + xxhash32(p.x)))));
	return pcg(cell_size + pcg(p.z + pcg(p.y + pcg(p.x)))) % gHashGridBucketCount;
}

struct ReservoirHashGrid<T> {
	RWStructuredBuffer<uint> mChecksums;
	RWStructuredBuffer<uint> mCounters;
	RWStructuredBuffer<uint> mStats;
	RWStructuredBuffer<ReservoirData> mReservoirs;
	RWStructuredBuffer<PathVertex> mSampleSources;
	RWStructuredBuffer<T> mReservoirSamples;
	RWStructuredBuffer<uint> mIndices;
	RWStructuredBuffer<uint2> mAppendIndices;
	RWStructuredBuffer<ReservoirData> mAppendReservoirs;
	RWStructuredBuffer<PathVertex> mAppendSampleSources;
	RWStructuredBuffer<T> mAppendReservoirSamples;

	uint find(const float3 pos, const float cell_size, const bool inserting = false) {
		uint checksum;
		uint bucket_index = hashgrid_bucket_index(pos, checksum, cell_size);
		for (uint i = 0; i < 32; i++) {
			if (inserting) {
				uint checksum_prev;
				InterlockedCompareExchange(mChecksums[bucket_index], 0, checksum, checksum_prev);
				if (checksum_prev == 0 || checksum_prev == checksum)
					return bucket_index;
			} else {
				if (mChecksums[bucket_index] == checksum)
					return bucket_index;
			}
			bucket_index++;
        }
        if (gPerformanceCounters && inserting)
			InterlockedAdd(mStats[0], 1); // failed inserts
		return -1;
	}
	void append(const float3 pos, const float cell_size, const ReservoirData r, const PathVertex v, const T y) {
		const uint bucket_index = find(pos, cell_size, true);
		if (bucket_index == -1) return;

		uint index_in_bucket;
		InterlockedAdd(mCounters[bucket_index], 1, index_in_bucket);

        if (gPerformanceCounters && index_in_bucket == 0)
			InterlockedAdd(mStats[1], 1); // buckets used

		uint append_index;
		InterlockedAdd(mAppendIndices[0][0], 1, append_index);
		mAppendReservoirs[1 + append_index] = r;
		mSampleSources[1 + append_index] = v;
		mAppendReservoirSamples[1 + append_index] = y;
		mAppendIndices[1 + append_index] = uint2(bucket_index, index_in_bucket);
	}

	void compute_indices(const uint bucket_index) {
		if (bucket_index >= gHashGridBucketCount) return;
		uint offset;
		InterlockedAdd(mAppendIndices[0][1], mCounters[bucket_index], offset);
		mIndices[bucket_index] = offset;
	}
	void swizzle(const uint append_index) {
		if (append_index >= mAppendIndices[0][0]) return;
		const uint2 data = mAppendIndices[1 + append_index];
		const uint bucket_index = data[0];
		const uint index_in_bucket = data[1];

		const uint dst_index = mIndices[bucket_index] + index_in_bucket;
		mReservoirs[dst_index]       = mAppendReservoirs[1 + append_index];
		mSampleSources[dst_index]    = mAppendSampleSources[1 + append_index];
		mReservoirSamples[dst_index] = mAppendReservoirSamples[1 + append_index];
	}
};
