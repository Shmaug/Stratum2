#pragma once

#include "rng.hlsli"

struct HashGrid<T> {
	RWStructuredBuffer<uint> mChecksums;
	RWStructuredBuffer<uint> mCounters;

	// stores (cellIndex, indexInCell) in append-order.
	// mAppendIndices[0][0] is the total number of appended items
	// mAppendIndices[0][1] is used in ComputeIndices() for the prefix sum.
	RWStructuredBuffer<uint2> mAppendIndices;
	RWStructuredBuffer<T> mAppendData;

	// mIndices[cellIndex] = offset in mData
	// mCounters[cellIndex] = number of items in cell
	RWStructuredBuffer<uint> mIndices;
	RWStructuredBuffer<T> mData;

	RWStructuredBuffer<uint> mStats;


	property uint mCellCount { get { return gPushConstants.mHashGridCellCount; } }


	float GetCellSize(const float3 aPosition) {
		return gPushConstants.VcmMergeRadius()*2;
		//if (aCellPixelSize <= 0)
		//	return aMinCellSize;
		//const float cameraDistance = length(gCamera.transform.transformPoint(0) - aPosition);
		//const float2 extent = gCamera.view.extent();
		//const float step = cameraDistance * tan(aCellPixelSize * gCamera.view.mProjection.mVerticalFoV * max(1/extent.y, extent.y/pow2(extent.x)));
		//return aMinCellSize * (1 << uint(log2(step / aMinCellSize)));
	}

	uint2 GetCellDataRange(const uint cellIndex) {
        const uint start = mIndices[cellIndex];
        return uint2(start, start + mCounters[cellIndex]);
    }

	uint FindCellIndex<let bInserting : bool>(const float3 aPosition, const int3 offset = 0) {
		const float cellSize = GetCellSize(aPosition);

		// compute index in hash grid
		const int3 p = int3(floor(aPosition/cellSize)) + offset;
		const uint checksum = max(1, xxhash32(asuint(cellSize) + xxhash32(p.z + xxhash32(p.y + xxhash32(p.x)))));
		uint cellIndex = pcg(asuint(cellSize) + pcg(p.z + pcg(p.y + pcg(p.x)))) % mCellCount;

		// resolve hash collisions with linear probing
		for (uint i = 0; i < 32; i++) {
			// find cell with matching checksum, or empty cell if inserting
			if (bInserting) {
				uint prevChecksum;
				InterlockedCompareExchange(mChecksums[cellIndex], 0, checksum, prevChecksum);
				if (prevChecksum == 0 || prevChecksum == checksum)
					return cellIndex;
			} else {
				if (mChecksums[cellIndex] == checksum)
					return cellIndex;
			}
			cellIndex = (cellIndex + 1) % mCellCount;
        }

		// failed to find cell (hashgrid full)
		return -1;
	}

	void Append(const float3 aPosition, const T data) {
        const uint cellIndex = FindCellIndex<true>(aPosition);
        if (cellIndex == -1) {
			if (gPerformanceCounters)
                InterlockedAdd(mStats[0], 1); // failed inserts
			return;
        }

		// append item to cell by incrementing cell counter
		uint indexInCell;
		InterlockedAdd(mCounters[cellIndex], 1, indexInCell);

		uint appendIndex;
		InterlockedAdd(mAppendIndices[0][0], 1, appendIndex);
		mAppendIndices[1 + appendIndex] = uint2(cellIndex, indexInCell);
		mAppendData[1 + appendIndex] = data;

		// first time the cell was used
        if (gPerformanceCounters && indexInCell == 0)
			InterlockedAdd(mStats[1], 1);
	}


	// Prefix sum over cell counter values to determine indices. To be called with 1 thread per cell.
	void ComputeIndices(const uint aCellIndex) {
		if (aCellIndex >= mCellCount) return;

		uint offset;
		InterlockedAdd(mAppendIndices[0][1], mCounters[aCellIndex], offset);

		mIndices[aCellIndex] = offset;
	}

	// Sort items from append order into cells. To be called with 1 thread per item.
	void Swizzle(const uint aAppendIndex) {
		if (aAppendIndex >= mAppendIndices[0][0]) return;

		const uint2 data = mAppendIndices[1 + aAppendIndex];
		const uint cellIndex   = data[0];
		const uint indexInCell = data[1];

		const uint dstIndex = mIndices[cellIndex] + indexInCell;
		mData[dstIndex] = mAppendData[1 + aAppendIndex];
	}
};
