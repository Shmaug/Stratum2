#pragma once

#include "rng.hlsli"

struct HashGridConstants {
    uint mCellCount;
    uint mCellPixelRadius;
    uint mMinCellSize;
    uint pad;
    float3 mCameraPosition;
    float mDistanceScale;
};

struct HashGrid<T> {
	RWStructuredBuffer<uint> mChecksums;
    // mCounters[cellIndex] = number of items in cell
    // mCounters[cellCount] = number of appended items
    // mCounters[cellCount+1] = prefix sum counter
    // mCounters[cellCount+2] = number of non-empty cells
	RWStructuredBuffer<uint> mCounters;

	// stores (cellIndex, indexInCell) in append-order.
	RWStructuredBuffer<uint2> mAppendDataIndices;
	RWStructuredBuffer<T> mAppendData;

	// mIndices[cellIndex] = offset in mData
	RWStructuredBuffer<uint> mIndices;
	RWStructuredBuffer<T> mData;

    RWStructuredBuffer<uint> mActiveCellIndices;

    ConstantBuffer<HashGridConstants> mConstants;

    uint GetCurrentSize() { return mCounters[mConstants.mCellCount]; }

    float GetCellSize(const float3 aPosition) {
        if (mConstants.mCellPixelRadius <= 0)
            return mConstants.mMinCellSize;
        const float cameraDistance = length(mConstants.mCameraPosition - aPosition);
        const float step = cameraDistance * mConstants.mDistanceScale;
		return mConstants.mMinCellSize * (1 << uint(log2(step / mConstants.mMinCellSize)));
	}

	uint2 GetCellDataRange(const uint cellIndex) {
        const uint start = mIndices[cellIndex];
        return uint2(start, start + mCounters[cellIndex]);
    }

    uint FindCellIndex<let bInsert : bool>(const float3 aPosition, float aCellSize = 0, const int3 aOffset = 0) {
        if (aCellSize == 0) aCellSize = GetCellSize(aPosition);
        // compute index in hash grid
        const int3 p = int3(floor(aPosition / aCellSize)) + aOffset;
        const uint checksum = max(1, xxhash32(xxhash32(asuint(aCellSize)) + xxhash32(p.z + xxhash32(p.y + xxhash32(p.x)))));
        const uint baseCellIndex = pcg(pcg(asuint(aCellSize)) + pcg(p.z + pcg(p.y + pcg(p.x)))) % mConstants.mCellCount;

        // resolve hash collisions with linear probing
        for (uint i = 0; i < 32; i++) {
            const uint cellIndex = (baseCellIndex + i) % mConstants.mCellCount;
            // find cell with matching checksum, or empty cell if inserting
            if (bInsert) {
                uint prevChecksum;
                InterlockedCompareExchange(mChecksums[cellIndex], 0, checksum, prevChecksum);
                if (prevChecksum == 0 || prevChecksum == checksum)
                    return cellIndex;
            } else {
                if (mChecksums[cellIndex] == checksum)
                    return cellIndex;
            }
        }

        // collision resolution failed - hashgrid full?
        return -1;
    }

    uint Append(const float3 aPosition, const T data, float aCellSize = 0) {
        if (aCellSize == 0) aCellSize = GetCellSize(aPosition);
        const uint cellIndex = FindCellIndex<true>(aPosition, aCellSize);
        if (cellIndex == -1) {
			return -1;
        }

		// append item to cell by incrementing cell counter
		uint indexInCell;
		InterlockedAdd(mCounters[cellIndex], 1, indexInCell);

        // keep track of non-empty cells
        if (indexInCell == 0) {
            uint count;
            InterlockedAdd(mCounters[mConstants.mCellCount + 2], 1, count);
            mActiveCellIndices[count] = cellIndex;
        }

		// store payload
		uint appendIndex;
		InterlockedAdd(mCounters[mConstants.mCellCount], 1, appendIndex);
		mAppendDataIndices[appendIndex] = uint2(cellIndex, indexInCell);
        mAppendData[appendIndex] = data;
        return appendIndex;
	}


	// Prefix sum over cell counter values to determine indices. Should be called with 1 thread per cell.
    void ComputeIndices(const uint aCellIndex) {
        if (aCellIndex >= mConstants.mCellCount) return;

        uint offset;
        InterlockedAdd(mCounters[mConstants.mCellCount + 1], mCounters[aCellIndex], offset);

		mIndices[aCellIndex] = offset;
	}

	// Sort items from append order into cells. Should be called with 1 thread per item.
    void Swizzle(const uint aAppendIndex) {
        if (aAppendIndex >= GetCurrentSize()) return;

		const uint2 data = mAppendDataIndices[aAppendIndex];
		const uint cellIndex   = data[0];
		const uint indexInCell = data[1];

        mData[mIndices[cellIndex] + indexInCell] = mAppendData[aAppendIndex];
	}
};
