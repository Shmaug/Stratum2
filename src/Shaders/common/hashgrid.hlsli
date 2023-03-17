#pragma once

#include "rng.hlsli"

struct HashGridConstants {
	float mCellPixelRadius;
	float mMinCellSize;
	uint mCellCount;
	uint pad;
	float3 mCameraPosition;
	float mDistanceScale;
};

struct HashGrid<T> {
	RWStructuredBuffer<uint> mChecksums;
    // mCellCounters[cellIndex] = number of items in cell
	RWStructuredBuffer<uint> mCellCounters;
    // mOtherCounters[0] = number of appended items
    // mOtherCounters[1] = prefix sum counter
	RWStructuredBuffer<uint> mOtherCounters;

	// stores (cellIndex, indexInCell) in append-order.
	RWStructuredBuffer<uint2> mAppendDataIndices;
	RWStructuredBuffer<T> mAppendData;

    // mIndices[cellIndex] = offset in mDataIndices
	RWStructuredBuffer<uint> mIndices;
	RWStructuredBuffer<uint> mDataIndices;

    ConstantBuffer<HashGridConstants> mConstants;

    uint GetCurrentElementCount() { return mOtherCounters[0]; }

    uint GetCellDataOffset(const uint aCellIndex) { return mIndices[aCellIndex]; }
    uint GetCellDataCount (const uint aCellIndex) { return mCellCounters[aCellIndex]; }

	// aIndex is GetCellDataOffset(cellIndex) + offset, where offset is [ 0, GetCellDataCount(cellIndex) )
    T Get(const uint aIndex) {
        return mAppendData[mDataIndices[aIndex]];
	}


    float GetCellSize(const float3 aPosition) {
        if (mConstants.mCellPixelRadius <= 0)
            return mConstants.mMinCellSize;
        const float cameraDistance = length(mConstants.mCameraPosition - aPosition);
        const float step = cameraDistance * mConstants.mDistanceScale;
		return mConstants.mMinCellSize * (1 << uint(log2(step / mConstants.mMinCellSize)));
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
		InterlockedAdd(mCellCounters[cellIndex], 1, indexInCell);

		// store payload
		uint appendIndex;
		InterlockedAdd(mOtherCounters[0], 1, appendIndex);
		mAppendDataIndices[appendIndex] = uint2(cellIndex, indexInCell);
        mAppendData[appendIndex] = data;
        return appendIndex;
	}


	// Prefix sum over cell counter values to determine indices. Should be called with 1 thread per cell.
    void ComputeIndices(const uint aCellIndex) {
        if (aCellIndex >= mConstants.mCellCount) return;

        uint offset;
        InterlockedAdd(mOtherCounters[1], mCellCounters[aCellIndex], offset);

		mIndices[aCellIndex] = offset;
	}

	// Sort items from append order into cells. Should be called with 1 thread per item.
    void Swizzle(const uint aAppendIndex) {
        if (aAppendIndex >= GetCurrentElementCount()) return;

		const uint2 data = mAppendDataIndices[aAppendIndex];
		const uint cellIndex   = data[0];
		const uint indexInCell = data[1];

        mDataIndices[GetCellDataOffset(cellIndex) + indexInCell] = aAppendIndex;
	}
};
