#ifdef N
struct DataType {
    float data[N];
};
#elif !defined(DataType)
#define DataType float4
#endif

#include "common/hashgrid.hlsli"

ParameterBlock<HashGrid<DataType>> gHashGrid;

// Prefix sum over cell counter values to determine indices. Should be called with 1 thread per cell.
[shader("compute")]
[numthreads(64,1,1)]
void ComputeIndices(uint3 index: SV_DispatchThreadID) {
    gHashGrid.ComputeIndices(index.y*1024 + index.x);
}

// Sort items from append order into cells. Should be called with 1 thread per item.
[shader("compute")]
[numthreads(64,1,1)]
void Swizzle(uint3 index: SV_DispatchThreadID) {
    gHashGrid.Swizzle(index.y*1024 + index.x);
}