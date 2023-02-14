#ifndef gHasMedia
#define gHasMedia (false)
#endif

#ifndef gAlphaTest
#define gAlphaTest (false)
#endif

#ifndef gShadingNormals
#define gShadingNormals (false)
#endif

#ifndef gNormalMaps
#define gNormalMaps (false)
#endif

#ifndef gMaxNullCollisions
#define gMaxNullCollisions (1024)
#endif

#ifndef gPerformanceCounters
#define gPerformanceCounters (false)
#endif

// Do only unidirectional path tracing with next event estimation
#ifndef gPathTraceOnly
#define gPathTraceOnly (false)
#endif
// Do only light tracing
#ifndef gLightTraceOnly
#define gLightTraceOnly (false)
#endif
// Vertex merging (of some form) is used
#ifndef gUseVM
#define gUseVM (false)
#endif
// Vertex connection (BPT) is used
#ifndef gUseVC
#define gUseVC (false)
#endif
// Light vertex cache (LVC) is used
#ifndef gUseLVC
#define gUseLVC (false)
#endif
// Do PPM, terminates camera after first merge
#ifndef gPpm
#define gPpm (false)
#endif
// show only specific path connections
#ifndef gDebugPaths
#define gDebugPaths (false)
#endif
// show weighted path connections
#ifndef gDebugPathWeights
#define gDebugPathWeights (false)
#endif

#ifndef gLVCHashGridSampling
#define gLVCHashGridSampling (false)
#endif

#ifndef gDIReservoirFlags
#define gDIReservoirFlags 0
#endif
#ifndef gLVCReservoirFlags
#define gLVCReservoirFlags 0
#endif
