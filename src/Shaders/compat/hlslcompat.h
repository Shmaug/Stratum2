#ifndef HLSLCOMPAT_H
#define HLSLCOMPAT_H

#if defined(__HLSL_VERSION) && !defined(__HLSL__)
#define __HLSL__
#endif


#ifdef __SLANG__

#define SLANG_MUTATING [mutating]
#define SLANG_CTOR(type) __init

#else // __SLANG__

#define SLANG_MUTATING
#define SLANG_CTOR(type) inline type

#endif


#ifdef __cplusplus

#define CONST_CPP const
#define __hlsl_in
#define __hlsl_out(T) T&
#define __hlsl_inout(T) T&
#define STM_NAMESPACE_BEGIN namespace stm2 {
#define STM_NAMESPACE_END }

#define POS_INFINITY  numeric_limits<float>::infinity()
#define NEG_INFINITY -numeric_limits<float>::infinity()

#else // __cplusplus

#define CONST_CPP
#define __hlsl_in
#define __hlsl_out(T) out T
#define __hlsl_inout(T) inout T
#define STM_NAMESPACE_BEGIN
#define STM_NAMESPACE_END

#define M_PI (3.1415926535897932)
#define M_1_PI (1/M_PI)

#define POS_INFINITY asfloat(0x7F800000)
#define NEG_INFINITY asfloat(0xFF800000)

#endif

#endif