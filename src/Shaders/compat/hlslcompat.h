#ifndef HLSLCOMPAT_H
#define HLSLCOMPAT_H

#if defined(__HLSL_VERSION) && !defined(__HLSL__)
#define __HLSL__
#endif

#ifdef __HLSL__
#define __hlsl_in
#define __hlsl_out(T) out T
#define __hlsl_inout(T) inout T
#endif

#ifdef __SLANG__
#define SLANG_MUTATING [mutating]
#define SLANG_SHADER(type) [shader(type)]
#else
#define SLANG_MUTATING
#define SLANG_SHADER(type)
#endif

#ifdef __cplusplus
#define CONST_CPP const
#define __hlsl_in
#define __hlsl_out(T) T&
#define __hlsl_inout(T) T&
#else
#define CONST_CPP
#endif



#endif