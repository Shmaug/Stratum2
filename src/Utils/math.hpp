#pragma once

#include <cstdint>
#include <concepts>
#include <algorithm>

#include <Eigen/Dense>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace tinyvkpt {

using namespace std;

using uint = uint32_t;

template<typename T, int N>        using VectorType = Eigen::Array<T, N, 1, Eigen::ColMajor, N, 1>;
template<typename T, int M, int N> using MatrixType = Eigen::Array<T, M, N, Eigen::RowMajor, M, N>;

using char2    	= VectorType<int8_t , 2>;
using char3    	= VectorType<int8_t , 3>;
using char4    	= VectorType<int8_t , 4>;
using uchar2   	= VectorType<int8_t , 2>;
using uchar3   	= VectorType<int8_t , 3>;
using uchar4   	= VectorType<int8_t , 4>;
using int2    	= VectorType<int32_t, 2>;
using int3    	= VectorType<int32_t, 3>;
using int4    	= VectorType<int32_t, 4>;
using uint2   	= VectorType<int32_t, 2>;
using uint3   	= VectorType<int32_t, 3>;
using uint4   	= VectorType<int32_t, 4>;
using long2    	= VectorType<int64_t, 2>;
using long3    	= VectorType<int64_t, 3>;
using long4    	= VectorType<int64_t, 4>;
using ulong2   	= VectorType<int64_t, 2>;
using ulong3   	= VectorType<int64_t, 3>;
using ulong4   	= VectorType<int64_t, 4>;
using float2  	= VectorType<float  , 2>;
using float3  	= VectorType<float  , 3>;
using float4  	= VectorType<float  , 4>;
using double2  	= VectorType<double , 2>;
using double3  	= VectorType<double , 3>;
using double4  	= VectorType<double , 4>;

using int2x2    = MatrixType<int32_t, 2, 2>;
using int2x3    = MatrixType<int32_t, 2, 3>;
using int2x4    = MatrixType<int32_t, 2, 4>;
using int3x2    = MatrixType<int32_t, 3, 2>;
using int3x3    = MatrixType<int32_t, 3, 3>;
using int3x4    = MatrixType<int32_t, 3, 4>;
using int4x2    = MatrixType<int32_t, 4, 2>;
using int4x3    = MatrixType<int32_t, 4, 3>;
using int4x4    = MatrixType<int32_t, 4, 4>;

using uint2x2   = MatrixType<uint32_t, 2, 2>;
using uint2x3   = MatrixType<uint32_t, 2, 3>;
using uint2x4   = MatrixType<uint32_t, 2, 4>;
using uint3x2   = MatrixType<uint32_t, 3, 2>;
using uint3x3   = MatrixType<uint32_t, 3, 3>;
using uint3x4   = MatrixType<uint32_t, 3, 4>;
using uint4x2   = MatrixType<uint32_t, 4, 2>;
using uint4x3   = MatrixType<uint32_t, 4, 3>;
using uint4x4   = MatrixType<uint32_t, 4, 4>;

using float2x2  = MatrixType<float, 2, 2>;
using float2x3  = MatrixType<float, 2, 3>;
using float2x4  = MatrixType<float, 2, 4>;
using float3x2  = MatrixType<float, 3, 2>;
using float3x3  = MatrixType<float, 3, 3>;
using float3x4  = MatrixType<float, 3, 4>;
using float4x2  = MatrixType<float, 4, 2>;
using float4x3  = MatrixType<float, 4, 3>;
using float4x4  = MatrixType<float, 4, 4>;

using double2x2 = MatrixType<double, 2, 2>;
using double2x3 = MatrixType<double, 2, 3>;
using double2x4 = MatrixType<double, 2, 4>;
using double3x2 = MatrixType<double, 3, 2>;
using double3x3 = MatrixType<double, 3, 3>;
using double3x4 = MatrixType<double, 3, 4>;
using double4x2 = MatrixType<double, 4, 2>;
using double4x3 = MatrixType<double, 4, 3>;
using double4x4 = MatrixType<double, 4, 4>;

using std::min;
using std::max;
using std::abs;
using std::clamp;

 template <typename T> inline constexpr T sign(const T x, std::false_type is_signed) { return (T)(T(0) < x); }
 template <typename T> inline constexpr T sign(const T x, std::true_type is_signed) { return (T)((T(0) < x) - (x < T(0))); }
 template <typename T> inline constexpr T sign(const T x) { return sign(x, std::is_signed<T>()); }

template<typename T,int M, int N> inline MatrixType<T,M,N> max(const T& a, const MatrixType<T,M,N>& b) { return MatrixType<T,M,N>::Constant(a).max(b); }
template<typename T,int M, int N> inline MatrixType<T,M,N> max(const MatrixType<T,M,N>& a, T& b) { return a.max(MatrixType<T,M,N>::Constant(b)); }
template<typename T,int M, int N> inline MatrixType<T,M,N> max(const MatrixType<T,M,N>& a, const MatrixType<T,M,N>& b) { return a.max(b); }
template<typename T,int M, int N> inline MatrixType<T,M,N> min(const MatrixType<T,M,N>& a, const MatrixType<T,M,N>& b) { return a.min(b); }
template<typename T,int M, int N> inline MatrixType<T,M,N> saturate(const MatrixType<T,M,N>& a) { return a.max(MatrixType<T,M,N>::Zero()).min(MatrixType<T,M,N>::Ones()); }
template<typename T,int M, int N> inline bool all(const MatrixType<T,M,N>& a) { return a.all(); }
template<typename T,int M, int N> inline bool any(const MatrixType<T,M,N>& a) { return a.any(); }

template<typename T, int N> inline VectorType<T,N> max(const T& a, const VectorType<T,N>& b) { return VectorType<T,N>::Constant(a).max(b); }
template<typename T, int N> inline VectorType<T,N> max(const VectorType<T,N>& a, T& b) { return a.max(VectorType<T,N>::Constant(b)); }
template<typename T, int N> inline VectorType<T,N> max(const VectorType<T,N>& a, const VectorType<T,N>& b) { return a.max(b); }
template<typename T, int N> inline VectorType<T,N> min(const VectorType<T,N>& a, const VectorType<T,N>& b) { return a.min(b); }
template<typename T, int N> inline VectorType<T,N> saturate(const VectorType<T,N>& a) { return a.max(VectorType<T,N>::Zero()).min(VectorType<T,N>::Ones()); }
template<typename T, int N> inline bool all(const VectorType<T,N>& a) { return a.all(); }
template<typename T, int N> inline bool any(const VectorType<T,N>& a) { return a.any(); }

template<floating_point T> inline T saturate(const T& a) { return clamp(a, T(0), T(1)); }
template<typename T,int M, int N> inline MatrixType<T,M,N> abs(const MatrixType<T,M,N> a) { return a.abs(); }
template<typename T, int N> inline VectorType<T,N> abs(const VectorType<T,N> a) { return a.abs(); }
template<typename T, int M, int N> inline T dot(const MatrixType<T,M,N> a, const MatrixType<T,M,N>& b) { return a.matrix().dot(b.matrix()); }
template<typename T, int M, int N> inline T length(const MatrixType<T,M,N> a) { return a.matrix().norm(); }
template<typename T, int M, int N> inline MatrixType<T,M,N> normalize(const MatrixType<T,M,N> a) { return a.matrix().normalized(); }
template<typename T, int N> inline T dot(const VectorType<T,N> a, const VectorType<T,N>& b) { return a.matrix().dot(b.matrix()); }
template<typename T, int N> inline T length(const VectorType<T,N> a) { return a.matrix().norm(); }
template<typename T, int N> inline VectorType<T,N> normalize(const VectorType<T,N> a) { return a.matrix().normalized(); }

template<typename T> inline VectorType<T,3> cross(const VectorType<T,3> a, const VectorType<T,3> b) { return a.matrix().cross(b.matrix()); }

template<typename T, int M, int N, int K> inline MatrixType<T, M, K> mul(const MatrixType<T, M, N> a, const MatrixType<T, N, K> b) { return a.matrix() * b.matrix(); }

inline float asfloat(uint32_t v) { return *reinterpret_cast<float*>(&v); }
inline uint32_t asuint(float v) { return *reinterpret_cast<uint32_t*>(&v); }
template<int M, int N> inline MatrixType<float, M, N> asfloat(const MatrixType<uint32_t, M, N> v) { return MatrixType<float, M, N>::Map(reinterpret_cast<float*>(v.data())); }
template<int M, int N> inline MatrixType<uint,  M, N> asuint (const MatrixType<float, M, N> v)    { return MatrixType<uint,  M, N>::Map(reinterpret_cast<uint32_t*>(v.data())); }
template<int N> inline VectorType<float, N> asfloat(const VectorType<uint32_t, N> v) { return VectorType<float, N>::Map(reinterpret_cast<float*>(v.data())); }
template<int N> inline VectorType<uint,  N> asuint (const VectorType<float, N> v)    { return VectorType<uint,  N>::Map(reinterpret_cast<uint32_t*>(v.data())); }

}