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

namespace stm2 {

using namespace std;

using uint = uint32_t;

template<typename T, int N>        using VectorType = Eigen::Array<T, N, 1, Eigen::ColMajor, N, 1>;
template<typename T, int M, int N> using MatrixType = Eigen::Array<T, M, N, Eigen::RowMajor, M, N>;

#define DECLARE_VECTOR_TYPES(T, tname) \
	using tname##1 = VectorType<T, 1>; \
	using tname##2 = VectorType<T, 2>; \
	using tname##3 = VectorType<T, 3>; \
	using tname##4 = VectorType<T, 4>; \
	using tname##2x2 = MatrixType<T, 2, 2>; \
	using tname##2x3 = MatrixType<T, 2, 3>; \
	using tname##2x4 = MatrixType<T, 2, 4>; \
	using tname##3x2 = MatrixType<T, 3, 2>; \
	using tname##3x3 = MatrixType<T, 3, 3>; \
	using tname##3x4 = MatrixType<T, 3, 4>; \
	using tname##4x2 = MatrixType<T, 4, 2>; \
	using tname##4x3 = MatrixType<T, 4, 3>; \
	using tname##4x4 = MatrixType<T, 4, 4>;

DECLARE_VECTOR_TYPES(int8_t, char)
DECLARE_VECTOR_TYPES(int8_t, uchar)
DECLARE_VECTOR_TYPES(int32_t, int)
DECLARE_VECTOR_TYPES(int32_t, uint)
DECLARE_VECTOR_TYPES(int64_t, long)
DECLARE_VECTOR_TYPES(int64_t, ulong)
DECLARE_VECTOR_TYPES(float, float)
DECLARE_VECTOR_TYPES(double, double)

#undef DECLARE_VECTOR_TYPES

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