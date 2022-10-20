#pragma once

#include <functional>

#include <vulkan/vulkan_hash.hpp>
#include <Eigen/Dense>

#include "common.hpp"

namespace tinyvkpt {

template<typename T> concept hashable = requires(T v) { { std::hash<T>()(v) } -> convertible_to<size_t>; };

constexpr size_t hash_combine(const size_t x, const size_t y) {
	return x ^ (y + 0x9e3779b9 + (x << 6) + (x >> 2));
}

// accepts string literals
template<typename T, size_t N>
constexpr size_t hash_array(const T(& arr)[N]) {
	hash<T> hasher;
	if constexpr (N == 0)
		return 0;
	else if constexpr (N == 1)
		return hasher(arr[0]);
	else
		return hash_combine(hash_array<T,N-1>(arr), hasher(arr[N-1]));
}

template<hashable Tx, hashable... Ty>
inline size_t hash_args(const Tx& x, const Ty&... y) {
	if constexpr (sizeof...(Ty) == 0)
		return hash<Tx>()(x);
	else
		return hash_combine(hash<Tx>()(x), hash_args<Ty...>(y...));
}

}

namespace std {

template<tinyvkpt::hashable T0, tinyvkpt::hashable T1>
struct hash<pair<T0,T1>> {
	inline size_t operator()(const pair<T0,T1>& v) const {
		return tinyvkpt::hash_combine(hash<T0>()(v.first), hash<T1>()(v.second));
	}
};

template<tinyvkpt::hashable... Types>
struct hash<tuple<Types...>> {
	inline size_t operator()(const tuple<Types...>& v) const {
		return tinyvkpt::hash_args<Types...>(get<Types>(v)...);
	}
};

template<tinyvkpt::hashable T, size_t N>
struct hash<std::array<T,N>> {
	constexpr size_t operator()(const std::array<T,N>& a) const {
		return tinyvkpt::hash_array<T,N>(a.data());
	}
};

template<ranges::range R> requires(tinyvkpt::hashable<ranges::range_value_t<R>>)
struct hash<R> {
	inline size_t operator()(const R& r) const {
		size_t h = 0;
		for (auto it = ranges::begin(r); it != ranges::end(r); ++it)
			h = tinyvkpt::hash_combine(h, hash<ranges::range_value_t<R>>()(*it));
		return h;
	}
};

template<typename T>
struct hash<weak_ptr<T>> {
	inline size_t operator()(const weak_ptr<T>& p) const {
		return hash<shared_ptr<T>>()(p.lock());
	}
};

template<tinyvkpt::hashable T, int Rows, int Cols, int Options, int MaxRows, int MaxCols> requires(Rows != Eigen::Dynamic && Cols != Eigen::Dynamic)
struct hash<Eigen::Array<T,Rows,Cols,Options,MaxRows,MaxCols>> {
  constexpr size_t operator()(const Eigen::Array<T,Rows,Cols,Options,MaxRows,MaxCols>& m) const {
		hash<T> hasher;
		size_t h = 0;
		for (size_t r = 0; r < Rows; r++)
			for (size_t c = 0; c < Cols; c++)
				h = tinyvkpt::hash_combine(h, hasher(m[r][c]));
		return h;
  }
};
template<tinyvkpt::hashable T, int Rows, int Cols, int Options, int MaxRows, int MaxCols> requires(Rows != Eigen::Dynamic && Cols != Eigen::Dynamic)
struct hash<Eigen::Matrix<T,Rows,Cols,Options,MaxRows,MaxCols>> {
  constexpr size_t operator()(const Eigen::Matrix<T,Rows,Cols,Options,MaxRows,MaxCols>& m) const {
		return hash<Eigen::Array<T,Rows,Cols,Options,MaxRows,MaxCols>>()(m);
  }
};

}