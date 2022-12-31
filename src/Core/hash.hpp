#pragma once

#include <functional>

#include <vulkan/vulkan_hash.hpp>
#include <Eigen/Dense>

namespace stm2 {

template<typename T> concept hashable = requires(T v) { { std::hash<T>()(v) } -> std::convertible_to<size_t>; };

constexpr size_t hashCombine(const size_t x, const size_t y) {
	return x ^ (y + 0x9e3779b9 + (x << 6) + (x >> 2));
}

// accepts string literals
template<typename T, size_t N>
constexpr size_t hashArray(const T(& arr)[N]) {
	std::hash<T> hasher;
	if constexpr (N == 0)
		return 0;
	else if constexpr (N == 1)
		return hasher(arr[0]);
	else
		return hashCombine(hashArray<T,N-1>(arr), hasher(arr[N-1]));
}

template<std::ranges::range R> requires(hashable<std::ranges::range_value_t<R>>)
inline size_t hashRange(const R& r) {
	size_t h = 0;
	for (auto it = std::ranges::begin(r); it != std::ranges::end(r); ++it)
		h = hashCombine(h, std::hash<std::ranges::range_value_t<R>>()(*it));
	return h;
}

template<hashable Tx, hashable... Ty>
inline size_t hashArgs(const Tx& x, const Ty&... y) {
	if constexpr (sizeof...(Ty) == 0)
		return std::hash<Tx>()(x);
	else
		return hashCombine(std::hash<Tx>()(x), hashArgs<Ty...>(y...));
}

}

namespace std {

template<stm2::hashable T0, stm2::hashable T1>
struct hash<pair<T0,T1>> {
	inline size_t operator()(const pair<T0,T1>& v) const {
		return stm2::hashCombine(hash<T0>()(v.first), hash<T1>()(v.second));
	}
};

template<stm2::hashable... Types>
struct hash<tuple<Types...>> {
	inline size_t operator()(const tuple<Types...>& v) const {
		return stm2::hashArgs<Types...>(get<Types>(v)...);
	}
};

template<stm2::hashable T, size_t N>
struct hash<std::array<T,N>> {
	constexpr size_t operator()(const std::array<T,N>& a) const {
		return stm2::hashArray<T,N>(a.data());
	}
};

template<stm2::hashable T, int Rows, int Cols, int Options, int MaxRows, int MaxCols> requires(Rows != Eigen::Dynamic && Cols != Eigen::Dynamic)
struct hash<Eigen::Array<T,Rows,Cols,Options,MaxRows,MaxCols>> {
  constexpr size_t operator()(const Eigen::Array<T,Rows,Cols,Options,MaxRows,MaxCols>& m) const {
		hash<T> hasher;
		size_t h = 0;
		for (size_t r = 0; r < Rows; r++)
			for (size_t c = 0; c < Cols; c++)
				h = stm2::hashCombine(h, hasher(m[r][c]));
		return h;
  }
};
template<stm2::hashable T, int Rows, int Cols, int Options, int MaxRows, int MaxCols> requires(Rows != Eigen::Dynamic && Cols != Eigen::Dynamic)
struct hash<Eigen::Matrix<T,Rows,Cols,Options,MaxRows,MaxCols>> {
  constexpr size_t operator()(const Eigen::Matrix<T,Rows,Cols,Options,MaxRows,MaxCols>& m) const {
		return hash<Eigen::Array<T,Rows,Cols,Options,MaxRows,MaxCols>>()(m);
  }
};

}