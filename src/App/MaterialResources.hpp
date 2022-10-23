#pragma once

#include <Utils/math.hpp>
#include <Core/Buffer.hpp>
#include <Core/Image.hpp>

namespace tinyvkpt {

struct ByteAppendBuffer : public vector<uint32_t> {
	inline uint Load(const uint32_t address) {
		return operator[](address / 4);
	}

	template<typename T, int N>
	inline void AppendN(const VectorType<T, N>& x) {
		for (uint i = 0; i < N; i++)
			emplace_back(x[i]);
	}
	template<int N>
	inline void AppendN(const VectorType<float, N>& x) {
		for (uint i = 0; i < N; i++)
			emplace_back(asuint(x[i]));
	}
	inline void Append(const uint32_t x) {
		emplace_back(x);
	}
	inline void Appendf(const float x) {
		emplace_back(asuint(x));
	}
};

struct MaterialResources {
	unordered_map<Image::View, uint32_t> image4s;
	unordered_map<Image::View, uint32_t> image1s;
	unordered_map<Buffer::View<byte>, uint32_t> volumeDataMap;

	inline uint32_t getIndex(const Image::View& image) {
		if (!image) return ~0u;
		const Image::View tmp(image.image(), image.subresourceRange(), image.type());
		if (channelCount(tmp.image()->format()) == 1) {
			auto it = image1s.find(tmp);
			return (it == image1s.end()) ? image1s.emplace(tmp, (uint32_t)image1s.size()).first->second : it->second;
		} else {
			auto it = image4s.find(tmp);
			return (it == image4s.end()) ? image4s.emplace(tmp, (uint32_t)image4s.size()).first->second : it->second;
		}
	}
	inline uint32_t getIndex(const Buffer::View<byte>& buf) {
		if (!buf) return ~0u;
		auto it = volumeDataMap.find(buf);
		return (it == volumeDataMap.end()) ? volumeDataMap.emplace(buf, (uint32_t)volumeDataMap.size()).first->second : it->second;
	}
};


}