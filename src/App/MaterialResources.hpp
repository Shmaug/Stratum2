#pragma once

#include <Core/math.hpp>
#include <Core/Buffer.hpp>
#include <Core/Image.hpp>

namespace stm2 {

struct ByteAppendBuffer : public vector<uint32_t> {
	inline size_t sizeBytes() const {
		return size() * sizeof(uint32_t);
	}

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
	ByteAppendBuffer mMaterialData;

	unordered_map<Image::View, uint32_t> mImage4s;
	unordered_map<Image::View, uint32_t> mImage1s;
	unordered_map<pair<shared_ptr<Buffer>, vk::DeviceSize>, uint32_t> mVolumeDataMap;

	inline void clear() {
		mMaterialData.clear();
		mImage4s.clear();
		mImage1s.clear();
		mVolumeDataMap.clear();
	}

	inline uint32_t getIndex(const Image::View& image) {
		if (!image) return ~0u;
		const Image::View tmp(image.image(), image.subresourceRange(), image.type());
		if (channelCount(tmp.image()->format()) == 1) {
			auto it = mImage1s.find(tmp);
			return (it == mImage1s.end()) ? mImage1s.emplace(tmp, (uint32_t)mImage1s.size()).first->second : it->second;
		} else {
			auto it = mImage4s.find(tmp);
			return (it == mImage4s.end()) ? mImage4s.emplace(tmp, (uint32_t)mImage4s.size()).first->second : it->second;
		}
	}
	inline uint32_t getIndex(const Buffer::View<byte>& buf) {
		if (!buf) return ~0u;
		const auto key = pair{buf.buffer(), buf.offset()};
		auto it = mVolumeDataMap.find(key);
		return (it == mVolumeDataMap.end()) ? mVolumeDataMap.emplace(key, (uint32_t)mVolumeDataMap.size()).first->second : it->second;
	}
};

}