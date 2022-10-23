#pragma once

#include "Device.hpp"

#include <Utils/hash.hpp>

namespace tinyvkpt {

class Buffer : public Device::Resource {
public:
	Buffer(Device& device, const string& name, const vk::BufferCreateInfo& createInfo, const vk::MemoryPropertyFlags memoryFlags = vk::MemoryPropertyFlagBits::eDeviceLocal, const bool randomHostAccess = false);
	inline Buffer(Device& device, const string& name, const vk::DeviceSize& size, const vk::BufferUsageFlags usage, const vk::MemoryPropertyFlags memoryFlags = vk::MemoryPropertyFlagBits::eDeviceLocal, const bool randomHostAccess = false) :
		Buffer(device, name, vk::BufferCreateInfo({}, size, usage), memoryFlags, randomHostAccess) {}
	~Buffer();

	DECLARE_DEREFERENCE_OPERATORS(vk::Buffer, mBuffer)

	inline operator bool() const { return mBuffer; }

	inline void* data() const { return mAllocationInfo.pMappedData; }
	inline vk::DeviceSize size() const { return mSize; }
	inline vk::BufferUsageFlags usage() const { return mUsage; }
	inline vk::MemoryPropertyFlags memoryUsage() const { return mMemoryFlags; }
	inline vk::SharingMode sharingMode() const { return mSharingMode; }
	inline vk::DeviceSize deviceAddress() const { return mDevice->getBufferAddress(mBuffer); }

	template<typename T = byte>
	class View {
	public:
		using value_type = T;
		using size_type = vk::DeviceSize;
		using reference = value_type&;
		using pointer = value_type*;
		using iterator = T*;

		View() = default;
		View(View&&) = default;
		inline View(const View& v, size_t elementOffset = 0, size_t elementCount = VK_WHOLE_SIZE) : mBuffer(v.mBuffer), mOffset(v.mOffset + elementOffset * sizeof(T)) {
			if (mBuffer) {
				mSize = (elementCount == VK_WHOLE_SIZE) ? (v.size() - elementOffset) : elementCount;
				if (mOffset + mSize * sizeof(T) > mBuffer->size())
					throw out_of_range("view size out of bounds");
			}
		}
		inline View(const shared_ptr<Buffer>& buffer, const vk::DeviceSize byteOffset = 0, const vk::DeviceSize elementCount = VK_WHOLE_SIZE) : mBuffer(buffer), mOffset(byteOffset) {
			if (mBuffer) {
				mSize = (elementCount == VK_WHOLE_SIZE) ? (mBuffer->size() - mOffset) / sizeof(T) : elementCount;
				if (mOffset + mSize * sizeof(T) > mBuffer->size())
					throw out_of_range("view size out of bounds");
			}
		}

		View& operator=(const View&) = default;
		View& operator=(View&&) = default;
		bool operator==(const View&) const = default;

		inline operator View<byte>() const { return View<byte>(mBuffer, mOffset, sizeBytes()); }
		inline operator bool() const { return !empty(); }

		inline const shared_ptr<Buffer>& buffer() const { return mBuffer; }
		inline vk::DeviceSize offset() const { return mOffset; }

		inline bool empty() const { return !mBuffer || mSize == 0; }
		inline void reset() { mBuffer.reset(); }
		inline vk::DeviceSize size() const { return mSize; }
		inline vk::DeviceSize sizeBytes() const { return mSize * sizeof(T); }
		inline pointer data() const { return reinterpret_cast<pointer>(reinterpret_cast<char*>(mBuffer->data()) + offset()); }
#if VK_KHR_buffer_device_address
		inline vk::DeviceSize deviceAddress() const {
			return mBuffer->deviceAddress() + mOffset;
		}
#endif

		inline T& at(size_type index) const { return data()[index]; }
		inline T& operator[](size_type index) const { return at(index); }

		inline reference front() { return at(0); }
		inline reference back() { return at(mSize - 1); }

		inline iterator begin() const { return data(); }
		inline iterator end() const { return data() + mSize; }

		template<typename Ty>
		inline View<Ty> cast() const {
			if ((sizeBytes() % sizeof(Ty)) != 0)
				throw logic_error("Buffer size must be divisible by sizeof(Ty)");
			return View<Ty>(buffer(), offset(), sizeBytes() / sizeof(Ty));
		}

	private:
		shared_ptr<Buffer> mBuffer;
		vk::DeviceSize mOffset;
		vk::DeviceSize mSize;
	};

	class StrideView : public View<byte> {
	private:
		vk::DeviceSize mStride;
	public:
		StrideView() = default;
		StrideView(StrideView&&) = default;
		StrideView(const StrideView&) = default;
		inline StrideView(const View<byte>& view, vk::DeviceSize stride) : View<byte>(view), mStride(stride) {}
		template<typename T>
		inline StrideView(const View<T>& v) : View<byte>(v.buffer(), v.offset(), v.sizeBytes()), mStride(sizeof(T)) {}
		inline StrideView(const shared_ptr<Buffer>& buffer, vk::DeviceSize stride, vk::DeviceSize byteOffset = 0, vk::DeviceSize byteLength = VK_WHOLE_SIZE)
			: View<byte>(buffer, byteOffset, byteLength), mStride(stride) {}

		StrideView& operator=(const StrideView&) = default;
		StrideView& operator=(StrideView&&) = default;
		bool operator==(const StrideView&) const = default;

		inline vk::DeviceSize stride() const { return mStride; }

		template<typename T>
		inline operator View<T>() const {
			if (sizeof(T) != mStride) throw logic_error("sizeof(T) must match stride");
			return Buffer::View<T>(buffer(), offset(), sizeBytes() / sizeof(T));
		}
	};

private:
	vk::Buffer mBuffer;
	VmaAllocation mAllocation;
	VmaAllocationInfo mAllocationInfo;
	vk::DeviceSize mSize;
	vk::BufferUsageFlags mUsage;
	vk::MemoryPropertyFlags mMemoryFlags;
	vk::SharingMode mSharingMode;
};

}