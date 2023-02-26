#include "load_serialized.hpp"

#include <miniz.h>

#define MTS_FILEFORMAT_VERSION_V3 0x0003
#define MTS_FILEFORMAT_VERSION_V4 0x0004

#define ZSTREAM_BUFSIZE 32768

namespace stm2 {

class z_iftream {
public:
	inline z_iftream(std::fstream& fs) : fs(fs) {
		std::streampos pos = fs.tellg();
		fs.seekg(0, fs.end);
		fsize = (size_t)fs.tellg();
		fs.seekg(pos, fs.beg);

		int windowBits = 15;
		m_inflateStream.zalloc = Z_NULL;
		m_inflateStream.zfree = Z_NULL;
		m_inflateStream.opaque = Z_NULL;
		m_inflateStream.avail_in = 0;
		m_inflateStream.next_in = Z_NULL;

		int retval = inflateInit2(&m_inflateStream, windowBits);
		if (retval != Z_OK) {
			throw runtime_error("Could not initialize ZLIB");
		}
	}
	inline virtual ~z_iftream() {
		inflateEnd(&m_inflateStream);
	}

	inline void read(void* ptr, size_t size) {
		uint8_t* targetPtr = (uint8_t*)ptr;
		while (size > 0) {
			if (m_inflateStream.avail_in == 0) {
				size_t remaining = fsize - fs.tellg();
				m_inflateStream.next_in = m_inflateBuffer;
				m_inflateStream.avail_in = (uint32_t)min(remaining, sizeof(m_inflateBuffer));
				if (m_inflateStream.avail_in == 0) {
					throw runtime_error("Read less data than expected");
				}

				fs.read((char*)m_inflateBuffer, m_inflateStream.avail_in);
			}

			m_inflateStream.avail_out = (uint32_t)size;
			m_inflateStream.next_out = targetPtr;

			int retval = inflate(&m_inflateStream, Z_NO_FLUSH);
			switch (retval) {
			case Z_STREAM_ERROR: {
				throw runtime_error("inflate(): stream error!");
			}
			case Z_NEED_DICT: {
				throw runtime_error("inflate(): need dictionary!");
			}
			case Z_DATA_ERROR: {
				throw runtime_error("inflate(): data error!");
			}
			case Z_MEM_ERROR: {
				throw runtime_error("inflate(): memory error!");
			}
			};

			size_t outputSize = size - (size_t)m_inflateStream.avail_out;
			targetPtr += outputSize;
			size -= outputSize;

			if (size > 0 && retval == Z_STREAM_END) {
				throw runtime_error("inflate(): attempting to read past the end of the stream!");
			}
		}
	}

private:
	std::fstream& fs;
	size_t fsize;
	z_stream m_inflateStream;
	uint8_t m_inflateBuffer[ZSTREAM_BUFSIZE];
};

Mesh loadSerialized(CommandBuffer& commandBuffer, const filesystem::path& filename, int shape_index) {
	std::fstream fs(filename.c_str(), std::fstream::in | std::fstream::binary);
	// Format magic number, ignore it
	fs.ignore(sizeof(short));
	// Version number
	short version = 0;
	fs.read((char*)&version, sizeof(short));
	if (shape_index > 0) {
		// Go to the end of the file to see how many components are there
		fs.seekg(-sizeof(uint32_t), fs.end);
		uint32_t count = 0;
		fs.read((char*)&count, sizeof(uint32_t));
		size_t offset = 0;
		if (version == MTS_FILEFORMAT_VERSION_V4) {
			fs.seekg(-sizeof(uint64_t) * (count - shape_index) - sizeof(uint32_t), fs.end);
			fs.read((char*)&offset, sizeof(size_t));
		} else {  // V3
			fs.seekg(-sizeof(uint32_t) * (count - shape_index + 1), fs.end);
			uint32_t upos = 0;
			fs.read((char*)&upos, sizeof(unsigned int));
			offset = upos;
		}
		fs.seekg(offset, fs.beg);
		// Skip the header
		fs.ignore(sizeof(short) * 2);
	}
	z_iftream zs(fs);

	enum ETriMeshFlags {
		EHasNormals = 0x0001,
		EHasTexcoords = 0x0002,
		EHasTangents = 0x0004,  // unused
		EHasColors = 0x0008,
		EFaceNormals = 0x0010,
		ESinglePrecision = 0x1000,
		EDoublePrecision = 0x2000
	};

	uint32_t flags;
	zs.read((char*)&flags, sizeof(uint32_t));
	std::string name;
	if (version == MTS_FILEFORMAT_VERSION_V4) {
		char c;
		while (true) {
			zs.read((char*)&c, sizeof(char));
			if (c == '\0')
				break;
			name.push_back(c);
		}
	}
	size_t vertex_count = 0;
	zs.read((char*)&vertex_count, sizeof(size_t));
	size_t triangle_count = 0;
	zs.read((char*)&triangle_count, sizeof(size_t));

	bool file_double_precision = flags & EDoublePrecision;
	// bool face_normals = flags & EFaceNormals;

	vk::BufferUsageFlags bufferUsage = vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer;
	if (commandBuffer.mDevice.accelerationStructureFeatures().accelerationStructure) {
		bufferUsage |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
		bufferUsage |= vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
	}

	Mesh::Vertices attributes;

	Buffer::View<float3> positions_tmp = make_shared<Buffer>(commandBuffer.mDevice, "tmp positions", sizeof(float3) * vertex_count, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
	{
		float3 vmin = float3::Constant(numeric_limits<float>::infinity());
		float3 vmax = float3::Constant(-numeric_limits<float>::infinity());
		if (file_double_precision) {
			for (uint32_t i = 0; i < vertex_count; i++) {
				double3 tmp;
				zs.read(tmp.data(), sizeof(double) * 3);
				positions_tmp[i] = tmp.cast<float>();
				vmin = min(vmin, positions_tmp[i]);
				vmax = max(vmax, positions_tmp[i]);
			}
		} else {
			for (uint32_t i = 0; i < vertex_count; i++) {
				zs.read(positions_tmp[i].data(), sizeof(float) * 3);
				vmin = min(vmin, positions_tmp[i]);
				vmax = max(vmax, positions_tmp[i]);
			}
		}
		attributes.mAabb = vk::AabbPositionsKHR(vmin[0], vmin[1], vmin[2], vmax[0], vmax[1], vmax[2]);
		Buffer::View<float3> positions = make_shared<Buffer>(commandBuffer.mDevice, filename.stem().string() + " positions", positions_tmp.sizeBytes(), bufferUsage|vk::BufferUsageFlagBits::eVertexBuffer);
		Buffer::copy(commandBuffer, positions_tmp, positions);
		attributes[Mesh::VertexAttributeType::ePosition].emplace_back(positions, Mesh::VertexAttributeDescription{ (uint32_t)sizeof(float3), vk::Format::eR32G32B32Sfloat, 0, vk::VertexInputRate::eVertex });
	}

	if (flags & EHasNormals) {
		Buffer::View<float3> normals_tmp = make_shared<Buffer>(commandBuffer.mDevice, "tmp normals", sizeof(float3) * vertex_count, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		if (file_double_precision) {
			for (uint32_t i = 0; i < vertex_count; i++) {
				double3 tmp;
				zs.read(tmp.data(), sizeof(double) * 3);
				normals_tmp[i] = tmp.cast<float>();
			}
		} else {
			for (uint32_t i = 0; i < vertex_count; i++)
				zs.read(normals_tmp[i].data(), sizeof(float) * 3);
		}
		Buffer::View<float3> normals = make_shared<Buffer>(commandBuffer.mDevice, filename.stem().string() + " normals", normals_tmp.sizeBytes(), bufferUsage|vk::BufferUsageFlagBits::eVertexBuffer);
		Buffer::copy(commandBuffer, normals_tmp, normals);
		attributes[Mesh::VertexAttributeType::eNormal].emplace_back(normals, Mesh::VertexAttributeDescription{ (uint32_t)sizeof(float3), vk::Format::eR32G32B32Sfloat, 0, vk::VertexInputRate::eVertex });
	}

	if (flags & EHasTexcoords) {
		Buffer::View<float2> uvs_tmp = make_shared<Buffer>(commandBuffer.mDevice, "tmp uvs", sizeof(float2) * vertex_count, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		if (file_double_precision) {
			for (uint32_t i = 0; i < vertex_count; i++) {
				double2 tmp;
				zs.read(tmp.data(), sizeof(double) * 2);
				uvs_tmp[i] = tmp.cast<float>();
			}
		} else {
			for (uint32_t i = 0; i < vertex_count; i++)
				zs.read(uvs_tmp[i].data(), sizeof(float) * 2);
		}
		Buffer::View<float2> uvs = make_shared<Buffer>(commandBuffer.mDevice, filename.stem().string() + " uvs", uvs_tmp.sizeBytes(), bufferUsage|vk::BufferUsageFlagBits::eVertexBuffer);
		Buffer::copy(commandBuffer, uvs_tmp, uvs);
		attributes[Mesh::VertexAttributeType::eTexcoord].emplace_back(uvs, Mesh::VertexAttributeDescription{ (uint32_t)sizeof(float2), vk::Format::eR32G32Sfloat, 0, vk::VertexInputRate::eVertex });
	}

	if (flags & EHasColors) {
		Buffer::View<float3> colors_tmp = make_shared<Buffer>(commandBuffer.mDevice, "tmp colors", sizeof(float3) * vertex_count, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		if (file_double_precision) {
			for (uint32_t i = 0; i < vertex_count; i++) {
				double3 tmp;
				zs.read(tmp.data(), sizeof(double) * 3);
				colors_tmp[i] = tmp.cast<float>();
			}
		} else {
			for (uint32_t i = 0; i < vertex_count; i++)
				zs.read(colors_tmp[i].data(), sizeof(float) * 3);
		}
		Buffer::View<float2> colors = make_shared<Buffer>(commandBuffer.mDevice, filename.stem().string() + " colors", colors_tmp.sizeBytes(), bufferUsage|vk::BufferUsageFlagBits::eVertexBuffer);
		Buffer::copy(commandBuffer, colors_tmp, colors);
		attributes[Mesh::VertexAttributeType::eColor].emplace_back(colors, Mesh::VertexAttributeDescription{ (uint32_t)sizeof(float3), vk::Format::eR32G32B32Sfloat, 0, vk::VertexInputRate::eVertex });
	}

	Buffer::View<uint32_t> indices_tmp = make_shared<Buffer>(commandBuffer.mDevice, "tmp inds", 3 * triangle_count * sizeof(uint32_t), vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
	zs.read(indices_tmp.data(), sizeof(uint32_t) * 3 * triangle_count);

	Buffer::View<uint32_t> indexBuffer = make_shared<Buffer>(commandBuffer.mDevice, filename.stem().string() + " indices", indices_tmp.sizeBytes(), bufferUsage | vk::BufferUsageFlagBits::eIndexBuffer);
	Buffer::copy(commandBuffer, indices_tmp, indexBuffer);

	return Mesh(attributes, indexBuffer, vk::PrimitiveTopology::eTriangleList);
}

}