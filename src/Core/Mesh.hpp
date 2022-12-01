#pragma once

#include <Utils/hash.hpp>

#include "Buffer.hpp"
#include "Shader.hpp"

namespace stm2 {

class Mesh {
public:
	enum class VertexAttributeType {
		ePosition,
		eNormal,
		eTangent,
		eBinormal,
		eColor,
		eTexcoord,
		ePointSize,
		eBlendIndex,
		eBlendWeight
	};
	struct VertexAttributeDescription {
		uint32_t mStride;
		vk::Format mFormat;
		uint32_t mOffset;
		vk::VertexInputRate mInputRate;
	};
	using VertexAttributeData = pair<Buffer::View<byte>, VertexAttributeDescription>;

	struct VertexLayoutDescription {
		unordered_map<VertexAttributeType, vector<pair<VertexAttributeDescription, uint32_t/*binding index*/>>> mAttributes;
		vk::PrimitiveTopology mTopology;
		vk::IndexType mIndexType;

		VertexLayoutDescription() = default;
		VertexLayoutDescription(const VertexLayoutDescription&) = default;
		VertexLayoutDescription(VertexLayoutDescription&&) = default;
		VertexLayoutDescription& operator=(const VertexLayoutDescription&) = default;
		VertexLayoutDescription& operator=(VertexLayoutDescription&&) = default;
		inline VertexLayoutDescription(vk::PrimitiveTopology topo, vk::IndexType indexType = vk::IndexType::eUint16) : mTopology(topo), mIndexType(indexType) {}
	};

	class Vertices : public unordered_map<VertexAttributeType, vector<VertexAttributeData>> {
	public:
		inline optional<VertexAttributeData> find(const VertexAttributeType t, uint32_t index = 0) const {
			auto it = unordered_map<VertexAttributeType, vector<VertexAttributeData>>::find(t);
			if (it != end() && it->second.size() > index)
				return it->second[index];
			else
				return nullopt;
		}

		void bind(CommandBuffer& commandBuffer) const;
	};

	Mesh() = default;
	Mesh(const Mesh&) = default;
	Mesh(Mesh&&) = default;
	Mesh& operator=(const Mesh&) = default;
	Mesh& operator=(Mesh&&) = default;
	inline Mesh(const Vertices& vertices, const Buffer::StrideView& indices, const vk::PrimitiveTopology topology = vk::PrimitiveTopology::eTriangleList)
		: mVertices(vertices), mIndices(indices), mTopology(topology) {}
	inline Mesh(Vertices&& vertices, const Buffer::StrideView& indices, const vk::PrimitiveTopology topology = vk::PrimitiveTopology::eTriangleList)
		: mVertices(move(vertices)), mIndices(indices), mTopology(topology) {}

	inline const Vertices& vertices() const { return mVertices; }
	inline const Buffer::StrideView& indices() const { return mIndices; }
	inline const vk::PrimitiveTopology topology() const { return mTopology; }
	inline const vk::IndexType indexType() const {
		return (mIndices.stride() == sizeof(uint32_t)) ? vk::IndexType::eUint32 : (mIndices.stride() == sizeof(uint16_t)) ? vk::IndexType::eUint16 : vk::IndexType::eUint8EXT;
	}

	VertexLayoutDescription vertexLayout(const Shader& vertexShader) const;

	void bind(CommandBuffer& commandBuffer) const;

	void drawGui();

private:
	Vertices mVertices;
	Buffer::StrideView mIndices;
	vk::PrimitiveTopology mTopology = vk::PrimitiveTopology::eTriangleList;
};

}

namespace std {
template<>
struct hash<stm2::Mesh::VertexAttributeDescription> {
	inline size_t operator()(const stm2::Mesh::VertexAttributeDescription& v) const {
		return stm2::hashArgs(v.mFormat, v.mOffset, v.mInputRate);
	}
};

template<>
struct hash<stm2::Mesh::VertexLayoutDescription> {
	inline size_t operator()(const stm2::Mesh::VertexLayoutDescription& v) const {
		size_t h = 0;
		for (const auto[type, attribs] : v.mAttributes) {
			h = stm2::hashArgs(h, type);
			for (const auto&[a,i] : attribs)
				h = stm2::hashArgs(h, a, i);
		}
		return stm2::hashArgs(h, v.mTopology, v.mIndexType);
	}
};

inline string to_string(const stm2::Mesh::VertexAttributeType& value) {
	switch (value) {
		case stm2::Mesh::VertexAttributeType::ePosition: return "Position";
		case stm2::Mesh::VertexAttributeType::eNormal: return "Normal";
		case stm2::Mesh::VertexAttributeType::eTangent: return "Tangent";
		case stm2::Mesh::VertexAttributeType::eBinormal: return "Binormal";
		case stm2::Mesh::VertexAttributeType::eBlendIndex: return "BlendIndex";
		case stm2::Mesh::VertexAttributeType::eBlendWeight: return "BlendWeight";
		case stm2::Mesh::VertexAttributeType::eColor: return "Color";
		case stm2::Mesh::VertexAttributeType::ePointSize: return "PointSize";
		case stm2::Mesh::VertexAttributeType::eTexcoord: return "Texcoord";
		default: return "invalid ( " + vk::toHexString( static_cast<uint32_t>( value ) ) + " )";
	}
}
}