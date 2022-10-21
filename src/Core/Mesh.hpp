#pragma once

#include <Utils/hash.hpp>

#include "Buffer.hpp"
#include "Shader.hpp"

namespace tinyvkpt {

class Mesh {
public:
	class Vertices {
	public:
		enum class AttributeType {
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
		struct AttributeDescription {
			uint32_t mStride;
			vk::Format mFormat;
			uint32_t mOffset;
			vk::VertexInputRate mInputRate;
		};
		using Attribute = pair<AttributeDescription, Buffer::View<byte>>;

		Vertices() = default;
		Vertices(Vertices&&) = default;
		Vertices(const Vertices&) = default;
		inline Vertices(const unordered_map<AttributeType, vector<Attribute>>& attributes) : mAttributes(attributes) {}

		Vertices& operator=(const Vertices&) = default;
		Vertices& operator=(Vertices&&) = default;

		inline auto begin() { return mAttributes.begin(); }
		inline auto end() { return mAttributes.end(); }
		inline auto begin() const { return mAttributes.begin(); }
		inline auto end() const { return mAttributes.end(); }

		inline optional<Attribute> find(const AttributeType t, uint32_t index = 0) const {
			auto it = mAttributes.find(t);
			if (it != mAttributes.end() && it->second.size() > index)
				return it->second[index];
			else
				return nullopt;
		}
		inline const vector<Attribute>& at(const AttributeType& t) const { return mAttributes.at(t); }

		inline vector<Attribute>& operator[](const AttributeType& t) { return mAttributes[t]; }
		inline const vector<Attribute>& operator[](const AttributeType& t) const { return mAttributes.at(t); }

		void bind(CommandBuffer& commandBuffer) const;

	private:
		unordered_map<AttributeType, vector<Attribute>> mAttributes;
	};

	struct VertexLayoutDescription {
		unordered_map<Vertices::AttributeType, vector<pair<Vertices::AttributeDescription, uint32_t/*binding index*/>>> mAttributes;
		vk::PrimitiveTopology mTopology;
		vk::IndexType mIndexType;

		VertexLayoutDescription() = default;
		VertexLayoutDescription(const VertexLayoutDescription&) = default;
		VertexLayoutDescription(VertexLayoutDescription&&) = default;
		VertexLayoutDescription& operator=(const VertexLayoutDescription&) = default;
		VertexLayoutDescription& operator=(VertexLayoutDescription&&) = default;
		inline VertexLayoutDescription(vk::PrimitiveTopology topo, vk::IndexType indexType = vk::IndexType::eUint16) : mTopology(topo), mIndexType(indexType) {}
	};

	Mesh() = default;
	Mesh(const Mesh&) = default;
	Mesh(Mesh&&) = default;
	Mesh& operator=(const Mesh&) = default;
	Mesh& operator=(Mesh&&) = default;
	inline Mesh(Vertices&& vertices, Buffer::StrideView indices, vk::PrimitiveTopology topology)
		: mVertices(move(vertices)), mIndices(indices), mTopology(topology) {}

	inline const Vertices& vertices() const { return mVertices; }
	inline const Buffer::StrideView& indices() const { return mIndices; }
	inline const vk::PrimitiveTopology topology() const { return mTopology; }
	inline const vk::IndexType indexType() const {
		return (mIndices.stride() == sizeof(uint32_t)) ? vk::IndexType::eUint32 : (mIndices.stride() == sizeof(uint16_t)) ? vk::IndexType::eUint16 : vk::IndexType::eUint8EXT;
	}

	VertexLayoutDescription vertexLayout(const Shader& vertexShader) const;

	void bind(CommandBuffer& commandBuffer) const;

private:
	Vertices mVertices;
	Buffer::StrideView mIndices;
	vk::PrimitiveTopology mTopology = vk::PrimitiveTopology::eTriangleList;
};

}

namespace std {
template<>
struct hash<tinyvkpt::Mesh::Vertices::AttributeDescription> {
	inline size_t operator()(const tinyvkpt::Mesh::Vertices::AttributeDescription& v) const {
		return tinyvkpt::hash_args(v.mFormat, v.mOffset, v.mInputRate);
	}
};

template<>
struct hash<tinyvkpt::Mesh::VertexLayoutDescription> {
	inline size_t operator()(const tinyvkpt::Mesh::VertexLayoutDescription& v) const {
		size_t h = 0;
		for (const auto[type, attribs] : v.mAttributes) {
			h = tinyvkpt::hash_args(h, type);
			for (const auto&[a,i] : attribs)
				h = tinyvkpt::hash_args(h, a, i);
		}
		return tinyvkpt::hash_args(h, v.mTopology, v.mIndexType);
	}
};

inline string to_string(const tinyvkpt::Mesh::Vertices::AttributeType& value) {
	switch (value) {
		case tinyvkpt::Mesh::Vertices::AttributeType::ePosition: return "Position";
		case tinyvkpt::Mesh::Vertices::AttributeType::eNormal: return "Normal";
		case tinyvkpt::Mesh::Vertices::AttributeType::eTangent: return "Tangent";
		case tinyvkpt::Mesh::Vertices::AttributeType::eBinormal: return "Binormal";
		case tinyvkpt::Mesh::Vertices::AttributeType::eBlendIndex: return "BlendIndex";
		case tinyvkpt::Mesh::Vertices::AttributeType::eBlendWeight: return "BlendWeight";
		case tinyvkpt::Mesh::Vertices::AttributeType::eColor: return "Color";
		case tinyvkpt::Mesh::Vertices::AttributeType::ePointSize: return "PointSize";
		case tinyvkpt::Mesh::Vertices::AttributeType::eTexcoord: return "Texcoord";
		default: return "invalid ( " + vk::toHexString( static_cast<uint32_t>( value ) ) + " )";
	}
}
}