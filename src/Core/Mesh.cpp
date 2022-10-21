#include "Mesh.hpp"
#include "CommandBuffer.hpp"

namespace tinyvkpt {

Mesh::VertexLayoutDescription Mesh::vertexLayout(const Shader& vertexShader) const {
	VertexLayoutDescription layout(mTopology, indexType());

	struct stride_view_hash {
		inline size_t operator()(const Buffer::StrideView& v) const {
			return hash_args(v.buffer().get(), v.offset(), v.sizeBytes(), v.stride());
		}
	};

	unordered_map<Buffer::StrideView, uint32_t, stride_view_hash> uniqueBuffers;
	for (const auto&[id, v] : vertexShader.inputVariables()) {
		Vertices::AttributeType attributeType;
		// TODO: attributeType from v.mSemantic
		optional<Vertices::Attribute> attrib = mVertices.find(attributeType, v.mSemanticIndex);
		if (!attrib) throw logic_error("Mesh does not contain required shader input " + to_string(attributeType) + "." + to_string(v.mSemanticIndex));

		auto& dstAttribs = layout.mAttributes[attributeType];
		if (dstAttribs.size() <= v.mSemanticIndex)
		dstAttribs.resize(v.mSemanticIndex + 1);

		dstAttribs[v.mSemanticIndex].first = attrib->first;
		if (auto it = uniqueBuffers.find(attrib->second); it != uniqueBuffers.end())
			dstAttribs[v.mSemanticIndex].second = it->second;
		else {
			dstAttribs[v.mSemanticIndex].second =  (uint32_t)uniqueBuffers.size();
			uniqueBuffers.emplace(attrib->second, dstAttribs[v.mSemanticIndex].second);
		}
	}

	return layout;
}

void Mesh::Vertices::bind(CommandBuffer& commandBuffer) const {
	// TODO
}
void Mesh::bind(CommandBuffer& commandBuffer) const {
	mVertices.bind(commandBuffer);
	commandBuffer->bindIndexBuffer(***mIndices.buffer(), mIndices.offset(), indexType());
}

}