#include "Mesh.hpp"
#include "CommandBuffer.hpp"

namespace tinyvkpt {

Mesh::VertexLayoutDescription Mesh::vertexLayout(const Shader& vertexShader) const {
	VertexLayoutDescription layout(mTopology, indexType());

	struct stride_view_hash {
		inline size_t operator()(const Buffer::StrideView& v) const {
			return hashArgs(v.buffer().get(), v.offset(), v.sizeBytes(), v.stride());
		}
	};

	unordered_map<Buffer::StrideView, uint32_t, stride_view_hash> uniqueBuffers;
	for (const auto&[id, v] : vertexShader.inputVariables()) {
		VertexAttributeType attributeType;
		// TODO: attributeType from v.mSemantic
		optional<VertexAttributeData> attrib = mVertices.find(attributeType, v.mSemanticIndex);
		if (!attrib) throw logic_error("Mesh does not contain required shader input " + to_string(attributeType) + "." + to_string(v.mSemanticIndex));

		// get/create attribute in attribute array
		auto& dstAttribs = layout.mAttributes[attributeType];
		if (dstAttribs.size() <= v.mSemanticIndex)
		dstAttribs.resize(v.mSemanticIndex + 1);

		auto&[vertexBuffer, attributeDescription] = *attrib;

		// store attribute description
		dstAttribs[v.mSemanticIndex].first = attributeDescription;

		// get unique binding index for buffer
		if (auto it = uniqueBuffers.find(vertexBuffer); it != uniqueBuffers.end())
			dstAttribs[v.mSemanticIndex].second = it->second;
		else {
			dstAttribs[v.mSemanticIndex].second = (uint32_t)uniqueBuffers.size();
			uniqueBuffers.emplace(vertexBuffer, dstAttribs[v.mSemanticIndex].second);
		}
	}

	return layout;
}

void Mesh::Vertices::bind(CommandBuffer& commandBuffer) const {
	// TODO: bind mesh
}
void Mesh::bind(CommandBuffer& commandBuffer) const {
	mVertices.bind(commandBuffer);
	commandBuffer->bindIndexBuffer(**mIndices.buffer(), mIndices.offset(), indexType());
}

}