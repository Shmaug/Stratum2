#include "Mesh.hpp"
#include "CommandBuffer.hpp"

#include <imgui/imgui.h>

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
		static const unordered_map<string, VertexAttributeType> attributeTypeMap {
			{ "position", VertexAttributeType::ePosition },
			{ "normal", VertexAttributeType::eNormal },
			{ "tangent", VertexAttributeType::eTangent },
			{ "binormal", VertexAttributeType::eBinormal },
			{ "color", VertexAttributeType::eColor },
			{ "texcoord", VertexAttributeType::eTexcoord },
			{ "pointsize", VertexAttributeType::ePointSize },
			{ "blendindex", VertexAttributeType::eBlendIndex },
			{ "blendweight", VertexAttributeType::eBlendWeight }
		};

		VertexAttributeType attributeType = VertexAttributeType::eTexcoord;
		string s = v.mSemantic;
		ranges::transform(v.mSemantic, s.begin(), static_cast<int(*)(int)>(std::tolower));
		if (auto it = attributeTypeMap.find(s); it != attributeTypeMap.end())
			attributeType = it->second;
		else
			cout << "Warning, unknown variable semantic " << v.mSemantic << endl;

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

void Mesh::drawGui() {
	ImGui::LabelText("Topology", to_string(mTopology).c_str());
	if (mIndices)
		ImGui::LabelText("Index stride", to_string(mIndices.stride()).c_str());
	for (const auto& [type, verts] : mVertices)
		for (uint32_t i = 0; i < verts.size(); i++) {
			const auto&[buf, desc] = verts[i];
			if (buf && ImGui::CollapsingHeader((to_string(type) + "_" + to_string(i)).c_str())) {
				ImGui::LabelText("Format", to_string(desc.mFormat).c_str());
				ImGui::LabelText("Stride", to_string(desc.mStride).c_str());
				ImGui::LabelText("Offset", to_string(desc.mOffset).c_str());
				ImGui::LabelText("Input rate", to_string(desc.mInputRate).c_str());
			}
		}
}

}