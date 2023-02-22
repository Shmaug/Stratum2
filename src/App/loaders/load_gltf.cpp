#include <App/Scene.hpp>
#include <Core/CommandBuffer.hpp>
#include <Core/Profiler.hpp>

#define TINYGLTF_USE_CPP14
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_IMPLEMENTATION
#include <tiny_gltf.h>

namespace stm2 {

shared_ptr<Node> Scene::loadGltf(CommandBuffer& commandBuffer, const filesystem::path& filename) {
	ProfilerScope ps("loadGltf", &commandBuffer);

	cout << "Loading " << filename << endl;

	tinygltf::Model model;
	tinygltf::TinyGLTF loader;
	string err, warn;
	if (
		(filename.extension() == ".glb" && !loader.LoadBinaryFromFile(&model, &err, &warn, filename.string())) ||
		(filename.extension() == ".gltf" && !loader.LoadASCIIFromFile(&model, &err, &warn, filename.string())) )
		throw runtime_error(filename.string() + ": " + err);
	if (!warn.empty()) cerr << filename.string() << ": " << warn << endl;

	cout << "Processing scene data..." << endl;

	Device& device = commandBuffer.mDevice;

	vector<shared_ptr<Buffer>> buffers(model.buffers.size());
	vector<Image::View> images(model.images.size());
	vector<shared_ptr<Material>> materials(model.materials.size());
	vector<vector<shared_ptr<Mesh>>> meshes(model.meshes.size());

	const shared_ptr<Node> rootNode = Node::create(filename.stem().string());

	auto getImage = [&](const uint32_t texture_index, const bool srgb) -> Image::View {
		if (texture_index >= model.textures.size()) return {};
		const uint32_t index = model.textures[texture_index].source;
		if (index >= images.size()) return {};
		if (images[index]) return images[index];

		const tinygltf::Image& image = model.images[index];
		Buffer::View<unsigned char> pixels = make_shared<Buffer>(device, image.name+"/Staging", image.image.size(), vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		ranges::uninitialized_copy(image.image, pixels);

		Image::Metadata md = {};
		if (srgb) {
			static const std::array<vk::Format,4> formatMap { vk::Format::eR8Srgb, vk::Format::eR8G8Srgb, vk::Format::eR8G8B8Srgb, vk::Format::eR8G8B8A8Srgb };
			md.mFormat = formatMap.at(image.component - 1);
		} else {
			static const unordered_map<int, std::array<vk::Format,4>> formatMap {
				{ TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE,  { vk::Format::eR8Unorm, vk::Format::eR8G8Unorm, vk::Format::eR8G8B8Unorm, vk::Format::eR8G8B8A8Unorm } },
				{ TINYGLTF_COMPONENT_TYPE_BYTE,           { vk::Format::eR8Snorm, vk::Format::eR8G8Snorm, vk::Format::eR8G8B8Snorm, vk::Format::eR8G8B8A8Snorm } },
				{ TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT, { vk::Format::eR16Unorm, vk::Format::eR16G16Unorm, vk::Format::eR16G16B16Unorm, vk::Format::eR16G16B16A16Unorm } },
				{ TINYGLTF_COMPONENT_TYPE_SHORT,          { vk::Format::eR16Snorm, vk::Format::eR16G16Snorm, vk::Format::eR16G16B16Snorm, vk::Format::eR16G16B16A16Snorm } },
				{ TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT,   { vk::Format::eR32Uint, vk::Format::eR32G32Uint, vk::Format::eR32G32B32Uint, vk::Format::eR32G32B32A32Uint } },
				{ TINYGLTF_COMPONENT_TYPE_INT,            { vk::Format::eR32Sint, vk::Format::eR32G32Sint, vk::Format::eR32G32B32Sint, vk::Format::eR32G32B32A32Sint } },
				{ TINYGLTF_COMPONENT_TYPE_FLOAT,          { vk::Format::eR32Sfloat, vk::Format::eR32G32Sfloat, vk::Format::eR32G32B32Sfloat, vk::Format::eR32G32B32A32Sfloat } },
				{ TINYGLTF_COMPONENT_TYPE_DOUBLE,         { vk::Format::eR64Sfloat, vk::Format::eR64G64Sfloat, vk::Format::eR64G64B64Sfloat, vk::Format::eR64G64B64A64Sfloat } }
			};
			md.mFormat = formatMap.at(image.pixel_type).at(image.component - 1);
		}

		md.mExtent = vk::Extent3D(image.width, image.height, 1);
		md.mLevels = Image::maxMipLevels(md.mExtent);
		const shared_ptr<Image> img = make_shared<Image>(device, image.name, md);

		pixels.copyToImage(commandBuffer, img);

		img->generateMipMaps(commandBuffer);

		images[index] = img;
		return img;
	};

	cout << "Loading buffers..." << endl;

	vk::BufferUsageFlags bufferUsage = vk::BufferUsageFlagBits::eVertexBuffer|vk::BufferUsageFlagBits::eIndexBuffer|vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eTransferSrc;
	#ifdef VK_KHR_buffer_device_address
	bufferUsage |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
	bufferUsage |= vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
	#endif
	ranges::transform(model.buffers, buffers.begin(), [&](const tinygltf::Buffer& buffer) {
		Buffer::View<unsigned char> tmp = make_shared<Buffer>(device, buffer.name+"/Staging", buffer.data.size(), vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		ranges::copy(buffer.data, tmp.begin());
		Buffer::View<unsigned char> dst = make_shared<Buffer>(device, buffer.name, buffer.data.size(), bufferUsage, vk::MemoryPropertyFlagBits::eDeviceLocal);
		Buffer::copy(commandBuffer, tmp, dst);
		dst.barrier(commandBuffer, vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eVertexInput|vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eVertexAttributeRead|vk::AccessFlagBits::eIndexRead|vk::AccessFlagBits::eShaderRead);
		return dst.buffer();
	});

	cout << "Loading materials..." << endl;
	ranges::transform(model.materials, materials.begin(), [&](const tinygltf::Material& material) {
		ImageValue<3> emission{
			double3::Map(material.emissiveFactor.data()).cast<float>(),
			getImage(material.emissiveTexture.index, true) };
		if (material.extensions.find("KHR_materials_emissive_strength") != material.extensions.end())
			emission.mValue *= (float)material.extensions.at("KHR_materials_emissive_strength").Get("emissiveStrength").GetNumberAsDouble();

		const ImageValue<3> baseColor{
			double3::Map(material.pbrMetallicRoughness.baseColorFactor.data()).cast<float>(),
			getImage(material.pbrMetallicRoughness.baseColorTexture.index, true) };
		const ImageValue<4> metallicRoughness{
			double4(0, material.pbrMetallicRoughness.roughnessFactor, material.pbrMetallicRoughness.metallicFactor, 0).cast<float>(),
			getImage(material.pbrMetallicRoughness.metallicRoughnessTexture.index, false) };
		const float eta = material.extensions.contains("KHR_materials_ior") ? (float)material.extensions.at("KHR_materials_ior").Get("ior").GetNumberAsDouble() : 1.5f;
		const float transmission = material.extensions.contains("KHR_materials_transmission") ? (float)material.extensions.at("KHR_materials_transmission").Get("transmissionFactor").GetNumberAsDouble() : 0;

		Material m = makeMetallicRoughnessMaterial(commandBuffer, baseColor, metallicRoughness, ImageValue<3>(float3::Constant(transmission), {}), eta, emission);

		if (material.extensions.contains("KHR_materials_clearcoat")) {
			const auto& v = material.extensions.at("KHR_materials_clearcoat");
			m.mMaterialData.setClearcoat((float)v.Get("clearcoatFactor").GetNumberAsDouble());
		}

		m.mBumpImage = getImage(material.normalTexture.index, false);
		m.mBumpStrength = 1;

		return make_shared<Material>(m);
	});

	cout << "Loading meshes...";
	for (uint32_t i = 0; i < model.meshes.size(); i++) {
		cout << "\rLoading meshes " << (i+1) << "/" << model.meshes.size() << "     ";
		meshes[i].resize(model.meshes[i].primitives.size());
		for (uint32_t j = 0; j < model.meshes[i].primitives.size(); j++) {
			const tinygltf::Primitive& prim = model.meshes[i].primitives[j];
			const auto& indicesAccessor = model.accessors[prim.indices];
			const auto& indexBufferView = model.bufferViews[indicesAccessor.bufferView];
			const size_t indexStride = tinygltf::GetComponentSizeInBytes(indicesAccessor.componentType);
			const Buffer::StrideView indexBuffer = Buffer::StrideView(buffers[indexBufferView.buffer], indexStride, indexBufferView.byteOffset + indicesAccessor.byteOffset, indicesAccessor.count * indexStride);

			Mesh::Vertices vertexData;

			vk::PrimitiveTopology topology;
			switch (prim.mode) {
				case TINYGLTF_MODE_POINTS: 			topology = vk::PrimitiveTopology::ePointList; break;
				case TINYGLTF_MODE_LINE: 			topology = vk::PrimitiveTopology::eLineList; break;
				case TINYGLTF_MODE_LINE_LOOP: 		topology = vk::PrimitiveTopology::eLineStrip; break;
				case TINYGLTF_MODE_LINE_STRIP: 		topology = vk::PrimitiveTopology::eLineStrip; break;
				case TINYGLTF_MODE_TRIANGLES: 		topology = vk::PrimitiveTopology::eTriangleList; break;
				case TINYGLTF_MODE_TRIANGLE_STRIP: 	topology = vk::PrimitiveTopology::eTriangleStrip; break;
				case TINYGLTF_MODE_TRIANGLE_FAN: 	topology = vk::PrimitiveTopology::eTriangleFan; break;
			}

			for (const auto&[attribName,attribIndex] : prim.attributes) {
				const tinygltf::Accessor& accessor = model.accessors[attribIndex];

				static const unordered_map<int, unordered_map<int, vk::Format>> formatMap {
					{ TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE, {
						{ TINYGLTF_TYPE_SCALAR, vk::Format::eR8Uint },
						{ TINYGLTF_TYPE_VEC2, 	vk::Format::eR8G8Uint },
						{ TINYGLTF_TYPE_VEC3, 	vk::Format::eR8G8B8Uint },
						{ TINYGLTF_TYPE_VEC4, 	vk::Format::eR8G8B8A8Uint },
					} },
					{ TINYGLTF_COMPONENT_TYPE_BYTE, {
						{ TINYGLTF_TYPE_SCALAR, vk::Format::eR8Sint },
						{ TINYGLTF_TYPE_VEC2, 	vk::Format::eR8G8Sint },
						{ TINYGLTF_TYPE_VEC3, 	vk::Format::eR8G8B8Sint },
						{ TINYGLTF_TYPE_VEC4, 	vk::Format::eR8G8B8A8Sint },
					} },
					{ TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT, {
						{ TINYGLTF_TYPE_SCALAR, vk::Format::eR16Uint },
						{ TINYGLTF_TYPE_VEC2, 	vk::Format::eR16G16Uint },
						{ TINYGLTF_TYPE_VEC3, 	vk::Format::eR16G16B16Uint },
						{ TINYGLTF_TYPE_VEC4, 	vk::Format::eR16G16B16A16Uint },
					} },
					{ TINYGLTF_COMPONENT_TYPE_SHORT, {
						{ TINYGLTF_TYPE_SCALAR, vk::Format::eR16Sint },
						{ TINYGLTF_TYPE_VEC2, 	vk::Format::eR16G16Sint },
						{ TINYGLTF_TYPE_VEC3, 	vk::Format::eR16G16B16Sint },
						{ TINYGLTF_TYPE_VEC4, 	vk::Format::eR16G16B16A16Sint },
					} },
					{ TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT, {
						{ TINYGLTF_TYPE_SCALAR, vk::Format::eR32Uint },
						{ TINYGLTF_TYPE_VEC2, 	vk::Format::eR32G32Uint },
						{ TINYGLTF_TYPE_VEC3, 	vk::Format::eR32G32B32Uint },
						{ TINYGLTF_TYPE_VEC4, 	vk::Format::eR32G32B32A32Uint },
					} },
					{ TINYGLTF_COMPONENT_TYPE_INT, {
						{ TINYGLTF_TYPE_SCALAR, vk::Format::eR32Sint },
						{ TINYGLTF_TYPE_VEC2, 	vk::Format::eR32G32Sint },
						{ TINYGLTF_TYPE_VEC3, 	vk::Format::eR32G32B32Sint },
						{ TINYGLTF_TYPE_VEC4, 	vk::Format::eR32G32B32A32Sint },
					} },
					{ TINYGLTF_COMPONENT_TYPE_FLOAT, {
						{ TINYGLTF_TYPE_SCALAR, vk::Format::eR32Sfloat },
						{ TINYGLTF_TYPE_VEC2, 	vk::Format::eR32G32Sfloat },
						{ TINYGLTF_TYPE_VEC3, 	vk::Format::eR32G32B32Sfloat },
						{ TINYGLTF_TYPE_VEC4, 	vk::Format::eR32G32B32A32Sfloat },
					} },
					{ TINYGLTF_COMPONENT_TYPE_DOUBLE, {
						{ TINYGLTF_TYPE_SCALAR, vk::Format::eR64Sfloat },
						{ TINYGLTF_TYPE_VEC2, 	vk::Format::eR64G64Sfloat },
						{ TINYGLTF_TYPE_VEC3, 	vk::Format::eR64G64B64Sfloat },
						{ TINYGLTF_TYPE_VEC4, 	vk::Format::eR64G64B64A64Sfloat },
					} }
				};
				vk::Format attributeFormat = formatMap.at(accessor.componentType).at(accessor.type);

				Mesh::VertexAttributeType attributeType;
				uint32_t typeIndex = 0;
				// parse typename & typeindex
				{
					string typeName;
					typeName.resize(attribName.size());
					ranges::transform(attribName, typeName.begin(), [&](char c) { return tolower(c); });
					size_t c = typeName.find_first_of("0123456789");
					if (c != string::npos) {
						typeIndex = stoi(typeName.substr(c));
						typeName = typeName.substr(0, c);
					}
					if (typeName.back() == '_') typeName.pop_back();
					static const unordered_map<string, Mesh::VertexAttributeType> semanticMap {
						{ "position", 	Mesh::VertexAttributeType::ePosition },
						{ "normal", 	Mesh::VertexAttributeType::eNormal },
						{ "tangent", 	Mesh::VertexAttributeType::eTangent },
						{ "bitangent", 	Mesh::VertexAttributeType::eBinormal },
						{ "texcoord", 	Mesh::VertexAttributeType::eTexcoord },
						{ "color", 		Mesh::VertexAttributeType::eColor },
						{ "psize", 		Mesh::VertexAttributeType::ePointSize },
						{ "pointsize", 	Mesh::VertexAttributeType::ePointSize },
						{ "joints",     Mesh::VertexAttributeType::eBlendIndex },
						{ "weights",    Mesh::VertexAttributeType::eBlendWeight }
					};
					attributeType = semanticMap.at(typeName);
				}

				auto& attribs = vertexData[attributeType];
				if (attribs.size() <= typeIndex) attribs.resize(typeIndex+1);
				const tinygltf::BufferView& bv = model.bufferViews[accessor.bufferView];
				const uint32_t stride = accessor.ByteStride(bv);
				attribs[typeIndex] = {
					Buffer::View<byte>(buffers[bv.buffer], bv.byteOffset + accessor.byteOffset, stride*accessor.count),
					Mesh::VertexAttributeDescription(stride, attributeFormat, 0, vk::VertexInputRate::eVertex) };

				if (attributeType == Mesh::VertexAttributeType::ePosition) {
					vertexData.mAabb.minX = (float)accessor.minValues[0];
					vertexData.mAabb.minY = (float)accessor.minValues[1];
					vertexData.mAabb.minZ = (float)accessor.minValues[2];
					vertexData.mAabb.maxX = (float)accessor.maxValues[0];
					vertexData.mAabb.maxY = (float)accessor.maxValues[1];
					vertexData.mAabb.maxZ = (float)accessor.maxValues[2];
				}
			}

			meshes[i][j] = make_shared<Mesh>(vertexData, indexBuffer, topology);
		}
	}
	cout << endl;

	cout << "Loading primitives...";
	vector<shared_ptr<Node>> nodes(model.nodes.size());
	for (size_t n = 0; n < model.nodes.size(); n++) {
		cout << "\rLoading primitives " << (n+1) << "/" << model.nodes.size() << "     ";

		const auto& node = model.nodes[n];
		const shared_ptr<Node> dst = rootNode->addChild(node.name);
		nodes[n] = dst;

		// compute transform

		if (!node.translation.empty() || !node.rotation.empty() || !node.scale.empty()) {
			float3 translate;
			float3 scale;
			if (node.translation.empty()) translate = float3::Zero();
			else translate = double3::Map(node.translation.data()).cast<float>();
			if (node.scale.empty()) scale = float3::Ones();
			else scale = double3::Map(node.scale.data()).cast<float>();
			const quatf rotate = node.rotation.empty() ? quatf::identity() : normalize(quatf((float)node.rotation[0], (float)node.rotation[1], (float)node.rotation[2], (float)node.rotation[3]));
			dst->makeComponent<TransformData>(translate, rotate, scale);
		} else if (!node.matrix.empty())
			dst->makeComponent<TransformData>(Eigen::Array<double,4,4>::Map(node.matrix.data()).block<3,4>(0,0).cast<float>());

		// make node for MeshPrimitive

		if (node.mesh < model.meshes.size())
			for (uint32_t i = 0; i < model.meshes[node.mesh].primitives.size(); i++) {
				const auto& prim = model.meshes[node.mesh].primitives[i];
				dst->addChild(model.meshes[node.mesh].name)->makeComponent<MeshPrimitive>(materials[prim.material], meshes[node.mesh][i]);
			}

		auto light_it = node.extensions.find("KHR_lights_punctual");
		if (light_it != node.extensions.end() && light_it->second.Has("light")) {
			const tinygltf::Light& l = model.lights[light_it->second.Get("light").GetNumberAsInt()];
			if (l.type == "point" && l.extras.Has("radius")) {
				auto sphere = dst->addChild(l.name)->makeComponent<SpherePrimitive>();
				sphere->mRadius = (float)l.extras.Get("radius").GetNumberAsDouble();
				Material m;
				const float3 emission = double3::Map(l.color.data()).cast<float>();
				m.mMaterialData.setBaseColor(float3::Zero());
				m.mMaterialData.setEmission(emission * (float)(l.intensity / (4*M_PI*sphere->mRadius*sphere->mRadius)));
				sphere->mMaterial = make_shared<Material>(m);
			}
		}
	}
	cout << endl;

	for (size_t i = 0; i < model.nodes.size(); i++)
		for (int c : model.nodes[i].children)
			nodes[i]->addChild(nodes[c]);

	cout << "Loaded " << filename << endl;

	return rootNode;
}

}