#include <App/Scene.hpp>

#ifdef STRATUM_ENABLE_ASSIMP
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#endif

#include <portable-file-dialogs.h>

namespace tinyvkpt {

#ifdef STRATUM_ENABLE_ASSIMP
NodePtr Scene::loadAssimp(CommandBuffer& commandBuffer, const filesystem::path& filename) {
	ProfilerScope ps("loadAssimp", &commandBuffer);

	Device& device = commandBuffer.mDevice;

	// Create an instance of the Importer class
	Assimp::Importer importer;
	// And have it read the given file with some example postprocessing
	// Usually - if speed is not the most important aspect for you - you'll
	// propably to request more postprocessing than we do in this example.
	uint32_t flags = aiProcessPreset_TargetRealtime_MaxQuality | aiProcess_TransformUVCoords;
	flags &= ~(aiProcess_CalcTangentSpace); // Never use Assimp's tangent gen code
	flags &= ~(aiProcess_FindDegenerates); // Avoid converting degenerated triangles to lines
	//flags &= ~(aiProcess_RemoveRedundantMaterials);
	flags &= ~(aiProcess_SplitLargeMeshes);

	int removeFlags = aiComponent_COLORS;
	for (uint32_t uvLayer = 1; uvLayer < AI_MAX_NUMBER_OF_TEXTURECOORDS; uvLayer++) removeFlags |= aiComponent_TEXCOORDSn(uvLayer);
	importer.SetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS, removeFlags);

	const aiScene* scene = importer.ReadFile(filename.string(), flags);

	// If the import failed, report it
	if (!scene) {
		cout << "Failed to load " << filename << ": " << importer.GetErrorString() << endl;
		return;
	}

	vector<component_ptr<Material>> materials;
	vector<component_ptr<Mesh>> meshes;
	unordered_map<string, Image::View> images;

	auto get_image = [&](fs::path path, bool srgb) -> Image::View {
		if (path.is_relative()) {
			fs::path cur = fs::current_path();
			fs::current_path(filename.parent_path());
			path = fs::absolute(path);
			fs::current_path(cur);
		}
		auto it = images.find(path.string());
		if (it != images.end()) return it->second;
		ImageData pixels = load_image_data(device, path, srgb);
		auto img = make_shared<Image>(commandBuffer, path.filename().string(), pixels, 1);
		commandBuffer.hold_resource(img);
		images.emplace(path.string(), img);
		return img;
	};

	if (scene->HasLights())
		cout << "Warning: punctual lights are unsupported" << endl;

	if (scene->HasMaterials()) {
		bool interpret_as_pbr = false;
		if (filename.extension().string() == ".fbx") {
			//pfd::message n("Load PBR materials?", "Interpret diffuse/specular as glTF basecolor/pbr textures?", pfd::choice::yes_no);
			//interpret_as_pbr = n.result() == pfd::button::yes;
			interpret_as_pbr = true;
		}

		cout << "Loading materials...";

		const NodePtr materials_node = root.addChild("materials");
		for (int i = 0; i < scene->mNumMaterials; i++) {
			cout << "\rLoading materials " << (i+1) << "/" << scene->mNumMaterials << "     ";

			aiMaterial* m = scene->mMaterials[i];
			Material& material = *materials.emplace_back(materials_node.make_child(m->GetName().C_Str()).makeComponent<Material>());

			ImageValue3 diffuse       = ImageValue3{ float3::Ones() };
			ImageValue4 specular      = ImageValue4{ interpret_as_pbr ? float4(1,.5f,0,0) : float4::Ones() };
			ImageValue3 transmittance = ImageValue3{ float3::Zero() };
			ImageValue3 emission      = ImageValue3{ float3::Zero() };
			float eta = 1.45f;

            aiColor3D tmp_color;
            if (m->Get(AI_MATKEY_COLOR_DIFFUSE, tmp_color) == AI_SUCCESS) diffuse.value  = float3(tmp_color.r, tmp_color.g, tmp_color.b);
            if (m->Get(AI_MATKEY_COLOR_SPECULAR, tmp_color) == AI_SUCCESS) specular.value = float4(tmp_color.r, tmp_color.g, tmp_color.b, 1);
            if (m->Get(AI_MATKEY_COLOR_TRANSPARENT, tmp_color) == AI_SUCCESS) transmittance.value = float3(tmp_color.r, tmp_color.g, tmp_color.b);
            if (m->Get(AI_MATKEY_COLOR_EMISSIVE, tmp_color) == AI_SUCCESS) emission.value = float3(tmp_color.r, tmp_color.g, tmp_color.b);
			m->Get(AI_MATKEY_REFRACTI, eta);

			if (m->GetTextureCount(aiTextureType_DIFFUSE) > 0) {
				aiString aiPath;
				m->GetTexture(aiTextureType_DIFFUSE, 0, &aiPath);
				diffuse.image = get_image(aiPath.C_Str(), true);
			}
			if (m->GetTextureCount(aiTextureType_SPECULAR) > 0) {
				aiString aiPath;
				m->GetTexture(aiTextureType_SPECULAR, 0, &aiPath);
				specular.image = get_image(aiPath.C_Str(), false);
			}
			if (m->GetTextureCount(aiTextureType_EMISSIVE) > 0) {
				aiString aiPath;
				m->GetTexture(aiTextureType_EMISSIVE, 0, &aiPath);
				emission = ImageValue3{float3::Ones(), get_image(aiPath.C_Str(), true)};
			}

			if (interpret_as_pbr)
				material = root.find_in_ancestor<Scene>()->make_metallic_roughness_material(commandBuffer, diffuse, specular, transmittance, eta, emission);
			else
				material = root.find_in_ancestor<Scene>()->make_diffuse_specular_material(commandBuffer, diffuse, ImageValue3{specular.image,specular.value.head<3>()}, ImageValue1{1.f}, transmittance, eta, emission);

			if (m->GetTextureCount(aiTextureType_NORMALS) > 0) {
				aiString aiPath;
				m->GetTexture(aiTextureType_NORMALS, 0, &aiPath);
				material.bump_image = get_image(aiPath.C_Str(), false);
			} else if (m->GetTextureCount(aiTextureType_HEIGHT) > 0) {
				aiString aiPath;
				m->GetTexture(aiTextureType_HEIGHT, 0, &aiPath);
				material.bump_image = get_image(aiPath.C_Str(), false);
			}

			material.bump_strength = 1;
            m->Get(AI_MATKEY_BUMPSCALING, material.bump_strength);
		}
		cout << endl;
	}

	if (scene->HasMeshes()) {
		vk::BufferUsageFlags bufferUsage = vk::BufferUsageFlagBits::eVertexBuffer|vk::BufferUsageFlagBits::eIndexBuffer|vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eTransferSrc;
		#ifdef VK_KHR_buffer_device_address
		bufferUsage |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
		bufferUsage |= vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
		#endif

		cout << "Loading meshes...";

		vector<Buffer::View<float3>> positions_tmp(scene->mNumMeshes);
		vector<Buffer::View<float3>> normals_tmp(scene->mNumMeshes);
		vector<Buffer::View<float2>> uvs_tmp(scene->mNumMeshes);
		vector<Buffer::View<uint32_t>> indices_tmp(scene->mNumMeshes);

		for (int i = 0; i < scene->mNumMeshes; i++) {
			cout << "\rCreating vertex buffers " << (i+1) << "/" << scene->mNumMeshes;

			const aiMesh* m = scene->mMeshes[i];

			if (!(m->mPrimitiveTypes & aiPrimitiveType_TRIANGLE) || (m->mPrimitiveTypes & ~aiPrimitiveType_TRIANGLE) != 0)
				continue;

			positions_tmp[i] = make_shared<Buffer>(commandBuffer.mDevice, "tmp vertices", m->mNumVertices*sizeof(float3), vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
			normals_tmp[i] = make_shared<Buffer>(commandBuffer.mDevice, "tmp normals" , m->mNumVertices*sizeof(float3), vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
			indices_tmp[i] = make_shared<Buffer>(commandBuffer.mDevice, "tmp indices" , m->mNumFaces*3*sizeof(uint32_t), vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
			if (m->GetNumUVChannels() >= 1)
				uvs_tmp[i] = make_shared<Buffer>(commandBuffer.mDevice, "tmp uvs", m->mNumVertices*sizeof(float2), vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		}
		cout << endl;

		auto copy_vertices_fn = [&](uint32_t i) {
			const aiMesh* m = scene->mMeshes[i];
			if (!(m->mPrimitiveTypes & aiPrimitiveType_TRIANGLE) || (m->mPrimitiveTypes & ~aiPrimitiveType_TRIANGLE) != 0)
				return;

			for (int vi = 0; vi < m->mNumVertices; vi++)
				positions_tmp[i][vi] = float3((float)m->mVertices[vi].x, (float)m->mVertices[vi].y, (float)m->mVertices[vi].z);
			for (int vi = 0; vi < m->mNumVertices; vi++)
				normals_tmp[i][vi] = float3((float)m->mNormals[vi].x, (float)m->mNormals[vi].y, (float)m->mNormals[vi].z);
			if (m->GetNumUVChannels() >= 1)
				for (int vi = 0; vi < m->mNumVertices; vi++)
					uvs_tmp[i][vi] = float2(m->mTextureCoords[0][vi].x, m->mTextureCoords[0][vi].y);

			for (int fi = 0; fi < m->mNumFaces; fi++) {
				const uint32_t idx = fi*3;
				indices_tmp[i][idx+0] = m->mFaces[fi].mIndices[0];
				indices_tmp[i][idx+1] = m->mFaces[fi].mIndices[1];
				indices_tmp[i][idx+2] = m->mFaces[fi].mIndices[2];
			}
		};

		vector<thread> threads;
		threads.reserve(scene->mNumMeshes);
		for (uint32_t i = 0; i < scene->mNumMeshes; i++)
			threads.push_back(thread(bind(copy_vertices_fn, i)));
		cout << "Copying vertex data...";
		for (thread& t : threads) if (t.joinable()) t.join();
		cout << endl;

		Node& meshes_node = root.make_child("meshes");
		for (int i = 0; i < scene->mNumMeshes; i++) {
			cout << "\rCreating meshes " << (i+1) << "/" << scene->mNumMeshes;

			const aiMesh* m = scene->mMeshes[i];

			if (!(m->mPrimitiveTypes & aiPrimitiveType_TRIANGLE) || (m->mPrimitiveTypes & ~aiPrimitiveType_TRIANGLE) != 0)
				continue;

			vk::BufferUsageFlags bufferUsage = vk::BufferUsageFlagBits::eTransferSrc|vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eStorageBuffer;
			#ifdef VK_KHR_buffer_device_address
			bufferUsage |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
			bufferUsage |= vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
			#endif

			auto vao = make_shared<VertexArrayObject>(unordered_map<VertexArrayObject::AttributeType, vector<VertexArrayObject::Attribute>>{
				{ VertexArrayObject::AttributeType::ePosition, { {
					VertexArrayObject::AttributeDescription{ (uint32_t)sizeof(float3), vk::Format::eR32G32B32Sfloat, 0, vk::VertexInputRate::eVertex },
					make_shared<Buffer>(commandBuffer.mDevice, filename.stem().string() + " vertices", positions_tmp[i].size_bytes(), bufferUsage|vk::BufferUsageFlagBits::eVertexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal) } } },
				{ VertexArrayObject::AttributeType::eNormal, { {
					VertexArrayObject::AttributeDescription{ (uint32_t)sizeof(float3), vk::Format::eR32G32B32Sfloat, 0, vk::VertexInputRate::eVertex },
					make_shared<Buffer>(commandBuffer.mDevice, filename.stem().string() + " normals", normals_tmp[i].size_bytes(), bufferUsage|vk::BufferUsageFlagBits::eVertexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal) } } },
			});

			Buffer::View<uint32_t> indexBuffer = make_shared<Buffer>(commandBuffer.mDevice, filename.stem().string() + " indices", indices_tmp[i].size_bytes(), bufferUsage|vk::BufferUsageFlagBits::eIndexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
			commandBuffer.copy_buffer(positions_tmp[i], vao->at(VertexArrayObject::AttributeType::ePosition)[0].second);
			commandBuffer.copy_buffer(normals_tmp[i], vao->at(VertexArrayObject::AttributeType::eNormal)[0].second);
			commandBuffer.copy_buffer(indices_tmp[i], indexBuffer);

			if (m->GetNumUVChannels() >= 1) {
				(*vao)[VertexArrayObject::AttributeType::eTexcoord].emplace_back(
					VertexArrayObject::AttributeDescription{ (uint32_t)sizeof(float2), vk::Format::eR32G32Sfloat, 0, vk::VertexInputRate::eVertex },
					make_shared<Buffer>(commandBuffer.mDevice, filename.stem().string() + " uvs", uvs_tmp[i].size_bytes(), bufferUsage|vk::BufferUsageFlagBits::eVertexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal));
				commandBuffer.copy_buffer(uvs_tmp[i], vao->at(VertexArrayObject::AttributeType::eTexcoord)[0].second);
			}

			Node& mesh_node = meshes_node.make_child(m->mName.C_Str());
			meshes.emplace_back( mesh_node.makeComponent<Mesh>(vao, indexBuffer, vk::PrimitiveTopology::eTriangleList) );
		}
		cout << endl;
	}

	stack<pair<aiNode*, Node*>> nodes;
	nodes.push(make_pair(scene->mRootNode, &root.make_child(scene->mRootNode->mName.C_Str())));
	while (!nodes.empty()) {
		auto[an, n] = nodes.top();
		nodes.pop();

		n->makeComponent<TransformData>( from_float3x4(Eigen::Array<ai_real,4,4,Eigen::RowMajor>::Map(&an->mTransformation.a1).block<3,4>(0,0).cast<float>()) );

		if (an->mNumMeshes == 1)
			n->makeComponent<MeshPrimitive>(materials[scene->mMeshes[an->mMeshes[0]]->mMaterialIndex], meshes[an->mMeshes[0]]);
		else if (an->mNumMeshes > 1)
			for (int i = 0; i < an->mNumMeshes; i++)
				n->make_child(scene->mMeshes[an->mMeshes[i]]->mName.C_Str()).makeComponent<MeshPrimitive>(materials[scene->mMeshes[an->mMeshes[i]]->mMaterialIndex], meshes[an->mMeshes[i]]);

		for (int i = 0; i < an->mNumChildren; i++)
			nodes.push(make_pair(an->mChildren[i], &n->make_child(an->mChildren[i]->mName.C_Str())));
	}

	cout << "Loaded " << filename << endl;
}
#endif

}