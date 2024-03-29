#ifdef ENABLE_ASSIMP

#include <App/Scene.hpp>
#include <Core/Profiler.hpp>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <portable-file-dialogs.h>

namespace stm2 {

shared_ptr<Node> Scene::loadAssimp(CommandBuffer& commandBuffer, const filesystem::path& filename) {
	cout << "Loading " << filename << endl;

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
		return nullptr;
	}

	cout << "Processing scene data..." << endl;

	vector<shared_ptr<Material>> materials;
	vector<shared_ptr<Mesh>> meshes;
	unordered_map<string, Image::View> images;

	auto getImage = [&](filesystem::path path, const bool srgb) -> Image::View {
		if (path.is_relative()) {
			filesystem::path cur = filesystem::current_path();
			filesystem::current_path(filename.parent_path());
			path = filesystem::absolute(path);
			filesystem::current_path(cur);
		}
		auto it = images.find(path.string());
		if (it != images.end()) return it->second;

		Image::Metadata md = {};
		shared_ptr<Buffer> pixels;
		tie(pixels, md.mFormat, md.mExtent) = Image::loadFile(commandBuffer.mDevice, path, srgb);
		md.mUsage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
		const shared_ptr<Image> img = make_shared<Image>(commandBuffer.mDevice, path.filename().string(), md);

		pixels->copyToImage(commandBuffer, img);
		commandBuffer.trackResource(pixels);

		images.emplace(path.string(), img);
		return img;
	};

	if (scene->HasLights())
		cout << "Warning: punctual lights are unsupported" << endl;

	const shared_ptr<Node> root = Node::create(filename.stem().string());

	if (scene->HasMaterials()) {
		bool interpret_as_pbr = false;
		if (filename.extension().string() == ".fbx") {
			//pfd::message n("Load PBR materials?", "Interpret diffuse/specular as glTF basecolor/pbr textures?", pfd::choice::yes_no);
			//interpret_as_pbr = n.result() == pfd::button::yes;
			interpret_as_pbr = true;
		}

		cout << "Loading materials...";

		const shared_ptr<Node> materialsNode = root->addChild("materials");
		for (int i = 0; i < scene->mNumMaterials; i++) {
			cout << "\rLoading materials " << (i+1) << "/" << scene->mNumMaterials << "     ";

			aiMaterial* m = scene->mMaterials[i];
			Material& material = *materials.emplace_back(materialsNode->addChild(m->GetName().C_Str())->makeComponent<Material>());

			ImageValue<3> diffuse       = ImageValue<3>{ float3::Ones() };
			ImageValue<4> specular      = ImageValue<4>{ interpret_as_pbr ? float4(1,.5f,0,0) : float4::Ones() };
			ImageValue<3> transmittance = ImageValue<3>{ float3::Zero() };
			ImageValue<3> emission      = ImageValue<3>{ float3::Zero() };
			float eta = 1.45f;

            aiColor3D tmp_color;
            if (m->Get(AI_MATKEY_COLOR_DIFFUSE    , tmp_color) == AI_SUCCESS) diffuse.mValue       = float3(tmp_color.r, tmp_color.g, tmp_color.b);
            if (m->Get(AI_MATKEY_COLOR_SPECULAR   , tmp_color) == AI_SUCCESS) specular.mValue      = float4(tmp_color.r, tmp_color.g, tmp_color.b, 1);
            if (m->Get(AI_MATKEY_COLOR_TRANSPARENT, tmp_color) == AI_SUCCESS) transmittance.mValue = float3(tmp_color.r, tmp_color.g, tmp_color.b);
            if (m->Get(AI_MATKEY_COLOR_EMISSIVE   , tmp_color) == AI_SUCCESS) emission.mValue      = float3(tmp_color.r, tmp_color.g, tmp_color.b);
			m->Get(AI_MATKEY_REFRACTI, eta);

			if (m->GetTextureCount(aiTextureType_DIFFUSE) > 0) {
				aiString aiPath;
				m->GetTexture(aiTextureType_DIFFUSE, 0, &aiPath);
				diffuse.mImage = getImage(aiPath.C_Str(), true);
			}
			if (m->GetTextureCount(aiTextureType_SPECULAR) > 0) {
				aiString aiPath;
				m->GetTexture(aiTextureType_SPECULAR, 0, &aiPath);
				specular.mImage = getImage(aiPath.C_Str(), false);
			}
			if (m->GetTextureCount(aiTextureType_EMISSIVE) > 0) {
				aiString aiPath;
				m->GetTexture(aiTextureType_EMISSIVE, 0, &aiPath);
				emission = ImageValue<3>{float3::Ones(), getImage(aiPath.C_Str(), true)};
			}

			if (interpret_as_pbr)
				material = makeMetallicRoughnessMaterial(commandBuffer, diffuse, specular, transmittance, eta, emission);
			else
				material = makeDiffuseSpecularMaterial(commandBuffer, diffuse, ImageValue<3>{specular.mValue.head<3>(), specular.mImage}, ImageValue<1>{float1(1.f)}, transmittance, eta, emission);

			if (m->GetTextureCount(aiTextureType_NORMALS) > 0) {
				aiString aiPath;
				m->GetTexture(aiTextureType_NORMALS, 0, &aiPath);
				material.mBumpImage = getImage(aiPath.C_Str(), false);
			} else if (m->GetTextureCount(aiTextureType_HEIGHT) > 0) {
				aiString aiPath;
				m->GetTexture(aiTextureType_HEIGHT, 0, &aiPath);
				material.mBumpImage = getImage(aiPath.C_Str(), false);
			}

			material.mBumpStrength = 1;
            m->Get(AI_MATKEY_BUMPSCALING, material.mBumpStrength);
		}
		cout << endl;
	}

	if (scene->HasMeshes()) {
		cout << "Loading meshes...";

		size_t vertexDataSize = 0;
		size_t indexDataSize = 0;

		vector<size_t> positionsOffsets(scene->mNumMeshes);
		vector<size_t> normalsOffsets(scene->mNumMeshes);
		vector<size_t> uvsOffsets(scene->mNumMeshes);
		vector<size_t> indicesOffsets(scene->mNumMeshes);

		for (int i = 0; i < scene->mNumMeshes; i++) {
			const aiMesh* m = scene->mMeshes[i];

			if (!(m->mPrimitiveTypes & aiPrimitiveType_TRIANGLE) || (m->mPrimitiveTypes & ~aiPrimitiveType_TRIANGLE) != 0)
				continue;

			positionsOffsets[i] = vertexDataSize;
			vertexDataSize += m->mNumVertices*3;

			normalsOffsets[i] = vertexDataSize;
			vertexDataSize += m->mNumVertices*3;

			if (m->GetNumUVChannels() > 0) {
				uvsOffsets[i] = vertexDataSize;
				vertexDataSize += m->mNumVertices*2;
			}

			indicesOffsets[i] = indexDataSize;
			indexDataSize += m->mNumFaces*3;
		}

		Buffer::View<float> vertexBufferTmp   = make_shared<Buffer>(commandBuffer.mDevice, "tmp vertices" , vertexDataSize*sizeof(float), vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		Buffer::View<uint32_t> indexBufferTmp = make_shared<Buffer>(commandBuffer.mDevice, "tmp indices" , indexDataSize*sizeof(uint32_t), vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);

		// copy vertex data to staging buffers
		auto copyVertices = [&](const uint32_t i) {
			const aiMesh* m = scene->mMeshes[i];
			if (!(m->mPrimitiveTypes & aiPrimitiveType_TRIANGLE) || (m->mPrimitiveTypes & ~aiPrimitiveType_TRIANGLE) != 0)
				return;

			const size_t offsetPos = positionsOffsets[i];
			for (int vi = 0; vi < m->mNumVertices; vi++)
				float3::Map(&vertexBufferTmp[offsetPos + 3*vi]) = float3((float)m->mVertices[vi].x, (float)m->mVertices[vi].y, (float)m->mVertices[vi].z);

			const size_t offsetNorm = normalsOffsets[i];
			for (int vi = 0; vi < m->mNumVertices; vi++)
				float3::Map(&vertexBufferTmp[offsetNorm + 3*vi]) = float3((float)m->mNormals[vi].x, (float)m->mNormals[vi].y, (float)m->mNormals[vi].z);

			if (m->GetNumUVChannels() >= 1) {
				const size_t offsetUv = uvsOffsets[i];
				for (int vi = 0; vi < m->mNumVertices; vi++)
					float2::Map(&vertexBufferTmp[offsetUv + 2*vi]) = float2((float)m->mTextureCoords[0][vi].x, (float)m->mTextureCoords[0][vi].y);
			}

			const size_t offsetIdx = indicesOffsets[i];
			for (int fi = 0; fi < m->mNumFaces; fi++) {
				const uint32_t idx = fi*3;
				indexBufferTmp[offsetIdx + idx+0] = m->mFaces[fi].mIndices[0];
				indexBufferTmp[offsetIdx + idx+1] = m->mFaces[fi].mIndices[1];
				indexBufferTmp[offsetIdx + idx+2] = m->mFaces[fi].mIndices[2];
			}
		};

		vector<thread> threads;
		threads.reserve(scene->mNumMeshes);
		for (uint32_t i = 0; i < scene->mNumMeshes; i++)
			threads.push_back(thread(bind(copyVertices, i)));
		cout << "Copying vertex data...";
		for (thread& t : threads) if (t.joinable()) t.join();
		cout << endl;


		vk::BufferUsageFlags bufferUsage = vk::BufferUsageFlagBits::eTransferSrc|vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eStorageBuffer;
		if (commandBuffer.mDevice.accelerationStructureFeatures().accelerationStructure) {
			bufferUsage |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
			bufferUsage |= vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
		}

		Buffer::View<float> vertexBuffer   = make_shared<Buffer>(commandBuffer.mDevice, filename.stem().string() + "/Vertices" , vertexDataSize*sizeof(float), bufferUsage|vk::BufferUsageFlagBits::eVertexBuffer);
		Buffer::View<uint32_t> indexBuffer = make_shared<Buffer>(commandBuffer.mDevice, filename.stem().string() + "/Indices" , indexDataSize*sizeof(uint32_t), bufferUsage|vk::BufferUsageFlagBits::eIndexBuffer);
		Buffer::copy(commandBuffer, vertexBufferTmp, vertexBuffer);
		Buffer::copy(commandBuffer, indexBufferTmp , indexBuffer);

		// construct meshes

		const shared_ptr<Node>& meshesNode = root->addChild("meshes");
		for (int i = 0; i < scene->mNumMeshes; i++) {
			cout << "\rCreating meshes " << (i+1) << "/" << scene->mNumMeshes;

			const aiMesh* m = scene->mMeshes[i];

			if (!(m->mPrimitiveTypes & aiPrimitiveType_TRIANGLE) || (m->mPrimitiveTypes & ~aiPrimitiveType_TRIANGLE) != 0)
				continue;

			Mesh::Vertices vertices;

			vertices[Mesh::VertexAttributeType::ePosition].emplace_back(
				Buffer::View<byte>(vertexBuffer.buffer(), positionsOffsets[i]*sizeof(float), m->mNumVertices*3*sizeof(float)),
				Mesh::VertexAttributeDescription{ (uint32_t)sizeof(float3), vk::Format::eR32G32B32Sfloat, 0, vk::VertexInputRate::eVertex });

			vertices[Mesh::VertexAttributeType::eNormal].emplace_back(
				Buffer::View<byte>(vertexBuffer.buffer(), normalsOffsets[i]*sizeof(float), m->mNumVertices*3*sizeof(float)),
				Mesh::VertexAttributeDescription{ (uint32_t)sizeof(float3), vk::Format::eR32G32B32Sfloat, 0, vk::VertexInputRate::eVertex });

			if (m->GetNumUVChannels() >= 1) {
				vertices[Mesh::VertexAttributeType::eTexcoord].emplace_back(
					Buffer::View<byte>(vertexBuffer.buffer(), uvsOffsets[i]*sizeof(float), m->mNumVertices*2*sizeof(float)),
					Mesh::VertexAttributeDescription{ (uint32_t)sizeof(float2), vk::Format::eR32G32Sfloat, 0, vk::VertexInputRate::eVertex } );
			}

			vertices.mAabb.minX = (float)m->mAABB.mMin.x;
			vertices.mAabb.minY = (float)m->mAABB.mMin.y;
			vertices.mAabb.minZ = (float)m->mAABB.mMin.z;
			vertices.mAabb.maxX = (float)m->mAABB.mMax.x;
			vertices.mAabb.maxY = (float)m->mAABB.mMax.y;
			vertices.mAabb.maxZ = (float)m->mAABB.mMax.z;

			const shared_ptr<Node>& meshNode = meshesNode->addChild(m->mName.C_Str());
			meshes.emplace_back( meshNode->makeComponent<Mesh>(vertices, Buffer::View<uint32_t>(indexBuffer, indicesOffsets[i], m->mNumFaces*3), vk::PrimitiveTopology::eTriangleList) );
		}
		cout << endl;
	}

	stack<pair<aiNode*, Node*>> nodes;
	nodes.push(make_pair(scene->mRootNode, root->addChild(scene->mRootNode->mName.C_Str()).get()));
	while (!nodes.empty()) {
		auto[an, n] = nodes.top();
		nodes.pop();

		n->makeComponent<TransformData>( Eigen::Array<ai_real,4,4,Eigen::RowMajor>::Map(&an->mTransformation.a1).block<3,4>(0,0).cast<float>() );

		if (an->mNumMeshes == 1)
			n->makeComponent<MeshPrimitive>(materials[scene->mMeshes[an->mMeshes[0]]->mMaterialIndex], meshes[an->mMeshes[0]]);
		else if (an->mNumMeshes > 1)
			for (int i = 0; i < an->mNumMeshes; i++)
				n->addChild(scene->mMeshes[an->mMeshes[i]]->mName.C_Str())->makeComponent<MeshPrimitive>(materials[scene->mMeshes[an->mMeshes[i]]->mMaterialIndex], meshes[an->mMeshes[i]]);

		for (int i = 0; i < an->mNumChildren; i++)
			nodes.push(make_pair(an->mChildren[i], n->addChild(an->mChildren[i]->mName.C_Str()).get()));
	}

	cout << "Loaded " << filename << endl;
	return root;
}

}

#endif