#include "Scene.hpp"
#include "Inspector.hpp"
#include "Gui.hpp"
#include <Core/Instance.hpp>
#include <Core/Profiler.hpp>
#include <Core/CommandBuffer.hpp>
#include <Core/Pipeline.hpp>
#include <Core/Window.hpp>

#include <future>
#include <portable-file-dialogs.h>

namespace stm2 {

tuple<shared_ptr<vk::raii::AccelerationStructureKHR>, Buffer::View<byte>> buildAccelerationStructure(CommandBuffer& commandBuffer, const string& name, const vk::AccelerationStructureTypeKHR type, const vk::ArrayProxy<const vk::AccelerationStructureGeometryKHR>& geometries, const vk::ArrayProxy<const vk::AccelerationStructureBuildRangeInfoKHR>& buildRanges) {
	vk::AccelerationStructureBuildGeometryInfoKHR buildGeometry(type, vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace, vk::BuildAccelerationStructureModeKHR::eBuild);
	buildGeometry.setGeometries(geometries);

	vk::AccelerationStructureBuildSizesInfoKHR buildSizes;
	if (buildRanges.size() > 0 && buildRanges.front().primitiveCount > 0) {
		vector<uint32_t> counts((uint32_t)geometries.size());
		for (uint32_t i = 0; i < geometries.size(); i++)
			counts[i] = (buildRanges.data() + i)->primitiveCount;
		buildSizes = commandBuffer.mDevice->getAccelerationStructureBuildSizesKHR(vk::AccelerationStructureBuildTypeKHR::eDevice, buildGeometry, counts);
	} else
		buildSizes.accelerationStructureSize = buildSizes.buildScratchSize = 4;

	Buffer::View<byte> buffer = make_shared<Buffer>(commandBuffer.mDevice, name, buildSizes.accelerationStructureSize, vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress);
	Buffer::View<byte> scratchData = make_shared<Buffer>(commandBuffer.mDevice, name + "/scratchData", buildSizes.buildScratchSize, vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eStorageBuffer);
	shared_ptr<vk::raii::AccelerationStructureKHR> accelerationStructure = make_shared<vk::raii::AccelerationStructureKHR>(*commandBuffer.mDevice, vk::AccelerationStructureCreateInfoKHR({}, **buffer.buffer(), buffer.offset(), buffer.sizeBytes(), type));

	buildGeometry.dstAccelerationStructure = **accelerationStructure;
	buildGeometry.scratchData = scratchData.deviceAddress();

	commandBuffer->buildAccelerationStructuresKHR(buildGeometry, buildRanges.data());

	commandBuffer.trackResource(buffer.buffer());
	commandBuffer.trackResource(scratchData.buffer());
	commandBuffer.trackVulkanResource(accelerationStructure);

	return tie(accelerationStructure, buffer);
}

TransformData nodeToWorld(const Node& node) {
	TransformData transform;
	if (auto c = node.getComponent<TransformData>(); c)
		transform = *c;
	else
		transform = TransformData(float3::Zero(), quatf::identity(), float3::Ones());
	shared_ptr<Node> p = node.parent();
	while (p) {
		if (auto c = p->getComponent<TransformData>())
			transform = tmul(*c, transform);
		p = p->parent();
	}
	return transform;
}


Scene::Scene(Node& node): mNode(node) {
	if (shared_ptr<Inspector> inspector = mNode.root()->findDescendant<Inspector>()) {
		inspector->setTypeCallback<Scene>();
		inspector->setTypeCallback<TransformData>();
		inspector->setTypeCallback<Camera>();
		inspector->setTypeCallback<MeshPrimitive>();
		inspector->setTypeCallback<SpherePrimitive>();
		inspector->setTypeCallback<Material>();
		inspector->setTypeCallback<Medium>();
		inspector->setTypeCallback<Environment>();
		inspector->setTypeCallback<nanovdb::GridMetaData>([](Node& n) {
		auto metadata = n.getComponent<nanovdb::GridMetaData>();
			ImGui::LabelText("Grid name", metadata->shortGridName());
			ImGui::LabelText("Grid count", "%u", metadata->gridCount());
			ImGui::LabelText("Grid type", nanovdb::toStr(metadata->gridType()));
			ImGui::LabelText("Grid class", nanovdb::toStr(metadata->gridClass()));
			ImGui::LabelText("Bounding box min", "%.02f %.02f %.02f", metadata->worldBBox().min()[0], metadata->worldBBox().min()[1], metadata->worldBBox().min()[2]);
			ImGui::LabelText("Bounding box max", "%.02f %.02f %.02f", metadata->worldBBox().max()[0], metadata->worldBBox().max()[1], metadata->worldBBox().max()[2]);
		});
	}

	ComputePipeline::Metadata md;
	md.mBindingFlags["gPositions"] = vk::DescriptorBindingFlagBits::ePartiallyBound;
	md.mBindingFlags["gNormals"]   = vk::DescriptorBindingFlagBits::ePartiallyBound;
	md.mBindingFlags["gTexcoords"] = vk::DescriptorBindingFlagBits::ePartiallyBound;

	const filesystem::path shaderPath = *mNode.findAncestor<Instance>()->findArgument("shaderKernelPath");
	mCopyVerticesPipeline                = ComputePipelineCache(shaderPath / "copy_vertices.hlsl", "main", "cs_6_7", {}, md);
	mConvertAlphaToRoughnessPipeline     = ComputePipelineCache(shaderPath / "roughness_convert.hlsl", "alpha_to_roughness");
	mConvertShininessToRoughnessPipeline = ComputePipelineCache(shaderPath / "roughness_convert.hlsl", "shininess_to_roughness");
	mConvertPbrPipeline                  = ComputePipelineCache(shaderPath / "material_convert.hlsl", "from_gltf_pbr");
	mConvertDiffuseSpecularPipeline      = ComputePipelineCache(shaderPath / "material_convert.hlsl", "from_diffuse_specular");

	for (const string arg : mNode.findAncestor<Instance>()->findArguments("scene"))
		mToLoad.emplace_back(arg);
}


shared_ptr<Node> Scene::loadEnvironmentMap(CommandBuffer& commandBuffer, const filesystem::path& filepath) {
	filesystem::path path = filepath;
	if (path.is_relative()) {
		filesystem::path cur = filesystem::current_path();
		filesystem::current_path(path.parent_path());
		path = filesystem::absolute(path);
		filesystem::current_path(cur);
	}

	Image::Metadata md = {};
	shared_ptr<Buffer> pixels;
	tie(pixels, md.mFormat, md.mExtent) = Image::loadFile(commandBuffer.mDevice, filepath, false);
	md.mUsage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage;
	const shared_ptr<Image> img = make_shared<Image>(commandBuffer.mDevice, filepath.filename().string(), md);

	pixels->copyToImage(commandBuffer, img);

	commandBuffer.trackResource(img);

	const shared_ptr<Node> node = Node::create(filepath.stem().string());
	node->makeComponent<Environment>(ImageValue<3>{ float3::Ones(), img });
	return node;
}

ImageValue<1> Scene::alphaToRoughness(CommandBuffer& commandBuffer, const ImageValue<1>& alpha) {
	ImageValue<1> roughness;
	roughness.mValue = sqrt(alpha.mValue);
	if (alpha.mImage) {
		Image::Metadata md;
		md.mFormat = alpha.mImage.image()->format();
		md.mExtent = alpha.mImage.extent();
		md.mLevels = Image::maxMipLevels(md.mExtent);
		md.mUsage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage;
		roughness.mImage = make_shared<Image>(commandBuffer.mDevice, "roughness", md);
		mConvertAlphaToRoughnessPipeline.get(commandBuffer.mDevice)->dispatchTiled(commandBuffer,
			alpha.mImage.extent(),
			Descriptors{
				{ { "gInput", 0 }, DescriptorValue{ ImageDescriptor{
						alpha.mImage,
						vk::ImageLayout::eShaderReadOnlyOptimal,
						vk::AccessFlagBits::eShaderRead,
						{} } } },
				{ { "gRoughnessRW", 0 }, DescriptorValue{ ImageDescriptor{
						roughness.mImage,
						vk::ImageLayout::eGeneral,
						vk::AccessFlagBits::eShaderWrite,
						{} } } }
			});
		roughness.mImage.image()->generateMipMaps(commandBuffer);
		cout << "Converted alpha to roughness: " << alpha.mImage.image()->resourceName() << endl;
	}
	return roughness;
}
ImageValue<1> Scene::shininessToRoughness(CommandBuffer& commandBuffer, const ImageValue<1>& shininess) {
	ImageValue<1> roughness;
	roughness.mValue = sqrt(2 / (shininess.mValue + 2));
	if (shininess.mImage) {
		Image::Metadata md;
		md.mFormat = shininess.mImage.image()->format();
		md.mExtent = shininess.mImage.extent();
		md.mLevels = Image::maxMipLevels(md.mExtent);
		md.mUsage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage;
		roughness.mImage = make_shared<Image>(commandBuffer.mDevice, "roughness", md);
		mConvertShininessToRoughnessPipeline.get(commandBuffer.mDevice)->dispatchTiled(commandBuffer,
			shininess.mImage.extent(),
			Descriptors{
				{ { "gInput", 0 }, ImageDescriptor{
					shininess.mImage,
					vk::ImageLayout::eShaderReadOnlyOptimal,
					vk::AccessFlagBits::eShaderRead,
					{} } },
				{ { "gRoughnessRW", 0 }, ImageDescriptor{
					roughness.mImage,
					vk::ImageLayout::eGeneral,
					vk::AccessFlagBits::eShaderWrite,
					{} } }
			});
		roughness.mImage.image()->generateMipMaps(commandBuffer);
		cout << "Converted shininess to roughness: " << shininess.mImage.image()->resourceName() << endl;
	}
	return roughness;
}

Material Scene::makeMetallicRoughnessMaterial(CommandBuffer& commandBuffer, const ImageValue<3>& baseColor, const ImageValue<4>& metallic_roughness, const ImageValue<3>& transmission, const float eta, const ImageValue<3>& emission) {
	Material m;
	if ((emission.mValue > 0).any()) {
		m.mValues[0].mImage = emission.mImage;
		m.baseColor() = emission.mValue / luminance(emission.mValue);
		m.emission() = luminance(emission.mValue);
		m.eta() = 0; // eta
		return m;
	}
	m.baseColor() = baseColor.mValue;
	m.emission() = 0;
	m.metallic() = metallic_roughness.mValue.z(); // metallic
	m.roughness() = metallic_roughness.mValue.y(); // roughness
	m.anisotropic() = 0; // anisotropic
	m.subsurface() = 0; // subsurface
	m.clearcoat() = 0; // clearcoat
	m.clearcoatGloss() = 1; // clearcoat gloss
	m.transmission() = luminance(transmission.mValue);
	m.eta() = eta;
	if (baseColor.mImage || metallic_roughness.mImage || transmission.mImage) {
		Descriptors descriptors;

		Image::View d = baseColor.mImage ? baseColor.mImage : metallic_roughness.mImage ? metallic_roughness.mImage : transmission.mImage;
		Image::Metadata md;
		md.mExtent = d.extent();
		md.mLevels = Image::maxMipLevels(md.mExtent);
		md.mFormat = vk::Format::eR8G8B8A8Unorm;
		md.mUsage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage;
		for (int i = 0; i < DisneyMaterialData::gDataSize; i++) {
			m.mValues[i].mImage = make_shared<Image>(commandBuffer.mDevice, "material data", md);
			descriptors[{ "gOutput", i }] = ImageDescriptor{ m.mValues[i].mImage, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };
		}
		if (baseColor.mImage) {
			md.mLevels = 1;
			md.mFormat = vk::Format::eR8Unorm;
			md.mUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage;
			m.mAlphaMask = make_shared<Image>(commandBuffer.mDevice, "alpha mask", md);
		}
		descriptors[{ "gOutputAlphaMask", 0 }] = ImageDescriptor{ m.mAlphaMask ? m.mAlphaMask : m.mValues[0].mImage, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };

		m.mMinAlpha = make_shared<Buffer>(commandBuffer.mDevice, "min_alpha", sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
		m.mMinAlpha[0] = 0xFFFFFFFF;
		descriptors[{ "gOutputMinAlpha", 0 }] = m.mMinAlpha;

		descriptors[{ "gDiffuse", 0 }] = ImageDescriptor{ baseColor.mImage ? baseColor.mImage : d, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{ "gSpecular", 0 }] = ImageDescriptor{ metallic_roughness.mImage ? metallic_roughness.mImage : d, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{ "gTransmittance", 0 }] = ImageDescriptor{ transmission.mImage ? transmission.mImage : d, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{ "gRoughness", 0 }] = ImageDescriptor{ d, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		Defines defs;
		if (baseColor.mImage) defs["gUseDiffuse"] = "true";
		if (metallic_roughness.mImage) defs["gUseSpecular"] = "true";
		if (transmission.mImage) defs["gUseTransmittance"] = "true";
		mConvertPbrPipeline.get(commandBuffer.mDevice, defs)->dispatchTiled(commandBuffer, d.extent(), descriptors);

		for (int i = 0; i < DisneyMaterialData::gDataSize; i++)
			m.mValues[i].mImage.image()->generateMipMaps(commandBuffer);
	}
	return m;
}
Material Scene::makeDiffuseSpecularMaterial(CommandBuffer& commandBuffer, const ImageValue<3>& diffuse, const ImageValue<3>& specular, const ImageValue<1>& roughness, const ImageValue<3>& transmission, const float eta, const ImageValue<3>& emission) {
	Material m;
	if ((emission.mValue > 0).any()) {
		m.mValues[0].mImage = emission.mImage;
		m.baseColor() = emission.mValue / luminance(emission.mValue);
		m.emission() = luminance(emission.mValue);
		m.eta() = 0; // eta
		return m;
	}
	const float ld = luminance(diffuse.mValue);
	const float ls = luminance(specular.mValue);
	const float lt = luminance(transmission.mValue);
	m.baseColor() = (diffuse.mValue * ld + specular.mValue * ls + transmission.mValue * lt) / (ld + ls + lt);
	m.emission() = 0;
	m.metallic() = ls / (ld + ls + lt);
	m.roughness() = roughness.mValue[0];
	m.anisotropic() = 0;
	m.subsurface() = 0;
	m.clearcoat() = 0;
	m.clearcoatGloss() = 1;
	m.transmission() = lt / (ld + ls + lt);
	m.eta() = eta;
	if (diffuse.mImage || specular.mImage || transmission.mImage || roughness.mImage) {
		Descriptors descriptors;

		Image::View d = diffuse.mImage ? diffuse.mImage : specular.mImage ? specular.mImage : transmission.mImage ? transmission.mImage : roughness.mImage;
		Image::Metadata md;
		md.mExtent = d.extent();
		md.mLevels = Image::maxMipLevels(md.mExtent);
		md.mFormat = vk::Format::eR8G8B8A8Unorm;
		md.mUsage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage;
		for (int i = 0; i < DisneyMaterialData::gDataSize; i++) {
			m.mValues[i].mImage = make_shared<Image>(commandBuffer.mDevice, "material data", md);
			descriptors[{"gOutput", i}] = ImageDescriptor{ m.mValues[i].mImage, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };
		}
		if (diffuse.mImage) {
			md.mLevels = 1;
			md.mFormat = vk::Format::eR8Unorm;
			md.mUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage;
			m.mAlphaMask = make_shared<Image>(commandBuffer.mDevice, "alpha mask", md);
		}
		descriptors[{"gOutputAlphaMask", 0}] = ImageDescriptor{ m.mAlphaMask ? m.mAlphaMask : m.mValues[0].mImage, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };

		m.mMinAlpha = make_shared<Buffer>(commandBuffer.mDevice, "min_alpha", sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
		m.mMinAlpha[0] = 0xFFFFFFFF;
		descriptors[{"gOutputMinAlpha", 0}] = m.mMinAlpha;

		descriptors[{"gDiffuse", 0}] = ImageDescriptor{ diffuse.mImage ? diffuse.mImage : d, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{"gSpecular", 0}] = ImageDescriptor{ specular.mImage ? specular.mImage : d, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{"gTransmittance", 0}] = ImageDescriptor{ transmission.mImage ? transmission.mImage : d, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{"gRoughness", 0}] = ImageDescriptor{ roughness.mImage ? roughness.mImage : d, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		Defines defs;
		if (diffuse.mImage)      defs["gUseDiffuse"] = "true";
		if (specular.mImage)     defs["gUseSpecular"] = "true";
		if (transmission.mImage) defs["gUseTransmittance"] = "true";
		if (roughness.mImage)    defs["gUseRoughness"] = "true";

		mConvertDiffuseSpecularPipeline.get(commandBuffer.mDevice, defs)->dispatchTiled(commandBuffer, d.extent(), descriptors);

		for (int i = 0; i < DisneyMaterialData::gDataSize; i++)
			m.mValues[i].mImage.image()->generateMipMaps(commandBuffer);
	}
	return m;
}

void Scene::update(CommandBuffer& commandBuffer, const float deltaTime) {
	ProfilerScope s("Scene::update", &commandBuffer);

	if (mAnimatedTransform) {
		TransformData& t = *mAnimatedTransform->getComponent<TransformData>();
		const float r = length(mAnimateRotate);
		const quatf rotate = (r > 0) ? quatf::angleAxis(r * deltaTime, mAnimateRotate / r) : quatf::identity();
		t = tmul(t, TransformData(mAnimateTranslate * deltaTime, rotate, float3::Ones()));
		if (!mAnimateWiggleOffset.isZero()) {
			t.m.topRightCorner(3, 1) = mAnimateWiggleBase + mAnimateWiggleOffset * sin(mAnimateWiggleTime);
			mAnimateWiggleTime += deltaTime * mAnimateWiggleSpeed;
		}
		mUpdateOnce = true;
	}

	// open file dialog
	if (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_O, false)) {
		auto f = pfd::open_file("Open scene", "", loaderFilters());
		for (const string& filepath : f.result())
			mToLoad.emplace_back(filepath);
	}

	// load input files

	if (auto w = mNode.root()->findDescendant<Window>(); w)
		for (const string& file : w->inputState().files())
			mToLoad.emplace_back(file);

	bool update = mAlwaysUpdate || mUpdateOnce;
	bool loaded = false;
	for (const string& file : mToLoad) {
		const filesystem::path filepath = file;
		mNode.addChild(load(commandBuffer, filepath));
		loaded = true;
		update = true;
	}
	mToLoad.clear();

	if (mResources)
		mResources->markUsed();

	if (!update) return;

	// Update scene data based on node graph
	// always update once after load so that motion transforms are valid

	mUpdateOnce = loaded && !mAlwaysUpdate;

	auto prevFrame = mResources;

	mResources = mResourcePool.get();
	if (!mResources)
		mResources = mResourcePool.emplace(make_shared<FrameResources>(commandBuffer.mDevice));

	mResources->update(*this, commandBuffer, prevFrame);
}

void Scene::FrameResources::update(Scene& scene, CommandBuffer& commandBuffer, const shared_ptr<FrameResources>& prevFrame) {
	// Construct resources used by renderers (mesh/material data buffers, image arrays, etc.)

	uint32_t totalVertexCount = 0;
	uint32_t totalIndexBufferSize = 0;
	Descriptors vertexCopyDescriptors;
	vector<uint4> vertexCopyInfos;
	vector<Buffer::View<byte>> indexBuffers;

	vector<pair<MeshPrimitive*, uint32_t>> meshInstanceIndices;
	vector<InstanceData> instanceDatas;
	vector<TransformData> instanceTransforms;
	vector<TransformData> instanceInverseTransforms;
	vector<TransformData> instanceMotionTransforms;
	vector<uint32_t> lightInstanceMap;

	mInstanceIndexMap = make_shared<Buffer>(commandBuffer.mDevice, "InstanceIndexMap", max<size_t>(1, prevFrame ? prevFrame->mInstances.size() * sizeof(uint32_t) : 0), vk::BufferUsageFlagBits::eStorageBuffer);
	ranges::fill(mInstanceIndexMap, -1);

	mMaterialCount = 0;
	mEmissivePrimitiveCount = 0;

	unordered_map<const void*, uint32_t> materialMap;

	if (prevFrame) {
		instanceDatas.reserve(prevFrame->mInstances.size());
		instanceTransforms.reserve(prevFrame->mInstances.size());
		instanceInverseTransforms.reserve(prevFrame->mInstances.size());
		instanceMotionTransforms.reserve(prevFrame->mInstances.size());
		mMaterialResources.mMaterialData.reserve(prevFrame->mMaterialData ? prevFrame->mMaterialData.size() / sizeof(uint32_t) : 1);
	}

	vector<vk::AccelerationStructureInstanceKHR> instancesAS;
	vector<vk::BufferMemoryBarrier> blasBarriers;


	// 'material' is either a Material or Medium
	auto appendMaterialData = [&](const auto* material) {
		// append unique materials to materials list
		auto materialMap_it = materialMap.find(material);
		if (materialMap_it == materialMap.end()) {
			materialMap_it = materialMap.emplace(material, (uint32_t)mMaterialResources.mMaterialData.sizeBytes()).first;
			material->store(mMaterialResources);
			mMaterialCount++;
		}
		return materialMap_it->second;
	};

	auto appendInstanceData = [&](Node& node, const void* primPtr, const InstanceData& instance, const TransformData& transform, const float emissivePower) {
		const uint32_t instanceIndex = (uint32_t)instanceDatas.size();
		instanceDatas.emplace_back(instance);
		mInstanceNodes.emplace_back(node.getPtr());

		// add light if emissive

		uint32_t lightIndex = INVALID_INSTANCE;
		if (emissivePower > 0) {
			lightIndex = (uint32_t)lightInstanceMap.size();
			BF_SET(instanceDatas[instanceIndex].packed[1], lightIndex, 0, 12);
			lightInstanceMap.emplace_back(instanceIndex);
		}

		// transforms

		TransformData prevTransform;
		if (prevFrame) {
			if (auto it = prevFrame->mInstanceTransformMap.find(primPtr); it != prevFrame->mInstanceTransformMap.end()) {
				prevTransform = it->second.first;
				mInstanceIndexMap[it->second.second] = instanceIndex;
			}
		}
		mInstanceTransformMap.emplace(primPtr, make_pair(transform, instanceIndex));

		const TransformData invTransform = transform.inverse();
		instanceTransforms.emplace_back(transform);
		instanceInverseTransforms.emplace_back(invTransform);
		instanceMotionTransforms.emplace_back(InstanceData::makeMotionTransform(invTransform, prevTransform));
		return instanceIndex;
	};



	{ // mesh instances
		ProfilerScope s("Process mesh instances", &commandBuffer);
		scene.mNode.forEachDescendant<MeshPrimitive>([&](Node& primNode, const shared_ptr<MeshPrimitive>& prim) {
			if (!prim->mMesh || !prim->mMaterial) return;

			if (prim->mMesh->topology() != vk::PrimitiveTopology::eTriangleList ||
				(prim->mMesh->indexType() != vk::IndexType::eUint32 && prim->mMesh->indexType() != vk::IndexType::eUint16) ||
				!prim->mMesh->vertices().find(Mesh::VertexAttributeType::ePosition) ||
				!prim->mMesh->vertices().find(Mesh::VertexAttributeType::eNormal) ||
				!prim->mMesh->vertices().find(Mesh::VertexAttributeType::eTexcoord)) {
				cout << "Skipping unsupported mesh in node " << primNode.name() << endl;
				return;
			}

			const uint32_t materialAddress = appendMaterialData(prim->mMaterial.get());

			// write vertexCopyDescriptors
			auto [positions, positionsDesc] = prim->mMesh->vertices().at(Mesh::VertexAttributeType::ePosition)[0];
			auto [normals, normalsDesc] = prim->mMesh->vertices().at(Mesh::VertexAttributeType::eNormal)[0];
			auto [texcoords, texcoordsDesc] = prim->mMesh->vertices().at(Mesh::VertexAttributeType::eTexcoord)[0];
			const uint32_t index = (uint32_t)vertexCopyInfos.size();
			vertexCopyDescriptors[{ "gPositions", index }] = Buffer::View(positions, positionsDesc.mOffset);
			vertexCopyDescriptors[{ "gNormals", index }] = Buffer::View(normals, normalsDesc.mOffset);
			vertexCopyDescriptors[{ "gTexcoords", index }] = Buffer::View(texcoords, texcoordsDesc.mOffset);
			const uint32_t vertexCount = (uint32_t)(positions.sizeBytes() / positionsDesc.mStride);
			vertexCopyInfos.emplace_back(uint4(
				vertexCount,
				positionsDesc.mStride,
				normalsDesc.mStride,
				texcoordsDesc.mStride));

			// track total vertices/indices
			const uint32_t firstVertex = totalVertexCount;
			const uint32_t indexByteOffset = totalIndexBufferSize;
			totalVertexCount += vertexCount;
			totalIndexBufferSize += prim->mMesh->indices().sizeBytes();

			// get/build BLAS
			const size_t key = hashArgs(prim->mMesh.get(), prim->mMaterial->alphaTest());
			auto it = scene.mMeshAccelerationStructures.find(key);
			if (it == scene.mMeshAccelerationStructures.end()) {
				vk::AccelerationStructureGeometryTrianglesDataKHR triangles;
				triangles.vertexFormat = positionsDesc.mFormat;
				triangles.vertexData = positions.deviceAddress();
				triangles.vertexStride = positionsDesc.mStride;
				triangles.maxVertex = vertexCount;
				triangles.indexType = prim->mMesh->indexType();
				triangles.indexData = prim->mMesh->indices().deviceAddress();
				vk::AccelerationStructureGeometryKHR triangleGeometry(vk::GeometryTypeKHR::eTriangles, triangles, prim->mMaterial->alphaTest() ? vk::GeometryFlagBitsKHR{} : vk::GeometryFlagBitsKHR::eOpaque);
				vk::AccelerationStructureBuildRangeInfoKHR range(prim->mMesh->indices().size() / (prim->mMesh->indices().stride() * 3));

				auto [as, asbuf] = buildAccelerationStructure(commandBuffer, primNode.name() + "/BLAS", vk::AccelerationStructureTypeKHR::eBottomLevel, triangleGeometry, range);
				blasBarriers.emplace_back(
					vk::AccessFlagBits::eAccelerationStructureWriteKHR, vk::AccessFlagBits::eAccelerationStructureReadKHR,
					VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
					**asbuf.buffer(), asbuf.offset(), asbuf.sizeBytes());
				it = scene.mMeshAccelerationStructures.emplace(key, make_pair(as, asbuf)).first;
			}

			const uint32_t triCount = prim->mMesh->indices().sizeBytes() / (prim->mMesh->indices().stride() * 3);
			const TransformData transform = nodeToWorld(primNode);
			const float area = 1;

			if (prim->mMaterial->emission() > 0)
				mEmissivePrimitiveCount += triCount;

			vk::AccelerationStructureInstanceKHR& instance = instancesAS.emplace_back();
			float3x4::Map(&instance.transform.matrix[0][0]) = transform.to_float3x4();
			instance.instanceCustomIndex = appendInstanceData(primNode, prim.get(), TrianglesInstanceData(materialAddress, triCount, firstVertex, indexByteOffset, prim->mMesh->indices().stride()), transform, prim->mMaterial->emission() * area);
			instance.mask = BVH_FLAG_TRIANGLES;
			instance.accelerationStructureReference = commandBuffer.mDevice->getAccelerationStructureAddressKHR(**it->second.first);

			meshInstanceIndices.emplace_back(prim.get(), (uint32_t)instance.instanceCustomIndex);
		});
	}

	{ // sphere instances
		ProfilerScope s("Process sphere instances", &commandBuffer);
		scene.mNode.forEachDescendant<SpherePrimitive>([&](Node& primNode, const shared_ptr<SpherePrimitive>& prim) {
			if (!prim->mMaterial) return;

			const uint32_t materialAddress = appendMaterialData(prim->mMaterial.get());

			TransformData transform = nodeToWorld(primNode);
			const float radius = prim->mRadius * transform.m.block<3, 3>(0, 0).matrix().determinant();
			// remove scale/rotation from transform
			transform = TransformData(transform.m.col(3).head<3>(), quatf::identity(), float3::Ones());

			// get/build BLAS
			const float3 mn = -float3::Constant(radius);
			const float3 mx = float3::Constant(radius);
			const size_t key = hashArgs(mn[0], mn[1], mn[2], mx[0], mx[1], mx[2], prim->mMaterial->alphaTest());
			auto aabb_it = scene.mAABBs.find(key);
			if (aabb_it == scene.mAABBs.end()) {
				Buffer::View<vk::AabbPositionsKHR> aabb = make_shared<Buffer>(commandBuffer.mDevice, "aabb data", sizeof(vk::AabbPositionsKHR), vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
				aabb[0].minX = mn[0];
				aabb[0].minY = mn[1];
				aabb[0].minZ = mn[2];
				aabb[0].maxX = mx[0];
				aabb[0].maxY = mx[1];
				aabb[0].maxZ = mx[2];
				vk::AccelerationStructureGeometryAabbsDataKHR aabbs(aabb.deviceAddress(), sizeof(vk::AabbPositionsKHR));
				vk::AccelerationStructureGeometryKHR aabbGeometry(vk::GeometryTypeKHR::eAabbs, aabbs, prim->mMaterial->alphaTest() ? vk::GeometryFlagBitsKHR{} : vk::GeometryFlagBitsKHR::eOpaque);
				vk::AccelerationStructureBuildRangeInfoKHR range(1);

				commandBuffer.trackResource(aabb.buffer());
				auto [as, asbuf] = buildAccelerationStructure(commandBuffer, "aabb BLAS", vk::AccelerationStructureTypeKHR::eBottomLevel, aabbGeometry, range);
				blasBarriers.emplace_back(
					vk::AccessFlagBits::eAccelerationStructureWriteKHR, vk::AccessFlagBits::eAccelerationStructureReadKHR,
					VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
					**asbuf.buffer(), asbuf.offset(), asbuf.sizeBytes());
				aabb_it = scene.mAABBs.emplace(key, make_pair(as, asbuf)).first;
			}

			if (prim->mMaterial->emission() > 0)
				mEmissivePrimitiveCount++;

			vk::AccelerationStructureInstanceKHR& instance = instancesAS.emplace_back();
			float3x4::Map(&instance.transform.matrix[0][0]) = transform.to_float3x4();
			instance.instanceCustomIndex = appendInstanceData(primNode, prim.get(), SphereInstanceData(materialAddress, radius), transform, prim->mMaterial->emission() * (4 * M_PI * radius * radius));
			instance.mask = BVH_FLAG_SPHERES;
			instance.accelerationStructureReference = commandBuffer.mDevice->getAccelerationStructureAddressKHR(**aabb_it->second.first);
		});
	}

	{ // medium instances
		ProfilerScope s("Process media", &commandBuffer);
		scene.mNode.forEachDescendant<Medium>([&](Node& primNode, const shared_ptr<Medium>& vol) {
			if (!vol) return;

			const uint32_t materialAddress = appendMaterialData(vol.get());

			auto densityGrid = vol->mDensityGrid->grid<float>();

			// get/build BLAS
			const nanovdb::Vec3R& mn = densityGrid->worldBBox().min();
			const nanovdb::Vec3R& mx = densityGrid->worldBBox().max();
			const size_t key = hashArgs((float)mn[0], (float)mn[1], (float)mn[2], (float)mx[0], (float)mx[1], (float)mx[2]);
			auto aabb_it = scene.mAABBs.find(key);
			if (aabb_it == scene.mAABBs.end()) {
				Buffer::View<vk::AabbPositionsKHR> aabb = make_shared<Buffer>(commandBuffer.mDevice, "aabb data", sizeof(vk::AabbPositionsKHR), vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
				aabb[0].minX = (float)mn[0];
				aabb[0].minY = (float)mn[1];
				aabb[0].minZ = (float)mn[2];
				aabb[0].maxX = (float)mx[0];
				aabb[0].maxY = (float)mx[1];
				aabb[0].maxZ = (float)mx[2];
				vk::AccelerationStructureGeometryAabbsDataKHR aabbs(aabb.deviceAddress(), sizeof(vk::AabbPositionsKHR));
				vk::AccelerationStructureGeometryKHR aabbGeometry(vk::GeometryTypeKHR::eAabbs, aabbs, vk::GeometryFlagBitsKHR::eOpaque);
				vk::AccelerationStructureBuildRangeInfoKHR range(1);
				auto [as, asbuf] = buildAccelerationStructure(commandBuffer, "aabb BLAS", vk::AccelerationStructureTypeKHR::eBottomLevel, aabbGeometry, range);

				commandBuffer.trackResource(aabb.buffer());
				blasBarriers.emplace_back(
					vk::AccessFlagBits::eAccelerationStructureWriteKHR, vk::AccessFlagBits::eAccelerationStructureReadKHR,
					VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
					**asbuf.buffer(), asbuf.offset(), asbuf.sizeBytes());
				aabb_it = scene.mAABBs.emplace(key, make_pair(as, asbuf)).first;
			}

			// append to instance list
			const TransformData transform = nodeToWorld(primNode);
			vk::AccelerationStructureInstanceKHR& instance = instancesAS.emplace_back();
			float3x4::Map(&instance.transform.matrix[0][0]) = transform.to_float3x4();
			instance.instanceCustomIndex = appendInstanceData(primNode, vol.get(), VolumeInstanceData(materialAddress, mMaterialResources.mVolumeDataMap.at(vol->mDensityBuffer)), transform, 0);
			instance.mask = BVH_FLAG_VOLUME;
			instance.accelerationStructureReference = commandBuffer.mDevice->getAccelerationStructureAddressKHR(**aabb_it->second.first);
		});
	}

	{ // environment material
		ProfilerScope s("Process environment", &commandBuffer);
		mEnvironmentMaterialAddress = -1;
		scene.mNode.forEachDescendant<Environment>([&](Node& node, const shared_ptr<Environment> environment) {
			if (environment->mEmission.mValue.isZero()) return true;
			mEnvironmentMaterialAddress = mMaterialResources.mMaterialData.sizeBytes();
			mMaterialCount++;
			environment->store(mMaterialResources);
			return false;
		});
	}

	{ // Build TLAS
		ProfilerScope s("Build TLAS", &commandBuffer);
		commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, vk::DependencyFlagBits::eByRegion, {}, blasBarriers, {});

		vk::AccelerationStructureGeometryKHR geom{ vk::GeometryTypeKHR::eInstances, vk::AccelerationStructureGeometryInstancesDataKHR() };
		vk::AccelerationStructureBuildRangeInfoKHR range{ (uint32_t)instancesAS.size() };
		if (!instancesAS.empty()) {
			shared_ptr<Buffer> buf = make_shared<Buffer>(commandBuffer.mDevice, "TLAS instance buffer",
				sizeof(vk::AccelerationStructureInstanceKHR) * instancesAS.size() + 16, // extra 16 bytes for alignment
				vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress,
				vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

			const size_t address = (size_t)buf->deviceAddress();
			const size_t offset = (-address & 15); // aligned = unaligned + (-unaligned & (alignment - 1))

			ranges::copy(instancesAS, (vk::AccelerationStructureInstanceKHR*)((byte*)buf->data() + offset));

			geom.geometry.instances.data = buf->deviceAddress() + offset;
			commandBuffer.trackResource(buf);
		}
		tie(mAccelerationStructure, mAccelerationStructureBuffer) = buildAccelerationStructure(commandBuffer, scene.mNode.name() + "/TLAS", vk::AccelerationStructureTypeKHR::eTopLevel, geom, range);
		mAccelerationStructureBuffer.barrier(commandBuffer,
			vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, vk::PipelineStageFlagBits::eComputeShader,
			vk::AccessFlagBits::eAccelerationStructureWriteKHR, vk::AccessFlagBits::eAccelerationStructureReadKHR);
	}

	// pack mesh data
	{
		ProfilerScope s("Pack mesh vertices/indices", &commandBuffer);

		if (!mVertices || mVertices.size() < totalVertexCount)
			mVertices = make_shared<Buffer>(commandBuffer.mDevice, "gVertices", max(totalVertexCount, 1u) * sizeof(PackedVertexData), vk::BufferUsageFlagBits::eStorageBuffer);
		if (!mIndices || mIndices.size() < totalIndexBufferSize)
			mIndices = make_shared<Buffer>(commandBuffer.mDevice, "gIndices", max(totalIndexBufferSize, 4u), vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer);

		if (!vertexCopyInfos.empty()) {
			Buffer::View<uint4> infos = make_shared<Buffer>(commandBuffer.mDevice, "gInfos", vertexCopyInfos.size() * sizeof(uint4), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
			ranges::copy(vertexCopyInfos, infos.begin());
			vertexCopyDescriptors[{ "gInfos", 0 }] = infos;
			vertexCopyDescriptors[{ "gVertices", 0 }] = mVertices;

			// pack vertices

			auto copyPipeline = scene.mCopyVerticesPipeline.get(commandBuffer.mDevice);
			commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, ***copyPipeline);
			copyPipeline->getDescriptorSets(vertexCopyDescriptors)->bind(commandBuffer, {});
			for (uint32_t i = 0; i < vertexCopyInfos.size(); i++) {
				copyPipeline->pushConstants(commandBuffer, PushConstants{ { "gBufferIndex", PushConstantValue(i) } });
				const vk::Extent3D dim = copyPipeline->calculateDispatchDim(vk::Extent3D(vertexCopyInfos[i][0], 1, 1));
				commandBuffer->dispatch(dim.width, dim.height, dim.depth);
			}
		}

		// copy indices

		for (const auto& [prim, instanceIndex] : meshInstanceIndices) {
			const TrianglesInstanceData& instance = *(TrianglesInstanceData*)&instanceDatas[instanceIndex];
			commandBuffer->copyBuffer(
				**prim->mMesh->indices().buffer(),
				**mIndices.buffer(),
				vk::BufferCopy(prim->mMesh->indices().offset(), instance.indicesByteOffset(), prim->mMesh->indices().sizeBytes()));
		}
		mIndices.barrier(commandBuffer, vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead);
	}

	{ // upload instance/material data
		ProfilerScope s("Upload scene data buffers");
		if (!mInstances || mInstances.size() < instanceDatas.size()) {
			mInstances = make_shared<Buffer>(commandBuffer.mDevice, "mInstances", max(1ull, instanceDatas.size() * sizeof(InstanceData)), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst);
			mInstanceTransforms = make_shared<Buffer>(commandBuffer.mDevice, "mInstanceTransforms", max(1ull, instanceTransforms.size() * sizeof(TransformData)), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst);
			mInstanceInverseTransforms = make_shared<Buffer>(commandBuffer.mDevice, "mInstanceInverseTransforms", max(1ull, instanceInverseTransforms.size() * sizeof(TransformData)), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst);
			mInstanceMotionTransforms = make_shared<Buffer>(commandBuffer.mDevice, "mInstanceMotionTransforms", max(1ull, instanceMotionTransforms.size() * sizeof(TransformData)), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst);
		}
		if (!mMaterialData || mMaterialData.size() < mMaterialResources.mMaterialData.sizeBytes())
			mMaterialData = make_shared<Buffer>(commandBuffer.mDevice, "mMaterialData", max(1ull, mMaterialResources.mMaterialData.sizeBytes()), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst);
		if (!mLightInstanceMap || mLightInstanceMap.size() < lightInstanceMap.size())
			mLightInstanceMap = make_shared<Buffer>(commandBuffer.mDevice, "mLightInstanceMap", max(1ull, lightInstanceMap.size() * sizeof(uint32_t)), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst);

		if (!instanceDatas.empty()) {
			ranges::copy(instanceDatas, mInstances.begin());
			ranges::copy(instanceTransforms, mInstanceTransforms.begin());
			ranges::copy(instanceInverseTransforms, mInstanceInverseTransforms.begin());
			ranges::copy(instanceMotionTransforms, mInstanceMotionTransforms.begin());
		}
		if (!mMaterialResources.mMaterialData.empty())
			ranges::copy(mMaterialResources.mMaterialData, mMaterialData.begin());
		if (!lightInstanceMap.empty())
			ranges::copy(lightInstanceMap, mLightInstanceMap.begin());
	}
}

Descriptors Scene::FrameResources::getDescriptors() const {
	Descriptors descriptors;
	descriptors[{ "mAccelerationStructure", 0u }]     = mAccelerationStructure;
	descriptors[{ "mMaterialData", 0u }]              = mMaterialData;
	descriptors[{ "mVertices", 0u }]                  = mVertices;
	descriptors[{ "mIndices", 0u }]                   = mIndices;
	descriptors[{ "mInstances", 0u }]                 = mInstances;
	descriptors[{ "mInstanceTransforms", 0u }]        = mInstanceTransforms;
	descriptors[{ "mInstanceInverseTransforms", 0u }] = mInstanceInverseTransforms;
	descriptors[{ "mInstanceMotionTransforms", 0u }]  = mInstanceMotionTransforms;
	descriptors[{ "mLightInstances", 0u }]            = mLightInstanceMap;
	for (const auto& [vol, index] : mMaterialResources.mVolumeDataMap)
		descriptors[{"mVolumes", index}] = vol;
	for (const auto& [image, index] : mMaterialResources.mImage4s)
		descriptors[{"mImages", index}] = ImageDescriptor{ image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
	for (const auto& [image, index] : mMaterialResources.mImage1s)
		descriptors[{"mImage1s", index}] = ImageDescriptor{ image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
	return descriptors;
}


template<unsigned int N>
void ImageValue<N>::drawGui(const string& label) {
	ImGui::DragScalarN(label.c_str(), ImGuiDataType_Float, mValue.data(), N, .01f);

	if (mImage) {
		ImGui::PushID(this);
		const bool d = ImGui::Button("x");
		ImGui::PopID();
		if (d)
			mImage = {};
		else {
			const uint32_t w = ImGui::GetWindowSize().x;
			ImGui::Image(Gui::getTextureID(mImage), ImVec2(w, w * (float)mImage.extent().height / (float)mImage.extent().width));
		}
	}
}

void Scene::drawGui() {
	if (mResources) {
		ImGui::Text("%llu instances", mResources->mInstances.size());
		ImGui::Text("%llu light instances", mResources->mLightInstanceMap.size());
		ImGui::Text("%u emissive primitives", mResources->mEmissivePrimitiveCount);
		ImGui::Text("%u materials", mResources->mMaterialCount);
	}
	if (ImGui::Button("Load file")) {
		auto f = pfd::open_file("Open scene", "", loaderFilters());
		for (const string& filepath : f.result())
			mToLoad.emplace_back(filepath);
	}
	ImGui::Checkbox("Always update", &mAlwaysUpdate);
}

void TransformData::drawGui(Node& node) {
	TransformData prev = *this;

	float3 translate = m.topRightCorner(3, 1);
	float3 scale;
	scale.x() = m.block(0, 0, 3, 1).matrix().norm();
	scale.y() = m.block(0, 1, 3, 1).matrix().norm();
	scale.z() = m.block(0, 2, 3, 1).matrix().norm();
	const Eigen::Matrix3f r = m.block<3, 3>(0, 0).matrix() * Eigen::DiagonalMatrix<float, 3, 3>(1 / scale.x(), 1 / scale.y(), 1 / scale.z());
	Eigen::Quaternionf rotation(r);

	bool changed = false;

	if (ImGui::DragFloat3("Translation", translate.data(), .1f) && !translate.hasNaN()) {
		m.topRightCorner(3, 1) = translate;
		changed = true;
	}

	bool v = ImGui::DragFloat3("Rotation (XYZ)", rotation.vec().data(), .1f);
	v |= ImGui::DragFloat("Rotation (W)", &rotation.w(), .1f);
	if (v && !rotation.vec().hasNaN()) {
		m.block<3, 3>(0, 0) = rotation.normalized().matrix();
		changed = true;
	}

	if (ImGui::DragFloat3("Scale", scale.data(), .1f) && !rotation.vec().hasNaN() && !scale.hasNaN() && (scale != float3::Zero()).all()) {
		m.block<3, 3>(0, 0) = rotation.normalized().matrix() * Eigen::DiagonalMatrix<float, 3, 3>(scale.x(), scale.y(), scale.z());
		changed = true;
	}


	if (const auto scene = node.findAncestor<Scene>()) {
		if (changed)
			scene->markDirty();

		// animate hack
		if (scene->mAnimatedTransform.get() == &node) {
			if (ImGui::Button("Stop Animating")) scene->mAnimatedTransform = nullptr;
			ImGui::DragFloat3("Move", scene->mAnimateTranslate.data(), .01f);
			ImGui::DragFloat3("Rotate", scene->mAnimateRotate.data(), .01f);
			const bool was_zero = scene->mAnimateWiggleOffset.isZero();
			ImGui::DragFloat3("Wiggle Offset", scene->mAnimateWiggleOffset.data(), .01f);
			if (!scene->mAnimateWiggleOffset.isZero()) {
				if (ImGui::Button("Set Wiggle Anchor") || was_zero) {
					scene->mAnimateWiggleBase = translate;
					scene->mAnimateWiggleTime = 0;
				}
				ImGui::DragFloat("Wiggle Speed", &scene->mAnimateWiggleSpeed);
			}
		} else if (ImGui::Button("Animate"))
			scene->mAnimatedTransform = node.getPtr();
	}
}

void MeshPrimitive::drawGui(Node& node) {
	if (mMesh) {
		ImGui::Text("%s", type_index(typeid(Mesh)).name());
		ImGui::SameLine();
		if (ImGui::Button("Mesh"))
			node.root()->findDescendant<Inspector>()->pin(node, mMesh);
	}
	if (mMaterial) {
		ImGui::Text("%s", type_index(typeid(Material)).name());
		ImGui::SameLine();
		if (ImGui::Button("Material"))
			node.root()->findDescendant<Inspector>()->pin(node, mMaterial);
	}
}
void SpherePrimitive::drawGui(Node& node) {
	if (ImGui::DragFloat("Radius", &mRadius, .01f))
		if (auto scene = node.findAncestor<Scene>())
			scene->markDirty();
	if (mMaterial) {
		ImGui::Text("%s", type_index(typeid(Material)).name());
		ImGui::SameLine();
		if (ImGui::Button("Material"))
			node.root()->findDescendant<Inspector>()->pin(node, mMaterial);
	}
}

void Camera::drawGui() {
	bool ortho = mProjection.isOrthographic();
	if (ImGui::Checkbox("Orthographic", reinterpret_cast<bool*>(&ortho)))
		mProjection.mVerticalFoV = -mProjection.mVerticalFoV;
	ImGui::DragFloat("Near Plane", &mProjection.mNearPlane, 0.01f, -1, 1);
	if (mProjection.isOrthographic()) {
		ImGui::DragFloat("Far Plane", &mProjection.mFarPlane, 0.01f, -1, 1);
		ImGui::DragFloat2("Projection Scale", mProjection.mScale.data(), 0.01f, -1, 1);
	} else {
		float fovy = degrees(2 * atan(1 / mProjection.mScale[1]));
		if (ImGui::DragFloat("Vertical FoV", &fovy, 0.01f, 1, 179)) {
			const float aspect = mProjection.mScale[0] / mProjection.mScale[1];
			mProjection.mScale[1] = 1 / tan(radians(fovy / 2));
			mProjection.mScale[0] = mProjection.mScale[1] * aspect;
		}
	}
	ImGui::DragFloat2("Projection Offset", mProjection.mOffset.data(), 0.01f, -1, 1);
	ImGui::InputInt2("ImageRect Offset", &mImageRect.offset.x);
	ImGui::InputInt2("ImageRect Extent", reinterpret_cast<int32_t*>(&mImageRect.extent.width));
}

void Material::drawGui(Node& node) {
	bool changed = false;

	if (ImGui::ColorEdit3("Base Color", baseColor().data())) changed = true;
	ImGui::PushItemWidth(80);
	if (ImGui::DragFloat("Emission", &emission())) changed = true;
	if (ImGui::DragFloat("Metallic", &metallic(), 0.1, 0, 1)) changed = true;
	if (ImGui::DragFloat("Roughness", &roughness(), 0.1, 0, 1)) changed = true;
	if (ImGui::DragFloat("Anisotropic", &anisotropic(), 0.1, 0, 1)) changed = true;
	if (ImGui::DragFloat("Subsurface", &subsurface(), 0.1, 0, 1)) changed = true;
	if (ImGui::DragFloat("Clearcoat", &clearcoat(), 0.1, 0, 1)) changed = true;
	if (ImGui::DragFloat("Clearcoat Gloss", &clearcoatGloss(), 0.1, 0, 1)) changed = true;
	if (ImGui::DragFloat("Transmission", &transmission(), 0.1, 0, 1)) changed = true;
	if (ImGui::DragFloat("Index of Refraction", &eta(), 0.1, 0, 2)) changed = true;
	if (mBumpImage) if (ImGui::DragFloat("Bump Strength", &mBumpStrength, 0.1, 0, 10)) changed = true;
	ImGui::PopItemWidth();

	const float w = ImGui::CalcItemWidth() - 4;
	for (uint i = 0; i < DisneyMaterialData::gDataSize; i++)
		if (mValues[i].mImage) {
			ImGui::Text(mValues[i].mImage.image()->resourceName().c_str());
			ImGui::Image(Gui::getTextureID(mValues[i].mImage), ImVec2(w, w * mValues[i].mImage.extent().height / (float)mValues[i].mImage.extent().width));
		}
	if (mAlphaMask) {
		ImGui::Text(mAlphaMask.image()->resourceName().c_str());
		ImGui::Image(Gui::getTextureID(mAlphaMask), ImVec2(w, w * mAlphaMask.extent().height / (float)mAlphaMask.extent().width));
	}
	if (mBumpImage) {
		ImGui::Text(mBumpImage.image()->resourceName().c_str());
		ImGui::Image(Gui::getTextureID(mBumpImage), ImVec2(w, w * mBumpImage.extent().height / (float)mBumpImage.extent().width));
	}

	if (changed)
		if (const auto scene = node.findAncestor<Scene>(); scene)
			scene->markDirty();
}
void Medium::drawGui(Node& node) {
	bool changed = false;

	if (ImGui::ColorEdit3("Density", mDensityScale.data(), ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float)) changed = true;
	if (ImGui::ColorEdit3("Albedo", mAlbedoScale.data(), ImGuiColorEditFlags_Float)) changed = true;
	if (ImGui::SliderFloat("Anisotropy", &mAnisotropy, -.999f, .999f)) changed = true;
	if (ImGui::SliderFloat("Attenuation Unit", &mAttenuationUnit, 0, 1)) changed = true;

	if (changed) {
		const auto scene = node.findAncestor<Scene>();
		if (scene)
			scene->markDirty();
	}
}
}