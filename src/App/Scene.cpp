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

#include <ImGuizmo.h>

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

	Buffer::View<byte> buffer = make_shared<Buffer>(
		commandBuffer.mDevice,
		name,
		buildSizes.accelerationStructureSize,
		vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress);

	Buffer::View<byte> scratchData = make_shared<Buffer>(
		commandBuffer.mDevice,
		name + "/scratchData",
		buildSizes.buildScratchSize,
		vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eStorageBuffer);

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
		inspector->setInspectCallback<Scene>();
		inspector->setInspectCallback<TransformData>();
		inspector->setInspectCallback<Camera>();
		inspector->setInspectCallback<MeshPrimitive>();
		inspector->setInspectCallback<SpherePrimitive>();
		inspector->setInspectCallback<Material>();
		inspector->setInspectCallback<Medium>();
		inspector->setInspectCallback<EnvironmentMap>();
		inspector->setInspectCallback<nanovdb::GridMetaData>([](Node& n) {
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
	md.mBindingFlags["gNormals"] = vk::DescriptorBindingFlagBits::ePartiallyBound;
	md.mBindingFlags["gTexcoords"] = vk::DescriptorBindingFlagBits::ePartiallyBound;

	const filesystem::path shaderPath = *mNode.findAncestor<Instance>()->findArgument("shaderKernelPath");
	mConvertAlphaToRoughnessPipeline = ComputePipelineCache(shaderPath / "convert_roughness.slang", "alpha_to_roughness");
	mConvertShininessToRoughnessPipeline = ComputePipelineCache(shaderPath / "convert_roughness.slang", "shininess_to_roughness");
	mConvertPbrPipeline = ComputePipelineCache(shaderPath / "convert_material.slang", "from_gltf_pbr");
	mConvertDiffuseSpecularPipeline = ComputePipelineCache(shaderPath / "convert_material.slang", "from_diffuse_specular");

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
	commandBuffer.trackResource(pixels);

	const shared_ptr<Node> node = Node::create(filepath.stem().string());
	node->makeComponent<EnvironmentMap>(ImageValue<3>{ float3::Ones(), img });
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
	m.mMaterialData.setBaseColor(baseColor.mValue);
	m.mMaterialData.setEmission(emission.mValue);
	m.mMaterialData.setMetallic(metallic_roughness.mValue.z());
	m.mMaterialData.setRoughness(metallic_roughness.mValue.y());
	m.mMaterialData.setAnisotropic(0);
	m.mMaterialData.setSubsurface(0);
	m.mMaterialData.setClearcoat(0);
	m.mMaterialData.setClearcoatGloss(0);
	m.mMaterialData.setTransmission(luminance(transmission.mValue));
	m.mMaterialData.setEta(eta - 1);
	if (baseColor.mImage || metallic_roughness.mImage || transmission.mImage || emission.mImage) {
		Descriptors descriptors;

		Image::View d = baseColor.mImage ? baseColor.mImage :
		                metallic_roughness.mImage ? metallic_roughness.mImage :
		                emission.mImage ? emission.mImage :
		                transmission.mImage;
		Image::Metadata md;
		md.mExtent = d.extent();
		md.mLevels = Image::maxMipLevels(md.mExtent);
		md.mFormat = vk::Format::eR8G8B8A8Unorm;
		md.mUsage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage;
		for (int i = 0; i < m.mImages.size(); i++) {
			m.mImages[i] = make_shared<Image>(commandBuffer.mDevice, "PackedMaterialData[" + to_string(i) + "]", md);
			descriptors[{ "gOutput", i }] = ImageDescriptor{ m.mImages[i], vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };
		}
		if (baseColor.mImage) {
			md.mLevels = 1;
			md.mFormat = vk::Format::eR8Unorm;
			md.mUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage;
			m.mAlphaMask = make_shared<Image>(commandBuffer.mDevice, "mAlphaMask", md);
		}
		descriptors[{ "gOutputAlphaMask", 0 }] = ImageDescriptor{ m.mAlphaMask ? m.mAlphaMask : d, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };

		m.mMinAlpha = make_shared<Buffer>(commandBuffer.mDevice, "mMinAlpha", sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
		m.mMinAlpha[0] = 255;
		descriptors[{ "gOutputMinAlpha", 0 }] = m.mMinAlpha;
		m.mMinAlpha.barrier(commandBuffer,
			vk::PipelineStageFlagBits::eHost, vk::PipelineStageFlagBits::eComputeShader,
			vk::AccessFlagBits::eHostWrite, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);

		descriptors[{ "gDiffuse", 0 }]       = ImageDescriptor{ baseColor.mImage          ? baseColor.mImage          : d, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{ "gSpecular", 0 }]      = ImageDescriptor{ metallic_roughness.mImage ? metallic_roughness.mImage : d, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{ "gTransmittance", 0 }] = ImageDescriptor{ transmission.mImage       ? transmission.mImage       : d, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{ "gEmission", 0 }]      = ImageDescriptor{ emission.mImage           ? emission.mImage           : d, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{ "gRoughness", 0 }]     = ImageDescriptor{                                                         d, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		Defines defs;
		if (baseColor.mImage) defs["gUseDiffuse"] = "true";
		if (metallic_roughness.mImage) defs["gUseSpecular"] = "true";
		if (transmission.mImage) defs["gUseTransmittance"] = "true";
		if (emission.mImage)     defs["gUseEmission"] = "true";

		mConvertPbrPipeline.get(commandBuffer.mDevice, defs)->dispatchTiled(commandBuffer, d.extent(), descriptors);

		for (const Image::View& img : m.mImages)
			img.image()->generateMipMaps(commandBuffer);
	}
	return m;
}
Material Scene::makeDiffuseSpecularMaterial(CommandBuffer& commandBuffer, const ImageValue<3>& diffuse, const ImageValue<3>& specular, const ImageValue<1>& roughness, const ImageValue<3>& transmission, const float eta, const ImageValue<3>& emission) {
	Material m;
	const float ld = luminance(diffuse.mValue);
	const float ls = luminance(specular.mValue);
	const float lt = luminance(transmission.mValue);
	m.mMaterialData.setBaseColor((diffuse.mValue * ld + specular.mValue * ls + transmission.mValue * lt) / (ld + ls + lt));
	m.mMaterialData.setEmission(emission.mValue);
	m.mMaterialData.setMetallic(ls / (ld + ls + lt));
	m.mMaterialData.setRoughness(roughness.mValue[0]);
	m.mMaterialData.setAnisotropic(0);
	m.mMaterialData.setSubsurface(0);
	m.mMaterialData.setClearcoat(0);
	m.mMaterialData.setClearcoatGloss(1);
	m.mMaterialData.setTransmission(lt / (ld + ls + lt));
	m.mMaterialData.setEta(eta - 1);
	if (diffuse.mImage || specular.mImage || transmission.mImage || roughness.mImage) {
		Descriptors descriptors;

		Image::View d = diffuse.mImage ? diffuse.mImage :
		                specular.mImage ? specular.mImage :
						transmission.mImage ? transmission.mImage :
						emission.mImage ? emission.mImage :
						roughness.mImage;
		Image::Metadata md;
		md.mExtent = d.extent();
		md.mLevels = Image::maxMipLevels(md.mExtent);
		md.mFormat = vk::Format::eR8G8B8A8Unorm;
		md.mUsage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage;
		for (int i = 0; i < m.mImages.size(); i++) {
			m.mImages[i] = make_shared<Image>(commandBuffer.mDevice, "material data", md);
			descriptors[{"gOutput", i}] = ImageDescriptor{ m.mImages[i], vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };
		}
		if (diffuse.mImage) {
			md.mLevels = 1;
			md.mFormat = vk::Format::eR8Unorm;
			md.mUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage;
			m.mAlphaMask = make_shared<Image>(commandBuffer.mDevice, "alpha mask", md);
		}
		descriptors[{"gOutputAlphaMask", 0}] = ImageDescriptor{ m.mAlphaMask ? m.mAlphaMask : d, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };

		m.mMinAlpha = make_shared<Buffer>(commandBuffer.mDevice, "min_alpha", sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
		m.mMinAlpha[0] = 255;
		descriptors[{"gOutputMinAlpha", 0}] = m.mMinAlpha;
		m.mMinAlpha.barrier(commandBuffer,
			vk::PipelineStageFlagBits::eHost, vk::PipelineStageFlagBits::eComputeShader,
			vk::AccessFlagBits::eHostWrite, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);

		descriptors[{"gDiffuse", 0}]       = ImageDescriptor{ diffuse.mImage      ? diffuse.mImage      : d, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{"gSpecular", 0}]      = ImageDescriptor{ specular.mImage     ? specular.mImage     : d, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{"gTransmittance", 0}] = ImageDescriptor{ transmission.mImage ? transmission.mImage : d, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{"gRoughness", 0}]     = ImageDescriptor{ roughness.mImage    ? roughness.mImage    : d, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{"gEmission", 0}]      = ImageDescriptor{ diffuse.mImage      ? diffuse.mImage      : d, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		Defines defs;
		if (diffuse.mImage)      defs["gUseDiffuse"] = "true";
		if (specular.mImage)     defs["gUseSpecular"] = "true";
		if (transmission.mImage) defs["gUseTransmittance"] = "true";
		if (roughness.mImage)    defs["gUseRoughness"] = "true";
		if (emission.mImage)     defs["gUseEmission"] = "true";

		mConvertDiffuseSpecularPipeline.get(commandBuffer.mDevice, defs)->dispatchTiled(commandBuffer, d.extent(), descriptors);

		for (const Image::View& img : m.mImages)
			img.image()->generateMipMaps(commandBuffer);
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

	if (auto w = mNode.root()->findDescendant<Window>()) {
		for (const string& file : w->droppedFiles())
			mToLoad.emplace_back(file);
		w->droppedFiles().clear();
	}

	bool update = mAlwaysUpdate || mUpdateOnce;
	bool loaded = false;
	for (const string& file : mToLoad) {
		const filesystem::path filepath = file;
		mNode.addChild(load(commandBuffer, filepath));
		loaded = true;
		update = true;
	}
	mToLoad.clear();

	if (!update) return;

	// Update scene data based on node graph
	// always update once after load so that motion transforms are valid

	mUpdateOnce = loaded && !mAlwaysUpdate;

	updateFrameData(commandBuffer);
}

void Scene::updateFrameData(CommandBuffer& commandBuffer) {
	mLastUpdate = chrono::high_resolution_clock::now();

	auto prevInstanceTransforms = move(mFrameData.mInstanceTransformMap);
	mFrameData.clear();
	mFrameData.mResourcePool.clean();

	// Construct resources used by renderers (mesh/material data buffers, image arrays, etc.)

	vector<pair<MeshPrimitive*, uint32_t>> meshInstanceIndices;
	vector<InstanceData> instanceDatas;
	vector<TransformData> instanceTransforms;
	vector<TransformData> instanceInverseTransforms;
	vector<TransformData> instanceMotionTransforms;
	vector<uint32_t> lightInstanceMap; // light index -> instance index
	vector<uint32_t> instanceLightMap; // instance index -> light index
	vector<uint32_t> instanceIndexMap; // current frame instance index -> previous frame instance index

	vector<MeshVertexInfo> meshVertexInfos;
	unordered_map<Buffer*, uint32_t> vertexBufferMap;

	unordered_map<const void*, uint32_t> materialMap;

	mFrameData.mAabbMin = float3::Constant(numeric_limits<float>::infinity());
	mFrameData.mAabbMax = float3::Constant(-numeric_limits<float>::infinity());

	vector<vk::AccelerationStructureInstanceKHR> instancesAS;
	vector<vk::BufferMemoryBarrier> blasBarriers;

	auto appendVertexBuffer = [&](const shared_ptr<Buffer>& buf) -> uint32_t {
		if (!buf)
			return 0xFFFF;

		if (auto it = vertexBufferMap.find(buf.get()); it != vertexBufferMap.end())
			return it->second;
		const uint32_t idx = mFrameData.mVertexBuffers.size();
		mFrameData.mVertexBuffers.emplace_back(buf);
		vertexBufferMap.emplace(buf.get(), idx);
		return idx;
	};

	// 'material' is either a Material or Medium
	auto appendMaterialData = [&](const auto* material) {
		// append unique materials to materials list
		auto materialMap_it = materialMap.find(material);
		if (materialMap_it == materialMap.end()) {
			materialMap_it = materialMap.emplace(material, (uint32_t)mFrameData.mMaterialResources.mMaterialData.sizeBytes()).first;
			material->store(mFrameData.mMaterialResources);
			mFrameData.mMaterialCount++;
		}
		return materialMap_it->second;
	};

	auto appendInstanceData = [&](Node& node, const void* primPtr, const InstanceData& instance, const TransformData& transform, const float emissivePower) {
		const uint32_t instanceIndex = (uint32_t)instanceDatas.size();
		instanceDatas.emplace_back(instance);
		mFrameData.mInstanceNodes.emplace_back(node.getPtr());
		uint32_t& lightIndex = instanceLightMap.emplace_back(INVALID_INSTANCE);
		uint32_t& prevInstanceIndex = instanceIndexMap.emplace_back(-1);

		// add light if emissive

		if (emissivePower > 0) {
			lightIndex = (uint32_t)lightInstanceMap.size();
			//instanceDatas[instanceIndex].lightIndex(lightIndex);
			lightInstanceMap.emplace_back(instanceIndex);
			mFrameData.mLightCount++;
		}

		// transforms

		TransformData prevTransform;
		if (auto it = prevInstanceTransforms.find(primPtr); it != prevInstanceTransforms.end()) {
			uint32_t idx;
			tie(prevTransform, prevInstanceIndex) = it->second;
		}
		mFrameData.mInstanceTransformMap.emplace(primPtr, make_pair(transform, instanceIndex));

		const TransformData invTransform = transform.inverse();
		instanceTransforms.emplace_back(transform);
		instanceInverseTransforms.emplace_back(invTransform);
		instanceMotionTransforms.emplace_back(makeMotionTransform(invTransform, prevTransform));
		return instanceIndex;
	};

	auto getAabbBlas = [&](const float3 mn, const float3 mx, const bool opaque) {
		const size_t key = hashArgs(mn[0], mn[1], mn[2], mx[0], mx[1], mx[2], opaque);
		auto aabb_it = mAABBs.find(key);
		if (aabb_it != mAABBs.end())
			return aabb_it->second;

		Buffer::View<vk::AabbPositionsKHR> aabb = make_shared<Buffer>(
			commandBuffer.mDevice,
			"aabb data",
			sizeof(vk::AabbPositionsKHR),
			vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
		aabb[0].minX = mn[0];
		aabb[0].minY = mn[1];
		aabb[0].minZ = mn[2];
		aabb[0].maxX = mx[0];
		aabb[0].maxY = mx[1];
		aabb[0].maxZ = mx[2];
		vk::AccelerationStructureGeometryAabbsDataKHR aabbs(aabb.deviceAddress(), sizeof(vk::AabbPositionsKHR));
		vk::AccelerationStructureGeometryKHR aabbGeometry(vk::GeometryTypeKHR::eAabbs, aabbs, opaque ? vk::GeometryFlagBitsKHR::eOpaque : vk::GeometryFlagBitsKHR{});
		vk::AccelerationStructureBuildRangeInfoKHR range(1);
		commandBuffer.trackResource(aabb.buffer());

		auto [as, asbuf] = buildAccelerationStructure(commandBuffer, "aabb BLAS", vk::AccelerationStructureTypeKHR::eBottomLevel, aabbGeometry, range);

		blasBarriers.emplace_back(
			vk::AccessFlagBits::eAccelerationStructureWriteKHR, vk::AccessFlagBits::eAccelerationStructureReadKHR,
			VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
			**asbuf.buffer(), asbuf.offset(), asbuf.sizeBytes());

		return mAABBs.emplace(key, make_pair(as, asbuf)).first->second;
	};

	{ // mesh instances
		ProfilerScope s("Process mesh instances", &commandBuffer);
		mNode.forEachDescendant<MeshPrimitive>([&](Node& primNode, const shared_ptr<MeshPrimitive>& prim) {
			if (!prim->mMesh || !prim->mMaterial) return;

			if (prim->mMesh->topology() != vk::PrimitiveTopology::eTriangleList ||
				(prim->mMesh->indexType() != vk::IndexType::eUint32 && prim->mMesh->indexType() != vk::IndexType::eUint16) ||
				!prim->mMesh->vertices().find(Mesh::VertexAttributeType::ePosition)) {
				cout << "Skipping unsupported mesh in node " << primNode.name() << endl;
				return;
			}

			auto [positions, positionsDesc] = prim->mMesh->vertices().at(Mesh::VertexAttributeType::ePosition)[0];

			const uint32_t vertexCount = (uint32_t)((positions.sizeBytes() - positionsDesc.mOffset) / positionsDesc.mStride);
			const uint32_t primitiveCount = prim->mMesh->indices().size() / (prim->mMesh->indices().stride() * 3);

			// get/build BLAS
			const size_t key = hashArgs(positions.buffer(), positions.offset(), positions.sizeBytes(), positionsDesc, prim->mMaterial->alphaTest());
			auto it = mMeshAccelerationStructures.find(key);
			if (it == mMeshAccelerationStructures.end()) {
				ProfilerScope ps("Build acceleration structure", &commandBuffer);

				vk::AccelerationStructureGeometryTrianglesDataKHR triangles;
				triangles.vertexFormat = positionsDesc.mFormat;
				triangles.vertexData = positions.deviceAddress();
				triangles.vertexStride = positionsDesc.mStride;
				triangles.maxVertex = vertexCount;
				triangles.indexType = prim->mMesh->indexType();
				triangles.indexData = prim->mMesh->indices().deviceAddress();
				vk::AccelerationStructureGeometryKHR triangleGeometry(vk::GeometryTypeKHR::eTriangles, triangles, prim->mMaterial->alphaTest() ? vk::GeometryFlagBitsKHR{} : vk::GeometryFlagBitsKHR::eOpaque);
				vk::AccelerationStructureBuildRangeInfoKHR range(primitiveCount);

				auto [as, asbuf] = buildAccelerationStructure(commandBuffer, primNode.name() + "/BLAS", vk::AccelerationStructureTypeKHR::eBottomLevel, triangleGeometry, range);

				blasBarriers.emplace_back(
					vk::AccessFlagBits::eAccelerationStructureWriteKHR, vk::AccessFlagBits::eAccelerationStructureReadKHR,
					VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
					**asbuf.buffer(), asbuf.offset(), asbuf.sizeBytes());

				it = mMeshAccelerationStructures.emplace(key, make_pair(as, asbuf)).first;
			}

			Buffer::View<byte> normals, texcoords;
			Mesh::VertexAttributeDescription normalsDesc, texcoordsDesc;
			if (auto attrib = prim->mMesh->vertices().find(Mesh::VertexAttributeType::eNormal))
				tie(normals, normalsDesc) = *attrib;
			if (auto attrib = prim->mMesh->vertices().find(Mesh::VertexAttributeType::eTexcoord))
				tie(texcoords, texcoordsDesc) = *attrib;

			const uint32_t vertexInfoIndex = (uint32_t)meshVertexInfos.size();

			meshVertexInfos.emplace_back(
				appendVertexBuffer(prim->mMesh->indices().buffer()), (uint32_t)prim->mMesh->indices().offset(), (uint32_t)prim->mMesh->indices().stride(),
				appendVertexBuffer(positions.buffer()), (uint32_t)positions.offset() + positionsDesc.mOffset, positionsDesc.mStride,
				appendVertexBuffer(normals.buffer())  , (uint32_t)normals.offset()   + normalsDesc.mOffset  , normalsDesc.mStride,
				appendVertexBuffer(texcoords.buffer()), (uint32_t)texcoords.offset() + texcoordsDesc.mOffset, texcoordsDesc.mStride);

			const uint32_t materialAddress = appendMaterialData(prim->mMaterial.get());

			const uint32_t triCount = prim->mMesh->indices().sizeBytes() / (prim->mMesh->indices().stride() * 3);
			const TransformData transform = nodeToWorld(primNode);
			const float area = 1;

			if (!prim->mMaterial->mMaterialData.getEmission().isZero())
				mFrameData.mEmissivePrimitiveCount += triCount;

			vk::AccelerationStructureInstanceKHR& instance = instancesAS.emplace_back();
			float3x4::Map(&instance.transform.matrix[0][0]) = transform.to_float3x4();
			instance.instanceCustomIndex = appendInstanceData(primNode, prim.get(), MeshInstanceData(materialAddress, vertexInfoIndex, primitiveCount), transform, luminance(prim->mMaterial->mMaterialData.getEmission()) * area);
			instance.mask = BVH_FLAG_TRIANGLES;
			instance.accelerationStructureReference = commandBuffer.mDevice->getAccelerationStructureAddressKHR(**it->second.first);

			const vk::AabbPositionsKHR& aabb = prim->mMesh->vertices().mAabb;
			for (uint32_t i = 0; i < 8; i++) {
				const int3 idx(i % 2, (i % 4) / 2, i / 4);
				float3 corner(
					idx[0] == 0 ? aabb.minX : aabb.maxX,
					idx[1] == 0 ? aabb.minY : aabb.maxY,
					idx[2] == 0 ? aabb.minZ : aabb.maxZ);
				corner = transform.transformPoint(corner);
				mFrameData.mAabbMin = min(mFrameData.mAabbMin, corner);
				mFrameData.mAabbMax = max(mFrameData.mAabbMax, corner);
			}

			meshInstanceIndices.emplace_back(prim.get(), (uint32_t)instance.instanceCustomIndex);
		});
	}

	{ // sphere instances
		ProfilerScope s("Process sphere instances", &commandBuffer);
		mNode.forEachDescendant<SpherePrimitive>([&](Node& primNode, const shared_ptr<SpherePrimitive>& prim) {
			if (!prim->mMaterial) return;

			const uint32_t materialAddress = appendMaterialData(prim->mMaterial.get());

			TransformData transform = nodeToWorld(primNode);
			const float radius = prim->mRadius * transform.m.block<3, 3>(0, 0).matrix().determinant();
			// remove scale/rotation from transform
			transform = TransformData(transform.m.col(3).head<3>(), quatf::identity(), float3::Ones());

			if (!prim->mMaterial->mMaterialData.getEmission().isZero())
				mFrameData.mEmissivePrimitiveCount++;

			const auto& [as, asbuf] = getAabbBlas(-float3::Constant(radius), float3::Constant(radius), !prim->mMaterial->alphaTest());

			vk::AccelerationStructureInstanceKHR& instance = instancesAS.emplace_back();
			float3x4::Map(&instance.transform.matrix[0][0]) = transform.to_float3x4();
			instance.instanceCustomIndex = appendInstanceData(primNode, prim.get(), SphereInstanceData(materialAddress, radius), transform, luminance(prim->mMaterial->mMaterialData.getEmission()) * (4 * M_PI * radius * radius));
			instance.mask = BVH_FLAG_SPHERES;
			instance.accelerationStructureReference = commandBuffer.mDevice->getAccelerationStructureAddressKHR(**as);

			const float3 center = transform.transformPoint(float3::Zero());
			mFrameData.mAabbMin = min<float, 3>(mFrameData.mAabbMin, center - float3::Constant(radius));
			mFrameData.mAabbMax = max<float, 3>(mFrameData.mAabbMax, center + float3::Constant(radius));
		});
	}

	{ // medium instances
		ProfilerScope s("Process media", &commandBuffer);
		mNode.forEachDescendant<Medium>([&](Node& primNode, const shared_ptr<Medium>& vol) {
			if (!vol) return;

			const uint32_t materialAddress = appendMaterialData(vol.get());

			auto densityGrid = vol->mDensityGrid->grid<float>();

			// get/build BLAS
			const nanovdb::Vec3R& mn = densityGrid->worldBBox().min();
			const nanovdb::Vec3R& mx = densityGrid->worldBBox().max();
			const size_t key = hashArgs((float)mn[0], (float)mn[1], (float)mn[2], (float)mx[0], (float)mx[1], (float)mx[2]);
			auto aabb_it = mAABBs.find(key);
			if (aabb_it == mAABBs.end()) {
				Buffer::View<vk::AabbPositionsKHR> aabb = make_shared<Buffer>(
					commandBuffer.mDevice,
					"aabb data",
					sizeof(vk::AabbPositionsKHR),
					vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress,
					vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
				aabb[0].minX = (float)mn[0];
				aabb[0].minY = (float)mn[1];
				aabb[0].minZ = (float)mn[2];
				aabb[0].maxX = (float)mx[0];
				aabb[0].maxY = (float)mx[1];
				aabb[0].maxZ = (float)mx[2];
				vk::AccelerationStructureGeometryAabbsDataKHR aabbs(aabb.deviceAddress(), sizeof(vk::AabbPositionsKHR));
				vk::AccelerationStructureGeometryKHR aabbGeometry(vk::GeometryTypeKHR::eAabbs, aabbs, vk::GeometryFlagBitsKHR::eOpaque);
				vk::AccelerationStructureBuildRangeInfoKHR range(1);
				commandBuffer.trackResource(aabb.buffer());

				auto [as, asbuf] = buildAccelerationStructure(commandBuffer, "aabb BLAS", vk::AccelerationStructureTypeKHR::eBottomLevel, aabbGeometry, range);

				blasBarriers.emplace_back(
					vk::AccessFlagBits::eAccelerationStructureWriteKHR, vk::AccessFlagBits::eAccelerationStructureReadKHR,
					VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
					**asbuf.buffer(), asbuf.offset(), asbuf.sizeBytes());

				aabb_it = mAABBs.emplace(key, make_pair(as, asbuf)).first;
			}

			// append to instance list
			const TransformData transform = nodeToWorld(primNode);
			vk::AccelerationStructureInstanceKHR& instance = instancesAS.emplace_back();
			float3x4::Map(&instance.transform.matrix[0][0]) = transform.to_float3x4();
			instance.instanceCustomIndex = appendInstanceData(primNode, vol.get(), VolumeInstanceData(materialAddress, mFrameData.mMaterialResources.mVolumeDataMap.at({ vol->mDensityBuffer.buffer(),vol->mDensityBuffer.offset() })), transform, 0);
			instance.mask = BVH_FLAG_VOLUME;
			instance.accelerationStructureReference = commandBuffer.mDevice->getAccelerationStructureAddressKHR(**aabb_it->second.first);

			for (uint32_t i = 0; i < 8; i++) {
				const int3 idx(i % 2, (i % 4) / 2, i / 4);
				float3 corner(idx[0] == 0 ? mn[0] : mx[0],
					idx[1] == 0 ? mn[1] : mx[1],
					idx[2] == 0 ? mn[2] : mx[2]);
				corner = transform.transformPoint(corner);
				mFrameData.mAabbMin = min(mFrameData.mAabbMin, corner);
				mFrameData.mAabbMax = max(mFrameData.mAabbMax, corner);
			}
		});
	}

	{ // environment material
		ProfilerScope s("Process environment", &commandBuffer);
		mFrameData.mEnvironmentMaterialAddress = -1;
		mNode.forEachDescendant<EnvironmentMap>([&](Node& node, const shared_ptr<EnvironmentMap> environment) {
			if (environment->mValue.isZero()) return true;
			mFrameData.mEnvironmentMaterialAddress = mFrameData.mMaterialResources.mMaterialData.sizeBytes();
			environment->store(mFrameData.mMaterialResources);
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

		const auto&[ as, asbuf ] = buildAccelerationStructure(commandBuffer, mNode.name() + "/TLAS", vk::AccelerationStructureTypeKHR::eTopLevel, geom, range);
		mFrameData.mDescriptors[{ "mAccelerationStructure", 0u }] = as;
		mFrameData.mAccelerationStructureBuffer = asbuf;
		mFrameData.mAccelerationStructureBuffer.barrier(commandBuffer,
			vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, vk::PipelineStageFlagBits::eComputeShader,
			vk::AccessFlagBits::eAccelerationStructureWriteKHR, vk::AccessFlagBits::eAccelerationStructureReadKHR);
	}

	{ // upload data
		ProfilerScope s("Upload scene data buffers");

		auto emptyBuffer = make_shared<Buffer>(commandBuffer.mDevice, "Empty", sizeof(TransformData), vk::BufferUsageFlagBits::eStorageBuffer);

		auto uploadOrEmpty = [&]<typename T>(const string& name, const vk::ArrayProxy<T>& data) -> Buffer::View<T> {
			if (data.empty())
				return emptyBuffer;
			else
				return mFrameData.mResourcePool.uploadData<T>(commandBuffer, name, data);
		};

		mFrameData.mDescriptors[{ "mInstances", 0u }]                 = uploadOrEmpty.operator()<InstanceData>  ("mInstances", instanceDatas);
		mFrameData.mDescriptors[{ "mInstanceTransforms", 0u }]        = uploadOrEmpty.operator()<TransformData> ("mInstanceTransforms", instanceTransforms);
		mFrameData.mDescriptors[{ "mInstanceInverseTransforms", 0u }] = uploadOrEmpty.operator()<TransformData> ("mInstanceInverseTransforms", instanceInverseTransforms);
		mFrameData.mDescriptors[{ "mInstanceMotionTransforms", 0u }]  = uploadOrEmpty.operator()<TransformData> ("mInstanceMotionTransforms", instanceMotionTransforms);
		mFrameData.mDescriptors[{ "mLightInstanceMap", 0u }]          = uploadOrEmpty.operator()<uint32_t>      ("mLightInstanceMap", lightInstanceMap);
		mFrameData.mDescriptors[{ "mInstanceLightMap", 0u }]          = uploadOrEmpty.operator()<uint32_t>      ("mInstanceLightMap", instanceLightMap);
		mFrameData.mDescriptors[{ "mMaterialData", 0u }]              = uploadOrEmpty.operator()<uint32_t>      ("mMaterialData", mFrameData.mMaterialResources.mMaterialData);
		mFrameData.mDescriptors[{ "mMeshVertexInfo", 0u }]            = uploadOrEmpty.operator()<MeshVertexInfo>("mMeshVertexInfo", meshVertexInfos);
		if (!instanceIndexMap.empty())
			mFrameData.mResourcePool.uploadData<uint32_t>(commandBuffer, "mInstanceIndexMap", instanceIndexMap);
	}

	for (uint32_t i = 0; i < mFrameData.mVertexBuffers.size(); i++)
		mFrameData.mDescriptors[{ "mVertexBuffers", i }] = mFrameData.mVertexBuffers[i];
	for (const auto& [image, index] : mFrameData.mMaterialResources.mImage4s)
		mFrameData.mDescriptors[{"mImages", index}] = ImageDescriptor{ image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
	for (const auto& [image, index] : mFrameData.mMaterialResources.mImage2s)
		mFrameData.mDescriptors[{"mImage2s", index}] = ImageDescriptor{ image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
	for (const auto& [image, index] : mFrameData.mMaterialResources.mImage1s)
		mFrameData.mDescriptors[{"mImage1s", index}] = ImageDescriptor{ image, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
	for (const auto& [vol, index] : mFrameData.mMaterialResources.mVolumeDataMap)
		mFrameData.mDescriptors[{"mVolumes", index}] = vol.first;
}


// drawGui functions

void Scene::drawGui() {
	ImGui::PushID(this);
	if (ImGui::Button("Load file")) {
		auto f = pfd::open_file("Open scene", "", loaderFilters());
		for (const string& filepath : f.result())
			mToLoad.emplace_back(filepath);
	}
	ImGui::Checkbox("Always update", &mAlwaysUpdate);

	if (!mFrameData.mMaterialResources.mImage4s.empty() || !mFrameData.mMaterialResources.mImage2s.empty() || !mFrameData.mMaterialResources.mImage1s.empty()) {
		const uint32_t w = ImGui::GetWindowSize().x;
		if (ImGui::CollapsingHeader("Image4s")) {
			vector<Image::View> images(mFrameData.mMaterialResources.mImage4s.size());
			for (const auto& [img, idx] : mFrameData.mMaterialResources.mImage4s)
				images[idx] = img;
			for (uint32_t i = 0; i < images.size(); i++) {
				const string label = "" + to_string(i) + ": " + images[i].image()->resourceName();
				if (ImGui::CollapsingHeader(label.c_str())) {
					ImGui::Text("%ux%u %s", images[i].extent().width, images[i].extent().height, to_string(images[i].image()->format()).c_str());
					ImGui::Image(Gui::getTextureID(images[i]), ImVec2(w, w * (float)images[i].extent().height / (float)images[i].extent().width));
				}
			}
		}
		if (ImGui::CollapsingHeader("Image2s")) {
			vector<Image::View> images(mFrameData.mMaterialResources.mImage2s.size());
			for (const auto& [img, idx] : mFrameData.mMaterialResources.mImage2s)
				images[idx] = img;
			for (uint32_t i = 0; i < images.size(); i++) {
				const string label = "" + to_string(i) + ": " + images[i].image()->resourceName();
				if (ImGui::CollapsingHeader(label.c_str())) {
					ImGui::Text("%ux%u %s", images[i].extent().width, images[i].extent().height, to_string(images[i].image()->format()).c_str());
					ImGui::Image(Gui::getTextureID(images[i]), ImVec2(w, w * (float)images[i].extent().height / (float)images[i].extent().width));
				}
			}
		}
		if (ImGui::CollapsingHeader("Image1s")) {
			vector<Image::View> images(mFrameData.mMaterialResources.mImage1s.size());
			for (const auto& [img, idx] : mFrameData.mMaterialResources.mImage1s)
				images[idx] = img;
			for (uint32_t i = 0; i < images.size(); i++) {
				const string label = "" + to_string(i) + ": " + images[i].image()->resourceName();
				if (ImGui::CollapsingHeader(label.c_str())) {
					ImGui::Text("%ux%u %s", images[i].extent().width, images[i].extent().height, to_string(images[i].image()->format()).c_str());
					ImGui::Image(Gui::getTextureID(images[i]), ImVec2(w, w * (float)images[i].extent().height / (float)images[i].extent().width));
				}
			}
		}
	}

	if (ImGui::CollapsingHeader("Resources")) {
		ImGui::Indent();
		mFrameData.mResourcePool.drawGui();
		ImGui::Unindent();
	}
	ImGui::PopID();
}

template<int N>
bool ImageValue<N>::drawGui(const string& label) {
	bool changed = ImGui::DragScalarN(label.c_str(), ImGuiDataType_Float, mValue.data(), N, .01f);

	if (mImage) {
		ImGui::PushID(this);
		const bool d = ImGui::Button("x");
		ImGui::PopID();
		if (d) {
			mImage = {};
			changed = true;
		} else {
			const uint32_t w = ImGui::GetWindowSize().x;
			ImGui::Image(Gui::getTextureID(mImage), ImVec2(w, w * (float)mImage.extent().height / (float)mImage.extent().width));
		}
	}

	return changed;
}

void Frustum(float left, float right, float bottom, float top, float znear, float zfar, float* m16) {
   float temp, temp2, temp3, temp4;
   temp = 2.0f * znear;
   temp2 = right - left;
   temp3 = top - bottom;
   temp4 = zfar - znear;
   m16[0] = temp / temp2;
   m16[1] = 0.0;
   m16[2] = 0.0;
   m16[3] = 0.0;
   m16[4] = 0.0;
   m16[5] = temp / temp3;
   m16[6] = 0.0;
   m16[7] = 0.0;
   m16[8] = (right + left) / temp2;
   m16[9] = (top + bottom) / temp3;
   m16[10] = (-zfar - znear) / temp4;
   m16[11] = -1.0f;
   m16[12] = 0.0;
   m16[13] = 0.0;
   m16[14] = (-temp * zfar) / temp4;
   m16[15] = 0.0;
}
void Perspective(float fovy, float aspectRatio, float znear, float zfar, float* m16) {
   float ymax, xmax;
   ymax = znear * tanf(fovy / 2);
   xmax = ymax * aspectRatio;
   Frustum(-xmax, xmax, -ymax, ymax, znear, zfar, m16);
}
void LookAt(const float3 eye, const float3 fwd, const float3 up, float* m16) {
	float3 Z = normalize(fwd);
	float3 Y = normalize(up);

	float3 X = normalize(cross(Y, Z));
	Y = normalize(cross(Z, X));

	m16[0] = X[0];
	m16[1] = Y[0];
	m16[2] = Z[0];
	m16[3] = 0.0f;
	m16[4] = X[1];
	m16[5] = Y[1];
	m16[6] = Z[1];
	m16[7] = 0.0f;
	m16[8] = X[2];
	m16[9] = Y[2];
	m16[10] = Z[2];
	m16[11] = 0.0f;
	m16[12] = -dot(X, eye);
	m16[13] = -dot(Y, eye);
	m16[14] = -dot(Z, eye);
	m16[15] = 1.0f;
}
bool EditTransform(float* view, float* projection, Eigen::Matrix4f& parent, Eigen::Matrix4f& matrix) {
	static ImGuizmo::OPERATION mCurrentGizmoOperation = ImGuizmo::ROTATE;
	static ImGuizmo::MODE mCurrentGizmoMode = ImGuizmo::LOCAL;
	static bool useSnap = false;
	static float3 snapTranslation = float3::Constant(0.05f);
	static float3 snapAngle = float3::Constant(M_PI/8);
	static float3 snapScale = float3::Constant(0.1f);

	if (ImGui::IsKeyPressed(ImGuiKey_T))
		mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
	if (ImGui::IsKeyPressed(ImGuiKey_R))
		mCurrentGizmoOperation = ImGuizmo::ROTATE;
	if (ImGui::IsKeyPressed(ImGuiKey_Y))
		mCurrentGizmoOperation = ImGuizmo::SCALE;

	if (ImGui::RadioButton("Translate (T)", mCurrentGizmoOperation == ImGuizmo::TRANSLATE))
		mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
	ImGui::SameLine();
	if (ImGui::RadioButton("Rotate (R)", mCurrentGizmoOperation == ImGuizmo::ROTATE))
		mCurrentGizmoOperation = ImGuizmo::ROTATE;
	ImGui::SameLine();
	if (ImGui::RadioButton("Scale (Y)", mCurrentGizmoOperation == ImGuizmo::SCALE))
		mCurrentGizmoOperation = ImGuizmo::SCALE;

	bool changed = false;

	float matrixTranslation[3], matrixRotation[3], matrixScale[3];
	ImGuizmo::DecomposeMatrixToComponents(matrix.data(), matrixTranslation, matrixRotation, matrixScale);
	if (ImGui::InputFloat3("T", matrixTranslation)) changed = true;
	if (ImGui::InputFloat3("R", matrixRotation)) changed = true;
	if (ImGui::InputFloat3("S", matrixScale)) changed = true;
	if (changed)
		ImGuizmo::RecomposeMatrixFromComponents(matrixTranslation, matrixRotation, matrixScale, matrix.data());

	if (ImGui::RadioButton("Local", mCurrentGizmoMode == ImGuizmo::LOCAL))
		mCurrentGizmoMode = ImGuizmo::LOCAL;
	ImGui::SameLine();
	if (ImGui::RadioButton("World", mCurrentGizmoMode == ImGuizmo::WORLD))
		mCurrentGizmoMode = ImGuizmo::WORLD;

	if (ImGui::IsKeyPressed(ImGuiKey_U))
		useSnap = !useSnap;
	ImGui::Checkbox("Snap (U)", &useSnap);
	ImGui::SameLine();
	float3* snap;
	switch (mCurrentGizmoOperation) {
	case ImGuizmo::TRANSLATE:
		snap = &snapTranslation;
		ImGui::InputFloat3("Snap", snap->data());
		break;
	case ImGuizmo::ROTATE:
		snap = &snapAngle;
		ImGui::InputFloat("Angle Snap", snap->data());
		break;
	case ImGuizmo::SCALE:
		snap = &snapScale;
		ImGui::InputFloat("Scale Snap", snap->data());
		break;
	}

	ImGuiIO& io = ImGui::GetIO();
	ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);

	matrix = parent * matrix;

	ImGuizmo::SetID(0);
	if (ImGuizmo::Manipulate(view, projection, mCurrentGizmoOperation, mCurrentGizmoMode, matrix.data(), NULL, useSnap ? snap->data() : NULL))
		changed = true;

	matrix = parent.inverse() * matrix;

	return changed;
}

void TransformData::drawGui(Node& node) {
	Eigen::Matrix4f m4x4;
	m4x4.topRows<3>() = m;
	m4x4.bottomRows<1>() = float4(0,0,0,1);

	shared_ptr<Node> sceneNode;
	if (const shared_ptr<Scene> scene = node.findAncestor<Scene>(&sceneNode)) {
		// animate hack
		if (scene->mAnimatedTransform.get() == &node) {
			if (ImGui::Button("Stop animating")) scene->mAnimatedTransform = nullptr;
			ImGui::DragFloat3("Move", scene->mAnimateTranslate.data(), .01f);
			ImGui::DragFloat3("Rotate", scene->mAnimateRotate.data(), .01f);
			const bool was_zero = scene->mAnimateWiggleOffset.isZero();
			ImGui::DragFloat3("Wiggle offset", scene->mAnimateWiggleOffset.data(), .01f);
			if (!scene->mAnimateWiggleOffset.isZero()) {
				if (ImGui::Button("Set wiggle anchor") || was_zero) {
					scene->mAnimateWiggleBase = m.rightCols<1>();
					scene->mAnimateWiggleTime = 0;
				}
				ImGui::DragFloat("Wiggle speed", &scene->mAnimateWiggleSpeed);
			}
		} else if (ImGui::Button("Animate"))
			scene->mAnimatedTransform = node.getPtr();

		shared_ptr<Node> cameraNode;
		if (const shared_ptr<Camera> camera = sceneNode->findDescendant<Camera>(&cameraNode)) {

			Eigen::Matrix4f parentTransform = Eigen::Matrix4f::Identity();
			if (shared_ptr<Node> p = node.parent())
				parentTransform.topRows<3>() = nodeToWorld(*p).m;

			const TransformData cameraTransform = nodeToWorld(*cameraNode);

			Eigen::Matrix4f view;
			Eigen::Matrix4f proj;
			LookAt(cameraTransform.transformPoint(float3::Zero()), cameraTransform.transformVector(float3(0,0,1)), cameraTransform.transformVector(float3(0,1,0)), view.data());
			Perspective(camera->mProjection.mVerticalFoV, camera->mImageRect.extent.width / (float)camera->mImageRect.extent.height, abs(camera->mProjection.mNearPlane), abs(camera->mProjection.mFarPlane), proj.data());
			if (EditTransform(view.data(), proj.data(), parentTransform, m4x4)) {
				m = m4x4.topRows<3>();
				scene->markDirty();
			}

			return;
		}
	}

	// transform not in scene, or no camera
	bool changed = false;
	float matrixTranslation[3], matrixRotation[3], matrixScale[3];
	ImGuizmo::DecomposeMatrixToComponents(m4x4.data(), matrixTranslation, matrixRotation, matrixScale);
	if (ImGui::InputFloat3("Tr", matrixTranslation)) changed = true;
	if (ImGui::InputFloat3("Rt", matrixRotation)) changed = true;
	if (ImGui::InputFloat3("Sc", matrixScale)) changed = true;
	if (changed) {
		ImGuizmo::RecomposeMatrixFromComponents(matrixTranslation, matrixRotation, matrixScale, m4x4.data());
		m = m4x4.topRows<3>();
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

void EnvironmentMap::drawGui(Node& node) {
	if (ImageValue<3>::drawGui("Emission"))
		if (const shared_ptr<Scene> scene = node.findAncestor<Scene>())
			scene->markDirty();
}

void Material::drawGui(Node& node) {
	bool changed = false;

	auto slider = [&](const char* label, float value, function<void(float)> setter) {
		if (ImGui::SliderFloat(label, &value, 0, 1)) {
			setter(value);
			changed = true;
		}
	};

	float3 color = mMaterialData.getBaseColor();
	if (ImGui::ColorEdit3("Base Color", color.data(), ImGuiColorEditFlags_Float|ImGuiColorEditFlags_PickerHueBar)) {
		mMaterialData.setBaseColor(color);
		changed = true;
	}
	float3 emission = mMaterialData.getEmission();
	if (ImGui::ColorEdit3("Emission", emission.data(), ImGuiColorEditFlags_Float|ImGuiColorEditFlags_HDR|ImGuiColorEditFlags_PickerHueBar)) {
		mMaterialData.setEmission(emission);
		changed = true;
	}

	ImGui::PushItemWidth(80);
	slider("Metallic",         mMaterialData.getMetallic(),       bind_front(&PackedMaterialData::setMetallic, &mMaterialData));
	slider("Roughness",        mMaterialData.getRoughness(),      bind_front(&PackedMaterialData::setRoughness, &mMaterialData));
	slider("Anisotropic",      mMaterialData.getAnisotropic(),    bind_front(&PackedMaterialData::setAnisotropic, &mMaterialData));
	slider("Subsurface",       mMaterialData.getSubsurface(),     bind_front(&PackedMaterialData::setSubsurface, &mMaterialData));
	slider("Clearcoat",        mMaterialData.getClearcoat(),      bind_front(&PackedMaterialData::setClearcoat, &mMaterialData));
	slider("Clearcoat gloss",  mMaterialData.getClearcoatGloss(), bind_front(&PackedMaterialData::setClearcoatGloss, &mMaterialData));
	slider("Transmission",     mMaterialData.getTransmission(),   bind_front(&PackedMaterialData::setTransmission, &mMaterialData));
	slider("Refraction index", mMaterialData.getEta(),            bind_front(&PackedMaterialData::setEta, &mMaterialData));

	if (mBumpImage) {
		if (ImGui::DragFloat("Bump Strength", &mBumpStrength, 0.1, 0, 10))
			changed = true;
	}
	if (mAlphaMask) {
		if (ImGui::SliderFloat("Alpha cutoff", &mAlphaCutoff, 0, 1))
			changed = true;
	}
	ImGui::PopItemWidth();

	const float w = ImGui::CalcItemWidth() - 4;
	for (const Image::View& image : mImages)
		if (image) {
			ImGui::Text(image.image()->resourceName().c_str());
			ImGui::Image(Gui::getTextureID(image), ImVec2(w, w * image.extent().height / (float)image.extent().width));
		}
	if (mAlphaMask) {
		if (ImGui::SliderFloat("Alpha cutoff", &mAlphaCutoff, 0, 1))
			changed = true;
		ImGui::Text(mAlphaMask.image()->resourceName().c_str());
		ImGui::Image(Gui::getTextureID(mAlphaMask), ImVec2(w, w * mAlphaMask.extent().height / (float)mAlphaMask.extent().width));
	}
	if (mBumpImage) {
		ImGui::Text(mBumpImage.image()->resourceName().c_str());
		ImGui::Image(Gui::getTextureID(mBumpImage), ImVec2(w, w * mBumpImage.extent().height / (float)mBumpImage.extent().width));
	}

	if (changed)
		if (const auto scene = node.findAncestor<Scene>())
			scene->markDirty();
}
void Medium::drawGui(Node& node) {
	bool changed = false;

	if (ImGui::ColorEdit3("Density", mDensityScale.data(), ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float)) changed = true;
	if (ImGui::ColorEdit3("Albedo", mAlbedoScale.data(), ImGuiColorEditFlags_Float)) changed = true;
	if (ImGui::SliderFloat("Anisotropy", &mAnisotropy, -.999f, .999f)) changed = true;

	if (changed) {
		const auto scene = node.findAncestor<Scene>();
		if (scene)
			scene->markDirty();
	}
}

}