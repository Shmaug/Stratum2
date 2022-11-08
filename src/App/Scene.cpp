#include "Scene.hpp"
#include <Core/Instance.hpp>
#include <Core/Profiler.hpp>
#include <Core/CommandBuffer.hpp>
#include <Core/Pipeline.hpp>
#include <Core/Swapchain.hpp>
#include <Core/Window.hpp>

#include <future>
#include <portable-file-dialogs.h>

namespace tinyvkpt {

tuple<shared_ptr<vk::raii::AccelerationStructureKHR>, Buffer::View<byte>> buildAccelerationStructure(CommandBuffer& commandBuffer, const string& name, const vk::AccelerationStructureTypeKHR type, const vk::ArrayProxy<const vk::AccelerationStructureGeometryKHR>& geometries,  const vk::ArrayProxy<const vk::AccelerationStructureBuildRangeInfoKHR>& buildRanges) {
	vk::AccelerationStructureBuildGeometryInfoKHR buildGeometry(type,vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace, vk::BuildAccelerationStructureModeKHR::eBuild);
	buildGeometry.setGeometries(geometries);

	vk::AccelerationStructureBuildSizesInfoKHR buildSizes;
	if (buildRanges.size() > 0 && buildRanges.front().primitiveCount > 0) {
		vector<uint32_t> counts((uint32_t)geometries.size());
		for (uint32_t i = 0; i < geometries.size(); i++)
			counts[i] = (buildRanges.data() + i)->primitiveCount;
		buildSizes = commandBuffer.mDevice->getAccelerationStructureBuildSizesKHR(vk::AccelerationStructureBuildTypeKHR::eDevice, buildGeometry, counts);
	} else
		buildSizes.accelerationStructureSize = buildSizes.buildScratchSize = 4;

	Buffer::View<byte> buffer      = make_shared<Buffer>(commandBuffer.mDevice, name,                  buildSizes.accelerationStructureSize, vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR|vk::BufferUsageFlagBits::eShaderDeviceAddress);
	Buffer::View<byte> scratchData = make_shared<Buffer>(commandBuffer.mDevice, name + "/scratchData", buildSizes.buildScratchSize         , vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR|vk::BufferUsageFlagBits::eShaderDeviceAddress|vk::BufferUsageFlagBits::eStorageBuffer);
	shared_ptr<vk::raii::AccelerationStructureKHR> accelerationStructure = make_shared<vk::raii::AccelerationStructureKHR>(*commandBuffer.mDevice, vk::AccelerationStructureCreateInfoKHR({}, **buffer.buffer(), buffer.offset(), buffer.sizeBytes(), type));

	buildGeometry.dstAccelerationStructure = **accelerationStructure;
	buildGeometry.scratchData = scratchData.deviceAddress();

	commandBuffer->buildAccelerationStructuresKHR(buildGeometry, buildRanges.data());
	commandBuffer.trackResource(buffer.buffer());
	commandBuffer.trackResource(scratchData.buffer());
	commandBuffer.trackVulkanResource(accelerationStructure);

	return tie(accelerationStructure, buffer);
}

static float3 gAnimateTranslate = float3::Zero();
static float3 gAnimateRotate = float3::Zero();
static float3 gAnimateWiggleBase = float3::Zero();
static float3 gAnimateWiggleOffset = float3::Zero();
static float gAnimateWiggleSpeed = 1;
static float gAnimateWiggleTime = 0;
static TransformData* gAnimatedTransform = nullptr;

static bool gChanged = false;
/*
inline void drawGui(Camera& cam) {
	bool ortho = cam.mProjection.orthographic();
	if (ImGui::Checkbox("Orthographic", reinterpret_cast<bool*>(&ortho)))
		cam.mProjection.vertical_fov = -cam.mProjection.vertical_fov;
	ImGui::DragFloat("Near Plane", &cam.mProjection.near_plane, 0.01f, -1, 1);
	if (cam.mProjection.orthographic()) {
		ImGui::DragFloat("Far Plane", &cam.mProjection.far_plane, 0.01f, -1, 1);
		ImGui::DragFloat2("Projection Scale", cam.mProjection.scale.data(), 0.01f, -1, 1);
	} else {
		float fovy = degrees(2 * atan(1 / cam.mProjection.scale[1]));
		if (ImGui::DragFloat("Vertical FoV", &fovy, 0.01f, 1, 179)) {
			const float aspect = cam.mProjection.scale[0] / cam.mProjection.scale[1];
			cam.mProjection.scale[1] = 1 / tan(radians(fovy / 2));
			cam.mProjection.scale[0] = cam.mProjection.scale[1] * aspect;
		}
	}
	ImGui::DragFloat2("Projection Offset", cam.mProjection.offset.data(), 0.01f, -1, 1);
	ImGui::InputInt2("ImageRect Offset", &cam.mImageRect.offset.x);
	ImGui::InputInt2("ImageRect Extent", reinterpret_cast<int32_t*>(&cam.mImageRect.extent.width));
}
inline void drawGui(TransformData& t) {
	TransformData prev = t;

	float3 translate = t.m.topRightCorner(3, 1);
	float3 scale;
	scale.x() = t.m.block(0, 0, 3, 1).matrix().norm();
	scale.y() = t.m.block(0, 1, 3, 1).matrix().norm();
	scale.z() = t.m.block(0, 2, 3, 1).matrix().norm();
	const Eigen::Matrix3f r = t.m.block<3, 3>(0, 0).matrix() * Eigen::DiagonalMatrix<float, 3, 3>(1 / scale.x(), 1 / scale.y(), 1 / scale.z());
	Eigen::Quaternionf rotation(r);

	if (ImGui::DragFloat3("Translation", translate.data(), .1f) && !translate.hasNaN()) {
		t.m.topRightCorner(3, 1) = translate;
		gChanged = true;
	}

	bool v = ImGui::DragFloat3("Rotation (XYZ)", rotation.vec().data(), .1f);
	v |= ImGui::DragFloat("Rotation (W)", &rotation.w(), .1f);
	if (v && !rotation.vec().hasNaN()) {
		t.m.block<3, 3>(0, 0) = rotation.normalized().matrix();
		gChanged = true;
	}

	if (ImGui::DragFloat3("Scale", scale.data(), .1f) && !rotation.vec().hasNaN() && !scale.hasNaN()) {
		t.m.block<3, 3>(0, 0) = rotation.normalized().matrix() * Eigen::DiagonalMatrix<float, 3, 3>(scale.x() == 0 ? 1 : scale.x(), scale.y() == 0 ? 1 : scale.y(), scale.z() == 0 ? 1 : scale.z());
		gChanged = true;
	}

	if (gAnimatedTransform == &t) {
		if (ImGui::Button("Stop Animating")) gAnimatedTransform = nullptr;
		ImGui::DragFloat3("Move", gAnimateTranslate.data(), .01f);
		ImGui::DragFloat3("Rotate", gAnimateRotate.data(), .01f);
		const bool was_zero = gAnimateWiggleOffset.isZero();
		ImGui::DragFloat3("Wiggle Offset", gAnimateWiggleOffset.data(), .01f);
		if (!gAnimateWiggleOffset.isZero()) {
			if (ImGui::Button("Set Wiggle Anchor") || was_zero) {
				gAnimateWiggleBase = translate;
				gAnimateWiggleTime = 0;
			}
			ImGui::DragFloat("Wiggle Speed", &gAnimateWiggleSpeed);
		}
	} else if (ImGui::Button("Animate"))
		gAnimatedTransform = &t;
}
inline void drawGui(MeshPrimitive& mesh) {
	if (mesh.mMesh) {
		ImGui::Text("%s", type_index(typeid(Mesh)).name());
		ImGui::SameLine();
		inspector.component_ptr_field(mesh.mMesh);
	}
	if (mesh.mMaterial) {
		ImGui::Text("%s", type_index(typeid(Material)).name());
		ImGui::SameLine();
		inspector.component_ptr_field(mesh.mMaterial);
	}
}
inline void drawGui(SpherePrimitive& sphere) {
	if (ImGui::DragFloat("Radius", &sphere.mRadius, .01f)) gChanged = true;
	if (sphere.mMaterial) {
		ImGui::Text("%s", type_index(typeid(Material)).name());
		ImGui::SameLine();
		inspector.component_ptr_field(sphere.mMaterial);
	}
}
*/
TransformData nodeToWorld(const Node& node) {
	TransformData transform;
	if (auto c = node.getComponent<TransformData>(); c)
		transform = *c;
	else
		transform = TransformData(float3::Zero(), quatf_identity(), float3::Ones());
	NodePtr p = node.parent();
	while (p) {
		if (auto c = p->getComponent<TransformData>())
			transform = tmul(*c, transform);
		p = p->parent();
	}
	return transform;
}

NodePtr Scene::loadEnvironmentMap(CommandBuffer& commandBuffer, const filesystem::path& filepath) {
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

	const NodePtr node = Node::create(filepath.stem().string());
	node->makeComponent<Environment>(ImageValue3{float3::Ones(), img});
	return node;
}

ImageValue1 Scene::alphaToRoughness(CommandBuffer& commandBuffer, const ImageValue1& alpha) {
	ImageValue1 roughness;
	roughness.value = sqrt(alpha.value);
	if (alpha.image) {
		Image::Metadata md;
		md.mFormat = alpha.image.image()->format();
		md.mExtent = alpha.image.extent();
		md.mLevels = Image::maxMipLevels(md.mExtent);
		md.mUsage = vk::ImageUsageFlagBits::eTransferSrc|vk::ImageUsageFlagBits::eTransferDst|vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage;
		roughness.image = make_shared<Image>(commandBuffer.mDevice, "roughness", md);
		mConvertAlphaToRoughnessPipeline.get(commandBuffer.mDevice)->dispatchTiled(commandBuffer,
			alpha.image.extent(),
			Descriptors{
				{ { "gInput", 0 }, DescriptorValue{ ImageDescriptor{
						alpha.image,
						vk::ImageLayout::eShaderReadOnlyOptimal,
						vk::AccessFlagBits::eShaderRead,
						{} } } },
				{ { "gRoughnessRW", 0 }, DescriptorValue{ ImageDescriptor{
						roughness.image,
						vk::ImageLayout::eGeneral,
						vk::AccessFlagBits::eShaderWrite,
						{} } } }
			} );
		roughness.image.image()->generateMipMaps(commandBuffer);
		cout << "Converted alpha to roughness: " << alpha.image.image()->resourceName() << endl;
	}
	return roughness;
}
ImageValue1 Scene::shininessToRoughness(CommandBuffer& commandBuffer, const ImageValue1& shininess) {
	ImageValue1 roughness;
	roughness.value = sqrt(2 / (shininess.value + 2));
	if (shininess.image) {
		Image::Metadata md;
		md.mFormat = shininess.image.image()->format();
		md.mExtent = shininess.image.extent();
		md.mLevels = Image::maxMipLevels(md.mExtent);
		md.mUsage = vk::ImageUsageFlagBits::eTransferSrc|vk::ImageUsageFlagBits::eTransferDst|vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage;
		roughness.image = make_shared<Image>(commandBuffer.mDevice, "roughness", md);
		mConvertShininessToRoughnessPipeline.get(commandBuffer.mDevice)->dispatchTiled(commandBuffer,
			shininess.image.extent(),
			Descriptors{
				{ { "gInput", 0 }, ImageDescriptor{
					shininess.image,
					vk::ImageLayout::eShaderReadOnlyOptimal,
					vk::AccessFlagBits::eShaderRead,
					{} } },
				{ { "gRoughnessRW", 0 }, ImageDescriptor{
					roughness.image,
					vk::ImageLayout::eGeneral,
					vk::AccessFlagBits::eShaderWrite,
					{} } }
			} );
		roughness.image.image()->generateMipMaps(commandBuffer);
		cout << "Converted shininess to roughness: " << shininess.image.image()->resourceName() << endl;
	}
	return roughness;
}

Material Scene::makeMetallicRoughnessMaterial(CommandBuffer& commandBuffer, const ImageValue3& base_color, const ImageValue4& metallic_roughness, const ImageValue3& transmission, const float eta, const ImageValue3& emission) {
	Material m;
	if ((emission.value > 0).any()) {
		m.mValues[0].image = emission.image;
		m.baseColor() = emission.value/luminance(emission.value);
		m.emission() = luminance(emission.value);
		m.eta() = 0; // eta
		return m;
	}
	m.baseColor() = base_color.value;
	m.emission() = 0;
	m.metallic() = metallic_roughness.value.z(); // metallic
	m.roughness() = metallic_roughness.value.y(); // roughness
	m.anisotropic() = 0; // anisotropic
	m.subsurface() = 0; // subsurface
	m.clearcoat() = 0; // clearcoat
	m.clearcoatGloss() = 1; // clearcoat gloss
	m.transmission() = luminance(transmission.value);
	m.eta() = eta;
	if (base_color.image || metallic_roughness.image || transmission.image) {
		Descriptors descriptors;

		Image::View d = base_color.image ? base_color.image : metallic_roughness.image ? metallic_roughness.image : transmission.image;
		Image::Metadata md;
		md.mExtent = d.extent();
		md.mLevels = Image::maxMipLevels(md.mExtent);
		md.mFormat = vk::Format::eR8G8B8A8Unorm;
		md.mUsage = vk::ImageUsageFlagBits::eTransferSrc|vk::ImageUsageFlagBits::eTransferDst|vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage;
		for (int i = 0; i < DISNEY_DATA_N; i++) {
			m.mValues[i].image = make_shared<Image>(commandBuffer.mDevice, "material data", md);
			descriptors[{ "gOutput", i }] = ImageDescriptor{m.mValues[i].image, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {}};
		}
		if (base_color.image) {
			md.mLevels = 1;
			md.mFormat = vk::Format::eR8Unorm;
			md.mUsage = vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage;
			m.mAlphaMask = make_shared<Image>(commandBuffer.mDevice, "alpha mask", md);
		}
		descriptors[{ "gOutputAlphaMask", 0 }] = ImageDescriptor{m.mAlphaMask ? m.mAlphaMask : m.mValues[0].image, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {}};

		m.mMinAlpha = make_shared<Buffer>(commandBuffer.mDevice, "min_alpha", sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		m.mMinAlpha[0] = 0xFFFFFFFF;
		descriptors[{ "gOutputMinAlpha", 0 }] = m.mMinAlpha;

		descriptors[{ "gDiffuse", 0 }] = ImageDescriptor{base_color.image ? base_color.image : d, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {}};
		descriptors[{ "gSpecular", 0 }] = ImageDescriptor{metallic_roughness.image ? metallic_roughness.image : d, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {}};
		descriptors[{ "gTransmittance", 0 }] = ImageDescriptor{transmission.image ? transmission.image : d, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {}};
		Defines defs;
		if (base_color.image) defs["gUseDiffuse"] = "1";
		if (metallic_roughness.image) defs["gUseSpecular"] = "1";
		if (transmission.image) defs["gUseTransmittance"] = "1";
		mConvertPbrPipeline.get(commandBuffer.mDevice, defs)->dispatchTiled(commandBuffer, d.extent(), descriptors);

		for (int i = 0; i < DISNEY_DATA_N; i++)
			m.mValues[i].image.image()->generateMipMaps(commandBuffer);
	}
	return m;
}
Material Scene::makeDiffuseSpecularMaterial(CommandBuffer& commandBuffer, const ImageValue3& diffuse, const ImageValue3& specular, const ImageValue1& roughness, const ImageValue3& transmission, const float eta, const ImageValue3& emission) {
	Material m;
	if ((emission.value > 0).any()) {
		m.mValues[0].image = emission.image;
		m.baseColor() = emission.value/luminance(emission.value);
		m.emission() = luminance(emission.value);
		m.eta()  = 0; // eta
		return m;
	}
	const float ld = luminance(diffuse.value);
	const float ls = luminance(specular.value);
	const float lt = luminance(transmission.value);
	m.baseColor() = (diffuse.value*ld + specular.value*ls + transmission.value*lt) / (ld + ls + lt);
	m.emission() = 0;
	m.metallic() = ls / (ld + ls + lt);
	m.roughness() = roughness.value;
	m.anisotropic() = 0;
	m.subsurface() = 0;
	m.clearcoat() = 0;
	m.clearcoatGloss() = 1;
	m.transmission() = lt / (ld + ls + lt);
	m.eta() = eta;
	if (diffuse.image || specular.image || transmission.image || roughness.image) {
		Descriptors descriptors;

		Image::View d = diffuse.image ? diffuse.image : specular.image ? specular.image : transmission.image ? transmission.image : roughness.image;
		Image::Metadata md;
		md.mExtent = d.extent();
		md.mLevels = Image::maxMipLevels(md.mExtent);
		md.mFormat = vk::Format::eR8G8B8A8Unorm;
		md.mUsage = vk::ImageUsageFlagBits::eTransferSrc|vk::ImageUsageFlagBits::eTransferDst|vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage;
		for (int i = 0; i < DISNEY_DATA_N; i++) {
			m.mValues[i].image = make_shared<Image>(commandBuffer.mDevice, "material data", md);
			descriptors[{"gOutput",i}] = ImageDescriptor{ m.mValues[i].image, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };
		}
		if (diffuse.image) {
			md.mLevels = 1;
			md.mFormat = vk::Format::eR8Unorm;
			md.mUsage = vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage;
			m.mAlphaMask = make_shared<Image>(commandBuffer.mDevice, "alpha mask", md);
		}
		descriptors[{"gOutputAlphaMask",0}] = ImageDescriptor{ m.mAlphaMask ? m.mAlphaMask : m.mValues[0].image, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite, {} };

		m.mMinAlpha = make_shared<Buffer>(commandBuffer.mDevice, "min_alpha", sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		m.mMinAlpha[0] = 0xFFFFFFFF;
		descriptors[{"gOutputMinAlpha",0}] = m.mMinAlpha;

		descriptors[{"gDiffuse",0}]       = ImageDescriptor{ diffuse.image      ? diffuse.image      : d, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{"gSpecular",0}]      = ImageDescriptor{ specular.image     ? specular.image     : d, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{"gTransmittance",0}] = ImageDescriptor{ transmission.image ? transmission.image : d, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		descriptors[{"gRoughness",0}]     = ImageDescriptor{ roughness.image    ? roughness.image    : d, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead, {} };
		Defines defs;
		if (diffuse.image)      defs["gUseDiffuse"] = "1";
		if (specular.image)     defs["gUseSpecular"] = "1";
		if (transmission.image) defs["gUseTransmittance"] = "1";
		if (roughness.image)    defs["gUseRoughness"] = "1";

		mConvertDiffuseSpecularPipeline.get(commandBuffer.mDevice, defs)->dispatchTiled(commandBuffer, d.extent(), descriptors);

		for (int i = 0; i < DISNEY_DATA_N; i++)
			m.mValues[i].image.image()->generateMipMaps(commandBuffer);
	}
	return m;
}

Scene::Scene(Device& device) {
	ComputePipeline::Metadata md;
	md.mBindingFlags["gPositions"] = vk::DescriptorBindingFlagBits::ePartiallyBound;
	mCopyVerticesPipeline = ComputePipelineCache("kernels/copy_vertices.hlsl");

	mConvertDiffuseSpecularPipeline 	 = ComputePipelineCache("kernels/material_convert.hlsl", "alpha_to_roughness");
	mConvertPbrPipeline 				 = ComputePipelineCache("kernels/material_convert.hlsl", "shininess_to_roughness");
	mConvertAlphaToRoughnessPipeline 	 = ComputePipelineCache("kernels/material_convert.hlsl", "from_gltf_pbr");
	mConvertShininessToRoughnessPipeline = ComputePipelineCache("kernels/material_convert.hlsl", "from_diffuse_specular");

	for (const string arg : device.mInstance.findArguments("scene"))
		mToLoad.emplace_back(arg);

	mRootNode = Node::create("Scene Root");
}

void Scene::drawGui() {
	if (mResources) {
		ImGui::Text("%lu instances"          , mResources->mInstances.size());
		ImGui::Text("%lu light instances"    , mResources->mLightInstanceMap.size());
		ImGui::Text("%lu emissive primitives", mResources->mEmissivePrimitiveCount);
		ImGui::Text("%u materials"           , mResources->mMaterialCount);
	}
	if (ImGui::Button("Load File")) {
		auto f = pfd::open_file("Open scene", "", loaderFilters());
		for (const string& filepath : f.result())
			mToLoad.emplace_back(filepath);
	}
	ImGui::Checkbox("Always Update", &mAlwaysUpdate);
}

void Scene::update(CommandBuffer& commandBuffer, const float deltaTime) {
	ProfilerScope s("Scene::update", &commandBuffer);

	if (gAnimatedTransform) {
		const float r = length(gAnimateRotate);
		const quatf rotate = (r > 0) ? angle_axis(r * deltaTime, gAnimateRotate / r) : quatf_identity();
		*gAnimatedTransform = tmul(*gAnimatedTransform, TransformData(gAnimateTranslate * deltaTime, rotate, float3::Ones()));
		if (!gAnimateWiggleOffset.isZero()) {
			gAnimatedTransform->m.topRightCorner(3, 1) = gAnimateWiggleBase + gAnimateWiggleOffset*sin(gAnimateWiggleTime);
			gAnimateWiggleTime += deltaTime*gAnimateWiggleSpeed;
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

	for (const string& file : mRootNode->findAncestor<Swapchain>()->mWindow.inputState().files())
		mToLoad.emplace_back(file);

	if (gChanged) {
		mUpdateOnce = true;
		gChanged = false;
	}

	bool update = mAlwaysUpdate || mUpdateOnce;
	bool loaded = false;
	for (const string& file : mToLoad) {
		const filesystem::path filepath = file;
		try {
			const NodePtr n = load(commandBuffer, filepath);
			mRootNode->addChild(n);
			loaded = true;
			update = true;
		} catch (exception e) {
			cout << "Failed to load " << filepath << ": " << e.what() << endl;
		}
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
	if (!mResources) {
		mResources = make_shared<RenderResources>(*this, commandBuffer, prevFrame);
		mResourcePool.emplace(mResources);
	}
}

Scene::RenderResources::RenderResources(Scene& scene, CommandBuffer& commandBuffer, const shared_ptr<RenderResources>& prevFrame) :
	Device::Resource(commandBuffer.mDevice, "RenderResources"), mAccelerationStructure(nullptr) {

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
	if (prevFrame) {
		instanceDatas.reserve(prevFrame->mInstances.size());
		instanceTransforms.reserve(prevFrame->mInstances.size());
		instanceInverseTransforms.reserve(prevFrame->mInstances.size());
		instanceMotionTransforms.reserve(prevFrame->mInstances.size());
	}
	vector<uint32_t> lightInstanceMap;

	mInstanceIndexMap = make_shared<Buffer>(commandBuffer.mDevice, "InstanceIndexMap", sizeof(uint32_t) * max<size_t>(1, prevFrame ? prevFrame->mInstances.size() : 0), vk::BufferUsageFlagBits::eStorageBuffer);
	ranges::fill(mInstanceIndexMap, -1);

	vector<vk::AccelerationStructureInstanceKHR> instancesAS;
	vector<vk::BufferMemoryBarrier> blasBarriers;

	mMaterialCount = 0;
	mEmissivePrimitiveCount = 0;

	ByteAppendBuffer materialData;
	materialData.reserve(prevFrame && prevFrame->mMaterialData ? prevFrame->mMaterialData.size() / sizeof(uint32_t) : 1);
	unordered_map<const Material*, uint32_t> materialMap;

	// 'material' is either a Material or Medium
	auto appendMaterialData = [&](const auto* material) {
		// append unique materials to materials list
		auto materialMap_it = materialMap.find(reinterpret_cast<const Material*>(material));
		if (materialMap_it == materialMap.end()) {
			materialMap_it = materialMap.emplace(reinterpret_cast<const Material*>(material), materialData.sizeBytes()).first;

			material->store(materialData, mMaterialResources);
			mMaterialCount++;
		}
		return materialMap_it->second;
	};

	auto appendInstanceData = [&](const NodePtr& primNode, const void* primPtr, const InstanceData& instance, const TransformData& transform, const float emissivePower) {
		const uint32_t instanceIndex = (uint32_t)instanceDatas.size();
		instanceDatas.emplace_back(instance);
		mInstanceNodes.emplace_back(primNode);

		uint32_t light_index = INVALID_INSTANCE;
		if (emissivePower > 0) {
			light_index = (uint32_t)lightInstanceMap.size();
			BF_SET(instanceDatas[instanceIndex].packed[1], light_index, 0, 12);
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
		instanceMotionTransforms.emplace_back(make_instance_motion_transform(invTransform, prevTransform));
		return instanceIndex;
	};

	{ // mesh instances
		ProfilerScope s("Process mesh instances", &commandBuffer);
		scene.node()->forEachDescendant<MeshPrimitive>([&](const NodePtr& primNode, const shared_ptr<MeshPrimitive>& prim) {
			if (!prim->mMesh || !prim->mMaterial)
				return;
			if (prim->mMesh->topology() != vk::PrimitiveTopology::eTriangleList ||
				(prim->mMesh->indexType() != vk::IndexType::eUint32 && prim->mMesh->indexType() != vk::IndexType::eUint16) ||
				!prim->mMesh->vertices().find(Mesh::VertexAttributeType::ePosition) ||
				!prim->mMesh->vertices().find(Mesh::VertexAttributeType::eNormal) ||
				!prim->mMesh->vertices().find(Mesh::VertexAttributeType::eTexcoord)) {
				cout << "Skipping unsupported mesh in node " << primNode->name() << endl;
				return;
			}

			const uint32_t materialAddress = appendMaterialData(prim->mMaterial.get());

			// write vertexCopyDescriptors
			auto[positions, positionsDesc] = prim->mMesh->vertices().at(Mesh::VertexAttributeType::ePosition)[0];
			auto[normals, normalsDesc]     = prim->mMesh->vertices().at(Mesh::VertexAttributeType::eNormal)[0];
			auto[texcoords, texcoordsDesc] = prim->mMesh->vertices().at(Mesh::VertexAttributeType::eTexcoord)[0];
			const uint32_t index = (uint32_t)vertexCopyInfos.size();
			vertexCopyDescriptors[{ "gPositions", index }] = Buffer::View(positions, positionsDesc.mOffset);
			vertexCopyDescriptors[{ "gNormals"  , index }] = Buffer::View(normals, normalsDesc.mOffset);
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

				auto[as, asbuf] = buildAccelerationStructure(commandBuffer, primNode->name() + "/BLAS", vk::AccelerationStructureTypeKHR::eBottomLevel, triangleGeometry, range);
				blasBarriers.emplace_back(
					vk::AccessFlagBits::eAccelerationStructureWriteKHR, vk::AccessFlagBits::eAccelerationStructureReadKHR,
					VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
					**asbuf.buffer(), asbuf.offset(), asbuf.sizeBytes());
				it = scene.mMeshAccelerationStructures.emplace(key, make_pair(as, asbuf)).first;
			}

			const uint32_t triCount = prim->mMesh->indices().sizeBytes() / (prim->mMesh->indices().stride() * 3);
			const TransformData transform = nodeToWorld(*primNode);
			const float area = 1;

			if (prim->mMaterial->emission() > 0)
				mEmissivePrimitiveCount += triCount;

			vk::AccelerationStructureInstanceKHR& instance = instancesAS.emplace_back();
			float3x4::Map(&instance.transform.matrix[0][0]) = transform.to_float3x4();
			instance.instanceCustomIndex = appendInstanceData(primNode, prim.get(), make_instance_triangles(materialAddress, triCount, firstVertex, indexByteOffset, prim->mMesh->indices().stride()), transform, prim->mMaterial->emission() * area);
			instance.mask = BVH_FLAG_TRIANGLES;
			instance.accelerationStructureReference = commandBuffer.mDevice->getAccelerationStructureAddressKHR(**it->second.first);

			meshInstanceIndices.emplace_back(prim.get(), (uint32_t)instance.instanceCustomIndex);
		});
	}

	{ // sphere instances
		ProfilerScope s("Process sphere instances", &commandBuffer);
		scene.node()->forEachDescendant<SpherePrimitive>([&](const NodePtr& primNode, const shared_ptr<SpherePrimitive>& prim) {
			if (!prim->mMaterial) return;

			const uint32_t materialAddress = appendMaterialData(prim->mMaterial.get());

			TransformData transform = nodeToWorld(*primNode);
			const float radius = prim->mRadius * transform.m.block<3, 3>(0, 0).matrix().determinant();
			// remove scale/rotation from transform
			transform = TransformData(transform.m.col(3).head<3>(), quatf_identity(), float3::Ones());

			// get/build BLAS
			const float3 mn = -float3::Constant(radius);
			const float3 mx = float3::Constant(radius);
			const size_t key = hashArgs(mn[0], mn[1], mn[2], mx[0], mx[1], mx[2], prim->mMaterial->alphaTest());
			auto aabb_it = scene.mAABBs.find(key);
			if (aabb_it == scene.mAABBs.end()) {
				Buffer::View<vk::AabbPositionsKHR> aabb = make_shared<Buffer>(commandBuffer.mDevice, "aabb data", sizeof(vk::AabbPositionsKHR), vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
				aabb[0].minX = mn[0];
				aabb[0].minY = mn[1];
				aabb[0].minZ = mn[2];
				aabb[0].maxX = mx[0];
				aabb[0].maxY = mx[1];
				aabb[0].maxZ = mx[2];
				vk::AccelerationStructureGeometryAabbsDataKHR aabbs(aabb.deviceAddress(), sizeof(vk::AabbPositionsKHR));
				vk::AccelerationStructureGeometryKHR aabbGeometry(vk::GeometryTypeKHR::eAabbs, aabbs, prim->mMaterial->alphaTest() ? vk::GeometryFlagBitsKHR{} : vk::GeometryFlagBitsKHR::eOpaque);
				vk::AccelerationStructureBuildRangeInfoKHR range(1);

				auto [ as, asbuf ] = buildAccelerationStructure(commandBuffer, "aabb BLAS", vk::AccelerationStructureTypeKHR::eBottomLevel, aabbGeometry, range);
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
			instance.instanceCustomIndex = appendInstanceData(primNode, prim.get(), make_instance_sphere(materialAddress, radius), transform, prim->mMaterial->emission() * (4*M_PI*radius*radius));
			instance.mask = BVH_FLAG_SPHERES;
			instance.accelerationStructureReference = commandBuffer.mDevice->getAccelerationStructureAddressKHR(**aabb_it->second.first);
		});
	}

	{ // media
		ProfilerScope s("Process media", &commandBuffer);
		scene.node()->forEachDescendant<Medium>([&](const NodePtr& primNode, const shared_ptr<Medium>& vol) {
			if (!vol) return;

			const uint32_t materialAddress = appendMaterialData(vol.get());

			auto densityGrid = vol->mDensityGrid->grid<float>();

			// get/build BLAS
			const nanovdb::Vec3R& mn = densityGrid->worldBBox().min();
			const nanovdb::Vec3R& mx = densityGrid->worldBBox().max();
			const size_t key = hashArgs((float)mn[0], (float)mn[1], (float)mn[2], (float)mx[0], (float)mx[1], (float)mx[2]);
			auto aabb_it = scene.mAABBs.find(key);
			if (aabb_it == scene.mAABBs.end()) {
				Buffer::View<vk::AabbPositionsKHR> aabb = make_shared<Buffer>(commandBuffer.mDevice, "aabb data", sizeof(vk::AabbPositionsKHR), vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
				aabb[0].minX = (float)mn[0];
				aabb[0].minY = (float)mn[1];
				aabb[0].minZ = (float)mn[2];
				aabb[0].maxX = (float)mx[0];
				aabb[0].maxY = (float)mx[1];
				aabb[0].maxZ = (float)mx[2];
				vk::AccelerationStructureGeometryAabbsDataKHR aabbs(aabb.deviceAddress(), sizeof(vk::AabbPositionsKHR));
				vk::AccelerationStructureGeometryKHR aabbGeometry(vk::GeometryTypeKHR::eAabbs, aabbs, vk::GeometryFlagBitsKHR::eOpaque);
				vk::AccelerationStructureBuildRangeInfoKHR range(1);
				auto [ as, asbuf ] = buildAccelerationStructure(commandBuffer, "aabb BLAS", vk::AccelerationStructureTypeKHR::eBottomLevel, aabbGeometry, range);

				blasBarriers.emplace_back(
					vk::AccessFlagBits::eAccelerationStructureWriteKHR, vk::AccessFlagBits::eAccelerationStructureReadKHR,
					VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
					**asbuf.buffer(), asbuf.offset(), asbuf.sizeBytes());
				aabb_it = scene.mAABBs.emplace(key, make_pair(as, asbuf)).first;
			}

			// append to instance list
			const TransformData transform = nodeToWorld(*primNode);
			vk::AccelerationStructureInstanceKHR& instance = instancesAS.emplace_back();
			float3x4::Map(&instance.transform.matrix[0][0]) = transform.to_float3x4();
			instance.instanceCustomIndex = appendInstanceData(primNode, vol.get(), make_instance_volume(materialAddress, mMaterialResources.mVolumeDataMap.at(vol->mDensityBuffer)), transform, 0);
			instance.mask = BVH_FLAG_VOLUME;
			instance.accelerationStructureReference = commandBuffer.mDevice->getAccelerationStructureAddressKHR(**aabb_it->second.first);
		});
	}

	{ // Build TLAS
		ProfilerScope s("Build TLAS", &commandBuffer);
		commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, vk::DependencyFlagBits::eByRegion, {}, blasBarriers, {});

		vk::AccelerationStructureGeometryKHR geom{ vk::GeometryTypeKHR::eInstances, vk::AccelerationStructureGeometryInstancesDataKHR() };
		vk::AccelerationStructureBuildRangeInfoKHR range{ (uint32_t)instancesAS.size() };
		if (!instancesAS.empty()) {
			Buffer::View<vk::AccelerationStructureInstanceKHR> instanceBuf(make_shared<Buffer>(commandBuffer.mDevice, "TLAS instance buffer",
				sizeof(vk::AccelerationStructureInstanceKHR) * instancesAS.size(),
				vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress,
				vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent ));

			ranges::copy(instancesAS, instanceBuf.begin());

			geom.geometry.instances.data = instanceBuf.deviceAddress();
			commandBuffer.trackResource(instanceBuf.buffer());
		}
		tie(mAccelerationStructure, mAccelerationStructureBuffer) = buildAccelerationStructure(commandBuffer, scene.node()->name() + "/TLAS", vk::AccelerationStructureTypeKHR::eTopLevel, geom, range);
		mAccelerationStructureBuffer.barrier(commandBuffer,
			vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, vk::PipelineStageFlagBits::eComputeShader,
			vk::AccessFlagBits::eAccelerationStructureWriteKHR, vk::AccessFlagBits::eAccelerationStructureReadKHR);
	}

	{ // environment material
		ProfilerScope s("Process environment", &commandBuffer);
		mEnvironmentMaterialAddress = -1;
		scene.node()->forEachDescendant<Environment>([&](const NodePtr& node, const shared_ptr<Environment> environment) {
			if (environment->emission.value.isZero()) return true;
			mEnvironmentMaterialAddress = materialData.sizeBytes();
			mMaterialCount++;
			environment->store(materialData, mMaterialResources);
			return false;
		});
	}

	{ // pack mesh vertices/indices
		ProfilerScope s("Pack mesh vertices/indices", &commandBuffer);

		if (!mVertices || mVertices.size() < totalVertexCount)
			mVertices = make_shared<Buffer>(commandBuffer.mDevice, "gVertices", max(totalVertexCount, 1u) * sizeof(PackedVertexData), vk::BufferUsageFlagBits::eStorageBuffer);
		if (!mIndices || mIndices.size() < totalIndexBufferSize)
			mIndices = make_shared<Buffer>(commandBuffer.mDevice, "gIndices", max(totalIndexBufferSize, 4u), vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer);

		// TODO: buffer memory usage flags everywhere

		Buffer::View<uint4> infos = make_shared<Buffer>(commandBuffer.mDevice, "gInfos", vertexCopyInfos.size()*sizeof(uint4), vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible);
		ranges::copy(vertexCopyInfos, infos.begin());
		vertexCopyDescriptors[{ "gInfos", 0 }] = infos;

		// pack vertices


		auto copyPipeline = scene.mCopyVerticesPipeline.get(commandBuffer.mDevice);
		auto descriptors = copyPipeline->getDescriptorSets(vertexCopyDescriptors);
		commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, ***copyPipeline);
		descriptors->bind(commandBuffer, {});
		for (uint32_t i = 0; i < vertexCopyInfos.size(); i++) {
			copyPipeline->pushConstants(commandBuffer, PushConstants{ { "gBufferIndex", PushConstantValue(i) } });
			const vk::Extent3D dim = copyPipeline->dispatchDimensions(vk::Extent3D(vertexCopyInfos[i][0],1,1));
			commandBuffer->dispatch(dim.width, dim.height, dim.depth);
		}

		// copy indices

		for (const auto& [prim, instanceIndex] : meshInstanceIndices) {
			const InstanceData& instance = instanceDatas[instanceIndex];
			commandBuffer->copyBuffer(
				**prim->mMesh->indices().buffer(),
				**mIndices.buffer(),
				vk::BufferCopy(prim->mMesh->indices().offset(), instance.indices_byte_offset(), prim->mMesh->indices().sizeBytes()));
		}
		mIndices.barrier(commandBuffer, vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead);
	}

	{ // upload data
		ProfilerScope s("Upload scene data buffers");
		vector<future<void>> tasks;
		tasks.reserve(5);
		tasks.emplace_back( async(launch::async, [&](){ ranges::copy(instanceDatas, mInstances.begin()); }) );
		tasks.emplace_back( async(launch::async, [&](){ ranges::copy(instanceTransforms, mInstanceTransforms.begin()); }) );
		tasks.emplace_back( async(launch::async, [&](){ ranges::copy(instanceInverseTransforms, mInstanceInverseTransforms.begin()); }) );
		tasks.emplace_back( async(launch::async, [&](){ ranges::copy(instanceMotionTransforms, mInstanceMotionTransforms.begin()); }) );
		tasks.emplace_back( async(launch::async, [&](){ ranges::copy(materialData, mMaterialData.begin()); }) );
		if (lightInstanceMap.size())
			tasks.emplace_back( async(launch::async, [&](){ ranges::copy(lightInstanceMap, mLightInstanceMap.begin()); }) );
		for (auto& task : tasks)
			task.wait();
	}
}

}