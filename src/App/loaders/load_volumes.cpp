#include "../Scene.hpp"
#include <Core/CommandBuffer.hpp>

#ifdef ENABLE_OPENVDB
#include <openvdb/openvdb.h>
#endif

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/GridBuilder.h>
#include <nanovdb/util/IO.h>
#ifdef ENABLE_OPENVDB
#include <nanovdb/util/OpenToNanoVDB.h>
#endif

namespace stm2 {

void createTransform(Node& dst, Medium& h) {
	const nanovdb::Vec3R bboxMax = h.mDensityGrid->grid<float>()->worldBBox().max();
	const nanovdb::Vec3R bboxMin = h.mDensityGrid->grid<float>()->worldBBox().min();
	const nanovdb::Vec3R center = (bboxMax + bboxMin) / 2;
	const nanovdb::Vec3R extent = bboxMax - bboxMin;
	const float scale = 1 / (float)(bboxMax - bboxMin).max();
	dst.makeComponent<TransformData>(
		-VectorType<nanovdb::Vec3R::ValueType, 3>::Map(&center[0]).cast<float>() * scale,
		quatf::identity(),
		float3::Constant(scale));
}

Medium createVolume(CommandBuffer& commandBuffer, const string& name, const shared_ptr<nanovdb::GridHandle<nanovdb::HostBuffer>>& density = {}, const shared_ptr<nanovdb::GridHandle<nanovdb::HostBuffer>>& albedo = {}) {
	Medium h;
	h.mDensityScale = float3::Ones();
	h.mAnisotropy = 0.f;
	h.mAlbedoScale = float3::Ones();
	h.mAttenuationUnit = 0.1f;
	h.mDensityGrid = density;
	h.mAlbedoGrid = albedo;
	if (density) {
		Buffer::View<byte> staging = make_shared<Buffer>(commandBuffer.mDevice, name + "/staging", density->size(), vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		memcpy(staging.data(), density->data(), density->size());
		h.mDensityBuffer = make_shared<Buffer>(commandBuffer.mDevice, name + "/density", density->size(), vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer);
		Buffer::copy(commandBuffer, staging, h.mDensityBuffer);
	}
	if (albedo) {
		Buffer::View<byte> staging = make_shared<Buffer>(commandBuffer.mDevice, name + "/staging", albedo->size(), vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		memcpy(staging.data(), albedo->data(), albedo->size());
		h.mAlbedoBuffer = make_shared<Buffer>(commandBuffer.mDevice, name + "/albedo", albedo->size(), vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer);
		Buffer::copy(commandBuffer, staging, h.mAlbedoBuffer);
	}
	return h;
}

nanovdb::GridHandle<nanovdb::HostBuffer> loadVol(const filesystem::path& filename) {
	// code from https://github.com/mitsuba-renderer/mitsuba/blob/master/src/volume/gridvolume.cpp#L217
	enum EVolumeType {
		EFloat32 = 1,
		EFloat16 = 2,
		EUInt8 = 3,
		EQuantizedDirections = 4
	};

	fstream fs(filename.c_str(), fstream::in | fstream::binary);
	char header[3];
	fs.read(header, 3);
	if (header[0] != 'V' || header[1] != 'O' || header[2] != 'L') throw runtime_error("Error loading volume from a file (incorrect header). Filename:" + filename.string());
	uint8_t version;
	fs.read((char*)&version, 1);
	if (version != 3) throw runtime_error("Error loading volume from a file (incorrect header). Filename:" + filename.string());

	int type;
	fs.read((char*)&type, sizeof(int));
	if (type != EFloat32) throw runtime_error("Unsupported volume format (only support Float32). Filename:" + filename.string());

	int xres, yres, zres;
	fs.read((char*)&xres, sizeof(int));
	fs.read((char*)&yres, sizeof(int));
	fs.read((char*)&zres, sizeof(int));

	int channels;
	fs.read((char*)&channels, sizeof(int));
	if (type != EFloat32) throw runtime_error("Unsupported volume format (not Float32). Filename:" + filename.string());

	float3 pmin, pmax;
	fs.read((char*)&pmin, sizeof(float3));
	fs.read((char*)&pmax, sizeof(float3));

	if (channels == 1) {
		vector<float> data(xres * yres * zres);
		fs.read((char*)data.data(), sizeof(float) * xres * yres * zres);
		nanovdb::GridBuilder<float> builder(0, nanovdb::GridClass::FogVolume);
		builder([&](const nanovdb::Coord& ijk) -> float {
			return data[(ijk[2] * yres + ijk[1]) * xres + ijk[0]];
			}, nanovdb::CoordBBox(nanovdb::Coord(0), nanovdb::Coord(xres, yres, zres) - nanovdb::Coord(1)));
		return builder.getHandle<>(1.0, nanovdb::Vec3d(0), filename.stem().string(), nanovdb::GridClass::FogVolume);
	} else
		throw runtime_error("Unsupported volume format (wrong number of channels). Filename:" + filename.string());
}

shared_ptr<Node> Scene::loadVol(CommandBuffer& commandBuffer, const filesystem::path& filename) {
	nanovdb::GridHandle<nanovdb::HostBuffer> densityHandle = stm2::loadVol(filename);
	if (densityHandle) {
		// create nvdb so the file can be loaded faster next time
		if (!filesystem::exists(filename.string() + ".nvdb"))
			nanovdb::io::writeGrid(filename.string() + ".nvdb", densityHandle);
		const shared_ptr<Node> node = Node::create(filename.stem().string());
		Medium& h = *node->makeComponent<Medium>(createVolume(commandBuffer, filename.stem().string(), node->makeComponent<nanovdb::GridHandle<nanovdb::HostBuffer>>(move(densityHandle))));
		createTransform(*node, h);
		return node;
	}
	return nullptr;
}

shared_ptr<Node> Scene::loadNvdb(CommandBuffer& commandBuffer, const filesystem::path& filename) {
	nanovdb::GridHandle<nanovdb::HostBuffer> densityHandle = nanovdb::io::readGrid(filename.string().c_str());
	if (densityHandle) {
		const shared_ptr<Node> node = Node::create(filename.stem().string());
		Medium& h = *node->makeComponent<Medium>(createVolume(commandBuffer, filename.stem().string(), node->makeComponent<nanovdb::GridHandle<nanovdb::HostBuffer>>(move(densityHandle))));
		createTransform(*node, h);
		return node;
	}
	return nullptr;
}

#ifdef ENABLE_OPENVDB
shared_ptr<Node> Scene::loadVdb(CommandBuffer& commandBuffer, const filesystem::path& filename) {
	static bool initialized = false;
	if (!initialized) {
		openvdb::initialize();
		initialized = true;
	}
	openvdb::io::File file(filename.string().c_str());
	if (!file.open()) return nullptr;

	unordered_map<string, shared_ptr<nanovdb::GridHandle<nanovdb::HostBuffer>>> grids;
	for (auto it = file.beginName(); name_it != file.endName(); ++it) {
		auto grid = root.addChild(*name_it).makeComponent<nanovdb::GridHandle<nanovdb::HostBuffer>>(nanovdb::openToNanoVDB(file.readGrid(*it)));
		grids.emplace(*it, grid);
		// cache NVDB
		if (!filesystem::exists(filename.string() + *it + ".nvdb"))
			nanovdb::io::writeGrid(filename.string() + *it + ".nvdb", *grid);
	}
	file.close();

	if (grids.empty()) return nullptr;

	if (auto grid = grids.begin()->second)
		return createVolume(commandBuffer, filename.stem().string(), grid);

	return nullptr;
}
#endif

}