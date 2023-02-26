#include <App/Scene.hpp>
#include <Core/Profiler.hpp>

#include "load_obj.hpp"
#include "load_serialized.hpp"

#include <pugixml.hpp>
#include <regex>

namespace stm2 {

vector<string> split_string(const string& str, const regex& delim_regex) {
	sregex_token_iterator first{ begin(str), end(str), delim_regex, -1 }, last;
	return vector<string>{ first, last };
}

float3 parse_vector3(const string& value) {
	vector<string> list = split_string(value, regex("(,| )+"));
	float3 v;
	if (list.size() == 1) {
		v[0] = stof(list[0]);
		v[1] = stof(list[0]);
		v[2] = stof(list[0]);
	} else if (list.size() == 3) {
		v[0] = stof(list[0]);
		v[1] = stof(list[1]);
		v[2] = stof(list[2]);
	} else {
		throw runtime_error("parse_vector3 failed");
	}
	return v;
}

float3 parse_srgb(const string& value) {
	float3 srgb;
	if (value.size() == 7 && value[0] == '#') {
		char* end_ptr = NULL;
		// parse hex code (#abcdef)
		int encoded = strtol(value.c_str() + 1, &end_ptr, 16);
		if (*end_ptr != '\0') {
			throw runtime_error("Invalid sRGB value: " + value);
		}
		srgb[0] = ((encoded & 0xFF0000) >> 16) / 255.0f;
		srgb[1] = ((encoded & 0x00FF00) >> 8) / 255.0f;
		srgb[2] = (encoded & 0x0000FF) / 255.0f;
	} else {
		throw runtime_error("Unsupported sRGB format: " + value);
	}
	return srgb;
}

vector<pair<float, float>> parse_spectrum(const string& value) {
	vector<string> list = split_string(value, regex("(,| )+"));
	vector<pair<float, float>> s;
	if (list.size() == 1 && list[0].find(":") == string::npos) {
		// a single uniform value for all wavelength
		s.push_back(make_pair(float(-1), stof(list[0])));
	} else {
		for (auto val_str : list) {
			vector<string> pair = split_string(val_str, regex(":"));
			if (pair.size() < 2) {
				throw runtime_error("parse_spectrum failed");
			}
			s.push_back(make_pair(float(stof(pair[0])), float(stof(pair[1]))));
		}
	}
	return s;
}

Eigen::Matrix4f parse_matrix4x4(const string& value) {
	vector<string> list = split_string(value, regex("(,| )+"));
	if (list.size() != 16)
		throw runtime_error("parse_matrix4x4 failed");

	Eigen::Matrix4f m;
	int k = 0;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			m(i, j) = stof(list[k++]);
	return m;
}

TransformData parse_transform(pugi::xml_node node) {
	TransformData t;
	for (auto child : node.children()) {
		string name = child.name();
		for (char& c : name) c = tolower(c);

		if (name == "scale") {
			float x = 1;
			float y = 1;
			float z = 1;
			if (!child.attribute("x").empty())
				x = stof(child.attribute("x").value());
			if (!child.attribute("y").empty())
				y = stof(child.attribute("y").value());
			if (!child.attribute("z").empty())
				z = stof(child.attribute("z").value());
			if (!child.attribute("value").empty())
				x = y = z = stof(child.attribute("value").value());
			t = tmul(TransformData(float3::Zero(), quatf::identity(), float3(x, y, z)), t);
		} else if (name == "translate") {
			float x = 0;
			float y = 0;
			float z = 0;
			if (!child.attribute("x").empty())
				x = stof(child.attribute("x").value());
			if (!child.attribute("y").empty())
				y = stof(child.attribute("y").value());
			if (!child.attribute("z").empty())
				z = stof(child.attribute("z").value());
			t = tmul(TransformData(float3(x, y, z), quatf::identity(), float3::Ones()), t);
		} else if (name == "rotate") {
			float x = 0;
			float y = 0;
			float z = 0;
			float angle = 0;
			if (!child.attribute("x").empty())
				x = stof(child.attribute("x").value());
			if (!child.attribute("y").empty())
				y = stof(child.attribute("y").value());
			if (!child.attribute("z").empty())
				z = stof(child.attribute("z").value());
			if (!child.attribute("angle").empty())
				angle = radians(stof(child.attribute("angle").value()));
			t = tmul(TransformData(float3::Zero(), quatf::angleAxis(angle, float3(x, y, z)), float3::Ones()), t);
		} else if (name == "lookat") {
			const float3 pos    = parse_vector3(child.attribute("origin").value());
			const float3 target = parse_vector3(child.attribute("target").value());
			float3 up = parse_vector3(child.attribute("up").value());

			const float3 fwd = (target - pos).matrix().normalized();
			up = (up - dot(up, fwd) * fwd).matrix().normalized();
			const float3 r = normalize(cross(up, fwd));
			t = tmul(TransformData(pos, quatf(r, up, fwd), float3::Ones()), t);
		} else if (name == "matrix") {
			t = tmul(TransformData(parse_matrix4x4(string(child.attribute("value").value())).block<3,4>(0,0)), t);
		}
	}
	return t;
}

float3 XYZintegral_coeff(const float wavelength) {
	// To support spectral data, we need to convert spectral measurements (how much energy at each wavelength) to
	// RGB. To do this, we first convert the spectral data to CIE XYZ, by
	// integrating over the XYZ response curve. Here we use an analytical response
	// curve proposed by Wyman et al.: https://jcgt.org/published/0002/02/01/
	float3 rgb;
	{
		const float t1 = (wavelength - 442.0f) * ((wavelength < 442.0f) ? 0.0624f : 0.0374f);
		const float t2 = (wavelength - 599.8f) * ((wavelength < 599.8f) ? 0.0264f : 0.0323f);
		const float t3 = (wavelength - 501.1f) * ((wavelength < 501.1f) ? 0.0490f : 0.0382f);
		rgb[0] = 0.362f * exp(-0.5f * t1 * t1) + 1.056f * exp(-0.5f * t2 * t2) - 0.065f * exp(-0.5f * t3 * t3);
	}
	{
		const float t1 = (wavelength - 568.8) * ((wavelength < 568.8) ? 0.0213 : 0.0247f);
		const float t2 = (wavelength - 530.9) * ((wavelength < 530.9) ? 0.0613 : 0.0322f);
		rgb[1] = 0.821f * exp(-0.5f * t1 * t1) + 0.286f * exp(-0.5f * t2 * t2);
	}
	{
		const float t1 = (wavelength - 437.0) * ((wavelength < 437.0) ? 0.0845 : 0.0278f);
		const float t2 = (wavelength - 459.0) * ((wavelength < 459.0) ? 0.0385 : 0.0725f);
		rgb[2] = 1.217f * exp(-0.5f * t1 * t1) +0.681f * exp(-0.5f * t2 * t2);
	}
	return rgb;
}

float3 integrate_XYZ(const std::vector<std::pair<float, float>>& data) {
	static const float CIE_Y_integral = 106.856895f;
	static const float wavelength_beg = 400;
	static const float wavelength_end = 700;
	if (data.size() == 0) {
		return float3::Zero();
	}
	float3 ret = float3::Zero();
	int data_pos = 0;
	// integrate from wavelength 400 nm to 700 nm, increment by 1nm at a time
	// linearly interpolate from the data
	for (float wavelength = wavelength_beg; wavelength <= wavelength_end; wavelength += float(1)) {
		// assume the spectrum data is sorted by wavelength
		// move data_pos such that wavelength is between two data or at one end
		while(data_pos < (int)data.size() - 1 && !((data[data_pos].first <= wavelength && data[data_pos + 1].first > wavelength) || data[0].first > wavelength)) {
			data_pos += 1;
		}
		float measurement = 0;
		if (data_pos < (int)data.size() - 1 && data[0].first <= wavelength) {
			const float curr_data = data[data_pos].second;
			const float next_data = data[std::min(data_pos + 1, (int)data.size() - 1)].second;
			const float curr_wave = data[data_pos].first;
			const float next_wave = data[std::min(data_pos + 1, (int)data.size() - 1)].first;
			// linearly interpolate
			measurement = curr_data * (next_wave - wavelength) / (next_wave - curr_wave) +
										next_data * (wavelength - curr_wave) / (next_wave - curr_wave);
		} else {
			// assign the endpoint
			measurement = data[data_pos].second;
		}
		const float3 coeff = XYZintegral_coeff(wavelength);
		ret += coeff * measurement;
	}
	const float wavelength_span = wavelength_end - wavelength_beg;
	ret *= (wavelength_span / (CIE_Y_integral * (wavelength_end - wavelength_beg)));
	return ret;
}

float3 parse_color(pugi::xml_node node) {
	string type = node.name();
	if (type == "spectrum") {
		vector<pair<float, float>> spec = parse_spectrum(node.attribute("value").value());
		if (spec.size() > 1) {
			return xyzToRgb(integrate_XYZ(spec));
		} else if (spec.size() == 1) {
			return float3::Ones();
		} else {
			return float3::Zero();
		}
	} else if (type == "rgb") {
		return parse_vector3(node.attribute("value").value());
	} else if (type == "srgb") {
		float3 srgb = parse_srgb(node.attribute("value").value());
		return srgbToRgb(srgb);
	} else if (type == "float") {
		return float3::Constant(stof(node.attribute("value").value()));
	} else {
		throw runtime_error("Unsupported color type: " + type);
		return float3::Zero();
	}
}


Mesh create_mesh(CommandBuffer& commandBuffer, const vector<float3>& vertices, const vector<float3>& normals, const vector<float2>& uvs, const vector<uint32_t>& indices) {
	vk::BufferUsageFlags bufferUsage = vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer;
	if (commandBuffer.mDevice.accelerationStructureFeatures().accelerationStructure) {
		bufferUsage |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
		bufferUsage |= vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
	}

	// create staging buffers
	Buffer::View<float3> positions_tmp = make_shared<Buffer>(commandBuffer.mDevice, "positions_tmp", vertices.size() * sizeof(float3), vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
	Buffer::View<float3> normals_tmp   = make_shared<Buffer>(commandBuffer.mDevice, "normals_tmp", normals.size() * sizeof(float3)   , vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
	Buffer::View<float2> texcoords_tmp = make_shared<Buffer>(commandBuffer.mDevice, "texcoords_tmp", uvs.size() * sizeof(float2)     , vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
	Buffer::View<uint32_t> indices_tmp = make_shared<Buffer>(commandBuffer.mDevice, "tmp indices", indices.size() * sizeof(uint32_t) , vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);

	// compute aabb
	float3 vmin = float3::Constant(numeric_limits<float>::infinity());
	float3 vmax = float3::Constant(-numeric_limits<float>::infinity());
	for (const float3& p : vertices) {
		vmin = min(vmin, p);
		vmax = min(vmax, p);
	}

	// copy vertex data to staging buffers
	memcpy(positions_tmp.data(), vertices.data(), positions_tmp.sizeBytes());
	memcpy(normals_tmp.data(), vertices.data(), normals_tmp.sizeBytes());
	memcpy(texcoords_tmp.data(), vertices.data(), texcoords_tmp.sizeBytes());
	memcpy(indices_tmp.data(), indices.data(), indices_tmp.sizeBytes());

	// copy vertex data to gpu buffers
	Buffer::View<float3> positions_buf = make_shared<Buffer>(commandBuffer.mDevice, "positions", positions_tmp.sizeBytes(), bufferUsage|vk::BufferUsageFlagBits::eVertexBuffer);
	Buffer::View<float3> normals_buf   = make_shared<Buffer>(commandBuffer.mDevice, "normals"  , normals_tmp.sizeBytes()  , bufferUsage|vk::BufferUsageFlagBits::eVertexBuffer);
	Buffer::View<float2> texcoords_buf = make_shared<Buffer>(commandBuffer.mDevice, "texcoords", texcoords_tmp.sizeBytes(), bufferUsage|vk::BufferUsageFlagBits::eVertexBuffer);
	Buffer::View<uint32_t> indices_buf = make_shared<Buffer>(commandBuffer.mDevice, "indices"  , indices_tmp.sizeBytes()  , bufferUsage|vk::BufferUsageFlagBits::eIndexBuffer);
	Buffer::copy(commandBuffer, positions_tmp, positions_buf);
	Buffer::copy(commandBuffer, normals_tmp, normals_buf);
	Buffer::copy(commandBuffer, texcoords_tmp, texcoords_buf);
	Buffer::copy(commandBuffer, indices_tmp, indices_buf);

	// construct mesh object
	Mesh::Vertices vertexArray;
	vertexArray[Mesh::VertexAttributeType::ePosition].emplace_back(positions_buf, Mesh::VertexAttributeDescription{ (uint32_t)sizeof(float3), vk::Format::eR32G32B32Sfloat, 0, vk::VertexInputRate::eVertex });
	vertexArray[Mesh::VertexAttributeType::eNormal].emplace_back(normals_buf, Mesh::VertexAttributeDescription{ (uint32_t)sizeof(float3), vk::Format::eR32G32B32Sfloat, 0, vk::VertexInputRate::eVertex });
	vertexArray[Mesh::VertexAttributeType::eTexcoord].emplace_back(texcoords_buf, Mesh::VertexAttributeDescription{ (uint32_t)sizeof(float2), vk::Format::eR32G32Sfloat, 0, vk::VertexInputRate::eVertex });
	vertexArray.mAabb = vk::AabbPositionsKHR(vmin[0], vmin[1], vmin[2], vmax[0], vmax[1], vmax[2]);
	return Mesh(vertexArray, indices_buf, vk::PrimitiveTopology::eTriangleList);
}

// parse "texture" node
Image::View parse_texture(CommandBuffer& commandBuffer, pugi::xml_node node) {
	const string type = node.attribute("type").value();

	filesystem::path filename;
	float3 color0 = float3::Constant(0.4f);
	float3 color1 = float3::Constant(0.2f);
	float uscale = 1;
	float vscale = 1;
	float uoffset = 0;
	float voffset = 0;

	for (auto child : node.children()) {
		string name = child.attribute("name").value();
		if (name == "filename") {
			filename = child.attribute("value").value();
		} else if (name == "color0") {
			color0 = parse_color(child);
		} else if (name == "color1") {
			color1 = parse_color(child);
		} else if (name == "uvscale") {
			uscale = vscale = stof(child.attribute("value").value());
		} else if (name == "uscale") {
			uscale = stof(child.attribute("value").value());
		} else if (name == "vscale") {
			vscale = stof(child.attribute("value").value());
		} else if (name == "uoffset") {
			uoffset = stof(child.attribute("value").value());
		} else if (name == "voffset") {
			voffset = stof(child.attribute("value").value());
		}
	}

	if (type == "bitmap") {
		Image::Metadata metadata;
		const Image::PixelData pixels = Image::loadFile(commandBuffer.mDevice, filename, 0, 4);
		metadata.mExtent = get<vk::Extent3D>(pixels);
		metadata.mFormat = get<vk::Format>(pixels);
		metadata.mUsage = vk::ImageUsageFlagBits::eTransferDst|vk::ImageUsageFlagBits::eTransferSrc|vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage;
		auto img = make_shared<Image>(commandBuffer.mDevice, filename.stem().string(), metadata);
		commandBuffer.trackResource(img);
		img->upload(commandBuffer, pixels);
		img->generateMipMaps(commandBuffer);
		return img;
	} else if (type == "checkerboard") {
		Image::Metadata metadata;
		metadata.mExtent = vk::Extent3D(512, 512, 1);
		metadata.mFormat = vk::Format::eR8G8B8A8Unorm;
		metadata.mUsage = vk::ImageUsageFlagBits::eTransferDst|vk::ImageUsageFlagBits::eTransferSrc|vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage;

		Buffer::View<byte> buf = make_shared<Buffer>(commandBuffer.mDevice, "checkerboard pixels", metadata.mExtent.width*metadata.mExtent.height*4, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);

		for (uint32_t y = 0; y < metadata.mExtent.height; y++)
			for (uint32_t x = 0; x < metadata.mExtent.width; x++) {
				const float u = uoffset + uscale * (x + 0.5f) / (float)metadata.mExtent.width;
				const float v = voffset + vscale * (y + 0.5f) / (float)metadata.mExtent.height;
				const float3 c = fmodf(fmodf(floorf(u/2), 2.f) + fmodf(floorf(v/2), 2.f), 2.f) < 1 ? color0 : color1;
				const size_t addr = 4*(y*metadata.mExtent.width + x);
				buf[addr+0] = (byte)(c[0] * 0xFF);
				buf[addr+1] = (byte)(c[1] * 0xFF);
				buf[addr+2] = (byte)(c[2] * 0xFF);
				buf[addr+3] = (byte)(0xFF);
			}

		auto img = make_shared<Image>(commandBuffer.mDevice, "checkerboard", metadata);
		commandBuffer.trackResource(img);
		img->upload(commandBuffer, Image::PixelData { buf.buffer(), metadata.mFormat, metadata.mExtent });
		img->generateMipMaps(commandBuffer);
		return img;
	}
	throw runtime_error("Unsupported texture type: " + type + " for " + node.attribute("name").value());
}

ImageValue<3> parse_spectrum_texture(CommandBuffer& commandBuffer, pugi::xml_node node, unordered_map<string /* name id */, Image::View>& texture_map) {
	const string type = node.name();

	if (type == "spectrum") {
		vector<pair<float, float>> spec =
			parse_spectrum(node.attribute("value").value());
		if (spec.size() > 1) {
			float3 xyz = integrate_XYZ(spec);
			return ImageValue<3>{xyzToRgb(xyz), {}};
		} else if (spec.size() == 1) {
			return ImageValue<3>{float3::Ones(), {}};
		} else {
			return ImageValue<3>{float3::Zero(), {}};
		}
	} else if (type == "rgb") {
		return ImageValue<3>{parse_vector3(node.attribute("value").value()), {}};
	} else if (type == "srgb") {
		float3 srgb = parse_srgb(node.attribute("value").value());
		return ImageValue<3>{srgbToRgb(srgb), {}};
	} else if (type == "ref") {
		// referencing a texture
		string ref_id = node.attribute("id").value();
		auto t_it = texture_map.find(ref_id);
		if (t_it == texture_map.end()) {
			throw runtime_error("Texture not found: " + ref_id);
		}
		return ImageValue<3>{float3::Ones(), t_it->second};
	} else if (type == "texture") {
		Image::View t = parse_texture(commandBuffer, node);
		if (!node.attribute("id").empty()) {
			string id = node.attribute("id").value();
			if (texture_map.find(id) != texture_map.end()) throw runtime_error("Duplicate texture ID: " + id);
			texture_map.emplace(id, t);
		}
		return ImageValue<3>{float3::Ones(), t};
	}

	throw runtime_error("Unsupported spectrum texture type: " + type);
}

ImageValue<1> parse_float_texture(CommandBuffer& commandBuffer, pugi::xml_node node, unordered_map<string /* name id */, Image::View>& texture_map) {
	const string type = node.name();

	if (type == "ref") {
		// referencing a texture
		string ref_id = node.attribute("id").value();
		auto t_it = texture_map.find(ref_id);
		if (t_it == texture_map.end()) throw runtime_error("Texture not found: " + ref_id);
		return { float1::Ones(), t_it->second };
	} else if (type == "float") {
		return { float1::Constant(stof(node.attribute("value").value())), {} };
	} else if (type == "texture") {
		Image::View t = parse_texture(commandBuffer, node);
		if (!node.attribute("id").empty()) {
			string id = node.attribute("id").value();
			if (texture_map.find(id) != texture_map.end()) throw runtime_error("Duplicate texture ID: " + id);
			texture_map.emplace(id, t);
		}
		return { float1::Ones(), t };
	}

	throw runtime_error("Unsupported float texture type: " + type);
}

shared_ptr<Material> parse_bsdf(Scene& scene, Node& dst, CommandBuffer& commandBuffer, pugi::xml_node node, unordered_map<string /* name id */, shared_ptr<Material>>& material_map, unordered_map<string /* name id */, Image::View>& texture_map) {
	string type = node.attribute("type").value();

	unordered_set<string> ids;
	if (!node.attribute("id").empty()) ids.emplace(node.attribute("id").value());
	while (type == "twosided" || type == "bumpmap" || type == "mask") {
		if (node.child("bsdf").empty()) throw runtime_error(type + " has no child BSDF");
		node = node.child("bsdf");
		type = node.attribute("type").value();
		if (!node.attribute("id").empty()) ids.emplace(node.attribute("id").value());
	}

	string name = !ids.empty() ? *ids.begin() : (type + " BSDF");

	if (type == "diffuse") {
		ImageValue<3> diffuse { float3::Constant(0.5f) };
		for (auto child : node.children()) {
			string name = child.attribute("name").value();
			if (name == "reflectance")
				diffuse = parse_spectrum_texture(commandBuffer, child, texture_map);
		}
		auto m = dst.addChild(name)->makeComponent<Material>();
		m->mMaterialData.setBaseColor(diffuse.mValue);
		m->mImages[0] = diffuse.mImage;
		m->mMaterialData.setMetallic(0);
		m->mMaterialData.setEta(1.5f);
		for (const string& id : ids)
			if (!id.empty()) material_map[id] = m;
		return m;
	} else if (type == "roughplastic" || type == "plastic" || type == "conductor" || type == "roughconductor") {
		ImageValue<3> diffuse  { float3::Ones(), {} };
		ImageValue<3> specular { float3::Zero(), {} };
		ImageValue<1> roughness { float1::Constant((type == "plastic" || type == "conductor") ? 0 : 0.1f), {} };

		float intIOR = 1.49;
		float extIOR = 1.000277;
		float eta = intIOR / extIOR;
		for (auto child : node.children()) {
			string name = child.attribute("name").value();
			if (name == "diffuseReflectance") {
				diffuse = parse_spectrum_texture(commandBuffer, child, texture_map);
			} else if (name == "specularReflectance") {
				specular = parse_spectrum_texture(commandBuffer, child, texture_map);
			} else if (name == "alpha") {
				// Alpha requires special treatment since we need to convert
				// the values to roughness
				string type = child.name();
				if (type == "ref") {
					// referencing a texture
					string ref_id = child.attribute("id").value();
					auto t_it = texture_map.find(ref_id);
					if (t_it == texture_map.end()) throw runtime_error("Texture not found: " + ref_id);
					roughness = scene.alphaToRoughness(commandBuffer, { float1::Ones(), t_it->second });
				} else if (type == "float") {
					float alpha = stof(child.attribute("value").value());
					roughness.mValue = sqrt(alpha);
				} else
					throw runtime_error("Unsupported float texture type: " + type);
			} else if (name == "roughness") {
				roughness = parse_float_texture(commandBuffer, child, texture_map);
			} else if (name == "intIOR") {
				intIOR = stof(child.attribute("value").value());
				eta = intIOR / extIOR;
			} else if (name == "extIOR") {
				extIOR = stof(child.attribute("value").value());
				eta = intIOR / extIOR;
			}
		}
		auto m = dst.addChild(name)->makeComponent<Material>(scene.makeDiffuseSpecularMaterial(commandBuffer, diffuse, specular, roughness, { float3::Zero(), {} }, eta, { float3::Zero(), {} }));
		for (const string& id : ids)
			if (!id.empty()) material_map[id] = m;
		return m;
	} else if (type == "roughdielectric" || type == "dielectric" || type == "thindielectric") {
		ImageValue<3> diffuse       { float3::Zero(), {} };
		ImageValue<3> specular      { float3::Zero(), {} };
		ImageValue<3> transmittance { float3::Ones(), {} };
		ImageValue<1> roughness     { float1::Constant((type == "dielectric") ? 0 : 0.1f), {} };
		float intIOR = 1.5046;
		float extIOR = 1.000277;
		float eta = intIOR / extIOR;
		for (auto child : node.children()) {
			string name = child.attribute("name").value();
			if (name == "specularReflectance") {
				specular = parse_spectrum_texture(commandBuffer, child, texture_map);
			} else if (name == "specularTransmittance") {
				transmittance = parse_spectrum_texture(commandBuffer, child, texture_map);
			} else if (name == "alpha") {
				string type = child.name();
				if (type == "ref") {
					// referencing a texture
					string ref_id = child.attribute("id").value();
					auto t_it = texture_map.find(ref_id);
					if (t_it == texture_map.end()) throw runtime_error("Texture not found: " + ref_id);
					roughness.mImage = scene.alphaToRoughness(commandBuffer, { float1::Ones(), t_it->second }).mImage;
				} else if (type == "float") {
					roughness.mValue = sqrt(stof(child.attribute("value").value()));
				} else
					throw runtime_error("Unsupported float texture type: " + type);
			} else if (name == "roughness") {
				roughness = parse_float_texture(commandBuffer, child, texture_map);
			} else if (name == "intIOR") {
				intIOR = stof(child.attribute("value").value());
				eta = intIOR / extIOR;
			} else if (name == "extIOR") {
				extIOR = stof(child.attribute("value").value());
				eta = intIOR / extIOR;
			}
		}
		auto m = dst.addChild(name)->makeComponent<Material>(scene.makeDiffuseSpecularMaterial(commandBuffer, diffuse, specular, roughness, transmittance, eta, { float3::Zero(), {} }));
		for (const string& id : ids)
			if (!id.empty()) material_map[id] = m;
		return m;
	}

	string idstr;
	for (const string& id : ids)
		idstr += id + " ";

	cerr << "Unsupported BSDF type: \"" + type + "\" with IDs " + idstr << endl;
	throw runtime_error("Unsupported BSDF type: \"" + type + "\" with IDs " + idstr);
}

void parse_shape(Scene& scene, CommandBuffer& commandBuffer, Node& dst, pugi::xml_node node, unordered_map<string, shared_ptr<Material>>& material_map, unordered_map<string, Image::View>& texture_map, unordered_map<string, shared_ptr<Mesh>>& obj_map, unordered_map<pair<string,uint32_t>, shared_ptr<Mesh>>& serialized_map) {
	shared_ptr<Material> material;
	string filename;
	int shape_index = -1;

	// parse children first, to load material

	for (auto child : node.children()) {
		const string name = child.name();
		if (name == "ref") {
			const string name_value = child.attribute("name").value();
			pugi::xml_attribute id = child.attribute("id");
			if (id.empty()) throw runtime_error("Material reference id not specified.");
			auto it = material_map.find(id.value());
			if (it == material_map.end()) throw runtime_error("Material reference " + string(id.value()) + " not found.");
			if (!material)
				material = it->second;
		} else if (name == "bsdf") {
			// store previously-parsed emission
			optional<float3> emission;
			if (material) emission = material->mMaterialData.getEmission();
			// parse material
			material = parse_bsdf(scene, dst, commandBuffer, child, material_map, texture_map);
			// override emission with previously parsed emission
			if (emission) material->mMaterialData.setEmission(*emission);
		} else if (name == "emitter") {
			// make emission material
			float3 radiance = float3::Ones();
			for (auto grand_child : child.children()) {
				string name = grand_child.attribute("name").value();
				if (name == "radiance") {
					string rad_type = grand_child.name();
					if (rad_type == "spectrum") {
						vector<pair<float, float>> spec = parse_spectrum(grand_child.attribute("value").value());
						if (spec.size() == 1) {
							// For a light source, the white point is
							// XYZ(0.9505, 1.0, 1.0888) instead
							// or XYZ(1, 1, 1). We need to handle this special case when
							// we don't have the full spectrum data.
							const float3 xyz = float3(0.9505f, 1.0f, 1.0888f);
							radiance = xyzToRgb(xyz * spec[0].second);
						} else {
							const float3 xyz = integrate_XYZ(spec);
							radiance = xyzToRgb(xyz);
						}
					} else if (rad_type == "rgb") {
						radiance = parse_vector3(grand_child.attribute("value").value());
					} else if (rad_type == "srgb") {
						const string value = grand_child.attribute("value").value();
						const float3 srgb = parse_srgb(value);
						radiance = srgbToRgb(srgb);
					}
				}
			}
			if (material) {
				material->mMaterialData.setEmission(radiance);
			} else {
				material = dst.makeComponent<Material>();
				material->mMaterialData.setBaseColor(float3::Zero());
				material->mMaterialData.setEmission(radiance);
			}
		}

		const string name_attrib = child.attribute("name").value();
		if (name == "string" && name_attrib == "filename") {
			filename = child.attribute("value").value();
		} else if (name == "transform" && name_attrib == "toWorld") {
			dst.makeComponent<TransformData>(parse_transform(child));
		} if (name == "integer" && name_attrib == "shapeIndex") {
			shape_index = stoi(child.attribute("value").value());
		}
	}

	string type = node.attribute("type").value();
	if (type == "obj") {
		shared_ptr<Mesh> m;
		if (auto m_it = obj_map.find(filename); m_it != obj_map.end()) {
			m = m_it->second;
		} else {
			m = dst.makeComponent<Mesh>(loadObj(commandBuffer, filename));
			obj_map.emplace(filename, m);
		}
		dst.makeComponent<MeshPrimitive>(material, m);
	} else if (type == "serialized") {
		const auto key = make_pair(filename, shape_index);
		shared_ptr<Mesh> m;
		if (auto m_it = serialized_map.find(key); m_it != serialized_map.end()) {
			m = m_it->second;
		} else {
			m = dst.makeComponent<Mesh>(loadSerialized(commandBuffer, filename, shape_index));
			serialized_map.emplace(key, m);
		}
		dst.makeComponent<MeshPrimitive>(material, m);
	} else if (type == "sphere") {
		optional<float3> center;
		float radius = 1;
		for (auto child : node.children()) {
			const string name = child.attribute("name").value();
			if (name == "center") {
				center = float3{
						stof(child.attribute("x").value()),
						stof(child.attribute("y").value()),
						stof(child.attribute("z").value()) };
			} else if (name == "radius") {
				radius = stof(child.attribute("value").value());
			}
		}
		if (center) {
			if (auto ptr = dst.getComponent<TransformData>())
				*ptr = tmul(*ptr, TransformData(*center, quatf::identity(), float3::Ones()));
			else
				dst.makeComponent<TransformData>(*center, quatf::identity(), float3::Ones());
		}
		dst.makeComponent<SpherePrimitive>(material, radius);
	} else if (type == "rectangle") {
		const vector<float3> vertices = { float3(-1,-1,0), float3(-1,1,0), float3(1,-1,0), float3(1,1,0) };
		const vector<float3> normals = { float3(0,0,1), float3(0,0,1), float3(0,0,1), float3(0,0,1) };
		const vector<float2> uvs = { float2(0,0), float2(0,1), float2(1,0), float2(1,1) };
		const vector<uint32_t> indices = { 0, 1, 2, 1, 3, 2 };
		dst.makeComponent<MeshPrimitive>(material, dst.makeComponent<Mesh>(create_mesh(commandBuffer, vertices, normals, uvs, indices)));
	}/* else if (type == "cube") {
		vector<float3> vertices(16);
		vector<float3> normals(16);
		vector<float2> uvs(16);
		vector<uint32_t> indices(36);
		for (uint32_t face = 0; face < 6; face++) {
			const uint32_t i = face*4;
			const int s = face%2 == 0 ? 1 : -1;

			float3 n = float3::Zero();
			n[face/2] = s;

			float3 v0 = n;
			float3 v1 = n;
			float3 v2 = n;
			float3 v3 = n;
			v0[(face/2 + 1) % 3] = -s;
			v0[(face/2 + 2) % 3] = -s;
			v1[(face/2 + 1) % 3] = s;
			v1[(face/2 + 2) % 3] = -s;
			v2[(face/2 + 1) % 3] = -s;
			v2[(face/2 + 2) % 3] = s;
			v3[(face/2 + 1) % 3] = s;
			v3[(face/2 + 2) % 3] = s;

			vertices[i+0] = v0;
			vertices[i+1] = v1;
			vertices[i+2] = v2;
			vertices[i+3] = v3;
			normals[i+0] = n;
			normals[i+1] = n;
			normals[i+2] = n;
			normals[i+3] = n;
			uvs[i+0] = float2(0,0);
			uvs[i+1] = float2(1,0);
			uvs[i+2] = float2(0,1);
			uvs[i+3] = float2(1,1);

			indices[face*6 + 0] = i + 0;
			indices[face*6 + 1] = i + 1;
			indices[face*6 + 2] = i + 2;
			indices[face*6 + 3] = i + 1;
			indices[face*6 + 4] = i + 3;
			indices[face*6 + 5] = i + 2;
		}
		dst.makeComponent<MeshPrimitive>(material, dst.makeComponent<Mesh>(create_mesh(commandBuffer, vertices, normals, uvs, indices)));
	}*/
	else throw runtime_error("Unsupported shape: " + type);
}

shared_ptr<Node> parse_scene(Scene& scene, CommandBuffer& commandBuffer, pugi::xml_node node) {
	unordered_map<string /* name id */, shared_ptr<Material>> material_map;
	unordered_map<string /* name id */, Image::View> texture_map;
	unordered_map<string /* filename */, shared_ptr<Mesh>> obj_map;
	unordered_map<pair<string /* filename */, uint32_t /* shape index */>, shared_ptr<Mesh>> serialized_map;

	int envmap_light_id = -1;

	const shared_ptr<Node> root = Node::create(node.name());

	for (auto child : node.children()) {
		string name = child.name();
		if (name == "bsdf") {
			parse_bsdf(scene, *root, commandBuffer, child, material_map, texture_map);
		} else if (name == "shape") {
			parse_shape(scene, commandBuffer, *root->addChild("shape"), child, material_map, texture_map, obj_map, serialized_map);
		} else if (name == "texture") {
			string id = child.attribute("id").value();
			if (texture_map.find(id) != texture_map.end()) throw runtime_error("Duplicate texture ID: " + id);
			texture_map[id] = parse_texture(commandBuffer, child);
		} else if (name == "emitter") {
			string type = child.attribute("type").value();
			if (type == "envmap") {
				optional<TransformData> transform;
				string filename;
				float scale = 1;
				for (auto grand_child : child.children()) {
					string name = grand_child.attribute("name").value();
					if (name == "filename") {
						filename = grand_child.attribute("value").value();
					} else if (name == "toWorld") {
						transform = parse_transform(grand_child);
					} else if (name == "scale") {
						scale = stof(grand_child.attribute("value").value());
					}
				}
				if (filename.size() > 0) {
					const shared_ptr<Node> envNode = scene.load(commandBuffer, filename);
					envNode->getComponent<EnvironmentMap>()->mValue *= scale;
					if (transform)
						envNode->makeComponent<TransformData>(*transform);
					root->addChild(envNode);
				} else {
					throw runtime_error("Filename unspecified for envmap.");
				}
			} else {
				throw runtime_error("Unsupported emitter type:" + type);
			}
		}
	}

	return root;
}

shared_ptr<Node> Scene::loadMitsuba(CommandBuffer& commandBuffer, const filesystem::path& filename) {
	ProfilerScope ps("Scene::loadMitsuba", &commandBuffer);

	pugi::xml_document doc;
	pugi::xml_parse_result result = doc.load_file(filename.c_str());
	if (!result) {
		cerr << "Error description: " << result.description() << endl;
		cerr << "Error offset: " << result.offset << endl;
		throw runtime_error("Parse error");
	}
	// back up the current working directory and switch to the parent folder of the file
	filesystem::path old_path = filesystem::current_path();
	filesystem::current_path(filename.parent_path());

	auto root = parse_scene(*this, commandBuffer, doc.child("scene"));

	// switch back to the old current working directory
	filesystem::current_path(old_path);

	cout << "Loaded " << filename << endl;

	return root;
}

}