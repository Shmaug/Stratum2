#pragma once

#include "VCM.hpp"
#include "TestRenderer.hpp"
#include "RasterRenderer.hpp"
#include "NonEuclidianRenderer.hpp"

namespace stm2 {

using Renderer = variant<
	shared_ptr<RasterRenderer>,
	shared_ptr<VCM>,
	shared_ptr<TestRenderer>,
	shared_ptr<NonEuclidianRenderer>>;

enum RendererType {
	eRaster,
	eVCM,
	eTest,
	eNonEuclidian,
	eRendererTypeCount
};

inline Renderer make_renderer(const RendererType type, Node& node) {
	switch (type) {
		default:
		case RendererType::eRaster:
			return node.makeComponent<RasterRenderer>(node);
		case RendererType::eVCM:
			return node.makeComponent<VCM>(node);
		case RendererType::eTest:
			return node.makeComponent<TestRenderer>(node);
		case RendererType::eNonEuclidian:
			return node.makeComponent<NonEuclidianRenderer>(node);
	}
}

inline static unordered_map<string, RendererType> StringToRendererTypeMap = {
	{ "Raster", RendererType::eRaster },
	{ "VCM", RendererType::eVCM },
	{ "Test", RendererType::eTest },
	{ "NonEuclidian", RendererType::eNonEuclidian } };

}

namespace std {
inline string to_string(stm2::RendererType type) {
	switch (type) {
		default:
		case stm2::RendererType::eRaster: return "Raster";
		case stm2::RendererType::eVCM: return "VCM";
		case stm2::RendererType::eTest: return "Test";
		case stm2::RendererType::eNonEuclidian: return "Non euclidian";
	}
}
}