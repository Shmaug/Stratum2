#pragma once

#include "TestRenderer.hpp"
#include "RasterRenderer.hpp"
#include "NonEuclidianRenderer.hpp"

namespace stm2 {

using Renderer = variant<
	shared_ptr<TestRenderer>,
	shared_ptr<RasterRenderer>,
	shared_ptr<NonEuclidianRenderer>>;

enum RendererType {
	eTest,
	eRaster,
	eNonEuclidian,
	eRendererTypeCount
};

Renderer make_renderer(const RendererType type, Node& node) {
	switch (type) {
		default:
		case RendererType::eTest:
			return node.makeComponent<TestRenderer>(node);
		case RendererType::eRaster:
			return node.makeComponent<RasterRenderer>(node);
		case RendererType::eNonEuclidian:
			return node.makeComponent<NonEuclidianRenderer>(node);
	}
}

inline static unordered_map<string, RendererType> StringToRendererTypeMap = {
	{ "Test", RendererType::eTest },
	{ "Raster", RendererType::eRaster },
	{ "NonEuclidian", RendererType::eNonEuclidian } };

}

namespace std {
string to_string(stm2::RendererType type) {
	switch (type) {
		default:
		case stm2::RendererType::eTest: return "Test";
		case stm2::RendererType::eRaster: return "Raster";
		case stm2::RendererType::eNonEuclidian: return "Non euclidian";
	}
}
}