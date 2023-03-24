#pragma once

#include "RasterRenderer.hpp"
#include "ReSTIRPT.hpp"
#include "TestRenderer.hpp"
#include "VCM.hpp"

namespace stm2 {

using Renderer = variant<
	shared_ptr<RasterRenderer>,
	shared_ptr<ReSTIRPT>,
	shared_ptr<TestRenderer>,
	shared_ptr<VCM>
>;

enum RendererType {
	eRaster,
	eReSTIRPT,
	eTest,
	eVCM,
	eRendererTypeCount
};

inline Renderer make_renderer(const RendererType type, Node& node) {
	switch (type) {
		default:
		case RendererType::eRaster:   return node.makeComponent<RasterRenderer>(node);
		case RendererType::eReSTIRPT: return node.makeComponent<ReSTIRPT>(node);
		case RendererType::eTest:     return node.makeComponent<TestRenderer>(node);
		case RendererType::eVCM:      return node.makeComponent<VCM>(node);
	}
}

inline static unordered_map<string, RendererType> StringToRendererTypeMap = {
	{ "Raster",   RendererType::eRaster },
	{ "ReSTIRPT", RendererType::eReSTIRPT },
	{ "Test",     RendererType::eTest },
	{ "VCM",      RendererType::eVCM },
};

}

namespace std {
inline string to_string(stm2::RendererType type) {
	switch (type) {
		default:
		case stm2::RendererType::eRaster:   return "Raster";
		case stm2::RendererType::eReSTIRPT: return "ReSTIRPT";
		case stm2::RendererType::eTest:     return "Test";
		case stm2::RendererType::eVCM:      return "VCM";
	}
}
}