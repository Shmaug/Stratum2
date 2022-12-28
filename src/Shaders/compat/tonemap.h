#pragma once

#include "hlslcompat.h"
#ifdef __cplusplus
#include <unordered_set>
#endif

STM_NAMESPACE_BEGIN

enum class TonemapMode {
	eRaw = 0,
	eReinhard,
	eReinhardExtended,
	eReinhardLuminance,
	eReinhardLuminanceExtended,
	eUncharted2,
	eFilmic,
	eACES,
	eACESApprox,
	eViridisR,
	eViridisLengthRGB,
	eTonemapModeCount
};

#ifdef __cplusplus
static const std::unordered_set<TonemapMode> gTonemapModeNeedsMax = {
	TonemapMode::eViridisR,
	TonemapMode::eReinhardExtended,
	TonemapMode::eReinhardLuminanceExtended,
	TonemapMode::eUncharted2
};
#endif

STM_NAMESPACE_END


#ifdef __cplusplus
namespace std {
inline string to_string(const stm2::TonemapMode& m) {
	switch (m) {
		default: return "Unknown";
		case stm2::TonemapMode::eRaw: return "Raw";
		case stm2::TonemapMode::eReinhard: return "Reinhard";
		case stm2::TonemapMode::eReinhardExtended: return "Reinhard Extended";
		case stm2::TonemapMode::eReinhardLuminance: return "Reinhard (Luminance)";
		case stm2::TonemapMode::eReinhardLuminanceExtended: return "Reinhard Extended (Luminance)";
		case stm2::TonemapMode::eUncharted2: return "Uncharted 2";
		case stm2::TonemapMode::eFilmic: return "Filmic";
		case stm2::TonemapMode::eACES: return "ACES";
		case stm2::TonemapMode::eACESApprox: return "ACES (approximated)";
		case stm2::TonemapMode::eViridisR: return "Viridis Colormap - R channel";
		case stm2::TonemapMode::eViridisLengthRGB: return "Viridis Colormap - length(RGB)";
	}
}
}
#endif