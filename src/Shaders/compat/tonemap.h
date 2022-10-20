#ifndef TONEMAP_H
#define TONEMAP_H

#ifdef __cplusplus
namespace tinyvkpt {
#endif

enum class TonemapMode {
	eRaw,
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

static const unordered_set<TonemapMode> gTonemapModeNeedsMax = {
	TonemapMode::eViridisR,
	TonemapMode::eReinhardExtended,
	TonemapMode::eReinhardLuminanceExtended,
	TonemapMode::eUncharted2
};

}

namespace std {
inline string to_string(const tinyvkpt::TonemapMode& m) {
	switch (m) {
		default: return "Unknown";
		case tinyvkpt::TonemapMode::eRaw: return "Raw";
		case tinyvkpt::TonemapMode::eReinhard: return "Reinhard";
		case tinyvkpt::TonemapMode::eReinhardExtended: return "Reinhard Extended";
		case tinyvkpt::TonemapMode::eReinhardLuminance: return "Reinhard (Luminance)";
		case tinyvkpt::TonemapMode::eReinhardLuminanceExtended: return "Reinhard Extended (Luminance)";
		case tinyvkpt::TonemapMode::eUncharted2: return "Uncharted 2";
		case tinyvkpt::TonemapMode::eFilmic: return "Filmic";
		case tinyvkpt::TonemapMode::eACES: return "ACES";
		case tinyvkpt::TonemapMode::eACESApprox: return "ACES (approximated)";
		case tinyvkpt::TonemapMode::eViridisR: return "Viridis Colormap - R channel";
		case tinyvkpt::TonemapMode::eViridisLengthRGB: return "Viridis Colormap - length(RGB)";
	}
};
}
#endif

#endif