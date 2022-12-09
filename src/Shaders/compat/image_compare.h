#ifndef IMAGE_COMPARE_H
#define IMAGE_COMPARE_H

#include "hlslcompat.h"

STM_NAMESPACE_BEGIN

enum class ImageCompareMode {
	eSMAPE = 0,
	eMSE,
	eAverage,
	eCompareMetricCount
};

STM_NAMESPACE_END

#ifdef __cplusplus
namespace std {
inline string to_string(const stm2::ImageCompareMode& m) {
	switch (m) {
		default: return "Unknown";
        case stm2::ImageCompareMode::eSMAPE:   return "SMAPE";
        case stm2::ImageCompareMode::eMSE:     return "MSE";
        case stm2::ImageCompareMode::eAverage: return "Average";
	}
};
}
#endif

#endif