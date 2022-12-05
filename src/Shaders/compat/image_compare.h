#ifndef IMAGE_COMPARE_H
#define IMAGE_COMPARE_H

#include "hlslcompat.h"

STM_NAMESPACE_BEGIN

enum class CompareMetric {
	eSMAPE,
	eMSE,
	eAverage,
	eCompareMetricCount
};

STM_NAMESPACE_END

#ifdef __cplusplus
namespace std {
inline string to_string(const stm2::CompareMetric& m) {
	switch (m) {
		default: return "Unknown";
		case stm2::CompareMetric::eSMAPE:   return "SMAPE";
		case stm2::CompareMetric::eMSE:     return "MSE";
		case stm2::CompareMetric::eAverage: return "Average";
	}
};
}
#endif

#endif