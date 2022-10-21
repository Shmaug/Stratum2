#ifndef MSE_H
#define MSE_H

#ifdef __cplusplus
namespace tinyvkpt {
#endif

enum class CompareMetric {
	eSMAPE,
	eMSE,
	eAverage,
	eCompareMetricCount
};

#ifdef __cplusplus
} // namespace tinyvkpt
namespace std {
inline string to_string(const tinyvkpt::CompareMetric& m) {
	switch (m) {
		default: return "Unknown";
		case tinyvkpt::CompareMetric::eSMAPE:   return "SMAPE";
		case tinyvkpt::CompareMetric::eMSE:     return "MSE";
		case tinyvkpt::CompareMetric::eAverage: return "Average";
	}
};
}
#endif

#endif