#ifndef FILTER_KERNEL_TYPE_H
#define FILTER_KERNEL_TYPE_H

#ifdef __cplusplus
namespace tinyvkpt {
#endif

enum class FilterKernelType {
  eAtrous,
  eBox3,
  eBox5,
  eSubsampled,
  eBox3Subsampled,
  eBox5Subsampled,
  eFilterKernelTypeCount
};

#ifdef __cplusplus
} // namespace tinyvkpt
namespace std {
inline string to_string(const tinyvkpt::FilterKernelType& t) {
  switch (t) {
    default: return "Unknown";
    case tinyvkpt::FilterKernelType::eAtrous:         return "Atrous";
    case tinyvkpt::FilterKernelType::eBox3:           return "3x3 Box";
    case tinyvkpt::FilterKernelType::eBox5:           return "5x5 Box";
    case tinyvkpt::FilterKernelType::eSubsampled:     return "Subsampled";
    case tinyvkpt::FilterKernelType::eBox3Subsampled: return "3x3 Box, then Subsampled";
    case tinyvkpt::FilterKernelType::eBox5Subsampled: return "5x5 Box, then Subsampled";
  }
}
}
#endif

#endif