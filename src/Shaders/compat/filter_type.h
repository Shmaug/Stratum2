#pragma once

STM_NAMESPACE_BEGIN

enum class FilterKernelType {
  eAtrous,
  eBox3,
  eBox5,
  eSubsampled,
  eBox3Subsampled,
  eBox5Subsampled,
  eFilterKernelTypeCount
};

STM_NAMESPACE_END

#ifdef __cplusplus
namespace std {
inline string to_string(const stm2::FilterKernelType& t) {
  switch (t) {
    default: return "Unknown";
    case stm2::FilterKernelType::eAtrous:         return "Atrous";
    case stm2::FilterKernelType::eBox3:           return "3x3 Box";
    case stm2::FilterKernelType::eBox5:           return "5x5 Box";
    case stm2::FilterKernelType::eSubsampled:     return "Subsampled";
    case stm2::FilterKernelType::eBox3Subsampled: return "3x3 Box, then Subsampled";
    case stm2::FilterKernelType::eBox5Subsampled: return "5x5 Box, then Subsampled";
  }
}
}
#endif