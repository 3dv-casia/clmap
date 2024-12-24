#pragma once

#include <vector>

namespace limap {

namespace math {

// Determine median value in vector. Returns NaN for empty vectors.
template <typename T>
double Median(const std::vector<T>& elems);

// Determine mean value in a vector.
template <typename T>
double Mean(const std::vector<T>& elems);

////////////////////////////////////////////////////////////////////////////////
// Implement
////////////////////////////////////////////////////////////////////////////////

template <typename T>
double Median(const std::vector<T>& elems) {
  CHECK(!elems.empty());

  const size_t mid_idx = elems.size() / 2;

  std::vector<T> ordered_elems = elems;
  std::nth_element(ordered_elems.begin(), ordered_elems.begin() + mid_idx,
                   ordered_elems.end());

  if (elems.size() % 2 == 0) {
    const T mid_element1 = ordered_elems[mid_idx];
    const T mid_element2 = *std::max_element(ordered_elems.begin(),
                                             ordered_elems.begin() + mid_idx);
    return (mid_element1 + mid_element2) / 2.0;
  } else {
    return ordered_elems[mid_idx];
  }
}

template <typename T>
double Mean(const std::vector<T>& elems) {
  CHECK(!elems.empty());
  double sum = 0;
  for (const auto el : elems) {
    sum += static_cast<double>(el);
  }
  return sum / elems.size();
}
}  // namespace math

}  // namespace limap
