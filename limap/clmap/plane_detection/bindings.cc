#include "clmap/plane_detection/robust_statistics_approach.h"

#include "_limap/helpers.h"

namespace py = pybind11;

namespace limap {

void bind_clmap_plane_detection(py::module& m) {
  m.def("RobustStatisticsApproach", &plane_detection::RobustStatisticsApproach,
        py::arg("inputFileName"), py::arg("outputFileName"),
        py::arg("minNormalDiff") = 0.5, py::arg("maxDist") = 0.258819,
        py::arg("outlierRatio") = 0.75, py::arg("normalsNeighborSize") = 30);
}

}  // namespace limap
