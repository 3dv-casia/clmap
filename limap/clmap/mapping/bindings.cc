#include "clmap/mapping/line_mapper.h"
#include "clmap/mapping/merging.h"

#include <vector>

#include "_limap/helpers.h"
#include <Eigen/Core>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

namespace limap {

void bind_clmap_line_mapper(py::module& m) {
  py::class_<LineMapper>(m, "LineMapper")
      .def(py::init<>())
      .def(py::init<const LineMapperConfig&>())
      .def(py::init<py::dict>())
      .def("SetImageCollection", &LineMapper::SetImageCollection)
      .def("SetAllLine2D", &LineMapper::SetAllLine2D)
      .def("SetRanges", &LineMapper::SetRanges)
      .def("SetPLBipartite2d", &LineMapper::SetPLBipartite2d)
      .def("SetSfMPoints", &LineMapper::SetSfMPoints)
      .def("SetVPResults", &LineMapper::SetVPResults)
      .def("GetLocalBestProposalIdx", &LineMapper::GetLocalBestProposalIdx)
      .def("GetGlobalBestProposalIdx", &LineMapper::GetGlobalBestProposalIdx)
      .def("Initialize", &LineMapper::Initialize)
      .def("MatchLines2dByEpipolarIoU", &LineMapper::MatchLines2dByEpipolarIoU)
      .def("TriangulateAllImages", &LineMapper::TriangulateAllImages)
      .def("LoadBidirectionalMatches", &LineMapper::LoadBidirectionalMatches)
      .def("TriangulateImageWithInitialMatches",
           &LineMapper::TriangulateImageWithInitialMatches)
      .def("TriangulateImageWithExhaustiveMatches",
           &LineMapper::TriangulateImageWithExhaustiveMatches)
      .def("SelectLocalBestProposal", &LineMapper::SelectLocalBestProposal)
      .def("InitializeGlobalBestProposal",
           &LineMapper::InitializeGlobalBestProposal)
      .def("SelectGlobalBestProposal", &LineMapper::SelectGlobalBestProposal)
      .def("ComputeDistanceConsistencyPercentage",
           &LineMapper::ComputeDistanceConsistencyPercentage)
      .def("ComputeAngleConsistencyPercentage",
           &LineMapper::ComputeAngleConsistencyPercentage)
      .def("BuildGlobalLineTrack", &LineMapper::BuildGlobalLineTrack)
      .def("GetNumProposal", &LineMapper::GetNumProposal)
      .def("GetNumLines2dHavingProposal",
           &LineMapper::GetNumLines2dHavingProposal)
      .def("ComputeAngleDistanceConsistencyPercentage",
           &LineMapper::ComputeAngleDistanceConsistencyPercentage);
}

void bind_clmap_merging(py::module& m) {
  m.def("RemergeLineTracks", &RemergeLineTracks, py::arg("linetracks"),
        py::arg("linker3d"), py::arg("num_outliers") = 2,
        py::arg("use_collinearity2D") = true, py::arg("overlap_th") = 0.0,
        py::arg("angle_th") = 2.0, py::arg("perp_dist_th") = 2.0);
}

void bind_clmap_mapping(py::module& m) {
  bind_clmap_line_mapper(m);
  bind_clmap_merging(m);
}

}  // namespace limap
