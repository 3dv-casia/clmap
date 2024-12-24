#include <vector>

#include "_limap/helpers.h"
#include <Eigen/Core>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

#include "clmap/optim/plp_associator.h"

namespace limap {

void bind_clmap_plp_association(py::module& m) {
  py::class_<PLPAssociatorConfig>(m, "PLPAssociatorConfig")
      .def(py::init<>())
      .def(py::init<py::dict>());

  py::class_<PLPAssociator>(m, "PLPAssociator")
      .def(py::init<>())
      .def(py::init<const PLPAssociatorConfig&>())
      .def("InitImagecols", &PLPAssociator::InitImagecols)
      .def("InitPointTracks",
           static_cast<void (PLPAssociator::*)(const std::vector<PointTrack>&)>(
               &PLPAssociator::InitPointTracks))
      .def("InitPointTracks", static_cast<void (PLPAssociator::*)(
                                  const std::map<int, PointTrack>&)>(
                                  &PLPAssociator::InitPointTracks))
      .def("InitLineTracks",
           static_cast<void (PLPAssociator::*)(const std::vector<LineTrack>&)>(
               &PLPAssociator::InitLineTracks))
      .def(
          "InitLineTracks",
          static_cast<void (PLPAssociator::*)(const std::map<int, LineTrack>&)>(
              &PLPAssociator::InitLineTracks))
      .def("InitVPTracks", static_cast<void (PLPAssociator::*)(
                               const std::vector<limap::vplib::VPTrack>&)>(
                               &PLPAssociator::InitVPTracks))
      .def("InitVPTracks", static_cast<void (PLPAssociator::*)(
                               const std::map<int, limap::vplib::VPTrack>&)>(
                               &PLPAssociator::InitVPTracks))
      .def("Init2DBipartites_PointLine",
           &PLPAssociator::Init2DBipartites_PointLine)
      .def("Init2DBipartites_VPLine", &PLPAssociator::Init2DBipartites_VPLine)
      .def("InitPlanes", &PLPAssociator::InitPlanes)
      .def("InitPP_Bipartite3d", &PLPAssociator::InitPP_Bipartite3d)
      .def("InitLP_Bipartite3d", &PLPAssociator::InitLP_Bipartite3d)
      .def("ComputePointsUncertainty", &PLPAssociator::ComputePointsUncertainty)
      .def("ComputePointsOnLineUncertainty",
           &PLPAssociator::ComputePointsOnLineUncertainty)
      .def("SetUp", &PLPAssociator::SetUp)
      .def("Solve", &PLPAssociator::Solve)
      .def("UpdatePointTracks", &PLPAssociator::UpdatePointTracks)
      .def("UpdateLineTracks", &PLPAssociator::UpdateLineTracks)
      .def("UpdateUncertainty", &PLPAssociator::UpdateUncertainty)
      .def("UpdatePL_Bipartite3d", &PLPAssociator::UpdatePL_Bipartite3d)
      .def("UpdateVPLine_Bipartite3d", &PLPAssociator::UpdateVPLine_Bipartite3d)
      .def("UpdatePlanes", &PLPAssociator::UpdatePlanes)
      .def("UpdatePP_Bipartite3d", &PLPAssociator::UpdatePP_Bipartite3d)
      .def("UpdateLP_Bipartite3d", &PLPAssociator::UpdateLP_Bipartite3d)
      .def("GetOutputImagecols", &PLPAssociator::GetOutputImagecols)
      .def("GetPointTracks", &PLPAssociator::GetPointTracks)
      .def("GetLineTracks", &PLPAssociator::GetLineTracks)
      .def("GetPL_Bipartite3d", &PLPAssociator::GetPL_Bipartite3d)
      .def("GetVPLine_Bipartite3d", &PLPAssociator::GetVPLine_Bipartite3d)
      .def("GetPlanes", &PLPAssociator::GetPlanes)
      .def("GetPP_Bipartite3d", &PLPAssociator::GetPP_Bipartite3d)
      .def("GetLP_Bipartite3d", &PLPAssociator::GetLP_Bipartite3d)
      .def("GetOutputVPTracks", &PLPAssociator::GetOutputVPTracks)
      .def("GetPointInitialCost", &PLPAssociator::GetPointInitialCost)
      .def("GetLineInitialCost", &PLPAssociator::GetLineInitialCost)
      .def("GetPointLineInitialCost", &PLPAssociator::GetPointLineInitialCost)
      .def("GetVPLineInitialCost", &PLPAssociator::GetVPLineInitialCost)
      .def("GetVPOrthogonalityInitialCost",
           &PLPAssociator::GetVPOrthogonalityInitialCost)
      .def("GetVPCollinearityInitialCost",
           &PLPAssociator::GetVPCollinearityInitialCost)
      .def("GetPointPlaneInitialCost", &PLPAssociator::GetPointPlaneInitialCost)
      .def("GetLinePlaneDistanceInitialCost",
           &PLPAssociator::GetLinePlaneDistanceInitialCost)
      .def("GetLinePlaneAngleInitialCost",
           &PLPAssociator::GetLinePlaneAngleInitialCost)
      .def("GetPlaneOrthogonalityInitialCost",
           &PLPAssociator::GetPlaneOrthogonalityInitialCost)
      .def("GetPlaneParallelismInitialCost",
           &PLPAssociator::GetPlaneParallelismInitialCost)
      .def("ComputeAdaptiveLossWeight",
           &PLPAssociator::ComputeAdaptiveLossWeight)
      .def("PrintLossWeight", &PLPAssociator::PrintLossWeight)
      .def("GetAllLossWeight", &PLPAssociator::GetAllLossWeight);
}

void bind_clmap_optim(py::module& m) { bind_clmap_plp_association(m); }

}  // namespace limap
