#include <vector>

#include "_limap/helpers.h"
#include <Eigen/Core>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

#include "clmap/base/bipartite.h"
#include "clmap/base/infinite_plane3d.h"

namespace limap {

void bind_clmap_infinite_plane3d(py::module& m) {
  py::class_<InfinitePlane3d>(m, "InfinitePlane3d")
      .def(py::init<>())
      .def(py::init<V3D, double>(), py::arg("n0"), py::arg("d"))
      .def("GetHomogeneousVec", &InfinitePlane3d::GetHomogeneousVec)
      .def_readwrite("n0", &InfinitePlane3d::n0)
      .def_readwrite("d", &InfinitePlane3d::d);
}

void bind_clmap_bipartite(py::module& m) {
  py::class_<PP_Bipartite3d>(m, "PP_Bipartite3d")
      .def(py::init<>())
      .def(py::init<const PP_Bipartite3d&>())
      .def(py::init<py::dict>())
      .def("as_dict", &PP_Bipartite3d::as_dict)
      .def("AddObj1", &PP_Bipartite3d::AddObj1)
      .def("AddObj2", &PP_Bipartite3d::AddObj2)
      .def("AddEdge", &PP_Bipartite3d::AddEdge)
      .def("ExistObj1", &PP_Bipartite3d::ExistObj1)
      .def("ExistObj2", &PP_Bipartite3d::ExistObj2)
      .def("ClearEdges", &PP_Bipartite3d::ClearEdges)
      .def("UpdateObj1", &PP_Bipartite3d::UpdateObj1)
      .def("UpdateObj2", &PP_Bipartite3d::UpdateObj2)
      .def("GetObj1Neighbors", &PP_Bipartite3d::GetObj1Neighbors)
      .def("GetObj2Neighbors", &PP_Bipartite3d::GetObj2Neighbors)
      .def("GetObj1Ids", &PP_Bipartite3d::GetObj1Ids)
      .def("GetObj2Ids", &PP_Bipartite3d::GetObj2Ids)
      .def_readwrite("obj1_map", &PP_Bipartite3d::obj1_map)
      .def_readwrite("obj2_map", &PP_Bipartite3d::obj2_map)
      .def_readwrite("n_1_to_2", &PP_Bipartite3d::n_1_to_2)
      .def_readwrite("n_2_to_1", &PP_Bipartite3d::n_2_to_1);

  py::class_<LP_Bipartite3d>(m, "LP_Bipartite3d")
      .def(py::init<>())
      .def(py::init<const LP_Bipartite3d&>())
      .def(py::init<py::dict>())
      .def("as_dict", &LP_Bipartite3d::as_dict)
      .def("AddObj1", &LP_Bipartite3d::AddObj1)
      .def("AddObj2", &LP_Bipartite3d::AddObj2)
      .def("AddEdge", &LP_Bipartite3d::AddEdge)
      .def("ExistObj1", &LP_Bipartite3d::ExistObj1)
      .def("ExistObj2", &LP_Bipartite3d::ExistObj2)
      .def("ClearEdges", &LP_Bipartite3d::ClearEdges)
      .def("UpdateObj1", &LP_Bipartite3d::UpdateObj1)
      .def("UpdateObj2", &LP_Bipartite3d::UpdateObj2)
      .def("GetObj1Neighbors", &LP_Bipartite3d::GetObj1Neighbors)
      .def("GetObj2Neighbors", &LP_Bipartite3d::GetObj2Neighbors)
      .def("GetObj1Ids", &LP_Bipartite3d::GetObj1Ids)
      .def("GetObj2Ids", &LP_Bipartite3d::GetObj2Ids)
      .def_readwrite("obj1_map", &LP_Bipartite3d::obj1_map)
      .def_readwrite("obj2_map", &LP_Bipartite3d::obj2_map)
      .def_readwrite("n_1_to_2", &LP_Bipartite3d::n_1_to_2)
      .def_readwrite("n_2_to_1", &LP_Bipartite3d::n_2_to_1);
}

void bind_clmap_base(py::module& m) {
  bind_clmap_infinite_plane3d(m);
  bind_clmap_bipartite(m);
}

}  // namespace limap
