#include "clmap/structure/bipartite_builder.h"

#include "_limap/helpers.h"

namespace py = pybind11;

namespace limap {

void bind_clmap_bipartite_builder(py::module& m) {
  m.def("build_init_PP_Bipartite3d", &BuildInitPP_Bipartite3d);
  m.def("build_init_LP_Bipartite3d", &BuildInitLP_Bipartite3d);
  m.def("filter_inf_plane3d_with_PP_LP_Bipartite3d",
        &FilterInfPlane3dWithPP_LP_Bipartite3d);
}

void bind_clmap_structure(py::module& m) { bind_clmap_bipartite_builder(m); }

}  // namespace limap
