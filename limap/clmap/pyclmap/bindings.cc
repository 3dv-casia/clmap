
#include "clmap/base/bindings.cc"

#include "clmap/mapping/bindings.cc"
#include "clmap/optim/bindings.cc"
#include "clmap/plane_detection/bindings.cc"
#include "clmap/structure/bindings.cc"

#include "_limap/helpers.h"

namespace py = pybind11;

void bind_clmap_base(py::module&);
void bind_clmap_mapping(py::module&);
void bind_clmap_optim(py::module&);
void bind_clmap_structure(py::module&);
void bind_clmap_plane_detection(py::module& m);

namespace limap {

void bind_clmap(py::module& m) {
  bind_clmap_base(m);
  bind_clmap_mapping(m);
  bind_clmap_optim(m);
  bind_clmap_structure(m);
  bind_clmap_plane_detection(m);
}

}  // namespace limap
