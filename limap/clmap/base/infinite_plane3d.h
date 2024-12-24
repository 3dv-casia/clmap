#pragma once

#include "clmap/util/types.h"

namespace limap {

// A 3D point r on the plane iif r.dot(n0) = d, where n0 is an unit
// normal vector and d >= 0 or d < 0.
class InfinitePlane3d {
 public:
  InfinitePlane3d() {}
  InfinitePlane3d(const V3D& _n0, double _d);

  V4D GetHomogeneousVec() const;

  // normalized normal
  V3D n0;

  // d >= 0 or d < 0
  double d;
};

// Minimal InfinitePlane3d used for ceres optimization.
class MinimalInfinitePlane3d {
 public:
  enum class MinimalType { CP = 0, QUATERNION, HOMOGENEOUS };

  MinimalInfinitePlane3d() {}
  MinimalInfinitePlane3d(const std::vector<double>& values,
                         const MinimalType& type);
  MinimalInfinitePlane3d(const InfinitePlane3d& inf_plane3d,
                         const MinimalType& type);

  InfinitePlane3d GetInfinitePlane3d() const;

  // type0: closet-point vector, dof = 3
  V3D cpvec;

  // type1: quaternion vector for SO(3), dof = 3
  V4D qvec;

  // type2: homogehous vector, dof = 3
  V4D hvec;

  // Minimal type
  MinimalType minimal_type;
};

}  // namespace limap
