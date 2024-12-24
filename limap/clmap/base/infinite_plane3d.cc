#include "clmap/base/infinite_plane3d.h"

#include <colmap/base/pose.h>
#include <glog/logging.h>

namespace limap {

InfinitePlane3d::InfinitePlane3d(const V3D& _n0, double _d) : n0(_n0), d(_d) {
  CHECK_LE(std::abs(n0.norm() - 1), EPS);
}

V4D InfinitePlane3d::GetHomogeneousVec() const {
  return V4D(n0(0), n0(1), n0(2), -d);
}

MinimalInfinitePlane3d::MinimalInfinitePlane3d(
    const std::vector<double>& values, const MinimalType& type) {
  if (type == MinimalType::CP) {
    cpvec(0) = values.at(0);
    cpvec(1) = values.at(1);
    cpvec(2) = values.at(2);
    CHECK_GT(cpvec.norm(), 0);
  } else if (type == MinimalType::QUATERNION) {
    qvec(0) = values.at(0);  // w
    qvec(1) = values.at(1);  // x
    qvec(2) = values.at(2);  // y
    qvec(3) = values.at(3);  // z
    CHECK_GT(qvec.norm(), 0);
    qvec = colmap::NormalizeQuaternion(qvec);
  } else if (type == MinimalType::HOMOGENEOUS) {
    hvec(0) = values.at(0);
    hvec(1) = values.at(1);
    hvec(2) = values.at(2);
    hvec(3) = values.at(3);
    CHECK_GT(hvec.norm(), 0);
  } else {
    throw std::invalid_argument(
        "Error: minimal_type must be CP, QUATERNION or HOMOGENEOUS.");
  }
  minimal_type = type;
}

MinimalInfinitePlane3d::MinimalInfinitePlane3d(
    const InfinitePlane3d& inf_plane3d, const MinimalType& type) {
  double d = inf_plane3d.d;
  V3D n0 = inf_plane3d.n0;

  if (type == MinimalType::CP) {
    cpvec = d * n0;  // from origion to the shortest point on the plane.
  } else if (type == MinimalType::QUATERNION) {
    qvec = colmap::NormalizeQuaternion(
        V4D(-d, n0(0), n0(1), n0(2)));  // (w, x, y, z)
  } else if (type == MinimalType::HOMOGENEOUS) {
    hvec = V4D(n0(0), n0(1), n0(2), -d);
  } else {
    throw std::invalid_argument(
        "Error: minimal_type must be CP, QUATERNION or HOMOGENEOUS.");
  }
  minimal_type = type;
}

InfinitePlane3d MinimalInfinitePlane3d::GetInfinitePlane3d() const {
  if (minimal_type == MinimalType::CP) {
    double d = cpvec.norm();
    if (d == 0) {
      return InfinitePlane3d(V3D(1.0, 0.0, 0.0), 0.0);
    } else {
      return InfinitePlane3d(cpvec / d, d);
    }
  } else if (minimal_type == MinimalType::QUATERNION) {
    V3D normal = V3D(qvec(1), qvec(2), qvec(3));
    double norm = normal.norm();
    CHECK_GT(norm, 0);
    V3D n0 = normal / norm;
    double d = -qvec(0) / norm;
    return InfinitePlane3d(n0, d);
  } else if (minimal_type == MinimalType::HOMOGENEOUS) {
    V3D normal = V3D(hvec(0), hvec(1), hvec(2));
    double norm = normal.norm();
    CHECK_GT(norm, 0);
    V3D n0 = normal / norm;
    double d = -hvec(3) / norm;
    return InfinitePlane3d(n0, d);
  } else {
    throw std::invalid_argument(
        "Error: minimal_type must be CP, QUATERNION or HOMOGENEOUS.");
  }
}

}  // namespace limap
