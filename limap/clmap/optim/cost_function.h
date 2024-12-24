#pragma once

#include "clmap/base/infinite_plane3d.h"
#include "clmap/util/types.h"

#include "ceresbase/line_dists.h"
#include "ceresbase/line_transforms.h"

namespace limap {

template <typename T>
void MinimalPlaneToHesse(const T* const plane_vec,
                         MinimalInfinitePlane3d::MinimalType minimal_type,
                         T n0[3], T d[1]) {
  if (minimal_type == MinimalInfinitePlane3d::MinimalType::CP) {
    T norm =
        ceres::sqrt(plane_vec[0] * plane_vec[0] + plane_vec[1] * plane_vec[1] +
                    plane_vec[2] * plane_vec[2]);
    if (norm == T(0)) {
      n0[0] = T(1.0);
      n0[1] = T(0.0);
      n0[2] = T(0.0);
      d[0] = T(0.0);
    } else {
      n0[0] = plane_vec[0] / norm;
      n0[1] = plane_vec[1] / norm;
      n0[2] = plane_vec[2] / norm;
      d[0] = norm;
    }
  } else if (minimal_type == MinimalInfinitePlane3d::MinimalType::QUATERNION) {
    T norm =
        ceres::sqrt(plane_vec[1] * plane_vec[1] + plane_vec[2] * plane_vec[2] +
                    plane_vec[3] * plane_vec[3] + EPS);
    n0[0] = plane_vec[1] / norm;
    n0[1] = plane_vec[2] / norm;
    n0[2] = plane_vec[3] / norm;
    d[0] = -plane_vec[0] / norm;
  } else if (minimal_type == MinimalInfinitePlane3d::MinimalType::HOMOGENEOUS) {
    T norm =
        ceres::sqrt(plane_vec[0] * plane_vec[0] + plane_vec[1] * plane_vec[1] +
                    plane_vec[2] * plane_vec[2] + EPS);
    n0[0] = plane_vec[0] / norm;
    n0[1] = plane_vec[1] / norm;
    n0[2] = plane_vec[2] / norm;
    d[0] = -plane_vec[3] / norm;
  } else {
    throw std::invalid_argument("Error: minimal_type is invalid");
  }
}

////////////////////////////////////////////////////////////
// point-line association on 3D
////////////////////////////////////////////////////////////

struct PointLineAssociation3dFunctor {
 public:
  PointLineAssociation3dFunctor(double uncertainty)
      : uncertainty_(uncertainty) {
    THROW_CHECK_GT(uncertainty, 0);
  }

  static ceres::CostFunction* Create(double uncertainty) {
    return new ceres::AutoDiffCostFunction<PointLineAssociation3dFunctor, 3, 3,
                                           4, 2>(
        new PointLineAssociation3dFunctor(uncertainty));
  }

  template <typename T>
  bool operator()(const T* const point_vec, const T* const uvec,
                  const T* const wvec, T* residuals) const {
    T dir3d[3], b[3];
    MinimalPluckerToPlucker<T>(uvec, wvec, dir3d, b);

    // Reference: page 4 in the following link:
    // [LINK]:
    // https://faculty.sites.iastate.edu/jia/files/inline-files/plucker-coordinates.pdf
    T b_point[3];
    ceres::CrossProduct(dir3d, point_vec, b_point);
    for (size_t i = 0; i < 3; ++i) {
      b_point[i] = b_point[i] + b[i];
    }
    T disp[3];
    ceres::CrossProduct(dir3d, b_point, disp);
    residuals[0] = disp[0] / uncertainty_;
    residuals[1] = disp[1] / uncertainty_;
    residuals[2] = disp[2] / uncertainty_;
    return true;
  }

 private:
  double uncertainty_;
};

////////////////////////////////////////////////////////////
// point-plane association on 3D
////////////////////////////////////////////////////////////

struct PointPlaneAssociation3dFunctor {
 public:
  PointPlaneAssociation3dFunctor(
      const MinimalInfinitePlane3d::MinimalType& minimal_type,
      double uncertainty)
      : minimal_type_(minimal_type), uncertainty_(uncertainty) {
    THROW_CHECK_GT(uncertainty, 0);
  }

  static ceres::CostFunction* Create(
      const MinimalInfinitePlane3d::MinimalType& minimal_type,
      double uncertainty = 1.0) {
    if (minimal_type == MinimalInfinitePlane3d::MinimalType::CP) {
      return new ceres::AutoDiffCostFunction<PointPlaneAssociation3dFunctor, 1,
                                             3, 3>(
          new PointPlaneAssociation3dFunctor(minimal_type, uncertainty));
    } else if (minimal_type ==
                   MinimalInfinitePlane3d::MinimalType::QUATERNION ||
               minimal_type ==
                   MinimalInfinitePlane3d::MinimalType::HOMOGENEOUS) {
      return new ceres::AutoDiffCostFunction<PointPlaneAssociation3dFunctor, 1,
                                             3, 4>(
          new PointPlaneAssociation3dFunctor(minimal_type, uncertainty));
    } else {
      throw std::invalid_argument("Error: minimal_type is invalid");
    }
  }

  template <typename T>
  bool operator()(const T* const point_vec, const T* const plane_vec,
                  T* residuals) const {
    T n0[3], d[1];

    MinimalPlaneToHesse<T>(plane_vec, minimal_type_, n0, d);

    residuals[0] = n0[0] * point_vec[0] + n0[1] * point_vec[1] +
                   n0[2] * point_vec[2] - d[0];

    residuals[0] /= uncertainty_;

    return true;
  }

 protected:
  MinimalInfinitePlane3d::MinimalType minimal_type_;
  double uncertainty_;
};

////////////////////////////////////////////////////////////
// line-plane association on 3D
////////////////////////////////////////////////////////////

struct LinePlaneAssociation3dDistanceFunctor {
 public:
  LinePlaneAssociation3dDistanceFunctor(
      const MinimalInfinitePlane3d::MinimalType& minimal_type,
      const std::vector<std::pair<V3D, double>>& un_points_on_line)
      : minimal_type_(minimal_type), un_points_on_line_(un_points_on_line) {
    for (const auto& point : un_points_on_line) {
      THROW_CHECK_GT(point.second, 0);
    }
  }

  static ceres::CostFunction* Create(
      const MinimalInfinitePlane3d::MinimalType& minimal_type,
      const std::vector<std::pair<V3D, double>>& un_points_on_line =
          std::vector<std::pair<V3D, double>>()) {
    size_t n = un_points_on_line.size();
    if (minimal_type == MinimalInfinitePlane3d::MinimalType::CP) {
      if (n < 2) {
        return new ceres::AutoDiffCostFunction<
            LinePlaneAssociation3dDistanceFunctor, 1, 4, 2, 3>(
            new LinePlaneAssociation3dDistanceFunctor(minimal_type,
                                                      un_points_on_line));
      } else if (n == 2) {
        return new ceres::AutoDiffCostFunction<
            LinePlaneAssociation3dDistanceFunctor, 2, 4, 2, 3>(
            new LinePlaneAssociation3dDistanceFunctor(minimal_type,
                                                      un_points_on_line));
      } else {
        throw std::runtime_error(
            "Error: the number of points on line only supports 0, 1 or 2.");
      }
    } else if (minimal_type ==
                   MinimalInfinitePlane3d::MinimalType::QUATERNION ||
               minimal_type ==
                   MinimalInfinitePlane3d::MinimalType::HOMOGENEOUS) {
      if (n < 2) {
        return new ceres::AutoDiffCostFunction<
            LinePlaneAssociation3dDistanceFunctor, 1, 4, 2, 4>(
            new LinePlaneAssociation3dDistanceFunctor(minimal_type,
                                                      un_points_on_line));
      } else if (n == 2) {
        return new ceres::AutoDiffCostFunction<
            LinePlaneAssociation3dDistanceFunctor, 2, 4, 2, 4>(
            new LinePlaneAssociation3dDistanceFunctor(minimal_type,
                                                      un_points_on_line));
      } else {
        throw std::runtime_error(
            "Error: the number of points on line only supports 0, 1 or 2.");
      }
    } else {
      throw std::invalid_argument("Error: minimal_type is invalid");
    }
  }

  template <typename T>
  bool operator()(const T* const uvec, const T* const wvec,
                  const T* const plane_vec, T* residuals) const {
    T dir3d[3], m[3];
    MinimalPluckerToPlucker<T>(uvec, wvec, dir3d, m);

    T n0[3], d[1];
    MinimalPlaneToHesse<T>(plane_vec, minimal_type_, n0, d);

    size_t num_points_on_line = un_points_on_line_.size();
    if (num_points_on_line == 0) {
      // Compute the perpendicular distance between the point3d (the closest
      // 3D point from origion to current infinite 3D line) on current infinite
      // 3D line to plane.
      T point3d[3];
      ceres::CrossProduct(dir3d, m, point3d);
      residuals[0] =
          n0[0] * point3d[0] + n0[1] * point3d[1] + n0[2] * point3d[2] - d[0];
      // the default uncertainty is 1.0 here
      return true;
    } else {
      // Compute the perpendicular distance between some points3d (the 3D points
      // projected from some fixed 3D points on the initial 3D line segment) on
      // current infinite 3D line to plane.
      //
      // Note: only supports 1 or 2 points3d now
      for (size_t i = 0; i < num_points_on_line; i++) {
        // Reference: page 4 in the following link:
        // [LINK]:
        // https://faculty.sites.iastate.edu/jia/files/inline-files/plucker-coordinates.pdf

        // compute $m_q$ using Eq.(5)
        V3D point_xyz = un_points_on_line_.at(i).first;  // a fixed 3D point $q$
        T point3d[3] = {T(point_xyz(0)), T(point_xyz(1)), T(point_xyz(2))};
        T res1[3];  // $\hat l \times q$
        ceres::CrossProduct(dir3d, point3d, res1);
        T m_q[3];  // $m_q$
        m_q[0] = m[0] + res1[0];
        m_q[1] = m[1] + res1[1];
        m_q[2] = m[2] + res1[2];

        // compute $q_{\perp}$ using Eq.(6)
        T res2[3];  // $\hat l \times m_q$
        ceres::CrossProduct(dir3d, m_q, res2);
        T proj_point3d[3];  // $q_{\perp}$
        proj_point3d[0] = point3d[0] + res2[0];
        proj_point3d[1] = point3d[1] + res2[1];
        proj_point3d[2] = point3d[2] + res2[2];

        // compute the perpendicular distance between $q_{\perp}$ to plane
        residuals[i] = n0[0] * proj_point3d[0] + n0[1] * proj_point3d[1] +
                       n0[2] * proj_point3d[2] - d[0];

        double uncertainty = un_points_on_line_.at(i).second;
        residuals[i] /= uncertainty;
      }
      return true;
    }
  }

 protected:
  MinimalInfinitePlane3d::MinimalType minimal_type_;

  // (xyz of point3d, uncertainty of point3d)
  std::vector<std::pair<V3D, double>> un_points_on_line_;
};

struct LinePlaneAssociation3dAngleFunctor {
 public:
  LinePlaneAssociation3dAngleFunctor(
      const MinimalInfinitePlane3d::MinimalType& minimal_type)
      : minimal_type_(minimal_type) {}

  static ceres::CostFunction* Create(
      const MinimalInfinitePlane3d::MinimalType& minimal_type) {
    if (minimal_type == MinimalInfinitePlane3d::MinimalType::CP) {
      return new ceres::AutoDiffCostFunction<LinePlaneAssociation3dAngleFunctor,
                                             1, 4, 2, 3>(
          new LinePlaneAssociation3dAngleFunctor(minimal_type));
    } else if (minimal_type ==
                   MinimalInfinitePlane3d::MinimalType::QUATERNION ||
               minimal_type ==
                   MinimalInfinitePlane3d::MinimalType::HOMOGENEOUS) {
      return new ceres::AutoDiffCostFunction<LinePlaneAssociation3dAngleFunctor,
                                             1, 4, 2, 4>(
          new LinePlaneAssociation3dAngleFunctor(minimal_type));
    } else {
      throw std::invalid_argument("Error: minimal_type is invalid");
    }
  }

  template <typename T>
  bool operator()(const T* const uvec, const T* const wvec,
                  const T* const plane_vec, T* residuals) const {
    T dir3d[3], m[3];
    MinimalPluckerToPlucker<T>(uvec, wvec, dir3d, m);

    T n0[3], d[1];
    MinimalPlaneToHesse<T>(plane_vec, minimal_type_, n0, d);

    residuals[0] = CeresComputeDist3D_cosine(dir3d, n0);

    return true;
  }

 protected:
  MinimalInfinitePlane3d::MinimalType minimal_type_;
};

////////////////////////////////////////////////////////////
// plane-plane orthogonality on 3D
////////////////////////////////////////////////////////////

struct PlaneOrthogonalityFunctor {
 public:
  PlaneOrthogonalityFunctor(
      const MinimalInfinitePlane3d::MinimalType& minimal_type)
      : minimal_type_(minimal_type) {}

  static ceres::CostFunction* Create(
      const MinimalInfinitePlane3d::MinimalType& minimal_type) {
    if (minimal_type == MinimalInfinitePlane3d::MinimalType::CP) {
      return new ceres::AutoDiffCostFunction<PlaneOrthogonalityFunctor, 1, 3,
                                             3>(
          new PlaneOrthogonalityFunctor(minimal_type));
    } else if (minimal_type ==
                   MinimalInfinitePlane3d::MinimalType::QUATERNION ||
               minimal_type ==
                   MinimalInfinitePlane3d::MinimalType::HOMOGENEOUS) {
      return new ceres::AutoDiffCostFunction<PlaneOrthogonalityFunctor, 1, 4,
                                             4>(
          new PlaneOrthogonalityFunctor(minimal_type));
    } else {
      throw std::invalid_argument("Error: minimal_type is invalid");
    }
  }

  template <typename T>
  bool operator()(const T* const plane1_vec, const T* const plane2_vec,
                  T* residuals) const {
    T n1[3], d1[1];
    MinimalPlaneToHesse<T>(plane1_vec, minimal_type_, n1, d1);
    T n2[3], d2[1];
    MinimalPlaneToHesse<T>(plane2_vec, minimal_type_, n2, d2);

    residuals[0] = CeresComputeDist3D_cosine(n1, n2);
    return true;
  }

 protected:
  MinimalInfinitePlane3d::MinimalType minimal_type_;
};

////////////////////////////////////////////////////////////
// plane-plane parallelism on 3D
////////////////////////////////////////////////////////////

struct PlaneParallelismFunctor {
 public:
  PlaneParallelismFunctor(
      const MinimalInfinitePlane3d::MinimalType& minimal_type)
      : minimal_type_(minimal_type) {}

  static ceres::CostFunction* Create(
      const MinimalInfinitePlane3d::MinimalType& minimal_type) {
    if (minimal_type == MinimalInfinitePlane3d::MinimalType::CP) {
      return new ceres::AutoDiffCostFunction<PlaneParallelismFunctor, 1, 3, 3>(
          new PlaneParallelismFunctor(minimal_type));
    } else if (minimal_type ==
                   MinimalInfinitePlane3d::MinimalType::QUATERNION ||
               minimal_type ==
                   MinimalInfinitePlane3d::MinimalType::HOMOGENEOUS) {
      return new ceres::AutoDiffCostFunction<PlaneParallelismFunctor, 1, 4, 4>(
          new PlaneParallelismFunctor(minimal_type));
    } else {
      throw std::invalid_argument("Error: minimal_type is invalid");
    }
  }

  template <typename T>
  bool operator()(const T* const plane1_vec, const T* const plane2_vec,
                  T* residuals) const {
    T n1[3], d1[1];
    MinimalPlaneToHesse<T>(plane1_vec, minimal_type_, n1, d1);
    T n2[3], d2[1];
    MinimalPlaneToHesse<T>(plane2_vec, minimal_type_, n2, d2);

    residuals[0] = CeresComputeDist3D_sine(n1, n2);
    return true;
  }

 protected:
  MinimalInfinitePlane3d::MinimalType minimal_type_;
};

}  // namespace limap
