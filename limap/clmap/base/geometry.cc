#include "clmap/base/geometry.h"

#include "clmap/util/math.h"

#include <cmath>

#include "base/linetrack.h"
#include <glog/logging.h>

namespace limap {

namespace geometry {

double ComputeAngle(const V3D& dir1, const V3D& dir2) {
  return std::acos(std::min(std::abs(dir1.normalized().dot(dir2.normalized())),
                            1.0)) *
         180.0 / M_PI;
}

double ComputeDistance(const V3D& line2d, const V3D& point2d) {
  V3D line = line2d / (line2d(2) + EPS);
  V3D point = point2d / (point2d(2) + EPS);

  double num = std::abs(line.dot(point));
  double den = std::sqrt(line(0) * line(0) + line(1) * line(1));

  return num / (den + EPS);
}

double ComputeSignedDistancePoint3dToPlane3d(
    const V3D& point3d_xyz, const InfinitePlane3d& inf_plane3d) {
  return point3d_xyz.dot(inf_plane3d.n0) - inf_plane3d.d;
}

double ComputeDistancePoint3dToPlane3d(const V3D& point3d_xyz,
                                       const InfinitePlane3d& inf_plane3d) {
  return std::abs(
      ComputeSignedDistancePoint3dToPlane3d(point3d_xyz, inf_plane3d));
}

void ComputeLines3dUncertainty(const std::vector<LineTrack>& line_tracks,
                               const ImageCollection& img_cols,
                               std::vector<std::pair<V3D, double>>* u_start,
                               std::vector<std::pair<V3D, double>>* u_end,
                               std::vector<std::pair<V3D, double>>* u_median,
                               double var2d, const std::string& u_method) {
  CHECK_NOTNULL(u_start);
  CHECK_NOTNULL(u_end);
  CHECK_NOTNULL(u_median);
  u_start->clear();
  u_end->clear();
  u_median->clear();

  //////////////////////////////////////////////////////////////////////////////
  // Compute the midian point and two endpoints uncertainty.
  //////////////////////////////////////////////////////////////////////////////

  size_t num_lines3D = line_tracks.size();

  u_start->resize(num_lines3D);
  u_end->resize(num_lines3D);
  u_median->resize(num_lines3D);

  // #pragma omp parallel for
  for (size_t i = 0; i < num_lines3D; i++) {
    const LineTrack& line_track = line_tracks[i];
    ComputeLine3dUncertainty(line_track, img_cols, &((*u_start)[i]),
                             &((*u_end)[i]), &((*u_median)[i]), var2d,
                             u_method);
  }
}

void ComputeLines3dUncertainty(
    const std::unordered_map<int, LineTrack>& line_tracks,
    const ImageCollection& img_cols,
    std::unordered_map<int, std::pair<V3D, double>>* u_start,
    std::unordered_map<int, std::pair<V3D, double>>* u_end,
    std::unordered_map<int, std::pair<V3D, double>>* u_median, double var2d,
    const std::string& u_method) {
  CHECK_NOTNULL(u_start);
  CHECK_NOTNULL(u_end);
  CHECK_NOTNULL(u_median);
  u_start->clear();
  u_end->clear();
  u_median->clear();

  //////////////////////////////////////////////////////////////////////////////
  // Compute the midian point and two endpoints uncertainty.
  //////////////////////////////////////////////////////////////////////////////

  size_t num_lines3D = line_tracks.size();

  u_start->reserve(num_lines3D);
  u_end->reserve(num_lines3D);
  u_median->reserve(num_lines3D);

  for (auto it = line_tracks.begin(); it != line_tracks.end(); it++) {
    int line_id = it->first;
    u_start->emplace(line_id, std::pair<V3D, double>());
  }
  *u_end = *u_start;
  *u_median = *u_start;

  for (auto it = line_tracks.begin(); it != line_tracks.end(); it++) {
    int line_id = it->first;
    const LineTrack& line_track = it->second;
    ComputeLine3dUncertainty(line_track, img_cols, &(u_start->at(line_id)),
                             &(u_end->at(line_id)), &(u_median->at(line_id)),
                             var2d, u_method);
  }
}

void ComputeLine3dUncertainty(const LineTrack& line_track,
                              const ImageCollection& img_cols,
                              std::pair<V3D, double>* u_start,
                              std::pair<V3D, double>* u_end,
                              std::pair<V3D, double>* u_median, double var2d,
                              const std::string& u_method) {
  CHECK_NOTNULL(u_start);
  CHECK_NOTNULL(u_end);
  CHECK_NOTNULL(u_median);

  // Compute the midian point and two endpoints uncertainty.
  const Line3d& line3D = line_track.line;
  const V3D& start = line3D.start;
  const V3D& end = line3D.end;
  const V3D median = (start + end) / 2.0;
  const std::vector<int>& image_ids = line_track.GetSortedImageIds();
  const size_t num_images = image_ids.size();

  THROW_CHECK_GE(num_images, 1)

  std::vector<double> all_u_start;
  all_u_start.reserve(num_images);
  std::vector<double> all_u_end;
  all_u_end.reserve(num_images);
  std::vector<double> all_u_median;
  all_u_median.reserve(num_images);

  for (const auto& image_id : image_ids) {
    CameraView view = img_cols.camview(image_id);
    double d1 = view.pose.projdepth(start);
    double d2 = view.pose.projdepth(end);
    double d3 = view.pose.projdepth(median);
    // TODO: the track element whose depth < 0.0 should be filtered.
    all_u_start.push_back(std::abs(view.cam.uncertainty(d1, var2d)));
    all_u_end.push_back(std::abs(view.cam.uncertainty(d2, var2d)));
    all_u_median.push_back(std::abs(view.cam.uncertainty(d3, var2d)));
  }

  if (u_method == "median") {
    *u_start = std::make_pair(start, math::Median<double>(all_u_start));
    *u_end = std::make_pair(end, math::Median<double>(all_u_end));
    *u_median = std::make_pair(median, math::Median<double>(all_u_median));
  } else if (u_method == "average") {
    *u_start = std::make_pair(start, math::Mean<double>(all_u_start));
    *u_end = std::make_pair(end, math::Mean<double>(all_u_end));
    *u_median = std::make_pair(median, math::Mean<double>(all_u_median));
  } else if (u_method == "min") {
    *u_start = std::make_pair(
        start, *std::min_element(all_u_start.begin(), all_u_start.end()));
    *u_end = std::make_pair(
        end, *std::min_element(all_u_end.begin(), all_u_end.end()));
    *u_median = std::make_pair(
        median, *std::min_element(all_u_median.begin(), all_u_median.end()));
  } else {
    throw std::invalid_argument("Error: Not Implemented.");
  }
}

void ComputePoints3dUncertainty(const std::vector<PointTrack>& point_tracks,
                                const ImageCollection& img_cols,
                                std::vector<double>* uncertainties,
                                double var2d, const std::string& u_method) {
  CHECK_NOTNULL(uncertainties);
  uncertainties->clear();
  size_t num_points3D = point_tracks.size();
  uncertainties->resize(num_points3D);

  for (size_t i = 0; i < num_points3D; i++) {
    const PointTrack& point_track = point_tracks[i];
    ComputePoint3dUncertainty(point_track, img_cols, &((*uncertainties)[i]),
                              var2d, u_method);
  }
}

void ComputePoints3dUncertainty(
    const std::unordered_map<int, PointTrack>& point_tracks,
    const ImageCollection& img_cols,
    std::unordered_map<int, double>* uncertainties, double var2d,
    const std::string& u_method) {
  CHECK_NOTNULL(uncertainties);
  uncertainties->clear();
  size_t num_points3D = point_tracks.size();
  uncertainties->reserve(num_points3D);

  // init
  for (auto it = point_tracks.begin(); it != point_tracks.end(); it++) {
    int point_id = it->first;
    uncertainties->emplace(point_id, 0.0);
  }

  for (auto it = point_tracks.begin(); it != point_tracks.end(); it++) {
    int point_id = it->first;
    const PointTrack& point_track = it->second;
    ComputePoint3dUncertainty(point_track, img_cols,
                              &(uncertainties->at(point_id)), var2d, u_method);
  }
}

void ComputePoint3dUncertainty(const PointTrack& point_track,
                               const ImageCollection& img_cols,
                               double* uncertainty, double var2d,
                               const std::string& u_method) {
  CHECK_NOTNULL(uncertainty);

  const V3D& xyz = point_track.p;
  const std::vector<int>& image_ids = point_track.image_id_list;
  const size_t num_images = image_ids.size();

  THROW_CHECK_GE(num_images, 1)

  std::vector<double> all_u;
  all_u.reserve(num_images);

  for (const auto& image_id : image_ids) {
    CameraView view = img_cols.camview(image_id);
    double depth = view.pose.projdepth(xyz);
    // TODO: the point whose depth < 0.0 should be filtered.
    all_u.push_back(std::abs(view.cam.uncertainty(depth, var2d)));
  }

  if (u_method == "median") {
    *uncertainty = math::Median<double>(all_u);
  } else if (u_method == "average") {
    *uncertainty = math::Mean<double>(all_u);
  } else if (u_method == "min") {
    *uncertainty = *std::min_element(all_u.begin(), all_u.end());
  } else {
    throw std::invalid_argument("Error: Not Implemented.");
  }
}

bool TestLine3dOnInfPlane3d(const Line3d& line3d,
                            const InfinitePlane3d& inf_plane3d,
                            double start_uncertainty, double end_uncertainty,
                            double max_si_dist, double min_angle,
                            double* scale_inv_dist) {
  CHECK_NOTNULL(scale_inv_dist);
  THROW_CHECK_GT(start_uncertainty, 0);
  THROW_CHECK_GT(end_uncertainty, 0);

  const V3D& start = line3d.start;
  const V3D& end = line3d.end;

  double start_dist = ComputeDistancePoint3dToPlane3d(start, inf_plane3d);
  double end_dist = ComputeDistancePoint3dToPlane3d(end, inf_plane3d);
  double start_scale_inv_dist = start_dist / (start_uncertainty + EPS);
  double end_scale_inv_dist = end_dist / (end_uncertainty + EPS);
  *scale_inv_dist = std::max(start_scale_inv_dist, end_scale_inv_dist);  // > 0

  if (*scale_inv_dist > max_si_dist) {
    return false;
  }

  double angle = geometry::ComputeAngle(line3d.direction(), inf_plane3d.n0);
  if (angle < min_angle) {
    return false;
  }
  return true;
}

bool TestPointOnInfPlane3d(const V3D& point3d_xyz,
                           const InfinitePlane3d& inf_plane3d,
                           double uncertainty,
                           double th_hard_pointplane_si_dist3d,
                           double* scale_inv_dist) {
  THROW_CHECK_GT(uncertainty, 0);
  double d = geometry::ComputeDistancePoint3dToPlane3d(point3d_xyz,
                                                       inf_plane3d);  // > 0
  *scale_inv_dist = d / uncertainty;                                  // > 0
  THROW_CHECK_GE(*scale_inv_dist, 0);
  if (*scale_inv_dist > th_hard_pointplane_si_dist3d) {
    return false;
  }
  return true;
}

}  // namespace geometry

}  // namespace limap
