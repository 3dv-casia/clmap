#include "clmap/mapping/line_triangulation.h"

#include "base/line_dists.h"
#include "merging/aggregator.h"
#include "triangulation/functions.h"

namespace limap {

bool TriangulateLineWithPlanes(const CameraView& view1, const CameraView& view2,
                               const Line2d& l1, const Line2d& l2,
                               Line3d* line) {
  CHECK_NOTNULL(line);

  *line = limap::triangulation::triangulate(l1, view1, l2, view2);
  if (line->score < 0) {
    return false;
  }
  return true;
}

bool TriangulateLineWithPoints(const CameraView& view1, const Line2d& l1,
                               const std::vector<V3D>& points, Line3d* line) {
  CHECK_NOTNULL(line);

  if (points.size() < 2) {
    return false;
  }

  InfiniteLine3d inf_line;
  FitInfLineWithPointsPCA(points, &inf_line);

  *line =
      limap::triangulation::triangulate_with_infinite_line(l1, view1, inf_line);

  if (line->score < 0) {
    return false;
  }
  return true;
}

void FindSharedPoints3DId(const CameraView& view1, const CameraView& view2,
                          const line2d_t& l1_idx, const line2d_t& l2_idx,
                          const structures::PL_Bipartite2d& bqt1,
                          const structures::PL_Bipartite2d& bqt2,
                          const std::unordered_map<point3d_t, V3D>& sfm_points,
                          std::unordered_set<int>* point_ids) {
  CHECK_NOTNULL(point_ids);
  point_ids->clear();

  // (point3D_id, corresponding Point2d)
  std::map<point3d_t, Point2d> points1;

  // ids of 3D points of the 2D points on ref line
  std::set<point3d_t> set1;

  // (point3D_id, (ref_line2D_point2D, ng_line2D_point2D))
  std::map<point3d_t, std::pair<V2D, V2D>> points_info;

  for (const point2d_t& point_idx : bqt1.neighbor_points(l1_idx)) {
    // point2d on ref line2d
    const Point2d& p = bqt1.point(point_idx);
    set1.insert(p.point3D_id);
    points1.insert(std::make_pair(p.point3D_id, p));
  }

  for (const point2d_t& point_idx : bqt2.neighbor_points(l2_idx)) {
    // point2d on neighbor line2d
    auto p = bqt2.point(point_idx);
    if (set1.find(p.point3D_id) != set1.end()) {
      // coordinates of Point2d on ref line2d
      V2D p1 = points1.at(p.point3D_id).p;
      points_info.insert(
          std::make_pair(p.point3D_id, std::pair<V2D, V2D>(p1, p.p)));
    }
  }

  // save shared 3D points with respect to ref and neighbor line
  for (auto it = points_info.begin(); it != points_info.end(); ++it) {
    point_ids->insert(it->first);
  }
}

void FindPoints3D(const line2d_t& line2d_idx,
                  const structures::PL_Bipartite2d& bqt,
                  std::unordered_set<int>* point3D_ids) {
  CHECK_NOTNULL(point3D_ids);
  point3D_ids->clear();

  for (const point2d_t& point_idx : bqt.neighbor_points(line2d_idx)) {
    const Point2d& p = bqt.point(point_idx);
    point3D_ids->insert(p.point3D_id);
  }
}

void FitInfLineWithPointsPCA(const std::vector<V3D>& points,
                             InfiniteLine3d* inf_line) {
  CHECK_NOTNULL(inf_line);
  THROW_CHECK_GE(points.size(), 2);

  // compute 3D points center
  V3D center(0.0, 0.0, 0.0);
  for (size_t i = 0; i < points.size(); ++i) {
    center += points[i];
  }
  center /= points.size();

  // compute infinite 3D line direction
  Eigen::MatrixXd epoints;
  epoints.resize(points.size(), 3);
  for (size_t i = 0; i < points.size(); ++i) {
    epoints.row(i) = points[i] - center;
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(epoints, Eigen::ComputeThinV);
  V3D direc = svd.matrixV().col(0).normalized();

  *inf_line = InfiniteLine3d(center, direc, true);
}

bool TestDifferentInfLines3D(const Line2d& line2d1, const Line2d& line2d2,
                             const double overlap_th, const double angle_th,
                             const double perp_dist_th) {
  double angle = compute_angle<Line2d>(line2d1, line2d2);
  if (angle > angle_th) return true;

  double overlap = compute_bioverlap<Line2d>(line2d1, line2d2);
  if (overlap > overlap_th) return true;

  // compute max-max perpendicular distance
  double perp_dist =
      std::max(compute_distance<Line2d>(line2d1, line2d2,
                                        LineDistType::PERPENDICULAR_ONEWAY),
               compute_distance<Line2d>(line2d2, line2d1,
                                        LineDistType::PERPENDICULAR_ONEWAY));

  if (perp_dist > perp_dist_th) return true;

  return false;
}

Line3d AggregateLine3dList(const std::vector<Line3d>& lines,
                           const std::vector<double>& scores,
                           const std::string& method, const int& num_outliers) {
  THROW_CHECK_EQ(lines.size(), scores.size());
  size_t num_lines = lines.size();

  if (num_lines < 4) {
    return limap::merging::Aggregator::aggregate_line3d_list_takebest(lines,
                                                                      scores);
  }

  V3D center(0.0, 0.0, 0.0);
  V3D direc(0.0, 0.0, 0.0);

  if (method == "point_PCA") {
    // compute center
    for (size_t i = 0; i < num_lines; ++i) {
      center += lines[i].start;
      center += lines[i].end;
    }
    center = center / (2 * num_lines);
    // compute direction
    Eigen::MatrixXd endpoints;
    endpoints.resize(num_lines * 2, 3);
    for (size_t i = 0; i < num_lines; ++i) {
      endpoints.row(2 * i) = lines[i].start - center;
      endpoints.row(2 * i + 1) = lines[i].end - center;
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(endpoints, Eigen::ComputeThinV);
    direc = svd.matrixV().col(0).normalized();
  } else if (method == "point_weighted_PCA") {
    // compute center
    for (size_t i = 0; i < num_lines; ++i) {
      center += scores[i] * lines[i].start;
      center += scores[i] * lines[i].end;
    }
    center = center /
             (2.0 * std::accumulate(scores.begin(), scores.end(), 0.0) + EPS);
    // compute direction
    Eigen::MatrixXd endpoints;
    endpoints.resize(num_lines * 2, 3);
    for (size_t i = 0; i < num_lines; ++i) {
      double sqrt_w = std::sqrt(scores[i]);
      endpoints.row(2 * i) = sqrt_w * (lines[i].start - center);
      endpoints.row(2 * i + 1) = sqrt_w * (lines[i].end - center);
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(endpoints, Eigen::ComputeThinV);
    direc = svd.matrixV().col(0).normalized();
  } else if (method == "direction_PCA") {
    // compute center
    for (size_t i = 0; i < num_lines; ++i) {
      center += lines[i].start;
      center += lines[i].end;
    }
    center = center / (2 * num_lines);
    // compute direction
    Eigen::MatrixXd directions;
    directions.resize(num_lines, 3);
    for (size_t i = 0; i < num_lines; ++i) {
      directions.row(i) = lines[i].direction().normalized();
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(directions, Eigen::ComputeThinV);
    direc = svd.matrixV().col(0).normalized();
  } else if (method == "direction_weighted_PCA") {
    // compute center
    for (size_t i = 0; i < num_lines; ++i) {
      center += scores[i] * lines[i].start;
      center += scores[i] * lines[i].end;
    }
    center = center /
             (2.0 * std::accumulate(scores.begin(), scores.end(), 0.0) + EPS);
    // compute direction
    Eigen::MatrixXd directions;
    directions.resize(num_lines, 3);
    for (size_t i = 0; i < num_lines; ++i) {
      double sqrt_w = std::sqrt(scores[i]);
      directions.row(i) = sqrt_w * lines[i].direction().normalized();
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(directions, Eigen::ComputeThinV);
    direc = svd.matrixV().col(0).normalized();
  } else {
    throw std::invalid_argument("Error: Not Implemented.");
  }

  // projection
  std::vector<double> projections;
  for (size_t i = 0; i < num_lines; ++i) {
    projections.push_back((lines[i].start - center).dot(direc));
    projections.push_back((lines[i].end - center).dot(direc));
  }
  std::sort(projections.begin(), projections.end());

  // uncertainty
  double min_uncertainty = std::numeric_limits<double>::max();
  for (size_t i = 0; i < num_lines; ++i) {
    if (lines[i].uncertainty < min_uncertainty)
      min_uncertainty = lines[i].uncertainty;
  }

  // construct final line
  Line3d final_line;
  final_line.start = center + direc * projections[num_outliers];
  final_line.end =
      center + direc * projections[num_lines * 2 - 1 - num_outliers];
  final_line.uncertainty = min_uncertainty;
  return final_line;
}

}  // namespace limap