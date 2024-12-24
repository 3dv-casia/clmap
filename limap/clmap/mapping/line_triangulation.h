#pragma once

#include "clmap/base/bipartite.h"
#include "clmap/util/types.h"

#include <unordered_map>
#include <vector>

#include "base/image_collection.h"
#include "base/line_linker.h"
#include "base/linebase.h"
#include "structures/pl_bipartite.h"

namespace limap {

// Triangulate a 3D line segment by intersecting two back-projected planes from
// two 2D line segments.
bool TriangulateLineWithPlanes(const CameraView& view1, const CameraView& view2,
                               const Line2d& l1, const Line2d& l2,
                               Line3d* line);

// Triangulate a 3D line segment by fitting 3D points.
bool TriangulateLineWithPoints(const CameraView& view1, const Line2d& l1,
                               const std::vector<V3D>& points, Line3d* line);

void FindSharedPoints3DId(const CameraView& view1, const CameraView& view2,
                          const line2d_t& l1_idx, const line2d_t& l2_idx,
                          const structures::PL_Bipartite2d& bqt1,
                          const structures::PL_Bipartite2d& bqt2,
                          const std::unordered_map<point3d_t, V3D>& sfm_points,
                          std::unordered_set<int>* point_ids);

// point3D_ids: the ID of 3D sfm points whose 2D observations are on the 2D line
// segment.
void FindPoints3D(const line2d_t& line2d_idx,
                  const structures::PL_Bipartite2d& bqt,
                  std::unordered_set<int>* point3D_ids);

void FitInfLineWithPointsPCA(const std::vector<V3D>& points,
                             InfiniteLine3d* inf_line);

// Return true if two Line2D in the same image correspond to different
// 3D infinite lines.
bool TestDifferentInfLines3D(const Line2d& line2d1, const Line2d& line2d2,
                             const double overlap_th, const double angle_th,
                             const double perp_dist_th);

Line3d AggregateLine3dList(const std::vector<Line3d>& lines,
                           const std::vector<double>& scores,
                           const std::string& method,
                           const int& num_outliers = 2);

}  // namespace limap
