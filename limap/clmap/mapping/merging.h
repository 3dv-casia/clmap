#pragma once

#include "clmap/util/types.h"

#include "base/graph.h"
#include "base/line_linker.h"
#include "base/linebase.h"
#include "base/linetrack.h"

namespace limap {

// Cluster 2D line segments to form line tracks according to 2D collinearity
// constraint: The track elements (2D line segments) in the same image should
// be collinear as soon as possible.
std::vector<int> ComputeLineTrackLabelsCollinearity2D(
    const Graph& graph, const std::vector<Line3d>& line3d_list_nodes,
    const std::unordered_map<image_t, std::vector<Line2d>>& all_line2D,
    const double& overlap_th = 0.0, const double& angle_th = 2.0,
    const double& perp_dist_th = 2.0);

// Remerge line tracks with the 2D collinearity constraint.
std::vector<LineTrack> RemergeLineTracks(
    const std::vector<LineTrack>& linetracks, LineLinker3d linker3d,
    const int num_outliers = 2, bool use_collinearity2D = true,
    const double& overlap_th = 0.0, const double& angle_th = 2.0,
    const double& perp_dist_th = 2.0);

}  // namespace limap
