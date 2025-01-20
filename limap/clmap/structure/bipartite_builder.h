#pragma once

#include "clmap/base/bipartite.h"

#include "base/image_collection.h"
#include "base/linetrack.h"
#include "base/pointtrack.h"

namespace limap {

PP_Bipartite3d BuildInitPP_Bipartite3d(
    const std::unordered_map<int, PointTrack>& point_tracks,
    const std::unordered_map<int, InfinitePlane3d>& inf_planes3d,
    const std::unordered_map<int, int>& candidate_point_plane_bpt,
    const ImageCollection& img_cols, double var2d = 1.0,
    const std::string& u_method = "median",
    double th_hard_pointplane_si_dist3d = 2.0);

LP_Bipartite3d BuildInitLP_Bipartite3d(
    const std::unordered_map<int, LineTrack>& line_tracks,
    const std::unordered_map<int, InfinitePlane3d>& inf_planes3d,
    const std::unordered_map<int, std::unordered_set<int>>&
        candidate_line_planes_bpt,
    const ImageCollection& img_cols, double var2d = 4.0,
    const std::string& u_method = "median",
    double th_hard_lineplane_si_dist3d = 2.0,
    double th_hard_lineplane_angle = 30.0);

std::tuple<size_t, size_t, std::unordered_map<int, InfinitePlane3d>,
           PP_Bipartite3d, LP_Bipartite3d>
FilterInfPlane3dWithPP_LP_Bipartite3d(
    const std::unordered_map<int, InfinitePlane3d>& inf_planes3d,
    const PP_Bipartite3d& pp_bpt3d, const LP_Bipartite3d& lp_bpt3d,
    int min_support_lines = 2);

std::vector<V3D> GetInlierPoint3dsFromPP_Bipartite3d(
    const PP_Bipartite3d& pp_bpt3d);

std::vector<Line3d> GetInlierLine3dsFromLP_Bipartite3d(
    const LP_Bipartite3d& lp_bpt3d);

}  // namespace limap
